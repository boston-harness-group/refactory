"""CLI entry point for offline SkillOpt replay.

Run B on the LiveMath graph with A's seed, A's config, and A's recorded model
outputs in place of model calls.  Produces a comparison report naming the first
point where B diverges from A.

Uses B's actual code paths:
  - livemath/mode.py + livemath/ops.py for solver prompt construction
  - skillopt/trainer.py for the 6-stage SkillOpt training loop
  - skillopt/replay.py for the ReplayClient that feeds A's recorded outputs

Usage::

    python -m refactory_packages.skillopt.replay_runner \\
        --recording-dir ~/skillopt/runs/skillopt-train-20261023-e10-s42 \\
        --data-dir ~/skillopt/data/livemathematicianbench_split \\
        --output-dir /tmp/replay_s42
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from refactory_packages.skillopt.compare import compare_runs, format_report
from refactory_packages.skillopt.replay import (
    CallMatcher,
    ReplayClient,
    ReplayExhausted,
    ReplayMismatch,
)
from refactory_packages.skillopt.trainer import SkillOptTrainer


# ── config mapping ─────────────────────────────────────────────────────────


def load_a_config(a_out_dir: Path) -> dict[str, Any]:
    """Map A's ``config.json`` parameters to B's training kwargs.

    A's key → B's kwarg:
      num_epochs → max_epochs
      early_stop_patience → early_stop_patience
      early_stop_at_full_val → early_stop_at_full_val
      edit_budget → edit_budget_max
      min_edit_budget → edit_budget_min
      seed → seed
      slow_update_samples → slow_update_samples
      minibatch_size → group_size
      merge_batch_size → merge_batch_size
    """
    with open(a_out_dir / "config.json") as f:
        raw = json.load(f)

    return {
        "max_epochs": raw["num_epochs"],
        "early_stop_patience": raw["early_stop_patience"],
        "early_stop_at_full_val": raw["early_stop_at_full_val"],
        "edit_budget_max": raw["edit_budget"],
        "edit_budget_min": raw["min_edit_budget"],
        "seed": raw["seed"],
        "slow_update_samples": raw["slow_update_samples"],
        "group_size": raw["minibatch_size"],
        "merge_batch_size": raw["merge_batch_size"],
    }


# ── data loading ───────────────────────────────────────────────────────────


def _load_initial_skill(recording_dir: Path) -> str:
    """Load the initial skill from A's env data or the package default."""
    config_path = recording_dir / "out" / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        skill_init = cfg.get("skill_init", "")
        if skill_init:
            # Try absolute path first
            resolved = Path(skill_init)
            if resolved.is_absolute() and resolved.exists():
                return resolved.read_text()
            # SkillOpt resolves relative to its own project root ~/skillopt
            for base in [Path("/home/lab/skillopt"), Path.home() / "skillopt"]:
                candidate = base / skill_init
                if candidate.exists():
                    return candidate.read_text()
    return ""


def _load_train_items(data_dir: Path) -> list[dict[str, Any]]:
    """Load training items from the split directory."""
    items_file = data_dir / "train" / "items.json"
    if items_file.exists():
        with open(items_file) as f:
            return json.load(f)
    raise FileNotFoundError(f"No training items found at {items_file}")


def _load_val_items(data_dir: Path) -> list[dict[str, Any]]:
    """Load validation items from the split directory."""
    items_file = data_dir / "val" / "items.json"
    if items_file.exists():
        with open(items_file) as f:
            return json.load(f)
    return []


# ── runner ─────────────────────────────────────────────────────────────────


def run_replay(
    recording_dir: Path,
    output_dir: Path,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the offline replay: drive B's SkillOpt training loop with A's recordings.

    Creates a SkillOptTrainer with a ReplayClient and runs it through the full
    6-stage pipeline (rollout, reflect, aggregate, select, update, evaluate)
    plus epoch-end stages (slow update, meta skill).

    When a ReplayMismatch occurs (B requests a call not in A's recordings),
    that mismatch itself is the report — it identifies where B's code diverges.

    ``data_dir`` points at the 35/17/125 parts from the official id split.
    """
    a_out_dir = recording_dir / "out"
    a_config = load_a_config(a_out_dir)

    # Build replay client
    matcher = CallMatcher(recording_dir / "calls.jsonl")
    client = ReplayClient(matcher)

    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "recording_dir": str(recording_dir),
        "output_dir": str(output_dir),
        "data_dir": str(data_dir) if data_dir else None,
        "a_config": a_config,
        "calls_matched": 0,
        "calls_total": matcher.total_recordings,
        "replay_status": "not_started",
        "mismatch": None,
        "unused_recordings": [],
        "b_history": [],
        "comparison": None,
    }

    try:
        # Load initial skill
        skill_text = _load_initial_skill(recording_dir)

        # Load training items
        if data_dir is None:
            with open(a_out_dir / "config.json") as f:
                cfg = json.load(f)
            split_dir = cfg.get("split_dir", "")
            if split_dir:
                candidate = Path.home() / "skillopt" / split_dir
                if candidate.exists():
                    data_dir = candidate
        if data_dir is None:
            raise FileNotFoundError(
                "No data_dir provided and could not resolve from A's config"
            )

        train_items = _load_train_items(data_dir)
        val_items = _load_val_items(data_dir)

        # Add use_slow_update and use_meta_skill from A's config
        with open(a_out_dir / "config.json") as f:
            full_cfg = json.load(f)
        a_config["use_slow_update"] = full_cfg.get("use_slow_update", True)
        a_config["use_meta_skill"] = full_cfg.get("use_meta_skill", True)

        # Build and run the trainer
        trainer = SkillOptTrainer(
            model=client,
            config=a_config,
            train_items=train_items,
            val_items=val_items,
            initial_skill=skill_text,
            output_dir=output_dir,
            a_out_dir=a_out_dir,
        )
        b_history = trainer.run()

        report["b_history"] = b_history
        report["replay_status"] = "completed"
        report["calls_matched"] = matcher.consumed_count()

        # Report unused recordings
        unused = matcher.unused_recordings()
        report["unused_recordings"] = [
            {"system": u[0][:80], "prompt": u[1][:80], "remaining": u[2]}
            for u in unused
        ]

        # Run comparison
        comparison = compare_runs(a_out_dir, b_history)
        report["comparison"] = comparison

    except ReplayMismatch as e:
        report["replay_status"] = "mismatch"
        report["calls_matched"] = matcher.consumed_count()
        report["mismatch"] = {
            "request_system": e.request_system[:500],
            "request_prompt": e.request_prompt[:500],
            "nearest_system": e.nearest_system[:500],
            "nearest_prompt": e.nearest_prompt[:500],
            "diff": e.diff_text[:5000],
        }
        # Still run comparison on whatever B produced
        if report.get("b_history"):
            try:
                comparison = compare_runs(a_out_dir, report["b_history"])
                report["comparison"] = comparison
            except Exception:
                pass
    except ReplayExhausted as e:
        report["replay_status"] = "exhausted"
        report["calls_matched"] = matcher.consumed_count()
        report["mismatch"] = {
            "error": str(e),
            "system": e.system[:500],
            "prompt": e.prompt[:500],
            "consumed_count": e.consumed_count,
        }
    except Exception as e:
        report["replay_status"] = "error"
        report["calls_matched"] = matcher.consumed_count()
        report["mismatch"] = {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # Write report
    report_path = output_dir / "replay_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline SkillOpt replay"
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        required=True,
        help="Path to A's recording directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to the 35/17/125 id split directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path for B's output",
    )
    args = parser.parse_args()

    report = run_replay(args.recording_dir, args.output_dir, args.data_dir)

    # Print report summary
    print(f"Replay status: {report['replay_status']}")
    print(f"Calls total in recording: {report['calls_total']}")
    print(f"Calls matched: {report['calls_matched']}")

    if report.get("unused_recordings"):
        n_unused = sum(u["remaining"] for u in report["unused_recordings"])
        print(
            f"Unused recordings: {n_unused} across "
            f"{len(report['unused_recordings'])} keys"
        )

    if report.get("mismatch"):
        print("\n=== MISMATCH ===")
        print(json.dumps(report["mismatch"], indent=2)[:3000])

    if report.get("comparison"):
        text = format_report(report["comparison"])
        print("\n" + text)

    sys.exit(0 if report["replay_status"] == "completed" else 1)


if __name__ == "__main__":
    main()
