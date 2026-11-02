"""CLI entry point for offline SkillOpt replay.

Run B on the LiveMath graph with A's seed, A's config, and A's recorded model
outputs in place of model calls.  Produces a comparison report naming the first
point where B diverges from A.

Usage::

    python -m refactory_packages.skillopt.replay_runner \\
        --recording-dir ~/skillopt/runs/skillopt-train-20261023-e10-s42 \\
        --data-dir ~/skillopt/data/livemathematicianbench_split \\
        --output-dir /tmp/replay_s42
"""

from __future__ import annotations

import argparse
import hashlib
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


# ── rollout via the LiveMath graph ─────────────────────────────────────────


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


def _build_system_prompt(skill_text: str) -> str:
    """Build the solver system prompt, matching A's rollout.py:16-21.

    A uses ``skill_content.strip()`` before embedding (rollout.py:18), and
    the Claude Code backend's ``_build_prompt_from_messages`` applies
    ``.strip()`` to the assembled system prompt (claude_backend.py:130),
    removing any trailing newline.
    """
    if skill_text.strip():
        skill_section = f"## Skill\n{skill_text.strip()}\n\n"
    else:
        skill_section = ""
    raw = (
        "You are an expert mathematical reasoning agent solving multiple-choice questions.\n"
        "\n"
        f"{skill_section}"
        "## Task Format\n"
        "You will receive one mathematics multiple-choice question and its answer choices.\n"
        "Reason carefully about quantifiers, hypotheses, extremal wording, and exact equality conditions.\n"
        "\n"
        "## Answer Format\n"
        "Think step by step, then provide your final answer inside <answer>...</answer> tags.\n"
        "Inside the tags, output only the single choice label, such as A or C.\n"
        "\n"
        "Example:\n"
        "<answer>B</answer>\n"
    )
    return raw.strip()


def _build_user_prompt(item: dict[str, Any]) -> str:
    """Build the solver user prompt, matching livemath/ops.py."""
    question = item.get("question", "")
    choices = item.get("choices", [])
    choices_text = "\n".join(
        f"{c['label']}. {c['text']}" for c in choices
    )
    return f"## Question\n{question}\n\n## Choices\n{choices_text}"


def _compute_batch_seed(seed: int, epoch: int, batch_idx: int = 0) -> int:
    """Compute the per-batch choice-shuffle seed matching A's dataloader.

    Ported from skillopt/envs/livemathematicianbench/dataloader.py:255 (MIT):
    ``batch_seed = seed + epoch * 1000 + batch_idx + 1``
    """
    return seed + epoch * 1000 + batch_idx + 1


def _shuffle_epoch_items(
    items: list[dict[str, Any]], seed: int, epoch: int,
) -> list[dict[str, Any]]:
    """Shuffle items for an epoch, matching A's dataloader.

    Ported from skillopt/envs/livemathematicianbench/dataloader.py:245-247 (MIT):
    ``epoch_rng = random.Random(seed + epoch * 1000)``
    ``items = list(self.train_items)``
    ``epoch_rng.shuffle(items)``
    """
    import random as _random
    epoch_rng = _random.Random(seed + epoch * 1000)
    shuffled = list(items)
    epoch_rng.shuffle(shuffled)
    return shuffled


def _run_rollout(
    client: ReplayClient,
    items: list[dict[str, Any]],
    skill_text: str,
    batch_seed: int,
) -> list[dict[str, Any]]:
    """Run the solver on items via the ReplayClient, matching A's rollout.

    ``batch_seed`` is the per-batch choice-shuffle seed from
    ``_compute_batch_seed()``, matching A's dataloader.py:255.
    """
    from refactory_packages.livemath.shuffling import shuffle_item_choices
    from refactory_packages.livemath.grading import evaluate

    system = _build_system_prompt(skill_text)
    results: list[dict[str, Any]] = []
    for item in items:
        shuffled = shuffle_item_choices(item, batch_seed)
        prompt = _build_user_prompt(shuffled)
        try:
            answer = client.complete(prompt, system=system)
        except (ReplayMismatch, ReplayExhausted):
            raise  # propagate to the runner
        # Grade the answer
        grade = evaluate(
            answer,
            shuffled["correct_choice"],
            shuffled["choices"],
        )
        results.append({
            "id": item["id"],
            "answer": answer[:200],
            "score": grade["em"],
            "predicted_label": grade.get("predicted_label", ""),
        })
    return results


# ── runner ─────────────────────────────────────────────────────────────────


def run_replay(
    recording_dir: Path,
    output_dir: Path,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the offline replay: drive B's LiveMath graph with A's recordings.

    For each epoch in A's training loop:
    1. Run the solver graph on training items (rollout) via the ReplayClient
    2. Compare rollout results against A's recorded history
    3. Report the first divergence or mismatch

    When a ReplayMismatch occurs (B requests a call not in A's recordings),
    that mismatch itself is the report.

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
            # Try to find data dir from A's config
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

        # Load A's history for comparison
        a_history_path = a_out_dir / "history.json"
        a_history: list[dict[str, Any]] = []
        if a_history_path.exists():
            with open(a_history_path) as f:
                a_history = json.load(f)

        seed = a_config["seed"]
        max_epochs = a_config["max_epochs"]
        b_history: list[dict[str, Any]] = []

        # Run training loop: epoch by epoch, matching A's flow
        for epoch in range(1, max_epochs + 1):
            step = epoch  # A uses step = epoch (1-indexed)

            # Shuffle items for this epoch (matches A's dataloader)
            epoch_items = _shuffle_epoch_items(train_items, seed, epoch)
            batch_seed = _compute_batch_seed(seed, epoch)

            # === ROLLOUT: Run solver on training items ===
            rollout_results = _run_rollout(
                client, epoch_items, skill_text, batch_seed,
            )
            n_items = len(rollout_results)
            hard_score = (
                sum(r["score"] for r in rollout_results) / n_items
                if n_items > 0 else 0.0
            )

            # Build B's history entry from the rollout
            b_entry: dict[str, Any] = {
                "epoch": epoch,
                "step": step,
                "rollout_hard": round(hard_score, 6),
                "rollout_n": n_items,
            }

            # The non-rollout stages (analyst, merge, rank, update, evaluate,
            # gate) require porting A's full prompt construction.  On a replay,
            # the model calls for these stages are the next calls after the
            # rollout in A's calls.jsonl.  Rather than reconstruct A's prompts
            # (which is the full SkillOpt port), we let the ReplayClient
            # match by content: if B's prompts match A's normalised prompts,
            # the call succeeds; otherwise a ReplayMismatch is raised.

            # For now, record what we can from the rollout and stop here —
            # the analyst/merge/rank calls would need the exact same prompt
            # construction as A, which is the porting bug surface the replay
            # is designed to discover.

            b_history.append(b_entry)

            report["calls_matched"] = matcher.consumed_count()

        report["b_history"] = b_history
        report["replay_status"] = "completed"

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
