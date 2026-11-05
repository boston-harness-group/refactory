"""Structure ablation grid: evaluate solver blocks via the existing testeval path.

A thin loop that, for each requested block/mode, runs the existing
``Run.evaluate`` machinery (the same path ``refactory optimize livemath/solver
--strategy testeval`` uses) with forwarded route flags, then writes a JSON
results table.

Usage::

    refactory grid livemath/solver --skill skill.txt --parts parts.json \\
        --blocks baseline solve_then_verify --seed 42 --part val \\
        --worker-backend claude_code --model claude-opus-4-6 --effort medium

Every route flag works exactly as in a live run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


# Block name → registered mode name.
_BLOCK_TO_MODE = {
    "baseline": "livemath/solver",
    "solve_then_verify": "livemath/solver_verify",
    "two_attempts_then_judge": "livemath/solver_judge",
    "solve_then_self_check": "livemath/solver_self_check",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a structure-ablation grid over solver blocks",
    )
    parser.add_argument("mode", help="the mode family, e.g. livemath/solver")
    parser.add_argument("--skill", default="", help="path to a skill file")
    parser.add_argument("--part", default="val", choices=["val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blocks", nargs="+", default=None,
                        help="which blocks to evaluate (default: all)")
    parser.add_argument("--output", default="grid_results.json")
    parser.add_argument("--parts", required=True, help="path to parts.json manifest")
    parser.add_argument("--data-path", default="", help="override LIVEMATH_DATA_PATH")
    # Route flags forwarded to the evaluator.
    parser.add_argument("--worker-backend", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None)
    parser.add_argument("--permission-mode", default=None)
    parser.add_argument("--setting-sources", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--data-parallelism", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    return parser


def _shortcut_split(
    item_records: list[dict[str, Any]],
) -> tuple[float, float, int, int]:
    """Split accuracy into shortcut vs other items.

    A shortcut item's ``correct_choice["text"]`` contains
    ``"stronger result can be proven"`` (case-sensitive).
    """
    shortcut_correct = 0
    shortcut_total = 0
    other_correct = 0
    other_total = 0
    for rec in item_records:
        correct_text = ""
        meta = rec.get("metadata") or {}
        cc = meta.get("correct_choice") or {}
        correct_text = cc.get("text", "")
        is_shortcut = "stronger result can be proven" in correct_text
        passed = rec.get("passed", rec.get("score", 0) > 0)
        if is_shortcut:
            shortcut_total += 1
            if passed:
                shortcut_correct += 1
        else:
            other_total += 1
            if passed:
                other_correct += 1
    shortcut_acc = shortcut_correct / shortcut_total if shortcut_total else 0.0
    other_acc = other_correct / other_total if other_total else 0.0
    return shortcut_acc, other_acc, shortcut_total, other_total


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``refactory grid``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode != "livemath/solver":
        print(f"refactory grid: only livemath/solver is supported, got {args.mode!r}",
              file=sys.stderr)
        return 2

    from .blocks import BLOCK_NAMES

    block_names = args.blocks or list(BLOCK_NAMES)
    for name in block_names:
        if name not in _BLOCK_TO_MODE:
            print(f"refactory grid: unknown block {name!r}; known: {list(_BLOCK_TO_MODE)}",
                  file=sys.stderr)
            return 2

    # Read skill text.
    skill_text = ""
    if args.skill:
        skill_path = Path(args.skill)
        if not skill_path.exists():
            print(f"refactory grid: skill file not found: {args.skill}", file=sys.stderr)
            return 1
        skill_text = skill_path.read_text()

    # Read parts manifest.
    parts_path = Path(args.parts)
    if not parts_path.exists():
        print(f"refactory grid: parts manifest not found: {args.parts}", file=sys.stderr)
        return 1
    parts_manifest = json.loads(parts_path.read_text())
    item_ids = parts_manifest.get(args.part, [])
    if not item_ids:
        print(f"refactory grid: no items in part {args.part!r}", file=sys.stderr)
        return 1

    import os
    import tempfile

    from refactory_dsh.evaluators import WalkerEvaluator
    from refactory_dsh.nn import RunResult
    from refactory_dsh.task import write_subset

    # Build the model client with forwarded route flags.
    model = None
    route_flags: dict[str, str] = {}
    for flag in ("worker_backend", "model", "effort", "permission_mode",
                 "setting_sources", "provider"):
        value = getattr(args, flag, None)
        if value is not None:
            route_flags[flag] = value

    if route_flags:
        from refactory_dsh.model import DshModelClient

        model = DshModelClient(
            worker_backend=route_flags.get("worker_backend", "dsh"),
            model=route_flags.get("model", ""),
            effort=route_flags.get("effort", ""),
            permission_mode=route_flags.get("permission_mode", ""),
            setting_sources=route_flags.get("setting_sources", ""),
            provider=route_flags.get("provider", ""),
        )

    if args.data_path:
        os.environ["LIVEMATH_DATA_PATH"] = args.data_path

    results: list[dict[str, Any]] = []

    from .mode import Solver, SolverJudge, SolverSelfCheck, SolverVerify

    _BLOCK_TO_CLASS = {
        "baseline": Solver,
        "solve_then_verify": SolverVerify,
        "two_attempts_then_judge": SolverJudge,
        "solve_then_self_check": SolverSelfCheck,
    }

    for block_name in block_names:
        mode_cls = _BLOCK_TO_CLASS[block_name]
        mode = mode_cls()

        # Set the skill knob.
        if skill_text:
            from refactory_dsh.nn import edits

            state = mode.state_dict()
            state = edits.configure(state, "skill", skill_text)
            mode.load_state_dict(state)

        # Set data_parallelism and max_completion_tokens if specified.
        if args.data_parallelism is not None:
            from refactory_dsh.nn import edits

            state = mode.state_dict()
            state = edits.configure(state, "data_parallelism", str(args.data_parallelism))
            mode.load_state_dict(state)

        if args.max_completion_tokens is not None:
            from refactory_dsh.nn import edits

            state = mode.state_dict()
            state = edits.configure(state, "max_completion_tokens", str(args.max_completion_tokens))
            mode.load_state_dict(state)

        # Create a temporary workdir for this evaluation.
        with tempfile.TemporaryDirectory(prefix=f"grid-{block_name}-") as tmpdir:
            workdir = Path(tmpdir) / "run"
            workdir.mkdir()

            # Write the subset file for this part/seed.
            write_subset(workdir, args.part, 0.0, seed=args.seed)

            evaluator = WalkerEvaluator(
                model=model,
                project=Path("."),
                max_steps=20000,
            )

            t0 = time.time()
            result = evaluator.evaluate(mode, mode.state_dict(), workdir)
            wall_time = round(time.time() - t0, 2)

        # Extract accuracy.
        item_scores = result.facts.get("item_scores", [])
        accuracy = mode.score(result)

        # Count errors/timeouts.
        errors = sum(1 for r in result.item_records if r.get("error_kind"))
        timeouts = sum(1 for r in result.item_records
                       if r.get("error_kind") == "timeout")

        # Shortcut split.
        shortcut_acc, other_acc, shortcut_count, other_count = _shortcut_split(
            result.item_records)

        results.append({
            "block": block_name,
            "accuracy": round(accuracy, 4),
            "shortcut_accuracy": round(shortcut_acc, 4),
            "other_accuracy": round(other_acc, 4),
            "shortcut_count": shortcut_count,
            "other_count": other_count,
            "model_calls": result.model_calls,
            "errors": errors,
            "timeouts": timeouts,
            "wall_time_s": wall_time,
            "item_records": result.item_records,
        })

    output = {
        "seed": args.seed,
        "part": args.part,
        "blocks": results,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"Grid results written to {output_path}")

    # Print summary table.
    print(f"\n{'Block':<30} {'Acc':>6} {'Shortcut':>8} {'Other':>6} "
          f"{'Calls':>6} {'Err':>4} {'Time':>7}")
    print("-" * 75)
    for row in results:
        print(f"{row['block']:<30} {row['accuracy']:>6.3f} "
              f"{row['shortcut_accuracy']:>8.3f} {row['other_accuracy']:>6.3f} "
              f"{row['model_calls']:>6} {row['errors']:>4} "
              f"{row['wall_time_s']:>6.1f}s")

    return 0
