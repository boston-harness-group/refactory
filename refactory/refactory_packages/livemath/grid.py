"""Structure ablation grid: evaluate solver blocks via the existing loop path.

A thin loop that, for each requested block/mode, runs the existing
``Run.evaluate`` machinery (the same path ``refactory optimize livemath/solver
--strategy testeval`` uses) with forwarded route flags, then writes a JSON
results table.

Usage::

    refactory grid livemath/solver --skill skill.txt --parts parts.json \\
        --blocks baseline solve_then_verify --seed 42 --part val \\
        --worker-backend claude_code --model claude-opus-4-6 --effort medium

Every route flag works exactly as in a live run: ``--worker-backend``,
``--model``, ``--effort``, ``--permission-mode``, ``--setting-sources``,
``--provider``, ``data_parallelism``, ``max_completion_tokens``, and the spine
runtime (``--evaluator dsh``).
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
    # Route flags forwarded to the loop.
    parser.add_argument("--worker-backend", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None)
    parser.add_argument("--permission-mode", default=None)
    parser.add_argument("--setting-sources", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--data-parallelism", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--evaluator", default=None,
                        help="which evaluator: walker (default) or dsh (spine)")
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


def _register_grid_strategy(part: str, seed: int) -> str:
    """Register a one-shot strategy that evaluates the seed on the given part.

    Returns the strategy name to use in the loop config.  Each (part, seed) pair
    gets a unique name so registrations do not collide.
    """
    from refactory_dsh import strategies
    from refactory_dsh.outerloop import OuterLoop, Run

    name = f"_grid_eval_{part}_{seed}"
    if name in strategies.STRATEGIES:
        return name

    class _GridEval(OuterLoop):
        """Evaluate the seed on one part with one seed, then stop."""

        def strategy(self, run: Run) -> None:
            run.evaluate(run.seed(), part=part, seed=seed)
            run.stop("grid evaluated")

    _GridEval.name = name
    strategies.register(name, _GridEval)
    return name


def _run_one_block(
    mode_name: str,
    *,
    part: str,
    seed: int,
    parts_manifest: str,
    skill_text: str,
    runs_root: Path,
    route_flags: dict[str, Any],
    knobs: dict[str, str],
    evaluator: str | None,
    model: Any = None,
) -> dict[str, Any]:
    """Evaluate one mode through the loop machinery and return the result.

    This is the same path ``refactory optimize <mode> --strategy testeval``
    uses: ``strategies.create()`` → ``loop.run()`` → ``Run.evaluate()``.

    ``model`` may be passed by a caller (or test) that already has a client;
    when omitted, one is built from *route_flags*, matching what
    ``loop.main()`` does.
    """
    from refactory_dsh.loop import LoopConfig
    from refactory_dsh import strategies

    strategy_name = _register_grid_strategy(part, seed)

    # Build LoopConfig from flags — same fields ``loop.main()`` sets.
    config_dict: dict[str, Any] = {
        "runs_root": str(runs_root),
        "strategy": strategy_name,
        "epochs": 1,
        "max_evaluations": 1,
        "holdout": 0,
        "seed": seed,
        "parts_manifest": parts_manifest,
        "print_events": False,
        "verbose": False,
        "knobs": {},
    }
    for flag in ("worker_backend", "model", "effort", "permission_mode",
                 "setting_sources", "provider"):
        value = route_flags.get(flag)
        if value is not None:
            config_dict[flag] = value
    if evaluator is not None:
        config_dict["evaluator"] = evaluator

    # Set knobs
    for name, value in knobs.items():
        config_dict["knobs"][name] = value

    config = LoopConfig.from_dict(config_dict)

    # Build the model client when one is not already provided.
    if model is None:
        has_route = any(
            route_flags.get(f) is not None
            for f in ("worker_backend", "model", "effort", "permission_mode",
                       "setting_sources", "provider")
        )
        if has_route:
            from refactory_dsh.model import DshModelClient
            model = DshModelClient(
                worker_backend=route_flags.get("worker_backend", "dsh"),
                model=route_flags.get("model", ""),
                effort=route_flags.get("effort", ""),
                permission_mode=route_flags.get("permission_mode", ""),
                setting_sources=route_flags.get("setting_sources", ""),
                provider=route_flags.get("provider", ""),
            )

    loop = strategies.create(
        strategy_name, mode_name, config,
        project=Path("."), model=model,
    )
    report = loop.run()

    # Extract results from the loop.
    seed_candidate = loop.population[0] if loop.population else None
    item_records: list[dict[str, Any]] = []
    item_scores: list[dict[str, Any]] = []
    model_calls = 0
    duration_s = 0.0
    accuracy = 0.0
    errors = 0
    timeouts = 0

    if seed_candidate is not None:
        # Read part results from the loop's occasion directory.
        part_results = loop._part_results.get(seed_candidate.id, {}).get(part, [])
        if part_results:
            occasion = part_results[0]
            # Read item records from the occasion directory.
            occasion_num = 0
            occasion_dir = runs_root / "runs" / seed_candidate.id / "occasions" / f"{part}_{occasion_num}"
            records_path = occasion_dir / "item_records.json"
            if records_path.exists():
                item_records = json.loads(records_path.read_text())
            facts_path = occasion_dir / "facts.json"
            if facts_path.exists():
                facts = json.loads(facts_path.read_text())
                item_scores = facts.get("item_scores", [])
            duration_s = occasion.get("duration_s") or 0.0
        # Model calls come from the seed candidate evaluation.
        model_calls = seed_candidate.model_calls or 0
        # Compute accuracy from item_scores.
        if item_scores:
            accuracy = sum(s.get("score", 0) for s in item_scores) / len(item_scores)

    # Count errors and timeouts from item_records.
    for rec in item_records:
        if rec.get("error_kind"):
            errors += 1
            if rec.get("error_kind") == "timeout":
                timeouts += 1

    # Shortcut split.
    shortcut_acc, other_acc, shortcut_count, other_count = _shortcut_split(
        item_records)

    return {
        "block": None,  # set by caller
        "accuracy": round(accuracy, 4),
        "shortcut_accuracy": round(shortcut_acc, 4),
        "other_accuracy": round(other_acc, 4),
        "shortcut_count": shortcut_count,
        "other_count": other_count,
        "model_calls": model_calls,
        "errors": errors,
        "timeouts": timeouts,
        "wall_time_s": round(duration_s, 2),
        "item_records": item_records,
    }


def main(argv: list[str] | None = None, *, _model: Any = None) -> int:
    """Entry point for ``refactory grid``.

    ``_model`` is an internal hook for tests: when provided it is passed to the
    loop instead of building a ``DshModelClient`` from the route flags.
    """
    import os
    import tempfile

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

    if args.data_path:
        os.environ["LIVEMATH_DATA_PATH"] = args.data_path

    # Collect route flags.
    route_flags: dict[str, Any] = {}
    for flag in ("worker_backend", "model", "effort", "permission_mode",
                 "setting_sources", "provider"):
        value = getattr(args, flag, None)
        if value is not None:
            route_flags[flag] = value

    # Build knobs dict.
    knobs: dict[str, str] = {}
    if skill_text:
        knobs["skill"] = skill_text
    if args.data_parallelism is not None:
        knobs["data_parallelism"] = str(args.data_parallelism)
    if args.max_completion_tokens is not None:
        knobs["max_completion_tokens"] = str(args.max_completion_tokens)

    results: list[dict[str, Any]] = []

    for block_name in block_names:
        mode_name = _BLOCK_TO_MODE[block_name]

        with tempfile.TemporaryDirectory(prefix=f"grid-{block_name}-") as tmpdir:
            runs_root = Path(tmpdir) / "loop"

            t0 = time.time()
            result = _run_one_block(
                mode_name,
                part=args.part,
                seed=args.seed,
                parts_manifest=str(parts_path),
                skill_text=skill_text,
                runs_root=runs_root,
                route_flags=route_flags,
                knobs=knobs,
                evaluator=args.evaluator,
                model=_model,
            )
            wall_time = round(time.time() - t0, 2)

        result["block"] = block_name
        result["wall_time_s"] = wall_time
        results.append(result)

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
