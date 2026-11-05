"""Structure ablation grid: evaluate solver blocks via the existing loop path.

A thin loop that, for each requested block/mode, runs ``refactory optimize``'s
own entry point (``refactory_dsh.loop.main``) with a one-shot strategy that calls
``Run.evaluate`` on the seed, so route resolution, the model-choice gate, the
evaluator and the run directory are optimize's. Each block's run directory is
``<output>.runs/<block>/``, and the JSON results table is rewritten after every
block.

Usage::

    refactory grid livemath/solver --skill skill.txt --parts parts.json \\
        --blocks baseline solve_then_verify --seed 42 --part val \\
        --worker-backend claude_code --model claude-opus-4-6 --effort medium

Every route flag works exactly as in a live run: ``--worker-backend``,
``--model``, ``--effort``, ``--permission-mode``, ``--setting-sources``,
``--provider``, ``data_parallelism``, ``max_completion_tokens``, and the spine
runtime (``--evaluator dsh``). With no route flag a block runs on the route
optimize would pick (the stored model choice). ``model_calls`` counts completed
agent calls (one recorded exchange each).
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


#: Route flags, forwarded to ``refactory optimize``'s own parser unchanged.
ROUTE_FLAGS = ("worker_backend", "model", "effort", "permission_mode",
               "setting_sources", "provider")
#: Knob flags, forwarded as ``--knob NAME=VALUE``.
KNOB_FLAGS = ("data_parallelism", "max_completion_tokens")


def runs_dir_for(output: Path) -> Path:
    """Where the grid keeps its block run directories: ``<output>.runs/`` beside it."""
    return output.with_name(output.stem + ".runs")


def loop_argv(mode_name: str, *, strategy: str, runs_root: Path, parts: str,
              seed: int, flags: dict[str, Any], skill_text: str) -> list[str]:
    """The ``refactory optimize`` (``refactory_dsh.loop``) arguments for one block.

    Route flags travel as optimize's own flags and the knob flags as ``--knob``,
    so the block's route goes through optimize's resolution (``profiles.resolve``,
    ``require_model_choice``); a flag nobody gave is left out, as optimize leaves it.
    """
    argv = [mode_name, "--strategy", strategy, "--runs-root", str(runs_root),
            "--parts", parts, "--holdout", "0", "--epochs", "1",
            "--max-evaluations", "1", "--seed", str(seed)]
    for flag in ROUTE_FLAGS + ("evaluator",):
        value = flags.get(flag)
        if value is not None:
            argv += [f"--{flag.replace('_', '-')}", str(value)]
    if skill_text:
        argv += ["--knob", f"skill={skill_text}"]
    for flag in KNOB_FLAGS:
        value = flags.get(flag)
        if value is not None:
            argv += ["--knob", f"{flag}={value}"]
    return argv


def model_calls(item_records: list[dict[str, Any]]) -> int:
    """Completed model calls: one recorded exchange per agent call (both runtimes)."""
    return sum(len(rec.get("exchanges") or ()) for rec in item_records)


def _occasion_dir(runs_root: Path, part: str) -> Path | None:
    found = sorted(runs_root.glob(f"runs/*/occasions/{part}_0"))
    return found[0] if found else None


def block_row(block: str, runs_root: Path, part: str) -> dict[str, Any] | None:
    """One block's table row, read from its run directory; None if nothing was evaluated."""
    occasion = _occasion_dir(runs_root, part)
    if occasion is None:
        return None

    def load(name: str, default: Any) -> Any:
        path = occasion / name
        return json.loads(path.read_text()) if path.exists() else default

    item_records = load("item_records.json", [])
    item_scores = load("facts.json", {}).get("item_scores", [])
    trials = load("measurements.json", {}).get("trials") or []
    accuracy = (sum(s.get("score", 0) for s in item_scores) / len(item_scores)
                if item_scores else 0.0)
    errors = sum(1 for rec in item_records if rec.get("error_kind"))
    timeouts = sum(1 for rec in item_records if rec.get("error_kind") == "timeout")
    shortcut_acc, other_acc, shortcut_count, other_count = _shortcut_split(item_records)
    return {
        "block": block,
        "accuracy": round(accuracy, 4),
        "shortcut_accuracy": round(shortcut_acc, 4),
        "other_accuracy": round(other_acc, 4),
        "shortcut_count": shortcut_count,
        "other_count": other_count,
        "model_calls": model_calls(item_records),
        "errors": errors,
        "timeouts": timeouts,
        "eval_duration_s": round(sum(t.get("duration_s") or 0 for t in trials), 2),
        "run_dir": str(runs_root),
        "item_records": item_records,
    }


def _write(output_path: Path, payload: dict[str, Any]) -> None:
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(output_path)


def main(argv: list[str] | None = None, *, _client_factory: Any = None) -> int:
    """Entry point for ``refactory grid``.

    Each block runs as ``refactory optimize <block mode>`` would, in this process
    (``refactory_dsh.loop.main``), in ``<output>.runs/<block>/``; the results file
    is rewritten after every block (``complete`` false until the last one).
    ``_client_factory`` is passed through to ``loop.main`` (tests answer with a stub).
    """
    import contextlib
    import io
    import os

    from refactory_dsh import loop

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

    skill_text = ""
    if args.skill:
        skill_path = Path(args.skill)
        if not skill_path.exists():
            print(f"refactory grid: skill file not found: {args.skill}", file=sys.stderr)
            return 1
        skill_text = skill_path.read_text()

    parts_path = Path(args.parts)
    if not parts_path.exists():
        print(f"refactory grid: parts manifest not found: {args.parts}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    runs_dir = runs_dir_for(output_path)
    taken = [name for name in block_names if (runs_dir / name).exists()]
    if taken:
        print(f"refactory grid: {runs_dir} already holds runs for {taken}; "
              "choose another --output or remove them", file=sys.stderr)
        return 2
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.data_path:
        os.environ["LIVEMATH_DATA_PATH"] = args.data_path

    flags = {flag: getattr(args, flag) for flag in ROUTE_FLAGS + KNOB_FLAGS + ("evaluator",)}
    strategy = _register_grid_strategy(args.part, args.seed)
    results: list[dict[str, Any]] = []

    def payload(complete: bool) -> dict[str, Any]:
        return {"seed": args.seed, "part": args.part, "complete": complete,
                "runs_dir": str(runs_dir), "blocks": results}

    for block_name in block_names:
        runs_root = runs_dir / block_name
        t0 = time.time()
        printed = io.StringIO()
        try:
            # A Ctrl-C reaches the loop as it reaches `refactory optimize`: the data
            # pass cancels its items and each model client shuts its call down.
            with contextlib.redirect_stdout(printed):
                rc = loop.main(loop_argv(
                    _BLOCK_TO_MODE[block_name], strategy=strategy, runs_root=runs_root,
                    parts=str(parts_path.resolve()), seed=args.seed, flags=flags,
                    skill_text=skill_text,
                ), _client_factory=_client_factory)
        except KeyboardInterrupt:
            print(f"refactory grid: interrupted during {block_name}; "
                  f"finished blocks are in {output_path}", file=sys.stderr)
            return 130
        finally:
            if runs_root.is_dir():
                (runs_root / "optimize.json").write_text(printed.getvalue())
        row = block_row(block_name, runs_root, args.part)
        if row is None:
            _write(output_path, payload(False))
            print(f"refactory grid: {block_name} evaluated nothing (exit {rc}); "
                  f"see {runs_root}", file=sys.stderr)
            return rc or 1
        row["wall_time_s"] = round(time.time() - t0, 2)
        results.append(row)
        _write(output_path, payload(False))

    _write(output_path, payload(True))
    print(f"Grid results written to {output_path}")

    print(f"\n{'Block':<30} {'Acc':>6} {'Shortcut':>8} {'Other':>6} "
          f"{'Calls':>6} {'Err':>4} {'Time':>7}")
    print("-" * 75)
    for row in results:
        print(f"{row['block']:<30} {row['accuracy']:>6.3f} "
              f"{row['shortcut_accuracy']:>8.3f} {row['other_accuracy']:>6.3f} "
              f"{row['model_calls']:>6} {row['errors']:>4} "
              f"{row['wall_time_s']:>6.1f}s")

    return 0
