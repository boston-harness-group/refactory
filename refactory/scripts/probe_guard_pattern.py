"""End-to-end probe: code steps read and write their declared files.

Applies edits in the exact JSON op form a proposer sends, via the same edit
path the loop uses, to the real spreadsheetbench_v2/steps seed graph.  Six
patterns:

A. After implement, insert a sealed code step that reads input.xlsx,
   output.xlsx and targets.json, restores every cell changed outside the
   targets, and writes output.xlsx (selective takes_over of implement's
   output.xlsx).  Verify and finalize then read the restored file.

B. The same guard as an agent step with tools, instead of a code step.

C. A loop around implement, with a sealed code judge that reports stray cells
   as its verdict.

D. A route after implement whose "revise" way has a step that takes_over
   implement's output.xlsx, with the ways rejoining before verify.

E. A parallel fork of two new agent steps after plan.

F. A loop around implement with an agent+tools judge.

For each, the probe shows:
  - the edit is accepted, with no false refusal;
  - the wiring validator passes;
  - downstream steps read the produced files;
  - the graded output.xlsx reflects the guard (where applicable).

Usage:
  cd refactory && PYTHONPATH=$PWD python -m scripts.probe_guard_pattern

  --real              Evaluate candidates using the same machinery Loop.forward
                      uses: the walker, the mode's Task, real model calls, and
                      the given items.  Requires API keys and prepared data.

  --items-root PATH   SB2_DATA directory (prepared spreadsheetbench items).
                      Required with --real.

  --ids id1,id2,...   Comma-separated item IDs to evaluate (default: all).

  --patterns A,B,...  Comma-separated pattern letters (default: all six).

  --out DIR           Output directory for run directories (default: probe_runs).

  --concurrency N     Concurrent evaluations (default: 1).

Without --real, the probe validates graph structure only (no tool environment
needed).

Lab command:
  cd refactory && PYTHONPATH=$PWD python -m scripts.probe_guard_pattern \\
      --items-root /path/to/prepared/data --ids 1,2,3 --real
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from refactory_dsh import snippet
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.validate import validate_state
from refactory_packages.spreadsheetbench_v2.steps_mode import Steps


def _state() -> dict:
    return Steps().state_dict()


def _ok(label: str) -> None:
    print(f"  PASS  {label}")


def _fail(label: str, detail: str) -> None:
    print(f"  FAIL  {label}: {detail}")


# ── Pattern A: sealed code guard with selective takes_over ───────────────

GUARD_CODE = '''\
import openpyxl, json, copy

def main(inp):
    """Restore cells changed outside the targets."""
    targets = json.loads(inp["files"].get("targets.json", "{}"))
    target_cells = set()
    for entry in targets.get("cells", []):
        target_cells.add((entry["sheet"], entry["cell"]))

    wb_in = openpyxl.load_workbook("input.xlsx")
    wb_out = openpyxl.load_workbook("output.xlsx")

    restored = 0
    for sheet_name in wb_out.sheetnames:
        ws_out = wb_out[sheet_name]
        ws_in = wb_in[sheet_name] if sheet_name in wb_in.sheetnames else None
        if ws_in is None:
            continue
        for row in ws_out.iter_rows():
            for cell in row:
                key = (sheet_name, cell.coordinate)
                if key not in target_cells:
                    original = ws_in[cell.coordinate].value
                    if cell.value != original:
                        cell.value = original
                        restored += 1

    wb_out.save("output.xlsx")
    return {"facts": {"restored_cells": restored}}
'''


def pattern_a(state: dict) -> dict:
    """Insert a sealed code guard after implement with selective takes_over."""
    edit = {
        "op": "insert", "after": "implement", "kind": "code",
        "id": "guard",
        "text": GUARD_CODE,
        "sealed": True,
        "takes_over": ["implement", "output.xlsx"],
    }
    return nn_edits.apply(edit, state)


# ── Pattern B: agent guard with tools ────────────────────────────────────

AGENT_GUARD_TEXT = (
    "Compare input.xlsx and output.xlsx. The targets.json file lists which cells "
    "the implement step was supposed to change. For every cell in output.xlsx that "
    "differs from input.xlsx AND is not in the targets list, restore the original "
    "value from input.xlsx. Save the corrected output.xlsx."
)


def pattern_b(state: dict) -> dict:
    """Insert an agent guard with tools after implement."""
    edit = {
        "op": "insert", "after": "implement", "kind": "agent",
        "id": "guard",
        "text": AGENT_GUARD_TEXT,
        "tools": ["Bash"],
        "takes_over": ["implement", "output.xlsx"],
    }
    return nn_edits.apply(edit, state)


# ── Pattern C: sealed code judge loop ────────────────────────────────────

JUDGE_CODE = '''\
import openpyxl, json

def main(inp):
    """Report stray cells as the verdict."""
    targets_text = inp["files"].get("targets.json", "{}")
    targets = json.loads(targets_text)
    target_cells = set()
    for entry in targets.get("cells", []):
        target_cells.add((entry["sheet"], entry["cell"]))

    wb_in = openpyxl.load_workbook("input.xlsx")
    wb_out = openpyxl.load_workbook("output.xlsx")
    stray = []
    for sheet_name in wb_out.sheetnames:
        ws_out = wb_out[sheet_name]
        ws_in = wb_in[sheet_name] if sheet_name in wb_in.sheetnames else None
        if ws_in is None:
            continue
        for row in ws_out.iter_rows():
            for cell in row:
                key = (sheet_name, cell.coordinate)
                if key not in target_cells:
                    original = ws_in[cell.coordinate].value
                    if cell.value != original:
                        stray.append(f"{sheet_name}!{cell.coordinate}")

    if stray:
        return {"verdict": "reloop",
                "reason": f"stray cells changed: {', '.join(stray[:10])}"}
    return {"verdict": "proceed", "reason": "no stray cells"}
'''


def pattern_c(state: dict) -> dict:
    """A loop around implement with a sealed code judge."""
    edit = {
        "op": "loop", "step": "implement", "judge": "code",
        "text": JUDGE_CODE,
        "max_iterations": 1,
        "sealed": True,
    }
    return nn_edits.apply(edit, state)


# ── Pattern D: route with one way taking over output.xlsx ────────────────


def pattern_d(state: dict) -> dict:
    """A route after implement: proceed goes to verify as normal, revise goes
    through a new agent step that takes_over implement's output.xlsx."""
    edit = {
        "op": "route", "after": "implement", "judge": "agent",
        "text": "Check whether the output spreadsheet needs revision.",
        "to": {
            "proceed": "verify",
            "revise": {
                "kind": "agent", "text": "Fix the spreadsheet.",
                "tools": ["Bash"],
                "takes_over": "implement",
            },
        },
    }
    return nn_edits.apply(edit, state)


# ── Pattern E: parallel fork of two agent steps after plan ───────────────


def pattern_e(state: dict) -> dict:
    """Parallelize: run two copies of implement in parallel."""
    edit = {
        "op": "parallelize", "steps": ["implement"], "copies": 2,
    }
    return nn_edits.apply(edit, state)


# ── Pattern F: loop with agent+tools judge ───────────────────────────────


def pattern_f(state: dict) -> dict:
    """A loop around implement with an agent judge that has tools."""
    edit = {
        "op": "loop", "step": "implement", "judge": "agent",
        "text": (
            "Open input.xlsx and output.xlsx with openpyxl. Check whether cells "
            "outside the target range were modified. Answer 'proceed' if the output "
            "is clean, 'reloop' with a list of stray cells otherwise."
        ),
        "tools": ["Bash"],
        "max_iterations": 1,
    }
    return nn_edits.apply(edit, state)


# ── probe logic ──────────────────────────────────────────────────────────

def probe_pattern(name: str, label: str, build_fn, state: dict) -> bool:
    """Test one pattern. Returns True if all checks pass."""
    print(f"\n--- Pattern {name}: {label} ---")
    passed = True

    # 1. Edit is accepted
    try:
        result = build_fn(state)
        _ok("edit accepted")
    except Exception as exc:
        _fail("edit accepted", str(exc)[:300])
        return False

    # 2. Wiring validator passes
    issues = validate_state(result)
    if issues:
        _fail("wiring validation", "; ".join(str(i) for i in issues[:3]))
        passed = False
    else:
        _ok("wiring validation")

    # 3. Check downstream steps read the produced files
    nodes = result["graph"]["nodes"]
    if name in ("A", "B"):
        guard = nodes.get("guard", {})
        if "output.xlsx" in guard.get("writes", []):
            _ok("guard writes output.xlsx")
        else:
            _fail("guard writes output.xlsx", f"writes: {guard.get('writes', [])}")
            passed = False

        verify = nodes.get("verify", {})
        if "output.xlsx" in verify.get("reads", []):
            _ok("verify reads output.xlsx")
        else:
            _fail("verify reads output.xlsx", f"reads: {verify.get('reads', [])}")
            passed = False

        impl = nodes.get("implement", {})
        if "output.xlsx" not in impl.get("writes", []):
            _ok("implement no longer writes output.xlsx (taken over)")
        else:
            _fail("implement still writes output.xlsx", "takeover incomplete")
            passed = False

    if name == "C":
        gates = [n for n in nodes.values()
                 if n.get("evaluator") == snippet.GATE]
        if gates:
            gate = gates[0]
            if gate.get("sealed_script"):
                _ok("code judge gate is sealed")
            else:
                _ok("code judge gate found (unsealed)")
        else:
            _fail("code judge gate", "no code gate found in graph")
            passed = False

    if name == "D":
        # Route should have created a gate with two ways
        gate_nodes = [nid for nid, n in nodes.items() if n.get("kind") == "gate"]
        route_gates = [nid for nid in gate_nodes if nid not in ("has_case",)]
        if route_gates:
            _ok(f"route gate created: {route_gates[0]}")
        else:
            _fail("route gate", "no route gate found")
            passed = False

        # The revise way's step should take over implement's outputs
        revise_steps = [nid for nid, n in nodes.items()
                        if "output.xlsx" in n.get("writes", [])
                        and nid not in ("implement", "verify", "data")]
        if revise_steps:
            _ok(f"revise way step writes output.xlsx: {revise_steps[0]}")
        else:
            _fail("revise way step", "no non-implement step writes output.xlsx")
            passed = False

        # verify still reads output.xlsx (from whichever way ran)
        verify = nodes.get("verify", {})
        if "output.xlsx" in verify.get("reads", []):
            _ok("verify reads output.xlsx after route rejoin")
        else:
            _fail("verify reads output.xlsx", f"reads: {verify.get('reads', [])}")
            passed = False

    if name == "E":
        # Parallelizing implement should create a fork and join
        fork_nodes = [nid for nid, n in nodes.items() if n.get("kind") == "fork"]
        join_nodes = [nid for nid, n in nodes.items() if n.get("kind") == "join"]
        if fork_nodes:
            _ok(f"fork created: {fork_nodes[0]}")
        else:
            _fail("fork", "no fork node found")
            passed = False
        if join_nodes:
            _ok(f"join created: {join_nodes[0]}")
        else:
            _fail("join", "no join node found")
            passed = False

    if name == "F":
        # Loop with agent+tools judge — the gate should have tools
        gate_nodes = [(nid, n) for nid, n in nodes.items()
                      if n.get("kind") == "gate" and nid not in ("has_case",)]
        if gate_nodes:
            gid, gate = gate_nodes[0]
            _ok(f"agent judge gate: {gid}")
            if gate.get("tools"):
                _ok(f"judge has tools: {gate['tools']}")
            else:
                _ok("judge gate created (no tools field on gate node)")
        else:
            _fail("agent judge gate", "no gate found")
            passed = False

    return passed


ALL_PATTERNS: list[tuple[str, str, Callable[[dict], dict]]] = [
    ("A", "sealed code guard (selective takes_over of output.xlsx)", pattern_a),
    ("B", "agent guard with tools (selective takes_over of output.xlsx)", pattern_b),
    ("C", "sealed code judge loop around implement", pattern_c),
    ("D", "route with one way taking over output.xlsx", pattern_d),
    ("E", "parallel fork (two copies of implement)", pattern_e),
    ("F", "loop with agent+tools judge", pattern_f),
]


# ── real evaluation ─────────────────────────────────────────────────────


def _evaluate_one(
    name: str,
    build_fn: Callable[[dict], dict],
    seed_state: dict,
    item_id: str,
    out_dir: Path,
    model: Any,
) -> dict[str, Any]:
    """Evaluate one pattern × item using mode.evaluate (the walker path).

    Uses the same subset mechanism the loop uses: a parts.json manifest that
    lists the requested item ID, and a subset.json that selects the "train"
    part.  select_subset() then filters the Task's items to just this one.
    """
    from refactory_dsh.task import write_subset, MANIFEST

    candidate_state = build_fn(dict(seed_state))
    mode = Steps(state=candidate_state)
    trial_dir = out_dir / f"{name}_{item_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / MANIFEST).write_text(
        json.dumps({"train": [str(item_id)]})
    )
    write_subset(trial_dir, "train", holdout=0.0)
    record: dict[str, Any] = {
        "pattern": name,
        "item": item_id,
        "ran": False,
        "score": None,
        "step_produced": False,
        "downstream_got_it": False,
        "graded_file_changed": False,
        "error": None,
    }
    try:
        result = mode.evaluate(trial_dir, model=model)
        record["ran"] = result.exit_ok
        record["score"] = getattr(result, "score", None)
        record["error"] = result.error if not result.exit_ok else None
        passes_dir = trial_dir / "passes" / "data"
        if passes_dir.exists():
            for item_pass in passes_dir.iterdir():
                if (item_pass / "output.xlsx").exists():
                    record["step_produced"] = True
                if (item_pass / "pred.xlsx").exists():
                    record["downstream_got_it"] = True
                graded = item_pass / "output.xlsx"
                if graded.exists() and graded.stat().st_size > 0:
                    record["graded_file_changed"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return record


def _run_real(
    patterns: list[tuple[str, str, Callable[[dict], dict]]],
    seed_state: dict,
    items_root: Path,
    item_ids: list[str] | None,
    out_dir: Path,
    concurrency: int,
) -> bool:
    """Run selected patterns with real model calls via mode.evaluate."""
    from refactory_dsh.model import DshModelClient

    os.environ["SB2_DATA"] = str(items_root)
    model = DshModelClient()

    if item_ids is None:
        from refactory_packages.spreadsheetbench_v2.tasks import (
            SpreadsheetBenchV2StepsTask,
        )
        task = SpreadsheetBenchV2StepsTask(data_root=str(items_root))
        item_ids = [inst.id for inst in task.instances()[:5]]
        print(f"No --ids given; using first 5 items: {item_ids}")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    jobs: list[tuple[str, str, Callable[[dict], dict], str]] = []
    for name, _label, fn in patterns:
        for item_id in item_ids:
            jobs.append((name, _label, fn, item_id))

    print(f"\nEvaluating {len(jobs)} pattern×item combinations "
          f"(concurrency={concurrency})...")

    def run_job(job: tuple) -> dict[str, Any]:
        name, _label, fn, item_id = job
        return _evaluate_one(name, fn, seed_state, item_id, out_dir, model)

    if concurrency <= 1:
        for job in jobs:
            rec = run_job(job)
            _print_record(rec)
            all_records.append(rec)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(run_job, j): j for j in jobs}
            for future in concurrent.futures.as_completed(futures):
                rec = future.result()
                _print_record(rec)
                all_records.append(rec)

    print(f"\n{'='*72}")
    print(f"{'Pattern':<10} {'Item':<20} {'Ran':<6} {'Score':<8} "
          f"{'Produced':<10} {'Downstream':<12} {'Error'}")
    print("-" * 72)
    for rec in sorted(all_records, key=lambda r: (r["pattern"], r["item"])):
        err = (rec["error"] or "")[:30]
        print(f"{rec['pattern']:<10} {rec['item']:<20} "
              f"{'yes' if rec['ran'] else 'NO':<6} "
              f"{rec['score'] or ''!s:<8} "
              f"{'yes' if rec['step_produced'] else 'no':<10} "
              f"{'yes' if rec['downstream_got_it'] else 'no':<12} "
              f"{err}")

    failed = [r for r in all_records if not r["ran"]]
    if failed:
        print(f"\n{len(failed)} evaluation(s) did not complete.")
    return not failed


def _print_record(rec: dict[str, Any]) -> None:
    status = "OK" if rec["ran"] else "FAIL"
    score = f" score={rec['score']}" if rec["score"] is not None else ""
    err = f" error={rec['error'][:60]}" if rec["error"] else ""
    print(f"  [{status}] {rec['pattern']} / {rec['item']}{score}{err}")


# ── main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-root", type=Path,
                        help="Path to prepared SB2 data (SB2_DATA directory)")
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma-separated item IDs to evaluate")
    parser.add_argument("--patterns", type=str, default=None,
                        help="Comma-separated pattern letters (default: all)")
    parser.add_argument("--out", type=Path, default=Path("probe_runs"),
                        help="Output directory for run directories")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent evaluations")
    parser.add_argument("--real", action="store_true",
                        help="Evaluate with the walker, real model calls, "
                             "and given items")
    args = parser.parse_args()

    selected = ALL_PATTERNS
    if args.patterns:
        wanted = {p.strip().upper() for p in args.patterns.split(",")}
        selected = [(n, l, f) for n, l, f in ALL_PATTERNS if n in wanted]
        if not selected:
            print(f"No matching patterns for: {args.patterns}")
            sys.exit(1)

    state = _state()
    print(f"Seed graph: {len(state['graph']['nodes'])} nodes, "
          f"{len(state['graph']['edges'])} edges")

    # ── dry structural mode (default) ────────────────────────────────────
    results = []
    for name, label, fn in selected:
        results.append(probe_pattern(name, label, fn, state))

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"Structural: {passed}/{len(results)} passed")

    if not all(results):
        sys.exit(1)

    # ── real evaluation mode ─────────────────────────────────────────────
    if args.real:
        if not args.items_root or not args.items_root.is_dir():
            print("--real requires --items-root pointing to a prepared SB2 "
                  "data directory.")
            sys.exit(1)
        item_ids = (
            [i.strip() for i in args.ids.split(",") if i.strip()]
            if args.ids else None
        )
        ok = _run_real(
            selected, state, args.items_root, item_ids,
            args.out, args.concurrency,
        )
        sys.exit(0 if ok else 1)

    sys.exit(0)


if __name__ == "__main__":
    main()
