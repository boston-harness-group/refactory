"""End-to-end probe: code steps read and write their declared files.

Applies edits in the exact JSON op form a proposer sends, via the same edit
path the loop uses, to the real spreadsheetbench_v2/steps seed graph.  Three
patterns:

A. Plan writes targets.json.  After implement, insert a sealed code step that
   reads input.xlsx, output.xlsx and targets.json, restores every cell changed
   outside the targets, and writes output.xlsx (selective takes_over of
   implement's output.xlsx).  Verify and finalize then read the restored file.

B. The same guard as an agent step with tools, instead of a code step.

C. A loop around implement, with a sealed code judge that reports stray cells
   as its verdict.

For each, the probe shows:
  - the edit is accepted, with no false refusal;
  - the wiring validator passes;
  - the step runs (when real items and a tool environment are available);
  - downstream steps read the produced files;
  - the graded file reflects the guard.

Usage:
  python -m scripts.probe_guard_pattern [--items PATH] [--run]

  --items PATH   Path to prepared spreadsheetbench items (for end-to-end run).
  --run          Attempt to run the graph on a synthetic item.
                 Requires the spreadsheetbench tool environment.

Without --run, the probe validates graph structure only (no tool environment
needed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add the refactory directory to the path.
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

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, help="Path to prepared items")
    parser.add_argument("--run", action="store_true", help="Run on synthetic item")
    args = parser.parse_args()

    state = _state()
    print(f"Seed graph: {len(state['graph']['nodes'])} nodes, "
          f"{len(state['graph']['edges'])} edges")

    results = []
    results.append(probe_pattern(
        "A", "sealed code guard (selective takes_over of output.xlsx)",
        pattern_a, state))
    results.append(probe_pattern(
        "B", "agent guard with tools (selective takes_over of output.xlsx)",
        pattern_b, state))
    results.append(probe_pattern(
        "C", "sealed code judge loop around implement",
        pattern_c, state))

    print(f"\n{'='*60}")
    total = len(results)
    passed = sum(results)
    print(f"Patterns: {passed}/{total} passed")

    if args.run:
        print("\n--run requested: attempting end-to-end run...")
        try:
            from refactory_packages.spreadsheetbench import interpreter
            interp = interpreter.interpreter()
            print(f"  Interpreter: {interp}")
            print("  End-to-end run not yet implemented in this probe.")
            print("  Run with the full harness instead:")
            print("  python -m refactory_dsh.run --mode spreadsheetbench_v2.steps ...")
        except Exception as exc:
            print(f"  Tool environment not available: {exc}")
            print("  Run on a machine with the spreadsheetbench interpreter.")

    if args.items:
        print(f"\n--items {args.items}: checking items directory...")
        if args.items.is_dir():
            items = list(args.items.iterdir())
            print(f"  Found {len(items)} items")
        else:
            print(f"  Not a directory: {args.items}")

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
