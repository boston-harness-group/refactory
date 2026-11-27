"""The code-run pre-check (``E-code-run``) skips sealed script steps.

A sealed code step runs in the Task's tool environment with no import
allowlist.  ``code_dry_run`` must skip it rather than running it through
``snippet.execute``, which applies the V18 allowlist and would refuse any
import outside it (openpyxl, pandas, …).

The companion probe pattern H in ``scripts/probe_guard_pattern`` confirms
the same with the real steps graph.  These tests use the real edit path and
``code_dry_run`` directly so they catch regressions without a lab run.
"""
from __future__ import annotations

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim.checks import code_dry_run
from refactory_packages.spreadsheetbench_v2.steps_mode import Steps


def _steps_state() -> dict:
    return Steps().state_dict()


OPENPYXL_GUARD = """\
import openpyxl, json

def main(inp):
    wb = openpyxl.load_workbook("output.xlsx")
    return {"facts": {"sheets": len(wb.sheetnames)}}
"""


def test_code_dry_run_skips_sealed_script_step():
    """A sealed code step importing openpyxl is skipped (not refused) by
    ``code_dry_run``.  Without the fix, ``snippet.execute`` applies V18 and
    refuses ``import openpyxl``."""
    state = _steps_state()
    result = nn_edits.apply(
        {"op": "insert", "after": "implement", "kind": "code",
         "text": OPENPYXL_GUARD,
         "id": "guard", "sealed": True,
         "takes_over": ["implement", "output.xlsx"]},
        state)

    def _fake_recorded(_parent_id: str) -> dict:
        return {}

    refusals, notes = code_dry_run(
        "parent", state, result, ["guard"], recorded=_fake_recorded)
    assert not refusals, f"sealed step was refused: {refusals}"
    sealed_notes = [n for n in notes if "sealed script step" in n]
    assert sealed_notes, f"expected a sealed skip note, got: {notes}"


def test_code_dry_run_skips_sealed_with_no_recorded_callback():
    """When recorded is None (no run directory at all), the whole dry run is
    skipped before reaching per-node checks.  The sealed step is not refused."""
    state = _steps_state()
    result = nn_edits.apply(
        {"op": "insert", "after": "implement", "kind": "code",
         "text": OPENPYXL_GUARD,
         "id": "guard", "sealed": True,
         "takes_over": ["implement", "output.xlsx"]},
        state)

    refusals, notes = code_dry_run(
        "parent", state, result, ["guard"], recorded=None)
    assert not refusals, f"sealed step was refused: {refusals}"
    assert notes, "expected a skip note"


def test_unsealed_code_step_importing_openpyxl_would_be_caught():
    """Sanity check: an UNSEALED code step importing openpyxl would hit V18.
    This is not tested through code_dry_run (needs a real sandbox), but through
    the validator's static check."""
    from refactory_dsh.nn.validate import validate_state

    state = _steps_state()
    result = nn_edits.apply(
        {"op": "insert", "after": "implement", "kind": "code",
         "text": OPENPYXL_GUARD,
         "id": "guard",
         "takes_over": ["implement", "output.xlsx"]},
        state)

    issues = validate_state(result)
    v18 = [i for i in issues if "V18" in str(i) or "import" in str(i).lower()]
    assert v18, f"unsealed openpyxl should trigger V18, got: {issues}"
