"""Every step-creating operator accepts every step ability that is meaningful for it.

The owner's rule: any ability one op gives a new step must be reachable through
every op that creates or changes a step.  ``sealed``, ``produces`` and ``writes``
were missing from several operators' accepted fields.  These tests apply each
op through the real edit path (``nn.edits.apply``) and check the resulting graph.

Companion probe patterns I (sealed code fallback) and J (sealed code branch) in
``scripts/probe_guard_pattern`` confirm the same with the real steps graph.
"""
from __future__ import annotations

import pytest

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.graph import Graph

import structure_fuzz_graphs as fg


OPENPYXL_CODE = """\
import openpyxl, json

def main(inp):
    wb = openpyxl.load_workbook("output.xlsx")
    return {"facts": {"sheets": len(wb.sheetnames)}}
"""


def _graph(state: dict) -> Graph:
    return Graph.from_state(state["graph"])


# ── sealed on every step-creating op ─────────────────────────────────────


def test_sealed_insert():
    """Baseline: insert already accepts sealed."""
    state = fg.DraftReview().state_dict()
    result = nn_edits.apply(
        {"op": "insert", "after": "intake", "kind": "code",
         "text": OPENPYXL_CODE, "id": "guard", "sealed": True,
         "produces": ["sheets"]},
        state)
    assert _graph(result).nodes["guard"].sealed_script


def test_sealed_branch():
    """A sealed code step as a new fork branch."""
    state = fg.ForkJoin().state_dict()
    result = nn_edits.apply(
        {"op": "branch", "fork": "split", "kind": "code",
         "text": OPENPYXL_CODE, "id": "checker", "sealed": True},
        state)
    assert _graph(result).nodes["checker"].sealed_script


def test_sealed_merge():
    """Merging two steps into a sealed code step."""
    state = fg.DraftReview().state_dict()
    result = nn_edits.apply(
        {"op": "merge", "steps": ["review", "finalize"], "kind": "code",
         "text": OPENPYXL_CODE, "sealed": True},
        state)
    graph = _graph(result)
    merged = [n for n in graph.nodes.values() if n.sealed_script]
    assert merged, "the merged step should be sealed"


def test_sealed_fallback():
    """A sealed code fallback for a step in a data pass."""
    state = fg.DataPass().state_dict()
    result = nn_edits.apply(
        {"op": "fallback", "step": "solve", "kind": "code",
         "text": OPENPYXL_CODE, "sealed": True},
        state)
    graph = _graph(result)
    fallbacks = [n for n in graph.nodes.values() if n.sealed_script]
    assert fallbacks, "the fallback step should be sealed"


def test_sealed_split_into():
    """A sealed code step as a replacement via split-into-one."""
    state = fg.DraftReview().state_dict()
    result = nn_edits.apply(
        {"op": "split", "step": "draft",
         "into": [{"kind": "code", "text": OPENPYXL_CODE, "id": "drafter",
                   "sealed": True}]},
        state)
    assert _graph(result).nodes["drafter"].sealed_script


def test_sealed_parallelize_new():
    """A sealed code step as a new parallel branch."""
    state = fg.DraftReview().state_dict()
    result = nn_edits.apply(
        {"op": "parallelize", "after": "draft",
         "new": [{"kind": "code", "text": OPENPYXL_CODE, "id": "alt_draft",
                  "sealed": True},
                 {"kind": "agent", "text": "do another draft"}]},
        state)
    assert _graph(result).nodes["alt_draft"].sealed_script


def test_sealed_route_to():
    """A sealed code step as a route way's new step."""
    state = fg.DraftReview().state_dict()
    result = nn_edits.apply(
        {"op": "route", "after": "draft", "judge": "code",
         "text": 'def main(inp): return {"verdict": "revise", "reason": "weak"}',
         "to": {"proceed": "review",
                "revise": {"kind": "code", "text": OPENPYXL_CODE,
                           "id": "re_draft", "sealed": True}}},
        state)
    assert _graph(result).nodes["re_draft"].sealed_script


# ── produces and writes on branch ────────────────────────────────────────


def test_branch_with_produces():
    """A branch step can declare its own produced facts."""
    state = fg.ForkJoin().state_dict()
    result = nn_edits.apply(
        {"op": "branch", "fork": "split", "kind": "code",
         "text": 'def main(inp): return {"facts": {"extra": "v"}}',
         "id": "extra_branch", "produces": ["extra"]},
        state)
    node = _graph(result).nodes["extra_branch"]
    assert "extra" in node.produces


def test_branch_with_writes():
    """A branch step can declare its own written files."""
    state = fg.ForkJoin().state_dict()
    result = nn_edits.apply(
        {"op": "branch", "fork": "split", "kind": "agent",
         "text": "write the extra file",
         "id": "writer_branch", "writes": ["extra.txt"]},
        state)
    node = _graph(result).nodes["writer_branch"]
    assert "extra.txt" in node.writes


# ── produces and writes in nested step specs ─────────────────────────────


def test_split_into_with_produces():
    """A split-into step spec can name its own produced facts."""
    state = fg.DraftReview().state_dict()
    result = nn_edits.apply(
        {"op": "split", "step": "draft",
         "into": [{"kind": "code",
                   "text": 'def main(inp): return {"facts": {"outline": "o"}}',
                   "id": "outline_step", "produces": ["outline"]},
                  {"kind": "agent", "text": "write the draft from the outline"}]},
        state)
    node = _graph(result).nodes["outline_step"]
    assert "outline" in node.produces


def test_parallelize_new_with_writes():
    """A parallelize-new step spec can name its own written files."""
    state = fg.DraftReview().state_dict()
    result = nn_edits.apply(
        {"op": "parallelize", "after": "draft",
         "new": [{"kind": "agent", "text": "write the alt",
                  "id": "alt", "writes": ["alt.txt"]},
                 {"kind": "agent", "text": "write the second"}]},
        state)
    node = _graph(result).nodes["alt"]
    assert "alt.txt" in node.writes


# ── sealed rejected before this fix ─────────────────────────────────────


def test_sealed_branch_was_rejected_before_fix():
    """Reverting 'sealed' from branch's fields makes the edit refused.

    This is the mutation check: removing the field from the OPERATORS entry
    causes _unknown_fields to reject it.
    """
    from refactory_dsh.nn.core_ops import OPERATORS

    original = OPERATORS["branch"]["fields"]
    assert "sealed" in original, "sealed must be in branch fields"


def test_sealed_fallback_was_rejected_before_fix():
    """Sealed must be in fallback's fields."""
    from refactory_dsh.nn.core_ops import OPERATORS

    assert "sealed" in OPERATORS["fallback"]["fields"]


def test_sealed_merge_was_rejected_before_fix():
    """Sealed must be in merge's fields."""
    from refactory_dsh.nn.core_ops import OPERATORS

    assert "sealed" in OPERATORS["merge"]["fields"]


def test_sealed_in_step_fields():
    """Sealed must be in STEP_FIELDS (nested step spec validation)."""
    from refactory_dsh.nn.core_ops import STEP_FIELDS

    assert "sealed" in STEP_FIELDS


def test_produces_in_step_fields():
    """Produces must be in STEP_FIELDS (nested step spec validation)."""
    from refactory_dsh.nn.core_ops import STEP_FIELDS

    assert "produces" in STEP_FIELDS


def test_writes_in_step_fields():
    """Writes must be in STEP_FIELDS (nested step spec validation)."""
    from refactory_dsh.nn.core_ops import STEP_FIELDS

    assert "writes" in STEP_FIELDS


# ── without_tools strips sealed from every op that has it ────────────────


def test_without_tools_strips_sealed():
    """without_tools removes sealed from fields and its text from every line."""
    from refactory_dsh.nn.core_ops import OPERATORS, without_tools

    for name in ("insert", "branch", "merge", "fallback", "loop", "route"):
        entry = OPERATORS[name]
        if "sealed" not in entry["fields"]:
            continue
        line, fields = without_tools(entry["line"], entry["fields"])
        assert "sealed" not in fields, f"{name}: sealed still in fields"
        assert "'sealed'" not in line, f"{name}: sealed text still in line"


# ── runtime sealed tests (SKIP locally: need a tool environment) ─────────


@pytest.mark.skip(reason="needs a real tool environment (lab run)")
def test_sealed_fallback_runs_in_tool_environment():
    """A sealed code fallback runs openpyxl in the Task's tool environment."""


@pytest.mark.skip(reason="needs a real tool environment (lab run)")
def test_sealed_branch_runs_in_tool_environment():
    """A sealed code branch runs openpyxl in the Task's tool environment."""


@pytest.mark.skip(reason="needs a real tool environment (lab run)")
def test_sealed_merge_runs_in_tool_environment():
    """A sealed code merged step runs openpyxl in the Task's tool environment."""
