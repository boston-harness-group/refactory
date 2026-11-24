"""The proposal's input reports and never prescribes, and says each thing once.

- Analysts and their merges say what happens, where, on which items, with quotes and
  impact; no answer format, parsed finding or rendered line carries a change to make
  (``change``) or a thing to keep (``preserve``).
- The memory note is a record of facts: an answer's advice sections are dropped
  (`memory.without_advice`) and an answer that is only advice leaves no note.
- The request's reference lists every operation of the catalogue and every node field
  exactly once, so a dropped or doubled one fails here.
"""

from __future__ import annotations

import re

import pytest

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.node_schema import SAID_ELSEWHERE, _public_fields
from refactory_dsh.optim import analysts, memory

from test_request_order import _ctx, _many
from test_structure_first_request import _ctx3

ADVICE_WORDS = re.compile(r'"change"|"preserve"|what change|must preserve|must keep|'
                          r"what to keep|would fix it", re.IGNORECASE)


# ── analysts report ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    analysts.FAILURE_SYSTEM, analysts.SUCCESS_SYSTEM, analysts.FAILURE_ANSWER,
    analysts.SUCCESS_ANSWER, analysts.MERGE_FAILURE_SYSTEM, analysts.MERGE_SUCCESS_SYSTEM,
    analysts.MERGE_FINAL_SYSTEM, analysts.MERGE_ANSWER])
def test_no_analyst_or_merge_prompt_asks_for_a_change_or_a_thing_to_keep(text):
    assert not ADVICE_WORDS.search(text), ADVICE_WORDS.search(text).group(0)


@pytest.mark.parametrize("text", [analysts.FAILURE_SYSTEM, analysts.SUCCESS_SYSTEM,
                                  analysts.MERGE_FINAL_SYSTEM])
def test_the_analysts_are_told_to_say_what_happens_not_what_to_do(text):
    assert "Do not propose changes: say what happens, not what to do." in text


def test_a_finding_that_carries_advice_keeps_only_its_report():
    raw = {"finding": "solve drops the units", "items": ["a", "b"], "impact": "high",
           "where": {"nodes": ["solve"]}, "evidence": ["12 m"],
           "change": "ADVICE-CHANGE", "preserve": "ADVICE-KEEP"}
    found = analysts.finding(raw, "failure", {"a", "b"})
    assert found is not None and "change" not in found and "preserve" not in found
    assert found["items"] == ["a", "b"] and found["evidence"] == ["12 m"]
    merged = analysts._for_merge([{**found, "change": "ADVICE-CHANGE"}])
    assert "change" not in merged[0]
    line = analysts.finding_line(1, {**found, "change": "ADVICE-CHANGE",
                                     "preserve": "ADVICE-KEEP"})
    assert "ADVICE" not in line and "solve drops the units" in line and "12 m" in line
    section = "\n".join(analysts.section_lines(
        {"findings": [{**found, "change": "ADVICE-CHANGE"}], "batches": [], "group_size": 8}))
    assert "ADVICE" not in section and "change would fix" not in section


def test_an_analyst_request_shows_no_operations_to_choose_from():
    records = _many(4, 2, size=200)
    ctx = _ctx(records)
    pipeline = analysts.pipeline_text(ctx)
    for kind in analysts.KINDS:
        group = [r for r in records if bool(r["passed"]) == (kind == "success")]
        text = analysts.analyst_batch(ctx, kind, 1, 1, group, pipeline)[0]
        assert "The changes the optimizer can make" not in text
        assert "| set_knob |" not in text and "`insert` (" not in text


# ── memory keeps facts ────────────────────────────────────────────────────────


NOTE = """## Epoch 3 record

### Judge redesign (epoch 3)
Changed the judge; 2 fixed / 4 broken of 66. It said proceed on every item.

**Key lesson**: judges in a loop are unreliable here.

### Persistent failures
1. Tax rate sign flipped on 6 items (sb2-1, sb2-2).

### What to try next — prioritized
1. Invest in the solver instead.
   - a sub-point

### Regression risks
- Guard the skill knob.

### Meta-lesson on change design
- The LLM-as-judge pattern is unreliable for this task.

### What works
Explicit tasks score 100% (sb2-3)."""


def test_the_record_drops_every_advice_section_and_keeps_the_facts():
    kept = memory.without_advice(NOTE)
    for gone in ("What to try next", "Invest in the solver", "a sub-point",
                 "Regression risks", "Guard the skill knob", "Meta-lesson",
                 "LLM-as-judge", "Key lesson"):
        assert gone not in kept, gone
    for fact in ("Judge redesign", "2 fixed / 4 broken of 66", "Tax rate sign flipped",
                 "Explicit tasks score 100%"):
        assert fact in kept, fact


def test_a_note_that_is_only_advice_is_no_note():
    assert memory.parse_note({"memory": "### Next steps\n- try X\n### Recommendations\n- Y"}) \
        is None
    parsed = memory.parse_note({"memory": NOTE, "reasoning": "r"})
    assert parsed and "What to try next" not in parsed["text"]


@pytest.mark.parametrize("text", [memory.MEMORY_SYSTEM, memory.SYNTHESIS_SYSTEM])
def test_the_memory_prompt_asks_for_facts_and_forbids_advice(text):
    flat = " ".join(text.split())
    assert "Write nothing about what to do next" in flat
    assert "Make no claim about what can or cannot work" in flat
    assert "What to try next" not in text and "regression risks future" not in flat


def test_the_proposal_shows_the_record_without_repeating_the_tally():
    text = _ctx3(memory={"text": "RECORD", "epoch": 2, "tally": ["TALLY-LINE"]}).render()
    section = text.split("## Optimizer memory", 1)[1]
    assert "RECORD" in section and "TALLY-LINE" not in text
    assert "Use it to improve" not in section


# ── the request says each thing once ──────────────────────────────────────────


@pytest.fixture(scope="module")
def request_text() -> str:
    return _ctx(_many(4, 2, size=200)).render()


def _op_entries(text: str, op: str) -> int:
    """How often ``op`` is described: a table row of the written-out operations or a
    structural operator's line."""
    return (len(re.findall(rf"^\| {re.escape(op)} \|", text, re.MULTILINE))
            + len(re.findall(rf"^- `{re.escape(op)}` \(", text, re.MULTILINE)))


@pytest.mark.parametrize("op", sorted({c["op"] for c in nn_edits.catalogue("broad")}))
def test_the_reference_describes_every_operation_once(request_text, op):
    reference = request_text.split("## Reference:", 1)[1]
    assert _op_entries(request_text, op) == 1, op
    assert _op_entries(reference, op) == 1, op


# `route` is said in the operators' route line, only when the run has routes to offer.
@pytest.mark.parametrize("field", [f.name for f in _public_fields()
                                   if f.name not in SAID_ELSEWHERE])
def test_the_reference_describes_every_node_field_once(request_text, field):
    assert len(re.findall(rf"^\| {re.escape(field)} \|", request_text, re.MULTILINE)) == 1


def test_the_request_has_one_reference_and_no_prescriptive_or_repeated_sections(request_text):
    for gone in ("Where the pipeline fails", "What each step carries",
                 "Compositions built only from these operators", "Structural operators"):
        assert gone not in request_text, gone
    top = [line for line in request_text.splitlines() if line.startswith("## ")]
    assert "## How nodes compose" not in top and "## The operations you may use" not in top
    reference = request_text.split("## Reference:", 1)[1]
    for part in ("### The operations you may use", "### How nodes compose",
                 "### Node schemas", "### The rules the validator checks"):
        assert reference.count(part) == 1, part
    assert request_text.count("Compute policy:") <= 1


def test_the_last_edit_shows_its_diff_and_counts_and_one_line_per_changed_item():
    parent = _many(3, 3, size=200)
    child = [{**r, "passed": not r["passed"], "score": float(not r["passed"])} for r in parent]
    edit = [{"op": "set_knob", "name": "solve_task", "value": "NEW-TEXT"}]
    text = _ctx(parent, edit_pairs=[{"child": "c1", "parent": "p0", "edit": edit,
                                     "parent_records": parent, "child_records": child,
                                     "prediction": "f0 passes", "steps": []}]).render()
    section = text.split("## What the last edit changed", 1)[1].split("\n## ", 1)[0]
    assert "NEW-TEXT" in section and "3 fixed" in section and "3 broken" in section
    assert "Its prediction: f0 passes" in section
    lines = [line for line in section.splitlines() if line.startswith("- Item ")]
    assert sorted(line.split()[2] for line in lines) == sorted(r["id"] for r in parent)
    assert "#### Item" not in section and "Parent's reply" not in section
