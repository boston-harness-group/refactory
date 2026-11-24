"""The proposal's input reports and never prescribes, and says each thing once.

- Analysts and their merges say what happens, where, on which items, with quotes and
  impact; no answer format, parsed finding or rendered line carries a change to make
  (``change``) or a thing to keep (``preserve``).
- The memory note is a record of facts: the prompt asks for each fact the record holds
  and forbids advice; the note is kept as written.
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
                                  analysts.MERGE_FAILURE_SYSTEM, analysts.MERGE_SUCCESS_SYSTEM,
                                  analysts.MERGE_FINAL_SYSTEM])
def test_the_analysts_are_told_to_say_what_happens_not_what_to_do(text):
    assert "Do not propose changes: say what happens, not what to do." in text


@pytest.mark.parametrize("text", [analysts.MERGE_FAILURE_SYSTEM, analysts.MERGE_SUCCESS_SYSTEM,
                                  analysts.MERGE_FINAL_SYSTEM, analysts.FAILURE_SYSTEM])
def test_no_merge_or_analyst_prompt_ranks_by_priority_or_lists_fixes(text):
    flat = " ".join(text.split()).lower()
    for word in ("priority", "take priority", "work that should run", "is absent",
                 "keep the failure one"):
        assert word not in flat, word


def test_the_final_merge_keeps_a_success_finding_beside_a_failure_finding():
    flat = " ".join(analysts.MERGE_FINAL_SYSTEM.split())
    assert "Keep every finding of both lists" in flat and "both stay" in flat


def test_every_merged_finding_is_shown_whole():
    found = [{"source": "failure", "finding": f"FINDING-{k} " + "F" * 3000,
              "items": [f"i{j}" for j in range(30)], "evidence": ["q" * 500]}
             for k in range(25)]
    text = "\n".join(analysts.section_lines({"findings": found, "batches": []}))
    assert all(f"FINDING-{k} " + "F" * 3000 in text for k in range(25))
    assert "i29" in text and "not shown" not in text and "[cut:" not in text


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


#: What the record must hold, as the memory prompt asks for it.
RECORD_FACTS = (
    "Each change tried, this epoch and as the previous record holds earlier ones",
    "whether it was kept, not kept or refused, and why",
    "which train items it fixed and which it broke",
    "What keeps failing: the items, and where in the pipeline the traces show it starts.",
    "At most a few short observations of what the traces show happened",
)
#: What the record must not hold.
RECORD_FORBIDS = (
    "Write nothing about what to do next: no recommendations, next steps, priorities, "
    "lessons, meta-lessons, principles or regression risks to guard against",
    "Make no claim about what can or cannot work, be built or be edited, and do not judge "
    "a kind of change in general.",
    "Do not output instructions for the pipeline's own steps",
)


@pytest.mark.parametrize("system", [memory.MEMORY_SYSTEM, memory.SYNTHESIS_SYSTEM])
@pytest.mark.parametrize("rule", [*RECORD_FACTS, *RECORD_FORBIDS])
def test_the_memory_prompt_asks_for_each_fact_and_forbids_advice(system, rule):
    assert rule in " ".join(system.split()), rule


def test_a_note_is_kept_as_written():
    """No keyword filter: the prompt forbids advice, and the note is kept whole."""
    note = "### Risk-check step inserted (epoch 3)\n- fixed 3, broke 1"
    assert memory.parse_note({"memory": note, "reasoning": "r"})["text"] == note


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


def _flipped(records, prefix):
    return [{**r, "passed": not r["passed"], "score": float(not r["passed"]),
             "exchanges": [{**e, "response": f"{prefix}-{r['id']}-{e['node']} "
                                             + "y" * 6000} for e in r["exchanges"]]}
            for r in records]


def test_the_last_edit_shows_its_diff_counts_and_how_it_moved_two_items_replies():
    """Every changed item in one line; for the first broken and the first fixed item, the
    replies that moved, the parent's and the child's, each within the fixed cap."""
    from refactory_dsh.optim import base

    parent = _many(3, 3, size=200)
    child = _flipped(parent, "CHILD")
    edit = [{"op": "set_knob", "name": "solve_task", "value": "NEW-TEXT"}]
    text = _ctx(parent, edit_pairs=[{"child": "c1", "parent": "p0", "edit": edit,
                                     "parent_records": parent, "child_records": child,
                                     "prediction": "f0 passes", "steps": []}]).render()
    section = text.split("## What the last edit changed", 1)[1].split("\n## ", 1)[0]
    assert "NEW-TEXT" in section and "3 fixed" in section and "3 broken" in section
    assert "Its prediction: f0 passes" in section
    lines = [line for line in section.splitlines() if line.startswith("- Item ")]
    assert sorted(line.split()[2] for line in lines) == sorted(r["id"] for r in parent)
    blocks = [line for line in section.splitlines() if line.startswith("#### Replies")]
    assert blocks == ["#### Replies that moved on item p0 (broken, child c1)",
                      "#### Replies that moved on item f0 (fixed, child c1)"]
    for item in ("p0", "f0"):
        assert f"CHILD-{item}-draft" in section and f"CHILD-{item}-final" in section
    assert "CHILD-p1-draft" not in text, "only the first of each kind"
    # Each 6,000-character reply is cut to the cap (4,000), head and tail, the cut said.
    assert base.EDIT_REPLY_CHARS == 4000
    assert "y" * 4001 not in section and "characters omitted" in section
    # A parent reply "Trajectories" shows whole is named by where it is, never repeated.
    assert "Parent's reply: as under \"Trajectories of the candidate being edited\"" in section
    assert "Parent's reply:\n" not in section or "Parent's reply: as" in section
