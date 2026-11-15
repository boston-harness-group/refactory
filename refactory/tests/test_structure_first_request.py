"""The proposal request puts the pipeline first and treats structure as an equal lever.

Evidence from twelve runs: analysts named structural causes in almost every epoch, yet
the proposer chose a structural edit once in 61 proposals, citing risk, complexity and
cost -- words our own request supplied. So the request now:

- opens with the pipeline (nodes, edges), a per-step failure map from the analysts, the
  operations and how nodes compose, and only then each step's settings (every knob's
  exact text still shown), the node reference and the validator's rules, introduced as
  fixable rather than as a wall of refusals;
- words its guidance kind-neutrally (no "few changes", and no sentence that raised the
  bar a new step has to clear);
- states the compute policy from the scoring settings (parsimony 0: extra steps and
  model calls per item are not penalised);
- says in plain words what the run has done with structure ("no structural change has
  been tried in this run yet"), not ``structural_yield: n/a``;
- carries the analysts' optional step attribution (first went wrong / replaced good work
  from an earlier step with worse / repeated other steps' output without adding to it) into a
  per-step table, for any graph shape (tool-using agents, gates and loops, forks, code
  ops), tolerating findings without it and malformed ones.

Domain-free: the fixtures are a generic pipeline and the request's new words pass the
presentation tests' denylist. Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from refactory_dsh.nn import edits
from refactory_dsh.optim import ProposalContext, analysts, feedback
from refactory_dsh.optim import base
from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer

from test_analysts import Client, _ctx, _rec
from test_proposal_presentation import DENYLIST

#: A generic pipeline: a tool-using agent that works in turns, a draft, a review, and a
#: finalising step.
NODES = [{"id": "explore", "kind": "agent", "knobs": ["explore__task"]},
         {"id": "draft", "kind": "agent", "knobs": ["draft__task"]},
         {"id": "review", "kind": "agent", "knobs": ["review__task"]},
         {"id": "finalize", "kind": "agent", "knobs": ["finalize__task"]}]
EDGES = [{"source": "explore", "target": "draft"}, {"source": "draft", "target": "review"},
         {"source": "review", "target": "finalize"}]
STATES = [{"id": n["id"], "kind": "agent", "writes": [f"{n['id']}.txt"], "produces": [n["id"]],
           "prompt_template": f"PROMPT-{n['id']} {{{{input}}}}",
           **({"tools": ["Read", "Bash"]} if n["id"] == "explore" else {}),
           "field_bindings": [["prompt_template", f"{n['id']}__task"]]} for n in NODES]
KNOBS = [{"name": f"{n['id']}__task", "value": f"PROMPT-{n['id']} {{{{input}}}}",
          "kind": "str", "frozen": False} for n in NODES]
FINDINGS = [
    {"source": "failure", "finding": "F-UNDONE", "items": ["i1", "i2"],
     "where": {"nodes": ["finalize"], "edges": ["review -> finalize"]},
     "steps": {"first_wrong": "review", "overrode": ["finalize"], "repeated": ["review"]}},
    {"source": "failure", "finding": "F-EARLY", "items": ["i2", "i3"],
     "where": {"nodes": ["draft"]}, "steps": {"first_wrong": "draft"}},
    {"source": "failure", "finding": "F-OLD", "items": ["i4"], "where": {"nodes": ["draft"]}},
    {"source": "success", "finding": "S-KEEP", "items": ["i5"], "where": {"nodes": ["finalize"]},
     "steps": {"first_wrong": "finalize"}}]
#: Another shape: a step in a loop behind a gate, then a fork of two branches and a join,
#: then a code op -- listed alphabetically, as a mode lists its nodes, not in flow order.
LOOPED = sorted([{"id": "attempt", "kind": "agent"}, {"id": "attempt_check", "kind": "gate"},
                 {"id": "fork_1", "kind": "fork"}, {"id": "branch_a", "kind": "agent"},
                 {"id": "branch_b", "kind": "agent"}, {"id": "join_1", "kind": "join"},
                 {"id": "tally", "kind": "op", "call": "refactory_dsh.snippet:run"},
                 {"id": "begin", "kind": "agent"}], key=lambda n: n["id"])
LOOPED_EDGES = [{"source": "begin", "target": "attempt"},
                {"source": "attempt", "target": "attempt_check"},
                {"source": "attempt_check", "target": "attempt", "condition": "reloop"},
                {"source": "attempt_check", "target": "fork_1", "condition": "proceed"},
                {"source": "fork_1", "target": "branch_b"}, {"source": "fork_1", "target": "branch_a"},
                {"source": "branch_a", "target": "join_1"}, {"source": "branch_b", "target": "join_1"},
                {"source": "join_1", "target": "tally"}]


def _ctx3(**fields: Any) -> ProposalContext:
    fields.setdefault("catalogue", edits.catalogue("broad"))
    fields.setdefault("knobs", KNOBS)
    return ProposalContext(mode_name="m", nodes=NODES, edges=EDGES, node_states=STATES,
                           **fields)


def _headings(text: str) -> list[str]:
    return [line[3:] for line in text.splitlines() if line.startswith("## ")]


def _section(text: str, heading: str) -> str:
    return text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]


# ── (a) the pipeline first, the settings after the operations ─────────────────


def test_the_request_leads_with_the_graph_and_shows_the_settings_after_the_operations():
    text = _ctx3(analysis={"findings": FINDINGS, "batches": []}, parsimony=0.0).render()
    heads = _headings(text)
    order = ["What the score measures", "The pipeline you may edit",
             "Where the pipeline fails, step by step", "The operations you may use",
             "How nodes compose", "Each step's settings",
             "Each step in full, and the node schema", "The rules the validator checks",
             "What the analysts found, ranked", "Answer"]
    assert [h for h in heads if h in order] == order
    assert "Parameters you may move" not in heads
    # Reorganised, not dropped: every knob's exact text is in the settings table.
    settings = _section(text, "Each step's settings")
    for knob in KNOBS:
        assert f"| {knob['name']} | {knob['value']} |" in settings
    # The pipeline section says both levers are open, before any knob is listed.
    pipeline = _section(text, "The pipeline you may edit")
    assert "Both are levers of the same standing" in pipeline
    assert "| draft | review | null |" in pipeline
    assert text.index("## The pipeline you may edit") < text.index("PROMPT-draft")


def test_a_settings_only_run_says_so_and_the_rules_are_introduced_as_fixable():
    text = _ctx3(catalogue=edits.catalogue("knob_only")).render()
    assert "This run permits changes to the steps' settings only" in _section(
        text, "The pipeline you may edit")
    rules = _section(text, "The rules the validator checks")
    assert "so it can be fixed in a later answer" in rules
    assert "A proposal that breaks one of these is refused" not in text
    assert "`V4`" in rules


# ── (b) kind-neutral wording ───────────────────────────────────────────────────


OLD_SENTENCES = (
    "Few well-argued changes, not many guesses",
    "a concise set of edits, only the ones",
    "tends to repeat their mistakes",
    "Prefer a structural change that changes how the work is done over one that delivers",
)


@pytest.mark.parametrize("preset", ["broad", "knob_only"])
def test_the_answer_rules_are_kind_neutral(preset):
    text = " ".join(_ctx3(catalogue=edits.catalogue(preset), max_edits=4,
                          structure_guidance=True).render().split())
    for sentence in OLD_SENTENCES:
        assert sentence not in text
    assert ("a setting, a prompt or a structural change, whose effect you can state" in text)
    assert "neither kind is the safer default" in text
    # Deliberate, evidence-grounded design of either kind: the failure addressed, the
    # change to what a step sees or how the work is done, a checkable prediction.
    assert ("the failure it addresses (the step where it starts), how it changes what a "
            "step sees or how the work is done, and a prediction that can be checked") in text


def test_the_structure_guidance_paragraph_is_kind_neutral():
    for sentence in OLD_SENTENCES:
        assert sentence not in feedback.STRUCTURE_GUIDANCE
    assert "neither is the safer default" in feedback.STRUCTURE_GUIDANCE
    assert "Design a structural change from the evidence" in feedback.STRUCTURE_GUIDANCE
    # Example-framed: critics and reformatting steps are legitimate, not argued against.
    assert "For example, a step can add" in feedback.STRUCTURE_GUIDANCE
    assert ("a critic of an earlier step's work, or a step that reformats or summarises for "
            "the next one") in feedback.STRUCTURE_GUIDANCE
    assert "rather than only the earlier steps" not in feedback.STRUCTURE_GUIDANCE
    composed = _section(_ctx3(structure_guidance=True).render(), "How nodes compose")
    assert feedback.STRUCTURE_GUIDANCE in composed


# ── (c) the compute policy follows the scoring settings ───────────────────────


def test_the_compute_policy_follows_parsimony():
    def said(**fields: Any) -> str:
        text = ProposalContext(**fields).render()
        return _section(text, "What the score measures") if "## What the score measures" in text else ""

    mean = said(parsimony=0.0, score_is_item_mean=True)
    assert ("Compute policy: this run does not penalise extra steps or model calls per item. "
            "Per-item cost is reported separately and is not part of the score") in mean
    other = said(parsimony=0.0)
    assert "Compute policy: the loop does not penalise extra steps or model calls" in other
    assert "cost counts only as far as the task's own score measures it" in other
    sized = said(parsimony=0.01, score_is_item_mean=True)
    assert "Compute policy" not in sized and "size cost of 0.01 per node" in sized
    assert said() == "", "unknown scoring settings say nothing"
    # The analyst calls carry the objective, not the policy line: only the request does.
    assert "Compute policy" not in base.objective_text(0.0, item_mean=True)


# ── structural evidence in plain words ─────────────────────────────────────────


def test_structural_evidence_is_said_in_plain_words():
    none = {"structural_candidates": 0, "knob_candidates": 4, "structural_refusals": 0,
            "structural_yield": "n/a", "accepted_ops": {"set_knob": 4}}
    text = _ctx3(traces=[{"epoch": 2, "knob_attribution": {"draft__task": {}},
                          "structural": none}]).render()
    evidence = _section(text, "Evidence from the traces")
    rows = [line for line in evidence.splitlines() if line.startswith("| ") and "---" not in line][1:]
    assert rows[0].startswith("| structural changes | No structural change has been tried in "
                              "this run yet; every candidate so far (4) keeps the starting "
                              "pipeline's structure. Operations in the candidates so far: "
                              "`set_knob` x4. |")
    assert "structural_yield" not in evidence and "n/a" not in evidence
    assert "risk" not in evidence.lower()
    refused = base.structural_text({"structural_candidates": 0, "knob_candidates": 2,
                                    "structural_refusals": 3})
    assert refused.startswith("No structural change has been evaluated in this run yet")
    assert "3 structural proposal(s) were refused by the validator before running" in refused
    tried = base.structural_text({"structural_candidates": 2, "knob_candidates": 5})
    assert tried == "2 candidate(s) so far made a structural change and 5 did not."
    # Malformed counts read as 0; never raise.
    assert base.structural_text({"structural_candidates": "x", "knob_candidates": None,
                                 "accepted_ops": "bad"}).startswith("No structural change")


# ── (d) step attribution: parsed, merged, aggregated, tolerated ───────────────


def test_step_attribution_is_parsed_and_malformed_parts_are_ignored():
    payload = {"findings": [
        {"finding": "A", "items": ["i1"],
         "steps": {"first_wrong": ["review", "draft"], "overrode": "finalize",
                   "repeated": ["review", 3, {"x": 1}, "", "review"]}},
        {"finding": "B", "items": ["i1"], "steps": "review went wrong"},
        {"finding": "C", "items": ["i1"], "steps": {"first_wrong": 7, "overrode": [None]}},
        {"finding": "D", "items": ["i1"]}]}
    found = analysts.parse_findings(payload, "failure", ["i1"])
    assert found[0]["steps"] == {"first_wrong": "review", "overrode": ["finalize"],
                                 "repeated": ["review"]}
    assert [f["steps"] for f in found[1:]] == [{}, {}, {}]
    # Merge input carries the attribution when there is one, and nothing when not.
    merged = analysts._for_merge(found)
    assert merged[0]["steps"]["overrode"] == ["finalize"] and "steps" not in merged[3]
    assert "steps" in analysts.FAILURE_ANSWER and "steps" in analysts.MERGE_ANSWER
    assert "first went wrong" in analysts.FAILURE_SYSTEM
    # Shape-free: loops, parallel branches and runs where it cannot be told are named.
    assert "its own earlier attempt in a loop" in analysts.FAILURE_SYSTEM
    assert "a parallel branch" in analysts.FAILURE_SYSTEM
    assert "Leave out what the items cannot show" in analysts.FAILURE_SYSTEM


def test_the_step_map_counts_items_once_per_step_and_role():
    rows = {r["step"]: r for r in analysts.step_map(FINDINGS, NODES)}
    assert list(rows) == ["draft", "review", "finalize"], "the pipeline's order"
    assert rows["draft"]["named"] == {"items": 3, "findings": 2}
    assert rows["draft"]["first_wrong"] == {"items": 2, "findings": 1}
    assert rows["review"]["first_wrong"] == {"items": 2, "findings": 1}
    assert rows["review"]["repeated"] == {"items": 2, "findings": 1}
    assert rows["finalize"]["overrode"] == {"items": 2, "findings": 1}
    # Success findings are not failures: S-KEEP's `finalize` is not counted.
    assert "first_wrong" not in rows["finalize"]
    # A step the pipeline does not have is ignored; odd entries never raise.
    odd = [{"source": "failure", "items": "i1", "where": "finalize",
            "steps": {"first_wrong": "ghost"}}, "junk", None,
           {"items": [["i"]], "where": {"nodes": [1, "draft"]}, "steps": []}]
    assert [r["step"] for r in analysts.step_map(odd, NODES)] == ["draft"]


def test_the_step_map_takes_any_graph_shape():
    """A gate, a step that repeats its own earlier attempt in a loop, parallel branches
    that echo each other, a code op: every node id can fill a role."""
    found = [{"source": "failure", "items": ["i1", "i2"],
              "where": {"nodes": ["attempt", "attempt_check"]},
              "steps": {"first_wrong": "attempt_check", "repeated": ["attempt"]}},
             {"source": "failure", "items": ["i3"], "where": {"nodes": ["branch_b"]},
              "steps": {"repeated": ["branch_a", "branch_b"], "overrode": ["tally"]}}]
    rows = {r["step"]: r for r in analysts.step_map(found, LOOPED, LOOPED_EDGES)}
    assert list(rows) == ["attempt", "attempt_check", "branch_a", "branch_b", "tally"]
    assert rows["attempt_check"]["first_wrong"] == {"items": 2, "findings": 1}
    assert rows["attempt"]["repeated"]["items"] == 2
    assert rows["tally"]["overrode"]["items"] == 1
    # Without the graph (nodes unknown), every id the findings give is kept, sorted.
    assert [r["step"] for r in analysts.step_map(found)] == sorted(rows)


def test_the_step_map_follows_the_flow_not_the_listed_order():
    """A mode lists its nodes sorted by id; the map follows the edges, loops included."""
    assert analysts.flow_order(LOOPED, LOOPED_EDGES) == [
        "begin", "attempt", "attempt_check", "fork_1", "branch_a", "branch_b", "join_1", "tally"]
    found = [{"source": "failure", "items": ["i1"],
              "where": {"nodes": ["tally", "begin", "attempt_check"]}}]
    assert [r["step"] for r in analysts.step_map(found, LOOPED, LOOPED_EDGES)] == [
        "begin", "attempt_check", "tally"]
    # In the request too: the edges order the rows, not the node list.
    text = ProposalContext(nodes=list(reversed(NODES)), edges=EDGES, catalogue=edits.catalogue(),
                           analysis={"findings": FINDINGS, "batches": []}).render()
    rows = [line.split(" | ")[0][2:] for line in _section(
        text, "Where the pipeline fails, step by step").splitlines()
        if line.startswith("| ") and not line.startswith(("| step", "|---"))]
    assert rows == ["draft", "review", "finalize"]
    # Malformed edges or nodes are skipped, never raise.
    assert analysts.flow_order([{"id": "a"}, "x", {"id": "b"}],
                               [None, {"source": "b", "target": "a"}, {"source": "z"}]) == ["b", "a"]


def _kinds(*ids: str, gates: tuple[str, ...] = (), forks: tuple[str, ...] = (),
           joins: tuple[str, ...] = ()) -> list[dict[str, str]]:
    kind = {**{g: "gate" for g in gates}, **{f: "fork" for f in forks}, **{j: "join" for j in joins}}
    return [{"id": i, "kind": kind.get(i, "agent")} for i in sorted(ids)]


def _wires(*pairs: str) -> list[dict[str, str]]:
    return [dict(zip(("source", "target"), pair.split(">"))) for pair in pairs]


@pytest.mark.parametrize("nodes,edges,flow", [
    # A gate whose id sorts before the step it guards, with no entry step.
    (_kinds("solve", "check", "z_end", gates=("check",)),
     _wires("solve>check", "check>solve", "check>z_end"), ["solve", "check", "z_end"]),
    # ... and with an entry step.
    (_kinds("prep", "solve", "check", "z_end", gates=("check",)),
     _wires("prep>solve", "solve>check", "check>solve", "check>z_end"),
     ["prep", "solve", "check", "z_end"]),
    (_kinds("x_start", "m_work", "b_gate", "z_end", gates=("b_gate",)),
     _wires("x_start>m_work", "m_work>b_gate", "b_gate>m_work", "b_gate>z_end"),
     ["x_start", "m_work", "b_gate", "z_end"]),
    # A fork inside a loop: the gate after the join sends the work back to the fork's step.
    (_kinds("a_entry", "work", "fork", "left", "right", "join", "a_gate", "out",
            gates=("a_gate",), forks=("fork",), joins=("join",)),
     _wires("a_entry>work", "work>fork", "fork>left", "fork>right", "left>join",
            "right>join", "join>a_gate", "a_gate>work", "a_gate>out"),
     ["a_entry", "work", "fork", "left", "right", "join", "a_gate", "out"]),
])
def test_a_loops_gate_comes_after_the_steps_it_guards_whatever_its_id(nodes, edges, flow):
    assert analysts.flow_order(nodes, edges) == flow


def test_an_item_cited_by_two_findings_for_the_same_step_and_role_counts_once():
    found = [{"source": "failure", "items": ["i1", "i2"], "where": {"nodes": ["review"]},
              "steps": {"first_wrong": "review"}},
             {"source": "failure", "items": ["i2", "i3"], "where": {"nodes": ["review"]},
              "steps": {"first_wrong": " `review` "}}]
    row = analysts.step_map(found, NODES)[0]
    assert row["step"] == "review"
    assert row["first_wrong"] == {"items": 3, "findings": 2}
    assert row["named"] == {"items": 3, "findings": 2}


CAUSE = ("the cause can lie in what the step is told (its instructions) or in what it is "
         "given and how the work is split between steps. For example: instructions on what "
         "to keep from earlier steps and when to depart from it; independent attempts that do "
         "not see each other's output; a vote or a check across attempts; a check that sends "
         "a step back with a reason; a deterministic code step. Choose by the evidence.")


def test_the_request_shows_the_step_map_and_the_cause_line_only_when_steps_interact():
    text = _ctx3(analysis={"findings": FINDINGS, "batches": []}, code_nodes=True).render()
    section = _section(text, "Where the pipeline fails, step by step")
    assert "| finalize | 2 (1) | - | 2 (1) | - |" in section
    assert "| review | - | 2 (1) | - | 2 (1) |" in section
    assert "(1 of 3 failure finding(s) say only where they start" in section
    assert "Counted over all 3 failure finding(s)." in section
    # Both causes stay equal; structures are examples beside an instruction fix.
    assert CAUSE in " ".join(section.split())
    assert "remedy" not in section and "not more instructions" not in section
    assert ("steps: first went wrong at: `review`; replaced good work from an earlier step "
            "with worse: `finalize`") in text
    # Old findings (no attribution) still map by where they start, with no cause line.
    old = [{k: v for k, v in f.items() if k != "steps"} for f in FINDINGS]
    plain = _section(_ctx3(analysis={"findings": old, "batches": []}).render(),
                     "Where the pipeline fails, step by step")
    assert "| draft | 3 (2) | - | - | - |" in plain and "the cause can lie" not in plain
    # A settings-only run is not handed structural examples.
    knob_only = _ctx3(analysis={"findings": FINDINGS, "batches": []},
                      catalogue=edits.catalogue("knob_only")).render()
    assert "the cause can lie" not in _section(knob_only, "Where the pipeline fails, step by step")
    # No analysis, or no failure finding that names a step: no section.
    assert "Where the pipeline fails" not in _ctx3().render()
    assert "Where the pipeline fails" not in _ctx3(
        analysis={"findings": [FINDINGS[3]], "batches": []}).render()


def test_only_first_wrong_gives_no_cause_line():
    found = [{"source": "failure", "items": ["i1"], "where": {"nodes": ["draft"]},
              "steps": {"first_wrong": "draft"}}]
    section = _section(_ctx3(analysis={"findings": found, "batches": []}).render(),
                       "Where the pipeline fails, step by step")
    assert "| draft | 1 (1) | 1 (1) | - | - |" in section
    assert "the cause can lie" not in section


def test_unknown_ids_do_not_count_as_attributed_and_success_lines_show_no_roles():
    found = [{"source": "failure", "items": ["i1"], "where": {"nodes": ["draft"]},
              "steps": {"first_wrong": "ghost", "overrode": ["phantom"]}},
             {"source": "failure", "items": ["i2"], "where": {"nodes": ["draft"]},
              "steps": {"first_wrong": "draft"}}]
    section = _section(_ctx3(analysis={"findings": found, "batches": []}).render(),
                       "Where the pipeline fails, step by step")
    assert "(1 of 2 failure finding(s) say only where they start" in section
    line = analysts.finding_line(1, {"source": "success", "finding": "S", "items": ["i5"],
                                     "steps": {"first_wrong": "finalize"}})
    assert "first went wrong" not in line and "steps:" not in line


def test_the_caption_says_when_the_map_counts_more_than_the_list_shows():
    many = [{"source": "failure", "finding": f"F{n}", "items": [f"i{n}"],
             "where": {"nodes": ["draft"]}} for n in range(analysts.SHOWN_FINDINGS + 2)]
    section = _section(_ctx3(analysis={"findings": many, "batches": []}).render(),
                       "Where the pipeline fails, step by step")
    assert (f"Counted over all {len(many)} failure finding(s); the list below shows the top "
            f"{analysts.SHOWN_FINDINGS} findings of either kind.") in section


def test_without_analysts_the_map_comes_from_the_readers_patterns():
    readers = [{"failure_patterns": [
                    {"pattern": "p1", "step": "review", "items": ["i1", "i2"]},
                    {"pattern": "p2", "step": "unclear", "items": ["i3"]},
                    {"pattern": "p3", "step": "`draft`", "items": ["i4"]}],
                "success_patterns": [{"pattern": "s", "step": "finalize", "items": ["i5"]}]},
               {"failure_patterns": [{"pattern": "p4", "step": "review", "items": ["i2", "i6"]},
                                     "junk", {"step": None}]}]
    text = _ctx3(reader_findings=readers).render()
    section = _section(text, "Where the pipeline fails, step by step")
    assert ("From the readers' failure patterns for this candidate (in full under \"What the "
            "readers found\")") in section
    # Readers give no roles: only the first column, no attribution line, and the patterns
    # that name no step ("unclear", none) are not counted.
    assert "| step | named in failure findings |\n" in section
    assert "| review | 3 (2) |\n" in section and "| draft | 1 (1) |\n" in section
    assert "first went wrong" not in section and "say only where they start" not in section
    assert "Counted over all 3 failure pattern(s)." in section
    assert "finalize" not in section and "the cause can lie" not in section
    assert (text.index("## The pipeline you may edit")
            < text.index("## Where the pipeline fails, step by step")
            < text.index("## The operations you may use"))
    # "unclear" names no step, with or without a known graph.
    assert [f["where"]["nodes"] for f in analysts.reader_failures(readers)] == [
        ["review"], [], ["draft"], ["review"], []]
    # With an analysis the analysts' findings are used, not the readers'.
    both = _section(_ctx3(reader_findings=readers,
                          analysis={"findings": FINDINGS, "batches": []}).render(),
                    "Where the pipeline fails, step by step")
    assert "From the analysts' failure findings" in both


def test_a_text_setting_with_line_breaks_or_a_bar_is_shown_exactly():
    value = "First line | with a bar\n  indented second line\n```inner fence```\nlast"
    knobs = [*KNOBS, {"name": "draft__system_prompt", "value": value, "kind": "str",
                      "frozen": False}]
    settings = _section(_ctx3(knobs=knobs).render(), "Each step's settings")
    assert value in settings, "the exact text, line breaks and bar included"
    assert ("| draft__system_prompt | (shown whole below, "
            f"{len(value)} characters) |") in settings
    assert "````\n" + value + "\n````" in settings, "a fence longer than any run inside"
    # One-line texts stay in the table as they are.
    assert f"| draft__task | {KNOBS[1]['value']} |" in settings


def test_the_attribution_survives_the_analyst_stage_into_the_request():
    def analyst(system: str, prompt: str) -> str:
        ids = [line.split()[2] for line in prompt.splitlines() if line.startswith("### Item ")]
        if system == analysts.SUCCESS_SYSTEM:
            return json.dumps({"findings": [{"finding": "ok", "items": ids, "preserve": "p"}]})
        return json.dumps({"findings": [{
            "finding": "final overrides draft", "items": ids, "change": "c",
            "where": {"nodes": ["final"]},
            "steps": {"first_wrong": "final", "overrode": ["final"]}}]})

    client = Client(analyst=analyst)
    records = [_rec(f"f{i}", False) for i in range(3)] + [_rec("s0", True)]
    optimizer = AutoresearchOptimizer(client)
    optimizer.propose(_ctx(records, group_size=2), 1, [])
    assert all(f["steps"] == {"first_wrong": "final", "overrode": ["final"]}
               for f in optimizer.last_analysis["findings"] if f["source"] == "failure")
    request = client.proposal()
    section = _section(request, "Where the pipeline fails, step by step")
    assert re.search(r"\| final \| 3 \(\d\) \| 3 \(\d\) \| 3 \(\d\) \| - \|", section), section


# ── domain-free ────────────────────────────────────────────────────────────────


def test_the_new_wording_names_no_task_or_dataset():
    text = _ctx3(analysis={"findings": FINDINGS, "batches": []}, parsimony=0.0,
                 score_is_item_mean=True, structure_guidance=True,
                 traces=[{"structural": {"structural_candidates": 0}}]).render().lower()
    new = " ".join([_section(text, "where the pipeline fails, step by step"),
                    _section(text, "the pipeline you may edit"),
                    _section(text, "what the score measures"),
                    base.structural_text({"structural_candidates": 0}),
                    feedback.STRUCTURE_GUIDANCE, analysts.FAILURE_SYSTEM,
                    analysts.FAILURE_ANSWER, analysts.MERGE_ANSWER])
    for word in DENYLIST:
        assert not re.search(rf"\b{re.escape(word)}\b", new, re.IGNORECASE), word


# ── every outer loop that asks the shared model proposer ──────────────────────


@pytest.mark.parametrize("name,overrides", [
    *((name, {}) for name in __import__("refactory_dsh.strategies", fromlist=["names"]).names()),
    ("default", {"parent_picker": "evolutionary", "portfolio": "prompt+pipeline"}),
])
def test_every_model_proposer_strategy_gets_the_graph_first_request(
        tmp_path, monkeypatch, name, overrides):
    """The order, the kind-neutral rules, the structural evidence and the step map live in
    the shared request (`ProposalContext.render`), so every registered strategy that asks
    the model proposer gets them, end to end on the search-combo toy. (The SkillOpt package
    strategy writes its own text-only requests and is not among them.)"""
    from test_analysts import ANALYST_SYSTEMS, MERGE_SYSTEMS
    from test_search_combo import ComboModel, _parts, _run

    original = ComboModel.complete

    def complete(self, prompt, *, system="", **kwargs):  # type: ignore[no-untyped-def]
        if system in ANALYST_SYSTEMS:
            ids = [line.split()[2] for line in prompt.splitlines() if line.startswith("### Item ")]
            return json.dumps({"findings": [{
                "finding": "F", "items": ids, "where": {"nodes": ["solve"]}, "change": "c",
                "steps": {"first_wrong": "solve", "repeated": ["solve"]}}]})
        if system in MERGE_SYSTEMS:
            return Client().complete(prompt, system=system)
        return original(self, prompt, system=system, **kwargs)

    monkeypatch.setattr(ComboModel, "complete", complete)
    model = ComboModel()
    loop = _run(name, tmp_path, model, val_gate=None, epochs=3, parsimony=0.0,
                parts_manifest=str(_parts(tmp_path)), analysts=True, analyst_group_size=2,
                **overrides)
    assert loop.optimizer.proposal_failures == []
    assert model.requests, name
    order = ["## The pipeline you may edit", "## The operations you may use",
             "## Each step's settings", "## The rules the validator checks",
             "## Evidence from the traces", "## Answer"]
    for request in model.requests:
        text = request.text
        assert all(heading in text for heading in order), name
        assert [text.index(h) for h in order] == sorted(text.index(h) for h in order), name
        assert "Parameters you may move" not in text
        assert "Few well-argued changes" not in text
        assert "neither kind is the safer default" in " ".join(text.split())
        assert "Compute policy:" in text, "this run has parsimony 0"
    evidence = [_section(r.text, "Evidence from the traces") for r in model.requests]
    assert any("| structural changes |" in e for e in evidence), name
    assert all("structural_yield" not in e for e in evidence), name
    mapped = [r.text for r in model.requests if "## Item results" in r.text]
    assert mapped and all("## Where the pipeline fails, step by step" in t for t in mapped), name
    for text in mapped:
        assert (text.index("## The pipeline you may edit")
                < text.index("## Where the pipeline fails, step by step")
                < text.index("## The operations you may use")), name


def test_before_the_first_gradient_the_structure_row_is_counted_from_the_history():
    history = [{"id": "c0", "structural": False}, {"id": "c1", "structural": False}]
    evidence = _section(_ctx3(history=history).render(), "Evidence from the traces")
    assert ("| structural changes | No structural change has been tried in this run yet; "
            "every candidate so far (2) keeps the starting pipeline's structure. |") in evidence
    mixed = _section(_ctx3(history=[*history, {"id": "c2", "structural": True}]).render(),
                     "Evidence from the traces")
    assert "1 candidate(s) so far made a structural change and 2 did not." in mixed
    # A truncated history cannot say what the whole run did: nothing is claimed.
    cut = _ctx3(history=history, history_truncated=True).render()
    assert "structural changes" not in _section(cut, "Evidence from the traces")
    assert "structural changes" not in _section(_ctx3().render(), "Evidence from the traces")


def test_the_skillopt_strategy_builds_its_own_requests_from_none_of_this():
    """The SkillOpt package strategy is text-only by design and writes its own prompts:
    none of its modules imports the shared request, its wording or the loop's analysts, so
    this change cannot reach its requests (their bytes stay as they were)."""
    import ast
    from pathlib import Path

    import refactory_packages.skillopt as skillopt

    shared = {"ProposalContext", "answer_instructions", "composition_section",
              "bundle_instructions", "objective_text", "objective_lines", "compute_policy",
              "structural_text", "STRUCTURE_GUIDANCE", "analysts", "feedback", "base"}
    for path in Path(skillopt.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "refactory_dsh.optim"):
                assert not node.module.endswith((".base", ".analysts", ".feedback")), path
                assert not {a.name for a in node.names} & shared, (path, node.module)
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith(("refactory_dsh.optim.base",
                                                  "refactory_dsh.optim.analysts"))
                               for a in node.names), path
