"""The request's "what each step carries, and which operators apply to it" section.

Each row is worked out by applying the core operators with their default arguments to the
request's own state, so what it says applies must apply, what it says is refused must be
refused with those words, and the alternative a refusal names must itself apply. These
tests hold the section to that on the S4 start state (livemath/skill_graph) and on a nested
fork, and pin the loop operator's line to what a loop actually builds.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import structure_fuzz as sf
from refactory_dsh import registry
from refactory_dsh.nn import core_ops
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import base
from refactory_dsh.optim.memory import state_view

MODE = "livemath/skill_graph"


def _start_state() -> dict:
    _package, mode = registry.resolve(Path("."), MODE)
    return mode().state_dict()


def _section(state: dict, **kwargs) -> list[str]:
    view = state_view(state)
    return base.operator_fit_section(nn_edits.catalogue("broad"), view.node_states, view.edges,
                                     view.knobs, **kwargs)


def _rows(lines: list[str]) -> dict[str, list[str]]:
    table = next(line for line in lines if line.startswith("| step |"))
    rows = {}
    for line in table.splitlines()[2:]:
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        rows[cells[0]] = cells[1:]
    return rows


def _outcome(edit: dict, state: dict) -> str | None:
    reason = nn_edits.precheck(edit, state)
    if reason is None:
        try:
            nn_edits.apply(edit, copy.deepcopy(state))
        except nn_edits.EditError as exc:
            reason = str(exc)
    return reason


def _edits_by_label(state: dict, step: str) -> dict[str, dict]:
    from refactory_dsh.nn.graph import Graph

    graph = Graph.from_state(state["graph"])
    return dict(base._fit_edits(graph.nodes[step], graph, set(nn_edits.CORE_OPS)))


def _check_rows_are_true(state: dict) -> dict[str, list[str]]:
    """Every row's verdicts agree with applying the operator: an applying label applies,
    a refused one is refused with the reason the row gives."""
    rows = _rows(_section(state))
    assert rows
    for step, (_carries, applies, refused) in rows.items():
        edits = _edits_by_label(state, step)
        for label in filter(None, applies.split(", ")) if applies != "-" else ():
            assert _outcome(edits[label], state) is None, (step, label)
        for entry in refused.split(" | ") if refused != "-" else ():
            labels, reason = entry.split(": ", 1)
            for label in labels.split(", "):
                got = _outcome(edits[label], state)
                assert got is not None and got.removeprefix("E-precondition: ") == reason, (
                    step, label, got)
    return rows


def test_each_row_on_the_s4_start_state_says_what_the_operators_do():
    rows = _check_rows_are_true(_start_state())
    model_steps = {"compare", "decide", "hypotheses", "precision"}
    assert model_steps <= set(rows)
    assert "fact" not in " ".join(rows)  # rows are steps, not facts
    assert rows["step_fork"][1] == "branch, serialize"


def test_a_model_step_s_row_follows_its_prompt_back_through_the_plain_op_that_built_it():
    rows = _rows(_section(_start_state()))
    assert rows["decide"][0].startswith(
        "decide_prompt = op decide_prompt from compare_notes.txt (compare), "
        "hypotheses_notes.txt (hypotheses), precision_notes.txt (precision)")
    assert rows["precision"][0].startswith(
        "precision_prompt = op notes_prompts from compare_notes.txt (compare)")
    assert rows["notes_prompts"][0] == "-"  # only model steps' prompts are traced
    # The text a plain op builds a prompt from is often a knob: named as one.
    assert "precision_prompt = op notes_prompts from compare_notes.txt (compare), " \
        "knob precision_task, knob hypotheses_task" in rows["precision"][0], rows["precision"][0]
    _package, mode = registry.resolve(Path("."), "livemath/solver")
    assert "knob skill" in _rows(_section(mode().state_dict()))["solve"][0]


def test_a_refusal_names_the_exact_replacing_edit_and_that_edit_applies():
    state = _start_state()
    refused = _rows(_section(state))["decide"][2]
    hint = re.search(r"replace it with (\{.*?\}\]\})", refused)
    assert hint, refused
    edit = json.loads(hint.group(1))
    assert edit == {"op": "split", "step": "decide",
                    "into": [{"kind": "agent", "text": "<what it does>", "sees": ["task"]}]}
    assert _outcome(edit, state) is None
    # The same exact edit for a data pass's ends, where the replacement is code.
    reason = _outcome({"op": "remove", "step": "prepare"}, state)
    edit = json.loads(re.search(r"replace it with (\{.*\})$", reason).group(1))
    assert edit["into"][0]["kind"] == "code" and edit["step"] == "prepare"
    assert _outcome(edit, state) is None


def test_a_refusal_never_names_a_replacing_edit_that_would_be_refused():
    """Where the replacement is refused too (a fork's branch; a step alone reading a frozen
    knob), the hint says so with that refusal instead of naming the edit."""
    state = _start_state()
    refused = _rows(_section(state))["hypotheses"][2]
    assert "replacing it is refused too ('hypotheses' is a fork's branch" in refused, refused
    assert '"op": "split"' not in refused
    _package, mode = registry.resolve(Path("."), "livemath/solver")
    solver = mode().state_dict()
    reason = _outcome({"op": "remove", "step": "solve"}, solver)
    assert "replacing it is refused too ('solve' alone reads the frozen knob(s)" in reason
    for row in _rows(_section(solver)).values():
        for found in re.findall(r"replace it with (\{.*?\}\]\})", row[2]):
            assert _outcome(json.loads(found), solver) is None, found


def test_a_step_alone_reading_a_frozen_knob_is_shown_unsplittable():
    state = _start_state()
    assert "split" in _rows(_section(state))["compare"][1].split(", ")
    state["knobs"]["compare__system_prompt"]["spec"]["frozen"] = True
    rows = _check_rows_are_true(state)
    applies, refused = rows["compare"][1].split(", "), rows["compare"][2]
    assert "split" not in applies and "frozen knob(s) ['compare__system_prompt']" in refused
    assert "frozen knob" in _section(state)[0]


def test_each_row_on_a_nested_fork_says_what_the_operators_do():
    state = sf.Applier(sf.seeds()["fuzz_nested_fork"]).seed_state
    rows = _check_rows_are_true(state)
    forks = [n for n, node in state["graph"]["nodes"].items() if node.get("kind") == "fork"]
    assert len(forks) >= 2 and set(forks) <= set(rows)


def test_a_protected_step_is_shown_untouchable():
    state = _start_state()
    rows = _rows(_section(state, protected=["decide"]))
    assert rows["decide"][1:] == ["-", "every operator: protected"]


def test_the_section_stays_within_its_limit_and_names_what_it_left_out():
    state = _start_state()
    full = _section(state)
    assert sum(map(len, full)) <= base.OPERATOR_FIT_CHARS
    small = _section(state, limit=2500)
    assert sum(map(len, small)) <= 2500
    assert small[2].startswith("(") and "not listed" in small[2]


def test_no_section_when_the_run_permits_no_core_operator():
    view = state_view(_start_state())
    assert base.operator_fit_section(nn_edits.catalogue("knob_only"), view.node_states,
                                     view.edges, view.knobs) == []


def test_the_loop_line_states_the_cap_and_verdict_the_loop_builds():
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    after = nn_edits.apply({"op": "loop", "step": "review", "judge": "agent"}, state)
    knob = next(k for k in after["knobs"] if k.endswith("__max_iterations"))
    spec = after["knobs"][knob]["spec"]
    line = core_ops.OPERATORS["loop"]["line"]
    assert knob == "review_check__max_iterations" and "<gate>__max_iterations" in line
    assert "<step>_check" in line
    assert f"default {spec['default']}, {spec['lo']} to {spec['hi']}" in line
    judge = after["graph"]["nodes"]["review_check_judge"]
    assert "proceed" in json.dumps(judge) and "reloop" in json.dumps(judge)
    assert "proceed or reloop" in line


def test_an_edit_whose_preconditions_pass_but_whose_expansion_fails_is_shown_refused(
        monkeypatch):
    """`precheck` checks preconditions only; the row goes by the full expansion."""
    state = _start_state()
    real = nn_edits.apply

    def apply(edit, state, *args, **kwargs):
        if edit.get("op") == "insert" and edit.get("after") == "compare":
            raise nn_edits.EditError("E-wiring: the new step would read nothing")
        return real(edit, state, *args, **kwargs)

    monkeypatch.setattr(nn_edits, "apply", apply)
    applies, refused = _rows(_section(state))["compare"][1:]
    assert "insert" not in applies.split(", ")
    assert "insert: E-wiring: the new step would read nothing" in refused


def test_a_replacing_hint_whose_edit_fails_to_expand_says_it_is_refused(monkeypatch):
    state = _start_state()
    real = nn_edits.apply

    def apply(edit, state, *args, **kwargs):
        if edit.get("op") == "split" and edit.get("step") == "prepare":
            raise nn_edits.EditError("E-wiring: the code step would read nothing")
        return real(edit, state, *args, **kwargs)

    monkeypatch.setattr(nn_edits, "apply", apply)
    reason = nn_edits.precheck({"op": "remove", "step": "prepare"}, state)
    assert "replacing it is refused too (E-wiring: the code step would read nothing)" in reason
