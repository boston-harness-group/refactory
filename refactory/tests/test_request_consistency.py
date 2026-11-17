"""What the request says in one place agrees with what it says in another, and with the code.

A fact the request shows in several places -- a loop's cap, a frozen knob, a protected
step, the text the system appends to a judge -- is generated from one source each time;
these tests hold the places to each other, so a renderer that drifts fails here instead
of telling the proposer two different things. And each core operator's one-line claim
(`nn.core_ops.OPERATORS`) is checked against what the operator actually builds.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import pytest

import structure_fuzz as sf
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.nn import core_ops, pipeline_view
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.graph import Graph
from refactory_dsh.optim import Candidate, feedback
from test_core_operators import _apply, _ok


def _graph(candidate: Candidate) -> Graph:
    return Graph.from_state(candidate.state["graph"])


def _succ(graph: Graph, node: str) -> set[str]:
    return {e.target for e in graph.edges if e.source == node}


def _pred(graph: Graph, node: str) -> set[str]:
    return {e.source for e in graph.edges if e.target == node}


def _edges(graph: Graph) -> set[tuple[str, str, str | None]]:
    return {(e.source, e.target, e.condition) for e in graph.edges}


def _new(before: Graph, after: Graph) -> list[str]:
    return sorted(set(after.nodes) - set(before.nodes))


def _outputs(graph: Graph, node: str) -> set[tuple[str, str]]:
    n = graph.nodes[node]
    return {("file", w) for w in n.writes} | {("fact", p) for p in n.produces}


def _consumed(candidate: Candidate, node: str) -> set[tuple[str, str]]:
    graph = _graph(candidate)
    return set(core_ops._consumed(graph.nodes[node], candidate.state))


def _reaches(graph: Graph, start: str, goal: str) -> bool:
    seen, todo = set(), [start]
    while todo:
        node = todo.pop()
        if node == goal:
            return True
        if node not in seen:
            seen.add(node)
            todo += _succ(graph, node)
    return False


def _seed(name: str) -> Graph:
    return Graph.from_state(sf.Applier(sf.seeds()[name]).seed_state["graph"])


# ── each core operator's line, against what it builds ─────────────────────


def _takes(out: Candidate, node: str) -> bool:
    """Whether a step's prompt can take a new fact (`core_ops.REACHES`): an agent's own
    prompt knob, or the one text knob an op fills facts into."""
    n = _graph(out).nodes[node]
    if n.kind == "agent":
        return core_ops._task_knob(n) is not None
    return len(core_ops._filled_text_knobs(n, out.state)) == 1


def _reaches_the_reader(out: Candidate, step: str, after: str) -> None:
    """`core_ops.REACHES`: into the nearest prompt after ``after`` that can take it, else the
    candidate says nothing reads it."""
    graph = _graph(out)
    readers = [r for r in core_ops._readers_from(graph, after, out.state) if r != step]
    taker = next((r for r in readers if _takes(out, r)), None)
    if taker is not None:
        assert _outputs(graph, step) & _consumed(out, taker), f"its output reaches {taker}"
    else:
        assert any(f"no step reads what {step}" in note for note in out.notes), out.notes


def _insert() -> None:
    """'it sees what that step sees plus its outputs; its output goes into the nearest prompt
    after it that can take it ...'"""
    before = _seed("fuzz_draft_review")
    _a, out = _ok("fuzz_draft_review", {"op": "insert", "after": "draft"})
    graph = _graph(out)
    step = next(n for n in _new(before, graph) if graph.nodes[n].kind == "agent")
    assert _outputs(before, "draft") & _consumed(out, step), "it sees draft's outputs"
    assert _reaches(graph, "draft", step) and _reaches(graph, step, "review")
    _reaches_the_reader(out, step, step)


def _remove() -> None:
    """'take a step out and reconnect around it'"""
    before = _seed("fuzz_draft_review")
    _a, inserted = _ok("fuzz_draft_review", {"op": "insert", "after": "draft"})
    added = _new(before, _graph(inserted))
    step = next(n for n in added if _graph(inserted).nodes[n].kind == "agent")
    _a, removed = _ok("fuzz_draft_review", {"op": "remove", "step": step}, inserted.state)
    assert _edges(_graph(removed)) == _edges(before)


def _parallelize() -> None:
    """'one step beside 'copies' of itself ... between a fork and a join'"""
    before = _seed("fuzz_draft_review")
    _a, out = _ok("fuzz_draft_review", {"op": "parallelize", "steps": ["draft"], "copies": 1})
    graph = _graph(out)
    fork = next(n for n in graph.nodes.values() if n.kind == "fork")
    join = next(n for n in graph.nodes.values() if n.kind == "join")
    copies = [t for t in fork.targets if t != "draft"]
    assert "draft" in fork.targets and len(copies) == 1 and copies[0] in _new(before, graph)
    assert all(_reaches(graph, t, join.id) for t in fork.targets)


def _serialize() -> None:
    """'run a fork's branches one after another, in their order'"""
    before = _seed("fuzz_fork_join")
    fork = before.nodes["split"]
    _a, out = _ok("fuzz_fork_join", {"op": "serialize", "fork": "split"})
    graph = _graph(out)
    assert not [n for n in graph.nodes.values() if n.kind in ("fork", "join")]
    branches = list(fork.targets)
    for first, second in zip(branches, branches[1:]):
        assert _reaches(graph, first, second) and not _reaches(graph, second, first)


def _branch() -> None:
    """'a new step as another branch of a fork ...; its output goes into the nearest prompt
    after the join that can take it ...'"""
    before = _seed("fuzz_fork_join")
    _a, out = _ok("fuzz_fork_join", {"op": "branch", "fork": "split"})
    graph = _graph(out)
    fork = graph.nodes["split"]
    join = next(n for n in graph.nodes.values() if n.kind == "join")
    heads = [t for t in fork.targets if t not in before.nodes["split"].targets]
    new = [s for s in join.sources if s not in before.nodes["merge"].sources]
    assert len(heads) == 1 and len(new) == 1 and _reaches(graph, heads[0], new[0])
    _reaches_the_reader(out, new[0], join.id)


def _rewire() -> None:
    """'"task" (the task's own inputs, no model output)'"""
    _a, out = _ok("fuzz_draft_review", {"op": "rewire", "step": "review", "sees": ["task"]})
    graph = _graph(out)
    model_outputs = set().union(*(_outputs(graph, n.id) for n in graph.nodes.values()
                                  if n.kind == "agent"))
    assert not _consumed(out, "review") & model_outputs


def _loop() -> None:
    """'it is sent back at most 'max_iterations' times, so it runs at most that many times
    plus one (default D: up to D+1 runs; LO to HI; afterwards the knob
    <gate>__max_iterations, the gate being <step>_check unless 'id' names it), then the run
    goes on'; an agent judge's 'answer starts with proceed or reloop'; and 'with 'back_to',
    send the work back to an earlier agent step, so every step from there to the judge runs
    again (a fork in between whole)'."""
    default, lo, hi = core_ops.LOOP_ITERATIONS
    line = core_ops.OPERATORS["loop"]["line"]
    assert f"default {default}: up to {default + 1} runs; {lo} to {hi}" in line
    for edit, gate_id, cap in (
            ({"op": "loop", "step": "draft"}, "draft_check", default),
            ({"op": "loop", "step": "draft", "id": "gatekeeper", "max_iterations": hi},
             "gatekeeper", hi)):
        before = _seed("fuzz_draft_review")
        _a, out = _ok("fuzz_draft_review", edit)
        graph = _graph(out)
        gate = graph.nodes[gate_id]
        assert gate.max_iterations == cap and gate.on_exhausted == "continue"
        knob = out.state["knobs"][f"{gate_id}__max_iterations"]
        assert knob["value"] == cap
        assert (gate_id, "draft", "reloop") in _edges(graph)
        old_next = _succ(before, "draft")
        assert {e.target for e in graph.edges
                if e.source == gate_id and e.condition == "proceed"} == old_next
        drawn = pipeline_view.render(out.state)
        assert pipeline_view.LOOP_LINE.format(
            step="draft", judge="", cap=cap, runs=cap + 1, knob="",
            exhausted="").split(",")[0] in drawn
        assert f"max {cap} -> {cap + 1} runs" in drawn
    assert "with 'back_to', send the work back to an earlier agent step, so every step " \
        "from there to the judge runs again (a fork in between whole)" in line
    _a, out = _ok("fuzz_draft_review", {"op": "loop", "step": "review", "back_to": "draft"})
    graph = _graph(out)
    assert ("review_check", "draft", "reloop") in _edges(graph)
    assert {e.target for e in graph.edges if e.source == "review_check"
            and e.condition == "proceed"} == _succ(_seed("fuzz_draft_review"), "review")
    assert "loop(draft … review, judge " in pipeline_view.render(out.state)
    _a, refused = _apply("fuzz_draft_review", {"op": "loop", "step": "draft",
                                               "back_to": "review"})
    assert isinstance(refused, list) and "comes after 'draft'" in refused[0], refused
    for bad in (lo - 1, hi + 1):
        _a, refused = _apply("fuzz_draft_review",
                             {"op": "loop", "step": "draft", "max_iterations": bad})
        assert isinstance(refused, list) and f"from {lo} to {hi}" in refused[0], refused
    _a, out = _ok("fuzz_draft_review", {"op": "loop", "step": "draft", "judge": "agent",
                                        "text": "Check the draft."})
    judge = next(n for n in _graph(out).nodes.values()
                 if n.kind == "agent" and n.id.startswith("draft_check"))
    own, appended = feedback.step_appended(out.state, judge.id)
    assert own.startswith("Check the draft.")
    assert appended and "proceed" in appended and "reloop" in appended


def _route() -> None:
    """'a judge after a step chooses one labelled next step ...; the branches rejoin'"""
    _a, out = _ok("fuzz_draft_review", {"op": "route", "after": "review"})
    graph = _graph(out)
    gate = next(n for n in graph.nodes.values() if n.kind == "gate" and "review" in
                {p for p in _pred(graph, n.id)} | {p for q in _pred(graph, n.id)
                                                   for p in _pred(graph, q)})
    labelled = [e for e in graph.edges if e.source == gate.id]
    assert len({e.condition for e in labelled}) >= 2 and all(e.condition for e in labelled)
    ends = [e.target for e in labelled]
    joint = [n for n in graph.nodes if all(_reaches(graph, t, n) for t in ends)]
    assert joint, "the branches rejoin"


def _split() -> None:
    """'replace a step by two new steps run in turn ...; the last takes over its outputs'"""
    before = _seed("fuzz_draft_review")
    _a, out = _ok("fuzz_draft_review", {"op": "split", "step": "finalize"})
    graph = _graph(out)
    new = [n for n in _new(before, graph) if graph.nodes[n].kind == "agent"]
    assert "finalize" not in graph.nodes or graph.nodes["finalize"].kind == "agent"
    steps = [n for n in graph.nodes if n in new or n == "finalize"]
    assert len(steps) == 2
    last = next(n for n in steps if not any(_reaches(graph, n, m) for m in steps if m != n))
    assert _outputs(before, "finalize") <= _outputs(graph, last)


def _merge() -> None:
    """'two consecutive steps as one, with the first's inputs and the second's outputs'"""
    before = _seed("fuzz_draft_review")
    _a, out = _ok("fuzz_draft_review", {"op": "merge", "steps": ["review", "finalize"]})
    graph = _graph(out)
    steps = [n for n in ("review", "finalize", *_new(before, graph))
             if n in graph.nodes and graph.nodes[n].kind == "agent"]
    assert len(steps) == 1
    assert _outputs(before, "finalize") <= _outputs(graph, steps[0])
    assert _outputs(before, "draft") & _consumed(out, steps[0]), "the first's inputs"


#: Each core operator's line, checked against what it builds.
CLAIMS: dict[str, Callable[[], None]] = {
    "insert": _insert, "remove": _remove, "parallelize": _parallelize,
    "serialize": _serialize, "branch": _branch, "rewire": _rewire, "loop": _loop,
    "route": _route, "split": _split, "merge": _merge}


def test_every_core_operator_s_line_is_checked():
    assert set(CLAIMS) == set(core_ops.CORE_OPS) == set(core_ops.OPERATORS)


@pytest.mark.parametrize("op", sorted(CLAIMS))
def test_a_core_operator_builds_what_its_line_says(op):
    CLAIMS[op]()


# ── a fact the request shows in several places says the same thing ───────


def _request(tmp_path: Path, mode: str, edit: dict | None = None) -> tuple[str, dict]:
    loop = Loop(mode, LoopConfig(runs_root=tmp_path / "run", optimizer="random"),
                project=Path("."))
    if edit is not None:
        state = nn_edits.apply(edit, loop.mode.state_dict())
        loop.mode.load_state_dict(state)
    return loop._context().render(), loop.mode.state_dict()


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:end if end >= 0 else len(text)]


def _fenced(text: str) -> str:
    """The lines of ``text`` inside its code fence (the view itself, not its legend)."""
    return text.split("```text\n", 1)[1].split("\n```", 1)[0]


def _settings(text: str) -> dict[str, list[str]]:
    rows = {}
    for line in _section(text, "## Each step's settings").splitlines():
        cells = [c.strip() for c in line.strip("|").split(" | ")]
        if line.startswith("| ") and len(cells) >= 6 and cells[0] not in ("name", "---"):
            rows[cells[0]] = cells
    return rows


MODES = ["livemath/solver", "livemath/skill_graph"]


@pytest.mark.parametrize("mode", MODES)
def test_frozen_knobs_agree_across_the_view_the_settings_and_the_fit_table(tmp_path, mode):
    text, state = _request(tmp_path, mode)
    frozen = pipeline_view._frozen(state)
    assert frozen, "the fixture has frozen knobs"
    settings = _settings(text)
    assert {name for name, cells in settings.items() if cells[5] == "true"} == \
        frozen & set(settings)
    view = _fenced(_section(text, "## The pipeline you may edit"))
    starred = set(re.findall(r"([A-Za-z_][\w]*)\*", view))
    told = set(re.findall(r"([A-Za-z_][\w]*)[*?]? \"", view))
    assert starred == frozen & told, (starred, frozen & told)
    for named in re.findall(r"frozen knob\(s\) \[([^\]]*)\]", text):
        assert set(re.findall(r"'([^']+)'", named)) <= frozen


@pytest.mark.parametrize("cap", [1, 3])
def test_a_loop_s_cap_agrees_across_the_view_the_settings_and_the_node(tmp_path, cap):
    text, state = _request(tmp_path, "livemath/skill_graph",
                           {"op": "loop", "step": "compare", "max_iterations": cap})
    gate = state["graph"]["nodes"]["compare_check"]
    assert gate["max_iterations"] == cap
    assert f"max {cap} -> {cap + 1} runs" in _fenced(_section(text, "## The pipeline you may edit"))
    assert _settings(text)["compare_check__max_iterations"][1] == str(cap)


def test_protected_steps_agree_across_the_view_and_the_fit_table(tmp_path):
    loop = Loop("livemath/skill_graph",
                LoopConfig(runs_root=tmp_path / "run", optimizer="random"), project=Path("."))
    text = loop._context().render()
    view = _fenced(_section(text, "## The pipeline you may edit"))
    marked = {m for m in re.findall(r"^\s*(?:\(fallback\) )?(\S+) .*\[protected\]", view, re.M)}
    shown = {p for p in loop.protected if re.search(rf"^\s*{re.escape(p)} ", view, re.M)}
    assert marked == shown


# ── the text the system appends to a judge, beside its own (S1-s42) ───────


S42_JUDGE = ("Read the solver's answer and check each step of its working.\n"
             "Put your verdict on the last line.")


def test_a_judge_s_own_words_and_the_appended_rule_are_shown_side_by_side(tmp_path):
    """The S1-s42 judge said 'verdict on the last line' while the system appends 'start
    your answer with proceed or reloop': the wiring record (and the attempt notes, which
    carry it), the pipeline view and the step review each show both, each as what it is."""
    _text, before = _request(tmp_path, "livemath/solver")
    after = nn_edits.apply({"op": "loop", "step": "solve", "judge": "agent",
                            "text": S42_JUDGE}, before)
    judge = next(n for n in after["graph"]["nodes"] if n.startswith("solve_check_judge"))
    own, appended = feedback.step_appended(after, judge)
    assert own.startswith(S42_JUDGE) and "proceed" in appended and "reloop" in appended
    wiring = pipeline_view.changed_lines(before, after)
    line = next(line for line in wiring if judge in line)
    assert "Read the solver's answer" in line
    assert "appended by the system" in line and "proceed or reloop" in line
    assert any(judge in line and "appended by the system" in line
               for line in pipeline_view.render(after).splitlines())
    review = feedback.review_prompt({"step": judge, "candidate": "c", "kind": "loop",
                                     "own_text": own, "appended": appended,
                                     "samples": [{"item": "i1", "input": "x",
                                                  "output": "proceed"}]}, "livemath/solver")
    assert f"Its own text:\n{S42_JUDGE}" in review
    assert f"Appended by the system after it:\n{appended}" in review


def test_as_built_asks_for_the_description_to_be_checked_against_it():
    from refactory_dsh.optim import base

    entry = {"flow_change": "a judge after solve", "wiring": ["+ solve_check gate"]}
    assert base._as_built_lines(entry)[-1] == "Check the description above against As built."
    assert "Check the description" not in "\n".join(base._as_built_lines(
        {"wiring": entry["wiring"]})), "only where there is a description to check"


# ── a knob pointer copied into a core operator's text ──────────────────────


def _text_knob(state: dict) -> tuple[str, str]:
    return next((name, entry["value"]) for name, entry in state["knobs"].items()
                if isinstance(entry.get("value"), str) and len(entry["value"]) > 5)


@pytest.mark.parametrize("seed, edit", [
    ("fuzz_draft_review", {"op": "insert", "after": "draft", "kind": "agent"}),
    ("fuzz_fork_join", {"op": "branch", "fork": "split", "kind": "agent"}),
    ("fuzz_draft_review", {"op": "split", "step": "finalize",
                           "into": [{"kind": "agent"}, {"kind": "agent", "text": "Second."}]}),
], ids=["insert", "branch", "split"])
def test_a_pointer_in_a_core_operator_s_text_becomes_the_knob_s_text(seed, edit):
    """`nn.edits.KNOB_POINTER` copied from the node reference into an operator's 'text' is
    the knob's current text in the new step's knob, never the pointer itself; a pointer to
    a knob the graph lacks is refused."""
    state = sf.Applier(sf.seeds()[seed]).seed_state
    name, value = _text_knob(state)
    pointer = nn_edits.KNOB_POINTER.format(knob=name)
    with_text = ({**edit, "into": [{**edit["into"][0], "text": pointer}, edit["into"][1]]}
                 if "into" in edit else {**edit, "text": pointer})
    _a, out = _ok(seed, with_text)
    texts = [str(e.get("value")) for e in out.state["knobs"].values()]
    assert not any(pointer in t for t in texts), "the pointer was stored as a prompt"
    new = [str(e.get("value")) for n, e in out.state["knobs"].items() if n not in state["knobs"]]
    head = value.split("{{", 1)[0].strip()  # a slot the new step cannot see is taken out
    assert head and any(head in t for t in new), "the new step's knob holds the knob's text"
    missing = nn_edits.KNOB_POINTER.format(knob="no_such_knob")
    bad = ({**edit, "into": [{**edit["into"][0], "text": missing}, edit["into"][1]]}
           if "into" in edit else {**edit, "text": missing})
    _a, refused = _apply(seed, bad)
    assert isinstance(refused, list) and "no_such_knob" in " ".join(refused), refused


def test_only_a_judge_s_prompt_is_split_at_the_system_s_verdict_rule():
    """The system appends its verdict rule where it builds a judge; a step elsewhere whose
    own prompt ends with the very same words is shown as its own text, in the view and in
    the step review's split."""
    from refactory_dsh.optim import feedback

    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    state = nn_edits.apply({"op": "loop", "step": "draft", "judge": "agent",
                            "text": "Check the draft."}, state)
    judge = next(n for n in state["graph"]["nodes"] if n.startswith("draft_check_judge"))
    rule = feedback.step_appended(state, judge)[1]
    assert rule
    task = dict(state["graph"]["nodes"]["review"]["field_bindings"])["prompt_template"]
    state["knobs"][task]["value"] = state["knobs"][task]["value"] + "\n\n" + rule
    # Matching the text alone would split review's prompt too: the position decides.
    assert core_ops.system_appended(state["knobs"][task]["value"])[1] == rule
    assert feedback.step_appended(state, "review") == ("", "")
    drawn = pipeline_view.render(state)
    assert drawn.count("appended by the system") == 1, drawn
    review = next(line for line in drawn.splitlines() if line.strip().startswith("review "))
    assert "appended by the system" not in review
