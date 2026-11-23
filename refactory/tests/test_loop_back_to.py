"""The ``loop`` operator's ``back_to``: a judge after one step that sends the work back to an
earlier step, so every step from there to the judge runs again.

- **Refusals**, each naming what to use instead: ``back_to`` after the judged step, not on
  every path to it, outside its per-item pass, outside its fork branch, an op (no prompt for
  the reason), and a cycle the validator refuses.
- **Both runtimes** (Python walker and TS spine), on LiveMath S4 (``skill_graph``): a judge
  after ``decide`` sends back to ``compare``; the fork between them runs again whole, the
  judge's reason reaches ``compare``'s prompt, the cap holds, and past it the last attempt
  goes on to grading.
- LiveMath S1 (``solver``): an agent inserted before ``solve`` inside the data pass, and a
  judge after ``solve`` that sends back to it.
- **Nested**: a loop on one step inside a span another loop sends back over.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import structure_fuzz as sf
from refactory_dsh import registry
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import pipeline_view as pv
from refactory_dsh.nn.graph import Graph
from refactory_dsh.nn.validate import validate_state
from test_livemath_blocks import _facts, _patch_task, _scores, _spine, needs_spine
from test_livemath_skill_graph import LABEL_B

S4_AGENTS = ("compare", "precision", "hypotheses", "decide")
JUDGE = "decide_check_judge"
REASON = "compare missed the units of the second choice"
#: The S4 loop: a judge after decide that sends back to compare.
S4_LOOP = {"op": "loop", "step": "decide", "back_to": "compare", "judge": "agent",
           "text": "Is the decision sound?"}


import copy as _copy
import functools as _functools


@_functools.cache
def _state_cached(name: str) -> dict[str, Any]:
    return registry.resolve(Path("."), name)[1]().state_dict()


def _state(name: str) -> dict[str, Any]:
    return _copy.deepcopy(_state_cached(name))


@_functools.cache
def _mode_cached(name: str):
    return registry.resolve(Path("."), name)[1]


def _mode(name: str, state: dict[str, Any]):
    mode = _mode_cached(name)()
    mode.load_state_dict(state)
    return mode


def _refused(edit: dict, state: dict) -> str:
    reason = nn_edits.precheck(edit, state)
    assert reason is not None, edit
    return reason


def _graph(state: dict) -> Graph:
    return Graph.from_state(state["graph"])


class _Script:
    """Answers per node; a list is answered in turn, starting over after the last, counted
    per (node, prompt) -- a step run again on the same inputs sees the same prompt, so each
    item keeps its own count however items interleave. Logs (node, prompt) for every call."""

    def __init__(self, answers: dict[str, Any], log: list | None = None, node: str = "",
                 turns: dict | None = None):
        self.answers, self.node = answers, node
        self.log = [] if log is None else log
        self.turns = {} if turns is None else turns

    def for_node(self, node_id: str, role: str = "") -> "_Script":
        return _Script(self.answers, self.log, node_id, self.turns)

    def with_tool_policy(self, policy: str) -> "_Script":
        return self

    def with_wall(self, wall: Any) -> "_Script":
        return self

    def with_budget(self, budget: Any) -> "_Script":
        return self

    def context(self) -> dict:
        return {}

    def route(self) -> dict:
        return {"worker_backend": "dsh"}

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        self.log.append((self.node, prompt))
        answer = self.answers.get(self.node, "<answer>B</answer>")
        if isinstance(answer, list):
            turn = self.turns.get((self.node, prompt), 0)
            self.turns[(self.node, prompt)] = turn + 1
            answer = answer[turn % len(answer)]
        return answer


_S4_ANSWERS = {"compare": "compare notes <answer>A</answer>",
               "precision": "precision notes <answer>A</answer>",
               "hypotheses": "hypotheses notes <answer>A</answer>",
               "decide": "<answer>B</answer>"}


def _walk(mode, model, tmp_path, monkeypatch):
    _patch_task(monkeypatch)
    workdir = tmp_path / "walker"
    workdir.mkdir(parents=True, exist_ok=True)
    result = mode.run(workdir, model=model)
    assert result.exit_ok, result.error
    return result


def _s4(cap: int) -> dict:
    return nn_edits.apply({**S4_LOOP, "max_iterations": cap}, _state("livemath/skill_graph"))


# ── what the edit builds ──────────────────────────────────────────────────


def test_back_to_puts_the_judge_after_the_step_and_sends_back_to_the_earlier_step():
    state = _s4(2)
    assert validate_state(state) == []
    graph = _graph(state)
    gate = graph.nodes["decide_check"]
    assert {(e.target, e.condition) for e in graph.out_edges("decide_check")} == {
        ("compare", "reloop"), ("extract_answer", "proceed")}
    # decide -> the judge's collect step -> the judge -> the gate.
    assert [e.target for e in graph.out_edges("decide")] == [f"{JUDGE}_inputs"]
    assert [e.target for e in graph.out_edges(JUDGE)] == ["decide_check"]
    assert gate.max_iterations == 2 and gate.on_exhausted == "continue"
    assert state["knobs"]["decide_check__max_iterations"]["value"] == 2
    # The judge sees what decide read and wrote, as a loop on decide alone would.
    plain = nn_edits.apply({**S4_LOOP, "back_to": None}, _state("livemath/skill_graph"))
    assert graph.nodes[JUDGE].to_state() == _graph(plain).nodes[JUDGE].to_state()
    line = next(line.text for line in pv.view_lines(state) if line.key == "decide_check")
    assert line.startswith(f"loop(compare … decide, judge {JUDGE} ")
    assert line.endswith("max 2 -> 3 runs, knob decide_check__max_iterations)")


def test_back_to_the_step_itself_is_the_plain_loop():
    base = _state("livemath/skill_graph")
    assert nn_edits.apply({**S4_LOOP, "back_to": "decide"}, base) == nn_edits.apply(
        {k: v for k, v in S4_LOOP.items() if k != "back_to"}, base)


# ── refusals ──────────────────────────────────────────────────────────────


def test_back_to_after_the_judged_step_is_refused():
    reason = _refused({**S4_LOOP, "step": "compare", "back_to": "decide"},
                      _state("livemath/skill_graph"))
    assert reason == ("'back_to' 'decide' comes after 'compare'; it names an earlier step, "
                      "where the work goes back to when the judge after 'compare' says reloop")


def test_back_to_not_on_every_path_is_refused_and_names_the_steps_that_are():
    reason = _refused({**S4_LOOP, "back_to": "precision"}, _state("livemath/skill_graph"))
    assert reason == ("'back_to' 'precision' is not on every path to 'decide' (a branch or a "
                      "route's way passes it by); it names a step every run passes before "
                      "'decide': compare")
    # A route's way: polish is passed by when the route goes to rough.
    gates = sf.cached_applier("fuzz_gates").seed_state
    assert "is not on every path to 'redo'" in _refused(
        {"op": "loop", "step": "redo", "back_to": "polish"}, gates)


def test_back_to_outside_the_pass_is_refused():
    state = _state("livemath/skill_graph")
    reason = _refused({"op": "loop", "step": "finish", "back_to": "compare"}, state)
    assert reason == ("'back_to' 'compare' is outside the pass 'finish' is in ('finish' is "
                      "outside any per-item pass); a loop sends back only within its pass")
    reason = _refused({**S4_LOOP, "back_to": "finish"}, state)
    assert reason.startswith("'back_to' 'finish' is outside the pass 'decide' is in ('decide' "
                             "is inside the per-item pass 'data')")


def test_back_to_outside_the_judged_step_s_fork_branch_is_refused():
    reason = _refused({**S4_LOOP, "step": "precision"}, _state("livemath/skill_graph"))
    assert reason == ("'precision' is inside a branch of fork 'step_fork' and 'back_to' "
                      "'compare' is not; a loop in a branch sends back only within it (judge a "
                      "step after the fork's join to re-run the whole fork)")


def test_back_to_inside_a_branch_from_after_the_join_is_refused():
    """With one branch, its head is on every path past the join: the branch rule refuses
    it, since a send-back from after the join re-runs the fork whole."""
    import copy

    state = copy.deepcopy(sf.cached_applier("fuzz_fork_join").seed_state)
    graph = state["graph"]
    graph["nodes"]["split"]["targets"] = ["left"]
    graph["nodes"]["merge"]["sources"] = ["left"]
    for gone in ("right_a", "right_b"):
        del graph["nodes"][gone]
    graph["edges"] = [e for e in graph["edges"] if not {e["source"], e["target"]}
                      & {"right_a", "right_b"}]
    graph["nodes"]["decide"]["reads"] = ["left.txt"]
    graph["nodes"]["decide"]["prompt_template"] = "Decide."
    assert validate_state(state) == []
    assert _refused({"op": "loop", "step": "decide", "back_to": "left"}, state) == (
        "'back_to' 'left' is inside a branch of fork 'split' and 'decide' is not; a send-back "
        "from after the join re-runs the whole fork, so it names a step before 'split'")


def test_the_written_out_loop_refuses_back_to():
    state = sf.cached_applier("fuzz_draft_review").seed_state
    edit = {"op": "loop", "target": "review", "back_to": "draft",
            "gate": {"id": "g", "kind": "gate", "evaluator": "structure_fuzz_graphs:proceed"}}
    assert _refused(edit, state) == ("'back_to' goes with the one-line loop ('step' and "
                                     "'judge'); written out, the gate sends back to 'target'")


def test_back_to_an_op_is_refused_naming_the_model_step_to_use():
    """The runtimes hand a reloop's reason only to an agent's prompt (the spine's
    ``feedback[target]``, the walker's ``with_feedback``), so an op could only lose it."""
    state = _state("livemath/skill_graph")
    reason = _refused({**S4_LOOP, "back_to": "prepare"}, state)
    assert reason == ("'back_to' 'prepare' is an op, which has no prompt for the judge's "
                      "reason; send back to the model step that should act on it: compare")
    reason = _refused({**S4_LOOP, "back_to": "notes_prompts"}, state)
    assert reason.endswith("act on it: hypotheses, precision")


def test_a_cycle_the_validator_refuses_is_refused_with_its_rule(monkeypatch):
    """The cycle is held to the validator before the edit applies: a refusal it adds to the
    graph (here V16, stubbed onto the loop's own gate) is the edit's refusal."""
    import importlib

    validate_mod = importlib.import_module("refactory_dsh.nn.validate")

    real = validate_mod.validate

    def strict(graph, values, **kw):
        issues = real(graph, values, **kw)
        if "decide_check" in graph.nodes:
            issues.append(validate_mod.Issue("V16", "a gate in a branch reloops only within "
                                                    "it", node="decide_check"))
        return issues

    monkeypatch.setattr(validate_mod, "validate", strict)
    reason = _refused(S4_LOOP, _state("livemath/skill_graph"))
    assert reason == ("the validator refuses the cycle back to 'compare': V16 [decide_check]: "
                      "a gate in a branch reloops only within it")


def test_back_to_must_be_a_step():
    reason = _refused({**S4_LOOP, "back_to": "step_fork"}, _state("livemath/skill_graph"))
    assert reason == ("'step_fork' is a fork; a loop's 'back_to' works on a step (agent, op "
                      "or code)")


# ── both runtimes: S4 ─────────────────────────────────────────────────────


def _s4_runs_each_step(records: list[dict], runs: int) -> None:
    for record in records:
        nodes = [e["node"] for e in record.get("exchanges") or []]
        for step in S4_AGENTS:
            assert nodes.count(step) == runs, (record["id"], nodes)


@pytest.mark.parametrize("parallelism", [1, 3])
def test_s4_walker_sends_back_to_compare_reruns_the_fork_and_hands_compare_the_reason(
        parallelism, tmp_path, monkeypatch):
    state = nn_edits.configure(_s4(2), "data_parallelism", str(parallelism))
    log: list = []
    model = _Script({**_S4_ANSWERS, JUDGE: [f"reloop: {REASON}", "proceed"]}, log)
    result = _walk(_mode("livemath/skill_graph", state), model, tmp_path, monkeypatch)
    _s4_runs_each_step(result.item_records, 2)
    compares = [prompt for node, prompt in log if node == "compare"]
    assert len(compares) == 10 and sum(REASON in p for p in compares) == 5
    assert _scores(result.facts["item_scores"]) == sorted(
        (str(r["id"]), 1.0 if str(r["id"]) in LABEL_B else 0.0)
        for r in result.facts["item_scores"])


def test_s4_walker_honours_the_cap_and_goes_on_with_the_last_attempt(tmp_path, monkeypatch):
    """The judge always says reloop: with max 1, each step runs twice, then decide's last
    answer is graded (label-B items score 1)."""
    log: list = []
    model = _Script({**_S4_ANSWERS, "decide": ["<answer>A</answer>", "<answer>B</answer>"],
                     JUDGE: [f"reloop: {REASON}"]}, log)
    result = _walk(_mode("livemath/skill_graph", _s4(1)), model, tmp_path, monkeypatch)
    _s4_runs_each_step(result.item_records, 2)
    assert {str(r["id"]): r["score"] for r in result.facts["item_scores"]} == {
        str(r["id"]): 1.0 if str(r["id"]) in LABEL_B else 0.0
        for r in result.facts["item_scores"]}


def _drain_record(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@needs_spine
def test_s4_spine_sends_back_to_compare_reruns_the_fork_and_hands_compare_the_reason(
        tmp_path, monkeypatch):
    """At parallelism 1: the drain answers each node's list in turn across the run, so the
    items go one at a time for each to get reloop, then proceed."""
    _patch_task(monkeypatch)
    state = _s4(2)
    record = tmp_path / "drain.jsonl"
    monkeypatch.setenv("DRAIN_RECORD", str(record))
    responses = {**_S4_ANSWERS, JUDGE: [f"reloop: {REASON}", "proceed"]}
    spine_dir = tmp_path / "spine"
    assert _spine(_graph(state), spine_dir, tmp_path, agent_responses=responses,
                  mode_state=state) == 0
    delivered = [r for r in _drain_record(record) if "prompt" in r]
    for step in S4_AGENTS:
        assert sum(r["node"] == step for r in delivered) == 10, step
    compares = [r for r in delivered if r["node"] == "compare"]
    assert sum(REASON in (r.get("feedback") or "") for r in compares) == 5
    records = json.loads((spine_dir / "item_records.json").read_text())
    _s4_runs_each_step(records, 2)
    assert _scores(_facts(spine_dir)["item_scores"]) == sorted(
        (str(r["id"]), 1.0 if str(r["id"]) in LABEL_B else 0.0) for r in records)


@needs_spine
def test_s4_spine_honours_the_cap_and_goes_on_with_the_last_attempt(tmp_path, monkeypatch):
    _patch_task(monkeypatch)
    state = _s4(1)
    responses = {**_S4_ANSWERS, "decide": ["<answer>A</answer>", "<answer>B</answer>"],
                 JUDGE: [f"reloop: {REASON}"]}
    spine_dir = tmp_path / "spine"
    assert _spine(_graph(state), spine_dir, tmp_path, agent_responses=responses,
                  mode_state=state) == 0
    records = json.loads((spine_dir / "item_records.json").read_text())
    _s4_runs_each_step(records, 2)
    assert {str(r["id"]): r["score"] for r in _facts(spine_dir)["item_scores"]} == {
        str(r["id"]): 1.0 if str(r["id"]) in LABEL_B else 0.0 for r in records}


# ── S1: inside the data pass ──────────────────────────────────────────────


def _s1() -> dict:
    state = nn_edits.apply({"op": "insert", "before": "solve", "kind": "agent", "id": "plan",
                            "text": "Plan how to solve it."}, _state("livemath/solver"))
    return nn_edits.apply({"op": "loop", "step": "solve", "back_to": "plan", "judge": "agent",
                           "text": "Does the answer follow from the plan?"}, state)


def test_s1_sends_back_to_an_earlier_model_step_inside_the_pass(tmp_path, monkeypatch):
    state = _s1()
    assert validate_state(state) == []
    graph = _graph(state)
    assert ("solve_check", "plan", "reloop") in {e.key() for e in graph.edges}
    assert "loop(plan … solve, judge solve_check_judge" in pv.render(state)
    log: list = []
    model = _Script({"plan": "a plan", "solve": "<answer>B</answer>",
                     "solve_check_judge": [f"reloop: {REASON}", "proceed"]}, log)
    result = _walk(_mode("livemath/solver", state), model, tmp_path, monkeypatch)
    plans = [prompt for node, prompt in log if node == "plan"]
    assert len(plans) == 10 and sum(REASON in p for p in plans) == 5
    assert [node for node, _ in log].count("solve") == 10
    assert len(result.facts["item_scores"]) == 5


@needs_spine
def test_s1_spine_sends_back_to_an_earlier_model_step_inside_the_pass(tmp_path, monkeypatch):
    _patch_task(monkeypatch)
    state = _s1()
    record = tmp_path / "drain.jsonl"
    monkeypatch.setenv("DRAIN_RECORD", str(record))
    spine_dir = tmp_path / "spine"
    assert _spine(_graph(state), spine_dir, tmp_path, mode_state=state, agent_responses={
        "plan": "a plan", "solve": "<answer>B</answer>",
        "solve_check_judge": [f"reloop: {REASON}", "proceed"]}) == 0
    delivered = [r for r in _drain_record(record) if "prompt" in r]
    plans = [r for r in delivered if r["node"] == "plan"]
    assert len(plans) == 10 and sum(REASON in (r.get("feedback") or "") for r in plans) == 5
    assert sum(r["node"] == "solve" for r in delivered) == 10
    assert len(_facts(spine_dir)["item_scores"]) == 5


# ── nested: a loop inside a looped span ───────────────────────────────────


def _nested() -> dict:
    """S4 with a loop on precision alone (inside its branch), then the judge after decide
    sending back to compare over it."""
    state = nn_edits.apply({"op": "loop", "step": "precision", "judge": "agent"},
                           _state("livemath/skill_graph"))
    return nn_edits.apply(S4_LOOP, state)


def test_nested_loops_validate_and_read_as_both_loops():
    state = _nested()
    assert validate_state(state) == []
    text = pv.render(state)
    assert "loop(precision, judge precision_check_judge" in text
    assert "loop(compare … decide, judge decide_check_judge" in text


def test_nested_loops_run_on_the_walker(tmp_path, monkeypatch):
    """The inner judge sends precision back once per outer run; the outer judge sends the
    span back once: precision runs 4 times, the other steps twice, per item."""
    log: list = []
    model = _Script({**_S4_ANSWERS, "precision_check_judge": ["reloop: tighten", "proceed"],
                     JUDGE: [f"reloop: {REASON}", "proceed"]}, log)
    result = _walk(_mode("livemath/skill_graph", _nested()), model, tmp_path, monkeypatch)
    for record in result.item_records:
        nodes = [e["node"] for e in record.get("exchanges") or []]
        assert nodes.count("precision") == 4, nodes
        for step in ("compare", "hypotheses", "decide"):
            assert nodes.count(step) == 2, (step, nodes)


@needs_spine
def test_nested_loops_run_on_the_spine(tmp_path, monkeypatch):
    _patch_task(monkeypatch)
    state = _nested()
    spine_dir = tmp_path / "spine"
    assert _spine(_graph(state), spine_dir, tmp_path, mode_state=state, agent_responses={
        **_S4_ANSWERS, "precision_check_judge": ["reloop: tighten", "proceed"],
        JUDGE: [f"reloop: {REASON}", "proceed"]}) == 0
    for record in json.loads((spine_dir / "item_records.json").read_text()):
        nodes = [e["node"] for e in record.get("exchanges") or []]
        assert nodes.count("precision") == 4, nodes
        for step in ("compare", "hypotheses", "decide"):
            assert nodes.count(step) == 2, (step, nodes)
