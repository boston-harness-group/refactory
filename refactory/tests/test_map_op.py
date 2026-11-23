"""The ``map`` operator: a new agent step run once per element of a per-item list, its
results combined into one output for the next step.

Neither runtime runs a data pass inside another (`test_neither_runtime_nests_a_data_pass`),
so a map is not one. It is built from what both runtimes already run: an ``elements`` op
(`operators.MAP_CALL`) writes one prompt per element (the map's one text knob, then the
element); a fork runs one optional branch per element up to ``max``, each an agent that
reads its element's file, so a branch with no element is refused before any model call and
recorded as not needed, never as a missing answer; a join; and the combine step, which
sees how many ran and every result (one not written said so) through a collect step.

- **What the edit builds** on LiveMath S1 (``solver``) and S4 (``skill_graph``), each with a
  code step before it that lists the item's options; the view's one line; the lowering;
  the refusals; the elements op's lists.
- **Both runtimes** (the walker; the spine with dispatched workers) on S1 and S4 with a
  scripted model: one run per option, none for the fifth branch, the combine sees each
  result and its output reaches decide (S4); a list longer than ``max`` runs its first
  ``max``, said so; a map in a loop never reruns an old element; a short list leaves no
  "no answer" in what the optimizer is told; ``remove`` takes a map out whole.
- **A map the driver runs** (the structure fuzz's draft graph, no data pass): on both
  runtimes a short, one-element, empty or over-max list runs only the runs with an
  element; the others are recorded missing and not needed, never handed to a model or the
  driver.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import lowering, nn, registry
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import pipeline_view as pv
from refactory_dsh.nn.core_ops import MAP_ELEMENT_PROMPT
from refactory_dsh.nn.graph import Graph
from refactory_dsh.nn.operators import MAP_CALL, list_elements, map_of
from refactory_dsh.nn.validate import validate_state
from test_crashed_step_retry import Scripted, _data, _spine, _walker, needs_spine
from test_livemath_blocks import _patch_task
from test_livemath_skill_graph import LABEL_B

ITEMS = 5  # the LiveMath fixture, four options each
CHECK = "Is this option correct? Say yes or no, and why."
WEIGH = "Weigh the verdicts on each option."
#: A code step listing the item's options (the item the steps see has no answer).
OPTIONS = ("import json\n"
           "def main(inp):\n"
           "    item = json.loads(inp['files']['current_item.json'])\n"
           "    return {'facts': {'options': json.dumps([c['label'] + ': ' + c['text']\n"
           "                                             for c in item['choices']])}}\n")


import copy as _copy
import functools as _functools


@_functools.cache
def _state_cached(name: str) -> dict[str, Any]:
    return registry.resolve(Path("."), name)[1]().state_dict()


def _state(name: str) -> dict[str, Any]:
    return _copy.deepcopy(_state_cached(name))


def _mode(name: str, state: dict[str, Any]):
    mode = registry.resolve(Path("."), name)[1]()
    mode.load_state_dict(state)
    return mode


def _graph(state: dict[str, Any]) -> Graph:
    return Graph.from_state(state["graph"])


def _mapped(name: str, before: str, **edit: Any) -> dict[str, Any]:
    """``name``'s state with a code step listing the options before ``before``, and a map
    over them after it."""
    state = nn_edits.apply({"op": "insert", "before": before, "kind": "code", "id": "options",
                            "sees": ["current_item.json"], "text": OPTIONS}, _state(name))
    return nn_edits.apply({"op": "map", "after": "options", "over": "options", "text": CHECK,
                           "max": 5, "combine": {"kind": "agent", "text": WEIGH}, **edit}, state)


def _s1(**edit: Any) -> dict[str, Any]:
    return _mapped("livemath/solver", "solve", **edit)


def _s4(**edit: Any) -> dict[str, Any]:
    return _mapped("livemath/skill_graph", "compare", to="decide_task", **edit)


RUNS = [f"options_map_{i}" for i in range(1, 6)]


# ── what the edit builds ──────────────────────────────────────────────────


@pytest.mark.parametrize("state", [_s1, _s4], ids=["S1", "S4"])
def test_the_map_is_an_elements_step_a_fork_of_optional_runs_a_join_and_a_combine(state):
    state = state()
    assert validate_state(state) == []
    graph = _graph(state)
    assert map_of(graph, "options_map_fork") == "options_map"
    op = graph.nodes["options_map"]
    assert op.call == MAP_CALL and op.inputs[0] == "options"
    assert op.writes == tuple(f"{r}_element.txt" for r in RUNS)
    assert op.knobs == ("options_map__task",)
    assert state["knobs"]["options_map__task"]["value"].startswith(CHECK)
    fork = graph.nodes["options_map_fork"]
    assert fork.targets == fork.optional_targets == tuple(RUNS)
    for run in RUNS:
        node = graph.nodes[run]
        assert node.reads == (f"{run}_element.txt",) and node.writes == (f"{run}.txt",)
        assert node.prompt_template == MAP_ELEMENT_PROMPT
        assert [e.target for e in graph.out_edges(run)] == ["options_map_join"]
    combine = graph.nodes["options_map_combine_inputs"]
    assert {f"{r}.txt" for r in RUNS} <= set(combine.reads)
    line = next(l for l in pv.view_lines(state) if l.key == "options_map_fork")
    assert line.reader == "map"
    assert line.text.startswith("map(options_map_1 … options_map_5 over options@options, max 5)")
    assert 'told options_map_1__system_prompt "You are one step of a pipeline."; ' \
        f'options_map__task "{CHECK[:20]}' in line.text


def test_s4_the_combined_output_reaches_decide():
    graph = _graph(_s4())
    assert "options_map_combine" in graph.nodes["decide_prompt"].inputs or \
        "{{options_map_combine}}" in _s4()["knobs"]["decide_task"]["value"]


def test_the_lowering_hands_the_elements_step_its_list_and_knob_and_marks_runs_optional():
    state = _s4()
    declared = lowering.node_declarations(state)["options_map"]["snippet"]
    assert declared["inputs"][0] == "options" and declared["task_knob"] == "options_map__task"
    workflow = lowering.to_workflow(state)
    assert workflow.nodes["options_map_fork"].optional_targets == RUNS


def test_the_view_reads_as_exactly_the_graph_s_edges():
    """Every succession the drawn lines imply is an edge, and every edge is read
    (`test_pipeline_view._reading_problems`), on S1, S4 and a map on every step of the
    fuzz seeds that takes one."""
    import structure_fuzz as sf
    from test_pipeline_view import _reading_problems

    states = [_s1(), _s4()]
    for name in ("fuzz_draft_review", "fuzz_data_pass", "fuzz_fork_join"):
        seed = sf.cached_applier(name).seed_state
        for step in sorted(_graph(seed).nodes):
            edit = {"op": "map", "after": step, "over": step}
            if nn_edits.precheck(edit, seed) is None:
                states.append(nn_edits.apply(edit, seed))
    assert len(states) > 6
    for state in states:
        assert validate_state(state) == []
        assert _reading_problems(state) == (set(), set()), pv.render(state)


# ── refusals ──────────────────────────────────────────────────────────────


def _refused(edit: dict, state: dict) -> str:
    reason = nn_edits.precheck(edit, state)
    assert reason is not None, edit
    return reason


def test_refusals():
    state = nn_edits.apply({"op": "insert", "before": "compare", "kind": "code", "id": "options",
                            "sees": ["current_item.json"], "text": OPTIONS},
                           _state("livemath/skill_graph"))
    edit = {"op": "map", "after": "options", "over": "options"}
    assert _refused({**edit, "over": "ghost"}, state).startswith(
        "'over' names a list there after 'options': a fact, a file or a step whose output it "
        "is, not 'ghost'; there: ")
    assert _refused({**edit, "over": "decide"}, state).startswith("'over' names a list")
    assert _refused({**edit, "max": 9}, state) == (
        "'max' is a whole number from 2 to 8 (the most elements, one run each)")
    assert _refused({**edit, "max": 1}, state).startswith("'max' is a whole number")
    assert _refused({**edit, "after": "step_fork"}, state) == (
        "'step_fork' is a fork; map works on a step (agent, op or code)")
    assert _refused({**edit, "kind": "code"}, state).startswith(
        "'map' does not take 'kind'; it would be ignored. Its fields: after, over, text, ")
    assert _refused({**edit, "combine": {"kind": "agent", "role": "x"}}, state).startswith(
        "a new step does not take 'role'")
    assert _refused({**edit, "combine": {"kind": "gate"}}, state).startswith(
        "a new step's kind is one of")
    assert _refused({**edit, "id": "compare"}, state) == "node 'compare' already exists"
    assert _refused({**edit, "text": 3}, state) == "'text' is a string: what to do with one element"


def test_the_elements_of_a_list():
    """As a model writes them: a fenced JSON array, a bulleted list with wrapped lines and
    an introduction, lettered options, numbered steps, an object, prose."""
    assert list_elements('["A: x", "B: y"]') == ["A: x", "B: y"]
    assert list_elements('[{"label": "A"}, 2]') == ['{"label":"A"}', "2"]
    assert list_elements('Here they are:\n```json\n["a", "b", "c"]\n```\nHope that helps.'
                         ) == ["a", "b", "c"]  # the one fence in the text
    assert list_elements("~~~\n1. x\n2. y\n~~~") == ["1. x", "2. y"]
    assert list_elements('Sure, the options are ["a", "b"] as you asked.') == ["a", "b"]
    assert list_elements("- first\n  - its detail\n  - another\n- second\n\nThat is all.") == [
        "- first\n  - its detail\n  - another", "- second"]  # nested kept, outro left out
    assert list_elements("- see [1]\n- and [2]") == ["- see [1]", "- and [2]"]
    assert list_elements("1. a\n2. b\n   more of b\n\n   still b\n3. c") == [
        "1. a", "2. b\nmore of b\nstill b", "3. c"]
    assert list_elements('```json\n[\n  "a",\n  "b",\n  "c"\n]\n```') == ["a", "b", "c"]
    assert list_elements("The claims:\n- the series diverges, since its terms\n  do not tend "
                         "to zero\n- it converges\n* by comparison") == [
        "- the series diverges, since its terms\ndo not tend to zero", "- it converges",
        "* by comparison"]
    assert list_elements("A) Diverges\nB) Converges conditionally\nC) Oscillates") == [
        "A) Diverges", "B) Converges conditionally", "C) Oscillates"]
    assert list_elements("1. Read it.\n2. Solve it.\n   Show the work.") == [
        "1. Read it.", "2. Solve it.\nShow the work."]
    assert list_elements('{\n  "a": 1,\n  "b": 2\n}') == ['{\n  "a": 1,\n  "b": 2\n}']
    assert list_elements("The answer is A.\nIt is the only one.") == [
        "The answer is A.", "It is the only one."]
    assert list_elements(["a", {"b": 1}]) == ["a", '{"b":1}']
    assert list_elements("") == []


# ── both runtimes ─────────────────────────────────────────────────────────


class _Model:
    """Answers by node, logging (node, prompt)."""

    def __init__(self, log: list, node: str = ""):
        self.log, self.node = log, node

    def for_node(self, node_id: str, role: str = "") -> "_Model":
        return _Model(self.log, node_id)

    def with_tool_policy(self, policy: str) -> "_Model":
        return self

    def with_wall(self, wall: Any) -> "_Model":
        return self

    def with_budget(self, budget: Any) -> "_Model":
        return self

    def context(self) -> dict:
        return {}

    def route(self) -> dict:
        return {"worker_backend": "dsh"}

    def complete(self, prompt: str, **_: Any) -> str:
        self.log.append((self.node, prompt))
        if self.node.startswith("options_map_") and self.node != "options_map_combine":
            return f"verdict of {self.node}"
        return "<answer>B</answer>"


def _walk(name: str, state: dict, tmp_path: Path, monkeypatch) -> tuple[Any, list]:
    _patch_task(monkeypatch)
    log: list = []
    run = tmp_path / "walker"
    run.mkdir()
    return _mode(name, state).run(run, model=_Model(log)), log


def _graded_b(records: list[dict]) -> None:
    assert len(records) == ITEMS
    for record in records:
        assert record["error_kind"] is None, record
        assert record["score"] == (1.0 if str(record["id"]) in LABEL_B else 0.0), record


def _one_run_per_option(prompts: dict[str, list[str]]) -> None:
    """Each item's four options reach four runs, one each, and the fifth runs never; the
    combine sees each run's result, the fifth said missing."""
    for index, run in enumerate(RUNS[:4]):
        assert len(prompts[run]) == ITEMS, run
        assert all(f"## Element {index + 1} of 4\n" in p and CHECK in p for p in prompts[run])
    assert prompts.get(RUNS[4], []) == []
    assert len(prompts["options_map_combine"]) == ITEMS
    for prompt in prompts["options_map_combine"]:
        assert WEIGH in prompt and "### options_map_5.txt\n(not written" in prompt
        assert "ran 4 of 4 elements (max 5)" in prompt


@pytest.mark.parametrize("name, state", [("livemath/solver", _s1),
                                         ("livemath/skill_graph", _s4)], ids=["S1", "S4"])
def test_walker_runs_the_step_once_per_option_and_combines(name, state, tmp_path, monkeypatch):
    result, log = _walk(name, state(), tmp_path, monkeypatch)
    assert result.exit_ok, result.error
    _graded_b(result.facts["item_scores"])
    # The unused fifth run is no missing step: it had nothing to do.
    assert all("missing_steps" not in r for r in result.facts["item_scores"])
    prompts: dict[str, list[str]] = {}
    for node, prompt in log:
        prompts.setdefault(node, []).append(prompt)
    _one_run_per_option(prompts)
    for index, run in enumerate(RUNS[:4]):
        assert all(f"verdict of {run}" in p for p in prompts["options_map_combine"])
    if name == "livemath/skill_graph":
        assert all("## Options map combine\n<answer>B</answer>" in p.replace("  ", "")
                   for p in prompts["decide"])


@needs_spine
@pytest.mark.parametrize("name, state", [("livemath/solver", _s1),
                                         ("livemath/skill_graph", _s4)], ids=["S1", "S4"])
def test_spine_runs_the_step_once_per_option_and_combines(name, state, tmp_path):
    records, dsh = _dispatched(name, state(), {WEIGH: ["<answer>B</answer>"],
                                               CHECK: ["a verdict on this option"],
                                               "": ["<answer>B</answer>"]}, tmp_path)
    _graded_b(records)
    assert all("missing_steps" not in r for r in records)
    assert dsh.calls(CHECK) == 4 * ITEMS and dsh.calls(WEIGH) == ITEMS
    combines = dsh.prompts(WEIGH)
    assert all(p.count("a verdict on this option") == 4 for p in combines)
    assert all("### options_map_5.txt\n(not written" in p for p in combines)
    assert all("ran 4 of 4 elements (max 5)" in p for p in combines)
    checks = dsh.prompts(CHECK)
    for index in range(1, 5):
        assert sum(f"## Element {index} of 4\n" in p for p in checks) == ITEMS
    if name == "livemath/skill_graph":
        decides = [p for p in dsh.prompts("") if "Weigh the notes above" in p]
        assert len(decides) == ITEMS and all("Options map combine" in p for p in decides)


@pytest.mark.parametrize("runtime", ["walker", pytest.param("spine", marks=needs_spine)])
def test_a_list_longer_than_max_runs_the_first_max_and_says_so(runtime, tmp_path, monkeypatch):
    """Four options, max 3: the first three run, the item is scored, and the combine and
    the item's record say "ran 3 of 4 elements (max 3)" -- a design limit, not an error."""
    state = _s1(max=3)
    if runtime == "walker":
        result, log = _walk("livemath/solver", state, tmp_path, monkeypatch)
        assert result.exit_ok, result.error
        records = result.facts["item_scores"]
        runs = [n for n, _p in log if n.startswith("options_map_") and n[-1].isdigit()]
        combines = [p for n, p in log if n == "options_map_combine"]
    else:
        records, dsh = _dispatched("livemath/solver", state, {
            WEIGH: ["<answer>B</answer>"], CHECK: ["a verdict"], "": ["<answer>B</answer>"]},
            tmp_path)
        runs = ["run"] * dsh.calls(CHECK)
        combines = dsh.prompts(WEIGH)
    _graded_b(records)
    assert len(runs) == 3 * ITEMS and len(combines) == ITEMS
    assert all("ran 3 of 4 elements (max 3)" in p for p in combines)
    for record in records:
        assert "missing_steps" not in record
    if runtime == "walker":
        _every_visit_and_call_recorded(records)


def _every_visit_and_call_recorded(records: list[dict], runs: int = 3) -> None:
    """Each item's record holds the elements step's visit ("ran K of N elements (max K)")
    and every model call: each run and the combine."""
    for record in records:
        assert any(f"ran {runs} of" in str(step.get("output")) and f"(max {runs})" in
                   str(step.get("output")) for step in record["steps"]), record["steps"]
        called = sorted(e["node"] for e in record.get("exchanges") or ()
                        if e["node"].startswith("options_map_"))
        assert called == sorted([*(f"options_map_{i}" for i in range(1, runs + 1)),
                                 "options_map_combine"]), called


@needs_spine
def test_spine_records_every_visit_and_call_of_a_map(tmp_path):
    """Eight runs at once on every item, five items at once: the spine's records keep every
    visit and call (#692 locks the item's record against parallel writers)."""
    # Eight runs at once on every item, five items at once: many writers per record.
    eight = OPTIONS.replace("for c in item['choices']]", "for c in item['choices'] * 2]")
    state = nn_edits.apply({"op": "insert", "before": "solve", "kind": "code", "id": "options",
                            "sees": ["current_item.json"], "text": eight},
                           _state("livemath/solver"))
    state = nn_edits.apply({"op": "map", "after": "options", "over": "options", "text": CHECK,
                            "max": 8, "combine": {"kind": "agent", "text": WEIGH}}, state)
    state = nn_edits.configure(state, "data_parallelism", "5")
    # A lost entry is a race: three runs, so a regression shows even under light load.
    for attempt in range(3):
        records, _dsh = _dispatched("livemath/solver", state, {
            WEIGH: ["<answer>B</answer>"], CHECK: ["a verdict"], "": ["<answer>B</answer>"]},
            tmp_path / f"attempt{attempt}")
        _every_visit_and_call_recorded(records, 8)


def _dispatched(name: str, state: dict, plan: dict, tmp_path: Path, *, ok: bool = True
                ) -> tuple[list, Scripted]:
    """The mode state on the spine drain, its agents dispatched to workers (`worker_node`,
    which refuses a run whose element was not written) against a scripted dsh."""
    import os
    import subprocess

    from refactory_dsh import workers
    from test_livemath_blocks import DRAIN_AGENT, FIXTURE_DIR, NODE, PYTHON_BIN, REPO_ROOT

    dsh = Scripted(tmp_path / "dsh", plan)
    run = tmp_path / "spine"

    def dispatched(workflow: Any) -> Any:
        return workers.lower_workflow(workflow, tmp_path / "specs", dsh.client().context(),
                                      lowering.node_declarations(state))

    run.mkdir()
    lowering.write_declarations(state, run)
    graph_path, _ = lowering.write_mode_graph(state, tmp_path / "graphs", name="map",
                                              transform=dispatched)
    env = {key: value for key, value in os.environ.items() if key != "REFACTORY_WORKER_CONTEXT"}
    env.update(REFACTORY_WORKDIR=str(run), PYTHONPATH=str(Path(__file__).resolve().parent),
               LIVEMATH_DATA_PATH=str(FIXTURE_DIR), PATH=f"{PYTHON_BIN}:{env.get('PATH', '')}")
    # The run's knobs, as a DSH run hands them to every node command (`dsh_runtime`).
    env.update(lowering.knob_environment(state))
    done = subprocess.run([str(NODE), "--import", "tsx/esm", str(DRAIN_AGENT), str(graph_path),
                           str(run), f"{PYTHON_BIN}/python", "120000"],
                          capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300, env=env)
    # Every item failing fails the run: what a list longer than 'max' on every item does.
    assert (done.returncode == 0) is ok, (
        f"stdout: {done.stdout[-2000:]}\nstderr: {done.stderr[-3000:]}")
    return json.loads((run / "facts.json").read_text())["item_scores"], dsh


# ── why a map is no nested data pass ──────────────────────────────────────


def _nested() -> nn.Graph:
    """A data pass whose body holds another data node over two inline items."""
    graph = nn.Graph()
    graph.add(_data("entry", "inner"))
    graph.add(nn.Node("entry", "op", call="loop_ops:noop"))
    graph.add(nn.Node("inner", "data", subgraph_entry="each", subgraph_exit="each",
                      inline_items=({"id": "a"}, {"id": "b"}), writes=("inner_item.json",)))
    graph.add(nn.Node("each", "op", call="loop_ops:noop"))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("entry", "inner")
    graph.connect("data", "report")
    return graph


@pytest.mark.parametrize("runtime", ["walker", pytest.param("spine", marks=needs_spine)])
def test_neither_runtime_nests_a_data_pass(runtime, tmp_path):
    """The validator takes it, but the walker runs the inner data node as an op (no
    callable) and the spine refuses a pass while one is iterating (`Spine` tracks one
    pass): so a map is built from a fork, which both run."""
    graph = _nested()
    assert validate_state({"graph": graph.to_state()}) == []
    dsh = Scripted(tmp_path / "dsh", {"": ["right"]})
    with pytest.raises(AssertionError) as failed:
        if runtime == "walker":
            _walker(graph, tmp_path / "w", dsh)
        else:
            _spine(graph, tmp_path / "s", tmp_path, dsh)
    said = str(failed.value)
    assert ("empty callable target" in said if runtime == "walker"
            else "inner cannot start: data is iterating" in said), said


def test_an_edit_to_one_part_of_a_map_is_refused():
    """A map's runs take one element each, so the fuzz found a serialized map fork running
    a branch with no element: every core operator naming a part of a map (its elements
    step, fork or runs) is refused, saying what to edit instead."""
    state = _s4()
    reason = ("'{}' is part of the map 'options_map', whose runs each take one element; "
              "change what they do with its knob options_map__task, work on the step "
              "before the map or its combine step, or remove the map by its elements step")
    for edit, part in (({"op": "serialize", "fork": "options_map_fork"}, "options_map_fork"),
                       ({"op": "insert", "after": "options_map_2"}, "options_map_2"),
                       ({"op": "branch", "fork": "options_map_fork"}, "options_map_fork"),
                       ({"op": "remove", "step": "options_map_join"}, "options_map_join"),
                       ({"op": "insert", "after": "options_map"}, "options_map"),
                       ({"op": "map", "after": "options_map_combine", "over": "options_map_3"},
                        "options_map_3"),
                       ({"op": "merge", "steps": ["options_map_5", "options_map_combine"]},
                        "options_map_5")):
        assert _refused(edit, state) == reason.format(part), edit
    assert nn_edits.precheck({"op": "insert", "after": "options_map_combine"}, state) is None
    combine = ("'options_map_combine' is the combine of the map 'options_map'; a step goes "
               "after it (insert after, loop, route), or remove it to take the map out; change "
               "what it does with its own knobs")
    for edit in ({"op": "insert", "before": "options_map_combine"},
                 {"op": "split", "step": "options_map_combine"},
                 {"op": "rewire", "step": "options_map_combine", "sees": ["task"]},
                 {"op": "parallelize", "steps": ["options_map_combine"], "copies": 1}):
        assert _refused(edit, state) == combine, edit
    assert _refused({"op": "loop", "target": "options_map", "gate": {
        "id": "g", "kind": "gate", "evaluator": "structure_fuzz_graphs:proceed"}},
        state).startswith("'options_map' is part of the map")


# ── a map run again, short lists, removal, the knob's slots ────────────────


def _looped() -> dict[str, Any]:
    """S1 with an agent listing things before solve, a map over its list, and a loop judging
    the combine that sends the work back to the lister: a pass of the map runs again."""
    state = nn_edits.apply({"op": "insert", "before": "solve", "kind": "agent", "id": "lister",
                            "text": "List things."}, _state("livemath/solver"))
    state = nn_edits.apply({"op": "map", "after": "lister", "over": "lister", "text": "Check",
                            "combine": {"kind": "agent", "text": "Combine."}}, state)
    return nn_edits.apply({"op": "loop", "step": "lister_map_combine", "back_to": "lister",
                           "judge": "agent", "text": "ok?", "max_iterations": 1}, state)


class _Lister(_Model):
    """Three things on an item's first list, one on its second; the judge sends back once."""

    def for_node(self, node_id: str, role: str = "") -> "_Lister":
        return _Lister(self.log, node_id)

    def complete(self, prompt: str, **_: Any) -> str:
        self.log.append((self.node, prompt))
        seen = sum(node == self.node for node, _p in self.log)
        if self.node == "lister":
            return "x1\nx2\nx3" if seen % 2 else "only"
        if self.node == "lister_map_combine_check_judge":
            return "reloop: again" if seen % 2 else "proceed"
        if self.node.startswith("lister_map_") and self.node[-1].isdigit():
            return f"VERD of {self.node}"
        return "<answer>B</answer>"


def _second_pass(runs: list[str], combines: list[str]) -> None:
    """Per item: the first pass runs three elements, the second one; the second combine
    sees one result and no old one."""
    assert len(combines) == 2 * ITEMS
    assert sorted(p.count("VERD") for p in combines) == [1] * ITEMS + [3] * ITEMS
    assert sum("ran 1 of 1 elements (max 4)" in p for p in combines) == ITEMS
    assert sorted(runs) == sorted([*["## Element 1 of 3\nx1", "## Element 2 of 3\nx2",
                                     "## Element 3 of 3\nx3"] * ITEMS,
                                   *["## Element 1 of 1\nonly"] * ITEMS])


def test_walker_a_map_run_again_never_shows_an_old_element_or_result(tmp_path, monkeypatch):
    import re

    _patch_task(monkeypatch)
    log: list = []
    run = tmp_path / "walker"
    run.mkdir()
    result = _mode("livemath/solver", _looped()).run(run, model=_Lister(log))
    assert result.exit_ok, result.error
    runs = [re.search(r"## Element \d+ of \d+\n\S+", p).group(0) for n, p in log
            if n.startswith("lister_map_") and n[-1].isdigit()]
    _second_pass(runs, [p for n, p in log if n == "lister_map_combine"])
    assert all("missing_steps" not in r for r in result.facts["item_scores"])


@needs_spine
def test_spine_a_map_run_again_never_shows_an_old_element_or_result(tmp_path):
    import re

    # The dsh plan answers by key across the run: items one after another (parallelism 1).
    state = nn_edits.configure(_looped(), "data_parallelism", "1")
    records, dsh = _dispatched("livemath/solver", state, {
        "ok?": ["reloop: again", "proceed"] * ITEMS, "Combine.": ["CC"], "Check": ["VERD"],
        "List things": ["x1\nx2\nx3", "only"] * ITEMS, "": ["<answer>B</answer>"]}, tmp_path)
    runs = [re.search(r"## Element \d+ of \d+\n\S+", p).group(0) for p in dsh.prompts("Check")]
    _second_pass(runs, dsh.prompts("Combine."))
    assert all("missing_steps" not in r for r in records)


#: Each item's options cut short, so some runs have nothing to do (item 202503:1: none).
SHORT = ("import json\n"
         "def main(inp):\n"
         "    item = json.loads(inp['files']['current_item.json'])\n"
         "    n = {'202501:1': 4, '202501:2': 1, '202503:1': 0, '202503:2': 4,\n"
         "         '202503:3': 2}[str(item.get('id'))]\n"
         "    return {'facts': {'options': json.dumps([c['label'] + ': ' + c['text']\n"
         "                                             for c in item['choices']][:n])}}\n")


def test_a_short_list_is_no_missing_answer_to_the_optimizer(tmp_path, monkeypatch):
    """Runs the list had no element for are not needed: no item lists them as missing, the
    proposer's item results count no item without an answer, and an analyst's trace says
    no "No answer from step" -- an item answered wrong stays answered wrong."""
    import test_map_op as module
    from refactory_dsh.optim import analysts, base

    monkeypatch.setattr(module, "OPTIONS", SHORT)
    result, log = _walk("livemath/solver", _s1(), tmp_path, monkeypatch)
    assert result.exit_ok, result.error
    records = result.facts["item_scores"]
    assert all("missing_steps" not in r and r["error_text"] is None for r in records)
    ran = Counter(n for n, _p in log if n.startswith("options_map_") and n[-1].isdigit())
    assert sum(ran.values()) == 4 + 1 + 0 + 4 + 2
    combines = [p for n, p in log if n == "options_map_combine"]
    assert sorted(re.search(r"ran (\d) of \d", p).group(1) for p in combines) == \
        ["0", "1", "2", "4", "4"]
    evidence = "\n".join(base._item_evidence(records))
    wrong = sum(1 for r in records if not r["passed"])
    assert f"{wrong} answered wrong" in evidence and "no answer" not in evidence
    for record in records:
        assert "No answer from step" not in analysts.item_text(record)


@pytest.mark.parametrize("state, original", [
    (_s1, lambda: nn_edits.apply({"op": "insert", "before": "solve", "kind": "code",
                                  "id": "options", "sees": ["current_item.json"],
                                  "text": OPTIONS}, _state("livemath/solver"))),
    (_s4, lambda: nn_edits.apply({"op": "insert", "before": "compare", "kind": "code",
                                  "id": "options", "sees": ["current_item.json"],
                                  "text": OPTIONS}, _state("livemath/skill_graph")))],
    ids=["S1", "S4"])
@pytest.mark.parametrize("step", ["options_map", "options_map_combine"])
def test_remove_by_the_elements_step_or_the_combine_takes_the_map_out_whole(
        state, original, step):
    """As before the map: its nodes and knobs gone, the step before it leading to the step
    after it, its output taken out of the prompt it was put in (S4: decide_task)."""
    removed = nn_edits.apply({"op": "remove", "step": step}, state())
    assert validate_state(removed) == []
    assert _same(removed, original())


def _same(a: dict, b: dict) -> bool:
    def key(state: dict) -> tuple:
        graph = state["graph"]
        return (graph["nodes"], sorted(json.dumps(e, sort_keys=True) for e in graph["edges"]),
                graph.get("start"), state.get("knobs"))
    return key(a) == key(b)


def test_a_slot_in_the_map_s_knob_is_filled_from_the_facts(tmp_path, monkeypatch):
    """The elements step fills its knob's ``{{name}}`` slots, as an agent's prompt is
    filled: the validator takes the slot (V20, `operators.elements` fills facts) and each
    element's prompt carries the fact's value."""
    from refactory_dsh.nn.validate import consumed

    state = nn_edits.apply({"op": "set_knob", "name": "options_map__task",
                            "value": "The question: {{solver_prompt}} Is this one right?"},
                           _s1())
    assert validate_state(state) == []
    values = {name: entry["value"] for name, entry in state["knobs"].items()}
    # The slot is one of the step's inputs, so the validator holds it to its maker (V15).
    assert ("fact", "solver_prompt") in consumed(_graph(state).nodes["options_map"], values)
    result, log = _walk("livemath/solver", state, tmp_path, monkeypatch)
    assert result.exit_ok, result.error
    runs = [p for n, p in log if n == "options_map_1"]
    assert len(runs) == ITEMS and all("The question: History:" in p for p in runs)
    assert sum("Which of the following series" in p for p in runs) == 1


def test_a_loop_s_hint_never_names_a_map_s_run():
    reason = nn_edits.precheck({"op": "loop", "step": "decide", "back_to": "options",
                                "judge": "agent"}, _s4())
    assert reason.endswith("act on it: options_map_combine"), reason


def test_every_core_operator_aimed_at_a_map_is_refused_or_leaves_it_removable():
    """Drawn as the fuzz draws them, every core operator aimed at the map's parts, its
    combine and the steps around it: refused, or the map still comes out whole by its
    elements step -- or says plainly which later step would lose its input, or which loop's
    gate to remove first."""
    import random

    import structure_fuzz as sf
    from refactory_dsh.nn import core_ops

    state = _s4()
    graph = _graph(state)
    targets = sorted(n for n in graph.nodes if n.startswith("options") or n in (
        "compare", "prepare"))
    generator = sf.EditGenerator(random.Random("map-parts"))
    taken = 0
    for op in core_ops.CORE_OPS:
        for target in targets:
            for _ in range(3):
                edit = generator.core_op(state, op, target)
                if nn_edits.precheck(edit, state) is not None:
                    continue
                try:
                    after = nn_edits.apply(edit, state)
                except nn_edits.EditError:
                    continue
                if validate_state(after) or "options_map" not in after["graph"]["nodes"]:
                    continue
                taken += 1
                reason = nn_edits.precheck({"op": "remove", "step": "options_map"}, after)
                if reason is not None:
                    # Said plainly: a later step reading its output, or a loop's gate to
                    # remove first.
                    assert ("leaves steps without an input only it makes" in reason
                            or "remove that loop's gate first" in reason), (edit, reason)
                    continue
                removed = nn_edits.apply({"op": "remove", "step": "options_map"}, after)
                assert validate_state(removed) == [], edit
                assert not set(removed["graph"]["nodes"]) & set(_maps_nodes(after)), edit
    assert taken >= 20, taken


def _maps_nodes(state: dict) -> list[str]:
    from refactory_dsh.nn.core_ops import _maps

    return _maps(_graph(state), state)["options_map"]["nodes"]


@pytest.mark.parametrize("kind", ["code", "agent"])
def test_removing_a_map_a_later_step_reads_says_which_step_loses_its_input(kind):
    """A later step given the combine's output (a code step's input, or a slot in an
    agent's own prompt) is said by name; its slot is never taken out of its prompt."""
    state = nn_edits.apply({"op": "insert", "after": "options_map_combine", "kind": kind,
                            "id": "uses_it", "text": "Use {{options_map_combine}}."}
                           if kind == "agent" else
                           {"op": "insert", "after": "options_map_combine", "kind": kind,
                            "id": "uses_it", "sees": ["options_map_combine"]}, _s4())
    reason = nn_edits.precheck({"op": "remove", "step": "options_map"}, state)
    assert reason.startswith("removing map 'options_map' leaves steps without an input only "
                             "it makes: 'uses_it' needs "), reason
    assert "options_map_combine" in reason and "set_knob" not in reason


def test_remove_takes_out_the_slot_and_heading_wherever_they_were_moved():
    """The optimizer moved the combine's section to the top of decide_task, and wrote the
    slot inline once more: removing the map leaves decide_task as it was before the map."""
    state = _s4()
    task = state["knobs"]["decide_task"]["value"]
    section = "## Options map combine\n{{options_map_combine}}"
    assert section in task
    moved = section + "\n\n" + task.replace(section, "").rstrip() + \
        " Mind {{options_map_combine}}."
    state = nn_edits.apply({"op": "set_knob", "name": "decide_task", "value": moved}, state)
    removed = nn_edits.apply({"op": "remove", "step": "options_map"}, state)
    before = nn_edits.apply({"op": "insert", "before": "compare", "kind": "code",
                             "id": "options", "sees": ["current_item.json"], "text": OPTIONS},
                            _state("livemath/skill_graph"))
    assert removed["knobs"]["decide_task"]["value"] == \
        before["knobs"]["decide_task"]["value"] + " Mind."


def test_map_runs_take_sealed_tools_as_insert_s_step_does():
    """'tools' on a map: each run gets the sealed list in a fresh workspace and reads its
    element's file itself; a list outside the sealed set is refused as insert refuses it."""
    from refactory_dsh.nn.core_ops import MAP_ELEMENT_TOOLS_PROMPT

    state = _s4(tools=["bash"])
    assert validate_state(state) == []
    graph = _graph(state)
    for run in RUNS:
        node = graph.nodes[run]
        assert node.tools == ("bash",) and node.tool_policy is None
        assert node.workspace["kind"] == "scratch" and node.reads == (f"{run}_element.txt",)
        assert node.prompt_template == MAP_ELEMENT_TOOLS_PROMPT.format(
            file=f"{run}_element.txt")
    assert graph.nodes["options_map_combine"].tools == ()
    base = {"op": "map", "after": "options", "over": "options"}
    plain = nn_edits.apply({"op": "insert", "before": "compare", "kind": "code",
                            "id": "options", "sees": ["current_item.json"], "text": OPTIONS},
                           _state("livemath/skill_graph"))
    insert = nn_edits.precheck({"op": "insert", "after": "options", "tools": ["curl"]}, plain)
    assert _refused({**base, "tools": ["curl"]}, plain) == insert
    assert insert.startswith("'tools' lists the tools a new step uses")


def test_only_a_map_s_runs_read_its_element_files():
    """The fuzz found a step added by hand reading an element file a short list never
    wrote: the validator refuses it (V6), naming the map step."""
    state = nn_edits.apply({"op": "add_node", "position": {"after": "options_map_combine"},
                            "node": {"id": "peek", "kind": "agent", "prompt_template": "Look.",
                                     "reads": ["options_map_4_element.txt"],
                                     "writes": ["peek.txt"]}}, _s4())
    assert [(i.code, i.node) for i in validate_state(state)
            if "map step" in i.message] == [("V6", "peek")]


def test_a_fallback_goes_on_a_map_s_combine_and_never_inside_the_map():
    """As COMBINE_OPS says: a fallback for the combine (after it) is taken, and the map
    still comes out whole with it; one for the elements step or a run is refused."""
    state = _s4()
    for step in ("options_map", "options_map_2"):
        assert _refused({"op": "fallback", "step": step}, state).startswith(
            f"'{step}' is part of the map 'options_map'")
    fell = nn_edits.apply({"op": "fallback", "step": "options_map_combine",
                           "kind": "agent", "text": "Weigh them simply."}, state)
    assert validate_state(fell) == [] and _graph(fell).falls_back("options_map_combine")
    removed = nn_edits.apply({"op": "remove", "step": "options_map"}, fell)
    assert validate_state(removed) == []
    assert _same(removed, nn_edits.apply(
        {"op": "insert", "before": "compare", "kind": "code", "id": "options",
         "sees": ["current_item.json"], "text": OPTIONS}, _state("livemath/skill_graph")))


def test_no_agent_requires_a_map_run_s_result():
    """The fuzz found an agent added by hand reading a run's result, which a short list
    never writes: the validator refuses it (V6); the combine reads them through its collect
    step, an op that says one missing."""
    state = nn_edits.apply({"op": "add_node", "position": {"after": "options_map_combine"},
                            "node": {"id": "peek", "kind": "agent", "prompt_template": "Look.",
                                     "reads": ["options_map_2.txt"], "writes": ["peek.txt"]}},
                           _s4())
    assert [(i.code, i.node) for i in validate_state(state)
            if "results of map step" in i.message] == [("V6", "peek")]
    assert validate_state(_s4()) == []


# ── a map the driver runs (no data pass, no dispatched workers) ─────────────


def _driven(cap: int) -> dict[str, Any]:
    """The structure fuzz's draft -> review -> finalize with a map after draft over draft's
    reply, ``cap`` runs: its agents are handed to the driver on the spine (`wf_next`), not
    to dispatched workers as LiveMath's are."""
    import structure_fuzz as sf

    edit = {"op": "map", "after": "draft", "over": "draft", "max": cap}
    return nn_edits.apply(edit, sf.cached_applier("fuzz_draft_review").seed_state)


def _driven_run(runtime: str, state: dict, draft: str, tmp_path: Path, monkeypatch
                ) -> tuple[dict[str, list[str]], dict[str, Any], Path]:
    """Each agent's prompts, the run's facts and its directory, on ``runtime``, with draft
    answering ``draft`` (the list) and every other agent its structure-fuzz answer."""
    import structure_fuzz as sf
    from test_livemath_blocks import _spine as spine_drain

    graph = _graph(state)
    answers = {**sf._answers(graph), "draft": draft}
    run = tmp_path / runtime
    prompts: dict[str, list[str]] = {}
    if runtime == "walker":
        model = sf.ScriptedModel(answers)
        result = sf.run_walker(sf.seeds()["fuzz_draft_review"], state, run, model)
        assert result.exit_ok and not result.error, result.error
        for node, prompt, _system in model.log:
            prompts.setdefault(node, []).append(prompt)
        return prompts, result.facts, run
    record = tmp_path / "delivered.jsonl"
    monkeypatch.setenv("DRAIN_RECORD", str(record))
    assert spine_drain(graph, run, tmp_path, agent_responses=answers, mode_state=state) == 0
    for line in record.read_text().splitlines():
        entry = json.loads(line)
        if "prompt" in entry:
            prompts.setdefault(entry["node"], []).append(entry["prompt"])
    return prompts, json.loads((run / "facts.json").read_text()), run


@pytest.mark.parametrize("runtime", ["walker", pytest.param("spine", marks=needs_spine)])
@pytest.mark.parametrize("elements, cap", [(2, 4), (1, 4), (0, 3), (5, 3)],
                         ids=["short", "one", "empty", "over-max"])
def test_a_driven_map_runs_only_its_elements_runs_on_both_runtimes(
        runtime, elements, cap, tmp_path, monkeypatch):
    """A run the list had no element for is never handed to a model or the driver: it is
    recorded missing and not needed before any call (its element file was never written),
    on the walker and on the spine alike, and no item lists it as a missing answer. Each run
    that ran saw its own element; the combine ran once and was told how many ran."""
    from refactory_dsh.items import missing_steps
    from refactory_dsh.nn.operators import NOT_NEEDED_DIR

    items = [f"element-{index}" for index in range(1, elements + 1)]
    prompts, facts, run = _driven_run(runtime, _driven(cap), json.dumps(items),
                                      tmp_path, monkeypatch)
    runs = [f"draft_map_{index}" for index in range(1, cap + 1)]
    ran = min(elements, cap)
    assert [r for r in runs if r in prompts] == runs[:ran], sorted(prompts)
    for index, name in enumerate(runs[:ran]):
        (prompt,) = prompts[name]
        assert f"## Element {index + 1} of {ran}\n{items[index]}" in prompt
    assert len(prompts["draft_map_combine"]) == 1
    said = f"ran {ran} of {elements} elements (max {cap})"
    assert facts["draft_map"] == said
    assert said in prompts["draft_map_combine"][0]
    for name in runs[ran:]:
        assert (run / NOT_NEEDED_DIR / f"{name}.txt").is_file(), name
        assert (run / "reviews" / f"{name}-missing.txt").is_file(), name
    assert missing_steps(run) == {}
    assert len(prompts["finalize"]) == 1


@needs_spine
def test_the_fuzz_s_map_over_a_draft_runs_alike_on_the_walker_and_the_spine():
    """The map the structure fuzz builds by default (after draft, over draft, max 4) on a
    one-line draft: the spine drain once failed it ("no-tools input is not a readable
    file") when it handed the second run, which had no element, to the driver."""
    import structure_fuzz as sf

    applier = sf.cached_applier("fuzz_draft_review")
    state = nn_edits.apply({"op": "map", "after": "draft", "over": "draft"},
                           applier.seed_state)
    assert sf.check_state(applier, state, walker=True, spine=True) == []
