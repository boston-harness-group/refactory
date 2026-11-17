"""The ``map`` operator: a new agent step run once per element of a per-item list, its
results combined into one output for the next step.

Neither runtime runs a data pass inside another (`test_neither_runtime_nests_a_data_pass`),
so a map is not one. It is built from what both runtimes already run: an ``elements`` op
(`operators.MAP_CALL`) writes one prompt per element (the map's one text knob, then the
element); a fork runs one optional branch per element up to ``max``, each an agent that
reads its element's file, so a branch with no element is refused before any model call and
is missing, never a failure; a join; and the combine step, which sees every result (a
missing one said so) through a collect step.

- **What the edit builds** on LiveMath S1 (``solver``) and S4 (``skill_graph``), each with a
  code step before it that lists the item's options; the view's one line; the lowering;
  the refusals; the elements op's lists.
- **Both runtimes** (the walker; the spine with dispatched workers) on S1 and S4 with a
  scripted model: one run per option, none for the fifth branch, the combine sees each
  result and its output reaches decide (S4); a list longer than ``max`` fails the item.
"""
from __future__ import annotations

import json
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


def _state(name: str) -> dict[str, Any]:
    return registry.resolve(Path("."), name)[1]().state_dict()


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
        seed = sf.Applier(sf.seeds()[name]).seed_state
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
    assert list_elements('["A: x", "B: y"]') == ["A: x", "B: y"]
    assert list_elements('[{"label": "A"}, 2]') == ['{"label": "A"}', "2"]
    assert list_elements("first\n\n  second  \n") == ["first", "second"]
    assert list_elements(["a", {"b": 1}]) == ["a", '{"b": 1}']
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


@pytest.mark.parametrize("name, state", [("livemath/solver", _s1),
                                         ("livemath/skill_graph", _s4)], ids=["S1", "S4"])
def test_walker_runs_the_step_once_per_option_and_combines(name, state, tmp_path, monkeypatch):
    result, log = _walk(name, state(), tmp_path, monkeypatch)
    assert result.exit_ok, result.error
    _graded_b(result.facts["item_scores"])
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
    assert dsh.calls(CHECK) == 4 * ITEMS and dsh.calls(WEIGH) == ITEMS
    combines = dsh.prompts(WEIGH)
    assert all(p.count("a verdict on this option") == 4 for p in combines)
    assert all("### options_map_5.txt\n(not written" in p for p in combines)
    checks = dsh.prompts(CHECK)
    for index in range(1, 5):
        assert sum(f"## Element {index} of 4\n" in p for p in checks) == ITEMS
    if name == "livemath/skill_graph":
        decides = [p for p in dsh.prompts("") if "Weigh the notes above" in p]
        assert len(decides) == ITEMS and all("Options map combine" in p for p in decides)


@pytest.mark.parametrize("runtime", ["walker", pytest.param("spine", marks=needs_spine)])
def test_a_list_longer_than_max_fails_the_item_naming_the_cap(runtime, tmp_path, monkeypatch):
    state = _s1(max=3)
    if runtime == "walker":
        result, log = _walk("livemath/solver", state, tmp_path, monkeypatch)
        assert not result.exit_ok  # every item failing fails the run
        records = result.facts["item_scores"]
        assert not [node for node, _p in log if node.startswith("options_map")]
    else:
        records, dsh = _dispatched("livemath/solver", state, {"": ["<answer>B</answer>"]},
                                   tmp_path, ok=False)
        assert dsh.calls() == 0
    assert len(records) == ITEMS
    failed = [r for r in records if r.get("error_text")]
    # The all-failed rule ends the spine's pass at its last item (`items`), recorded without it.
    assert len(failed) >= ITEMS - 1 and not any(r.get("prediction") for r in records)
    for record in failed:
        assert "'options' has 4 elements, more than its max of 3" in record["error_text"], record


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
              "change what they do with its knob options_map__task, or work on the step "
              "before the map or its combine step")
    for edit, part in (({"op": "serialize", "fork": "options_map_fork"}, "options_map_fork"),
                       ({"op": "insert", "after": "options_map_2"}, "options_map_2"),
                       ({"op": "branch", "fork": "options_map_fork"}, "options_map_fork"),
                       ({"op": "remove", "step": "options_map"}, "options_map"),
                       ({"op": "merge", "steps": ["options_map_5", "options_map_combine"]},
                        "options_map_5")):
        assert _refused(edit, state) == reason.format(part), edit
    assert nn_edits.precheck({"op": "insert", "after": "options_map_combine"}, state) is None
