"""The ``fallback`` operator: a step that answers instead when a per-item step genuinely
fails -- it raises, or its session crashes and its retries are spent -- and never when it
runs out of time.

It reuses what the graph already has for a step that may go without an answer: the step
is recorded missing (``reviews/<step>-missing.txt``, `items.missing_steps`) rather than
failing its item, and a gate after it (`graph.FALLBACK_GATE_CALL`) reads why: ``fallback``
for a failure, ``proceed`` for an answer or a timeout. The fallback step writes the failed
step's outputs, so what reads them still gets an answer.

- **What the edit builds** on LiveMath S1 (``solver``: the step ends its pass) and S4
  (``skill_graph``), the view's line, the lowering, and the refusals.
- **Both runtimes**, on S1 and S4 with a scripted model: a step whose sessions crash is
  answered by its fallback; a timeout (inactivity, the harness backstop, the node's wall,
  or timeout retries spent) never reaches the fallback; an answer goes on as before.
- **Both runtimes with dispatched workers** (`worker_node`, `run.run_node`): a crash, a
  transport failure and a raising op fall back; a declared timeout fails its item.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import lowering, registry, workers
from refactory_dsh.model import CompletionFailed, CompletionTimeout, DeclaredTimeout, HarnessBackstop
from refactory_dsh.nn import Graph, Node
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import pipeline_view as pv
from refactory_dsh.nn.graph import FALLBACK_GATE_CALL
from refactory_dsh.nn.validate import validate_state
from test_crashed_step_retry import (FAKE_DSH, OUTPUT_LIMIT, PASS_RETRY_TIMES, WORKER_CRASH,
                                     Scripted,
                                     _data, _keys, _pass, _solver, _spine, _walker, needs_spine)
from test_livemath_blocks import DRAIN_AGENT, FIXTURE_DIR, NODE, PYTHON_BIN, REPO_ROOT, _patch_task
from test_livemath_skill_graph import LABEL_B

ITEMS = 5  # the LiveMath fixture
SESSIONS = 1 + PASS_RETRY_TIMES
SIMPLE = "Answer with just the letter."
S1_FALLBACK = {"op": "fallback", "step": "solve", "kind": "agent", "text": SIMPLE}
S4_FALLBACK = {"op": "fallback", "step": "decide", "kind": "agent", "text": SIMPLE}


def _state(name: str) -> dict[str, Any]:
    return registry.resolve(Path("."), name)[1]().state_dict()


def _mode(name: str, state: dict[str, Any]):
    mode = registry.resolve(Path("."), name)[1]()
    mode.load_state_dict(state)
    return mode


def _graph(state: dict[str, Any]) -> Graph:
    return Graph.from_state(state["graph"])


def _refused(edit: dict, state: dict) -> str:
    reason = nn_edits.precheck(edit, state)
    assert reason is not None, edit
    return reason


def _s1() -> dict[str, Any]:
    return nn_edits.apply(S1_FALLBACK, _state("livemath/solver"))


def _s4() -> dict[str, Any]:
    return nn_edits.apply(S4_FALLBACK, _state("livemath/skill_graph"))


# ── what the edit builds ──────────────────────────────────────────────────


def test_s1_fallback_after_the_pass_s_last_step():
    """solve ends its pass: a no-op step after it becomes the pass's end, so both ways
    rejoin at a step (V12); the fallback sees what solve sees and writes answer.txt."""
    state = _s1()
    assert validate_state(state) == []
    graph = _graph(state)
    assert {e.key() for e in graph.edges if "solve" in e.source} == {
        ("solve", "solve_failed", ""), ("solve_failed", "solve_done", "proceed"),
        ("solve_failed", "solve_fallback", "fallback"), ("solve_fallback", "solve_done", "")}
    assert graph.nodes["data"].subgraph_exit == "solve_done"
    assert graph.nodes["solve_failed"].evaluator == FALLBACK_GATE_CALL
    fallback = graph.nodes["solve_fallback"]
    assert fallback.writes == ("answer.txt",) and fallback.role == "solver"
    assert graph.falls_back("solve") and not graph.falls_back("solve_fallback")
    assert pv.FORMS["fallback"] == "fallback(step fails -> step)"
    assert "fallback(solve fails -> solve_fallback)" in pv.render(state).splitlines()[2]


def test_s4_fallback_rejoins_before_what_reads_the_answer():
    state = _s4()
    assert validate_state(state) == []
    graph = _graph(state)
    assert {(e.target, e.condition) for e in graph.out_edges("decide_failed")} == {
        ("extract_answer", "proceed"), ("decide_fallback", "fallback")}
    assert [e.target for e in graph.out_edges("decide_fallback")] == ["extract_answer"]
    # It sees what decide sees: the decide prompt the notes are assembled into.
    assert graph.nodes["decide_fallback"].prompt_template.startswith(SIMPLE)
    assert "{{decide_prompt}}" in state["knobs"]["decide_fallback__task"]["value"]
    line = next(l for l in pv.view_lines(state) if l.key == "decide_failed")
    assert (line.text, line.reader) == ("fallback(decide fails -> decide_fallback)", "fallback")


def test_the_lowering_marks_the_step_and_names_it_to_its_gate():
    state = _s1()
    workflow = lowering.to_workflow(state)
    assert workflow.nodes["solve"].falls_back is True
    assert workflow.nodes["solve_fallback"].falls_back is None
    declared = lowering.node_declarations(state)
    assert declared["solve"]["falls_back"] is True
    assert declared["solve_failed"]["snippet"]["step"] == "solve"
    assert declared["solve_failed"]["snippet"]["labels"] == ["fallback", "proceed"]
    assert "falls_back" not in lowering.node_declarations(_state("livemath/solver"))["solve"]


def test_a_code_fallback_takes_the_step_s_outputs():
    state = nn_edits.apply({"op": "fallback", "step": "decide", "kind": "code"},
                           _state("livemath/skill_graph"))
    assert validate_state(state) == []
    node = _graph(state).nodes["decide_fallback"]
    assert node.writes == ("answer.txt",) and node.call == "refactory_dsh.snippet:run"


def test_the_view_reads_as_exactly_the_graph_s_edges():
    """Read from the drawn lines alone, every succession they imply is an edge of the graph
    and every edge is read (`test_pipeline_view._reading_problems`), on S1 and S4 and on
    every fallback a fuzz seed with a per-item pass takes."""
    import structure_fuzz as sf
    from test_pipeline_view import _reading_problems

    states = [_s1(), _s4()]
    for name in ("fuzz_data_pass", "livemath_skill_graph"):
        seed = sf.Applier(sf.seeds()[name]).seed_state
        graph = _graph(seed)
        for step in sorted(graph.nodes):
            edit = {"op": "fallback", "step": step}
            if nn_edits.precheck(edit, seed) is None:
                states.append(nn_edits.apply(edit, seed))
    assert len(states) > 6
    for state in states:
        assert validate_state(state) == []
        assert _reading_problems(state) == (set(), set()), pv.render(state)


# ── refusals ──────────────────────────────────────────────────────────────


def test_refusals():
    s1, s4 = _state("livemath/solver"), _state("livemath/skill_graph")
    assert _refused({"op": "fallback", "step": "finish"}, s1) == (
        "'finish' runs outside any per-item pass; a fallback answers for a step that runs "
        "once per item")
    assert _refused({"op": "fallback", "step": "step_fork"}, s4) == (
        "'step_fork' is a fork; fallback works on a step (agent, op or code)")
    assert _refused({"op": "fallback", "step": "precision"}, s4) == (
        "'precision' ends a fork branch; insert a step after it first, or give the fallback "
        "to a step inside the branch")
    assert _refused({**S1_FALLBACK, "id": "format_prompt"}, s1) == \
        "node 'format_prompt' already exists"
    assert _refused({**S1_FALLBACK, "retry": {"times": 1}}, s1).startswith(
        "'fallback' does not take 'retry'; it would be ignored. Its fields: step, kind, ")
    assert _refused(S1_FALLBACK, _s1()) == "'solve' has a fallback already, after 'solve_failed'"


def test_a_step_with_no_outputs_and_a_two_file_step_are_refused():
    bare = {"graph": _pass_graph(Node("solve", "op", call="loop_ops:noop")).to_state()}
    assert _refused({"op": "fallback", "step": "solve"}, bare) == \
        "'solve' has no outputs for a fallback to write"
    state = _state("livemath/solver")
    state["graph"]["nodes"]["solve"]["writes"] = ["answer.txt", "work.txt"]
    assert _refused(S1_FALLBACK, state).startswith("'solve' writes 2 files, and a new agent")


def test_a_gate_between_the_step_and_its_fallback_gate_is_refused_by_the_validator():
    """The fallback gate reads whether the one step before it failed, so a loop's gate put
    between them leaves it answering for nothing (V5)."""
    state = nn_edits.apply({"op": "loop", "step": "decide", "judge": "code"}, _s4())
    assert any(i.code == "V5" and i.node == "decide_failed" for i in validate_state(state))


# ── both runtimes, S1 and S4, a scripted model ─────────────────────────────


class _Model:
    """Answers per node: an exception class (raised every session) or text."""

    def __init__(self, answers: dict[str, Any], log: list | None = None, node: str = ""):
        self.answers, self.node = answers, node
        self.log = [] if log is None else log

    def for_node(self, node_id: str, role: str = "") -> "_Model":
        return _Model(self.answers, self.log, node_id)

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
        self.log.append(self.node)
        answer = self.answers.get(self.node, "<answer>B</answer>")
        if isinstance(answer, BaseException):
            raise answer
        return answer


CRASH = CompletionFailed("model completion failed (exit 1): " + WORKER_CRASH, kind="worker")
TIMEOUTS = {
    "timeout": CompletionTimeout("model completion timed out after 1800s of inactivity"),
    "harness_incident": HarnessBackstop("model session reached the harness backstop"),
    "declared_timeout": DeclaredTimeout("model session reached its node's declared timeout_s"),
}


def _walk(name: str, state: dict, answers: dict, tmp_path: Path, monkeypatch, *,
          ok: bool = True):
    _patch_task(monkeypatch)
    model = _Model(answers)
    run = tmp_path / "walker"
    run.mkdir()
    result = _mode(name, state).run(run, model=model)
    # Every item failing fails the run: what a timeout on every item does.
    assert result.exit_ok is ok, result.error
    return result.facts["item_scores"], model.log


def _graded_b(records: list[dict]) -> None:
    """Every item graded on answer B (label-B items right), with no error."""
    assert len(records) == ITEMS
    for record in records:
        assert record["error_kind"] is None, record
        assert record["score"] == (1.0 if str(record["id"]) in LABEL_B else 0.0), record


@pytest.mark.parametrize("name, state, step", [("livemath/solver", _s1, "solve"),
                                               ("livemath/skill_graph", _s4, "decide")])
def test_walker_a_crashed_step_is_answered_by_its_fallback(name, state, step, tmp_path,
                                                            monkeypatch):
    records, log = _walk(name, state(), {step: CRASH}, tmp_path, monkeypatch)
    _graded_b(records)
    assert log.count(step) == ITEMS * SESSIONS and log.count(f"{step}_fallback") == ITEMS
    for record in records:
        assert record["missing_steps"][step]["cause"] == "crash"


@pytest.mark.parametrize("kind", sorted(TIMEOUTS))
@pytest.mark.parametrize("name, state, step", [("livemath/solver", _s1, "solve"),
                                               ("livemath/skill_graph", _s4, "decide")])
def test_walker_a_timeout_never_reaches_the_fallback(kind, name, state, step, tmp_path,
                                                      monkeypatch):
    """A timeout fails the item as it does with no fallback (its error kind kept)."""
    records, log = _walk(name, state(), {step: TIMEOUTS[kind]}, tmp_path, monkeypatch,
                         ok=False)
    assert f"{step}_fallback" not in log
    assert [r["error_kind"] for r in records] == [kind] * ITEMS


def test_walker_timeout_retries_spent_go_on_without_an_answer(tmp_path, monkeypatch):
    """solve declares a retry on timeout: spent, it is missing for time, so its gate goes
    on and the item has no answer, as with no fallback."""
    state = _s1()
    state["graph"]["nodes"]["solve"].update(timeout_s=600, retry={"on": ["timeout"], "times": 1})
    records, log = _walk("livemath/solver", state, {"solve": TIMEOUTS["timeout"]}, tmp_path,
                         monkeypatch)
    assert "solve_fallback" not in log and log.count("solve") == 2 * ITEMS
    for record in records:
        assert record["score"] == 0.0 and record["missing_steps"]["solve"]["cause"] == "timeout"


@pytest.mark.parametrize("name, state, step", [("livemath/solver", _s1, "solve"),
                                               ("livemath/skill_graph", _s4, "decide")])
def test_walker_an_answer_goes_on_and_the_fallback_never_runs(name, state, step, tmp_path,
                                                              monkeypatch):
    records, log = _walk(name, state(), {}, tmp_path, monkeypatch)
    _graded_b(records)
    assert f"{step}_fallback" not in log and log.count(step) == ITEMS
    assert all("missing_steps" not in record for record in records)


def _on_spine(name: str, state: dict, plan: dict[str, list[str]], tmp_path: Path) -> tuple[
        list[dict], Scripted]:
    """The real mode state on the spine drain, its agents dispatched to workers against a
    scripted dsh that answers by prompt text."""
    dsh = Scripted(tmp_path / "dsh", plan)
    run = tmp_path / "spine"

    def dispatched(workflow: Any) -> Any:
        return workers.lower_workflow(workflow, tmp_path / "specs", dsh.client().context(),
                                      lowering.node_declarations(state))

    run.mkdir()
    lowering.write_declarations(state, run)
    graph_path, _ = lowering.write_mode_graph(state, tmp_path / "graphs", name="fallback",
                                              transform=dispatched)
    env = {key: value for key, value in os.environ.items() if key != "REFACTORY_WORKER_CONTEXT"}
    env.update(REFACTORY_WORKDIR=str(run), PYTHONPATH=str(Path(__file__).resolve().parent),
               LIVEMATH_DATA_PATH=str(FIXTURE_DIR), PATH=f"{PYTHON_BIN}:{env.get('PATH', '')}")
    done = subprocess.run([str(NODE), "--import", "tsx/esm", str(DRAIN_AGENT), str(graph_path),
                           str(run), f"{PYTHON_BIN}/python", "120000"],
                          capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300, env=env)
    assert done.returncode == 0, f"stdout: {done.stdout[-2000:]}\nstderr: {done.stderr[-3000:]}"
    return json.loads((run / "facts.json").read_text())["item_scores"], dsh


@needs_spine
def test_spine_s1_a_crashed_solve_is_answered_by_its_fallback(tmp_path):
    records, dsh = _on_spine("livemath/solver", _s1(), {SIMPLE: ["<answer>B</answer>"],
                                                        "": [WORKER_CRASH]}, tmp_path)
    _graded_b(records)
    assert dsh.calls(SIMPLE) == ITEMS and dsh.calls() == ITEMS * SESSIONS
    assert all(r["missing_steps"]["solve"]["cause"] == "crash" for r in records)


@needs_spine
def test_spine_s4_a_crashed_decide_is_answered_by_its_fallback(tmp_path):
    """decide is told apart by its task text, which its fallback also sees (in the decide
    prompt): the fallback's own words come first."""
    records, dsh = _on_spine("livemath/skill_graph", _s4(), {
        SIMPLE: ["<answer>B</answer>"], "Weigh the notes above": [WORKER_CRASH],
        "": ["notes <answer>A</answer>"]}, tmp_path)
    _graded_b(records)
    assert dsh.calls(SIMPLE) == ITEMS and dsh.calls("Weigh the notes above") == ITEMS * SESSIONS
    assert all(r["missing_steps"]["decide"]["cause"] == "crash" for r in records)


@needs_spine
def test_spine_s4_an_answer_goes_on_and_the_fallback_never_runs(tmp_path):
    records, dsh = _on_spine("livemath/skill_graph", _s4(), {
        SIMPLE: ["<answer>A</answer>"], "": ["<answer>B</answer>"]}, tmp_path)
    _graded_b(records)
    assert dsh.calls(SIMPLE) == 0


# ── both runtimes, dispatched workers: crash, transport, raising op, timeout ──


SLOW_DSH = FAKE_DSH.replace('if step.startswith("dsh:"):',
                            'if step.startswith("sleep:"):\n'
                            '    import time\n'
                            '    time.sleep(float(step[6:]))\n'
                            'if step.startswith("dsh:"):')


class _Slow(Scripted):
    """`Scripted`, where an answer ``sleep:N`` sleeps N seconds first."""

    def __init__(self, root: Path, plan: dict[str, list[str]]):
        super().__init__(root, plan)
        (root / "fake_dsh.py").write_text(SLOW_DSH.format(root=str(root)))


def _pass_graph(solver: Node | None = None) -> Graph:
    """data -> [solve -> score] -> report, ``solve`` an agent (or the given step), with a
    fallback agent that answers ``right``."""
    graph = Graph()
    graph.add(_data("solve", "score"))
    graph.add(solver or _solver())
    graph.add(Node("score", "op", call="crash_ops:score", reads=("answer.txt",),
                   writes=("item_result.json",)))
    graph.add(Node("report", "op", call="loop_ops:noop"))
    graph.connect("solve", "score")
    graph.connect("data", "report")
    return graph


def _with_fallback(graph: Graph, **spec: Any) -> Graph:
    state = nn_edits.apply({"op": "fallback", "step": "solve", "kind": "agent", "text": SIMPLE,
                            **spec}, {"graph": graph.to_state()})
    assert validate_state(state) == [], validate_state(state)
    return Graph.from_state(state["graph"])


def _both(runtime: str, graph: Graph, tmp_path: Path, dsh: Scripted) -> tuple[list, Path]:
    run = tmp_path / runtime
    if runtime == "walker":
        return _walker(graph, run, dsh), run
    return _spine(graph, run, tmp_path, dsh), run


RUNTIMES = ["walker", pytest.param("spine", marks=needs_spine)]
RIGHT = [(str(i), True, 1.0, None) for i in range(4)]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_crashed_agent_falls_back_on_both_runtimes(runtime, tmp_path):
    """Item 0's solve crashes every session (its retries spent): the fallback answers it."""
    plan = {SIMPLE: ["right"], "": [WORKER_CRASH] * SESSIONS + ["right"]}
    dsh = Scripted(tmp_path / "dsh", plan)
    scores, run = _both(runtime, _with_fallback(_pass_graph()), tmp_path, dsh)
    assert _keys(scores) == RIGHT
    assert dsh.calls(SIMPLE) == 1
    assert (_pass(run, "0") / "reviews/solve-missing.txt").exists()


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_transport_failure_fails_its_item_and_never_falls_back(runtime, tmp_path):
    """A wire that kept failing is the harness's fault, not the step's: item 0 fails as it
    would with no fallback, and the fallback is never asked."""
    plan = {SIMPLE: ["right"], "": ["dsh: TRANSPORT: 500 Internal Server Error", "right"]}
    dsh = Scripted(tmp_path / "dsh", plan)
    scores, run = _both(runtime, _with_fallback(_pass_graph()), tmp_path, dsh)
    assert [k[:3] for k in _keys(scores)] == [("0", False, 0.0), *(k[:3] for k in RIGHT[1:])]
    assert dsh.calls(SIMPLE) == 0
    assert not (_pass(run, "0") / "reviews/solve-missing.txt").exists()


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_raising_op_falls_back_on_both_runtimes(runtime, tmp_path):
    """solve is an op that raises on item 0: the fallback answers it."""
    graph = _with_fallback(_pass_graph(Node("solve", "op", call="crash_ops:right_unless_first",
                                            writes=("answer.txt",))))
    dsh = Scripted(tmp_path / "dsh", {"": ["right"]})
    scores, run = _both(runtime, graph, tmp_path, dsh)
    assert _keys(scores) == RIGHT
    assert dsh.calls() == 1
    assert "item 0 breaks" in (_pass(run, "0") / "reviews/solve-missing.txt").read_text()


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_declared_timeout_fails_its_item_and_never_falls_back(runtime, tmp_path):
    """Item 0's solve runs past its declared wall: the item fails ``declared_timeout``, as
    with no fallback, and the fallback is never asked."""
    graph = _with_fallback(_pass_graph(_solver(timeout_s=2)))
    dsh = _Slow(tmp_path / "dsh", {SIMPLE: ["right"], "": ["sleep:30", "right"]})
    scores, _run = _both(runtime, graph, tmp_path, dsh)
    assert _keys(scores) == [("0", False, 0.0, "declared_timeout"), *RIGHT[1:]]
    assert dsh.calls(SIMPLE) == 0


# ── review fixes: a stale record, harness failures, the op retry, partial outputs ──


def _looped_op(call: str) -> Graph:
    """data -> [solve (an op) -> solve_failed (fallback: solve_fallback, answering
    ``wrong``) -> check (reloop solve, at most 3 runs | proceed score)] -> report."""
    graph = Graph()
    graph.add(_data("solve", "score"))
    graph.add(Node("solve", "op", call=call, writes=("answer.txt",)))
    graph.add(Node("check", "gate", evaluator="crash_ops:check", reads=("answer.txt",),
                   max_iterations=3))
    graph.add(Node("score", "op", call="crash_ops:score", reads=("answer.txt",),
                   writes=("item_result.json",)))
    graph.add(Node("report", "op", call="loop_ops:noop"))
    graph.connect("solve", "check")
    graph.connect("check", "solve", condition="reloop")
    graph.connect("check", "score", condition="proceed")
    graph.connect("data", "report")
    return _with_fallback(graph)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_step_that_answers_on_a_later_visit_is_not_answered_by_its_fallback(runtime,
                                                                               tmp_path):
    """Item 0's solve fails its first visit (both runs), so its fallback answers ``wrong``;
    the checker sends it back, and solve now answers ``right``: the fallback gate reads only
    this visit, so the right answer stands (1.0, not the fallback's 0.25)."""
    dsh = Scripted(tmp_path / "dsh", {SIMPLE: ["wrong"], "": ["right"]})
    scores, run = _both(runtime, _looped_op("crash_ops:right_from_third_try"), tmp_path, dsh)
    assert _keys(scores) == RIGHT
    assert dsh.calls(SIMPLE) == 1
    assert (_pass(run, "0") / ".solve-tries").read_text() == "3"
    assert not (_pass(run, "0") / "reviews/solve-missing.txt").exists()


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_step_with_a_fallback_gets_the_same_op_retry_on_both_runtimes(runtime, tmp_path):
    """An op that fails once is answered by the runner's one retry, on both runtimes (the
    walker with recovery off too): the fallback is never asked."""
    graph = _with_fallback(_pass_graph(Node("solve", "op", call="crash_ops:right_from_second_try",
                                            writes=("answer.txt",))))
    dsh = Scripted(tmp_path / "dsh", {"": ["wrong"]})
    scores, run = _both(runtime, graph, tmp_path, dsh)
    assert _keys(scores) == RIGHT
    assert dsh.calls() == 0
    assert (_pass(run, "0") / ".solve-tries").read_text() == "2"


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_failed_step_s_partial_answer_is_never_graded(runtime, tmp_path):
    """Item 0's solve writes half an answer and raises; its fallback runs out of output in
    its one session: the item has no answer (0.0), not the partial one (0.25)."""
    graph = _with_fallback(_pass_graph(Node("solve", "op", call="crash_ops:partial_unless_first",
                                            writes=("answer.txt",))))
    dsh = Scripted(tmp_path / "dsh", {"": [OUTPUT_LIMIT]})
    scores, run = _both(runtime, graph, tmp_path, dsh)
    assert [k[:3] for k in _keys(scores)] == [("0", False, 0.0), *(k[:3] for k in RIGHT[1:])]
    assert dsh.calls() == 1
    assert not (_pass(run, "0") / "answer.txt").exists()


WRITING_DSH = FAKE_DSH.replace('if step.startswith("dsh:"):',
                               'if step.startswith("write:"):\n'
                               '    pathlib.Path("answer.txt").write_text("partial\\n")\n'
                               '    step = step[6:]\n'
                               'if step.startswith("dsh:"):')


class _Writing(Scripted):
    """`Scripted`, where an answer ``write:<answer>`` first writes a partial ``answer.txt``
    where the session runs (a tool session that wrote, then crashed)."""

    def __init__(self, root: Path, plan: dict[str, list[str]]):
        super().__init__(root, plan)
        (root / "fake_dsh.py").write_text(WRITING_DSH.format(root=str(root)))


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_failed_agent_s_partial_answer_is_never_graded(runtime, tmp_path):
    """Item 0's solve (an agent) writes half an answer and crashes in every session; its
    fallback runs out of output: the item has no answer (0.0), not the partial one."""
    dsh = _Writing(tmp_path / "dsh", {SIMPLE: [OUTPUT_LIMIT],
                                      "": [f"write:{WORKER_CRASH}"] * SESSIONS + ["right"]})
    scores, run = _both(runtime, _with_fallback(_pass_graph()), tmp_path, dsh)
    assert [k[:3] for k in _keys(scores)] == [("0", False, 0.0), *(k[:3] for k in RIGHT[1:])]
    assert dsh.calls(SIMPLE) == 1
    assert not (_pass(run, "0") / "answer.txt").exists()


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("call, kind", [("crash_ops:snippet_timeout_on_first", "RuntimeError"),
                                        ("crash_ops:sandbox_down_on_first", "harness_incident")])
def test_a_code_step_s_wall_or_sandbox_never_falls_back(runtime, call, kind, tmp_path):
    """A code step past its own wall (`SnippetTimeout`: its code's fault, recorded as the
    pipeline's, never a harness timeout) and a sandbox that did not start
    (`SandboxUnavailable`, the harness's) both fail item 0 as with no fallback, and the
    fallback is never asked."""
    graph = _with_fallback(_pass_graph(Node("solve", "op", call=call, writes=("answer.txt",))))
    dsh = Scripted(tmp_path / "dsh", {"": ["right"]})
    scores, _run = _both(runtime, graph, tmp_path, dsh)
    assert _keys(scores) == [("0", False, 0.0, kind), *RIGHT[1:]]
    assert dsh.calls() == 0


def _errors() -> dict[str, BaseException]:
    from refactory_dsh.budgets import BudgetExhausted
    from refactory_dsh.nn.executor import OpOutcomeUnknown, OpTimeoutError
    from refactory_dsh.node_retry import RetriesExhausted
    from refactory_dsh.snippet import NoSandbox, SandboxUnavailable, SnippetError, SnippetTimeout

    return {
        "raises": ValueError("bad input"),
        "crash spent": RetriesExhausted("solve crashed 3 times", kind="crash"),
        "output limit": CompletionFailed("model completion failed: output limit",
                                         kind="output_limit"),
        "code error": SnippetError("the code raised: KeyError"),
        "inactivity": TIMEOUTS["timeout"],
        "backstop": TIMEOUTS["harness_incident"],
        "declared wall": TIMEOUTS["declared_timeout"],
        "timeout spent": RetriesExhausted("solve timed out twice", kind="timeout"),
        "declared spent": RetriesExhausted("solve timed out twice", kind="timeout",
                                           cause="declared_timeout"),
        "snippet wall": SnippetTimeout("the code ran over its 10 s limit"),
        "sandbox": SandboxUnavailable("the sandbox did not start within 60 s"),
        "op timeout": OpTimeoutError("solve", 3600),
        "op outcome": OpOutcomeUnknown("child exited before reporting its outcome"),
        "transport": CompletionFailed("model completion failed: 503", kind="transport"),
        "transport text": RuntimeError("model completion failed on the wire after 5 tries"),
        "killed": RuntimeError("model completion failed (exit -9)"),
        "subprocess": subprocess.TimeoutExpired(["pytest"], 10),
        "budget": BudgetExhausted("the run's budget is spent"),
        "snippet wall text": RuntimeError("solve: SnippetTimeout: the code ran over its 10 s limit"),
        "no sandbox": NoSandbox("code nodes cannot run here: no user namespaces"),
        "auth": CompletionFailed("model completion failed: authentication_error", kind="setup"),
        "model name": CompletionFailed("model completion failed: model not found", kind="setup"),
        "permission": CompletionFailed("model completion failed: permission_error", kind="setup"),
        "prompt too long": CompletionFailed("model completion failed: prompt is too long",
                                            kind="setup"),
    }


GENUINE = {"raises", "crash spent", "output limit", "code error", "prompt too long"}


@pytest.mark.parametrize("name", sorted(_errors()))
def test_only_a_genuine_failure_falls_back(name):
    """The step's own failures (it raised, crashed past its retries, hit its output limit
    or a prompt too long, its code raised) fall back; time, harness and configuration
    failures of every kind never do."""
    from refactory_dsh.node_retry import falls_back_on

    assert falls_back_on(_errors()[name]) is (name in GENUINE)


@pytest.mark.parametrize("text, kind", [
    ("solve: SnippetTimeout: the code ran over its 10 s limit", "RuntimeError"),
    ("solve: SandboxUnavailable: the sandbox did not start within 60 s", "harness_incident"),
    ("solve exited with code 1\nSandboxUnavailable: code nodes cannot run here", "harness_incident"),
    ("check: ValueError: the reply said 'SnippetTimeout: the code ran over its 1 s limit'",
     "RuntimeError"),
])
def test_a_code_step_s_wall_and_sandbox_read_from_a_record_s_text(text, kind):
    """How the walker (``<node>: Class: ``) and the spine's runner stderr (``Class: ``)
    carry them into an item record, anchored at a line's start: an unstarted sandbox is
    the harness's; a code step's own wall stays the pipeline's (a target)."""
    from refactory_dsh.items import error_kind

    assert error_kind(RuntimeError(text)) == kind


@pytest.mark.parametrize("name, fallback, step", [("livemath/solver", S1_FALLBACK, "solve"),
                                                  ("livemath/skill_graph", S4_FALLBACK,
                                                   "decide")])
def test_insert_after_a_step_with_a_fallback_goes_where_its_ways_rejoin(name, fallback, step):
    """The step keeps its fallback, and the new step runs after whichever answered;
    inserting before the fallback gate is the same edit."""
    with_fallback = nn_edits.apply(fallback, _state(name))
    onward = {e.condition: e.target for e in _graph(with_fallback).out_edges(f"{step}_failed")}
    state = nn_edits.apply({"op": "insert", "after": step, "kind": "code"}, with_fallback)
    assert validate_state(state) == []
    graph = _graph(state)
    new = f"{step}_next"
    assert graph.falls_back(step) and graph.fallback_gate(step) == f"{step}_failed"
    assert {(e.target, e.condition) for e in graph.out_edges(f"{step}_failed")} == {
        (new, "proceed"), (f"{step}_fallback", "fallback")}
    assert [e.target for e in graph.out_edges(f"{step}_fallback")] == [new]
    assert [e.target for e in graph.out_edges(new)] == [onward["proceed"]]
    assert not graph.falls_back(new)
    assert nn_edits.apply({"op": "insert", "before": f"{step}_failed", "kind": "code"},
                          with_fallback) == state


@pytest.mark.parametrize("name, fallback, step", [("livemath/solver", S1_FALLBACK, "solve"),
                                                  ("livemath/skill_graph", S4_FALLBACK,
                                                   "decide")])
def test_removing_a_fallback_restores_the_graph_exactly(name, fallback, step):
    """Removing the fallback step, or its gate, takes the whole fallback out."""
    original = _state(name)
    with_fallback = nn_edits.apply(fallback, original)
    for removed in (f"{step}_fallback", f"{step}_failed"):
        assert nn_edits.apply({"op": "remove", "step": removed}, with_fallback) == original
    assert _refused({"op": "remove", "step": f"{step}_failed", "keep": "fallback"},
                    with_fallback) == (
        f"'{step}_failed' is the fallback gate of '{step}'; remove it (or the fallback step) "
        f"to take the fallback out, or remove '{step}' itself")


def test_removing_the_step_takes_its_fallback_with_it():
    """The fallback never moves to the step before: removing the step it answers for
    removes it as with no fallback."""
    import structure_fuzz as sf

    seed = sf.Applier(sf.seeds()["fuzz_data_pass"]).seed_state
    graph = _graph(seed)
    tried = 0
    for step in sorted(graph.nodes):
        edit = {"op": "fallback", "step": step}
        if nn_edits.precheck(edit, seed) is not None:
            continue
        state = nn_edits.apply(edit, seed)
        plain = {"op": "remove", "step": step}
        if nn_edits.precheck(plain, seed) is not None:
            continue
        tried += 1
        assert nn_edits.apply(plain, state) == nn_edits.apply(plain, seed), step
    assert tried


def test_a_visit_s_missing_record_says_only_its_own_cause(tmp_path):
    """A step that crashed past its retries on one visit and raised on the next reads
    ``failed`` for the latest, not the earlier ``crash``; answering clears both."""
    from refactory_dsh.items import missing_steps
    from refactory_dsh.nn.fork import _record_missing
    from refactory_dsh.node_retry import RetriesExhausted, clear_spent, record_spent

    for error in (RetriesExhausted("solve crashed 3 times", kind="crash"), ValueError("bad")):
        _record_missing(tmp_path, "solve", f"{type(error).__name__}: {error}")
        record_spent(tmp_path, "solve", error)
    assert missing_steps(tmp_path)["solve"]["cause"] == "failed"
    clear_spent(tmp_path, "solve")
    assert missing_steps(tmp_path) == {}


def test_a_step_s_own_time_limit_is_its_own_cause_never_a_harness_timeout(tmp_path):
    """An optional op's own subprocess limit, or a code step's wall, records
    ``step_timeout``: the step's fault, never the harness ``timeout`` a request leaves out
    (`optim.base.harness_timeout`); a model's inactivity is still ``timeout``."""
    from refactory_dsh.model import InactivityTimeout
    from refactory_dsh.node_retry import missing_cause
    from refactory_dsh.optim.base import harness_timeout
    from refactory_dsh.snippet import SnippetTimeout

    assert missing_cause(subprocess.TimeoutExpired(["pytest"], 10)) == "step_timeout"
    assert missing_cause(SnippetTimeout("the code ran over its 10 s limit")) == "step_timeout"
    assert missing_cause(InactivityTimeout(["dsh"], 600)) == "timeout"
    record = {"passed": False, "error_kind": None,
              "missing_steps": {"solve": {"cause": "step_timeout"}}}
    assert not harness_timeout(record)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_an_optional_op_s_own_timeout_is_recorded_as_its_own(runtime, tmp_path):
    """On both runtimes an optional branch op whose own subprocess times out is missing
    with cause ``step_timeout``."""
    graph = Graph()
    graph.add(_data("fork", "pick"))
    graph.add(Node("fork", "fork", targets=("a", "b"), optional_targets=("a",)))
    graph.add(Node("a", "op", call="crash_ops:own_timeout", writes=("a.txt",)))
    graph.add(Node("b", "op", call="crash_ops:right_b", writes=("b.txt",)))
    graph.add(Node("join", "join", sources=("a", "b")))
    graph.add(Node("pick", "op", call="crash_ops:pick", reads=("a.txt", "b.txt"),
                   writes=("item_result.json",)))
    graph.add(Node("report", "op", call="loop_ops:noop"))
    for source, target in [("fork", "join"), ("a", "join"), ("b", "join"), ("join", "pick"),
                           ("data", "report")]:
        graph.connect(source, target)
    dsh = Scripted(tmp_path / "dsh", {"": ["right"]})
    _scores, run = _both(runtime, graph, tmp_path, dsh)
    from refactory_dsh.items import missing_steps

    assert missing_steps(_pass(run, "0"))["a"]["cause"] == "step_timeout"


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_fall_back_puts_back_what_the_step_only_rewrote(runtime, tmp_path):
    """draft writes answer.txt; solve reads it, rewrites it and raises on item 0: its
    fallback sees draft's answer as it was (a code fallback that keeps it), not nothing."""
    graph = Graph()
    graph.add(_data("draft", "score"))
    graph.add(Node("draft", "op", call="crash_ops:draft_right", writes=("answer.txt",)))
    graph.add(Node("solve", "op", call="crash_ops:rewrite_unless_first", reads=("answer.txt",),
                   writes=("answer.txt",)))
    graph.add(Node("score", "op", call="crash_ops:score", reads=("answer.txt",),
                   writes=("item_result.json",)))
    graph.add(Node("report", "op", call="loop_ops:noop"))
    graph.connect("draft", "solve")
    graph.connect("solve", "score")
    graph.connect("data", "report")
    state = nn_edits.apply({"op": "fallback", "step": "solve", "kind": "op",
                            "call": "crash_ops:keep_answer"}, {"graph": graph.to_state()})
    assert validate_state(state) == [], validate_state(state)
    dsh = Scripted(tmp_path / "dsh", {"": ["right"]})
    scores, run = _both(runtime, Graph.from_state(state["graph"]), tmp_path, dsh)
    assert _keys(scores) == RIGHT
    assert (_pass(run, "0") / "kept.txt").read_text() == "right\n"
