"""A gate out of reloops ends its item, never the run, unless the graph says ``halt``.

A check-and-retry loop inside a data pass cannot settle an item its solver keeps
getting wrong. Before ``on_exhausted``, the reloop past ``max_iterations`` halted the
whole pass on both runtimes, so the candidate got no score at all (``E-exam``). Now:

- undeclared, a gate in a pass continues along ``proceed`` and the item is graded as it
  stands; ``fail_item`` fails that item only; ``halt`` still halts the run;
- the walker (`Executor._visit_gate`) and the spine (`Spine.verdict`, through the
  lowering's resolved ``on_exhausted``) agree, sequentially, with items at once, and
  with the loop inside a fork in the pass;
- ``loop`` works on the last step of a pass.

The graph's solver gets item ``1`` right on its second attempt and item ``3`` never
(`loop_ops.EXHAUST_SOLVES_AT`); a right answer scores 1.0, a wrong one graded as it
stands 0.25, a failed item 0.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import lowering, nn
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.executor import CURRENT_ITEM, execute
from refactory_dsh.nn.graph import GraphError

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
NODE = shutil.which("node")
DRAIN_AGENT = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
PYTHON_BIN = str(REPO_ROOT / ".venv/bin")

needs_spine = pytest.mark.skipif(
    NODE is None
    or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists(),
    reason="the spine drain needs Node and the installed workspace (pnpm install)",
)

TASK = "loop_ops:ExhaustTask"
#: The no-op step `loop` adds after its gate on a pass's last step (`nn.loops.LOOP_END_CALL`).
LOOP_END_CALL = "refactory_dsh.nn.loops:loop_end"
#: Every item right but ``3``, which is still wrong after its retries and graded as is.
CONTINUED = [("0", True, 1.0, None), ("1", True, 1.0, None), ("2", True, 1.0, None),
             ("3", False, 0.25, None)]
#: The same, with item ``3`` failed rather than graded.
FAILED = [*CONTINUED[:3], ("3", False, 0.0, "RuntimeError")]


def _loop_graph(*, parallelism: int = 1, on_exhausted: str = "", in_fork: bool = False,
                max_iterations: int = 2) -> nn.Graph:
    """data -> [solve -> check (reloop solve | proceed score)] -> report.

    ``in_fork`` puts the loop on one branch of a fork inside the pass:
    ``entry -> fork -> (solve -> check, side) -> join -> score``.
    """
    graph = nn.Graph()
    entry = "entry" if in_fork else "solve"
    graph.add(nn.Node("data", "data", subgraph_entry=entry, subgraph_exit="score",
                      writes=(CURRENT_ITEM,), produces=("item_scores",), task_ref=TASK,
                      parallelism=parallelism))
    graph.add(nn.Node("solve", "op", call="loop_ops:exhaust_solve",
                      reads=(CURRENT_ITEM, "attempts.json"), writes=("attempts.json", "answer.txt")))
    graph.add(nn.Node("check", "gate", evaluator="loop_ops:exhaust_check", reads=("answer.txt",),
                      max_iterations=max_iterations,
                      **({"on_exhausted": on_exhausted} if on_exhausted else {})))
    graph.add(nn.Node("score", "op", call="loop_ops:exhaust_score", reads=("answer.txt",),
                      writes=("item_result.json",)))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("solve", "check")
    graph.connect("check", "solve", condition="reloop")
    if in_fork:
        graph.add(nn.Node("entry", "op", call="loop_ops:noop"))
        graph.add(nn.Node("fork", "fork", targets=("solve", "side")))
        graph.add(nn.Node("side", "op", call="loop_ops:noop"))
        graph.add(nn.Node("join", "join", sources=("check", "side")))
        graph.connect("entry", "fork")
        graph.connect("fork", "join")
        graph.connect("check", "join", condition="proceed")
        graph.connect("side", "join")
        graph.connect("join", "score")
    else:
        graph.connect("check", "score", condition="proceed")
    graph.connect("data", "report")
    return graph


def _state(graph: nn.Graph) -> dict[str, Any]:
    return {"version": nn.STATE_VERSION, "graph": graph.to_state(), "knobs": {}}


def _keys(records: list[dict[str, Any]]) -> list[tuple[str, bool, float, Any]]:
    return sorted((str(r["id"]), bool(r["passed"]), float(r.get("score") or 0), r.get("error_kind"))
                  for r in records)


def _walker(graph: nn.Graph, run: Path, **kw: Any) -> Any:
    return execute(graph, run, max_steps=400, **kw)


def _spine(graph: nn.Graph, run: Path, tmp_path: Path) -> tuple[int, str]:
    """Lower ``graph`` into ``run``, drive it through the real spine drain, and return
    the drain's exit code and the run's recorded status (``spine-status.json``)."""
    run.mkdir(parents=True, exist_ok=True)
    state = {"graph": graph.to_state()}
    lowering.write_declarations(state, run)
    graph_path, _ = lowering.write_mode_graph(state, tmp_path / f"graphs-{run.name}", name="exhaust")
    responses = tmp_path / f"responses-{run.name}.json"
    responses.write_text("{}")
    env = {**os.environ, "REFACTORY_WORKDIR": str(run), "PYTHONPATH": str(HERE)}
    env["PATH"] = f"{PYTHON_BIN}:{env.get('PATH', '')}"
    done = subprocess.run(
        [str(NODE), "--import", "tsx/esm", str(DRAIN_AGENT), str(graph_path), str(run),
         f"{PYTHON_BIN}/python", "30000", str(responses)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180, env=env)
    if done.returncode != 0:
        sys.stderr.write(f"spine drain rc={done.returncode}\nstdout: {done.stdout[-2000:]}\n"
                         f"stderr: {done.stderr[-3000:]}\n")
    status_file = run / ".refactory" / "spine-status.json"
    status = json.loads(status_file.read_text()).get("status", "") if status_file.exists() else ""
    return done.returncode, status


def _attempts(run: Path, item: str) -> int:
    [path] = list(run.glob(f"passes/data/{item}_*/attempts.json"))
    return int(json.loads(path.read_text()))


# ── the walker ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("in_fork", [False, True])
@pytest.mark.parametrize("parallelism", [1, 3])
def test_an_exhausted_loop_in_a_pass_scores_every_item_on_the_walker(tmp_path, parallelism, in_fork):
    """[walker] Undeclared, the item still wrong after its retries goes on with its last
    attempt and is graded as it stands; every item is scored and the run is not halted,
    sequentially, with items at once, and with the loop inside a fork in the pass."""
    run = tmp_path / "run"
    result = _walker(_loop_graph(parallelism=parallelism, in_fork=in_fork), run)
    assert result.exit_ok and not result.halted, result.error
    assert _keys(result.facts["item_scores"]) == CONTINUED
    # One attempt plus its two reloops, and no more.
    assert _attempts(run, "3") == 3
    if parallelism == 1:
        # The last item's proceed came from running out of reloops, and says so.
        assert result.gate_verdicts.get("check") == "proceed"
        assert result.gate_exhausted == ["check"]


def test_an_exhausted_proceed_is_recorded_apart_from_a_pass(tmp_path, monkeypatch):
    """[walker] The gate's step and the run record mark a proceed taken on exhaustion
    (``exhausted``), and a later proceed that passed clears it, as the spine's
    ``gate_exhausted`` does; the mark survives the progress record."""
    from refactory_dsh.nn.executor import Executor
    from refactory_dsh.nn.record import RunRecord

    walked: list[Executor] = []
    run = Executor.run

    def keep(self: Executor, **kw: Any) -> Any:
        walked.append(self)
        return run(self, **kw)

    monkeypatch.setattr(Executor, "run", keep)
    graph = _loop_graph()
    result = _walker(graph, tmp_path / "run")
    assert result.exit_ok, result.error
    [executor] = walked
    proceeds = [step.exhausted for step in executor.steps
                if step.node == "check" and step.verdict == "proceed"]
    # Items 0, 1 and 2 passed the check; item 3 ran out of reloops.
    assert proceeds == [False, False, False, True]
    assert RunRecord.from_progress(executor.record.to_progress()).gate_exhausted == ["check"]
    executor.record.gate(graph, "check", "proceed", target="score", retry=False)
    assert executor.record.gate_exhausted == []
    assert "gate_exhausted" not in executor.record.to_progress()


@pytest.mark.parametrize("in_fork", [False, True])
@pytest.mark.parametrize("parallelism", [1, 3])
def test_fail_item_fails_only_that_item_on_the_walker(tmp_path, parallelism, in_fork):
    """[walker] ``fail_item``: the stuck item gets a failed record; its siblings are
    scored and the run goes on."""
    result = _walker(_loop_graph(parallelism=parallelism, on_exhausted="fail_item", in_fork=in_fork),
                     tmp_path / "run")
    assert result.exit_ok and not result.halted, result.error
    assert _keys(result.facts["item_scores"]) == FAILED
    [stuck] = [r for r in result.facts["item_scores"] if r["id"] == "3"]
    assert "max_iterations of 2" in stuck["error_text"]
    assert "on_exhausted: fail_item" in stuck["error_text"]


@pytest.mark.parametrize("parallelism", [1, 3])
def test_a_declared_halt_still_halts_the_run_on_the_walker(tmp_path, parallelism):
    """[walker] ``halt`` keeps the old behaviour: the stuck item halts the whole run."""
    result = _walker(_loop_graph(parallelism=parallelism, on_exhausted="halt"), tmp_path / "run")
    assert result.halted and not result.exit_ok
    assert "gate 'check' halted the run" in str(result.error)


def test_an_exhausted_loop_outside_a_pass_still_halts(tmp_path):
    """[walker] A gate outside any pass halts as before, whatever it declares but
    ``continue``: there is no item to end (design mode's gates are such gates)."""
    def top(on_exhausted: str) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("body", "op", call="loop_ops:count_visit",
                          writes=("visits.json",), produces=("visits",)))
        graph.add(nn.Node("gate", "gate", evaluator="loop_ops:always_retry", max_iterations=2,
                          on_exhausted=on_exhausted))
        graph.add(nn.Node("done", "op", call="loop_ops:noop"))
        graph.connect("body", "gate")
        graph.connect("gate", "body", condition="retry")
        graph.connect("gate", "done", condition="proceed")
        return graph

    for declared in ("", "fail_item", "halt"):
        result = _walker(top(declared), tmp_path / f"run-{declared or 'default'}")
        assert result.halted, (declared, result.error)
    result = _walker(top("continue"), tmp_path / "run-continue")
    assert result.exit_ok and not result.halted, result.error
    assert json.loads((tmp_path / "run-continue" / "visits.json").read_text()) == 3


# ── both runtimes ──────────────────────────────────────────────────────────


@needs_spine
@pytest.mark.parametrize("case", ["continue", "fail_item", "fork-continue", "fork-fail_item"])
@pytest.mark.parametrize("parallelism", [1, 3])
def test_an_exhausted_loop_ends_its_item_on_both_runtimes(tmp_path, parallelism, case):
    """[both runtimes] One graph, walked and lowered onto the spine's drain: the stuck
    item is graded as it stands (undeclared) or failed (``fail_item``), its siblings are
    scored, and neither runtime halts; sequentially and with items at once, with the
    loop in the pass or inside a fork in it."""
    in_fork = case.startswith("fork-")
    declared = "" if case.endswith("continue") else "fail_item"
    graph = _loop_graph(parallelism=parallelism, on_exhausted=declared, in_fork=in_fork)
    expected = CONTINUED if not declared else FAILED

    walker = _walker(graph, tmp_path / "walker")
    assert walker.exit_ok and not walker.halted, walker.error
    rc, status = _spine(graph, tmp_path / "spine", tmp_path)
    assert (rc, status) == (0, "done")
    spine = json.loads((tmp_path / "spine" / "facts.json").read_text())

    assert _keys(walker.facts["item_scores"]) == expected
    assert _keys(spine["item_scores"]) == expected
    assert _attempts(tmp_path / "walker", "3") == _attempts(tmp_path / "spine", "3") == 3


@needs_spine
@pytest.mark.parametrize("parallelism", [1, 3])
def test_a_declared_halt_halts_on_both_runtimes(tmp_path, parallelism):
    """[both runtimes] ``halt`` still halts the whole run on each runtime."""
    graph = _loop_graph(parallelism=parallelism, on_exhausted="halt")
    assert _walker(graph, tmp_path / "walker").halted
    rc, status = _spine(graph, tmp_path / "spine", tmp_path)
    assert status == "halted", (rc, status)


def test_the_lowering_hands_the_resolved_choice_to_the_spine():
    """[spine, via lowering] The spine gets each gate's resolved ``on_exhausted``; a gate
    that halts lowers without the field, byte for byte as before."""
    lowered = lowering.to_workflow({"graph": _loop_graph().to_state()})
    assert lowered.nodes["check"].on_exhausted == "continue"
    assert lowering.to_workflow(
        {"graph": _loop_graph(on_exhausted="fail_item").to_state()}).nodes["check"].on_exhausted == "fail_item"
    halting = lowering.to_workflow({"graph": _loop_graph(on_exhausted="halt").to_state()})
    assert halting.nodes["check"].on_exhausted is None
    assert "on_exhausted" not in halting.to_dict()["nodes"]["check"]


# ── the declaration ────────────────────────────────────────────────────────


def test_on_exhausted_resolves_from_the_graph():
    """`Graph.on_exhausted`, the one rule both runtimes read."""
    assert _loop_graph().on_exhausted("check") == "continue"
    assert _loop_graph(in_fork=True).on_exhausted("check") == "continue"
    assert _loop_graph(on_exhausted="fail_item").on_exhausted("check") == "fail_item"
    assert _loop_graph(on_exhausted="halt").on_exhausted("check") == "halt"
    # No proceed edge to go on by: the item fails instead.
    graph = _loop_graph()
    graph.edges = [e for e in graph.edges if e.condition != "proceed"]
    graph.connect("check", "score", condition="accept")
    assert graph.on_exhausted("check") == "fail_item"
    # A proceed edge that is itself the retry (an inferred single-gate cycle) leads back,
    # not on: continuing along it would loop past the cap, so the item fails instead.
    graph = _loop_graph(on_exhausted="continue")
    graph.edges = [e for e in graph.edges if e.source != "check"]
    graph.connect("check", "solve", condition="proceed")
    graph.connect("check", "score", condition="accept")
    assert graph.exhaustion_exit("check") is None
    assert graph.on_exhausted("check") == "fail_item"
    # Outside a pass: halt, unless it declares continue.
    graph = _loop_graph()
    graph.nodes["data"] = nn.Node("data", "data", subgraph_entry="report", subgraph_exit="report",
                                  writes=(CURRENT_ITEM,), task_ref=TASK)
    assert graph.on_exhausted("check") == "halt"
    graph.nodes["check"] = nn.Node("check", "gate", evaluator="loop_ops:exhaust_check",
                                   max_iterations=2, on_exhausted="fail_item")
    assert graph.on_exhausted("check") == "halt"


def test_on_exhausted_is_checked():
    """Only a gate declares it, only with a known value, and ``continue`` needs a
    ``proceed`` edge that leads on (V5)."""
    with pytest.raises(GraphError, match="not a gate but declares on_exhausted"):
        nn.Node("a", "op", call="loop_ops:noop", on_exhausted="continue")
    with pytest.raises(GraphError, match="unknown on_exhausted 'retry'"):
        nn.Node("g", "gate", evaluator="loop_ops:exhaust_check", on_exhausted="retry")
    graph = _loop_graph(on_exhausted="continue")
    assert not [i for i in nn.validate_state(_state(graph), resolve_calls=False) if i.code == "V5"]
    graph.edges = [e for e in graph.edges if e.condition != "proceed"]
    graph.connect("check", "score", condition="accept")
    assert [i for i in nn.validate_state(_state(graph), resolve_calls=False)
            if i.code == "V5" and "on_exhausted 'continue'" in i.message]
    # Round trip: written only when declared, so other states keep their ids.
    assert nn.Node.from_state(graph.nodes["check"].to_state()).on_exhausted == "continue"
    assert "on_exhausted" not in _loop_graph().nodes["check"].to_state()


# ── loop on the last step of a pass ────────────────────────────────────────


def _ends_at_solver(*, agent: bool, extract: bool = False) -> nn.Graph:
    """data -> [format -> solve] -> report: the pass ends at its solver, as LiveMath's does.
    ``extract`` adds a step after the solver, as the Countdown seed's ``extract_answer``."""
    graph = nn.Graph()
    graph.add(nn.Node("data", "data", subgraph_entry="format",
                      subgraph_exit="extract" if extract else "solve",
                      writes=(CURRENT_ITEM,), produces=("item_scores",), task_ref=TASK))
    graph.add(nn.Node("format", "op", call="loop_ops:noop", reads=(CURRENT_ITEM,)))
    if agent:
        graph.add(nn.Node("solve", "agent", prompt_template="Solve it.", writes=("answer.txt",)))
    else:
        graph.add(nn.Node("solve", "op", call="loop_ops:exhaust_solve",
                          reads=(CURRENT_ITEM, "attempts.json"),
                          writes=("attempts.json", "answer.txt")))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("format", "solve")
    if extract:
        graph.add(nn.Node("extract", "op", call="loop_ops:noop", reads=("answer.txt",)))
        graph.connect("solve", "extract")
    graph.connect("data", "report")
    return graph


_GATE = {"id": "check", "kind": "gate", "evaluator": "loop_ops:exhaust_check",
         "reads": ["answer.txt"], "max_iterations": 3}


def _looped(graph: nn.Graph) -> dict[str, Any]:
    return nn_edits.apply({"op": "loop", "target": "solve", "gate": dict(_GATE)}, _state(graph))


class _Answering:
    """A model whose every agent reply is ``answer``."""

    thread_safe = True

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def complete(self, prompt: str, *, system: str = "", **kw: Any) -> str:
        self.calls += 1
        return self.answer

    def for_node(self, node_id: str, role: str | None = None) -> "_Answering":
        return self

    def with_wall(self, seconds: float) -> "_Answering":
        return self

    def route(self) -> dict[str, Any]:
        return {"worker_backend": "dsh"}


def test_loop_wraps_the_last_agent_of_a_pass(tmp_path):
    """``loop`` on an agent with no out-edge (the pass's exit) is accepted: a no-op step
    after the gate becomes the exit, the gate leads there on ``proceed``, and the state
    validates and runs, retrying the agent and grading every item."""
    seed = _ends_at_solver(agent=True)
    assert seed.out_edges("solve") == []
    looped = _looped(seed)
    graph = nn.Graph.from_state(looped["graph"])
    assert graph.nodes["data"].subgraph_exit == "solve_done"
    assert graph.nodes["solve_done"].call == LOOP_END_CALL
    assert {(e.source, e.target, e.condition) for e in graph.edges} >= {
        ("solve", "check", None), ("check", "solve", "reloop"), ("check", "solve_done", "proceed")}
    assert not nn.validate_state(looped)

    wrong = _Answering("wrong")
    result = _walker(graph, tmp_path / "wrong", model=wrong)
    assert result.exit_ok and not result.halted, result.error
    assert _keys(result.facts["item_scores"]) == [(str(i), False, 0.25, None) for i in range(4)]
    assert wrong.calls == 4 * 4  # one attempt and three retries per item
    right = _Answering("right")
    result = _walker(graph, tmp_path / "right", model=right)
    assert _keys(result.facts["item_scores"]) == [(str(i), True, 1.0, None) for i in range(4)]
    assert right.calls == 4


def test_loop_on_a_terminal_outside_a_pass_adds_the_end_step():
    """The same on the run's last step outside any pass: the end step is the new terminal."""
    graph = nn.Graph()
    graph.add(nn.Node("solve", "op", call="loop_ops:exhaust_solve",
                      reads=("attempts.json",), writes=("attempts.json", "answer.txt")))
    state = nn_edits.apply({"op": "loop", "target": "solve", "gate": dict(_GATE)}, _state(graph))
    assert nn.Graph.from_state(state["graph"]).terminals() == ["solve_done"]
    assert not nn.validate_state(state)


@needs_spine
@pytest.mark.parametrize("parallelism", [1, 3])
def test_loop_on_the_last_step_runs_on_both_runtimes(tmp_path, parallelism):
    """[both runtimes] The looped last-step graph scores every item the same on each."""
    seed = _ends_at_solver(agent=False)
    seed.nodes["data"] = nn.Node("data", "data", subgraph_entry="format", subgraph_exit="solve",
                                 writes=(CURRENT_ITEM,), produces=("item_scores",), task_ref=TASK,
                                 parallelism=parallelism)
    graph = nn.Graph.from_state(_looped(seed)["graph"])
    walker = _walker(graph, tmp_path / "walker")
    assert walker.exit_ok, walker.error
    rc, status = _spine(graph, tmp_path / "spine", tmp_path)
    assert (rc, status) == (0, "done")
    spine = json.loads((tmp_path / "spine" / "facts.json").read_text())
    assert _keys(walker.facts["item_scores"]) == _keys(spine["item_scores"]) == CONTINUED


# ── the outer loop ─────────────────────────────────────────────────────────


class _EndsAtSolver(nn.Mode):
    """A scored seed whose pass ends at its solver, as LiveMath's does."""

    name = "ends_at_solver"
    extract = False

    def build(self) -> nn.Graph:
        return _ends_at_solver(agent=False, extract=self.extract)

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


class _Extracts(_EndsAtSolver):
    """The same with a step after the solver, as the Countdown seed has."""

    name = "extracts"
    extract = True


@contextlib.contextmanager
def _registered(mode_class: type) -> Any:
    import refactory_dsh.registry as registry

    class Package:
        name = "fake"
        description = ""

        def mode(self, name: str) -> type:
            return mode_class

    original = registry.resolve
    registry.resolve = lambda project, identifier: (Package(), mode_class)
    try:
        yield
    finally:
        registry.resolve = original


@pytest.mark.parametrize("mode_class", [_Extracts, _EndsAtSolver], ids=["extracts", "ends_at_solver"])
def test_the_outer_loop_scores_a_candidate_whose_loop_runs_out(tmp_path, mode_class):
    """The search's likeliest first structural move -- one ``loop`` edit with a checker
    gate around the solver -- gets a score when an item never passes the check: the stuck
    item is graded as it stands instead of the candidate scoring nothing. On a seed with
    a step after the solver (Countdown's shape), and on one whose pass ends at it."""
    from refactory_dsh.loop import Loop, LoopConfig
    from refactory_dsh.optim import Candidate

    with _registered(mode_class):
        loop = Loop(f"fake/{mode_class.name}",
                    LoopConfig(epochs=1, batch_size=1, optimizer="random", seed=1, holdout=0.0,
                               runs_root=tmp_path / "run"),
                    project=Path("."))
        loop.init()
        seed = loop.population[0]
        edit = {"op": "loop", "target": "solve", "gate": dict(_GATE)}
        child = Candidate.from_state(nn_edits.apply(edit, seed.state), parent=seed.id, edits=[edit])
        loop._evaluate(child)

    assert child.exit_ok, child.error
    assert child.raw_score == pytest.approx((1.0 + 1.0 + 1.0 + 0.25) / 4)
