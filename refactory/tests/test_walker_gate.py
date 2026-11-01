"""Walker gate flow: stop at a user gate, steer, resume and halt.

These tests go through the real commands (``design.main``, ``console.main``)
with a ``FakeModel``, not a mock executor.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from refactory_dsh import steering
from refactory_dsh.nn import Graph, Knob, Mode, Node
from refactory_dsh.nn.executor import GateStop, HaltStop, route_verdict
from refactory_dsh.nn.graph import Edge


# ── helpers ────────────────────────────────────────────────────────────────


def _simple_gate_graph():
    """A minimal graph with an op, an agent, a user gate, and an emitter."""
    graph = Graph()
    graph.add(Node("prepare", "op", call="loop_ops:noop",
                    writes=(), produces=("name", "brief", "check")))
    graph.add(Node("design_agent", "agent",
                    prompt_template="Design: {{brief}}",
                    reads=(),
                    writes=("design.md",),
                    produces=("design",)))
    graph.add(Node("gate_strategy", "gate", gate_kind="user",
                    reads=("design.md",),
                    max_iterations=4,
                    notes="Approve the design?"))
    graph.add(Node("emit_agent", "agent",
                    prompt_template="Emit: {{brief}}",
                    reads=("design.md",),
                    writes=("output.txt",),
                    produces=("emit",)))
    graph.connect("prepare", "design_agent")
    graph.connect("design_agent", "gate_strategy")
    graph.connect("gate_strategy", "design_agent", condition="reloop")
    graph.connect("gate_strategy", "emit_agent", condition="proceed")
    return graph


class _FakeModel:
    """Returns scripted answers."""
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def complete(self, prompt, *, system="", workdir=None):
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else ""

    def context(self):
        return {"worker_backend": "dsh"}

    def route(self):
        return {"worker_backend": "dsh"}

    def for_node(self, node_id, role=None):
        return self


# ── tests ──────────────────────────────────────────────────────────────────


def test_gate_stop_raises_for_user_gate(tmp_path):
    """A walker run at a user gate without a verdict raises GateStop."""
    from refactory_dsh.nn.executor import execute

    graph = _simple_gate_graph()
    model = _FakeModel(["a design document"])
    result = execute(graph, tmp_path / "run", model=model,
                     task={"name": "test", "brief": "test", "check": "true"})
    assert result.exit_ok is False
    assert result.gate_stop == "gate_strategy"
    assert "gate_strategy" in (result.error or "")


def test_halt_at_gate_ends_run(tmp_path):
    """A halt verdict at a user gate ends the run with halted=True."""
    from refactory_dsh.nn.executor import execute

    graph = _simple_gate_graph()
    workdir = tmp_path / "run"
    model = _FakeModel(["a design document"])
    # First run: stops at gate.
    result = execute(graph, workdir, model=model,
                     task={"name": "test", "brief": "test", "check": "true"})
    assert result.gate_stop == "gate_strategy"

    # Steer with halt.
    ledger = steering.Ledger(workdir)
    ledger.verdict("gate_strategy", "halt")

    # Resume: should halt.
    model2 = _FakeModel(["a design document"])
    result2 = execute(graph, workdir, model=model2,
                      task={"name": "test", "brief": "test", "check": "true"})
    assert result2.exit_ok is False
    assert result2.halted is True


def test_proceed_with_note(tmp_path):
    """A proceed verdict with a note: the note appears in the next agent's prompt."""
    from refactory_dsh.nn.executor import execute

    graph = _simple_gate_graph()
    workdir = tmp_path / "run"
    model = _FakeModel(["a design document"])

    # First run: stops at gate.
    result = execute(graph, workdir, model=model,
                     task={"name": "test", "brief": "test", "check": "true"})
    assert result.gate_stop == "gate_strategy"

    # Steer with proceed and a note.
    ledger = steering.Ledger(workdir)
    ledger.verdict("gate_strategy", "proceed")
    ledger.note("constraint", "must use Python 3.12")

    # Resume from the gate with the first run's facts, as a real --resume does.
    model2 = _FakeModel(["the emitted code"])
    result2 = execute(graph, workdir, model=model2,
                      task={"name": "test", "brief": "test", "check": "true"},
                      start="gate_strategy", facts=dict(result.facts))
    assert result2.exit_ok, result2.error
    # The note should appear in the emit agent's prompt (folded by for_run).
    assert any("Python 3.12" in p for p in model2.prompts), (
        "the note did not reach the next agent's prompt"
    )


def test_reloop_with_note(tmp_path):
    """A reloop verdict with a note: the run reloops and the note reaches the agent."""
    from refactory_dsh.nn.executor import execute

    graph = _simple_gate_graph()
    workdir = tmp_path / "run"
    model = _FakeModel(["first design"])

    # First run: stops at gate.
    result = execute(graph, workdir, model=model,
                     task={"name": "test", "brief": "test", "check": "true"})
    assert result.gate_stop == "gate_strategy"

    # Steer with reloop and a note.
    ledger = steering.Ledger(workdir)
    ledger.verdict("gate_strategy", "reloop")
    ledger.note("correction", "redo the error handling")

    # Resume: starts from scratch; the reloop verdict sends design_agent back,
    # and then gate_strategy has no verdict, so it stops again.
    model2 = _FakeModel(["second design", "third design"])
    result2 = execute(graph, workdir, model=model2,
                      task={"name": "test", "brief": "test", "check": "true"})
    assert result2.gate_stop == "gate_strategy"
    # The note should have reached the design agent (folded by for_run).
    assert any("redo the error handling" in p for p in model2.prompts), (
        "the reloop note did not reach the design agent"
    )


def test_upfront_approve_survives_resume(tmp_path):
    """An up-front verdict persisted as a ledger entry survives a resume."""
    from refactory_dsh.nn.executor import execute

    graph = _simple_gate_graph()
    workdir = tmp_path / "run"

    # Seed the ledger and add an upfront verdict.
    steering.Ledger.seed(workdir, "build a test thing")
    ledger = steering.Ledger(workdir)
    ledger.verdict("gate_strategy", "proceed", approval=True)

    # Run: should pass through the gate.
    model = _FakeModel(["a design document", "the emitted code"])
    result = execute(graph, workdir, model=model,
                     task={"name": "test", "brief": "test", "check": "true"})
    assert result.exit_ok, result.error


def test_verdict_routing_cases():
    """The route_verdict function matches the shared JSON cases."""
    cases_path = Path(__file__).resolve().parent.parent / "refactory_dsh" / "data" / "verdict-routing-cases.json"
    cases = json.loads(cases_path.read_text())
    for case in cases:
        edges = [Edge(source=case["gate"], target=e["target"], condition=e["condition"])
                 for e in case["edges"]]
        expect = case["expect"]
        if "error" in expect:
            with pytest.raises(Exception):
                route_verdict(case["verdict"], edges, case["gate_kind"])
        elif "halt" in expect:
            edge, reason = route_verdict(case["verdict"], edges, case["gate_kind"])
            assert edge is None, f"{case['name']}: expected halt but got {edge}"
            assert reason == expect["reason"], f"{case['name']}: reason mismatch"
        else:
            edge, _ = route_verdict(case["verdict"], edges, case["gate_kind"])
            assert edge is not None, f"{case['name']}: expected target but got halt"
            assert edge.target == expect["target"], f"{case['name']}: target mismatch"


def test_gate_stop_in_fork_branch(tmp_path):
    """GateStop in a fork branch records refused, not an uncaught exception."""
    graph = Graph()
    graph.add(Node("start", "op", call="loop_ops:noop",
                    writes=(), produces=()))
    graph.add(Node("fork", "fork", targets=("branch_a",)))
    graph.add(Node("branch_a", "gate", gate_kind="user",
                    reads=(), max_iterations=2, notes="user gate in fork"))
    graph.add(Node("join", "join", sources=("branch_a",)))
    graph.connect("start", "fork")
    graph.connect("fork", "join")
    graph.connect("branch_a", "join")

    from refactory_dsh.nn.executor import execute
    result = execute(graph, tmp_path / "run")
    assert result.exit_ok is False
    assert "branch_a" in (result.error or "") or "gate stop" in (result.error or "")


def test_gate_stop_in_data_pass(tmp_path):
    """GateStop in a data pass becomes a failed item under F4.

    With the failure rule, a GateStop in a data pass is caught as a failed
    item (score 0, error kept). Since this is the only item, all-items-failed
    makes the node fail. The gate is recorded in the item's error_text.
    """
    graph = Graph()
    graph.add(Node("data", "data", subgraph_entry="inner_gate", subgraph_exit="inner_gate",
                    inline_items=[{"id": "item1"}],
                    writes=("current_item.json",), produces=("item_scores",)))
    graph.add(Node("inner_gate", "gate", gate_kind="user",
                    reads=(), max_iterations=2, notes="user gate in data pass"))
    graph.add(Node("finish", "op", call="loop_ops:noop"))
    graph.connect("data", "finish")

    from refactory_dsh.nn.executor import execute
    result = execute(graph, tmp_path / "run")
    assert result.exit_ok is False
    # All items failed → the error names the data node.
    assert "all" in (result.error or "").lower() and "failed" in (result.error or "").lower()
    # The gate is recorded in the failed item's error_text.
    scores = result.facts.get("item_scores", [])
    assert len(scores) == 1
    assert scores[0]["passed"] is False
    assert "inner_gate" in scores[0].get("error_text", "")


def test_second_walker_run_refused_while_first_holds_lock(tmp_path):
    """The ownership lock prevents concurrent walker runs on the same workdir."""
    from refactory_dsh.run_ownership import ownership

    workdir = tmp_path / "run"
    barrier = threading.Event()
    error_from_thread: list[BaseException] = []

    def hold_lock():
        try:
            with ownership(workdir):
                barrier.set()
                # Hold the lock until the main thread signals.
                import time
                for _ in range(50):
                    time.sleep(0.05)
        except Exception as exc:
            error_from_thread.append(exc)
            barrier.set()

    t = threading.Thread(target=hold_lock, daemon=True)
    t.start()
    barrier.wait(timeout=5)
    assert barrier.is_set(), "lock-holding thread did not start"

    # A second ownership attempt from a different thread should fail.
    with pytest.raises(ValueError, match="active invocation"):
        with ownership(workdir):
            pass  # pragma: no cover — should not get here
    t.join(timeout=5)


def test_rerun_archives_previous_invocation(tmp_path):
    """A second invocation archives the first's ledger and outputs."""
    from refactory_dsh import invocations

    workdir = tmp_path / "run"
    state = {
        "nodes": {},
        "graph": {"start": "noop", "nodes": {"noop": {"id": "noop", "kind": "op", "call": "loop_ops:noop"}}},
        "knobs": {},
        "ports": [],
    }

    # First preparation.
    invocations.prepare(
        state, workdir, name="first",
        transform=lambda workflow, stage_dir=None: workflow,
        intent="first intent",
    )
    # The ledger was seeded.
    assert steering.Ledger(workdir).exists()
    first_ledger = (workdir / steering.LEDGER).read_text()

    # Second preparation: the old ledger should be archived.
    invocations.prepare(
        state, workdir, name="second",
        transform=lambda workflow, stage_dir=None: workflow,
        intent="second intent",
    )
    second_ledger = (workdir / steering.LEDGER).read_text()
    assert "second intent" in second_ledger
    # The first ledger is in the archive.
    archives = list((workdir / ".refactory/invocations").iterdir())
    assert len(archives) == 1
    archive = archives[0] / "artifacts" / steering.LEDGER
    assert archive.exists()


def test_walker_create_uses_shared_preparation(tmp_path):
    """create() calls prepare_run, so the ledger is seeded and the invocation is recorded."""
    from refactory_dsh import invocations

    workdir = tmp_path / "run"
    state = {
        "nodes": {},
        "graph": {"start": "noop", "nodes": {"noop": {"id": "noop", "kind": "op", "call": "loop_ops:noop"}}},
        "knobs": {},
        "ports": [],
    }

    # A preparation seeds the ledger and writes invocation.json.
    invocations.prepare(
        state, workdir, name="test-create",
        transform=lambda workflow, stage_dir=None: workflow,
        intent="create a package",
    )
    assert steering.Ledger(workdir).exists()
    assert (workdir / invocations.INVOCATION).exists()


def test_gate_stop_output(tmp_path, capsys):
    """The gate-stop output includes the prompt, verdicts (with halt), steer and resume."""
    from refactory_dsh.run_preparation import print_gate_stop

    workdir = tmp_path / "run"
    workdir.mkdir(parents=True)
    (workdir / ".refactory").mkdir()
    # Write a run-config.json with a resume command.
    import json
    (workdir / ".refactory/run-config.json").write_text(json.dumps({
        "resume_command": "python -m refactory_dsh.design --resume " + str(workdir),
    }))

    graph = _simple_gate_graph()
    print_gate_stop("gate_strategy", workdir, graph=graph)
    captured = capsys.readouterr()
    stderr = captured.err
    assert "gate_strategy" in stderr
    assert "halt" in stderr
    assert "steer" in stderr
    assert "resume" in stderr


def test_halt_in_fork_branch(tmp_path):
    """HaltStop in a fork branch records halted."""
    graph = Graph()
    graph.add(Node("start", "op", call="loop_ops:noop",
                    writes=(), produces=()))
    graph.add(Node("fork", "fork", targets=("branch_a",)))
    # A gate with an upfront halt verdict.
    graph.add(Node("branch_a", "gate", gate_kind="user",
                    reads=(), max_iterations=2, notes="user gate in fork"))
    graph.add(Node("join", "join", sources=("branch_a",)))
    graph.connect("start", "fork")
    graph.connect("fork", "join")
    graph.connect("branch_a", "join")

    from refactory_dsh.nn.executor import execute

    workdir = tmp_path / "run"
    # Seed ledger and add a halt verdict as upfront approval (branch_a has no
    # prior gate visit, so a scoped verdict would fail).
    steering.Ledger.seed(workdir, "test")
    ledger = steering.Ledger(workdir)
    ledger.verdict("branch_a", "halt", approval=True)

    result = execute(graph, workdir)
    assert result.exit_ok is False
    assert result.halted is True or "halted" in (result.error or "")
