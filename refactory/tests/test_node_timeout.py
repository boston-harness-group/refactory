"""An agent node's own bound on its model session holds on both runtimes.

`Node.timeout_s` is lowered to the IR's `AgentNode.timeout`. On the spine,
`workers.lower_workflow` gives the dispatched session that wall (`node_wall_s`) and
the node's command the usual dispatch overhead beyond it. On the walker, the
executor gives the node's model client the same wall. The verifier declares one, so
a slow verifier is missing rather than holding up the round.
"""
from __future__ import annotations

import subprocess

import pytest

from refactory_dsh import lowering, workers
from refactory_dsh.design import DesignMode
from refactory_dsh.fast import DesignFastMode
from refactory_dsh.model import NODE_DISPATCH_OVERHEAD_S, DshModelClient
from refactory_dsh.nn import Graph, GraphError, Node
from refactory_dsh.verification import DESIGN_VERIFIER_TIMEOUT_S, VERIFIER_TIMEOUT_S


def test_only_an_agent_declares_a_positive_timeout():
    with pytest.raises(GraphError, match="not an agent"):
        Node("x", "op", call="m:f", timeout_s=10)
    with pytest.raises(GraphError, match="timeout_s must be >= 1"):
        Node("x", "agent", prompt_template="p", timeout_s=0)


def test_the_timeout_survives_the_graph_state_round_trip():
    node = Node("x", "agent", prompt_template="p", timeout_s=42)
    assert Node.from_state(node.to_state()).timeout_s == 42
    # An unset timeout leaves the state as it was, so recorded candidate ids hold.
    assert "timeout_s" not in Node("y", "agent", prompt_template="p").to_state()


def test_the_verifier_declares_its_bound_in_both_modes():
    # Design mode's verifier has a bound a medium change fits, and a retry; design_fast
    # keeps the short one, without a retry.
    assert DesignMode().build().nodes["verifier"].timeout_s == DESIGN_VERIFIER_TIMEOUT_S
    assert DesignFastMode().build().nodes["verifier"].timeout_s == VERIFIER_TIMEOUT_S
    assert DesignFastMode().build().nodes["verifier"].retry is None


def test_an_inline_verifier_keeps_the_route_envelope():
    # fix_pack's verifier is on the main path, not an optional fork branch: running out
    # of time there would fail the run, so it has no bound of its own.
    from refactory_dsh import registry
    from refactory_dsh.paths import REPO_ROOT
    graph = registry.resolve(REPO_ROOT, "fix_pack/make_pass")[1]().graph()
    assert graph.nodes["verifier"].timeout_s is None


def test_the_spine_gives_the_session_the_bound_and_the_command_its_overhead(tmp_path):
    state = DesignMode().state_dict()
    workflow = lowering.to_workflow(state)
    assert workflow.nodes["verifier"].timeout == DESIGN_VERIFIER_TIMEOUT_S
    assert workflow.nodes["health_checker"].timeout is None
    lowered = workers.lower_workflow(workflow, tmp_path, DshModelClient().context())
    verifier = lowered.nodes["verifier"]
    # Its bound, then its declared retry at twice the bound, then the dispatch overhead.
    assert verifier.command_timeout_s == 3 * DESIGN_VERIFIER_TIMEOUT_S + NODE_DISPATCH_OVERHEAD_S
    import json
    spec = json.loads(open(verifier.dispatch_command.rsplit(" ", 1)[1].strip("'")).read())
    assert spec["model_context"]["node_wall_s"] == DESIGN_VERIFIER_TIMEOUT_S
    # The dispatched worker rebuilds its client from that context with the bound.
    client = DshModelClient(**spec["model_context"])
    assert client.context()["node_wall_s"] == DESIGN_VERIFIER_TIMEOUT_S


def _budget_of(client, monkeypatch, tmp_path):
    seen = {}

    def capture(self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):
        import time
        seen["budget_s"] = budget_s
        seen["deadline_in"] = stop_at - time.monotonic() if stop_at is not None else None
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setenv("REFACTORY_MODEL_SCRATCH", str(tmp_path))
    monkeypatch.setattr(DshModelClient, "_run_completion", capture)
    client.complete("hello", workdir=tmp_path)
    return seen


def test_a_node_wall_caps_the_model_session(monkeypatch, tmp_path):
    node_client = DshModelClient().for_node("verifier", role="code_reviewer")
    unbounded = _budget_of(node_client, monkeypatch, tmp_path)
    assert unbounded["budget_s"] > VERIFIER_TIMEOUT_S
    bounded = _budget_of(node_client.with_wall(VERIFIER_TIMEOUT_S), monkeypatch, tmp_path)
    assert bounded["budget_s"] == VERIFIER_TIMEOUT_S
    # The deadline the session is actually stopped at, not only the reported budget.
    assert VERIFIER_TIMEOUT_S - 5 < bounded["deadline_in"] <= VERIFIER_TIMEOUT_S


def test_the_walker_gives_a_bounded_node_its_wall(tmp_path):
    from refactory_dsh.nn.executor import Executor, RunContext

    walls: list[tuple[str, int | None]] = []

    class Client:
        node = ""
        wall = None

        def for_node(self, node, role="worker"):
            bound = Client()
            bound.node = node
            return bound

        def with_wall(self, seconds):
            bound = Client()
            bound.node, bound.wall = self.node, seconds
            return bound

        def complete(self, prompt, *, system="", workdir=None):
            walls.append((self.node, self.wall))
            return "done"

    graph = Graph()
    graph.add(Node("bounded", "agent", prompt_template="p", timeout_s=123))
    graph.add(Node("free", "agent", prompt_template="p"))
    graph.connect("bounded", "free")
    ctx = RunContext(tmp_path, knobs={}, facts={}, task={})
    ctx.model = Client()
    Executor(graph, ctx).run(start="bounded")
    assert walls == [("bounded", 123), ("free", None)]
