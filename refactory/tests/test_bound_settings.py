"""Two settings the search moves with ``set_knob`` once they are bound to a knob: a loop
gate's ``max_iterations`` and a fork branch's ``optional``.

Both runtimes read a bound setting in one place (`Node.bound`, `Graph.optional_targets_of`):
the walker's executor directly, the spine through lowering. So the value the walker uses,
the value in the lowered graph and what the spine does with it agree for every value the
knob takes, which is what these tests hold.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import structure_fuzz as sf
from refactory_dsh.lowering import node_declarations, to_workflow
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.graph import Graph
from refactory_dsh.optim import Candidate
from test_core_operators import _ok, needs_code
from test_livemath_blocks import needs_spine


def _knobs(state: dict) -> dict:
    return {name: entry["value"] for name, entry in state["knobs"].items()}


def _optional_branch(**extra) -> tuple[sf.Applier, Candidate, str, str]:
    """``fuzz_fork_join`` with a branch added as optional: the applier, the candidate, the
    branch's first step and its knob."""
    applier, out = _ok("fuzz_fork_join", {"op": "branch", "fork": "split", "optional": True,
                                          **extra})
    graph = Graph.from_state(out.state["graph"])
    head = graph.nodes["split"].targets[-1]
    return applier, out, head, graph.nodes[head].bound_knob("optional")


@pytest.mark.parametrize("value", [True, False])
def test_a_branchs_optional_is_one_value_on_the_walker_and_in_the_lowered_graph(value):
    _applier, out, head, knob = _optional_branch()
    assert knob == f"{head}__optional"
    state = nn_edits.configure(out.state, knob, value)
    graph = Graph.from_state(state["graph"])
    walker = graph.optional_targets_of(graph.nodes["split"], _knobs(state))
    lowered = to_workflow(state).nodes["split"].optional_targets
    assert (head in walker) is value and list(walker) == list(lowered)
    assert bool(node_declarations(state)[head].get("optional")) is value


def test_a_branch_given_optional_false_is_still_a_setting_and_its_readers_are_guarded():
    """'optional': false makes the knob too, so the search can turn it on later; what reads
    the branch's output is guarded from the start, as it is for an optional branch."""
    applier, out = _ok("fuzz_draft_review", {"op": "parallelize", "steps": ["draft"], "copies": 1})
    more = applier.attempt(out, {"op": "branch", "fork": "draft_fork", "optional": False,
                                 "kind": "agent", "text": "Try a bolder draft."}, {out.id})
    assert isinstance(more, Candidate), more
    graph = Graph.from_state(more.state["graph"])
    head = graph.nodes["draft_fork"].targets[-1]
    assert graph.optional_targets_of(graph.nodes["draft_fork"], _knobs(more.state)) == ()
    assert more.state["knobs"][f"{head}__optional"]["value"] is False
    assert "{% if draft_fork_branch %}" in more.state["knobs"]["review_task"]["value"]


def test_an_optional_binding_off_a_branch_head_is_refused():
    applier, out, head, knob = _optional_branch()
    state = {**out.state, "graph": dict(out.state["graph"])}
    nodes = dict(state["graph"]["nodes"])
    join = next(n for n, node in nodes.items() if node["kind"] == "join")
    nodes[join] = {**nodes[join], "knobs": [knob], "field_bindings": [["optional", knob]]}
    state["graph"] = {**state["graph"], "nodes": nodes}
    import importlib

    issues = importlib.import_module("refactory_dsh.nn.validate").validate_state(
        state, invariants=applier.invariants)
    assert any("optional binding is only for a fork branch" in str(issue) for issue in issues)


def _loop() -> tuple[sf.Applier, Candidate, str, str]:
    applier, out = _ok("fuzz_draft_review", {"op": "loop", "step": "review", "judge": "agent"})
    gate = next(n for n in Graph.from_state(out.state["graph"]).nodes.values() if n.kind == "gate")
    return applier, out, gate.id, gate.bound_knob("max_iterations")


@pytest.mark.parametrize("value", [1, 4])
def test_a_loops_limit_is_one_value_on_the_walker_and_in_the_lowered_graph(value):
    _applier, out, gate, knob = _loop()
    state = nn_edits.configure(out.state, knob, value)
    graph = Graph.from_state(state["graph"])
    assert graph.nodes[gate].bound("max_iterations", _knobs(state)) == value
    assert to_workflow(state).nodes[gate].max_iterations == value


@needs_spine
@pytest.mark.parametrize("setting", ["loop limit", "optional", "required"])
def test_a_bound_setting_runs_alike_on_the_walker_and_the_spine(setting):
    if setting == "loop limit":
        applier, out, _gate, knob = _loop()
        state = nn_edits.configure(out.state, knob, 1)
    else:
        applier, out, _head, knob = _optional_branch()
        state = nn_edits.configure(out.state, knob, setting == "optional")
    assert sf.check_state(applier, state, walker=True, spine=True) == []


@needs_spine
@needs_code
def test_a_failing_branch_fails_the_item_only_when_its_knob_says_required():
    """The same failing branch: optional, the item runs on; set back to required by its
    knob, the item fails. The walker and the spine agree on both."""
    applier, out, head, knob = _optional_branch(kind="code", id="flaky",
                                                text="def main(inp):\n    raise ValueError('no')\n")
    for optional in (True, False):
        state = nn_edits.configure(out.state, knob, optional)
        tmp = Path(tempfile.mkdtemp(prefix="bound-settings-"))
        walker = sf.run_walker(applier.seed, state, tmp / "walker")
        assert walker.exit_ok is optional, (optional, walker.error)
        assert sf.check_spine(applier.seed, state, walker, tmp) == []


def test_set_knob_through_the_attempt_path_moves_a_branchs_optional():
    """What the coverage map says: an existing branch's optional changes by set_knob."""
    applier, out, head, knob = _optional_branch()
    moved = applier.attempt(out, {"op": "set_knob", "name": knob, "value": False}, {out.id})
    assert isinstance(moved, Candidate), moved
    graph = Graph.from_state(moved.state["graph"])
    assert head not in graph.optional_targets_of(graph.nodes["split"], _knobs(moved.state))
