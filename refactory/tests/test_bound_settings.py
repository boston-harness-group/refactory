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
    branch = {"op": "branch", "fork": "split", "optional": True, **extra}
    # decide reads the branch's output: a step nothing reads is refused (E-unread).
    reader = {"op": "rewire", "step": "decide",
              "sees": ["right_a", "left", "right_b", extra.get("id", "split_branch")]}
    applier, out = _ok("fuzz_fork_join", {"edits": [branch, reader]})
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
        found = sf.check_spine(applier.seed, state, walker, tmp)
        # Required, the spine's drain fails too (`check_spine` says so, comparing with a
        # walker that finished).
        assert ([(v.invariant, "rc=" in v.message) for v in found]
                == ([] if optional else [("spine", True)])), found


def test_set_knob_through_the_attempt_path_moves_a_branchs_optional():
    """What the coverage map says: an existing branch's optional changes by set_knob."""
    applier, out, head, knob = _optional_branch()
    moved = applier.attempt(out, {"op": "set_knob", "name": knob, "value": False}, {out.id})
    assert isinstance(moved, Candidate), moved
    graph = Graph.from_state(moved.state["graph"])
    assert head not in graph.optional_targets_of(graph.nodes["split"], _knobs(moved.state))


# ── an optional binding survives every operator that reshapes its fork ──────


def _bound(state: dict) -> dict[str, str]:
    graph = Graph.from_state(state["graph"])
    return {node.id: node.bound_knob("optional") for node in graph.nodes.values()
            if node.bound_knob("optional")}


@pytest.mark.parametrize("optional", [True, False])
@pytest.mark.parametrize("edit, survives", [
    ({"op": "serialize", "fork": "split"}, False),
    ({"edits": [{"op": "rewire", "step": "decide", "sees": ["right_a", "left", "right_b"]},
                {"op": "remove", "step": "split_branch"}]}, False),
    ({"op": "loop", "step": "split_branch", "judge": "agent"}, True),
    ({"op": "route", "after": "split_branch", "judge": "agent"}, True),
    ({"op": "parallelize", "steps": ["split_branch"], "copies": 1}, True),
], ids=["serialize", "remove the branch", "loop in it", "route in it", "parallelize in it"])
def test_an_operator_reshaping_the_fork_leaves_no_stranded_optional_binding(edit, survives,
                                                                             optional):
    """Where the branch's first step stays, its binding and value stay; where the fork or the
    branch goes, so do the binding and its knob, and the edit is not refused (V14)."""
    applier, out, head, knob = _optional_branch(kind="agent", text="Third.")
    out = applier.attempt(out, {"op": "set_knob", "name": knob, "value": optional}, {out.id}) \
        if not optional else out
    child = applier.attempt(out, edit, {out.id})
    assert isinstance(child, Candidate), child
    if survives:
        assert _bound(child.state) == {head: knob}
        assert child.state["knobs"][knob]["value"] is optional
    else:
        assert _bound(child.state) == {} and knob not in child.state["knobs"]
    assert sf.check_state(applier, child.state, walker=False) == []


@pytest.mark.parametrize("optional", [True, False])
def test_an_insert_before_the_branch_moves_its_optional_to_the_new_first_step(optional):
    applier, out, head, knob = _optional_branch(kind="agent", text="Third.")
    out = applier.attempt(out, {"op": "set_knob", "name": knob, "value": optional}, {out.id}) \
        if not optional else out
    child = applier.attempt(out, {"op": "insert", "before": head, "kind": "agent"}, {out.id})
    assert isinstance(child, Candidate), child
    graph = Graph.from_state(child.state["graph"])
    first = graph.nodes["split"].targets[-1]
    assert first != head and _bound(child.state) == {first: f"{first}__optional"}
    assert child.state["knobs"][f"{first}__optional"]["value"] is optional
    assert (first in graph.optional_targets_of(graph.nodes["split"], _knobs(child.state))) is optional


def test_a_first_step_rebuilt_under_its_id_gets_its_binding_back():
    """An edit that rebuilds a branch's first step under its own id (rewire's collect step)
    without its binding: the binding and the knob's value are put back."""
    from refactory_dsh.nn import core_ops

    _applier, out, head, knob = _optional_branch(kind="agent", text="Third.")
    bare = Graph.from_state(out.state["graph"]).nodes[head].to_state()
    bare["knobs"] = [k for k in bare.get("knobs") or [] if k != knob]
    bare["field_bindings"] = [p for p in bare.get("field_bindings") or [] if p[0] != "optional"]
    # The step rebuilt with its own knobs declared again at their defaults, as rewire does.
    steps = [step for step in core_ops._respec(out.state, head, bare)
             if not (step["op"] == "set_knob" and step["name"] == knob)]
    settled = nn_edits.apply_many(steps + core_ops._settle_optional(out.state, steps), out.state)
    assert _bound(settled) == {head: knob} and settled["knobs"][knob]["value"] is True


def _failing_op_branch(optional: bool) -> tuple[sf.Applier, dict]:
    """``fuzz_fork_join`` with a third branch whose op always fails, its optional bound to a
    knob set to ``optional`` (built directly: an operator adds no new callable)."""
    import copy

    from refactory_dsh.nn import Knob

    applier = sf.cached_applier("fuzz_fork_join")
    state = copy.deepcopy(applier.seed_state)
    graph = state["graph"]
    graph["nodes"]["flaky"] = {"id": "flaky", "kind": "op", "call": sf.fg.FAIL,
                               "writes": ["flaky.txt"], "knobs": ["flaky__optional"],
                               "field_bindings": [["optional", "flaky__optional"]]}
    graph["nodes"]["split"]["targets"].append("flaky")
    graph["nodes"]["merge"]["sources"].append("flaky")
    graph["edges"].append({"source": "flaky", "target": "merge", "condition": None})
    state["knobs"]["flaky__optional"] = {"spec": Knob(False, "bool").spec(), "value": optional}
    return applier, state


@pytest.mark.parametrize("optional", [True, False])
def test_the_walker_fails_an_item_on_a_failing_branch_only_when_its_knob_says_required(
        optional, tmp_path):
    """No code step needed, so this holds on every machine: the walker goes on past the
    failing branch when its knob says optional, and fails the item when it says required."""
    applier, state = _failing_op_branch(optional)
    walker = sf.run_walker(applier.seed, state, tmp_path / "walker")
    assert walker.exit_ok is optional, walker.error


@needs_spine
@pytest.mark.parametrize("optional", [True, False])
def test_the_spine_agrees_with_the_walker_on_a_failing_branchs_knob(optional, tmp_path):
    """Optional, the spine's run ends as the walker's does; required, its drain fails too
    (`check_spine` compares with a walker that finished, so it says the drain failed)."""
    applier, state = _failing_op_branch(optional)
    walker = sf.run_walker(applier.seed, state, tmp_path / "walker")
    found = sf.check_spine(applier.seed, state, walker, tmp_path)
    if optional:
        assert found == []
    else:
        assert [(v.invariant, "rc=" in v.message) for v in found] == [("spine", True)], found
