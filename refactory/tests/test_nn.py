"""The mode library: state round-trip, the validator, and the edit algebra.

Phase 1 and phase 2 gates of ``docs/plans/pythonic-modes.md``.
"""

from __future__ import annotations

import copy
import json
import random

import pytest

from refactory_dsh import nn


class Fixture(nn.Mode):
    """A four-node mode with a retry gate, used as the valid-state baseline."""

    name = "fixture"
    description = "four nodes, one gate, two files of data flow"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "rounds": nn.Knob(3, "int", lo=1, hi=8, description="how many rounds"),
            "label": nn.Knob("a", "enum", enum=("a", "b")),
            "seed": nn.Knob(7, "int", frozen=True, description="reproducibility"),
        }

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("start", "op", call="json:dumps",
                          writes=("a.json",), knobs=("rounds",)))
        graph.add(nn.Node("mid", "op", call="json:loads",
                          reads=("a.json",), writes=("b.json",), knobs=("label",)))
        graph.add(nn.Node("gate", "gate", evaluator="json:dumps", max_iterations=3))
        graph.add(nn.Node("stop", "op", call="json:dumps",
                          reads=("b.json",), writes=("best.json",), produces=("best",),
                          knobs=("seed",)))
        graph.connect("start", "mid")
        graph.connect("mid", "gate")
        graph.connect("gate", "mid", condition="reloop")
        graph.connect("gate", "stop", condition="proceed")
        return graph

    def score(self, run: nn.RunResult) -> float:
        return float(run.facts.get("best", 0.0))

    def invariants(self) -> tuple[nn.Invariant, ...]:
        return (nn.fact_produced("best"),)

    def insertable(self) -> tuple[nn.Node, ...]:
        return (
            nn.Node("probe", "op", call="json:dumps", writes=("probe.json",)),
            nn.Node("retry_gate", "gate", evaluator="json:dumps", max_iterations=3),
        )


def _state() -> dict:
    return Fixture().state_dict()


def _validate(state: dict, *, resolve_calls: bool = False) -> list[nn.Issue]:
    return nn.validate(
        nn.Graph.from_state(state["graph"]),
        {name: entry["value"] for name, entry in state["knobs"].items()},
        declared_knobs={n: nn.Knob.from_spec(e["spec"]) for n, e in state["knobs"].items()},
        invariants=Fixture().invariants(),
        resolve_calls=resolve_calls,
    )


def _codes(state: dict, **kwargs) -> set[str]:
    return {issue.code for issue in _validate(state, **kwargs)}


def _audit(graph: nn.Graph) -> list[str]:
    """Structural facts a runnable mode must have, written without the validator.

    This exists so the property test below can compare the validator against an
    independently expressed expectation instead of against itself.
    """
    problems: list[str] = []
    if graph.start not in graph.nodes:
        problems.append("the start node does not exist")
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        out = graph.out_edges(node_id)
        if node.kind == "gate":
            if len(out) < 2:
                problems.append(f"{node_id}: a gate with {len(out)} out-edges")
            if any(edge.condition is None for edge in out):
                problems.append(f"{node_id}: a gate with an unconditional out-edge")
        else:
            if len(out) > 1:
                problems.append(f"{node_id}: a non-gate node with {len(out)} out-edges")
            if any(edge.condition is not None for edge in out):
                problems.append(f"{node_id}: a non-gate node with a verdict edge")
    if not graph.terminals():
        problems.append("no terminal node")
    produced = {name for node in graph.nodes.values() for name in node.writes}
    for node_id in sorted(graph.nodes):
        for name in graph.nodes[node_id].reads:
            if name not in produced:
                problems.append(f"{node_id}: reads {name!r}, nothing produces it")
    return problems


# ── phase 1: state ─────────────────────────────────────────────────────────
def test_genesis_state_validates_cleanly():
    assert _validate(_state()) == []


def test_state_round_trips_through_a_new_instance():
    original = _state()
    rebuilt = Fixture(state=copy.deepcopy(original))
    assert rebuilt.state_dict() == original


def test_state_is_canonical():
    state = _state()
    graph = nn.Graph.from_state(state["graph"])
    assert graph.to_state() == state["graph"]
    assert nn.Graph.from_state(graph.to_state()).to_state() == graph.to_state()


def test_state_carries_no_scoring_surface():
    """D4: nothing an optimizer can write can reach score() or invariants().

    Checked structurally rather than as a substring: a node field named
    `score_effect` says whether a node can change the number, which is the opposite
    of exposing the number.
    """
    state = _state()

    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            found = set(node)
            for value in node.values():
                found |= keys(value)
            return found
        if isinstance(node, list):
            return {k for item in node for k in keys(item)}
        return set()

    found = keys(state)
    assert "score" not in found, "the state exposes a score"
    assert "invariants" not in found, "the state exposes the invariants"
    assert "score()" not in json.dumps(state, sort_keys=True)


def test_default_score_means_unscored():
    class Unscored(nn.Mode):
        def build(self) -> nn.Graph:
            graph = nn.Graph()
            graph.add(nn.Node("only", "op", call="json:dumps"))
            return graph

    mode = Unscored()
    assert mode.is_scored() is False
    assert mode.score(nn.RunResult()) == 0.0
    assert Fixture().is_scored() is True


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda s: s.update({"extra": 1}), "unknown state dict field"),
        (lambda s: s["knobs"]["rounds"].update({"value": 99}), "upper bound"),
        (lambda s: s["knobs"]["rounds"].update({"value": 2.5}), "integer"),
        (lambda s: s["knobs"]["rounds"].pop("spec"), "needs a 'value' and a 'spec'"),
        (lambda s: s["graph"]["nodes"]["start"].update({"wat": 1}), "unknown node field"),
        (lambda s: s.update({"version": 99}), "different layout"),
        (lambda s: s.pop("graph"), "needs a 'graph'"),
    ],
)
def test_strict_load_refuses_a_bad_state(mutate, expected):
    state = _state()
    mutate(state)
    with pytest.raises(nn.StateError) as excinfo:
        Fixture(state=state)
    assert expected in str(excinfo.value)


def test_a_knob_bad_default_is_an_authoring_error():
    with pytest.raises(nn.KnobError):
        nn.Knob(10, "int", lo=1, hi=5)


# ── phase 1: the validator ─────────────────────────────────────────────────
def test_v1_missing_start():
    assert "V1" in _codes(nn.edits.apply({"op": "remove_node", "id": "start"}, _state()))


def test_v2_unreachable_node():
    codes = _codes(
        nn.edits.apply({"op": "remove_edge", "source": "start", "target": "mid"}, _state())
    )
    assert "V2" in codes


def test_v3_no_terminal():
    state = _state()
    state = nn.edits.apply(
        {"op": "remove_edge", "source": "gate", "target": "stop", "condition": "proceed"}, state
    )
    state = nn.edits.apply(
        {"op": "add_edge", "source": "stop", "target": "start", "condition": None}, state
    )
    graph = nn.Graph.from_state(state["graph"])
    assert graph.terminals() == []
    assert "V3" in _codes(state)


def test_v4_non_gate_fan_out():
    state = nn.edits.apply(
        {"op": "add_edge", "source": "mid", "target": "stop", "condition": None}, _state()
    )
    assert "V4" in _codes(state)


def test_v5_gate_needs_two_verdicts():
    state = nn.edits.apply(
        {"op": "remove_edge", "source": "gate", "target": "stop", "condition": "proceed"},
        _state(),
    )
    assert "V5" in _codes(state)


def test_v6_dangling_read():
    codes = _codes(nn.edits.apply({"op": "remove_node", "id": "mid"}, _state()))
    assert "V6" in codes


def test_v7_undeclared_knob():
    state = nn.edits.apply(
        {"op": "set_node_field", "id": "mid", "field": "knobs", "value": ["label", "nope"]},
        _state(),
    )
    assert "V7" in _codes(state)


def test_v7_dead_knob():
    """A knob nothing consumes is refused, so the search cannot move nothing."""
    state = nn.edits.apply(
        {"op": "set_node_field", "id": "start", "field": "knobs", "value": []}, _state()
    )
    codes = _codes(state)
    assert "V7" in codes


def test_v8_unresolvable_call():
    state = nn.edits.apply(
        {"op": "set_node_field", "id": "start", "field": "call",
         "value": "no.such.module:function"},
        _state(),
    )
    assert "V8" in _codes(state, resolve_calls=True)
    assert "V8" not in _codes(state, resolve_calls=False)


def test_v9_out_of_bounds_value():
    state = _state()
    state["knobs"]["rounds"]["value"] = 99
    assert "V9" in _codes(state)


def test_v10_cycle_without_gate():
    state = nn.edits.apply(
        {"op": "add_edge", "source": "mid", "target": "start", "condition": None}, _state()
    )
    assert "V10" in _codes(state)


def test_mode_invariant_guards_the_metric():
    """Removing the node that writes the scored fact is refused, with a reason."""
    state = nn.edits.apply({"op": "remove_node", "id": "stop"}, _state())
    issues = _validate(state)
    named = [i for i in issues if i.code == "I:fact_produced"]
    assert named, [str(i) for i in issues]
    assert "score reads" in named[0].message


def test_every_issue_carries_a_reason():
    state = nn.edits.apply({"op": "remove_node", "id": "mid"}, _state())
    for issue in _validate(state):
        assert issue.code
        assert len(issue.message) > 10
        assert str(issue)


# ── phase 2: the edit algebra ──────────────────────────────────────────────
def test_edit_never_mutates_its_input():
    state = _state()
    before = copy.deepcopy(state)
    nn.edits.apply({"op": "set_knob", "name": "rounds", "value": 5}, state)
    assert state == before


def test_frozen_knob_cannot_be_moved():
    with pytest.raises(nn.edits.EditError, match="frozen"):
        nn.edits.apply({"op": "set_knob", "name": "seed", "value": 8}, _state())


def test_remove_node_is_naive_and_leaves_the_hole():
    """No operator rewires to keep the graph valid; the validator reports it."""
    state = nn.edits.apply({"op": "remove_node", "id": "mid"}, _state())
    graph = nn.Graph.from_state(state["graph"])
    assert "mid" not in graph.nodes
    assert "V6" in _codes(state)


def test_reorder_refuses_a_data_dependency():
    reason = nn.edits.precheck({"op": "reorder", "first": "start", "second": "mid"}, _state())
    assert reason is not None
    assert "data dependency" in reason


def test_insert_after_splices_cleanly():
    state = _state()
    node = {"id": "probe", "kind": "op", "call": "json:dumps", "writes": ["probe.json"]}
    moved = nn.edits.apply({"op": "insert_after", "target": "start", "node": node}, state)
    graph = nn.Graph.from_state(moved["graph"])
    assert sorted(graph.nodes) == ["gate", "mid", "probe", "start", "stop"]
    assert [e.target for e in graph.out_edges("start")] == ["probe"]
    assert [e.target for e in graph.out_edges("probe")] == ["mid"]
    assert _audit(graph) == []
    assert _validate(moved) == []


def test_loop_macro_adds_a_bounded_retry():
    state = _state()
    gate = {"id": "again", "kind": "gate", "evaluator": "json:dumps", "max_iterations": 2}
    moved = nn.edits.apply({"op": "loop", "target": "mid", "gate": gate}, state)
    graph = nn.Graph.from_state(moved["graph"])
    verdicts = {e.condition: e.target for e in graph.out_edges("again")}
    assert verdicts == {"reloop": "mid", "proceed": "gate"}, (
        "the retry edge has to be the explicit 'reloop' the validator and the TS "
        "spine recognise; an unlabelled edge leaves V10 to infer the cycle's budget"
    )
    assert _validate(moved) == []


def test_wrap_on_a_gate_needs_a_gate():
    state = _state()
    plain = {"id": "probe", "kind": "op", "call": "json:dumps", "writes": ["probe.json"]}
    reason = nn.edits.precheck({"op": "wrap", "target": "gate", "node": plain}, state)
    assert reason is not None
    assert "gate" in reason
    wrapper = {"id": "probe", "kind": "gate", "evaluator": "json:dumps", "max_iterations": 2}
    moved = nn.edits.apply({"op": "wrap", "target": "gate", "node": wrapper}, state)
    assert _validate(moved) == []


def test_bypass_reconnects_predecessor_to_successor():
    """Bypass rewires the edges and leaves the data hole for the validator.

    Removing ``b`` reconnects ``a`` to ``c``, but ``c`` still reads what ``b``
    produced. That dangling read is reported rather than papered over, which is
    the difference between a naive operator and a repairing one.
    """
    chain = nn.Graph()
    chain.add(nn.Node("a", "op", call="json:dumps", writes=("a.json",)))
    chain.add(nn.Node("b", "op", call="json:loads", reads=("a.json",), writes=("b.json",)))
    chain.add(nn.Node("c", "op", call="json:loads", reads=("b.json",), writes=("best",)))
    chain.connect("a", "b")
    chain.connect("b", "c")
    state = {"version": nn.STATE_VERSION, "graph": chain.to_state(), "knobs": {}}
    moved = nn.edits.apply({"op": "bypass", "target": "b"}, state)
    graph = nn.Graph.from_state(moved["graph"])
    assert sorted(graph.nodes) == ["a", "c"]
    assert [e.target for e in graph.out_edges("a")] == ["c"]
    assert _audit(graph) == ["c: reads 'b.json', nothing produces it"]


def test_bypass_refuses_the_start_node():
    reason = nn.edits.precheck({"op": "bypass", "target": "start"}, _state())
    assert reason is not None
    assert "start node" in reason


def test_bypass_refuses_to_create_a_self_loop():
    """A predecessor that is also a successor would make a node feed itself."""
    reason = nn.edits.precheck({"op": "bypass", "target": "mid"}, _state())
    assert reason is not None
    assert "loop" in reason


def test_every_macro_expands_to_primitives_only():
    state = _state()
    samples = {
        "insert_after": {"target": "start",
                         "node": {"id": "p1", "kind": "op", "call": "json:dumps"}},
        "insert_before": {"target": "start",
                          "node": {"id": "p2", "kind": "op", "call": "json:dumps"}},
        "wrap": {"target": "mid",
                 "node": {"id": "p3", "kind": "op", "call": "json:dumps"}},
        "bypass": {"target": "mid"},
        "duplicate": {"target": "mid"},
        "split": {"target": "start",
                  "into": [{"id": "p4", "kind": "op", "call": "json:dumps"},
                           {"id": "p5", "kind": "op", "call": "json:dumps"}]},
        "reorder": {"first": "mid", "second": "gate"},
        "loop": {"target": "mid",
                 "gate": {"id": "p6", "kind": "gate", "evaluator": "json:dumps"}},
    }
    macros = [name for name in nn.edits.names() if nn.edits.is_macro(name)]
    assert sorted(macros) == sorted(samples)
    for name, edit in samples.items():
        steps = next(
            spec.expand({"op": name, **edit}, state)
            for spec in [nn.edits._specs()[name]]  # noqa: SLF001 - the catalogue is the subject
        )
        assert steps, name
        for step in steps:
            assert not nn.edits.is_macro(step["op"]), (name, step)


def test_presets_are_closed():
    # text_patch is deliberately excluded from every preset that random search
    # samples from (including broad) so seeded runs stay unchanged — correction 3.
    all_names = nn.edits.names()
    broad = sorted(nn.edits.PRESETS["broad"])
    assert "text_patch" in all_names, "text_patch must be a registered edit"
    assert "text_patch" not in broad, "text_patch must not be in 'broad' preset"
    assert sorted(set(all_names) - {"text_patch"}) == broad
    for name in nn.edits.PRESETS["knob_only"]:
        assert nn.edits._specs()[name].kind == "knob"  # noqa: SLF001
    with pytest.raises(nn.edits.EditError):
        nn.edits.enabled("nonsense")


def test_random_edit_streams_are_never_repaired():
    """Phase 2 gate.

    A random stream of edits may produce states the validator refuses. The two
    properties that must hold throughout are: the input state is never mutated,
    and any state the validator accepts also passes an independently written
    structural audit.
    """
    state = _state()
    rng = random.Random(20260913)
    applied = refused = invalid = 0
    for _ in range(400):
        edit = nn.edits.random_edit(
            state, preset="broad", insertable=Fixture().insertable(), rng=rng
        )
        if edit is None:
            continue
        before = copy.deepcopy(state)
        try:
            candidate = nn.edits.apply(edit, state)
        except nn.edits.EditError:
            refused += 1
            assert state == before, edit
            continue
        applied += 1
        assert state == before, edit
        graph = nn.Graph.from_state(candidate["graph"])
        assert nn.Graph.from_state(graph.to_state()).to_state() == graph.to_state()
        issues = _validate(candidate)
        if not issues:
            assert _audit(graph) == [], (edit, _audit(graph))
        else:
            invalid += 1
            for issue in issues:
                assert issue.message
        state = candidate
    assert applied > 100, f"only {applied} edits applied"
    assert refused > 0, "no edit was refused; the stream is not exercising preconditions"
    assert invalid > 0, "no candidate was invalid; the stream is not exercising the validator"
