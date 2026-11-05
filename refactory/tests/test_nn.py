"""The mode library: state round-trip, the validator, and the edit algebra.

Phase 1 and phase 2 gates of ``docs/plans/pythonic-modes.md``.
"""

from __future__ import annotations

import copy
import json
import random
import shutil
from pathlib import Path

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
    # Directly mutate the state dict to give a node an undeclared knob.
    state = _state()
    state["graph"]["nodes"]["mid"]["knobs"] = ["label", "nope"]
    assert "V7" in _codes(state)


def test_v7_dead_knob():
    """A knob nothing consumes is refused, so the search cannot move nothing."""
    state = _state()
    state["graph"]["nodes"]["start"]["knobs"] = []
    codes = _codes(state)
    assert "V7" in codes


def test_v8_unresolvable_call():
    state = _state()
    state["graph"]["nodes"]["start"]["call"] = "no.such.module:function"
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


# ── F14: field bindings and auto system-prompt knob ────────────────────────


def test_field_bindings_roundtrip():
    """field_bindings survive to_state -> from_state. Python runtime."""
    node = nn.Node("a", "agent", prompt_template="t",
                   knobs=("my_knob",),
                   field_bindings=(("system_prompt_template", "my_knob"),))
    state = node.to_state()
    assert state["field_bindings"] == [["system_prompt_template", "my_knob"]]
    restored = nn.Node.from_state(state)
    assert restored.field_bindings == (("system_prompt_template", "my_knob"),)


def test_field_bindings_sparse_serialization():
    """Empty field_bindings omitted from to_state(). Python runtime."""
    node = nn.Node("a", "agent", prompt_template="t")
    state = node.to_state()
    assert "field_bindings" not in state


def _v14_state(field_name, kind="op", knob_kind="str"):
    """A minimal state with a field_binding for V14 testing."""
    graph = nn.Graph()
    if kind == "agent":
        graph.add(nn.Node("n", "agent", prompt_template="t", knobs=("k",),
                          field_bindings=((field_name, "k"),)))
    else:
        graph.add(nn.Node("n", "op", call="json:dumps", knobs=("k",),
                          field_bindings=((field_name, "k"),)))
    value = "x" if knob_kind == "str" else 2
    state = {"version": nn.STATE_VERSION, "graph": graph.to_state(), "knobs": {
        "k": {"value": value, "spec": {"default": value, "kind": knob_kind,
              "lo": None, "hi": None, "enum": [], "values": [],
              "frozen": False, "description": "", "required": False}},
    }}
    return state


def test_v14_unbindable_field():
    """role is not in BINDABLE_FIELDS and is refused. Python runtime."""
    state = _v14_state("role")
    issues = nn.validate_state(state, resolve_calls=False)
    assert any(i.code == "V14" for i in issues)


def test_v14_unbindable_tools():
    """tools is not in BINDABLE_FIELDS. Python runtime."""
    state = _v14_state("tools", kind="agent")
    issues = nn.validate_state(state, resolve_calls=False)
    assert any(i.code == "V14" for i in issues)


def test_v14_unbindable_tool_policy():
    """tool_policy is not in BINDABLE_FIELDS. Python runtime."""
    state = _v14_state("tool_policy", kind="agent")
    issues = nn.validate_state(state, resolve_calls=False)
    assert any(i.code == "V14" for i in issues)


def test_v14_unbindable_notes():
    """notes is not in BINDABLE_FIELDS. Python runtime."""
    state = _v14_state("notes")
    issues = nn.validate_state(state, resolve_calls=False)
    assert any(i.code == "V14" for i in issues)


def test_v14_unbindable_limit():
    """limit is not in BINDABLE_FIELDS. Python runtime."""
    state = _v14_state("limit")
    issues = nn.validate_state(state, resolve_calls=False)
    assert any(i.code == "V14" for i in issues)


def test_v14_parallelism_only_data_nodes():
    """parallelism binding on a non-data node is refused. Python runtime."""
    state = _v14_state("parallelism", knob_kind="int")
    issues = nn.validate_state(state, resolve_calls=False)
    assert [i.message for i in issues if i.code == "V14"] == [
        "parallelism binding is only for data nodes"]


def test_v14_valid_binding():
    """A valid system_prompt_template binding passes V14. Python runtime."""
    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="t",
                      knobs=("my_knob",),
                      field_bindings=(("system_prompt_template", "my_knob"),)))
    state = {"version": nn.STATE_VERSION, "graph": graph.to_state(), "knobs": {
        "my_knob": {"value": "hello", "spec": {"default": "hello", "kind": "str",
                    "lo": None, "hi": None, "enum": [], "values": [],
                    "frozen": False, "description": "", "required": False}},
    }}
    issues = nn.validate_state(state, resolve_calls=False)
    v14 = [i for i in issues if i.code == "V14"]
    assert not v14, v14


def test_v14_nonexistent_knob():
    """Binding to an undeclared knob is refused. Python runtime."""
    state = _state()
    state["graph"]["nodes"]["start"]["knobs"] = ["rounds", "ghost"]
    state["graph"]["nodes"]["start"]["field_bindings"] = [["system_prompt_template", "ghost"]]
    assert "V14" in _codes(state)


def test_auto_system_prompt_knob_static_template():
    """Agent node gets auto system-prompt knob with static template. Python runtime."""
    class StaticMode(nn.Mode):
        name = "static_test"
        def build(self):
            g = nn.Graph()
            g.add(nn.Node("a", "agent", prompt_template="Do it",
                          system_prompt_template="You are helpful"))
            return g

    mode = StaticMode()
    state = mode.state_dict()
    assert "a__system_prompt" in state["knobs"]
    assert state["knobs"]["a__system_prompt"]["value"] == "You are helpful"
    node_state = state["graph"]["nodes"]["a"]
    assert "a__system_prompt" in node_state["knobs"]
    assert ["system_prompt_template", "a__system_prompt"] in node_state["field_bindings"]


def test_auto_system_prompt_knob_slotted_template():
    """Agent node with slotted system_prompt_template gets auto-knob (not suppressed).
    Editing it to a value missing a slot is refused. Python runtime."""
    class SlottedMode(nn.Mode):
        name = "slotted_test"
        @classmethod
        def declare(cls):
            return {}
        def build(self):
            g = nn.Graph()
            g.add(nn.Node("a", "agent", prompt_template="Do it",
                          system_prompt_template="You are {{role}}"))
            return g

    mode = SlottedMode()
    state = mode.state_dict()
    assert "a__system_prompt" in state["knobs"]
    assert state["knobs"]["a__system_prompt"]["value"] == "You are {{role}}"
    # Editing to a value missing {{role}} is refused
    with pytest.raises(nn.edits.EditError, match="role"):
        nn.edits.apply({"op": "set_knob", "name": "a__system_prompt",
                        "value": "You are a helper"}, state)
    # Editing to a value keeping {{role}} works
    new_state = nn.edits.apply({"op": "set_knob", "name": "a__system_prompt",
                                "value": "Be an expert {{role}}"}, state)
    assert new_state["knobs"]["a__system_prompt"]["value"] == "Be an expert {{role}}"


def test_auto_system_prompt_knob_empty_template():
    """Agent node with empty system_prompt_template gets auto-knob with default ''.
    Python runtime."""
    class EmptyMode(nn.Mode):
        name = "empty_test"
        def build(self):
            g = nn.Graph()
            g.add(nn.Node("a", "agent", prompt_template="Do it"))
            return g

    mode = EmptyMode()
    state = mode.state_dict()
    assert "a__system_prompt" in state["knobs"]
    assert state["knobs"]["a__system_prompt"]["value"] == ""


def test_auto_knob_collision_detection():
    """Auto-knob is NOT added when the name collides with a user-declared knob.
    Python runtime."""
    class CollisionMode(nn.Mode):
        name = "collision_test"
        @classmethod
        def declare(cls):
            return {"agent__system_prompt": nn.Knob("custom", "str")}
        def build(self):
            g = nn.Graph()
            g.add(nn.Node("agent", "agent", prompt_template="Do it",
                          knobs=("agent__system_prompt",)))
            return g

    mode = CollisionMode()
    state = mode.state_dict()
    assert state["knobs"]["agent__system_prompt"]["value"] == "custom"
    # The node should not have a field_binding since the auto-knob was skipped
    node_state = state["graph"]["nodes"]["agent"]
    bindings = node_state.get("field_bindings", [])
    assert not bindings


def test_lowering_resolves_field_bindings():
    """Lowering resolves field_bindings so the TS spine gets the resolved value.
    TS runtime (via lowering)."""
    from refactory_dsh.lowering import to_workflow

    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="t",
                      knobs=("my_knob",),
                      field_bindings=(("system_prompt_template", "my_knob"),)))
    state = {"graph": graph.to_state(), "knobs": {
        "my_knob": {"value": "You are a helper.", "spec": {"default": "", "kind": "str",
                    "lo": None, "hi": None, "enum": [], "values": [],
                    "frozen": False, "description": "", "required": False}},
    }}
    wf = to_workflow(state)
    assert wf.nodes["a"].system_prompt == "You are a helper."


def test_lowering_strips_field_bindings():
    """Lowered IR does not contain field_bindings. TS runtime."""
    from refactory_dsh.lowering import to_workflow
    import json as _json

    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="t",
                      knobs=("my_knob",),
                      field_bindings=(("system_prompt_template", "my_knob"),)))
    state = {"graph": graph.to_state(), "knobs": {
        "my_knob": {"value": "hello", "spec": {"default": "", "kind": "str",
                    "lo": None, "hi": None, "enum": [], "values": [],
                    "frozen": False, "description": "", "required": False}},
    }}
    wf = to_workflow(state)
    ir_json = _json.dumps(wf.to_dict())
    assert "field_bindings" not in ir_json


def test_field_binding_precedence():
    """Binding value wins over the node's direct field value. Both runtimes."""
    from refactory_dsh.lowering import to_workflow

    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="t",
                      system_prompt_template="X",
                      knobs=("my_knob",),
                      field_bindings=(("system_prompt_template", "my_knob"),)))
    state = {"graph": graph.to_state(), "knobs": {
        "my_knob": {"value": "Y", "spec": {"default": "X", "kind": "str",
                    "lo": None, "hi": None, "enum": [], "values": [],
                    "frozen": False, "description": "", "required": False}},
    }}
    wf = to_workflow(state)
    assert wf.nodes["a"].system_prompt == "Y"


def test_data_slot_preservation_prompt_template():
    """set_knob on a knob bound to prompt_template with slots refuses missing slots.
    Python runtime."""
    graph = nn.Graph()
    graph.add(nn.Node("a", "agent",
                      prompt_template="Tell me about {{user_prompt}} and {{context}}",
                      knobs=("pt_knob",),
                      field_bindings=(("prompt_template", "pt_knob"),)))
    state = {"version": nn.STATE_VERSION, "graph": graph.to_state(), "knobs": {
        "pt_knob": {"value": "Tell me about {{user_prompt}} and {{context}}",
                    "spec": {"default": "Tell me about {{user_prompt}} and {{context}}",
                             "kind": "str", "lo": None, "hi": None, "enum": [], "values": [],
                             "frozen": False, "description": "", "required": False}},
    }}
    with pytest.raises(nn.edits.EditError, match="user_prompt"):
        nn.edits.apply({"op": "set_knob", "name": "pt_knob",
                        "value": "Tell me about {{context}}"}, state)


def test_data_slot_preservation_system_prompt_template():
    """set_knob on a knob bound to system_prompt_template with slots refuses missing
    slots. Python runtime."""
    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="t",
                      system_prompt_template="Use {{skill_section}} and {{context}}",
                      knobs=("sp_knob",),
                      field_bindings=(("system_prompt_template", "sp_knob"),)))
    state = {"version": nn.STATE_VERSION, "graph": graph.to_state(), "knobs": {
        "sp_knob": {"value": "Use {{skill_section}} and {{context}}",
                    "spec": {"default": "Use {{skill_section}} and {{context}}",
                             "kind": "str", "lo": None, "hi": None, "enum": [], "values": [],
                             "frozen": False, "description": "", "required": False}},
    }}
    with pytest.raises(nn.edits.EditError, match="skill_section"):
        nn.edits.apply({"op": "set_knob", "name": "sp_knob",
                        "value": "Use {{context}}"}, state)


def test_frozen_knob_field_binding_refused():
    """A frozen knob bound to a field is refused by set_knob. Python runtime."""
    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="t",
                      knobs=("fk",),
                      field_bindings=(("system_prompt_template", "fk"),)))
    state = {"version": nn.STATE_VERSION, "graph": graph.to_state(), "knobs": {
        "fk": {"value": "hello", "spec": {"default": "hello", "kind": "str",
               "lo": None, "hi": None, "enum": [], "values": [],
               "frozen": True, "description": "", "required": False}},
    }}
    with pytest.raises(nn.edits.EditError, match="frozen"):
        nn.edits.apply({"op": "set_knob", "name": "fk", "value": "new"}, state)


def test_configure_frozen_field_binding_succeeds():
    """configure() on a frozen knob succeeds (person, not the optimizer). Python runtime."""
    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="t",
                      knobs=("fk",),
                      field_bindings=(("system_prompt_template", "fk"),)))
    state = {"version": nn.STATE_VERSION, "graph": graph.to_state(), "knobs": {
        "fk": {"value": "hello", "spec": {"default": "hello", "kind": "str",
               "lo": None, "hi": None, "enum": [], "values": [],
               "frozen": True, "description": "", "required": False}},
    }}
    new_state = nn.edits.configure(state, "fk", "new value")
    assert new_state["knobs"]["fk"]["value"] == "new value"


def test_patch_on_system_prompt_knob():
    """set_knob's patch works on the auto system-prompt knob. Python runtime."""
    class PatchMode(nn.Mode):
        name = "patch_test"
        def build(self):
            g = nn.Graph()
            g.add(nn.Node("a", "agent", prompt_template="Do it",
                          system_prompt_template="Be helpful"))
            return g

    state = PatchMode().state_dict()
    assert "a__system_prompt" in state["knobs"]
    new_state = nn.edits.apply({"op": "set_knob", "name": "a__system_prompt",
                                "patch": [{"kind": "replace", "target": "Be helpful",
                                           "content": "Be concise"}]}, state)
    assert new_state["knobs"]["a__system_prompt"]["value"] == "Be concise"


def test_patch_and_value_reach_both_runtimes_alike():
    """A system prompt moved by set_knob's patch is the same state as the same text
    set by value, so the walker sends it and lowering hands it to the spine. Both runtimes."""
    from refactory_dsh import lowering
    from refactory_dsh.nn.executor import execute

    class PatchMode(nn.Mode):
        name = "patch_two_runtime"
        def build(self):
            g = nn.Graph()
            g.add(nn.Node("a", "agent", prompt_template="Do it", writes=("out.txt",),
                          system_prompt_template="Be helpful"))
            return g

    state = PatchMode().state_dict()
    patched = nn.edits.apply({"op": "set_knob", "name": "a__system_prompt",
                              "patch": [{"kind": "append", "content": "Be brief."}]}, state)
    text = patched["knobs"]["a__system_prompt"]["value"]
    assert text == "Be helpful\n\nBe brief.\n"
    valued = nn.edits.apply({"op": "set_knob", "name": "a__system_prompt", "value": text}, state)
    assert patched == valued
    # Spine: the lowered node carries the patched prompt.
    assert lowering.to_workflow(patched).nodes["a"].system_prompt == text
    # Walker: the model receives it.
    captured = {}

    class Client:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            captured["system"] = system
            return "reply"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    import tempfile
    graph = nn.Graph.from_state(patched["graph"])
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "out.txt").write_text("")
        result = execute(graph, Path(tmp), model=Client(),
                         knobs={k: v["value"] for k, v in patched["knobs"].items()})
    assert result.exit_ok, result.error
    assert captured["system"] == text


def test_patch_slot_preservation():
    """A patch on a template-bound knob refuses if slots are removed. Python runtime."""
    graph = nn.Graph()
    graph.add(nn.Node("a", "agent",
                      prompt_template="Do it",
                      system_prompt_template="Use {{skill_section}} and {{context}}",
                      writes=("out.txt",),
                      knobs=("sp",),
                      field_bindings=(("system_prompt_template", "sp"),)))
    state = {"graph": graph.to_state(), "knobs": {
        "sp": {"value": "Use {{skill_section}} and {{context}}",
               "spec": {"default": "Use {{skill_section}} and {{context}}", "kind": "str",
                        "lo": None, "hi": None, "enum": [], "values": [],
                        "frozen": False, "description": "", "required": False}},
    }}
    # Replacing a slot with empty should fail
    with pytest.raises(nn.edits.EditError, match="template slot"):
        nn.edits.apply({"op": "set_knob", "name": "sp",
                        "patch": [{"kind": "replace", "target": " and {{context}}", "content": ""}]}, state)
    # Keeping both slots should succeed
    nn.edits.apply({"op": "set_knob", "name": "sp",
                    "patch": [{"kind": "replace", "target": "Use", "content": "Apply"}]}, state)


def test_is_structural_after_removal():
    """set_knob candidates are non-structural; add_node candidates are structural.
    Python runtime."""
    from refactory_dsh.optim.base import Candidate

    c1 = Candidate(id="a", parent=None, state={},
                   edits=[{"op": "set_knob", "name": "x", "value": 1}])
    assert not c1.is_structural()

    c2 = Candidate(id="b", parent=None, state={},
                   edits=[{"op": "add_node", "node": {}, "position": {"start": True}}])
    assert c2.is_structural()

    c3 = Candidate(id="c", parent=None, state={},
                   edits=[{"op": "remove_node", "id": "x"}])
    assert c3.is_structural()

    c4 = Candidate(id="d", parent=None, state={},
                   edits=[{"op": "remove_edge"}])
    assert c4.is_structural()


def test_livemath_knob_names_unchanged():
    """LiveMath's knob set includes the frozen auto-knob. Python runtime."""
    from refactory_packages.livemath.mode import Solver

    state = Solver().state_dict()
    knob_names = set(state["knobs"].keys())
    expected = {"data_parallelism", "max_completion_tokens", "skill",
                "use_sketch", "use_theorem", "solve__system_prompt"}
    assert knob_names == expected
    # The auto-knob is frozen
    assert state["knobs"]["solve__system_prompt"]["spec"]["frozen"] is True


def test_executor_resolves_field_bindings(tmp_path):
    """Executor resolves field_bindings for system_prompt_template. Python runtime."""
    from refactory_dsh.nn.mode import Mode

    captured = {}

    class ScriptedClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            captured["system"] = system
            return "reply"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    class BoundMode(Mode):
        name = "bound_test"
        unscored_reason = "test"
        @classmethod
        def declare(cls):
            return {"my_knob": nn.Knob("Override system", "str")}
        def build(self):
            g = nn.Graph()
            g.add(nn.Node("a", "agent", prompt_template="Do it",
                          system_prompt_template="Original",
                          writes=("out.txt",),
                          knobs=("my_knob",),
                          field_bindings=(("system_prompt_template", "my_knob"),)))
            return g

    mode = BoundMode()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "out.txt").write_text("")
    from refactory_dsh.evaluators import WalkerEvaluator

    evaluator = WalkerEvaluator(model=ScriptedClient(), project=tmp_path)
    result = evaluator.evaluate(mode, mode.state_dict(), run_dir)
    assert result.exit_ok, result.error
    assert captured["system"] == "Override system"


@pytest.mark.skipif(
    not shutil.which("node")
    or not (Path(__file__).resolve().parent.parent.parent
            / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists(),
    reason="the spine drain needs Node and the installed workspace (pnpm install)",
)
def test_two_runtime_system_prompt_knob(tmp_path, monkeypatch):
    """Both runtimes resolve a system-prompt knob (non-default, with {{slot}}) identically.

    Correction 3: runs the same graph with a bound system-prompt knob on the walker
    and through the real spine drain path (data-drain-agent.mts), and asserts the same
    rendered system prompt reaches the model and the same records.
    Both runtimes.
    """
    import os
    import shutil as _shutil
    import subprocess
    import sys

    from refactory_dsh import lowering
    from refactory_dsh.nn.executor import execute
    from refactory_dsh.nn.mode import Mode

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    DRAIN_AGENT = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
    NODE_BIN = _shutil.which("node")

    # ── Build the graph ──
    # Agent node with system_prompt_template containing a {{slot}},
    # bound to a knob set to a non-default value.
    captured = {}

    class ScriptedClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            captured["system"] = system
            captured["prompt"] = prompt
            return "reply"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    # Build a graph with an agent node whose system-prompt is bound to a knob
    # containing a {{slot}}, set to a non-default value.
    # Walker graph: writes out.txt (the walker captures the scripted response there)
    walker_graph = nn.Graph()
    walker_graph.add(nn.Node("a", "agent", prompt_template="Do it",
                             system_prompt_template="Default with {{topic}}",
                             writes=("out.txt",),
                             knobs=("sp_knob", "topic"),
                             field_bindings=(("system_prompt_template", "sp_knob"),)))

    state = {"graph": walker_graph.to_state(), "knobs": {
        "sp_knob": {"value": "You are an expert on {{topic}}",
                    "spec": {"default": "Default with {{topic}}", "kind": "str",
                             "lo": None, "hi": None, "enum": [], "values": [],
                             "frozen": False, "description": "", "required": False}},
        "topic": {"value": "physics",
                  "spec": {"default": "math", "kind": "str",
                           "lo": None, "hi": None, "enum": [], "values": [],
                           "frozen": False, "description": "", "required": False}},
    }}

    # ── Walker run ──
    walker_dir = tmp_path / "walker"
    walker_dir.mkdir()
    (walker_dir / "out.txt").write_text("")

    from refactory_dsh.nn.executor import execute
    walker_result = execute(walker_graph, walker_dir, model=ScriptedClient(),
                            knobs={"sp_knob": "You are an expert on {{topic}}",
                                   "topic": "physics"})
    assert walker_result.exit_ok, walker_result.error
    walker_system = captured["system"]
    assert walker_system == "You are an expert on physics", (
        f"walker system prompt: {walker_system!r}")

    # ── Spine lowering: verify the IR carries the resolved system prompt ──
    wf = lowering.to_workflow(state)
    spine_system_prompt = wf.nodes["a"].system_prompt
    # The IR carries the knob value (before fact expansion) since {{topic}} is runtime
    assert spine_system_prompt == "You are an expert on {{topic}}", (
        f"lowered system_prompt: {spine_system_prompt!r}")

    # ── Real spine drain run via data-drain-agent.mts ──
    # For the spine, use writes=() since the drain provides scripted responses
    # via wf_done without writing files (same pattern as test_records_equivalence).
    spine_graph = nn.Graph()
    spine_graph.add(nn.Node("a", "agent", prompt_template="Do it",
                            system_prompt_template="Default with {{topic}}",
                            writes=(),
                            knobs=("sp_knob", "topic"),
                            field_bindings=(("system_prompt_template", "sp_knob"),)))
    spine_state = {"graph": spine_graph.to_state(), "knobs": state["knobs"]}

    spine_dir = tmp_path / "spine"
    spine_dir.mkdir(parents=True, exist_ok=True)
    lowering.write_declarations(spine_state, spine_dir)
    graph_path, _wf = lowering.write_mode_graph(
        spine_state, tmp_path / "graphs", name="slotted_two_runtime",
    )

    python_bin = str(REPO_ROOT / ".venv/bin")
    responses = {"a": "reply"}
    responses_file = spine_dir / "responses.json"
    responses_file.write_text(json.dumps(responses))

    record = tmp_path / "spine-record.jsonl"
    env = {**os.environ, "REFACTORY_WORKDIR": str(spine_dir), "DRAIN_RECORD": str(record)}
    env["PATH"] = f"{python_bin}:{env.get('PATH', '')}"
    spine_result = subprocess.run(
        [str(NODE_BIN), "--import", "tsx/esm", str(DRAIN_AGENT),
         str(graph_path), str(spine_dir), f"{python_bin}/python",
         "10000", str(responses_file)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
        env=env,
    )
    assert spine_result.returncode == 0, (
        f"spine drain failed:\nstdout: {spine_result.stdout[-2000:]}\n"
        f"stderr: {spine_result.stderr[-2000:]}"
    )

    # Verify: the lowered graph.json carries the resolved system_prompt from the knob
    lowered_graph = json.loads(graph_path.read_text())
    lowered_nodes = lowered_graph.get("nodes", {})
    spine_node = lowered_nodes.get("a", {})
    assert spine_node.get("system_prompt") == "You are an expert on {{topic}}", (
        f"graph.json system_prompt wrong: {spine_node.get('system_prompt')!r}")

    # Both runtimes resolve the system prompt from the same knob value.
    # Walker: resolved field_bindings → rendered {{topic}} → "You are an expert on physics"
    # Spine: resolved field_bindings at lowering → graph.json has "You are an expert on {{topic}}"
    # → spine worker renders {{topic}} from knob values in REFACTORY_KNOBS env.
    knob_env = lowering.knob_environment(state)
    knobs_json = json.loads(knob_env["REFACTORY_KNOBS"])
    assert knobs_json.get("topic") == "physics", (
        f"knob environment must carry topic: {knobs_json}")
    assert walker_system == "You are an expert on physics"

    # What the spine handed the worker for node a (the drain records it): the bound
    # knob's text, not the node's own template.
    sent = [json.loads(line) for line in record.read_text().splitlines()]
    assert [entry["node"] for entry in sent] == ["a"]
    assert sent[0]["system_prompt"] == "You are an expert on {{topic}}", sent[0]
    # Rendered as the dispatched worker renders it (worker_node.py: run._knobs from
    # REFACTORY_KNOBS, run._read_facts from the run folder), it is the walker's text.
    from refactory_dsh.nn.executor import RunContext, render
    from refactory_dsh.run import _knobs, _read_facts
    monkeypatch.setenv("REFACTORY_KNOBS", knob_env["REFACTORY_KNOBS"])
    worker_system = render(sent[0]["system_prompt"],
                           RunContext(spine_dir, knobs=_knobs(), facts=_read_facts(spine_dir)))
    assert worker_system == walker_system


def test_replay_refuses_removed_ops(tmp_path):
    """A journal with set_node_field/replace_node edits is refused on replay.
    Python runtime."""
    from refactory_dsh.loop import Loop, LoopConfig, LoopError, JOURNAL

    seed_state = Fixture().state_dict()
    from refactory_dsh.optim.base import Candidate
    parent = Candidate.from_state(seed_state)

    loop = Loop("circle_packing/adaptive", LoopConfig(
        runs_root=tmp_path, resume=True, optimizer="random", batch_size=1,
    ), project=Path("."))
    loop.root.mkdir(parents=True, exist_ok=True)
    (loop.root / JOURNAL).write_text("\n".join(json.dumps(r) for r in [
        {"event": "original_seed", "state": seed_state},
        {"event": "candidate", "state": parent.state,
         "edits": [{"op": "set_node_field", "id": "start", "field": "notes", "value": "x"}]},
    ]) + "\n")
    with pytest.raises(LoopError, match="set_node_field.*removed in F14"):
        loop._replay_if_asked()


def test_replay_refuses_text_patch_journal(tmp_path):
    """A journal recorded with the old text_patch op cannot be resumed; the refusal
    names set_knob's patch form. A set_knob patch in the journal resumes. Python runtime."""
    from refactory_dsh.loop import Loop, LoopConfig, LoopError, JOURNAL
    from refactory_dsh.optim.base import Candidate

    seed_state = Fixture().state_dict()
    parent = Candidate.from_state(seed_state)

    def _loop(root, op_edit):
        loop = Loop("circle_packing/adaptive", LoopConfig(
            runs_root=root, resume=True, optimizer="random", batch_size=1,
        ), project=Path("."))
        loop.root.mkdir(parents=True, exist_ok=True)
        (loop.root / JOURNAL).write_text("\n".join(json.dumps(r) for r in [
            {"event": "original_seed", "state": seed_state},
            {"event": "candidate", "state": parent.state, "edits": [op_edit]},
        ]) + "\n")
        return loop

    old = _loop(tmp_path / "old", {"op": "text_patch", "name": "skill",
                                   "edits": [{"kind": "append", "content": "x"}]})
    with pytest.raises(LoopError, match="cannot resume: journal contains edit op "
                                        "'text_patch'.*folded into set_knob"):
        old._replay_if_asked()
    new = _loop(tmp_path / "new", {"op": "set_knob", "name": "skill",
                                   "patch": [{"kind": "append", "content": "x"}]})
    assert new._replay_if_asked() is True


def test_presets_are_closed():
    # broad is the whole catalogue; the patch form is part of set_knob, which
    # random search samples only with a value (test_set_knob_patch.py).
    all_names = nn.edits.names()
    broad = sorted(nn.edits.PRESETS["broad"])
    assert "text_patch" not in all_names, "text_patch was folded into set_knob"
    assert sorted(all_names) == broad
    for name in nn.edits.PRESETS["knob_only"]:
        assert nn.edits._specs()[name].kind == "knob"  # noqa: SLF001
    with pytest.raises(nn.edits.EditError):
        nn.edits.enabled("nonsense")


def test_catalogue_excludes_removed_free_form_ops():
    """F14 guard: replace_node and set_node_field must never reappear in the
    edit catalogue.  Without this test, re-adding their ``_primitive()``
    registrations would silently pass ``test_presets_are_closed`` and
    ``test_random_edit_streams_are_never_repaired``.
    """
    all_names = set(nn.edits.names())
    assert "replace_node" not in all_names, (
        "replace_node must not be in the edit catalogue (removed in F14)"
    )
    assert "set_node_field" not in all_names, (
        "set_node_field must not be in the edit catalogue (removed in F14)"
    )
    # Also verify they are absent from every preset
    for preset_name, ops in nn.edits.PRESETS.items():
        assert "replace_node" not in ops, f"replace_node in preset {preset_name!r}"
        assert "set_node_field" not in ops, f"set_node_field in preset {preset_name!r}"


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
