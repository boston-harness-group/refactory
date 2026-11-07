"""The edited graph's nodes in full, and the node schemas, in the optimizer's prompt.

A structure edit needs a whole node object. ``ProposalContext.render()`` shows each
node of the edited graph in node-state form (long texts clipped, bound knobs named),
a schema per node kind generated from ``Node`` and its validator, and which edit
fields take a node. Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import node_schema, validate_state
from refactory_dsh.nn.graph import GATE_KINDS, NODE_KINDS, GraphError, Node
from refactory_dsh.optim import ProposalContext
from refactory_packages.livemath.skill_graph import SkillGraph

MODE = "livemath/skill_graph"


def _prompt(tmp_path: Path) -> str:
    loop = Loop(MODE, LoopConfig(runs_root=tmp_path / "run", optimizer="random"), project=Path("."))
    return loop._context().render()


def _rendered_node(text: str, node_id: str) -> dict[str, Any]:
    match = re.search(rf"#### `{re.escape(node_id)}` \(\w+\)\n```json\n(.*?)\n```", text, re.S)
    assert match, node_id
    return json.loads(match.group(1))


def test_the_loop_shows_every_node_of_the_graph_in_full(tmp_path):
    """Through the loop: each skill_graph node's state, reads, writes, produces,
    knobs, prompts and bindings included, as add_node takes it."""
    text = _prompt(tmp_path)
    states = SkillGraph().state_dict()["graph"]["nodes"]
    assert "### Each node in full" in text
    for node_id, state in states.items():
        assert _rendered_node(text, node_id) == state, node_id


def test_a_copied_node_is_accepted_by_insert_after_and_validates(tmp_path):
    """A node copied from the prompt with a new id and output is spliced in by
    insert_after on the real graph, and the result passes validation."""
    text = _prompt(tmp_path)
    copy = _rendered_node(text, "compare")
    copy.update(id="compare_check", writes=["compare_check_notes.txt"])
    mode = SkillGraph()
    state = nn_edits.apply({"op": "insert_after", "target": "compare", "node": copy},
                           mode.state_dict())
    assert "compare_check" in state["graph"]["nodes"]
    assert validate_state(state, invariants=mode.invariants(), resolve_calls=True) == []


def test_long_texts_are_clipped_with_the_bound_knob_named_and_the_cap_holds():
    mode = SkillGraph()
    states = list(mode.state_dict()["graph"]["nodes"].values())
    clipped = ProposalContext(node_states=states, node_spec_text_chars=300).render()
    assert ("`system_prompt_template` is clipped here (its full value is the knob "
            "`compare__system_prompt`, which it is bound to).") in clipped
    assert "characters omitted" in clipped

    def section(text: str) -> str:
        start = text.index("### Each node in full")
        return text[start:text.index("\n\n## What the author says", start)]

    full = len(section(ProposalContext(node_states=states).render()))
    for cap in range(full - 4000, full + 1, 251):
        assert len(section(ProposalContext(node_states=states, node_spec_section_chars=cap)
                           .render())) <= cap, cap
    tight = ProposalContext(node_states=states, node_spec_section_chars=12000).render()
    assert "not shown in full to stay within 12000 characters:" in tight
    assert "### Node schemas" in tight, "the schemas stay when nodes are left out"


def test_the_schema_covers_every_kind_and_its_examples_are_valid_nodes():
    schema = node_schema.node_schema()
    expected = [k for k in NODE_KINDS if k != "gate"] + [f"gate ({g})" for g in GATE_KINDS]
    assert sorted(schema) == sorted(expected)
    for name, entry in schema.items():
        Node.from_state(entry["example"])  # a valid node, as add_node would build it
        assert {"id", "kind"} <= set(entry["required"])
    assert "call" in schema["op"]["required"] and "prompt_template" in schema["agent"]["required"]
    assert {"subgraph_entry", "subgraph_exit"} <= set(schema["data"]["required"])
    assert "targets" in schema["fork"]["required"] and "targets" in schema["agent"]["refused"]
    assert "evaluator" in schema["gate (fn)"]["required"]
    assert "evaluator" in schema["gate (user)"]["refused"]
    assert "gate_kind" in schema["gate (agent)"]["required"]
    assert "retry" in schema["agent"]["optional"] and "retry" in schema["op"]["refused"]
    meanings = node_schema.field_meanings()
    assert all(meanings.get(f.name) for f in dataclasses.fields(Node)), "every field has a meaning"
    text = "\n".join(node_schema.render_schema())
    assert "(rule V8), so a new op or fn-gate node can only reuse a callable" in text
    assert "`gate (user)`: required id, kind, gate_kind" in text


def test_the_schema_is_generated_from_the_node_definition(monkeypatch):
    """A field added to the node, with a rule that requires it for ops, shows up as
    required for ops, with its comment as its meaning."""

    @dataclasses.dataclass
    class Extended(Node):
        #: A field added only in this test.
        flavour: str = ""

        def __post_init__(self) -> None:
            super().__post_init__()
            if self.kind == "op" and not self.flavour:
                raise GraphError(f"op node {self.id!r} declares no flavour")

    monkeypatch.setattr(node_schema, "Node", Extended)
    node_schema.node_schema.cache_clear()
    node_schema.field_meanings.cache_clear()
    try:
        assert "flavour" in node_schema.node_schema()["op"]["required"]
        assert "flavour" in node_schema.node_schema()["agent"]["optional"]
        assert node_schema.field_meanings()["flavour"] == "A field added only in this test."
    finally:
        node_schema.node_schema.cache_clear()
        node_schema.field_meanings.cache_clear()


def test_catalogue_rows_say_which_fields_take_a_node():
    catalogue = [op for op in nn_edits.catalogue("broad")
                 if op["op"] in ("insert_after", "split", "loop", "add_node", "bypass")]
    text = ProposalContext(catalogue=catalogue).render()
    rows = {line.split("|")[1].strip(): line for line in text.splitlines()
            if line.startswith("| ") and line.split("|")[1].strip() in
            ("insert_after", "split", "loop", "add_node", "bypass")}
    assert "node: a full node object (see the node schemas above)" in rows["insert_after"]
    assert "node: a full node object" in rows["add_node"]
    assert "into: a list of two full node objects" in rows["split"]
    assert "gate: a full node object" in rows["loop"]
    assert "full node object" not in rows["bypass"]
