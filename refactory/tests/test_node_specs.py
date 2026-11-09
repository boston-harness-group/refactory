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


def _copyable(state: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    """A node state as the prompt shows it: bound fields at their knob's value, and
    no bindings (nor the bound knobs) to copy."""
    bound = {field: knob for field, knob in state.get("field_bindings") or ()}
    shown = {k: v for k, v in state.items() if k != "field_bindings"}
    shown["knobs"] = [k for k in state["knobs"] if k not in bound.values()]
    for field, knob in bound.items():
        shown[field] = knobs[knob]["value"]
    return shown


def test_the_loop_shows_every_node_of_the_graph_in_full(tmp_path):
    """Through the loop: each skill_graph node's state, reads, writes, produces,
    knobs and prompts included, as add_node takes it."""
    text = _prompt(tmp_path)
    state = SkillGraph().state_dict()
    assert "### Each node in full" in text
    for node_id, node in state["graph"]["nodes"].items():
        assert _rendered_node(text, node_id) == _copyable(node, state["knobs"]), node_id


def test_a_bound_field_shows_its_knobs_current_value():
    """After set_knob on compare's system-prompt knob, the compare node shows the
    new prompt, not the stored seed text, and says to edit the knob."""
    state = nn_edits.apply({"op": "set_knob", "name": "compare__system_prompt",
                            "value": "NEW COMPARE PROMPT"}, SkillGraph().state_dict())
    described = SkillGraph(state=state).describe()
    text = ProposalContext(
        knobs=[{"name": name, **spec} for name, spec in described["knobs"].items()],
        node_states=list(state["graph"]["nodes"].values())).render()
    assert _rendered_node(text, "compare")["system_prompt_template"] == "NEW COMPARE PROMPT"
    assert ("`system_prompt_template` is bound to the knob `compare__system_prompt` and shows "
            "its current value: edit the knob, not this field.") in text
    assert "field_bindings" not in _rendered_node(text, "compare")


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


class _SystemLog:
    """A scripted model for skill_graph that logs each step's system prompt."""

    def __init__(self, log: list[tuple[str, str]] | None = None, node: str = "") -> None:
        self.log = [] if log is None else log
        self.node = node

    def for_node(self, node_id: str, role: str = "") -> "_SystemLog":
        return _SystemLog(self.log, node_id)

    def with_tool_policy(self, policy: str) -> "_SystemLog":
        return self

    def with_wall(self, wall: Any) -> "_SystemLog":
        return self

    def with_budget(self, budget: Any) -> "_SystemLog":
        return self

    def context(self) -> dict:
        return {}

    def route(self) -> dict:
        return {"worker_backend": "dsh"}

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        self.log.append((self.node, system))
        return "<answer>B</answer>" if self.node == "decide" else f"{self.node} notes"


def test_an_inserted_copy_runs_its_own_prompt_on_its_own_knob(tmp_path, monkeypatch):
    """End to end on the real skill_graph: a copy of compare inserted with its own
    system prompt gets its own verify__system_prompt knob holding that prompt, and
    the run sends that prompt; a copy that keeps compare's binding is refused."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_livemath_blocks import _patch_task

    copy = _rendered_node(_prompt(tmp_path), "compare")
    copy.update(id="verify", writes=["verify_notes.txt"],
                system_prompt_template="VERIFY SYSTEM PROMPT")
    state = nn_edits.apply({"op": "insert_after", "target": "compare", "node": copy},
                           SkillGraph().state_dict())
    assert state["knobs"]["verify__system_prompt"]["value"] == "VERIFY SYSTEM PROMPT"
    # Its task text (prompt_template) is a knob of its own too, for a later set_knob.
    assert state["knobs"]["verify__task"]["value"] == copy["prompt_template"]
    assert state["graph"]["nodes"]["verify"]["field_bindings"] == [
        ["system_prompt_template", "verify__system_prompt"], ["prompt_template", "verify__task"]]
    assert validate_state(state, invariants=SkillGraph().invariants(), resolve_calls=True) == []
    _patch_task(monkeypatch)
    model = _SystemLog()
    (tmp_path / "pass").mkdir()
    result = SkillGraph(state=state).run(tmp_path / "pass", model=model)
    assert result.exit_ok, result.error
    assert {system for node, system in model.log if node == "verify"} == {"VERIFY SYSTEM PROMPT"}

    shared = dict(copy, id="verify2", knobs=[*copy["knobs"], "compare__system_prompt"],
                  field_bindings=[["system_prompt_template", "compare__system_prompt"]])
    with pytest.raises(nn_edits.EditError, match="binds knob 'compare__system_prompt', which "
                       "node 'compare' owns") as refused:
        nn_edits.apply({"op": "insert_after", "target": "compare", "node": shared},
                       SkillGraph().state_dict())
    assert refused.value.code == "E-binding"


def test_a_duplicate_gets_its_own_knob_at_the_originals_value():
    state = nn_edits.apply({"op": "set_knob", "name": "compare__system_prompt",
                            "value": "CURRENT COMPARE PROMPT"}, SkillGraph().state_dict())
    out = nn_edits.apply({"op": "duplicate", "target": "compare", "new_id": "compare2"}, state)
    assert out["knobs"]["compare2__system_prompt"]["value"] == "CURRENT COMPARE PROMPT"
    assert out["graph"]["nodes"]["compare2"]["field_bindings"] == [
        ["system_prompt_template", "compare2__system_prompt"], ["prompt_template", "compare2__task"]]
    assert "compare__system_prompt" not in out["graph"]["nodes"]["compare2"]["knobs"]
    assert out["knobs"]["compare__system_prompt"] == state["knobs"]["compare__system_prompt"]
    # The copy writes compare's notes file again without reading it, so compare's own
    # notes are lost (V19); that is the only thing the validator says.
    issues = validate_state(out, invariants=SkillGraph().invariants(), resolve_calls=True)
    assert [(i.code, i.node) for i in issues] == [("V19", "compare")]


@pytest.mark.parametrize("text, reason", [
    ("keep this\n[... 412 characters omitted ...]\nand this", "clipped text"),
    ("Answer {{question", "never closed"),
])
def test_a_node_carrying_a_clipped_or_broken_text_is_refused(tmp_path, text, reason):
    copy = _rendered_node(_prompt(tmp_path), "compare")
    copy.update(id="verify", writes=["verify_notes.txt"], prompt_template=text)
    with pytest.raises(nn_edits.EditError, match=reason) as refused:
        nn_edits.apply({"op": "insert_after", "target": "compare", "node": copy},
                       SkillGraph().state_dict())
    assert refused.value.code == "E-node-text"


def test_long_texts_are_clipped_with_the_bound_knob_named_and_the_cap_holds():
    mode = SkillGraph()
    states = list(mode.state_dict()["graph"]["nodes"].values())
    clipped = ProposalContext(node_states=states, node_spec_text_chars=300).render()
    assert "`system_prompt_template` is clipped here (" in clipped and "do not copy it" in clipped
    assert "characters omitted" in clipped
    assert "must not be copied" in clipped

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
    assert "(rule V8), so a new op or fn-gate node reuses a callable an existing node" in text
    assert "or is one of the generic ones in `refactory_dsh.snippet`" in text
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


def test_a_kind_the_schema_cannot_probe_does_not_break_the_request(monkeypatch):
    """A refusal that names no field: that kind says to see the validator, and the
    rest of the schema still renders."""

    @dataclasses.dataclass
    class Odd(Node):
        def __post_init__(self) -> None:
            super().__post_init__()
            if self.kind == "join":
                raise GraphError("an odd rule")

    monkeypatch.setattr(node_schema, "Node", Odd)
    node_schema.node_schema.cache_clear()
    node_schema.field_meanings.cache_clear()
    try:
        text = "\n".join(node_schema.render_schema())
        assert "`join`: required fields: see the validator" in text
        assert "`op`: required id, kind, call" in text
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


def test_a_prompt_with_json_braces_is_accepted(tmp_path):
    """Nested JSON in a prompt (`}}` with no `{{`) is not a broken template slot."""
    copy = _rendered_node(_prompt(tmp_path), "compare")
    copy.update(id="verify", writes=["verify_notes.txt"],
                prompt_template='{{compare_prompt}}\nReply as {"a": {"b": 1}}')
    state = nn_edits.apply({"op": "insert_after", "target": "compare", "node": copy},
                           SkillGraph().state_dict())
    assert state["graph"]["nodes"]["verify"]["prompt_template"].endswith('{"b": 1}}')


def _with_verify(tmp_path: Path) -> dict[str, Any]:
    copy = _rendered_node(_prompt(tmp_path), "compare")
    copy.update(id="verify", writes=["verify_notes.txt"],
                system_prompt_template="VERIFY SYSTEM PROMPT")
    return nn_edits.apply({"op": "insert_after", "target": "compare", "node": copy},
                          SkillGraph().state_dict())


def test_removing_a_node_takes_its_own_knob_and_leaves_declared_ones(tmp_path):
    """An added node's own knob goes with it, so the id can be added again; a
    declared knob only the removed node read stays, and setting it is refused."""
    state = _with_verify(tmp_path)
    removed = nn_edits.apply({"op": "bypass", "target": "verify"}, state)
    assert "verify__system_prompt" not in removed["knobs"]
    assert validate_state(removed, invariants=SkillGraph().invariants(), resolve_calls=True) == []
    again = nn_edits.apply({"op": "insert_after", "target": "compare",
                            "node": state["graph"]["nodes"]["verify"] | {"field_bindings": [],
                            "knobs": ["max_completion_tokens"]}}, removed)
    assert again["knobs"]["verify__system_prompt"]["value"] == "VERIFY SYSTEM PROMPT"

    orphaned = nn_edits.apply({"op": "remove_node", "id": "decide_prompt"},
                              SkillGraph().state_dict())
    assert "decide_task" in orphaned["knobs"], "a declared knob is never dropped by an edit"
    with pytest.raises(nn_edits.EditError, match="read by no node") as refused:
        nn_edits.apply({"op": "set_knob", "name": "decide_task", "value": "x"}, orphaned)
    assert refused.value.code == "E-knob"


def test_a_new_node_may_not_bind_a_knob_another_node_reads(tmp_path):
    """compare_task is read (not bound) by prepare-side nodes; binding it is refused."""
    state = SkillGraph().state_dict()
    readers = SkillGraph().graph().knob_consumers()
    knob = "compare_task"
    assert knob in readers, "setup: another node reads it"
    copy = _rendered_node(_prompt(tmp_path), "compare")
    copy.update(id="verify", writes=["verify_notes.txt"], knobs=[*copy["knobs"], knob],
                field_bindings=[["prompt_template", knob]])
    with pytest.raises(nn_edits.EditError, match=f"binds knob '{knob}', which node .* reads"):
        nn_edits.apply({"op": "insert_after", "target": "compare", "node": copy}, state)


def test_lowering_binds_an_added_agents_own_system_prompt_knob(tmp_path):
    """The spine's graph for a state with an edit-added agent node: lowering succeeds
    and the node's system prompt is its own knob's value."""
    from refactory_dsh import lowering

    state = _with_verify(tmp_path)
    state = nn_edits.apply({"op": "set_knob", "name": "verify__system_prompt",
                            "value": "TUNED VERIFY PROMPT"}, state)
    path, _workflow = lowering.write_mode_graph(state, tmp_path / "lowered", name="verify-graph")
    text = path.read_text()
    assert "TUNED VERIFY PROMPT" in text
    lowered = json.loads(text)
    found = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("id") == "verify":
                found.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(lowered)
    nodes = [node for node in found if "kind" in node or "reads" in node]
    assert nodes and all("TUNED VERIFY PROMPT" in json.dumps(n) and "VERIFY SYSTEM PROMPT"
                         not in json.dumps(n) for n in nodes), nodes


def test_the_spine_runs_an_added_agent_node(tmp_path):
    """The real spine drain runs the edited skill_graph: the added node is called on
    every item, and the pass completes."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_livemath_blocks import NODE, REPO_ROOT, _spine

    if NODE is None or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists():
        pytest.skip("the spine drain needs Node and the installed workspace")
    state = _with_verify(tmp_path)
    graph = SkillGraph(state=state).graph()
    answers = {"compare": "compare notes", "verify": "verify notes", "precision": "p notes",
               "hypotheses": "h notes", "decide": "<answer>B</answer>"}
    rc = _spine(graph, tmp_path / "spine", tmp_path, agent_responses=answers, mode_state=state)
    assert rc == 0
    records = json.loads((tmp_path / "spine" / "item_records.json").read_text())
    assert records and all("verify" in [e.get("node") for e in r.get("exchanges") or ()]
                           for r in records)


def test_a_new_agents_task_text_is_a_knob_the_next_round_can_move(tmp_path):
    """A new agent's prompt_template is bound to its own `<id>__task` knob holding the
    text the edit gave it, as its system prompt is to `<id>__system_prompt`: the next
    round's request lists the knob, a set_knob moves it, and the run sends the new text.
    Removing the node removes both knobs."""
    from refactory_dsh.nn.graph import Graph

    copy = _rendered_node(_prompt(tmp_path), "compare")
    copy.update(id="verify", writes=["verify_notes.txt"], prompt_template="CHECK {{compare_prompt}}")
    state = nn_edits.apply({"op": "insert_after", "target": "compare", "node": copy},
                           SkillGraph().state_dict())
    assert state["knobs"]["verify__task"]["value"] == "CHECK {{compare_prompt}}"
    knobs = SkillGraph(state=state).describe()["knobs"]
    assert "verify__task" in knobs and not knobs["verify__task"]["frozen"]
    listed = ProposalContext(knobs=[{"name": n, **k} for n, k in knobs.items()]).render()
    parameters = listed[listed.index("## Parameters you may move"):listed.index("## The pipeline")]
    assert "| verify__task |" in parameters
    moved = nn_edits.apply({"op": "set_knob", "name": "verify__task",
                            "value": "RECHECK {{compare_prompt}}"}, state)
    assert validate_state(moved, invariants=SkillGraph().invariants(), resolve_calls=True) == []
    graph = Graph.from_state(moved["graph"])
    assert graph.nodes["verify"].bound("prompt_template", {
        n: e["value"] for n, e in moved["knobs"].items()}) == "RECHECK {{compare_prompt}}"
    gone = nn_edits.apply({"op": "remove_node", "id": "verify"}, moved)
    assert not [k for k in gone["knobs"] if k.startswith("verify__")]
