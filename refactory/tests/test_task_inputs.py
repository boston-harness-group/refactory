"""What ``"sees": ["task"]`` gives: the task's own input, once, as the model steps receive
it -- never a knob's text or a step's output, never an item field no step receives.

An op's output carries the knobs it declares in ``knob_outputs``; an op that reads a knob
and declares nothing carries it in every output. Both runtimes accept the declaration
and refuse a bad one (the walker's ``Node``, the spine's ``validateGraph``)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from refactory_dsh import lowering, nn
from refactory_dsh.nn import core_ops, operators
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import pipeline_view as pv
from refactory_dsh.nn.executor import render
from refactory_dsh.nn.graph import Graph, GraphError
from refactory_dsh.registry import resolve

QUESTION = "Does the series sum_n (-1)^n / sqrt(n) converge?"
#: An item field no seed step receives; the task never gives it.
THEOREM_TYPE = "THEOREM-TYPE-MARK"
ITEM = {"id": "202501:1", "question": QUESTION, "theorem_type": [THEOREM_TYPE],
        "choices": [{"label": "A", "text": "Diverges"},
                    {"label": "B", "text": "Converges conditionally"}],
        "correct_choice": {"label": "B", "text": "Converges conditionally"}}


def _state(name: str) -> dict:
    _package, mode = resolve(Path("."), name)
    return mode().state_dict()


def _values(state: dict) -> dict:
    return {name: entry.get("value") for name, entry in state["knobs"].items()}


def _facts(state: dict, op_id: str) -> dict:
    """The facts op ``op_id`` makes for ``ITEM`` with the state's knob values."""
    node = Graph.from_state(state["graph"]).nodes[op_id]
    module, fn = node.call.split(":")
    values = _values(state)
    ctx = types.SimpleNamespace(
        knobs=values, facts={},
        read_json=lambda _name, _default=None: dict(ITEM),
        knob=lambda name, default="": values.get(name, default))
    return getattr(__import__(module, fromlist=[fn]), fn)(ctx)


def _prompt(state: dict, node_id: str, facts: dict) -> str:
    """What node ``node_id`` is sent: its system prompt and prompt, filled as the walker
    fills them, and the files it reads (none for a step that sees only the task)."""
    node = Graph.from_state(state["graph"]).nodes[node_id]
    values = _values(state)
    ctx = types.SimpleNamespace(knobs=values, facts=facts)
    assert node.reads == (), node.reads
    return "\n".join(render(str(node.bound(field, values)), ctx)
                     for field in ("system_prompt_template", "prompt_template"))


def _assert_task_only(state: dict, node_id: str, facts: dict) -> None:
    sent = _prompt(state, node_id, facts)
    assert sent.count(QUESTION) == 1, sent
    assert THEOREM_TYPE not in sent and "theorem_type" not in sent, sent
    own = set(Graph.from_state(state["graph"]).nodes[node_id].knobs)
    for name, value in _values(state).items():
        text = str(value or "").strip()
        if name not in own and isinstance(value, str) and len(text) > 20:
            assert text[:60] not in sent, (name, sent)


def test_an_s4_branch_that_sees_task_gets_the_question_once_and_nothing_else():
    s4 = _state("livemath/skill_graph")
    graph = Graph.from_state(s4["graph"])
    assert core_ops._task_inputs(graph, s4) == {("fact", "user_prompt")}
    branched = nn_edits.apply({"op": "branch", "fork": "step_fork", "kind": "agent",
                               "sees": ["task"], "id": "fresh"}, s4)
    _assert_task_only(branched, "fresh", _facts(s4, "prepare"))
    assert pv.view_lines(branched)  # drawn
    line = next(t for t in pv.render(branched).splitlines() if "fresh agent" in t)
    assert "sees user_prompt (via prepare: current_item.json@data)" in line, line
    assert "compare_task" not in line and "current_item.json@data," not in line, line


def test_an_s1_insert_that_sees_task_gets_the_framed_question_once_and_no_skill():
    s1 = _state("livemath/solver")
    graph = Graph.from_state(s1["graph"])
    # solver_prompt frames the bare question alone (ops.format_prompt): no skill in it.
    assert core_ops._task_inputs(graph, s1) == {("fact", "solver_prompt")}
    s1["knobs"]["skill"]["value"] = "Always check the alternating series test first."
    inserted = nn_edits.apply({"op": "insert", "after": "solve", "kind": "agent",
                               "sees": ["task"], "id": "check"}, s1)
    _assert_task_only(inserted, "check", _facts(s1, "format_prompt"))


def test_the_seeds_declare_which_outputs_carry_their_knobs():
    s4, s1 = (Graph.from_state(_state(n)["graph"])
              for n in ("livemath/skill_graph", "livemath/solver"))
    assert s4.nodes["prepare"].knob_outputs == {"user_prompt": (),
                                                "compare_prompt": ("compare_task",)}
    assert s1.nodes["format_prompt"].knob_outputs == {
        "user_prompt": (), "solver_prompt": (), "skill_section": ("skill",)}
    # The view says which knob reaches each step; no `?` once it is declared.
    s4_text = pv.render(_state("livemath/skill_graph"))
    assert 'compare_task "' in s4_text and "compare_task?" not in s4_text


def _undeclared(state: dict, op_id: str) -> dict:
    graph = Graph.from_state(state["graph"])
    node = graph.nodes[op_id].to_state()
    node.pop("knob_outputs")
    graph.nodes[op_id] = nn.Node.from_state(node)
    return {**state, "graph": graph.to_state()}


@pytest.mark.parametrize("name,op_id", [("livemath/skill_graph", "prepare"),
                                        ("livemath/solver", "format_prompt")])
def test_an_op_that_reads_a_knob_and_declares_nothing_carries_it_in_every_output(name, op_id):
    state = _undeclared(_state(name), op_id)
    graph = Graph.from_state(state["graph"])
    node = graph.nodes[op_id]
    assert set(operators.carried_knobs(node).values()) == {node.knobs}
    made = {("fact", n) for n in node.produces}
    assert not core_ops._task_inputs(graph, state) & made
    assert "?" in pv.render(state)


def test_task_gives_one_copy_of_a_text_and_no_data_record():
    graph = Graph()
    graph.add(nn.Node("data", "data", subgraph_entry="intake", subgraph_exit="solve",
                      inline_items=({"id": "1", "question": "q"},),
                      writes=("current_item.json",)))
    graph.add(nn.Node("intake", "op", call="json:dumps", reads=("current_item.json",),
                      writes=("brief.txt",), produces=("brief",)))
    graph.add(nn.Node("solve", "agent", prompt_template="{{brief}}", writes=("a.txt",)))
    graph.connect("intake", "solve")
    graph.start = "data"
    # brief.txt and brief are the same text; the step receives brief, so only it.
    assert operators.task_inputs(graph) == {("fact", "brief")}
    # A step that receives neither: both, still never the data node's record.
    graph.nodes["solve"] = nn.Node("solve", "agent", prompt_template="Go.", writes=("a.txt",))
    assert operators.task_inputs(graph) == {("fact", "brief"), ("file", "brief.txt")}


def test_a_pass_through_copy_of_the_item_is_never_the_task():
    """A collect step that hands a step the item record is that record again: the step
    receives it, but it is not the task's input; the op-built brief is."""
    graph = Graph()
    graph.add(nn.Node("data", "data", subgraph_entry="intake", subgraph_exit="solve",
                      inline_items=({"id": "1", "question": "q"},),
                      writes=("current_item.json",)))
    graph.add(nn.Node("intake", "op", call="json:dumps", reads=("current_item.json",),
                      produces=("brief",)))
    graph.add(nn.Node.from_state(operators.collect_node("raw", ["current_item.json"], "raw")))
    graph.add(nn.Node("solve", "agent", prompt_template="{{brief}} {{raw}}", writes=("a.txt",)))
    graph.connect("intake", "raw")
    graph.connect("raw", "solve")
    graph.start = "data"
    assert operators.task_inputs(graph) == {("fact", "brief")}


def test_the_operator_line_the_table_and_the_view_say_task_in_one_set_of_words():
    """The operator line says what "task" is in the words the step-carries table uses
    (`test_operator_fit`); the view's ``@maker`` for a name no step makes is ``@run``, so
    it is never mistaken for what "task" gives."""
    from refactory_dsh.optim import base

    assert operators.TASK_MEANING in core_ops.OPERATORS["rewire"]["line"]
    assert "`@run`: no step makes it" in base._view_lead()
    assert "@task" not in base._view_lead()


# ── the declaration, refused when wrong, on both runtimes ───────────────────


def _op(**fields) -> nn.Node:
    return nn.Node("build", "op", call="json:dumps", knobs=("a", "b"), writes=("x.txt",),
                   produces=("x", "y"), **fields)


@pytest.mark.parametrize("declared,why", [
    ({"x.txt": [], "x": ["a"], "y": [], "z": []}, "neither writes nor produces"),
    ({"x.txt": [], "x": ["a"]}, r"leaves out \['y'\]"),
    ({"x.txt": [], "x": ["c"], "y": []}, "does not read"),
    ({"x.txt": [], "x": ["a", "a"], "y": []}, "names a knob twice"),
    ({"x.txt": [], "x": "a", "y": []}, "is a list of knob names"),
    (["x"], "is a mapping"),
])
def test_the_walker_refuses_a_bad_declaration(declared, why):
    with pytest.raises(GraphError, match=why):
        _op(knob_outputs=declared)


def test_only_an_op_declares_it_and_it_round_trips():
    with pytest.raises(GraphError, match="only an op does"):
        nn.Node("s", "agent", prompt_template="p", writes=("x.txt",),
                knob_outputs={"x.txt": []})
    node = _op(knob_outputs={"x.txt": [], "x": ["a"], "y": ["a", "b"]})
    assert node.knob_outputs == {"x.txt": (), "x": ("a",), "y": ("a", "b")}
    assert nn.Node.from_state(json.loads(json.dumps(node.to_state()))) == node
    assert "knob_outputs" not in _op().to_state()


@pytest.mark.parametrize("name,op_id", [("livemath/skill_graph", "prepare"),
                                        ("livemath/solver", "format_prompt")])
def test_both_lowerings_carry_the_declaration_as_the_graph_states_it(name, op_id):
    state = _state(name)
    node = Graph.from_state(state["graph"]).nodes[op_id]
    declared = {out: list(knobs) for out, knobs in node.knob_outputs.items()}
    ir = lowering.to_workflow(state).to_dict()["nodes"][op_id]
    assert ir["knob_outputs"] == declared
    assert ir["knobs"] == list(node.knobs) and ir["produces"] == list(node.produces)
    assert lowering.node_declarations(state)[op_id]["knob_outputs"] == declared
    # An op that declares none lowers as before.
    assert not {"knob_outputs", "knobs", "produces"} & set(
        lowering.to_workflow(state).to_dict()["nodes"]["finish"])


@pytest.mark.parametrize("name,op_id,edit", [
    ("livemath/skill_graph", "prepare",
     {"op": "branch", "fork": "step_fork", "kind": "agent", "sees": ["task"]}),
    ("livemath/solver", "format_prompt",
     {"op": "insert", "after": "solve", "kind": "agent", "sees": ["task"]}),
])
def test_task_that_gives_nothing_is_refused_naming_the_op_to_declare(name, op_id, edit):
    """With only an undeclared multi-output op to build it from, "task" is empty: the edit
    is refused with the op to declare knob_outputs on, and no step that sees nothing is
    built."""
    state = _undeclared(_state(name), op_id)
    with pytest.raises(nn_edits.EditError) as refused:
        nn_edits.apply(edit, state)
    reason = str(refused.value)
    assert '"task" gives nothing before this step' in reason, reason
    # Only the op whose declaration would help is named (not notes_prompts, whose
    # outputs carry compare's notes whatever it declares).
    assert f"Op {op_id} reads a knob and does not declare" in reason, reason
    assert f"declare knob_outputs ({{output: [knobs]}}) on {op_id}." in reason, reason
    # Declared, the same edit applies.
    assert nn_edits.apply(edit, _state(name))


def test_task_that_gives_nothing_without_an_undeclared_op_says_to_name_inputs():
    graph = Graph()
    graph.add(nn.Node("solve", "agent", prompt_template="Go.", writes=("a.txt",),
                      produces=("a",), tool_policy="none"))
    graph.add(nn.Node("check", "agent", prompt_template="{{a}}", writes=("b.txt",),
                      tool_policy="none"))
    graph.connect("solve", "check")
    graph.start = "solve"
    state = {"graph": graph.to_state(), "knobs": {}}
    reason = nn_edits.precheck({"op": "rewire", "step": "check", "sees": ["task"]}, state)
    assert reason and "gives nothing" in reason and "knob_outputs" not in reason, reason
