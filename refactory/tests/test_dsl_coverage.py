"""The DSL coverage map (`nn.coverage`): every construct of the graph language has an
entry, every operation it names exists, every gap gives a reason from the fixed list, the
covered constructs are made by the operations named (applied through the attempt path the
outer loop uses), and the table in the docs is the map's. A new node kind, gate kind,
bindable field or ``Node`` field without an entry fails here."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from refactory_dsh import snippet
from refactory_dsh.nn import coverage
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.graph import Graph
from refactory_dsh.optim import Candidate

import structure_fuzz as sf
import structure_fuzz_graphs as fg

DOC = Path(__file__).resolve().parent.parent / "docs" / "outer-loop.md"
START, END = "<!-- dsl-coverage:start -->", "<!-- dsl-coverage:end -->"


def _aspects(entry: coverage.Entry) -> list[coverage.Aspect]:
    return [entry.create, entry.change, entry.remove]


def test_every_construct_has_exactly_one_entry():
    wanted = coverage.constructs()
    found: dict[str, list[str]] = {}
    for entry in coverage.entries():
        found.setdefault(entry.group, []).append(entry.name)
    for group, names in wanted.items():
        assert sorted(found.get(group, [])) == sorted(names), (group, set(names) ^ set(found.get(group, [])))
    assert set(found) == set(wanted)


def test_every_operation_named_exists_and_every_gap_has_a_known_reason():
    ops = set(nn_edits.names())
    for entry in coverage.entries():
        assert entry.create is not None and entry.change is not None, entry
        for aspect in _aspects(entry):
            if isinstance(aspect, coverage.Gap):
                assert aspect.reason in coverage.REASONS, (entry, aspect)
                if aspect.reason in ("deferred", "awaiting a user decision"):
                    assert aspect.detail, f"{entry}: a deferred gap says what is deferred"
            elif aspect is not None:
                assert aspect and set(aspect) <= ops, (entry, set(aspect) - ops)


def test_nothing_is_left_planned_each_gap_is_covered_or_excluded_on_purpose():
    """Strict: no construct waits on a later change. What the user deferred is excluded on
    purpose, and the decision names each deferred thing; the settings bound to knobs (a
    loop's limit, a branch's optional) are covered."""
    assert "planned" not in coverage.REASONS
    deferred = {(e.group, e.name) for e in coverage.entries()
                for a in _aspects(e) if isinstance(a, coverage.Gap) and a.reason == "deferred"}
    assert deferred == {("node field", name) for name in (
        "knobs", "field_bindings", "role", "on_exhausted", "timeout_s", "retry", "workspace")} | {
        ("composition", "a step per sub-item")}, deferred
    # Deferred is deferred: none of these settings can be bound to a knob yet.
    from refactory_dsh.nn.graph import BINDABLE_FIELDS

    assert not {"timeout_s", "retry", "on_exhausted", "workspace", "role"} & BINDABLE_FIELDS
    decision = coverage.DECISIONS["deferred"]
    for named in ("timeout_s", "retry", "on_exhausted", "workspace kind", "model role",
                  "per-sub-item"):
        assert named in decision, named
    assert not {("node kind", "fork"), ("node kind", "join"), ("node field", "reads"),
                ("node field", "inputs"), ("node field", "call"), ("node field", "targets"),
                ("node field", "sources"), ("node field", "optional_targets"),
                ("node field", "max_iterations"), ("bindable field", "optional"),
                ("composition", "nested fork"),
                ("composition", "gate routing to several steps")} & deferred


def _apply(seed: str, edit: Any) -> tuple[Graph, Graph]:
    applier = sf.Applier(sf.seeds()[seed])
    parent = Candidate.from_state(applier.seed_state)
    child = applier.attempt(parent, edit, {parent.id})
    assert isinstance(child, Candidate), (seed, edit, child)
    return Graph.from_state(parent.state["graph"]), Graph.from_state(child.state["graph"])


def _gate(node_id: str) -> dict[str, Any]:
    return {"id": node_id, "kind": "gate", "evaluator": fg.PROCEED, "max_iterations": 2}


def _kinds(graph: Graph, kind: str) -> int:
    return sum(1 for n in graph.nodes.values() if n.kind == kind)


def _conditions(graph: Graph) -> list[str]:
    return sorted(str(e.condition) for e in graph.edges if e.condition)


#: One example per covered construct: the seed, the edit (with its operations from the
#: entry's ``create``), and what the child must show that the parent did not.
EXAMPLES: dict[tuple[str, str], tuple[str, Any, Callable[[Graph, Graph], bool]]] = {
    ("node kind", "op"): ("fuzz_dataflow", {"op": "insert_after", "target": "clean", "node": fg.op(
        "tidy", reads=("clean.txt",), produces=("tidied",)).to_state()},
        lambda p, c: _kinds(c, "op") == _kinds(p, "op") + 1),
    ("node kind", "agent"): ("fuzz_dataflow", {"op": "insert_after", "target": "clean", "node": fg.agent(
        "read", "Read {{cleaned}}", writes="read.txt", produces=("read",)).to_state()},
        lambda p, c: _kinds(c, "agent") == _kinds(p, "agent") + 1),
    ("node kind", "gate"): ("fuzz_dataflow", {"op": "loop", "target": "summarise",
                                              "gate": _gate("again")},
                            lambda p, c: _kinds(c, "gate") == _kinds(p, "gate") + 1),
    ("node kind", "fork"): ("fuzz_dataflow", {"op": "parallel", "target": "clean", "branches": [
        fg.op("left", writes=("left.txt",)).to_state(), fg.op("right", writes=("right.txt",)).to_state()]},
        lambda p, c: _kinds(c, "fork") == _kinds(p, "fork") + 1),
    ("node kind", "join"): ("fuzz_dataflow", {"op": "parallel", "target": "clean", "branches": [
        fg.op("left", writes=("left.txt",)).to_state(), fg.op("right", writes=("right.txt",)).to_state()]},
        lambda p, c: _kinds(c, "join") == _kinds(p, "join") + 1),
    ("gate kind", "fn"): ("fuzz_dataflow", {"op": "loop", "target": "summarise", "gate": _gate("again")},
                          lambda p, c: c.nodes["again"].gate_kind == "fn"),
    ("edge form", "plain"): ("fuzz_dataflow", {"op": "insert_after", "target": "clean", "node": fg.op(
        "tidy", reads=("clean.txt",), produces=("tidied",)).to_state()},
        lambda p, c: any(e.source == "clean" and e.target == "tidy" and e.condition is None
                         for e in c.edges)),
    ("edge form", "verdict"): ("fuzz_dataflow", {"op": "loop", "target": "summarise",
                                                 "gate": _gate("again")},
                               lambda p, c: "proceed" in _conditions(c) and "proceed" not in _conditions(p)),
    ("edge form", "reloop"): ("fuzz_dataflow", {"op": "loop", "target": "summarise", "gate": _gate("again")},
                              lambda p, c: "reloop" in _conditions(c)),
    ("composition", "nested fork"): ("fuzz_fork_join", {"op": "parallel", "target": "right_a", "branches": [
        fg.op("deep_a", writes=("deep_a.txt",)).to_state(),
        fg.op("deep_b", writes=("deep_b.txt",)).to_state()]},
        lambda p, c: _kinds(c, "fork") == 2),
    ("composition", "loop inside a fork branch"): ("fuzz_fork_join", {
        "op": "loop", "target": "right_a", "gate": _gate("right_check")},
        lambda p, c: any(e.source == "right_check" and e.target == "right_a" for e in c.edges)),
    ("composition", "loop inside a data pass"): ("fuzz_data_pass", {
        "op": "loop", "target": "solve", "gate": _gate("solve_check")},
        lambda p, c: "solve_check" in c.data_body("items")),
}


# One-line core operators (`nn.core_ops`) making the same constructs.
EXAMPLES.update({
    ("node kind", "agent"): ("fuzz_draft_review", {"op": "insert", "after": "draft",
                                                   "kind": "agent", "text": "Weak points."},
                             lambda p, c: _kinds(c, "agent") == _kinds(p, "agent") + 1),
    ("node kind", "fork"): ("fuzz_draft_review", {"op": "parallelize", "steps": ["draft"],
                                                  "copies": 1},
                            lambda p, c: _kinds(c, "fork") == _kinds(p, "fork") + 1),
    ("node kind", "join"): ("fuzz_draft_review", {"op": "parallelize", "after": "draft", "new": [
        {"kind": "agent", "text": "One."}, {"kind": "agent", "text": "Two."}]},
        lambda p, c: _kinds(c, "join") == _kinds(p, "join") + 1),
    ("gate kind", "fn"): ("fuzz_draft_review", {"op": "loop", "step": "draft", "judge": "agent"},
                          lambda p, c: c.nodes["draft_check"].gate_kind == "fn"),
    ("edge form", "verdict"): ("fuzz_draft_review", {"op": "route", "after": "review", "to": {
        "proceed": "finalize", "again": {"kind": "agent", "text": "Again."}}},
        lambda p, c: "again" in _conditions(c) and "again" not in _conditions(p)),
    ("composition", "nested fork"): ("livemath_skill_graph", {
        "op": "parallelize", "after": "hypotheses", "new": [{"kind": "agent", "text": "One."},
                                                            {"kind": "agent", "text": "Two."}]},
        lambda p, c: "hypotheses_fork" in c.nodes and _kinds(c, "fork") == 2),
    ("composition", "gate routing to several steps"): ("fuzz_draft_review", {
        "op": "route", "after": "review", "to": {"proceed": "finalize",
                                                 "rewrite": {"kind": "agent", "text": "Redo."}}},
        lambda p, c: len(c.out_edges("review_route")) == 2),
    ("node field", "optional_targets"): ("livemath_skill_graph", {
        "op": "branch", "fork": "step_fork", "optional": True},
        # The new branch's first step binds its optional.
        lambda p, c: [t for t in c.nodes["step_fork"].targets if c.nodes[t].bound_knob("optional")]
        == [c.nodes["step_fork"].targets[-1]]),
})


#: Covered changes and removals of an existing construct: the seed, the edit, which of the
#: entry's aspects names its operation, and what the child shows.
CHANGES: dict[tuple[str, str, str], tuple[str, Any, Callable[[Graph, Graph], bool]]] = {
    ("node kind", "fork", "change"): ("livemath_skill_graph", {"op": "branch", "fork": "step_fork"},
                                      lambda p, c: len(c.nodes["step_fork"].targets) == 3),
    ("node kind", "fork", "remove"): ("fuzz_fork_join", {"op": "serialize", "fork": "split"},
                                      lambda p, c: "split" not in c.nodes),
    ("node kind", "join", "remove"): ("fuzz_nested_fork", {"op": "serialize", "fork": "inner"},
                                      lambda p, c: "inner_join" not in c.nodes),
    ("node kind", "agent", "remove"): ("fuzz_nested_fork", {"op": "remove", "step": "solo"},
                                       lambda p, c: "solo" not in c.nodes),
    ("node kind", "gate", "remove"): ("fuzz_gate_loop", {"op": "remove", "step": "review",
                                                          "keep": "proceed"},
                                      lambda p, c: _kinds(c, "gate") == 0),
    ("node field", "reads", "change"): ("fuzz_draft_review", {"op": "rewire", "step": "finalize",
                                                               "sees": ["task"]},
                                        lambda p, c: c.nodes["finalize"].prompt_template
                                        != p.nodes["finalize"].prompt_template),
    # A step's callable changes when an op step takes its place (any machine), or a code
    # step (where code steps can run: `test_a_step_split_into_a_code_step_runs_that_code`).
    ("node field", "call", "change"): ("fuzz_draft_review", {"op": "split", "step": "review",
                                                              "into": [{"kind": "op", "call": fg.STEP}]},
                                       lambda p, c: "review" not in c.nodes and any(
                                           n.call == fg.STEP and n.id not in p.nodes
                                           for n in c.nodes.values())),
    ("node field", "targets", "change"): ("livemath_skill_graph", {
        "op": "insert", "before": "hypotheses"},
        lambda p, c: "hypotheses" not in c.nodes["step_fork"].targets),
    ("node field", "sources", "change"): ("fuzz_nested_fork", {"op": "remove", "step": "deep_b"},
                                          lambda p, c: "inner_join" not in c.nodes),
    ("composition", "gate routing to several steps", "remove"): (
        "fuzz_multi_terminal", {"op": "remove", "step": "route", "keep": "proceed"},
        lambda p, c: "route" not in c.nodes),
}


@pytest.mark.parametrize("key", sorted(CHANGES), ids=lambda key: ":".join(key))
def test_a_covered_change_or_removal_is_made_by_an_operation_its_entry_names(key):
    group, name, aspect = key
    seed, edit, shows = CHANGES[key]
    entry = next(e for e in coverage.entries() if (e.group, e.name) == (group, name))
    ops = getattr(entry, aspect)
    assert isinstance(ops, tuple) and edit["op"] in ops, (entry, edit["op"])
    parent, child = _apply(seed, edit)
    assert shows(parent, child), key


@pytest.mark.skipif(not snippet.available()[0], reason="this machine cannot sandbox code steps "
                    "(`snippet.available`); the op-step case above covers the change here")
def test_a_step_split_into_a_code_step_runs_that_code():
    """Where code steps can run, ``split`` into a code step changes a step's callable to the
    code op, as the coverage map's ``call`` entry also says."""
    parent, child = _apply("fuzz_draft_review", {"op": "split", "step": "review",
                                                 "into": [{"kind": "code"}]})
    assert "review" not in child.nodes
    assert any(n.call == snippet.RUN for n in child.nodes.values() if n.id not in parent.nodes)


@pytest.mark.parametrize("key", sorted(EXAMPLES), ids=lambda key: f"{key[0]}:{key[1]}")
def test_a_covered_construct_is_made_by_an_operation_its_entry_names(key):
    seed, edit, shows = EXAMPLES[key]
    entry = next(e for e in coverage.entries() if (e.group, e.name) == key)
    ops = {step.get("op") for step in (edit.get("edits") or [edit])}
    assert isinstance(entry.create, tuple) and ops <= set(entry.create), (entry, ops)
    parent, child = _apply(seed, edit)
    assert shows(parent, child), key


def test_every_structural_node_kind_and_edge_form_the_map_covers_has_an_example():
    covered = {(e.group, e.name) for e in coverage.entries()
               if e.group in ("node kind", "edge form") and isinstance(e.create, tuple)}
    assert covered - set(EXAMPLES) <= {("edge form", "halt")}, covered - set(EXAMPLES)


def test_the_docs_carry_the_maps_table():
    text = DOC.read_text()
    assert START in text and END in text, "docs/outer-loop.md has the coverage markers"
    shown = text[text.index(START) + len(START):text.index(END)].strip()
    assert shown == coverage.table_markdown(), (
        "the table in docs/outer-loop.md is not the map's; paste coverage.table_markdown()")


# ── every operation an entry names for a field can set it (no overclaim) ──────────


#: A new sealed agent step written out in full, for the operations that take a node object.
_LOOK_NODE = {"id": "look", "kind": "agent", "prompt_template": "Look.", "writes": ["look.txt"],
              "produces": ["look"], "tools": ["bash"], "tool_policy": None,
              "workspace": {"kind": "scratch"}}
_LOOK = {"op": "insert", "before": "solve", "id": "look", "kind": "agent", "text": "Look.",
         "to": "solve"}
_POLISH = {"op": "insert", "after": "solve", "id": "polish", "kind": "agent",
           "text": "Polish.", "takes_over": "solve"}
_PAIR = {"op": "parallelize", "after": "look", "new": [
    {"kind": "agent", "text": "A.", "id": "pa", "to": "solve"},
    {"kind": "agent", "text": "B.", "id": "pb", "to": "solve"}]}


def _with(spec: dict, **field) -> dict:
    return {**spec, **field}


def _field_examples(field: dict) -> dict[str, tuple[str, list[dict]]]:
    """For each operation, a proposal on the per-item pass (`item_files_ops.AgentPass`, or
    SpreadsheetBench's Solver for ``map``, which needs a per-item list) whose new step
    carries ``field`` (``{"tools": [...]}`` or ``{"route": {...}}``)."""
    node = {**_LOOK_NODE, **({"route": field["route"]} if "route" in field else {})}
    return {
        "add_node": ("agent_pass", [{"op": "add_node", "position": {"before": "solve"},
                                     "node": node}]),
        "insert_before": ("agent_pass", [{"op": "insert_before", "target": "solve", "node": node}]),
        "insert_after": ("agent_pass", [{"op": "insert_after", "target": "solve", "node": node}]),
        "wrap": ("agent_pass", [_LOOK, {"op": "wrap", "target": "look", "node": {
            **node, "id": "watch", "writes": ["watch.txt"], "produces": ["watch"]}}]),
        "parallel": ("agent_pass", [_LOOK, {"op": "parallel", "target": "look", "branches": [
            {**node, "id": "left", "writes": ["left.txt"], "produces": ["left"]},
            {**node, "id": "right", "writes": ["right.txt"], "produces": ["right"]}]}]),
        "split": ("agent_pass", [{"op": "split", "step": "solve", "into": [
            _with({"kind": "agent", "text": "Plan."}, **field),
            {"kind": "agent", "text": "Answer."}]}]),
        "loop": ("agent_pass", [_with({"op": "loop", "step": "solve", "judge": "agent"},
                                      **field)]),
        "insert": ("agent_pass", [_with(_LOOK, **field)]),
        "parallelize": ("agent_pass", [_LOOK, {**_PAIR, "new": [
            _with(_PAIR["new"][0], **field), _PAIR["new"][1]]}]),
        "branch": ("agent_pass", [_LOOK, _PAIR, _with({"op": "branch", "fork": "look_fork",
                                                       "kind": "agent", "text": "C.",
                                                       "id": "pc", "to": "solve"}, **field)]),
        "route": ("agent_pass", [_POLISH, _with({"op": "route", "after": "solve",
                                                 "judge": "agent"}, **field)]),
        "merge": ("agent_pass", [_LOOK, {"op": "insert", "after": "look", "id": "look2",
                                         "kind": "agent", "text": "Again.", "takes_over": "look"},
                                 _with({"op": "merge", "steps": ["look", "look2"]}, **field)]),
        "fallback": ("agent_pass", [_with({"op": "fallback", "step": "solve", "kind": "agent",
                                           "text": "Simpler."}, **field)]),
        "map": ("solver", [_with({"op": "map", "after": "frame", "over": "solver_prompt",
                                  "text": "Do one.", "to": "solve"}, **field)]),
    }


def _field_attempt(seed: str, edits: list[dict]):
    import item_files_ops as ifo
    from refactory_dsh.optim import Optimizer, Proposal
    from refactory_dsh.workers import StepRoutes
    from refactory_packages.spreadsheetbench import mode as sb_mode

    mode = ifo.AgentPass() if seed == "agent_pass" else sb_mode.Solver()
    optimizer = Optimizer()
    optimizer.sealed_tools = (True, "")
    optimizer.step_routes = StepRoutes(sf.ROUTE_BASE, node_routes=sf.ROUTE_MAPS)
    parent = Candidate.from_state(mode.state_dict())
    out = optimizer.attempt(Proposal(edits=edits), parent, mode.invariants(), True, (),
                            protected=mode.edit_protected(), record=False)
    return parent, out


def _created(field: str) -> list[str]:
    entry = next(e for e in coverage.entries() if (e.group, e.name) == ("node field", field))
    assert isinstance(entry.create, tuple), entry
    return list(entry.create)


@pytest.mark.parametrize("op", _created("tools"))
def test_every_operation_the_map_names_for_tools_makes_a_sealed_tool_step(op):
    """The map says each of these makes a step with tools: each one does, sealed."""
    from refactory_dsh.workspace import is_sealed

    seed, edits = _field_examples({"tools": ["bash"]})[op]
    parent, out = _field_attempt(seed, edits)
    assert isinstance(out, Candidate), (op, out)
    new = [n for k, n in out.state["graph"]["nodes"].items()
           if k not in parent.state["graph"]["nodes"] and n.get("tools")]
    assert new and all(is_sealed(n) for n in new), (op, new)


@pytest.mark.parametrize("op", [op for op in _created("sealed_script") if op != "add_node"])
def test_every_operation_the_map_names_for_sealed_script_makes_one(op):
    """The map says each of these makes a sealed script step: each one does (``insert`` with
    kind 'code' and 'sealed' true; ``loop`` and ``route`` with judge 'code' and 'sealed')."""
    _POLISH_CODE = {"op": "insert", "after": "solve", "id": "polish", "kind": "code",
                    "takes_over": "solve"}
    if op == "insert":
        edits = [{"op": "insert", "after": "solve", "kind": "code", "sealed": True,
                  "takes_over": "solve"}]
    elif op == "loop":
        edits = [{"op": "loop", "step": "solve", "judge": "code", "sealed": True}]
    elif op == "route":
        edits = [_POLISH_CODE, {"op": "route", "after": "solve", "judge": "code",
                                "sealed": True}]
    else:
        pytest.skip(f"{op} not wired for sealed_script test")
    parent, out = _field_attempt("agent_pass", edits)
    assert isinstance(out, Candidate), (op, out)
    new = [n for k, n in out.state["graph"]["nodes"].items()
           if k not in parent.state["graph"]["nodes"] and n.get("sealed_script")]
    assert new, (op, [n for k, n in out.state["graph"]["nodes"].items()
                      if k not in parent.state["graph"]["nodes"]])


@pytest.mark.parametrize("op", _created("route"))
def test_every_operation_the_map_names_for_route_gives_a_new_step_its_own_route(op):
    """The map says each of these may give a new agent step its own route: each one does
    (``route`` itself through a step in its 'to'; a judge takes its step's)."""
    route = {"model": "m-small"}
    examples = _field_examples({"route": route})
    if op == "route":
        examples["route"] = ("agent_pass", [_POLISH, {"op": "route", "after": "solve",
                                                      "judge": "agent", "to": {
            "keep": "polish", "redo": {"kind": "agent", "text": "Redo.", "route": route,
                                       "takes_over": "solve"}}}])
    seed, edits = examples[op]
    parent, out = _field_attempt(seed, edits)
    assert isinstance(out, Candidate), (op, out)
    new = [n for k, n in out.state["graph"]["nodes"].items()
           if k not in parent.state["graph"]["nodes"] and n.get("route")]
    assert new and all(n["route"].get("model") == "m-small" for n in new), (op, new)


# ── the request names the operations from the map, never by hand ──────────────────


def _named_in_tools_line(ops) -> set[str]:
    """The operations the request's tools bullet names (`optim.base.tools_line`)."""
    import re

    from refactory_dsh.optim import base

    found = re.search(r"\(made by (.*?): in the edit's own", base.tools_line(ops))
    return set(re.findall(r"`([a-z_]+)`", found.group(1))) if found else set()


_EVERY_OP = [c["op"] for c in nn_edits.catalogue("broad")]


@pytest.mark.parametrize("op", _EVERY_OP)
def test_the_tools_bullet_names_an_operation_exactly_when_it_makes_a_tool_step(op):
    """[request text] The tools bullet names every operation that can make a new agent step
    with tools (the map's, each held to making a sealed one above) and nothing else; it once
    named three when nine could."""
    assert (op in _named_in_tools_line(_EVERY_OP)) == (op in _created("tools")), op


@pytest.mark.parametrize("op", _EVERY_OP)
def test_an_operation_whose_fields_take_tools_is_one_the_map_says_makes_a_tool_step(op):
    """An operation's own 'tools' field and the map agree, so the bullet (from the map)
    never leaves out an operation the operations table offers 'tools' on."""
    fields = next(c["fields"] for c in nn_edits.catalogue("broad") if c["op"] == op)
    if "tools" in fields:
        assert op in _created("tools"), op


@pytest.mark.parametrize("preset", ["broad", "structure", "conservative"])
def test_the_tools_bullet_names_only_the_operations_this_run_offers(preset):
    offered = {c["op"] for c in nn_edits.catalogue(preset)}
    assert _named_in_tools_line(offered) == offered & set(_created("tools")), preset


@pytest.mark.parametrize("op", _EVERY_OP)
def test_the_node_reference_names_an_operation_exactly_when_it_takes_a_node(op):
    """[request text] "Each node in full" says which operations take a copied node: those
    whose fields have `node`, generated from the catalogue."""
    import re

    from refactory_dsh.optim import base

    catalogue = nn_edits.catalogue("broad")
    text = "\n".join(base._node_specs([{"id": "a", "kind": "agent", "prompt_template": "x",
                                        "writes": ["a.txt"]}], catalogue=catalogue))
    said = re.search(r"use it as the `node` field of (.*?)\. Knob", text)
    named = set(re.findall(r"`([a-z_]+)`", said.group(1))) if said else set()
    fields = next(c["fields"] for c in catalogue if c["op"] == op)
    assert (op in named) == ("node" in fields), (op, named)


# ── the composition examples: one of each shape, each one an attempt takes ────────


def _composition_modes():
    from refactory_packages.livemath import mode as lm_mode, skill_graph
    from refactory_packages.spreadsheetbench import agent_mode, mode as sb_mode
    from refactory_packages.spreadsheetbench_v2 import steps_mode, swe_mode

    return {"sb_solver": sb_mode.Solver, "sb_agent": agent_mode.Agent, "sb2_swe": swe_mode.Swe,
            "sb2_steps": steps_mode.Steps,
            "livemath_solver": lm_mode.Solver, "livemath_skill_graph": skill_graph.SkillGraph}


_GRAPHS = ["sb_solver", "sb_agent", "sb2_swe", "sb2_steps", "livemath_solver",
           "livemath_skill_graph"]


def _composition_lines(mode) -> list[str]:
    from refactory_dsh.optim import base
    from refactory_dsh.optim.memory import state_view

    view = state_view(mode.state_dict())
    return base.composition_section(
        nn_edits.catalogue("broad"), view.node_states, view.edges, code_nodes=True,
        tools=True, protected=mode.edit_protected(),
        knob_values={k["name"]: k["value"] for k in view.knobs},
        frozen=[k["name"] for k in view.knobs if k.get("frozen")])


def _shown_and_refused(lines: list[str]) -> tuple[list[str], list[str]]:
    from refactory_dsh.optim import base

    shown = []
    if base.COMPOSITION_LEAD in lines:
        shown = [line[2:].split(": {", 1)[0]
                 for line in lines[lines.index(base.COMPOSITION_LEAD) + 1:]
                 if line.startswith("- ")]
    said = next((line for line in lines if line.startswith(base.REFUSED_LEAD)), "")
    refused = [shape for shape in base.COMPOSITION_SHAPES if f"{shape}: " in said]
    return shown, refused


def test_the_composition_leads_are_neutral_facts():
    from refactory_dsh.optim import base

    assert base.COMPOSITION_LEAD == ("One example of each composition shape, with this graph's "
                                     "ids (how to write it; none is preferred):")
    assert base.REFUSED_LEAD == "Shapes not shown here, with the refusal of the forms tried: "


@pytest.mark.parametrize("name", _GRAPHS)
def test_every_composition_shape_is_shown_once_or_said_refused(name):
    """[request text] Each shape (new agent step, fork and join, verdict loop, route, code op,
    tool step) is shown once or named among the refused with the refusal, so the list is
    never read as the full set; titles name the shape, never when to use it."""
    from refactory_dsh.optim import base

    shown, refused = _shown_and_refused(_composition_lines(_composition_modes()[name]()))
    assert len(shown) == len(set(shown)) and not set(shown) & set(refused), (shown, refused)
    assert sorted(shown + refused) == sorted(base.COMPOSITION_SHAPES), (shown, refused)
    if name in ("sb_solver", "livemath_solver", "livemath_skill_graph"):
        assert not refused, refused  # graphs that take every shape show each one


@pytest.mark.parametrize("name", ["sb2_swe", "sb_agent"])
def test_a_frozen_prompt_is_said_and_a_tool_step_taking_over_is_shown(name):
    """On a graph whose solve prompt is frozen, the refusal said is the real one, and the
    tool step that can write what it takes over is shown, its text saying it writes them."""
    import json

    from refactory_dsh.optim import base

    lines = _composition_lines(_composition_modes()[name]())
    said = next(line for line in lines if line.startswith(base.REFUSED_LEAD))
    assert ("a new agent step: knob 'solve__prompt_template' is frozen and cannot be moved"
            in said), said
    assert "nothing reads the output of new step 'solve_next'" in said, said  # every form's
    assert base.COMPOSITION_SHAPES[5] not in said, said
    tool = next(l for l in lines if l.startswith(f"- {base.COMPOSITION_SHAPES[5]}: "))
    edit = json.loads(tool.split(": ", 1)[1])
    assert edit["takes_over"] == "solve" and edit["tools"]
    assert "`reply.txt`" in edit["text"] and "the outputs of `solve` it takes over" in edit["text"]


def _able_to_write(node: dict, file: str) -> bool:
    """Whether a new step can produce ``file``: an agent with tools writes any file, one
    without writes only its reply (its first file), a code op returns text files only."""
    if node.get("kind") == "agent":
        return bool(node.get("tools")) or (node.get("writes") or [None])[0] == file
    return file.endswith((".txt", ".json", ".md"))


def _writers(state: dict, files) -> dict[str, set[str]]:
    nodes = state["graph"]["nodes"]
    return {f: {k for k, n in nodes.items() if f in (n.get("writes") or ())} for f in files}


@pytest.mark.parametrize("name", _GRAPHS)
@pytest.mark.parametrize("shape", list(range(6)))
def test_each_composition_example_applies_and_leaves_the_graph_runnable(name, shape):
    """Each example, exactly as the request shows it, is a valid proposal on that graph:
    `Optimizer.attempt` takes it, it adds what its shape says, every file the protected
    steps read is still written by the same steps (no step's outputs are taken over by a
    step that could not write them), and the code example returns the fact it declares."""
    import json

    from refactory_dsh.optim import Optimizer, Proposal
    from refactory_dsh.optim import base
    from refactory_dsh.workers import StepRoutes

    mode = _composition_modes()[name]()
    title = base.COMPOSITION_SHAPES[shape]
    line = next((l for l in _composition_lines(mode) if l.startswith(f"- {title}: ")), None)
    if line is None:
        pytest.skip(f"{name} refuses {title} (said in the request)")
    shown = json.loads(line.split(": ", 1)[1])
    edits = shown["edits"] if "edits" in shown else [shown]
    optimizer = Optimizer()
    optimizer.sealed_tools = (True, "")
    optimizer.step_routes = StepRoutes(sf.ROUTE_BASE, node_routes=sf.ROUTE_MAPS)
    parent = Candidate.from_state(mode.state_dict())
    out = optimizer.attempt(Proposal(edits=edits), parent, mode.invariants(), True, (),
                            protected=mode.edit_protected(), record=False)
    assert isinstance(out, Candidate), (title, out)
    before = parent.state["graph"]["nodes"]
    new = {k: n for k, n in out.state["graph"]["nodes"].items() if k not in before}
    kinds = {n.get("kind") for n in new.values()}
    expected = {0: {"agent"}, 1: {"fork", "join"}, 2: {"gate"}, 3: {"gate"}, 4: {"op"}}
    assert expected.get(shape, {"agent"}) <= kinds, (title, kinds)
    read = {f for p in mode.edit_protected() if before.get(p, {}).get("kind") != "data"
            for f in before[p].get("reads") or ()}
    after = _writers(out.state, read)
    for file, writers in after.items():
        assert writers, (title, file)  # still written
        for writer in writers - _writers(parent.state, read)[file]:
            assert _able_to_write(out.state["graph"]["nodes"][writer], file), (title, writer, file)
    if shape == 4:
        code = next(n for n in new.values() if n.get("code") or n.get("kind") == "op"
                    and "snippet" in str(n.get("call")))
        text = code.get("code") or out.state["knobs"][f"{code['id']}__code"]["value"]
        scope: dict = {}
        exec(text, scope)  # the example's own code, run as the step would run it
        returned = scope["main"]({"facts": {}, "files": {}})
        assert set(returned["facts"]) == set(code.get("produces") or ()), returned
    if shape == 5:
        assert any(n.get("tools") for n in new.values()), new


def test_the_architecture_doc_names_the_shapes_the_request_shows():
    from refactory_dsh.optim import base

    doc = " ".join((Path(__file__).resolve().parent.parent / "docs" / "architecture.md")
                   .read_text().split())
    for shape in base.COMPOSITION_SHAPES:
        assert shape.split(" (")[0] in doc, shape
