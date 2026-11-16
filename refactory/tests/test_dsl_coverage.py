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
    ("node kind", "join"): ("fuzz_dataflow", {"op": "parallelize", "after": "clean", "new": [
        {"kind": "agent", "text": "One."}, {"kind": "agent", "text": "Two."}]},
        lambda p, c: _kinds(c, "join") == _kinds(p, "join") + 1),
    ("gate kind", "fn"): ("fuzz_draft_review", {"op": "loop", "step": "draft", "judge": "agent"},
                          lambda p, c: c.nodes["draft_check"].gate_kind == "fn"),
    ("edge form", "verdict"): ("fuzz_data_pass", {"op": "route", "after": "solve", "to": {
        "proceed": "check", "again": {"kind": "agent", "text": "Again."}}},
        lambda p, c: "again" in _conditions(c) and "again" not in _conditions(p)),
    ("composition", "nested fork"): ("fuzz_nested_fork", {"op": "branch", "fork": "inner",
                                                          "kind": "agent", "text": "Deeper."},
                                     lambda p, c: len(c.nodes["inner"].targets) == 3),
    ("composition", "gate routing to several steps"): ("fuzz_draft_review", {
        "op": "route", "after": "review", "to": {"proceed": "finalize",
                                                 "rewrite": {"kind": "agent", "text": "Redo."}}},
        lambda p, c: len(c.out_edges("review_route")) == 2),
    ("node field", "optional_targets"): ("fuzz_fork_join", {"op": "branch", "fork": "split",
                                                            "optional": True},
                                         # The new branch's first step binds its optional.
                                         lambda p, c: [t for t in c.nodes["split"].targets
                                                       if c.nodes[t].bound_knob("optional")]
                                         == [c.nodes["split"].targets[-1]]),
})


#: Covered changes and removals of an existing construct: the seed, the edit, which of the
#: entry's aspects names its operation, and what the child shows.
CHANGES: dict[tuple[str, str, str], tuple[str, Any, Callable[[Graph, Graph], bool]]] = {
    ("node kind", "fork", "change"): ("fuzz_fork_join", {"op": "branch", "fork": "split"},
                                      lambda p, c: len(c.nodes["split"].targets) == 3),
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
    ("node field", "targets", "change"): ("fuzz_fork_join", {"op": "insert", "before": "left"},
                                          lambda p, c: "left" not in c.nodes["split"].targets),
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
