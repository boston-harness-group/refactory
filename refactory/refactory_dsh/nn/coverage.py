"""The DSL coverage map: what the outer loop's edits can reach, construct by construct.

Every construct of the graph language -- each node kind, gate kind, bindable field,
``Node`` field, edge form and composition pattern both runtimes run -- has one entry
here, saying which edit operations create it, change it on an existing node and remove
it. Where none does, the entry names why, from a fixed list (`REASONS`): left out by
design (the grader side, a gate nobody answers in a headless run, an infrastructure
flag, a step's identity, a user's decision), or ``planned``, with the change that will
cover it. ``tests/test_dsl_coverage.py`` generates the list of constructs from the code
and fails when one has no entry, so a new feature cannot land unreachable by accident;
the table in ``docs/outer-loop.md`` is generated from it (`table_markdown`).

The standing rule holds: an existing node changes only through knobs (bound fields) and
topology. A field outside that is reachable only by removing a node and adding another.

The user's decisions the map records (`DECISIONS`): what an existing step reads may be
rewired; a copy of a step that uses tools may keep its tools when it gets its own fresh
workspace; ``configure`` may change timeouts, retries, loop limits, optional branches,
the workspace kind and the model role, never tools or ``outputs_only``; the per-sub-item
experiment comes after the core operators.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Union

from .graph import BINDABLE_FIELDS, GATE_KINDS, NODE_KINDS, Node

#: Why a construct, or one way of editing it, has no operation. Fixed: an entry must use
#: one of these.
REASONS: dict[str, str] = {
    "grader-side": "the data pass and the grading stay as the mode's author built them",
    "no headless answerer": "nobody answers it while an outer loop's candidates run",
    "infrastructure flag": "a runtime or lowering setting with no search value",
    "identity": "what a step is: a different one is a different step (remove it, add one)",
    "excluded by decision": "the user decided the search may not change it",
    "awaiting a user decision": "reachable only once the user decides",
    "planned": "a later change adds the operation named",
}


#: Decisions the user made about what the search may change, which later operators follow.
DECISIONS: dict[str, str] = {
    "rewire": "an operator may change what an existing step reads (its reads and inputs)",
    "tool copies": "a copy of a step that uses tools keeps its tools only with its own fresh "
                   "workspace",
    "configure": "timeout_s, retry, max_iterations, on_exhausted, a branch's optional, the "
                 "workspace kind and the role among the run's routes; never tools, tool_policy "
                 "or outputs_only",
    "per-sub-item": "a step per sub-item is tried as an experiment on both runtimes after the "
                    "core operators",
}


@dataclass(frozen=True)
class Gap:
    """No operation does this, and why: a key of `REASONS`, and what it means here."""

    reason: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.reason}: {self.detail}" if self.detail else self.reason


#: The operations that do one thing to a construct, or why none does; ``None`` where it
#: does not apply (a field is not removed from a node).
Aspect = Union[tuple[str, ...], Gap, None]


@dataclass(frozen=True)
class Entry:
    """One construct and how the edits reach it."""

    group: str
    name: str
    create: Aspect
    change: Aspect
    remove: Aspect = None


#: Operations that add a node object of the given kind.
_ADD = ("add_node", "insert_after", "insert_before", "wrap", "split", "parallel", "duplicate")
_GATE_ADD = ("add_node", "wrap", "loop")
_NEW = ("add_node", "insert_after", "insert_before", "wrap", "split", "parallel", "loop")
_PLANNED_REWIRE = Gap("planned", "rewire, which changes what an existing step reads (PR2)")
_PLANNED_CONFIGURE = Gap("planned", "configure, which binds the setting to a knob (PR3)")
_PLANNED_FORK = Gap("planned", "branch / unbranch and serialize (PR2)")
_OUTPUTS = Gap("identity", "a step's outputs are what its readers rely on: replace the step "
                           "(split) to change them")
_GRADER = Gap("grader-side")
_FLAG = Gap("infrastructure flag")


def _node_kinds() -> list[Entry]:
    return [
        Entry("node kind", "op", _ADD, ("set_knob",), ("remove_node", "bypass")),
        Entry("node kind", "agent", _ADD, ("set_knob",), ("remove_node", "bypass")),
        Entry("node kind", "gate", _GATE_ADD, ("set_knob",), ("remove_node", "add_edge")),
        Entry("node kind", "fork", ("parallel", "add_node"), _PLANNED_FORK, _PLANNED_FORK),
        Entry("node kind", "join", ("parallel", "add_node"), _PLANNED_FORK, _PLANNED_FORK),
        Entry("node kind", "data", _GRADER, ("set_knob",), _GRADER),
    ]


def _gate_kinds() -> list[Entry]:
    nobody = Gap("no headless answerer", "refused E-gate-kind, and left out of the schema")
    return [
        Entry("gate kind", "fn", _GATE_ADD, ("set_knob",), ("remove_node", "add_edge")),
        Entry("gate kind", "user", nobody, nobody, ("remove_node", "add_edge")),
        Entry("gate kind", "agent", nobody, nobody, ("remove_node", "add_edge")),
    ]


def _bindable() -> list[Entry]:
    return [
        Entry("bindable field", "prompt_template", _NEW, ("set_knob",)),
        Entry("bindable field", "system_prompt_template", _NEW, ("set_knob",)),
        Entry("bindable field", "code", _NEW, ("set_knob",)),
        Entry("bindable field", "max_iterations", _GATE_ADD, ("set_knob",)),
        Entry("bindable field", "parallelism", _GRADER, ("set_knob",)),
    ]


#: Each ``Node`` field: set on a node an edit adds, changed on an existing node.
_FIELDS: dict[str, tuple[Aspect, Aspect]] = {
    "id": (_NEW, Gap("identity")),
    "kind": (_NEW, Gap("identity")),
    "call": (_NEW, Gap("planned", "split with one new step, which replaces a step (PR2)")),
    "evaluator": (_NEW, Gap("planned", "split with one new step, which replaces a step (PR2)")),
    "gate_kind": (_GATE_ADD, Gap("identity")),
    "knobs": (_NEW, _PLANNED_REWIRE),
    "reads": (_NEW, _PLANNED_REWIRE),
    "required_reads": (_NEW, _PLANNED_REWIRE),
    "inputs": (_NEW, _PLANNED_REWIRE),
    "writes": (_NEW, _OUTPUTS),
    "produces": (_NEW, _OUTPUTS),
    "prompt_template": (_NEW, ("set_knob",)),
    "system_prompt_template": (_NEW, ("set_knob",)),
    "code": (_NEW, ("set_knob",)),
    "role": (_NEW, Gap("planned", "configure, among the run's declared routes (PR3)")),
    "tools": (_NEW, Gap("excluded by decision", "tools are not a configure setting")),
    "tool_policy": (_NEW, Gap("excluded by decision", "tools are not a configure setting")),
    "outputs_only": (_NEW, Gap("excluded by decision", "not a configure setting")),
    "max_iterations": (_GATE_ADD, ("set_knob",)),
    "on_exhausted": (_GATE_ADD, _PLANNED_CONFIGURE),
    "timeout_s": (_NEW, _PLANNED_CONFIGURE),
    "retry": (_NEW, _PLANNED_CONFIGURE),
    "workspace": (_NEW, Gap("planned", "configure, the workspace kind (PR3)")),
    "targets": (("parallel", "add_node"), _PLANNED_FORK),
    "optional_targets": (("add_node",), Gap("planned", "branch with optional (PR2)")),
    "sources": (("parallel", "add_node"), _PLANNED_FORK),
    "field_bindings": (_NEW, _PLANNED_CONFIGURE),
    "feedback_keys": (_NEW, _FLAG),
    "notes": (_NEW, _FLAG),
    "blocking": (_NEW, _FLAG),
    "score_effect": (_NEW, _FLAG),
    "resource_role": (_NEW, _FLAG),
    "subgraph_entry": (_GRADER, _GRADER),
    "subgraph_exit": (_GRADER, _GRADER),
    "task_ref": (_GRADER, _GRADER),
    "source_path": (_GRADER, _GRADER),
    "source_format": (_GRADER, _GRADER),
    "inline_items": (_GRADER, _GRADER),
    "limit": (_GRADER, _GRADER),
    "parallelism": (_GRADER, ("set_knob",)),
    "parallelism_knob": (_GRADER, _GRADER),
}


def _node_fields() -> list[Entry]:
    return [Entry("node field", field.name, *_FIELDS[field.name])
            for field in dataclasses.fields(Node) if field.name in _FIELDS]


#: Every way control leaves a node.
EDGE_FORMS = ("plain", "verdict", "reloop", "halt")


def _edges() -> list[Entry]:
    return [
        Entry("edge form", "plain", ("add_edge", "insert_after", "insert_before"),
              ("remove_edge", "add_edge"), ("remove_edge",)),
        Entry("edge form", "verdict", ("add_edge", "loop", "wrap"),
              ("remove_edge", "add_edge"), ("remove_edge",)),
        Entry("edge form", "reloop", ("loop", "add_edge"), ("remove_edge", "add_edge"),
              ("remove_edge",)),
        # A verdict that ends the run: a code gate's answer, with no edge of its own.
        Entry("edge form", "halt", ("add_node", "loop"), ("set_knob",), ("remove_node",)),
    ]


#: The shapes both runtimes run that an edit composes from several constructs.
COMPOSITIONS = ("nested fork", "loop inside a fork branch", "loop inside a data pass",
                "gate routing to several steps", "a step per sub-item")


def _compositions() -> list[Entry]:
    return [
        Entry("composition", "nested fork", ("parallel",), _PLANNED_FORK, _PLANNED_FORK),
        Entry("composition", "loop inside a fork branch", ("loop",), ("set_knob",),
              ("remove_node", "add_edge")),
        Entry("composition", "loop inside a data pass", ("loop",), ("set_knob",),
              ("remove_node", "add_edge")),
        Entry("composition", "gate routing to several steps", ("add_node", "add_edge"),
              ("remove_edge", "add_edge"), Gap("planned", "remove with keep, which unroutes (PR2)")),
        Entry("composition", "a step per sub-item",
              Gap("planned", "a per-sub-item experiment on both runtimes, after PR2"),
              Gap("planned", "after the per-sub-item experiment"),
              Gap("planned", "after the per-sub-item experiment")),
    ]


def entries() -> list[Entry]:
    """The whole map."""
    return [*_node_kinds(), *_gate_kinds(), *_bindable(), *_node_fields(), *_edges(),
            *_compositions()]


def constructs() -> dict[str, set[str]]:
    """Every construct the map must cover, by group, generated from the code."""
    return {
        "node kind": set(NODE_KINDS),
        "gate kind": set(GATE_KINDS),
        "bindable field": set(BINDABLE_FIELDS),
        "node field": {field.name for field in dataclasses.fields(Node)},
        "edge form": set(EDGE_FORMS),
        "composition": set(COMPOSITIONS),
    }


def _cell(aspect: Aspect) -> str:
    if aspect is None:
        return "-"
    if isinstance(aspect, Gap):
        return f"*{aspect}*"
    return ", ".join(f"`{op}`" for op in aspect)


def table_markdown() -> str:
    """The map as the Markdown table ``docs/outer-loop.md`` carries."""
    rows = ["| construct | create | change on an existing node | remove |",
            "|---|---|---|---|"]
    for entry in entries():
        rows.append(f"| {entry.group}: `{entry.name}` | {_cell(entry.create)} | "
                    f"{_cell(entry.change)} | {_cell(entry.remove)} |")
    return "\n".join(rows)
