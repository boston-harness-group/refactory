"""The DSL coverage map: what the outer loop's edits can reach, construct by construct.

Every construct of the graph language -- each node kind, gate kind, bindable field,
``Node`` field, edge form and composition pattern both runtimes run -- has one entry
here, saying which edit operations create it, change it on an existing node and remove
it. Where none does, the entry names why, from a fixed list (`REASONS`): left out by
design (the grader side, a gate nobody answers in a headless run, an infrastructure
flag, a step's identity, a user's decision), or ``deferred`` by the user, with why in
`DECISIONS`. Nothing is left planned: the core operators (`nn.core_ops`) cover the forks and
joins, rewiring what a step reads, replacing a step's callable and un-routing a gate, and a
loop's limit and a branch's optional are settings bound to knobs (``loop``, ``branch`` with
``optional``) that ``set_knob`` moves. ``tests/test_dsl_coverage.py`` generates the list of
constructs from the code and fails when one has no entry or an entry is left planned, so a
new feature cannot land unreachable by accident; the table in ``docs/outer-loop.md`` is
generated from it (`table_markdown`).

The standing rule holds: an existing node changes only through knobs (bound fields) and
topology. A field outside that is reachable only by removing a node and adding another.

The user's decisions the map records (`DECISIONS`): what an existing step reads may be
rewired; a copy of a step that uses tools may keep its tools when it gets its own fresh
workspace; settings bound to knobs may be loop limits and optional branches, never tools or
``outputs_only``; a new step may choose its own model, effort and output cap among the run's
routes; timeouts, retries, the workspace kind, the model role and the per-sub-item
experiment are deferred.
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
    "deferred": "the user deferred it; DECISIONS['deferred'] says why",
}


#: Decisions the user made about what the search may change, which later operators follow.
DECISIONS: dict[str, str] = {
    "rewire": "an operator may change what an existing step reads (its reads and inputs)",
    "tool copies": "a copy of a step that uses tools keeps its tools only with its own fresh "
                   "workspace",
    "configure": "settings bound to knobs, which set_knob moves: a loop gate's max_iterations "
                 "(`loop` makes its knob) and a branch's optional (`branch` with 'optional'); "
                 "never tools, tool_policy or outputs_only",
    "step routes": "approved 2026-11-17: a new agent step may set its own model, effort "
                   "and output cap (`route`), each a value one of the run's routes already "
                   "uses (`workers.StepRoutes`), never one it runs on already; failing fast "
                   "earns nothing, as parsimony is 0 and a time-out is a harness timeout, "
                   "not a scored failure. A step's role stays its neighbour's",
    "deferred": "left for later on 2026-11-16: timeout_s, retry and on_exhausted as knobs "
                "(they mostly trade cost and reliability, invite failing fast to game the "
                "score against 'a time-out counts as a failure', and touch node_retry on both "
                "runtimes); the workspace kind and the model role; binding a new setting on an "
                "existing step; and the per-sub-item experiment",
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
_ADD = ("add_node", "insert_after", "insert_before", "wrap", "split", "parallel", "duplicate",
        "insert", "parallelize", "branch", "route", "merge")
_GATE_ADD = ("add_node", "wrap", "loop", "route")
_NEW = ("add_node", "insert_after", "insert_before", "wrap", "split", "parallel", "loop",
        "insert", "parallelize", "branch", "route", "merge")
_STEP_REMOVE = ("remove_node", "bypass", "remove")
_GATE_REMOVE = ("remove_node", "add_edge", "remove")
_FORK_CREATE = ("parallel", "parallelize", "add_node")
_FORK_CHANGE = ("branch", "remove", "parallelize", "insert")
_FORK_REMOVE = ("serialize", "remove")
_REWIRE = ("rewire",)
_REPLACE = ("split",)
_DEFERRED_SETTING = Gap("deferred", "a timeout or retry setting as a knob")
_DEFERRED_BINDING = Gap("deferred", "binding a new setting on an existing step")
_OUTPUTS = Gap("identity", "a step's outputs are what its readers rely on: replace the step "
                           "(split) to change them")
_GRADER = Gap("grader-side")
_FLAG = Gap("infrastructure flag")


def _node_kinds() -> list[Entry]:
    return [
        Entry("node kind", "op", _ADD, ("set_knob", "rewire"), _STEP_REMOVE),
        Entry("node kind", "agent", _ADD, ("set_knob", "rewire"), _STEP_REMOVE),
        Entry("node kind", "gate", _GATE_ADD, ("set_knob",), _GATE_REMOVE),
        Entry("node kind", "fork", _FORK_CREATE, _FORK_CHANGE, _FORK_REMOVE),
        Entry("node kind", "join", _FORK_CREATE, _FORK_CHANGE, _FORK_REMOVE),
        Entry("node kind", "data", _GRADER, ("set_knob",), _GRADER),
    ]


def _gate_kinds() -> list[Entry]:
    nobody = Gap("no headless answerer", "refused E-gate-kind, and left out of the schema")
    return [
        Entry("gate kind", "fn", _GATE_ADD, ("set_knob",), _GATE_REMOVE),
        Entry("gate kind", "user", nobody, nobody, _GATE_REMOVE),
        Entry("gate kind", "agent", nobody, nobody, _GATE_REMOVE),
    ]


def _bindable() -> list[Entry]:
    return [
        Entry("bindable field", "prompt_template", _NEW, ("set_knob",)),
        Entry("bindable field", "system_prompt_template", _NEW, ("set_knob",)),
        Entry("bindable field", "code", _NEW, ("set_knob",)),
        Entry("bindable field", "max_iterations", _GATE_ADD, ("set_knob",)),
        Entry("bindable field", "parallelism", _GRADER, ("set_knob",)),
        Entry("bindable field", "optional", ("branch",), ("set_knob",)),
        *(Entry("bindable field", f"route.{name}", _NEW, ("set_knob",))
          for name in ("max_completion_tokens", "max_thinking_tokens", "max_turns",
                       "bash_timeout_ms", "bash_output_chars")),
    ]


#: Each ``Node`` field: set on a node an edit adds, changed on an existing node.
_FIELDS: dict[str, tuple[Aspect, Aspect]] = {
    "id": (_NEW, Gap("identity")),
    "kind": (_NEW, Gap("identity")),
    "call": (_NEW, _REPLACE),
    "evaluator": (_NEW, _REPLACE),
    "gate_kind": (_GATE_ADD, Gap("identity")),
    "knobs": (_NEW, _DEFERRED_BINDING),
    "reads": (_NEW, _REWIRE),
    "required_reads": (_NEW, _REWIRE),
    "inputs": (_NEW, _REWIRE),
    "writes": (_NEW, _OUTPUTS),
    "produces": (_NEW, _OUTPUTS),
    "prompt_template": (_NEW, ("set_knob",)),
    "system_prompt_template": (_NEW, ("set_knob",)),
    "code": (_NEW, ("set_knob",)),
    "role": (_NEW, Gap("deferred", "the model role among the run's routes")),
    # A step's own model, effort and cap, among the run's routes (DECISIONS['step routes']).
    "route": (("add_node", "insert", "branch", "merge", "split", "parallelize", "route"),
              _REPLACE),
    "tools": (_NEW, Gap("excluded by decision", "tools are not a configure setting")),
    "tool_policy": (_NEW, Gap("excluded by decision", "tools are not a configure setting")),
    "outputs_only": (_NEW, Gap("excluded by decision", "not a configure setting")),
    "keeps_tool_markup": (_NEW, Gap("excluded by decision", "not a configure setting")),
    "knob_outputs": (_NEW, Gap("excluded by decision",
                               "the mode's declaration of what its op does")),
    # What a code step's run records its returned outputs under: set by an edit that gives
    # code new output names (a copy of a code step; 'takes_over' of one), never by hand.
    "renames": (("add_node", "insert", "split", "branch", "parallelize"), ("insert",)),
    "max_iterations": (_GATE_ADD, ("set_knob",)),
    "on_exhausted": (_GATE_ADD, _DEFERRED_SETTING),
    "timeout_s": (_NEW, _DEFERRED_SETTING),
    "retry": (_NEW, _DEFERRED_SETTING),
    "workspace": (_NEW, Gap("deferred", "the workspace kind")),
    "targets": (_FORK_CREATE, _FORK_CHANGE),
    # A branch made with 'optional' binds it to a knob (`Graph.optional_targets_of`).
    "optional_targets": (("branch", "add_node"), ("set_knob",)),
    "sources": (_FORK_CREATE, _FORK_CHANGE),
    "field_bindings": (_NEW, _DEFERRED_BINDING),
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
        Entry("edge form", "plain", ("add_edge", "insert_after", "insert_before", "insert"),
              ("remove_edge", "add_edge", "serialize", "remove"), ("remove_edge", "remove")),
        Entry("edge form", "verdict", ("add_edge", "loop", "wrap", "route"),
              ("remove_edge", "add_edge"), ("remove_edge", "remove")),
        Entry("edge form", "reloop", ("loop", "add_edge"), ("remove_edge", "add_edge"),
              ("remove_edge", "remove")),
        # A verdict that ends the run: a code gate's answer, with no edge of its own.
        Entry("edge form", "halt", ("add_node", "loop"), ("set_knob",), ("remove_node", "remove")),
    ]


#: The shapes both runtimes run that an edit composes from several constructs.
COMPOSITIONS = ("nested fork", "loop inside a fork branch", "loop inside a data pass",
                "gate routing to several steps", "a step per sub-item")


def _compositions() -> list[Entry]:
    return [
        Entry("composition", "nested fork", ("parallel", "parallelize", "branch"),
              _FORK_CHANGE, _FORK_REMOVE),
        Entry("composition", "loop inside a fork branch", ("loop",), ("set_knob",),
              _GATE_REMOVE),
        Entry("composition", "loop inside a data pass", ("loop",), ("set_knob",),
              _GATE_REMOVE),
        Entry("composition", "gate routing to several steps", ("route", "add_node", "add_edge"),
              ("remove_edge", "add_edge"), ("remove",)),
        Entry("composition", "a step per sub-item", *(Gap("deferred", "the per-sub-item "
                                                                  "experiment"),) * 3),
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
