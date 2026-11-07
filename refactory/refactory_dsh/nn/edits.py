"""The edit algebra: every structural change is typed data.

An edit is a JSON object naming an operation, and applying it produces a new
state. Two properties make this the write surface the outer loop can be trusted
with:

**Closed and enumerable.** Seven primitives are the algebra. Every macro is a
literal composition of them, so a macro cannot express anything the primitives
cannot, and the set of reachable edits is enumerable and reviewable.

**Naive.** No operator repairs, auto-fixes, or maintains an invariant as it
edits. ``remove_node`` deletes the node and its edges and leaves the hole; the
validator then reports the hole. Repair is what makes a search dishonest: if a
proposal is fixed up before it runs, the score belongs to something the optimizer
never proposed.

A macro whose precondition does not hold is refused with a reason, and the state
is untouched. A macro that applies but produces a broken graph is a validation
failure. Those are two different messages, and both are fed back to the optimizer.
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from .graph import Edge, Graph, GraphError, Node

State = dict[str, Any]


class EditError(ValueError):
    """An edit cannot be applied to this state. The state is left untouched.

    ``code`` groups refusals so a run can report which rule refused what. A
    histogram of prose is noise; a histogram of rules is the gradient on the edit
    catalogue itself.
    """

    def __init__(self, message: str, code: str = "E-shape") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EditSpec:
    """One operation in the catalogue."""

    name: str
    kind: str
    description: str
    fields: tuple[str, ...]
    macro: bool = False
    precondition: Callable[[dict, State], str | None] | None = None
    expand: Callable[[dict, State], list[dict]] | None = None
    apply: Callable[[State, dict], State] | None = None


_PRIMITIVES: dict[str, EditSpec] = {}
_MACROS: dict[str, EditSpec] = {}

#: Named subsets. ``broad`` is the default: the whole catalogue.
PRESETS: dict[str, tuple[str, ...]] = {
    "broad": (),  # filled in below with every registered name
    "structure": (),  # filled in below with primitives + macros
    "conservative": ("set_knob", "insert_after", "insert_before", "wrap"),
    "knob_only": ("set_knob", "add_knob", "remove_knob"),
}


def _primitive(name: str, kind: str, description: str, fields: Sequence[str],
               fn: Callable[[State, dict], State]) -> None:
    _PRIMITIVES[name] = EditSpec(name, kind, description, tuple(fields), apply=fn)


def _macro(name: str, description: str, fields: Sequence[str],
           precondition: Callable[[dict, State], str | None],
           expand: Callable[[dict, State], list[dict]]) -> None:
    _MACROS[name] = EditSpec(name, "topology", description, tuple(fields),
                             macro=True, precondition=precondition, expand=expand)


# ── state helpers ──────────────────────────────────────────────────────────
def _graph(state: Mapping[str, Any]) -> Graph:
    try:
        return Graph.from_state(state["graph"])
    except (KeyError, GraphError) as exc:
        raise EditError(f"state is not readable: {exc}", code="E-state") from exc


def _finish(state: Mapping[str, Any], graph: Graph) -> State:
    _refuse_baked_items(graph)
    out = dict(state)
    out["graph"] = graph.to_state()
    out["knobs"] = copy.deepcopy(dict(state.get("knobs") or {}))
    return out


#: The node fields that decide *which* items a harness reads. A produced package is
#: refused for baking its items in, and a search that could set these would be held to
#: a weaker rule than the thing it edits: `inline_items` scores the items it was given,
#: and a moved `task_ref` or `source_path` changes the exam the candidate is graded on.
ITEM_SOURCE_FIELDS = ("task_ref", "source_path", "source_format", "inline_items", "limit", "produces")


def _refuse_baked_items(graph: Graph) -> None:
    """No state an edit produces bakes its items into the graph.

    Every primitive returns through here, so this holds for the ops that set a field,
    the ones that add or replace a node, and the macros built out of them. A person's
    route to different items is a run's `--source`, not a graph that carries them.
    """
    baked = sorted(
        node.id for node in graph.nodes.values() if getattr(node, "inline_items", ())
    )
    if baked:
        raise EditError(
            f"node(s) {', '.join(baked)} declare inline_items, which writes the items "
            "into the graph; a harness for a class of work reads its items at run time, "
            "so an edit may not bake them in",
            code="E-items",
        )


def _require(graph: Graph, node_id: Any) -> Node:
    if not isinstance(node_id, str) or node_id not in graph.nodes:
        known = ", ".join(sorted(graph.nodes)) or "none"
        raise EditError(f"no node {node_id!r}; existing nodes: {known}", code="E-node")
    return graph.nodes[node_id]


def _drop(graph: Graph, edge: Edge) -> None:
    graph.edges = [e for e in graph.edges if e.key() != edge.key()]


def _sole_out_edge(graph: Graph, node_id: str) -> Edge | None:
    """The one out-edge a splice may rewrite, or ``None`` for a terminal.

    Raises when the node's out-edges leave the choice ambiguous, because picking
    one would be the operator deciding something the optimizer did not say.
    """
    out = graph.out_edges(node_id)
    if not out:
        return None
    if len(out) > 1:
        raise EditError(
            f"node {node_id!r} has {len(out)} out-edges, so 'after' does not say "
            "which one to splice; use add_edge/remove_edge for an explicit rewrite"
        )
    return out[0]


def _dependency_violation(graph: Graph, first: str, second: str) -> str | None:
    """Reason a node may not run immediately before another it feeds."""
    produced = set(graph.nodes[first].writes)
    consumed = set(graph.nodes[second].reads)
    shared = sorted(produced & consumed)
    if shared:
        return (
            f"data dependency: {second!r} reads {shared}, which only {first!r} "
            "produces, so the two cannot be swapped"
        )
    return None


# ── primitives ─────────────────────────────────────────────────────────────
def _position(position: Any) -> tuple[str, Any]:
    if not isinstance(position, dict) or len(position) != 1:
        raise EditError(
            "a position must be exactly one of {'after': id}, {'before': id}, "
            "{'start': true}, {'terminal': true} or {'detached': true}"
        )
    (kind, value), = position.items()
    if kind not in ("after", "before", "start", "terminal", "detached"):
        raise EditError(f"unknown position {kind!r}")
    return kind, value


#: What a clipped text looks like in an optimizer's prompt (``optim.base._clip_middle``).
_CLIP_MARKER = re.compile(r"\[\.\.\. \d+ characters omitted \.\.\.\]")
#: A template slot opened (``{{``) and never closed before the next brace.
_UNCLOSED_SLOT = re.compile(r"\{\{(?![^{}]*\}\})")


def _check_new_node_texts(node_state: Mapping[str, Any]) -> None:
    """A new node's texts carry no clip marker and no unclosed ``{{``."""
    for key, value in node_state.items():
        if not isinstance(value, str):
            continue
        if _CLIP_MARKER.search(value):
            raise EditError(
                f"node {node_state.get('id')!r}: {key} holds a clipped text (it contains "
                "'[... N characters omitted ...]'); write the text yourself or edit the "
                "knob it is bound to, rather than copying a clipped one", code="E-node-text")
        if _UNCLOSED_SLOT.search(value):
            raise EditError(
                f"node {node_state.get('id')!r}: {key} opens a template slot '{{{{' that "
                "is never closed with '}}'", code="E-node-text")


def _own_bindings(node: Node, graph: Graph, state: Mapping[str, Any]) -> tuple[Node, dict[str, Any]]:
    """The new node with knobs of its own, and the knobs to declare for it.

    A new node may not bind a knob another node binds: it would silently run that
    node's value (a copied agent running the original's system prompt). A new agent
    node with no system-prompt binding gets its own ``<id>__system_prompt`` knob, as
    a mode's agent nodes do when it is built (``Mode._auto_system_prompt_knobs``),
    holding the system prompt the edit gave it.
    """
    owners = {knob: owner.id for owner in graph.nodes.values()
              for _field, knob in owner.field_bindings}
    readers = graph.knob_consumers()
    for _field, knob in node.field_bindings:
        if knob in owners or knob in readers:
            other = owners.get(knob) or readers[knob][0]
            how = "owns" if knob in owners else "reads"
            raise EditError(
                f"new node {node.id!r} binds knob {knob!r}, which node {other!r} {how}; "
                f"drop field_bindings (and the knob from its knobs) -- a new agent node gets "
                f"its own {node.id}__system_prompt knob from its system_prompt_template",
                code="E-binding")
    if node.kind != "agent" or any(f == "system_prompt_template" for f, _ in node.field_bindings):
        return node, {}
    from .knob import Knob

    name = f"{node.id}__system_prompt"
    if name in (state.get("knobs") or {}):
        raise EditError(f"knob {name!r} already exists, so node {node.id!r} cannot get its own "
                        "system-prompt knob; choose another node id", code="E-knob")
    knob = Knob(node.system_prompt_template or "", "str", description=f"system prompt for {node.id}")
    node = replace(node, knobs=(*node.knobs, name),
                   field_bindings=(*node.field_bindings, ("system_prompt_template", name)))
    return node, {name: {"value": knob.default, "spec": knob.spec()}}


def _add_node(state: State, edit: dict) -> State:
    node_state = edit.get("node")
    if not isinstance(node_state, dict):
        raise EditError("add_node needs a 'node' object")
    _check_new_node_texts(node_state)
    try:
        node = Node.from_state(node_state)
    except GraphError as exc:
        raise EditError(str(exc)) from exc
    kind, value = _position(edit.get("position"))
    graph = _graph(state)
    if node.id in graph.nodes:
        raise EditError(f"node {node.id!r} already exists")
    node, own_knobs = _own_bindings(node, graph, state)

    if kind == "start":
        previous = graph.start
        graph.add(node)
        graph.start = node.id
        if previous:
            graph.connect(node.id, previous)
    elif kind == "detached":
        # No edges at all. The validator reports the node as unreachable until a
        # later step of the same macro wires it, which is what makes a multi-step
        # rewrite expressible without the operator inventing a temporary edge.
        graph.add(node)
    elif kind == "terminal":
        # The run's terminals, not a data pass's exit: an exit has no out-edge
        # because its pass ends there, and a node wired after it would run once
        # per item inside the pass rather than once at the end of the run.
        in_passes = _data_pass_nodes(graph)
        terminals = [terminal for terminal in graph.terminals() if terminal not in in_passes]
        graph.add(node)
        for terminal in terminals:
            graph.connect(terminal, node.id)
    elif kind == "after":
        target = _require(graph, value)
        edge = _sole_out_edge(graph, target.id)
        graph.add(node)
        if edge is not None:
            _drop(graph, edge)
            graph.connect(target.id, node.id)
            graph.connect(node.id, edge.target, edge.condition)
        else:
            graph.connect(target.id, node.id)
        # A node added after a data pass's exit is the pass's new last node, so
        # the exit moves to it: both runtimes end a pass at its exit, and a node
        # left after it would never run on the spine.
        _move_exits(graph, target.id, node.id)
    else:  # before
        target = _require(graph, value)
        incoming = list(graph.in_edges(target.id))
        graph.add(node)
        for edge in incoming:
            _drop(graph, edge)
            graph.connect(edge.source, node.id, edge.condition)
        graph.connect(node.id, target.id)
        if graph.start == target.id:
            graph.start = node.id
        # Likewise a node added before a data pass's entry is its new first node.
        for data_id, data in list(graph.nodes.items()):
            if data.kind == "data" and data.subgraph_entry == target.id:
                graph.nodes[data_id] = replace(data, subgraph_entry=node.id)
    out = _finish(state, graph)
    out["knobs"].update(own_knobs)
    return out


def _data_pass_nodes(graph: Graph) -> set[str]:
    """Every node some data node runs once per item (``Graph.data_body``)."""
    nodes: set[str] = set()
    for node in graph.nodes.values():
        if node.kind == "data":
            nodes |= graph.data_body(node.id)
    return nodes


def _move_exits(graph: Graph, old_exit: str, new_exit: str) -> None:
    """Make ``new_exit`` the exit of every data pass that ended at ``old_exit``."""
    for data_id, node in list(graph.nodes.items()):
        if node.kind == "data" and node.subgraph_exit == old_exit:
            graph.nodes[data_id] = replace(node, subgraph_exit=new_exit)


def _remove_node(state: State, edit: dict) -> State:
    """Delete a node, and the knobs that were its own.

    A node's own knobs are the ones it binds that carry its id (``<id>__...``: the
    system-prompt knob every agent node gets, a duplicate's copies) and no remaining
    node reads. They go with it, so the state stays valid (V7) and the same id can
    be added again. Any other knob it alone read stays declared, and the validator
    refuses the state, so a mode's declared knobs are never dropped by an edit.
    """
    graph = _graph(state)
    target = _require(graph, edit.get("id"))
    del graph.nodes[target.id]
    graph.edges = [e for e in graph.edges if target.id not in (e.source, e.target)]
    if graph.start == target.id:
        # Left empty on purpose: the validator reports the missing start rather
        # than this operator choosing a successor to promote.
        graph.start = ""
    readers = graph.knob_consumers()
    out = _finish(state, graph)
    for _field, knob in target.field_bindings:
        if knob.startswith(f"{target.id}__") and knob not in readers:
            out["knobs"].pop(knob, None)
    return out


def _add_edge(state: State, edit: dict) -> State:
    graph = _graph(state)
    source = _require(graph, edit.get("source"))
    target = _require(graph, edit.get("target"))
    try:
        graph.connect(source.id, target.id, edit.get("condition"))
    except GraphError as exc:
        raise EditError(str(exc)) from exc
    return _finish(state, graph)


def _remove_edge(state: State, edit: dict) -> State:
    graph = _graph(state)
    source = _require(graph, edit.get("source"))
    target = _require(graph, edit.get("target"))
    condition = edit.get("condition", None)
    matches = [
        e
        for e in graph.edges
        if e.source == source.id
        and e.target == target.id
        and ("condition" not in edit or e.condition == condition)
    ]
    if not matches:
        raise EditError(f"no edge {source.id!r} -> {target.id!r} to remove", code="E-edge")
    for edge in matches:
        _drop(graph, edge)
    return _finish(state, graph)


def _set_knob(state: State, edit: dict) -> State:
    from .knob import Knob, KnobError

    name = edit.get("name")
    entries = dict(state.get("knobs") or {})
    if not isinstance(name, str) or name not in entries:
        known = ", ".join(sorted(entries)) or "none"
        raise EditError(f"no knob {name!r}; declared knobs: {known}", code="E-knob")
    knob = Knob.from_spec(entries[name]["spec"])
    if knob.frozen:
        raise EditError(f"knob {name!r} is frozen and cannot be moved", code="E-knob")
    graph_state = state.get("graph") or {}
    if graph_state.get("nodes") and name not in _graph(state).knob_consumers():
        # Moving it would spend an evaluation on a run nothing in it can see.
        raise EditError(f"knob {name!r} is read by no node of this graph (its node was "
                        "removed), so setting it cannot change the run", code="E-knob")
    # One edit moves a knob, in one of two forms: the whole new value, or a patch
    # (an ordered list of text edits) to a str knob's current text.
    has_value, has_patch = "value" in edit, "patch" in edit
    if has_value == has_patch:
        raise EditError(
            f"set_knob on knob {name!r} needs exactly one of 'value' (the whole new "
            "value) or 'patch' (a list of text edits to a str knob); got "
            + ("both" if has_value else "neither"),
            code="E-knob",
        )
    if has_patch:
        return _patch_knob(state, edit, name, knob, entries)
    if "config" in edit:
        raise EditError(
            f"set_knob on knob {name!r}: 'config' configures a 'patch' and means "
            "nothing with 'value'",
            code="E-knob",
        )
    try:
        value = knob.coerce(edit.get("value"))
    except KnobError as exc:
        raise EditError(f"knob {name!r}: {exc}", code="E-knob") from exc
    # Data-slot preservation check for template bindings.
    import re as _re_slots
    graph = _graph(state)
    for node in graph.nodes.values():
        for field_name, knob_name in getattr(node, "field_bindings", ()):
            if knob_name == name and field_name in ("prompt_template", "system_prompt_template"):
                current_tpl = getattr(node, field_name, "")
                required = set(_re_slots.findall(r'\{\{\s*([A-Za-z_]\w*)\s*\}\}', current_tpl))
                if not required:
                    continue
                present = set(_re_slots.findall(r'\{\{\s*([A-Za-z_]\w*)\s*\}\}', str(value)))
                missing = required - present
                if missing:
                    raise EditError(
                        f"knob {name!r}: template slot(s) {sorted(missing)} required by "
                        f"node {node.id!r}'s {field_name} are missing from the new value",
                        code="E-knob",
                    )
    out = _finish(state, graph)
    out["knobs"][name]["value"] = value
    return out


def _add_knob(state: State, edit: dict) -> State:
    from .knob import Knob, KnobError

    name = edit.get("name")
    if not isinstance(name, str) or not name:
        raise EditError("add_knob needs a 'name'", code="E-knob")
    entries = dict(state.get("knobs") or {})
    if name in entries:
        raise EditError(f"knob {name!r} already exists", code="E-knob")
    try:
        knob = Knob.from_spec(dict(edit.get("spec") or {}))
    except (KnobError, KeyError, TypeError) as exc:
        raise EditError(f"knob {name!r}: {exc}", code="E-knob") from exc
    out = _finish(state, _graph(state))
    out["knobs"][name] = {"value": knob.default, "spec": knob.spec()}
    return out


def _remove_knob(state: State, edit: dict) -> State:
    name = edit.get("name")
    entries = dict(state.get("knobs") or {})
    if not isinstance(name, str) or name not in entries:
        raise EditError(f"no knob {name!r} to remove", code="E-knob")
    out = _finish(state, _graph(state))
    del out["knobs"][name]
    return out


# Text-patch edit semantics ported from microsoft/SkillOpt (MIT License).
# Copyright (c) Microsoft Corporation.
# References:
#   skillopt/optimizer/skill.py — _apply_edit_with_report(), apply_patch_with_report()
#   skillopt/types.py — Edit, Patch, EditOp
# insert_before is a refactory extension not present in SkillOpt's four-op set
# (EditOp = Literal['append', 'insert_after', 'replace', 'delete'] at types.py:26).

#: SkillOpt's own per-kind on_missing choices as one named configuration.
#: Users pass ``config=SKILLOPT_PATCH_CONFIG`` (or a copy with overrides) to
#: get SkillOpt-compatible patch behaviour.
SKILLOPT_PATCH_CONFIG: dict[str, Any] = {
    "match": "exact",
    "on_missing": {
        "insert_after": "append",   # skill.py:109-117 missing target falls back to append
        "insert_before": "skip",    # refactory extension; no SkillOpt analogue
        "replace": "skip",          # skill.py:128-130
        "delete": "skip",           # skill.py:138-140
    },
    "honour_protected": True,
    "protected_regions": [
        # skill.py:14-20 SLOW_UPDATE_START/END and APPENDIX_START/END, protected
        # together as _PROTECTED_REGIONS (skill.py:27-30).
        ["<!-- SLOW_UPDATE_START -->", "<!-- SLOW_UPDATE_END -->"],
        ["<!-- APPENDIX_START -->", "<!-- APPENDIX_END -->"],
    ],
    # skill.py:43-58, 94-96: an edit is skipped when its target's first occurrence
    # starts inside [first start marker, end of first end marker) of any region,
    # checked before the op is dispatched (append and unknown ops included).
    "protected_check": "target_start",
    "strip_markers": True,           # skill.py:66-73
    "insert_position": "end_of_line",
    # skill.py:118-122: content goes on its own line after the target's line:
    # line + "\n" + content + "\n" + rest.
    "insert_after_separator": "\n\n",
    # skill.py:120: a target on a last line with no newline gets text + "\n" +
    # content + "\n".
    "insert_after_eof": "line",
    "trailing_newline": True,        # skill.py:106
    "append_strip_trailing": True,   # skill.py:106 skill.rstrip()
    # skill.py:79: content is .strip()ped, then markers removed (skill.py:78).
    "content_strip": True,
    "unknown_kind_policy": "skip",   # skill.py:144-145
    # skill.py:173-184: an exception in one edit is reported as status "error"
    # and the patch continues.
    "edit_error_policy": "report",
    # SkillOpt's own status names (skill.py _apply_edit_with_report).
    "status_names": {
        "applied_replace": "applied_replace",
        "applied_delete": "applied_delete",
        "applied_insert_after": "applied_insert_after",
        "applied_insert_before": "applied_insert_before",
        "applied_append": "applied_append",
        "applied_append_before_protected_region": "applied_append_before_protected_region",
        "applied_insert_after_fallback_append": "applied_insert_after_fallback_append",
        "applied_insert_after_fallback_before_protected_region":
            "applied_insert_after_fallback_before_protected_region",
        "skipped_protected_region": "skipped_protected_region",
        "skipped_unknown_kind": "skipped_unknown_op",
        "skipped_replace_missing_target": "skipped_replace_missing_target",
        "skipped_replace_target_not_found": "skipped_replace_target_not_found",
        "skipped_delete_missing_target": "skipped_delete_missing_target",
        "skipped_delete_target_not_found": "skipped_delete_target_not_found",
        "skipped": "skipped",
        "error": "error",
    },
    # skill.py:101-104, 111-115: blank lines around content placed before a region.
    "append_protected_separator": "\n\n",
    # skill.py:109: insert_after with an empty target falls back like a missing one.
    "insert_after_empty_target_policy": "append",
}


_VALID_ON_MISSING = ("skip", "append", "refuse")


def _resolve_on_missing(config_on_missing: str | dict, edit_kind: str) -> str:
    """Resolve the on_missing policy for a given edit kind."""
    if isinstance(config_on_missing, str):
        return config_on_missing
    if isinstance(config_on_missing, dict):
        return config_on_missing.get(edit_kind, "skip")
    return "skip"


def _patch_knob(state: State, edit: dict, name: str, knob: Any,
                entries: dict[str, Any]) -> State:
    """``set_knob`` with a ``patch``: apply its text edits to a str knob as one patch.

    The caller has already checked the name and the freeze, as for a ``value``.
    """
    from .knob import KnobError

    if knob.kind != "str":
        raise EditError(
            f"knob {name!r} has kind {knob.kind!r}; set_knob's 'patch' needs a str "
            "knob, so give the new 'value' instead",
            code="E-knob",
        )
    text = str(entries[name]["value"])
    original_text = text

    config = dict(edit.get("config") or {})
    match_mode = config.get("match", "exact")
    if match_mode != "exact":
        raise EditError(f"unsupported match mode {match_mode!r}; only 'exact' is supported", code="E-patch")
    on_missing = config.get("on_missing", "skip")
    # Validate on_missing
    if isinstance(on_missing, str):
        if on_missing not in _VALID_ON_MISSING:
            raise EditError(f"unsupported on_missing policy {on_missing!r}", code="E-patch")
    elif isinstance(on_missing, dict):
        for policy_val in on_missing.values():
            if policy_val not in _VALID_ON_MISSING:
                raise EditError(f"unsupported on_missing policy {policy_val!r}", code="E-patch")
    else:
        raise EditError("on_missing must be a string or dict", code="E-patch")

    honour_protected = config.get("honour_protected", True)
    protected_regions = list(config.get("protected_regions") or [])
    # Reject empty markers — text.find("", pos) always returns pos, causing infinite loops.
    for pair in protected_regions:
        if len(pair) != 2 or not pair[0] or not pair[1]:
            raise EditError(
                "protected_regions markers must be non-empty [start, end] pairs",
                code="E-patch",
            )
    strip_markers = config.get("strip_markers", True)
    insert_position = config.get("insert_position", "end_of_line")
    if insert_position not in ("end_of_line", "after_target"):
        raise EditError(f"unsupported insert_position {insert_position!r}", code="E-patch")

    # SkillOpt-fidelity parameters (brief fix 2).
    insert_after_sep: str = config.get("insert_after_separator", "\n")
    trailing_newline: bool = config.get("trailing_newline", True)
    append_strip_trailing: bool = config.get("append_strip_trailing", False)
    content_strip: bool = config.get("content_strip", False)
    unknown_kind_policy: str = config.get("unknown_kind_policy", "refuse")
    status_names: dict[str, str] = config.get("status_names") or {}
    append_protected_sep: str = config.get("append_protected_separator", "\n\n")
    insert_after_empty_target_policy: str = config.get(
        "insert_after_empty_target_policy", "skip",
    )
    protected_check: str = config.get("protected_check", "overlap")
    if protected_check not in ("overlap", "target_start"):
        raise EditError(f"unsupported protected_check {protected_check!r}", code="E-patch")
    insert_after_eof: str = config.get("insert_after_eof", "separator")
    if insert_after_eof not in ("separator", "line"):
        raise EditError(f"unsupported insert_after_eof {insert_after_eof!r}", code="E-patch")
    edit_error_policy: str = config.get("edit_error_policy", "raise")
    if edit_error_policy not in ("raise", "report"):
        raise EditError(f"unsupported edit_error_policy {edit_error_policy!r}", code="E-patch")

    # Collect all marker strings for stripping
    all_markers: list[str] = []
    for pair in protected_regions:
        all_markers.extend(pair)

    edits_list = edit.get("patch")
    if not isinstance(edits_list, list):
        raise EditError("set_knob's 'patch' must be a list of text edits", code="E-patch")

    report: list[dict[str, Any]] = []

    _KNOWN_KINDS = {"replace", "delete", "insert_after", "insert_before", "append"}

    def _strip_content_markers(content: str) -> str:
        if not strip_markers or not all_markers:
            return content
        for marker in all_markers:
            content = content.replace(marker, "")
        return content

    def _find_protected_spans() -> list[tuple[int, int]]:
        """Find all protected spans in the current text."""
        spans: list[tuple[int, int]] = []
        if not honour_protected or not protected_regions:
            return spans
        for start_marker, end_marker in protected_regions:
            pos = 0
            while True:
                start_idx = text.find(start_marker, pos)
                if start_idx < 0:
                    break
                end_idx = text.find(end_marker, start_idx)
                if end_idx < 0:
                    break
                spans.append((start_idx, end_idx + len(end_marker)))
                pos = end_idx + len(end_marker)
        return spans

    def _overlaps_protected(edit_start: int, edit_end: int) -> bool:
        spans = _find_protected_spans()
        for prot_start, prot_end in spans:
            if not (edit_end <= prot_start or edit_start >= prot_end):
                return True
        return False

    def _missing_policy(kind: str) -> str:
        """The on_missing policy for *kind*; raises when it is 'refuse'."""
        policy = _resolve_on_missing(on_missing, kind)
        if policy == "refuse":
            raise EditError(
                f"target not found for {kind} edit and on_missing is 'refuse'",
                code="E-patch",
            )
        return policy

    def _status(canonical: str, fallback: str | None = None) -> str:
        """Map a canonical status to the configured name (SkillOpt compat).

        *fallback* is the canonical status used when *canonical* has no
        configured name, so finer-grained statuses keep their old names under
        the general defaults.
        """
        if canonical in status_names:
            return status_names[canonical]
        if fallback is not None:
            return status_names.get(fallback, fallback)
        return canonical

    def _target_start_protected(target: str) -> bool:
        """SkillOpt's test (skill.py:43-58): first occurrence of *target* starts
        inside [first start marker, end of first end marker) of any region."""
        target_idx = text.find(target)
        if target_idx == -1:
            return False
        for start_marker, end_marker in protected_regions:
            start_idx = text.find(start_marker)
            end_idx = text.find(end_marker)
            if start_idx == -1 or end_idx == -1:
                continue
            if start_idx <= target_idx < end_idx + len(end_marker):
                return True
        return False

    def _do_append(content: str) -> bool:
        """Append content, respecting protected regions.

        Returns True when the content went before a protected region.
        """
        nonlocal text
        if honour_protected and protected_regions:
            # Find earliest protected start marker position
            earliest = len(text)
            for start_marker, _end_marker in protected_regions:
                idx = text.find(start_marker)
                if idx >= 0 and idx < earliest:
                    earliest = idx
            if earliest < len(text):
                text = (text[:earliest].rstrip() + append_protected_sep + content
                        + append_protected_sep + text[earliest:])
                return True
        if append_strip_trailing:
            text = text.rstrip()
        if trailing_newline:
            text = text + "\n\n" + content + "\n"
        else:
            text = text + "\n\n" + content
        return False

    def _fallback_append(kind: str, content: str) -> str:
        """A missing target's append fallback; returns the status."""
        if _do_append(content):
            return _status(f"applied_{kind}_fallback_before_protected_region", "applied_append")
        return _status(f"applied_{kind}_fallback_append", "applied_append")

    def _one(i: int, edit_item: Any) -> dict[str, Any]:
        nonlocal text
        kind = edit_item.get("kind")
        target = edit_item.get("target", "")
        content = edit_item.get("content", "")
        skillopt_order = protected_check == "target_start"
        if not skillopt_order and kind not in _KNOWN_KINDS:
            if unknown_kind_policy == "skip":
                return {
                    "index": i, "kind": kind, "target": "", "content_preview": "",
                    "status": _status("skipped_unknown_kind"),
                }
            raise EditError(f"unknown edit kind {kind!r}", code="E-patch")

        # skill.py:78-79 strips whitespace first, then removes markers.
        if content_strip:
            content = content.strip()
        content = _strip_content_markers(content)
        outcome: dict[str, Any] = {
            "index": i,
            "kind": kind,
            # skill.py:89 slices the target as given (a non-str target is an error).
            "target": target[:200] if skillopt_order else (str(target)[:200] if target else ""),
            "content_preview": str(content)[:200],
            "status": "",
        }

        if skillopt_order:
            # skill.py:94-96 protected check precedes op dispatch.
            if honour_protected and target and _target_start_protected(target):
                outcome["status"] = _status("skipped_protected_region")
                return outcome
            if kind not in _KNOWN_KINDS:
                if unknown_kind_policy == "skip":
                    outcome["status"] = _status("skipped_unknown_kind")
                    return outcome
                raise EditError(f"unknown edit kind {kind!r}", code="E-patch")

        if kind == "append":
            if _do_append(content):
                outcome["status"] = _status(
                    "applied_append_before_protected_region", "applied_append",
                )
            else:
                outcome["status"] = _status("applied_append")
            return outcome

        # For kinds that need a target: empty-target guard
        if not target:
            # insert_after with an empty target can fall back to append
            if kind == "insert_after" and insert_after_empty_target_policy == "append":
                outcome["status"] = _fallback_append(kind, content)
                return outcome
            outcome["status"] = _status(f"skipped_{kind}_missing_target", "skipped")
            return outcome

        idx = text.find(target)

        if idx < 0:
            # Target not found — apply on_missing policy
            if _missing_policy(kind) == "append":
                outcome["status"] = _fallback_append(kind, content)
            else:
                outcome["status"] = _status(f"skipped_{kind}_target_not_found")
            return outcome

        # Check protected region
        if not skillopt_order and _overlaps_protected(idx, idx + len(target)):
            outcome["status"] = _status("skipped_protected_region")
            return outcome

        if kind == "replace":
            text = text[:idx] + content + text[idx + len(target):]
            outcome["status"] = _status("applied_replace")
        elif kind == "delete":
            text = text[:idx] + text[idx + len(target):]
            outcome["status"] = _status("applied_delete")
        elif kind == "insert_after":
            if insert_position == "after_target":
                text = text[:idx + len(target)] + content + text[idx + len(target):]
            else:
                # end_of_line: insert after the end of the line containing the target
                eol = text.find("\n", idx + len(target))
                if eol >= 0:
                    text = text[:eol] + insert_after_sep + content + text[eol:]
                elif insert_after_eof == "line":
                    # skill.py:120-122: no newline after the target's line.
                    text = text + "\n" + content + "\n"
                else:
                    text = text + insert_after_sep + content
            outcome["status"] = _status("applied_insert_after")
        elif kind == "insert_before":
            sol = text.rfind("\n", 0, idx)
            sol = sol + 1 if sol >= 0 else 0
            text = text[:sol] + content + "\n" + text[sol:]
            outcome["status"] = _status("applied_insert_before")
        return outcome

    for i, edit_item in enumerate(edits_list):
        if edit_error_policy == "report":
            # skill.py:173-184: one failing edit is reported and the patch goes on.
            try:
                report.append(_one(i, edit_item))
            except EditError:
                raise
            except Exception as exc:  # noqa: BLE001
                report.append({
                    "index": i, "kind": "", "target": "", "content_preview": "",
                    "status": _status("error"), "error": str(exc),
                })
        else:
            report.append(_one(i, edit_item))

    # Coerce
    try:
        value = knob.coerce(text)
    except KnobError as exc:
        raise EditError(f"knob {name!r}: {exc}", code="E-knob") from exc

    # Data-slot preservation check for template bindings (same as _set_knob).
    import re as _re_slots_tp
    graph = _graph(state)
    for _tp_node in graph.nodes.values():
        for _tp_field, _tp_knob in getattr(_tp_node, "field_bindings", ()):
            if _tp_knob == name and _tp_field in ("prompt_template", "system_prompt_template"):
                _tp_current_tpl = getattr(_tp_node, _tp_field, "")
                _tp_required = set(_re_slots_tp.findall(r'\{\{\s*([A-Za-z_]\w*)\s*\}\}', _tp_current_tpl))
                if not _tp_required:
                    continue
                _tp_present = set(_re_slots_tp.findall(r'\{\{\s*([A-Za-z_]\w*)\s*\}\}', str(value)))
                _tp_missing = _tp_required - _tp_present
                if _tp_missing:
                    raise EditError(
                        f"knob {name!r}: template slot(s) {sorted(_tp_missing)} required by "
                        f"node {_tp_node.id!r}'s {_tp_field} are missing from the new value",
                        code="E-knob",
                    )

    out = _finish(state, graph)
    out["knobs"][name]["value"] = value

    # Return the report alongside the state.  The caller (apply()) places it
    # in the _edit_reports collector when one is provided, rather than mutating
    # the edit dict — see brief fix 1.
    out["_patch_report"] = {"outcomes": report, "noop": original_text == text}

    return out


_primitive(
    "add_node", "node",
    "Add a node. 'position' is {after|before: id}, {start: true}, {terminal: true} "
    "or {detached: true} to add it without edges.",
    ("node", "position"), _add_node,
)
_primitive(
    "remove_node", "node",
    "Delete a node and every edge touching it, with the knobs that were its own "
    "(`<id>__...`, e.g. its system prompt). Leaves the hole for the validator to report.",
    ("id",), _remove_node,
)
_primitive(
    "add_edge", "edge",
    "Add an edge. A 'condition' makes it a gate verdict.",
    ("source", "target", "condition"), _add_edge,
)
_primitive(
    "remove_edge", "edge",
    "Remove the edge between two nodes, optionally only one condition.",
    ("source", "target", "condition"), _remove_edge,
)
_primitive(
    "set_knob", "knob",
    "Set a parameter: exactly one of 'value' (the whole new value) or 'patch' (an "
    "ordered list of text edits -- replace, delete, insert_after, insert_before, "
    "append -- applied to a str knob as one patch, configured by 'config'). Refused "
    "for a frozen knob or a value outside its bounds.",
    ("name", "value", "patch", "config"), _set_knob,
)
_primitive(
    "add_knob", "knob",
    "Declare a new parameter. A node must consume it or the validator refuses the state.",
    ("name", "spec"), _add_knob,
)
_primitive(
    "remove_knob", "knob",
    "Remove a parameter. Any node still consuming it becomes a validation failure.",
    ("name",), _remove_knob,
)


# ── macros ─────────────────────────────────────────────────────────────────
def _pre_insert_after(edit: dict, state: State) -> str | None:
    graph = _graph(state)
    target = graph.nodes.get(edit.get("target"))
    if target is None:
        return f"no node {edit.get('target')!r} to insert after"
    if target.kind == "gate" and graph.out_edges(target.id):
        return (
            f"{target.id!r} is a gate whose out-edges carry verdicts; 'after' does "
            "not say which verdict to splice onto"
        )
    try:
        _sole_out_edge(graph, target.id)
    except EditError as exc:
        return str(exc)
    return None


def _expand_insert_after(edit: dict, _state: State) -> list[dict]:
    return [{"op": "add_node", "node": edit["node"], "position": {"after": edit["target"]}}]


def _pre_insert_before(edit: dict, state: State) -> str | None:
    graph = _graph(state)
    if edit.get("target") not in graph.nodes:
        return f"no node {edit.get('target')!r} to insert before"
    return None


def _expand_insert_before(edit: dict, _state: State) -> list[dict]:
    return [{"op": "add_node", "node": edit["node"], "position": {"before": edit["target"]}}]


def _pre_wrap(edit: dict, state: State) -> str | None:
    graph = _graph(state)
    target = graph.nodes.get(edit.get("target"))
    if target is None:
        return f"no node {edit.get('target')!r} to wrap"
    out = graph.out_edges(target.id)
    if not out:
        return f"{target.id!r} has no out-edge, so wrapping it would enclose nothing"
    node = edit.get("node")
    if len(out) > 1 and (not isinstance(node, dict) or node.get("kind") != "gate"):
        return (
            f"{target.id!r} emits {len(out)} verdicts, so only a gate can take them "
            "over; the replacement node must declare kind 'gate'"
        )
    return None


def _expand_wrap(edit: dict, state: State) -> list[dict]:
    graph = _graph(state)
    target = edit["target"]
    node = dict(edit["node"])
    out = graph.out_edges(target)
    steps: list[dict] = [
        {"op": "add_node", "node": node, "position": {"detached": True}}
    ]
    for edge in out:
        steps.append({"op": "remove_edge", "source": edge.source, "target": edge.target,
                      "condition": edge.condition})
    # The replacement takes over the exact verdicts the target emitted, so it is
    # wired with the same conditions rather than a fresh label.
    for edge in out:
        steps.append({"op": "add_edge", "source": target, "target": node["id"],
                      "condition": edge.condition})
        steps.append({"op": "add_edge", "source": node["id"], "target": edge.target,
                      "condition": edge.condition})
    return steps


def _pre_bypass(edit: dict, state: State) -> str | None:
    graph = _graph(state)
    target = graph.nodes.get(edit.get("target"))
    if target is None:
        return f"no node {edit.get('target')!r} to bypass"
    if target.id == graph.start:
        return (
            f"{target.id!r} is the start node; bypassing it would leave the graph "
            "with no entry, so it cannot be removed this way"
        )
    incoming = graph.in_edges(target.id)
    outgoing = graph.out_edges(target.id)
    if not incoming:
        return f"{target.id!r} has no predecessor to reconnect from"
    if not outgoing:
        return f"{target.id!r} has no successor to reconnect to"
    overlapping = sorted(
        {e.source for e in incoming} & {e.target for e in outgoing}
    )
    if overlapping:
        return (
            f"{target.id!r} sits inside a loop: {overlapping} leads back to it, so "
            "reconnecting predecessors to successors would make that node feed "
            "itself"
        )
    return None


def _expand_bypass(edit: dict, state: State) -> list[dict]:
    graph = _graph(state)
    target = edit["target"]
    steps: list[dict] = []
    for incoming in graph.in_edges(target):
        for outgoing in graph.out_edges(target):
            if incoming.source == target or outgoing.target == target:
                continue
            steps.append({"op": "add_edge", "source": incoming.source,
                          "target": outgoing.target, "condition": incoming.condition})
    steps.append({"op": "remove_node", "id": target})
    return steps


def _pre_duplicate(edit: dict, state: State) -> str | None:
    graph = _graph(state)
    if edit.get("target") not in graph.nodes:
        return f"no node {edit.get('target')!r} to duplicate"
    return None


def _expand_duplicate(edit: dict, state: State) -> list[dict]:
    """A copy of the node, with knobs of its own for what the original binds.

    Each bound field gets a new knob (``<new id>__<suffix>``) declared with the
    original knob's spec and current value, so the copy starts out the same and is
    tuned on its own; it never shares the original's knob."""
    graph = _graph(state)
    source = graph.nodes[edit["target"]]
    node = source.to_state()
    new_id = str(edit.get("new_id") or _fresh_id(graph, f"{source.id}_copy"))
    node["id"] = new_id
    knobs = state.get("knobs") or {}
    steps: list[dict] = []
    bindings: list[list[str]] = []
    for field_name, knob in source.field_bindings:
        prefix = f"{source.id}__"
        own = (f"{new_id}__{knob[len(prefix):]}" if knob.startswith(prefix)
               else f"{new_id}__{field_name}")
        entry = knobs.get(knob) or {}
        spec = {**dict(entry.get("spec") or {}), "default": entry.get("value")}
        steps.append({"op": "add_knob", "name": own, "spec": spec})
        bindings.append([field_name, own])
    bound = {knob for _field, knob in source.field_bindings}
    node["knobs"] = [k for k in node.get("knobs", []) if k not in bound] + [own for _f, own in bindings]
    node.pop("field_bindings", None)
    if bindings:
        node["field_bindings"] = bindings
    return [*steps, {"op": "add_node", "node": node, "position": {"after": source.id}}]


def _pre_split(edit: dict, state: State) -> str | None:
    graph = _graph(state)
    if edit.get("target") not in graph.nodes:
        return f"no node {edit.get('target')!r} to split"
    into = edit.get("into")
    if not isinstance(into, list) or len(into) != 2:
        return "'into' must be exactly two node objects"
    return None


def _expand_split(edit: dict, state: State) -> list[dict]:
    first, second = edit["into"]
    return [
        {"op": "add_node", "node": first, "position": {"after": edit["target"]}},
        {"op": "add_node", "node": second, "position": {"after": first["id"]}},
    ]


def _pre_reorder(edit: dict, state: State) -> str | None:
    graph = _graph(state)
    first = graph.nodes.get(edit.get("first"))
    second = graph.nodes.get(edit.get("second"))
    if first is None or second is None:
        return f"reorder needs two existing nodes; got {edit.get('first')!r}, {edit.get('second')!r}"
    edge = graph.out_edges(first.id)
    if len(edge) != 1 or edge[0].target != second.id or edge[0].condition is not None:
        return f"{first.id!r} does not unconditionally lead only to {second!r}"
    return _dependency_violation(graph, first.id, second.id)


def _expand_reorder(edit: dict, state: State) -> list[dict]:
    graph = _graph(state)
    first, second = edit["first"], edit["second"]
    steps: list[dict] = [{"op": "remove_edge", "source": first, "target": second}]
    for edge in graph.out_edges(second):
        steps.append({"op": "remove_edge", "source": second, "target": edge.target,
                      "condition": edge.condition})
    steps.append({"op": "add_edge", "source": second, "target": first, "condition": None})
    for edge in graph.out_edges(second):
        steps.append({"op": "add_edge", "source": first, "target": edge.target,
                      "condition": edge.condition})
    return steps


def _pre_loop(edit: dict, state: State) -> str | None:
    graph = _graph(state)
    target = graph.nodes.get(edit.get("target"))
    if target is None:
        return f"no node {edit.get('target')!r} to loop over"
    if not graph.out_edges(target.id):
        return f"{target.id!r} has no out-edge, so there is nothing to loop back from"
    try:
        _sole_out_edge(graph, target.id)
    except EditError as exc:
        return str(exc)
    return None


def _expand_loop(edit: dict, state: State) -> list[dict]:
    graph = _graph(state)
    target = edit["target"]
    gate = dict(edit["gate"])
    successor = _sole_out_edge(graph, target)
    steps: list[dict] = [
        {"op": "add_node", "node": gate, "position": {"detached": True}}
    ]
    if successor is not None:
        steps.append({"op": "remove_edge", "source": successor.source,
                      "target": successor.target, "condition": successor.condition})
    steps.append({"op": "add_edge", "source": target, "target": gate["id"], "condition": None})
    steps.append({"op": "add_edge", "source": gate["id"], "target": target, "condition": "reloop"})
    if successor is not None:
        steps.append({"op": "add_edge", "source": gate["id"], "target": successor.target,
                      "condition": "proceed"})
    return steps


def _fresh_id(graph: Graph, base: str) -> str:
    if base not in graph.nodes:
        return base
    for n in range(2, 1000):
        candidate = f"{base}{n}"
        if candidate not in graph.nodes:
            return candidate
    raise EditError(f"cannot find a free id based on {base!r}")


_macro("insert_after", "Splice a new node onto the out-edge of an existing node.",
       ("target", "node"), _pre_insert_after, _expand_insert_after)
_macro("insert_before", "Splice a new node in front of an existing node.",
       ("target", "node"), _pre_insert_before, _expand_insert_before)
_macro("wrap", "Put a new node on every out-edge of an existing node, so it sees "
       "the same verdicts (an observer, a verifier, a retry wrapper).",
       ("target", "node"), _pre_wrap, _expand_wrap)
_macro("bypass", "Remove a node and reconnect each predecessor to each successor.",
       ("target",), _pre_bypass, _expand_bypass)
_macro("duplicate", "Insert a copy of a node directly after it; the copy gets its own "
       "knobs (same current values) for every field the original binds to a knob.",
       ("target", "new_id"), _pre_duplicate, _expand_duplicate)
_macro("split", "Replace one step with two consecutive steps.",
       ("target", "into"), _pre_split, _expand_split)
_macro("reorder", "Swap two adjacent steps. Refused when the first produces what "
       "the second reads.",
       ("first", "second"), _pre_reorder, _expand_reorder)
_macro("loop", "Wrap a step in a gate that can send control back to it, so the step "
       "gains a retry.",
       ("target", "gate"), _pre_loop, _expand_loop)


PRESETS["broad"] = tuple(sorted(list(_PRIMITIVES) + list(_MACROS)))
PRESETS["structure"] = PRESETS["broad"]

#: Operations that were folded into another one. They are not in the catalogue, so
#: an edit naming one is refused as unknown, with where its form went.
_REMOVED: dict[str, str] = {
    "text_patch": (
        "text_patch was folded into set_knob: write {\"op\": \"set_knob\", "
        "\"name\": <knob>, \"patch\": [<text edits>], \"config\": {...}} (the old "
        "'edits' list is set_knob's 'patch')"
    ),
}


# ── public surface ─────────────────────────────────────────────────────────
def _specs() -> dict[str, EditSpec]:
    return {**_PRIMITIVES, **_MACROS}


def names() -> list[str]:
    return sorted(_specs())


def unknown(op: object) -> str | None:
    """The refusal for an op the catalogue does not have, or None when it does.

    One home for the rule: an op that does not exist is a different fact from one
    that exists and is outside this run's permitted edits. The callers differ only
    in what they do with the answer -- reject and continue, or reject and return --
    so neither spells the predicate or the message a second time.
    """
    if op in _specs():
        return None
    return f"E-op: {_unknown_message(op)}"


def _unknown_message(op: object) -> str:
    removed = _REMOVED.get(op) if isinstance(op, str) else None
    if removed is not None:
        return f"unknown edit {op!r}: {removed}"
    return f"unknown edit {op!r}; known: {names()}"


def removed(op: object) -> str | None:
    """Where a folded-away operation went, or None for any other name."""
    return _REMOVED.get(op) if isinstance(op, str) else None


def enabled(preset: str = "broad") -> list[str]:
    """The operations a preset allows. An unknown preset is an error."""
    if preset not in PRESETS:
        raise EditError(f"unknown edit preset {preset!r}; known: {sorted(PRESETS)}", code="E-preset")
    chosen = PRESETS[preset]
    return sorted(chosen if chosen else _specs())


def catalogue(preset: str = "broad") -> list[dict[str, Any]]:
    """The enabled operations, described for an optimizer's prompt."""
    specs = _specs()
    return [
        {
            "op": name,
            "kind": specs[name].kind,
            "macro": specs[name].macro,
            "fields": list(specs[name].fields),
            "description": specs[name].description,
        }
        for name in enabled(preset)
    ]


def is_macro(name: str) -> bool:
    return name in _MACROS


def precheck(edit: dict, state: Mapping[str, Any]) -> str | None:
    """The reason ``edit`` cannot be applied, for the checks that need no state.

    **Not a dry run.** This consults the edit's own shape and, for a macro, that
    macro's declared precondition; a primitive is checked no further, so a
    ``set_knob`` on an undeclared knob — and a ``remove_node`` of the start node —
    pass here and are refused by :func:`apply`, which raises :class:`EditError`
    carrying the code. :meth:`~refactory_dsh.optim.base.Optimizer.attempt` is the
    pattern that turns that into a refusal reason.

    The narrow scope is the point: a caller that wants a verdict without building
    a state it will throw away gets one cheaply, and the authority for whether an
    edit is applicable stays in one place.
    """
    if not isinstance(edit, dict):
        return f"an edit must be an object, got {type(edit).__name__}"
    name = edit.get("op")
    spec = _specs().get(name)
    if spec is None:
        return _unknown_message(name)
    if spec.macro and spec.precondition is not None:
        try:
            return spec.precondition(edit, dict(state))
        except EditError as exc:
            return str(exc)
    return None


def _walk_successors(graph: Graph, node: Node) -> list[str]:
    """The nodes a walk goes on to after ``node``: its out-edges' targets."""
    return [edge.target for edge in graph.out_edges(node.id)]


def _fork_branch_members(graph: Graph, fork: Node) -> set[str]:
    """Every node a fork's branches visit before its join (``Graph.fork_join``)."""
    barrier = graph.fork_join(fork.id)
    seen: set[str] = set()
    pending = list(fork.targets)
    while pending:
        current = pending.pop()
        if current == barrier or current in seen or current not in graph.nodes:
            continue
        seen.add(current)
        pending.extend(_walk_successors(graph, graph.nodes[current]))
        member = graph.nodes[current]
        if member.kind == "fork":
            pending.extend(member.targets)
        if member.kind == "data" and member.subgraph_entry:
            pending.append(member.subgraph_entry)
    return seen


def _completes_without(graph: Graph, start: str, banned: str, end: str | None,
                       _passes: frozenset[str] = frozenset()) -> bool:
    """Whether a run from ``start`` can finish without visiting ``banned``.

    Finishing is reaching ``end`` (a data pass's exit), or with ``end`` None, any node
    with no out-edge that is not a pass's exit. A fork runs every branch, so it is
    passed only when no branch visits ``banned``; a data node is passed only when its
    pass can finish without it; a gate may take any of its verdicts. A data node met
    again inside its own pass (an edge back into it) cannot be passed.
    """
    in_passes = _data_pass_nodes(graph) if end is None else set()
    seen: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in seen or current not in graph.nodes or current == banned:
            continue
        seen.add(current)
        node = graph.nodes[current]
        if end is not None and current == end:
            return True
        if node.kind == "fork" and banned in _fork_branch_members(graph, node):
            continue
        if node.kind == "data" and node.subgraph_entry and banned in graph.data_body(node.id) and (
                node.id in _passes or not _completes_without(
                    graph, node.subgraph_entry, banned, node.subgraph_exit, _passes | {node.id})):
            continue
        following = _walk_successors(graph, node)
        if not following:
            if end is None and current not in in_passes:
                return True
            continue
        stack.extend(following)
    return False


def _unavoidable(graph: Graph, node_id: str) -> bool:
    """Whether every run that finishes visits ``node_id``.

    For a node inside a data pass, every finished pass of that data node visits it.
    """
    if node_id not in graph.nodes or graph.start not in graph.nodes:
        return False
    for data in graph.nodes.values():
        if data.kind == "data" and node_id in graph.data_body(data.id):
            return not _completes_without(graph, data.subgraph_entry, node_id, data.subgraph_exit)
    return not _completes_without(graph, graph.start, node_id, None)


def protected_violation(
    before: Mapping[str, Any], after: Mapping[str, Any], protected: Iterable[str],
) -> str | None:
    """Why ``after`` removes, rewires or replaces a protected node of ``before``, or None.

    A protected node (``Mode.protected_nodes``) keeps its declaration, its out-edges and
    the values of the knobs it reads, and a node every finished run visited is still
    visited by every finished run. New nodes may be inserted in front of it: its
    in-edges may change, as long as it is still reached as before.
    """
    protected = sorted(set(protected or ()))
    if not protected:
        return None
    old_graph, new_graph = _graph(before), _graph(after)
    old_knobs, new_knobs = before.get("knobs") or {}, after.get("knobs") or {}
    for node_id in protected:
        old = old_graph.nodes.get(node_id)
        if old is None:
            continue
        new = new_graph.nodes.get(node_id)
        if new is None:
            return f"node {node_id!r} is protected, and the edit removes it"
        if new.to_state() != old.to_state():
            old_state, new_state = old.to_state(), new.to_state()
            changed = sorted(key for key in set(old_state) | set(new_state)
                             if old_state.get(key) != new_state.get(key))
            return (f"node {node_id!r} is protected, and the edit changes its declaration "
                    f"({', '.join(changed)})")
        if sorted(e.key() for e in old_graph.out_edges(node_id)) != sorted(
                e.key() for e in new_graph.out_edges(node_id)):
            return f"node {node_id!r} is protected, and the edit rewires its out-edges"
        for knob in old.knobs:
            if old_knobs.get(knob) != new_knobs.get(knob):
                return (f"node {node_id!r} is protected, and the edit changes the knob "
                        f"{knob!r} it reads")
        if node_id not in new_graph.reachable() or (
                _unavoidable(old_graph, node_id) and not _unavoidable(new_graph, node_id)):
            return (f"node {node_id!r} is protected, and after the edit a run can finish "
                    "without it")
    return None


def apply(
    edit: dict,
    state: Mapping[str, Any],
    *,
    edit_reports: list[dict[str, Any]] | None = None,
    protected: Iterable[str] = (),
) -> State:
    """Apply ``edit`` and return a new state, leaving ``state`` untouched.

    If *edit_reports* is given it collects per-edit reports (currently only
    ``set_knob`` with a ``patch`` produces one).  Reports are **not** mutated onto the edit
    dict.

    *protected* names nodes the edit may not remove, rewire or replace
    (``Mode.protected_nodes``); an edit that does is refused with ``E-protected``. It is
    checked on the whole edit, so a macro's intermediate steps may pass through states a
    finished edit could not leave.
    """
    protected = tuple(protected or ())
    if protected:
        out = apply(edit, state, edit_reports=edit_reports)
        violation = protected_violation(state, out, protected)
        if violation:
            raise EditError(violation, code="E-protected")
        return out
    if not isinstance(edit, dict):
        raise EditError(f"an edit must be an object, got {type(edit).__name__}", code="E-op")
    name = edit.get("op")
    spec = _specs().get(name)
    if spec is None:
        raise EditError(_unknown_message(name), code="E-op")
    if spec.macro:
        reason = precheck(edit, state)
        if reason:
            raise EditError(reason, code="E-precondition")
        assert spec.expand is not None
        out = dict(state)
        for step in spec.expand(edit, dict(state)):
            out = apply(step, out, edit_reports=edit_reports)
        return out
    assert spec.apply is not None
    out = spec.apply(dict(state), edit)
    # A patching set_knob stashes its report on the returned state dict under a
    # private key.  Pull it out so the state stays clean and the caller sees the
    # report in the collector, not on the edit dict.
    report = out.pop("_patch_report", None)
    if report is not None and edit_reports is not None:
        edit_reports.append(report)
    return out


#: What a person's `NAME=VALUE` says for a bool knob.
_BOOL_TEXT = {"1": True, "true": True, "yes": True, "on": True,
              "0": False, "false": False, "no": False, "off": False}


def _from_text(knob: Any, raw: str) -> Any:
    """A command line's text as the value this knob's kind wants.

    The editor's values are already typed (the optimizer proposes typed edits); a
    command line's are not, so `--knob target=7` has to become the integer 7 or every
    numeric knob is unsettable from the CLI. A value that will not parse is left for
    `coerce` to refuse, with the same message it would have given.
    """
    text = raw.strip()
    try:
        if knob.kind == "int":
            return int(text)
        if knob.kind == "float":
            return float(text)
        if knob.kind == "bool":
            if text.lower() in _BOOL_TEXT:
                return _BOOL_TEXT[text.lower()]
            return raw
        if knob.kind == "enum":
            for member in knob.enum:
                if str(member) == text or member == raw:
                    return member
            if all(isinstance(member, int) for member in knob.enum):
                return int(text)
            if all(isinstance(member, float) for member in knob.enum):
                return float(text)
    except ValueError:
        return raw
    return raw


def configure(state: Mapping[str, Any], name: str, value: Any) -> State:
    """Set a knob for the run being made, refusing a bad name or value but not a freeze.

    `set_knob` is the *optimizer's* edit, and a frozen knob is frozen against the
    search — that is why it refuses one. A person saying what this run's input is is
    not the search: `refactory ceo <mode> --knob source_root=…` is how a mode whose
    input is a frozen knob gets used at all, and routing it through the editor made
    every such mode unusable ("knob 'source_root' is frozen and cannot be moved").
    Everything else about the refusal is kept: the name must exist and the value must
    be one the knob accepts.
    """
    from .knob import Knob, KnobError

    entries = dict(state.get("knobs") or {})
    if not isinstance(name, str) or name not in entries:
        known = ", ".join(sorted(entries)) or "none"
        raise EditError(f"no knob {name!r}; declared knobs: {known}", code="E-knob")
    knob = Knob.from_spec(entries[name]["spec"])
    try:
        coerced = knob.coerce(_from_text(knob, value) if isinstance(value, str) else value)
    except KnobError as exc:
        raise EditError(f"knob {name!r}: {exc}", code="E-knob") from exc
    out = _finish(state, _graph(state))
    out["knobs"][name]["value"] = coerced
    return out


def apply_many(edits: Iterable[dict], state: Mapping[str, Any]) -> State:
    out = dict(state)
    for edit in edits:
        out = apply(edit, out)
    return out


# ── random sampling, for the random optimizer ──────────────────────────────
def random_edit(
    state: Mapping[str, Any],
    *,
    preset: str = "broad",
    insertable: Sequence[Any] = (),
    rng: random.Random | None = None,
) -> dict | None:
    """A random edit from the enabled set, or ``None`` if none can be built.

    Structural edits need something to insert, so a mode that wants them declares
    an insertable catalogue (``Node`` objects or their states). Without one, only
    the parameter edits are reachable, and this says so by returning ``None``
    rather than by inventing a node.
    """
    rng = rng or random
    graph = _graph(state)
    entries = dict(state.get("knobs") or {})
    templates = [
        node.to_state() if hasattr(node, "to_state") else dict(node) for node in insertable
    ]
    movable = [
        name
        for name, entry in entries.items()
        if not entry["spec"].get("frozen", False)
    ]
    node_ids = sorted(graph.nodes)
    choices = enabled(preset)

    for _ in range(40):
        op = rng.choice(choices)
        edit: dict | None
        if op == "set_knob":
            if not movable:
                continue
            name = rng.choice(movable)
            from .knob import Knob

            knob = Knob.from_spec(entries[name]["spec"])
            value = _random_value(knob, rng, entries[name]["value"])
            if value is None:
                continue
            edit = {"op": "set_knob", "name": name, "value": value}
        elif op in ("add_knob", "remove_knob"):
            continue  # parameter-set surgery needs a named knob; not sampled
        elif op in ("insert_after", "insert_before", "wrap"):
            if not templates or not node_ids:
                continue
            target = rng.choice(node_ids)
            node = dict(rng.choice(templates))
            node["id"] = _fresh_id(graph, str(node.get("id") or "step"))
            edit = {"op": op, "target": target, "node": node}
        elif op == "bypass":
            candidates = [n for n in node_ids if n != graph.start]
            if not candidates:
                continue
            edit = {"op": "bypass", "target": rng.choice(candidates)}
        elif op == "duplicate":
            if not node_ids:
                continue
            edit = {"op": "duplicate", "target": rng.choice(node_ids)}
        elif op == "split":
            if len(templates) < 2 or not node_ids:
                continue
            first, second = rng.sample(templates, 2)
            first = dict(first, id=_fresh_id(graph, str(first.get("id") or "step")))
            second = dict(second, id=_fresh_id(graph, str(second.get("id") or "step")))
            edit = {"op": "split", "target": rng.choice(node_ids), "into": [first, second]}
        elif op == "reorder":
            pairs = [(e.source, e.target) for e in graph.edges if e.condition is None]
            if not pairs:
                continue
            first, second = rng.choice(pairs)
            edit = {"op": "reorder", "first": first, "second": second}
        elif op == "loop":
            gates = [n for n in templates if n.get("kind") == "gate"]
            if not gates or not node_ids:
                continue
            gate = dict(rng.choice(gates))
            gate["id"] = _fresh_id(graph, str(gate.get("id") or "retry_gate"))
            edit = {"op": "loop", "target": rng.choice(node_ids), "gate": gate}
        elif op == "add_node":
            if not templates:
                continue
            node = dict(rng.choice(templates))
            node["id"] = _fresh_id(graph, str(node.get("id") or "step"))
            edit = {"op": "add_node", "node": node,
                    "position": {"after": rng.choice(node_ids)} if node_ids else {"start": True}}
        elif op == "remove_node":
            candidates = [n for n in node_ids if n != graph.start]
            if not candidates:
                continue
            edit = {"op": "remove_node", "id": rng.choice(candidates)}
        elif op in ("add_edge", "remove_edge"):
            if not graph.edges:
                continue
            edge = rng.choice(graph.edges)
            edit = {"op": op, "source": edge.source, "target": edge.target,
                    "condition": edge.condition}
        else:  # pragma: no cover - the catalogue is closed
            continue

        if precheck(edit, state) is None:
            return edit
    return None


def _random_value(knob: Any, rng: random.Random, current: Any = None) -> Any:
    """An acceptable value other than the one the knob currently holds.

    Every kind is movable, including an open ``str`` knob: a prompt a sampler
    cannot reach is a prompt the random optimizer silently cannot search.
    """
    return knob.sample(rng, knob.default if current is None else current)
