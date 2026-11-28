from __future__ import annotations

from typing import Any, Collection, Mapping, Sequence

from refactory_dsh.nn.graph import Graph, Node

from ._common import NEUTRAL_STEP, STEP_KINDS, _Refused, _default_wired, _edits, verdict_rule
from .state import _forks_naming, _fresh, _kind_of, _sealed, _text_of, _uses_tools
from .visibility import _with_sections

import sys as _sys


class _Proxy:
    """Late-bound lookup on the core_ops package for monkeypatch compat."""
    __slots__ = ()
    def __getattr__(self, name):
        return getattr(_sys.modules['refactory_dsh.nn.core_ops'], name)


_co = _Proxy()


def _spec(raw: Any, *, default_kind: str = "agent", default_text: str = NEUTRAL_STEP) -> dict:
    """A step spec ``{"kind", "text", ...}`` with its defaults; refused when malformed."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise _Refused("a new step is an object such as {\"kind\": \"agent\", \"text\": \"...\"}")
    kind = raw.get("kind") or default_kind
    if kind not in STEP_KINDS:
        raise _Refused(f"a new step's kind is one of {list(STEP_KINDS)}, not {kind!r}")
    if kind == "op" and not isinstance(raw.get("call"), str):
        raise _Refused("an op step names its callable: {\"kind\": \"op\", \"call\": \"module:function\"}")
    text = raw.get("text")
    if text is not None and not isinstance(text, str):
        raise _Refused("a new step's 'text' is a string (its prompt, or its code)")
    for key in ("id", "to", "system"):
        if raw.get(key) is not None and not isinstance(raw.get(key), str):
            raise _Refused(f"'{key}' is a string")
    if raw.get("system") is not None and kind != "agent":
        # Only an agent has a system prompt: on a code or op step it would be ignored.
        raise _Refused("only an agent step takes a 'system'")
    route = None
    if raw.get("route") is not None:
        if kind != "agent":
            raise _Refused("only an agent step takes a 'route'")
        from refactory_dsh.nn.graph import route_declaration

        try:
            route = route_declaration(raw["route"])
        except ValueError as exc:
            raise _Refused(str(exc)) from exc
    return {**raw, "kind": kind, "route": route, "text": text if text is not None else (
        default_text if kind == "agent" else "")}




def _default_code(facts: Sequence[str], files: Sequence[str] = ()) -> str:
    """Code that joins what the step was given into each fact and file it declares."""
    return ("def main(inp):\n"
            "    parts = [str(v) for v in inp['facts'].values()]\n"
            "    parts += [str(v) for v in inp['files'].values()]\n"
            "    text = '\\n\\n'.join(parts)\n"
            f"    return {{'facts': {{name: text for name in {list(facts)!r}}},\n"
            f"            'files': {{name: text for name in {list(files)!r}}}}}\n")




def _copy_spec(node: Node, state: Mapping[str, Any]) -> dict:
    """A step spec copying ``node``'s kind and text (an agent's tools come along)."""
    kind = _kind_of(node)
    if kind not in ("agent", "code"):
        return {"kind": "agent", "text": NEUTRAL_STEP}
    spec = {"kind": kind, "text": _text_of(node, state), "_like": node.id}
    if kind == "code":
        spec["_copied"] = spec["text"]
    if kind == "agent" and _sealed(node):
        # Its sealed tools, in the spec itself, so every check that reads a spec's tools
        # (a take-over of several files, a fallback's) sees the copy as the tool step it is.
        spec["tools"] = list(node.tools)
    return spec




def _output_renames(source: Node, facts: Sequence[str], files: Sequence[str]
                    ) -> dict[str, dict[str, str | None]] | None:
    """The ``renames`` (`graph.Node.renames`) of a code step that runs ``source``'s code
    but declares ``facts`` and ``files`` in place of ``source``'s outputs, in declaration
    order (an output past the new lists is dropped): what the code returns -- under
    ``source``'s names, or the names ``source``'s own renames map -- recorded under the new
    ones. The code is never changed, so it stays right however it is rewritten later."""
    out: dict[str, dict[str, str | None]] = {}
    for key, declared, new in (("facts", source.produces, facts),
                               ("files", source.writes, files)):
        before = dict((source.renames or {}).get(key) or {})
        returned_as = {name: returned for returned, name in before.items() if name is not None}
        table: dict[str, str | None] = {r: None for r, name in before.items() if name is None}
        for index, name in enumerate(declared):
            table[returned_as.get(name, name)] = new[index] if index < len(new) else None
        table = {returned: name for returned, name in table.items() if returned != name}
        if table:
            out[key] = table
    return out or None




def _step_nodes(graph: Graph, state: Mapping[str, Any], spec: dict, node_id: str,
                seen: Sequence[tuple[str, str]], *, writes: Sequence[str] | None = None,
                produces: Sequence[str] | None = None, settings_from: Node | None = None,
                verdict: Sequence[str] = (), knobs: Sequence[str] = ()) -> list[dict[str, Any]]:
    """The node states a new step takes, in run order: a ``collect`` step for the files it
    sees (agents only) and the step itself.

    ``writes``/``produces`` are its outputs when it takes over another step's; by default
    one file ``<id>.txt`` and one fact ``<id>``. ``settings_from`` is the agent whose role,
    timeout and route a new agent takes, unless the spec names a route of its own; a copy (``spec["_like"]``) of an agent that uses tools
    keeps them, in a fresh workspace of its own. ``verdict`` makes it a judge: its answer
    starts with one of the labels, and it writes its reply only, for the gate to read."""
    from refactory_dsh import snippet

    from refactory_dsh.nn.operators import COLLECT_CALL, collect_node

    facts = [name for kind, name in seen if kind == "fact"]
    files = [name for kind, name in seen if kind == "file"]
    kind = spec["kind"]
    like = graph.nodes.get(spec.get("_like") or "") if spec.get("_like") else None
    source = like or settings_from
    out: list[dict[str, Any]] = []
    if kind == "agent":
        text = _co._strip_unseen(spec["text"] or NEUTRAL_STEP,
                             set(facts) | set(state.get("knobs") or {}))
        tools = spec.get("tools")
        if not tools and like is not None and _sealed(like):
            # A copy of a sealed tool step is sealed as it is: its files in its own
            # workspace (as it reads them), never pasted in as text by a collect step.
            tools = list(like.tools)
        if files and not tools:
            gather = _fresh(graph, f"{node_id}_inputs", (node_id,))
            out.append(collect_node(gather, files, gather,
                                    notes=f"Hands {node_id} the files it sees."))
            facts = [*facts, gather]
        text = _with_sections(text, facts)
        if verdict:
            text = f"{text}\n\n{verdict_rule(verdict)}"
        node: dict[str, Any] = {
            "id": node_id, "kind": "agent", "prompt_template": text,
            "system_prompt_template": spec.get("system") or "You are one step of a pipeline.",
            "writes": [f"{node_id}.txt"] if verdict else list(
                writes or spec.get("writes") or [f"{node_id}.txt"]),
            "tool_policy": "none",
        }
        if not verdict:
            node["produces"] = (list(produces) if produces is not None
                                else list(spec.get("produces") or [node_id]))
        if source is not None and source.kind == "agent":
            if source.role:
                node["role"] = source.role
            if source.timeout_s:
                node["timeout_s"] = source.timeout_s
        route = spec.get("route") or (source.route if source is not None
                                      and source.kind == "agent" else None)
        if route:
            node["route"] = dict(route)
        if like is not None and like.kind == "agent" and like.keeps_tool_markup:
            # A copy takes a reply that writes out tool calls as the step it copies does
            # (`graph.Node.keeps_tool_markup`); a new step takes the platform's refusal.
            node["keeps_tool_markup"] = True
        if like is not None and _uses_tools(like):
            # A copy of a step that uses tools keeps them only in a fresh workspace of its
            # own, so it never works where the step it copies works (`coverage.DECISIONS`).
            node["tool_policy"] = like.tool_policy
            if like.tools:
                node["tools"] = list(like.tools)
            node["workspace"] = {"kind": "scratch"}
        if tools:
            # Its tools run sealed, in a fresh workspace holding only the files it reads
            # (`workspace.SEALED_TOOLS`): no other item, grader file or network.
            node["tool_policy"] = None
            node["tools"] = list(tools)
            node["workspace"] = {"kind": "scratch"}
            if files:
                node["reads"] = files
        if (tools or (like is not None and _uses_tools(like))) and spec.get("sees") is None \
                and not spec.get("reads"):
            # Wired by default, so given the item's input files by default too.
            _default_wired(node_id)
        carried = _co._carried_knobs("agent", text, node["system_prompt_template"], knobs)
        if carried:
            node["knobs"] = carried
        out.append(node)
        return out
    if kind == "code":
        made = (list(produces) if produces is not None
                else list(spec.get("produces") or [node_id]))
        if writes is None:
            writes = spec.get("writes")
        code = spec["text"] or _default_code(made, list(writes or ()))
        renames = None
        if like is not None and spec.get("_copied") is not None and code == spec["_copied"]:
            renames = _output_renames(like, made, list(writes or ()))
        node = {"id": node_id, "kind": "op", "call": snippet.RUN, "code": code,
                "reads": files, "inputs": facts, "produces": made}
        if renames:
            node["renames"] = renames
        if writes:
            node["writes"] = list(writes)
        if spec.get("sealed"):
            node["sealed_script"] = True
        out.append(node)
        return out
    call = spec["call"]
    if call == COLLECT_CALL:
        out.append(collect_node(node_id, list(spec.get("reads") or files),
                                list(produces)[0] if produces else node_id))
        return out
    node = {"id": node_id, "kind": "op", "call": call,
            "reads": list(spec.get("reads") or files),
            "inputs": list(spec.get("inputs") or facts)}
    for key, given in (("writes", writes or spec.get("writes") or [f"{node_id}.txt"]),
                       ("produces", produces if produces is not None
                        else spec.get("produces") or [node_id]),
                       ("required_reads", spec.get("required_reads"))):
        if given:
            node[key] = list(given)
    out.append(node)
    return out




def _shared_knobs(*nodes: Node) -> list[str]:
    """The knobs these steps read that are not their own (``<id>__...``): a mode's knobs,
    which the step taking their place reads on, so none is left read by nothing (V7)."""
    out: list[str] = []
    for node in nodes:
        bound = {knob for _f, knob in node.field_bindings}
        out += [k for k in node.knobs if not k.startswith(f"{node.id}__") and k not in bound
                and k not in out]
    return out




def _carried_knobs(kind: str, text: str | None, system: str | None,
                   shared: Sequence[str]) -> list[str]:
    """The shared knobs (:func:`_shared_knobs`) a step taking others' place reads on: an
    agent those its prompts name. A code step's code never sees the knobs (`snippet`), and
    a new op step is given none. The rest go with the steps they belonged to, and the
    candidate's notes say so (:func:`wiring_notes`)."""
    if kind != "agent":
        return []
    from refactory_dsh.nn.validate import template_names

    named = template_names(text or NEUTRAL_STEP) | template_names(system or "")
    return [k for k in shared if k in named]




def _orphaned_knobs(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """Knobs a step of ``before`` read that no step of ``after`` reads: the settings of the
    steps an operator removed or replaced, which go with them."""
    old, new = _edits()._graph(before), _edits()._graph(after)
    read_before = old.knob_consumers()
    read_after = new.knob_consumers()
    declared = set(after.get("knobs") or {})
    return sorted(k for k in read_before if k not in read_after and k in declared)




def _drop_orphans(build: "_Steps") -> None:
    build.add([{"op": "remove_knob", "name": k} for k in _orphaned_knobs(build.start, build.state)])




def _frozen_settings(graph: Graph, state: Mapping[str, Any], gone: set[str]) -> list[str]:
    """Frozen knobs that only steps in ``gone`` read, which removing them would drop."""
    readers = graph.knob_consumers()
    knobs = state.get("knobs") or {}
    return sorted(k for k, ids in readers.items() if set(ids) <= gone
                  and ((knobs.get(k) or {}).get("spec") or {}).get("frozen"))




def _chain(nodes: Sequence[dict[str, Any]]) -> tuple[str, str]:
    """The first and last id of a new step's nodes (its collect step, then itself)."""
    return str(nodes[0]["id"]), str(nodes[-1]["id"])




def _add_chain(nodes: Sequence[dict[str, Any]], position: dict[str, Any]) -> list[dict]:
    """Primitive steps adding a step's nodes at ``position`` in run order: the step at the
    position, then its collect step in front of it."""
    *before, step = nodes
    steps = [{"op": "add_node", "node": step, "position": position}]
    for node in reversed(before):
        steps.append({"op": "add_node", "node": node, "position": {"before": step["id"]}})
    return steps




def _add_detached(nodes: Sequence[dict[str, Any]], exact: Collection[str] = ()) -> list[dict]:
    steps = [{"op": "add_node", "node": node, "position": {"detached": True},
              **({"_exact": True} if node["id"] in exact else {})} for node in nodes]
    for first, second in zip(nodes, nodes[1:]):
        steps.append({"op": "add_edge", "source": first["id"], "target": second["id"],
                      "condition": None})
    return steps




def _respec(state: Mapping[str, Any], node_id: str, new_state: dict[str, Any],
            values: Mapping[str, Any] | None = None) -> list[dict]:
    """Primitive steps that give an existing node ``new_state`` under the same id, judged
    on ``state`` (the state they apply to): removed and added again, with its own knobs
    (those ``remove_node`` takes with it) declared again at their values, and every edge it
    has in ``state``. Used for a fork's targets, a join's sources and what a rewired step
    reads: topology, never a setting -- unless ``values`` gives one of those knobs a new
    value, which it is declared again at (a rewire's new prompt, holding its new slots)."""
    graph = _edits()._graph(state)
    node = graph.nodes[node_id]
    if new_state.get("kind") == "agent" and "tool_policy" not in new_state \
            and node.tool_policy is None:
        # An agent that runs with tools keeps them: a state leaves out a policy of None,
        # which `add_node` would read as a new agent's default, no tools.
        new_state = {**new_state, "tool_policy": None}
    knobs = state.get("knobs") or {}
    readers = graph.knob_consumers()
    steps: list[dict] = [{"op": "remove_node", "id": node_id}]
    restore: list[dict] = []
    for _field, knob in node.field_bindings:
        if knob.startswith(f"{node_id}__") and set(readers.get(knob, ())) <= {node_id} \
                and knob in knobs:
            entry = knobs[knob]
            steps.append({"op": "add_knob", "name": knob, "spec": dict(entry.get("spec") or {})})
            value = (values or {}).get(knob, entry.get("value"))
            if value != (entry.get("spec") or {}).get("default"):
                restore.append({"op": "set_knob", "name": knob, "value": value})
    position = {"start": True} if graph.start == node_id else {"detached": True}
    steps.append({"op": "add_node", "node": new_state, "position": position, "_respec": True})
    for edge in graph.edges:
        if node_id in (edge.source, edge.target):
            steps.append({"op": "add_edge", "source": edge.source, "target": edge.target,
                          "condition": edge.condition})
    return steps + restore




def _renamed(state: Mapping[str, Any], old: str, new: Sequence[str], *, joins: bool = True,
             forks: bool = True) -> list[dict]:
    """Respec steps, judged on ``state``, for every fork naming ``old`` as a branch and every
    join naming it as a source, so they name ``new`` instead (none: the name is dropped).
    A branch keeps its optional mark under its new name."""
    graph = _edits()._graph(state)
    steps: list[dict] = []
    current = dict(state)
    for node_id in sorted(n.id for n in _forks_naming(graph, old)):
        node = _edits()._graph(current).nodes[node_id]
        changed = node.to_state()
        if node.kind == "fork" and forks and old in node.targets:
            changed["targets"] = [t for name in node.targets
                                  for t in ([*new] if name == old else [name])]
            optional = [t for name in node.optional_targets
                        for t in ([*new] if name == old else [name])]
            if optional:
                changed["optional_targets"] = optional
            else:
                changed.pop("optional_targets", None)
        elif node.kind == "join" and joins and old in node.sources:
            changed["sources"] = [s for name in node.sources
                                  for s in ([*new] if name == old else [name])]
        else:
            continue
        more = _co._respec(current, node_id, changed)
        steps += more
        current = _simulate(current, more)
    return steps




def _simulate(state: Mapping[str, Any], steps: Sequence[dict]) -> dict[str, Any]:
    return _edits().apply_many(steps, state)




class _Steps:
    """Primitive steps built in order, each judged on the state the ones before it leave."""

    def __init__(self, state: Mapping[str, Any]) -> None:
        self.start = dict(state)
        self.state = dict(state)
        self.steps: list[dict] = []

    def add(self, steps: Sequence[dict]) -> None:
        if steps:
            self.state = _simulate(self.state, steps)
            self.steps += list(steps)

    @property
    def graph(self) -> Graph:
        return _edits()._graph(self.state)
