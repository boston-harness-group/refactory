from __future__ import annotations

from typing import Any, Mapping, Sequence

from refactory_dsh.nn.graph import Graph, Node

from ._common import MAX_BRANCHES, _Refused, _edits
from .state import _available, _consumed, _forks_naming, _fresh, _outputs, _require_step, _uses_tools
from .build import _Steps, _add_detached, _chain, _spec, _step_nodes
from .placement import _new_facts, _place_all
from .checks import _head
from .insert import _neighbour_view

import sys as _sys


class _Proxy:
    """Late-bound lookup on the core_ops package for monkeypatch compat."""
    __slots__ = ()
    def __getattr__(self, name):
        return getattr(_sys.modules['refactory_dsh.nn.core_ops'], name)


_co = _Proxy()


def _chain_of(graph: Graph, ids: Sequence[str], state: Mapping[str, Any] | None = None
              ) -> None:
    """Refused unless ``ids`` follow each other straight, in run order. With ``state``, a
    step's own collect step in front of it (`_head`, folded into the step in the view)
    counts as part of it."""
    for first, second in zip(ids, ids[1:]):
        head = _head(graph, second, state) if state is not None else second
        out = graph.out_edges(first)
        if len(out) != 1 or out[0].target != head or out[0].condition is not None:
            raise _Refused(f"{first!r} does not lead straight to {second!r}; give consecutive "
                           "steps, in run order")
        if len(graph.in_edges(head)) != 1:
            raise _Refused(f"{second!r} is reached from more than one step")




def _tools_beside(node: Node) -> bool:
    return _uses_tools(node) and node.workspace is None and not node.outputs_only




def _pre_parallelize(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    if edit.get("steps") is not None and edit.get("new") is not None:
        raise _Refused("parallelize takes 'steps' (existing steps, or one with 'copies') or "
                       "'after' with 'new' (new steps), not both")
    if edit.get("new") is not None:
        anchor = _require_step(graph, edit.get("after"))
        if anchor.kind in ("gate", "fork"):
            raise _Refused(f"{anchor.id!r} is a {anchor.kind}; put new steps after a step")
        _co._sole_out(graph, anchor.id)
        new = edit["new"]
        if not isinstance(new, list) or not 2 <= len(new) <= MAX_BRANCHES:
            raise _Refused(f"'new' is a list of 2 to {MAX_BRANCHES} steps")
        for raw in new:
            _spec(raw)
        return None
    ids = edit.get("steps")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise _Refused("'steps' is a list of step ids")
    nodes = [_co._plain_step(graph, i, "parallelize") for i in ids]
    for node in nodes:
        if _forks_naming(graph, node.id) and node.id not in (ids[0], ids[-1]):
            raise _Refused(f"{node.id!r} is named by a fork or join")
    copies = edit.get("copies")
    if len(ids) == 1:
        if not isinstance(copies, int) or isinstance(copies, bool) \
                or not 1 <= copies < MAX_BRANCHES:
            raise _Refused(f"one step runs beside 1 to {MAX_BRANCHES - 1} copies of itself: "
                           "give 'copies'")
        node = nodes[0]
        if node.kind != "agent":
            raise _Refused(f"{node.id!r} is not a model step; a copy of an op or of code would "
                           "compute the same thing again. Copy an agent")
        if edit.get("sees") is not None:
            raise _Refused("a copy is sent exactly what the step is sent, so it sees what the "
                           "step sees; for a new attempt with other inputs, give 'new' steps "
                           "('after' and 'new', each with its 'sees')")
        if _tools_beside(node):
            raise _Refused(f"{node.id!r} uses tools in the run directory, so a copy beside it "
                           "could see its half-done work; copy a step without tools, or one "
                           "that works in a workspace of its own")
    else:
        if copies is not None:
            raise _Refused("'copies' is for one step; list several steps without it")
        if len(ids) > MAX_BRANCHES or len(set(ids)) != len(ids):
            raise _Refused(f"'steps' names 2 to {MAX_BRANCHES} distinct steps")
        _chain_of(graph, ids)
        for node in nodes:
            if _tools_beside(node):
                raise _Refused(f"{node.id!r} uses tools in the run directory, so it cannot run "
                               "beside another step; give it a workspace of its own first")
        made: dict[tuple[str, str], str] = {}
        for node in nodes:
            for key in _outputs(node):
                if key in made:
                    raise _Refused(f"{made[key]!r} and {node.id!r} both make {key[1]!r}")
                made[key] = node.id
        for node in nodes:
            needs = sorted(name for key, name in _consumed(node, state)
                           if made.get((key, name)) not in (None, node.id))
            if needs:
                raise _Refused(f"{node.id!r} reads {needs}, which a step it would run beside "
                               "makes; they cannot run at once")
    if nodes[0].id == graph.start:
        raise _Refused(f"{nodes[0].id!r} is the start step; insert a step before it first")
    _co._sole_out(graph, ids[-1])
    return None




def _fork_join_ids(graph: Graph, base: str) -> tuple[str, str]:
    fork = _fresh(graph, f"{base}_fork")
    return fork, _fresh(graph, f"{base}_join", (fork,))




def _clone_agent(node: Node, state: Mapping[str, Any], copy_id: str
                 ) -> tuple[list[dict], dict[str, Any]]:
    """A copy of agent ``node`` as ``copy_id``: the same prompts, system prompt, role,
    limits and tools, so it is sent byte for byte what ``node`` is sent, with outputs of its
    own (``<copy_id>.txt``, ``<copy_id>``). A text bound to one of ``node``'s knobs is bound
    to a knob of the copy's own (`<copy_id>__...`) holding the same value, frozen when
    ``node``'s is -- also when ``node`` shares it with other steps, so tuning the copy never
    tunes them. A text ``node`` holds in its declaration stays so in the copy: the copy has
    no knob ``node`` lacks (its add_node is ``_exact``). Returns the primitive steps
    declaring those knobs, and the copy's node state."""
    knobs = state.get("knobs") or {}
    copy = node.to_state()
    copy["id"] = copy_id
    if node.writes:
        copy["writes"] = [f"{copy_id}.txt", *(f"{copy_id}_{name}" for name in node.writes[1:])]
    if node.produces:
        copy["produces"] = [copy_id, *(f"{copy_id}_{name}" for name in node.produces[1:])]
    steps: list[dict] = []
    renamed: dict[str, str] = {}
    suffixes = {"prompt_template": "task", "system_prompt_template": "system_prompt",
                "code": "code"}
    for field_name, knob in node.field_bindings:
        own = (knob[len(node.id) + 2:] if knob.startswith(f"{node.id}__")
               else suffixes.get(field_name, field_name))
        new_knob = f"{copy_id}__{own}"
        entry = knobs.get(knob) or {}
        # Declared at the original's value, so a frozen knob needs no set_knob.
        spec = {**dict(entry.get("spec") or {}), "default": entry.get("value")}
        steps.append({"op": "add_knob", "name": new_knob, "spec": spec})
        renamed[knob] = new_knob
    copy["knobs"] = [renamed.get(k, k) for k in node.knobs]
    copy["field_bindings"] = [[f, renamed.get(k, k)] for f, k in node.field_bindings]
    # Its tools as the original's, said outright: a new agent without a tool_policy is
    # given none (`edits._add_node`).
    copy["tool_policy"] = node.tool_policy
    if node.workspace is not None or _uses_tools(node):
        # A copy never works where the step it copies works (`coverage.DECISIONS`).
        copy["workspace"] = {"kind": "scratch"}
    copy.pop("notes", None)
    return steps, copy




def _expand_parallelize(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    avail = _available(graph)
    if edit.get("new") is not None:
        anchor = graph.nodes[edit["after"]]
        fork, join = _fork_join_ids(graph, anchor.id)
        there = avail.get(anchor.id, set()) | _outputs(anchor)
        view = _neighbour_view(graph, state, anchor)
        branches: list[list[dict]] = []
        tos: list[str | None] = []
        taken: list[str] = [fork, join]
        for index, raw in enumerate(edit["new"]):
            spec = _spec(raw)
            tos.append(spec.get("to"))
            node_id = spec.get("id") or _fresh(graph, f"{anchor.id}_branch{index + 1}", taken)
            taken.append(node_id)
            nodes = _step_nodes(graph, state, spec, node_id,
                                _co._seen(graph, state, there, view, spec.get("sees")),
                                settings_from=anchor if anchor.kind == "agent" else None)
            taken += [str(n["id"]) for n in nodes]
            branches.append(nodes)
        build.add([
            {"op": "add_node", "node": {"id": join, "kind": "join",
                                        "sources": [_chain(n)[1] for n in branches]},
             "position": {"after": anchor.id}},
            {"op": "add_node", "node": {"id": fork, "kind": "fork",
                                        "targets": [_chain(n)[0] for n in branches]},
             "position": {"after": anchor.id}},
        ])
        for nodes in branches:
            build.add(_add_detached(nodes))
            build.add([{"op": "add_edge", "source": nodes[-1]["id"], "target": join,
                        "condition": None}])
        for nodes, to in zip(branches, tos):
            _place_all(build, _new_facts(nodes), after=join, to=to)
        return build.steps
    ids = list(edit["steps"])
    first, last = graph.nodes[ids[0]], graph.nodes[ids[-1]]
    fork, join = _fork_join_ids(graph, first.id)
    copies: list[list[dict]] = []
    exact: set[str] = set()
    if len(ids) == 1:
        taken = [fork, join]
        there = set(avail.get(first.id, set()))
        for index in range(int(edit["copies"])):
            copy_id = _fresh(graph, f"{first.id}_copy{index + 1}", taken)
            taken.append(copy_id)
            knob_steps, node = _clone_agent(first, state, copy_id)
            build.add(knob_steps)
            copies.append([node])
            exact.add(copy_id)
    heads = [ids[0], *(_chain(n)[0] for n in copies)] if copies else list(ids)
    ends = [ids[-1], *(_chain(n)[1] for n in copies)] if copies else list(ids)
    # The new fork stands where the first step stood in an enclosing fork, and the new
    # join where the last step ended an enclosing branch.
    build.add(_co._renamed(build.state, first.id, [fork], joins=False))
    build.add(_co._renamed(build.state, last.id, [join], forks=False))
    build.add([
        {"op": "add_node", "node": {"id": join, "kind": "join", "sources": ends},
         "position": {"after": last.id}},
        {"op": "add_node", "node": {"id": fork, "kind": "fork", "targets": heads},
         "position": {"before": first.id}},
        {"op": "remove_edge", "source": fork, "target": first.id, "condition": None},
        *({"op": "remove_edge", "source": a, "target": b, "condition": None}
          for a, b in zip(ids, ids[1:])),
        *({"op": "add_edge", "source": a, "target": join, "condition": None} for a in ids[:-1]),
        {"op": "add_edge", "source": fork, "target": join, "condition": None},
    ])
    for nodes in copies:
        build.add(_add_detached(nodes, exact=exact))
        build.add([{"op": "add_edge", "source": nodes[-1]["id"], "target": join,
                    "condition": None}])
    _place_all(build, [f for nodes in copies for f in _new_facts(nodes)], after=join)
    return build.steps
