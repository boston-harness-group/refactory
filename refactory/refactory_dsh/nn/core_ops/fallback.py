from __future__ import annotations

from typing import Any, Mapping, Sequence

from refactory_dsh.nn.graph import Graph, Node

from ._common import TOOLS_HINTS, _Refused, _edits
from .state import _available, _consumed, _fresh, _sole_out
from .build import _Steps, _add_detached, _chain, _respec, _spec, _step_nodes
from .loop import _pass_of
from .split_op import _verdict_read
from .visibility import _seen
from .checks import _plain_step


def _fallback_spec(edit: dict, step: Node) -> dict:
    spec = _spec({k: v for k, v in edit.items() if k not in ("op", "step")})
    if spec["kind"] == "agent" and not spec.get("tools") and len(step.writes) > 1:
        raise _Refused(f"{step.id!r} writes {len(step.writes)} files, and a new agent step "
                       "without tools writes only its reply; give it a code step"
                       f"{TOOLS_HINTS[1]}")
    return spec


def _pre_fallback(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    node = _plain_step(graph, edit.get("step"), "fallback")
    if _pass_of(graph, node.id) is None:
        raise _Refused(f"{node.id!r} runs outside any per-item pass; a fallback answers for "
                       "a step that runs once per item")
    if graph.falls_back(node.id):
        raise _Refused(f"{node.id!r} has a fallback already, after "
                       f"{graph.fallback_gate(node.id)!r}")
    if not (node.writes or node.produces):
        raise _Refused(f"{node.id!r} has no outputs for a fallback to write")
    onward = _sole_out(graph, node.id)
    if onward is not None and graph.nodes[onward].kind == "join":
        raise _Refused(f"{node.id!r} ends a fork branch; insert a step after it first, or "
                       "give the fallback to a step inside the branch")
    spec = _fallback_spec(edit, node)
    _verdict_read(graph, node, spec)
    if spec.get("id") is not None and spec["id"] in graph.nodes:
        raise _Refused(f"node {spec['id']!r} already exists")
    return None


def _expand_fallback(edit: dict, state: dict) -> list[dict]:
    """A fallback gate after the step (`graph.FALLBACK_GATE_CALL`): ``proceed`` to what
    came next, ``fallback`` to the new step, which sees what the step sees, writes its
    outputs and leads to the same next step. A step that ends its pass first gets a no-op
    step after it (`LOOP_END_CALL`), as a loop's gate does, so the pass ends at a step."""
    from refactory_dsh.nn.graph import FALLBACK, FALLBACK_GATE_CALL, PROCEED
    from refactory_dsh.nn.loops import LOOP_END_CALL

    build = _Steps(state)
    graph = build.graph
    node = graph.nodes[edit["step"]]
    spec = _fallback_spec(edit, node)
    gate_id = _fresh(graph, f"{node.id}_failed")
    node_id = spec.get("id") or _fresh(graph, f"{node.id}_fallback", (gate_id,))
    there = _available(graph).get(node.id, set())
    seen = _seen(graph, state, there, _consumed(node, state), spec.get("sees"))
    nodes = _step_nodes(graph, state, spec, node_id, seen, writes=list(node.writes),
                        produces=list(node.produces),
                        settings_from=node if node.kind == "agent" else None)
    onward = _sole_out(graph, node.id)
    if onward is None:
        onward = _fresh(graph, f"{node.id}_done", (gate_id, *(str(n["id"]) for n in nodes)))
        build.add([{"op": "add_node", "position": {"after": node.id}, "node": {
            "id": onward, "kind": "op", "call": LOOP_END_CALL, "score_effect": "none",
            "notes": f"Ends what {node.id!r} ended, after its fallback gate {gate_id!r}."}}])
    build.add([{"op": "add_node", "position": {"detached": True}, "node": {
                   "id": gate_id, "kind": "gate", "evaluator": FALLBACK_GATE_CALL,
                   "notes": f"Sends the item to {node_id!r} when {node.id!r} failed."}},
               {"op": "remove_edge", "source": node.id, "target": onward, "condition": None},
               {"op": "add_edge", "source": node.id, "target": gate_id, "condition": None},
               {"op": "add_edge", "source": gate_id, "target": onward, "condition": PROCEED}])
    build.add(_add_detached(nodes))
    build.add([{"op": "add_edge", "source": gate_id, "target": nodes[0]["id"],
                "condition": FALLBACK},
               {"op": "add_edge", "source": nodes[-1]["id"], "target": onward,
                "condition": None}])
    return build.steps


def _fallback_parts(graph: Graph, step: str) -> tuple[str, str, set[str], list]:
    """A step with a fallback's gate, the step both ways rejoin at (the gate's ``proceed``
    target), the fallback's way (the steps from the gate's ``fallback`` edge up to the
    rejoin) and the edges that way rejoins by."""
    from refactory_dsh.nn.graph import FALLBACK, PROCEED

    gate = graph.fallback_gate(step)
    assert gate is not None, step
    out = {e.condition: e.target for e in graph.out_edges(gate)}
    onward = out[PROCEED]
    way: set[str] = set()
    pending = [out[FALLBACK]]
    rejoins = []
    while pending:
        current = pending.pop()
        if current == onward or current in way:
            continue
        way.add(current)
        for edge in graph.out_edges(current):
            if edge.target == onward:
                rejoins.append(edge)
            else:
                pending.append(edge.target)
    return gate, onward, way, rejoins


def _at_rejoin(graph: Graph, step: str, nodes: Sequence[dict[str, Any]]) -> list[dict]:
    """Primitive steps adding a new step's nodes where ``step``'s two ways (it answered, or
    its fallback did) rejoin: both lead to the new step, which leads where they led."""
    from refactory_dsh.nn.graph import PROCEED

    gate, onward, _way, rejoins = _fallback_parts(graph, step)
    head, end = _chain(nodes)
    steps = _add_detached(nodes)
    for source, condition in [(gate, PROCEED), *((e.source, e.condition) for e in rejoins)]:
        steps += [{"op": "remove_edge", "source": source, "target": onward,
                   "condition": condition},
                  {"op": "add_edge", "source": source, "target": head, "condition": condition}]
    steps.append({"op": "add_edge", "source": end, "target": onward, "condition": None})
    return steps


def _fallback_owner(graph: Graph, node_id: str) -> str | None:
    """The step with a fallback that ``node_id`` is, or whose fallback gate or fallback's
    way it is on; else None."""
    for step in sorted(graph.nodes):
        if graph.falls_back(step):
            gate, _onward, way, _rejoins = _fallback_parts(graph, step)
            if node_id in {step, gate, *way}:
                return step
    return None


def _unfallback(state: Mapping[str, Any], step: str) -> list[dict]:
    """Primitive steps taking ``step``'s fallback out, judged on ``state``: its gate and
    its way go, and ``step`` leads where they rejoined -- or, when the fallback added a
    no-op step to end the pass (`_expand_fallback`), ends the pass again."""
    from refactory_dsh.nn.loops import LOOP_END_CALL

    build = _Steps(state)
    graph = build.graph
    gate, onward, way, rejoins = _fallback_parts(graph, step)
    build.add([{"op": "remove_node", "id": node} for node in sorted(way)])
    build.add([{"op": "remove_node", "id": gate}])
    after = graph.nodes[onward]
    exits = [d.id for d in graph.nodes.values()
             if d.kind == "data" and d.subgraph_exit == onward]
    own_end = (after.kind == "op" and after.call == LOOP_END_CALL and exits
               and not graph.out_edges(onward)
               and {e.source for e in graph.in_edges(onward)} <= {gate, *(e.source for e in rejoins)})
    if own_end:
        build.add([{"op": "remove_node", "id": onward}])
        for data_id in exits:
            data = build.graph.nodes[data_id].to_state()
            build.add(_respec(build.state, data_id, {**data, "subgraph_exit": step}))
    else:
        build.add([{"op": "add_edge", "source": step, "target": onward, "condition": None}])
    return build.steps
