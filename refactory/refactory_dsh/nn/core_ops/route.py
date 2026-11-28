from __future__ import annotations

import re
from typing import Any, Mapping

from refactory_dsh.nn.graph import Graph, GraphError, Node

from ._common import MAX_ROUTES, NEUTRAL_REVIEW, NEUTRAL_ROUTE, _Refused, _edits
from .state import _available, _downstream, _fresh, _outputs, _sole_out
from .visibility import _not_on_every_path_hint, _readable, _seen
from .build import _Steps, _add_chain, _add_detached, _copy_spec, _spec, _step_nodes
from .placement import _new_facts, _place, _place_all, _readers_from
from .checks import _head, _plain_step, _upstream
from .insert import _neighbour_view, _runs_first_on_every_path, _unscored_take_over
from .loop import _forward_flow, _gate_state, _judge, _judge_spec, _reach
from .fallback import _at_rejoin, _fallback_parts


_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def _route_next(graph: Graph, step: str) -> str | None:
    """The step a route after ``step`` leads on to: its next step, or, for a step with a
    fallback, where its two ways rejoin."""
    if graph.falls_back(step):
        return _fallback_parts(graph, step)[1]
    return _sole_out(graph, step)


def _routes(edit: dict, graph: Graph, anchor: Node,
            state: Mapping[str, Any]) -> dict[str, Any]:
    """The route's ways: its 'to', or by default 'proceed' to the next step and 'revise',
    a new agent step that redoes an agent step's work -- taking it over in place
    (`_route_take_over`) where that reaches the result, else writing its own."""
    to = edit.get("to")
    if to is None:
        succ = _route_next(graph, anchor.id)
        if succ is None:
            raise _Refused(f"{anchor.id!r} has no next step to route to")
        # A copy of an agent step (its role, timeout, route and sealed tools), told to
        # review; a neutral agent step after anything else.
        revise = ({**_copy_spec(anchor, state), "text": NEUTRAL_REVIEW}
                  if anchor.kind == "agent" else {"kind": "agent", "text": NEUTRAL_REVIEW})
        if anchor.kind == "agent":
            try:
                _route_take_over(graph, state, anchor, _spec({**revise, "takes_over": anchor.id}))
                revise["takes_over"] = anchor.id
            except (_Refused, GraphError):
                pass  # refused here, or a graph whose cycle the edit's own checks refuse
        return {"proceed": succ, "revise": revise}
    if not isinstance(to, dict) or not 2 <= len(to) <= MAX_ROUTES:
        raise _Refused(f"'to' maps 2 to {MAX_ROUTES} one-word labels to a step id or a new step")
    return to


def _pre_route(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    anchor = _plain_step(graph, edit.get("after"), "route")
    if _sole_out(graph, anchor.id) is None:
        raise _Refused(f"{anchor.id!r} is the last step of its pass or run; route after a step "
                       "that has a next step")
    _judge(edit)
    # What runs before the route on a run's way forward: a way to one of those would send
    # work back. Retry edges are left out (`_forward_flow`), so a route inside a loop (a
    # step a gate sends back to on reloop) may still lead on to the steps after it.
    try:
        flow = _forward_flow(graph)
    except GraphError:
        # A cycle no gate reads: the state is refused by name further on (E-state).
        upstream = _upstream(graph, anchor.id) | {anchor.id}
    else:
        upstream = {key for key in graph.nodes if anchor.id in _reach(flow, key)}
    seen_labels: set[str] = set()
    for label, target in _routes(edit, graph, anchor, state).items():
        if not isinstance(label, str) or not _LABEL.fullmatch(label) or label.lower() == "halt":
            raise _Refused(f"route label {label!r} is not one word (and 'halt' ends the run)")
        if label.lower() in seen_labels:
            raise _Refused(f"route label {label!r} is given twice")
        seen_labels.add(label.lower())
        if isinstance(target, str):
            if target not in graph.nodes:
                raise _Refused(f"route {label!r} names no step {target!r}")
            if target in upstream:
                raise _Refused(f"route {label!r} leads back to {target!r}; use 'loop' to send "
                               "work back")
        else:
            spec = _spec(target)
            if spec.get("takes_over") is not None:
                _route_take_over(graph, state, anchor, spec)
    return None


def _route_take_over(graph: Graph, state: Mapping[str, Any], anchor: Node,
                     spec: Mapping[str, Any]) -> Node:
    """The step a route way's new step takes over: one every run passes before the route,
    whose outputs the new step writes again, in place, on its way only (the other ways
    keep the step's own; the ways rejoin, so a reader after them reads whichever ran).
    Refused as insert's 'takes_over' is when nothing after the route reads them."""
    target_id = spec.get("takes_over")
    if not isinstance(target_id, str) or target_id not in graph.nodes:
        raise _Refused(f"'takes_over' names an existing step, not {target_id!r}")
    target = graph.nodes[target_id]
    if target.kind not in ("agent", "op"):
        raise _Refused(f"{target_id!r} is a {target.kind}; only agent or op steps can be taken over")
    if not (target.writes or target.produces):
        raise _Refused(f"{target_id!r} has no outputs to take over")
    if target_id != anchor.id and not _runs_first_on_every_path(graph, target_id, [anchor.id]):
        hint = _not_on_every_path_hint(graph, target_id)
        raise _Refused(f"{target_id!r} does not run before the route on every path "
                       f"({hint}); a way's "
                       "new step takes over a step every run passes before it")
    if spec["kind"] == "agent" and not spec.get("tools") and len(target.writes) > 1:
        raise _Refused(f"{target_id!r} writes {len(target.writes)} files, and a new agent step "
                       "without tools writes only its reply; take it over with a code step")
    succ = _route_next(graph, anchor.id)
    unscored = _unscored_take_over(graph, state, target,
                                   _downstream(graph, succ) if succ is not None else set())
    if unscored:
        raise _Refused(f"nothing after the route reads {target_id!r}'s outputs ({unscored}), "
                       "so taking it over changes nothing")
    return target


def _expand_route(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    anchor = graph.nodes[edit["after"]]
    succ = _route_next(graph, anchor.id)
    assert succ is not None
    # After a step with a fallback the route goes where its two ways rejoin, so the
    # fallback still answers for the step (as insert's new step does, `_at_rejoin`).
    rejoin = graph.falls_back(anchor.id)
    judge = _judge(edit)
    routes = _routes(edit, graph, anchor, state)
    avail = _available(graph)
    there = avail.get(anchor.id, set()) | _outputs(anchor)
    view = _neighbour_view(graph, state, anchor)
    seen = _seen(graph, state, there, view, edit.get("sees"))
    gate_id = edit.get("id") or _fresh(graph, f"{anchor.id}_route")
    taken = [gate_id]
    is_sealed = bool(edit.get("sealed"))
    if judge == "agent":
        judge_id = _fresh(graph, f"{gate_id}_judge", taken)
        nodes = _step_nodes(graph, state, _judge_spec(edit, NEUTRAL_ROUTE),
                            judge_id, seen, verdict=list(routes),
                            settings_from=anchor if anchor.kind == "agent" else None)
        taken += [str(n["id"]) for n in nodes]
        build.add(_at_rejoin(build.graph, anchor.id, nodes) if rejoin
                  else _add_chain(nodes, {"after": anchor.id}))
        gate = _gate_state(gate_id, judge, "", seen, label=next(iter(routes)),
                           reads=[f"{judge_id}.txt"])
        build.add([{"op": "add_node", "node": gate, "position": {"after": judge_id}}])
    else:
        gate = _gate_state(gate_id, judge, edit.get("text") or "", seen,
                           label=next(iter(routes)), sealed=is_sealed)
        build.add(_at_rejoin(build.graph, anchor.id, [gate]) if rejoin else
                  [{"op": "add_node", "node": gate, "position": {"after": anchor.id}}])
    build.add([{"op": "remove_edge", "source": gate_id, "target": succ, "condition": None}])
    new_facts: list[str] = []
    for label, target in routes.items():
        if isinstance(target, str):
            # Into the step's own collect step, when it has one (`_head`).
            target = _head(build.graph, target, build.state)
            build.add([{"op": "add_edge", "source": gate_id, "target": target,
                        "condition": label}])
            continue
        spec = _spec(target)
        node_id = spec.get("id") or _fresh(graph, f"{anchor.id}_{label}", taken)
        taken.append(node_id)
        over = (_route_take_over(graph, state, anchor, spec)
                if spec.get("takes_over") is not None else None)
        way_seen = _seen(graph, state, there, view, spec.get("sees"))
        if over is not None and spec.get("sees") is None:
            # By default it sees what it rewrites, wherever before the route that was made.
            from refactory_dsh.workspace import is_session_log

            # What the step it redoes left, but never its session log (nothing reads it).
            redone = {key for key in _readable(graph, state, _outputs(over)) & there
                      if not (key[0] == "file" and is_session_log(key[1]))}
            way_seen = sorted(set(way_seen) | redone, key=lambda key: (key[0] != "fact", key[1]))
        nodes = _step_nodes(graph, state, spec, node_id, way_seen,
                            settings_from=anchor if anchor.kind == "agent" else None,
                            **({"writes": list(over.writes), "produces": list(over.produces)}
                               if over is not None else {}))
        taken += [str(n["id"]) for n in nodes]
        build.add(_add_detached(nodes))
        build.add([{"op": "add_edge", "source": gate_id, "target": nodes[0]["id"],
                    "condition": label},
                   {"op": "add_edge", "source": nodes[-1]["id"], "target": succ,
                    "condition": None}])
        if over is not None:
            pass  # it writes the taken step's names, which the steps after the rejoin read
        elif spec.get("to") is not None:
            # Where its 'to' says, as an inserted step's (`_place_all`).
            _place_all(build, _new_facts(nodes), after=node_id, to=spec["to"], guarded=True)
        else:
            new_facts += _new_facts(nodes)
    for fact in new_facts:
        build.add(_place(build.state, fact, readers=_readers_from(build.graph, succ, build.state),
                         guarded=True))
    return build.steps
