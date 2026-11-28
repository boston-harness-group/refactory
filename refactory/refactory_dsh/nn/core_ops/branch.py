from __future__ import annotations

from typing import Any, Mapping

from refactory_dsh.nn.graph import Graph, Node

from ._common import MAX_BRANCHES, NEUTRAL_STEP, _Refused, _edits
from .state import _available, _consumed, _fresh, _join_of, _outputs, _require_step, _task_inputs
from .build import _Steps, _add_detached, _chain, _copy_spec, _respec, _spec, _step_nodes
from .placement import _new_facts, _place_all
from .visibility import _seen


def _pre_branch(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    fork = _require_step(graph, edit.get("fork"), "fork")
    if fork.kind != "fork":
        raise _Refused(f"{fork.id!r} is a {fork.kind}, not a fork; 'parallelize' runs steps "
                       "beside a step that is not one")
    _join_of(graph, fork)
    if len(fork.targets) >= MAX_BRANCHES:
        raise _Refused(f"fork {fork.id!r} already has {len(fork.targets)} branches")
    if edit.get("optional") is not None and not isinstance(edit.get("optional"), bool):
        raise _Refused("'optional' is true or false")
    spec = _branch_spec(edit, graph, fork, state)
    if spec.get("id") is not None and spec["id"] in graph.nodes:
        raise _Refused(f"node {spec['id']!r} already exists")
    return None


def _branch_spec(edit: dict, graph: Graph, fork: Node, state: Mapping[str, Any]) -> dict:
    first = next((graph.nodes[t] for t in fork.targets if t in graph.nodes
                  and graph.nodes[t].kind == "agent"), None)
    default = _copy_spec(first, state) if first is not None else {"kind": "agent",
                                                                   "text": NEUTRAL_STEP}
    given = {k: v for k, v in edit.items() if k not in ("op", "fork", "optional")}
    copied = default if given.get("text") is None and given.get("kind") is None else {}
    return _spec({**copied, **given}, default_kind=default["kind"],
                 default_text=default["text"] if given.get("kind") in (None, default["kind"])
                 else NEUTRAL_STEP)


def _expand_branch(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    fork = graph.nodes[edit["fork"]]
    join = _join_of(graph, fork)
    preds = [e.source for e in graph.in_edges(fork.id)]
    there = set(_available(graph).get(fork.id, set()))
    view: set[tuple[str, str]] = set()
    for pred in preds:
        view |= _consumed(graph.nodes[pred], state) | _outputs(graph.nodes[pred])
    if not preds:
        view = _task_inputs(graph, state)
    spec = _branch_spec(edit, graph, fork, state)
    node_id = spec.get("id") or _fresh(graph, f"{fork.id}_branch")
    nodes = _step_nodes(graph, state, spec, node_id,
                        _seen(graph, state, there, view, spec.get("sees")))
    head, end = _chain(nodes)
    # Given at all, 'optional' is a setting the search can move: the branch's first step
    # binds it to a knob of its own (`Graph.optional_targets_of`), and what reads the
    # branch's output is guarded, since the knob may let the branch fail later.
    settable = edit.get("optional") is not None
    fork_state = fork.to_state()
    fork_state["targets"] = [*fork.targets, head]
    knob_steps: list[dict] = []
    set_steps: list[dict] = []
    if settable:
        knob = f"{head}__optional"
        from refactory_dsh.nn.knob import Knob

        knob_steps = [{"op": "add_knob", "name": knob, "spec": Knob(
            False, "bool", description=f"whether branch {head} may fail without failing the "
                                       "item").spec()}]
        set_steps = [{"op": "set_knob", "name": knob, "value": bool(edit["optional"])}]
        nodes[0] = {**nodes[0], "knobs": [*(nodes[0].get("knobs") or []), knob],
                    "field_bindings": [*(nodes[0].get("field_bindings") or []), ["optional", knob]]}
    join_state = join.to_state()
    join_state["sources"] = [*join.sources, end]
    build.add(knob_steps)
    build.add(_respec(build.state, fork.id, fork_state))
    build.add(_respec(build.state, join.id, join_state))
    build.add(_add_detached(nodes))
    build.add([{"op": "add_edge", "source": end, "target": join.id, "condition": None}])
    build.add(set_steps)
    _place_all(build, _new_facts(nodes), after=join.id, to=spec.get("to"), guarded=settable)
    return build.steps
