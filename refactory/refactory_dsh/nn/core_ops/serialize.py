from __future__ import annotations

from typing import Any, Mapping

from refactory_dsh.nn.graph import Graph, Node

from ._common import _Refused, _edits
from .state import _data_ends, _join_of, _require_step
from .build import _Steps, _renamed


def _pre_serialize(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    fork = _require_step(graph, edit.get("fork"), "fork")
    if fork.kind != "fork":
        raise _Refused(f"{fork.id!r} is a {fork.kind}, not a fork")
    _serialize_plan(graph, fork)
    return None


def _serialize_plan(graph: Graph, fork: Node) -> tuple[Node, list[list[tuple[str, Any]]]]:
    from refactory_dsh.nn.validate import fork_branches

    join = _join_of(graph, fork)
    if fork.id == graph.start:
        raise _Refused(f"fork {fork.id!r} is the start step; insert a step before it first")
    if _data_ends(graph, fork.id) or _data_ends(graph, join.id):
        raise _Refused(f"fork {fork.id!r} or its join starts or ends a data pass")
    ends = [[(e.source, e.condition) for e in graph.in_edges(join.id) if e.source in branch]
            for branch in fork_branches(graph, fork, join.id)]
    return join, ends


def _serialize_steps(state: Mapping[str, Any], fork_id: str) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    fork = graph.nodes[fork_id]
    join, ends = _serialize_plan(graph, fork)
    heads = list(fork.targets)
    onward = [(e.target, e.condition) for e in graph.out_edges(join.id)]
    last_ends = [source for source, _c in (ends[-1] if ends else [])]
    build.add(_renamed(build.state, fork.id, heads[:1], joins=False))
    build.add(_renamed(build.state, join.id, last_ends, forks=False))
    edges: list[dict] = []
    for edge in graph.in_edges(fork.id):
        targets = [(heads[0], None)] if heads else onward
        edges += [{"op": "add_edge", "source": edge.source, "target": target,
                   "condition": edge.condition} for target, _c in targets]
    for index, branch_ends in enumerate(ends):
        for source, condition in branch_ends:
            if index + 1 < len(heads):
                edges.append({"op": "add_edge", "source": source, "target": heads[index + 1],
                              "condition": condition})
            else:
                edges += [{"op": "add_edge", "source": source, "target": target,
                           "condition": condition if condition is not None else cond}
                          for target, cond in onward]
    build.add(edges)
    build.add([{"op": "remove_node", "id": fork.id}, {"op": "remove_node", "id": join.id}])
    return build.steps


def _expand_serialize(edit: dict, state: dict) -> list[dict]:
    return _serialize_steps(state, edit["fork"])
