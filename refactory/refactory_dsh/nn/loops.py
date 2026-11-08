"""Derive retry transitions without assigning forward outcomes a retry cost."""
from __future__ import annotations

from typing import Any

from .graph import Graph, GraphError

#: The op a ``loop`` edit adds after its gate when the step it loops over was the last
#: of its data pass (or of the run): the gate leaves by its ``proceed`` edge to this
#: step, which ends the pass, since a pass may not end at a gate (V12).
LOOP_END_CALL = "refactory_dsh.nn.loops:loop_end"


def loop_end(ctx: Any) -> dict[str, Any]:
    """`LOOP_END_CALL`: does nothing; it is where a loop at the end of a pass leaves."""
    return {}


def loop_edges(graph: Graph) -> set[tuple[str, str, str]]:
    """Explicit reloops, plus single-gate cycles after explicit reloops are cut.

    Multiple gates in a residual cycle do not identify which decision retries the
    work. Refuse that ambiguity rather than choose from node names or DFS order.
    Fork/join links participate in control flow; Data passes keep their own scope.
    """
    gates = {key for key, node in graph.nodes.items() if node.kind == "gate"}
    loops = {edge.key() for edge in graph.edges
             if edge.source in gates and (edge.condition or "").strip().lower() == "reloop"}
    adjacent: dict[str, list[str]] = {key: [] for key in graph.nodes}
    for edge in graph.edges:
        if edge.key() not in loops and edge.source in adjacent and edge.target in adjacent:
            adjacent[edge.source].append(edge.target)
    for key, node in graph.nodes.items():
        if node.kind == "fork":
            adjacent[key].extend(target for target in node.targets if target in adjacent)
        elif node.kind == "join":
            for source in node.sources:
                if source in adjacent:
                    adjacent[source].append(key)
        elif node.kind == "data" and node.subgraph_entry in adjacent:
            adjacent[key].append(node.subgraph_entry)

    # A single-gate SCC can still contain an internal cycle that avoids its
    # gate entirely. Preserve the stronger boundedness rule before inference.
    visiting: set[str] = set()
    finished: set[str] = set()

    def check_guarded(key: str) -> None:
        if key in gates or key in finished:
            return
        if key in visiting:
            raise GraphError(
                f"cycle through {key!r} avoids every gate; route every cycle "
                "through a gate with a bounded retry transition"
            )
        visiting.add(key)
        for target in adjacent[key]:
            check_guarded(target)
        visiting.remove(key)
        finished.add(key)

    for key in sorted(adjacent):
        check_guarded(key)

    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()

    def visit(key: str) -> None:
        indices[key] = low[key] = len(indices)
        stack.append(key)
        active.add(key)
        for target in adjacent[key]:
            if target not in indices:
                visit(target)
                low[key] = min(low[key], low[target])
            elif target in active:
                low[key] = min(low[key], indices[target])
        if low[key] != indices[key]:
            return
        component: set[str] = set()
        while True:
            target = stack.pop()
            active.remove(target)
            component.add(target)
            if target == key:
                break
        if len(component) == 1 and key not in adjacent[key]:
            return
        owners = gates & component
        if len(owners) != 1:
            raise GraphError(
                f"cycle {sorted(component)} has {len(owners)} gates after explicit reloop "
                "edges are removed; label the intended retry transitions 'reloop' "
                "so their max_iterations budget is unambiguous"
            )
        owner = next(iter(owners))
        loops.update(edge.key() for edge in graph.out_edges(owner) if edge.target in component)

    for key in sorted(adjacent):
        if key not in indices:
            visit(key)
    return loops
