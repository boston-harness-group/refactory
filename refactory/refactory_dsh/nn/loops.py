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


def blind_reloops(graph: Graph, knob_values: Any = None) -> dict[str, dict[str, list[str]]]:
    """The reloops whose redone step cannot see what their gate read, as
    ``{gate: {condition: [files]}}``: the gate's read files (``reads`` and
    ``required_reads``) that the step it sends back to is not shown.

    Only a retry edge (`loop_edges`) to an agent counts (any other target has no prompt;
    `validate.wiring_warnings` names it). That agent sees the gate's inputs, and the
    reloop is wired, when any of these holds; the reloop is then left out:

    - it declares the file in ``reads`` or ``required_reads`` (it is handed the file);
    - it has ``feedback_keys`` (its facts are re-derived from the run's files before each
      dispatch, as design mode's steps are);
    - a name its prompts read (a ``{{name}}`` slot or an ``{% if %}`` condition) is a fact
      the gate sets, or one set by a node that reads or writes one of the gate's files
      (the review reaches it as a fact), or it reads a file such a node writes.

    Both runtimes hand only these files over (`run.with_unseen_reads`): the walker from
    the graph, the spine's node runner from the gate's ``blind_reloops`` declaration
    (`lowering.declarations`), so a wired loop's prompts are unchanged.
    """
    from .validate import TEMPLATE_FIELDS, template_names

    knob_values = dict(knob_values or {})

    def prompt_names(node: Any) -> set[str]:
        """Every name the node's prompts read, as sent (bound fields), conditions too."""
        names: set[str] = set()
        for field_name in TEMPLATE_FIELDS:
            try:
                text = node.bound(field_name, knob_values)
            except Exception:  # noqa: BLE001 - a bad binding is V14's to report
                text = getattr(node, field_name, "")
            names |= template_names(str(text or ""))
        return names

    out: dict[str, dict[str, list[str]]] = {}
    try:
        retries = sorted(loop_edges(graph))
    except GraphError:  # an ambiguous cycle is the validator's to refuse (V10)
        return out
    for gate_id, target_id, condition in retries:
        gate, target = graph.nodes.get(gate_id), graph.nodes.get(target_id)
        if gate is None or target is None or target.kind != "agent" or gate.gate_kind != "fn":
            continue
        files = list(dict.fromkeys((*gate.reads, *gate.required_reads)))
        unseen = [name for name in files
                  if name not in target.reads and name not in target.required_reads]
        if not unseen or target.feedback_keys:
            continue
        # What the gate's files become: the facts and files of the gate and of any node
        # that reads or writes one of them. The target sees the review when its prompt reads one
        # of those facts, or it reads one of those files.
        readers = [gate, *(node for node in graph.nodes.values()
                           if any(name in (*node.reads, *node.required_reads, *node.writes)
                                  for name in files))]
        facts = {fact for node in readers for fact in node.produces}
        written = {name for node in readers for name in node.writes}
        if prompt_names(target) & facts or written & {*target.reads, *target.required_reads}:
            continue
        out.setdefault(gate_id, {})[condition.strip().lower()] = unseen
    return out


def cycle_nodes(graph: Graph, target: str, gate: str) -> set[str]:
    """The nodes a retry from ``gate`` back to ``target`` runs again, the gate left out:
    every node reachable from ``target`` that can also reach ``gate`` along non-retry
    edges, fork targets and join sources (`RunRecord._uncomplete_between`'s rule)."""
    retries = loop_edges(graph)
    forward: dict[str, list[str]] = {}
    backward: dict[str, list[str]] = {}

    def link(source: str, to: str) -> None:
        forward.setdefault(source, []).append(to)
        backward.setdefault(to, []).append(source)

    for edge in graph.edges:
        if edge.key() not in retries:
            link(edge.source, edge.target)
    for node_id, node in graph.nodes.items():
        if node.kind == "fork":
            for child in node.targets:
                link(node_id, child)
        if node.kind == "join":
            for source in node.sources:
                link(source, node_id)

    def reach(start: str, following: dict[str, list[str]]) -> set[str]:
        seen, queue = {start}, [start]
        for current in queue:
            for other in following.get(current, ()):
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        return seen

    after, before = reach(target, forward), reach(gate, backward)
    return {node for node in after if node != gate and (node == target or node in before)}
