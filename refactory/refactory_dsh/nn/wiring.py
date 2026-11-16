"""What an edit built, read from the expanded graph.

A proposal says in prose what its edit does (``flow_change``); the edit's expansion can
differ (a loop's cap left at its default, a branch that sees the task taking in more than
the question). `as_built` states the candidate's own wiring against its parent's, from the
two graphs alone, never from the edit's text: each new step with what it takes in (and
which step makes each input, through any collect step the edit added), what it writes and
which steps read that; each new or changed gate with the step it sends back to and its
effective cap; each existing step whose inputs changed; each step removed. Any graph shape:
nested forks, gates, code steps and data passes are read the same way.
"""

from __future__ import annotations

from typing import Any, Mapping

from .graph import Graph
from .operators import made_by
from .validate import consumed

#: The most lines one candidate's wiring takes in a request; the rest are counted.
WIRING_LINES = 12


def _knob_values(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: (entry or {}).get("value") if isinstance(entry, Mapping) else entry
            for name, entry in (state.get("knobs") or {}).items()}


def _upstream(graph: Graph, node_id: str) -> set[str]:
    seen: set[str] = set()
    todo = [node_id]
    while todo:
        for edge in graph.in_edges(todo.pop()):
            if edge.source not in seen:
                seen.add(edge.source)
                todo.append(edge.source)
    return seen


def _kind(graph: Graph, node_id: str) -> str:
    from refactory_dsh import snippet

    node = graph.nodes[node_id]
    return "code" if node.kind == "op" and snippet.is_code_node(node) else node.kind


def as_built(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """The wiring ``after`` has that ``before`` did not, one line per new or changed
    step, gate or removal; empty when the graphs are the same or cannot be read."""
    try:
        old = Graph.from_state(dict(before.get("graph") or {}))
        new = Graph.from_state(dict(after.get("graph") or {}))
    except Exception:  # noqa: BLE001 - an unreadable graph is the validator's to report
        return []
    old_values, values = _knob_values(before), _knob_values(after)
    makers = made_by(new)
    added = [n for n in sorted(new.nodes) if n not in old.nodes]
    inputs_of = {n: set(consumed(new.nodes[n], values)) for n in new.nodes}
    # A collect or file_text op the edit added to hand a step its files: said as the way
    # in ("via ..."), not as a step of its own.
    from .operators import COLLECT_CALL, FILE_TEXT_CALL

    helpers = {n for n in added if new.nodes[n].kind == "op"
               and new.nodes[n].call in (COLLECT_CALL, FILE_TEXT_CALL)}

    def source(kind: str, name: str) -> list[str]:
        made = [m for m in makers.get((kind, name)) or [] if m not in helpers]
        via = [m for m in makers.get((kind, name)) or [] if m in helpers]
        if via:
            return [f"{inner} ({', '.join(makers.get((k, inner)) or []) or 'task input'}, "
                    f"via {', '.join(via)})"
                    for helper in via for k, inner in sorted(inputs_of[helper])]
        return [f"{name} ({', '.join(made) or 'task input'})"]

    def takes_in(node_id: str) -> str:
        parts = [part for kind, name in sorted(inputs_of[node_id],
                                               key=lambda key: (key[0] != "fact", key[1]))
                 for part in source(kind, name)]
        return ", ".join(dict.fromkeys(parts)) or "nothing"

    lines: list[str] = []
    for node_id in added:
        if node_id in helpers:
            continue
        node, kind = new.nodes[node_id], _kind(new, node_id)
        if kind == "gate":
            lines += _gate_lines(new, node_id, values, None)
            continue
        if kind in ("fork", "join"):
            targets = [e.target for e in new.out_edges(node_id)] if kind == "fork" else []
            lines.append(f"new {kind} {node_id}" + (f" runs {', '.join(targets)} at once"
                                                     if targets else ""))
            continue
        if not (inputs_of[node_id] or node.writes or node.produces):
            before_it = sorted(_before(new, node_id, helpers))
            lines.append(f"new no-op step {node_id}" + (f" after {', '.join(before_it)}"
                                                        if before_it else ""))
            continue
        forks = sorted(n for n in _before(new, node_id, helpers) if new.nodes[n].kind == "fork")
        place = f", a branch of fork {', '.join(forks)}" if forks else ""
        outputs = _outputs(new, node_id)
        readers = sorted({r for out in outputs for r in _readers(new, inputs_of, out)}
                         - {node_id})
        gates = sorted({e.target for e in new.out_edges(node_id)
                        if new.nodes[e.target].kind == "gate"})
        read_by = ", ".join(readers) or ("the gate " + ", ".join(gates) if gates
                                         else "no step")
        lines.append(
            f"new {kind} step {node_id}{place}: takes in {takes_in(node_id)}; writes "
            f"{', '.join(node.writes) or 'no file'}"
            + (f", makes fact {', '.join(node.produces)}" if node.produces else "")
            + f"; read by {read_by}")
    for node_id in sorted(set(new.nodes) & set(old.nodes)):
        node = new.nodes[node_id]
        if node.kind == "gate":
            old_cap = old.nodes[node_id].bound("max_iterations", old_values)
            if node.bound("max_iterations", values) != old_cap:
                lines += _gate_lines(new, node_id, values, old_cap)
            continue
        was = set(consumed(old.nodes[node_id], old_values))
        now = inputs_of[node_id]
        if was != now and node.kind in ("agent", "op"):
            gained = [part for key in sorted(now - was) for part in source(*key)]
            lost = sorted(name for _kind_, name in was - now)
            lines.append(f"step {node_id} now takes in {takes_in(node_id)}"
                         + (f" (new: {', '.join(gained)})" if gained else "")
                         + (f" (no longer: {', '.join(lost)})" if lost else ""))
    lines += [f"removed step {node_id}" for node_id in sorted(set(old.nodes) - set(new.nodes))]
    return lines


def _before(graph: Graph, node_id: str, skip: set[str]) -> set[str]:
    """The steps right before ``node_id`` (by an edge, or a fork naming it among its
    targets), looking through the ones in ``skip``."""
    sources = {edge.source for edge in graph.in_edges(node_id)} | {
        n for n, node in graph.nodes.items() if node.kind == "fork" and node_id in node.targets}
    out: set[str] = set()
    for source in sources:
        out |= _before(graph, source, skip) if source in skip else {source}
    return out


def _outputs(graph: Graph, node_id: str) -> list[tuple[str, str]]:
    node = graph.nodes[node_id]
    return [("fact", f) for f in node.produces] + [("file", f) for f in node.writes]


def _readers(graph: Graph, inputs_of: Mapping[str, set], output: tuple[str, str]) -> list[str]:
    return [n for n, taken in inputs_of.items() if output in taken]


def _gate_lines(graph: Graph, gate_id: str, values: Mapping[str, Any],
                old_cap: Any) -> list[str]:
    """A gate's send-back: the step it returns to, its effective cap (the bound knob's
    value, else its own field), how often that step runs at most, and what follows."""
    gate = graph.nodes[gate_id]
    above = _upstream(graph, gate_id)
    back = sorted(e.target for e in graph.out_edges(gate_id) if e.target in above)
    cap = gate.bound("max_iterations", values)
    knob = gate.bound_knob("max_iterations")
    judged = ", ".join(f"{name} (written by {', '.join(made_by(graph).get(('file', name)) or [])})"
                       for name in gate.reads) or "its own check"
    if not back:
        return [f"new gate {gate_id} on {judged}: chooses among "
                f"{', '.join(sorted({e.target for e in graph.out_edges(gate_id)}))}"]
    exhausted = graph.on_exhausted(gate_id) or "continue"
    lines = []
    for target in back:
        lines.append(
            f"{'new gate' if old_cap is None else 'gate'} {gate_id} on {judged}: sends work back "
            f"to {target} at most {cap} time(s)"
            + (f" (knob {knob} = {cap}" + (f", was {old_cap}" if old_cap is not None else "")
               + ")" if knob else "")
            + f", so {target} runs at most {cap + 1} times per item; then "
            + ("the run goes on with the last attempt" if exhausted == "continue"
               else exhausted))
    return lines


def clipped(lines: list[str], limit: int = WIRING_LINES) -> list[str]:
    """``lines`` cut to ``limit``, the rest counted."""
    if len(lines) <= limit:
        return list(lines)
    return [*lines[:limit], f"(and {len(lines) - limit} more wiring line(s))"]
