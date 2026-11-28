from __future__ import annotations

from typing import Any, Mapping

from refactory_dsh.nn.graph import Graph, Node

from ._common import _Refused, _edits
from .state import _consumed, _outputs, _require_step, _task_knob

import sys as _sys


class _Proxy:
    """Late-bound lookup on the core_ops package for monkeypatch compat."""
    __slots__ = ()
    def __getattr__(self, name):
        return getattr(_sys.modules['refactory_dsh.nn.core_ops'], name)


_co = _Proxy()


def _plain_step(graph: Graph, node_id: Any, what: str) -> Node:
    node = _require_step(graph, node_id)
    if node.kind not in ("agent", "op"):
        raise _Refused(f"{node.id!r} is a {node.kind}; {what} works on a step (agent, op or code)")
    return node




def _slots_in_field(node: Node) -> set[str]:
    from refactory_dsh.nn.validate import template_names

    return template_names(str(node.prompt_template or ""))




def _losses(graph: Graph, state: Mapping[str, Any], gone: set[str]
            ) -> tuple[list[str], dict[str, set[str]], dict[str, set[str]]]:
    """What removing ``gone`` takes from the steps left: the steps that need something only
    ``gone`` made (each said); the prompt slots that can simply be taken out of a step's own
    prompt knob (a slot an operator put there, not one its declared prompt has); and the
    optional reads of ops and gates that name a file nothing makes any more."""
    made: set[tuple[str, str]] = set()
    for node_id in gone:
        made |= _outputs(graph.nodes[node_id])
    others: set[tuple[str, str]] = set()
    for node in graph.nodes.values():
        if node.id not in gone:
            others |= _outputs(node)
    lost = made - others
    hard: list[str] = []
    strip: dict[str, set[str]] = {}
    drop: dict[str, set[str]] = {}
    for node in sorted(graph.nodes.values(), key=lambda n: n.id):
        if node.id in gone or node.kind == "data":
            continue
        needs = _consumed(node, state)
        if node.kind in ("op", "gate"):
            # An op's optional read of a file nothing makes any more is dropped from its
            # reads (V6 refuses a read no path makes); its required reads cannot be.
            optional = {("file", name) for name in node.reads if name not in node.required_reads}
            if optional & lost:
                drop[node.id] = {name for _k, name in optional & lost}
            needs = needs - optional
        hit = needs & lost
        knob = _task_knob(node)
        own = (knob is not None and len(graph.knob_consumers().get(knob, ())) == 1)
        loose = {name for kind, name in hit if kind == "fact" and own
                 and name not in _slots_in_field(node) and name not in node.inputs
                 and name not in node.feedback_keys}
        if loose:
            strip[node.id] = loose
        rest = sorted(name for kind, name in hit if not (kind == "fact" and name in loose))
        if rest:
            hard.append(f"{node.id!r} needs {', '.join(rest)}")
    return hard, strip, drop




def _strip_steps(state: Mapping[str, Any], strip: Mapping[str, set[str]]) -> list[dict]:
    graph = _edits()._graph(state)
    steps = []
    for node_id, names in sorted(strip.items()):
        knob = _task_knob(graph.nodes[node_id])
        value = str(((state.get("knobs") or {}).get(knob) or {}).get("value") or "")
        from refactory_dsh.nn.validate import template_names

        keep = template_names(value) - set(names)
        new = _co._strip_unseen(value, keep | set(state.get("knobs") or {})).rstrip()
        if new != value:
            steps.append({"op": "set_knob", "name": knob, "value": new})
    return steps




def _owned(graph: Graph, step: str, state: Mapping[str, Any]) -> list[str]:
    """Helper steps that exist only for ``step``: the collect step an operator put in front
    of it, and the judge a ``loop`` or ``route`` put in front of its gate, when only
    ``step`` reads what they make."""
    out = []
    for helper in (f"{step}_inputs", f"{step}_judge"):
        node = graph.nodes.get(helper)
        if node is None or [e.target for e in graph.out_edges(helper)] != [step]:
            continue
        readers = {n.id for n in graph.nodes.values()
                   if _consumed(n, state) & _outputs(node)} - {helper}
        if readers <= {step}:
            out.append(helper)
    return out




def _inputs_helper(graph: Graph, step: str, state: Mapping[str, Any]) -> str | None:
    """The ``collect`` step an operator put in front of ``step`` for it alone
    (`_owned`), which the pipeline view folds into the step; ``None`` when it has none."""
    helper = f"{step}_inputs"
    return helper if helper in _owned(graph, step, state) else None




def _head(graph: Graph, step: str, state: Mapping[str, Any]) -> str:
    """Where a way into ``step`` enters: its own collect step when it has one (the view
    shows the two as one step), else the step itself. An edge to the step that skipped its
    collect would run it without the files it is handed."""
    return _inputs_helper(graph, step, state) or step




def _upstream(graph: Graph, node_id: str) -> set[str]:
    """Every node a walk can pass before ``node_id``: along edges, from a fork to its
    branch heads, from a join's sources to the join and from a data node to its pass's
    entry -- the control flow `loops.loop_edges` reads cycles from -- so an edge back to
    one of them closes a cycle (a fork branch's head included)."""
    def before(key: str) -> list[str]:
        node = graph.nodes.get(key)
        out = [e.source for e in graph.in_edges(key)]
        out += [n.id for n in graph.nodes.values()
                if (n.kind == "fork" and key in n.targets)
                or (n.kind == "data" and n.subgraph_entry == key)]
        if node is not None and node.kind == "join":
            out += list(node.sources)
        return out

    seen: set[str] = set()
    stack = before(node_id)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(before(current))
    return seen
