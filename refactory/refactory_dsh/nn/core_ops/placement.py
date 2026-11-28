from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from refactory_dsh.nn.graph import Graph, Node

from ._common import _edits
from .state import _available, _consumed, _downstream, _task_knob
from .visibility import _heading
from .build import _Steps, _respec


def _filled_text_knobs(node: Node, state: Mapping[str, Any],
                       graph: Graph | None = None) -> list[str]:
    """The text knobs an op that fills facts fills (`validate.filled_knobs`); with
    ``graph``, only those whose text still reaches a step (not `_dead_in`)."""
    from refactory_dsh.nn.validate import filled_knobs

    try:
        names = filled_knobs(node)
    except Exception:  # noqa: BLE001 - an unresolvable callable fills nothing known
        return []
    knobs = state.get("knobs") or {}
    dead = _dead_in(graph, node, state) if graph is not None else {}
    return [name for name in names
            if ((knobs.get(name) or {}).get("spec") or {}).get("kind") == "str"
            and name not in dead]


def _dead_in(graph: Graph, node: Node, state: Mapping[str, Any]) -> dict[str, list[str]]:
    """The text knobs ``node`` fills whose text reaches no step any more, each with the
    outputs that carry it (`operators.carried_knobs`), none of which any other step reads
    and none of which ends a data pass. A knob carried by an output still read (an op
    that does not declare ``knob_outputs`` carries each knob in every output) is not
    dead."""
    from refactory_dsh.nn.operators import carried_knobs
    from refactory_dsh.nn.validate import filled_knobs

    try:
        names = filled_knobs(node)
    except Exception:  # noqa: BLE001 - an unresolvable callable fills nothing known
        return {}
    if not names or any(d.kind == "data" and d.subgraph_exit == node.id
                        for d in graph.nodes.values()):
        return {}
    read = set().union(*(_consumed(n, state) for n in graph.nodes.values() if n.id != node.id))
    carried = carried_knobs(node)
    out: dict[str, list[str]] = {}
    for name in names:
        carriers = [key for key, knobs in carried.items() if name in knobs]
        if carriers and not any(key in read for key in carriers):
            out[name] = [output for _kind, output in carriers]
    return out


def dead_text_knobs(state: Mapping[str, Any]) -> dict[str, tuple[str, list[str]]]:
    """Every text knob an op fills whose text reaches no step any more: knob -> (an op
    filling it, the outputs carrying it that nothing reads), e.g. a prompt an op builds for
    a step that was rewired to see something else. A knob is dead only when every node
    reading it is an op or gate filling it into outputs nothing reads (`_dead_in`): one
    other op whose prompt is read, or any other reader, keeps it alive."""
    graph = _edits()._graph(state)
    knobs = state.get("knobs") or {}
    dead_by: dict[str, list[tuple[str, list[str]]]] = {}
    for node in graph.nodes.values():
        if node.kind in ("op", "gate"):
            for name, outputs in _dead_in(graph, node, state).items():
                dead_by.setdefault(name, []).append((node.id, outputs))
    readers = graph.knob_consumers()
    out: dict[str, tuple[str, list[str]]] = {}
    for name, found in dead_by.items():
        if ((knobs.get(name) or {}).get("spec") or {}).get("kind") != "str":
            continue
        if set(readers.get(name, ())) <= {op for op, _outputs in found}:  # noqa: F811
            out[name] = found[0]
    return out


def dead_knob_reason(name: str, state: Mapping[str, Any],
                     before: Mapping[str, Any] | None = None) -> str | None:
    """Why knob ``name`` changes no result in ``state`` (`dead_text_knobs`), naming the
    steps it reached in ``before`` when it did; None when it still reaches a step."""
    dead = dead_text_knobs(state).get(name)
    if dead is None:
        return None
    op, outputs = dead
    reached = ""
    if before is not None:
        try:
            old = _edits()._graph(before)
            readers = sorted(n.id for n in old.nodes.values() if n.id != op and any(
                ("fact", o) in _consumed(n, before) or ("file", o) in _consumed(n, before)
                for o in outputs))
        except Exception:  # noqa: BLE001 - only the wording
            readers = []
        if readers:
            reached = f" (it reached {', '.join(readers)} before)"
    return (f"knob {name!r} reaches no step: op {op!r} fills it into {', '.join(outputs)}, "
            f"which no step reads any more{reached}")


def _fills(node: Node, state: Mapping[str, Any], graph: Graph | None = None) -> bool:
    return node.kind in ("op", "gate") and bool(_filled_text_knobs(node, state, graph))


def _readers_from(graph: Graph, node_id: str, state: Mapping[str, Any]) -> list[str]:
    """The steps from ``node_id`` on whose prompts are filled with facts -- agents, and ops
    that fill facts into the knobs they read -- itself first when it is one, then the rest
    downstream, nearest first (through fork branches and verdicts, not back along a
    reloop)."""
    from refactory_dsh.nn.loops import loop_edges

    retries = {(source, target) for source, target, _c in loop_edges(graph)}
    order: list[str] = []
    seen, queue = {node_id}, [node_id]
    first = graph.nodes[node_id]
    if first.kind == "agent" or _fills(first, state, graph):
        order.append(node_id)
    while queue:
        current = queue.pop(0)
        node = graph.nodes.get(current)
        if node is None:
            continue
        following = [e.target for e in graph.out_edges(current)
                     if (current, e.target) not in retries]
        if node.kind == "fork":
            following = [*node.targets, *following]
        for target in following:
            if target in seen or target not in graph.nodes:
                continue
            seen.add(target)
            if graph.nodes[target].kind == "agent" or _fills(graph.nodes[target], state, graph):
                order.append(target)
            queue.append(target)
    return order


def _place(state: Mapping[str, Any], fact: str, *, readers: Sequence[str],
           to: str | None = None, guarded: bool = False) -> list[dict]:
    """The ``set_knob`` that puts ``{{fact}}`` where the next model step's prompt is filled:
    ``to`` (a step, an agent's prompt knob or a knob an op fills facts into), else the first
    of ``readers`` that is an agent whose prompt is a knob of its own, or an op filling
    facts into exactly one text knob (an op filling several is a choice for ``to``);
    guarded by ``{% if %}`` when the fact may not be there on every path. None when no such
    place is found and ``to`` was not given."""
    edits = _edits()
    graph = edits._graph(state)
    knob_to: str | None = None
    if to is not None and to not in graph.nodes:
        dead = dead_knob_reason(to, state)
        if dead is not None:
            raise edits.EditError(f"'to' {to!r}: {dead}, so the output would reach nothing; "
                                  "name a knob a step still reads", code="E-wiring")
    if to is not None:
        if to in graph.nodes:
            readers = [to]
        else:
            owners = [n.id for n in graph.nodes.values()
                      if _task_knob(n) == to or to in _filled_text_knobs(n, state)]
            if not owners:
                raise edits.EditError(f"'to' {to!r} is neither a step nor a prompt knob a step "
                                      "fills", code="E-wiring")
            readers, knob_to = owners[:1], to
    avail = _available(graph)
    for reader in readers:
        node = graph.nodes.get(reader)
        knob: str | None = None
        if node is not None and node.kind == "agent":
            knob = _task_knob(node)
            if knob is not None and len(graph.knob_consumers().get(knob, ())) > 1:
                knob = None
        elif node is not None:
            filled = _filled_text_knobs(node, state, graph)
            knob = knob_to if knob_to in filled else (filled[0] if len(filled) == 1 else None)
            if knob is None and to is None and filled:
                return []  # it fills several prompts: which one is the proposer's to say
        section = f"## {_heading(fact)}\n{{{{{fact}}}}}"
        if guarded or ("fact", fact) not in avail.get(reader, set()):
            section = f"{{% if {fact} %}}{section}{{% endif %}}"
        if knob is None:
            if to is not None and knob_to is None and node is not None \
                    and node.kind == "agent" and _task_knob(node) is None:
                # An agent 'to' names whose prompt is no knob (a template a step's fact
                # fills, say) is re-slotted in place, as `rewire` re-slots one: the same
                # step, with a section for the fact after its prompt.
                return _reslotted(state, node, fact, section)
            if to is not None:
                raise edits.EditError(f"step {reader!r} has no prompt knob of its own to put "
                                      f"{{{{{fact}}}}} in; name the knob with 'to'",
                                      code="E-wiring")
            continue
        current = str(((state.get("knobs") or {}).get(knob) or {}).get("value") or "")
        if re.search(r"\{\{\s*" + re.escape(fact) + r"\s*\}\}", current):
            return []
        return [{"op": "set_knob", "name": knob, "value": f"{current.rstrip()}\n\n{section}".lstrip()}]
    return []


def _reslotted(state: Mapping[str, Any], node: Node, fact: str, section: str) -> list[dict]:
    """The steps that give agent ``node``, whose prompt is no knob, ``section`` (the slot
    for ``fact``) after its prompt, in place (`_respec`): its id, outputs, settings and
    place in any fork, join, loop or pass stay; its prompt becomes a knob of its own
    (``<id>__task``) when it is added again (`edits._own_bindings`), as a rewired one's
    does. Nothing when the prompt already fills ``fact``."""
    text = str(node.prompt_template or "")
    if re.search(r"\{\{\s*" + re.escape(fact) + r"\s*\}\}", text):
        return []
    new_state = node.to_state()
    new_state["prompt_template"] = f"{text.rstrip()}\n\n{section}".lstrip()
    return _respec(state, node.id, new_state)


def _place_all(build: _Steps, facts: Sequence[str], *, after: str, to: str | None = None,
               guarded: bool = False) -> None:
    """`_place` each fact, readers from the first model step after ``after`` on. A 'to'
    whose reader never runs after ``after`` is refused: the output would never reach it."""
    if to is not None and facts:
        graph = build.graph
        owners = [to] if to in graph.nodes else [
            n.id for n in graph.nodes.values()
            if _task_knob(n) == to or to in _filled_text_knobs(n, build.state)]
        later = _downstream(graph, after, reloops=True) - {after}
        if owners and not set(owners) & later:
            raise _edits().EditError(
                f"'to' {to!r} is read by {owners[0]}, which runs before {after!r}, so the output "
                f"would never reach it; insert before {owners[0]} instead", code="E-wiring")
    for fact in facts:
        graph = build.graph
        readers = _readers_from(graph, after, build.state)
        if readers and readers[0] == after:
            readers = readers[1:]
        build.add(_place(build.state, fact, readers=readers, to=to, guarded=guarded))


def _new_facts(nodes: Sequence[dict[str, Any]]) -> list[str]:
    """The facts a new step makes for later steps (not its collect step's)."""
    return list(nodes[-1].get("produces") or ()) if nodes else []
