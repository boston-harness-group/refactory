from __future__ import annotations

from typing import Any, Mapping

from refactory_dsh.nn.graph import Graph, Node

from ._common import NEUTRAL_REVIEW, _Refused, _edits
from .state import _available, _consumed, _data_ends, _forks_naming, _fresh, _sole_out
from .build import _Steps, _add_chain, _carried_knobs, _copy_spec, _drop_orphans, _frozen_settings, _shared_knobs, _spec, _step_nodes
from .visibility import _seen
from .checks import _plain_step


def is_one_line_split(edit: Mapping[str, Any]) -> bool:
    """Whether a ``split`` edit is the one-line form (step specs, or one step), not the
    written-out one (a ``target`` and two node objects with ids)."""
    if "target" not in edit:
        return True
    into = edit.get("into")
    return isinstance(into, list) and any(
        isinstance(n, dict) and ("text" in n or "id" not in n) for n in into)


def _split_specs(node: Node, edit: dict, state: Mapping[str, Any]) -> list[dict]:
    into = edit.get("into")
    if into is None:
        return [_copy_spec(node, state), {"kind": "agent", "text": NEUTRAL_REVIEW}]
    if not isinstance(into, list) or not 1 <= len(into) <= 2:
        raise _Refused("'into' is one step (replace) or two steps (split)")
    return [_spec(raw) for raw in into]


def _verdict_read(graph: Graph, node: Node, spec: Mapping[str, Any]) -> None:
    """Refused when a new code step with no text of its own would write what a verdict gate
    reads from ``node`` (a judge's reply): its default code joins its inputs, which never
    starts with a verdict, so the gate would fail on every run."""
    from refactory_dsh import snippet

    if spec.get("kind") != "code" or spec.get("text"):
        return
    for gate in graph.nodes.values():
        if gate.kind == "gate" and gate.evaluator == snippet.VERDICT \
                and set(gate.reads) & set(node.writes):
            raise _Refused(f"{gate.id!r} reads its verdict from what {node.id!r} writes; a code "
                           "step taking that over needs its own 'text' (code whose output "
                           "starts with the verdict), or make it an agent")


def _pre_split_line(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    node = _plain_step(graph, edit.get("step", edit.get("target")), "split")
    _sole_out(graph, node.id)
    specs = _split_specs(node, edit, state)
    if _forks_naming(graph, node.id):
        # A new step would take a new id, which the fork's targets, the join's sources and
        # the role names its readers resolve by (V21) would all have to follow; an agent's
        # replacement in place is a rewire with 'text' (its refusal names it), which keeps it.
        raise _Refused(f"{node.id!r} is a fork's branch or a join's source; insert a step before "
                       "or after it instead")
    last = specs[-1]
    if last["kind"] == "agent" and len(node.writes) != 1:
        raise _Refused(f"{node.id!r} writes {list(node.writes) or 'no file'}; a new agent writes "
                       "one file (its reply), so the step taking its place should be code")
    ends = _data_ends(graph, node.id)
    if len(specs) == 1 and "exit" in ends and ("entry" in ends or node.id == graph.start):
        raise _Refused(f"{node.id!r} both starts and ends its pass; split it into two steps")
    for spec in specs:
        if spec.get("id") is not None and spec["id"] in graph.nodes:
            raise _Refused(f"node {spec['id']!r} already exists")
    last = specs[-1]
    _verdict_read(graph, node, last)
    carried = _carried_knobs(last["kind"], last.get("text"), last.get("system"), _shared_knobs(node))
    frozen = [k for k in _frozen_settings(graph, state, {node.id}) if k not in carried]
    if frozen:
        raise _Refused(f"{node.id!r} alone reads the frozen knob(s) {frozen}, which keep their "
                       "value and declaration; split another step, or insert one beside it")
    return None


def _expand_split_line(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    node = graph.nodes[edit.get("step", edit.get("target"))]
    specs = _split_specs(node, edit, state)
    there = set(_available(graph).get(node.id, set()))
    view = _consumed(node, state)
    taken: list[str] = []
    built: list[list[dict]] = []
    for index, spec in enumerate(specs):
        last = index == len(specs) - 1
        base = f"{node.id}_new" if len(specs) == 1 else f"{node.id}_{'ab'[index]}"
        node_id = spec.get("id") or _fresh(graph, base, taken)
        taken.append(node_id)
        if index == 1:
            made = built[0][-1]
            there = there | {("fact", n) for n in made.get("produces") or ()} | {
                ("file", n) for n in made.get("writes") or ()}
            view = view | {("fact", n) for n in made.get("produces") or ()}
        nodes = _step_nodes(graph, state, spec, node_id,
                            _seen(graph, state, there, view, spec.get("sees")),
                            writes=list(node.writes) if last else None,
                            produces=list(node.produces) if last else None,
                            settings_from=node if node.kind == "agent" else None,
                            knobs=_shared_knobs(node) if last else ())
        taken += [str(n["id"]) for n in nodes]
        built.append(nodes)
    if len(built) == 1:
        nodes = built[0]
        if "exit" in _data_ends(graph, node.id):
            build.add(_add_chain(nodes, {"after": node.id}))
            build.add([{"op": "add_edge", "source": e.source, "target": nodes[0]["id"],
                        "condition": e.condition} for e in graph.in_edges(node.id)])
        else:
            build.add(_add_chain(nodes, {"before": node.id}))
            build.add([{"op": "add_edge", "source": nodes[-1]["id"], "target": e.target,
                        "condition": e.condition} for e in graph.out_edges(node.id)])
        build.add([{"op": "remove_node", "id": node.id}])
        _drop_orphans(build)
        return build.steps
    first, second = built
    build.add(_add_chain(first, {"before": node.id}))
    build.add(_add_chain(second, {"after": node.id}))
    build.add([{"op": "remove_node", "id": node.id},
               {"op": "add_edge", "source": first[-1]["id"], "target": second[0]["id"],
                "condition": None}])
    _drop_orphans(build)
    return build.steps
