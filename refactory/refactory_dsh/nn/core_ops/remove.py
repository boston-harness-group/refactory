from __future__ import annotations

import json
from typing import Any, Mapping

from refactory_dsh.nn.graph import Node

from ._common import _Refused, _edits
from .state import _consumed, _data_ends, _outputs, _require_step
from .build import _Steps, _frozen_settings, _simulate
from .checks import _losses, _owned, _upstream
from .map_op import _map_removed, _pre_remove_map, _remove_map
from .fallback import _fallback_owner, _unfallback

import sys as _sys


class _Proxy:
    """Late-bound lookup on the core_ops package for monkeypatch compat."""
    __slots__ = ()
    def __getattr__(self, name):
        return getattr(_sys.modules['refactory_dsh.nn.core_ops'], name)


_co = _Proxy()


#: What a replacing hint adds when the step taking the place is code.
CODE_HINT_NOTE = (" (its default code joins what it is given; give 'text', your def main(inp), "
                  "for your own)")




def _replace_hint(node: Node, state: dict, sees: Any = None) -> str:
    """How to replace ``node`` by one step of the proposer's own text (an agent for an
    agent, code for anything else: a new agent writes only its reply): the exact edit when
    it applies to the graph, else why replacing it is refused too. An attempt holds the
    named edit to its own checks as well (`optim.base.checked_hint`)."""
    step: dict[str, Any] = ({"kind": "agent", "text": "<what it does>"} if node.kind == "agent"
                            else {"kind": "code"})
    if sees is not None:
        step["sees"] = sees
    edit = {"op": "split", "step": node.id, "into": [step]}
    refused = _edits().precheck(edit, state)
    if refused is None:
        try:
            _edits().apply(edit, state)
        except _edits().EditError as exc:
            refused = str(exc)
    if refused is not None:
        return ("replacing it is refused too ("
                f"{refused.removeprefix('E-precondition: ')})")
    return f"replace it with {json.dumps(edit)}" + ("" if node.kind == "agent" else CODE_HINT_NOTE)




def _pre_remove(edit: dict, state: dict) -> str | None:
    from refactory_dsh.nn.graph import FALLBACK

    graph = _edits()._graph(state)
    node = _require_step(graph, edit.get("step"))
    owner = _fallback_owner(graph, node.id)
    if owner is not None:
        # A fallback's gate or way goes as a whole (`_unfallback`); the step it answers for
        # goes as it would with no fallback.
        if node.kind == "gate" and edit.get("keep") == FALLBACK:
            raise _Refused(f"{node.id!r} is the fallback gate of {owner!r}; remove it (or "
                           "the fallback step) to take the fallback out, or remove "
                           f"{owner!r} itself")
        if node.id != owner:
            return None
        return _pre_remove(edit, _simulate(state, _unfallback(state, owner)))
    found = _map_removed(graph, state, node.id)
    if found is not None:
        if graph.falls_back(found["combine"]):
            # The combine's fallback goes first, as it does when the combine is removed.
            return _pre_remove(edit, _simulate(state, _unfallback(state, found["combine"])))
        _pre_remove_map(graph, state, found)
        return None
    if node.kind in ("fork", "join"):
        raise _Refused(f"{node.id!r} is a {node.kind}; 'serialize' the fork to run its branches "
                       "one after another, or remove a branch's step")
    if node.kind == "data":
        raise _Refused(f"{node.id!r} is a data node: the data pass is the grader's side")
    if node.id == graph.start:
        raise _Refused(f"{node.id!r} is the start step; insert a step before it, or "
                       f"{_replace_hint(node, state)}")
    ends = _data_ends(graph, node.id)
    if ends:
        raise _Refused(f"{node.id!r} is the {' and '.join(ends)} of a data pass, so it stays; "
                       f"{_replace_hint(node, state)}")
    if node.kind == "gate":
        labels = sorted({str(e.condition) for e in graph.out_edges(node.id) if e.condition})
        keep = edit.get("keep")
        if keep is None and len(graph.out_edges(node.id)) > 1:
            raise _Refused(f"{node.id!r} is a gate; say which verdict's path to keep with "
                           f"'keep' (one of {labels})")
        if keep is not None and keep not in labels:
            raise _Refused(f"gate {node.id!r} has no verdict {keep!r}; its verdicts: {labels}")
        target = next((e.target for e in graph.out_edges(node.id)
                       if keep is None or e.condition == keep), None)
        if target is not None and (target == node.id or target in _upstream(graph, node.id)):
            raise _Refused(f"verdict {keep!r} of {node.id!r} leads back to {target!r}; keep the "
                           "verdict that goes on")
        return None
    if len(graph.out_edges(node.id)) > 1:
        raise _Refused(f"{node.id!r} has several out-edges")
    overlapping = {e.source for e in graph.in_edges(node.id)} & {
        e.target for e in graph.out_edges(node.id)}
    if overlapping:
        raise _Refused(f"{node.id!r} sits inside a loop with {sorted(overlapping)}; remove the "
                       "loop's gate first (remove it with 'keep')")
    if any(e.source == node.id and graph.nodes[e.target].kind == "gate"
           and any(back.target == node.id for back in graph.out_edges(e.target))
           for e in graph.out_edges(node.id)) or any(
            graph.nodes[e.source].kind == "gate" and e.condition
            and e.source in {o.target for o in graph.out_edges(node.id)}
            for e in graph.in_edges(node.id)):
        raise _Refused(f"a gate sends work back to {node.id!r}; remove that gate first (remove "
                       "it with 'keep')")
    gone = {node.id, *_owned(graph, node.id, state)}
    frozen = _frozen_settings(graph, state, gone)
    if frozen:
        raise _Refused(f"{node.id!r} alone reads the frozen knob(s) {frozen}, which keep their "
                       "value and declaration; remove another step")
    hard, _strip, _drop = _losses(graph, state, gone)
    if hard:
        raise _Refused(f"removing {node.id!r} leaves steps without an input only it makes: "
                       f"{'; '.join(hard)}; remove or rewire those first")
    return None




def _bypass(state: Mapping[str, Any], node_id: str) -> list[dict]:
    """Primitive steps that take one plain step out, judged on ``state``: forks and joins
    that name it name its neighbours on the branch instead, then each predecessor leads to
    its successor (a branch end's predecessor to the join), then the step goes."""
    build = _Steps(state)
    graph = build.graph
    preds = graph.in_edges(node_id)
    succ = graph.out_edges(node_id)
    ends_join = [j.id for j in graph.nodes.values() if j.kind == "join" and node_id in j.sources]
    heads_in = [f.id for f in graph.nodes.values() if f.kind == "fork" and node_id in f.targets]
    for fork_id in heads_in:
        join = graph.fork_join(fork_id)
        heads = [e.target for e in succ if e.target != join]
        build.add(_co._renamed(build.state, node_id, heads, joins=False))
    if ends_join:
        build.add(_co._renamed(build.state, node_id, [e.source for e in preds], forks=False))
    edges = []
    for incoming in preds:
        for outgoing in succ:
            if incoming.source != outgoing.target:
                edges.append({"op": "add_edge", "source": incoming.source,
                              "target": outgoing.target, "condition": incoming.condition})
    have = {e.key() for e in build.graph.edges}
    build.add([e for e in edges if (e["source"], e["target"], e["condition"] or "") not in have
               and (e["source"], e["target"], e["condition"]) not in have])
    build.add([{"op": "remove_node", "id": node_id}])
    return build.steps




def _fold_forks(state: Mapping[str, Any]) -> list[dict]:
    """Serialize every fork left with one branch or none."""
    graph = _edits()._graph(state)
    for fork in sorted(graph.nodes.values(), key=lambda n: n.id):
        if fork.kind == "fork" and len(fork.targets) <= 1:
            steps = _co._serialize_steps(state, fork.id)
            return steps + _co._fold_forks(_simulate(state, steps))
    return []




def _expand_remove(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    owner = _fallback_owner(build.graph, edit["step"])
    if owner is not None:
        # Removing a fallback's gate or step takes the whole fallback out, so the graph is
        # what it was before it; removing the step it answers for takes it out first.
        build.add(_unfallback(state, owner))
        if edit["step"] != owner:
            _co._drop_orphans(build)
            return build.steps
        state = build.state
    graph = build.graph
    node = graph.nodes[edit["step"]]
    found = _map_removed(graph, state, node.id)
    if found is not None:
        if graph.falls_back(found["combine"]):
            build.add(_unfallback(state, found["combine"]))
            found = _map_removed(build.graph, build.state, node.id)
        build.add(_remove_map(build.state, found))
        return build.steps
    if node.kind == "gate":
        keep = edit.get("keep")
        kept = [e for e in graph.out_edges(node.id) if keep is None or e.condition == keep][:1]
        before = graph.reachable()
        build.add([{"op": "add_edge", "source": incoming.source, "target": outgoing.target,
                    "condition": incoming.condition}
                   for incoming in graph.in_edges(node.id) for outgoing in kept
                   if incoming.source != outgoing.target])
        build.add([{"op": "remove_node", "id": node.id}])
        after = build.graph
        build.add([{"op": "remove_node", "id": orphan}
                   for orphan in sorted((before & set(after.nodes)) - after.reachable())])
        judge = f"{node.id}_judge"
        after = build.graph
        if judge in after.nodes and not any(
                _consumed(n, build.state) & _outputs(after.nodes[judge])
                for n in after.nodes.values() if n.id != judge):
            for helper in [*_owned(after, judge, build.state), judge]:
                build.add(_bypass(build.state, helper))
    else:
        gone = {node.id, *_owned(graph, node.id, state)}
        _hard, strip, drop = _losses(graph, state, gone)
        build.add(_co._strip_steps(state, strip))
        for reader, names in sorted(drop.items()):
            changed = build.graph.nodes[reader].to_state()
            changed["reads"] = [r for r in changed.get("reads") or () if r not in names]
            build.add(_co._respec(build.state, reader, changed))
        for helper in [*_owned(graph, node.id, state), node.id]:
            build.add(_bypass(build.state, helper))
    build.add(_co._fold_forks(build.state))
    _co._drop_orphans(build)
    return build.steps
