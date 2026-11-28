from __future__ import annotations

import re
from typing import Any, Mapping

from refactory_dsh.nn.graph import Node

from ._common import _Refused, _edits
from .state import _available, _consumed, _data_ends, _forks_naming, _fresh, _outputs, _sealed, _sole_out, _text_of, _uses_tools
from .build import _Steps, _add_chain, _carried_knobs, _drop_orphans, _frozen_settings, _shared_knobs, _spec, _step_nodes
from .checks import _inputs_helper, _plain_step
from .remove import _bypass
from .parallelize import _chain_of
from .visibility import _seen


def _merge_tools(edit: Mapping[str, Any], a: Node, b: Node) -> list[str]:
    """The merged step's tools: the edit's, else those of the sealed steps it merges (a
    sealed step's tools come along, so the merged step stays sealed)."""
    if edit.get("tools"):
        return list(edit["tools"])
    return list(dict.fromkeys(t for n in (a, b) if _sealed(n) for t in n.tools))


def _merge_kind(edit: dict, a: Node, b: Node) -> str:
    return edit.get("kind") or ("agent" if "agent" in (a.kind, b.kind) else "code")


def _merge_text(edit: dict, a: Node, b: Node, kind: str, state: Mapping[str, Any]) -> str:
    """The merged step's text: the one given, else for an agent the two agents' prompts,
    with what ``a`` made named in words (the merged step now makes it itself, so it is no
    longer an input), after a line saying what a non-agent ``a`` did."""
    if edit.get("text"):
        return str(edit["text"])
    if kind != "agent":
        return ""
    made = sorted({name for _k, name in _outputs(a)})
    parts = [_text_of(n, state) for n in (a, b) if n.kind == "agent"]
    if a.kind != "agent":
        parts.insert(0, f"First do what step {a.id!r} did, which made {', '.join(made) or 'nothing named'}.")
    text = "\n\n".join(part for part in parts if part)
    for name in made:
        text = re.sub(r"\{\{\s*" + re.escape(name) + r"\s*\}\}", name, text)
    return text


def _pre_merge(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    ids = edit.get("steps")
    if not isinstance(ids, list) or len(ids) != 2 or not all(isinstance(i, str) for i in ids):
        raise _Refused("'steps' names two consecutive steps")
    a, b = (_plain_step(graph, i, "merge") for i in ids)
    _chain_of(graph, [a.id, b.id], state)
    _sole_out(graph, b.id)
    for node in (a, b):
        if _forks_naming(graph, node.id):
            raise _Refused(f"{node.id!r} is a fork's branch or a join's source")
        if _uses_tools(node) and not _sealed(node):
            raise _Refused(f"{node.id!r} uses tools outside a sealed workspace, which a "
                           "merged step could not keep; merge steps that do not, or sealed ones")
    # The steps' own collect steps go with them (`_expand_merge`).
    own = {a.id, b.id} | {h for h in (_inputs_helper(graph, a.id, state),
                                      _inputs_helper(graph, b.id, state)) if h}
    readers = sorted(n.id for n in graph.nodes.values()
                     if n.id not in own and _consumed(n, state) & (_outputs(a) - _outputs(b)))
    if readers:
        raise _Refused(f"{', '.join(readers)} read(s) what {a.id!r} makes, which a merged step "
                       "would no longer make")
    kind = _merge_kind(edit, a, b)
    if kind == "agent" and len(b.writes) != 1 and not _merge_tools(edit, a, b):
        raise _Refused(f"{b.id!r} writes {list(b.writes) or 'no file'}; a merged agent writes "
                       "one file")
    if (graph.start == a.id or "entry" in _data_ends(graph, a.id)) \
            and "exit" in _data_ends(graph, b.id):
        raise _Refused("the two steps start and end the same pass or run")
    _spec({**edit, "kind": kind})
    carried = _carried_knobs(kind, _merge_text(edit, a, b, kind, state), edit.get("system"),
                             _shared_knobs(a, b))
    frozen = [k for k in _frozen_settings(graph, state, {a.id, b.id}) if k not in carried]
    if frozen:
        raise _Refused(f"{a.id!r} and {b.id!r} alone read the frozen knob(s) {frozen}")
    if edit.get("id") is not None and edit["id"] in graph.nodes:
        raise _Refused(f"node {edit['id']!r} already exists")
    return None


def _expand_merge(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    a, b = (graph.nodes[i] for i in edit["steps"])
    kind = _merge_kind(edit, a, b)
    a_made = {name for _k, name in _outputs(a)}
    text = _merge_text(edit, a, b, kind, state)
    there = set(_available(graph).get(a.id, set()))
    view = _consumed(a, state) | {key for key in _consumed(b, state) if key[1] not in a_made}
    # Each step's own collect step goes with it: the merged step sees the files it handed
    # on (made before the merged step), and gets a collect step of its own for them.
    helpers = [h for h in (_inputs_helper(graph, a.id, state), _inputs_helper(graph, b.id, state))
               if h is not None]
    for helper in helpers:
        view = (view - _outputs(graph.nodes[helper])) | {
            ("file", name) for name in graph.nodes[helper].reads if name not in a_made}
        there |= {("file", name) for name in graph.nodes[helper].reads
                  if ("file", name) in set(_available(graph).get(helper, set()))
                  and name not in a_made}
    for helper in helpers:
        build.add(_bypass(build.state, helper))
    graph = build.graph
    tools = _merge_tools(edit, a, b) if kind == "agent" else []
    spec = _spec({**{k: v for k, v in edit.items() if k not in ("op", "steps")},
                  "kind": kind, "text": text, **({"tools": tools} if tools else {})})
    node_id = spec.get("id") or _fresh(graph, f"{a.id}_{b.id}")
    nodes = _step_nodes(graph, state, spec, node_id,
                        _seen(graph, state, there, view, spec.get("sees")),
                        writes=list(b.writes), produces=list(b.produces),
                        settings_from=a if a.kind == "agent" else (b if b.kind == "agent" else None),
                        knobs=_shared_knobs(a, b))
    if "exit" in _data_ends(graph, b.id):
        build.add(_add_chain(nodes, {"after": b.id}))
        build.add([{"op": "add_edge", "source": e.source, "target": nodes[0]["id"],
                    "condition": e.condition} for e in graph.in_edges(a.id)])
    else:
        build.add(_add_chain(nodes, {"before": a.id}))
        build.add([{"op": "add_edge", "source": nodes[-1]["id"], "target": e.target,
                    "condition": e.condition} for e in graph.out_edges(b.id)])
    build.add([{"op": "remove_node", "id": a.id}, {"op": "remove_node", "id": b.id}])
    _drop_orphans(build)
    return build.steps
