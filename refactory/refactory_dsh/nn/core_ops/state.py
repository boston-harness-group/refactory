from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from refactory_dsh.nn.graph import Graph, Node
from refactory_dsh.tool_policy import POLICIES as TOOL_POLICIES

from ._common import _Refused, _edits


def _knob_values(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: (entry or {}).get("value") for name, entry in (state.get("knobs") or {}).items()}




def _outputs(node: Node) -> set[tuple[str, str]]:
    return {("file", n) for n in node.writes} | {("fact", n) for n in node.produces}




def _consumed(node: Node, state: Mapping[str, Any]) -> set[tuple[str, str]]:
    from refactory_dsh.nn.validate import consumed

    try:
        return consumed(node, _knob_values(state))
    except Exception:  # noqa: BLE001 - a node that cannot be read is the validator's to say
        return set()




def _available(graph: Graph) -> dict[str, set[tuple[str, str]]]:
    from refactory_dsh.nn.validate import available_inputs

    return available_inputs(graph)




def _grader_side(graph: Graph, state: Mapping[str, Any]) -> set[tuple[str, str]]:
    from refactory_dsh.nn.validate import grader_derived

    try:
        return set(grader_derived(graph, _knob_values(state)))
    except Exception:  # noqa: BLE001 - an unloadable task is another rule's to report
        return set()




def _task_inputs(graph: Graph, state: Mapping[str, Any]) -> set[tuple[str, str]]:
    """What ``"sees": ["task"]`` gives (`operators.task_inputs`). A node that cannot be
    read refuses the edit with the real error, never as a task that gives nothing."""
    from refactory_dsh.nn.operators import task_inputs

    try:
        return task_inputs(graph, _knob_values(state))
    except Exception as exc:  # noqa: BLE001 - said in the refusal
        raise _Refused(f"what \"task\" gives cannot be worked out: {type(exc).__name__}: "
                       f"{exc}") from exc




def _taken_names(graph: Graph) -> set[str]:
    names = set(graph.nodes)
    for node in graph.nodes.values():
        names |= set(node.writes) | set(node.produces)
        names |= {name.rsplit(".", 1)[0] for name in node.writes}
    return names




def _fresh(graph: Graph, base: str, also: Iterable[str] = ()) -> str:
    """An id no node has and no file or fact is named after (nor any in ``also``)."""
    taken = _taken_names(graph) | set(also)
    base = re.sub(r"\W", "_", base).strip("_") or "step"
    if base not in taken and f"{base}.txt" not in taken:
        return base
    for n in range(2, 1000):
        candidate = f"{base}{n}"
        if candidate not in taken:
            return candidate
    raise _edits().EditError(f"cannot find a free id based on {base!r}")




def _task_knob(node: Node) -> str | None:
    return next((knob for field_name, knob in node.field_bindings
                 if field_name == "prompt_template"), None)




def _text_of(node: Node, state: Mapping[str, Any]) -> str:
    """What a step does, as text: an agent's task prompt, a code step's code."""
    values = _knob_values(state)
    field_name = "code" if node.code or any(f == "code" for f, _ in node.field_bindings) \
        else "prompt_template"
    try:
        return str(node.bound(field_name, values) or "")
    except Exception:  # noqa: BLE001
        return str(getattr(node, field_name, "") or "")




def _is_code(node: Node) -> bool:
    from refactory_dsh.snippet import is_code_node

    return is_code_node(node)




def _kind_of(node: Node) -> str:
    """A step's kind as an operator names it."""
    if node.kind == "agent":
        return "agent"
    if node.kind == "op":
        return "code" if _is_code(node) else "op"
    return node.kind




def _sealed(node: Node) -> bool:
    from refactory_dsh.workspace import is_sealed

    return is_sealed(node)




def _uses_tools(node: Node) -> bool:
    return node.kind == "agent" and (node.tool_policy not in TOOL_POLICIES or bool(node.tools))




def _forks_naming(graph: Graph, node_id: str) -> list[Node]:
    return [n for n in graph.nodes.values()
            if node_id in (*n.targets, *n.optional_targets, *n.sources)]




def _data_ends(graph: Graph, node_id: str) -> list[str]:
    """``entry``/``exit`` for each data pass that starts or ends at ``node_id``."""
    out = []
    for node in graph.nodes.values():
        if node.kind == "data":
            if node.subgraph_entry == node_id:
                out.append("entry")
            if node.subgraph_exit == node_id:
                out.append("exit")
    return out




def _sole_out(graph: Graph, node_id: str) -> str | None:
    """The target of a node's one plain out-edge; refused for any other shape."""
    out = graph.out_edges(node_id)
    if not out:
        return None
    if len(out) > 1 or out[0].condition is not None:
        raise _Refused(f"{node_id!r} has {len(out)} out-edge(s) carrying verdicts, so the "
                       "operator cannot tell which one to follow")
    return out[0].target




def _require_step(graph: Graph, node_id: Any, what: str = "step") -> Node:
    if not isinstance(node_id, str) or node_id not in graph.nodes:
        raise _Refused(f"no {what} {node_id!r}; existing steps: {', '.join(sorted(graph.nodes))}")
    return graph.nodes[node_id]




def _join_of(graph: Graph, fork: Node) -> Node:
    join = graph.fork_join(fork.id)
    if join is None or join not in graph.nodes or graph.nodes[join].kind != "join":
        raise _Refused(f"fork {fork.id!r} has no join after it")
    return graph.nodes[join]




def _downstream(graph: Graph, start: str, *, reloops: bool = False) -> set[str]:
    """``start`` and every step after it (fork branches and a data pass's steps included,
    no reloop back unless ``reloops``)."""
    from refactory_dsh.nn.loops import loop_edges

    back = set() if reloops else {(source, target) for source, target, _c in loop_edges(graph)}
    out, queue = {start}, [start]
    while queue:
        node = graph.nodes.get(queue.pop())
        if node is None:
            continue
        following = [e.target for e in graph.out_edges(node.id) if (node.id, e.target) not in back]
        for target in [*(node.targets if node.kind == "fork" else ()),
                       *((node.subgraph_entry,) if node.kind == "data" and node.subgraph_entry
                         else ()), *following]:
            if target not in out:
                out.add(target)
                queue.append(target)
    return out
