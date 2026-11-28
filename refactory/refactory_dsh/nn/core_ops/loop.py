from __future__ import annotations

from typing import Any, Mapping, Sequence

from refactory_dsh.nn.graph import Graph, Node

from ._common import JUDGE_KINDS, LOOP_ITERATIONS, NEUTRAL_CHECK, _Refused, _edits
from .state import _available, _consumed, _fresh, _is_code, _knob_values, _outputs, _sole_out
from .build import _add_chain, _simulate, _spec, _step_nodes
from .visibility import _seen
from .checks import _plain_step


def _judge(edit: dict) -> str:
    judge = edit.get("judge", "agent")
    if not isinstance(judge, str) or not (judge in JUDGE_KINDS or ":" in judge):
        raise _Refused(f"'judge' is one of {list(JUDGE_KINDS)} or a 'module:function' callable")
    if edit.get("tools") and judge != "agent":
        raise _Refused(f"'tools' are an agent judge's; a {judge!r} judge runs no tools")
    return judge


def _judge_spec(edit: Mapping[str, Any], default_text: str) -> dict:
    """The step spec of a ``loop`` or ``route`` agent judge: its text, and its 'tools' and
    'sees' as an inserted agent step takes them (`_step_nodes`): with tools it runs sealed,
    in a fresh workspace holding only the files it sees -- without 'sees', in the per-item
    part, the item's input files too (`_with_item_files`) -- and its reply is still the
    verdict its gate reads (`VERDICT_RULE`)."""
    raw: dict[str, Any] = {"kind": "agent", "text": edit.get("text") or default_text}
    if edit.get("tools"):
        raw["tools"] = list(edit["tools"])
    if edit.get("sees") is not None:
        raw["sees"] = edit["sees"]
    return _spec(raw)


def is_one_line_loop(edit: Mapping[str, Any]) -> bool:
    """Whether a ``loop`` edit is the one-line form (a step and a judge), not the written-out
    one (a ``target`` and a ``gate`` node object)."""
    return "gate" not in edit and "target" not in edit


def _pre_loop_line(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    node = _plain_step(graph, edit.get("step"), "loop")
    _sole_out(graph, node.id)
    _judge(edit)
    if edit.get("text") is not None and not isinstance(edit.get("text"), str):
        raise _Refused("'text' is a string")
    if edit.get("id") is not None and edit["id"] in graph.nodes:
        raise _Refused(f"node {edit['id']!r} already exists")
    _loop_cap(edit)
    if _back_to(graph, edit, node).id != node.id:
        _loop_cycle_valid(edit, state)
    return None


def _pass_of(graph: Graph, node_id: str) -> str | None:
    """The innermost data pass whose body holds ``node_id`` (``None``: the top level)."""
    bodies = [(len(body), data_id) for data_id, node in graph.nodes.items()
              if node.kind == "data" and node_id in (body := graph.data_body(data_id))]
    return min(bodies)[1] if bodies else None


def _forward_flow(graph: Graph) -> dict[str, list[str]]:
    """Each node's successors as a run walks them, retries left out: plain and verdict
    edges, a fork to its branch heads (not its own edge to its join, which a run reaches
    only through them), a branch's end to its join. A data node leads on past its pass
    (whose body is a scope of its own)."""
    from refactory_dsh.nn.loops import loop_edges

    retries = loop_edges(graph)
    out: dict[str, list[str]] = {key: [] for key in graph.nodes}
    for edge in graph.edges:
        source = graph.nodes.get(edge.source)
        if source is not None and source.kind == "fork" and source.targets:
            continue
        if edge.key() not in retries and edge.source in out and edge.target in out:
            out[edge.source].append(edge.target)
    for key, node in graph.nodes.items():
        if node.kind == "fork":
            out[key] += [t for t in node.targets if t in out]
        elif node.kind == "join":
            for source in node.sources:
                if source in out:
                    out[source].append(key)
    return out


def _reach(flow: Mapping[str, Sequence[str]], start: str, without: str | None = None
           ) -> set[str]:
    seen, todo = {start}, [start]
    while todo:
        for nxt in flow.get(todo.pop(), ()):
            if nxt not in seen and nxt != without:
                seen.add(nxt)
                todo.append(nxt)
    return seen


def _back_to(graph: Graph, edit: dict, node: Node) -> Node:
    """The step a loop sends back to: ``node`` itself, or the edit's ``back_to`` -- an
    agent (only an agent's prompt takes the judge's reason, on both runtimes) that every
    run passes before ``node``, in the same per-item pass and fork branch as ``node``, so
    a send-back re-runs exactly the steps from it to ``node`` (a fork in between whole)."""
    back = edit.get("back_to")
    if back is None or back == node.id:
        return node
    target = _plain_step(graph, back, "a loop's 'back_to'")
    where, here = _pass_of(graph, target.id), _pass_of(graph, node.id)
    if where != here:
        inside = f"inside the per-item pass {here!r}" if here else "outside any per-item pass"
        raise _Refused(f"'back_to' {target.id!r} is outside the pass {node.id!r} is in "
                       f"({node.id!r} is {inside}); a loop sends back only within its pass")
    flow = _forward_flow(graph)
    if target.id in _reach(flow, node.id):
        raise _Refused(f"'back_to' {target.id!r} comes after {node.id!r}; it names an earlier "
                       f"step, where the work goes back to when the judge after {node.id!r} "
                       "says reloop")
    start = _scope_start(graph, here)
    if target.id != start and node.id in _reach(flow, start, without=target.id):
        on_every = sorted(n for n in _reach(flow, start) - {node.id} - _map_runs(graph)
                          if graph.nodes[n].kind == "agent" and n != target.id
                          and (n == start or node.id not in _reach(flow, start, without=n)))
        raise _Refused(f"'back_to' {target.id!r} is not on every path to {node.id!r} (a branch "
                       "or a route's way passes it by); it names a step every run passes "
                       f"before {node.id!r}: " + (", ".join(on_every) or "none here"))
    from refactory_dsh.nn.validate import fork_branches

    for fork in (n for n in graph.nodes.values() if n.kind == "fork"):
        for branch in fork_branches(graph, fork, graph.fork_join(fork.id)):
            if node.id in branch and target.id not in branch:
                raise _Refused(f"{node.id!r} is inside a branch of fork {fork.id!r} and "
                               f"'back_to' {target.id!r} is not; a loop in a branch sends "
                               "back only within it (judge a step after the fork's join to "
                               "re-run the whole fork)")
            if target.id in branch and node.id not in branch:
                raise _Refused(f"'back_to' {target.id!r} is inside a branch of fork "
                               f"{fork.id!r} and {node.id!r} is not; a send-back from after "
                               "the join re-runs the whole fork, so it names a step before "
                               f"{fork.id!r}")
    if target.kind != "agent":
        nearest = _agents_after(graph, flow, target.id, node.id)
        raise _Refused(f"'back_to' {target.id!r} is {'code' if _is_code(target) else 'an op'}, "
                       "which has no prompt for the judge's reason; send back to the model "
                       "step that should act on it: "
                       + (", ".join(nearest) or f"none between it and {node.id!r}"))
    return target


def _map_runs(graph: Graph) -> set[str]:
    """The runs of every map (`operators.map_of`): never a step to name alone."""
    from refactory_dsh.nn.operators import map_of

    return {run for fork in graph.nodes if map_of(graph, fork)
            for run in graph.nodes[fork].targets}


def _scope_start(graph: Graph, data_id: str | None) -> str:
    """Where a run of the scope begins: the pass's entry, or the run's start."""
    return graph.nodes[data_id].subgraph_entry if data_id else graph.start


def _agents_after(graph: Graph, flow: Mapping[str, Sequence[str]], start: str, end: str
                  ) -> list[str]:
    """The first agents after ``start`` on the way to ``end`` (``end`` included)."""
    leads = {n for n in _reach(flow, start) if end in _reach(flow, n)}
    runs = _map_runs(graph)
    out, seen, todo = [], {start}, list(flow.get(start, ()))
    while todo:
        current = todo.pop(0)
        if current in seen or current not in leads:
            continue
        seen.add(current)
        if graph.nodes[current].kind == "agent" and current not in runs:
            out.append(current)
        else:
            todo += list(flow.get(current, ()))
    return sorted(out)


def _loop_cycle_valid(edit: dict, state: dict) -> None:
    """Refused when the validator refuses the loop's cycle (V10, V12, V16) and did not
    refuse the graph before it."""
    from refactory_dsh.nn.validate import validate

    rules = ("V10", "V12", "V16")

    def issues(of: Mapping[str, Any]) -> list:
        return [issue for issue in validate(_edits()._graph(of), _knob_values(of),
                                            declared_knobs={}, resolve_calls=False)
                if issue.code in rules]

    before = {str(issue) for issue in issues(state)}
    new = [issue for issue in issues(_simulate(state, _expand_loop_line(edit, state)))
           if str(issue) not in before]
    if new:
        raise _Refused(f"the validator refuses the cycle back to {edit['back_to']!r}: {new[0]}")


def _loop_cap(edit: dict) -> int:
    """The loop's ``max_iterations`` (how often the judge may send the step back), the
    default when the edit leaves it out; refused outside the knob's bounds."""
    default, lo, hi = LOOP_ITERATIONS
    cap = edit.get("max_iterations", default)
    if isinstance(cap, bool) or not isinstance(cap, int) or not lo <= cap <= hi:
        raise _Refused(f"'max_iterations' is a whole number from {lo} to {hi} (how often the "
                       "judge may send the step back)")
    return cap


def _iterations_knob(gate_id: str, cap: int = LOOP_ITERATIONS[0]) -> tuple[str, dict]:
    from refactory_dsh.nn.knob import Knob

    _default, lo, hi = LOOP_ITERATIONS
    return f"{gate_id}__max_iterations", Knob(
        cap, "int", lo=lo, hi=hi,
        description=f"how often {gate_id} may send the work back").spec()


def _gate_state(gate_id: str, judge: str, text: str, seen: Sequence[tuple[str, str]], *,
                label: str, reads: Sequence[str] = (), iterations: bool = False,
                cap: int = LOOP_ITERATIONS[0], sealed: bool = False) -> dict[str, Any]:
    from refactory_dsh import snippet

    gate: dict[str, Any] = {"id": gate_id, "kind": "gate"}
    if judge == "agent":
        gate.update({"evaluator": snippet.VERDICT, "reads": list(reads),
                     "required_reads": list(reads)})
    elif judge == "code":
        gate.update({"evaluator": snippet.GATE, "code": text or _default_gate_code(label),
                     "inputs": [n for k, n in seen if k == "fact"],
                     "reads": [n for k, n in seen if k == "file"]})
        if sealed:
            gate["sealed_script"] = True
    else:
        gate.update({"evaluator": judge, "reads": [n for k, n in seen if k == "file"]})
    if iterations:
        name, _spec_ = _iterations_knob(gate_id, cap)
        gate.update({"max_iterations": cap, "on_exhausted": "continue",
                     "knobs": [name], "field_bindings": [["max_iterations", name]]})
    return gate


def _default_gate_code(label: str) -> str:
    return ("def main(inp):\n"
            f"    return {{'verdict': {label!r}, 'reason': 'no check written yet'}}\n")


def _expand_loop_line(edit: dict, state: dict) -> list[dict]:
    edits = _edits()
    graph = edits._graph(state)
    node = graph.nodes[edit["step"]]
    back = _back_to(graph, edit, node).id
    judge = _judge(edit)
    there = _available(graph).get(node.id, set()) | _outputs(node)
    seen = _seen(graph, state, there, _consumed(node, state) | _outputs(node), edit.get("sees"))
    gate_id = edit.get("id") or _fresh(graph, f"{node.id}_check")
    cap = _loop_cap(edit)
    knob, knob_spec = _iterations_knob(gate_id, cap)
    steps: list[dict] = [{"op": "add_knob", "name": knob, "spec": knob_spec}]
    is_sealed = bool(edit.get("sealed"))
    if judge == "agent":
        judge_id = _fresh(graph, f"{gate_id}_judge", (gate_id,))
        nodes = _step_nodes(graph, state, _judge_spec(edit, NEUTRAL_CHECK),
                            judge_id, seen, verdict=("proceed", "reloop"),
                            settings_from=node if node.kind == "agent" else None)
        gate = _gate_state(gate_id, judge, "", seen, label="proceed",
                           reads=[f"{judge_id}.txt"], iterations=True, cap=cap)
        steps += edits._loop_steps(graph, node.id, gate, back)
        steps += _add_chain(nodes, {"before": gate_id})
    else:
        gate = _gate_state(gate_id, judge, edit.get("text") or "", seen, label="proceed",
                           iterations=True, cap=cap, sealed=is_sealed)
        steps += edits._loop_steps(graph, node.id, gate, back)
    return steps
