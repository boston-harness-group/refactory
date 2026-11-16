"""The pipeline as one line per step, in the core operators' own words.

A read-only view generated from the graph the edits act on: each model or code step on one
line with what it ``sees`` (each input named as a ``sees`` handle, with the step that makes
it), what it is ``told`` (the knobs whose text reaches it, their opening words) and what it
writes; forks as ``parallel { ... } -> join``; a judge and its gate as ``loop(step, judge
..., max N -> N+1 runs)`` or ``route(after step, judge ..., {label: step})``. Plain ops that
only assemble a prompt or collect files are folded into the ``sees`` of the step they feed
(``name@op(its inputs)``). Protected steps and frozen knobs are marked. The same lines,
compared between a candidate and its parent, say what an edit built (`changed_lines`).
Any graph shape: nested forks, data passes, gates and code steps are read the same way,
and a step the walk does not reach is still listed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .graph import Graph
from .operators import made_by
from .validate import consumed

#: How much of a knob's text a ``told`` entry shows.
TOLD_CHARS = 48
#: How many folded ops deep an input's making is shown.
FOLD_DEPTH = 3


@dataclass(frozen=True)
class Line:
    """One line of the view: the node (or gate, or fork) it is keyed by, its depth, and
    its text."""

    key: str
    depth: int
    text: str


def _values(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: (entry or {}).get("value") if isinstance(entry, Mapping) else entry
            for name, entry in (state.get("knobs") or {}).items()}


def _frozen(state: Mapping[str, Any]) -> set[str]:
    return {name for name, entry in (state.get("knobs") or {}).items()
            if isinstance(entry, Mapping) and ((entry.get("spec") or {}).get("frozen"))}


class _View:
    def __init__(self, state: Mapping[str, Any], protected: Sequence[str]) -> None:
        from refactory_dsh import snippet

        self.graph = Graph.from_state(dict(state.get("graph") or {}))
        self.values = _values(state)
        self.frozen = _frozen(state)
        self.protected = set(protected)
        self.makers = made_by(self.graph)
        g = self.graph
        self.inputs = {n: set(consumed(g.nodes[n], self.values)) for n in g.nodes}
        self.code = {n for n, node in g.nodes.items() if snippet.is_code_node(node)}
        self.folded = self._folded()
        self.judges = self._judges()
        self.back = self._back_edges()

    # ── what is folded ───────────────────────────────────────────────────
    def _readers(self, node_id: str) -> set[str]:
        node = self.graph.nodes[node_id]
        outs = {("fact", f) for f in node.produces} | {("file", f) for f in node.writes}
        return {n for n, taken in self.inputs.items() if taken & outs}

    def _folded(self) -> set[str]:
        """Plain ops that only hand their outputs to model or code steps (or to other such
        ops): they are said inside the ``sees`` of the steps they feed."""
        g = self.graph
        folded: set[str] = set()
        changed = True
        while changed:
            changed = False
            for node_id, node in g.nodes.items():
                if (node_id in folded or node.kind != "op" or node_id in self.code
                        or node_id in self.protected):
                    continue
                readers = self._readers(node_id)
                if readers and all(g.nodes[r].kind == "agent" or r in self.code or r in folded
                                   for r in readers):
                    folded.add(node_id)
                    changed = True
        return folded

    def _judges(self) -> dict[str, str]:
        """Each agent whose reply only a gate reads, by gate: said in that gate's line."""
        g = self.graph
        out: dict[str, str] = {}
        for gate_id, gate in g.nodes.items():
            if gate.kind != "gate":
                continue
            for name in gate.reads:
                for maker in self.makers.get(("file", name)) or []:
                    if g.nodes[maker].kind == "agent" and self._readers(maker) <= {gate_id}:
                        out[gate_id] = maker
        return out

    # ── one input, one step ──────────────────────────────────────────────
    def _input(self, kind: str, name: str, depth: int = 0) -> str:
        made = self.makers.get((kind, name)) or []
        folded = [m for m in made if m in self.folded]
        if not folded or depth >= FOLD_DEPTH:
            return f"{name}@{','.join(made) or 'task'}"
        parts = []
        for op_id in folded:
            inner = [self._input(k, n, depth + 1) for k, n in self._sorted(self.inputs[op_id])]
            parts.append(f"{name}@{op_id}" + (f"({', '.join(inner)})" if inner else ""))
        return ", ".join(parts)

    @staticmethod
    def _sorted(keys: set[tuple[str, str]]) -> list[tuple[str, str]]:
        return sorted(keys, key=lambda key: (key[0] != "fact", key[1]))

    def _told(self, node_id: str) -> list[str]:
        """The knobs whose text reaches the step: its own bound text fields, and the text
        knobs of the folded ops that build its inputs."""
        g = self.graph
        names: list[str] = []
        node = g.nodes[node_id]
        for field in ("system_prompt_template", "prompt_template", "code"):
            knob = node.bound_knob(field)
            if knob:
                names.append(knob)
        todo = [m for k, n in self._sorted(self.inputs[node_id])
                for m in self.makers.get((k, n)) or [] if m in self.folded]
        seen: set[str] = set()
        shared: set[str] = set()
        while todo:
            op_id = todo.pop(0)
            if op_id in seen:
                continue
            seen.add(op_id)
            op = g.nodes[op_id]
            texts = [k for k in op.knobs if isinstance(self.values.get(k), str)]
            names += texts
            # An op with several outputs reads its knobs for all of them; which one a
            # knob feeds is not declared, so the knob is marked "?".
            if len(op.produces) + len(op.writes) > 1:
                shared.update(texts)
            todo += [m for k, n in self._sorted(self.inputs[op_id])
                     for m in self.makers.get((k, n)) or [] if m in self.folded]
        out = []
        for name in dict.fromkeys(names):
            text = " ".join(str(self.values.get(name) or "").split())
            text = text if len(text) <= TOLD_CHARS else text[:TOLD_CHARS - 1] + "…"
            out.append(f"{name}{'*' if name in self.frozen else ''}"
                       f"{'?' if name in shared else ''} \"{text}\"")
        return out

    def step(self, node_id: str) -> str:
        g = self.graph
        node = g.nodes[node_id]
        kind = "code" if node_id in self.code else node.kind
        sees = [self._input(k, n) for k, n in self._sorted(self.inputs[node_id])]
        told = self._told(node_id)
        outs = [*node.writes, *(f for f in node.produces if f not in node.writes)]
        return ("[protected] " if node_id in self.protected else "") + "  ".join(
            part for part in (
                f"{node_id} {kind}",
                f"sees {', '.join(sees)}" if sees else "",
                f"told {'; '.join(told)}" if told else "",
                f"-> {', '.join(outs)}" if outs else "") if part)

    def judge_part(self, gate_id: str) -> str:
        gate = self.graph.nodes[gate_id]
        judge = self.judges.get(gate_id)
        if judge:
            body = self.step(judge).split("  ", 1)
            return f"judge {judge}" + (f" ({body[1]})" if len(body) > 1 else "")
        if gate_id in self.code:
            return "judge code"
        return f"judge {gate.evaluator or gate.gate_kind}"

    # ── the walk ─────────────────────────────────────────────────────────
    def _back_edges(self) -> set[tuple[str, str]]:
        """The edges that go back to a step already on the way to their source (a loop's
        send-back): found by a depth-first walk from the start, then from any step it did
        not reach."""
        g = self.graph
        state: dict[str, int] = {}  # 1 on the walk's stack, 2 done
        back: set[tuple[str, str]] = set()

        def successors(node_id: str) -> list[str]:
            node = g.nodes[node_id]
            out = [e.target for e in g.out_edges(node_id)]
            if node.kind == "fork":
                out += list(node.targets)
            if node.kind == "data" and node.subgraph_entry:
                out.append(node.subgraph_entry)
            return [t for t in dict.fromkeys(out) if t in g.nodes]

        for root in [g.start, *sorted(g.nodes)]:
            if root not in g.nodes or root in state:
                continue
            stack = [(root, iter(successors(root)))]
            state[root] = 1
            while stack:
                node_id, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    state[node_id] = 2
                    stack.pop()
                elif state.get(nxt) == 1:
                    back.add((node_id, nxt))
                elif nxt not in state:
                    state[nxt] = 1
                    stack.append((nxt, iter(successors(nxt))))
        return back

    def _forward(self, node_id: str) -> list[str]:
        return [e.target for e in self.graph.out_edges(node_id)
                if (node_id, e.target) not in self.back]

    def _next(self, node_id: str) -> str | None:
        g = self.graph
        node = g.nodes[node_id]
        if node.kind == "gate":
            forward = self._forward(node_id)
            return forward[0] if len(forward) == 1 else None
        outs = [e.target for e in g.out_edges(node_id) if e.condition is None]
        return outs[0] if len(outs) == 1 else None

    def _hidden(self, node_id: str) -> bool:
        """Said inside another line: a folded op, a judge, a no-op step."""
        node = self.graph.nodes[node_id]
        return (node_id in self.folded or node_id in self.judges.values()
                or (node.kind == "op" and not (self.inputs[node_id] or node.writes
                                               or node.produces)))

    def _shown_before(self, node_id: str) -> set[str]:
        """The shown steps right before ``node_id``, looking through hidden ones."""
        out: set[str] = set()
        for edge in self.graph.in_edges(node_id):
            source = edge.source
            out |= self._shown_before(source) if self._hidden(source) else {source}
        return out

    def _shown_from(self, node_id: str) -> str:
        """The first shown step from ``node_id`` on, looking through hidden ones."""
        seen = set()
        while node_id in self.graph.nodes and self._hidden(node_id) and node_id not in seen:
            seen.add(node_id)
            nxt = self._next(node_id)
            if nxt is None:
                break
            node_id = nxt
        return node_id

    def _reach(self, node_id: str) -> list[str]:
        """The nodes from ``node_id`` on along forward edges, in walk order."""
        order, todo, seen = [], [node_id], set()
        while todo:
            current = todo.pop(0)
            if current in seen or current not in self.graph.nodes:
                continue
            seen.add(current)
            order.append(current)
            node = self.graph.nodes[current]
            todo += self._forward(current) + (list(node.targets) if node.kind == "fork" else [])
        return order

    def _join_of(self, fork_id: str) -> str | None:
        """The join a fork's branches meet at: the first join along its first branch that
        has no fork of its own still open."""
        g = self.graph
        targets = list(g.nodes[fork_id].targets)
        if not targets:
            return None
        node_id, depth, seen = targets[0], 0, set()
        while node_id and node_id in g.nodes and node_id not in seen:
            seen.add(node_id)
            kind = g.nodes[node_id].kind
            if kind == "fork":
                depth += 1
            elif kind == "join":
                if depth == 0:
                    return node_id
                depth -= 1
            outs = [e.target for e in g.out_edges(node_id) if e.condition is None]
            if kind == "gate":
                outs = self._forward(node_id)
            node_id = outs[0] if outs else None
        return None

    def lines(self) -> list[Line]:
        out: list[Line] = []
        self.shown: set[str] = set()
        self._chain(self.graph.start, None, 0, out)
        rest = [n for n in sorted(self.graph.nodes) if n not in self.shown
                and n not in self.folded and self.graph.nodes[n].kind not in ("join",)
                and n not in self.judges.values()]
        if rest:
            out.append(Line("(other)", 0, "not on the main path:"))
            for node_id in self._in_order(rest):
                if node_id not in self.shown:
                    self._chain(node_id, None, 1, out)
        return out

    def _in_order(self, nodes: list[str]) -> list[str]:
        """``nodes`` with each after the ones that lead to it (edges among them), ties by id;
        what a cycle leaves goes last, by id."""
        g = self.graph
        left = set(nodes)
        order: list[str] = []
        while left:
            ready = sorted(n for n in left if not any(
                e.source in left and e.source != n and (e.source, n) not in self.back
                for e in g.in_edges(n)))
            if not ready:
                order += sorted(left)
                break
            order.append(ready[0])
            left.discard(ready[0])
        return order

    def _one(self, node_id: str, depth: int, out: list[Line]) -> None:
        g = self.graph
        node = g.nodes[node_id]
        self.shown.add(node_id)
        if node_id in self.folded or node_id in self.judges.values():
            return
        if node.kind == "gate":
            back = sorted(e.target for e in g.out_edges(node_id)
                          if (node_id, e.target) in self.back)
            if back:
                cap = node.bound("max_iterations", self.values)
                knob = node.bound_knob("max_iterations")
                out.append(Line(node_id, depth, f"loop({', '.join(back)}, "
                                f"{self.judge_part(node_id)}, max {cap}"
                                f"{'*' if knob in self.frozen else ''} -> {cap + 1} runs"
                                f"{f', knob {knob}' if knob else ''})"))
            else:
                before = sorted(self._shown_before(node_id))
                labels = ", ".join(f"{e.condition}: {self._shown_from(e.target)}"
                                   for e in g.out_edges(node_id) if e.condition)
                out.append(Line(node_id, depth, f"route(after {', '.join(before) or '-'}, "
                                f"{self.judge_part(node_id)}, {{{labels}}})"))
            return
        if node.kind in ("agent", "op"):
            if node.kind == "op" and not (self.inputs[node_id] or node.writes or node.produces):
                return  # a no-op step (the end of a pass after a gate)
            out.append(Line(node_id, depth, self.step(node_id)))

    def _chain(self, start: str | None, stop: str | None, depth: int,
               out: list[Line]) -> str | None:
        """Walk from ``start`` until ``stop`` (a join) or the end; returns where it ended."""
        g = self.graph
        node_id = start
        while node_id and node_id != stop and node_id in g.nodes:
            if node_id in self.shown and g.nodes[node_id].kind != "join":
                return node_id
            node = g.nodes[node_id]
            if node.kind == "data":
                self.shown.add(node_id)
                out.append(Line(node_id, depth, f"per item ({node_id}"
                                f"{' -> ' + ', '.join(node.writes) if node.writes else ''}):"))
                self._chain(node.subgraph_entry, None, depth + 1, out)
            elif node.kind == "fork":
                self.shown.add(node_id)
                join = self._join_of(node_id)
                out.append(Line(node_id, depth, "parallel {"))
                for target in node.targets:
                    self._chain(target, join, depth + 1, out)
                out.append(Line(f"{node_id}/", depth, f"}} -> {join or 'end'}"))
                if join:
                    self.shown.add(join)
                    node_id = self._next(join)
                    continue
                return None
            elif node.kind == "gate" and not any((node_id, e.target) in self.back
                                                 for e in g.out_edges(node_id)) and len(
                    [e for e in g.out_edges(node_id) if e.condition]) > 1:
                self._one(node_id, depth, out)
                # Each label's steps until the paths meet again, then on from there.
                edges = [e for e in g.out_edges(node_id) if e.condition]
                reaches = [self._reach(e.target) for e in edges]
                common = set(reaches[0]).intersection(*map(set, reaches[1:])) if reaches else set()
                rejoin = next((n for n in reaches[0] if n in common), None) if reaches else None
                for edge in edges:
                    if edge.target != rejoin and edge.target not in self.shown:
                        out.append(Line(f"{node_id}:{edge.condition}", depth + 1,
                                        f"on {edge.condition}:"))
                        self._chain(edge.target, rejoin, depth + 2, out)
                node_id = rejoin
                continue
            else:
                self._one(node_id, depth, out)
            node_id = self._next(node_id)
        return node_id


def view_lines(state: Mapping[str, Any], protected: Sequence[str] = ()) -> list[Line]:
    """The pipeline view of ``state``, line by line; empty when the graph cannot be read."""
    try:
        return _View(state, protected).lines()
    except Exception:  # noqa: BLE001 - an unreadable graph is the validator's to report
        return []


def render(state: Mapping[str, Any], protected: Sequence[str] = ()) -> str:
    """The view as text, two spaces of indent per level."""
    return "\n".join("  " * line.depth + line.text for line in view_lines(state, protected))


def changed_lines(before: Mapping[str, Any], after: Mapping[str, Any],
                  protected: Sequence[str] = ()) -> list[str]:
    """What an edit built, as the view's own lines: each new (``+``) or changed (``~``)
    line of ``after``'s view against ``before``'s, keyed by step, gate or fork, and each
    one gone (``- removed``)."""
    old = {line.key: line.text for line in view_lines(before, protected)}
    new = view_lines(after, protected)
    out = []
    for line in new:
        if line.key.endswith("/") or re.fullmatch(r".+:[^:]+", line.key):
            continue  # a block's closing line or a route's label line
        if line.key not in old:
            out.append(f"+ {line.text}")
        elif old[line.key] != line.text:
            out.append(f"~ {line.text}")
    gone = sorted(set(old) - {line.key for line in new} - {"(other)"})
    out += [f"- removed {key}" for key in gone if not key.endswith("/")
            and not re.fullmatch(r".+:[^:]+", key)]
    return out
