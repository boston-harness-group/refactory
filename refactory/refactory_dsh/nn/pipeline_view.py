"""The pipeline as one line per step, read in the core operators' own words.

A read-only view generated from the graph the edits act on. Reading mirrors writing:
every operation in the catalogue names the readers that render what it builds
(``EditSpec.reads_as``), and `READERS` holds them, so a new operation without a reader,
or a reader no operation names, fails ``tests/test_pipeline_view.py``:

- ``step``: a model, code or plain step on one line -- what it ``sees``, what it is
  ``told``, what it writes (``->``);
- ``sees``: a step's inputs as ``sees`` handles, ``name@maker``; a plain op that only
  assembles them folded in as ``name (via op: its inputs)``; an agent with tools sees the
  run directory (or its own workspace);
- ``told``: each knob whose text reaches the step, its opening words (``*`` frozen, ``?``
  read by an op with several outputs, for all of them); a code step's code;
- ``parallel``: a fork's branches, ``parallel fork { ... } -> join``, ``[optional]``;
- ``loop``: a gate that sends work back, ``loop(step, judge ..., max N -> N+1 runs)``;
- ``route``: a gate choosing a way on, ``route(after step, judge ..., {label: step})``,
  each label's steps until the ways meet;
- ``edges``: the order of the lines; what the walk cannot place is a fallback line.

A plain op is folded into the step it feeds only when it is a pure assembler (an op with
no tools that is not code, protected or the grader's) whose outputs only model steps read,
directly or through the collect step an operator put in front of one;
graders, ops doing real work and code steps keep their own lines. Anything no reader
recognises is drawn generically (kind, sees, writes, edges as ``->``) and counted
(`fallback_count`). The same lines, compared between a candidate and its parent, say what
an edit built (`changed_lines`), with the block each new line sits in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .graph import Graph
from .operators import made_by
from .validate import consumed

#: How much of a knob's text a ``told`` entry shows.
TOLD_CHARS = 48
#: How many folded ops deep an input's making is shown.
FOLD_DEPTH = 3
#: A loop's line: the step it sends back to, its judge, its cap and how often the step then
#: runs at most (cap + 1). The legend's form is made from the same template.
LOOP_LINE = "loop({step}, {judge}, max {cap} -> {runs} runs{knob})"
#: The key prefix of a line no reader recognised.
FALLBACK = "(fallback)"


@dataclass(frozen=True)
class Line:
    """One line of the view: the node (or gate, fork, block end) it is keyed by, its
    depth, its text, the block it sits in ("" at the top) and the reader that drew it."""

    key: str
    depth: int
    text: str
    within: str = ""
    reader: str = "step"


def _values(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: (entry or {}).get("value") if isinstance(entry, Mapping) else entry
            for name, entry in (state.get("knobs") or {}).items()}


def _frozen(state: Mapping[str, Any]) -> set[str]:
    return {name for name, entry in (state.get("knobs") or {}).items()
            if isinstance(entry, Mapping) and ((entry.get("spec") or {}).get("frozen"))}


def _clip(text: Any) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= TOLD_CHARS else text[:TOLD_CHARS - 1] + "…"


class _View:
    def __init__(self, state: Mapping[str, Any], protected: Sequence[str],
                 graders: Sequence[str] = ()) -> None:
        from refactory_dsh import snippet

        self.graph = Graph.from_state(dict(state.get("graph") or {}))
        self.values = _values(state)
        self.frozen = _frozen(state)
        self.protected = set(protected)
        self.graders = set(graders)
        self.makers = made_by(self.graph)
        g = self.graph
        self.inputs = {n: set(consumed(g.nodes[n], self.values)) for n in g.nodes}
        self.code = {n for n, node in g.nodes.items() if snippet.is_code_node(node)}
        self.optional = {t for node in g.nodes.values() if node.kind == "fork"
                         for t in node.optional_targets} | {
            n for n, node in g.nodes.items() if node.bound_flag("optional", self.values)}
        self.folded = self._folded()
        self.judges = self._judges()
        self.back = self._back_edges()
        self.shown: set[str] = set()
        # An optional branch's flag may sit on its collect step: the mark goes to the
        # step it leads to.
        self.optional |= {self._shown_from(n) for n in self.optional if n in g.nodes}

    # ── what is folded, what is a judge ──────────────────────────────────
    def _readers(self, node_id: str) -> set[str]:
        node = self.graph.nodes[node_id]
        outs = {("fact", f) for f in node.produces} | {("file", f) for f in node.writes}
        return {n for n, taken in self.inputs.items() if taken & outs}

    def _model(self, node_id: str) -> bool:
        return self.graph.nodes[node_id].kind == "agent"

    def _folded(self) -> set[str]:
        """Pure assemblers: plain ops (no tools, not code, not protected or the grader's)
        whose outputs only model steps read -- directly, or through the collect or
        file_text step an operator put in front of one."""
        from .operators import COLLECT_CALL, FILE_TEXT_CALL

        # A step a gate's verdict leads to (a route's label, a loop's send-back) is named in
        # that gate's line, so it is never folded away.
        # (A collect step in front of one is looked through, so it may still fold.)
        gated = {e.target for e in self.graph.edges if e.condition}
        candidates = {n for n, node in self.graph.nodes.items()
                      if node.kind == "op" and n not in self.code and not node.tools
                      and n not in self.protected and n not in self.graders}
        helpers = {n for n in candidates
                   if self.graph.nodes[n].call in (COLLECT_CALL, FILE_TEXT_CALL)}
        candidates -= gated - helpers
        out: set[str] = set()
        changed = True
        while changed:
            changed = False
            for node_id in sorted(candidates - out):
                readers = self._readers(node_id)
                if readers and all(self._model(r) or (r in out and r in helpers)
                                   for r in readers):
                    out.add(node_id)
                    changed = True
        return out

    def _judges(self) -> dict[str, str]:
        """Each agent right before a gate whose reply only that gate reads, by gate: said
        in that gate's line."""
        out: dict[str, str] = {}
        for gate_id, gate in self.graph.nodes.items():
            if gate.kind != "gate":
                continue
            before = {e.source for e in self.graph.in_edges(gate_id)}
            for name in gate.reads:
                for maker in self.makers.get(("file", name)) or []:
                    if (maker in before and self._model(maker)
                            and self._readers(maker) <= {gate_id}):
                        out[gate_id] = maker
        return out

    def _hidden(self, node_id: str) -> bool:
        """Said inside another line: a folded op, a judge, a no-op step, a join."""
        node = self.graph.nodes[node_id]
        return (node_id in self.folded or node_id in self.judges.values()
                or node.kind == "join"
                or (node.kind == "op" and not (self.inputs[node_id] or node.writes
                                               or node.produces)))

    # ── readers ──────────────────────────────────────────────────────────
    @staticmethod
    def _sorted(keys: set[tuple[str, str]]) -> list[tuple[str, str]]:
        return sorted(keys, key=lambda key: (key[0] != "fact", key[1]))

    def _input(self, kind: str, name: str, depth: int = 0) -> str:
        made = self.makers.get((kind, name)) or []
        folded = [m for m in made if m in self.folded]
        if not folded or depth >= FOLD_DEPTH:
            return f"{name}@{','.join(made) or 'task'}"
        parts = []
        for op_id in folded:
            inner = [self._input(k, n, depth + 1) for k, n in self._sorted(self.inputs[op_id])]
            parts.append(f"{name} (via {op_id}" + (f": {', '.join(inner)})" if inner else ")"))
        return ", ".join(parts)

    def sees(self, node_id: str) -> str:
        node = self.graph.nodes[node_id]
        parts = [self._input(k, n) for k, n in self._sorted(self.inputs[node_id])]
        if node.kind == "agent" and (node.tool_policy != "none" or node.tools):
            where = "own workspace" if node.workspace is not None else "run dir"
            parts.insert(0, f"{where} (tools{': ' + ', '.join(node.tools) if node.tools else ''})")
        return ", ".join(parts)

    def told(self, node_id: str) -> str:
        """The knobs whose text reaches the step: its bound text fields (a code step's
        code), and the text knobs of the folded ops that build its inputs."""
        g = self.graph
        node = g.nodes[node_id]
        names: list[str] = []
        code_knob = ""
        for field in ("system_prompt_template", "prompt_template", "code"):
            knob = node.bound_knob(field)
            if knob and field == "code":
                code_knob = knob
            elif knob:
                names.append(knob)
        told: list[str] = []
        if code_knob:
            first = str(self.values.get(code_knob) or "").strip().splitlines()
            told.append(f"{code_knob} (code) \"{_clip(first[0] if first else '')}\"")
        elif node_id in self.code:
            first = str(getattr(node, "code", "") or "").strip().splitlines()
            told.append(f"code \"{_clip(first[0] if first else '')}\"")
        shared: set[str] = set()
        todo = [m for k, n in self._sorted(self.inputs[node_id])
                for m in self.makers.get((k, n)) or [] if m in self.folded]
        seen: set[str] = set()
        while todo:
            op_id = todo.pop(0)
            if op_id in seen:
                continue
            seen.add(op_id)
            op = g.nodes[op_id]
            texts = [k for k in op.knobs if isinstance(self.values.get(k), str)]
            names += texts
            if len(op.produces) + len(op.writes) > 1:
                shared.update(texts)
            todo += [m for k, n in self._sorted(self.inputs[op_id])
                     for m in self.makers.get((k, n)) or [] if m in self.folded]
        for name in dict.fromkeys(names):
            told.append(f"{name}{'*' if name in self.frozen else ''}"
                        f"{'?' if name in shared else ''} \"{_clip(self.values.get(name))}\"")
        return "; ".join(told)

    def step(self, node_id: str) -> str:
        node = self.graph.nodes[node_id]
        kind = "code" if node_id in self.code else node.kind
        sees, told = self.sees(node_id), self.told(node_id)
        outs = [*node.writes, *(f for f in node.produces if f not in node.writes)]
        marks = "".join(f"[{m}] " for m, on in (("protected", node_id in self.protected),
                                                ("optional", node_id in self.optional)) if on)
        return marks + "  ".join(part for part in (
            f"{node_id} {kind}", f"sees {sees}" if sees else "", f"told {told}" if told else "",
            f"-> {', '.join(outs)}" if outs else "") if part)

    def judge(self, gate_id: str) -> str:
        gate = self.graph.nodes[gate_id]
        judge = self.judges.get(gate_id)
        if judge:
            body = self.step(judge).split("  ", 1)
            return f"judge {judge}" + (f" ({body[1]})" if len(body) > 1 else "")
        if gate_id in self.code:
            return f"judge code ({self.told(gate_id) or 'its code'})"
        return f"judge {gate.evaluator or gate.gate_kind}"

    def loop(self, gate_id: str) -> str:
        node = self.graph.nodes[gate_id]
        back = sorted(e.target for e in self.graph.out_edges(gate_id)
                      if (gate_id, e.target) in self.back)
        cap = node.bound("max_iterations", self.values)
        knob = node.bound_knob("max_iterations")
        return LOOP_LINE.format(step=", ".join(self._shown_from(t) for t in back),
                                judge=self.judge(gate_id),
                                cap=f"{cap}{'*' if knob in self.frozen else ''}",
                                runs=cap + 1, knob=f", knob {knob}" if knob else "")

    def route(self, gate_id: str) -> str:
        before = sorted(self._shown_before(gate_id))
        labels = ", ".join(f"{e.condition}: {self._shown_from(e.target)}"
                           for e in self.graph.out_edges(gate_id) if e.condition)
        return (f"route(after {', '.join(before) or '-'}, {self.judge(gate_id)}, "
                f"{{{labels}}})")

    def parallel(self, fork_id: str) -> str:
        return f"parallel {fork_id} {{"

    def edges(self, node_id: str) -> str:
        """A step the walk could not place: generically, with its edges."""
        node = self.graph.nodes[node_id]
        outs = [f"{e.target}" + (f" [{e.condition}]" if e.condition else "")
                for e in self.graph.out_edges(node_id)]
        base = self.step(node_id) if node.kind in ("agent", "op") else f"{node_id} {node.kind}"
        return base + (f"  edges -> {', '.join(outs)}" if outs else "")

    # ── the walk ─────────────────────────────────────────────────────────
    def _succ(self, node_id: str) -> list[str]:
        g = self.graph
        node = g.nodes[node_id]
        out = [e.target for e in g.out_edges(node_id)]
        if node.kind == "fork":
            out += list(node.targets)
        if node.kind == "data" and node.subgraph_entry:
            out.append(node.subgraph_entry)
        return [t for t in dict.fromkeys(out) if t in g.nodes]

    def _back_edges(self) -> set[tuple[str, str]]:
        """Edges back to a step already on the way to their source (a loop's send-back):
        a depth-first walk from the start, then from any step it did not reach."""
        g = self.graph
        state: dict[str, int] = {}
        back: set[tuple[str, str]] = set()
        for root in [g.start, *sorted(g.nodes)]:
            if root not in g.nodes or root in state:
                continue
            stack = [(root, iter(self._succ(root)))]
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
                    stack.append((nxt, iter(self._succ(nxt))))
        return back

    def _forward(self, node_id: str) -> list[str]:
        return [e.target for e in self.graph.out_edges(node_id)
                if (node_id, e.target) not in self.back]

    def _next(self, node_id: str) -> str | None:
        node = self.graph.nodes[node_id]
        if node.kind == "gate":
            forward = self._forward(node_id)
            return forward[0] if len(forward) == 1 else None
        outs = [e.target for e in self.graph.out_edges(node_id) if e.condition is None]
        return outs[0] if len(outs) == 1 else None

    def _shown_before(self, node_id: str) -> set[str]:
        out: set[str] = set()
        sources = {e.source for e in self.graph.in_edges(node_id)} | {
            n for n, node in self.graph.nodes.items()
            if node.kind == "fork" and node_id in node.targets}
        for source in sources:
            # A loop's gate belongs to the step it sends back to: look through it too.
            looped = self.graph.nodes[source].kind == "gate" and any(
                (source, e.target) in self.back for e in self.graph.out_edges(source))
            out |= (self._shown_before(source) if self._hidden(source) or looped
                    else {source})
        return out

    def _shown_from(self, node_id: str) -> str:
        seen = set()
        while node_id in self.graph.nodes and self._hidden(node_id) and node_id not in seen:
            seen.add(node_id)
            nxt = self._next(node_id)
            if nxt is None:
                break
            node_id = nxt
        return node_id

    def reach(self, node_id: str) -> list[str]:
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
        """The join a fork's branches meet at: the first join along its first branch with
        no inner fork still open."""
        g = self.graph
        targets = list(g.nodes[fork_id].targets)
        node_id, depth, seen = (targets[0] if targets else None), 0, set()
        while node_id and node_id in g.nodes and node_id not in seen:
            seen.add(node_id)
            kind = g.nodes[node_id].kind
            if kind == "fork":
                depth += 1
            elif kind == "join":
                if depth == 0:
                    return node_id
                depth -= 1
            outs = (self._forward(node_id) if kind == "gate"
                    else [e.target for e in g.out_edges(node_id) if e.condition is None])
            node_id = outs[0] if outs else None
        return None

    def _in_order(self, nodes: list[str]) -> list[str]:
        g, left, order = self.graph, set(nodes), []
        while left:
            ready = sorted(n for n in left if not any(
                e.source in left and e.source != n and (e.source, n) not in self.back
                for e in g.in_edges(n)))
            if not ready:
                return order + sorted(left)
            order.append(ready[0])
            left.discard(ready[0])
        return order

    def lines(self) -> list[Line]:
        out: list[Line] = []
        self.shown = set()
        self._chain(self.graph.start, None, 0, "", out)
        for node_id in self._in_order([n for n in self.graph.nodes if n not in self.shown]):
            if node_id not in self.shown and not self._hidden(node_id):
                self.shown.add(node_id)
                out.append(Line(f"{FALLBACK}{node_id}", 0, self.edges(node_id), "", "edges"))
        return out

    def _one(self, node_id: str, depth: int, within: str, out: list[Line]) -> None:
        node = self.graph.nodes[node_id]
        self.shown.add(node_id)
        if self._hidden(node_id):
            return
        if node.kind == "gate":
            if any((node_id, e.target) in self.back for e in self.graph.out_edges(node_id)):
                out.append(Line(node_id, depth, self.loop(node_id), within, "loop"))
            elif len([e for e in self.graph.out_edges(node_id) if e.condition]) > 1:
                out.append(Line(node_id, depth, self.route(node_id), within, "route"))
            else:
                out.append(Line(f"{FALLBACK}{node_id}", depth, self.edges(node_id), within,
                                "edges"))
        elif node.kind in ("agent", "op"):
            out.append(Line(node_id, depth, self.step(node_id), within, "step"))
        else:
            out.append(Line(f"{FALLBACK}{node_id}", depth, self.edges(node_id), within,
                            "edges"))

    def _chain(self, start: str | None, stop: str | None, depth: int, within: str,
               out: list[Line]) -> None:
        g = self.graph
        node_id = start
        while node_id and node_id != stop and node_id in g.nodes:
            node = g.nodes[node_id]
            if node_id in self.shown and node.kind != "join":
                return
            if node.kind == "data":
                self.shown.add(node_id)
                out.append(Line(node_id, depth, f"per item ({node_id}"
                                + (f" -> {', '.join(node.writes)}" if node.writes else "")
                                + ") {", within, "step"))
                self._chain(node.subgraph_entry, None, depth + 1, f"per item {node_id}", out)
                out.append(Line(f"{node_id}/", depth, "}", within, "step"))
            elif node.kind == "fork":
                self.shown.add(node_id)
                join = self._join_of(node_id)
                out.append(Line(node_id, depth, self.parallel(node_id), within, "parallel"))
                for target in node.targets:
                    self._chain(target, join, depth + 1, f"parallel {node_id}", out)
                out.append(Line(f"{node_id}/", depth, f"}} -> {join or 'end'}", within,
                                "parallel"))
                if not join:
                    return
                self.shown.add(join)
                node_id = self._next(join)
                continue
            elif node.kind == "gate" and not any((node_id, e.target) in self.back
                                                 for e in g.out_edges(node_id)) and len(
                    [e for e in g.out_edges(node_id) if e.condition]) > 1:
                self._one(node_id, depth, within, out)
                edges = [e for e in g.out_edges(node_id) if e.condition]
                reaches = [self.reach(e.target) for e in edges]
                common = set(reaches[0]).intersection(*map(set, reaches[1:]))
                rejoin = next((n for n in reaches[0] if n in common), None)
                for edge in edges:
                    if edge.target != rejoin and edge.target not in self.shown:
                        out.append(Line(f"{node_id}:{edge.condition}", depth + 1,
                                        f"on {edge.condition}:", within, "route"))
                        self._chain(edge.target, rejoin, depth + 2,
                                    f"route {node_id} on {edge.condition}", out)
                node_id = rejoin
                continue
            else:
                self._one(node_id, depth, within, out)
            node_id = self._next(node_id)


#: Each reader by the name operations give it in ``EditSpec.reads_as``: what draws that
#: kind of line. ``tests/test_pipeline_view.py`` holds the two sets equal.
READERS: dict[str, Callable[..., str]] = {
    "step": _View.step, "sees": _View.sees, "told": _View.told, "loop": _View.loop,
    "route": _View.route, "parallel": _View.parallel, "edges": _View.edges,
}


#: The key of the one line a view that could not be drawn has.
UNAVAILABLE = "(unavailable)"

#: How each block of the view reads, said once in the request's legend; the test holds
#: each to what its reader draws (a loop of cap N runs its step N+1 times).
FORMS = {
    "parallel": "parallel fork { ... } -> join",
    "loop": LOOP_LINE.format(step="step", judge="judge ...", cap="N", runs="N+1", knob=""),
    "route": "route(after step, judge ..., {label: step})",
    "per item": "per item (data) { ... }",
}


def view_lines(state: Mapping[str, Any], protected: Sequence[str] = (),
               graders: Sequence[str] = ()) -> list[Line]:
    """The pipeline view of ``state``, line by line. A graph the view cannot draw gives one
    line saying so, with why (and the error is logged), never nothing: the request and a
    candidate's record then say the view is missing rather than silently leaving it out."""
    try:
        return _View(state, protected, graders).lines()
    except Exception as exc:  # noqa: BLE001 - said, never raised: the validator judges graphs
        import logging

        logging.getLogger(__name__).warning("pipeline view unavailable: %s", exc, exc_info=True)
        return [Line(UNAVAILABLE, 0, f"pipeline view unavailable: {type(exc).__name__}: {exc}")]


def render(state: Mapping[str, Any], protected: Sequence[str] = (),
           graders: Sequence[str] = ()) -> str:
    """The view as text, two spaces of indent per level."""
    return "\n".join("  " * line.depth + line.text
                     for line in view_lines(state, protected, graders))


def fallback_count(state: Mapping[str, Any], protected: Sequence[str] = (),
                   graders: Sequence[str] = ()) -> int:
    """How many lines no reader recognised (drawn generically)."""
    return sum(line.key.startswith(FALLBACK)
               for line in view_lines(state, protected, graders))


def _structural(key: str) -> bool:
    return key.endswith("/") or (":" in key and not key.startswith(FALLBACK))


def changed_lines(before: Mapping[str, Any], after: Mapping[str, Any],
                  protected: Sequence[str] = (), graders: Sequence[str] = ()) -> list[str]:
    """What an edit built, as the view's own lines: each new (``+``) or changed (``~``)
    line of ``after``'s view against ``before``'s, keyed by step, gate or fork, with the
    block it sits in, and each one gone (``- removed``)."""
    before_lines = view_lines(before, protected, graders)
    new = view_lines(after, protected, graders)
    missing = [line.text for line in (*before_lines, *new) if line.key == UNAVAILABLE]
    if missing:  # what an edit built cannot be said without both views: said so
        return [f"! {missing[0]}"]
    old = {line.key: line for line in before_lines}
    out = []
    for line in new:
        if _structural(line.key):
            continue
        where = f"[in {line.within}] " if line.within else ""
        if line.key not in old:
            out.append(f"+ {where}{line.text}")
        elif old[line.key].text != line.text or old[line.key].within != line.within:
            out.append(f"~ {where}{line.text}")
    gone = sorted(set(old) - {line.key for line in new})
    out += [f"- removed {key.removeprefix(FALLBACK)}" for key in gone if not _structural(key)]
    return out
