"""The pipeline view (`nn.pipeline_view`): one line per step, read in the operators' words.

Reading mirrors writing and must not drift: every operation in the catalogue names the
readers that render what it builds (``EditSpec.reads_as``) and every reader is named by an
operation; applying an operation changes the view only in the lines it describes, which
name its handles; a graph built from core operators needs no fallback line. The request
opens its structure part with the view, and a candidate's record and the next request say
what its edit built as the view's new and changed lines. Shapes: nested forks, optional
branches, a loop inside a fork branch, a route whose ways rejoin, per-item bodies, several
data passes, code-verdict gates, tools, code steps, frozen and protected marks.
"""

from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path

import pytest

import structure_fuzz as sf
from refactory_dsh import registry
from refactory_dsh.nn import core_ops, validate_state
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import pipeline_view as pv
from refactory_dsh.nn.graph import Graph
from refactory_dsh.nn.validate import consumed
from refactory_dsh.optim import Candidate, base


def _mode(name: str):
    _package, mode = registry.resolve(Path("."), name)
    return mode()


def _seed(name: str) -> dict:
    return sf.Applier(sf.seeds()[name]).seed_state


def _lines(state, protected=()):
    return {line.key: line.text for line in pv.view_lines(state, protected)}


# ── the drift guard: every operation has a reader, every reader an operation ──


def test_every_operation_names_a_reader_and_every_reader_is_named():
    specs = nn_edits._specs()
    missing = sorted(name for name, spec in specs.items() if not spec.reads_as)
    assert not missing, f"operations without a reader (EditSpec.reads_as): {missing}"
    named = {reader for spec in specs.values() for reader in spec.reads_as}
    assert named - set(pv.READERS) == set(), f"unknown readers: {named - set(pv.READERS)}"
    assert set(pv.READERS) - named == set(), f"readers no operation names: {set(pv.READERS) - named}"
    for name, spec in core_ops.OPERATORS.items():
        assert spec["reads_as"] == specs[name].reads_as, name


def _part(text: str, word: str) -> str:
    """A step line's ``sees ...`` or ``told ...`` part."""
    found = re.search(rf"  {word} (.*?)(?=  (?:sees|told|-> )|$)", text)
    return found.group(1) if found else ""


def _drawn_by(reader: str, old: "pv.Line | None", new: "pv.Line | None") -> bool:
    """Whether a changed line is what ``reader`` draws: its own lines; for ``sees`` and
    ``told``, a step line whose part of that name changed; for ``parallel``, a line in a
    parallel block or the block itself."""
    line = new or old
    if reader in ("sees", "told"):
        return line.reader == "step" and _part(old.text if old else "", reader) != _part(
            new.text if new else "", reader)
    if reader == "parallel":
        return line.reader == "parallel" or line.within.startswith("parallel")
    return line.reader == reader


#: Each operation the random draws do not reach, applied by hand to a seed it fits.
_BY_HAND = {
    "add_edge": ("fuzz_gate_loop", lambda s: [
        {"op": "add_node", "position": {"detached": True},
         "node": {"id": "alt", "kind": "op", "call": s["graph"]["nodes"]["finish"]["call"],
                  "reads": ["work.txt"], "writes": ["alt.txt"], "produces": ["alt_done"]}},
        {"op": "add_edge", "source": "review", "target": "alt", "condition": "escalate"}]),
    "insert_after": ("fuzz_draft_review", lambda s: [
        {"op": "insert_after", "target": "draft", "node": {
            **{k: v for k, v in s["graph"]["nodes"]["review"].items()
               if k not in ("field_bindings",)},
            "id": "review2", "writes": ["review2.txt"], "produces": ["review2"]}}]),
    "insert_before": ("fuzz_draft_review", lambda s: [
        {"op": "insert_before", "target": "finalize", "node": {
            **{k: v for k, v in s["graph"]["nodes"]["review"].items()
               if k not in ("field_bindings",)},
            "id": "review2", "writes": ["review2.txt"], "produces": ["review2"]}}]),
    "add_node": ("fuzz_draft_review", lambda s: [
        {"op": "add_node", "position": {"after": "draft"}, "node": {
            **{k: v for k, v in s["graph"]["nodes"]["review"].items()
               if k not in ("field_bindings",)},
            "id": "review2", "writes": ["review2.txt"], "produces": ["review2"]}}]),
    "wrap": ("fuzz_draft_review", lambda s: [
        {"op": "wrap", "target": "review", "node": {
            **{k: v for k, v in s["graph"]["nodes"]["review"].items()
               if k not in ("field_bindings",)},
            "id": "watch", "writes": ["watch.txt"], "produces": ["watch"]}}]),
    "parallel": ("fuzz_draft_review", lambda s: [
        {"op": "parallel", "target": "draft", "branches": [
            {"id": "side_a", "kind": "agent", "writes": ["side_a.txt"],
             "prompt_template": "First angle on {{brief}}"},
            {"id": "side_b", "kind": "agent", "writes": ["side_b.txt"],
             "prompt_template": "Second angle on {{brief}}"}]}]),
    # A knob alone is never a valid state (an unread knob is refused): each is checked in
    # the edit it completes -- a new step told by a new knob, a step's knob removed with it.
    "add_knob": ("fuzz_draft_review", lambda s: [
        {"op": "add_knob", "name": "careful", "spec": {"default": "Be careful.", "kind": "str"}},
        {"op": "insert_after", "target": "draft", "node": {
            **{k: v for k, v in s["graph"]["nodes"]["review"].items()
               if k not in ("field_bindings",)},
            "id": "review2", "writes": ["review2.txt"], "produces": ["review2"],
            "knobs": ["careful"], "field_bindings": [["system_prompt_template", "careful"]]}}]),
}

#: Edits that make the state the by-hand case starts from (``remove_knob`` takes away a
#: step and then its own knob).
_BY_HAND_SETUP = {
    "remove_knob": ("fuzz_draft_review", lambda s: [
        {"op": "add_knob", "name": "careful", "spec": {"default": "Be careful.", "kind": "str"}},
        {"op": "insert_after", "target": "finalize", "node": {
            **{k: v for k, v in s["graph"]["nodes"]["review"].items()
               if k not in ("field_bindings",)},
            "id": "review2", "writes": ["review2.txt"], "produces": ["review2"],
            "knobs": ["careful"], "field_bindings": [["system_prompt_template", "careful"]]}}],
        lambda s: [{"op": "remove_node", "id": "review2"},
                   {"op": "add_edge", "source": "finalize", "target": "ship", "condition": None},
                   {"op": "remove_knob", "name": "careful"}]),
}


def _op_draws(seed: str) -> list[tuple[str, dict, dict]]:
    state = _seed(seed)
    rng = random.Random(f"readers:{seed}")
    out = []
    for _ in range(160):
        edit = nn_edits.random_edit(state, preset="broad", rng=rng, operators=rng.random() < 0.5)
        if edit is None or nn_edits.precheck(edit, state) is not None:
            continue
        try:
            after = nn_edits.apply(edit, state)
        except nn_edits.EditError:
            continue
        if not validate_state(after):
            out.append((edit["op"], state, after))
    return out


def test_each_operation_s_reader_draws_what_it_changes():
    """Reading mirrors writing in what is drawn, not only in names: after each operation
    (random draws on every seed, and by hand for those the draws miss), some changed line
    is drawn by a reader the operation declares (`EditSpec.reads_as`)."""
    specs = nn_edits._specs()
    applied = [draw for seed in sorted(sf.seeds()) for draw in _op_draws(seed)]
    cases = [(name, seed, lambda s: [], edits) for name, (seed, edits) in _BY_HAND.items()]
    cases += [(name, seed, setup, edits)
              for name, (seed, setup, edits) in _BY_HAND_SETUP.items()]
    for name, seed, setup, edits in cases:
        state = _seed(seed)
        for edit in setup(state):
            state = nn_edits.apply(edit, state)
        after = state
        for edit in edits(state):
            after = nn_edits.apply(edit, after)
        assert validate_state(after) == [], (name, validate_state(after))
        applied.append((name, state, after))
    checked: set[str] = set()
    for name, before, after in applied:
        old = {line.key: line for line in pv.view_lines(before)}
        new = {line.key: line for line in pv.view_lines(after)}
        changed = [(old.get(k), new.get(k)) for k in set(old) | set(new)
                   if (old.get(k) and old[k].text) != (new.get(k) and new[k].text)]
        if not changed:
            continue  # nothing drawn changed (a knob no step reads, say)
        checked.add(name)
        assert any(_drawn_by(reader, o, n) for reader in specs[name].reads_as
                   for o, n in changed), (name, specs[name].reads_as,
                                          sorted({(n or o).reader for o, n in changed}))
    # Every operation in the catalogue is checked by what it drew.
    assert checked == set(specs), sorted(set(specs) - checked)


# ── the view ────────────────────────────────────────────────────────────────


def test_the_s4_and_s1_views_say_each_step_in_one_line_under_2k():
    s4 = _mode("livemath/skill_graph")
    text = pv.render(s4.state_dict(), list(s4.edit_protected()))
    assert len(text) < 2000, len(text)
    lines = text.splitlines()
    assert lines[0] == "per item (data -> current_item.json) {"
    assert lines[1].startswith("  compare agent  sees compare_prompt (via prepare: "
                               "current_item.json@data)  told compare__system_prompt \"")
    # prepare declares which output carries compare_task: no `?`.
    assert "; compare_task \"" in lines[1] and lines[1].endswith("-> compare_notes.txt")
    assert lines[2] == "  parallel step_fork {"
    assert lines[3].startswith("    precision agent  sees precision_prompt (via notes_prompts: "
                               "compare_notes.txt@compare)")
    assert lines[5] == "  } -> step_join"
    assert lines[6].startswith("  decide agent  sees decide_prompt (via decide_prompt: "
                               "compare_notes.txt@compare, hypotheses_notes.txt@hypotheses, "
                               "precision_notes.txt@precision)")
    assert lines[7] == ("  [protected] extract_answer op  sees answer.txt@decide, "
                        "current_item.json@data  -> answer_label, answer_valid")
    assert lines[8:] == ["}", "[protected] finish op  -> result.json, finished"]
    s1 = _mode("livemath/solver")
    text1 = pv.render(s1.state_dict(), list(s1.edit_protected()))
    assert len(text1) < 2000
    assert "solve agent  sees skill_section (via format_prompt: current_item.json@data), " \
        "solver_prompt (via format_prompt: current_item.json@data)  told " \
        "solve__system_prompt* \"" in text1, text1  # * frozen
    assert pv.fallback_count(s4.state_dict()) == pv.fallback_count(s1.state_dict()) == 0


def test_folding_is_generic_assemblers_into_model_steps_only():
    """A plain op is folded only when model steps alone read it; one another op reads (or a
    protected step) keeps its own line, as code does; an agent with tools sees the run dir."""
    flow = _lines(_seed("fuzz_dataflow"))  # load is read by clean and emit, both ops
    assert "load" in flow and "emit" in flow
    assert "setup" in _lines(_seed("fuzz_fork_join"))  # feeds an agent and an op
    s4 = _mode("livemath/skill_graph")
    assert "prepare" in _lines(s4.state_dict(), ("data", "extract_answer", "finish", "prepare"))
    tools = _lines(_seed("fuzz_tool_step"))
    assert any(re.search(r"sees (run dir|own workspace) \(tools", line)
               for line in tools.values()), tools


def _draws(seed: str, count: int = 40, *, core_only: bool = False) -> list[dict]:
    state = _seed(seed)
    rng = random.Random(f"view:{seed}:{core_only}")
    states = [state]
    for _ in range(count):
        edit = nn_edits.random_edit(state, preset="broad", rng=rng, operators=True)
        if edit is None or (core_only and edit["op"] not in nn_edits.CORE_OPS) \
                or nn_edits.precheck(edit, state) is not None:
            continue
        try:
            after = nn_edits.apply(edit, state)
        except nn_edits.EditError:
            continue
        if not validate_state(after):  # a graph the validator refuses is never shown
            states.append(after)
    return states


@pytest.mark.parametrize("seed", sorted(sf.seeds()))
def test_every_input_gate_fork_and_edge_is_shown_on_seeds_and_random_edits(seed):
    """On each seed graph and 40 operator draws from it (those the validator accepts): each
    model or code step has a line (a judge's is inside its gate's) with every input it takes
    in; every gate is a loop or route line and every fork a parallel block; along every
    forward edge the walk reaches, the target's line comes after the source's."""
    for current in _draws(seed):
        view = pv._View(current, ())
        lines = view.lines()
        text = {line.key: line.text for line in lines}
        order = {line.key: index for index, line in enumerate(lines)}
        graph = view.graph
        for node_id, node in graph.nodes.items():
            if node.kind == "agent" or node_id in view.code:
                gate = next((g for g, j in view.judges.items() if j == node_id), None)
                line = text.get(gate) if gate else text.get(node_id)
                assert line is not None, (seed, node_id, list(text))
                for _kind, name in consumed(node, view.values):
                    assert f"{name}@" in line or f"{name} (via " in line, (seed, node_id, name)
            if node.kind == "gate":
                assert re.match(r"(loop|route)\(", text.get(node_id, "")), (seed, node_id)
            if node.kind == "fork":
                assert text.get(node_id) == f"parallel {node_id} {{", (seed, node_id)
        reached = set(view.reach(graph.start))
        for edge in graph.edges:
            a, b = edge.source, edge.target
            if a in order and b in order and a in reached and (a, b) not in view.back:
                assert order[b] > order[a], (seed, a, b)


def _gate_with_ways() -> dict:
    """fuzz_gate_loop's review gate with a send-back and two ways forward (validates)."""
    state = _seed("fuzz_gate_loop")
    finish = state["graph"]["nodes"]["finish"]
    state = nn_edits.apply({"op": "add_node", "position": {"detached": True},
                            "node": {"id": "alt", "kind": "op", "call": finish["call"],
                                     "reads": ["work.txt"], "writes": ["alt.txt"],
                                     "produces": ["alt_done"]}}, state)
    return nn_edits.apply({"op": "add_edge", "source": "review", "target": "alt",
                           "condition": "escalate"}, state)


def _ways_meet_above() -> dict:
    """The gate with ways: escalate goes alt -> finish, proceed goes to finish (drawn above
    by then), and a third way ends: proceed says ``-> finish (above)``."""
    state = _gate_with_ways()
    finish = state["graph"]["nodes"]["finish"]
    state = nn_edits.apply({"op": "add_edge", "source": "alt", "target": "finish",
                            "condition": None}, state)
    state = nn_edits.apply({"op": "add_node", "position": {"detached": True},
                            "node": {"id": "stop_here", "kind": "op", "call": finish["call"],
                                     "reads": ["work.txt"], "writes": ["stopped.txt"],
                                     "produces": ["stopped"]}}, state)
    return nn_edits.apply({"op": "add_edge", "source": "review", "target": "stop_here",
                           "condition": "drop"}, state)


def _shapes() -> dict[str, dict]:
    return {"two passes": _two_passes(_seed("fuzz_data_pass")), "gate ways": _gate_with_ways(),
            "ways meet above": _ways_meet_above(),
            **{name: _mode(name).state_dict()
               for name in ("livemath/skill_graph", "livemath/solver")}}


class _Reading:
    """What a reader of the view's lines takes as the graph's edges: each line after
    another at its depth follows it; a block's first line follows its header; a route's or
    loop's labelled ways follow the gate; what comes after a block follows the tails of its
    ways (a way ending in ``-> end`` has none); ``-> X (above)`` goes to X; a loop line
    sends back to its step; a fallback line lists its edges. Steps the view folds into
    another line (folded ops, judges, joins, no-op steps) may stand between two read
    steps. Read from the lines (keys, depths, the route/loop/fallback text), never from the
    drawer's own records."""

    def __init__(self, state: dict) -> None:
        self.view = view = pv._View(state, ())
        self.lines = view.lines()
        self.graph = view.graph
        self.wrong: set[tuple[str, str]] = set()   # read, but no such path in the graph
        self.covered: set[tuple[str, str]] = set()  # graph edges some reading uses
        self.i = 0
        start = self.graph.start
        first = self._first(start)
        if first and first != start:  # the run starts at steps folded into the first line
            self.covered.update(self._path(start, first) or [])
        self._end(self._seq(0, set()))
        for line in self.lines:
            if line.key.startswith(pv.FALLBACK) and "edges -> " in line.text:
                source = line.key.removeprefix(pv.FALLBACK)
                for target in re.findall(r"(\w+)(?: \[[^\]]*\])?",
                                         line.text.split("edges -> ", 1)[1]):
                    self._follow({source}, target)
            if line.reader == "loop" and "loop(" in line.text:
                gate = line.key
                for e in self.graph.out_edges(gate):
                    if (gate, e.target) in view.back and f"loop({view._shown_from(e.target)}" \
                            in line.text:
                        self.covered.add((gate, e.target))

    def _path(self, a: str, b: str) -> list[tuple[str, str]] | None:
        """The edges from ``a`` to ``b`` through folded steps only (forward edges)."""
        todo, seen = [(a, [])], {a}
        while todo:
            node, path = todo.pop(0)
            for e in self.graph.out_edges(node):
                if (node, e.target) in self.view.back:
                    continue
                step = [*path, (node, e.target)]
                if e.target == b:
                    return step
                if e.target not in seen and self.view._hidden(e.target):
                    seen.add(e.target)
                    todo.append((e.target, step))
            if self.graph.nodes[node].kind == "fork":
                for t in self.graph.nodes[node].targets:
                    if t == b:
                        return [*path, (node, t)]
                    if t not in seen and self.view._hidden(t):
                        seen.add(t)
                        todo.append((t, [*path, (node, t)]))
        return None

    def _follow(self, tails: set[str], b: str) -> None:
        for t in tails:
            path = self._path(t, b)
            if path is None:
                self.wrong.add((t, b))
            else:
                self.covered.update(path)

    def _node(self, key: str) -> str | None:
        return key if key in self.graph.nodes else None

    def _seq(self, depth: int, tails: set[str], heads: tuple[str, ...] = ()) -> set[str]:
        """Read the lines at ``depth`` from here; ``tails`` lead into the first. Returns
        what leads on after the last (empty when the chain ended)."""
        g = self.graph
        while self.i < len(self.lines):
            line = self.lines[self.i]
            if line.depth < depth or (line.depth == depth and heads and line.key in heads
                                       and self._started):
                return tails
            if line.depth > depth:  # belongs to an item handled below; never expected here
                self.i += 1
                continue
            key = line.key
            self.i += 1
            self._started = True
            if key.startswith(pv.FALLBACK):
                tails = set()
            elif key.startswith("->"):
                self._follow(tails, key[2:])
                tails = set()
            elif key.endswith(":end"):
                self._end(tails)
                tails = set()
            elif key.endswith("/"):
                self._end(tails)
                return set()  # a block's closing line: its header's reading continues
            elif self._node(key) and g.nodes[key].kind == "fork":
                self._follow(tails, key)
                join = self.view._join_of(key)
                out: set[str] = set()
                firsts = [self._first(t) for t in g.nodes[key].targets]
                for first in firsts:
                    if first is None:
                        continue
                    self._started = False
                    out |= self._seq(depth + 1, {key}, tuple(f for f in firsts if f))
                closing = self.lines[self.i] if self.i < len(self.lines) else None
                if closing is not None and closing.key == f"{key}/":
                    self.i += 1
                if join:
                    self._follow(out, join) if out else None
                    self.covered.add((key, join)) if any(
                        e.target == join for e in g.out_edges(key)) else None
                    tails = {join}
                else:
                    tails = set()
            elif self._node(key) and g.nodes[key].kind == "data":
                self._follow(tails, key)
                entry = g.nodes[key].subgraph_entry
                self._started = False
                first = self._first(entry) if entry else None
                if first and first != entry:
                    path = self._path(entry, first)
                    self.covered.update(path or [])
                self._end(self._seq(depth + 1, set()))
                if self.i < len(self.lines) and self.lines[self.i].key == f"{key}/":
                    self.i += 1
                tails = {key}
            elif self._node(key):
                if tails:
                    self._follow(tails, key)
                tails = {key}
                node = g.nodes[key]
                if node.kind == "gate":
                    labelled = {e.condition: e.target for e in g.out_edges(key)
                                if e.condition and (key, e.target) not in self.view.back}
                    drawn: set[str] = set()
                    out = set()
                    if self.i < len(self.lines) and self.lines[self.i].key == f"{key}:ways":
                        self.i += 1  # "then, by its verdict:", at the gate's own depth
                    while self.i < len(self.lines) and self.lines[self.i].depth == depth + 1 \
                            and self.lines[self.i].key.startswith(f"{key}:"):
                        way = self.lines[self.i].key.split(":", 1)[1]
                        self.i += 1
                        if way == "ways":
                            continue
                        drawn.add(way)
                        self._started = False
                        out |= self._seq(depth + 2, {key})
                    if self.i < len(self.lines) and self.lines[self.i].key == f"{key}:ways":
                        self.i += 1
                    if "route(" in line.text or drawn:
                        # Undrawn labels go to what follows; drawn ways' tails go on too.
                        tails = out | ({key} if set(labelled) - drawn else set())
                        if not (set(labelled) - drawn):
                            tails = out
            else:
                tails = set()
        return tails

    def _end(self, tails: set[str]) -> None:
        """A chain that stops with ``tails``: what they lead to through folded steps that
        end the pass (a no-op step) reads as the chain ending there."""
        for t in tails:
            todo = [t]
            seen = {t}
            while todo:
                node = todo.pop()
                for e in self.graph.out_edges(node):
                    if (node, e.target) in self.view.back or e.target in seen:
                        continue
                    if self.view._hidden(e.target) and not any(
                            (e.target, x.target) not in self.view.back and not
                            self.view._hidden(x.target)
                            for x in self.graph.out_edges(e.target)):
                        self.covered.add((node, e.target))
                        seen.add(e.target)
                        todo.append(e.target)

    def _first(self, node_id: str | None) -> str | None:
        """The first step a chain from ``node_id`` draws: itself, or past folded steps."""
        seen = set()
        while node_id and node_id in self.graph.nodes and self.view._hidden(node_id) \
                and node_id not in seen:
            seen.add(node_id)
            node_id = self.view._next(node_id)
        return node_id


def _reading_problems(state: dict) -> tuple[set, set]:
    reading = _Reading(state)
    edges = {(e.source, e.target) for e in reading.graph.edges}
    fork_targets = {(n.id, t) for n in reading.graph.nodes.values() if n.kind == "fork"
                    for t in n.targets}
    return reading.wrong, (edges | fork_targets) - reading.covered


@pytest.mark.parametrize("seed", sorted(sf.seeds()))
def test_the_lines_read_as_exactly_the_graph_s_edges_on_seeds_and_random_edits(seed):
    """Read from the drawn lines alone: every edge they imply is in the graph (soundness:
    a way that ends says ``-> end``, so nothing after its block reads as following it) and
    every edge of the graph is read from some line (completeness)."""
    for current in _draws(seed):
        wrong, unread = _reading_problems(current)
        assert wrong == set(), (seed, sorted(wrong), pv.render(current))
        assert unread == set(), (seed, sorted(unread), pv.render(current))


def test_the_lines_read_as_exactly_the_graph_s_edges_on_the_shapes():
    for name, state in _shapes().items():
        assert validate_state(state) == [] or name == "two passes", name
        wrong, unread = _reading_problems(state)
        assert wrong == set() and unread == set(), (name, wrong, unread, pv.render(state))


def test_routes_rejoining_inside_another_route_keep_every_way_and_end_where_they_end():
    """fuzz_gates: approve's reject way (redo) rejoins at drop, which triage's reject way
    also reaches; triage's proceed way (archive) ends the run and says so, so drop does not
    read as following archive."""
    state = _seed("fuzz_gates")
    text = pv.render(state)
    assert "on reject:\n    redo agent" in text and text.rstrip().endswith("drop op  -> dropped")
    assert "archive op  sees answer.txt@ship  -> finished\n        -> end" in text
    wrong, unread = _reading_problems(state)
    assert ("archive", "drop") not in wrong and wrong == set() and unread == set()


def test_a_way_that_meets_a_step_drawn_above_says_so():
    text = pv.render(_ways_meet_above())
    assert "-> finish (above)" in text, text


def test_a_loop_gate_s_ways_forward_are_shown_beside_its_loop_line():
    text = pv.render(_gate_with_ways())
    assert "loop(work, judge structure_fuzz_graphs:proceed, max 3 -> 4 runs, past the cap: " \
        "halt)\nthen, by its " \
        "verdict:" in text
    assert "on escalate:\n    alt op" in text and "on proceed:\n    finish op" in text


# ── round trip: an operation changes the lines it describes, naming its handles ──


def _touched(before: dict, after: dict) -> set[str]:
    """The nodes whose own state or told text changed, that came or went, or that read
    what such a node makes (an input's maker changed)."""
    old, new = before["graph"]["nodes"], after["graph"]["nodes"]
    knobs_old, knobs_new = before.get("knobs") or {}, after.get("knobs") or {}
    out = set(old) ^ set(new)

    def edges(state):
        return {(e["source"], e["target"], e.get("condition")) for e in state["graph"]["edges"]}
    out |= {n for e in edges(before) ^ edges(after) for n in e[:2]}  # its edges changed
    for node_id in set(old) & set(new):
        knobs = set(old[node_id].get("knobs") or ()) | {
            k for _f, k in old[node_id].get("field_bindings") or ()}
        if old[node_id] != new[node_id] or any(knobs_old.get(k) != knobs_new.get(k)
                                               for k in knobs):
            out.add(node_id)
    view, was = pv._View(after, ()), pv._View(before, ())
    for _ in range(4):  # what a changed, new or gone node makes: its readers' lines show it
        made = {key for graph in (view.graph, was.graph) for n in out if n in graph.nodes
                for key in ({("fact", f) for f in graph.nodes[n].produces}
                            | {("file", f) for f in graph.nodes[n].writes})}
        out |= {n for n, taken in view.inputs.items() if taken & made}
    # A step whose readers the edit changed: whether it is folded into them may change,
    # and so may the lines of those it feeds.
    common = set(view.graph.nodes) & set(was.graph.nodes)
    flipped = {n for n in common if (n in view.folded) != (n in was.folded)}
    out |= flipped | {r for n in flipped for r in view._readers(n) | was._readers(n)
                      if r in view.graph.nodes}
    # A gate naming a step whose fold changed names the next shown step instead.
    out |= {e.source for e in view.graph.edges if e.target in flipped}
    return out


@pytest.mark.parametrize("seed", sorted(sf.seeds()))
def test_an_operation_changes_only_the_lines_it_describes_and_names_its_handles(seed):
    """For each core-operator draw on the seed: every changed line belongs to a step the
    edit touched (or one fed by it); the changes are said in the operator's own reader
    (a loop line, a route line, a line in the fork's parallel block); and they name the
    step the operator was given."""
    state = _seed(seed)
    rng = random.Random(f"roundtrip:{seed}")
    checked = 0
    for _ in range(80):
        edit = nn_edits.random_edit(state, preset="broad", rng=rng, operators=True)
        if edit is None or edit["op"] not in nn_edits.CORE_OPS \
                or nn_edits.precheck(edit, state) is not None:
            continue
        try:
            after = nn_edits.apply(edit, state)
        except nn_edits.EditError:
            continue
        if validate_state(after):
            continue
        checked += 1
        old, new = _lines(state), _lines(after)
        changed = {k.removeprefix(pv.FALLBACK) for k in set(old) | set(new)
                   if old.get(k) != new.get(k) and not k.endswith("/") and ":" not in k}
        touched = _touched(state, after)
        assert changed <= touched, (seed, edit, changed - touched)
        text = "\n".join(pv.changed_lines(state, after))
        assert text, (seed, edit)
        reader = nn_edits._specs()[edit["op"]].reads_as[0]
        said = {"loop": "loop(", "route": "route(after ",
                "parallel": f"[in parallel {edit.get('fork', '')}"}
        if edit["op"] == "serialize":
            assert f"- removed {edit['fork']}" in text, (seed, edit, text)
        elif reader in said and edit["op"] != "parallelize":
            assert said[reader] in text, (seed, edit, text)
        step = edit.get("step") or edit.get("after")
        if isinstance(step, str) and edit["op"] in ("loop", "route", "rewire", "insert"):
            assert step in text, (seed, edit, text)
        if edit["op"] == "remove":
            assert f"- removed {edit['step']}" in text, (seed, edit, text)
    assert checked >= 3, (seed, checked)


def test_graphs_built_from_core_operators_need_no_fallback_and_seeds_are_counted():
    """Hand-written seeds are counted (the PR reports it); a graph only core operators
    changed from a seed adds no fallback line."""
    counts = {name: pv.fallback_count(_seed(name)) for name in sf.seeds()}
    assert counts == {name: 0 for name in sf.seeds()}, counts
    for name in sf.seeds():
        for current in _draws(name, 40, core_only=True):
            assert pv.fallback_count(current) == 0, (name, pv.render(current))


def test_a_loop_line_round_trips_its_step_and_cap():
    s1 = _mode("livemath/solver").state_dict()
    edit = {"op": "loop", "step": "solve", "judge": "agent", "text": "Check it.",
            "max_iterations": 3}
    line = _lines(nn_edits.apply(edit, s1))["solve_check"]
    found = re.match(r"loop\((\w+), judge (\w+) .*, max (\d) -> (\d) runs, knob (\w+)\)$", line)
    assert found, line
    step, judge, cap, runs, knob = found.groups()
    assert (step, judge, int(cap), int(runs), knob) == ("solve", "solve_check_judge", 3, 4,
                                                        "solve_check__max_iterations")
    again = {"op": "loop", "step": step, "judge": "agent", "text": "Check it.",
             "max_iterations": int(cap)}
    assert _lines(nn_edits.apply(again, s1))["solve_check"] == line


def test_a_route_line_round_trips_its_handles_and_its_ways_rejoin():
    state = _seed("fuzz_draft_review")
    edit = {"op": "route", "after": "draft", "judge": "agent", "text": "Choose.",
            "to": {"proceed": "review", "revise": {"kind": "agent", "text": "Revise it.",
                                                   "sees": ["task"]}}}
    after = nn_edits.apply(edit, state)
    text = pv.render(after)
    shown = _lines(after)
    line = next(t for t in shown.values() if t.startswith("route("))
    found = re.match(r"route\(after (\w+), judge \w+ .*, \{(.*)\}\)$", line)
    assert found and found.group(1) == "draft", line
    labels = dict(part.split(": ") for part in found.group(2).split(", "))
    assert labels["proceed"] == "review"
    assert shown[labels["revise"]].startswith(f"{labels['revise']} agent")  # not its collect step
    assert re.search(r"\n  on revise:\n    draft_revise agent .*\nreview agent", text), text
    again = {**edit, "to": {"proceed": labels["proceed"], "revise": edit["to"]["revise"]}}
    assert _lines(nn_edits.apply(again, state)) == shown


def test_a_branch_s_sees_line_names_exactly_what_it_was_given():
    s4 = _mode("livemath/skill_graph").state_dict()
    edit = {"op": "branch", "fork": "step_fork", "kind": "agent", "text": "{{user_prompt}}",
            "sees": ["user_prompt"], "id": "fresh"}
    after = nn_edits.apply(edit, s4)
    graph = Graph.from_state(after["graph"])
    values = {k: e["value"] for k, e in after["knobs"].items()}
    assert consumed(graph.nodes["fresh"], values) == {("fact", "user_prompt")}
    line = _lines(after)["fresh"]
    assert line.startswith("fresh agent  sees user_prompt (via prepare: current_item.json@data)")
    names = re.findall(r"(\w[\w.]*)(?:@| \(via )",
                       line.split("  sees ", 1)[1].split("  told")[0])
    assert names[0] == "user_prompt"
    assert _lines(nn_edits.apply({**edit, "sees": [names[0]]}, s4))["fresh"] == line
    # "task" is the question once: the same line as naming user_prompt.
    assert _lines(nn_edits.apply({**edit, "sees": ["task"]}, s4))["fresh"] == line


def test_sees_takes_a_file_name_on_a_nested_fork_and_lists_names_when_refused():
    nested = _seed("fuzz_nested_fork")
    graph = Graph.from_state(nested["graph"])
    fork, files = next((f, sorted(n for k, n in core_ops._available(graph)[f] if k == "file"))
                       for f, node in sorted(graph.nodes.items()) if node.kind == "fork"
                       and any(k == "file" for k, _n in core_ops._available(graph).get(f, ())))
    edit = {"op": "branch", "fork": fork, "kind": "agent", "sees": [files[0]], "id": "reader"}
    line = _lines(nn_edits.apply(edit, nested))["reader"]
    assert f"{files[0]}@" in line or f"{files[0]} (via " in line, line
    with pytest.raises(nn_edits.EditError) as refused:
        nn_edits.apply({**edit, "sees": ["no_such_input"]}, nested)
    message = str(refused.value)
    assert "'sees' names 'no_such_input'" in message and files[0] in message.split("there: ")[1]


def test_the_catalogue_says_sees_takes_single_names():
    lines = {op["op"]: json.dumps(op) for op in nn_edits.catalogue("broad")}
    assert "single fact or file names" in lines["rewire"]
    assert "single fact or file names" in lines["branch"]


# ── the shapes ─────────────────────────────────────────────────────────────


def test_shapes_nested_forks_optional_branch_and_a_loop_inside_a_branch():
    text = pv.render(_seed("fuzz_nested_fork"))
    outer, inner = re.findall(r"parallel (\w+) \{", text)
    assert re.search(rf"\n  parallel {inner} {{\n    ", text), text  # one level deeper
    assert text.index(f"parallel {outer}") < text.index(f"parallel {inner}")
    fork = _seed("fuzz_fork_join")
    optional = nn_edits.apply({"op": "branch", "fork": "split", "kind": "agent",
                               "optional": True, "id": "extra"}, fork)
    assert pv.changed_lines(fork, optional)[0].startswith(
        "+ [in parallel split] [optional] extra agent")
    looped = nn_edits.apply({"op": "loop", "step": "left", "judge": "code"}, fork)
    assert pv.changed_lines(fork, looped) == [
        '+ [in parallel split] loop(left, judge code (left_check__code (code) "def main(inp):"), '
        "max 2 -> 3 runs, knob left_check__max_iterations)"]


def test_shapes_per_item_bodies_and_several_data_passes():
    single = pv.render(_seed("fuzz_data_pass")).splitlines()
    assert single[0] == "per item (items -> current_item.json) {" and "}" in single
    two = _two_passes(_seed("fuzz_data_pass"))
    text = pv.render(two)
    assert text.count("per item (") == 2
    assert text.index("per item (items ") < text.index("per item (items_b ")
    assert pv.fallback_count(two) == 0


def _two_passes(state: dict) -> dict:
    """``state`` with its data pass run twice in turn: a copy of the pass (its nodes
    renamed ``*_b``) between the first pass and the step after it."""
    out = copy.deepcopy(state)
    nodes, edges = out["graph"]["nodes"], out["graph"]["edges"]
    inside = ["read_item", "solve", "check"]
    rename = {n: f"{n}_b" for n in [*inside, "items"]}
    for node_id in [*inside, "items"]:
        twin = copy.deepcopy(nodes[node_id])
        twin["id"] = rename[node_id]
        for key in ("subgraph_entry", "subgraph_exit"):
            if twin.get(key):
                twin[key] = rename[twin[key]]
        nodes[rename[node_id]] = twin
    edges += [{"source": rename[e["source"]], "target": rename[e["target"]],
               "condition": e.get("condition")} for e in list(edges)
              if e["source"] in inside and e["target"] in inside]
    out["graph"]["edges"] = [e for e in edges if not (e["source"] == "items"
                                                      and e["target"] == "report")]
    out["graph"]["edges"] += [{"source": "items", "target": "items_b", "condition": None},
                              {"source": "items_b", "target": "report", "condition": None}]
    return out


def test_shapes_code_verdict_gates_code_steps_frozen_and_protected():
    assert "loop(work, judge structure_fuzz_graphs:proceed, max 3 -> 4 runs, past the cap: " \
        "halt)" in pv.render(
        _seed("fuzz_gate_loop"))
    flow = _seed("fuzz_dataflow")
    coded = nn_edits.apply({"op": "insert", "after": "summarise", "kind": "code", "id": "tally"},
                           flow)
    assert any(line.startswith("+ tally code  sees ") and
               'told tally__code (code) "def main(inp):"' in line
               for line in pv.changed_lines(flow, coded))
    s4 = _mode("livemath/skill_graph")
    state = s4.state_dict()
    state["knobs"]["decide_task"]["spec"]["frozen"] = True
    text = pv.render(state, list(s4.edit_protected()))
    assert 'decide_task* "' in text and "[protected] extract_answer op" in text


# ── as built, the record and the request ─────────────────────────────────────


def test_what_an_edit_built_is_its_new_and_changed_view_lines():
    s1 = _mode("livemath/solver").state_dict()
    looped = nn_edits.apply({"op": "loop", "step": "solve", "judge": "agent"}, s1)
    built = pv.changed_lines(s1, looped)
    assert len(built) == 1 and built[0].startswith(
        "+ [in per item data] loop(solve, judge solve_check_judge (sees ")
    assert built[0].endswith("max 2 -> 3 runs, knob solve_check__max_iterations)")
    lowered = nn_edits.apply({"op": "set_knob", "name": "solve_check__max_iterations",
                              "value": 1}, looped)
    changed = pv.changed_lines(looped, lowered)
    assert len(changed) == 1 and changed[0].startswith("~ [in per item data] loop(solve,")
    assert "max 1 -> 2 runs" in changed[0]
    s4 = _mode("livemath/skill_graph").state_dict()
    branched = nn_edits.apply({"op": "branch", "fork": "step_fork", "kind": "agent",
                               "sees": ["task"], "id": "independent"}, s4)
    built = pv.changed_lines(s4, branched)
    assert any(line.startswith("+ [in parallel step_fork] independent agent  sees "
                               "user_prompt (via prepare: current_item.json@data)  told")
               for line in built), built
    assert any(line.startswith("~ [in per item data] decide agent") and
               "independent@independent" in line for line in built), built
    draft = _seed("fuzz_draft_review")
    step = next(n for n in sorted(draft["graph"]["nodes"])
                if nn_edits.precheck({"op": "remove", "step": n}, draft) is None)
    assert f"- removed {step}" in pv.changed_lines(
        draft, nn_edits.apply({"op": "remove", "step": step}, draft))


def test_the_request_shows_the_view_first_and_as_built_beside_the_words():
    pair = {"child": "c1", "parent": "p0", "edit": [{"op": "loop", "step": "solve"}],
            "flow_change": "the judge sends solve back once",
            "wiring": ["+ loop(solve, judge j, max 2 -> 3 runs)"]}
    lines = base._edit_change_lines([pair])
    said = lines.index("Flow change, as proposed: the judge sends solve back once")
    assert lines[said + 1].startswith("As built (its pipeline lines")
    assert lines[said + 2:said + 5] == ["```text", "+ loop(solve, judge j, max 2 -> 3 runs)",
                                        "```"]
    text = "\n".join(base._reasoning_lines([{**pair, "id": "c1", "epoch": 1}]))
    assert "Flow change, as proposed: the judge sends solve back once\nAs built" in text
    many = base._as_built_lines({"wiring": [f"+ line {i}" for i in range(20)]})
    assert f"(and {20 - base.AS_BUILT_LINES} more line(s))" in many


def _repeated_blocks(text: str, lines: int = 3, chars: int = 120) -> list[str]:
    """Each run of ``lines`` consecutive non-blank lines (``chars``+ characters together)
    that appears more than once in ``text``: a block said twice, whatever its line share."""
    rows = [line for line in text.splitlines() if line.strip()]
    seen: dict[str, int] = {}
    for i in range(len(rows) - lines + 1):
        block = "\n".join(rows[i:i + lines])
        if len(block) >= chars:
            seen[block] = seen.get(block, 0) + 1
    return [block for block, n in seen.items() if n > 1]


def test_as_built_is_said_once_when_the_last_edit_s_section_shows_it():
    """The latest child is in "What the last edit changed" (its edits, flow change and As
    built) and in "Your earlier reasoning": the second points to the first."""
    from refactory_dsh.optim import ProposalContext

    pair = {"child": "c1", "parent": "p0", "edit": [{"op": "loop", "step": "solve"}],
            "flow_change": "the judge sends solve back once when the answer is unsure",
            "wiring": [f"+ loop(solve, judge j, max 2 -> 3 runs) line {i}" for i in range(4)]}
    text = ProposalContext(edit_pairs=[pair], reasoning=[{**pair, "id": "c1", "epoch": 1,
                                                          "rationale": "R"}]).render()
    assert text.count("Flow change, as proposed: the judge sends") == 1
    assert text.count("line 0") == 1
    assert 'under "What the last edit changed" above' in text
    assert _repeated_blocks(text) == [], _repeated_blocks(text)


def test_the_loop_s_request_opens_with_the_view_and_shows_what_was_built(tmp_path):
    from refactory_dsh import strategies
    from refactory_dsh.loop import LoopConfig

    loop = strategies.create("default", "livemath/solver", LoopConfig(
        epochs=1, batch_size=1, optimizer="random", seed=1, runs_root=tmp_path,
        edit_preset="broad"), project=Path("."))
    seed = Candidate.from_state(loop.mode.state_dict())
    edit = {"op": "loop", "step": "solve", "judge": "agent"}
    child = Candidate.from_state(nn_edits.apply(edit, seed.state), parent=seed.id, edits=[edit])
    child.evaluated, child.score = True, 0.5
    seed.evaluated, seed.score = True, 0.4
    loop.population += [seed, child]
    loop._editing_parent = child.id
    loop.proposals[child.id] = {"id": child.id, "parent": seed.id, "epoch": 1,
                                "origin": "optimizer",
                                "flow_change": "solve is sent back once at most"}
    text = loop._context().render()
    pipeline = text.split("## The pipeline you may edit", 1)[1].split(base.REFERENCE_HEADING, 1)[0]
    assert pipeline.index(base.PIPELINE_VIEW_LEAD) < pipeline.index("```text\nper item (data")
    assert "loop(solve, judge solve_check_judge" in pipeline
    changed = text.split("## What the last edit changed", 1)[1].split("\n## ", 1)[0]
    assert "Flow change, as proposed: solve is sent back once at most" in changed
    assert "loop(solve, judge solve_check_judge" in changed and "max 2 -> 3 runs" in changed
    # Said once: the reasoning points to it.
    reasoning = text.split("## Your earlier reasoning", 1)[1].split("\n## ", 1)[0]
    assert 'under "What the last edit changed" above' in reasoning
    assert text.count("Flow change, as proposed: solve is sent back once at most") == 1


def test_a_round_records_what_was_built_in_the_proposal_record(tmp_path, monkeypatch):
    from refactory_dsh.loop import Loop
    from refactory_dsh.optim import Optimizer, Proposal
    from test_compound_proposals import TwoLevels
    from test_skillopt import _patch_mode
    from test_train_minibatch import _events
    from test_val_gate import _loop

    edit = {"op": "loop", "step": "work", "judge": "loop_ops:always_proceed",
            "max_iterations": 1}

    class Scripted(Optimizer):
        name = "scripted_view"

        def __init__(self) -> None:
            super().__init__(k=1)
            self.plan = [[Proposal(edit=edit, flow_change="work is checked once")], []]

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            return self.plan.pop(0) if self.plan else []

    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(TwoLevels):
        loop = _loop(tmp_path, epochs=2, batch_size=1, allowed_ops=(), edit_preset="broad")
        loop.optimizer = Scripted()
        loop.run()
    child = next(c for c in loop.population if c.parent)
    built = loop.proposals[child.id]["wiring"]
    assert any("loop(work, judge loop_ops:always_proceed, max 1 -> 2 runs, knob "
               "work_check__max_iterations)" in line for line in built), built
    journaled = [r for r in _events(loop, "proposal") if r["id"] == child.id]
    assert journaled and journaled[0]["wiring"] == built


def test_the_mode_s_graders_reach_the_wiring_record_and_the_request(tmp_path, monkeypatch):
    """The view reads the mode's grader nodes (never folded) in the proposal's wiring
    record, the history's As built and the request's pipeline view alike."""
    from refactory_dsh.loop import Loop
    from refactory_dsh.optim import Optimizer, Proposal
    from test_compound_proposals import TwoLevels
    from test_skillopt import _patch_mode
    from test_val_gate import _loop

    calls: list[tuple] = []
    for name in ("changed_lines", "render"):
        real = getattr(pv, name)

        def spy(*args, _real=real, _name=name, **kwargs):  # type: ignore[no-untyped-def]
            import inspect

            bound = inspect.signature(_real).bind(*args, **kwargs)
            calls.append((_name, tuple(bound.arguments.get("graders") or ())))
            return _real(*args, **kwargs)
        monkeypatch.setattr(pv, name, spy)
    edit = {"op": "loop", "step": "work", "judge": "loop_ops:always_proceed"}

    class Scripted(Optimizer):
        name = "scripted_graders"

        def __init__(self) -> None:
            super().__init__(k=1)
            self.plan = [[Proposal(edit=edit)], []]

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            return self.plan.pop(0) if self.plan else []

    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    monkeypatch.setattr(TwoLevels, "grader_nodes", lambda self: ("GRADER",), raising=False)
    with _patch_mode(TwoLevels):
        loop = _loop(tmp_path, epochs=2, batch_size=1, allowed_ops=(), edit_preset="broad")
        loop.optimizer = Scripted()
        loop.run()
        assert loop.graders == ("GRADER",)
        loop._context().render()
    kinds = {name for name, _args in calls}
    assert kinds == {"changed_lines", "render"}, kinds
    assert all(graders == ("GRADER",) for _name, graders in calls), calls


def test_steps_a_gate_names_are_never_folded_and_judges_are_only_judges():
    """A route's label and a loop's send-back name real steps; a step whose output reaches a
    later agent (through its collect step) keeps its line instead of reading as a judge; a
    route after a looped step names that step, not the loop's gate."""
    multi = _seed("fuzz_multi_terminal")
    looped = _lines(nn_edits.apply({"op": "loop", "step": "reject", "judge": "agent"}, multi))
    route = next(t for t in looped.values() if t.startswith("route("))
    assert "reject: reject" in route, route
    assert any(t.startswith("loop(reject, judge reject_check_judge") for t in looped.values())
    gates = _seed("fuzz_gates")
    inserted = _lines(nn_edits.apply({"op": "insert", "after": "archive", "kind": "agent"}, gates))
    assert any("proceed: archive" in t for t in inserted.values() if t.startswith("route(")), inserted
    gate_loop = _seed("fuzz_gate_loop")
    split = _lines(nn_edits.apply({"op": "split", "step": "finish",
                                   "into": [{"kind": "agent"}, {"kind": "agent"}]}, gate_loop))
    assert "work" in split and split["work"].startswith("work agent")
    rough = _lines(nn_edits.apply({"op": "loop", "step": "rough", "judge": "agent"}, gates))
    assert any(t.startswith("route(after polish, rough, ") for t in rough.values()), rough


def test_steps_the_walk_cannot_place_are_fallback_lines_and_counted():
    flow = _seed("fuzz_dataflow")
    cut = nn_edits.apply({"op": "remove_edge", "source": "load", "target": "clean"}, flow)
    lines = pv.view_lines(cut)
    fallback = [line for line in lines if line.key.startswith(pv.FALLBACK)]
    assert pv.fallback_count(cut) == len(fallback) > 0
    assert {line.key.removeprefix(pv.FALLBACK) for line in fallback} >= {"summarise"}
    assert all("edges -> " in line.text or line.key.endswith("emit") for line in fallback)


def test_an_agent_only_a_later_gate_reads_keeps_its_own_line():
    """A judge is the agent right before its gate; an earlier agent whose file only that
    gate reads is a step of its own."""
    def agent(node_id, writes, prompt="Work."):
        return {"id": node_id, "kind": "agent", "writes": [writes], "prompt_template": prompt}
    state = {"graph": {"start": "a", "nodes": {
        "a": agent("a", "a.txt"), "b": agent("b", "b.txt"),
        "g": {"id": "g", "kind": "gate", "evaluator": "loop_ops:always_proceed",
              "reads": ["a.txt"], "max_iterations": 2},
        "end": {"id": "end", "kind": "op", "call": "loop_ops:noop", "reads": ["b.txt"],
                "produces": ["done"]}},
        "edges": [{"source": "a", "target": "b", "condition": None},
                  {"source": "b", "target": "g", "condition": None},
                  {"source": "g", "target": "b", "condition": "reloop"},
                  {"source": "g", "target": "end", "condition": "proceed"}]}, "knobs": {}}
    text = pv.render(state)
    assert text.splitlines()[0].startswith("a agent"), text
    assert "loop(b, judge loop_ops:always_proceed, max 2 -> 3 runs, past the cap: halt)" \
        in text, text


def test_a_view_that_cannot_be_drawn_says_so_in_the_request_and_the_record(caplog):
    broken = {"graph": {"start": "a", "nodes": {"a": {"id": "a", "kind": "no_such_kind"}},
                        "edges": []}, "knobs": {}}
    with caplog.at_level("WARNING"):
        lines = pv.view_lines(broken)
    assert len(lines) == 1 and lines[0].key == pv.UNAVAILABLE
    assert lines[0].text.startswith("pipeline view unavailable: ")
    assert any("pipeline view unavailable" in r.getMessage() for r in caplog.records)
    good = _seed("fuzz_draft_review")
    assert pv.changed_lines(good, broken)[0].startswith("! pipeline view unavailable: ")
    unreadable = base.pipeline_view_lines([{"id": "x", "kind": "nope"}], [], [], ())
    assert "pipeline view unavailable: the graph does not read back" in unreadable


def test_the_legend_s_forms_are_what_the_readers_draw():
    """Each block form the request's legend gives (`FORMS`) matches its reader's output on
    a graph that has it: a loop of cap N says N+1 runs."""
    assert all(form in base.PIPELINE_VIEW_LEAD for form in pv.FORMS.values())
    s1 = _mode("livemath/solver").state_dict()
    for cap in (1, 3):
        looped = nn_edits.apply({"op": "loop", "step": "solve", "judge": "agent",
                                 "max_iterations": cap}, s1)
        line = _lines(looped)["solve_check"]
        found = re.match(r"loop\((\w+), judge .*, max (\d+) -> (\d+) runs", line)
        assert found and (int(found.group(2)), int(found.group(3))) == (cap, cap + 1), line
    text = pv.render(_seed("fuzz_fork_join"))
    assert re.search(r"(?m)^parallel \w+ \{$", text) and re.search(r"(?m)^\} -> \w+$", text)
    routed = pv.render(nn_edits.apply({"op": "route", "after": "draft", "judge": "agent",
                                       "to": {"proceed": "review", "revise": {"kind": "agent"}}},
                                      _seed("fuzz_draft_review")))
    assert re.search(r"(?m)^route\(after \w+, judge .*, \{\w+: \w+, \w+: \w+\}\)$", routed)
    assert re.search(r"(?m)^per item \(\w+( -> [^)]*)?\) \{$", pv.render(_seed("fuzz_data_pass")))


def test_the_fallback_count_reads_the_graders_as_the_view_does():
    state = _seed("fuzz_data_pass")
    for graders in ((), ("check",)):
        assert pv.fallback_count(state, (), graders) == sum(
            line.key.startswith(pv.FALLBACK) for line in pv.view_lines(state, (), graders))


def test_a_loop_says_what_happens_past_its_cap_unless_it_goes_on():
    """`Graph.on_exhausted`, as both runtimes act: a loop in a data pass that goes on with
    its last attempt says nothing more; one that fails the item, or halts the run, says
    so."""
    solver = _mode("livemath/solver").state_dict()
    looped = nn_edits.apply({"op": "loop", "step": "solve", "judge": "code"}, solver)
    line = next(l for l in pv.render(looped).splitlines() if "loop(solve" in l)
    assert "past the cap" not in line, line
    gate = next(n for n, node in looped["graph"]["nodes"].items()
                if node.get("kind") == "gate")
    looped["graph"]["nodes"][gate]["on_exhausted"] = "fail_item"
    line = next(l for l in pv.render(looped).splitlines() if "loop(solve" in l)
    assert line.rstrip().endswith(", past the cap: fail_item)"), line
    assert "past the cap: fail_item" in base.PIPELINE_VIEW_LEAD
