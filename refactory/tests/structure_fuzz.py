"""An invariant harness for structure edits: every graph the edit layer accepts is sound.

From a seed, `Fuzzer` generates random structure edits over a set of seed graphs and
applies each through the path the outer loop uses (`Optimizer.attempt`: precheck, apply,
the protected-node check, validate). Every state it ACCEPTS is checked against the
invariants below, each a promise the optimizer's results depend on:

- ``lowering``  the state lowers to the spine's workflow (`lowering.to_workflow`);
- ``walker``    the walker runs it to a terminal with a scripted model (no exception,
                not halted, not stopped at a gate);
- ``spine``     on a subset (graphs with no user or agent gate), the spine runs it
                with the same scripted answers and ends with the walker's facts and
                item scores;
- ``knobs``     every agent node runs its own system prompt: each binds a knob of its
                own, no two nodes bind the same knob, every bound knob is declared and
                every declared knob is read;
- ``dataflow``  every file a node reads and every ``{{fact}}`` an agent's prompt uses
                is produced upstream on every path that reaches it;
- ``forks``     forks and joins are paired: a fork's one out-edge leads to its join,
                every branch arrives there and nowhere else, and the join waits for
                exactly the branch ends that lead to it;
- ``protected`` the mode's protected nodes are as the seed declared them.

With ``compound`` > 0 some steps of a walk are *compound proposals* (``Proposal.edits``):
an ordered list of 2-4 mixed edits -- text ``set_knob`` (whole value or ``patch``),
structure macros and primitives, node copies and then their own prompt knobs, now and
then a malformed one or one too many -- applied as one candidate through the same
attempt path. Every accepted combined state is checked against every invariant above,
and the ``compound`` invariant checks the bundle itself:

- ``compound``  an accepted bundle is exactly its edits applied in order and records
                all of them (``Candidate.own_edits``); a refused one names the edit
                that fails on its own (``[edit k of n``: the first not an object or not
                in the catalogue, else the first whose precheck or apply fails in
                order), and a refusal of the combined state names none.

A violation is a bug. `Fuzzer` shrinks it to a minimal case (the seed's name and the
accepted edits that reach it) and `save_case` writes it to the regression corpus
(``fixtures/structure_fuzz``), which `test_structure_fuzz` replays on every run.

Everything is seeded: the same ``seed`` tries the same edits in the same order.
"""
from __future__ import annotations

import contextlib
import json
import random
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from refactory_dsh import lowering, nn
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.executor import execute
from refactory_dsh.nn.graph import Graph
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.optim.base import (
    DEFAULT_MAX_EDITS, _node_spec, proposal_refusal, reason_code,
)

import structure_fuzz_graphs as fg

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "fixtures" / "structure_fuzz"
INVARIANTS = ("lowering", "walker", "spine", "grader", "slots", "required", "knobs", "dataflow",
              "forks", "protected", "tools")
#: Checks that run the real code (lowering, the walker, the spine) and look at what it
#: did; the rest are spec checks, a second statement of the rules beside the validator.
OBSERVED = ("lowering", "walker", "spine", "grader", "slots", "required")
#: How both runtimes refuse a node whose required read is missing (`run.missing_inputs`).
REQUIRED_MISSING = "required input is missing"
#: Run facts the harness counts but does not call violations (an op's optional read
#: that was not there: V6 allows it).
NOTES: Counter = Counter()
#: Files the grader writes after an item is judged: its records carry the whole item.
GRADER_RECORDS = ("item_records.json", "facts.json", ".item_record.json", ".done")
SLOT = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
#: A walk restarts from its seed once its graph has this many nodes.
MAX_NODES = 28
#: Verdict labels the generator puts on gate edges; `fg.proceed` answers ``proceed``.
CONDITIONS = ("proceed", "reloop", "reject")
#: How the walker says a gate's verdict has no edge (`nn.executor.route_verdict`).
SCRIPTED_LABEL = "verdict 'proceed' is not one of"


# ── seeds ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Seed:
    """A seed graph: the mode it comes from, and how to run it."""

    name: str
    mode_class: type
    livemath: bool = False

    def mode(self) -> nn.Mode:
        return self.mode_class()

    def state(self) -> dict[str, Any]:
        return self.mode().state_dict()


def _livemath_skill_graph() -> type:
    from refactory_packages.livemath.skill_graph import SkillGraph

    return SkillGraph


def seeds() -> dict[str, Seed]:
    out = {cls.name: Seed(cls.name, cls) for cls in fg.SYNTHETIC}
    out["livemath_skill_graph"] = Seed("livemath_skill_graph", _livemath_skill_graph(),
                                       livemath=True)
    return out


# ── the attempt path ──────────────────────────────────────────────────────


class Applier:
    """Applies edits to one seed's states exactly as the outer loop's attempt does."""

    def __init__(self, seed: Seed) -> None:
        self.seed = seed
        mode = seed.mode()
        self.seed_state = mode.state_dict()
        self.invariants = mode.invariants()
        self.protected = tuple(mode.protected_nodes())
        self.optimizer = Optimizer()

    def attempt(self, parent: Candidate, edit: Any, seen: set[str]) -> Candidate | list[str]:
        """`Run.attempt`'s checks, then `Optimizer.attempt` with this mode's protection.

        ``edit`` is one edit, or a compound proposal in its ``{"edits": [...]}`` form.
        """
        proposal = Proposal(edit=edit, parent=parent.id)
        refused = proposal_refusal(proposal, max_edits=DEFAULT_MAX_EDITS)
        if refused:
            return refused
        return self.optimizer.attempt(
            proposal, parent, self.invariants, True, (),
            sorted(seen), seed_state=self.seed_state, protected=self.protected,
        )

    def replay(self, edits: list[Any]) -> tuple[list[Candidate], list[str] | None]:
        """Apply ``edits`` in order: the states accepted, and the refusal that stopped it."""
        current = Candidate.from_state(self.seed_state)
        seen = {current.id}
        states = [current]
        for edit in edits:
            outcome = self.attempt(current, edit, seen)
            if isinstance(outcome, list):
                return states, outcome
            current = outcome
            seen.add(current.id)
            states.append(current)
        return states, None


# ── the scripted model ────────────────────────────────────────────────────


#: What each agent node answers, on both runtimes. Any node not named answers B.
ANSWERS = {"compare": "compare notes", "precision": "precision notes",
           "hypotheses": "hypotheses notes", "decide": "<answer>B</answer>"}


class ScriptedModel:
    """A model client that answers by node id (`test_livemath_blocks._NodeScriptedModel`)."""

    def __init__(self, answers: dict[str, str], node: str = "", log: list | None = None):
        self.answers = answers
        self.node = node
        self.log = log if log is not None else []

    def for_node(self, node_id: str, role: str = "") -> "ScriptedModel":
        return ScriptedModel(self.answers, node_id, self.log)

    def with_tool_policy(self, policy: str) -> "ScriptedModel":
        return self

    def with_wall(self, wall: Any) -> "ScriptedModel":
        return self

    def with_budget(self, budget: Any) -> "ScriptedModel":
        return self

    def context(self) -> dict:
        return {}

    def route(self) -> dict:
        return {"worker_backend": "dsh"}

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        self.log.append((self.node, prompt, system))
        return self.answers.get(self.node, "<answer>B</answer>")


# ── invariants ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Violation:
    invariant: str
    message: str

    def __str__(self) -> str:
        return f"{self.invariant}: {self.message}"


def _graph_of(state: dict[str, Any]) -> Graph:
    return Graph.from_state(state["graph"])


def _knob_values(state: dict[str, Any]) -> dict[str, Any]:
    return {name: entry["value"] for name, entry in (state.get("knobs") or {}).items()}


def check_lowering(state: dict[str, Any]) -> list[Violation]:
    try:
        workflow = lowering.to_workflow(state, name="fuzz")
        workflow.to_dict()
    except Exception as exc:  # noqa: BLE001 - any failure to lower is the finding
        return [Violation("lowering", f"{type(exc).__name__}: {exc}")]
    return []


def check_knobs(state: dict[str, Any]) -> list[Violation]:
    graph = _graph_of(state)
    declared = set(state.get("knobs") or {})
    out: list[Violation] = []
    binders: dict[str, list[str]] = {}
    for node in graph.nodes.values():
        for field_name, knob in node.field_bindings:
            binders.setdefault(knob, []).append(node.id)
            if knob not in declared:
                out.append(Violation("knobs", f"{node.id} binds {field_name} to undeclared {knob}"))
        if node.kind == "agent" and not node.bound_knob("system_prompt_template"):
            out.append(Violation("knobs", f"agent {node.id} has no system-prompt knob of its own"))
    for knob, owners in sorted(binders.items()):
        if len(owners) > 1:
            out.append(Violation("knobs", f"knob {knob} is bound by {sorted(owners)}"))
    readers = graph.knob_consumers()
    for knob in sorted(declared - set(readers)):
        out.append(Violation("knobs", f"declared knob {knob} is read by no node"))
    for node in graph.nodes.values():
        for knob in node.knobs:
            if knob not in declared:
                out.append(Violation("knobs", f"{node.id} reads undeclared knob {knob}"))
    return out


def _branch(graph: Graph, target: str, barrier: str | None) -> tuple[set[str], bool]:
    """A fork branch's members, walked as the runtimes walk it, and whether it arrives
    at ``barrier``."""
    seen: set[str] = set()
    arrives = False
    pending = [target]
    while pending:
        current = pending.pop()
        if current == barrier:
            arrives = True
            continue
        if current in seen or current not in graph.nodes:
            continue
        seen.add(current)
        node = graph.nodes[current]
        pending.extend(e.target for e in graph.out_edges(current))
        if node.kind == "fork":
            pending.extend(node.targets)
    return seen, arrives


def _after(graph: Graph, join: str, retries: set[tuple[str, str, str]]) -> set[str]:
    """Every node the run goes on to after ``join``, not following retry edges back."""
    seen: set[str] = set()
    stack = [e.target for e in graph.out_edges(join) if e.key() not in retries]
    while stack:
        current = stack.pop()
        if current in seen or current not in graph.nodes:
            continue
        seen.add(current)
        stack.extend(e.target for e in graph.out_edges(current) if e.key() not in retries)
        node = graph.nodes[current]
        if node.kind == "fork":
            stack.extend(node.targets)
        if node.kind == "data" and node.subgraph_entry:
            stack.append(node.subgraph_entry)
    return seen


def _may_skip(graph: Graph, start: str, banned: str, join: str) -> bool:
    """Whether a branch walk from ``start`` can arrive at ``join``, or end, without
    visiting ``banned``."""
    seen: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current == join:
            return True
        if current in seen or current == banned or current not in graph.nodes:
            continue
        seen.add(current)
        following = [e.target for e in graph.out_edges(current)]
        if graph.nodes[current].kind == "fork":
            following += list(graph.nodes[current].targets)
        if not following:
            return True
        stack.extend(following)
    return False


def check_forks(state: dict[str, Any]) -> list[Violation]:
    graph = _graph_of(state)
    out: list[Violation] = []
    joins_of: dict[str, list[str]] = {}
    reachable = graph.reachable()
    try:
        from refactory_dsh.nn.loops import loop_edges

        retries = loop_edges(graph)
    except Exception:  # noqa: BLE001 - V10 reports an ambiguous cycle
        retries = set()
    for fork in sorted((n for n in graph.nodes.values() if n.kind == "fork"), key=lambda n: n.id):
        if fork.id not in reachable:
            continue
        edges = graph.out_edges(fork.id)
        if len(edges) != 1 or edges[0].condition is not None:
            out.append(Violation("forks", f"fork {fork.id} has {len(edges)} out-edges, not one "
                                          "unconditional edge to its join"))
            continue
        join = graph.nodes.get(edges[0].target)
        if join is None or join.kind != "join":
            out.append(Violation("forks", f"fork {fork.id} leads to {edges[0].target}, not a join"))
            continue
        joins_of.setdefault(join.id, []).append(fork.id)
        after = _after(graph, join.id, retries)
        members: set[str] = set()
        for target in fork.targets:
            branch, _arrives = _branch(graph, target, join.id)
            if fork.id in branch:
                out.append(Violation("forks", f"fork {fork.id}: branch {target} leads back to the fork"))
            escaped = sorted(branch & after)
            if escaped:
                out.append(Violation("forks", f"fork {fork.id}: branch {target} runs {escaped}, "
                                              f"which come after its join {join.id}"))
            if branch & members:
                out.append(Violation("forks", f"fork {fork.id}: branch {target} shares "
                                              f"{sorted(branch & members)} with another branch"))
            members |= branch
        outside = sorted({e.source for e in graph.in_edges(join.id)} - members - {fork.id})
        if outside:
            out.append(Violation("forks", f"join {join.id} is entered from {outside}, outside the "
                                          f"branches of fork {fork.id}"))
        for source in join.sources:
            owner = next((t for t in fork.targets if source in _branch(graph, t, join.id)[0]), None)
            if owner is None:
                out.append(Violation("forks", f"join {join.id} waits for {source}, which is in no "
                                              f"branch of fork {fork.id}"))
            elif _may_skip(graph, owner, source, join.id):
                out.append(Violation("forks", f"join {join.id} waits for {source}, which a run of "
                                              f"branch {owner} can finish without"))
    for join in sorted((n for n in graph.nodes.values() if n.kind == "join"), key=lambda n: n.id):
        if join.id in reachable and join.id not in joins_of:
            out.append(Violation("forks", f"join {join.id} belongs to no fork"))
        elif len(joins_of.get(join.id, ())) > 1:
            out.append(Violation("forks", f"join {join.id} is the join of {joins_of[join.id]}"))
    return out


TAG = re.compile(r"\{%[-\s]*(if|elif|else|endif)\b\s*(.*?)[-\s]*%\}", re.DOTALL)
GUARD = re.compile(r"""([A-Za-z_]\w*)\s*(==\s*(['"]).+\3)?""")


def _needed_slots(text: str) -> set[str]:
    """Slots a template needs: those outside an ``if``/``elif`` branch on their own name."""
    needed: set[str] = set()
    pieces = TAG.split(text)  # text, tag, expr, text, tag, expr, ...
    branch_names: list[str | None] = []
    for index in range(0, len(pieces), 3):
        for name in SLOT.findall(pieces[index]):
            if name not in branch_names:
                needed.add(name)
        if index + 2 < len(pieces):
            tag, expr = pieces[index + 1], pieces[index + 2].strip()
            word = GUARD.fullmatch(expr)
            if tag == "if":
                branch_names.append(word.group(1) if word else None)
            elif tag == "elif" and branch_names:
                branch_names[-1] = word.group(1) if word else None
            elif tag == "else" and branch_names:
                branch_names[-1] = None
            elif tag == "endif" and branch_names:
                branch_names.pop()
    return needed


def _consumed(node: nn.Node, knobs: dict[str, Any]) -> tuple[set[str], set[str]]:
    """The files a node reads and the facts its prompts need filled."""
    facts: set[str] = set()
    if node.kind == "agent":
        for field_name in ("prompt_template", "system_prompt_template"):
            facts |= _needed_slots(str(node.bound(field_name, knobs) or ""))
    facts -= set(knobs) | set(node.feedback_keys)
    return set(node.reads), facts


def _fork_of_join(graph: Graph) -> dict[str, str]:
    """Each join that is exactly one fork's join, and that fork."""
    owners: dict[str, list[str]] = {}
    for node in graph.nodes.values():
        if node.kind == "fork":
            join = graph.fork_join(node.id)
            if join in graph.nodes and graph.nodes[join].kind == "join":
                owners.setdefault(join, []).append(node.id)
    return {join: forks[0] for join, forks in owners.items() if len(forks) == 1}


def _branch_ends(graph: Graph, fork: nn.Node, join: str) -> list[set[str]]:
    """For each branch of ``fork``, the members a run of it can end at: the ones with an
    edge to ``join`` or with no out-edge at all."""
    out = []
    for target in fork.targets:
        members, _ = _branch(graph, target, join)
        out.append({m for m in members
                    if not graph.out_edges(m) or any(e.target == join for e in graph.out_edges(m))})
    return out


def available_before(graph: Graph) -> dict[str, set[tuple[str, str]]]:
    """What is available when each node starts, on every path that reaches it.

    The greatest fixpoint of: a node's inputs are what every predecessor left; a fork's
    branches start with what the fork had; a fork's join starts with what the fork had
    and what every run of each branch left (branches run to their end, whether that end
    leads to the join or nowhere); a data pass starts with what its data node had and
    the data node's own writes; the node after a data node gets what the data node had
    and declares, never what the pass made (a pass is isolated). Entries are
    ``("file", name)`` and ``("fact", name)``.
    """
    universe = {("file", n) for node in graph.nodes.values() for n in (*node.writes, *node.reads)}
    universe |= {("fact", n) for node in graph.nodes.values() for n in node.produces}
    universe |= {("fact", n) for node in graph.nodes.values()
                 for f in ("prompt_template", "system_prompt_template")
                 for n in SLOT.findall(str(getattr(node, f) or ""))}

    def produced(node: nn.Node) -> set[tuple[str, str]]:
        return {("file", n) for n in node.writes} | {("fact", n) for n in node.produces}

    fork_of = _fork_of_join(graph)
    preds: dict[str, list[tuple[str, str]]] = {n: [] for n in graph.nodes}
    for node in graph.nodes.values():
        if node.kind == "data":
            if node.subgraph_entry in preds:
                preds[node.subgraph_entry].append((node.id, "pass"))
            for edge in graph.out_edges(node.id):
                preds[edge.target].append((node.id, "after_data"))
            continue
        if node.kind == "fork":
            for target in node.targets:
                if target in preds:
                    preds[target].append((node.id, "in"))
        for edge in graph.out_edges(node.id):
            if edge.target in fork_of:
                continue  # a fork's join is fed by its fork and branches, below
            preds[edge.target].append((node.id, "out"))
    reachable = graph.reachable()
    preds = {n: [(s, how) for s, how in entries if s in reachable] for n, entries in preds.items()}
    avail_in: dict[str, set[tuple[str, str]]] = {n: set(universe) for n in graph.nodes}
    if graph.start in avail_in:
        avail_in[graph.start] = set()

    def avail_out(source: str, how: str = "out") -> set[tuple[str, str]]:
        node = graph.nodes[source]
        if how == "in":
            return set(avail_in[source])
        if how == "pass":
            return avail_in[source] | {("file", n) for n in node.writes}
        return avail_in[source] | produced(node)

    changed = True
    while changed:
        changed = False
        for node_id in sorted(graph.nodes):
            if node_id == graph.start:
                continue
            if node_id in fork_of:
                fork = graph.nodes[fork_of[node_id]]
                new = avail_in[fork.id]
                for ends in _branch_ends(graph, fork, node_id):
                    if ends:
                        new = new | set.intersection(*(avail_out(e) for e in ends))
            else:
                incoming = [avail_out(s, how) for s, how in preds[node_id]]
                if not incoming:
                    continue
                new = set.intersection(*incoming)
            if new != avail_in[node_id]:
                avail_in[node_id] = new
                changed = True
    return avail_in


def _on_some_run(graph: Graph) -> set[str]:
    """Every file and fact a node on some start-to-terminal path writes or produces
    (forward from the start, following fork targets and data entries; backward from
    the terminals along edges)."""
    forward = graph.reachable()
    back: dict[str, set[str]] = {}
    for edge in graph.edges:
        back.setdefault(edge.target, set()).add(edge.source)
    seen: set[str] = set()
    stack = list(graph.terminals())
    while stack:
        current = stack.pop()
        if current not in seen:
            seen.add(current)
            stack.extend(back.get(current, ()))
    return {n for node_id in forward & seen for n in (*graph.nodes[node_id].writes,
                                                      *graph.nodes[node_id].produces)}


def check_dataflow(state: dict[str, Any]) -> list[Violation]:
    """Spec check. An agent's inputs are required: every file it reads and every fact
    its prompts need is produced upstream on every path to it, and a data pass's own
    outputs stay in the pass. An op or a gate may read a file that is not there yet (no
    runtime checks an op's reads), so its reads need only be produced on some run; its
    ``required_reads`` are held to every path, as an agent's are."""
    graph = _graph_of(state)
    knobs = _knob_values(state)
    strict = available_before(graph)
    some_run = _on_some_run(graph)
    out: list[Violation] = []
    for node_id in sorted(graph.reachable()):
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        files, facts = _consumed(node, knobs)
        if node.kind == "agent":
            wanted = {("file", n) for n in files} | {("fact", n) for n in facts}
            missing = sorted(f"{k} {n}" for k, n in wanted - strict[node_id])
        else:
            missing = sorted({f"file {n}" for n in files if n not in some_run}
                             | {f"required file {n}" for n in node.required_reads
                                if ("file", n) not in strict[node_id]})
        if missing:
            out.append(Violation("dataflow", f"{node.kind} {node_id} reads {missing}, not "
                                             "produced where it needs to be"))
    return out


def _ends_without(graph: Graph, node_id: str) -> bool:
    """Whether, with ``node_id`` taken out, the run can still end: from the start to a
    terminal outside every pass, or for a node inside a pass, from its entry to its exit.
    Edges, fork targets and data entries are followed (an independent, simpler model
    than the edit layer's)."""
    passes = {d.id: graph.data_body(d.id) for d in graph.nodes.values() if d.kind == "data"}
    owner = next((d for d, body in passes.items() if node_id in body), None)
    if owner is not None:
        start, goal = graph.nodes[owner].subgraph_entry, {graph.nodes[owner].subgraph_exit}
    else:
        inside = set().union(*passes.values()) if passes else set()
        start, goal = graph.start, {n for n in graph.terminals() if n not in inside}
    seen: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current == node_id or current in seen or current not in graph.nodes:
            continue
        if current in goal:
            return True
        seen.add(current)
        node = graph.nodes[current]
        stack.extend(e.target for e in graph.out_edges(current))
        if node.kind == "fork":
            stack.extend(node.targets)
    return False


def check_protected(seed_state: dict[str, Any], state: dict[str, Any],
                    protected: tuple[str, ...]) -> list[Violation]:
    """Each protected node is as the seed declared it: there, the same declaration, the
    same out-edges and knob values, reached, and still on every run if it was."""
    if not protected:
        return []
    before, after = _graph_of(seed_state), _graph_of(state)
    out: list[Violation] = []
    for node_id in protected:
        old, new = before.nodes.get(node_id), after.nodes.get(node_id)
        if old is None:
            continue
        if new is None:
            out.append(Violation("protected", f"{node_id} was removed"))
            continue
        if new.to_state() != old.to_state():
            out.append(Violation("protected", f"{node_id}'s declaration changed"))
        if {e.key() for e in before.out_edges(node_id)} != {e.key() for e in after.out_edges(node_id)}:
            out.append(Violation("protected", f"{node_id}'s out-edges changed"))
        for knob in old.knobs:
            if (seed_state.get("knobs") or {}).get(knob) != (state.get("knobs") or {}).get(knob):
                out.append(Violation("protected", f"knob {knob} that {node_id} reads changed"))
        if node_id not in after.reachable():
            out.append(Violation("protected", f"{node_id} is no longer reached"))
        elif not _ends_without(before, node_id) and _ends_without(after, node_id):
            out.append(Violation("protected", f"a run can now end without {node_id}"))
    return out


def _verdicts(graph: Graph) -> dict[str, str]:
    return {n.id: "proceed" for n in graph.nodes.values()
            if n.kind == "gate" and n.gate_kind != "fn"}


@contextlib.contextmanager
def livemath_fixture() -> Iterator[None]:
    """LiveMath's task reads the test fixture's items, unshuffled, on the walker."""
    import refactory_packages.livemath.tasks as tasks_mod

    original = tasks_mod.LiveMathTask.__init__

    def patched(self, data_path=None, shuffle_choices=True):
        original(self, data_path=str(HERE / "fixtures" / "livemath"), shuffle_choices=False)

    tasks_mod.LiveMathTask.__init__ = patched
    try:
        yield
    finally:
        tasks_mod.LiveMathTask.__init__ = original


def run_walker(seed: Seed, state: dict[str, Any], workdir: Path,
               model: "ScriptedModel | None" = None) -> nn.RunResult:
    graph = _graph_of(state)
    model = model or ScriptedModel(_answers(graph))
    with livemath_fixture() if seed.livemath else contextlib.nullcontext():
        return execute(graph, workdir, knobs=_knob_values(state), verdicts=_verdicts(graph),
                       model=model, max_steps=600)


def grader_secrets(seed: Seed, state: dict[str, Any]) -> set[str]:
    """The values only the grader may see, for content search: each item's grader-only
    field values (strings of 8+ characters, and the strings inside a mapping) that no
    node-visible field of the same item also holds."""
    from refactory_dsh.task import load_task

    def strings(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value} if len(value) >= 8 else set()
        if isinstance(value, dict):
            return set().union(*(strings(v) for v in value.values())) if value else set()
        if isinstance(value, (list, tuple)):
            return set().union(*(strings(v) for v in value)) if value else set()
        return set()

    secrets: set[str] = set()
    graph = _graph_of(state)
    refs = sorted({n.task_ref for n in graph.nodes.values() if n.kind == "data" and n.task_ref})
    with livemath_fixture() if seed.livemath else contextlib.nullcontext():
        for ref in refs:
            task = load_task(ref)
            hidden = set(getattr(task, "grader_fields", ()) or ())
            for instance in task.instances():
                visible = strings({k: v for k, v in instance.metadata.items() if k not in hidden})
                for name in hidden:
                    secrets |= strings(instance.metadata.get(name)) - visible
    return secrets


def check_grader(secrets: set[str], hidden: set[str], workdir: Path,
                 model: "ScriptedModel | None") -> list[Violation]:
    """No file a node can read and no prompt holds a grader-only value (by content), and
    no ``current_item.json`` names a grader-only field."""
    out: list[Violation] = []
    for path in sorted(p for p in workdir.rglob("*") if p.is_file()):
        rel = path.relative_to(workdir)
        if path.name in GRADER_RECORDS or path.name.endswith(".done") or ".refactory" in rel.parts:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        leaked = sorted(secret for secret in secrets if secret in text)
        if leaked:
            out.append(Violation("grader", f"{rel} holds grader-only {leaked[0][:60]!r}"))
        if path.name == "current_item.json":
            try:
                named = sorted(hidden & set(json.loads(text)))
            except (ValueError, TypeError):
                named = []
            if named:
                out.append(Violation("grader", f"{rel} names grader-only {named}"))
    for node, prompt, system in (model.log if model is not None else ()):
        leaked = sorted(secret for secret in secrets if secret in prompt or secret in system)
        if leaked:
            out.append(Violation("grader", f"agent {node} was sent grader-only {leaked[0][:60]!r}"))
    return out


def check_slots(state: dict[str, Any], model: "ScriptedModel") -> list[Violation]:
    """No agent was sent a ``{{name}}`` the runtime left unfilled (a knob or a fact it
    needed and did not have)."""
    out = []
    for node, prompt, system in model.log:
        unfilled = sorted(set(SLOT.findall(prompt)) | set(SLOT.findall(system)))
        if unfilled:
            out.append(Violation("slots", f"agent {node} was sent unfilled {unfilled}"))
    return out


def _count_absent_reads(workdir: Path) -> None:
    for marker in workdir.rglob(f"{fg.MISSING_DIR}/*"):
        if marker.is_file():
            NOTES["op_read_absent"] += len(marker.read_text().splitlines())


def _terminals_reached(graph: Graph, workdir: Path) -> set[str]:
    try:
        progress = json.loads((workdir / ".refactory" / "progress.json").read_text())
    except (OSError, ValueError):
        return set()
    passes = nn_edits._data_pass_nodes(graph)  # noqa: SLF001
    ends = {n for n in graph.terminals() if n not in passes}
    return ends & set(progress.get("completed") or ())


def check_walker(seed: Seed, state: dict[str, Any], workdir: Path) -> tuple[list[Violation], Any]:
    graph = _graph_of(state)
    model = ScriptedModel(_answers(graph))
    try:
        result = run_walker(seed, state, workdir, model)
    except Exception as exc:  # noqa: BLE001
        return [Violation("walker", f"raised {type(exc).__name__}: {exc}")], None
    _count_absent_reads(workdir)
    hidden = {name for n in graph.nodes.values() if n.kind == "data" and n.task_ref
              for name in getattr(_task(seed, n.task_ref), "grader_fields", ())}
    seen = check_grader(grader_secrets(seed, state), hidden, workdir, model) + check_slots(state, model)
    if result.halted:
        # A halt is a defined ending: a gate out of reloops (the scripted evaluator always
        # answers proceed, which may be a loop's retry edge) or a halt verdict.
        return seen, result
    if SCRIPTED_LABEL in str(result.error or ""):
        # The scripted evaluator answers `proceed`; a gate with no proceed edge cannot
        # take that answer. What a real evaluator answers is its own business, so the
        # run is inconclusive rather than a finding.
        return seen, None
    if REQUIRED_MISSING in str(result.error or "") or _missing_required(workdir, graph):
        return [*seen, Violation("required", f"a required read was missing at run time: "
                                             f"{str(result.error)[:300]}")], result
    if not result.exit_ok or result.error:
        if _unguaranteed_op_reads(graph):
            # V6 lets an op read a file that may not be there (no runtime checks an op's
            # reads), and a real op that needs it then fails: LiveMath's extract_answer
            # placed before decide. The graph declares nothing that says the read is
            # required, so this is counted, not called a violation (a follow-up gives ops
            # a way to declare required reads).
            NOTES["op_unguaranteed_read_failed"] += 1
            return seen, None
        return [*seen, Violation("walker", f"failed: {result.error}")], result
    if result.gate_stop:
        return [*seen, Violation("walker", f"stopped at gate {result.gate_stop}")], result
    if not _terminals_reached(graph, workdir):
        return [*seen, Violation("walker", "finished without completing a terminal")], result
    return seen, result


def _unguaranteed_op_reads(graph: Graph) -> list[str]:
    """Ops and gates with an optional (not required) read that is not there on every path."""
    avail = available_before(graph)
    return sorted(n.id for n in graph.nodes.values() if n.kind in ("op", "gate")
                  and n.id in graph.reachable()
                  and any(("file", r) not in avail[n.id] for r in n.reads
                          if r not in n.required_reads))


def _missing_required(workdir: Path, graph: Graph) -> list[str]:
    """Required reads `step` found missing (it notes them under ``MISSING_DIR``)."""
    out = []
    for marker in workdir.rglob(f"{fg.MISSING_DIR}/*"):
        node = graph.nodes.get(marker.name)
        if node is not None and marker.is_file():
            out += [n for n in marker.read_text().split() if n in node.required_reads]
    return out


def _task(seed: Seed, ref: str) -> Any:
    from refactory_dsh.task import load_task

    with livemath_fixture() if seed.livemath else contextlib.nullcontext():
        return load_task(ref)


def spine_eligible(state: dict[str, Any]) -> bool:
    """The spine drain answers no user or agent gate, so those graphs are walker-only."""
    return not _verdicts(_graph_of(state))


def _answers(graph: Graph) -> dict[str, str]:
    """Every agent node's scripted answer, named, so both runtimes give each the same one."""
    return {n.id: ANSWERS.get(n.id, "<answer>B</answer>") for n in graph.nodes.values()
            if n.kind == "agent"}


def _compared_facts(graph: Graph, facts: dict[str, Any]) -> dict[str, Any]:
    """The facts both runtimes must agree on: every op's and data node's declared facts.
    An agent's produced fact is set by the spine's worker (`worker_node`), which the
    test drain stands in for without running, so it is left out."""
    declared = {n for node in graph.nodes.values() if node.kind != "agent" for n in node.produces}
    out = {k: v for k, v in facts.items() if k in declared and k != "item_scores"}
    out["item_scores"] = sorted((str(r.get("id")), r.get("score")) for r in facts.get("item_scores") or ())
    return out


def check_spine(seed: Seed, state: dict[str, Any], walker: Any, tmp: Path) -> list[Violation]:
    from test_livemath_blocks import _spine

    graph = _graph_of(state)
    spine_dir = tmp / "spine"
    try:
        rc = _spine(graph, spine_dir, tmp, agent_responses=_answers(graph), mode_state=state)
    except Exception as exc:  # noqa: BLE001
        return [Violation("spine", f"raised {type(exc).__name__}: {exc}")]
    if rc != 0:
        return [Violation("spine", f"the spine drain ended with rc={rc}")]
    try:
        spine_facts = json.loads((spine_dir / "facts.json").read_text())
    except (OSError, ValueError) as exc:
        return [Violation("spine", f"no facts: {exc}")]
    left, right = _compared_facts(graph, walker.facts), _compared_facts(graph, spine_facts)
    if left != right:
        keys = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
        return [Violation("spine", "walker and spine disagree on "
                                   + ", ".join(f"{k} (walker {left.get(k)!r}, spine {right.get(k)!r})"
                                               for k in keys))]
    return []


def check_tools(seed_state: dict[str, Any], state: dict[str, Any]) -> list[Violation]:
    """Spec check: an agent node the edits added runs without tools (the harness never
    allows tools for new nodes)."""
    seed_nodes = set(seed_state["graph"]["nodes"])
    return [Violation("tools", f"new agent {node.id} runs with tools")
            for node in _graph_of(state).nodes.values()
            if node.kind == "agent" and node.id not in seed_nodes
            and (node.tool_policy != "none" or node.tools)]


def check_static(seed_state: dict[str, Any], state: dict[str, Any],
                 protected: tuple[str, ...]) -> list[Violation]:
    return [*check_lowering(state), *check_knobs(state), *check_dataflow(state),
            *check_forks(state), *check_protected(seed_state, state, protected),
            *check_tools(seed_state, state)]


def check_state(applier: Applier, state: dict[str, Any], *, walker: bool = True,
                spine: bool = False, only: tuple[str, ...] = INVARIANTS) -> list[Violation]:
    """Every invariant in ``only`` for one accepted state."""
    found = [v for v in check_static(applier.seed_state, state, applier.protected)
             if v.invariant in only]
    if not (walker or spine) or not ({"walker", "spine", "grader", "slots"} & set(only)):
        return found
    tmp = Path(tempfile.mkdtemp(prefix="structure-fuzz-"))
    try:
        walked, result = check_walker(applier.seed, state, tmp / "walker")
        found += [v for v in walked if v.invariant in only]
        if spine and "spine" in only and result is not None and not walked \
                and spine_eligible(state):
            found += check_spine(applier.seed, state, result, tmp)
            graph = _graph_of(state)
            hidden = {name for n in graph.nodes.values() if n.kind == "data" and n.task_ref
                      for name in getattr(_task(applier.seed, n.task_ref), "grader_fields", ())}
            found += [v for v in check_grader(grader_secrets(applier.seed, state), hidden,
                                              tmp / "spine", None) if "grader" in only]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return found


# ── the edit generator ────────────────────────────────────────────────────


def _rendered_copy(state: dict[str, Any], node_id: str, limit: int = 100_000) -> dict[str, Any]:
    """A node as the optimizer is shown it (`ProposalContext.render`), parsed back."""
    node_state = state["graph"]["nodes"][node_id]
    text = _node_spec(node_state, limit, _knob_values(state))
    body = text.split("```json\n", 1)[1].split("\n```", 1)[0]
    return json.loads(body)


class EditGenerator:
    """Random structure edits: every primitive and macro, valid-looking and malformed."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.counter = 0

    def fresh_id(self, graph: Graph, base: str = "n") -> str:
        self.counter += 1
        candidate = f"{base}{self.counter}"
        while candidate in graph.nodes:
            self.counter += 1
            candidate = f"{base}{self.counter}"
        return candidate

    def _facts(self, graph: Graph) -> list[str]:
        return sorted({n for node in graph.nodes.values() for n in node.produces}) or ["nothing"]

    def _files(self, graph: Graph) -> list[str]:
        return sorted({n for node in graph.nodes.values() for n in node.writes}) or ["none.txt"]

    def new_node(self, state: dict[str, Any], kind: str | None = None) -> dict[str, Any]:
        """A node to insert: a copy of an existing one as rendered, or a fresh one."""
        rng, graph = self.rng, _graph_of(state)
        if graph.nodes and rng.random() < 0.55 and kind is None:
            source = rng.choice(sorted(graph.nodes))
            node = _rendered_copy(state, source)
            node["id"] = self.fresh_id(graph, f"{source}_c")
            if node["kind"] == "agent":
                node["system_prompt_template"] = f"You are copy {node['id']}."
                if rng.random() < 0.5:
                    node["prompt_template"] = f"New ask for {node['id']}: {{{{{rng.choice(self._facts(graph))}}}}}"
                if rng.random() < 0.6:
                    node["writes"] = [f"{node['id']}.txt"]
            # A real op returns the facts its code sets, so only the generic op's copies
            # may declare new outputs; changing a real op's declarations is not a wiring
            # edit, and its runtime refusal would not be a structure finding.
            if node["kind"] == "op" and node.get("call") == fg.STEP and rng.random() < 0.5:
                node["writes"] = [f"{node['id']}.txt"]
                node["produces"] = [f"{node['id']}_fact"]
            return node
        kind = kind or rng.choice(("op", "op", "agent", "agent", "gate", "fork", "join", "data"))
        new_id = self.fresh_id(graph, kind)
        files, facts, ids = self._files(graph), self._facts(graph), sorted(graph.nodes) or [new_id]
        if kind == "op":
            reads = rng.sample(files, k=min(len(files), rng.choice((0, 0, 1, 2))))
            required = [r for r in reads if rng.random() < 0.5]
            return {"id": new_id, "kind": "op", "call": fg.STEP, "reads": reads,
                    **({"required_reads": required} if required else {}),
                    "writes": [f"{new_id}.txt"], "produces": [f"{new_id}_fact"]}
        if kind == "agent":
            # Most new agents omit tool_policy (the edit layer gives them 'none'); some
            # say 'none'; a few ask for tools, which the attempt path refuses. Some read
            # a file, the item's included: what an agent added to cheat would read.
            policy = rng.choice(({},) * 5 + ({"tool_policy": "none"},) * 3 + ({"tool_policy": None},))
            return {"id": new_id, "kind": "agent", "role": "solver", **policy,
                    "reads": rng.sample(files, k=min(len(files), rng.choice((0, 0, 1)))),
                    "prompt_template": f"Think about {{{{{rng.choice(facts)}}}}}",
                    "system_prompt_template": f"You are {new_id}.",
                    "writes": rng.choice(([f"{new_id}.txt"],) * 6 + ([], [f"{new_id}.txt", f"{new_id}.md"])),
                    "produces": [f"{new_id}_fact"]}
        if kind == "gate":
            gate_kind = rng.choice(("fn", "fn", "user", "agent"))
            return {"id": new_id, "kind": "gate", "gate_kind": gate_kind,
                    "evaluator": fg.PROCEED if gate_kind == "fn" else "",
                    "max_iterations": rng.choice((1, 2))}
        if kind == "fork":
            return {"id": new_id, "kind": "fork", "targets": rng.sample(ids, k=min(len(ids), 2))}
        if kind == "join":
            return {"id": new_id, "kind": "join", "sources": rng.sample(ids, k=min(len(ids), 2))}
        return {"id": new_id, "kind": "data", "subgraph_entry": rng.choice(ids),
                "subgraph_exit": rng.choice(ids), "task_ref": fg.TASK_REF,
                "writes": ["current_item.json"], "produces": ["item_scores"]}

    def malformed(self, state: dict[str, Any]) -> Any:
        rng, graph = self.rng, _graph_of(state)
        ids = sorted(graph.nodes)
        some = rng.choice(ids) if ids else "x"
        node = self.new_node(state)
        choice = rng.randrange(12)
        if choice == 0:
            return rng.choice(([], "insert_after", None, 7, {"op": 3}))
        if choice == 1:
            return {"op": rng.choice(("teleport", "text_patch", "replace_node", ""))}
        if choice == 2:
            return {"op": "add_node", "node": {**node, "colour": "red"}, "position": {"after": some}}
        if choice == 3:
            return {"op": "insert_after", "target": some,
                    "node": {**node, "prompt_template": "Hi [... 120 characters omitted ...] there",
                             "kind": "agent", "writes": ["x.txt"]}}
        if choice == 4:
            return {"op": "insert_before", "target": some,
                    "node": {**node, "kind": "agent", "prompt_template": "unclosed {{slot",
                             "writes": ["y.txt"]}}
        if choice == 5:
            # A raw copy, bindings and all: it binds the original's knobs.
            raw = dict(state["graph"]["nodes"][some])
            raw["id"] = self.fresh_id(graph, "raw")
            return {"op": "insert_after", "target": some, "node": raw}
        if choice == 6:
            return {"op": "add_node", "node": {**node, "inline_items": [{"id": "x"}]},
                    "position": {"terminal": True}}
        if choice == 7:
            return {"op": rng.choice(("bypass", "duplicate", "loop", "wrap")), "target": "ghost"}
        if choice == 8:
            return {"op": "add_node", "node": node, "position": rng.choice(
                ({"after": some, "before": some}, {"sideways": True}, "after", {"after": "ghost"}))}
        if choice == 9:
            if rng.random() < 0.5:
                return {"op": "insert_after", "target": some,
                        "node": {"id": self.fresh_id(graph, "badreq"), "kind": "op", "call": fg.STEP,
                                 "reads": [], "required_reads": ["nowhere.txt"]}}
            return {"op": "split", "target": some, "into": [node]}
        if choice == 10:
            return {"op": "add_node", "node": {**node, "id": some}, "position": {"after": some}}
        return {"op": "remove_edge", "source": some, "target": rng.choice(ids) if ids else "y"}

    def text_knob_edit(self, state: dict[str, Any], name: str | None = None) -> dict[str, Any]:
        """A text ``set_knob``: a whole new value, or a ``patch`` of text edits."""
        rng = self.rng
        knobs = state.get("knobs") or {}
        texts = sorted(k for k, v in knobs.items() if (v.get("spec") or {}).get("kind") == "str")
        if name is None:
            if not texts:
                return {"op": "set_knob", "name": "ghost", "value": "x"}
            name = rng.choice(texts)
        self.counter += 1
        current = str((knobs.get(name) or {}).get("value") or "")
        form = rng.random()
        if form < 0.4:
            return {"op": "set_knob", "name": name, "value": f"{current}\nCompound edit {self.counter}."}
        if form < 0.7 or not current.strip():
            return {"op": "set_knob", "name": name,
                    "patch": [{"kind": "append", "content": f"Also check step {self.counter}."}]}
        start = rng.randrange(max(1, len(current) - 12))
        target = current[start:start + rng.choice((6, 12))]
        kind = rng.choice(("replace", "insert_after", "insert_before", "delete"))
        item = {"kind": kind, "target": target}
        if kind != "delete":
            item["content"] = f" (revised {self.counter}) "
        return {"op": "set_knob", "name": name, "patch": [item]}

    def compound(self, state: dict[str, Any]) -> dict[str, Any]:
        """A compound proposal: 2-4 mixed edits, each made against the state the ones
        before it leave (when they apply), so later edits can use what earlier ones
        added -- a copied node's own prompt knob above all. Now and then one edit is
        malformed, or the list is one longer than the run allows."""
        rng = self.rng
        size = rng.choice((2, 2, 3, 3, 4))
        if rng.random() < 0.04:
            size = DEFAULT_MAX_EDITS + 1
        edits: list[Any] = []
        current = state
        new_agents: list[str] = []
        for _ in range(size):
            roll = rng.random()
            if new_agents and roll < 0.6:
                edit = self.text_knob_edit(current, f"{new_agents.pop()}__system_prompt")
            elif roll < 0.4:
                edit = self.text_knob_edit(current)
            elif roll < 0.6 and _graph_of(current).nodes:
                target = rng.choice(sorted(_graph_of(current).nodes))
                agents = sorted(n for n, node in _graph_of(current).nodes.items()
                                if node.kind == "agent")
                if agents:
                    source = rng.choice(agents)
                    node = _rendered_copy(current, source)
                    node["id"] = self.fresh_id(_graph_of(current), f"{source}_c")
                    node["system_prompt_template"] = f"You are copy {node['id']}."
                    node["writes"] = [f"{node['id']}.txt"]
                    node["produces"] = [f"{node['id']}_fact"]
                    edit = {"op": rng.choice(("insert_after", "insert_before")),
                            "target": target, "node": node}
                else:
                    edit = self.edit(current)
            elif roll < 0.67:
                edit = self.malformed(current)
            else:
                edit = self.edit(current)
            edits.append(edit)
            try:
                after = nn_edits.apply(edit, current)
            except Exception:  # noqa: BLE001 - the attempt path refuses it; keep generating
                continue
            if isinstance(edit, dict):
                added = sorted(set(_graph_of(after).nodes) - set(_graph_of(current).nodes))
                new_agents += [n for n in added if f"{n}__system_prompt" in (after.get("knobs") or {})]
            current = after
        return {"edits": edits}

    def edit(self, state: dict[str, Any]) -> Any:
        rng, graph = self.rng, _graph_of(state)
        if rng.random() < 0.08:
            return self.malformed(state)
        ids = sorted(graph.nodes)
        if not ids:
            return {"op": "add_node", "node": self.new_node(state), "position": {"start": True}}
        op = rng.choice(sorted(nn_edits.enabled("structure")))
        target = rng.choice(ids)
        knobs = sorted(state.get("knobs") or {})
        if op == "add_node":
            position = rng.choice(({"after": target}, {"before": target}, {"start": True},
                                   {"terminal": True}, {"detached": True}))
            return {"op": "add_node", "node": self.new_node(state), "position": position}
        if op == "remove_node":
            return {"op": "remove_node", "id": target}
        if op == "add_edge":
            source = target
            is_gate = graph.nodes[source].kind == "gate"
            condition = rng.choice(CONDITIONS) if is_gate or rng.random() < 0.15 else None
            return {"op": "add_edge", "source": source, "target": rng.choice(ids),
                    **({"condition": condition} if condition or rng.random() < 0.5 else {})}
        if op == "remove_edge":
            if not graph.edges:
                return {"op": "remove_edge", "source": target, "target": target}
            edge = rng.choice(graph.edges)
            return {"op": "remove_edge", "source": edge.source, "target": edge.target,
                    **({"condition": edge.condition} if rng.random() < 0.6 else {})}
        if op == "set_knob":
            if not knobs:
                return {"op": "set_knob", "name": "ghost", "value": 1}
            name = rng.choice(knobs)
            spec = nn.Knob.from_spec(state["knobs"][name]["spec"])
            value = spec.sample(rng, state["knobs"][name]["value"])
            if spec.kind == "str" and rng.random() < 0.5:
                value = f"{state['knobs'][name]['value']}\nAlso, edit {self.counter}."
            return {"op": "set_knob", "name": name, "value": value}
        if op == "add_knob":
            self.counter += 1
            return {"op": "add_knob", "name": f"k{self.counter}",
                    "spec": {"default": "x", "kind": "str"}}
        if op == "remove_knob":
            return {"op": "remove_knob", "name": rng.choice(knobs) if knobs else "ghost"}
        if op in ("insert_after", "insert_before"):
            return {"op": op, "target": target, "node": self.new_node(state)}
        if op == "wrap":
            kind = "gate" if len(graph.out_edges(target)) > 1 or rng.random() < 0.3 else None
            return {"op": "wrap", "target": target, "node": self.new_node(state, kind)}
        if op == "bypass":
            return {"op": "bypass", "target": target}
        if op == "duplicate":
            return {"op": "duplicate", "target": target,
                    **({"new_id": self.fresh_id(graph, f"{target}_d")} if rng.random() < 0.5 else {})}
        if op == "split":
            first = self.new_node(state)
            second = self.new_node(state)
            if second["id"] == first["id"]:
                second["id"] = first["id"] + "b"
            return {"op": "split", "target": target, "into": [first, second]}
        if op == "reorder":
            plain = [e for e in graph.edges if e.condition is None]
            if plain and rng.random() < 0.85:
                edge = rng.choice(plain)
                return {"op": "reorder", "first": edge.source, "second": edge.target}
            return {"op": "reorder", "first": target, "second": rng.choice(ids)}
        if op == "loop":
            gate = self.new_node(state, "gate")
            return {"op": "loop", "target": target, "gate": gate}
        raise AssertionError(f"unhandled op {op}")  # pragma: no cover - catalogue is closed


# ── the run ───────────────────────────────────────────────────────────────


@dataclass
class Finding:
    seed: str
    edits: list[Any]
    violations: list[Violation]


@dataclass
class Stats:
    tried: int = 0
    accepted: int = 0
    refused: Counter = field(default_factory=Counter)
    violations: Counter = field(default_factory=Counter)
    walker_runs: int = 0
    spine_runs: int = 0
    findings: list[Finding] = field(default_factory=list)
    notes: Counter = field(default_factory=Counter)
    compound_tried: int = 0
    compound_accepted: int = 0
    compound_refused: Counter = field(default_factory=Counter)
    #: Compound refusals that named the edit that failed (``[edit k of n``).
    compound_named: int = 0

    def summary(self) -> dict[str, Any]:
        observed = {k: v for k, v in self.violations.items() if k in OBSERVED or k == "attempt"}
        spec = {k: v for k, v in self.violations.items() if k not in observed}
        return {"tried": self.tried, "accepted": self.accepted,
                "refused": dict(sorted(self.refused.items())),
                "violations": dict(sorted(self.violations.items())),
                "observed": dict(sorted(observed.items())),
                "spec": dict(sorted(spec.items())),
                "notes": dict(sorted(self.notes.items())),
                "walker_runs": self.walker_runs, "spine_runs": self.spine_runs,
                "compound": {"tried": self.compound_tried, "accepted": self.compound_accepted,
                             "refused": dict(sorted(self.compound_refused.items())),
                             "refusals_naming_the_edit": self.compound_named},
                "findings": len(self.findings)}


class Fuzzer:
    """Random walks of edits from each seed, every accepted state checked."""

    def __init__(self, seed: int, *, seed_names: tuple[str, ...] | None = None,
                 walk_length: int = 12, walker: bool = True, spine_every: int = 0,
                 shrink: bool = True, stop_on: str | None = None,
                 compound: float = 0.0) -> None:
        self.seed = seed
        self.seeds = {name: s for name, s in seeds().items()
                      if seed_names is None or name in seed_names}
        self.walk_length = walk_length
        self.walker = walker
        self.spine_every = spine_every
        self.shrink = shrink
        #: Stop once a violation of this invariant is found (the mutation checks).
        self.stop_on = stop_on
        #: The share of a walk's steps that are compound proposals (0 is none).
        self.compound = compound
        self.stats = Stats()

    def run(self, edits: int) -> Stats:
        names = sorted(self.seeds)
        NOTES.clear()
        walk = 0
        while self.stats.tried < edits and not self._stopped():
            name = names[walk % len(names)]
            self._walk(self.seeds[name], walk, edits)
            walk += 1
        self.stats.notes.update(NOTES)
        return self.stats

    def _walk(self, seed: Seed, walk: int, budget: int) -> None:
        rng = random.Random(f"{self.seed}:{seed.name}:{walk}")
        applier = Applier(seed)
        generator = EditGenerator(rng)
        current = Candidate.from_state(applier.seed_state)
        seen = {current.id}
        accepted: list[Any] = []
        for _ in range(self.walk_length):
            if self.stats.tried >= budget or self._stopped():
                return
            bundle = bool(self.compound) and rng.random() < self.compound
            edit = generator.compound(current.state) if bundle else generator.edit(current.state)
            self.stats.tried += 1
            self.stats.compound_tried += int(bundle)
            try:
                outcome = applier.attempt(current, edit, seen)
            except Exception as exc:  # noqa: BLE001 - the attempt path must never raise
                self._record(seed, applier, [*accepted, edit],
                             [Violation("attempt", f"raised {type(exc).__name__}: {exc}")])
                continue
            if bundle:
                wrong = check_compound(applier, current, edit["edits"], outcome)
                if isinstance(outcome, list) and outcome:
                    self.stats.compound_refused[reason_code(outcome[0])] += 1
                    self.stats.compound_named += int("[edit " in outcome[0])
                else:
                    self.stats.compound_accepted += int(not isinstance(outcome, list))
                if wrong:
                    self._record(seed, applier, [*accepted, edit], wrong, minimize=False)
                    return
            if isinstance(outcome, list):
                code = reason_code(outcome[0]) if outcome else "?"
                self.stats.refused[code] += 1
                if code == "E-internal":
                    # An operator that raised is a bug in the edit layer, not a refusal.
                    self._record(seed, applier, [*accepted, edit],
                                 [Violation("attempt", outcome[0])])
                continue
            self.stats.accepted += 1
            accepted.append(edit)
            seen.add(outcome.id)
            spine = bool(self.spine_every) and self.stats.accepted % self.spine_every == 0
            found = check_state(applier, outcome.state, walker=self.walker, spine=spine)
            self.stats.walker_runs += int(self.walker)
            self.stats.spine_runs += int(spine and spine_eligible(outcome.state))
            if found:
                self._record(seed, applier, list(accepted), found, spine=spine)
                return  # a broken state is not a parent worth editing further
            current = outcome
            if len(current.state["graph"]["nodes"]) > MAX_NODES:
                return

    def _stopped(self) -> bool:
        return self.stop_on is not None and self.stats.violations[self.stop_on] > 0

    def _record(self, seed: Seed, applier: Applier, edits: list[Any],
                found: list[Violation], *, spine: bool = False, minimize: bool = True) -> None:
        for violation in found:
            self.stats.violations[violation.invariant] += 1
        if self.shrink and minimize:
            edits = shrink(applier, edits, found[0].invariant, spine=spine)
        self.stats.findings.append(Finding(seed.name, edits, found))


def _first_failing(applier: Applier, parent: Candidate, edits: list[Any]
                   ) -> tuple[int | None, dict[str, Any]]:
    """The index of the edit that fails on its own, and the state the edits reach when
    none does. As the attempt path checks them: every edit's shape and op first (not an
    object, not in the catalogue), then each in order against the state the ones before
    it left (precheck, apply)."""
    state = parent.state
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict) or nn_edits.unknown(edit.get("op")) is not None:
            return index, state
    for index, edit in enumerate(edits):
        if nn_edits.precheck(edit, state):
            return index, state
        try:
            state = nn_edits.apply(edit, state, protected=applier.protected)
        except nn_edits.EditError:
            return index, state
    return None, state


def check_compound(applier: Applier, parent: Candidate, edits: list[Any],
                   outcome: Candidate | list[str]) -> list[Violation]:
    """The ``compound`` invariant for one bundle and what the attempt path made of it."""
    if len(edits) > DEFAULT_MAX_EDITS:
        if isinstance(outcome, list) and outcome and outcome[0].startswith("E-too-many-edits"):
            return []
        return [Violation("compound", f"{len(edits)} edits were not refused as too many: "
                                      f"{outcome if isinstance(outcome, list) else outcome.id}")]
    failing, state = _first_failing(applier, parent, edits)
    if isinstance(outcome, Candidate):
        found = []
        if failing is not None:
            found.append(Violation("compound", f"accepted although edit {failing + 1} of "
                                               f"{len(edits)} fails on its own"))
        elif outcome.state != state:
            found.append(Violation("compound", "the accepted state is not the edits applied "
                                               "in order"))
        if outcome.own_edits() != list(edits) or outcome.step_edits != len(edits):
            found.append(Violation("compound", "the candidate does not record every edit of "
                                               "its proposal"))
        return found
    reason = outcome[0] if outcome else ""
    if failing is None:
        if "[edit " in reason:
            return [Violation("compound", f"a refusal of the combined state names an edit: "
                                          f"{reason[:300]}")]
        return []
    if f"[edit {failing + 1} of {len(edits)}:" not in reason:
        return [Violation("compound", f"edit {failing + 1} of {len(edits)} fails on its own, "
                                      f"but the refusal does not name it: {reason[:300]}")]
    return []


def violations_of(applier: Applier, edits: list[Any], *, spine: bool = False,
                  only: tuple[str, ...] = INVARIANTS) -> list[Violation] | None:
    """The violations of the state ``edits`` reach, or None when one of them is refused.

    An edit list whose last edit makes the attempt path raise reports that as the
    ``attempt`` violation.
    """
    current = Candidate.from_state(applier.seed_state)
    seen = {current.id}
    for index, edit in enumerate(edits):
        try:
            outcome = applier.attempt(current, edit, seen)
        except Exception as exc:  # noqa: BLE001
            if index == len(edits) - 1:
                return [Violation("attempt", f"raised {type(exc).__name__}: {exc}")]
            return None
        if isinstance(outcome, list):
            if index == len(edits) - 1 and outcome and reason_code(outcome[0]) == "E-internal":
                return [Violation("attempt", outcome[0])]
            return None
        current = outcome
        seen.add(current.id)
    return check_state(applier, current.state,
                       walker=bool({"walker", "spine", "grader", "slots"} & set(only)),
                       spine=spine, only=only)


def shrink(applier: Applier, edits: list[Any], invariant: str, *, spine: bool = False) -> list[Any]:
    """The shortest sub-list of ``edits`` (greedy, one edit at a time) that still
    violates ``invariant``."""
    only = (invariant,) if invariant != "spine" else ("walker", "spine")

    def still(candidate: list[Any]) -> bool:
        found = violations_of(applier, candidate, spine=spine, only=only)
        return bool(found) and any(v.invariant == invariant for v in found)

    current = list(edits)
    index = 0
    while index < len(current) - 1:
        trial = current[:index] + current[index + 1:]
        if still(trial):
            current = trial
        else:
            index += 1
    return current


# ── the corpus ────────────────────────────────────────────────────────────


def save_case(finding: Finding, name: str, *, fixed_by: str = "") -> Path:
    CORPUS.mkdir(parents=True, exist_ok=True)
    path = CORPUS / f"{name}.json"
    path.write_text(json.dumps({
        "seed": finding.seed,
        "edits": finding.edits,
        "found": [str(v) for v in finding.violations],
        "fixed_by": fixed_by,
    }, indent=2, default=str) + "\n")
    return path


def corpus() -> list[Path]:
    return sorted(CORPUS.glob("*.json"))


def replay_case(path: Path, *, spine: bool = False) -> tuple[list[Violation], list[str] | None]:
    """A corpus case replayed: the violations of every state it reaches, and the refusal
    that stopped it (None when every edit was accepted)."""
    case = json.loads(path.read_text())
    applier = Applier(seeds()[case["seed"]])
    states, refusal = applier.replay(case["edits"])
    found: list[Violation] = []
    for candidate in states[1:]:
        found += check_state(applier, candidate.state, walker=True, spine=spine)
    return found, refusal
