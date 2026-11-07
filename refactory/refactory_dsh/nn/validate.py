"""The validator: what makes architecture-in-the-state safe.

PyTorch gets structural safety for free because ``nn.Module`` owns the
architecture and ``load_state_dict(strict=True)`` only has to check that the
parameters match it. We put the architecture in the state dict, which is what
makes architecture search possible, so something has to play that role. This
module is that something.

The validator answers one question: is this state a well-formed, runnable mode?
It never changes the state. An edit that is not well-formed is rejected with a
reason, because a repaired edit would be scored as if it were the edit that was
proposed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .graph import Graph, Node
from .knob import Knob, KnobError
from .resolve import ResolutionError, resolve


#: One line per rule, so the prompt and the code cannot drift apart.
#:
#: The optimizer is told these up front. Telling it only when it breaks one costs an
#: attempt per rule, which is the difference between learning the shape of a legal
#: edit and rediscovering it. A test asserts this table and the codes the validator
#: can actually emit are the same set, in both directions.
RULES: dict[str, str] = {
    "V0": "the state must be readable: it needs a graph, and the graph must parse",
    "V1": "the graph must name a start node, and it must exist",
    "V2": "every node must be reachable from the start node",
    "V3": "the graph must have a terminal (a node with no out-edge)",
    "V4": "a non-gate node has at most one out-edge, and it carries no verdict",
    "V5": "a gate has at least two out-edges, each with its own condition, and no "
          "unconditional out-edge",
    "V6": "every file a node reads must be written upstream of it on every path that "
          "reaches it; an op or a gate may also read what a node in a loop with it wrote "
          "the round before, or what its data node's pass wrote",
    "V7": "every knob a node consumes must be declared, and every declared knob "
          "must be consumed by some node",
    "V8": "every call and evaluator must resolve to a real callable",
    "V9": "every knob value must be of the declared kind and within its bounds; a required knob must be non-empty",
    "V10": "every cycle must pass through a gate; after explicit reloop edges are "
           "removed, each remaining cyclic region must have exactly one gate",
    "V11": "every declared port must be a file some node on a path writes",
    "V12": "a fork's targets, a join's sources and a data node's subgraph must name "
           "nodes that exist; a data node's entry reaches its exit, which is not a gate and "
           "has no out-edge (its pass ends there); and a pass is entered only through its "
           "data node, never by an edge from the data node or from outside, and never "
           "leads back to it",
    "V13": "a data node must not appear inside a fork branch",
    "V14": "every field binding must name a declared knob of the field's kind, once per "
           "field, and the field must be in the bindable allowlist",
    "V15": "an agent node's inputs are required: every file it reads and every {{name}} "
           "its prompts fill in that is not a knob must be produced upstream on every path "
           "that reaches it (a data pass's own files and facts stay inside the pass), and a "
           "tool_policy 'none' agent declares exactly one write",
    "V16": "a fork has one unconditional out-edge, to its join, and each join belongs to "
           "one fork; branches are disjoint, never lead back to the fork or past the join, "
           "declare no shared facts or conflicting files, and a gate in a branch reloops "
           "only within it; a join is entered only from its fork's branches and waits only "
           "for nodes every run of their branch visits",
    "I:": "the mode's own invariant, stated in Mode.invariants()",
}


def rules_text() -> str:
    """The rules, as the optimizer prompt shows them."""
    return "\n".join(f"- `{code}` — {text}" for code, text in RULES.items())


@dataclass(frozen=True)
class Issue:
    """One reason a state was refused."""

    code: str
    message: str
    node: str | None = None
    knob: str | None = None

    def __str__(self) -> str:
        where = f" [{self.node}]" if self.node else (f" [{self.knob}]" if self.knob else "")
        return f"{self.code}{where}: {self.message}"


@dataclass(frozen=True)
class Invariant:
    """A mode-declared rule, for the parts only the author can know.

    Metric integrity is the motivating example: the fact ``score()`` reads has to
    be produced by the graph, or the score is not about the mode any more. The
    check returns a violation message, or ``None`` when the state is acceptable.
    """

    name: str
    description: str
    check: Callable[[Graph, Mapping[str, Any]], str | None]


def on_path_nodes(graph: Graph) -> set[str]:
    """Nodes that lie on some ``start -> terminal`` path.

    A producer sitting off every path the run can take does not make a fact
    available, so a check that counts it is weaker than the promise it makes. This
    is a static approximation: nothing here can know which verdicts a gate will
    actually return, only which paths exist.
    """
    from_start = graph.reachable()
    reverse: dict[str, list[str]] = {}
    for edge in graph.edges:
        reverse.setdefault(edge.target, []).append(edge.source)

    reaches_terminal: set[str] = set()
    stack = list(graph.terminals())
    while stack:
        node_id = stack.pop()
        if node_id in reaches_terminal:
            continue
        reaches_terminal.add(node_id)
        stack.extend(reverse.get(node_id, ()))
    return from_start & reaches_terminal


def produced_on_path(graph: Graph) -> set[str]:
    """Every write and fact produced by a node on a ``start -> terminal`` path."""
    on_path = on_path_nodes(graph)
    return {
        name
        for node_id, node in graph.nodes.items()
        if node_id in on_path
        for name in (*node.writes, *node.produces)
    }


def validate(
    graph: Graph,
    values: Mapping[str, Any],
    *,
    declared_knobs: Mapping[str, Knob],
    invariants: tuple[Invariant, ...] = (),
    resolve_calls: bool = True,
    ports: tuple[Any, ...] = (),
) -> list[Issue]:
    """Every reason ``(graph, values)`` is not a runnable, scoreable mode."""
    issues: list[Issue] = []

    # V1 start node exists.
    if not graph.start:
        issues.append(Issue("V1", "the graph declares no start node"))
    elif graph.start not in graph.nodes:
        issues.append(Issue("V1", f"start node {graph.start!r} does not exist"))

    if not graph.nodes:
        issues.append(Issue("V1", "the graph has no nodes"))
        return _sorted(issues)

    # V2 reachability.
    reachable = graph.reachable()
    for node_id in sorted(set(graph.nodes) - reachable):
        issues.append(
            Issue("V2", "node is not reachable from the start node", node=node_id)
        )

    # V3 a terminal exists.
    if not graph.terminals():
        issues.append(Issue("V3", "the graph has no terminal node"))

    # V4/V5 edge shape around every node.
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        out = graph.out_edges(node_id)
        unconditional = [e for e in out if e.condition is None]
        conditioned = [e for e in out if e.condition is not None]
        if node.kind == "gate":
            if len(conditioned) < 2:
                issues.append(
                    Issue(
                        "V5",
                        f"a gate needs at least two conditioned out-edges, has {len(conditioned)}",
                        node=node_id,
                    )
                )
            if unconditional:
                issues.append(
                    Issue(
                        "V5",
                        "a gate must not have an unconditional out-edge; "
                        "every edge out of a gate carries a verdict",
                        node=node_id,
                    )
                )

        elif len(out) > 1:
            issues.append(
                Issue(
                    "V4",
                    f"a non-gate node needs at most one out-edge, has {len(out)}",
                    node=node_id,
                )
            )
        elif conditioned:
            issues.append(
                Issue(
                    "V4",
                    "a non-gate node must not branch; only a gate's out-edges "
                    "carry verdicts",
                    node=node_id,
                )
            )

    issues.extend(gate_label_issues(graph))

    # V12 the nodes a fork, a join or a data node names.
    #
    # These are how a fork reaches its branches and a join waits for them, so a
    # name that is not a node is a barrier that can never be satisfied — a failure
    # at run time that the state itself already knows about.
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        for kind, names in (
            ("fork target", node.targets),
            ("join source", node.sources),
            ("data subgraph", (node.subgraph_entry, node.subgraph_exit)),
        ):
            for name in names:
                if name and name not in graph.nodes:
                    issues.append(
                        Issue(
                            "V12",
                            f"{kind} {name!r} is not a node",
                            node=node_id,
                        )
                    )
        if node.kind == "data":
            issues.extend(_data_pass_issues(graph, node))
        # Both runtimes end an item's pass once its exit has run, so a node wired
        # after the exit would never run as part of it.
        if node.kind == "data" and node.subgraph_exit in graph.nodes:
            after = sorted(e.target for e in graph.out_edges(node.subgraph_exit))
            if after:
                issues.append(
                    Issue(
                        "V12",
                        f"data subgraph exit {node.subgraph_exit!r} has an out-edge to "
                        f"{after}; a pass ends at its exit, so nothing after it runs",
                        node=node_id,
                    )
                )

    # V13 data-inside-fork.
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if node.kind != "fork":
            continue
        barrier = graph.fork_join(node_id)
        for target in node.targets:
            seen: set[str] = set()
            pending = [target]
            while pending:
                current = pending.pop()
                if current == barrier or current in seen:
                    continue
                if current not in graph.nodes:
                    continue
                seen.add(current)
                member = graph.nodes[current]
                if member.kind == "data":
                    issues.append(Issue(
                        "V13",
                        f"fork {node_id!r}: data node {current!r} inside a fork branch "
                        "is not supported; move the data node outside the fork or "
                        "place the fork inside the data body",
                        node=node_id,
                    ))
                pending.extend(e.target for e in graph.out_edges(current))
                if member.kind == "fork" and member.targets:
                    pending.extend(member.targets)

    # V14 field bindings — allowlist, knob kind, one binding per field.
    from .graph import BINDABLE_FIELDS

    #: The knob kind each bindable field takes, so both runtimes refuse a mismatch
    #: here rather than one failing at lowering and the other at run time.
    _BINDING_KINDS = {
        "system_prompt_template": "str",
        "prompt_template": "str",
        "max_iterations": "int",
        "parallelism": "int",
    }

    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        seen_fields: set[str] = set()
        for pair in getattr(node, "field_bindings", ()):
            field_name = pair[0] if isinstance(pair, (tuple, list)) and len(pair) >= 1 else ""
            knob_name = pair[1] if isinstance(pair, (tuple, list)) and len(pair) >= 2 else ""
            if field_name not in BINDABLE_FIELDS:
                issues.append(
                    Issue(
                        "V14",
                        f"field {field_name!r} is not bindable; "
                        f"allowed: {sorted(BINDABLE_FIELDS)}",
                        node=node_id,
                    )
                )
            elif field_name == "parallelism" and node.kind != "data":
                issues.append(
                    Issue(
                        "V14",
                        f"parallelism binding is only for data nodes",
                        node=node_id,
                    )
                )
            if not knob_name:
                issues.append(
                    Issue(
                        "V14",
                        f"field {field_name!r} is bound to no knob (empty name)",
                        node=node_id,
                    )
                )
            elif knob_name not in declared_knobs:
                issues.append(
                    Issue(
                        "V14",
                        f"field binding references undeclared knob {knob_name!r}",
                        node=node_id,
                        knob=knob_name,
                    )
                )
            elif field_name in _BINDING_KINDS:
                kind = declared_knobs[knob_name].kind
                if kind != _BINDING_KINDS[field_name]:
                    issues.append(
                        Issue(
                            "V14",
                            f"field {field_name!r} takes a {_BINDING_KINDS[field_name]!r} "
                            f"knob, but {knob_name!r} is {kind!r}",
                            node=node_id,
                            knob=knob_name,
                        )
                    )
            if field_name == "parallelism" and node.parallelism_knob:
                issues.append(
                    Issue(
                        "V14",
                        "parallelism is bound by both parallelism_knob and a field binding",
                        node=node_id,
                    )
                )
            if field_name in seen_fields:
                issues.append(
                    Issue(
                        "V14",
                        f"field {field_name!r} is bound twice on this node",
                        node=node_id,
                    )
                )
            seen_fields.add(field_name)

    # V6 every read is written upstream; V15 an agent's inputs are there on every path.
    knob_values = dict(values)
    issues.extend(read_issues(graph))
    issues.extend(agent_input_issues(graph, knob_values))

    # V16 forks and joins are paired, and branches are independent.
    issues.extend(fork_issues(graph, knob_values))

    # V7 knobs: consumed ones exist, and declared ones are consumed.
    consumers = graph.knob_consumers()
    for node_id in sorted(graph.nodes):
        for name in graph.nodes[node_id].knobs:
            if name not in declared_knobs:
                issues.append(
                    Issue(
                        "V7",
                        f"consumes undeclared knob {name!r}",
                        node=node_id,
                        knob=name,
                    )
                )
    for name in sorted(declared_knobs):
        if name not in consumers:
            issues.append(
                Issue(
                    "V7",
                    "knob is declared but no node consumes it, so moving it "
                    "cannot change the run",
                    knob=name,
                )
            )

    # V8 callables resolve.
    if resolve_calls:
        for node_id in sorted(graph.nodes):
            node = graph.nodes[node_id]
            target = node.call if node.kind == "op" else (
                node.evaluator if node.kind == "gate" else ""
            )
            if not target:
                continue
            try:
                resolve(target)
            except ResolutionError as exc:
                issues.append(Issue("V8", str(exc), node=node_id))

    # V9 parameter values are in bounds.
    for name in sorted(values):
        knob = declared_knobs.get(name)
        if knob is None:
            issues.append(Issue("V9", f"value for undeclared knob {name!r}", knob=name))
            continue
        try:
            knob.coerce(values[name])
        except KnobError as exc:
            issues.append(Issue("V9", str(exc), knob=name))
    for name in sorted(set(declared_knobs) - set(values)):
        issues.append(Issue("V9", "knob has no value", knob=name))

    # V9 required-knob check.  allow_defaults=True so the starting state (whose
    # required knobs are empty placeholders) passes structural validation; only a
    # candidate that actively sets an empty value is refused.
    from refactory_dsh.nn.knob import check_required_knobs as _check_req

    for knob_name, message in _check_req(declared_knobs, values, allow_defaults=True):
        issues.append(Issue("V9", message, knob=knob_name))

    # V10: determine retry ownership before a candidate executes any work.
    from .loops import loop_edges

    try:
        loop_edges(graph)
    except ValueError as exc:
        issues.append(Issue("V10", str(exc)))

    # V11 every declared port is a file the run actually produces.
    if ports:
        produced = produced_on_path(graph)
        for port in ports:
            path = getattr(port, "path", None) or (port or {}).get("path", "")
            if path and path not in produced:
                issues.append(
                    Issue(
                        "V11",
                        f"declares the port {path!r}, which no node on a path from "
                        f"{graph.start!r} to a terminal writes, so the port would be "
                        "empty",
                        knob=getattr(port, "name", None),
                    )
                )

    # Mode-declared invariants.
    for invariant in invariants:
        try:
            violation = invariant.check(graph, values)
        except Exception as exc:  # a broken invariant is reported, not raised
            violation = f"invariant raised {type(exc).__name__}: {exc}"
        if violation:
            issues.append(Issue(f"I:{invariant.name}", violation))

    return _sorted(issues)


#: A ``{{name}}`` slot in a template, as both runtimes fill it.
_SLOT = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def template_facts(node: Node, knob_values: Mapping[str, Any]) -> set[str]:
    """The run facts an agent's prompts fill in: their ``{{name}}`` slots, less the knobs
    (rendered from the run's knob values) and the node's ``feedback_keys`` (re-derived
    before each dispatch). Read from the bound templates, as both runtimes send them."""
    if node.kind != "agent":
        return set()
    names: set[str] = set()
    for field_name in ("prompt_template", "system_prompt_template"):
        try:
            text = node.bound(field_name, knob_values)
        except Exception:  # a bad binding is V14's to report
            text = getattr(node, field_name)
        names |= set(_SLOT.findall(str(text or "")))
    return names - set(knob_values) - set(node.feedback_keys)


def fork_joins(graph: Graph) -> dict[str, list[str]]:
    """Each join some fork's unconditional out-edge leads to, and the forks that do."""
    out: dict[str, list[str]] = {}
    for node in graph.nodes.values():
        if node.kind != "fork":
            continue
        join = graph.fork_join(node.id)
        if join in graph.nodes and graph.nodes[join].kind == "join":
            out.setdefault(join, []).append(node.id)
    return out


def fork_branches(graph: Graph, fork: Node, join: str | None) -> list[set[str]]:
    """Each branch's members, walked as both runtimes walk it: edges and nested forks'
    targets, stopping at ``join``."""
    branches = []
    for target in fork.targets:
        seen: set[str] = set()
        pending = [target]
        while pending:
            current = pending.pop()
            if current == join or current in seen or current not in graph.nodes:
                continue
            seen.add(current)
            pending.extend(e.target for e in graph.out_edges(current))
            if graph.nodes[current].kind == "fork":
                pending.extend(graph.nodes[current].targets)
        branches.append(seen)
    return branches


def _branch_ends(graph: Graph, branch: set[str], join: str) -> set[str]:
    """The members a run of a branch can end at: an edge to the join, or no out-edge."""
    return {member for member in branch
            if not graph.out_edges(member) or any(e.target == join for e in graph.out_edges(member))}


def _successors(graph: Graph) -> dict[str, set[str]]:
    """Where the run can go next from each node, for dataflow: edges, a fork's targets, a
    data node's entry, a branch's end to its fork's join, and a pass's exit to the data
    node's successor."""
    succ: dict[str, set[str]] = {node_id: set() for node_id in graph.nodes}
    for node in graph.nodes.values():
        succ[node.id] |= {e.target for e in graph.out_edges(node.id) if e.target in succ}
        if node.kind == "fork":
            succ[node.id] |= {t for t in node.targets if t in succ}
        if node.kind == "data":
            if node.subgraph_entry in succ:
                succ[node.id].add(node.subgraph_entry)
            if node.subgraph_exit in succ:
                succ[node.subgraph_exit] |= {e.target for e in graph.out_edges(node.id)}
    for join, forks in fork_joins(graph).items():
        for fork_id in forks:
            for branch in fork_branches(graph, graph.nodes[fork_id], join):
                for end in _branch_ends(graph, branch, join):
                    succ[end].add(join)
    return succ


def cycle_mates(graph: Graph) -> dict[str, set[str]]:
    """For each node, the nodes in a cycle with it: each can reach the other."""
    succ = _successors(graph)
    reach: dict[str, set[str]] = {}
    for start in graph.nodes:
        seen: set[str] = set()
        stack = list(succ[start])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(succ[current])
        reach[start] = seen
    return {node_id: {m for m in reach[node_id] if node_id in reach[m]} for node_id in graph.nodes}


def _data_pass_issues(graph: Graph, data: Node) -> list[Issue]:
    """V12's rules for one data node's pass, as both runtimes run it."""
    entry, exit_ = data.subgraph_entry, data.subgraph_exit
    if entry not in graph.nodes or exit_ not in graph.nodes:
        return []  # a missing name is reported above
    issues: list[Issue] = []
    if graph.nodes[exit_].kind == "gate":
        issues.append(Issue("V12", f"data subgraph exit {exit_!r} is a gate; a pass ends by "
                                   "completing its exit", node=data.id))
    body = graph.data_body(data.id)
    if exit_ not in body:
        issues.append(Issue("V12", f"data subgraph entry {entry!r} cannot reach its exit "
                                   f"{exit_!r}", node=data.id))
    if any(e.target in body for e in graph.out_edges(data.id)):
        issues.append(Issue("V12", "the data node has an edge into its own pass, which would run "
                                   "it again outside the iteration", node=data.id))
    inside = body | {data.id}
    from_outside = sorted({e.source for member in body for e in graph.in_edges(member)
                           if e.source not in inside})
    if from_outside:
        issues.append(Issue("V12", f"its pass is entered from {from_outside}, outside the pass; a "
                                   "pass runs once per item, entered through its data node only",
                            node=data.id))
    back = sorted(member for member in body if any(e.target == data.id
                                                   for e in graph.out_edges(member)))
    if back:
        issues.append(Issue("V12", f"its pass leads back to the data node from {back}",
                            node=data.id))
    return issues


def available_inputs(graph: Graph, *, isolate_passes: bool = True) -> dict[str, set[tuple[str, str]]]:
    """What is there when each node starts, on every path that reaches it.

    ``("file", name)`` and ``("fact", name)`` entries; the greatest fixpoint of: a node
    starts with what every predecessor left; a fork's branches with what the fork had;
    a fork's join with what the fork had and what every run of each branch left; a data
    pass with what its data node had and the data node's writes; the data node's
    successor with what the data node had and declares -- never what its pass made,
    which stays in the pass on both runtimes. With ``isolate_passes`` false, the data
    node's successor also gets what its pass's exit left: what an op may look for.
    """
    def made(node: Node) -> set[tuple[str, str]]:
        return {("file", n) for n in node.writes} | {("fact", n) for n in node.produces}

    universe: set[tuple[str, str]] = set()
    for node in graph.nodes.values():
        universe |= made(node)
        universe |= {("file", n) for n in node.reads}
        universe |= {("fact", n) for f in ("prompt_template", "system_prompt_template")
                     for n in _SLOT.findall(str(getattr(node, f) or ""))}
    joined = {join: forks[0] for join, forks in fork_joins(graph).items() if len(forks) == 1}
    preds: dict[str, list[tuple[str, str]]] = {node_id: [] for node_id in graph.nodes}
    for node in graph.nodes.values():
        if node.kind == "data":
            if node.subgraph_entry in preds:
                preds[node.subgraph_entry].append((node.id, "pass"))
            for edge in graph.out_edges(node.id):
                preds[edge.target].append((node.id, "out"))
            continue
        if node.kind == "fork":
            for target in node.targets:
                if target in preds:
                    preds[target].append((node.id, "in"))
        for edge in graph.out_edges(node.id):
            if edge.target not in joined:
                preds[edge.target].append((node.id, "out"))
    ends = {join: [_branch_ends(graph, branch, join)
                   for branch in fork_branches(graph, graph.nodes[fork], join)]
            for join, fork in joined.items()}
    # Only what the run can reach feeds a node: an unreachable predecessor (V2's to
    # report) leaves nothing on a path, and nothing is guaranteed at an unreachable node.
    reachable = graph.reachable()
    preds = {node_id: [(s, how) for s, how in entries if s in reachable]
             for node_id, entries in preds.items()}
    avail: dict[str, set[tuple[str, str]]] = {node_id: set(universe) for node_id in graph.nodes}
    if graph.start in avail:
        avail[graph.start] = set()

    def left_by(node_id: str, how: str) -> set[tuple[str, str]]:
        node = graph.nodes[node_id]
        if how == "in":
            return avail[node_id]
        if how == "pass":
            return avail[node_id] | {("file", n) for n in node.writes}
        if node.kind == "data" and not isolate_passes and node.subgraph_exit in graph.nodes:
            exit_ = graph.nodes[node.subgraph_exit]
            return avail[exit_.id] | made(exit_) | made(node)
        return avail[node_id] | made(node)

    changed = True
    while changed:
        changed = False
        for node_id in sorted(graph.nodes):
            if node_id == graph.start:
                continue
            if node_id in joined:
                new = set(avail[joined[node_id]])
                for branch_ends in ends[node_id]:
                    if branch_ends:
                        new |= set.intersection(*(left_by(end, "out") for end in branch_ends))
            else:
                incoming = [left_by(source, how) for source, how in preds[node_id]]
                if not incoming:
                    continue
                new = set.intersection(*incoming)
            if new != avail[node_id]:
                avail[node_id] = new
                changed = True
    for node_id in set(graph.nodes) - reachable:
        avail[node_id] = set()
    return avail


def _skippable(graph: Graph, start: str, banned: str, join: str) -> bool:
    """Whether a run of a branch from ``start`` can end, or reach ``join``, without
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


def fork_branch_conflicts(graph: Graph, knob_values: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(fork, reason)`` for each fork whose branches cannot run side by side.

    The rules both runtimes enforce when a fork runs (``nn/fork.py``, and the spine
    through lowering): no branch writes executor metadata, and no two branches declare
    the same fact, depend on each other's facts in a prompt, or touch the same files (a
    directory and a file inside it overlap). One home, so lowering and the validator
    refuse the same shapes with the same words.
    """
    from pathlib import PurePosixPath

    def overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
        return left == right or left in right.parents or right in left.parents

    out: list[tuple[str, str]] = []
    metadata = PurePosixPath(".refactory")
    for fork_id in sorted(graph.nodes):
        fork = graph.nodes[fork_id]
        if fork.kind != "fork":
            continue
        barrier = graph.fork_join(fork_id)
        branches = [[graph.nodes[n] for n in sorted(members)]
                    for members in fork_branches(graph, fork, barrier)]
        facts = [{name for n in branch for name in n.produces} for branch in branches]
        consumed = [{match for n in branch for match in _SLOT.findall(
            str(n.bound("prompt_template", knob_values)))} for branch in branches]
        writes = [[PurePosixPath(p) for n in branch for p in n.writes] for branch in branches]
        reads = [[PurePosixPath(p) for n in branch for p in n.reads] for branch in branches]
        for index in range(len(branches)):
            if any(overlap(path, metadata) for path in writes[index]):
                out.append((fork_id, f"fork {fork_id!r}: a branch declares executor metadata as output"))
                break
            conflict = None
            for other in range(index):
                if facts[index] & facts[other]:
                    conflict = "branches declare the same produced facts"
                elif facts[index] & consumed[other] or facts[other] & consumed[index]:
                    conflict = "a branch prompt depends on a sibling's facts"
                elif (any(overlap(a, b) for a in writes[index] for b in writes[other] + reads[other])
                      or any(overlap(a, b) for a in writes[other] for b in reads[index])):
                    conflict = "branches have conflicting declared file access"
                if conflict:
                    break
            if conflict:
                out.append((fork_id, f"fork {fork_id!r}: {conflict}"))
                break
    return out


def gate_label_issues(graph: Graph) -> list[Issue]:
    """V5: no two out-edges of a gate carry the same verdict (compared as both runtimes
    route a verdict: trimmed, case-insensitively). A verdict names one edge; two leave the
    route to whichever runtime looks first, and the spine refuses the graph outright."""
    issues: list[Issue] = []
    for node_id in sorted(graph.nodes):
        if graph.nodes[node_id].kind != "gate":
            continue
        labels = [str(e.condition).strip().lower() for e in graph.out_edges(node_id)
                  if e.condition is not None]
        twice = sorted({label for label in labels if labels.count(label) > 1})
        if twice:
            issues.append(Issue("V5", f"two or more out-edges carry the verdict(s) {twice}; each "
                                      "verdict names exactly one edge", node=node_id))
    return issues


def read_issues(graph: Graph) -> list[Issue]:
    """V6: every file a node reads is written upstream of it on every path that reaches it.

    An op or a gate may read what a node in a loop with it wrote the round before (a
    retry reads its last attempt's feedback), and a file its data node's pass wrote: it
    may check for one. An agent may not; V15 holds its reads to every path.
    """
    issues: list[Issue] = []
    lenient = available_inputs(graph, isolate_passes=False)
    mates = cycle_mates(graph)
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        last_round = set() if node.kind == "agent" else {
            name for mate in mates.get(node_id, ()) for name in graph.nodes[mate].writes}
        for name in node.reads:
            if ("file", name) not in lenient.get(node_id, set()) and name not in last_round:
                issues.append(Issue(
                    "V6", f"reads {name!r}, which some path from {graph.start!r} to it does not "
                    "write first", node=node_id))
    return issues


def agent_input_issues(graph: Graph, knob_values: Mapping[str, Any]) -> list[Issue]:
    """V15: an agent's inputs -- its reads and its prompts' ``{{facts}}`` -- are required.

    Both runtimes refuse to dispatch an agent whose declared read is missing, and send a
    prompt with an unfilled ``{{name}}`` as it is, so each must be produced upstream on
    every path to the agent; a data pass's own files and facts stay inside the pass. A
    ``tool_policy`` 'none' agent's reply is its one output, so it declares one write.
    """
    issues: list[Issue] = []
    available = available_inputs(graph)
    for node_id in sorted(graph.reachable()):
        node = graph.nodes.get(node_id)
        if node is None or node.kind != "agent":
            continue
        if node.tool_policy == "none" and len(node.writes) != 1:
            issues.append(Issue(
                "V15", f"declares tool_policy 'none' and {len(node.writes)} writes; a no-tools "
                "agent's reply is its one declared output, so it declares exactly one",
                node=node_id))
        have = available.get(node_id, set())
        missing = sorted(f"file {name!r}" for name in node.reads if ("file", name) not in have)
        missing += sorted(f"{{{{{name}}}}}" for name in template_facts(node, knob_values)
                          if ("fact", name) not in have)
        if missing:
            issues.append(Issue(
                "V15", f"needs {', '.join(missing)}, which some path to it does not produce "
                "first; an agent's inputs are required (a data pass's own files and facts "
                "stay inside its pass)", node=node_id))
    return issues


def fork_issues(graph: Graph, knob_values: Mapping[str, Any]) -> list[Issue]:
    """V16: every fork is paired with one join, and its branches are independent."""
    from .loops import loop_edges

    issues: list[Issue] = []
    try:
        retries = loop_edges(graph)
    except Exception:  # an ambiguous cycle is V10's to report
        retries = set()
    owners = fork_joins(graph)
    for fork_id in sorted(graph.nodes):
        fork = graph.nodes[fork_id]
        if fork.kind != "fork":
            continue
        out = graph.out_edges(fork_id)
        join_id = out[0].target if len(out) == 1 and out[0].condition is None else None
        if join_id is None or graph.nodes.get(join_id) is None or graph.nodes[join_id].kind != "join":
            where = (f"leads to {out[0].target!r}, which is not a join" if join_id is not None
                     else f"has {len(out)} out-edge(s)")
            issues.append(Issue("V16", f"a fork's one unconditional out-edge leads to its join, "
                                       f"where its branches meet; this fork {where}", node=fork_id))
            continue
        join = graph.nodes[join_id]
        branches = fork_branches(graph, fork, join_id)
        # What runs after the join, not following a retry back over the fork.
        after: set[str] = set()
        stack = [e.target for e in graph.out_edges(join_id) if e.key() not in retries]
        while stack:
            current = stack.pop()
            if current in after or current not in graph.nodes:
                continue
            after.add(current)
            stack.extend(e.target for e in graph.out_edges(current) if e.key() not in retries)
            if graph.nodes[current].kind == "fork":
                stack.extend(graph.nodes[current].targets)
            if graph.nodes[current].kind == "data" and graph.nodes[current].subgraph_entry:
                stack.append(graph.nodes[current].subgraph_entry)
        members: set[str] = set()
        for target, branch in zip(fork.targets, branches):
            for gate_id in sorted(branch):
                if graph.nodes[gate_id].kind != "gate":
                    continue
                for edge in graph.out_edges(gate_id):
                    if edge.key() in retries and edge.target not in branch:
                        issues.append(Issue("V16", f"gate {gate_id!r} in branch {target!r} reloops "
                                                   f"to {edge.target!r}, outside the branch",
                                            node=fork_id))
            if fork_id in branch:
                issues.append(Issue("V16", f"branch {target!r} leads back to the fork", node=fork_id))
            if branch & members:
                issues.append(Issue("V16", f"branch {target!r} shares {sorted(branch & members)} "
                                           "with another branch", node=fork_id))
            if branch & after:
                issues.append(Issue("V16", f"branch {target!r} runs {sorted(branch & after)}, which "
                                           f"come after the join {join_id!r}", node=fork_id))
            members |= branch
        outside = sorted({e.source for e in graph.in_edges(join_id)} - members - {fork_id})
        if outside:
            issues.append(Issue("V16", f"join {join_id!r} is entered from {outside}, outside this "
                                       "fork's branches", node=fork_id))
        for source in join.sources:
            owner = next((t for t, b in zip(fork.targets, branches) if source in b), None)
            if owner is None:
                issues.append(Issue("V16", f"join {join_id!r} waits for {source!r}, which is in no "
                                           "branch of this fork", node=fork_id))
            elif _skippable(graph, owner, source, join_id):
                issues.append(Issue("V16", f"join {join_id!r} waits for {source!r}, which a run of "
                                           f"branch {owner!r} can finish without", node=fork_id))
    for join_id in sorted(graph.nodes):
        if graph.nodes[join_id].kind != "join":
            continue
        forks = owners.get(join_id, [])
        if len(forks) != 1:
            issues.append(Issue("V16", f"a join belongs to exactly one fork (the one whose out-edge "
                                       f"leads to it); this one belongs to {sorted(forks) or 'none'}",
                                node=join_id))
    for fork_id, reason in fork_branch_conflicts(graph, knob_values):
        issues.append(Issue("V16", reason, node=fork_id))
    return issues


def _sorted(issues: list[Issue]) -> list[Issue]:
    return sorted(issues, key=lambda i: (i.code, i.node or "", i.knob or "", i.message))


def declared_knobs(state: Mapping[str, Any]) -> dict[str, Knob]:
    """The knobs a state declares, as ``Knob`` objects."""
    out: dict[str, Knob] = {}
    for name, entry in (state.get("knobs") or {}).items():
        out[name] = Knob.from_spec(dict(entry["spec"]))
    return out


def validate_state(
    state: Mapping[str, Any],
    *,
    invariants: tuple[Invariant, ...] = (),
    resolve_calls: bool = True,
) -> list[Issue]:
    """Validate a state dict without building a mode around it.

    Everything the check needs is in the state, which is the point of putting the
    architecture there: a candidate can be judged before anything is run.
    """
    try:
        graph = Graph.from_state(state["graph"])
    except KeyError as exc:
        return [Issue("V0", f"the state has no {exc.args[0]!r}")]
    except Exception as exc:  # a malformed graph is an issue, not a crash
        return [Issue("V0", f"the graph is not readable: {exc}")]
    from .declarations import Port

    ports: list[Port] = []
    for entry in state.get("ports") or ():
        try:
            ports.append(Port.from_state(dict(entry)))
        except Exception:  # a malformed port is itself a problem, reported below
            continue
    return validate(
        graph,
        {name: entry["value"] for name, entry in (state.get("knobs") or {}).items()},
        declared_knobs=declared_knobs(state),
        invariants=invariants,
        resolve_calls=resolve_calls,
        ports=tuple(ports),
    )


def fact_produced(*facts: str) -> Invariant:
    """An invariant requiring the graph to produce the facts ``score()`` reads.

    Path-aware: the producer has to lie on a ``start -> terminal`` path, not merely
    exist somewhere in the graph.

    This is the metric-integrity rule, and it is the one that makes removing work
    a legitimate edit rather than a cheat: a candidate may drop any node it likes,
    as long as the number being maximized is still produced by the run.
    """

    wanted = tuple(facts)

    def check(graph: Graph, _values: Mapping[str, Any]) -> str | None:
        on_path = on_path_nodes(graph)
        produced = {
            name
            for node_id, node in graph.nodes.items()
            if node_id in on_path
            for name in node.produces
        }
        missing = [name for name in wanted if name not in produced]
        if missing:
            return (
                f"the score reads {missing}, which no node on a path from "
                f"{graph.start!r} to a terminal produces, so the number would not "
                "describe this mode's run; a producer that sits off every path the "
                "run can take does not count"
            )
        return None

    return Invariant("fact_produced", f"graph produces {list(wanted)}", check)


def artifact_produced(*paths: str, by: str) -> Invariant:
    """Keep a named artifact producer in the executable graph during search.

    Use this for a required output whose meaning is not measured by ``score()``.
    A fallback writer of the same filename does not replace the named producer.
    Like ``fact_produced``, this is a structural check: it does not prove the
    file's contents, or that every conditional route visits the producer. The
    mode's run checks and acceptance review must establish those properties.
    """
    wanted = tuple(paths)
    if not wanted or any(not path for path in wanted) or not by:
        raise ValueError("artifact_produced needs nonempty paths and a producer node")

    def check(graph: Graph, _values: Mapping[str, Any]) -> str | None:
        producer = graph.nodes.get(by)
        if producer is None or by not in on_path_nodes(graph):
            return (
                f"required artifact producer {by!r} for {list(wanted)} is absent "
                "from executable start-to-terminal paths; another writer or "
                "fallback of the same filenames does not satisfy this requirement"
            )
        missing = [path for path in wanted if path not in producer.writes]
        if missing:
            return f"required producer {by!r} no longer declares the artifacts {missing}"
        return None

    return Invariant("artifact_produced", f"{by!r} produces required artifacts {list(wanted)}", check)
