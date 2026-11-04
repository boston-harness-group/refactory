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

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .graph import Graph
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
    "V5": "a gate has at least two out-edges, each with a condition, and no "
          "unconditional out-edge",
    "V6": "every file a node reads must be written by a node on a path from the "
          "start node to a terminal",
    "V7": "every knob a node consumes must be declared, and every declared knob "
          "must be consumed by some node",
    "V8": "every call and evaluator must resolve to a real callable",
    "V9": "every knob value must be of the declared kind and within its bounds; a required knob must be non-empty",
    "V10": "every cycle must pass through a gate; after explicit reloop edges are "
           "removed, each remaining cyclic region must have exactly one gate",
    "V11": "every declared port must be a file some node on a path writes",
    "V12": "a fork's targets, a join's sources and a data node's subgraph must name "
           "nodes that exist",
    "V13": "a data node must not appear inside a fork branch",
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

    # V13 data-inside-fork.
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if node.kind != "fork":
            continue
        fork_edges = graph.out_edges(node_id)
        barrier = fork_edges[0].target if fork_edges else None
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

    # V6 every read is produced by a node on a path the run can take.
    produced = produced_on_path(graph)
    for node_id in sorted(graph.nodes):
        for name in graph.nodes[node_id].reads:
            if name not in produced:
                issues.append(
                    Issue(
                        "V6",
                        f"reads {name!r}, which no node on a path from "
                        f"{graph.start!r} to a terminal produces",
                        node=node_id,
                    )
                )

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
