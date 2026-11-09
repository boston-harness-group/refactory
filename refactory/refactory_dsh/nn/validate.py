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
from typing import Any, Callable, Iterable, Mapping

from refactory_dsh.snippet import ALLOWED_IMPORTS_TEXT, MAX_CODE_BYTES

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
          "unconditional out-edge (one is 'proceed' when it declares on_exhausted "
          "'continue'); a code or verdict gate's conditions are single words "
          "(no spaces or punctuation), as its answer is read as one word",
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
           "nodes that exist; a data node's entry reaches its exit, which is not a gate and "
           "has no out-edge (its pass ends there); and a pass is entered only through its "
           "data node, never by an edge from the data node or from outside, and never "
           "leads back to it",
    "V13": "a data node must not appear inside a fork branch",
    "V14": "every field binding must name a declared knob of the field's kind, once per "
           "field, and the field must be in the bindable allowlist",
    "V15": "a node's required inputs must be produced upstream on every path that "
           "reaches it (a data pass's own files and facts stay inside the pass): an agent's "
           "reads and every {{name}} its prompts fill in (not a knob, not in its "
           "feedback_keys, not inside an {% if name %} on the same name), an op's or a "
           "gate's required_reads and inputs (facts), and every {{name}} in a knob an op "
           "that fills facts reads; a tool_policy 'none' agent declares exactly one write",
    "V16": "a fork has one unconditional out-edge, to its join, and each join belongs to "
           "one fork; branches are disjoint, never lead back to the fork or past the join, "
           "declare no shared facts or conflicting files, and a gate in a branch reloops "
           "only within it; a join is entered only from its fork's branches and waits only "
           "for nodes every run of their branch visits",
    "V17": "no node outside the grader side consumes grader-derived data: what a data "
           "pass publishes (its graded records), the runtime's grading files, what a "
           "Task declares its grader writes (Task.grader_outputs), everything a node the "
           "mode declares as grading writes or produces (Mode.grader_nodes), and whatever "
           "a node that consumes any of these writes or produces in turn -- whether "
           "through a prompt slot or an {% if %} condition (bound fields included), feedback_keys, reads, "
           "required_reads or inputs. The grader side is the data nodes, the mode's protected nodes "
           "and its grader nodes",
    "V18": "a code node (call refactory_dsh.snippet:run, or evaluator refactory_dsh.snippet:gate) "
           f"has code that parses, defines main(inp), is at most {MAX_CODE_BYTES} bytes, imports "
           f"only these modules: {ALLOWED_IMPORTS_TEXT}; and uses no open, exec, eval, compile, __import__, getattr/setattr/delattr, "
           "globals/locals/vars or dunder name or attribute; no other node carries code; a "
           "verdict gate (evaluator refactory_dsh.snippet:verdict) reads its verdict from its "
           "first required_reads file or, with none, its first inputs fact",
    "V19": "a no-tools agent or a code op that sets no fact has a reader for what it writes: "
           "a step whose every file is written again by a no-tools agent, on every path from "
           "it, before any node reads it or may read it (a Python op or gate, an agent with "
           "tools, a data pass) and before the run or its pass ends, does work nothing sees",
    "V20": "a {{name}} slot an edit leaves in a knob is filled in only where a runtime fills it: in an "
           "agent's prompt_template and system_prompt_template (so in the knobs bound to "
           "them, such as <id>__task and <id>__system_prompt) and in a knob read by an op "
           "that fills facts (marked nn.fills_facts); a slot in a knob any other node reads "
           "would reach it as written, braces and all, so it is refused (one the parent "
           "already left unfilled for the same node and knob is not judged)",
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
    grader_side: Iterable[str] = (),
    graders: Iterable[str] = (),
    baseline: tuple[Graph, Mapping[str, Any]] | None = None,
) -> list[Issue]:
    """Every reason ``(graph, values)`` is not a runnable, scoreable mode.

    ``baseline`` is the graph and knob values an edit started from (its parent's):
    given, a ``{{name}}`` slot the edit leaves where nothing fills it, and the parent did
    not, is refused (V20). Without it -- a seed, a person's ``--knob`` -- V20 is not
    judged.

    ``grader_side`` names the nodes besides the data nodes that may consume
    grader-derived data (the mode's protected and grader nodes); ``graders`` the nodes
    the mode declares as grading, whose every output is grader-derived (V17).
    """
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
            if node.on_exhausted == "continue" and graph.exhaustion_exit(node_id) is None:
                issues.append(
                    Issue(
                        "V5",
                        "on_exhausted 'continue' takes the gate's 'proceed' out-edge, and "
                        "this gate has none that leads on (rather than back)",
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
    issues.extend(answer_label_issues(graph))

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
        "code": "str",
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
    issues.extend(agent_input_issues(graph, knob_values, resolve_calls=resolve_calls))
    issues.extend(required_read_issues(graph))
    # V20 a {{name}} slot an edit adds to a knob is somewhere a runtime fills it.
    if baseline is not None:
        issues.extend(unfilled_slot_issues(graph, knob_values, baseline=baseline,
                                           resolve_calls=resolve_calls))
    issues.extend(overwritten_write_issues(graph))

    # V16 forks and joins are paired, and branches are independent.
    issues.extend(fork_issues(graph, knob_values))

    # V17 nothing outside the grader side consumes grader-derived data.
    issues.extend(grader_issues(graph, knob_values, grader_side=grader_side, graders=graders,
                                resolve_tasks=resolve_calls))

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

    # V18 code nodes: their code passes the static check; nothing else carries code.
    issues.extend(code_issues(graph, knob_values))

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
        names |= required_slots(str(text or ""))
    return names - set(knob_values) - set(node.feedback_keys)


#: The fields both runtimes render as templates, filling each ``{{name}}`` slot.
TEMPLATE_FIELDS = ("prompt_template", "system_prompt_template")


def fills_facts(node: Node, *, resolve_calls: bool = True) -> bool | None:
    """Whether an op or a gate's callable fills the ``{{name}}`` slots in the knobs it
    reads (``nn.fills_facts``); ``None`` when it is not known: its callable does not
    resolve here, or ``resolve_calls`` is off. Read from the callable each time (no
    cache), so a reloaded package's mark is the one seen."""
    target = node.call if node.kind == "op" else (node.evaluator if node.kind == "gate" else "")
    if not target:
        return False
    if not resolve_calls:
        return None
    try:
        return bool(getattr(resolve(target), "fills_facts", False))
    except Exception:  # an unresolvable call is V8's to report
        return None


def filled_knobs(node: Node, *, resolve_calls: bool = True) -> tuple[str, ...]:
    """The knobs whose slots an op or gate fills: every knob it reads that is not bound
    to its code, when its callable fills facts; none otherwise (or when not known)."""
    if node.kind not in ("op", "gate") or not fills_facts(node, resolve_calls=resolve_calls):
        return ()
    code = {knob for field_name, knob in node.field_bindings if field_name == "code"}
    return tuple(name for name in node.knobs if name not in code)


def knob_slot_facts(node: Node, knob_values: Mapping[str, Any], *,
                    guarded: bool = False, resolve_calls: bool = True) -> set[str]:
    """The run facts the knobs an op fills (`filled_knobs`) take in: their ``{{name}}``
    slots, less the knobs; with ``guarded`` every name they read, slots inside an
    ``{% if %}`` on the same name and condition names included."""
    names: set[str] = set()
    for name in filled_knobs(node, resolve_calls=resolve_calls):
        value = knob_values.get(name)
        if isinstance(value, str):
            names |= template_names(value) if guarded else required_slots(value)
    return names - set(knob_values)


def unfilled_slots(graph: Graph, knob_values: Mapping[str, Any], *,
                   resolve_calls: bool = True) -> dict[tuple[str, str], set[str]]:
    """Every ``{{name}}`` slot nothing fills, by ``(node, knob)``: the slots of each knob
    a reachable node reads as written.

    An agent's template fields are rendered (so the knobs bound to them are), and an op
    or gate marked ``nn.fills_facts`` renders the knobs it reads. Any other reader -- an
    op that pastes its knob as it is, a gate, an agent reading an unbound knob -- sends
    the braces as written. A knob bound to a code node's code is Python, not a template.
    With ``resolve_calls`` off, an op whose callable is not resolved is not judged."""
    found: dict[tuple[str, str], set[str]] = {}
    for node_id in sorted(graph.reachable()):
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        if node.kind in ("op", "gate") and fills_facts(node, resolve_calls=resolve_calls) is None:
            continue
        bound = {knob: field_name for field_name, knob in node.field_bindings}
        filled = set(filled_knobs(node, resolve_calls=resolve_calls))
        for name in node.knobs:
            value = knob_values.get(name)
            if not isinstance(value, str) or name in filled:
                continue
            field_name = bound.get(name)
            if field_name == "code" or (node.kind == "agent" and field_name in TEMPLATE_FIELDS):
                continue
            slots = set(_SLOT.findall(value))
            if slots:
                found[(node_id, name)] = slots
    return found


def unfilled_slot_issues(graph: Graph, knob_values: Mapping[str, Any], *,
                         baseline: tuple[Graph, Mapping[str, Any]] | None = None,
                         resolve_calls: bool = True) -> list[Issue]:
    """V20: a ``{{name}}`` slot an edit leaves where nothing fills it.

    Judged by who reads the knob, not by the text alone: each (node, knob, slot) left
    unfilled (`unfilled_slots`) is refused unless ``baseline`` -- the graph and knob
    values the edit started from -- already had that same one unfilled. So a slot the
    search adds to a pasted knob is refused, and so is an existing slot it strands by
    removing or rewiring the step that filled it (replacing a fact-filling op with one
    that pastes the knob, unbinding an agent's prompt, a new op reading the knob). A
    slot the parent already left unfilled is the author's or the user's text -- a design
    brief asking for a ``{{name}}`` renderer -- and stays. With no ``baseline`` every
    unfilled slot is reported (a seed check)."""
    before = (unfilled_slots(baseline[0], baseline[1], resolve_calls=resolve_calls)
              if baseline is not None else {})
    issues: list[Issue] = []
    for (node_id, name), slots in sorted(unfilled_slots(
            graph, knob_values, resolve_calls=resolve_calls).items()):
        new = sorted(slots - before.get((node_id, name), set()))
        if not new:
            continue
        node = graph.nodes[node_id]
        shown = ", ".join(f"{{{{{slot}}}}}" for slot in new[:4])
        issues.append(Issue(
            "V20", f"knob {name!r} holds {shown}, and this {node.kind} reads the knob as "
            "written, so nothing fills the slot: the step would get the braces, not the "
            "value. A {{fact}} is filled in an agent's prompt_template and "
            "system_prompt_template (the <id>__task and <id>__system_prompt knobs) and in "
            "a knob an op that fills facts reads; move it to one of those, or drop it",
            node=node_id, knob=name))
    return issues


#: A ``{% if %}``/``{% elif %}``/``{% else %}``/``{% endif %}`` tag, as the renderer
#: reads it (``nn.executor._render_conditionals``).
_CONDITIONAL = re.compile(r"\{%[-\s]*(if|elif|else|endif)\b\s*(.*?)[-\s]*%\}", re.DOTALL)
#: A condition that is true only when its name has a value: ``name`` or
#: ``name == 'non-empty literal'``.
_GUARD = re.compile(r"""([A-Za-z_]\w*)(?:\s*==\s*(?:'[^']+'|"[^"]+"))?""")
#: Every condition the renderer evaluates (``nn.executor._eval_condition``): ``name``,
#: ``name == 'literal'`` and ``name != 'literal'``; the name is what it reads.
_CONDITION_NAME = re.compile(r"""\s*([A-Za-z_]\w*)\s*(?:(?:==|!=)\s*(?:'[^']*'|"[^"]*")\s*)?""")


def template_names(template: str) -> set[str]:
    """Every run value a template reads when rendered: its ``{{name}}`` slots and the
    name each ``{% if %}``/``{% elif %}`` condition tests (bare, ``==`` or ``!=``). A
    condition reads its value as surely as a slot does: ``{% if x == 'wrong' %}`` tells
    the agent something about ``x`` without printing it."""
    names = set(_SLOT.findall(template))
    for match in _CONDITIONAL.finditer(template):
        if match.group(1) in ("if", "elif"):
            found = _CONDITION_NAME.fullmatch(match.group(2))
            if found:
                names.add(found.group(1))
    return names


def required_slots(template: str) -> set[str]:
    """The ``{{name}}`` slots a template needs filled: all of them, except a slot inside
    an ``{% if name %}`` (or ``{% elif name %}``, or ``name == '...'``) branch on the same
    name, which renders only when the name has a value."""
    names: set[str] = set()
    guards: list[set[str]] = []  # per open conditional: the names its current branch guards
    pos = 0

    def take(text: str) -> None:
        guarded = set().union(*guards) if guards else set()
        names.update(name for name in _SLOT.findall(text) if name not in guarded)

    for match in _CONDITIONAL.finditer(template):
        take(template[pos:match.start()])
        pos = match.end()
        tag, expr = match.group(1), match.group(2).strip()
        guard = _GUARD.fullmatch(expr)
        if tag == "if":
            guards.append({guard.group(1)} if guard else set())
        elif tag == "elif" and guards:
            guards[-1] = {guard.group(1)} if guard else set()
        elif tag == "else" and guards:
            guards[-1] = set()
        elif tag == "endif" and guards:
            guards.pop()
    take(template[pos:])
    return names


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


def available_inputs(graph: Graph) -> dict[str, set[tuple[str, str]]]:
    """What is there when each node starts, on every path that reaches it.

    ``("file", name)`` and ``("fact", name)`` entries; the greatest fixpoint of: a node
    starts with what every predecessor left; a fork's branches with what the fork had;
    a fork's join with what the fork had and what every run of each branch left; a data
    pass with what its data node had and the data node's writes; the data node's
    successor with what the data node had and declares -- never what its pass made,
    which stays in the pass on both runtimes.
    """
    def made(node: Node) -> set[tuple[str, str]]:
        return {("file", n) for n in node.writes} | {("fact", n) for n in node.produces}

    universe: set[tuple[str, str]] = set()
    for node in graph.nodes.values():
        universe |= made(node)
        universe |= {("file", n) for n in node.reads}
        universe |= {("fact", n) for n in node.inputs}
        universe |= {("fact", n) for f in ("prompt_template", "system_prompt_template")
                     for n in template_names(str(getattr(node, f) or ""))}
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


def answer_label_issues(graph: Graph) -> list[Issue]:
    """V5 for code and verdict gates (`refactory_dsh.snippet`): each out-edge condition is
    a word their answer can be (`snippet.is_answer_word`), so the edit is refused here
    rather than every run failing when the gate answers."""
    from refactory_dsh import snippet

    issues: list[Issue] = []
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if node.kind != "gate" or node.evaluator not in (snippet.GATE, snippet.VERDICT):
            continue
        bad = sorted(str(e.condition) for e in graph.out_edges(node_id)
                     if e.condition is not None and not snippet.is_answer_word(e.condition))
        if bad:
            issues.append(Issue("V5", f"out-edge condition(s) {bad} cannot be this gate's answer: "
                                "a code or verdict gate answers one word, so each condition is "
                                "a single word with no spaces or punctuation", node=node_id))
    return issues


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
    """V6: every read is produced by a node on a path the run can take.

    Only that: an op or a gate may read a file that is not there yet (neither runtime
    checks an op's reads, and shipped modes read an optional file or the last round's
    feedback that way). An agent's reads are required, and V15 holds them to every
    path that reaches it.
    """
    issues: list[Issue] = []
    produced = produced_on_path(graph)
    for node_id in sorted(graph.nodes):
        for name in graph.nodes[node_id].reads:
            if name not in produced:
                issues.append(Issue(
                    "V6", f"reads {name!r}, which no node on a path from {graph.start!r} to a "
                    "terminal produces", node=node_id))
    return issues


def required_read_issues(graph: Graph) -> list[Issue]:
    """V15 for ops and gates: each of a node's ``required_reads`` is written upstream on
    every path that reaches it, as is each fact in its ``inputs``; both runtimes refuse
    to run the node without a required read, and a code node's ``main`` would be handed
    an input that is not there."""
    issues: list[Issue] = []
    available = available_inputs(graph)
    for node_id in sorted(graph.reachable()):
        node = graph.nodes.get(node_id)
        if node is None or node.kind not in ("op", "gate"):
            continue
        if not node.required_reads and not node.inputs:
            continue
        have = available.get(node_id, set())
        absent = sorted(name for name in node.required_reads if ("file", name) not in have)
        if absent:
            issues.append(Issue(
                "V15", f"declares {absent} as required_reads, which some path to it does not "
                "write first; both runtimes refuse to run it without them (a data pass's own "
                "files stay inside its pass)", node=node_id))
        unset = sorted(name for name in node.inputs if ("fact", name) not in have)
        if unset:
            issues.append(Issue(
                "V15", f"declares {unset} as inputs, which some path to it does not produce "
                "first (a data pass's own facts stay inside its pass)", node=node_id))
    return issues


def _replaces_its_writes(node: Node) -> bool:
    """Whether a node's whole effect is its declared outputs: a no-tools agent (its reply
    is its one write) or a code op (`main` returns the files it writes). Any other op or
    agent may append to a file, read one it does not declare, or act elsewhere."""
    from refactory_dsh import snippet

    if node.kind == "agent":
        return node.tool_policy == "none"
    return node.kind == "op" and snippet.is_code_node(node)


def _reads_only_declared(node: Node) -> bool:
    """Whether a node can read only the files it declares. A no-tools agent is handed its
    declared reads; a code op or code/verdict gate its declared inputs. Any other op or
    gate runs Python (or asks a person or an agent) that may open any file, since
    neither runtime checks an op's reads; an agent with tools may too; a data node's
    pass reads what is there."""
    from refactory_dsh import snippet

    if node.kind == "agent":
        return node.tool_policy == "none"
    if node.kind == "op":
        return snippet.is_code_node(node)
    if node.kind == "gate":
        return node.gate_kind == "fn" and node.evaluator in (snippet.GATE, snippet.VERDICT)
    return node.kind in ("fork", "join")


def _overwriters(graph: Graph, node_id: str, name: str) -> list[str] | None:
    """The steps that write ``name`` again on every path from ``node_id`` before any node
    may read it, or ``None`` when some path may read it first.

    A path may read it when it reaches a node declaring it in ``reads``, a node that may
    read files it does not declare (`_reads_only_declared` false: a Python op or gate, an
    agent with tools, a data node), or an end: a node with no out-edge -- the run's or
    its pass's last step -- leaves the file as the run's output. Only a no-tools agent
    writing it is a certain overwrite (its reply is the file); a code op writes only what
    its ``main`` returns, so the walk goes on past one. A fork's branches are walked as
    well as its join."""
    stack = [edge.target for edge in graph.out_edges(node_id)]
    if not stack:
        return None
    seen: set[str] = set()
    writers: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        node = graph.nodes.get(current)
        if node is None or name in node.reads or name in node.required_reads:
            return None
        if not _reads_only_declared(node):
            return None
        if name in node.writes and node.kind == "agent":
            writers.add(current)
            continue
        following = [edge.target for edge in graph.out_edges(current)]
        if node.kind == "fork":
            following += [*node.targets, *node.optional_targets]
        if not following:
            return None
        stack.extend(following)
    return sorted(writers)


def overwritten_write_issues(graph: Graph) -> list[Issue]:
    """V19: a step whose every write is overwritten before anything reads it.

    Only a step whose whole effect is its declared outputs is judged (a no-tools agent or
    a code op, `_replaces_its_writes`) and only one that sets no fact; a loop that
    rewrites its file on each pass, a reader in between, or a path to the end keeps it."""
    issues: list[Issue] = []
    for node_id in sorted(graph.reachable()):
        node = graph.nodes.get(node_id)
        if node is None or not node.writes or node.produces or not _replaces_its_writes(node):
            continue
        lost: dict[str, list[str]] = {}
        for name in node.writes:
            writers = _overwriters(graph, node_id, name)
            if writers is None:
                break
            lost[name] = writers
        else:
            detail = "; ".join(f"{name!r} by {', '.join(repr(w) for w in writers)}"
                               for name, writers in sorted(lost.items()))
            issues.append(Issue(
                "V19", f"everything it writes is written again before any node reads it "
                f"({detail}), on every path from it, so its work is lost; have a later step "
                "read it (`reads`), make its reply a fact (`produces`), write another file, "
                "or remove one of the two", node=node_id))
    return issues


def code_issues(graph: Graph, knob_values: Mapping[str, Any]) -> list[Issue]:
    """V18: a code node's code passes the static check (`snippet.static_issues`, which
    never runs it), read as both runtimes run it -- its bound knob's value; a node that
    runs no code carries none; a verdict gate has a source to read its verdict from."""
    from refactory_dsh import snippet

    issues: list[Issue] = []
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        bound = node.bound_knob("code")
        if snippet.is_code_node(node):
            try:
                code = node.bound("code", knob_values)
            except Exception:  # a bad binding is V14's to report
                code = node.code
            for problem in snippet.static_issues(code):
                issues.append(Issue("V18", f"code node: {problem}", node=node_id,
                                    knob=bound or None))
        elif node.code or bound:
            issues.append(Issue(
                "V18", "carries code but runs no code: only call "
                f"{snippet.RUN!r} or evaluator {snippet.GATE!r} runs a node's code", node=node_id))
        if node.kind == "gate" and node.evaluator == snippet.VERDICT \
                and not node.required_reads and not node.inputs:
            issues.append(Issue(
                "V18", "a verdict gate reads its verdict from its first required_reads file or "
                "its first inputs fact, and declares neither", node=node_id))
    return issues


def agent_input_issues(graph: Graph, knob_values: Mapping[str, Any], *,
                       resolve_calls: bool = True) -> list[Issue]:
    """V15: required inputs are there on every path: an op's or a gate's
    ``required_reads``, and an agent's inputs -- its reads and its prompts' ``{{facts}}``.

    Both runtimes refuse to dispatch an agent whose declared read is missing, and send a
    prompt with an unfilled ``{{name}}`` as it is, so each must be produced upstream on
    every path to the agent; a data pass's own files and facts stay inside the pass. A
    ``tool_policy`` 'none' agent's reply is its one output, so it declares one write.
    """
    issues: list[Issue] = []
    available = available_inputs(graph)
    for node_id in sorted(graph.reachable()):
        node = graph.nodes.get(node_id)
        if node is not None and node.kind in ("op", "gate"):
            have = available.get(node_id, set())
            missing = sorted(f"{{{{{name}}}}}" for name in knob_slot_facts(
                node, knob_values, resolve_calls=resolve_calls)
                             if ("fact", name) not in have)
            if missing:
                issues.append(Issue(
                    "V15", f"fills {', '.join(missing)} into the knobs it reads, which some "
                    "path to it does not produce first; an optional one goes inside "
                    "{% if name %}...{% endif %}", node=node_id))
            continue
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
                "stay inside its pass). A fact the runtime re-derives before each dispatch "
                "belongs in the node's feedback_keys, and an optional one goes inside "
                "{% if name %}...{% endif %}", node=node_id))
    return issues


# ── V17: grader-derived data stays on the grader side ─────────────────────


def _made(node: Node) -> set[tuple[str, str]]:
    return {("file", n) for n in node.writes} | {("fact", n) for n in node.produces}


def consumed(node: Node, knob_values: Mapping[str, Any], *,
             resolve_calls: bool = True) -> set[tuple[str, str]]:
    """Every fact and file a node takes in, as the graph declares it: each ``{{name}}``
    slot and each name a ``{% if %}``/``{% elif %}`` condition tests, in its prompts and
    in every field bound to a knob (rendered with the knob values, guarded slots
    included; a code node's code is Python, not a template), its ``feedback_keys``, its
    ``inputs``, its reads and its required_reads."""
    # A code node's code is Python, not a template: it takes in its ``inputs``.
    fields = ({"prompt_template", "system_prompt_template"}
              | {f for f, _k in node.field_bindings}) - {"code"}
    facts: set[str] = set(node.feedback_keys) | set(node.inputs)
    for field_name in sorted(fields):
        try:
            text = node.bound(field_name, knob_values)
        except Exception:  # a bad binding is V14's to report
            text = getattr(node, field_name, "")
        facts |= template_names(str(text or ""))
    # An op that fills facts takes in the slots of the knobs it reads.
    facts |= knob_slot_facts(node, knob_values, guarded=True, resolve_calls=resolve_calls)
    facts -= set(knob_values)
    files = set(node.reads) | set(getattr(node, "required_reads", ()) or ())
    return {("fact", n) for n in facts} | {("file", n) for n in files}


def _task_grader_outputs(ref: str) -> tuple[str, ...]:
    try:
        from refactory_dsh.task import grader_outputs
        return tuple(grader_outputs(ref))
    except Exception:  # an unloadable Task is another rule's to report
        return ()


def grader_derived(graph: Graph, knob_values: Mapping[str, Any], *,
                   graders: Iterable[str] = (), resolve_tasks: bool = True
                   ) -> set[tuple[str, str]]:
    """The provenance set: every ``("fact"|"file", name)`` that carries the grade.

    Computed from the graph and the declarations, never from a list of names:

    - what each data node publishes after its pass (its ``produces``: the pass's graded
      records), never the per-item input its pass reads;
    - the files the runtime's grading writes (``items.GRADER_FILES``);
    - what each data node's Task declares its grader writes (``Task.grader_outputs``),
      as a file and as a fact;
    - everything a node the mode declares as grading (``Mode.grader_nodes``) writes or
      produces;
    - and, to a fixpoint, everything any node that consumes one of these writes or
      produces: a grade copied is still the grade.
    """
    from refactory_dsh.items import GRADER_FILES

    derived: set[tuple[str, str]] = {("file", name) for name in GRADER_FILES}
    for node in graph.nodes.values():
        if node.kind == "data":
            derived |= {("fact", name) for name in node.produces}
            if resolve_tasks and node.task_ref:
                for name in _task_grader_outputs(node.task_ref):
                    derived |= {("file", name), ("fact", name)}
    for node_id in graders:
        node = graph.nodes.get(node_id)
        if node is not None:
            derived |= _made(node)
    changed = True
    while changed:
        changed = False
        for node in graph.nodes.values():
            if node.kind == "data" or not (
                    consumed(node, knob_values, resolve_calls=resolve_tasks) & derived):
                continue
            if not _made(node) <= derived:
                derived |= _made(node)
                changed = True
    return derived


def grader_issues(graph: Graph, knob_values: Mapping[str, Any], *,
                  grader_side: Iterable[str] = (), graders: Iterable[str] = (),
                  resolve_tasks: bool = True) -> list[Issue]:
    """V17: a node outside the grader side that consumes grader-derived data.

    The grader side is the data nodes, ``grader_side`` (the mode's protected nodes) and
    ``graders`` (the mode's grader nodes). Any other node -- an agent, an op, a gate --
    that takes in a grader-derived fact or file could carry the answers into a later
    pass, a prompt or the run's output, whatever the Task or the domain.
    """
    graders = tuple(graders)
    side = set(grader_side) | set(graders) | {n.id for n in graph.nodes.values()
                                              if n.kind == "data"}
    derived = grader_derived(graph, knob_values, graders=graders, resolve_tasks=resolve_tasks)
    issues: list[Issue] = []
    for node_id in sorted(graph.nodes):
        if node_id in side:
            continue
        taken = consumed(graph.nodes[node_id], knob_values, resolve_calls=resolve_tasks) & derived
        if taken:
            names = ", ".join(f"{{{{{n}}}}}" if kind == "fact" else f"file {n!r}"
                              for kind, n in sorted(taken))
            issues.append(Issue(
                "V17", f"consumes grader-derived {names} (the grade, or something made from "
                "it); only the data nodes and the nodes the mode declares protected or grading "
                "may read what grading produced", node=node_id))
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
    grader_side: Iterable[str] = (),
    graders: Iterable[str] = (),
    baseline: Mapping[str, Any] | None = None,
) -> list[Issue]:
    """Validate a state dict without building a mode around it. ``baseline`` is the
    state an edit started from (`validate`'s V20); an unreadable one counts as a parent
    that left nothing unfilled.

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
        grader_side=grader_side,
        graders=graders,
        baseline=_baseline(baseline),
    )


def _baseline(state: Mapping[str, Any] | None) -> tuple[Graph, Mapping[str, Any]] | None:
    """A parent state as `validate`'s ``baseline``: its graph and knob values."""
    if state is None:
        return None
    try:
        graph = Graph.from_state(state["graph"])
    except Exception:  # an unreadable parent left nothing unfilled
        graph = Graph()
    return graph, {name: entry.get("value") for name, entry in
                   (state.get("knobs") or {}).items()}


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
