"""Lower a mode state to refactory's IR, so DSH can run it.

The IR is the seam. A mode's state dict stays the optimizer's write surface; the
IR is what runs. Building real `refactory_dsh.ir` objects rather than emitting JSON
by hand means pydantic validates the result, so an IR that lowers is an IR that
loads.

Node mapping:

======================  ==================================================
`op`                    `FnNode` running ``python -m refactory_dsh.run <call>``
`agent`                 `AgentNode`, which DSH runs as a subagent
`gate`                  `GateNode`. ``fn`` runs its evaluator and routes by the
                        last stdout line, reporting the lines before it as the
                        reason; ``user`` and ``agent`` carry no command and are
                        answered by the person or by the driving agent
======================  ==================================================

The runner is told its own node through the spine's `DSH_WORKFLOW_NODE`, and the
knobs through the command environment, so a node command knows nothing about the
graph that invoked it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from refactory_dsh.ir import (
    AgentNode,
    AgentRole,
    DataNode,
    Edge,
    FnNode,
    ForkNode,
    GateNode,
    JoinNode,
    Workflow,
)
from refactory_dsh.nn import Graph, Knob
from refactory_dsh.nn.graph import op_io
from refactory_dsh.tool_policy import no_effect

#: How a node names the operation it runs. Resolved by the runtime, so no package
#: is named here and no absolute path is baked in: `--python-bin` supplies the PATH.
RUNNER = "python -m refactory_dsh.run"

#: The environment variable the knobs travel in.
KNOBS_ENV = "REFACTORY_KNOBS"

#: The graph file the science bundle loads, one per mode.
SUFFIX = ".graph.json"

#: Where the run's node declarations are left for the runner.
#:
#: `DSH_WORKFLOW_NODE` carries `{id, type, command, reads, writes}` -- the IR has no
#: representation for the facts a node produces, because the IR's nodes do not have
#: facts. So the declarations the IR cannot carry are written into the run directory,
#: and the runner reads them there. Without this the runner could not enforce the
#: produces check, and a node that lies about its facts would go unnoticed on the
#: DSH path while being caught on the walker.
DECLARATIONS = ".refactory/declarations.json"

#: The key the mode's declared memory travels under, beside the node declarations.
MEMORY_KEY = "memory"


def _optional_node_ids(graph: Graph, knob_values: Mapping[str, Any]) -> set[str]:
    """Node ids some fork lets fail without failing the item (`Graph.optional_targets_of`)."""
    ids: set[str] = set()
    for node in graph.nodes.values():
        if node.kind == "fork":
            ids.update(graph.optional_targets_of(node, knob_values))
    return ids


def node_declarations(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every node's kind and declared reads, writes and facts, keyed by node id.

    The kind travels because a reader of a finished run has the run directory and
    not the state: the citation chain has to know which nodes made a model call, or
    it would demand a citation from a deterministic op.

    A data node's *source* travels too. The IR carries it, but the runtime's node
    manifest does not, and the item loader is a command of its own — it has the run
    directory and must still know which items the node selected. Writing the source
    beside the declarations is what lets one loader serve both runtimes instead of
    the DSH path re-deriving it from an environment variable it cannot see.
    """
    graph = Graph.from_state(state.get("graph") or {})
    knob_values = {name: (entry or {}).get("value") for name, entry in (state.get("knobs") or {}).items()}
    optional_ids = _optional_node_ids(graph, knob_values)
    declarations: dict[str, dict[str, Any]] = {}
    # The reloops whose redone step cannot see what the gate read: the gate's runner
    # hands it those files (`run.with_unseen_reads`), as the walker does.
    from refactory_dsh.nn.loops import blind_reloops

    blind = blind_reloops(graph, knob_values)
    from refactory_dsh.nn.graph import GraphError
    from refactory_dsh.nn.loops import loop_edges

    try:
        reloops = loop_edges(graph)
    except GraphError:
        reloops = set()  # the validator refuses the graph; nothing to mark
    for node_id, node in graph.nodes.items():
        writes, reads, produces = op_io(node)
        entry: dict[str, Any] = {
            "kind": node.kind,
            "reads": list(reads),
            "writes": list(writes),
            "produces": list(produces),
            "resource_role": node.resource_phase,
        }
        if node_id in optional_ids:
            entry["optional"] = True
        if node.kind == "agent" and node.writes:
            # The output an agent's answer is captured into when it wrote none, and the
            # fact it produces is read from: its first declared write, as the walker takes
            # it. The IR keeps writes as a set, so their order cannot carry this.
            entry["answer_file"] = node.writes[0]
        if node.kind == "data":
            entry["data"] = {
                "subgraph_entry": node.subgraph_entry,
                "subgraph_exit": node.subgraph_exit,
                "task_ref": node.task_ref,
                "source_path": node.source_path,
                "source_format": node.source_format,
                "inline_items": [dict(item) for item in node.inline_items],
                "limit": node.limit,
            }
            # The nodes of the per-item pass, so the loader can tell a pass's own
            # writes from the run-level files it reads through (the walker takes the
            # same set from the graph: Graph.data_body).
            entry["subgraph_nodes"] = sorted(graph.data_body(node_id))
        if node.required_reads:
            # The node runner refuses to run the node while one is missing (`run.run_node`),
            # as the walker does.
            entry["required_reads"] = list(node.required_reads)
        if node.feedback_keys:
            entry["feedback_keys"] = list(node.feedback_keys)
        rule = graph.retry_rule(node_id)
        if rule is not None:
            # The rule the node runs under: declared, or a data pass's default.
            entry["retry"] = {**rule, "on": list(rule["on"])}
        elif node.retry is not None:
            entry["retry"] = {**node.retry, "on": list(node.retry["on"])}
        if graph.missing_on_crash(node_id):
            entry["missing_on_crash"] = True
        if graph.falls_back(node_id):
            # A genuine failure leaves it missing for its fallback gate (`run.run_node`,
            # `worker_node.main`), as the walker records it.
            entry["falls_back"] = True
        if node.workspace is not None:
            # Where the node works, for a reader of the run directory; the worker spec
            # carries the same declaration to the node's dispatch.
            entry["workspace"] = dict(node.workspace)
        if node.inputs:
            entry["inputs"] = list(node.inputs)
        if node.knob_outputs is not None:
            entry["knob_outputs"] = {out: list(k) for out, k in node.knob_outputs.items()}
        if (node.renames or {}).get("reads"):
            # The name each built file is read under (`nn.declared_files`), as the walker
            # hands it to its child.
            entry["read_renames"] = dict(node.renames["reads"])
        if node_id in blind:
            entry["blind_reloops"] = blind[node_id]
        verdicts = sorted({condition for gate, _target, condition in reloops if gate == node_id})
        if node.kind == "gate" and verdicts:
            # Its reloop verdicts and cap: the runner marks a reloop the cap refuses on the
            # item's record (`items.refused_at_cap`), as the walker does.
            entry["reloop"] = {"verdicts": verdicts,
                               "max_iterations": node.bound("max_iterations", knob_values)}
        if needs_snippet_spec(node):
            # What a code or verdict node's callable needs beyond the context (its code,
            # inputs, labels): the runner hands it over as the walker's child does.
            entry["snippet"] = snippet_spec(graph, node, knob_values)
        judged = judge_verdict(graph, node_id)
        if judged is not None:
            # The verdict gate that reads this agent's answer: its worker asks again when
            # the answer names no verdict (`node_retry.settle_verdict`), as the walker does.
            entry["verdict"] = judged
        declarations[node_id] = entry
    return declarations


def judge_verdict(graph: Graph, node_id: str) -> dict[str, Any] | None:
    """The verdict gate that reads an agent's answer, and what settling the answer needs.

    For an agent whose answer a verdict gate (`snippet.verdict`) reads its verdict from
    -- the gate's first ``required_reads`` is the agent's first write -- what settling
    that answer needs
    (`node_retry.settle_verdict`): ``{"gate", "labels", "unreadable"}``, the labels being
    the gate's out-edge conditions and ``unreadable`` the label taken when the answer
    names none even when asked again: the gate's ``proceed`` when it goes on there when
    its loop cannot settle (`Graph.on_exhausted` ``continue``, a loop's in a data pass
    unless it declares otherwise), else empty, and the gate fails as it always has
    (``fail_item``, ``halt``). ``None`` for any other node, and for an agent two verdict
    gates read: which gate's answers to ask for is not one question.

    Both runtimes read it here: the walker from the graph, the spine's worker from the
    declarations."""
    from refactory_dsh import snippet
    from refactory_dsh.nn.graph import PROCEED

    node = graph.nodes.get(node_id)
    if node is None or node.kind != "agent":
        return None
    if not node.writes:
        return None
    found = []
    for gate in graph.nodes.values():
        if gate.kind != "gate" or gate.evaluator != snippet.VERDICT:
            continue
        # The file the agent answers in, which `settle_verdict` settles: a gate fed the
        # agent's fact instead is not matched (its verdict is read as it always was).
        if not gate.required_reads or gate.required_reads[0] != node.writes[0]:
            continue
        labels = sorted({str(edge.condition) for edge in graph.out_edges(gate.id)
                         if edge.condition})
        onward = next((label for label in labels if label.lower() == PROCEED), "")
        found.append({"gate": gate.id, "labels": labels,
                      "unreadable": onward if graph.on_exhausted(gate.id) == "continue" else ""})
    return found[0] if len(found) == 1 else None


def needs_snippet_spec(node: Any) -> bool:
    """Whether ``node`` runs a `refactory_dsh.snippet` callable (a code op, a code gate or
    a verdict gate), or a map's elements step, which read their declaration from the
    runtime."""
    from refactory_dsh import snippet

    from refactory_dsh.nn.graph import FALLBACK_GATE_CALL
    from refactory_dsh.nn.operators import MAP_CALL

    target = node.call if node.kind == "op" else (node.evaluator if node.kind == "gate" else "")
    return target in (snippet.RUN, snippet.GATE, snippet.VERDICT, FALLBACK_GATE_CALL, MAP_CALL)


def snippet_spec(graph: Graph, node: Any, knob_values: dict[str, Any]) -> dict[str, Any]:
    """A code or verdict node's declaration (`snippet.spec`), the same on both runtimes:
    the walker builds it for its child, the lowering writes it for the spine's runner."""
    from refactory_dsh import snippet
    from refactory_dsh.nn.graph import FALLBACK_GATE_CALL
    from refactory_dsh.nn.operators import MAP_CALL, map_of

    found = snippet.spec(node, [edge.condition for edge in graph.out_edges(node.id)], knob_values)
    if node.kind == "gate" and node.evaluator == FALLBACK_GATE_CALL:
        # The step whose failure the fallback gate reads (`nn.operators.step_failed`).
        found["step"] = next((edge.source for edge in graph.in_edges(node.id)), "")
    if node.kind == "op" and node.call == MAP_CALL:
        # The text knob a map's elements step fills, and the map's runs with what each
        # writes, which it clears before a pass (`nn.operators.elements`).
        found["task_knob"] = node.knobs[0] if node.knobs else ""
        fork = next((e.target for e in graph.out_edges(node.id)
                     if map_of(graph, e.target) == node.id), None)
        found["runs"] = [{"id": run, "writes": list(graph.nodes[run].writes)}
                         for run in (graph.nodes[fork].targets if fork else ())]
    return found


def write_declarations(state: dict[str, Any], workdir: Path) -> Path:
    """Leave the declarations where the runner will look for them.

    Two things a node needs that the IR cannot carry: the facts it produces, and
    the memory its mode declares. Both travel beside the graph, keyed by node id
    for the first and under ``memory`` for the second.

    Memory has to travel, not be re-derived: the runner builds the store an op
    receives, and a store built from nothing is how a mode that keeps a cross-run
    prior works in the walker and silently does nothing on DSH.
    """
    path = Path(workdir) / DECLARATIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = node_declarations(state)
    declared = state.get("memory") or []
    if declared:
        payload[MEMORY_KEY] = declared
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def read_declarations(workdir: Path) -> dict[str, Any]:
    """The declarations the lowering left in the run directory, keyed by node id.

    A node entry carries its kind, its reads, writes and facts, and -- for a data
    node -- its source. The value type is loose because the entry is a document the
    lowering writes and several readers read parts of.
    """
    path = Path(workdir) / DECLARATIONS
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def read_memory_declarations(workdir: Path) -> list[Any]:
    """The mode's memory declarations, as the runner reads them back."""
    from refactory_dsh.nn import Memory

    loaded = read_declarations(workdir).get(MEMORY_KEY) or []
    out: list[Any] = []
    for entry in loaded:
        if isinstance(entry, dict):
            out.append(Memory.from_state(entry))
    return out


class LoweringError(ValueError):
    """A state cannot be expressed in the IR, and saying why is more useful than a traceback."""


def to_workflow(state: dict[str, Any], *, name: str = "", description: str = "") -> Workflow:
    """The refactory `Workflow` for a mode state."""
    graph = Graph.from_state(state.get("graph") or {})
    # The run's knob values, as the state carries them: a bound field reads its knob
    # from here (`Node.bound`), as the walker reads it from the run's knobs.
    knob_values = {name: entry.get("value") for name, entry in (state.get("knobs") or {}).items()}
    nodes: dict[str, Any] = {}
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        common = {
            "id": node.id,
            "reads": set(node.reads),
            "writes": set(node.writes),
            "blocking": node.blocking,
            "resource_role": node.resource_phase,
        }
        if node.kind == "op":
            nodes[node_id] = FnNode(
                command_timeout_s=command_timeout_s(node, state),
                **common,
                # Both fields, and both for a reason. `callable_name` is
                # the language's own way of naming what an `FnNode` runs
                # (`refactory_dsh/ir.py`), and it is what makes the IR
                # legible to refactory's tooling. DSH's spine does not model it —
                # its `WorkflowNode` has `command` and nothing else — so a graph
                # carrying only `callable_name` would be a node that never runs.
                # Emitting both keeps the IR faithful to the language it is written
                # in and still executable by the runtime we ship.
                callable_name=node.call,
                command=f"{RUNNER} {node.call}",
                notes=node.notes,
                **({"knob_outputs": {out: list(k) for out, k in node.knob_outputs.items()},
                    "knobs": list(node.knobs), "produces": list(node.produces)}
                   if node.knob_outputs is not None else {}),
                # The spine lets it fail and go on when its runner records it missing.
                **({"falls_back": True} if graph.falls_back(node_id) else {}),
            )
        elif node.kind == "gate":
            # Who answers it decides whether there is a command at all. `fn` runs its
            # evaluator; `user` and `agent` are handed to the driving agent by the
            # spine, which presents a user gate to the person. Emitting a command for
            # those would give the gate two answerers, and which one won would depend
            # on the runtime.
            gate: dict[str, Any] = {
                "evaluator_type": node.gate_kind,
                # A bound knob's value, as the walker reads it (`Node.bound`).
                "max_iterations": node.bound("max_iterations", knob_values),
            }
            # What a reloop past the cap does, resolved from the graph's shape as the
            # walker resolves it (`Graph.on_exhausted`). `halt` is the spine's own
            # default and is left out, so a graph whose gates all halt lowers as before.
            _exhausted = graph.on_exhausted(node.id)
            if _exhausted != "halt":
                gate["on_exhausted"] = _exhausted
            # The author's note is the gate's question: for a `user` gate it is what
            # the person is shown, and an empty prompt would ask them to choose
            # between labels with no statement of what they are choosing about.
            if node.notes:
                gate["gate_prompt"] = node.notes
            if node.gate_kind == "fn":
                gate["evaluator_command"] = f"{RUNNER} {node.evaluator}"
                _gate_timeout = command_timeout_s(node, state)
                if _gate_timeout is not None:
                    gate["command_timeout_s"] = _gate_timeout
            nodes[node_id] = GateNode(**common, **gate)
        elif node.kind == "fork":
            # The branches are reached through `targets`, not edges: one out-edge
            # per branch would let the successor run as soon as the first one
            # finished, which is the bug the barrier exists to prevent.
            fork_kwargs: dict[str, Any] = {"targets": list(node.targets)}
            optional = graph.optional_targets_of(node, knob_values)
            if optional:
                fork_kwargs["optional_targets"] = list(optional)
            nodes[node_id] = ForkNode(**common, **fork_kwargs)
        elif node.kind == "join":
            nodes[node_id] = JoinNode(**common, sources=list(node.sources))
        elif node.kind == "data":
            nodes[node_id] = DataNode(
                **common,
                subgraph_entry=node.subgraph_entry,
                subgraph_exit=node.subgraph_exit,
                task_ref=node.task_ref or None,
                source_path=node.source_path or None,
                source_format=node.source_format or None,
                inline_items=[dict(item) for item in node.inline_items],
                limit=node.limit or None,
                # A bound knob's value, as the state carries it (Node.effective_parallelism).
                parallelism=node.effective_parallelism(knob_values),
            )
        else:
            try:
                role = AgentRole(node.role)
            except ValueError as exc:
                known = ", ".join(member.value for member in AgentRole)
                raise LoweringError(
                    f"agent node {node.id!r} declares role {node.role!r}, which the IR "
                    f"does not have; known roles: {known}"
                ) from exc
            if no_effect(node.tool_policy) and len(node.writes) != 1:
                raise LoweringError(
                    f"agent node {node.id!r} declares tool_policy={node.tool_policy!r} but has "
                    f"{len(node.writes)} write(s); a no-tools node must declare exactly "
                    "one output for its captured reply"
                )
            # Bound fields carry their knob's value, as the walker reads them (`Node.bound`).
            _resolved_spt = node.bound("system_prompt_template", knob_values)
            _resolved_pt = node.bound("prompt_template", knob_values)
            _resolved_mi = node.bound("max_iterations", knob_values)
            _resolved_route = node.bound_route(knob_values)
            nodes[node_id] = AgentNode(
                **common,
                role=role,
                prompt_template=_resolved_pt,
                **({"system_prompt": _resolved_spt} if _resolved_spt else {}),
                tools=list(node.tools),
                max_iterations=_resolved_mi,
                # The node's model-session wall: `workers.lower_workflow` gives the
                # dispatched session this bound and the command it plus the dispatch
                # overhead; the walker gives its model client the same bound.
                **({"timeout": node.timeout_s} if node.timeout_s is not None else {}),
                # The node's recovery, declared or a data pass's default
                # (`Graph.retry_rule`): `workers.lower_workflow` reserves the spine's
                # envelope for every attempt and hands the rule to the worker, which
                # runs it (`node_retry.deliver`), as the walker does.
                **({"retry": {**rule, "on": list(rule["on"])}}
                   if (rule := graph.retry_rule(node_id)) is not None else {}),
                # Spent retries leave the step missing, not failed: the worker records
                # it so and the spine lets it fail and go on, as the walker does.
                **({"missing_on_crash": True} if graph.missing_on_crash(node_id) else {}),
                # A genuine failure (never a timeout) leaves a step with a fallback missing,
                # for its fallback gate, as the walker records it.
                **({"falls_back": True} if graph.falls_back(node_id) else {}),
                # Where the node works. Only a dispatched worker can give it one
                # (`workers.lower_workflow` refuses it on the CEO-subagent route).
                **({"workspace": dict(node.workspace)} if node.workspace is not None else {}),
                # The step's own route: `workers.lower_workflow` puts it on its worker's
                # model context, as the walker puts it on the step's client.
                **({"route": dict(_resolved_route)} if _resolved_route is not None else {}),
                **({"tool_policy": node.tool_policy} if node.tool_policy is not None else {}),
                **({"keeps_tool_markup": True} if node.keeps_tool_markup else {}),
            )

    # SWEEP-1: Validate fork-shape rules at lowering so the spine refuses the same
    # shapes the walker does (nn/fork.py:112-122). Branches that produce the same
    # facts, overlap in reads/writes, or read a sibling's facts are refused; the rules
    # live in one place (`nn.validate.fork_branch_conflicts`), which the validator's V16
    # also reads, so a search is refused them before anything is lowered.
    from refactory_dsh.nn.validate import fork_branch_conflicts, fork_branches

    for fork_id, fork_node in graph.nodes.items():
        if fork_node.kind != "fork":
            continue
        for branch in fork_branches(graph, fork_node, graph.fork_join(fork_id)):
            for current in sorted(branch):
                if graph.nodes[current].kind == "data":
                    raise LoweringError(
                        f"fork {fork_id!r}: data node {current!r} inside a fork branch "
                        "is not supported; move the data node outside the fork or "
                        "place the fork inside the data body"
                    )
    for _fork_id, reason in fork_branch_conflicts(graph, knob_values):
        raise LoweringError(reason)

    edges = [
        Edge(source=edge.source, target=edge.target, condition=edge.condition)
        for edge in sorted(graph.edges, key=lambda e: e.key())
    ]
    return Workflow(
        name=name or "mode",
        nodes=nodes,
        edges=edges,
        start_node=graph.start,
        terminal=True,
        knob_values=_ir_knobs(state.get("knobs") or {}),
        knob_specs={
            knob_name: dict(entry.get("spec") or {})
            for knob_name, entry in (state.get("knobs") or {}).items()
        },
    )


SMOKE_PRODUCES = frozenset(("smoke_score", "smoke_ok"))


def _is_smoke_node(node: Any) -> bool:
    """Whether this node runs a smoke validation and needs the smoke envelope.

    Two-factor discriminator: the node must both (1) produce ``smoke_score``
    and ``smoke_ok`` — facts only the smoke gate/op produces — and
    (2) declare ``smoke_timeout_s`` in its knobs.  This excludes
    ``prepare``, which declares ``smoke_timeout_s`` in its knobs (because it
    relays the value as a fact) but does not produce the smoke results.
    """
    return (
        SMOKE_PRODUCES.issubset(set(node.produces))
        and "smoke_timeout_s" in node.knobs
    )


def _smoke_budget(state: dict[str, Any]) -> int:
    """The smoke budget from the state's knobs, validated."""
    from refactory_dsh.sandbox import DEFAULT_MODEL_SMOKE_TIMEOUT_S

    budget = int((state.get("knobs") or {}).get("smoke_timeout_s", {}).get(
        "value", DEFAULT_MODEL_SMOKE_TIMEOUT_S,
    ))
    if not 1 <= budget <= 86400:
        raise LoweringError("smoke_timeout_s must be between 1 and 86400 seconds")
    return budget


def _is_check_node(node: Any) -> bool:
    """Whether this node runs repo_change's check and needs the check envelope.

    Every mode that uses ``repo_change.run_check`` declares
    ``*repo_change.CHECK_PRODUCES`` on the gate or op that calls it, and
    declares ``check_timeout_s`` in its knobs so the budget travels to the
    operation. Both conditions are required: ``materialise`` also produces
    ``commit`` and ``files`` (the check rewrites them after an install
    amends the attempt), but it does not declare the knob because it does
    not run the check. No fact-level discriminator (CHECK_WRITES or
    CHECK_PRODUCES alone) can exclude materialise, because materialise
    writes ``changes.diff`` and produces both ``commit`` and ``files``.

    A check that runs on a commit already prepared -- design mode's check
    beside its reviews, which leaves the commit facts to ``prepare_check`` --
    is known by the fact only a check produces, ``runtime_evidence``
    (`authoring_evidence.check_nodes`).
    """
    from refactory_dsh.invocations import CHECK_PRODUCES

    return (
        (set(CHECK_PRODUCES).issubset(set(node.produces))
         or "runtime_evidence" in node.produces)
        and "check_timeout_s" in node.knobs
    )


#: The commands one check op or gate can run one after another, each bounded by its own
#: inactivity limit and, however much it prints, by the shared backstop
#: (``sandbox.COMMAND_BACKSTOP_S``): the dependency install of the attempt (or of the check
#: copy beside the reviews), its ``generate`` and ``setup`` commands, the targeted
#: pre-pass, the check itself, then on the unchanged base its install, the new tests'
#: guardedness run, the base check and the targeted base re-run.
CHECK_COMMAND_PHASES = 9
#: A delivery runs the merged check's phases, plus the merged tree's install and its
#: regeneration.
DELIVERY_COMMAND_PHASES = CHECK_COMMAND_PHASES + 2


def _envelope_for_phases(phases: int) -> int:
    """The wall cap of an op that runs ``phases`` commands one after another.

    Each command can run up to the backstop while it keeps printing, so the op's own
    cap covers every one of them plus the dispatch overhead: the runtime never stops
    an op before the limits of the commands inside it have had their say.
    """
    from refactory_dsh.model import NODE_DISPATCH_OVERHEAD_S
    from refactory_dsh.sandbox import COMMAND_BACKSTOP_S

    return phases * COMMAND_BACKSTOP_S + NODE_DISPATCH_OVERHEAD_S


def _check_envelope_s(state: dict[str, Any]) -> int:
    """The command_timeout_s a check node needs, in seconds.

    The ``check_timeout_s`` knob is an inactivity limit: a command is stopped only when
    it has produced no output for that many seconds, and the backstop is each command's
    absolute wall-clock cap. The envelope covers ``CHECK_COMMAND_PHASES`` such commands
    in sequence (`_envelope_for_phases`), so the runtime does not stop the op while one
    of its commands is still within its own limits.
    """
    knob_entry = (state.get("knobs") or {}).get("check_timeout_s", {})
    budget = int(knob_entry.get(
        "value",
        (knob_entry.get("spec") or {}).get("default", 180),
    ))
    if not 1 <= budget <= 86400:
        raise LoweringError("check_timeout_s must be between 1 and 86400 seconds")
    return _envelope_for_phases(CHECK_COMMAND_PHASES)


def _is_deliver_node(node: Any) -> bool:
    """Whether this op needs the delivery envelope: it declares ``check_timeout_s``
    but does not produce the check's own facts.

    An op that runs the merged check declares ``check_timeout_s`` in its knobs,
    following the convention ``_is_check_node`` uses.  Every such op that is not
    itself a check node gets the delivery envelope.  This is generic: any future
    mode whose delivery runs a merged check declares the knob and gets the
    envelope automatically.
    """
    return "check_timeout_s" in node.knobs and not _is_check_node(node)


def _delivery_envelope_s(state: dict[str, Any]) -> int:
    """The command_timeout_s a deliver node needs, in seconds.

    The merged check's commands and the merged tree's install and regeneration, each up
    to the backstop: ``DELIVERY_COMMAND_PHASES`` in sequence (`_envelope_for_phases`).
    """
    return _envelope_for_phases(DELIVERY_COMMAND_PHASES)


def _smoke_envelope_s(state: dict[str, Any], runs: int) -> int:
    """The command_timeout_s of a smoke node that makes ``runs`` sandboxed runs.

    Each run is bounded by the smoke budget on a wall clock (`sandbox.run_sandboxed`),
    so the old ``runs × budget + 600`` still holds for them; a smoke node can also run
    a command, which is bounded by the backstop instead. The larger of the two.
    """
    from refactory_dsh.model import NODE_DISPATCH_OVERHEAD_S
    from refactory_dsh.sandbox import COMMAND_BACKSTOP_S

    budget = _smoke_budget(state)
    return max(runs * budget + 600, COMMAND_BACKSTOP_S + NODE_DISPATCH_OVERHEAD_S)


def command_timeout_s(node: Any, state: dict[str, Any]) -> int | None:
    """The per-node command_timeout_s for ops and fn gates, or None.

    Extracted from the inline classification in ``to_workflow()`` so both lowering
    and the walker executor share one formula.  A check or delivery covers each of the
    commands it runs in sequence up to the backstop (`_envelope_for_phases`); a smoke
    node covers its sandboxed runs' wall clock and the backstop (`_smoke_envelope_s`).
    """
    if node.kind == "op":
        if _is_smoke_node(node):
            return _smoke_envelope_s(state, 3)
        if _is_check_node(node):
            return _check_envelope_s(state)
        if _is_deliver_node(node):
            return _delivery_envelope_s(state)
    elif node.kind == "gate" and getattr(node, "gate_kind", "fn") == "fn":
        if _is_smoke_node(node):
            # The new mode's smoke, the empty-input run (`empty_input`) and the
            # revision source's smoke.
            return _smoke_envelope_s(state, 3)
        if _is_check_node(node):
            return _check_envelope_s(state)
    return None


def _ir_knobs(knobs: dict[str, Any]) -> dict[str, str | float]:
    """The knob values as the IR records them.

    The IR's field is ``str | float`` and pydantic is strict, so a bool or an int
    has to be projected. The authoritative transport is the JSON in the command
    environment, which keeps the types; this is the IR's own record of the run.
    """
    out: dict[str, str | float] = {}
    for knob_name, entry in knobs.items():
        value = entry.get("value")
        if isinstance(value, bool):
            out[knob_name] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            out[knob_name] = float(value)
        else:
            out[knob_name] = str(value)
    return out


def knob_environment(state: dict[str, Any], *, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a node command runs under: the knobs, and whatever else."""
    values = {name: entry.get("value") for name, entry in (state.get("knobs") or {}).items()}
    env = {KNOBS_ENV: json.dumps(values, default=str)}
    env.update(extra or {})
    return env


def write_mode_graph(
    state: dict[str, Any],
    out_dir: Path,
    *,
    name: str,
    description: str = "",
    transform: Any = None,
) -> tuple[Path, Workflow]:
    """Write ``<out_dir>/<name>.graph.json`` the way the science bundle loads it.

    ``transform`` is applied to the workflow before it is written, so a caller can
    layer something over the IR without this function knowing what the layer is. The
    default writes the mode unchanged.
    """
    workflow = to_workflow(state, name=name, description=description)
    if transform is not None:
        workflow = transform(workflow)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}{SUFFIX}"
    path.write_text(json.dumps(workflow.to_dict(), indent=2, default=str))
    return path, workflow


def node_manifest(state: dict[str, Any], node_id: str) -> dict[str, Any]:
    """What a node command is told about itself, mirroring `DSH_WORKFLOW_NODE`."""
    graph = Graph.from_state(state.get("graph") or {})
    node = graph.nodes[node_id]
    return {
        "id": node.id,
        "type": {"op": "FnNode", "agent": "AgentNode", "gate": "GateNode"}[node.kind],
        "command": node.call or node.evaluator,
        # As authored, as `node_declarations` carries them (`graph.op_io`).
        "reads": list(op_io(node)[1]),
        "writes": list(op_io(node)[0]),
    }


def knob_specs(state: dict[str, Any]) -> dict[str, Knob]:
    return {
        name: Knob.from_spec(dict(entry.get("spec") or {}))
        for name, entry in (state.get("knobs") or {}).items()
    }
