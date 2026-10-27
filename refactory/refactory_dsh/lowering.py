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
from typing import Any

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


def _optional_node_ids(graph: Graph) -> set[str]:
    """Node ids that appear in some fork's optional_targets."""
    ids: set[str] = set()
    for node in graph.nodes.values():
        if node.kind == "fork" and node.optional_targets:
            ids.update(node.optional_targets)
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
    optional_ids = _optional_node_ids(graph)
    declarations: dict[str, dict[str, Any]] = {}
    for node_id, node in graph.nodes.items():
        entry: dict[str, Any] = {
            "kind": node.kind,
            "reads": sorted(node.reads),
            "writes": sorted(node.writes),
            "produces": sorted(node.produces),
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
        if node.feedback_keys:
            entry["feedback_keys"] = list(node.feedback_keys)
        if node.workspace is not None:
            # Where the node works, for a reader of the run directory; the worker spec
            # carries the same declaration to the node's dispatch.
            entry["workspace"] = dict(node.workspace)
        declarations[node_id] = entry
    return declarations


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
            )
        elif node.kind == "gate":
            # Who answers it decides whether there is a command at all. `fn` runs its
            # evaluator; `user` and `agent` are handed to the driving agent by the
            # spine, which presents a user gate to the person. Emitting a command for
            # those would give the gate two answerers, and which one won would depend
            # on the runtime.
            gate: dict[str, Any] = {
                "evaluator_type": node.gate_kind,
                "max_iterations": node.max_iterations,
            }
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
            if node.optional_targets:
                fork_kwargs["optional_targets"] = list(node.optional_targets)
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
            nodes[node_id] = AgentNode(
                **common,
                role=role,
                prompt_template=node.prompt_template,
                tools=list(node.tools),
                max_iterations=node.max_iterations,
                # The node's model-session wall: `workers.lower_workflow` gives the
                # dispatched session this bound and the command it plus the dispatch
                # overhead; the walker gives its model client the same bound.
                **({"timeout": node.timeout_s} if node.timeout_s is not None else {}),
                # Where the node works. Only a dispatched worker can give it one
                # (`workers.lower_workflow` refuses it on the CEO-subagent route).
                **({"workspace": dict(node.workspace)} if node.workspace is not None else {}),
            )

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


def _check_envelope_s(state: dict[str, Any]) -> int:
    """The command_timeout_s a check node needs, in seconds.

    Covers every sub-operation ``run_check`` and ``_check_in_worktree`` perform
    sequentially: three dependency installs (each ``INSTALL_TIMEOUT_S``), setup,
    generate and the check command (each ``check_timeout_s``), and the dispatch
    overhead.
    """
    from refactory_dsh.model import NODE_DISPATCH_OVERHEAD_S
    from refactory_dsh.invocations import INSTALL_TIMEOUT_S

    knob_entry = (state.get("knobs") or {}).get("check_timeout_s", {})
    budget = int(knob_entry.get(
        "value",
        (knob_entry.get("spec") or {}).get("default", 180),
    ))
    if not 1 <= budget <= 86400:
        raise LoweringError("check_timeout_s must be between 1 and 86400 seconds")
    return 3 * INSTALL_TIMEOUT_S + 3 * budget + NODE_DISPATCH_OVERHEAD_S


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

    Covers the full merged-check envelope (``_check_envelope_s``) plus one
    additional install cycle for the merged tree's dependencies
    (``INSTALL_TIMEOUT_S``) and overhead for the merge and regeneration steps
    (``NODE_DISPATCH_OVERHEAD_S``).
    """
    from refactory_dsh.model import NODE_DISPATCH_OVERHEAD_S
    from refactory_dsh.invocations import INSTALL_TIMEOUT_S

    return _check_envelope_s(state) + INSTALL_TIMEOUT_S + NODE_DISPATCH_OVERHEAD_S


def command_timeout_s(node: Any, state: dict[str, Any]) -> int | None:
    """The per-node command_timeout_s for ops and fn gates, or None.

    Extracted from the inline classification in ``to_workflow()`` so both lowering
    and the walker executor share one formula.
    """
    if node.kind == "op":
        if _is_smoke_node(node):
            return 3 * _smoke_budget(state) + 600
        if _is_check_node(node):
            return _check_envelope_s(state)
        if _is_deliver_node(node):
            return _delivery_envelope_s(state)
    elif node.kind == "gate" and getattr(node, "gate_kind", "fn") == "fn":
        if _is_smoke_node(node):
            return 2 * _smoke_budget(state) + 600
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
        "reads": sorted(node.reads),
        "writes": sorted(node.writes),
    }


def knob_specs(state: dict[str, Any]) -> dict[str, Knob]:
    return {
        name: Knob.from_spec(dict(entry.get("spec") or {}))
        for name, entry in (state.get("knobs") or {}).items()
    }
