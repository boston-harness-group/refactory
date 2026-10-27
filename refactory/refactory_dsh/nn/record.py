"""Where a walked run stands, kept by the spine's rules and written in the spine's files.

The spine keeps three things about a run besides its facts: the nodes it has completed
(a reloop un-completes the ones it goes back over), each gate's last verdict, and the
feedback a reloop leaves for the node it goes back to (`Spine.completeDue`,
`Spine.verdict` and `Spine.uncompleteBetween` in
`packages/workflow/workflow-spine/src/spine.ts`). It writes the completed nodes to
`.refactory/spine-status.json` and the rest to its checkpoint, and every reader of a run
directory -- `refactory status`, the console, steer, the live view -- reads them there.

The walker keeps the same three here, by the same rules, so a walked run's status file
says what a driven run's would. They travel in the walker's own progress record
(`.refactory/progress.json`), which is how a resumed walk carries them on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph import Graph, Node

#: The spine's name for each walker node kind (`nodeKind` in workflow-spine's `index.ts`).
KINDS = {"agent": "agent", "gate": "gate", "op": "operation", "fork": "control", "join": "control",
         "data": "control"}


@dataclass
class RunRecord:
    """The completed nodes, the gates' last verdicts and the pending reloop feedback."""

    completed: list[str] = field(default_factory=list)
    gate_verdicts: dict[str, str] = field(default_factory=dict)
    feedback: dict[str, str] = field(default_factory=dict)

    def complete(self, node: Node) -> None:
        """``node`` finished. An agent's feedback was for this visit, so it is spent."""
        if node.id not in self.completed:
            self.completed.append(node.id)
        if node.kind == "agent":
            self.feedback.pop(node.id, None)

    def gate(self, graph: Graph, gate: str, outcome: str, *, target: str | None, retry: bool,
             reason: str = "") -> None:
        """``gate`` decided ``outcome``. A retry hands ``reason`` to ``target`` and
        un-completes the nodes between the two, which run again."""
        if gate not in self.completed:
            self.completed.append(gate)
        self.gate_verdicts[gate] = outcome.lower()
        # A retry's feedback for this gate was for this visit, as an agent's is (dm-226).
        self.feedback.pop(gate, None)
        if retry and target is not None:
            self.feedback[target] = reason
            self._uncomplete_between(graph, target, gate)

    def _uncomplete_between(self, graph: Graph, target: str, gate: str) -> None:
        """The nodes a retry from ``gate`` back to ``target`` runs again: exactly the cycle,
        every node reachable from ``target`` that can also reach ``gate``, along non-retry
        edges, fork targets and join sources, as the spine reopens them
        (`Spine.uncompleteBetween`). A target that is a gate (design mode's send-backs onto
        ``pick_corrections``, dm-226) reopens what each of its routes leads to in the cycle."""
        from .loops import loop_edges

        retries = loop_edges(graph)
        forward: dict[str, list[str]] = {}
        backward: dict[str, list[str]] = {}

        def link(source: str, to: str) -> None:
            forward.setdefault(source, []).append(to)
            backward.setdefault(to, []).append(source)

        for edge in graph.edges:
            if edge.key() not in retries:
                link(edge.source, edge.target)
        for node_id, node in graph.nodes.items():
            if node.kind == "fork":
                for child in node.targets:
                    link(node_id, child)
            if node.kind == "join":
                for source in node.sources:
                    link(source, node_id)
            if node.kind == "data" and node.subgraph_entry:
                link(node_id, node.subgraph_entry)

        def reach(start: str, following: dict[str, list[str]]) -> set[str]:
            seen, queue = {start}, [start]
            for node in queue:
                for other in following.get(node, ()):
                    if other not in seen:
                        seen.add(other)
                        queue.append(other)
            return seen

        after, before = reach(target, forward), reach(gate, backward)
        cycle = {node for node in after if node != gate and (node == target or node in before)}
        self.completed = [entry for entry in self.completed if entry not in cycle]
        for node in cycle:
            self.gate_verdicts.pop(node, None)

    def merge_branches(self, branches: list[RunRecord], members: list[set[str]]) -> None:
        """Fold in a fork's branch records once the branches are done.

        The spine dispatches a fork's branches together and completes each boundary in
        branch order, so a branch's first nodes are all completed before any second one:
        the branches' completions are taken wave by wave, in the fork's target order.
        """
        for depth in range(max((len(branch.completed) for branch in branches), default=0)):
            for branch in branches:
                if depth < len(branch.completed) and branch.completed[depth] not in self.completed:
                    self.completed.append(branch.completed[depth])
        for branch, owned in zip(branches, members):
            for gate, outcome in branch.gate_verdicts.items():
                if gate in owned:
                    self.gate_verdicts[gate] = outcome
            for node in owned:
                if node in branch.feedback:
                    self.feedback[node] = branch.feedback[node]
                else:
                    self.feedback.pop(node, None)

    def to_progress(self) -> dict[str, Any]:
        return {"completed": list(self.completed), "gate_verdicts": dict(self.gate_verdicts),
                "feedback": dict(self.feedback)}

    @classmethod
    def from_progress(cls, record: dict[str, Any]) -> RunRecord:
        """The record a progress file kept; empty for one written before it kept any."""
        completed = record.get("completed")
        verdicts = record.get("gate_verdicts")
        feedback = record.get("feedback")
        return cls(
            completed=[str(node) for node in completed] if isinstance(completed, list) else [],
            gate_verdicts={str(key): str(value) for key, value in verdicts.items()}
            if isinstance(verdicts, dict) else {},
            feedback={str(key): str(value) for key, value in feedback.items()}
            if isinstance(feedback, dict) else {},
        )

    def write_status(self, workdir: Path, graph: Graph, status: str, nodes: list[str], *,
                     error: str | None = None, failed: str | None = None) -> None:
        """Write ``.refactory/spine-status.json`` as the spine's `recordStatus` does."""
        from refactory_dsh.run_state import write_status

        running = [{"id": node, "kind": KINDS.get(getattr(graph.nodes.get(node), "kind", ""), "operation")}
                   for node in nodes]
        listed = ", ".join(f"{entry['id']} ({entry['kind']})" for entry in running)
        user_gate = any(getattr(graph.nodes.get(node), "gate_kind", "") == "user"
                        and getattr(graph.nodes.get(node), "kind", "") == "gate" for node in nodes)
        wait = (error if status == "halted" and error is not None
                else f"waiting for user gate: {listed or 'user gate'}" if status == "waiting_user" or user_gate
                else (f"running a graph check: {listed}" if listed else "running a graph check")
                if status == "executing" else None)
        write_status(workdir, status=status, nodes=nodes, running_nodes=running,
                     completed=list(self.completed), error=error, wait_reason=wait,
                     last_failure={"node": failed, "error": error or ""} if failed is not None else None)
