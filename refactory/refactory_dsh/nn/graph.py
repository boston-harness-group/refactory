"""The structure of a mode: nodes, edges, and their canonical serialization.

The graph is data. An optimizer edits this data, never Python source, which is
what makes a candidate revertible, diffable and exactly validatable.

Serialization is canonical: nodes are sorted by id and edges by
``(source, target, condition)``. Canonical form is what makes
``build() -> state -> build()`` an identity rather than an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

#: The node kinds a mode may declare. `fork`/`join` are a barrier pair and
#: `data` iterates a subgraph once per item; all three are executed by the
#: runtime, and they are kinds here because a mode has to be able to declare
#: them (create-v2 forks its three reviews, and a dataset mode is a DataNode).
NODE_KINDS = ("op", "agent", "gate", "fork", "join", "data")

#: Who answers a gate: the evaluated function, a person, or the running agent.
GATE_KINDS = ("fn", "user", "agent")

#: The failures an agent node may declare a retry on (`Node.retry`). Only a timeout: a
#: session that ran out of time is the one failure a fresh, longer session can fix. A
#: model that answered wrongly answers wrongly again, and a transport blip is already
#: retried inside the model client (`model.DEFAULT_ATTEMPTS`).
RETRY_ON = ("timeout",)

#: The most retries one node may declare, and the largest factor its limit may grow by
#: per retry. A bound, so a declared rule cannot hold a run for an unbounded time.
MAX_RETRY_TIMES = 3
MAX_TIMEOUT_FACTOR = 10


def retry_declaration(value: Any) -> dict[str, Any] | None:
    """A retry declaration in its one canonical form, or None for no retry.

    A mapping of ``on`` (the failures it answers, from `RETRY_ON`), ``times`` (how many
    retries) and ``timeout_factor`` (how much longer each retry's session limit is than
    the last one's). Unset keys take ``on: ["timeout"]``, ``times: 1`` and
    ``timeout_factor: 2``. Canonical -- ``on`` sorted, a whole factor an integer -- so a
    rule written two ways serializes, and therefore identifies a candidate, one way.
    Raises ``ValueError`` saying what is wrong.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"a retry is a mapping of on, times and timeout_factor, not {value!r}")
    unknown = sorted(set(value) - {"on", "times", "timeout_factor"})
    if unknown:
        raise ValueError(f"a retry takes on, times and timeout_factor; unknown: {unknown}")
    on = value.get("on", ["timeout"])
    if isinstance(on, str) or not isinstance(on, (list, tuple)) or not on:
        raise ValueError(f"a retry's 'on' is a non-empty list of failures, not {on!r}")
    strange = sorted({str(item) for item in on} - set(RETRY_ON))
    if strange:
        raise ValueError(f"a retry can answer {list(RETRY_ON)} only; unknown: {strange}")
    times = value.get("times", 1)
    if isinstance(times, bool) or not isinstance(times, int) or not 1 <= times <= MAX_RETRY_TIMES:
        raise ValueError(f"a retry's 'times' is a whole number from 1 to {MAX_RETRY_TIMES}, "
                         f"not {times!r}")
    factor = value.get("timeout_factor", 2)
    if (isinstance(factor, bool) or not isinstance(factor, (int, float))
            or not 1 <= factor <= MAX_TIMEOUT_FACTOR):
        raise ValueError(f"a retry's 'timeout_factor' is a number from 1 to {MAX_TIMEOUT_FACTOR}, "
                         f"not {factor!r}")
    if float(factor).is_integer():
        factor = int(factor)
    return {"on": sorted({str(item) for item in on}), "times": times, "timeout_factor": factor}

#: Where an agent node works (`Node.workspace`). ``none`` is the run directory, as
#: every node had before workspaces; ``scratch`` is a fresh empty folder per dispatch;
#: ``clone`` is a prepared clone of the run's worktree (`refactory_dsh.workspace`).
WORKSPACE_KINDS = ("none", "scratch", "clone")

#: What a ``clone`` workspace is a clone of: the run's last attempt, or its base.
WORKSPACE_OF = ("attempt", "base")

#: How a workspace's work becomes the node's output: its declared writes copied out of
#: it (``files``), or the clone's diff written as the node's answer file (``diff``).
WORKSPACE_COLLECT = ("files", "diff")


def workspace_declaration(value: Any) -> dict[str, str] | None:
    """A workspace declaration in its one canonical form, or None for ``none``.

    Accepts the kind alone (``"scratch"``) or a mapping of ``kind``, ``of`` and
    ``collect``. ``of`` defaults to ``attempt`` and belongs to a clone only;
    ``collect`` defaults to ``files``, and ``diff`` needs a clone, the only kind with a
    base to diff against. Canonical, so a declaration written two ways serializes, and
    therefore identifies a candidate, one way. Raises ``ValueError`` saying what is
    wrong.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = {"kind": value}
    if not isinstance(value, dict):
        raise ValueError(f"a workspace is a kind or a mapping of kind, of and collect, not {value!r}")
    unknown = sorted(set(value) - {"kind", "of", "collect"})
    if unknown:
        raise ValueError(f"a workspace takes kind, of and collect; unknown: {unknown}")
    kind = value.get("kind", "none")
    if kind not in WORKSPACE_KINDS:
        raise ValueError(f"unknown workspace kind {kind!r}; known: {list(WORKSPACE_KINDS)}")
    if kind == "none":
        if len(value) > 1:
            raise ValueError("a 'none' workspace takes no 'of' or 'collect': the node works in the run directory")
        return None
    collect = value.get("collect", "files")
    if collect not in WORKSPACE_COLLECT:
        raise ValueError(f"unknown workspace collect {collect!r}; known: {list(WORKSPACE_COLLECT)}")
    if kind == "clone":
        of = value.get("of", "attempt")
        if of not in WORKSPACE_OF:
            raise ValueError(f"unknown clone workspace 'of' {of!r}; known: {list(WORKSPACE_OF)}")
        return {"kind": kind, "of": of, "collect": collect}
    if "of" in value:
        raise ValueError("only a 'clone' workspace takes 'of'")
    if collect == "diff":
        raise ValueError("'collect: diff' needs a clone, which has a base to diff against; "
                         "a scratch workspace collects its files")
    return {"kind": kind, "collect": collect}

#: Who answers a gate. ``fn`` runs the evaluator and routes by its verdict; ``user``
#: is a pause the person answers while using the mode; ``agent`` is one the running
#: agent answers. Only ``fn`` can be answered with nobody present, which is why the
#: other two are refused by a headless run that has not been given a verdict.

_NODE_FIELDS = frozenset(
    {
        "id",
        "kind",
        "call",
        "evaluator",
        "gate_kind",
        "knobs",
        "reads",
        "writes",
        "produces",
        "prompt_template",
        "role",
        "resource_role",
        "tools",
        "max_iterations",
        "notes",
        "blocking",
        "score_effect",
        "targets",
        "optional_targets",
        "sources",
        "subgraph_entry",
        "subgraph_exit",
        "task_ref",
        "source_path",
        "source_format",
        "inline_items",
        "limit",
        "feedback_keys",
        "timeout_s",
        "retry",
        "outputs_only",
        "workspace",
    }
)


class GraphError(ValueError):
    """A graph could not be built or read from its serialized form."""


@dataclass
class Node:
    """One step of a mode.

    ``op`` runs a resolved Python callable, ``agent`` makes one model call, and
    ``gate`` routes by a verdict. ``gate_kind`` says who gives it: ``fn`` runs
    ``evaluator``, while ``user`` and ``agent`` are answered by a person or by the
    running agent, so they carry no evaluator at all.

    ``writes`` are the files the node produces and ``produces`` the run facts it
    sets. Both are declared rather than discovered, because the validator's
    statement that "every read is produced by some node" is only as true as the
    declarations, and the executor enforces them.
    """

    id: str
    kind: str
    call: str = ""
    evaluator: str = ""
    #: Who answers this gate: ``fn`` (the default), ``user`` or ``agent``.
    gate_kind: str = "fn"
    knobs: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    prompt_template: str = ""
    role: str = "ceo"
    tools: tuple[str, ...] = ()
    max_iterations: int = 1
    notes: str = ""
    blocking: bool = True
    #: Whether inserting this node can change the score. ``affects`` is the
    #: default; ``none`` says the node is advisory, which is worth stating because
    #: a block that cannot change the number is otherwise indistinguishable from
    #: one that can, and an optimizer given both as equals will try the useless one.
    score_effect: str = "affects"
    #: `fork`: the nodes whose branches start here. A fork reaches its branches
    #: through this list rather than through edges, because one out-edge per branch
    #: would schedule the successor as soon as the first branch finished.
    targets: tuple[str, ...] = ()
    #: `fork`: targets whose failure does not fail the fork.  Must be a subset
    #: of ``targets``.  A failed optional branch is skipped in the fact merge;
    #: the lowered graph marks these nodes ``optional: true`` in their worker
    #: spec so DSH and run.py also tolerate the failure.
    optional_targets: tuple[str, ...] = ()
    #: `join`: the nodes this barrier waits for. Declared for the same reason.
    sources: tuple[str, ...] = ()
    #: `data`: the subgraph it runs once per item, and where the items come from.
    subgraph_entry: str = ""
    subgraph_exit: str = ""
    task_ref: str = ""
    source_path: str = ""
    source_format: str = ""
    inline_items: tuple[dict[str, Any], ...] = ()
    limit: int = 0
    #: Resource phase is independent of worker routing. Unset preserves the
    #: enclosing phase, except the legacy validation role enters validation.
    resource_role: str | None = None
    feedback_keys: tuple[str, ...] = ()
    #: `agent`: the most wall time the node's model work may take, in seconds, on
    #: either runtime. Unset, the selected route's envelope applies. A node that
    #: runs over fails like any other failed node, so an optional fork branch is
    #: then missing rather than late.
    timeout_s: int | None = None
    #: `agent`: what the runtime does when the node's session fails, in its canonical
    #: form (`retry_declaration`), e.g. ``{"on": ["timeout"], "times": 1,
    #: "timeout_factor": 2}``: a session that times out is run once more, fresh, with
    #: twice its `timeout_s` (which a retry requires), inside the node's whole budget
    #: and the run's wall (`refactory_dsh.node_retry`). None is no retry, as before. A
    #: rule the graph declares, on both runtimes, with or without a CEO.
    retry: dict[str, Any] | None = None
    #: `agent`: the node's only effect is its declared writes, its own report files --
    #: a reviewer, a judge, a researcher -- never the run's worktree or the target. Only
    #: such a node may declare a retry: its outputs are set aside and the retry writes
    #: them afresh. A node that changes shared state would have its second session
    #: start from whatever the first one left. A timed-out attempt during which the
    #: target or the run's worktree changed is not retried (`node_retry.settle_for_retry`).
    outputs_only: bool = False
    #: `agent`: where the node changes files and runs commands (`WORKSPACE_KINDS`),
    #: in its canonical form (`workspace_declaration`). None is ``none``: the run
    #: directory, as before. The graph still decides what crosses into shared state:
    #: the runtime collects the node's output from its workspace before delivery
    #: checks it (`refactory_dsh.workspace`).
    workspace: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise GraphError(f"node {self.id!r}: unknown kind {self.kind!r}")
        if not self.id:
            raise GraphError("a node needs an id")
        if self.kind == "op" and not self.call:
            raise GraphError(f"op node {self.id!r} declares no call")
        if self.kind == "gate" and self.gate_kind not in GATE_KINDS:
            raise GraphError(
                f"gate node {self.id!r}: unknown gate_kind {self.gate_kind!r}; "
                f"known: {list(GATE_KINDS)}"
            )
        if self.kind == "gate" and self.gate_kind == "fn" and not self.evaluator:
            raise GraphError(f"gate node {self.id!r} declares no evaluator")
        if self.kind == "gate" and self.gate_kind != "fn" and self.evaluator:
            # A `user` or `agent` gate is answered by a person or by the running
            # agent. An evaluator would be a second answerer, and which one won
            # would depend on the runtime, so it is refused rather than ignored.
            raise GraphError(
                f"gate node {self.id!r} is answered by {self.gate_kind!r} and "
                "declares an evaluator too, so two things would answer it"
            )
        if self.kind == "agent" and not self.prompt_template:
            raise GraphError(f"agent node {self.id!r} declares no prompt_template")
        if self.kind == "fork" and not self.targets:
            raise GraphError(f"fork node {self.id!r} declares no targets")
        if self.kind == "join" and not self.sources:
            raise GraphError(f"join node {self.id!r} declares no sources")
        if self.kind == "data":
            if not self.subgraph_entry or not self.subgraph_exit:
                raise GraphError(
                    f"data node {self.id!r} needs subgraph_entry and subgraph_exit, "
                    "which say which part of the graph runs once per item"
                )
        if self.optional_targets:
            if self.kind != "fork":
                raise GraphError(f"node {self.id!r} is not a fork but declares optional_targets")
            if set(self.optional_targets) - set(self.targets):
                raise GraphError("optional_targets must be a subset of targets")
        if self.kind != "fork" and self.targets:
            raise GraphError(f"node {self.id!r} is not a fork but declares targets")
        if self.kind != "join" and self.sources:
            raise GraphError(f"node {self.id!r} is not a join but declares sources")
        if self.max_iterations < 1:
            raise GraphError(f"node {self.id!r}: max_iterations must be >= 1")
        if self.score_effect not in ("affects", "none"):
            raise GraphError(
                f"node {self.id!r}: unknown score_effect {self.score_effect!r}; "
                "known: affects, none"
            )
        if self.resource_role not in (None, "work", "final_validation"):
            raise GraphError(f"node {self.id!r}: unknown resource_role {self.resource_role!r}")
        if self.timeout_s is not None:
            if self.kind != "agent":
                raise GraphError(f"node {self.id!r} is not an agent but declares timeout_s")
            if self.timeout_s < 1:
                raise GraphError(f"node {self.id!r}: timeout_s must be >= 1")
        if self.outputs_only:
            if self.kind != "agent":
                raise GraphError(f"node {self.id!r} is not an agent but declares outputs_only")
            if not self.writes:
                raise GraphError(f"node {self.id!r} declares outputs_only and no writes, so it "
                                 "has no outputs to be the only effect of")
            rewritten = sorted(set(self.writes) & set(self.reads))
            if rewritten:
                raise GraphError(f"node {self.id!r} declares outputs_only but reads what it "
                                 f"writes ({rewritten}): its outputs are not its own reports")
        if self.workspace is not None:
            try:
                self.workspace = workspace_declaration(self.workspace)
            except ValueError as exc:
                raise GraphError(f"node {self.id!r}: {exc}") from exc
            if self.workspace is not None and self.kind != "agent":
                raise GraphError(f"node {self.id!r} is not an agent but declares a workspace")
            if self.workspace is not None and self.workspace["collect"] == "diff" and not self.writes:
                raise GraphError(f"node {self.id!r} collects its workspace's diff, which is written "
                                 "into its first declared write, and it declares none")

        if self.retry is not None:
            if self.kind != "agent":
                raise GraphError(f"node {self.id!r} is not an agent but declares a retry")
            try:
                self.retry = retry_declaration(self.retry)
            except ValueError as exc:
                raise GraphError(f"node {self.id!r}: {exc}") from exc
            if self.timeout_s is None:
                raise GraphError(
                    f"node {self.id!r} declares a retry but no timeout_s: a retry's limit is "
                    "its timeout_factor times the node's own bound, so the node needs one")
            if not self.outputs_only and self.workspace is None:
                # A node that may change the run's worktree or the target cannot simply
                # run again: the second session would start from what the first left.
                # A node with a workspace is safe: each attempt gets a fresh workspace
                # via workspace.dispatched(fresh=True), which discards the previous one.
                # A clone and a scratch folder both keep an attempt's writes inside the
                # workspace until collect, and collect runs only after the session
                # returns, so a timed-out attempt contributes nothing. The declaration is
                # canonical by here, so one that means "no workspace" is None and counts
                # as none.
                raise GraphError(
                    f"node {self.id!r} declares a retry but may change shared state (the "
                    "run's worktree or the target), so a second session would start from "
                    "whatever the first one left. Only a node whose outputs are its own "
                    "declared report files may retry: declare outputs_only=True or give "
                    "it a workspace")
    @property
    def resource_phase(self) -> str | None:
        """Explicit resource scope, with the historical validator fallback."""
        return self.resource_role or ("final_validation" if self.role == "validation" else None)

    def to_state(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "call": self.call,
            "evaluator": self.evaluator,
            # Only when it is not the default. A new field that always serialized
            # would change every mode's state dict, and therefore every candidate
            # id the loop has ever recorded -- which breaks resume and moves a
            # search for reasons that have nothing to do with the search.
            **({"gate_kind": self.gate_kind} if self.gate_kind != "fn" else {}),
            "knobs": list(self.knobs),
            "reads": list(self.reads),
            "writes": list(self.writes),
            "produces": list(self.produces),
            "prompt_template": self.prompt_template,
            "role": self.role,
            **({"resource_role": self.resource_role} if self.resource_role is not None else {}),
            "tools": list(self.tools),
            "max_iterations": self.max_iterations,
            "notes": self.notes,
            "blocking": self.blocking,
            "score_effect": self.score_effect,
            **({"targets": list(self.targets)} if self.targets else {}),
            **({"optional_targets": list(self.optional_targets)} if self.optional_targets else {}),
            **({"sources": list(self.sources)} if self.sources else {}),
            **({"subgraph_entry": self.subgraph_entry} if self.subgraph_entry else {}),
            **({"subgraph_exit": self.subgraph_exit} if self.subgraph_exit else {}),
            **({"task_ref": self.task_ref} if self.task_ref else {}),
            **({"source_path": self.source_path} if self.source_path else {}),
            **({"source_format": self.source_format} if self.source_format else {}),
            **({"inline_items": [dict(item) for item in self.inline_items]}
               if self.inline_items else {}),
            **({"limit": self.limit} if self.limit else {}),
            **({"feedback_keys": list(self.feedback_keys)} if self.feedback_keys else {}),
            **({"timeout_s": self.timeout_s} if self.timeout_s is not None else {}),
            **({"retry": {**self.retry, "on": list(self.retry["on"])}}
               if self.retry is not None else {}),
            **({"outputs_only": True} if self.outputs_only else {}),
            **({"workspace": dict(self.workspace)} if self.workspace is not None else {}),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Node":
        unknown = sorted(set(state) - _NODE_FIELDS)
        if unknown:
            raise GraphError(f"unknown node field(s): {unknown}")
        if "id" not in state or "kind" not in state:
            raise GraphError("a node state needs 'id' and 'kind'")
        return cls(
            id=str(state["id"]),
            kind=str(state["kind"]),
            call=str(state.get("call", "")),
            gate_kind=str(state.get("gate_kind", "fn")),
            evaluator=str(state.get("evaluator", "")),
            knobs=tuple(state.get("knobs") or ()),
            reads=tuple(state.get("reads") or ()),
            writes=tuple(state.get("writes") or ()),
            produces=tuple(state.get("produces") or ()),
            prompt_template=str(state.get("prompt_template", "")),
            role=str(state.get("role", "ceo")),
            resource_role=state.get("resource_role"),
            tools=tuple(state.get("tools") or ()),
            max_iterations=int(state.get("max_iterations", 1)),
            notes=str(state.get("notes", "")),
            blocking=bool(state.get("blocking", True)),
            score_effect=str(state.get("score_effect", "affects")),
            targets=tuple(state.get("targets") or ()),
            optional_targets=tuple(state.get("optional_targets") or ()),
            sources=tuple(state.get("sources") or ()),
            subgraph_entry=str(state.get("subgraph_entry", "")),
            subgraph_exit=str(state.get("subgraph_exit", "")),
            task_ref=str(state.get("task_ref", "")),
            source_path=str(state.get("source_path", "")),
            source_format=str(state.get("source_format", "")),
            inline_items=tuple(state.get("inline_items") or ()),
            limit=int(state.get("limit", 0)),
            feedback_keys=tuple(state.get("feedback_keys") or ()),
            timeout_s=(int(state["timeout_s"]) if state.get("timeout_s") is not None else None),
            retry=(dict(state["retry"]) if isinstance(state.get("retry"), dict)
                   else state.get("retry")),
            outputs_only=bool(state.get("outputs_only", False)),
            workspace=(dict(state["workspace"]) if isinstance(state.get("workspace"), dict)
                       else state.get("workspace")),
        )

    def with_field(self, field_name: str, value: Any) -> "Node":
        """A copy with one field replaced, refusing an unknown field."""
        if field_name not in _NODE_FIELDS:
            raise GraphError(f"node {self.id!r} has no field {field_name!r}")
        if field_name in ("knobs", "reads", "writes", "produces", "tools", "targets",
                          "optional_targets", "sources", "inline_items", "feedback_keys") and isinstance(value, list):
            value = tuple(value)
        return replace(self, **{field_name: value})


@dataclass(frozen=True)
class Edge:
    """One connection between two nodes, and the verdict that selects it.

    ``condition`` is the gate verdict this edge answers: ``None`` on the out-edge
    of a non-gate node, and a label on each out-edge of a gate. It is a label
    rather than an enum because a gate names its own outcomes.
    """

    source: str
    target: str
    condition: str | None = None

    def to_state(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "condition": self.condition}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Edge":
        return cls(
            source=str(state["source"]),
            target=str(state["target"]),
            condition=state.get("condition"),
        )

    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.condition or "")


class Graph:
    """A mutable graph. Edits mutate it; serialization canonicalizes it."""

    def __init__(self, *, auto_start: bool = True) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.start: str = ""
        # Authoring convenience only. A graph parsed from a state never guesses
        # its start node: guessing is a repair, and a repair hides the edit that
        # removed the real one.
        self._auto_start = auto_start

    # ── construction ───────────────────────────────────────────────────────
    def add(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise GraphError(f"node {node.id!r} already exists")
        self.nodes[node.id] = node
        if self._auto_start and not self.start:
            self.start = node.id
        return node

    def connect(self, source: str, target: str, condition: str | None = None) -> Edge:
        for node_id in (source, target):
            if node_id not in self.nodes:
                raise GraphError(f"cannot connect unknown node {node_id!r}")
        edge = Edge(source, target, condition)
        if any(existing.key() == edge.key() for existing in self.edges):
            raise GraphError(f"edge {edge.key()} already exists")
        self.edges.append(edge)
        return edge

    def copy(self) -> "Graph":
        return Graph.from_state(self.to_state())

    # ── queries ────────────────────────────────────────────────────────────
    def out_edges(self, node_id: str, condition: str | None = None) -> list[Edge]:
        return [
            e
            for e in self.edges
            if e.source == node_id and (condition is None or e.condition == condition)
        ]

    def in_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target == node_id]

    def terminals(self) -> list[str]:
        return sorted(n for n in self.nodes if not self.out_edges(n))

    def reachable(self) -> set[str]:
        """Every node the walk can reach from the start, following fork fan-out.

        A fork reaches its branches through `targets` rather than edges (one
        out-edge per branch would let the successor run as soon as the first branch
        finished), so a traversal that only follows edges would call every branch
        unreachable.
        """
        seen: set[str] = set()
        stack = [self.start] if self.start in self.nodes else []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(e.target for e in self.out_edges(current))
            node = self.nodes.get(current)
            if node is not None and node.kind == "fork":
                stack.extend(node.targets)
            if node is not None and node.kind == "data":
                # The subgraph is reached through the data node, not through an edge,
                # for the same reason a fork's branches are: the entry is the thing
                # the runtime starts, and an edge into it would run it once outside
                # the iteration.
                stack.append(node.subgraph_entry)
        return seen

    def knob_consumers(self) -> dict[str, list[str]]:
        """Which nodes consume each knob, for the dead-knob check."""
        consumers: dict[str, list[str]] = {}
        for node in self.nodes.values():
            for name in node.knobs:
                consumers.setdefault(name, []).append(node.id)
        for ids in consumers.values():
            ids.sort()
        return consumers

    # ── serialization ──────────────────────────────────────────────────────
    def to_state(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "nodes": {nid: self.nodes[nid].to_state() for nid in sorted(self.nodes)},
            "edges": [
                e.to_state()
                for e in sorted(self.edges, key=lambda edge: edge.key())
            ],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Graph":
        graph = cls(auto_start=False)
        if "start" not in state:
            raise GraphError("a graph state needs a 'start'")
        raw_nodes = state.get("nodes") or {}
        if not isinstance(raw_nodes, dict):
            raise GraphError("'nodes' must be an object keyed by node id")
        for node_id in sorted(raw_nodes):
            node_state = dict(raw_nodes[node_id])
            node_state.setdefault("id", node_id)
            if node_state["id"] != node_id:
                raise GraphError(
                    f"node key {node_id!r} disagrees with its id {node_state['id']!r}"
                )
            graph.nodes[node_id] = Node.from_state(node_state)
        graph.start = str(state["start"])
        if graph.start and graph.start not in graph.nodes:
            raise GraphError(f"start node {graph.start!r} does not exist")
        see: set[tuple[str, str, str]] = set()
        for raw in state.get("edges") or ():
            edge = Edge.from_state(raw)
            for node_id in (edge.source, edge.target):
                if node_id not in graph.nodes:
                    raise GraphError(f"edge references unknown node {node_id!r}")
            if edge.key() in see:
                raise GraphError(f"duplicate edge {edge.key()}")
            see.add(edge.key())
            graph.edges.append(edge)
        return graph

    @classmethod
    def from_nodes(cls, nodes: Iterable[Node], edges: Iterable[Edge], start: str) -> "Graph":
        graph = cls()
        for node in nodes:
            graph.add(node)
        for edge in edges:
            graph.connect(edge.source, edge.target, edge.condition)
        graph.start = start
        return graph
