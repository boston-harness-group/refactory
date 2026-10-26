"""The workflow IR this package owns and lowers a mode state into.

``Workflow.to_dict()`` is the JSON the shipped runtime reads, so its
serialization is behavior rather than a rendering detail: ``reads`` and
``writes`` are sorted for byte stability, every node carries a ``_type``
discriminator, and a key is emitted only when its value is non-empty.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── types formerly re-exported from remote-factory ────────────────


class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    STRATEGIST = "strategist"
    BUILDER = "builder"
    HEALTH_CHECKER = "health_checker"
    CODE_REVIEWER = "code_reviewer"
    ADVERSARIAL_TESTER = "adversarial_tester"
    FAILURE_ANALYST = "failure_analyst"
    CEO = "ceo"
    ARCHIVIST = "archivist"
    REFINER = "refiner"


class AgentConfig(BaseModel):
    """Configuration for an agent in the pool."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: AgentRole
    model: str
    timeout: int = 600


class VerdictType(str, Enum):
    PROCEED = "proceed"
    RELOOP = "reloop"
    HALT = "halt"


class Edge(BaseModel):
    """Directed edge in the workflow graph with an optional labelled condition.

    ``condition`` is normally one of the three ``VerdictType`` labels, but a gate
    may name its own forward outcomes (a switch rather than a binary decision).
    Arbitrary labels are stored verbatim and lowercased by consumers before
    matching, so ``"DRAFT"`` from an evaluator matches ``condition="draft"``.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    source: str
    target: str
    condition: VerdictType | str | None = None

    @property
    def condition_label(self) -> str | None:
        """Return the condition as a plain string, or ``None``."""
        if self.condition is None:
            return None
        return self.condition.value if isinstance(self.condition, VerdictType) else self.condition

# ── v2 extension fields ────────────────────────────────────────────
#
# v1's Node base and three node types lack fields v2 needs.  Rather than
# forking the entire class hierarchy, we subclass only the types that gain
# fields, keeping the same class names so ``_type`` discriminators and
# ``isinstance`` checks work unchanged.


class Node(BaseModel):
    """Base node in the workflow graph, with v2 extension fields."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    reads: set[str] = Field(default_factory=set)
    writes: set[str] = Field(default_factory=set)
    blocking: bool = True
    resource_role: Literal["work", "final_validation"] | None = None


class ArtifactCheck(BaseModel):
    """Validation rule for an agent-produced artifact."""

    model_config = ConfigDict(strict=True, extra="forbid")

    path: str
    must_exist: bool = True
    min_size: int = 0
    must_contain: list[str] = Field(default_factory=list)


class AgentNode(Node):
    """Node that invokes an agent."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: AgentRole
    model: str = ""
    prompt_template: str = ""
    tools: list[str] = Field(default_factory=list)
    timeout: int | None = None
    max_iterations: int = 1
    post_checks: list[ArtifactCheck] = Field(default_factory=list)
    dispatch_command: str | None = None
    command_timeout_s: int | None = Field(default=None, gt=0, le=2_147_483)
    #: The node's declared recovery (`nn.graph.retry_declaration`'s canonical form). None
    #: is no retry, and is left out of the serialized graph, so a graph that declares
    #: none is byte-identical to one lowered before the field existed.
    retry: dict[str, Any] | None = None
    #: Where the node works (`nn.graph.workspace_declaration`'s canonical form). None is
    #: the run directory, and is left out of the serialized graph, so a graph that
    #: declares no workspace is byte-identical to one lowered before the field existed.
    workspace: dict[str, str] | None = None


class FnNode(Node):
    """Node that runs a deterministic shell command or Python callable."""

    model_config = ConfigDict(strict=True, extra="forbid")

    command: str = ""
    callable_name: str | None = None
    notes: str = ""
    command_timeout_s: int | None = Field(default=None, gt=0, le=2_147_483)


class GateNode(Node):
    """Decision node that produces a verdict."""

    model_config = ConfigDict(strict=True, extra="forbid")

    evaluator_type: Literal["agent", "fn", "user"] = "agent"
    evaluator_role: AgentRole | None = None
    evaluator_command: str | None = None
    command_timeout_s: int | None = Field(default=None, gt=0, le=2_147_483)
    gate_prompt: str = ""
    max_iterations: int | None = None
    """Cap for this gate's reloop edge. ``None`` leaves the cap to the runtime."""


# ForkNode, JoinNode, DataNode need v2's Node base (which carries resource_role)
# even though v2 never sets resource_role on them — the lowering passes it in
# common kwargs and extra="forbid" would reject it without the field.


class ForkNode(Node):
    """Parallel execution node that launches all targets concurrently."""

    model_config = ConfigDict(strict=True, extra="forbid")

    targets: list[str]
    optional_targets: list[str] = Field(default_factory=list)


class JoinNode(Node):
    """Barrier node that waits for all sources to complete."""

    model_config = ConfigDict(strict=True, extra="forbid")

    sources: list[str]


class DataItem(BaseModel):
    """Standardized data payload for per-item workflow iteration."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    prompt: str = ""


class DataNode(Node):
    """Node that loads data items and drives per-item execution of a subgraph."""

    model_config = ConfigDict(strict=True, extra="forbid")

    task_ref: str | None = None
    source_path: str | None = None
    source_format: Literal["directory", "jsonl", "csv"] | None = None
    inline_items: list[DataItem] = Field(default_factory=list)
    subgraph_entry: str
    subgraph_exit: str
    parallelism: int = Field(default=1, ge=1)
    split: Literal["train", "val", "test", "all"] = "all"
    shuffle: bool = False
    shuffle_seed: int | None = None
    limit: int | None = None
    max_items: int = 500

    @model_validator(mode="after")
    def _validate_source(self) -> DataNode:
        sources = [
            self.task_ref is not None,
            self.source_path is not None,
            bool(self.inline_items),
        ]
        if sum(sources) != 1:
            raise ValueError(
                "Exactly one of task_ref, source_path, or inline_items must be set"
            )
        if self.source_path is not None and self.source_format is None:
            raise ValueError("source_path requires source_format to be set")
        return self


NodeType = AgentNode | FnNode | GateNode | ForkNode | JoinNode | DataNode


class Workflow(BaseModel):
    """A directed graph of typed nodes with labeled edges."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    nodes: dict[str, NodeType]
    edges: list[Edge]
    start_node: str
    terminal: bool = False
    task: str | None = Field(default=None)
    knob_values: dict[str, str | float] = Field(default_factory=dict)
    knob_bounds: dict[str, list[str | float]] = Field(default_factory=dict)
    knob_expandable: dict[str, str] = Field(default_factory=dict)
    knob_specs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    declared_capabilities: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workflow to a JSON-safe dict.

        ``reads`` and ``writes`` are sets, whose iteration order is not stable
        across processes; they are sorted here so that a given graph always
        serializes to identical bytes.
        """
        nodes_out: dict[str, Any] = {}
        for nid, node in self.nodes.items():
            d = node.model_dump(mode="json")
            if d.get("resource_role") is None:
                d.pop("resource_role", None)
            if d.get("command_timeout_s") is None:
                d.pop("command_timeout_s", None)
            if d.get("dispatch_command") is None:
                d.pop("dispatch_command", None)
            if "retry" in d and d["retry"] is None:
                d.pop("retry")
            if "workspace" in d and d["workspace"] is None:
                d.pop("workspace")
            for field_name in ("reads", "writes"):
                value = d.get(field_name)
                if isinstance(value, list):
                    d[field_name] = sorted(value)
            d["_type"] = type(node).__name__
            nodes_out[nid] = d

        edges_out = [e.model_dump(mode="json") for e in self.edges]

        result: dict[str, Any] = {
            "name": self.name,
            "nodes": nodes_out,
            "edges": edges_out,
            "start_node": self.start_node,
            "terminal": self.terminal,
        }
        if self.task is not None:
            result["task"] = self.task
        if self.knob_values:
            result["knob_values"] = dict(self.knob_values)
        if self.knob_bounds:
            result["knob_bounds"] = {k: list(v) for k, v in self.knob_bounds.items()}
        if self.knob_expandable:
            result["knob_expandable"] = dict(self.knob_expandable)
        if self.knob_specs:
            result["knob_specs"] = {k: dict(v) for k, v in self.knob_specs.items()}
        if self.declared_capabilities:
            result["declared_capabilities"] = sorted(self.declared_capabilities)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Workflow:
        """Reconstruct a Workflow from a dict produced by ``to_dict``."""
        _NODE_TYPE_MAP: dict[str, type[Node]] = {
            "AgentNode": AgentNode,
            "FnNode": FnNode,
            "GateNode": GateNode,
            "ForkNode": ForkNode,
            "JoinNode": JoinNode,
            "DataNode": DataNode,
        }
        _SET_FIELDS = {"reads", "writes"}

        nodes: dict[str, NodeType] = {}
        for nid, node_data in data["nodes"].items():
            node_data = dict(node_data)
            type_name = node_data.pop("_type", "FnNode")
            node_cls = _NODE_TYPE_MAP.get(type_name)
            if node_cls is None:
                raise ValueError(f"Unknown node type: {type_name}")
            for fld in _SET_FIELDS:
                if fld in node_data and isinstance(node_data[fld], list):
                    node_data[fld] = set(node_data[fld])
            nodes[nid] = node_cls.model_validate(node_data, strict=False)  # type: ignore[assignment]

        edges = [Edge.model_validate(e, strict=False) for e in data["edges"]]

        return cls(
            name=data["name"],
            nodes=nodes,
            edges=edges,
            start_node=data["start_node"],
            terminal=data.get("terminal", False),
            task=data.get("task"),
            knob_values=data.get("knob_values", {}),
            knob_bounds={k: list(v) for k, v in data.get("knob_bounds", {}).items()},
            knob_expandable=data.get("knob_expandable", {}),
            knob_specs=data.get("knob_specs", {}),
            declared_capabilities=frozenset(data.get("declared_capabilities", [])),
        )
