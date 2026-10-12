"""Independent source-derived oracle, deliberately outside production graph code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/create_fidelity/v1"
DIRECTORS = {"research_director", "strategy_director", "qa_director"}
KINDS = {"FnNode": "op", "Study": "op", "AgentNode": "agent", "GateNode": "gate",
         "ForkNode": "fork", "JoinNode": "join"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference() -> dict[str, Any]:
    return json.loads((FIXTURES / "reference.json").read_text())


def departures(graph: Any) -> list[dict[str, Any]]:
    """Compare original control flow and decision ownership, allowing only named mappings."""
    expected = reference()["graph"]
    rows = []

    def mismatch(requirement: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            rows.append({"requirement_id": requirement, "status": "failed",
                         "expected": wanted, "actual": actual})

    mismatch("G.start", graph.start, expected["start_node"])
    mismatch("G.stages", sorted(graph.nodes), sorted(expected["nodes"]))
    mismatch("G.edges", sorted((e.source, e.target, e.condition or "") for e in graph.edges),
             sorted((e["source"], e["target"], e.get("condition") or "") for e in expected["edges"]))
    for name, source in expected["nodes"].items():
        if name not in graph.nodes:
            continue
        node = graph.nodes[name]
        mismatch(f"G.{name}.kind", node.kind, "op" if name in DIRECTORS else KINDS[source["_type"]])
        mismatch(f"G.{name}.blocking", node.blocking, source["blocking"])
        if source["_type"] == "GateNode":
            mismatch(f"G.{name}.owner", node.gate_kind, source["evaluator_type"])
        if source.get("targets"):
            mismatch(f"G.{name}.targets", sorted(node.targets), sorted(source["targets"]))
        if source.get("sources"):
            mismatch(f"G.{name}.sources", sorted(node.sources), sorted(source["sources"]))
        if source.get("timeout"):
            mismatch(f"G.{name}.timeout", getattr(node, "timeout_s", None), source["timeout"])
        if source.get("role") and name not in DIRECTORS:
            mismatch(f"G.{name}.role", node.role, source["role"])
    return rows
