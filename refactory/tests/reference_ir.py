"""Snapshot-based IR field tracking.

Previously this module used the v1 subtree (remote-factory) as a live oracle
to verify that refactory-dsh's IR stayed parity with v1's ``primitives.py``.
With remote-factory deprecated, the oracle is replaced by frozen field-set
snapshots: if a field is added or removed from any IR class, the snapshot
test catches it.

``EXTENSION_FIELDS`` names the fields the local IR carries that v1 never had.
``strip_extension_keys`` removes them from a ``to_dict()`` result so round-trip
and lowering-snapshot tests compare only the shared vocabulary.
"""

from __future__ import annotations

from typing import Any

from refactory_dsh import ir

#: The names ``lowering`` binds to the IR.
IR_NAMES = (
    "AgentNode",
    "AgentRole",
    "DataNode",
    "Edge",
    "FnNode",
    "ForkNode",
    "GateNode",
    "JoinNode",
    "Workflow",
)

#: Fields the local IR declares that the original v1 primitives did not have.
EXTENSION_FIELDS: frozenset[str] = frozenset({
    "resource_role",
    "command_timeout_s",
    "dispatch_command",
    "retry",
    "optional_targets",
    "split",
    "subgraph_entry",
    "subgraph_exit",
    "system_prompt",
    "workspace",
})

#: Frozen expected field sets per IR class.  If a field is added or removed,
#: ``check_field_snapshot`` fails and the snapshot must be updated deliberately.
FIELD_SNAPSHOT: dict[str, frozenset[str]] = {
    "Node": frozenset({"id", "reads", "writes", "blocking", "resource_role"}),
    "AgentNode": frozenset({
        "id", "reads", "writes", "blocking", "resource_role",
        "role", "model", "prompt_template", "tools", "timeout",
        "max_iterations", "post_checks", "dispatch_command", "command_timeout_s",
        "retry", "workspace", "system_prompt",
    }),
    "FnNode": frozenset({
        "id", "reads", "writes", "blocking", "resource_role",
        "command", "callable_name", "notes", "command_timeout_s",
    }),
    "GateNode": frozenset({
        "id", "reads", "writes", "blocking", "resource_role",
        "evaluator_type", "evaluator_role", "evaluator_command",
        "command_timeout_s", "gate_prompt", "max_iterations",
    }),
    "ForkNode": frozenset({
        "id", "reads", "writes", "blocking", "resource_role",
        "targets", "optional_targets",
    }),
    "JoinNode": frozenset({
        "id", "reads", "writes", "blocking", "resource_role",
        "sources",
    }),
    "DataNode": frozenset({
        "id", "reads", "writes", "blocking", "resource_role",
        "task_ref", "source_path", "source_format", "inline_items",
        "subgraph_entry", "subgraph_exit", "parallelism", "split",
        "shuffle", "shuffle_seed", "limit", "max_items",
    }),
    "Edge": frozenset({"source", "target", "condition"}),
    "Workflow": frozenset({
        "name", "nodes", "edges", "start_node", "terminal", "task",
        "knob_values", "knob_bounds", "knob_expandable", "knob_specs",
        "declared_capabilities",
    }),
}


def strip_extension_keys(workflow_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove refactory's extension fields from a to_dict() result.

    Returns a shallow copy with the node dicts stripped of EXTENSION_FIELDS keys,
    so the result carries only the vocabulary the shared core defines.
    """
    out = dict(workflow_dict)
    nodes = {}
    for nid, node_data in out["nodes"].items():
        d = dict(node_data)
        for key in EXTENSION_FIELDS:
            d.pop(key, None)
        nodes[nid] = d
    out["nodes"] = nodes
    return out


def check_field_snapshot() -> None:
    """Assert every IR class has exactly the fields in FIELD_SNAPSHOT.

    Raises AssertionError with a detailed message if a field was added or
    removed from any IR class without updating the snapshot.
    """
    errors: list[str] = []
    for class_name, expected_fields in FIELD_SNAPSHOT.items():
        cls = getattr(ir, class_name)
        if not hasattr(cls, "model_fields"):
            continue
        actual_fields = set(cls.model_fields.keys())
        added = actual_fields - expected_fields
        removed = expected_fields - actual_fields
        if added:
            errors.append(
                f"{class_name} gained fields {sorted(added)} — update "
                f"FIELD_SNAPSHOT in reference_ir.py"
            )
        if removed:
            errors.append(
                f"{class_name} lost fields {sorted(removed)} — update "
                f"FIELD_SNAPSHOT in reference_ir.py"
            )
    if errors:
        raise AssertionError(
            "IR field snapshot is stale:\n  " + "\n  ".join(errors)
        )
