"""Smoke-predicate tests for lowering: the two-factor discriminator and
byte-identical graph.json snapshots for every shipped mode.

These tests assert that the ``_is_smoke_node`` predicate in ``lowering.py``
classifies exactly the same nodes as the old string-match special cases, and
that the lowered graph.json is byte-identical for every shipped mode at both
default and non-default ``smoke_timeout_s``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from refactory_dsh import lowering, registry
from refactory_dsh.nn import Graph, Node


# ── smoke predicate unit tests ─────────────────────────────────────────────


def test_smoke_predicate_is_sole_discriminator():
    """The predicate is the only path to the smoke envelope: removing it removes
    the timeout, and a product-string match never fires on the test node.
    """
    graph = Graph()
    graph.add(Node(
        "smoke", "gate", evaluator="test_ops:test_gate_smoke",
        knobs=("smoke_timeout_s",),
        produces=("smoke_score", "smoke_ok"),
        max_iterations=4,
    ))
    state = {
        "version": 1,
        "graph": graph.to_state(),
        "knobs": {"smoke_timeout_s": {"value": 3600, "spec": {"default": 3600}}},
        "ports": [],
        "memory": [],
    }
    from refactory_dsh.sandbox import COMMAND_BACKSTOP_S
    from refactory_dsh.model import NODE_DISPATCH_OVERHEAD_S

    # With the predicate active, the gate gets the smoke envelope (backstop-based).
    workflow = lowering.to_workflow(state, name="test_mode")
    expected = COMMAND_BACKSTOP_S + NODE_DISPATCH_OVERHEAD_S
    assert workflow.nodes["smoke"].command_timeout_s == expected, (
        f"expected {expected}, got {workflow.nodes['smoke'].command_timeout_s}"
    )

    # With the predicate disabled, the gate does NOT get the smoke envelope.
    with patch.object(lowering, '_is_smoke_node', return_value=False):
        workflow2 = lowering.to_workflow(state, name="test_mode")
    assert workflow2.nodes["smoke"].command_timeout_s is None, (
        "disabling _is_smoke_node must remove the smoke envelope"
    )


def test_smoke_predicate_excludes_prepare():
    """prepare has smoke_timeout_s in knobs but does NOT produce smoke_score/smoke_ok."""
    prepare_node = type('FakeNode', (), {
        'produces': ("name", "brief", "extends", "revision_of"),
        'knobs': ("name", "brief", "task_ref", "single_shot", "data_node",
                  "revision_of", "requires_input", "smoke_timeout_s"),
    })()
    assert not lowering._is_smoke_node(prepare_node)

    smoke_node = type('FakeNode', (), {
        'produces': ("smoke_score", "smoke_ok", "smoke_model"),
        'knobs': ("allow_regression", "sample_knobs", "smoke_timeout_s"),
    })()
    assert lowering._is_smoke_node(smoke_node)


def test_smoke_predicate_op_path():
    """An op (not gate) with the smoke predicate's properties gets the op smoke envelope."""
    graph = Graph()
    graph.add(Node(
        "smoke_op", "op", call="test_ops:test_smoke",
        knobs=("smoke_timeout_s",),
        produces=("smoke_score", "smoke_ok"),
    ))
    state = {
        "version": 1,
        "graph": graph.to_state(),
        "knobs": {"smoke_timeout_s": {"value": 3600, "spec": {"default": 3600}}},
        "ports": [],
        "memory": [],
    }
    from refactory_dsh.sandbox import COMMAND_BACKSTOP_S
    from refactory_dsh.model import NODE_DISPATCH_OVERHEAD_S

    workflow = lowering.to_workflow(state, name="test_mode")
    expected = COMMAND_BACKSTOP_S + NODE_DISPATCH_OVERHEAD_S
    assert workflow.nodes["smoke_op"].command_timeout_s == expected


def test_smoke_timeout_s_out_of_range_is_refused():
    """Out of range smoke_timeout_s is refused by lowering."""
    graph = Graph()
    graph.add(Node(
        "smoke", "gate", evaluator="test_ops:test_gate_smoke",
        knobs=("smoke_timeout_s",),
        produces=("smoke_score", "smoke_ok"),
        max_iterations=4,
    ))
    for bad_value in (0, 100000):
        state = {
            "version": 1,
            "graph": graph.to_state(),
            "knobs": {"smoke_timeout_s": {"value": bad_value, "spec": {"default": 3600}}},
            "ports": [],
            "memory": [],
        }
        with pytest.raises(lowering.LoweringError, match="smoke_timeout_s must be between 1 and 86400"):
            lowering.to_workflow(state, name="test_mode")


# ── byte-identical graph.json snapshots ────────────────────────────────────

SNAPSHOT_DIR = Path(__file__).resolve().parent / "lowering_snapshots"


def _mode_classes() -> list[tuple[str, type]]:
    """All shipped mode classes: discovered + the product mode classes."""
    from refactory_dsh.design import DesignMode
    from refactory_dsh.fast import DesignFastMode, CreateFastMode, EditFastMode
    from refactory_dsh.create import CreateMode
    from refactory_dsh.edit import EditMode

    # Discovered modes include fix_pack-make_pass
    discovered = registry.discover()
    results: dict[str, type] = {}
    for pack_name, pack in discovered.items():
        for mode_cls in pack.modes:
            identifier = f"{pack_name}/{mode_cls.name}"
            key = identifier.replace("/", "-")
            results[key] = mode_cls

    # Add the product mode classes for default_pack explicitly
    # (they are already discovered as default_pack/* but we ensure them)
    for cls, name in [
        (DesignMode, "default_pack-design"),
        (DesignFastMode, "default_pack-design_fast"),
        (CreateMode, "default_pack-create"),
        (CreateFastMode, "default_pack-create_fast"),
        (EditMode, "default_pack-edit"),
        (EditFastMode, "default_pack-edit_fast"),
    ]:
        results[name] = cls

    return sorted(results.items())


_SMOKE_MODES = frozenset({
    "default_pack-create", "default_pack-edit",
    "default_pack-create_fast", "default_pack-edit_fast",
})


def _lower_mode_bytes(mode_cls: type, *, name: str, smoke_timeout_s: int | None = None) -> bytes:
    """Lower a mode and return the graph.json bytes as write_mode_graph writes them.

    A trailing newline is appended when absent so the bytes match the
    committed snapshot files, which follow the text-file convention.
    """
    import tempfile
    mode = mode_cls()
    state = mode.state_dict()
    if smoke_timeout_s is not None:
        knobs = dict(state.get("knobs") or {})
        if "smoke_timeout_s" in knobs:
            entry = dict(knobs["smoke_timeout_s"])
            entry["value"] = smoke_timeout_s
            knobs["smoke_timeout_s"] = entry
            state["knobs"] = knobs
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _workflow = lowering.write_mode_graph(state, Path(tmpdir), name=name)
        data = path.read_bytes()
        if not data.endswith(b'\n'):
            data += b'\n'
        return data


def _snapshot_path(key: str, suffix: str = "") -> Path:
    """Path to a snapshot file."""
    return SNAPSHOT_DIR / f"{key}{suffix}.graph.json"


def regenerate_snapshots() -> None:
    """Regenerate all snapshot files from the current code.

    Run this once after a deliberate lowering change.  The files follow the
    text-file convention (trailing newline), matching ``_lower_mode_bytes``::

        ../.venv/bin/python -c "from tests.test_lowering_smoke import regenerate_snapshots; regenerate_snapshots()"
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for key, mode_cls in _mode_classes():
        data = _lower_mode_bytes(mode_cls, name=key)
        _snapshot_path(key).write_bytes(data)
        if key in _SMOKE_MODES:
            data1800 = _lower_mode_bytes(mode_cls, name=key, smoke_timeout_s=1800)
            _snapshot_path(key, "_smoke1800").write_bytes(data1800)
    print(f"wrote {len(list(SNAPSHOT_DIR.iterdir()))} snapshots to {SNAPSHOT_DIR}")


@pytest.fixture(scope="session")
def _ensure_snapshots():
    """Verify committed snapshot files exist; never regenerate inside the test."""
    if not SNAPSHOT_DIR.exists() or not any(SNAPSHOT_DIR.iterdir()):
        pytest.fail(
            f"Snapshot directory is missing or empty at {SNAPSHOT_DIR}.\n"
            f"Regenerate with:\n"
            f'  ../.venv/bin/python -c "from tests.test_lowering_smoke import regenerate_snapshots; regenerate_snapshots()"'
        )


@pytest.mark.usefixtures("_ensure_snapshots")
def test_lowered_graph_byte_identical_all_modes():
    """Every shipped mode lowers to a byte-identical graph.json against the snapshot."""
    failures: list[str] = []
    for key, mode_cls in _mode_classes():
        snap = _snapshot_path(key)
        if not snap.exists():
            failures.append(f"{key}: snapshot missing at {snap}")
            continue
        current = _lower_mode_bytes(mode_cls, name=key)
        expected = snap.read_bytes()
        if current != expected:
            failures.append(f"{key}: graph.json differs from snapshot ({len(current)} vs {len(expected)} bytes)")
    assert not failures, "byte-identity failures:\n" + "\n".join(f"  {f}" for f in failures)


@pytest.mark.usefixtures("_ensure_snapshots")
def test_lowered_graph_byte_identical_smoke_modes_at_1800():
    """Smoke modes at smoke_timeout_s=1800 lower to a byte-identical graph.json."""
    failures: list[str] = []
    for key, mode_cls in _mode_classes():
        if key not in _SMOKE_MODES:
            continue
        snap = _snapshot_path(key, "_smoke1800")
        if not snap.exists():
            failures.append(f"{key}_smoke1800: snapshot missing at {snap}")
            continue
        current = _lower_mode_bytes(mode_cls, name=key, smoke_timeout_s=1800)
        expected = snap.read_bytes()
        if current != expected:
            failures.append(f"{key}_smoke1800: graph.json differs from snapshot")
    assert not failures, "byte-identity failures at smoke_timeout_s=1800:\n" + "\n".join(f"  {f}" for f in failures)
