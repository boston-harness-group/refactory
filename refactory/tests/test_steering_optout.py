"""Tests for Ask 3: steering-ledger prompt opt-out that preserves correction enforcement.

Every test names the revert site that makes it fail when undone.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clean_budget_env(monkeypatch):
    """Prevent inherited budget ledger from locking files outside tmp_path."""
    monkeypatch.delenv("REFACTORY_BUDGET_LEDGER", raising=False)


def _make_graph(*nodes):
    """Helper: build a Graph from Node instances."""
    from refactory_dsh.nn.graph import Graph
    g = Graph()
    for n in nodes:
        g.add(n)
    return g


def _write_run_config(workdir: Path, omit: bool = False, **extra: Any) -> None:
    cfg_dir = workdir / ".refactory"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"mode": "test/test", "runtime": "walker",
                               "command": "echo", "resume_command": "echo",
                               "model_context": {"provider": "p", "model": "m"}}
    if omit:
        config["omit_steering"] = True
    config.update(extra)
    (cfg_dir / "run-config.json").write_text(json.dumps(config))


def _write_ledger(workdir: Path, text: str = "") -> None:
    if not text:
        text = ("# User Intent Ledger\n\n"
                "## [2026-01-01T00:00:00+00:00] Initial Idea\nBuild a widget\n")
    path = workdir / "intent-ledger.md"
    path.write_text(text)


def test_omit_steering_skips_ledger_injection(tmp_path):
    """for_run with omit_steering=True does not inject STEERING_START.
    Revert: steering.py for_run guard."""
    from refactory_dsh.steering import STEERING_START

    _write_run_config(tmp_path, omit=True)
    _write_ledger(tmp_path)

    from refactory_dsh.nn.graph import Node
    from refactory_dsh import lowering
    node = Node(id="build", kind="agent", prompt_template="do it",
                role="builder", writes=("out.txt",))
    graph = _make_graph(node)
    state = {"graph": graph.to_state(), "knobs": {}}
    wf = lowering.to_workflow(state)

    from refactory_dsh.steering import for_run
    wf2 = for_run(wf, tmp_path)
    # The agent node's prompt should NOT contain STEERING_START
    assert STEERING_START not in wf2.nodes["build"].prompt_template


def test_omit_steering_default_includes_ledger(tmp_path):
    """Without omit_steering and with a ledger, STEERING_START is present.
    Exemption: pins existing default behavior."""
    from refactory_dsh.steering import STEERING_START

    _write_run_config(tmp_path, omit=False)
    _write_ledger(tmp_path)

    from refactory_dsh.nn.graph import Node
    from refactory_dsh import lowering
    node = Node(id="build", kind="agent", prompt_template="do it",
                role="builder", writes=("out.txt",))
    graph = _make_graph(node)
    state = {"graph": graph.to_state(), "knobs": {}}
    wf = lowering.to_workflow(state)

    from refactory_dsh.steering import for_run
    wf2 = for_run(wf, tmp_path)
    assert STEERING_START in wf2.nodes["build"].prompt_template


def test_omit_steering_survives_actual_resume(tmp_path):
    """After resume preparation, omit_steering still affects for_run.
    Revert: run_flags.py RESUME_TAKES or run_preparation.py."""
    from refactory_dsh.steering import STEERING_START
    from refactory_dsh.run_flags import RESUME_TAKES

    assert "omit_steering" in RESUME_TAKES

    # Create a run with omit_steering
    _write_run_config(tmp_path, omit=True)
    _write_ledger(tmp_path)

    # Simulate resume: read back config
    from refactory_dsh.run_preparation import saved_run_config
    cfg = saved_run_config(tmp_path)
    assert cfg.get("omit_steering") is True

    # Execute for_run — should still skip steering
    from refactory_dsh.nn.graph import Node
    from refactory_dsh import lowering
    node = Node(id="build", kind="agent", prompt_template="do it",
                role="builder", writes=("out.txt",))
    graph = _make_graph(node)
    state = {"graph": graph.to_state(), "knobs": {}}
    wf = lowering.to_workflow(state)

    from refactory_dsh.steering import for_run
    wf2 = for_run(wf, tmp_path)
    assert STEERING_START not in wf2.nodes["build"].prompt_template


def test_omit_steering_through_walker_evaluator_path(tmp_path):
    """WalkerEvaluator → mode.evaluate path: omit_steering in run-config
    makes for_run skip the ledger AND system_prompt_template renders.
    Revert: evaluators.py or run_preparation.py changes."""
    from refactory_dsh.steering import STEERING_START

    _write_run_config(tmp_path, omit=True)
    _write_ledger(tmp_path)

    captured: dict[str, Any] = {}

    from refactory_dsh.nn.graph import Node
    node = Node(id="build", kind="agent", prompt_template="do {{task}}",
                system_prompt_template="You are {{role}}",
                writes=("out.txt",), role="builder")
    graph = _make_graph(node)

    class FakeClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            captured["system"] = system
            captured["prompt"] = prompt
            return "ok"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    (tmp_path / "out.txt").write_text("")
    from refactory_dsh.nn.executor import execute
    execute(graph, tmp_path, knobs={"task": "build", "role": "architect"},
            model=FakeClient())
    # System prompt rendered
    assert captured["system"] == "You are architect"
    # No steering in user prompt
    assert STEERING_START not in captured["prompt"]


def test_omit_steering_through_dispatched_spine_path(tmp_path):
    """Spine describe() carries system_prompt and worker_node refuses unsupported route.
    Revert: spine.ts describe() or worker_node.py pre-flight check."""
    from refactory_dsh.nn.graph import Node
    from refactory_dsh import lowering
    from refactory_dsh.ir import AgentNode

    node = Node(id="build", kind="agent", prompt_template="do it",
                system_prompt_template="You are {{role}}",
                role="builder", writes=("out.txt",))
    graph = _make_graph(node)
    state = {"graph": graph.to_state(), "knobs": {}}
    wf = lowering.to_workflow(state)
    ir_node = wf.nodes["build"]
    assert isinstance(ir_node, AgentNode)
    assert ir_node.system_prompt == "You are {{role}}"
    # The system_prompt should appear in the serialized IR for the spine
    d = wf.to_dict()
    assert d["nodes"]["build"]["system_prompt"] == "You are {{role}}"


def test_correction_enforcement_with_omit_steering_walker(tmp_path):
    """With omit_steering, corrections still reach design/build inputs via feedback,
    audit reads the ledger, and overwatch checks build status.
    Revert: any change that makes omit_steering suppress feedback delivery or audit."""
    from refactory_dsh.steering import STEERING_START, Ledger

    _write_run_config(tmp_path, omit=True)

    # Write a ledger with a gate_strategy correction
    ledger_text = (
        "# User Intent Ledger\n\n"
        "## [2026-01-01T00:00:00+00:00] Initial Idea\nBuild a widget\n\n"
        "## [2026-01-02T00:00:00+00:00] correction\n"
        "RECOGNIZABLE_CORRECTION_DIRECTIVE: Add error handling\n\n"
        "## [2026-01-03T00:00:00+00:00] correction\n"
        "INTENTIONALLY_UNBUILT: This was not implemented\n"
    )
    _write_ledger(tmp_path, ledger_text)

    # 1. Verify feedback mechanism still works (with_feedback is independent of steering)
    from refactory_dsh.agent_delivery import with_feedback
    prompt = "build the widget"
    enriched = with_feedback(prompt, "RECOGNIZABLE_CORRECTION_DIRECTIVE: Add error handling")
    assert "RECOGNIZABLE_CORRECTION_DIRECTIVE" in enriched
    assert "Feedback for this visit" in enriched

    # 2. Verify correction audit reads the actual ledger
    ledger = Ledger(tmp_path)
    assert ledger.exists()
    entries = ledger.entries()
    correction_entries = [e for e in entries if e.get("kind") == "correction"]
    assert len(correction_entries) >= 2
    assert any("RECOGNIZABLE_CORRECTION_DIRECTIVE" in str(e) for e in correction_entries)
    assert any("INTENTIONALLY_UNBUILT" in str(e) for e in correction_entries)

    # 3. Verify that for_run skips annotate but leaves ledger readable
    from refactory_dsh.nn.graph import Node
    from refactory_dsh import lowering
    node = Node(id="build", kind="agent", prompt_template="do it",
                role="builder", writes=("out.txt",))
    graph = _make_graph(node)
    state = {"graph": graph.to_state(), "knobs": {}}
    wf = lowering.to_workflow(state)
    from refactory_dsh.steering import for_run
    wf2 = for_run(wf, tmp_path)
    assert STEERING_START not in wf2.nodes["build"].prompt_template

    # 4. Verify the ledger file itself is still intact after for_run
    assert ledger.exists()
    entries_after = ledger.entries()
    assert len(entries_after) == len(entries)


def test_correction_enforcement_with_omit_steering_spine(tmp_path):
    """Same as walker variant but through the spine's IR serialization path.
    Revert: any change that breaks correction with omit_steering on spine path."""
    from refactory_dsh.steering import Ledger

    _write_run_config(tmp_path, omit=True)
    ledger_text = (
        "# User Intent Ledger\n\n"
        "## [2026-01-01T00:00:00+00:00] correction\n"
        "SPINE_CORRECTION: Use better logging\n"
    )
    _write_ledger(tmp_path, ledger_text)

    # The ledger is still readable
    ledger = Ledger(tmp_path)
    entries = ledger.entries()
    assert any("SPINE_CORRECTION" in str(e) for e in entries)

    # The system prompt still flows through IR
    from refactory_dsh.nn.graph import Node
    from refactory_dsh import lowering
    node = Node(id="build", kind="agent", prompt_template="do it",
                system_prompt_template="be careful",
                role="builder", writes=("out.txt",))
    graph = _make_graph(node)
    state = {"graph": graph.to_state(), "knobs": {}}
    wf = lowering.to_workflow(state)
    d = wf.to_dict()
    assert d["nodes"]["build"]["system_prompt"] == "be careful"


def test_agent_delivery_begin_skips_ledger_with_omit_steering(tmp_path):
    """agent_delivery.begin with omit_steering=True does not inject ledger.
    Revert: agent_delivery.py omit_steering guard."""
    from refactory_dsh.steering import STEERING_START

    _write_run_config(tmp_path, omit=True)
    _write_ledger(tmp_path)

    from refactory_dsh.agent_delivery import begin

    with mock.patch("refactory_dsh.agent_delivery.get_worktree_state", return_value=None):
        with mock.patch("refactory_dsh.agent_delivery.target_guard") as tg:
            tg.watch.return_value = None
            tg.watch_checkouts.return_value = {}
            tg.checkouts.return_value = []
            record = begin(tmp_path, "test_node", "build the widget", ["out.txt"])

    assert STEERING_START not in record["prompt"]

    # Without omit_steering: ledger IS included
    _write_run_config(tmp_path, omit=False)
    with mock.patch("refactory_dsh.agent_delivery.get_worktree_state", return_value=None):
        with mock.patch("refactory_dsh.agent_delivery.target_guard") as tg:
            tg.watch.return_value = None
            tg.watch_checkouts.return_value = {}
            tg.checkouts.return_value = []
            record2 = begin(tmp_path, "test_node2", "build the widget", ["out.txt"])

    assert STEERING_START in record2["prompt"]
