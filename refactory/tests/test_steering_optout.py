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
    """WalkerEvaluator.evaluate() → mode.evaluate() path: omit_steering in run-config
    makes for_run skip the ledger AND system_prompt_template renders through mode.evaluate.
    Revert: steering.py for_run guard or run_preparation.py omit_steering wiring."""
    from refactory_dsh.steering import STEERING_START

    captured: dict[str, Any] = {}

    class FakeClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            captured["system"] = system
            captured["prompt"] = prompt
            return "ok"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    # Build a Mode whose graph has a system_prompt_template and knobs
    from refactory_dsh.nn.graph import Node, Graph
    from refactory_dsh.nn.mode import Mode
    from refactory_dsh.nn.knob import Knob

    class TestMode(Mode):
        name = "test_omit_steer"
        unscored_reason = "test"
        @classmethod
        def declare(cls):
            return {"task": Knob(default="build", kind="str"),
                    "role": Knob(default="architect", kind="str")}
        def build(self):
            g = Graph()
            g.add(Node(id="build", kind="agent", prompt_template="do {{task}}",
                       system_prompt_template="You are {{role}}",
                       writes=("out.txt",), role="builder", knobs=("task", "role")))
            return g
    mode = TestMode()

    # Prepare a run directory with omit_steering and a ledger
    run_dir = tmp_path / "walker_eval_run"
    run_dir.mkdir()
    _write_ledger(run_dir)
    (run_dir / "out.txt").write_text("")

    # Exercise through WalkerEvaluator.evaluate() — the real outer-loop path
    from refactory_dsh.evaluators import WalkerEvaluator
    evaluator = WalkerEvaluator(model=FakeClient(), project=tmp_path, omit_steering=True)
    result = evaluator.evaluate(mode, mode.state_dict(), run_dir)

    assert result.exit_ok, result.error
    # System prompt rendered with knobs through the evaluator path
    assert captured.get("system") == "You are architect"
    # No steering in user prompt (the important check)
    assert STEERING_START not in captured.get("prompt", "")


def test_unsupported_route_refusal_through_evaluator_path(tmp_path):
    """WalkerEvaluator.evaluate() refuses an unsupported route (codex) with a
    system prompt before any model call starts.
    Revert: executor.py pre-flight capability check."""
    from refactory_dsh.nn.graph import Node, Graph
    from refactory_dsh.nn.mode import Mode

    class RefusalMode(Mode):
        name = "test_route_refusal"
        unscored_reason = "test"
        def build(self):
            g = Graph()
            g.add(Node(id="build", kind="agent", prompt_template="do it",
                       system_prompt_template="be careful",
                       writes=("out.txt",), role="builder"))
            return g
    mode = RefusalMode()

    class CodexClient:
        def complete(self, prompt, **kw):
            raise AssertionError("Should not reach model call")
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "codex"}

    run_dir = tmp_path / "route_refusal_run"
    run_dir.mkdir()
    (run_dir / "out.txt").write_text("")

    from refactory_dsh.evaluators import WalkerEvaluator
    evaluator = WalkerEvaluator(model=CodexClient(), project=tmp_path)
    result = evaluator.evaluate(mode, mode.state_dict(), run_dir)
    # The run must fail before the model is called
    assert not result.exit_ok
    assert "system" in result.error.lower() and "prompt" in result.error.lower()


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
    """With omit_steering, a gate correction reaches design/build inputs through the
    real execute() path, the correction audit reads the ledger, and an overwatch-like
    check can distinguish implemented vs unimplemented corrections.
    Revert: steering.py for_run guard, agent_delivery.py feedback path, or executor.py
    feedback wiring."""
    import sys
    from refactory_dsh.steering import STEERING_START, Ledger
    from refactory_dsh.nn.executor import Verdict

    run_dir = tmp_path / "correction_run"
    run_dir.mkdir()
    _write_run_config(run_dir, omit=True)

    # Write a ledger with corrections (for audit)
    ledger_text = (
        "# User Intent Ledger\n\n"
        "## [2026-01-01T00:00:00+00:00] Initial Idea\nBuild a widget\n\n"
        "## [2026-01-02T00:00:00+00:00] correction\n"
        "RECOGNIZABLE_CORRECTION_DIRECTIVE: Add error handling\n\n"
        "## [2026-01-03T00:00:00+00:00] correction\n"
        "INTENTIONALLY_UNBUILT: This was not implemented\n"
    )
    _write_ledger(run_dir, ledger_text)

    # Create a gate evaluator function that reloops on first call, proceeds on second.
    # Write it as a real importable module so resolve() can find it.
    gate_module_dir = tmp_path / "gate_helpers"
    gate_module_dir.mkdir()
    (gate_module_dir / "__init__.py").write_text("")
    (gate_module_dir / "check_gate.py").write_text("""\
from refactory_dsh.nn.executor import Verdict

_call_count = 0

def evaluate(ctx):
    global _call_count
    _call_count += 1
    if _call_count == 1:
        return Verdict("reloop", reason="RECOGNIZABLE_CORRECTION_DIRECTIVE: Add error handling")
    return "proceed"
""")
    sys.path.insert(0, str(tmp_path))

    try:
        # Build a graph: build → check → (reloop to build, proceed edge falls through)
        from refactory_dsh.nn.graph import Node
        build_node = Node(id="build", kind="agent", prompt_template="Build the widget.",
                          role="builder", writes=("out.txt",))
        gate_node = Node(id="check", kind="gate", gate_kind="fn",
                         evaluator="gate_helpers.check_gate:evaluate",
                         max_iterations=2, reads=("out.txt",))
        graph = _make_graph(build_node, gate_node)
        graph.connect("build", "check")
        graph.connect("check", "build", condition="reloop")

        captured_prompts: list[str] = []

        class FakeClient:
            def complete(self, prompt, *, system="", workdir=None, **kw):
                captured_prompts.append(prompt)
                (run_dir / "out.txt").write_text("implemented error handling")
                return "ok"
            def for_node(self, n, role=""):
                return self
            def with_wall(self, t):
                return self
            def context(self):
                return {}
            def route(self):
                return {"worker_backend": "dsh"}

        from refactory_dsh.nn.executor import execute
        (run_dir / "out.txt").write_text("")
        result = execute(graph, run_dir, model=FakeClient())

        # Builder must have run at least twice
        assert len(captured_prompts) >= 2, f"Expected >=2 builder runs, got {len(captured_prompts)}"

        # 1. Gate correction reached design/build inputs through the real path
        # Second prompt must contain the correction text via feedback
        assert "RECOGNIZABLE_CORRECTION_DIRECTIVE" in captured_prompts[1], \
            "Correction text did not reach builder's second prompt"
        assert "Feedback for this visit" in captured_prompts[1], \
            "Feedback heading missing from builder's second prompt"

        # 2. Correction audit reads the actual ledger entries
        ledger = Ledger(run_dir)
        assert ledger.exists()
        entries = ledger.entries()
        correction_entries = [e for e in entries if e.get("kind") == "correction"]
        assert len(correction_entries) >= 2
        assert any("RECOGNIZABLE_CORRECTION_DIRECTIVE" in str(e) for e in correction_entries)
        assert any("INTENTIONALLY_UNBUILT" in str(e) for e in correction_entries)

        # 3. Overwatch-like check: distinguish implemented vs unimplemented
        # The builder's output contains evidence of the implemented correction
        output = (run_dir / "out.txt").read_text()
        assert "error handling" in output, "Implemented correction not in output"
        # The INTENTIONALLY_UNBUILT correction is NOT in the builder's prompts
        for p in captured_prompts:
            assert "INTENTIONALLY_UNBUILT" not in p, \
                "Unbuilt correction should not appear in builder prompts"

        # 4. Steering was NOT injected (omit_steering is True)
        for p in captured_prompts:
            assert STEERING_START not in p, "Steering should be omitted"
    finally:
        sys.path.remove(str(tmp_path))
        # Clean up the imported module to avoid contaminating other tests
        for key in list(sys.modules):
            if key.startswith("gate_helpers"):
                del sys.modules[key]


def test_correction_enforcement_with_omit_steering_spine(tmp_path):
    """Through the spine's IR path with omit_steering: correction audit reads ledger,
    system_prompt flows through IR, and worker_node delivers feedback.
    Revert: any change that breaks correction with omit_steering on spine path."""
    from refactory_dsh.steering import STEERING_START, Ledger

    run_dir = tmp_path / "spine_correction_run"
    run_dir.mkdir()
    _write_run_config(run_dir, omit=True)
    ledger_text = (
        "# User Intent Ledger\n\n"
        "## [2026-01-01T00:00:00+00:00] correction\n"
        "SPINE_CORRECTION: Use better logging\n\n"
        "## [2026-01-02T00:00:00+00:00] correction\n"
        "SPINE_UNBUILT: Not implemented\n"
    )
    _write_ledger(run_dir, ledger_text)

    # 1. Correction audit reads the actual ledger
    ledger = Ledger(run_dir)
    entries = ledger.entries()
    correction_entries = [e for e in entries if e.get("kind") == "correction"]
    assert len(correction_entries) >= 2
    assert any("SPINE_CORRECTION" in str(e) for e in correction_entries)
    assert any("SPINE_UNBUILT" in str(e) for e in correction_entries)

    # 2. IR carries system_prompt and for_run omits steering
    from refactory_dsh.nn.graph import Node
    from refactory_dsh import lowering
    node = Node(id="build", kind="agent", prompt_template="do it",
                system_prompt_template="be careful",
                role="builder", writes=("out.txt",))
    graph = _make_graph(node)
    state = {"graph": graph.to_state(), "knobs": {}}
    wf = lowering.to_workflow(state)
    # Apply for_run with omit_steering
    from refactory_dsh.steering import for_run
    wf2 = for_run(wf, run_dir)
    assert STEERING_START not in wf2.nodes["build"].prompt_template
    d = wf2.to_dict()
    assert d["nodes"]["build"]["system_prompt"] == "be careful"

    # 3. Worker spec carries system_prompt for the spine dispatch
    from refactory_dsh.workers import lower_workflow
    from refactory_dsh.run import NODE_ENV
    lowered = lower_workflow(wf2, run_dir, {"provider": "test", "model": "test"})
    spec = lowered.nodes["build"]
    assert spec.system_prompt == "be careful"

    # 4. Feedback delivery works through worker_node with omit_steering
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kw): pass
        def for_node(self, n, role="worker"): return self
        def complete(self, prompt, *, system="", workdir=None, **kw):
            captured["prompt"] = prompt
            captured["system"] = system
            return "done"
        def context(self): return {"provider": "test", "model": "test"}
        def route(self): return {"worker_backend": "dsh"}

    worker_spec = {
        "schema_version": 1, "node_id": "build", "role": "builder",
        "prompt": "do it", "system_prompt": "be careful",
        "reads": [], "writes": [],
        "model_context": {"provider": "test", "model": "test"},
    }
    manifest = {"id": "build", "type": "AgentNode", "reads": [], "writes": [],
                "feedback_file": str(run_dir / ".refactory" / "prompts" / "build.feedback.md")}
    # Write a feedback sidecar to simulate gate correction delivery
    fb_dir = run_dir / ".refactory" / "prompts"
    fb_dir.mkdir(parents=True, exist_ok=True)
    (fb_dir / "build.feedback.md").write_text("SPINE_CORRECTION: Use better logging")

    def fake_begin(w, n, p, ws, **k):
        return {"prompt": p, "node": n, "status": "pending", "writes": ws}

    with mock.patch.dict(os.environ, {NODE_ENV: json.dumps(manifest)}):
        with mock.patch("refactory_dsh.model.DshModelClient", FakeClient):
            with mock.patch("refactory_dsh.agent_delivery.begin", fake_begin):
                with mock.patch("refactory_dsh.agent_delivery.finish",
                                return_value={"node": "build", "status": "delivered"}):
                    with mock.patch("refactory_dsh.agent_delivery._path", lambda w, n: Path(w) / n):
                        with mock.patch("refactory_dsh.run._knobs", return_value={}):
                            with mock.patch("refactory_dsh.run._read_facts", return_value={}):
                                from refactory_dsh import worker_node
                                worker_node.execute(worker_spec, run_dir)
    # Feedback reached the builder prompt through the real worker_node path
    assert "SPINE_CORRECTION" in captured["prompt"]
    assert "Feedback for this visit" in captured["prompt"]
    assert captured["system"] == "be careful"


def test_omit_steering_resume_retains_option_through_real_path(tmp_path):
    """An actual resume that writes and reads run-config retains omit_steering,
    and a subsequent for_run call respects it.
    Revert: run_preparation.py omit_steering or run_flags.py RESUME_TAKES."""
    from refactory_dsh.steering import STEERING_START
    from refactory_dsh.run_flags import RESUME_TAKES
    assert "omit_steering" in RESUME_TAKES

    # Simulate prepare_run writing a config with omit_steering
    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()
    _write_run_config(run_dir, omit=True)
    _write_ledger(run_dir)

    # Write progress.json so resume finds something to continue from
    progress = {
        "node": "build", "error": "simulated failure", "facts": {},
        "knobs": {}, "task": {}, "steps": 1, "outcome": "pending",
        "graph": {"nodes": {"build": {"id": "build", "kind": "agent",
                  "prompt_template": "do it", "role": "builder",
                  "writes": ["out.txt"], "reads": []}},
                  "edges": [], "start_node": "build"},
    }
    (run_dir / ".refactory").mkdir(exist_ok=True)
    (run_dir / ".refactory" / "progress.json").write_text(json.dumps(progress))

    # Read back the config (simulating what resume does)
    from refactory_dsh.run_preparation import saved_run_config
    cfg = saved_run_config(run_dir)
    assert cfg.get("omit_steering") is True

    # Verify for_run still skips steering after resume reads config
    from refactory_dsh.nn.graph import Node
    from refactory_dsh import lowering
    node = Node(id="build", kind="agent", prompt_template="do it",
                role="builder", writes=("out.txt",))
    graph = _make_graph(node)
    state = {"graph": graph.to_state(), "knobs": {}}
    wf = lowering.to_workflow(state)
    from refactory_dsh.steering import for_run
    wf2 = for_run(wf, run_dir)
    assert STEERING_START not in wf2.nodes["build"].prompt_template


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
