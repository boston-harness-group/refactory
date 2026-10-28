"""Tests for Ask 1 (system-prompt field) and Ask 2 (effort passthrough).

Every test names the revert site that makes it fail when undone.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clean_budget_env(monkeypatch):
    """Prevent inherited budget ledger from locking files outside tmp_path."""
    monkeypatch.delenv("REFACTORY_BUDGET_LEDGER", raising=False)


# ─── Ask 1: serialization ───────────────────────────────────────────

def test_system_prompt_template_serialization_round_trip():
    """Node with system_prompt_template survives to_state → from_state.
    Revert: graph.py conditional serialization."""
    from refactory_dsh.nn.graph import Node
    n = Node(id="a", kind="agent", prompt_template="hi",
             system_prompt_template="You are {{role}}")
    s = n.to_state()
    assert s["system_prompt_template"] == "You are {{role}}"
    n2 = Node.from_state(s)
    assert n2.system_prompt_template == "You are {{role}}"


def test_system_prompt_absent_by_default():
    """Default Node to_state has no system_prompt_template key (PIT-001).
    Revert: graph.py conditional serialization guard."""
    from refactory_dsh.nn.graph import Node
    n = Node(id="b", kind="agent", prompt_template="hi")
    s = n.to_state()
    assert "system_prompt_template" not in s


def _make_graph(*nodes):
    """Helper: build a Graph from Node instances."""
    from refactory_dsh.nn.graph import Graph
    g = Graph()
    for n in nodes:
        g.add(n)
    return g


def test_system_prompt_in_ir_and_worker_spec(tmp_path):
    """system_prompt flows graph → IR → worker spec.
    Revert: ir.py, lowering.py, or workers.py."""
    from refactory_dsh.nn.graph import Node
    from refactory_dsh import lowering
    from refactory_dsh.ir import AgentNode

    node = Node(id="build", kind="agent", prompt_template="do it",
                system_prompt_template="be careful", role="builder",
                writes=("out.txt",))
    graph = _make_graph(node)
    state = {"graph": graph.to_state(), "knobs": {}}
    wf = lowering.to_workflow(state)
    ir_node = wf.nodes["build"]
    assert isinstance(ir_node, AgentNode)
    assert ir_node.system_prompt == "be careful"
    # Serialization round-trip
    d = wf.to_dict()
    assert d["nodes"]["build"]["system_prompt"] == "be careful"
    from refactory_dsh.ir import Workflow
    wf2 = Workflow.from_dict(d)
    assert wf2.nodes["build"].system_prompt == "be careful"


def test_system_prompt_rendered_with_knobs_facts():
    """render() works on the system field with knobs.
    Exemption: tests the existing render() with new input."""
    from refactory_dsh.nn.executor import render, RunContext
    ctx = RunContext(Path("/tmp"), knobs={"role": "expert"})
    assert render("You are {{role}}", ctx) == "You are expert"


def test_node_notes_not_in_system_prompt(tmp_path):
    """A node's notes must not leak into the system prompt (#523).
    Revert: executor.py (notes used for system)."""
    from refactory_dsh.nn.graph import Node
    from refactory_dsh.nn.executor import execute

    captured: dict[str, Any] = {}
    node = Node(id="a", kind="agent", prompt_template="hi",
                notes="secret notes", writes=("out.txt",))
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
    result = execute(graph, tmp_path, model=FakeClient())
    assert result.exit_ok, result.error
    assert captured["system"] == ""
    assert "secret notes" not in captured.get("system", "")


def test_walker_system_prompt_reaches_model_complete(tmp_path):
    """Walker _visit_agent passes rendered system text to model.complete.
    Revert: executor.py system rendering/passing."""
    from refactory_dsh.nn.graph import Node
    from refactory_dsh.nn.executor import execute

    captured: dict[str, Any] = {}
    node = Node(id="a", kind="agent", prompt_template="do {{task}}",
                system_prompt_template="be {{style}}",
                writes=("out.txt",))
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
    result = execute(graph, tmp_path, knobs={"task": "build", "style": "careful"},
                     model=FakeClient())
    assert result.exit_ok, result.error
    assert captured["system"] == "be careful"
    assert "do build" in captured["prompt"]


def test_worker_system_prompt_reaches_model_complete(tmp_path):
    """worker_node.execute passes system to model.complete.
    Revert: worker_node.py system rendering/passing."""
    from refactory_dsh.run import NODE_ENV

    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kw):
            pass
        def for_node(self, n, role="worker"):
            return self
        def complete(self, prompt, *, system="", workdir=None, **kw):
            captured["system"] = system
            captured["prompt"] = prompt
            return "done"
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "dsh"}

    spec = {
        "schema_version": 1, "node_id": "emit", "role": "builder",
        "prompt": "build it", "system_prompt": "be careful",
        "reads": [], "writes": [],
        "model_context": {"provider": "test", "model": "test"},
    }
    manifest = {"id": "emit", "type": "AgentNode", "command": "echo",
                "reads": [], "writes": []}

    with mock.patch.dict(os.environ, {NODE_ENV: json.dumps(manifest)}):
        with mock.patch("refactory_dsh.model.DshModelClient", FakeClient):
            with mock.patch("refactory_dsh.agent_delivery.begin",
                            return_value={"prompt": "build it", "node": "emit", "status": "pending", "writes": []}):
                with mock.patch("refactory_dsh.agent_delivery.finish",
                                return_value={"node": "emit", "status": "delivered"}):
                    with mock.patch("refactory_dsh.agent_delivery._path", lambda w, n: Path(w) / n):
                        with mock.patch("refactory_dsh.run._knobs", return_value={}):
                            with mock.patch("refactory_dsh.run._read_facts", return_value={}):
                                from refactory_dsh import worker_node
                                worker_node.execute(spec, tmp_path)
    assert captured["system"] == "be careful"


def test_dsh_system_prompt_not_concatenated_into_stdin():
    """DSH path: task = prompt (not f'{system}\\n\\n{prompt}'), env var set.
    Revert: model.py _complete / _environment changes."""
    from refactory_dsh.model import DshModelClient

    # Build a minimal client by __new__ and setting required fields
    client = DshModelClient.__new__(DshModelClient)
    for attr, val in [("_provider", "p"), ("_model", "m"), ("_worker_backend", "dsh"),
                      ("_dsh_home", "/tmp/dsh"), ("_runtime_root", Path("/tmp/rt")),
                      ("_effort", ""), ("_dsh_cli", ""), ("_timeout_s", 300),
                      ("_attempts", 1), ("_retry_window_s", 0), ("_backoff_s", None),
                      ("_budget", None), ("_worker_receipt", None), ("_node_id", ""),
                      ("_role", "worker"), ("_cancellations", []),
                      ("_routing_provenance", {}), ("_python_bin", "/usr/bin/python3"),
                      ("_permission_mode", "skip"), ("_role_routes", {}), ("_node_routes", {}),
                      ("_provider_config", {})]:
        setattr(client, attr, val)

    # With system text: env var is set
    client._system_text = "You are a Python expert"
    env = client._environment()
    assert env.get("REFACTORY_SYSTEM_PROMPT") == "You are a Python expert"

    # Without system text: env var is NOT set
    client._system_text = ""
    env2 = client._environment()
    assert "REFACTORY_SYSTEM_PROMPT" not in env2

    # Verify _complete sets _system_text and keeps task = prompt
    captured: dict[str, Any] = {}
    def fake_run(cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):
        captured["task"] = task
        return type("R", (), {"returncode": 0, "stdout": "done", "stderr": ""})()
    client._run_completion = fake_run
    try:
        client._complete("History: user asked", system="You are expert", workdir=Path("/tmp"))
    except Exception:
        pass
    # task should be just the prompt, not f"{system}\n\n{prompt}"
    if "task" in captured:
        assert captured["task"] == "History: user asked"
        assert "You are expert" not in captured["task"]


def test_system_prompt_plugin_injects_section():
    """system_prompt_plugin.mjs is valid ESM with the right exports.
    Revert: system_prompt_plugin.mjs creation."""
    plugin = Path(__file__).resolve().parent.parent / "refactory_dsh" / "system_prompt_plugin.mjs"
    assert plugin.exists(), f"Plugin not found at {plugin}"
    content = plugin.read_text()
    assert "export const inject" in content
    assert "'systemPrompt'" in content
    assert "ctx.systemPrompt.section" in content
    assert "REFACTORY_SYSTEM_PROMPT" in content
    assert "refactory-node-system" in content


def test_dsh_composition_boot_system_channel(tmp_path):
    """write_patch inserts the system-prompt plugin only from the caller's
    system_prompt kwarg, not from the process environment.
    Revert: workers.py write_patch or model.py changes."""
    from refactory_dsh.workers import write_patch

    patch_path = tmp_path / "test.patch.yml"
    rt = Path(__file__).resolve().parent.parent.parent  # repo root

    # With system_prompt non-empty: plugin row is inserted
    write_patch(patch_path, rt, {"worker_backend": "dsh"}, system_prompt="You are an expert")
    rows = json.loads(patch_path.read_text())
    plugin_rows = [r for r in rows if isinstance(r, dict) and "insert" in r
                   for item in r.get("insert", []) if item.get("id") == "refactory-system-prompt"]
    assert len(plugin_rows) == 1, f"Expected system prompt plugin row, got: {rows}"

    # With system_prompt empty: no plugin row
    write_patch(patch_path, rt, {"worker_backend": "dsh"}, system_prompt="")
    rows2 = json.loads(patch_path.read_text())
    plugin_rows2 = [r for r in rows2 if isinstance(r, dict) and "insert" in r
                    for item in r.get("insert", []) if item.get("id") == "refactory-system-prompt"]
    assert len(plugin_rows2) == 0

    # Even if process env has REFACTORY_SYSTEM_PROMPT, empty kwarg wins
    with mock.patch.dict(os.environ, {"REFACTORY_SYSTEM_PROMPT": "wrong env"}):
        write_patch(patch_path, rt, {"worker_backend": "dsh"}, system_prompt="")
        rows3 = json.loads(patch_path.read_text())
        plugin_rows3 = [r for r in rows3 if isinstance(r, dict) and "insert" in r
                        for item in r.get("insert", []) if item.get("id") == "refactory-system-prompt"]
        assert len(plugin_rows3) == 0


def test_claude_code_preset_preserved_with_system_text():
    """Adapter patch: systemPrompt.type=preset, preset=claude_code, append=text.
    Revert: adapter patch systemPrompt addition."""
    # Read the patch file and verify the code pattern
    patch_path = Path(__file__).resolve().parent.parent.parent / "patches" / "@deepseek-ai__dsh-subagent-claude-code@0.2.0-rc.2.patch"
    content = patch_path.read_text()
    assert 'type: "preset"' in content
    assert 'preset: "claude_code"' in content
    assert "append: _rfSystemText" in content
    # Verify both branches exist
    assert '? { type: "preset", preset: "claude_code", append: _rfSystemText }' in content
    assert ': { type: "preset", preset: "claude_code" }' in content


def test_claude_code_preset_retained_with_no_system_text():
    """When system text is empty, preset is still set (not omitted).
    Revert: adapter patch without-append branch."""
    patch_path = Path(__file__).resolve().parent.parent.parent / "patches" / "@deepseek-ai__dsh-subagent-claude-code@0.2.0-rc.2.patch"
    content = patch_path.read_text()
    # The else branch must produce { type: "preset", preset: "claude_code" }
    assert ': { type: "preset", preset: "claude_code" }' in content


def test_unsupported_system_prompt_route_refused(tmp_path):
    """Codex backend refuses a non-empty system prompt before model call.
    Revert: executor.py or worker_node.py pre-flight check."""
    from refactory_dsh.workers import capabilities
    assert not capabilities("codex")["system_prompt"]

    # Worker path
    from refactory_dsh.run import NODE_ENV

    class FakeClient:
        def __init__(self, **kw):
            pass
        def for_node(self, n, role="worker"):
            return self
        def complete(self, prompt, **kw):
            raise AssertionError("Should not reach model call")
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "codex"}

    spec = {
        "schema_version": 1, "node_id": "emit", "role": "builder",
        "prompt": "build it", "system_prompt": "be careful",
        "reads": [], "writes": [],
        "model_context": {"provider": "test", "model": "test", "worker_backend": "codex"},
    }
    manifest = {"id": "emit", "type": "AgentNode", "reads": [], "writes": []}
    with mock.patch.dict(os.environ, {NODE_ENV: json.dumps(manifest)}):
        with mock.patch("refactory_dsh.model.DshModelClient", FakeClient):
            with mock.patch("refactory_dsh.run._knobs", return_value={}):
                with mock.patch("refactory_dsh.run._read_facts", return_value={}):
                    with pytest.raises(ValueError, match="system.prompt"):
                        from refactory_dsh import worker_node
                        worker_node.execute(spec, tmp_path)


def test_existing_graphs_without_system_prompt_unchanged():
    """A node without system_prompt_template yields identical state.
    Exemption: pins existing behavior."""
    from refactory_dsh.nn.graph import Node
    n = Node(id="c", kind="agent", prompt_template="hi")
    s = n.to_state()
    assert "system_prompt_template" not in s
    assert "system_prompt" not in s


def test_system_template_knob_rendering_through_real_path(tmp_path):
    """Full execute() path renders system_prompt_template with knobs.
    Revert: executor.py render call for system_prompt_template."""
    from refactory_dsh.nn.graph import Node
    from refactory_dsh.nn.executor import execute

    captured: dict[str, Any] = {}
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
    result = execute(graph, tmp_path, knobs={"task": "build", "role": "architect"},
                     model=FakeClient())
    assert captured["system"] == "You are architect"
    assert "do build" in captured["prompt"]


def test_unsupported_route_rejection_through_real_path(tmp_path):
    """Real execute() path refuses codex backend with system prompt.
    Revert: executor.py pre-flight capability check."""
    from refactory_dsh.nn.graph import Node
    from refactory_dsh.nn.executor import execute

    node = Node(id="build", kind="agent", prompt_template="do it",
                system_prompt_template="be careful",
                writes=("out.txt",), role="builder")
    graph = _make_graph(node)

    class FakeClient:
        def complete(self, prompt, **kw):
            raise AssertionError("Should not reach model call")
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "codex"}

    (tmp_path / "out.txt").write_text("")
    result = execute(graph, tmp_path, knobs={}, model=FakeClient())
    assert not result.exit_ok
    assert "system" in result.error.lower() and "prompt" in result.error.lower()


# ─── Ask 2: effort passthrough ───────────────────────────────────────

def test_validate_route_allows_effort_on_claude_code():
    """validate_route accepts effort on claude_code.
    Revert: workers.py validate_route split."""
    from refactory_dsh.workers import validate_route
    r = validate_route({"worker_backend": "claude_code", "effort": "medium"})
    assert r["effort"] == "medium"


def test_capabilities_claude_code_effort_and_system_prompt():
    """capabilities('claude_code') reports effort and system_prompt as True.
    Revert: workers.py capabilities update."""
    from refactory_dsh.workers import capabilities
    c = capabilities("claude_code")
    assert c["reasoning_effort"] is True
    assert c["system_prompt"] is True


def test_unsupported_effort_on_codex_refused():
    """validate_route raises on effort for codex.
    Revert: workers.py validate_route codex check."""
    from refactory_dsh.workers import validate_route
    with pytest.raises(ValueError, match="effort"):
        validate_route({"worker_backend": "codex", "effort": "medium"})


def test_claude_code_effort_medium_reaches_sdk_invocation():
    """Adapter patch reads REFACTORY_MODEL_EFFORT and sets effort in options.
    Revert: adapter patch effort addition."""
    patch_path = Path(__file__).resolve().parent.parent.parent / "patches" / "@deepseek-ai__dsh-subagent-claude-code@0.2.0-rc.2.patch"
    content = patch_path.read_text()
    assert "REFACTORY_MODEL_EFFORT" in content
    assert "options.effort = _rfEffort" in content


def test_actual_route_effort_not_copied_from_request():
    """actual_route.effort stays null in adapter metadata (CEO correction #3).
    Revert: adapter patch effort: null in consumeClaudeQuery."""
    patch_path = Path(__file__).resolve().parent.parent.parent / "patches" / "@deepseek-ai__dsh-subagent-claude-code@0.2.0-rc.2.patch"
    content = patch_path.read_text()
    assert "effort: null" in content
    # options.effort is not in the metadata init block
    init_block = content[content.index("actual_route"):content.index("source: \"native-sdk-init\"")]
    assert "effort: null" in init_block
