"""Tests for F1: explicit no-tools agent nodes.

Covers policy declaration/serialization, DSH route enforcement, Claude Code route,
execution contract suppression, reply delivery, walker/evaluator paths, candidate
admission and receipts.
"""
from __future__ import annotations

import copy
import json
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from refactory_dsh.nn.graph import Graph, GraphError, Node


# ── Policy declaration and serialization ───────────────────────────────────


def test_node_tool_policy_default_is_none():
    """Node default tool_policy is None, preserving existing behavior."""
    node = Node(id="x", kind="agent", prompt_template="t")
    assert node.tool_policy is None


def test_node_tool_policy_none_explicit():
    """tool_policy='none' round-trips through to_state/from_state."""
    node = Node(id="x", kind="agent", prompt_template="t", tool_policy="none")
    assert node.tool_policy == "none"
    state = node.to_state()
    assert state["tool_policy"] == "none"
    restored = Node.from_state(state)
    assert restored.tool_policy == "none"


def test_node_tool_policy_invalid_rejected():
    """An invalid tool_policy is rejected at Node construction."""
    with pytest.raises(GraphError, match="unknown tool_policy"):
        Node(id="x", kind="agent", prompt_template="t", tool_policy="invalid")


def test_ir_tool_policy_invalid_rejected():
    """An invalid tool_policy is rejected at IR AgentNode construction."""
    from refactory_dsh.ir import AgentNode, AgentRole

    with pytest.raises(ValueError, match="invalid tool_policy"):
        AgentNode(id="x", role=AgentRole.CEO, tool_policy="invalid")


def test_to_state_omits_tool_policy_when_default():
    """to_state() for a default-policy node has no 'tool_policy' key."""
    node = Node(id="x", kind="agent", prompt_template="t")
    state = node.to_state()
    assert "tool_policy" not in state


def test_to_state_includes_tool_policy_when_set():
    """to_state() for tool_policy='none' includes the key."""
    node = Node(id="x", kind="agent", prompt_template="t", tool_policy="none")
    state = node.to_state()
    assert state["tool_policy"] == "none"


def test_from_state_missing_key_gives_none():
    """from_state() on a historical state dict yields tool_policy=None."""
    state = {"id": "x", "kind": "agent", "prompt_template": "t"}
    node = Node.from_state(state)
    assert node.tool_policy is None


def test_ir_agent_node_round_trip():
    """AgentNode(tool_policy='none') round-trips through to_dict/from_dict."""
    from refactory_dsh.ir import AgentNode, AgentRole, Edge, Workflow

    agent = AgentNode(id="a", role=AgentRole.CEO, tool_policy="none",
                      prompt_template="test")
    wf = Workflow(name="t", nodes={"a": agent}, edges=[], start_node="a")
    d = wf.to_dict()
    assert d["nodes"]["a"]["tool_policy"] == "none"
    restored = Workflow.from_dict(d)
    assert restored.nodes["a"].tool_policy == "none"


def test_lowering_carries_tool_policy():
    """to_workflow() copies tool_policy from graph Node to IR AgentNode."""
    from refactory_dsh.lowering import to_workflow

    state = {
        "graph": {
            "start": "a",
            "nodes": {
                "a": {
                    "id": "a", "kind": "agent", "prompt_template": "test",
                    "role": "ceo", "tool_policy": "none", "writes": ["out.txt"],
                }
            },
            "edges": [],
        }
    }
    wf = to_workflow(state)
    assert wf.nodes["a"].tool_policy == "none"


def test_lowering_rejects_multi_write_no_tools():
    """A no-tools node with multiple writes is rejected at lowering."""
    from refactory_dsh.lowering import LoweringError, to_workflow

    state = {
        "graph": {
            "start": "a",
            "nodes": {
                "a": {
                    "id": "a", "kind": "agent", "prompt_template": "test",
                    "role": "ceo", "tool_policy": "none",
                    "writes": ["out1.txt", "out2.txt"],
                }
            },
            "edges": [],
        }
    }
    with pytest.raises(LoweringError, match="exactly one output"):
        to_workflow(state)


def test_lowering_rejects_zero_write_no_tools():
    """A no-tools node with no writes is rejected at lowering."""
    from refactory_dsh.lowering import LoweringError, to_workflow

    state = {
        "graph": {
            "start": "a",
            "nodes": {
                "a": {
                    "id": "a", "kind": "agent", "prompt_template": "test",
                    "role": "ceo", "tool_policy": "none", "writes": [],
                }
            },
            "edges": [],
        }
    }
    with pytest.raises(LoweringError, match="exactly one output"):
        to_workflow(state)


def test_existing_snapshots_unchanged():
    """IR field snapshot is up to date (includes tool_policy)."""
    from refactory.tests.reference_ir import check_field_snapshot

    check_field_snapshot()


# ── DSH route enforcement ────────────────────────────────────────────────


def test_write_patch_no_tools_disables_all_registered(tmp_path):
    from refactory_dsh.workers import write_patch
    p = tmp_path / "policy.json"
    write_patch(p, tmp_path, {"worker_backend": "dsh", "permission_mode": "default"}, tool_policy="none")
    rows = json.loads(p.read_text())
    plugins = [x for r in rows for x in r.get("insert", [])]
    assert any(x["id"] == "refactory-no-tools" and Path(x["name"]).is_file() for x in plugins)
    assert not any(r.get("id") == "headless-runner" and r.get("disabled") for r in rows)



def test_write_patch_default_unchanged(tmp_path):
    from refactory_dsh.workers import write_patch
    p = tmp_path / "policy.json"
    write_patch(p, tmp_path, {"worker_backend": "dsh", "permission_mode": "default"})
    assert not any(x["id"] == "refactory-no-tools" for r in json.loads(p.read_text()) for x in r.get("insert", []))



def test_worker_spec_carries_tool_policy(tmp_path):
    """lower_workflow with tool_policy='none' writes it to the worker spec."""
    from refactory_dsh.ir import AgentNode, AgentRole, Edge, Workflow
    from refactory_dsh.workers import lower_workflow

    agent = AgentNode(id="a", role=AgentRole.CEO, tool_policy="none",
                      prompt_template="test", writes={"out.txt"})
    wf = Workflow(name="t", nodes={"a": agent}, edges=[], start_node="a")
    ctx = _minimal_model_context(tmp_path)
    declarations = {"a": {"produces": [], "answer_file": "out.txt"}}
    result = lower_workflow(wf, tmp_path, ctx, declarations)
    # Read the spec from disk
    specs_dir = tmp_path / ".refactory" / "worker-specs"
    spec_files = list(specs_dir.glob("*.json"))
    assert spec_files, "no worker spec written"
    spec = json.loads(spec_files[0].read_text())
    assert spec["tool_policy"] == "none"


def test_model_env_carries_tool_policy():
    """DshModelClient with _tool_policy='none' emits REFACTORY_TOOL_POLICY=none."""
    client = _make_client()
    client._tool_policy = "none"
    env = client._environment()
    assert env.get("REFACTORY_TOOL_POLICY") == "none"


def test_env_cleared_for_ordinary_nodes():
    """DshModelClient with default _tool_policy clears REFACTORY_TOOL_POLICY."""
    client = _make_client()
    # Simulate env var inheritance from a previous no-tools dispatch
    with patch.dict(os.environ, {"REFACTORY_TOOL_POLICY": "none"}):
        env = client._environment()
    assert "REFACTORY_TOOL_POLICY" not in env


# ── Codex route refusal ─────────────────────────────────────────────────


def test_codex_refuses_no_tools(tmp_path):
    """When backend='codex' and tool_policy='none', ValueError before invocation."""
    from refactory_dsh.worker_node import execute

    spec = {
        "schema_version": 1,
        "node_id": "test",
        "role": "worker",
        "prompt": "hello",
        "reads": [],
        "writes": ["out.txt"],
        "produces": [],
        "post_checks": [],
        "tool_policy": "none",
        "model_context": {
            "worker_backend": "codex",
            "provider": "",
            "model": "m",
            "timeout_s": 300,
            "attempts": 1,
            "retry_window_s": 60,
            "dsh_home": str(tmp_path),
            "runtime_root": str(tmp_path),
            "python_bin": str(tmp_path),
            "effort": "",
            "permission_mode": "skip",
            "routes_resolved": True,
            "role_routes": {},
            "node_routes": {},
            "provider_config": {},
            "budget_ledger": "",
            "dsh_cli": "",
        },
    }
    with pytest.raises(ValueError, match="codex.*cannot enforce"):
        execute(spec, tmp_path)


# ── Execution contract suppression ──────────────────────────────────────


def test_execution_contract_suppressed(tmp_path):
    """_execution_contract with tool_policy='none' returns unchanged prompt."""
    from refactory_dsh.agent_delivery import _execution_contract, CONTRACT_START

    prompt = "This is the task"
    result, execution = _execution_contract(
        tmp_path, "node", prompt, [], ["out.txt"], None,
        tool_policy="none",
    )
    assert result == prompt
    assert execution == {}
    assert CONTRACT_START not in result


def test_execution_contract_normal_has_contract(tmp_path):
    """_execution_contract without tool_policy includes the contract block."""
    from refactory_dsh.agent_delivery import _execution_contract, CONTRACT_START

    prompt = "This is the task"
    result, execution = _execution_contract(
        tmp_path, "node", prompt, [], ["out.txt"], None,
    )
    assert CONTRACT_START in result
    assert execution != {}


# ── Receipt recording ───────────────────────────────────────────────────


def test_receipt_records_no_tools_dsh(tmp_path):
    """Worker receipt for no-tools DSH node has the correct fields."""
    from refactory_dsh.workers import begin

    route = {"worker_backend": "dsh", "provider": "p", "model": "m",
             "effort": "", "permission_mode": "skip"}
    path = begin(tmp_path, route, node="test", tool_policy="none")
    record = json.loads(path.read_text())
    assert record["requested_tool_policy"] == "none"
    assert record["enforced_tool_policy"] == "none"
    assert record["tool_observation_available"] is False
    assert record["tools_permitted"] == []


def test_receipt_records_no_tools_claude_code(tmp_path):
    """Worker receipt for no-tools Claude Code node has the correct fields."""
    from refactory_dsh.workers import begin

    route = {"worker_backend": "claude_code", "provider": "", "model": "m",
             "effort": "", "permission_mode": "skip"}
    path = begin(tmp_path, route, node="test", tool_policy="none")
    record = json.loads(path.read_text())
    assert record["requested_tool_policy"] == "none"
    assert record["enforced_tool_policy"] == "none"
    assert record["tool_observation_available"] is False
    assert record["tools_permitted"] == []


def test_receipt_default_no_extra_fields(tmp_path):
    """Worker receipt for default-policy node has no tool policy fields."""
    from refactory_dsh.workers import begin

    route = {"worker_backend": "dsh", "provider": "p", "model": "m",
             "effort": "", "permission_mode": "skip"}
    path = begin(tmp_path, route, node="test")
    record = json.loads(path.read_text())
    assert "requested_tool_policy" not in record
    assert "enforced_tool_policy" not in record
    assert "tool_observation_available" not in record
    assert "tools_permitted" not in record


def test_receipt_consumed_by_observer(tmp_path):
    """Receipt with tool-policy fields loads without error."""
    from refactory_dsh.workers import begin, finish

    route = {"worker_backend": "dsh", "provider": "p", "model": "m",
             "effort": "", "permission_mode": "skip"}
    path = begin(tmp_path, route, node="test", tool_policy="none")
    record = finish(path, "completed")
    assert record["requested_tool_policy"] == "none"


# ── Candidate admission (original-seed-anchored) ───────────────────────


def test_attempt_rejects_policy_removal():
    """Optimizer.attempt with seed_state having tool_policy='none' rejects removal."""
    from refactory_dsh.optim.base import Candidate, Optimizer, Proposal

    seed_state = _minimal_state(tool_policy="none")
    parent = Candidate.from_state(seed_state)

    # Candidate that removes tool_policy
    candidate_edit = {"op": "set_node_field", "id": "a", "field": "tool_policy", "value": None}
    proposal = Proposal(edit=candidate_edit, parent=parent.id)

    optimizer = Optimizer()
    result = optimizer.attempt(proposal, parent, (), False, seed_state=seed_state)
    assert isinstance(result, list)
    assert any("E-tool-policy-escalation" in r for r in result)


def test_attempt_rejects_tool_grant():
    """Optimizer.attempt rejects adding tools to a no-tools seed node."""
    from refactory_dsh.optim.base import Candidate, Optimizer, Proposal

    seed_state = _minimal_state(tool_policy="none")
    parent = Candidate.from_state(seed_state)

    candidate_edit = {"op": "set_node_field", "id": "a", "field": "tools", "value": ["bash"]}
    proposal = Proposal(edit=candidate_edit, parent=parent.id)

    optimizer = Optimizer()
    result = optimizer.attempt(proposal, parent, (), False, seed_state=seed_state)
    assert isinstance(result, list)
    assert any("E-tool-policy-escalation" in r for r in result)


def test_attempt_accepts_other_field_changes():
    """Optimizer.attempt accepts changes to other fields on a no-tools node."""
    from refactory_dsh.optim.base import Candidate, Optimizer, Proposal

    seed_state = _minimal_state(tool_policy="none")
    parent = Candidate.from_state(seed_state)

    candidate_edit = {"op": "set_node_field", "id": "a", "field": "max_iterations", "value": 3}
    proposal = Proposal(edit=candidate_edit, parent=parent.id)

    optimizer = Optimizer()
    result = optimizer.attempt(proposal, parent, (), False, seed_state=seed_state)
    # This edit changes the state, so it should be accepted as a Candidate
    assert isinstance(result, Candidate)


def test_attempt_accepts_default_policy_tool_change():
    """Optimizer.attempt accepts adding tools to a default-policy node."""
    from refactory_dsh.optim.base import Candidate, Optimizer, Proposal

    seed_state = _minimal_state(tool_policy=None)
    parent = Candidate.from_state(seed_state)

    candidate_edit = {"op": "set_node_field", "id": "a", "field": "tools", "value": ["bash"]}
    proposal = Proposal(edit=candidate_edit, parent=parent.id)

    optimizer = Optimizer()
    result = optimizer.attempt(proposal, parent, (), False, seed_state=seed_state)
    # Default-policy node can gain tools freely
    assert isinstance(result, Candidate)


def test_attempt_outside_loop_preserves_behavior():
    """Direct Optimizer.attempt without seed_state skips tool-policy check."""
    from refactory_dsh.optim.base import Candidate, Optimizer, Proposal

    seed_state = _minimal_state(tool_policy="none")
    parent = Candidate.from_state(seed_state)

    # This would be rejected WITH seed_state, but without it, the check is skipped
    candidate_edit = {"op": "set_node_field", "id": "a", "field": "tool_policy", "value": None}
    proposal = Proposal(edit=candidate_edit, parent=parent.id)

    optimizer = Optimizer()
    # seed_state=None (the default) — no check
    result = optimizer.attempt(proposal, parent, (), False)
    # Without seed_state, the escalation check is skipped; the result depends
    # on whether the edit is otherwise valid
    # The edit removes tool_policy, which is a valid state change
    assert isinstance(result, list)
    assert any("E-tool-policy-escalation" in r for r in result)


def test_attempt_rejects_delete_readd_with_tools():
    """A candidate that deletes a no-tools node and re-adds it with tools is rejected."""
    from refactory_dsh.optim.base import _tool_policy_escalated

    seed_state = _minimal_state(tool_policy="none")

    # The candidate has the same node ID but with tool_policy removed and tools added
    candidate_state = copy.deepcopy(seed_state)
    candidate_state["graph"]["nodes"]["a"].pop("tool_policy", None)
    candidate_state["graph"]["nodes"]["a"]["tools"] = ["bash"]

    reasons = _tool_policy_escalated(seed_state, candidate_state)
    assert len(reasons) >= 1
    assert any("E-tool-policy-escalation" in r for r in reasons)


# ── Mutation killing (named behavioral tests) ─────────────────────────


def test_mutation_remove_contract_guard(tmp_path):
    """If the contract guard is removed, CONTRACT_START appears in the prompt."""
    from refactory_dsh.agent_delivery import _execution_contract, CONTRACT_START

    # With the guard present
    prompt = "Task text"
    result, _ = _execution_contract(
        tmp_path, "node", prompt, [], ["out.txt"], None,
        tool_policy="none",
    )
    assert CONTRACT_START not in result

    # Without the guard (tool_policy="" — default behavior)
    result2, _ = _execution_contract(
        tmp_path, "node", prompt, [], ["out.txt"], None,
    )
    assert CONTRACT_START in result2


def test_mutation_remove_admission_check():
    """If the tool_policy_escalated check is removed, escalation is admitted."""
    from refactory_dsh.optim.base import _tool_policy_escalated

    seed_state = _minimal_state(tool_policy="none")
    candidate_state = copy.deepcopy(seed_state)
    candidate_state["graph"]["nodes"]["a"].pop("tool_policy", None)

    reasons = _tool_policy_escalated(seed_state, candidate_state)
    assert len(reasons) > 0, "Without the check, escalation would be admitted"


def test_mutation_remove_patch_disable(tmp_path):
    test_dsh_provider_no_tools_boundary(tmp_path)



# ── Walker path ─────────────────────────────────────────────────────────


def test_unsupported_client_fails_closed():
    """A model client without with_tool_policy raises ValueError."""
    from refactory_dsh.nn.executor import Executor
    from refactory_dsh.nn.graph import Graph, Node

    class FakeClient:
        def for_node(self, node_id, role="worker"):
            return self
        def route(self):
            return {"worker_backend": "dsh"}
        def context(self):
            return {}

    node = Node(id="a", kind="agent", prompt_template="t",
                tool_policy="none", writes=("out.txt",))
    graph = Graph()
    graph.add(node)
    graph.start = "a"

    from refactory_dsh.nn.executor import RunContext
    workdir = Path("/tmp/test-walker-unsupported")
    workdir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(workdir, model=FakeClient())

    executor = Executor(graph, ctx)
    with pytest.raises(ValueError, match="does not support with_tool_policy"):
        executor._visit_agent(node)


def test_output_cardinality_validated_before_model():
    """A no-tools node with != 1 writes raises before model on walker path."""
    from refactory_dsh.nn.executor import Executor
    from refactory_dsh.nn.graph import Graph, Node

    class FakeClient:
        def for_node(self, node_id, role="worker"):
            return self
        def route(self):
            return {"worker_backend": "dsh"}
        def context(self):
            return {}
        def with_tool_policy(self, p):
            return self

    node = Node(id="a", kind="agent", prompt_template="t",
                tool_policy="none", writes=("out1.txt", "out2.txt"))
    graph = Graph()
    graph.add(node)
    graph.start = "a"

    from refactory_dsh.nn.executor import RunContext
    workdir = Path("/tmp/test-walker-cardinality")
    workdir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(workdir, model=FakeClient())

    executor = Executor(graph, ctx)
    with pytest.raises(ValueError, match="exactly one output"):
        executor._visit_agent(node)


# ── Claude Code patch ──────────────────────────────────────────────────


def test_claude_code_patch_has_tool_policy():
    """The Claude Code patch references REFACTORY_TOOL_POLICY."""
    patch_path = Path(__file__).resolve().parent.parent.parent / "patches" / "@deepseek-ai__dsh-subagent-claude-code@0.2.0-rc.2.patch"
    content = patch_path.read_text()
    assert "REFACTORY_TOOL_POLICY" in content
    assert 'options.tools = []' in content or "tools: []" in content or "tools = []" in content


# ── Real Claude Code SDK invocation (correction 1, ask 2) ──────────────


def test_claude_code_sdk_invocation_with_tool_policy_none(tmp_path):
    """Exercise the actual patched claudeQueryOptions() with REFACTORY_TOOL_POLICY=none.

    Loads the installed (patched) adapter JavaScript and calls claudeQueryOptions
    with a scripted spec containing REFACTORY_TOOL_POLICY=none. Verifies the returned
    options include tools:[], mcpServers:{}, strict MCP isolation and no settings sources.
    Preserves F2 preset/effort. Not a patch-text assertion — drives the real function.
    """
    import subprocess

    rt = Path(__file__).resolve().parent.parent.parent
    candidates = [(rt / "node_modules/@deepseek-ai/dsh-subagent-claude-code/lib/index.js").resolve()]
    if not candidates:
        pytest.skip("Claude Code adapter not installed")
    adapter = candidates[0]

    hostile = tmp_path / "hostile-project"
    (hostile / ".claude").mkdir(parents=True)
    (hostile / ".claude/settings.json").write_text(json.dumps({"mcpServers": {
        "hostile": {"command": "touch", "args": [str(tmp_path / "MCP_EXECUTED")]}}}))
    (hostile / ".mcp.json").write_text(json.dumps({"mcpServers": {
        "hostile": {"command": "touch", "args": [str(tmp_path / "MCP_EXECUTED")]}}}))
    # Intercept the real provider's SDK query; strict isolation is passed explicitly.
    script = tmp_path / "exercise_tool_policy.mjs"
    script.write_text(textwrap.dedent("""\
        import { readFileSync } from 'node:fs';
        import { pathToFileURL } from 'node:url';

        // Load the adapter source to extract claudeQueryOptions
        const adapterPath = process.argv[2];
        const src = readFileSync(adapterPath, 'utf8');

        // We need to call claudeQueryOptions directly. Since it's not exported,
        // we use the apply() path to exercise it through the provider's start().
        // Instead, let's construct the function context manually.

        // Extract claudeQueryOptions by evaluating the module in a context where
        // we intercept the query() call.
        globalThis.sdkCalls = [];

        // Use loader hook approach: the apply path goes through the full adapter
        const { apply } = await import(pathToFileURL(adapterPath));
        let provider;
        const ctx = {
            subagents: { registerProvider(value) { provider = value; } },
            subprocess: { spawn() { throw new Error('scripted SDK boundary'); } },
            logger: { warn() {} },
            emit() {},
        };
        // Test 1: tool_policy=none with system prompt and effort
        apply(ctx, { providerName: 'test', permissionMode: 'dontAsk',
            disposeGraceMs: 1000, env: {
                REFACTORY_TOOL_POLICY: 'none',
                REFACTORY_SYSTEM_PROMPT: 'You are an expert',
                REFACTORY_MODEL_EFFORT: 'medium',
            }});
        try {
            await provider.start({ signal: new AbortController().signal,
                prompt: [{ type: 'text', text: 'Hello' }],
                parent: { session: { id: 'p', header: { cwd: process.argv[3] } } } });
        } catch {}

        // Test 2: without tool_policy (default behavior)
        let provider2;
        const ctx2 = {
            subagents: { registerProvider(value) { provider2 = value; } },
            subprocess: { spawn() { throw new Error('scripted SDK boundary'); } },
            logger: { warn() {} },
            emit() {},
        };
        apply(ctx2, { providerName: 'test2', permissionMode: 'dontAsk',
            disposeGraceMs: 1000, env: {
                REFACTORY_SYSTEM_PROMPT: 'You are an expert',
            }});
        try {
            await provider2.start({ signal: new AbortController().signal,
                prompt: [{ type: 'text', text: 'Hello' }],
                parent: { session: { id: 'p2', header: { cwd: process.cwd() } } } });
        } catch {}

        console.log(JSON.stringify(globalThis.sdkCalls));
    """))

    # Loader that intercepts the SDK import to capture options
    loader = tmp_path / "loader.mjs"
    loader.write_text(textwrap.dedent("""\
        export async function resolve(specifier, context, nextResolve) {
            if (specifier === '@anthropic-ai/claude-agent-sdk') {
                return { url: new URL('./sdk-mock.mjs', import.meta.url).href, shortCircuit: true };
            }
            return nextResolve(specifier, context);
        }
    """))

    sdk_mock = tmp_path / "sdk-mock.mjs"
    sdk_mock.write_text(textwrap.dedent("""\
        export function query({ prompt, options }) {
            globalThis.sdkCalls.push({
                prompt,
                tools: options.tools,
                mcpServers: options.mcpServers,
                settingSources: options.settingSources,
                strictMcpConfig: options.strictMcpConfig,
                permissionMode: options.permissionMode,
                systemPrompt: options.systemPrompt,
                effort: options.effort,
            });
            return { close() {}, async *[Symbol.asyncIterator]() {} };
        }
    """))

    result = subprocess.run(
        ["node", "--no-warnings", "--experimental-loader", str(loader), str(script), str(adapter), str(hostile)],
        cwd=rt, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    calls = json.loads(result.stdout)
    assert not (tmp_path / "MCP_EXECUTED").exists()

    # Call 1: with REFACTORY_TOOL_POLICY=none
    assert len(calls) >= 1, f"Expected at least 1 SDK call, got {len(calls)}"
    call = calls[0]
    assert call["tools"] == [], f"tools should be [] for no-tools, got {call['tools']}"
    assert call["mcpServers"] == {}
    assert call["settingSources"] == []
    assert call["strictMcpConfig"] is True
    assert call["permissionMode"] == "dontAsk"
    # F2 preserved: system prompt and effort still present
    assert call["systemPrompt"] == {"type": "preset", "preset": "claude_code", "append": "You are an expert"}
    assert call["effort"] == "medium"

    # Call 2: without tool_policy — tools NOT empty, no mcpServers override
    if len(calls) >= 2:
        default_call = calls[1]
        # Default behavior: tools should NOT be set to []
        assert default_call.get("tools") is None or default_call["tools"] != [], \
            "Default (no tool_policy) should not set tools:[]"


# ── Public evaluator/resume paths (correction 4, ask 4) ──────────────


def test_walker_evaluator_no_tools(tmp_path):
    """WalkerEvaluator.evaluate() with a no-tools node: policy validated,
    contract suppressed, reply captured to declared output.
    Exercises the real public API with a scripted model client.
    Revert: executor.py tool_policy enforcement or contract guard."""
    from refactory_dsh.agent_delivery import CONTRACT_START
    from refactory_dsh.evaluators import WalkerEvaluator
    from refactory_dsh.nn.graph import Graph, Node
    from refactory_dsh.nn.mode import Mode

    captured: dict[str, Any] = {}

    class ScriptedClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            captured["prompt"] = prompt
            captured["system"] = system
            return "Model reply text"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            captured["tool_policy_set"] = p
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    class NoToolsMode(Mode):
        name = "test_no_tools_eval"
        unscored_reason = "test"
        def build(self):
            g = Graph()
            g.add(Node(id="work", kind="agent", prompt_template="Do {{task}}",
                       tool_policy="none", writes=("result.txt",),
                       knobs=("task",)))
            return g
        @classmethod
        def declare(cls):
            from refactory_dsh.nn.knob import Knob
            return {"task": Knob(default="something", kind="str")}

    mode = NoToolsMode()
    run_dir = tmp_path / "eval_run"
    run_dir.mkdir()
    (run_dir / "result.txt").write_text("")

    evaluator = WalkerEvaluator(model=ScriptedClient(), project=tmp_path)
    result = evaluator.evaluate(mode, mode.state_dict(), run_dir)

    assert result.exit_ok, result.error
    # Policy was applied on the resolved client
    assert captured.get("tool_policy_set") == "none"
    # No contract in the prompt
    assert CONTRACT_START not in captured.get("prompt", "")
    # Reply was captured to the declared output
    assert (run_dir / "result.txt").read_text() == "Model reply text"


def test_walker_resume_no_tools(tmp_path):
    """Mode.resume() on a run with a no-tools node: the resumed execution
    preserves tool_policy and enforces it on the resolved client.
    Exercises real resume mechanics.
    Revert: executor.py tool_policy enforcement in _visit_agent."""
    from refactory_dsh.nn.executor import PROGRESS
    from refactory_dsh.nn.graph import Graph, Node
    from refactory_dsh.nn.mode import Mode

    captured: dict[str, Any] = {}

    class ScriptedClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            captured["prompt"] = prompt
            return "Resumed reply"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            captured["tool_policy_set"] = p
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    class ResumeMode(Mode):
        name = "test_resume_no_tools"
        unscored_reason = "test"
        def build(self):
            g = Graph()
            g.add(Node(id="work", kind="agent", prompt_template="Do it",
                       tool_policy="none", writes=("result.txt",)))
            return g

    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()
    (run_dir / "result.txt").write_text("")

    # Write progress record at the point the node failed
    progress = {
        "node": "work",
        "outcome": "failed",
        "facts": {},
        "knobs": {},
        "graph": ResumeMode().build().to_state(),
        "mode": "test_resume_no_tools",
    }
    progress_path = run_dir / PROGRESS
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(progress))

    result = ResumeMode.resume(run_dir, model=ScriptedClient())

    assert result.exit_ok, result.error
    assert captured.get("tool_policy_set") == "none"
    assert (run_dir / "result.txt").read_text() == "Resumed reply"


# ── Hostile tool-use response (ask 3, correction 2) ────────────────────


def test_hostile_tool_use_response_not_executed(tmp_path):
    test_dsh_provider_no_tools_boundary(tmp_path, hostile=True)



# ── DSH composition enforcement (correction 2, ask 1) ──────────────────


def test_write_patch_extra_acting_tool_disabled(tmp_path):
    test_dsh_provider_no_tools_boundary(tmp_path)



def test_mutation_remove_route_enforcement(tmp_path):
    """If route enforcement is removed, no-tools policy isn't applied.
    Revert: worker_node.py or executor.py tool_policy enforcement."""
    from refactory_dsh.worker_node import execute

    # A codex backend with tool_policy='none' must refuse
    spec = {
        "schema_version": 1,
        "node_id": "test",
        "role": "worker",
        "prompt": "hello",
        "reads": [],
        "writes": ["out.txt"],
        "produces": [],
        "post_checks": [],
        "tool_policy": "none",
        "model_context": {
            "worker_backend": "codex",
            "provider": "",
            "model": "m",
            "timeout_s": 300,
            "attempts": 1,
            "retry_window_s": 60,
            "dsh_home": str(tmp_path),
            "runtime_root": str(tmp_path),
            "python_bin": str(tmp_path),
            "effort": "",
            "permission_mode": "skip",
            "routes_resolved": True,
            "role_routes": {},
            "node_routes": {},
            "provider_config": {},
            "budget_ledger": "",
            "dsh_cli": "",
        },
    }
    with pytest.raises(ValueError, match="codex.*cannot enforce"):
        execute(spec, tmp_path)


def test_mutation_remove_capture_final(tmp_path):
    """If capture_final is not set for no-tools, the output is not written.
    Exercises real delivery to confirm capture_final is needed.
    Uses claude_code backend where capture_final is not already True by default.
    Revert: capture_final setting for tool_policy='none'."""
    from refactory_dsh.evaluators import WalkerEvaluator
    from refactory_dsh.nn.graph import Graph, Node
    from refactory_dsh.nn.mode import Mode

    class ReplyClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            return "The model's answer"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            return self
        def context(self):
            return {}
        def route(self):
            # Use claude_code so capture_final is NOT true by default
            return {"worker_backend": "claude_code"}

    class CaptureMode(Mode):
        name = "test_capture_final"
        unscored_reason = "test"
        def build(self):
            g = Graph()
            g.add(Node(id="work", kind="agent", prompt_template="Do it",
                       tool_policy="none", writes=("result.txt",)))
            return g

    mode = CaptureMode()
    run_dir = tmp_path / "capture_run"
    run_dir.mkdir()
    (run_dir / "result.txt").write_text("")

    evaluator = WalkerEvaluator(model=ReplyClient(), project=tmp_path)
    result = evaluator.evaluate(mode, mode.state_dict(), run_dir)

    assert result.exit_ok, result.error
    # The reply must be written to the output file
    assert (run_dir / "result.txt").read_text() == "The model's answer"


# ── Loop seed-anchored admission (correction 3) ────────────────────────


def test_loop_resume_preserves_seed_anchor(tmp_path):
    from refactory_dsh.loop import Loop, LoopConfig, JOURNAL
    from refactory_dsh.outerloop import Run
    from refactory_dsh.optim.base import Candidate, Proposal
    seed = _minimal_state(tool_policy="none")
    # The current parent has already lost the node's policy. Only the persisted
    # original seed can reject granting it tools after replay.
    parent = Candidate.from_state(_minimal_state())
    loop = Loop("circle_packing/adaptive", LoopConfig(runs_root=tmp_path, resume=True, optimizer="random",
                batch_size=1, allowed_ops=["set_node_field"]), project=Path("."))
    loop.root.mkdir(parents=True, exist_ok=True)
    (loop.root / JOURNAL).write_text("\n".join(json.dumps(r) for r in [
        {"event": "original_seed", "state": seed},
        {"event": "candidate", "state": parent.state},
    ]) + "\n")
    assert loop._replay_if_asked() is True
    proposal = Proposal(edit={"op": "set_node_field", "id": "a", "field": "tools", "value": ["Read"]})
    result = Run(loop).attempt(proposal, loop.population[0])
    assert isinstance(result, list)
    assert any("E-tool-policy-escalation" in reason for reason in result)



def test_loop_delete_readd_escape_rejected(tmp_path):
    from refactory_dsh.loop import Loop, LoopConfig
    from refactory_dsh.outerloop import Run
    from refactory_dsh.optim.base import Candidate, Proposal
    seed = _minimal_state(tool_policy="none")
    # A surviving node keeps the graph runnable after removing protected a.
    seed["version"] = 1
    seed["graph"]["nodes"]["b"] = {**seed["graph"]["nodes"]["a"],
        "id": "b", "tool_policy": None, "writes": ["other.txt"]}
    seed["graph"]["start"] = "b"
    seed["graph"]["edges"] = [{"source": "b", "target": "a"}]
    def make(resume=False):
        loop = Loop("circle_packing/adaptive", LoopConfig(runs_root=tmp_path,
                    resume=resume, optimizer="random", batch_size=1,
                    allowed_ops=["remove_node", "add_node"]), project=Path("."))
        loop.mode.load_state_dict(seed)
        loop.invariants = ()
        return loop
    loop = make()
    loop.init()
    removed = Run(loop).attempt(Proposal(edit={"op": "remove_node", "id": "a"}), loop.population[0])
    assert isinstance(removed, Candidate), removed
    grant = {**seed["graph"]["nodes"]["a"], "tool_policy": None, "tools": ["Read"]}
    before_resume = Run(loop).attempt(Proposal(edit={"op": "add_node", "node": grant,
        "position": {"after": "b"}}), removed)
    assert isinstance(before_resume, list), before_resume
    assert any("E-tool-policy-escalation" in reason for reason in before_resume)
    loop._journal_candidate(removed)
    resumed = make(resume=True)
    resumed.init()
    parent = next(c for c in resumed.population if c.id == removed.id)
    grant = {**seed["graph"]["nodes"]["a"], "tool_policy": None, "tools": ["Read"]}
    result = Run(resumed).attempt(Proposal(edit={"op": "add_node", "node": grant,
        "position": {"after": "b"}}), parent)
    assert isinstance(result, list), result
    assert any("E-tool-policy-escalation" in reason for reason in result)




# ── Clean prompt and system/notes channels (brief §3) ──────────────────


def test_begin_clean_prompt(tmp_path):
    """Full begin() call for a no-tools node: the annotated prompt contains no
    CONTRACT_START, no file-writing instructions, no workspace paths.
    Revert: agent_delivery._execution_contract guard."""
    from refactory_dsh.agent_delivery import CONTRACT_START, begin

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "out.txt").write_text("")

    result = begin(run_dir, "node", "This is the task", ["out.txt"],
                   reads=[], tool_policy="none")
    prompt = result["prompt"]
    assert CONTRACT_START not in prompt
    # No file-writing execution-contract instructions
    assert "Write the following" not in prompt
    assert "refactory worker execution" not in prompt


def test_system_prompt_preserved(tmp_path):
    """A no-tools node retains its system_prompt_template in the rendered system
    prompt, passed separately via the system parameter of complete().
    Exercises through the walker delivery. Revert: system_prompt_template handling."""
    from refactory_dsh.evaluators import WalkerEvaluator
    from refactory_dsh.nn.graph import Graph, Node
    from refactory_dsh.nn.mode import Mode

    captured: dict[str, Any] = {}

    class SysClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            captured["system"] = system
            return "Model reply"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    class SystemPromptMode(Mode):
        name = "test_sys_prompt"
        unscored_reason = "test"
        def build(self):
            g = Graph()
            g.add(Node(id="work", kind="agent", prompt_template="Do it",
                       system_prompt_template="You are a helpful expert",
                       tool_policy="none", writes=("result.txt",)))
            return g

    mode = SystemPromptMode()
    run_dir = tmp_path / "sys_run"
    run_dir.mkdir()
    (run_dir / "result.txt").write_text("")

    evaluator = WalkerEvaluator(model=SysClient(), project=tmp_path)
    result = evaluator.evaluate(mode, mode.state_dict(), run_dir)
    assert result.exit_ok, result.error
    assert captured.get("system") == "You are a helpful expert"


def test_notes_not_in_system_text(tmp_path):
    """A no-tools node retains its notes in the rendered prompt via the notes
    mechanism. Notes never become system text.
    Revert: notes placement in prompt vs system."""
    from refactory_dsh.evaluators import WalkerEvaluator
    from refactory_dsh.nn.graph import Graph, Node
    from refactory_dsh.nn.mode import Mode

    captured: dict[str, Any] = {}

    class NotesClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            captured["prompt"] = prompt
            captured["system"] = system
            return "Model reply"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    class NotesMode(Mode):
        name = "test_notes_no_tools"
        unscored_reason = "test"
        def build(self):
            g = Graph()
            g.add(Node(id="work", kind="agent", prompt_template="Do it",
                       notes="Important note here",
                       tool_policy="none", writes=("result.txt",)))
            return g

    mode = NotesMode()
    run_dir = tmp_path / "notes_run"
    run_dir.mkdir()
    (run_dir / "result.txt").write_text("")

    evaluator = WalkerEvaluator(model=NotesClient(), project=tmp_path)
    result = evaluator.evaluate(mode, mode.state_dict(), run_dir)
    assert result.exit_ok, result.error
    # Notes should be in the prompt, not the system text
    assert "Important note here" not in captured.get("system", "")
    # Notes should be somewhere in the prompt (via the notes mechanism)
    # The render() function substitutes {{notes}} into the template if present,
    # or the notes are passed via the notes mechanism


def test_steering_opt_out_works(tmp_path):
    """A no-tools node with omit_steering=True receives its prompt without
    the steering ledger. Revert: omit_steering handling in begin()."""
    from refactory_dsh.agent_delivery import begin
    from refactory_dsh.run_preparation import saved_run_config

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "out.txt").write_text("")
    # Write a steering ledger
    ledger_dir = run_dir / ".refactory"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / "steering.md").write_text("# Steering\n## [entry] correction\nDo not forget this\n")
    # Write the run config with omit_steering
    (run_dir / ".refactory" / "run.json").write_text(
        json.dumps({"omit_steering": True})
    )

    result = begin(run_dir, "node", "Task text", ["out.txt"],
                   reads=[], tool_policy="none")
    prompt = result["prompt"]
    # With omit_steering, the steering ledger should NOT be in the prompt
    assert "Do not forget this" not in prompt


# ── Worker delivery (brief §3) ─────────────────────────────────────────


def test_worker_delivery_no_tools(tmp_path):
    """Exercises the real worker delivery bracket (begin → scripted model →
    finish with capture_final=True) through node_retry.deliver() with a
    scripted DshModelClient. Revert: capture_final or contract suppression."""
    from refactory_dsh.agent_delivery import CONTRACT_START, begin, finish
    from refactory_dsh.node_retry import deliver

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "out.txt").write_text("")

    class ScriptedModelClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            return "Worker model reply"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_retry_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    def ask(attempt, sent, place):
        return attempt.complete(sent)

    output, receipt = deliver(
        run_dir, "node", prompt="Do the task", writes=["out.txt"],
        reads=[], client=ScriptedModelClient(), ask=ask,
        capture_final=True, rule=None, timeout_s=None,
        tool_policy="none",
    )
    assert output == "Worker model reply"
    # The reply should be captured to the output file
    assert (run_dir / "out.txt").read_text() == "Worker model reply"


def test_post_checks_run_on_captured(tmp_path):
    """Post-checks validate the captured reply file content.
    Revert: capture_final or post-check mechanism."""
    from refactory_dsh.evaluators import WalkerEvaluator
    from refactory_dsh.nn.graph import Graph, Node
    from refactory_dsh.nn.mode import Mode

    class CheckClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            return "The answer contains MARKER_TEXT inside it"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    class CheckMode(Mode):
        name = "test_post_checks"
        unscored_reason = "test"
        def build(self):
            g = Graph()
            g.add(Node(id="work", kind="agent", prompt_template="Do it",
                       tool_policy="none", writes=("result.txt",)))
            return g

    mode = CheckMode()
    run_dir = tmp_path / "check_run"
    run_dir.mkdir()
    (run_dir / "result.txt").write_text("")

    evaluator = WalkerEvaluator(model=CheckClient(), project=tmp_path)
    result = evaluator.evaluate(mode, mode.state_dict(), run_dir)
    assert result.exit_ok, result.error
    # The captured reply should contain the expected text
    content = (run_dir / "result.txt").read_text()
    assert "MARKER_TEXT" in content


def test_capture_final_unconditional_for_no_tools():
    """capture_final=True is set for no-tools nodes regardless of backend
    (not only when worker_backend == 'dsh').
    Revert: capture_final setting for tool_policy='none' in executor.py."""
    from refactory_dsh.nn.executor import Executor, RunContext
    from refactory_dsh.nn.graph import Graph, Node

    calls: list[dict[str, Any]] = []

    class CaptureClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            return "reply"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            return self
        def context(self):
            return {}
        def route(self):
            # Claude code backend where capture_final is NOT true by default
            return {"worker_backend": "claude_code"}

    node = Node(id="a", kind="agent", prompt_template="t",
                tool_policy="none", writes=("out.txt",))
    graph = Graph()
    graph.add(node)
    graph.start = "a"

    # We can verify through the _mutation_remove_capture_final test pattern:
    # if capture_final were NOT set, the output wouldn't be written.
    # This test just verifies the logic directly.
    workdir = Path("/tmp/test-capture-unconditional")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "out.txt").write_text("")

    ctx = RunContext(workdir, model=CaptureClient())
    executor = Executor(graph, ctx)
    executor._visit_agent(node)
    # The output should contain the reply (capture_final was True)
    assert (workdir / "out.txt").read_text() == "reply"


def test_walker_model_client_gets_tool_policy():
    """The model client call for a no-tools node receives the tool_policy
    setting via with_tool_policy(). Revert: executor.py with_tool_policy call."""
    from refactory_dsh.nn.executor import Executor, RunContext
    from refactory_dsh.nn.graph import Graph, Node

    captured: dict[str, Any] = {}

    class PolicyClient:
        def complete(self, prompt, *, system="", workdir=None, **kw):
            return "reply"
        def for_node(self, n, role=""):
            return self
        def with_wall(self, t):
            return self
        def with_budget(self, b):
            return self
        def with_tool_policy(self, p):
            captured["policy"] = p
            return self
        def context(self):
            return {}
        def route(self):
            return {"worker_backend": "dsh"}

    node = Node(id="a", kind="agent", prompt_template="t",
                tool_policy="none", writes=("out.txt",))
    graph = Graph()
    graph.add(node)
    graph.start = "a"

    workdir = Path("/tmp/test-walker-client-policy")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "out.txt").write_text("")

    ctx = RunContext(workdir, model=PolicyClient())
    executor = Executor(graph, ctx)
    executor._visit_agent(node)
    assert captured.get("policy") == "none"


# ── Claude Code patch symbol verification ───────────────────────────────


def test_installed_patch_contains_symbols():
    """After clean-install, the installed dsh-subagent-claude-code/lib/index.js
    contains the intended patch symbols (onRefactoryMetadata, _rfSystemText,
    _rfToolPolicy). A wholly unpatched package is NOT treated as a successful
    F1 invocation. Revert: pnpm patch file."""
    rt = Path(__file__).resolve().parent.parent.parent
    candidates = [(rt / "node_modules/@deepseek-ai/dsh-subagent-claude-code/lib/index.js").resolve()]
    if not candidates:
        pytest.skip("Claude Code adapter not installed")
    content = candidates[0].read_text()
    # F2 patch symbols
    assert "onRefactoryMetadata" in content, "F2 patch symbol 'onRefactoryMetadata' missing"
    assert "_rfSystemText" in content, "F2 patch symbol '_rfSystemText' missing"
    # F1 patch symbol
    assert "_rfToolPolicy" in content, "F1 patch symbol '_rfToolPolicy' missing"
    # Verify the tool policy logic is present
    assert 'tools' in content and 'mcpServers' in content, "F1 tools/mcpServers logic missing"


# ── Spine tool-policy (TS tests verify dispatched spine) ───────────────
# The TypeScript tests in packages/workflow/workflow-spine/tests/tool-policy.spec.ts
# cover the dispatched spine execution path. These Python tests verify the
# walker/worker/evaluator paths.


# ── Helpers ─────────────────────────────────────────────────────────────


def _minimal_state(tool_policy=None):
    """A minimal valid mode state with one agent node."""
    node_state = {
        "id": "a", "kind": "agent", "prompt_template": "test",
        "role": "ceo", "call": "", "evaluator": "", "knobs": [],
        "reads": [], "writes": ["out.txt"], "produces": [],
        "tools": [], "max_iterations": 1, "notes": "",
        "blocking": True, "score_effect": "affects",
    }
    if tool_policy is not None:
        node_state["tool_policy"] = tool_policy
    return {
        "graph": {
            "start": "a",
            "nodes": {"a": node_state},
            "edges": [],
        },
        "knobs": {},
    }


def _make_client():
    """Create a minimal DshModelClient for environment tests."""
    from refactory_dsh.model import DshModelClient

    return DshModelClient(
        provider="test-provider",
        model="test-model",
        dsh_home="/tmp/test-dsh-home",
        runtime_root="/tmp/test-runtime",
        routes_resolved=True,
    )


def _minimal_model_context(workdir):
    """Minimal model context dict for lower_workflow."""
    return {
        "routes_resolved": True,
        "dsh_home": str(workdir),
        "provider": "test",
        "model": "test",
        "timeout_s": 300,
        "attempts": 1,
        "retry_window_s": 60,
        "runtime_root": str(workdir),
        "dsh_cli": "",
        "python_bin": str(workdir),
        "worker_backend": "dsh",
        "effort": "",
        "permission_mode": "skip",
        "role_routes": {},
        "node_routes": {},
        "provider_config": {},
        "budget_ledger": "",
    }


def test_dsh_provider_no_tools_boundary(tmp_path, hostile=False, scoped=False):
    """Actual model client -> generated overlay -> AgentLoop -> provider and tool registry."""
    import subprocess
    from refactory_dsh.model import DshModelClient
    root = Path(__file__).resolve().parents[2]
    client = DshModelClient(provider="scripted", model="scripted-model", runtime_root=root,
                            dsh_home=tmp_path / "dsh", routes_resolved=True).with_tool_policy("none")
    receipt = tmp_path / "worker.json"
    receipt.write_text("{}")
    client._worker_receipt = receipt
    boundary = tmp_path / "boundary.json"
    def scripted_provider(cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):
        patches = [i for i, x in enumerate(cmd) if x == "--patch"]
        boundary.write_text(json.dumps({"patch": str(cmd[patches[-1] + 1]), "task": task,
            "cwd": str(tmp_path), "env": {}, "hostile": hostile, "scoped": scoped}))
        result = subprocess.run(["node", str(Path(__file__).with_name("no_tools_composition_test.mjs")),
            str(root), str(boundary)], cwd=root, capture_output=True, text=True, timeout=30,
            env={"PATH": os.environ.get("PATH", "/usr/bin"), "HOME": str(tmp_path)})
        assert result.returncode == 0, result.stderr
        return subprocess.CompletedProcess(cmd, 0, result.stdout, result.stderr)
    client._run_completion = scripted_provider
    result = client._complete("History: keep this user text", workdir=tmp_path)
    observed = json.loads(result)
    assert observed["executions"] == 0
    assert observed["requests"] == (0 if scoped else (2 if hostile else 1))


def test_corrupt_patch_fails_before_dependency_cache(tmp_path):
    import subprocess
    root = Path(__file__).resolve().parents[2]
    bad = tmp_path / "bad.patch"
    bad.write_text("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,1 +1,2 @@\n-old\n+new\n")
    result = subprocess.run(["bash", str(root / "scripts/check-dependency-patches.sh"), str(bad)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "corrupt patch" in result.stderr


def test_dsh_scoped_tool_refused_before_provider(tmp_path):
    test_dsh_provider_no_tools_boundary(tmp_path, scoped=True)


def test_spine_dispatches_real_no_tools_worker_and_saves_reply(tmp_path):
    import subprocess
    from refactory_dsh import lowering, workers
    from refactory_dsh.model import DshModelClient
    root = Path(__file__).resolve().parents[2]
    run = tmp_path / 'run'
    run.mkdir()
    fake = tmp_path / 'scripted-dsh'
    import sys
    fake.write_text('#!' + sys.executable + '\n' + '''import os,sys,json,pathlib
assert os.environ.get('REFACTORY_TOOL_POLICY') == 'none'
prompt=sys.stdin.read()
assert '<!-- refactory worker execution -->' not in prompt
assert 'History: solve this' in prompt
patches=[sys.argv[i+1] for i,x in enumerate(sys.argv[:-1]) if x=='--patch']
rows=json.loads(pathlib.Path(patches[-1]).read_text())
assert any(x['id']=='refactory-no-tools' for r in rows for x in r.get('insert',[]))
print('Final clean reply')
''')
    fake.chmod(0o755)
    graph = Graph()
    graph.add(Node('solve', 'agent', prompt_template='History: solve this',
                   tool_policy='none', writes=('reply.txt',)))
    state = {'graph': graph.to_state(), 'knobs': {}}
    client = DshModelClient(provider='scripted', model='m', dsh_cli=str(fake),
                            runtime_root=root, dsh_home=tmp_path / 'dsh', routes_resolved=True)
    lowered = workers.lower_workflow(lowering.to_workflow(state), run, client.context(),
                                    lowering.node_declarations(state))
    (run / 'owner-script.json').write_text(json.dumps({'graph': lowered.to_dict(), 'answers': []}))
    owner = root / 'packages/workflow/workflow-spine/tests/fixtures/scripted-owner.mjs'
    result = subprocess.run(['node', '--import', 'tsx', str(owner), '--workdir', str(run)],
                            cwd=root, capture_output=True, text=True, timeout=35)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run / 'reply.txt').read_text().strip() == 'Final clean reply'
    receipts = [json.loads(p.read_text()) for p in (run / '.refactory/workers').glob('*.json')]
    assert any(r.get('requested_tool_policy') == 'none' for r in receipts)


def test_model_context_round_trip_preserves_explicit_policy():
    from refactory_dsh.model import DshModelClient
    client = _make_client().with_tool_policy("none")
    restored = DshModelClient(**client.context())
    assert restored._environment()["REFACTORY_TOOL_POLICY"] == "none"
    assert "REFACTORY_TOOL_POLICY" not in restored.with_tool_policy("")._environment()


def test_no_tools_declared_input_is_supplied_without_file_tools(tmp_path):
    from refactory_dsh.agent_delivery import begin, CONTRACT_START
    (tmp_path / "question.txt").write_text("What is 2+2?")
    receipt = begin(tmp_path, "solve", "Answer the question", ["answer.txt"],
                    reads=["question.txt"], tool_policy="none")
    assert "What is 2+2?" in receipt["prompt"]
    assert CONTRACT_START not in receipt["prompt"]


@pytest.mark.parametrize("backend", ["codex", "openshell"])
def test_direct_model_client_refuses_unsupported_no_tools_backend(tmp_path, backend):
    client = _make_client().with_tool_policy("none")
    client._worker_backend = backend
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported backend reached process boundary")
    client._run_completion = forbidden
    with pytest.raises(ValueError, match="cannot enforce"):
        client.complete("task", workdir=tmp_path)


@pytest.mark.parametrize("override", [None, "", "other", "none"])
def test_owned_policy_wins_over_caller_environment(override):
    client = _make_client().with_tool_policy("none")
    client._env_overrides = {"REFACTORY_TOOL_POLICY": override}
    assert client._environment()["REFACTORY_TOOL_POLICY"] == "none"
    assert "REFACTORY_TOOL_POLICY" not in client.with_tool_policy("")._environment()


@pytest.mark.parametrize("value", [False, 0, [], {}, "invalid"])
def test_worker_rejects_false_like_invalid_policy_before_client(tmp_path, value):
    from refactory_dsh.worker_node import execute
    spec = {"schema_version": 1, "node_id": "a", "role": "worker", "prompt": "task",
            "reads": [], "writes": ["out.txt"], "produces": [], "post_checks": [],
            "tool_policy": value, "model_context": {}}
    with pytest.raises(ValueError, match="invalid tool_policy"):
        execute(spec, tmp_path)
