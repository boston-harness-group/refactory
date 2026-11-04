"""The claude_code route end to end: the real DSH command line, bridge and adapter.

A worker on ``--worker-backend claude_code`` once sent Claude Code the literal prompt
"-" and none of the node's system prompt, effort or tool policy (#493 G40). These run
the command `DshModelClient.complete` builds -- the prompt on stdin behind "-", the
worker patch, the bridge and the patched adapter -- with only the Agent SDK's
``query`` replaced by a fake that records what it was given. The walker
(`nn/executor.py`) and the spine (`worker_node.py`) both reach Claude Code through
`DshModelClient.complete`, so this is the path each of them takes.
"""
from __future__ import annotations

import json
import os
import shutil
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "node_modules" / "@deepseek-ai" / "dsh-subagent-claude-code" / "lib" / "index.js"

pytestmark = pytest.mark.skipif(not ADAPTER.exists() or shutil.which("node") is None,
                                reason="the Claude Code adapter is not installed")

_FAKE_SDK = textwrap.dedent("""\
    // Stands in for @anthropic-ai/claude-agent-sdk: records the query and answers once.
    import { writeFileSync } from 'node:fs'
    export function query({ prompt, options }) {
      const child = options.spawnClaudeCodeProcess({
        command: process.execPath, args: ['-e', 'setTimeout(() => {}, 60000)'],
        cwd: options.cwd, env: options.env, signal: options.abortController.signal })
      writeFileSync(process.env.FAKE_SDK_RECORD, JSON.stringify({
        prompt, systemPrompt: options.systemPrompt, effort: options.effort ?? null,
        tools: options.tools ?? null, mcpServers: options.mcpServers ?? null,
        settingSources: options.settingSources ?? null, strictMcpConfig: options.strictMcpConfig ?? null,
        permissionMode: options.permissionMode,
        systemPromptInChildEnv: 'REFACTORY_SYSTEM_PROMPT' in process.env,
        controlsInChildEnv: Object.keys(options.env).filter(key => key in CONTROLS) }))
      // FAKE_SDK_INIT_TOOLS: the tool list the SDK reports, as when it ignores `tools: []`.
      const tools = process.env.FAKE_SDK_INIT_TOOLS ? JSON.parse(process.env.FAKE_SDK_INIT_TOOLS)
        : options.tools ?? ['Bash', 'Read']
      return {
        close() { child.kill?.() },
        async *[Symbol.asyncIterator]() {
          yield { type: 'system', subtype: 'init', model: 'fake-claude', session_id: 'fake-session', tools }
          yield { type: 'result', subtype: 'success', is_error: false, result: 'PINEAPPLE',
                  modelUsage: { 'fake-claude': { inputTokens: 3, outputTokens: 1,
                                                  cacheReadInputTokens: 0, cacheCreationInputTokens: 0 } } }
        },
      }
    }
    const CONTROLS = { REFACTORY_SYSTEM_PROMPT: 1, REFACTORY_MODEL_EFFORT: 1,
                       REFACTORY_TOOL_POLICY: 1, REFACTORY_SETTING_SOURCES: 1 }
""")

_HOOKS = textwrap.dedent("""\
    export async function resolve(specifier, context, next) {
      if (specifier === '@anthropic-ai/claude-agent-sdk') {
        return { url: new URL('./fake-sdk.mjs', import.meta.url).href, shortCircuit: true }
      }
      return next(specifier, context)
    }
""")

_REGISTER = "import { register } from 'node:module'\nregister('./hooks.mjs', import.meta.url)\n"


@pytest.fixture
def fake_sdk(tmp_path, monkeypatch):
    """Every Node process DSH starts resolves the Agent SDK to the recording fake."""
    hooks = tmp_path / "sdk-hooks"
    hooks.mkdir()
    (hooks / "fake-sdk.mjs").write_text(_FAKE_SDK)
    (hooks / "hooks.mjs").write_text(_HOOKS)
    (hooks / "register.mjs").write_text(_REGISTER)
    record = tmp_path / "sdk-query.json"
    monkeypatch.setenv("NODE_OPTIONS", f"--import {(hooks / 'register.mjs').as_uri()}")
    monkeypatch.setenv("FAKE_SDK_RECORD", str(record))
    monkeypatch.delenv("REFACTORY_WORKER_CONTEXT", raising=False)
    return record


def _client(tmp_path, **route):
    from refactory_dsh.model import DshModelClient
    node_route = {key: route.pop(key) for key in ("setting_sources",) if key in route}
    client = DshModelClient(worker_backend="claude_code", model="claude-test", permission_mode="dontAsk",
                            runtime_root=ROOT, dsh_home=tmp_path / "dsh-home", routes_resolved=True,
                            timeout_s=120, node_routes={"solve": node_route}, **route)
    return client.for_node("solve")


def _receipt(workdir: Path) -> dict:
    [path] = (workdir / ".refactory" / "workers").glob("worker-*.json")
    return json.loads(path.read_text())


def test_claude_code_gets_the_prompt_system_prompt_effort_and_no_tools(tmp_path, fake_sdk):
    workdir = tmp_path / "run"
    client = _client(tmp_path, effort="low").with_tool_policy("none")
    answer = client.complete("Say a fruit", system="Always answer with the word PINEAPPLE", workdir=workdir)

    assert answer.strip() == "PINEAPPLE"
    sent = json.loads(fake_sdk.read_text())
    assert sent["prompt"] == "Say a fruit"  # not the "-" that put it on stdin
    assert sent["systemPrompt"] == {"type": "preset", "preset": "claude_code",
                                    "append": "Always answer with the word PINEAPPLE"}
    assert sent["effort"] == "low"
    assert sent["tools"] == [] and sent["mcpServers"] == {} and sent["strictMcpConfig"] is True
    assert sent["settingSources"] == []  # tool_policy none's default
    assert sent["permissionMode"] == "dontAsk"
    assert sent["controlsInChildEnv"] == []  # adapter controls never reach Claude Code's env

    receipt = _receipt(workdir)
    assert receipt["state"] == "completed"
    assert receipt["actual_route"]["effort"] == "low"
    assert receipt["actual_route"]["model"] == "fake-claude"
    assert receipt["requested_tool_policy"] == "none"
    assert receipt["enforced_tool_policy"] == "none"
    assert receipt["tools_permitted"] == []
    assert receipt["tool_policy_source"] == "claude-agent-sdk-options"


def test_claude_code_route_setting_sources_reach_the_sdk(tmp_path, fake_sdk):
    """A route's setting_sources (e.g. `--setting-sources user,project`) replace the
    default, tool_policy none included, and tools stay disabled."""
    workdir = tmp_path / "run"
    client = _client(tmp_path, setting_sources="user,project").with_tool_policy("none")
    client.complete("Say a fruit", workdir=workdir)

    sent = json.loads(fake_sdk.read_text())
    assert sent["settingSources"] == ["user", "project"]
    assert sent["tools"] == []
    assert sent["effort"] is None
    assert sent["systemPrompt"] == {"type": "preset", "preset": "claude_code"}
    receipt = _receipt(workdir)
    assert receipt["actual_route"]["setting_sources"] == ["user", "project"]
    assert receipt["actual_route"]["effort"] is None
    assert receipt["enforced_tool_policy"] == "none"


def test_claude_code_default_route_keeps_tools_and_settings(tmp_path, fake_sdk):
    workdir = tmp_path / "run"
    _client(tmp_path).complete("Say a fruit", workdir=workdir)

    sent = json.loads(fake_sdk.read_text())
    assert sent["prompt"] == "Say a fruit"
    assert sent["tools"] is None and sent["settingSources"] is None and sent["effort"] is None
    receipt = _receipt(workdir)
    assert "enforced_tool_policy" not in receipt
    assert "setting_sources" not in receipt["actual_route"]


def test_the_walker_reaches_claude_code_with_the_node_settings(tmp_path, fake_sdk):
    """The walker's agent node takes the same path (`nn/executor.py` -> complete)."""
    from refactory_dsh import nn

    graph = nn.Graph()
    graph.add(nn.Node("solve", "agent", prompt_template="Say a {{kind}}",
                      system_prompt_template="Always answer with the word PINEAPPLE",
                      tool_policy="none", writes=("answer.txt",)))
    result = nn.execute(graph, tmp_path / "run", facts={"kind": "fruit"},
                        model=_client(tmp_path, effort="low"))
    assert result.exit_ok, result.error

    sent = json.loads(fake_sdk.read_text())
    assert sent["prompt"].startswith("Say a fruit")
    assert sent["systemPrompt"]["append"] == "Always answer with the word PINEAPPLE"
    assert sent["effort"] == "low" and sent["tools"] == []
    assert _receipt(tmp_path / "run")["enforced_tool_policy"] == "none"


def test_the_spine_worker_reaches_claude_code_with_the_node_settings(tmp_path, fake_sdk):
    """The spine's dispatched worker (`worker_node.execute`) takes the same path, with
    the route carried across its worker spec's model context."""
    from refactory_dsh.worker_node import execute

    workdir = tmp_path / "run"
    workdir.mkdir()
    client = _client(tmp_path, effort="low", setting_sources="user,project")
    execute({"schema_version": 1, "node_id": "solve", "reads": [], "writes": ["answer.txt"],
             "produces": [], "prompt": "Say a fruit", "tool_policy": "none",
             "system_prompt": "Always answer with the word PINEAPPLE",
             "model_context": client.context()}, workdir)

    sent = json.loads(fake_sdk.read_text())
    assert sent["prompt"].startswith("Say a fruit")
    assert sent["systemPrompt"]["append"] == "Always answer with the word PINEAPPLE"
    assert sent["effort"] == "low" and sent["tools"] == []
    assert sent["settingSources"] == ["user", "project"]
    assert _receipt(workdir)["enforced_tool_policy"] == "none"


def test_a_system_prompt_over_the_environment_limit_reaches_claude_code_whole(tmp_path, fake_sdk):
    """A grown skill passes Linux's 128 KiB limit on one environment value; it rides the
    adapter config, never the DSH process environment, and arrives byte-exact."""
    system = ("Skill line \u2014 \u00e9t\u00e9 \u2713 " * 9000) + "END"
    assert len(system.encode("utf-8")) > 128 * 1024
    workdir = tmp_path / "run"
    _client(tmp_path).complete("Say a fruit", system=system, workdir=workdir)

    sent = json.loads(fake_sdk.read_text())
    assert sent["systemPrompt"]["append"] == system
    assert sent["systemPromptInChildEnv"] is False
    assert _receipt(workdir)["state"] == "completed"


def test_tools_the_sdk_reports_despite_no_tools_are_not_recorded_as_enforced(tmp_path, fake_sdk, monkeypatch):
    """The receipt reports what the SDK reported: tools in its init mean not enforced."""
    monkeypatch.setenv("FAKE_SDK_INIT_TOOLS", json.dumps(["Bash"]))
    workdir = tmp_path / "run"
    _client(tmp_path).with_tool_policy("none").complete("Say a fruit", workdir=workdir)

    assert json.loads(fake_sdk.read_text())["tools"] == []
    receipt = _receipt(workdir)
    assert receipt["requested_tool_policy"] == "none"
    assert receipt["enforced_tool_policy"] is None
    assert receipt["tools_permitted"] == ["Bash"]
    assert receipt["tool_observation_available"] is True
