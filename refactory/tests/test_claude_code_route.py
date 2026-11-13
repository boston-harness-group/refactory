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
    import { appendFileSync, writeFileSync } from 'node:fs'
    // Whether the prompt stream ends (the adapter released it) within `ms`.
    const closes = (input, ms) => Promise.race([input.next().then(step => step.done),
                                                new Promise(resolve => setTimeout(() => resolve(false), ms).unref())])
    export function query({ prompt, options }) {
      // FAKE_SDK_CALLS: one line per query, so a test can count the sessions started.
      if (process.env.FAKE_SDK_CALLS) appendFileSync(process.env.FAKE_SDK_CALLS, 'query\\n')
      const child = options.spawnClaudeCodeProcess({
        command: process.execPath, args: ['-e', 'setTimeout(() => {}, 60000)'],
        cwd: options.cwd, env: options.env, signal: options.abortController.signal })
      const record = {
        promptIsStream: typeof prompt !== 'string' && typeof prompt?.[Symbol.asyncIterator] === 'function',
        systemPrompt: options.systemPrompt, effort: options.effort ?? null,
        tools: options.tools ?? null, mcpServers: options.mcpServers ?? null,
        settingSources: options.settingSources ?? null, strictMcpConfig: options.strictMcpConfig ?? null,
        permissionMode: options.permissionMode,
        pathToClaudeCodeExecutable: options.pathToClaudeCodeExecutable ?? null,
        backgroundTasksInChildEnv: options.env.CLAUDE_CODE_DISABLE_BACKGROUND_TASKS ?? null,
        systemPromptInChildEnv: 'REFACTORY_SYSTEM_PROMPT' in process.env,
        controlsInChildEnv: Object.keys(options.env).filter(key => key in CONTROLS) }
      const save = () => writeFileSync(process.env.FAKE_SDK_RECORD, JSON.stringify(record))
      // FAKE_SDK_INIT_TOOLS: the tool list the SDK reports, as when it ignores `tools: []`.
      const tools = process.env.FAKE_SDK_INIT_TOOLS ? JSON.parse(process.env.FAKE_SDK_INIT_TOOLS)
        : options.tools ?? ['Bash', 'Read']
      const usage = { 'fake-claude': { inputTokens: 3, outputTokens: 1,
                                       cacheReadInputTokens: 0, cacheCreationInputTokens: 0 } }
      return {
        close() { child.kill?.() },
        async *[Symbol.asyncIterator]() {
          const input = record.promptIsStream ? prompt[Symbol.asyncIterator]() : null
          const first = input ? (await input.next()).value : null
          record.prompt = input ? first?.message?.content : prompt
          save()
          yield { type: 'system', subtype: 'init', model: 'fake-claude', session_id: 'fake-session', tools }
          if (process.env.FAKE_SDK_BACKGROUND === 'hang') {
            // A task that never reports (a server), and a CLI that stays alive and silent
            // even once its input is closed.
            yield { type: 'system', subtype: 'background_tasks_changed',
                    tasks: [{ task_id: 'server', task_type: 'local_bash', description: 'serve' }] }
            yield { type: 'result', subtype: 'success', is_error: false, result: 'server started', modelUsage: usage }
            await new Promise(() => {})
          } else if (process.env.FAKE_SDK_BACKGROUND) {
            // A turn that launches a background subagent and ends before it reports, as
            // Claude Code does: the input must stay open, and Claude Code continues.
            yield { type: 'system', subtype: 'background_tasks_changed',
                    tasks: [{ task_id: 'bg-1', task_type: 'local_agent', description: 'sleep' }] }
            yield { type: 'result', subtype: 'success', is_error: false, result: 'still waiting', modelUsage: usage }
            record.inputClosedWhileTasksRan = input ? await closes(input, 300) : null
            yield { type: 'system', subtype: 'background_tasks_changed', tasks: [] }
            yield { type: 'system', subtype: 'init', model: 'fake-claude', session_id: 'fake-session', tools }
            yield { type: 'result', subtype: 'success', is_error: false, result: 'FINAL PINEAPPLE', modelUsage: usage }
          } else {
            // FAKE_SDK_RESULT: the result message to end with, as when Claude Code fails.
            yield process.env.FAKE_SDK_RESULT ? JSON.parse(process.env.FAKE_SDK_RESULT)
              : { type: 'result', subtype: 'success', is_error: false, result: 'PINEAPPLE', modelUsage: usage }
          }
          record.inputClosedAfterAnswer = input ? await closes(input, 5000) : null
          save()
        },
      }
    }
    const CONTROLS = { REFACTORY_SYSTEM_PROMPT: 1, REFACTORY_MODEL_EFFORT: 1,
                       REFACTORY_TOOL_POLICY: 1, REFACTORY_SETTING_SOURCES: 1,
                       REFACTORY_CLAUDE_EXECUTABLE: 1, REFACTORY_BACKGROUND_QUIET_S: 1 }
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
    # The SDK's bundled Claude Code, with its background tasks left on.
    assert sent["pathToClaudeCodeExecutable"] is None and sent["backgroundTasksInChildEnv"] is None
    # The prompt is a stream that carries the text and is closed at the answer.
    assert sent["promptIsStream"] is True and sent["inputClosedAfterAnswer"] is True
    receipt = _receipt(workdir)
    assert "enforced_tool_policy" not in receipt
    assert "setting_sources" not in receipt["actual_route"]
    assert "claude_executable" not in receipt["actual_route"]


def test_a_chosen_claude_code_executable_reaches_the_sdk(tmp_path, fake_sdk, monkeypatch):
    """A route's `claude_executable` (`--claude-executable`) is the SDK's
    pathToClaudeCodeExecutable, with background tasks left on;
    REFACTORY_CLAUDE_EXECUTABLE does the same when the route names none."""
    workdir = tmp_path / "run"
    _client(tmp_path, claude_executable="/opt/claude/bin/claude").complete("Say a fruit", workdir=workdir)
    sent = json.loads(fake_sdk.read_text())
    assert sent["pathToClaudeCodeExecutable"] == "/opt/claude/bin/claude"
    assert sent["backgroundTasksInChildEnv"] is None
    assert sent["controlsInChildEnv"] == []
    assert _receipt(workdir)["actual_route"]["claude_executable"] == "/opt/claude/bin/claude"

    monkeypatch.setenv("REFACTORY_CLAUDE_EXECUTABLE", "/env/claude")
    _client(tmp_path).complete("Say a fruit", workdir=tmp_path / "env-run")
    sent = json.loads(fake_sdk.read_text())
    assert sent["pathToClaudeCodeExecutable"] == "/env/claude"
    assert sent["backgroundTasksInChildEnv"] is None


def test_the_walker_reaches_claude_code_with_the_node_settings(tmp_path, fake_sdk):
    """The walker's agent node takes the same path (`nn/executor.py` -> complete)."""
    from refactory_dsh import nn

    graph = nn.Graph()
    graph.add(nn.Node("solve", "agent", prompt_template="Say a {{kind}}",
                      system_prompt_template="Always answer with the word PINEAPPLE",
                      tool_policy="none", writes=("answer.txt",)))
    result = nn.execute(graph, tmp_path / "run", facts={"kind": "fruit"},
                        model=_client(tmp_path, effort="low", claude_executable="/opt/claude/bin/claude"))
    assert result.exit_ok, result.error

    sent = json.loads(fake_sdk.read_text())
    assert sent["prompt"].startswith("Say a fruit")
    assert sent["systemPrompt"]["append"] == "Always answer with the word PINEAPPLE"
    assert sent["effort"] == "low" and sent["tools"] == []
    assert sent["pathToClaudeCodeExecutable"] == "/opt/claude/bin/claude"
    assert sent["backgroundTasksInChildEnv"] is None
    assert _receipt(tmp_path / "run")["enforced_tool_policy"] == "none"


def test_the_spine_worker_reaches_claude_code_with_the_node_settings(tmp_path, fake_sdk):
    """The spine's dispatched worker (`worker_node.execute`) takes the same path, with
    the route carried across its worker spec's model context."""
    from refactory_dsh.worker_node import execute

    workdir = tmp_path / "run"
    workdir.mkdir()
    client = _client(tmp_path, effort="low", setting_sources="user,project",
                     claude_executable="/opt/claude/bin/claude")
    execute({"schema_version": 1, "node_id": "solve", "reads": [], "writes": ["answer.txt"],
             "produces": [], "prompt": "Say a fruit", "tool_policy": "none",
             "system_prompt": "Always answer with the word PINEAPPLE",
             "model_context": client.context()}, workdir)

    sent = json.loads(fake_sdk.read_text())
    assert sent["prompt"].startswith("Say a fruit")
    assert sent["systemPrompt"]["append"] == "Always answer with the word PINEAPPLE"
    assert sent["effort"] == "low" and sent["tools"] == []
    assert sent["settingSources"] == ["user", "project"]
    assert sent["pathToClaudeCodeExecutable"] == "/opt/claude/bin/claude"
    assert sent["backgroundTasksInChildEnv"] is None
    assert _receipt(workdir)["enforced_tool_policy"] == "none"


def test_a_step_waits_for_its_background_tasks_before_it_finishes(tmp_path, fake_sdk, monkeypatch):
    """A turn that launches a background subagent and ends early is not the answer:
    the prompt stays open until a result arrives with no task running, and that later
    result is the node's output (newer Claude Code killed the task and kept the early one)."""
    monkeypatch.setenv("FAKE_SDK_BACKGROUND", "1")
    monkeypatch.setenv("REFACTORY_BACKGROUND_QUIET_S", "30")  # an adapter control, never Claude Code's
    workdir = tmp_path / "run"
    answer = _client(tmp_path).complete("Say a fruit", workdir=workdir)

    assert answer.strip() == "FINAL PINEAPPLE"
    sent = json.loads(fake_sdk.read_text())
    assert sent["inputClosedWhileTasksRan"] is False
    assert sent["inputClosedAfterAnswer"] is True
    assert sent["backgroundTasksInChildEnv"] is None
    assert sent["controlsInChildEnv"] == []
    assert _receipt(workdir)["background_tasks_waited"] == 1


def test_a_silent_background_task_and_cli_end_the_step_with_its_last_answer(tmp_path, fake_sdk, monkeypatch):
    """A task that never reports does not hold the step past REFACTORY_BACKGROUND_QUIET_S,
    and a CLI that then stays alive is stopped: the step completes with its last answer,
    not as aborted or failed, and the record says what was given up."""
    monkeypatch.setenv("FAKE_SDK_BACKGROUND", "hang")
    monkeypatch.setenv("REFACTORY_BACKGROUND_QUIET_S", "0.2")
    workdir = tmp_path / "run"
    answer = _client(tmp_path).complete("Start a server", workdir=workdir)

    assert answer.strip() == "server started"
    receipt = _receipt(workdir)
    assert receipt["state"] == "completed"
    assert receipt["background_tasks_abandoned"] == 1
    assert receipt["background_stream_stopped"] is True


def test_the_adapter_releases_the_prompt_only_when_no_background_task_runs():
    """Scripted SDK streams through the patched `consumeClaudeQuery`: live tasks hold the
    step (a repeated init does not reset them), no tasks or ambient ones do not, and an
    error or a broken stream still releases the prompt."""
    import subprocess
    subprocess.run(["node", str(Path(__file__).with_name("claude_code_background_test.mjs")), str(ROOT)],
                   check=True)


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


def _failed_result(text: str, output_tokens: int) -> str:
    return json.dumps({"type": "result", "subtype": "success", "is_error": True, "result": text,
                       "modelUsage": {"fake-claude": {"inputTokens": 900, "outputTokens": output_tokens,
                                                      "cacheReadInputTokens": 0,
                                                      "cacheCreationInputTokens": 0}}})


#: The notice Claude Code ends a run with when a response is cut off at its output cap
#: and its own resumptions are spent (`claude` 0.3.263, apiError "max_output_tokens").
CUT_OFF = ("API Error: Claude's response exceeded the 32000 output token maximum. To configure "
           "this behavior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable.")


def test_an_output_limit_says_so_on_the_record(tmp_path, fake_sdk, monkeypatch):
    """A run Claude Code ends at its output cap was reported as a bare ``invalid-result``.
    The adapter now names the output limit and the tokens spent, the model client calls
    it an output limit, and the worker receipt and progress event carry the reason."""
    from refactory_dsh.node_retry import failure_kind

    monkeypatch.setenv("FAKE_SDK_RESULT", _failed_result(CUT_OFF, 128000))
    workdir = tmp_path / "run"
    with pytest.raises(RuntimeError) as caught:
        _client(tmp_path).with_tool_policy("none").complete("Solve it", workdir=workdir)
    message = str(caught.value)
    assert "category: output-limit" in message
    assert "output limit reached after 128000 output tokens" in message
    assert "capped at 32000 tokens" in message
    assert getattr(caught.value, "kind", None) == "output_limit"
    assert failure_kind(caught.value) == "output_limit"
    receipt = _receipt(workdir)
    assert receipt["state"] == "failed" and "output limit reached" in receipt["error"]
    events = [json.loads(line) for line in (workdir / ".refactory/events.jsonl").read_text().splitlines()]
    assert any("output limit reached" in (event.get("summary") or "") for event in events)


def test_another_error_result_says_what_it_was(tmp_path, fake_sdk, monkeypatch):
    monkeypatch.setenv("FAKE_SDK_RESULT", _failed_result("API Error: 529 overloaded", 12))
    with pytest.raises(RuntimeError) as caught:
        _client(tmp_path).with_tool_policy("none").complete("Solve it", workdir=tmp_path / "run")
    message = str(caught.value)
    assert "category: invalid-result" in message
    assert "the result was an error after 12 output tokens: API Error: 529 overloaded" in message


def test_a_rate_limited_result_never_replays_a_session_with_tools(tmp_path, fake_sdk, monkeypatch):
    """The adapter's detail quotes the provider ("API Error: 429 ... rate_limit_error"):
    through the real adapter, a session with tools runs once, and the failure is the
    wire's (``transport``), not a crash a node's retry answers. (The dsh-backend case,
    where the client could replay, is in test_crashed_step_retry.)"""
    from refactory_dsh.node_retry import failure_kind

    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_SDK_CALLS", str(calls))
    monkeypatch.setenv("FAKE_SDK_RESULT", _failed_result(
        'API Error: 429 {"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}', 0))
    client = _client(tmp_path, attempts=3, retry_window_s=0).with_tool_policy("")
    with pytest.raises(RuntimeError) as caught:
        client.complete("Edit the file", workdir=tmp_path / "run")
    assert calls.read_text().splitlines() == ["query"]
    assert "rate_limit_error" in str(caught.value)
    assert getattr(caught.value, "kind", None) == "transport"
    assert failure_kind(caught.value) is None
