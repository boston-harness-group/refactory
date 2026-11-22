"""SpreadsheetBench agent mode: SkillOpt's Claude Code agent harness, checked against SkillOpt.

Expected values come from SkillOpt's own code, never from the port:
``fixtures/spreadsheetbench_agent_parity.json`` was recorded by calling SkillOpt's
``_prepare_codex_workspace``, ``run_claude_code_exec`` (its CLI path, ``subprocess.run``
captured) and ``run_single`` (its exec branch) on ``sb_support`` workbooks.

Runtimes: the parity checks are pure. The smoke runs one item end to end with a stubbed
model on the walker (`nn.executor`, a scripted client that acts as the agent through the
seal it is handed) and on the TS spine (lowered, its solve step dispatched to the real
worker, `worker_node`, whose scripted ``dsh`` does the same). The adapter checks run the
real DSH command line, bridge and patched Claude Code adapter with only the Agent SDK's
``query`` faked (`test_claude_code_route`), on both runtimes' dispatch paths.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import task as task_module
from refactory_dsh.evaluators import WalkerEvaluator
from refactory_packages.spreadsheetbench import (
    agent_mode, agent_port, interpreter, skillopt_port as port, skillopt_view)
from sb_support import make_workbook
from test_spreadsheetbench import PARTS, sb_data, sb_env, sb_python  # noqa: F401 - fixtures

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "spreadsheetbench_agent_parity.json")
                     .read_text(encoding="utf-8"))
SPECS = json.loads((Path(__file__).parent / "fixtures" / "spreadsheetbench_parity.json")
                   .read_text(encoding="utf-8"))["previews"]
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
NODE = shutil.which("node")


# ── SkillOpt's workspace, prompt and command line, byte for byte ──────────────

@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: f"{c['book']}-{c['skill']}")
def test_workspace_files_and_prompt_are_skillopts(tmp_path, case):
    """SKILL.md, task.md, run_solution.py and the prompt, built from the same item and
    skill, are SkillOpt's; its workspace holds exactly the files the solve step reads."""
    book = make_workbook(tmp_path / f"{case['book']}.xlsx", SPECS[case["book"]]["spec"])
    skill = FIXTURE["skills"][case["skill"]]
    assert agent_port.build_codex_skill(skill) == case["skill_md"]
    assert agent_port.build_codex_task(FIXTURE["instruction"], str(book), FIXTURE["instruction_type"],
                                       FIXTURE["answer_position"]) == case["task_md"]
    assert agent_port.RUN_SOLUTION == case["run_solution"]
    assert agent_port.AGENT_PROMPT == case["prompt"]
    assert sorted(agent_mode.WORKSPACE_READS) == case["files"] and case["same_input"]
    # What SkillOpt's analysts are shown as the target's prompts in this harness.
    assert port.build_system(skill) == case["analyst_system"]
    assert port.build_user(FIXTURE["instruction"], str(book), FIXTURE["instruction_type"],
                           FIXTURE["answer_position"]) == case["analyst_user"]


def test_the_command_line_is_skillopts():
    """`agent_port.cli_argv` is SkillOpt's ``claude`` argv, and the solve step sends what
    it sends: the prompt after ``--``, Read and Bash offered and allowed, bypassPermissions,
    the thinking budget; no appended system prompt. Effort and model are the route's, as
    the docs say (SkillOpt's: medium; the run's model)."""
    (case, *_) = FIXTURE["cases"]
    first, retry = case["attempts"]
    argv = agent_port.cli_argv(path="<claude>", work_dir="<work_dir>", model="claude-opus-4-6",
                               prompt=agent_port.AGENT_PROMPT)
    assert argv == first["cmd"] and first["cwd"] == "<work_dir>"
    assert agent_port.SOLVE_PROMPT == first["cmd"][-1]
    assert FIXTURE["config"] == {"profile": "", "use_sdk": "cli", "effort": agent_port.EFFORT,
                                 "max_thinking_tokens": agent_port.MAX_THINKING_TOKENS,
                                 "empty_response_retries": agent_port.EMPTY_RESPONSE_RETRIES}
    # SkillOpt's one retry, after an empty answer, is the same call with a note appended.
    assert retry["cmd"][:-1] == first["cmd"][:-1] and retry["cmd"][-1].startswith(first["cmd"][-1])
    graph = agent_mode.Agent().build()
    solve = graph.nodes["solve"]
    values = agent_mode.Agent().values()
    assert solve.tools == tuple(first["cmd"][first["cmd"].index("--tools") + 1].split(","))
    assert solve.tools == tuple(first["cmd"][first["cmd"].index("--allowedTools") + 1].split(","))
    assert values[agent_mode.PROMPT_KNOB] == agent_port.SOLVE_PROMPT
    assert values[agent_mode.SYSTEM_PROMPT_KNOB] == ""  # no --append-system-prompt
    assert str(values[agent_mode.THINKING_KNOB]) == \
        first["cmd"][first["cmd"].index("--max-thinking-tokens") + 1]
    assert first["cmd"][first["cmd"].index("--permission-mode") + 1] == "bypassPermissions"
    assert graph.retry_rule("solve")["times"] == agent_port.EMPTY_RESPONSE_RETRIES
    frozen = {name for name, knob in agent_mode.Agent.declare().items() if knob.frozen}
    assert frozen == {"data_parallelism", agent_mode.THINKING_KNOB, agent_mode.PROMPT_KNOB,
                      agent_mode.SYSTEM_PROMPT_KNOB}


@pytest.mark.parametrize("case", FIXTURE["cases"][:1], ids=["one"])
def test_the_items_code_and_conversation_are_skillopts(case):
    """``run_single``'s exec branch: solution.py when the agent left one, else the reply's
    first fenced block; the conversation is the final message alone, one turn."""
    reply = "Done.\n```python\nprint('from the reply')\n```\n"
    for left, key in ((True, "with_solution"), (False, "without_solution")):
        expected = case["run_single"][key]
        solution = "print('from solution.py')\n" if left else None
        assert agent_port.item_code(solution, reply) == expected["code"]
        assert agent_port.conversation(reply) == expected["conversation"]
        assert expected["n_turns"] == 1
    # The CLI path's raw output is the final text: SkillOpt's analysts get no trace steps.
    assert case["trace_steps"] == ""


# ── one item end to end, stubbed model ────────────────────────────────────────

#: The solution the scripted agent writes: B1 from the instruction "Put N in B1."
SOLUTION = ("import openpyxl, re\n"
            "wb = openpyxl.load_workbook(INPUT_PATH)\n"
            "wb.active['B1'] = {n}\n"
            "wb.save(OUTPUT_PATH)\n")
#: What the scripted agent runs in its workspace, sealed: the validation driver, and a
#: look at the interpreter and at what it may read.
PROBE = ("python run_solution.py && test -f output.xlsx && echo VALIDATED; "
         "python -c 'import sys, openpyxl, pandas; print(\"PY\", sys.version_info[:2], "
         "openpyxl.__version__, pandas.__version__)'; "
         "ls -a; cat {gold} >/dev/null 2>&1 && echo GOLD-READ || echo GOLD-HIDDEN")


def _act(cwd: Path, runner: list[str], prompt: str, gold: str, write: bool) -> tuple[str, str]:
    """What the stub agent does in its workspace: write solution.py (unless ``write`` is
    off), then run the probe through the seal runner, as the adapter's hook does."""
    import re

    task = (cwd / agent_port.TASK_PATH).read_text()
    found = re.findall(r"Put (\d+) in B1\.", task)
    n = int(found[0]) if found else 0
    if write:
        (cwd / agent_port.SOLUTION_PATH).write_text(SOLUTION.format(n=n))
    proc = subprocess.run([*runner, "--", "/bin/bash", "-c", PROBE.format(gold=gold)],
                          cwd=cwd, capture_output=True, text=True, timeout=120)
    reply = ("solution.py was written." if write else
             f"Here.\n```python\n{SOLUTION.format(n=n)}```\n")
    return reply, proc.stdout + proc.stderr


class AgentStub:
    """The walker's model: acts as the agent in the workspace it is given, through the
    seal it is handed (`workspace.ask`), and records what it was sent and saw."""

    thread_safe = True

    def __init__(self, gold: str, write: bool = True) -> None:
        self.gold, self.write = gold, write
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, *, system: str = "", workdir: object = None,
                 cwd: object = None, env: object = None, confine: bool = False,
                 seal: dict[str, Any] | None = None) -> str:
        assert cwd is not None and seal is not None, "the solve step runs sealed, in a workspace"
        cwd = Path(str(cwd))
        listing = sorted(str(p.relative_to(cwd)) for p in cwd.rglob("*") if p.is_file())
        files = {name: (cwd / name).read_bytes() for name in listing}
        reply, seen = _act(cwd, list(seal["runner"]), prompt, self.gold, self.write)
        self.calls.append({"prompt": prompt, "system": system, "seal": seal, "files": files,
                           "seen": seen})
        return reply


def _agent_state(**knobs: Any) -> tuple[agent_mode.Agent, dict[str, Any]]:
    from refactory_dsh.nn.edits import configure

    state = agent_mode.Agent().state_dict()
    for name, value in {"data_parallelism": 1, **knobs}.items():
        state = configure(state, name, value)
    return agent_mode.Agent(state=state), state


def _walk(tmp_path: Path, part: str, model: Any, *, expect_ok: bool = True) -> Any:
    mode, state = _agent_state()
    work = tmp_path / f"walk-{part}"
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()
                                       if p != "train"}, train_ids=["t1"])
    task_module.write_subset(work, part, 0.0)
    result = WalkerEvaluator(model=model, project=Path("."), max_steps=400).evaluate(mode, state, work)
    assert result.exit_ok or not expect_ok, result.error
    return result


def _gold(sb_data: Path, item: str) -> str:
    return str(sb_data / "sb400" / "spreadsheet" / item / f"1_{item}_golden.xlsx")


def _sealed_and_pinned(seen: str) -> None:
    """The agent's commands ran sealed: the driver validated, ``python`` was the pinned
    interpreter, the workspace held SkillOpt's files and nothing of the run, and the
    golden file could not be read."""
    assert "VALIDATED" in seen, seen
    assert f"PY (3, 12) {interpreter.PINS['openpyxl']} {interpreter.PINS['pandas']}" in seen, seen
    assert "GOLD-HIDDEN" in seen and "GOLD-READ" not in seen, seen


def test_one_item_end_to_end_on_the_walker(tmp_path, sb_env):
    """Walker: one val item; the solve step is sent SkillOpt's prompt and no system prompt,
    works in a sealed copy of SkillOpt's workspace with the pinned interpreter, writes
    solution.py, and the item is run and graded from it."""
    model = AgentStub(_gold(sb_env, "v1"))
    result = _walk(tmp_path, "val", model)
    (record,) = result.item_records
    assert (record["id"], record["passed"], record["score"]) == ("v1", True, 1.0), record
    (call,) = model.calls
    assert call["prompt"] == agent_port.SOLVE_PROMPT and call["system"] == ""
    assert sorted(call["files"]) == sorted(agent_mode.WORKSPACE_READS)
    assert call["files"][agent_port.SKILL_PATH].decode() == \
        agent_port.build_codex_skill(port.INITIAL_SKILL)
    assert call["files"][agent_port.DRIVER_PATH].decode() == agent_port.RUN_SOLUTION
    assert call["files"][agent_port.TASK_PATH].decode().startswith("# Instruction\nPut 7 in B1.")
    assert call["seal"]["tools"] == ["Read", "Bash"] and call["seal"]["backend"] == "claude_code"
    _sealed_and_pinned(call["seen"])
    # One exchange, the reply; SkillOpt's analysts see that and the verification.
    assert [e["response"] for e in record["exchanges"]] == ["solution.py was written."]
    view = skillopt_view.rollout_result(record, agent_mode.Agent().values())
    assert view["n_turns"] == 1 and view["hard"] == 1
    assert [m["role"] for m in view["conversation"]] == ["assistant", "system"]
    assert view["conversation"][0]["content"] == "solution.py was written."
    assert view["conversation"][1]["content"].startswith(
        "[POST-EXECUTION VERIFICATION]\n\n## Eval Result (case 1): PASS")
    assert view["target_system_prompt"] == port.build_system(port.INITIAL_SKILL)
    assert view["target_user_prompt"].startswith("# Instruction\nPut 7 in B1.")
    assert view["spreadsheet_preview"].startswith("## Sheet: Sheet1")


def test_without_solution_py_the_replys_code_is_graded(tmp_path, sb_env):
    """codegen_agent.py:491-496: no solution.py, so the reply's fenced block is the code."""
    model = AgentStub(_gold(sb_env, "v1"), write=False)
    (record,) = _walk(tmp_path, "val", model).item_records
    assert record["passed"], record


def test_an_item_with_no_case_scores_zero_without_a_call(tmp_path, sb_env):
    model = AgentStub(_gold(sb_env, "v1"))
    (record,) = _walk(tmp_path, "test", model, expect_ok=False).item_records
    assert (record["passed"], record["fail_reason"]) == (False, "no-test-cases")
    assert model.calls == []


# ── the TS spine: lowered, the solve step dispatched to the real worker ───────

#: A scripted ``dsh`` that acts as the agent: it reads the worker patch the model client
#: wrote, takes the seal the claude_code adapter would be given from it, and does what
#: `_act` does in its working directory (the workspace).
AGENT_DSH = '''import json, os, pathlib, re, subprocess, sys
log = pathlib.Path({log!r})
prompt = sys.stdin.read()
argv = sys.argv[1:]
patches = [argv[i + 1] for i, a in enumerate(argv) if a == "--patch"]
config = {{}}
for patch in patches:
    try:
        rows = json.loads(pathlib.Path(patch).read_text())
    except ValueError:
        continue
    for row in rows:
        for inserted in row.get("insert", []) if isinstance(row, dict) else []:
            if inserted.get("id") == "refactory-external-worker":
                config = inserted.get("config", {{}})
env = config.get("env", {{}})
cwd = pathlib.Path.cwd()
runner = json.loads(env["REFACTORY_SEAL_RUNNER"])
files = sorted(str(p.relative_to(cwd)) for p in cwd.rglob("*") if p.is_file())
task = (cwd / "task.md").read_text()
n = int((re.findall(r"Put (\\d+) in B1\\.", task) or ["0"])[0])
(cwd / "solution.py").write_text({solution!r}.format(n=n))
proc = subprocess.run([*runner, "--", "/bin/bash", "-c", {probe!r}], cwd=cwd,
                      capture_output=True, text=True, timeout=120)
with log.open("a") as out:
    out.write(json.dumps({{"prompt": prompt, "config": config, "cwd": str(cwd), "files": files,
                          "seen": proc.stdout + proc.stderr}}) + "\\n")
print("solution.py was written.")
'''

#: The route SkillOpt's harness runs on: Claude Code, bypassPermissions (``skip``).
AGENT_ROUTE = {"worker_backend": "claude_code", "model": "claude-opus-4-6",
               "permission_mode": "skip", "effort": "medium"}


def _agent_dsh(tmp_path: Path, gold: str) -> tuple[Path, Path]:
    root = tmp_path / "dsh"
    root.mkdir(parents=True, exist_ok=True)
    log = root / "calls.jsonl"
    script = root / "agent_dsh.py"
    script.write_text(AGENT_DSH.format(log=str(log), solution=SOLUTION,
                                       probe=PROBE.format(gold=gold)))
    cli = root / "dsh"
    cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
    cli.chmod(0o755)
    return cli, log


needs_spine = pytest.mark.skipif(
    NODE is None or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/"
                         "@deepseek-ai/cordis").exists(),
    reason="the spine drain needs Node and the installed workspace")


def _drain(tmp_path: Path, client: Any) -> dict[str, Any]:
    """The val item, lowered with dispatched workers on ``client``'s route and drained by the
    real spine; its record."""
    from refactory_dsh import lowering, workers

    _, state = _agent_state()
    work = tmp_path / "spine"
    work.mkdir()
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()
                                       if p != "train"}, train_ids=["t1"])
    task_module.write_subset(work, "val", 0.0)
    lowering.write_declarations(state, work)

    def dispatched(workflow: Any) -> Any:
        return workers.lower_workflow(workflow, tmp_path / "specs", client.context(),
                                      lowering.node_declarations(state))

    graph_path, _ = lowering.write_mode_graph(state, tmp_path / "graphs", name="sba",
                                              transform=dispatched)
    (tmp_path / "responses.json").write_text("{}")
    env = {key: value for key, value in os.environ.items() if key != "REFACTORY_WORKER_CONTEXT"}
    env.update(lowering.knob_environment(state), REFACTORY_WORKDIR=str(work),
               PATH=f"{REPO_ROOT / '.venv/bin'}:{env.get('PATH', '')}")
    drain = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
    proc = subprocess.run(
        [str(NODE), "--import", "tsx/esm", str(drain), str(graph_path), str(work),
         str(REPO_ROOT / ".venv/bin/python"), "120000", str(tmp_path / "responses.json")],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600, env=env)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    (record,) = json.loads((work / "facts.json").read_text())["item_scores"]
    return record


#: The thinking budget `Agent.run` puts on the solve step's route; a lowered graph is
#: given the client's routes as the run resolved them.
THINKING_ROUTE = {"solve": {"max_thinking_tokens": str(agent_port.MAX_THINKING_TOKENS)}}


@needs_spine
def test_one_item_end_to_end_on_the_spine(tmp_path, sb_env):
    """Spine: the same item, lowered and drained, its solve step dispatched to the real
    worker (`worker_node`) on SkillOpt's route. The worker's patch hands the adapter the
    tools, the workspace, the seal runner and the thinking budget; the scripted agent
    works sealed there with the pinned interpreter; the item is graded from solution.py."""
    from refactory_dsh.model import DshModelClient

    cli, log = _agent_dsh(tmp_path, _gold(sb_env, "v1"))
    client = DshModelClient(**AGENT_ROUTE, dsh_cli=str(cli), attempts=1, timeout_s=120,
                            node_routes=THINKING_ROUTE)
    record = _drain(tmp_path, client)
    assert (record["id"], record["passed"]) == ("v1", True), record
    (call,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert call["prompt"] == agent_port.SOLVE_PROMPT
    assert call["files"] == sorted(agent_mode.WORKSPACE_READS)
    worker_env = call["config"]["env"]
    assert worker_env["REFACTORY_SEALED_TOOLS"] == "Read,Bash"
    assert worker_env["REFACTORY_SEAL_WORKSPACE"] == call["cwd"]
    assert worker_env["REFACTORY_MAX_THINKING_TOKENS"] == "16384"
    assert worker_env["REFACTORY_MODEL_EFFORT"] == "medium"
    assert "REFACTORY_SYSTEM_PROMPT" not in worker_env and "REFACTORY_TOOL_POLICY" not in worker_env
    assert call["config"]["permissionMode"] == "bypassPermissions"
    _sealed_and_pinned(call["seen"])
    assert record["exchanges"][-1]["response"].strip() == "solution.py was written."


# ── the real adapter: what Claude Code is started with, and the hook ──────────

ADAPTER = REPO_ROOT / "node_modules" / "@deepseek-ai" / "dsh-subagent-claude-code" / "lib" / "index.js"
needs_adapter = pytest.mark.skipif(not ADAPTER.exists() or NODE is None,
                                   reason="the Claude Code adapter is not installed")

#: Stands in for @anthropic-ai/claude-agent-sdk, as the agent: it records the options, then
#: asks the adapter's PreToolUse hook about each tool call the way Claude Code does and
#: runs the Bash calls as Claude Code would run them (the hook's command, by a shell, in
#: the session's working directory): one writes solution.py, one probes the seal. It also
#: asks about a Read inside the workspace, one outside it, one through a link that leads
#: out, and a tool the step does not list.
_SEALED_SDK = r"""
import { execSync } from 'node:child_process'
import { readFileSync, symlinkSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
export function query({ prompt, options }) {
  const child = options.spawnClaudeCodeProcess({
    command: process.execPath, args: ['-e', 'setTimeout(() => {}, 60000)'],
    cwd: options.cwd, env: options.env, signal: options.abortController.signal })
  const record = {
    tools: options.tools ?? null, allowedTools: options.allowedTools ?? null,
    permissionMode: options.permissionMode, cwd: options.cwd,
    additionalDirectories: options.additionalDirectories ?? null,
    settingSources: options.settingSources ?? null, mcpServers: options.mcpServers ?? null,
    strictMcpConfig: options.strictMcpConfig ?? null, effort: options.effort ?? null,
    maxThinkingTokens: options.maxThinkingTokens ?? null, systemPrompt: options.systemPrompt,
    model: options.model ?? null,
    controlsInChildEnv: Object.keys(options.env).filter(key => key.startsWith('REFACTORY_SEAL')
      || key === 'REFACTORY_SEALED_TOOLS' || key === 'REFACTORY_MAX_THINKING_TOKENS'),
    decisions: {}, bash: {} }
  const hook = options.hooks?.PreToolUse?.[0]?.hooks?.[0]
  const ask = async (tool_name, tool_input) => {
    if (!hook) return null
    const out = await hook({ hook_event_name: 'PreToolUse', tool_name, tool_input }, 'id', {})
    return out?.hookSpecificOutput ?? null
  }
  const bash = async (name, command) => {
    const decision = await ask('Bash', { command, description: name })
    record.decisions[name] = decision?.permissionDecision ?? null
    record.bash[name] = decision?.updatedInput?.command
      ? execSync(decision.updatedInput.command, { cwd: options.cwd, shell: '/bin/bash', encoding: 'utf8' })
      : null
  }
  return {
    close() { child.kill?.() },
    async *[Symbol.asyncIterator]() {
      const input = prompt[Symbol.asyncIterator]()
      record.prompt = (await input.next()).value?.message?.content
      yield { type: 'system', subtype: 'init', model: 'fake-claude', session_id: 'fake-session',
              tools: options.tools ?? [] }
      await bash('write', process.env.FAKE_AGENT_WRITE)
      await bash('probe', process.env.FAKE_AGENT_PROBE)
      const inside = await ask('Read', { file_path: join(options.cwd, 'task.md') })
      record.decisions.read_inside = inside?.permissionDecision
      record.readInsideGiven = inside?.updatedInput?.file_path ?? null
      record.readInsideText = record.readInsideGiven ? readFileSync(record.readInsideGiven, 'utf8') : null
      record.decisions.read_relative = (await ask('Read', { file_path: 'task.md' }))?.permissionDecision
      record.decisions.read_outside = (await ask('Read', { file_path: process.env.FAKE_AGENT_OUTSIDE }))?.permissionDecision
      symlinkSync(process.env.FAKE_AGENT_OUTSIDE, join(options.cwd, 'leak.xlsx'))
      record.decisions.read_link = (await ask('Read', { file_path: join(options.cwd, 'leak.xlsx') }))?.permissionDecision
      record.decisions.write_tool = (await ask('Write', { file_path: join(options.cwd, 'x.py'), content: '' }))?.permissionDecision
      writeFileSync(process.env.FAKE_SDK_RECORD, JSON.stringify(record))
      const usage = { 'fake-claude': { inputTokens: 3, outputTokens: 1, cacheReadInputTokens: 0, cacheCreationInputTokens: 0 } }
      yield { type: 'result', subtype: 'success', is_error: false, result: 'solution.py was written.', modelUsage: usage }
    },
  }
}
"""


@pytest.fixture
def sealed_sdk(tmp_path, monkeypatch, sb_env):
    """Every Node process DSH starts resolves the Agent SDK to the sealed agent above."""
    from test_claude_code_route import _HOOKS, _REGISTER

    hooks = tmp_path / "sdk-hooks"
    hooks.mkdir()
    (hooks / "fake-sdk.mjs").write_text(_SEALED_SDK)
    (hooks / "hooks.mjs").write_text(_HOOKS)
    (hooks / "register.mjs").write_text(_REGISTER)
    record = tmp_path / "sdk-query.json"
    gold = _gold(sb_env, "v1")
    monkeypatch.setenv("NODE_OPTIONS", f"--import {(hooks / 'register.mjs').as_uri()}")
    monkeypatch.setenv("FAKE_SDK_RECORD", str(record))
    # The answer is 7 (item v1): written through the seal, as the agent's Bash would.
    monkeypatch.setenv("FAKE_AGENT_WRITE", "cat > solution.py <<'EOF'\n" + SOLUTION.format(n=7) + "EOF")
    monkeypatch.setenv("FAKE_AGENT_PROBE", PROBE.format(gold=gold))
    monkeypatch.setenv("FAKE_AGENT_OUTSIDE", gold)
    monkeypatch.delenv("REFACTORY_WORKER_CONTEXT", raising=False)
    return record


def _claude(tmp_path: Path):
    from refactory_dsh.model import DshModelClient

    return DshModelClient(**AGENT_ROUTE, runtime_root=REPO_ROOT, dsh_home=tmp_path / "dsh-home",
                          routes_resolved=True, timeout_s=120)


def _adapter_sent(record: Path) -> dict[str, Any]:
    """What the adapter started Claude Code with, and what its hook decided: SkillOpt's
    tools, allowed, bypassPermissions, the workspace as cwd and added directory, no
    settings or MCP, the thinking budget and effort, Claude Code's own system prompt, the
    prompt byte for byte; each Bash run sealed with the pinned interpreter, a Read only in
    the workspace, any other tool refused."""
    sent = json.loads(record.read_text())
    assert sent["tools"] == sent["allowedTools"] == ["Read", "Bash"]
    assert sent["permissionMode"] == "bypassPermissions"
    assert sent["additionalDirectories"] == [sent["cwd"]]
    assert sent["settingSources"] == [] and sent["mcpServers"] == {} and sent["strictMcpConfig"]
    assert sent["maxThinkingTokens"] == agent_port.MAX_THINKING_TOKENS and sent["effort"] == "medium"
    assert sent["systemPrompt"] == {"type": "preset", "preset": "claude_code"}
    assert sent["model"] == "claude-opus-4-6"
    assert sent["prompt"] == agent_port.SOLVE_PROMPT
    assert sent["controlsInChildEnv"] == []
    assert sent["decisions"] == {"write": "allow", "probe": "allow", "read_inside": "allow",
                                 "read_relative": "allow", "read_outside": "deny",
                                 "read_link": "deny", "write_tool": "deny"}
    _sealed_and_pinned(sent["bash"]["probe"])
    # A Read inside is given a copy of the file it checked, beside the workspace (Linux).
    if sys.platform.startswith("linux"):
        given = Path(sent["readInsideGiven"])
        assert given.name == "task.md" and not given.is_relative_to(Path(sent["cwd"]).resolve())
        assert sent["readInsideText"].startswith("# Instruction\nPut 7 in B1.")
    return sent


@needs_adapter
def test_the_adapter_runs_the_walkers_solve_step_sealed(tmp_path, sealed_sdk):
    """Walker, through the real DSH command line, bridge and patched adapter: the item is
    solved by the (faked) Claude Code session, its Bash confined by the hook."""
    mode, state = _agent_state()
    work = tmp_path / "walk"
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()
                                       if p != "train"}, train_ids=["t1"])
    task_module.write_subset(work, "val", 0.0)
    result = WalkerEvaluator(model=_claude(tmp_path), project=Path("."), max_steps=400) \
        .evaluate(mode, state, work)
    assert result.exit_ok, result.error
    (record,) = result.item_records
    assert record["passed"], record
    _adapter_sent(sealed_sdk)


@needs_spine
@needs_adapter
def test_the_adapter_runs_the_spines_solve_step_sealed(tmp_path, sealed_sdk):
    """Spine: the dispatched worker (`worker_node`) through the real DSH command line,
    bridge and patched adapter, as on the walker."""
    from refactory_dsh.model import DshModelClient

    client = DshModelClient(**AGENT_ROUTE, runtime_root=REPO_ROOT, dsh_home=tmp_path / "dsh-home",
                            routes_resolved=True, timeout_s=120, node_routes=THINKING_ROUTE)
    record = _drain(tmp_path, client)
    assert record["passed"], record
    _adapter_sent(sealed_sdk)


# ── the outer loop on it: SkillOpt's strategy and the default loop ───────────

class LoopStub(AgentStub):
    """One client for a whole optimization: the agent in its workspace (`AgentStub`), and
    the optimizer's calls answered as the SkillOpt tests script them (`FakeClient`)."""

    def __init__(self, gold: str) -> None:
        from test_skillopt import FakeClient

        super().__init__(gold)
        self.optimizer = FakeClient()

    def complete(self, prompt: str, *, system: str = "", workdir: object = None,
                 cwd: object = None, env: object = None, confine: bool = False,
                 seal: dict[str, Any] | None = None) -> str:
        if cwd is None:
            return self.optimizer.complete(prompt, system=system)
        return super().complete(prompt, system=system, workdir=workdir, cwd=cwd, env=env,
                                confine=confine, seal=seal)


def _loop_config(tmp_path: Path, **overrides: Any) -> Any:
    from refactory_dsh.loop import LoopConfig

    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps({"train": PARTS["train"], "val": PARTS["val"]}))
    return LoopConfig(**{"epochs": 1, "batch_size": 1, "optimizer": "random", "seed": 3,
                         "evaluator": "walker", "runs_root": tmp_path / "run",
                         "parts_manifest": str(parts), "max_evaluations": 20, **overrides})


def test_skillopts_strategy_optimizes_the_skill(tmp_path, sb_env):
    """SkillOpt's strategy runs an epoch on the agent mode (walker): its train rollout
    reaches the agent, its analysts see SkillOpt's exec view of each item, and the skill
    it edits is the agent's SKILL.md; the prompts stay frozen."""
    from refactory_dsh import strategies

    model = LoopStub(_gold(sb_env, "v1"))
    loop = strategies.create("skillopt", "spreadsheetbench/agent",
                             _loop_config(tmp_path, strategy="skillopt", allowed_ops=("set_knob",)),
                             strategy_options={"max_epochs": 1}, project=Path("."), model=model)
    loop.run()
    analyst = [c for c in model.optimizer.calls if (c["call_key"] or ("",))[0].startswith("analyst_")]
    assert analyst, [c["call_key"] for c in model.optimizer.calls]
    # The analysts get the exec harness's view: the final message, then the verification.
    assert any("solution.py was written." in c["prompt"]
               and "[POST-EXECUTION VERIFICATION]" in c["prompt"] for c in analyst)
    # ... and, as the target's prompts, the chat path's (rollout.py:700-718), preview included.
    assert all("# Input spreadsheet preview\n## Sheet: Sheet1" in c["prompt"]
               and port.CODEGEN_SYSTEM in c["prompt"] for c in analyst)
    skills = {c["files"][agent_port.SKILL_PATH].decode() for c in model.calls}
    assert agent_port.build_codex_skill(port.INITIAL_SKILL) in skills
    assert any("## New Rule\nAlways check." in skill for skill in skills), \
        "a later rollout's SKILL.md carries the strategy's edit"
    assert {c["prompt"] for c in model.calls} == {agent_port.SOLVE_PROMPT}
    assert {c["system"] for c in model.calls} == {""}


def test_the_default_loop_evaluates_the_agent_mode(tmp_path, sb_env):
    """The default loop's own evaluations run the agent mode (walker): train (where the
    Task offers gold feedback, which this graph has no node to ask for: SkillOpt's single
    path takes none) and val."""
    from refactory_dsh.loop import Loop
    from refactory_dsh.outerloop import Run

    model = AgentStub(_gold(sb_env, "v1"))
    loop = Loop("spreadsheetbench/agent", _loop_config(tmp_path), project=Path("."), model=model)
    loop.init()
    seed = loop.population[0]
    loop.forward(epoch=0)
    assert seed.evaluated and len(model.calls) == 2  # t1 and t2
    assert all(c["prompt"] == agent_port.SOLVE_PROMPT for c in model.calls)
    Run(_loop=loop).evaluate(seed, part="val")
    assert len(model.calls) == 3
