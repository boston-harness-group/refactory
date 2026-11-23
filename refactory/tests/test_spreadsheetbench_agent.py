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
from refactory_packages.spreadsheetbench import mode as sb_mode
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


def _act(cwd: Path, runner: list[str], prompt: str, gold: str, write: bool,
         solution: str | None = None) -> tuple[str, str]:
    """What the stub agent does in its workspace: write solution.py (unless ``write`` is
    off), then run the probe through the seal runner, as the adapter's hook does."""
    import re

    task = (cwd / agent_port.TASK_PATH).read_text()
    found = re.findall(r"Put (\d+) in B1\.", task)
    n = int(found[0]) if found else 0
    if write:
        (cwd / agent_port.SOLUTION_PATH).write_text(solution or SOLUTION.format(n=n))
    proc = subprocess.run([*runner, "--", "/bin/bash", "-c", PROBE.format(gold=gold)],
                          cwd=cwd, capture_output=True, text=True, timeout=120)
    reply = ("solution.py was written." if write else
             f"Here.\n```python\n{SOLUTION.format(n=n)}```\n")
    return reply, proc.stdout + proc.stderr


class AgentStub:
    """The walker's model: acts as the agent in the workspace it is given, through the
    seal it is handed (`workspace.ask`), and records what it was sent and saw."""

    thread_safe = True

    def __init__(self, gold: str, write: bool = True, solution: str | None = None) -> None:
        self.gold, self.write, self.solution = gold, write, solution
        self.calls: list[dict[str, Any]] = []
        self.routes: list[Any] = []

    def with_step_route(self, route: Any) -> "AgentStub":
        """The step's own route (`Node.bound_route`), recorded."""
        self.routes.append(dict(route))
        return self

    def complete(self, prompt: str, *, system: str = "", workdir: object = None,
                 cwd: object = None, env: object = None, confine: bool = False,
                 seal: dict[str, Any] | None = None) -> str:
        assert cwd is not None and seal is not None, "the solve step runs sealed, in a workspace"
        cwd = Path(str(cwd))
        listing = sorted(str(p.relative_to(cwd)) for p in cwd.rglob("*") if p.is_file())
        files = {name: (cwd / name).read_bytes() for name in listing}
        reply, seen = _act(cwd, list(seal["runner"]), prompt, self.gold, self.write, self.solution)
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
    # The thinking budget reaches the step's route from its frozen knob (`Node.bound_route`).
    assert model.routes == [{"max_thinking_tokens": str(agent_port.MAX_THINKING_TOKENS)}]
    assert sorted(call["files"]) == sorted(agent_mode.WORKSPACE_READS)
    assert call["files"][agent_port.SKILL_PATH].decode() == \
        agent_port.build_codex_skill(port.INITIAL_SKILL)
    assert call["files"][agent_port.DRIVER_PATH].decode() == agent_port.RUN_SOLUTION
    assert call["files"][agent_port.TASK_PATH].decode().startswith("# Instruction\nPut 7 in B1.")
    assert call["seal"]["tools"] == ["Read", "Bash"] and call["seal"]["backend"] == "claude_code"
    _sealed_and_pinned(call["seen"])
    # One exchange, the reply; SkillOpt's analysts see that and the verification.
    assert [e["response"] for e in record["exchanges"]] == ["solution.py was written."]
    from refactory_packages.skillopt.env import SkillOptEnv

    view = SkillOptEnv.for_mode("spreadsheetbench/agent").rollout_result(
        record, agent_mode.Agent().values())
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


def _drain(tmp_path: Path, client: Any, expect_ok: bool = True, *, part: str = "val",
           train: tuple[str, ...] = ("t1",)) -> Any:
    """The part's items (the val item by default), lowered with dispatched workers on
    ``client``'s route and drained by the real spine; its record (all records, for a part
    of more than one item)."""
    from refactory_dsh import lowering, workers

    _, state = _agent_state()
    work = tmp_path / "spine"
    work.mkdir()
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()
                                       if p != "train"}, train_ids=list(train))
    task_module.write_subset(work, part, 0.0)
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
    assert proc.returncode == 0 or not expect_ok, proc.stdout[-2000:] + proc.stderr[-3000:]
    records = json.loads((work / "facts.json").read_text())["item_scores"]
    if part != "val":
        return records
    (record,) = records
    return record


@needs_spine
def test_one_item_end_to_end_on_the_spine(tmp_path, sb_env):
    """Spine: the same item, lowered and drained, its solve step dispatched to the real
    worker (`worker_node`) on SkillOpt's route. The worker's patch hands the adapter the
    tools, the workspace, the seal runner and the thinking budget; the scripted agent
    works sealed there with the pinned interpreter; the item is graded from solution.py."""
    from refactory_dsh.model import DshModelClient

    cli, log = _agent_dsh(tmp_path, _gold(sb_env, "v1"))
    client = DshModelClient(**AGENT_ROUTE, dsh_cli=str(cli), attempts=1, timeout_s=120)
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
import { mkdirSync, readFileSync, realpathSync, symlinkSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
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
      // FAKE_AGENT_FORBIDDEN: more paths outside the workspace (the dataset, another item):
      // a Read of each, and a `cat` of each through the hooked Bash.
      record.forbidden = {}
      for (const path of JSON.parse(process.env.FAKE_AGENT_FORBIDDEN || '[]')) {
        const read = (await ask('Read', { file_path: path }))?.permissionDecision
        const cat = await ask('Bash', { command: `cat '${path}' >/dev/null 2>&1 && echo READ || echo BLOCKED` })
        record.forbidden[path] = { read, cat: execSync(cat.updatedInput.command, { cwd: options.cwd, shell: '/bin/bash', encoding: 'utf8' }).trim() }
      }
      record.decisions.write_tool = (await ask('Write', { file_path: join(options.cwd, 'x.py'), content: '' }))?.permissionDecision
      // Claude Code's own project folder for this session (a large tool result's file).
      if (process.env.CLAUDE_CONFIG_DIR) {
        const slug = realpathSync(options.cwd).replace(/[^a-zA-Z0-9]/g, '-')
        const saved = join(process.env.CLAUDE_CONFIG_DIR, 'projects', slug, 'fake-session', 'tool-results', 'big.txt')
        mkdirSync(dirname(saved), { recursive: true })
        writeFileSync(saved, 'BIG OUTPUT')
        record.decisions.read_own_output = (await ask('Read', { file_path: saved }))?.permissionDecision
        // Claude Code 2.1.299 makes an empty memory folder in each project folder.
        mkdirSync(join(process.env.CLAUDE_CONFIG_DIR, 'projects', slug, 'memory'), { recursive: true })
        record.ownProject = join(process.env.CLAUDE_CONFIG_DIR, 'projects', slug)
      }
      if (process.env.CLAUDE_CONFIG_DIR && !process.env.FAKE_AGENT_ALONE) {
        const slug = realpathSync(options.cwd).replace(/[^a-zA-Z0-9]/g, '-')
        // Another session's folder of the same name (an earlier attempt's, or a workspace
        // whose name differs only in punctuation): its transcript and its tool results.
        const other = join(process.env.CLAUDE_CONFIG_DIR, 'projects', slug, 'other-session', 'tool-results', 'theirs.txt')
        mkdirSync(dirname(other), { recursive: true })
        writeFileSync(other, 'NOT YOURS')
        writeFileSync(join(process.env.CLAUDE_CONFIG_DIR, 'projects', slug, 'other-session.jsonl'), '{}')
        record.decisions.read_other_session = (await ask('Read', { file_path: other }))?.permissionDecision
        record.decisions.read_other_transcript = (await ask('Read', { file_path: join(process.env.CLAUDE_CONFIG_DIR, 'projects', slug, 'other-session.jsonl') }))?.permissionDecision
        record.ownProject = join(process.env.CLAUDE_CONFIG_DIR, 'projects', slug)
      }
      // Claude Code's own refusals of a call (an unoffered tool, input that does not validate)
      // never reach the hook; their results are the model's errors, not a breach.
      yield { type: 'user', message: { role: 'user', content: [
        { type: 'tool_result', tool_use_id: 'toolu_unoffered', is_error: true,
          content: '<tool_use_error>Error: No such tool available: Write. Write is disabled for this session, in subagents as well as here.</tool_use_error>' },
        { type: 'tool_result', tool_use_id: 'toolu_badinput', is_error: true,
          content: [{ type: 'text', text: '<tool_use_error>InputValidationError: Read failed due to the following issues:\nThe required parameter `file_path` is missing</tool_use_error>' }] }] } }
      // A tool result for a call no hook decided: the seal fails closed.
      if (process.env.FAKE_AGENT_UNHOOKED
          && readFileSync(join(options.cwd, 'task.md'), 'utf8').includes(process.env.FAKE_AGENT_UNHOOKED)) {
        yield { type: 'user', message: { role: 'user', content: [
          { type: 'tool_result', tool_use_id: 'toolu_unhooked', content: 'ran unconfined' }] } }
      }
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
    # The gold again, the dataset's index, another item's input and the item's own
    # golden folder: none is the agent's to read.
    data = sb_env / "sb400"
    monkeypatch.setenv("FAKE_AGENT_FORBIDDEN", json.dumps([
        gold, str(data / "dataset.json"), str(data / "spreadsheet" / "t1" / "1_t1_init.xlsx"),
        str(data / "spreadsheet" / "v1"), "~/.bashrc"]))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
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
                                 "read_link": "deny", "write_tool": "deny",
                                 "read_own_output": "allow", "read_other_session": "deny",
                                 "read_other_transcript": "deny"}
    # The session's own folders go with it; another session's of the same name stay.
    own = Path(sent["ownProject"])
    assert not (own / "fake-session").exists() and (own / "other-session").exists()
    assert not (own / "memory").exists()  # empty, so it went; the folder stays for the other
    _sealed_and_pinned(sent["bash"]["probe"])
    # The golden file, the dataset and other items: neither Bash nor Read reaches them.
    assert len(sent["forbidden"]) == 5
    assert all(v == {"read": "deny", "cat": "BLOCKED"} for v in sent["forbidden"].values()), \
        sent["forbidden"]
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
                            routes_resolved=True, timeout_s=120)
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


@needs_adapter
def test_the_agent_cannot_read_the_gold_the_dataset_or_another_item(tmp_path, sealed_sdk, sb_env):
    """From inside the agent's workspace (walker, real adapter): `cat` of the golden file,
    the dataset's index, another item's input and the item's own folder fails under the
    seal, and a Read of each is refused by the hook; so is a Read through a link to the
    gold. The workspace is SB's tool environment (`SpreadsheetBenchTask.tool_environment`,
    its `hidden` paths refused) like any sealed step's."""
    mode, state = _agent_state()
    work = tmp_path / "walk"
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()
                                       if p != "train"}, train_ids=["t1"])
    task_module.write_subset(work, "val", 0.0)
    result = WalkerEvaluator(model=_claude(tmp_path), project=Path("."), max_steps=400) \
        .evaluate(mode, state, work)
    assert result.exit_ok, result.error
    sent = json.loads(sealed_sdk.read_text())
    gold = _gold(sb_env, "v1")
    assert sent["forbidden"][gold] == {"read": "deny", "cat": "BLOCKED"}
    assert sent["decisions"]["read_outside"] == "deny" and sent["decisions"]["read_link"] == "deny"
    assert "GOLD-HIDDEN" in sent["bash"]["probe"]
    assert all(v == {"read": "deny", "cat": "BLOCKED"} for v in sent["forbidden"].values())
    dispatch = json.loads(next((work / "passes").rglob("workspaces/solve/dispatch.json")).read_text())
    assert dispatch["sealed"] and dispatch["backend"] == "claude_code"
    assert dispatch["tool_environment"]["python"] == os.environ[interpreter.ENV]


GUARD_RUNTIMES = ["walker", pytest.param("spine", marks=needs_spine)]


def _guarded_records(runtime: str, tmp_path: Path) -> dict[str, dict[str, Any]]:
    """Items t1 and v1 through the real adapter on ``runtime``, by id (two items, so a run
    whose one item is a harness incident still ends as runs do)."""
    from refactory_dsh.model import DshModelClient

    if runtime == "spine":
        records = _drain(tmp_path, DshModelClient(
            **AGENT_ROUTE, runtime_root=REPO_ROOT, dsh_home=tmp_path / "dsh-home",
            routes_resolved=True, timeout_s=120), part="train", train=("t1", "v1"))
    else:
        mode, state = _agent_state()
        work = tmp_path / "walk"
        task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()
                                           if p != "train"}, train_ids=["t1", "v1"])
        task_module.write_subset(work, "train", 0.0)
        records = WalkerEvaluator(model=_claude(tmp_path), project=Path("."), max_steps=400) \
            .evaluate(mode, state, work).item_records
    return {r["id"]: r for r in records}


@needs_adapter
@pytest.mark.parametrize("runtime", GUARD_RUNTIMES)
def test_claude_codes_own_refusals_are_not_seal_breaches(tmp_path, sealed_sdk, runtime):
    """A call to a tool the step is not offered, and one whose input does not validate, come
    back as Claude Code's ``<tool_use_error>`` without reaching the hook: the model's
    mistakes, which it sees and goes on from. The item is graded as usual."""
    record = _guarded_records(runtime, tmp_path)["v1"]
    assert record["passed"] and record.get("error_kind") is None, record


@needs_adapter
@pytest.mark.parametrize("runtime", GUARD_RUNTIMES)
def test_a_tool_call_no_hook_decided_fails_closed_as_a_harness_incident(tmp_path, sealed_sdk,
                                                                        monkeypatch, runtime):
    """If Claude Code ever ran a tool without the PreToolUse hook (a version that ignores
    it), the adapter's guard sees a tool result for a call the hook never decided: the
    seal records an incident and the step fails as the harness's failure, never graded."""
    monkeypatch.setenv("FAKE_AGENT_UNHOOKED", "Put 7 in B1.")  # item v1 only
    records = _guarded_records(runtime, tmp_path)
    record = records["v1"]
    assert not record["passed"] and record["error_kind"] == "harness_incident", json.dumps(record)[:3000]
    assert "hook did not decide tool call toolu_unhooked" in record["error_text"], record
    # The other item ran and was graded as usual (its answer, 7, is not t1's 42).
    assert records["t1"].get("error_kind") is None and records["t1"]["fail_reason"].startswith(
        "eval-mismatch"), records["t1"]


# ── route settings from frozen knobs, on the shipped runtime ─────────────────

def _dsh_specs(tmp_path: Path, monkeypatch: Any, mode: Any, client: Any) -> list[dict[str, Any]]:
    """What `DshEvaluator` lowers for ``mode`` with ``client`` (no route injected): the
    dispatched workers' specs. The DSH process itself is not started."""
    from refactory_dsh import dsh_profile, dsh_runtime
    from refactory_dsh.evaluators import DshEvaluator

    class _Exited:
        returncode, stdout, stderr = 1, iter(()), iter(())

        def __init__(self, *a: Any, **k: Any) -> None:
            self.pid = os.getpid()

        def poll(self) -> int:
            return 1

        def wait(self, timeout: Any = None) -> int:
            return 1

    monkeypatch.setattr(dsh_profile, "requirement", lambda **k: None)
    monkeypatch.setattr(dsh_runtime, "require_build", lambda **k: None)
    monkeypatch.setattr(dsh_runtime.subprocess, "Popen", _Exited)
    monkeypatch.setattr(dsh_runtime, "track_run", lambda process: None, raising=False)
    work = tmp_path / f"dsh-{mode.name}"
    DshEvaluator(model=client, timeout_s=5).evaluate(mode, mode.state_dict(), work)
    return [json.loads(p.read_text()) for p in (work / ".refactory" / "worker-specs").glob("*.json")]


def test_dsh_evaluator_carries_each_modes_bound_route_settings(tmp_path, monkeypatch, sb_data):
    """`DshEvaluator` (the spine) with no route injected: the agent's thinking budget and a
    solver given a nonzero output cap each reach the solve worker's route from their frozen
    knobs (`Node.bound_route`); a solver at its default (0, no cap) carries none."""
    from refactory_dsh.model import DshModelClient
    from refactory_dsh.nn.edits import configure
    from refactory_packages.spreadsheetbench import mode as sb_mode

    client = DshModelClient(**AGENT_ROUTE, runtime_root=REPO_ROOT, dsh_home=tmp_path / "home",
                            routes_resolved=True)
    (spec,) = [s for s in _dsh_specs(tmp_path, monkeypatch, agent_mode.Agent(), client)
               if s["node_id"] == "solve"]
    assert spec["model_context"]["max_thinking_tokens"] == str(agent_port.MAX_THINKING_TOKENS)
    solver_client = DshModelClient(worker_backend="claude_code", model="claude-opus-4-6",
                                   permission_mode="dontAsk", runtime_root=REPO_ROOT,
                                   dsh_home=tmp_path / "home", routes_resolved=True)
    (plain,) = [s for s in _dsh_specs(tmp_path, monkeypatch, sb_mode.Solver(), solver_client)
                if s["node_id"] == "solve"]
    assert "max_completion_tokens" not in plain["model_context"]
    capped = sb_mode.Solver(state=configure(sb_mode.Solver().state_dict(), "max_completion_tokens", 4096))
    capped.name = "solver-capped"
    (spec,) = [s for s in _dsh_specs(tmp_path, monkeypatch, capped, solver_client)
               if s["node_id"] == "solve"]
    assert spec["model_context"]["max_completion_tokens"] == "4096"


# ── what each arm's analysts see ──────────────────────────────────────────────

#: Code that fails to run: its error, cut at 500 characters, is what SkillOpt's analysts see.
FAILING = "raise RuntimeError('x' * 700)\n"


def test_an_exec_error_reaches_both_arms_as_skillopts_block(tmp_path, sb_env):
    """rollout.py:799-812: code that fails to run is shown as ``## Execution (case N)``
    and its first 500 characters of error -- the record's prediction, which SkillOpt's view
    puts after the final message and the default loop shows as what the item gave."""
    from refactory_dsh.optim import analysts
    from refactory_packages.skillopt.env import SkillOptEnv

    model = AgentStub(_gold(sb_env, "v1"), solution=FAILING)
    (record,) = _walk(tmp_path, "val", model, expect_ok=False).item_records
    assert record["fail_reason"].startswith("exec-error: RuntimeError"), record
    head = "## Execution (case 1)\nERROR: "
    assert record["prediction"].startswith(head), record
    error = record["prediction"][len(head):]
    assert len(error) == 500 and "Traceback" in error
    view = SkillOptEnv.for_mode("spreadsheetbench/agent").rollout_result(
        record, agent_mode.Agent().values())
    assert view["conversation"][-1]["content"] == f"[POST-EXECUTION VERIFICATION]\n\n{record['prediction']}"
    assert record["prediction"] in analysts.item_text(record)


def test_the_default_loop_is_shown_what_skillopts_analysts_are(tmp_path, sb_env):
    """Information parity for an exec rollout: SkillOpt's analysts see the item's
    instruction, the workbook preview, the final message and the verification (or error)
    block. The default loop's item now carries the preview as its context
    (`SpreadsheetBenchAgentTask.item_context`), shown in full after the input; the solver,
    whose call carries the preview in its prompt, declares none."""
    from refactory_dsh.optim import analysts
    from refactory_dsh.task import item_metadata_reader
    from refactory_packages.spreadsheetbench import mode as sb_mode

    model = AgentStub(_gold(sb_env, "v1"))
    (record,) = _walk(tmp_path, "val", model).item_records
    read = item_metadata_reader(agent_mode.Agent().state_dict(), context=True)
    context = read(record)
    preview = port.preview_workbook(str(sb_env / "sb400" / "spreadsheet" / "v1" / "1_v1_init.xlsx"))
    assert context == {"spreadsheet_preview": preview}
    text = analysts.item_text({**record, "item_context": context})
    assert "Given to the solver (spreadsheet_preview):" in text and preview.strip() in text
    assert "Put 7 in B1." in text and "solution.py was written." in text
    assert record["prediction"] in text
    assert item_metadata_reader(sb_mode.Solver().state_dict(), context=True) is None


# ── the outer loop's new steps ────────────────────────────────────────────────

def test_a_new_read_bash_step_is_refused_when_new_step_tools_is_off():
    """`is_sealed` now takes Claude Code's Read and Bash: a step the search adds with them
    in a fresh workspace is judged as a sealed step (on its route), and with
    ``new_step_tools`` off it is refused like any new step with tools."""
    from refactory_dsh.optim.base import _new_node_tools

    seed = agent_mode.Agent().state_dict()
    candidate = json.loads(json.dumps(seed))
    candidate["graph"]["nodes"]["extra"] = {
        "id": "extra", "kind": "agent", "tools": ["Read", "Bash"], "workspace": "scratch",
        "writes": ["extra.txt"], "prompt_template": "Look."}
    refused = _new_node_tools(seed, candidate, sealed=(True, ""), allowed=False)
    assert refused and "new_step_tools: false" in refused[0]
    assert _new_node_tools(seed, candidate, sealed=(True, ""), check=lambda *a: (True, "")) == []
    assert "cannot run a sealed step" in _new_node_tools(
        seed, candidate, sealed=(True, ""), check=lambda *a: (False, "not claude_code"))[0]


@pytest.mark.parametrize("mode_cls", [agent_mode.Agent, sb_mode.Solver], ids=["agent", "solver"])
def test_the_search_never_binds_a_route_setting_to_a_knob_of_its_own(mode_cls):
    """A proposal that adds an unfrozen knob and binds a step's thinking budget or output
    cap to it is refused (``E-route``); a copy of the solve step, whose binding is to the
    mode's frozen knob, is not refused for it."""
    import structure_fuzz as sf
    from refactory_dsh.optim import Candidate

    applier = sf.Applier(sf.Seed("sb-" + mode_cls.name, mode_cls))
    parent = Candidate.from_state(applier.seed_state)
    out = applier.attempt(parent, {"edits": [
        {"op": "add_knob", "name": "tk", "spec": {"kind": "int", "default": 100000, "lo": 1,
                                                 "hi": 10 ** 7}},
        {"op": "add_node", "node": {
            "id": "extra", "kind": "agent", "prompt_template": "x", "writes": ["e.txt"],
            "knobs": ["tk"], "field_bindings": [["route.max_thinking_tokens", "tk"],
                                                ["route.max_completion_tokens", "tk"]]},
         "position": {"before": "finalize"}}]}, {parent.id})
    assert isinstance(out, list) and any(
        r.startswith("E-route: step 'extra' binds its max_thinking_tokens to knob 'tk'")
        for r in out), out
    from refactory_dsh.optim.base import _route_binding_refusals

    copied = json.loads(json.dumps(applier.seed_state))
    copied["graph"]["nodes"]["solve2"] = dict(copied["graph"]["nodes"]["solve"], id="solve2")
    assert _route_binding_refusals(applier.seed_state, copied) == []
    knob = "max_thinking_tokens" if mode_cls is agent_mode.Agent else "max_completion_tokens"
    copied["knobs"][knob] = dict(copied["knobs"][knob], value=99999)
    assert _route_binding_refusals(applier.seed_state, copied)


@needs_adapter
def test_a_lone_sessions_project_folder_goes_with_it(tmp_path, sealed_sdk, monkeypatch):
    """With no other session of the same folder name, the whole project folder goes: the
    session's own files, then the empty memory folder Claude Code makes, then the folder."""
    monkeypatch.setenv("FAKE_AGENT_ALONE", "1")
    _guarded_records("walker", tmp_path)
    projects = tmp_path / "claude-config" / "projects"
    assert projects.is_dir() and list(projects.iterdir()) == []


def test_a_duplicate_of_the_solve_step_keeps_its_frozen_budget():
    """The real ``duplicate`` edit of the agent's solve step gives the copy a frozen copy of
    its thinking-budget knob (``solve2__route.max_thinking_tokens``, the same spec and
    value): accepted, so the search can copy a solve step without changing its budget; a
    copy whose knob is not frozen, or holds another value, is refused."""
    import structure_fuzz as sf
    from refactory_dsh.optim import Candidate
    from refactory_dsh.optim.base import _route_binding_refusals

    applier = sf.Applier(sf.Seed("sb-agent", agent_mode.Agent))
    parent = Candidate.from_state(applier.seed_state)
    out = applier.attempt(parent, {"op": "duplicate", "target": "solve", "new_id": "solve2"},
                          {parent.id})
    assert isinstance(out, Candidate), out  # the copy keeps its tools, sealed, and budget
    copy = out.state["graph"]["nodes"]["solve2"]
    assert copy["tools"] == ["Read", "Bash"] and copy.get("tool_policy") is None
    from refactory_dsh.nn import edits

    copied = edits.apply_many([{"op": "duplicate", "target": "solve", "new_id": "solve2"}],
                              applier.seed_state)
    knob = "solve2__route.max_thinking_tokens"
    assert copied["knobs"][knob]["spec"]["frozen"]
    assert _route_binding_refusals(applier.seed_state, copied) == []
    for change in ({"value": 99999}, {"spec": {**copied["knobs"][knob]["spec"], "frozen": False}}):
        changed = json.loads(json.dumps(copied))
        changed["knobs"][knob].update(change)
        assert _route_binding_refusals(applier.seed_state, changed), change
