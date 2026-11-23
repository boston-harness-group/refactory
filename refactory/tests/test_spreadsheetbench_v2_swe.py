"""SpreadsheetBench 2 swe mode: V2's own agent setup, checked against V2's own files.

Expected texts come from SpreadsheetBench 2's repository, never from the port:
``fixtures/spreadsheetbench_v2_swe_upstream.json`` holds its ``SWE-agent/config/spreadsheet.yaml``
templates as ``yaml.safe_load`` reads them and its ``view_xlsx`` and ``submit`` tools, with
the files' sha256.

Runtimes: the prompt checks are pure. The smoke runs one item end to end with a stubbed
model on the walker (`nn.executor`, a scripted agent acting through the seal it is handed)
and on the TS spine (lowered, its solve step dispatched to the real worker, whose scripted
``dsh`` does the same). The turn cap runs the real DSH command line, bridge and patched
Claude Code adapter with only the Agent SDK's ``query`` faked, on both runtimes.
LibreOffice is a stand-in (`office`), except in `test_libreoffice_runs_inside_the_seal`,
which needs a real one (``SB_SOFFICE``, ``SB2_REAL_SOFFICE`` or on the PATH).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import seal
from refactory_dsh import task as task_module
from refactory_dsh.evaluators import WalkerEvaluator
from refactory_packages.spreadsheetbench import recalc
from refactory_packages.spreadsheetbench_v2 import skillopt_view, swe_mode, swe_ops, swe_port, tasks
from sb2_support import make_workbook
from test_spreadsheetbench import sb_python  # noqa: F401 - fixture
from test_spreadsheetbench_agent import AGENT_ROUTE, _claude, needs_adapter, needs_spine
from test_spreadsheetbench_v2 import (  # noqa: F401 - fixtures
    PARTS, VAL, _gold, _rel, fake_lo, sb2_data, sb2_env)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
UPSTREAM = json.loads((HERE / "fixtures" / "spreadsheetbench_v2_swe_upstream.json")
                      .read_text(encoding="utf-8"))
#: The swe sessions get a /tmp of their own, which only Linux's seal gives.
needs_linux = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                 reason="the swe sessions' own /tmp is Linux's")


# ── the prompt, the system prompt and view_xlsx are V2's ──────────────────────

def _unported(text: str) -> str:
    """``text`` with every substitution undone, last first."""
    for old, new, _why in reversed(swe_port.SUBSTITUTIONS):
        assert text.count(new) == 1, new
        text = text.replace(new, old)
    return text


def test_the_prompt_is_v2s_instance_template_with_the_listed_substitutions():
    """The solve step's prompt is V2's instance template with `swe_port.SUBSTITUTIONS`
    applied and nothing else: undoing them gives V2's template back, byte for byte. The
    appended system prompt is V2's system template; the turn cap is V2's call cap."""
    assert swe_port.UPSTREAM_INSTANCE_TEMPLATE == UPSTREAM["instance_template"]
    assert swe_port.ported(UPSTREAM["instance_template"]) == swe_port.INSTANCE_TEMPLATE
    assert _unported(swe_port.INSTANCE_TEMPLATE) == UPSTREAM["instance_template"]
    values = swe_mode.Swe().values()
    assert values[swe_mode.PROMPT_KNOB] == swe_port.INSTANCE_TEMPLATE
    assert values[swe_mode.SYSTEM_PROMPT_KNOB] == UPSTREAM["system_template"]
    assert values[swe_mode.TURNS_KNOB] == UPSTREAM["per_instance_call_limit"] == 50
    solve = swe_mode.Swe().build().nodes["solve"]
    assert solve.tools == ("Read", "Bash") and solve.workspace["kind"] == "scratch"
    assert dict(solve.field_bindings)["route.max_turns"] == swe_mode.TURNS_KNOB
    frozen = {name for name, knob in swe_mode.Swe.declare().items() if knob.frozen}
    assert frozen == {"data_parallelism", swe_mode.TURNS_KNOB, swe_mode.PROMPT_KNOB,
                      swe_mode.SYSTEM_PROMPT_KNOB}
    # Upstream's Important, Environment and Recommended Workflow sections, unchanged.
    for section in ("## Important\n", "## Environment\n", "## Recommended Workflow\n"):
        head = UPSTREAM["instance_template"].split(section, 1)[1].split("\n\n", 1)[0]
        assert section + head in swe_port.INSTANCE_TEMPLATE


def test_view_xlsx_is_v2s_and_reads_a_small_workbook(tmp_path):
    """The shipped view_xlsx is V2's file byte for byte; run as the sessions run it, it
    lists the sheets and shows a sheet's values and formulas."""
    shipped = Path(swe_port.__file__).with_name("swe_tools") / "view_xlsx"
    assert hashlib.sha256(shipped.read_bytes()).hexdigest() == \
        UPSTREAM["sha256"]["SWE-agent/tools/view_xlsx/bin/view_xlsx"]
    assert swe_port.VIEW_XLSX == UPSTREAM["view_xlsx"]
    book = make_workbook(tmp_path / "input.xlsx", [
        {"title": "Data", "cells": {"A1": "x", "B1": 2, "B2": "=B1*3"}},
        {"title": "Notes", "cells": {"A1": "n"}}])
    script = tmp_path / "view_xlsx"
    script.write_text(swe_port.VIEW_XLSX)
    listed = subprocess.run([sys.executable, str(script), str(book), "list"],
                            capture_output=True, text=True, timeout=60)
    assert "Sheets: ['Data', 'Notes']" in listed.stdout, listed.stdout + listed.stderr
    shown = subprocess.run([sys.executable, str(script), str(book), "content", "Data"],
                           capture_output=True, text=True, timeout=60)
    assert "Row 1: ['x', 2]" in shown.stdout and "Row 2: [None, '=B1*3']" in shown.stdout, \
        shown.stdout + shown.stderr


def test_an_empty_skill_is_not_mentioned():
    """The default skill is empty, and with it the prompt is V2's template (the
    substitutions aside) with the item's values: no skill, no SKILL.md, no section. A
    skill adds one section before ``</task_context>`` that names SKILL.md."""
    assert swe_mode.Swe().values()["skill"] == ""
    bare = swe_port.render("Put 7 in B1.", "")
    assert "skill" not in bare.lower()
    expected = (swe_port.ported(UPSTREAM["instance_template"]).replace("{{skill_section}}", "")
                .replace("{{instruction}}", "Put 7 in B1.")
                .replace("{{spreadsheet_path}}", "input.xlsx")
                .replace("{{output_path}}", "output.xlsx"))
    assert bare == expected
    assert swe_port.render("Put 7 in B1.", "  \n") == bare
    skilled = swe_port.render("Put 7 in B1.", "Check the totals.")
    assert skilled == bare.replace("\n</task_context>",
                                   "\n" + swe_port.skill_section("x") + "</task_context>")
    assert "`SKILL.md`" in skilled and "Check the totals." not in skilled


# ── one item end to end, stubbed model ────────────────────────────────────────

#: What the scripted agent runs, each a separate sealed command as a session's Bash calls
#: are: view the input, write a script to /tmp, run it to write output.xlsx, look at
#: LibreOffice and at the golden file.
WRITE = ("cat > /tmp/script.py << 'EOF'\n"
         "import openpyxl\n"
         "wb = openpyxl.load_workbook('input.xlsx')\n"
         "wb.active['B1'] = {n}\n"
         "wb.save('output.xlsx')\n"
         "EOF")
COMMANDS = ("view_xlsx input.xlsx list", WRITE, "python3 /tmp/script.py && echo WROTE",
            "soffice --version; command -v libreoffice >/dev/null && echo HAS-LIBREOFFICE",
            "cat {gold} >/dev/null 2>&1 && echo GOLD-READ || echo GOLD-HIDDEN")


def _act(cwd: Path, runner: list[str], gold: str, n: int | None) -> list[str]:
    commands = COMMANDS if n is not None else (COMMANDS[0], COMMANDS[3], COMMANDS[4])
    seen = []
    for command in commands:
        proc = subprocess.run([*runner, "--", "/bin/bash", "-c",
                               command.format(n=n, gold=gold)],
                              cwd=cwd, capture_output=True, text=True, timeout=120)
        seen.append(proc.stdout + proc.stderr)
    return seen


class SweStub:
    """The walker's model: acts as the agent through the seal it is handed, and records
    what it was sent, given and saw. ``n``: the value it writes to B1; None writes no
    output."""

    thread_safe = True

    def __init__(self, gold: str, n: int | None = 7) -> None:
        self.gold, self.n = gold, n
        self.calls: list[dict[str, Any]] = []
        self.routes: list[Any] = []

    def with_step_route(self, route: Any) -> "SweStub":
        self.routes.append(dict(route))
        return self

    def complete(self, prompt: str, *, system: str = "", workdir: object = None,
                 cwd: object = None, env: object = None, confine: bool = False,
                 seal: dict[str, Any] | None = None) -> str:
        assert cwd is not None and seal is not None, "the solve step runs sealed, in a workspace"
        cwd = Path(str(cwd))
        files = {str(p.relative_to(cwd)): p.read_bytes() for p in cwd.rglob("*") if p.is_file()}
        seen = _act(cwd, list(seal["runner"]), self.gold, self.n)
        self.calls.append({"prompt": prompt, "system": system, "seal": seal, "files": files,
                           "seen": seen})
        return "Done."


@pytest.fixture()
def office(tmp_path, monkeypatch, sb2_env):
    """A LibreOffice stand-in installed as LibreOffice is (``<install>/program/soffice``, its
    ``python`` beside it copying the file as the grader's helper), outside the run: the
    sessions get its install folder."""
    program = tmp_path / "office" / "lo" / "program"
    program.mkdir(parents=True)
    (program / "soffice").write_text("#!/bin/sh\necho \"LibreOffice stand-in $MAX_CONCURRENCY\"\n")
    (program / "python").write_text('#!/bin/sh\nfor a; do src=$dst; dst=$a; done\n'
                                    'cp "$src" "$dst" && echo \'{"ok": true}\'\n')
    for name in ("soffice", "python"):
        (program / name).chmod(0o755)
    monkeypatch.setenv(recalc.SOFFICE_ENV, str(program / "soffice"))
    return program


def _state(**knobs: Any) -> dict[str, Any]:
    from refactory_dsh.nn.edits import configure

    state = swe_mode.Swe().state_dict()
    for name, value in {"data_parallelism": 1, **knobs}.items():
        state = configure(state, name, value)
    return state


def _walk(tmp_path: Path, model: Any, *, expect_ok: bool = True, **knobs: Any) -> Any:
    state = _state(**knobs)
    work = tmp_path / "walk"
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()})
    task_module.write_subset(work, "val", 0.0)
    result = WalkerEvaluator(model=model, project=Path("."), max_steps=400).evaluate(
        swe_mode.Swe(state=state), state, work)
    assert result.exit_ok or not expect_ok, result.error
    return result


@needs_linux
def test_one_item_end_to_end_on_the_walker(tmp_path, sb2_env, office):
    """Walker: the val item; the session is sent V2's prompt with the item's instruction and
    V2's system template, with the turn cap on its route; it works sealed in a workspace
    holding input.xlsx and an empty SKILL.md, with view_xlsx, LibreOffice and a /tmp kept
    across its commands; the output.xlsx it writes is graded by V2's grade."""
    model = SweStub(_gold(sb2_env))
    (record,) = _walk(tmp_path, model).item_records
    assert (record["id"], record["passed"], record["score"]) == (VAL, True, 1.0), record
    assert record["prediction"].startswith("## Eval Result: PASS")
    (call,) = model.calls
    assert call["prompt"] == swe_port.render("Put 7 in B1.", "")
    assert call["system"] == swe_port.SYSTEM_TEMPLATE
    assert model.routes == [{"max_turns": "50"}]
    assert sorted(call["files"]) == sorted(swe_mode.WORKSPACE_READS)
    assert call["files"][swe_port.SKILL_PATH] == b""
    assert call["seal"]["tools"] == ["Read", "Bash"] and call["seal"]["backend"] == "claude_code"
    listed, _write, wrote, office_seen, gold = call["seen"]
    assert "Sheets: ['Sheet1']" in listed, listed
    assert "WROTE" in wrote, wrote
    assert "LibreOffice stand-in 4" in office_seen and "HAS-LIBREOFFICE" in office_seen, office_seen
    assert "GOLD-HIDDEN" in gold and "GOLD-READ" not in gold, gold
    assert [e["response"] for e in record["exchanges"]] == ["Done."]


@needs_linux
def test_the_skill_is_the_agents_skill_md(tmp_path, sb2_env, office):
    model = SweStub(_gold(sb2_env))
    (record,) = _walk(tmp_path, model, skill="Check the totals.").item_records
    (call,) = model.calls
    assert call["files"][swe_port.SKILL_PATH] == b"Check the totals."
    assert call["prompt"] == swe_port.render("Put 7 in B1.", "Check the totals.")
    assert record["passed"], record


@needs_linux
def test_no_output_xlsx_fails_the_item(tmp_path, sb2_env, office):
    """A session that leaves no output.xlsx fails, saying so; both arms' analysts see why."""
    from refactory_packages.skillopt.env import SkillOptEnv

    (record,) = _walk(tmp_path, SweStub(_gold(sb2_env), n=None), expect_ok=False).item_records
    assert (record["passed"], record["score"]) == (False, 0.0)
    assert record["fail_reason"] == swe_ops.NO_OUTPUT == \
        "output-not-found: the agent left no output.xlsx"
    view = SkillOptEnv.for_mode("spreadsheetbench_v2/swe").rollout_result(
        record, swe_mode.Swe().values())
    assert view["conversation"] == [
        {"role": "assistant", "content": "Done."},
        {"role": "system", "content": f"[POST-EXECUTION VERIFICATION]\n\n{swe_ops.NO_OUTPUT}"}]


@needs_linux
def test_a_wrong_answer_reaches_both_arms(tmp_path, sb2_env, office):
    """V2's mismatch is the record's prediction; SkillOpt's view shows it after the final
    message, with V2's system template and the prompt the session was sent; the default
    loop's item context is the input's preview."""
    from refactory_dsh.task import item_metadata_reader
    from refactory_packages.skillopt.env import SkillOptEnv
    from refactory_packages.spreadsheetbench import skillopt_port as port

    (record,) = _walk(tmp_path, SweStub(_gold(sb2_env), n=8), expect_ok=False).item_records
    assert "Modification error at Sheet1!B1: answer=7, output=8" in record["prediction"]
    assert skillopt_view.MODE_VIEWS["swe"] is skillopt_view.swe_rollout_result
    view = SkillOptEnv.for_mode("spreadsheetbench_v2/swe").rollout_result(
        record, {**swe_mode.Swe().values(), "skill": "Check the totals."})
    assert view["conversation"][-1]["content"] == \
        f"[POST-EXECUTION VERIFICATION]\n\n{record['prediction']}"
    assert view["target_system_prompt"] == swe_port.SYSTEM_TEMPLATE
    assert view["target_user_prompt"] == swe_port.render("Put 7 in B1.", "Check the totals.")
    preview = port.preview_workbook(str(sb2_env / _rel(2)[0]))
    assert view["spreadsheet_preview"] == preview and view["task_type"] == "Template"
    context = item_metadata_reader(swe_mode.Swe().state_dict(), context=True)(record)
    assert context == {"spreadsheet_preview": preview}


# ── the TS spine: lowered, the solve step dispatched to the real worker ───────

#: A scripted ``dsh`` that acts as the agent: it takes the seal the claude_code adapter
#: would be given from the worker patch, runs the commands through it in its working
#: directory (the workspace), and logs what it was sent.
SWE_DSH = '''import json, pathlib, re, subprocess, sys
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
n = int((re.findall(r"Put (\\d+) in B1\\.", prompt) or ["0"])[0])
seen = []
for command in {commands!r}:
    proc = subprocess.run([*runner, "--", "/bin/bash", "-c", command.format(n=n, gold="/nonexistent")],
                          cwd=cwd, capture_output=True, text=True, timeout=120)
    seen.append(proc.stdout + proc.stderr)
with log.open("a") as out:
    out.write(json.dumps({{"prompt": prompt, "config": config, "seen": seen,
                          "files": files}}) + "\\n")
print("Done.")
'''


def _drain(tmp_path: Path, client: Any) -> Any:
    """The val item, lowered with dispatched workers on ``client``'s route and drained by
    the real spine (as `test_spreadsheetbench_v2._drain`, on the swe mode)."""
    from refactory_dsh import lowering, workers

    state = _state()
    work = tmp_path / "spine"
    work.mkdir()
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()})
    task_module.write_subset(work, "val", 0.0)
    lowering.write_declarations(state, work)

    def dispatched(workflow: Any) -> Any:
        return workers.lower_workflow(workflow, tmp_path / "specs", client.context(),
                                      lowering.node_declarations(state))

    graph_path, _ = lowering.write_mode_graph(state, tmp_path / "graphs", name="sb2s",
                                              transform=dispatched)
    (tmp_path / "responses.json").write_text("{}")
    env = {key: value for key, value in os.environ.items() if key != "REFACTORY_WORKER_CONTEXT"}
    env.update(lowering.knob_environment(state), REFACTORY_WORKDIR=str(work),
               PATH=f"{REPO_ROOT / '.venv/bin'}:{env.get('PATH', '')}")
    drain = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
    proc = subprocess.run(
        [shutil.which("node"), "--import", "tsx/esm", str(drain), str(graph_path), str(work),
         str(REPO_ROOT / ".venv/bin/python"), "120000", str(tmp_path / "responses.json")],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600, env=env)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    (record,) = json.loads((work / "facts.json").read_text())["item_scores"]
    return record


@needs_linux
@needs_spine
def test_one_item_end_to_end_on_the_spine(tmp_path, sb2_env, office):
    """Spine: the same item, its solve step dispatched to the real worker on the agent
    mode's route; the worker's patch hands the adapter the tools, the workspace, the seal
    runner, V2's system template and the turn cap; the output.xlsx is graded."""
    from refactory_dsh.model import DshModelClient

    root = tmp_path / "dsh"
    root.mkdir()
    log = root / "calls.jsonl"
    script = root / "swe_dsh.py"
    script.write_text(SWE_DSH.format(log=str(log), commands=list(COMMANDS)))
    cli = root / "dsh"
    cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
    cli.chmod(0o755)
    client = DshModelClient(**AGENT_ROUTE, dsh_cli=str(cli), attempts=1, timeout_s=120)
    record = _drain(tmp_path, client)
    assert (record["id"], record["passed"]) == (VAL, True), record
    (call,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert call["prompt"].strip() == swe_port.render("Put 7 in B1.", "").strip()
    worker_env = call["config"]["env"]
    assert worker_env["REFACTORY_MAX_TURNS"] == "50"
    assert worker_env["REFACTORY_SYSTEM_PROMPT"] == swe_port.SYSTEM_TEMPLATE
    assert worker_env["REFACTORY_SEALED_TOOLS"] == "Read,Bash"
    assert call["files"] == sorted(swe_mode.WORKSPACE_READS)
    assert "Sheets: ['Sheet1']" in call["seen"][0] and "WROTE" in call["seen"][2], call["seen"]


# ── the real adapter: the turn cap ────────────────────────────────────────────

#: Stands in for @anthropic-ai/claude-agent-sdk, as the agent: it records the options, runs
#: FAKE_SWE_COMMANDS as Claude Code runs Bash calls (the hook's command, by a shell, in the
#: session's working directory), then ends as Claude Code does at its turn cap
#: (``error_max_turns``, no text) or with a reply.
_SWE_SDK = r"""
import { execSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'
export function query({ prompt, options }) {
  const child = options.spawnClaudeCodeProcess({
    command: process.execPath, args: ['-e', 'setTimeout(() => {}, 60000)'],
    cwd: options.cwd, env: options.env, signal: options.abortController.signal })
  const record = { maxTurns: options.maxTurns ?? null, systemPrompt: options.systemPrompt,
    tools: options.tools ?? null, bash: [],
    controlsInChildEnv: Object.keys(options.env).filter(key => key === 'REFACTORY_MAX_TURNS'
      || key === 'REFACTORY_SYSTEM_PROMPT' || key.startsWith('REFACTORY_SEAL')) }
  const hook = options.hooks?.PreToolUse?.[0]?.hooks?.[0]
  return {
    close() { child.kill?.() },
    async *[Symbol.asyncIterator]() {
      const input = prompt[Symbol.asyncIterator]()
      record.prompt = (await input.next()).value?.message?.content
      yield { type: 'system', subtype: 'init', model: 'fake-claude', session_id: 'fake-session',
              tools: options.tools ?? [] }
      for (const [i, command] of JSON.parse(process.env.FAKE_SWE_COMMANDS || '[]').entries()) {
        const out = (await hook({ hook_event_name: 'PreToolUse', tool_name: 'Bash',
                                  tool_input: { command } }, `toolu_${i}`, {}))?.hookSpecificOutput
        record.bash.push(out?.updatedInput?.command
          ? execSync(out.updatedInput.command, { cwd: options.cwd, shell: '/bin/bash', encoding: 'utf8' })
          : null)
      }
      writeFileSync(process.env.FAKE_SDK_RECORD, JSON.stringify(record))
      const usage = { 'fake-claude': { inputTokens: 3, outputTokens: 1, cacheReadInputTokens: 0, cacheCreationInputTokens: 0 } }
      if (process.env.FAKE_SWE_END === 'max_turns') {
        yield { type: 'result', subtype: 'error_max_turns', is_error: true, errors: [], num_turns: 50, modelUsage: usage }
      } else {
        yield { type: 'result', subtype: 'success', is_error: false, result: 'Done.', modelUsage: usage }
      }
    },
  }
}
"""


@pytest.fixture
def swe_sdk(tmp_path, monkeypatch, sb2_env, office):
    """Every Node process DSH starts resolves the Agent SDK to the agent above, which writes
    the item's answer (7) to output.xlsx through the seal, then stops at its turn cap."""
    from test_claude_code_route import _HOOKS, _REGISTER

    hooks = tmp_path / "sdk-hooks"
    hooks.mkdir()
    (hooks / "fake-sdk.mjs").write_text(_SWE_SDK)
    (hooks / "hooks.mjs").write_text(_HOOKS)
    (hooks / "register.mjs").write_text(_REGISTER)
    record = tmp_path / "sdk-query.json"
    monkeypatch.setenv("NODE_OPTIONS", f"--import {(hooks / 'register.mjs').as_uri()}")
    monkeypatch.setenv("FAKE_SDK_RECORD", str(record))
    monkeypatch.setenv("FAKE_SWE_COMMANDS", json.dumps(
        [WRITE.format(n=7), "python3 /tmp/script.py && echo WROTE"]))
    monkeypatch.setenv("FAKE_SWE_END", "max_turns")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    monkeypatch.delenv("REFACTORY_WORKER_CONTEXT", raising=False)
    return record


def _capped(record: dict[str, Any], sent_path: Path) -> None:
    """The session was started with the turn cap and V2's system template appended,
    stopped at the cap, and the output.xlsx it left was graded; the item's answer is the
    adapter's note."""
    assert record["passed"], record
    assert record["exchanges"][-1]["response"].strip() == \
        "(Claude Code stopped the session at its turn limit of 50 turns.)"
    sent = json.loads(sent_path.read_text())
    assert sent["maxTurns"] == 50 and sent["controlsInChildEnv"] == []
    assert sent["systemPrompt"] == {"type": "preset", "preset": "claude_code",
                                    "append": swe_port.SYSTEM_TEMPLATE}
    assert sent["tools"] == ["Read", "Bash"]
    assert sent["prompt"] == swe_port.render("Put 7 in B1.", "")
    assert "WROTE" in sent["bash"][1], sent["bash"]


@needs_adapter
@needs_linux
def test_the_turn_cap_ends_the_session_and_its_output_is_graded_on_the_walker(tmp_path, swe_sdk):
    """Walker, through the real DSH command line, bridge and patched adapter: Claude Code is
    started with maxTurns 50 (``--max-turns``); stopped there, the step ends with the
    adapter's note, and the output.xlsx it left is graded."""
    state = _state()
    work = tmp_path / "walk"
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()})
    task_module.write_subset(work, "val", 0.0)
    result = WalkerEvaluator(model=_claude(tmp_path), project=Path("."), max_steps=400) \
        .evaluate(swe_mode.Swe(state=state), state, work)
    assert result.exit_ok, result.error
    (record,) = result.item_records
    _capped(record, swe_sdk)


@needs_linux
@needs_spine
@needs_adapter
def test_the_turn_cap_ends_the_session_and_its_output_is_graded_on_the_spine(tmp_path, swe_sdk):
    from refactory_dsh.model import DshModelClient

    client = DshModelClient(**AGENT_ROUTE, runtime_root=REPO_ROOT, dsh_home=tmp_path / "dsh-home",
                            routes_resolved=True, timeout_s=120)
    _capped(_drain(tmp_path, client), swe_sdk)


# ── the seal's /tmp, and a real LibreOffice inside it ─────────────────────────

@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="a private /tmp is Linux's")
def test_a_private_tmp_is_the_steps_own_and_otherwise_read_only(tmp_path):
    """With ``tmp``, a sealed command writes /tmp, a later command reads it, and it is the
    step's folder beside the workspace; without, /tmp is read only as before."""
    ws = tmp_path / "ws"
    ws.mkdir()
    write = "echo kept > /tmp/note && echo OK"
    plain = seal.tool_path(sys.executable, tmp_path / "bin-a", workspace=ws)
    proc = subprocess.run([*seal.runner_command(ws, None, plain), "--", "/bin/bash", "-c", write],
                          cwd=ws, capture_output=True, text=True, timeout=120)
    assert "OK" not in proc.stdout and "Read-only file system" in proc.stderr, proc.stderr
    own = seal.tool_path(sys.executable, tmp_path / "bin-b", workspace=ws, tmp=tmp_path / "own-tmp")
    runner = seal.runner_command(ws, None, own)
    first = subprocess.run([*runner, "--", "/bin/bash", "-c", write], cwd=ws,
                           capture_output=True, text=True, timeout=120)
    assert "OK" in first.stdout, first.stdout + first.stderr
    second = subprocess.run([*runner, "--", "/bin/bash", "-c", "cat /tmp/note"], cwd=ws,
                            capture_output=True, text=True, timeout=120)
    assert second.stdout.strip() == "kept", second.stdout + second.stderr
    assert (tmp_path / "own-tmp" / "note").read_text().strip() == "kept"


def test_a_tool_environment_may_not_show_the_run_or_shadow_python(tmp_path):
    """The further folders are refused as the interpreter's are (one holding the run), and
    a command may not take python's name."""
    ws = tmp_path / "run" / "ws"
    ws.mkdir(parents=True)
    with pytest.raises(seal.SealError, match="may not read"):
        seal.tool_path(sys.executable, tmp_path / "bin", workspace=ws, runs=[tmp_path / "run"],
                       reads=[str(tmp_path)])
    with pytest.raises(seal.SealError, match="cannot name a command"):
        seal.tool_path(sys.executable, tmp_path / "bin", workspace=ws, commands={"python3": ""})
    made = seal.tool_path(sys.executable, tmp_path / "bin", workspace=ws,
                          commands={"hello": "#!/bin/sh\necho hi\n"})
    assert os.access(Path(made.bin) / "hello", os.X_OK)


def _real_office() -> str | None:
    return os.environ.get("SB2_REAL_SOFFICE") or recalc.soffice()


@pytest.mark.skipif(not sys.platform.startswith("linux") or _real_office() is None,
                    reason="needs Linux and LibreOffice (or SB2_REAL_SOFFICE naming one)")
def test_libreoffice_runs_inside_the_seal(tmp_path, monkeypatch, sb_python):
    """With a real LibreOffice, a sealed command of the swe Task's tool environment
    converts a workbook headless, recalculating its formula, and view_xlsx reads the result;
    its profile is in the workspace (the command's HOME), its pipe in the step's /tmp."""
    monkeypatch.setenv(recalc.SOFFICE_ENV, _real_office())
    ws = tmp_path / "ws"
    ws.mkdir()
    make_workbook(ws / "input.xlsx", [{"title": "S", "cells": {"A1": 2, "A2": "=A1*3"}}])
    environment = tasks.SpreadsheetBenchV2SweTask.tool_environment
    made = seal.tool_path(sb_python, tmp_path / "bin", workspace=ws,
                          commands=environment.commands(), reads=environment.reads(),
                          tmp=tmp_path / "own-tmp" if environment.private_tmp else None)
    runner = seal.runner_command(ws, None, made)
    command = ("soffice --headless --calc --convert-to xlsx --outdir out input.xlsx >/dev/null 2>&1; "
               "python3 -c \"import openpyxl; print('A2 =', openpyxl.load_workbook('out/input.xlsx', "
               "data_only=True)['S']['A2'].value)\"; view_xlsx out/input.xlsx list; "
               "test -d .config/libreoffice && echo PROFILE-HERE")
    proc = subprocess.run([*runner, "--", "/bin/bash", "-c", command], cwd=ws,
                          capture_output=True, text=True, timeout=300)
    assert "A2 = 6" in proc.stdout, proc.stdout + proc.stderr
    assert "Sheets: ['S']" in proc.stdout and "PROFILE-HERE" in proc.stdout, proc.stdout
