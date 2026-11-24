"""SpreadsheetBench 2 steps mode: V2's agent setup with its workflow lifted into four steps.

Expected texts come from SpreadsheetBench 2's repository (``fixtures/
spreadsheetbench_v2_swe_upstream.json``, its ``spreadsheet.yaml`` as ``yaml.safe_load``
reads it), never from the port: each step's prompt is rebuilt from it and the swe
substitutions.

Runtimes: the prompt and graph checks are pure. One item runs end to end with a stubbed
model on the walker (`nn.executor`, a scripted agent per step acting through the seal it is
handed) and on the TS spine (lowered, each step dispatched to the real worker, whose
scripted ``dsh`` does the same). The turn counts and caps run the real DSH command line,
bridge and patched Claude Code adapter with only the Agent SDK's ``query`` faked, on both
runtimes. LibreOffice is a stand-in (`office`).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import task as task_module
from refactory_dsh.evaluators import WalkerEvaluator
from refactory_dsh.workspace import SEALED_SESSION, is_session_log
from refactory_packages.spreadsheetbench_v2 import (
    skillopt_view, steps_mode, steps_ops, steps_port, swe_mode, swe_port, tasks)
from test_spreadsheetbench import sb_python  # noqa: F401 - fixture
from test_spreadsheetbench_agent import AGENT_ROUTE, _claude, needs_adapter, needs_spine
from test_spreadsheetbench_v2 import (  # noqa: F401 - fixtures
    PARTS, VAL, _gold, fake_lo, sb2_data, sb2_env)
from test_spreadsheetbench_v2_swe import ROUTE, office  # noqa: F401 - fixture

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
UPSTREAM = json.loads((HERE / "fixtures" / "spreadsheetbench_v2_swe_upstream.json")
                      .read_text(encoding="utf-8"))
STEPS = ("inspect", "plan", "implement", "verify")
needs_linux = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                 reason="the sessions' own /tmp is Linux's")


# ── the seed graph and each step's prompt, from V2's template ─────────────────

def _upstream_lines(first: str, until: str | None) -> str:
    """Upstream's instance template from the line starting ``first`` up to the line
    starting ``until`` (or the end of the workflow, the blank line before
    ``</task_context>``), with the swe substitutions that fall in it."""
    text = UPSTREAM["instance_template"]
    start = text.index("\n" + first) + 1
    end = text.index("\n" + until) + 1 if until else text.index("\n</task_context>") + 1
    return text[start:end]


def test_the_seed_graph_is_four_sealed_steps_in_a_row():
    """The graph validates; prepare -> has_case -> inspect -> plan -> implement -> verify
    -> finalize, no loops, judges or routes; every step is swe's sealed solve setup with its
    own task and system-prompt knobs bound, the shared frozen limits on its route, and the
    files it reads and writes."""
    from refactory_dsh.nn.validate import validate_state

    mode = steps_mode.Steps()
    state = mode.state_dict()
    assert validate_state(state, invariants=mode.invariants()) == []
    graph = mode.build()
    assert sorted(graph.nodes) == sorted(["data", "prepare", "has_case", *STEPS, "finalize"])
    edges = sorted((e.source, e.target, e.condition or "") for e in graph.edges)
    assert edges == sorted([("prepare", "has_case", ""), ("has_case", "inspect", "proceed"),
                            ("has_case", "finalize", "skip"), ("inspect", "plan", ""),
                            ("plan", "implement", ""), ("implement", "verify", ""),
                            ("verify", "finalize", "")])
    solve = swe_mode.Swe().build().nodes["solve"]
    reads = {"inspect": [], "plan": ["inspect_notes.md"],
             "implement": ["inspect_notes.md", "plan.md"],
             "verify": ["inspect_notes.md", "plan.md", "output.xlsx"]}
    writes = {"inspect": ["inspect_notes.md"], "plan": ["plan.md"],
              "implement": ["output.xlsx"], "verify": ["verify_report.md", "output.xlsx"]}
    for step in STEPS:
        node = graph.nodes[step]
        assert (node.kind, node.tools, node.workspace, node.role, node.retry) == \
            ("agent", solve.tools, solve.workspace, solve.role, solve.retry)
        assert list(node.reads) == ["input.xlsx", "SKILL.md", *reads[step]]
        assert list(node.writes) == [f"{step}_reply.txt", *writes[step],
                                     f"{step}/{SEALED_SESSION}"]
        assert dict(node.field_bindings) == {
            "prompt_template": f"{step}__task", "system_prompt_template": f"{step}__system_prompt",
            "route.max_turns": "max_turns", "route.bash_timeout_ms": "bash_timeout_ms",
            "route.bash_output_chars": "bash_output_chars"}
    # Inspect and plan cannot leave an output.xlsx in the pass.
    assert "output.xlsx" not in graph.nodes["inspect"].writes + graph.nodes["plan"].writes
    specs = steps_mode.Steps.declare()
    frozen = {name for name, knob in specs.items() if knob.frozen}
    assert frozen == {"data_parallelism", "max_turns", "bash_timeout_ms", "bash_output_chars"}
    values = mode.values()
    assert values["max_turns"] == UPSTREAM["per_instance_call_limit"] == 50
    assert values["bash_timeout_ms"] == UPSTREAM["execution_timeout"] * 1000
    assert values["bash_output_chars"] == UPSTREAM["max_observation_length"]
    assert values["skill"] == ""
    assert mode.protected_nodes() == ("data", "finalize")
    assert steps_mode.Steps().claude_code_versions() == {
        step: swe_mode.Swe().claude_code_versions()["solve"] for step in STEPS}


def test_each_steps_prompt_is_v2s_template_with_its_own_stages():
    """Each step's ``<step>__task`` default is V2's instance template (with swe's
    substitutions) whose workflow keeps only the step's stages, byte for byte as V2 wrote
    them, then the step's own section; its ``<step>__system_prompt`` is V2's system
    template. Put back the whole workflow in place of the step's stages and section and
    the swe prompt comes back; the steps' stages in order are V2's whole workflow."""
    values = steps_mode.Steps().values()
    stages = {"inspect": ("1. **Inspect**", "2. **Plan**"),
              "plan": ("2. **Plan**", "3. **Implement**"),
              "implement": ("3. **Implement**", "5. **Verify**"),
              "verify": ("5. **Verify**", None)}
    ported_workflow = swe_port.ported(UPSTREAM["instance_template"]).split(
        "## Recommended Workflow\n", 1)[1].split("{{skill_section}}</task_context>", 1)[0]
    for step in STEPS:
        task = values[f"{step}__task"]
        assert values[f"{step}__system_prompt"] == UPSTREAM["system_template"]
        assert steps_port.WORKFLOW[step] in ported_workflow
        assert task == swe_port.INSTANCE_TEMPLATE.replace(
            ported_workflow, steps_port.WORKFLOW[step] + steps_port.STEP_SECTIONS[step])
        assert task.replace(steps_port.WORKFLOW[step] + steps_port.STEP_SECTIONS[step],
                            ported_workflow) == swe_port.INSTANCE_TEMPLATE
        # No other step's stages.
        for other in STEPS:
            if other != step:
                assert steps_port.WORKFLOW[other] not in task
    assert "".join(steps_port.WORKFLOW[s] for s in STEPS) == ported_workflow
    # Unported, each step's stages are V2's own lines.
    for step in STEPS:
        upstream = _upstream_lines(*stages[step])
        fragment = steps_port.WORKFLOW[step]
        for old, new, _why in swe_port.SUBSTITUTIONS:
            if new in fragment and old not in fragment:
                fragment = fragment.replace(new, old)
        assert fragment == upstream, step


def test_each_step_section_names_its_files_and_every_prompt_renders():
    """Rendered for an item, every step's prompt carries the instruction and paths, the
    skill section only with a skill, and its own section naming what it reads and writes;
    nothing is left unfilled."""
    for step in STEPS:
        bare = steps_port.render(step, "Put 7 in B1.")
        assert "**Instruction:** Put 7 in B1." in bare and "{{" not in bare
        assert "SKILL.md" not in bare and "## Skill" not in bare
        skilled = steps_port.render(step, "Put 7 in B1.", "Check the totals.")
        assert skilled == bare.replace("\n</task_context>",
                                       "\n" + swe_port.skill_section("x") + "</task_context>")
        assert f"step {STEPS.index(step) + 1} of 4" in bare
    assert "write your notes to `inspect_notes.md`" in steps_port.render("inspect", "x")
    assert "Do not write output.xlsx" in steps_port.render("inspect", "x")
    assert "Do not write output.xlsx" in steps_port.render("plan", "x")
    assert "`plan.md`" in steps_port.render("plan", "x")
    assert "then write output.xlsx" in steps_port.render("implement", "x")
    verify = steps_port.render("verify", "x")
    assert "correct output.xlsx where it is wrong" in verify
    assert "`verify_report.md`" in verify and "LibreOffice" in verify


def test_a_session_log_in_a_folder_is_a_session_log():
    """Each step keeps its own session facts (``<step>/.refactory-session.json``); the
    collector and the edit operators treat them as swe's one log."""
    assert is_session_log(SEALED_SESSION) and is_session_log(f"plan/{SEALED_SESSION}")
    assert not is_session_log("plan.md") and not is_session_log(f"{SEALED_SESSION}.bak")


def test_a_default_view_never_hands_on_a_steps_session_log():
    """A judge looped over a step sees what the step wrote, never its session log."""
    from test_judge_tools import _attempt, _nodes

    seed = steps_mode.Steps().state_dict()["graph"]["nodes"]
    for step in ("implement", "verify"):
        nodes = _nodes(_attempt([{"op": "loop", "step": step, "judge": "agent"}],
                                mode=steps_mode.Steps()))
        reads = [r for k, n in nodes.items() if k not in seed for r in (n.get("reads") or ())]
        assert "output.xlsx" in reads and not any(is_session_log(r) for r in reads), reads


# ── one item end to end, stubbed model ────────────────────────────────────────

def _step_of(prompt: str) -> str:
    (n,) = re.findall(r"This is step (\d) of 4", prompt)
    return STEPS[int(n) - 1]


def _write(n: int) -> str:
    return ("cat > /tmp/script.py << 'EOF'\n"
            "import openpyxl\n"
            "wb = openpyxl.load_workbook('input.xlsx')\n"
            f"wb.active['B1'] = {n}\n"
            "wb.save('output.xlsx')\n"
            "EOF\n"
            "python3 /tmp/script.py && echo WROTE")


def _commands(implement: int | None, verify: int | None, delete: bool = False) -> dict[str, list[str]]:
    """What each scripted step runs: inspect notes the sheets (and tries to leave an
    output.xlsx, which it does not declare), plan reads the notes and writes a plan,
    implement writes ``implement`` to B1 (None: nothing), verify reads B1, rewrites it to
    ``verify`` (None: leaves it) or deletes the output, and writes its report."""
    check = ("python3 -c \"import openpyxl; "
             "print('B1 IS', openpyxl.load_workbook('output.xlsx').active['B1'].value)\"")
    return {
        "inspect": ["view_xlsx input.xlsx list > inspect_notes.md && echo NOTED",
                    "cp input.xlsx output.xlsx && echo INSPECT-OUTPUT"],
        "plan": ["cat inspect_notes.md", "echo 'set B1' > plan.md && echo PLANNED"],
        "implement": ["cat plan.md", *([_write(implement)] if implement is not None else [])],
        "verify": [check, *([_write(verify)] if verify is not None else []),
                   *(["rm output.xlsx"] if delete else []),
                   "echo checked > verify_report.md"],
    }


class StepsStub:
    """The walker's model: acts as each step through the seal it is handed, choosing the
    step's commands from its prompt; records what it was sent, given and saw."""

    thread_safe = True

    def __init__(self, commands: dict[str, list[str]], silent: tuple[str, ...] = ()) -> None:
        self.commands = commands
        #: Steps whose every session ends with no reply text.
        self.silent = silent
        self.calls: list[dict[str, Any]] = []
        self.routes: list[Any] = []

    def with_step_route(self, route: Any) -> "StepsStub":
        self.routes.append(dict(route))
        return self

    def complete(self, prompt: str, *, system: str = "", workdir: object = None,
                 cwd: object = None, env: object = None, confine: bool = False,
                 seal: dict[str, Any] | None = None) -> str:
        assert cwd is not None and seal is not None, "every step runs sealed, in a workspace"
        cwd = Path(str(cwd))
        step = _step_of(prompt)
        files = {str(p.relative_to(cwd)): p.read_bytes() for p in cwd.rglob("*") if p.is_file()}
        seen = []
        for command in self.commands[step]:
            proc = subprocess.run([*seal["runner"], "--", "/bin/bash", "-c", command], cwd=cwd,
                                  capture_output=True, text=True, timeout=120)
            seen.append(proc.stdout + proc.stderr)
        self.calls.append({"step": step, "prompt": prompt, "system": system, "files": files,
                           "seen": seen, "tools": seal["tools"]})
        return "" if step in self.silent else f"{step} done."


def _state(**knobs: Any) -> dict[str, Any]:
    from refactory_dsh.nn.edits import configure

    state = steps_mode.Steps().state_dict()
    for name, value in {"data_parallelism": 1, **knobs}.items():
        state = configure(state, name, value)
    return state


def _walk(tmp_path: Path, model: Any, *, expect_ok: bool = True, **knobs: Any) -> Any:
    state = _state(**knobs)
    work = tmp_path / "walk"
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()})
    task_module.write_subset(work, "val", 0.0)
    result = WalkerEvaluator(model=model, project=Path("."), max_steps=400).evaluate(
        steps_mode.Steps(state=state), state, work)
    assert result.exit_ok or not expect_ok, result.error
    return result


def _final(root: Path) -> dict[str, Any]:
    (final,) = [json.loads(p.read_text()) for p in root.rglob("final.json")]
    return final


@needs_linux
def test_files_flow_through_the_steps_and_verify_corrects_the_output(tmp_path, sb2_env, office):
    """Walker: each step gets its own prompt and V2's system template, sealed with Bash
    alone and the frozen limits on its route; its workspace holds what it reads, the
    earlier steps' files among them (inspect's stray output.xlsx is never handed on);
    implement writes 8, verify reads it, corrects it to 7, and the corrected output is
    graded."""
    model = StepsStub(_commands(implement=8, verify=7))
    (record,) = _walk(tmp_path, model).item_records
    assert (record["id"], record["passed"], record["score"]) == (VAL, True, 1.0), record
    assert [c["step"] for c in model.calls] == list(STEPS)
    assert model.routes == [ROUTE] * 4
    by = {c["step"]: c for c in model.calls}
    for step in STEPS:
        assert by[step]["prompt"] == steps_port.render(step, "Put 7 in B1.")
        assert by[step]["system"] == swe_port.SYSTEM_TEMPLATE
        assert by[step]["tools"] == ["Bash"]
        assert by[step]["files"]["SKILL.md"] == b""
    assert sorted(by["inspect"]["files"]) == ["SKILL.md", "input.xlsx"]
    assert sorted(by["plan"]["files"]) == ["SKILL.md", "input.xlsx", "inspect_notes.md"]
    assert b"Sheets: ['Sheet1']" in by["plan"]["files"]["inspect_notes.md"]
    assert sorted(by["implement"]["files"]) == sorted(["SKILL.md", "inspect_notes.md",
                                                       "input.xlsx", "plan.md"])
    assert by["implement"]["files"]["plan.md"] == b"set B1\n"
    assert sorted(by["verify"]["files"]) == sorted(["SKILL.md", "inspect_notes.md", "input.xlsx",
                                                    "output.xlsx", "plan.md"])
    assert "B1 IS 8" in by["verify"]["seen"][0], by["verify"]["seen"]
    assert "INSPECT-OUTPUT" in by["inspect"]["seen"][1]
    # Every step's exchange is in the record, in order.
    assert [(e["node"], e["response"]) for e in record["exchanges"]] == \
        [(s, f"{s} done.") for s in STEPS]
    (pass_dir,) = {p.parent for p in (tmp_path / "walk").rglob("final.json")}
    assert (pass_dir / "verify_report.md").read_text() == "checked\n"
    assert (pass_dir / "inspect_notes.md").exists() and (pass_dir / "plan.md").exists()
    final = _final(tmp_path / "walk")
    # A stand-in model leaves no session facts: each step is flagged, turns unknown.
    assert final["step_turns"] == {s: None for s in STEPS} and final["n_turns"] is None
    assert final["session_facts_missing"] == list(STEPS)
    assert "turn_limits" not in final and "steps_without_reply" not in final
    assert record["prediction"].startswith("## Eval Result: PASS")
    assert record["prediction"].endswith(
        "(no session facts from Claude Code's adapter for the inspect, plan, implement, verify "
        "steps: their turn counts and whether they reached the turn cap are unknown)")
    (conversation,) = (tmp_path / "walk").rglob("conversation.json")
    assert [m["step"] for m in json.loads(conversation.read_text())] == list(STEPS)


@needs_linux
def test_without_verifys_correction_the_wrong_output_is_graded(tmp_path, sb2_env, office):
    """The grade reads the output left after the last step: implement's 8, unchanged by
    verify, fails with V2's mismatch."""
    (record,) = _walk(tmp_path, StepsStub(_commands(implement=8, verify=None)),
                      expect_ok=False).item_records
    assert (record["passed"], record["score"]) == (False, 0.0)
    assert "Modification error at Sheet1!B1: answer=7, output=8" in record["prediction"]


@needs_linux
def test_the_skill_is_one_skill_md_every_step_reads(tmp_path, sb2_env, office):
    model = StepsStub(_commands(implement=7, verify=None))
    (record,) = _walk(tmp_path, model, skill="Check the totals.").item_records
    assert record["passed"], record
    for call in model.calls:
        assert call["files"]["SKILL.md"] == b"Check the totals."
        assert call["prompt"] == steps_port.render(call["step"], "Put 7 in B1.",
                                                   "Check the totals.")


@needs_linux
def test_an_output_verify_removes_fails_the_item(tmp_path, sb2_env, office):
    """No output.xlsx after the last step fails the item, saying so."""
    (record,) = _walk(tmp_path, StepsStub(_commands(implement=7, verify=None, delete=True)),
                      expect_ok=False).item_records
    assert (record["passed"], record["score"]) == (False, 0.0)
    assert record["fail_reason"].startswith(steps_ops.NO_OUTPUT), record["fail_reason"]


@needs_linux
def test_no_output_from_implement_or_verify_fails_the_item(tmp_path, sb2_env, office):
    """Implement leaves no output.xlsx: verify still runs (a read only sealed steps may
    leave out is optional, `workspace.optional_reads`), its workspace without one; it
    writes none either, and the item fails with output-not-found after a full record."""
    model = StepsStub(_commands(implement=None, verify=None))
    (record,) = _walk(tmp_path, model, expect_ok=False).item_records
    assert (record["passed"], record["score"]) == (False, 0.0)
    assert record.get("error_text") is None, record
    assert record["fail_reason"].startswith(steps_ops.NO_OUTPUT), record["fail_reason"]
    assert [c["step"] for c in model.calls] == list(STEPS)
    assert "output.xlsx" not in model.calls[3]["files"]
    final = _final(tmp_path / "walk")
    assert final["fail_reason"] == steps_ops.NO_OUTPUT
    assert "steps_without_reply" not in final
    # Verify ran without output.xlsx, and the record says so.
    assert final["missing_inputs"] == {"verify": ["output.xlsx"]}
    assert "(the verify step ran without output.xlsx: no earlier step wrote it)" \
        in record["prediction"]


@needs_linux
def test_verify_writes_the_output_implement_left_out(tmp_path, sb2_env, office):
    """Verify keeps the monolith's last chance: implement leaves no output.xlsx, verify
    writes it, and it is graded."""
    (record,) = _walk(tmp_path, StepsStub(_commands(implement=None, verify=7))).item_records
    assert record["passed"], record


@needs_linux
def test_a_step_runs_without_notes_the_step_before_did_not_write(tmp_path, sb2_env, office):
    """Inspect writes no notes (a session stopped at its cap, say): plan runs without
    them, and the item goes on to be graded."""
    commands = _commands(implement=7, verify=None)
    commands["inspect"] = ["view_xlsx input.xlsx list"]
    commands["plan"] = ["echo 'set B1' > plan.md"]
    model = StepsStub(commands)
    (record,) = _walk(tmp_path, model).item_records
    assert record["passed"], record
    by = {c["step"]: c for c in model.calls}
    assert sorted(by["plan"]["files"]) == ["SKILL.md", "input.xlsx"]
    assert sorted(by["implement"]["files"]) == ["SKILL.md", "input.xlsx", "plan.md"]
    # Never silent: final.json and the verdict name each step that ran without the notes.
    final = _final(tmp_path / "walk")
    assert final["missing_inputs"] == {"plan": ["inspect_notes.md"],
                                       "implement": ["inspect_notes.md"],
                                       "verify": ["inspect_notes.md"]}
    for step in ("plan", "implement", "verify"):
        assert f"(the {step} step ran without inspect_notes.md: no earlier step wrote it)" \
            in record["prediction"], record["prediction"]


@needs_linux
def test_a_step_with_no_reply_is_named_and_the_item_goes_on(tmp_path, sb2_env, office):
    """Plan's session ends with no reply text and no plan.md: the item goes on
    (implement runs without plan.md), and final.json names plan in
    ``steps_without_reply`` and implement's absent plan.md in ``missing_inputs``."""
    commands = _commands(implement=7, verify=None)
    commands["plan"] = ["cat inspect_notes.md"]  # and no plan.md
    model = StepsStub(commands, silent=("plan",))
    (record,) = _walk(tmp_path, model).item_records
    assert record["passed"], record
    assert [c["step"] for c in model.calls] == list(STEPS)
    final = _final(tmp_path / "walk")
    assert final["steps_without_reply"] == ["plan"]
    assert final["missing_inputs"] == {"implement": ["plan.md"], "verify": ["plan.md"]}
    assert final["step_turns"]["plan"] is None
    assert "(the implement step ran without plan.md: no earlier step wrote it)" \
        in record["prediction"]


def test_only_reads_sealed_steps_may_leave_out_are_optional():
    """The optional reads: the hand-off files and output.xlsx, never what prepare lays."""
    from refactory_dsh.workspace import optional_reads

    graph = steps_mode.Steps().build()
    assert optional_reads(graph, "inspect") == ()
    assert optional_reads(graph, "plan") == ("inspect_notes.md",)
    assert optional_reads(graph, "implement") == ("inspect_notes.md", "plan.md")
    assert optional_reads(graph, "verify") == ("inspect_notes.md", "plan.md", "output.xlsx")
    # swe's solve reads only what prepare lays; nothing is optional.
    swe = swe_mode.Swe().build()
    assert optional_reads(swe, "solve") == ()
    from refactory_dsh import lowering

    declared = lowering.node_declarations(steps_mode.Steps().state_dict())
    assert declared["verify"]["optional_reads"] == ["inspect_notes.md", "plan.md", "output.xlsx"]
    assert "optional_reads" not in declared["inspect"]


@needs_linux
def test_the_skillopt_view_shows_the_steps_in_order(tmp_path, sb2_env, office):
    """SkillOpt's view: each step's final message headed by its step, then V2's verdict;
    the target's prompts are each step's prompt as sent and its system prompt."""
    from refactory_packages.skillopt.env import SkillOptEnv

    (record,) = _walk(tmp_path, StepsStub(_commands(implement=8, verify=None)),
                      expect_ok=False).item_records
    assert skillopt_view.MODE_VIEWS["steps"] is skillopt_view.steps_rollout_result
    view = SkillOptEnv.for_mode("spreadsheetbench_v2/steps").rollout_result(
        record, steps_mode.Steps().values())
    assert view["conversation"][:-1] == [{"role": "assistant", "content": f"[{s} step]\n{s} done."}
                                         for s in STEPS]
    assert view["conversation"][-1]["content"] == \
        f"[POST-EXECUTION VERIFICATION]\n\n{record['prediction']}"
    assert view["n_turns"] == 4
    assert view["target_user_prompt"] == "\n\n".join(
        f"### {s} step\n{steps_port.render(s, 'Put 7 in B1.')}" for s in STEPS)
    assert view["target_system_prompt"] == "\n\n".join(
        f"### {s} step\n{swe_port.SYSTEM_TEMPLATE}" for s in STEPS)
    # The default loop's item context is the input's preview, as swe's.
    from refactory_dsh.task import item_metadata_reader
    from refactory_packages.spreadsheetbench import skillopt_port as port
    from test_spreadsheetbench_v2 import _rel

    preview = port.preview_workbook(str(sb2_env / _rel(2)[0]))
    assert view["spreadsheet_preview"] == preview and view["task_type"] == "Template"
    context = item_metadata_reader(steps_mode.Steps().state_dict(), context=True)(record)
    assert context == {"spreadsheet_preview": preview}


# ── the TS spine: lowered, each step dispatched to the real worker ────────────

#: A scripted ``dsh`` that acts as each step: it takes the seal runner from the worker
#: patch, runs the step's commands (chosen by the prompt) through it in its working
#: directory, and logs what it was sent.
STEPS_DSH = '''import json, pathlib, re, subprocess, sys
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
step = {steps!r}[int(re.findall(r"This is step (\\d) of 4", prompt)[0]) - 1]
files = sorted(str(p.relative_to(cwd)) for p in cwd.rglob("*") if p.is_file())
seen = []
for command in {commands!r}[step]:
    proc = subprocess.run([*runner, "--", "/bin/bash", "-c", command],
                          cwd=cwd, capture_output=True, text=True, timeout=120)
    seen.append(proc.stdout + proc.stderr)
with log.open("a") as out:
    out.write(json.dumps({{"step": step, "prompt": prompt, "config": config, "seen": seen,
                          "files": files}}) + "\\n")
print(step + " done.")
'''


def _drain(tmp_path: Path, client: Any) -> Any:
    """The val item, lowered with dispatched workers on ``client``'s route and drained by
    the real spine (as `test_spreadsheetbench_v2_swe._drain`, on the steps mode)."""
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

    graph_path, _ = lowering.write_mode_graph(state, tmp_path / "graphs", name="sb2t",
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
def test_files_flow_and_verify_corrects_the_output_on_the_spine(tmp_path, sb2_env, office):
    """Spine: the same item, each step dispatched to the real worker on the agent mode's
    route; each worker's patch hands the adapter Bash, the step's system prompt and the
    frozen limits; files flow step to step and verify's correction is graded."""
    from refactory_dsh.model import DshModelClient

    root = tmp_path / "dsh"
    root.mkdir()
    log = root / "calls.jsonl"
    script = root / "steps_dsh.py"
    script.write_text(STEPS_DSH.format(log=str(log), steps=list(STEPS),
                                       commands=_commands(implement=8, verify=7)))
    cli = root / "dsh"
    cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
    cli.chmod(0o755)
    client = DshModelClient(**AGENT_ROUTE, dsh_cli=str(cli), attempts=1, timeout_s=120)
    record = _drain(tmp_path, client)
    assert (record["id"], record["passed"]) == (VAL, True), record
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [c["step"] for c in calls] == list(STEPS)
    for call in calls:
        assert call["prompt"].strip() == steps_port.render(call["step"], "Put 7 in B1.").strip()
        env = call["config"]["env"]
        assert (env["REFACTORY_MAX_TURNS"], env["REFACTORY_BASH_TIMEOUT_MS"],
                env["REFACTORY_BASH_OUTPUT_CHARS"]) == ("50", "60000", "10000")
        assert env["REFACTORY_SYSTEM_PROMPT"] == swe_port.SYSTEM_TEMPLATE
        assert env["REFACTORY_SEALED_TOOLS"] == "Bash"
    by = {c["step"]: c for c in calls}
    assert by["plan"]["files"] == sorted(["SKILL.md", "input.xlsx", "inspect_notes.md"])
    assert by["verify"]["files"] == sorted(["SKILL.md", "input.xlsx", "inspect_notes.md",
                                            "output.xlsx", "plan.md"])
    assert "B1 IS 8" in by["verify"]["seen"][0], by["verify"]["seen"]


@needs_linux
@needs_spine
def test_absent_hand_offs_are_optional_and_recorded_on_the_spine(tmp_path, sb2_env, office):
    """Spine: inspect leaves no notes and implement no output.xlsx; the workers run the
    later steps without them (as the walker does), record each one, and verify's own
    output is graded."""
    from refactory_dsh.model import DshModelClient

    root = tmp_path / "dsh"
    root.mkdir()
    log = root / "calls.jsonl"
    script = root / "steps_dsh.py"
    commands = _commands(implement=None, verify=7)
    commands["inspect"] = ["view_xlsx input.xlsx list"]
    script.write_text(STEPS_DSH.format(log=str(log), steps=list(STEPS), commands=commands))
    cli = root / "dsh"
    cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
    cli.chmod(0o755)
    client = DshModelClient(**AGENT_ROUTE, dsh_cli=str(cli), attempts=1, timeout_s=120)
    record = _drain(tmp_path, client)
    assert record["passed"], record
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [c["step"] for c in calls] == list(STEPS)
    assert "output.xlsx" not in calls[3]["files"]
    assert "inspect_notes.md" not in calls[1]["files"]
    # Recorded on the spine as on the walker: final.json and the verdict.
    final = _final(tmp_path / "spine")
    assert final["missing_inputs"] == {"plan": ["inspect_notes.md"],
                                       "implement": ["inspect_notes.md"],
                                       "verify": ["inspect_notes.md", "output.xlsx"]}
    assert ("(the verify step ran without inspect_notes.md, output.xlsx: no earlier step "
            "wrote them)") in record["prediction"], record["prediction"]


# ── the real adapter: turns per step, and a step stopped at its cap ───────────

#: Stands in for @anthropic-ai/claude-agent-sdk, as each step: it finds the step from the
#: prompt, runs that step's commands (FAKE_STEP_COMMANDS) as Claude Code runs Bash calls
#: (the hook's command, by a shell, in the session's working directory), appends what it
#: was started with to FAKE_SDK_RECORD, then ends as Claude Code does at its turn cap
#: (the step named by FAKE_CAPPED_STEP: ``error_max_turns``, 50 turns) or with a reply
#: after the step's FAKE_STEP_TURNS turns.
_STEPS_SDK = r"""
import { execSync } from 'node:child_process'
import { appendFileSync } from 'node:fs'
const STEPS = ['inspect', 'plan', 'implement', 'verify']
export function query({ prompt, options }) {
  const child = options.spawnClaudeCodeProcess({
    command: process.execPath, args: ['-e', 'setTimeout(() => {}, 60000)'],
    cwd: options.cwd, env: options.env, signal: options.abortController.signal })
  const record = { maxTurns: options.maxTurns ?? null, systemPrompt: options.systemPrompt,
    tools: options.tools ?? null }
  const hook = options.hooks?.PreToolUse?.[0]?.hooks?.[0]
  return {
    close() { child.kill?.() },
    async *[Symbol.asyncIterator]() {
      const input = prompt[Symbol.asyncIterator]()
      record.prompt = (await input.next()).value?.message?.content
      const step = STEPS[Number(/This is step (\d) of 4/.exec(record.prompt)[1]) - 1]
      record.step = step
      yield { type: 'system', subtype: 'init', model: 'fake-claude', session_id: 'fake-session',
              tools: options.tools ?? [] }
      const commands = JSON.parse(process.env.FAKE_STEP_COMMANDS)[step]
      for (const [i, command] of commands.entries()) {
        yield { type: 'assistant', message: { role: 'assistant', content: [
          { type: 'tool_use', id: `toolu_${i}`, name: 'Bash', input: { command, description: `step ${i}` } }] } }
        const out = (await hook({ hook_event_name: 'PreToolUse', tool_name: 'Bash',
                                  tool_input: { command } }, `toolu_${i}`, {}))?.hookSpecificOutput
        const text = out?.updatedInput?.command
          ? execSync(out.updatedInput.command, { cwd: options.cwd, shell: '/bin/bash', encoding: 'utf8' })
          : ''
        yield { type: 'user', message: { role: 'user', content: [
          { type: 'tool_result', tool_use_id: `toolu_${i}`, is_error: false, content: [{ type: 'text', text }] }] } }
      }
      appendFileSync(process.env.FAKE_SDK_RECORD, JSON.stringify(record) + '\n')
      const usage = { 'fake-claude': { inputTokens: 3, outputTokens: 1, cacheReadInputTokens: 0, cacheCreationInputTokens: 0 } }
      if (process.env.FAKE_CAPPED_STEP === step) {
        yield { type: 'result', subtype: 'error_max_turns', is_error: true, errors: [], num_turns: 50, modelUsage: usage }
      } else {
        const turns = JSON.parse(process.env.FAKE_STEP_TURNS)[step]
        yield { type: 'result', subtype: 'success', is_error: false, result: `${step} done.`, num_turns: turns, modelUsage: usage }
      }
    },
  }
}
"""

#: Each step's turn count when it ends itself.
STEP_TURNS = {"inspect": 2, "plan": 3, "implement": 5, "verify": 7}


@pytest.fixture
def steps_sdk(tmp_path, monkeypatch, sb2_env, office):
    """Every Node process DSH starts resolves the Agent SDK to the stand-in above, which
    runs each step's commands (implement writes 8, verify corrects it to 7)."""
    from test_claude_code_route import _HOOKS, _REGISTER

    hooks = tmp_path / "sdk-hooks"
    hooks.mkdir()
    (hooks / "fake-sdk.mjs").write_text(_STEPS_SDK)
    (hooks / "hooks.mjs").write_text(_HOOKS)
    (hooks / "register.mjs").write_text(_REGISTER)
    record = tmp_path / "sdk-query.jsonl"
    monkeypatch.setenv("NODE_OPTIONS", f"--import {(hooks / 'register.mjs').as_uri()}")
    monkeypatch.setenv("FAKE_SDK_RECORD", str(record))
    monkeypatch.setenv("FAKE_STEP_COMMANDS", json.dumps(_commands(implement=8, verify=7)))
    monkeypatch.setenv("FAKE_STEP_TURNS", json.dumps(STEP_TURNS))
    monkeypatch.setenv("FAKE_CAPPED_STEP", "implement")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    monkeypatch.delenv("REFACTORY_WORKER_CONTEXT", raising=False)
    return record


def _turns_recorded(record: dict[str, Any], sent_path: Path, work: Path) -> None:
    """Each step's session was started with the turn cap, Bash alone, V2's system template
    and its own prompt; the item's final.json records each step's turns (implement's at
    its cap) and their total; the verdict names the capped step; tool_calls.json keeps
    every step's calls in order; the corrected output is graded."""
    sent = [json.loads(line) for line in sent_path.read_text().splitlines()]
    assert [s["step"] for s in sent] == list(STEPS)
    for s in sent:
        assert s["maxTurns"] == 50 and s["tools"] == ["Bash"]
        assert s["systemPrompt"] == {"type": "preset", "preset": "claude_code",
                                     "append": swe_port.SYSTEM_TEMPLATE}
        assert s["prompt"] == steps_port.render(s["step"], "Put 7 in B1.")
    assert record["passed"], record
    final = _final(work)
    turns = {**STEP_TURNS, "implement": 50}
    assert final["step_turns"] == turns and final["n_turns"] == sum(turns.values()) == 62
    assert final["turn_limits"] == {"implement": 50}
    assert "session_facts_missing" not in final and "steps_without_reply" not in final
    assert record["prediction"].startswith("## Eval Result: PASS")
    assert record["prediction"].endswith(
        "\n(the implement step stopped at its turn cap: 50 turns, cap 50)")
    (path,) = work.rglob("tool_calls.json")
    kept = json.loads(path.read_text())
    assert kept["n_turns"] == 62
    assert [s["step"] for s in kept["steps"]] == list(STEPS)
    commands = _commands(implement=8, verify=7)
    for s in kept["steps"]:
        assert s["n_turns"] == turns[s["step"]]
        assert s["turn_limit_reached"] is (s["step"] == "implement")
        assert [c["input"] for c in s["calls"]] == commands[s["step"]]
    # Implement's session stopped at its cap still left output.xlsx; verify read it.
    verify_calls = kept["steps"][3]["calls"]
    assert "B1 IS 8" in verify_calls[0]["output"], verify_calls[0]
    assert not list(work.rglob("*.session.json"))


@needs_adapter
@needs_linux
def test_turns_per_step_and_in_total_on_the_walker(tmp_path, steps_sdk):
    state = _state()
    work = tmp_path / "walk"
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()})
    task_module.write_subset(work, "val", 0.0)
    result = WalkerEvaluator(model=_claude(tmp_path), project=Path("."), max_steps=400) \
        .evaluate(steps_mode.Steps(state=state), state, work)
    assert result.exit_ok, result.error
    (record,) = result.item_records
    _turns_recorded(record, steps_sdk, work)


@needs_linux
@needs_spine
@needs_adapter
def test_turns_per_step_and_in_total_on_the_spine(tmp_path, steps_sdk):
    from refactory_dsh.model import DshModelClient

    client = DshModelClient(**AGENT_ROUTE, runtime_root=REPO_ROOT, dsh_home=tmp_path / "dsh-home",
                            routes_resolved=True, timeout_s=120)
    _turns_recorded(_drain(tmp_path, client), steps_sdk, tmp_path / "spine")


def test_the_steps_task_names_each_capped_step_and_missing_facts():
    """The verdict's note: each capped step with its turns, and the steps with no facts."""
    task = tasks.SpreadsheetBenchV2StepsTask.__new__(tasks.SpreadsheetBenchV2StepsTask)
    assert task.session_note({}) == ""
    assert task.session_note({"missing_inputs": {"plan": ["inspect_notes.md"]}}) == \
        "(the plan step ran without inspect_notes.md: no earlier step wrote it)"
    assert task.session_note({"step_turns": {"plan": 50, "verify": None},
                              "turn_limits": {"plan": 50, "verify": 50}}) == (
        "(the plan step stopped at its turn cap: 50 turns, cap 50)\n"
        "(the verify step stopped at its turn cap: cap 50)")
    assert task.session_note({"session_facts_missing": ["plan"]}) == (
        "(no session facts from Claude Code's adapter for the plan step: its turn count "
        "and whether it reached the turn cap are unknown)")
