"""SpreadsheetBench: the port of SkillOpt's target agent, checked against SkillOpt.

Expected values come from SkillOpt's own code (ec9d021) or from SkillOpt's recorded
runs, never from the port: ``fixtures/spreadsheetbench_parity.json`` holds six target
calls recorded in the lab's ``runs/so-s42`` (with the messages SkillOpt saved for them)
and SkillOpt's outputs on synthetic workbooks built by ``sb_support.make_workbook``. The
loop runs on a synthetic dataset (`sb_support.make_dataset`) with a scripted model, on the
walker and on the TS spine. The checks on the real release data and run directories
(`refactory_packages.spreadsheetbench.parity`) run only where ``SB_DATA`` and
``SB_SKILLOPT_RUN`` name them.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn
from refactory_dsh import task as task_module
from refactory_dsh.evaluators import WalkerEvaluator
from refactory_dsh.nn.executor import RunContext
from refactory_dsh.nn.validate import validate
from refactory_packages.skillopt.claude_backend import (
    claude_backend_messages, claude_backend_prompt, unframe_claude_backend_messages)
from refactory_packages.spreadsheetbench import (
    interpreter, mode as sb_mode, ops, recalc, sandbox, skillopt_port as port, skillopt_view, tasks)
from sb_support import make_dataset, make_workbook
from tests_support import FakeModel

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "spreadsheetbench_parity.json")
                     .read_text(encoding="utf-8"))
EVAL_FEEDBACK_HEAD = "Your code executed successfully but produced incorrect results."


# ── SkillOpt's text, byte for byte ─────────────────────────────────────────

@pytest.mark.parametrize("case", FIXTURE["framing"], ids=lambda c: c["case"])
def test_framing_is_the_recorded_call(case):
    """§6.1: the system file and the History: prompt of a recorded so-s42 target call,
    from the messages SkillOpt saved for it: four first turns (an answer-sheet-prefixed
    position, a multi-sheet preview, a formula and a cut value, non-ASCII text), a turn
    after an execution error and a train turn after eval feedback."""
    assert claude_backend_messages(case["messages"]) == (case["system"], case["prompt"])
    assert unframe_claude_backend_messages(case["prompt"]) == [
        {"role": m["role"], "content": m["content"].strip()} for m in case["messages"][1:]]


def test_one_message_framing_is_the_multi_message_framing():
    for text in ("", "  a\nb  ", "x\n\ny"):
        assert claude_backend_prompt(text) == claude_backend_messages(
            [{"role": "user", "content": text}])[1]


def test_initial_skill_and_system_prompt_are_skillopts():
    assert port.build_system(port.INITIAL_SKILL) == FIXTURE["skill_initial_system"]
    rendered = sb_mode.SYSTEM_PROMPT.replace("{{skill_section}}",
                                             port.skill_section(port.INITIAL_SKILL))
    assert rendered == FIXTURE["skill_initial_system"]
    assert port.build_system("  ") == port.CODEGEN_SYSTEM
    assert not port.CODEGEN_SYSTEM.endswith("\n")


@pytest.mark.parametrize("name", ["one_sheet", "multi_sheet", "many_rows"])
def test_preview_and_first_message_are_skillopts(tmp_path, name):
    case = FIXTURE["previews"][name]
    path = make_workbook(tmp_path / f"{name}.xlsx", case["spec"])
    assert port.preview_workbook(str(path)) == case["preview"]
    assert port.build_user("Do the thing.\nTwo lines.", str(path), "Cell-Level Manipulation",
                           "'Data Sheet'!B1:B3") == case["user"]


def test_unreadable_workbook_preview_is_skillopts(tmp_path):
    broken = tmp_path / "broken.xlsx"
    broken.write_text("not a workbook")
    tail = FIXTURE["previews"]["broken"]["user_tail"]
    # The message names the file's path; only what follows it is compared.
    assert port.build_user("x", str(broken))[-200:] == tail[-200:]


@pytest.mark.parametrize("case", FIXTURE["extract_code"], ids=lambda c: repr(c["text"])[:20])
def test_extract_code_is_skillopts(case):
    """The first fence of any language; no usable fence is the whole reply."""
    assert port.extract_code(case["text"]) == case["code"]


@pytest.mark.parametrize("case", FIXTURE["runner"], ids=lambda c: c["code"][:10])
def test_runner_script_is_skillopts(case):
    assert port.runner_script(case["code"], "/in/1_a_init.xlsx", "/out/1_pred.xlsx") == case["script"]


def _value(v: Any) -> Any:
    from sb_support import _value as decode

    return decode(v)


@pytest.mark.parametrize("case", FIXTURE["compare"], ids=lambda c: f"{c['a']!r}-{c['b']!r}"[:30])
def test_cell_comparison_is_skillopts(case):
    assert port.compare_cell_value(_value(case["a"]), _value(case["b"])) is case["equal"]


@pytest.fixture(scope="module")
def graded(tmp_path_factory):
    root = tmp_path_factory.mktemp("graded")
    gold = make_workbook(root / "gold.xlsx", FIXTURE["gold"])
    preds = {name: make_workbook(root / f"{name}.xlsx", spec) for name, spec in FIXTURE["preds"].items()}
    return gold, preds, root


@pytest.mark.parametrize("index", range(len(FIXTURE["grading"])))
def test_grade_and_verification_report_are_skillopts(graded, index):
    gold, preds, root = graded
    case = FIXTURE["grading"][index]
    pred = preds[case["pred"]] if case["pred"] else root / "nope.xlsx"
    assert port.evaluate_cached(str(pred), str(gold), "t", case["position"]) == case["evaluate"]
    assert recalc.evaluate(str(pred), str(gold), "t", case["position"], choice="cached") \
        == case["evaluate"]
    assert port.auto_verify_output(str(pred), str(gold), case["position"]) == case["report"]


@pytest.mark.parametrize("index", range(len(FIXTURE["feedback"])))
def test_eval_feedback_is_skillopts(index):
    """§6.5 on SkillOpt's own outputs, its quirk included: a cell whose sheet name holds a
    space is dropped from the list (port spec §3, item 6)."""
    case = FIXTURE["feedback"][index]
    assert port.build_eval_feedback(case["report"]) == case["feedback"]


def test_no_code_and_exec_error_texts():
    assert port.NO_CODE == FIXTURE["no_code"]
    text = port.exec_error_feedback("x" * 4000)
    assert text.startswith("The code raised an error during execution:\n\n```\n" + "x" * 3000 + "\n```")
    assert "x" * 3001 not in text


# ── the pinned interpreter and the sandbox ──────────────────────────────────

def _require_sandbox(python: str) -> None:
    ok, why = sandbox.available(python)
    if not ok:
        if os.environ.get("REFACTORY_REQUIRE_CODE_SANDBOX"):
            pytest.fail(f"the sandbox must run here: {why}")
        pytest.skip(f"the model-code sandbox cannot run here: {why}")


@pytest.fixture(scope="session")
def sb_python():
    if shutil.which("uv") is None and not os.environ.get(interpreter.ENV):
        pytest.skip("building the pinned interpreter needs uv")
    python = os.environ.get(interpreter.ENV) or str(interpreter.provision())
    interpreter.check(python)
    _require_sandbox(python)
    return python


def test_interpreter_check_refuses_other_versions():
    """refactory's own venv has openpyxl 3.1.5 but not pandas: refused by name."""
    with pytest.raises(interpreter.InterpreterError, match="pandas"):
        interpreter.check.__wrapped__(sys.executable)


def test_code_runs_without_the_sandbox_never(monkeypatch):
    monkeypatch.setattr(sandbox, "available", lambda python: (False, "no namespaces"))
    with pytest.raises(sandbox.SandboxError, match="never run unsandboxed"):
        sandbox.require("python")


def test_hostile_code_sees_neither_the_answer_nor_secrets(tmp_path, monkeypatch, sb_python):
    """The golden file beside the input, the run, a secret in the home directory and the
    network are out of reach; the input copy reads and the output writes."""
    data = tmp_path / "data" / "spreadsheet" / "a"
    gold = make_workbook(data / "1_a_golden.xlsx", [{"title": "S", "cells": {"A1": "ANSWER"}}])
    run = tmp_path / "run"
    (run / "facts.json").parent.mkdir(parents=True)
    (run / "facts.json").write_text("{}")
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_test").write_text("SECRET")
    monkeypatch.setenv("HOME", str(home))
    work = Path(__import__("tempfile").mkdtemp(prefix="sb-item-"))
    try:
        (work / "in").mkdir()
        (work / "out").mkdir()
        source = make_workbook(work / "in" / "1_a_init.xlsx", [{"title": "S", "cells": {"A1": 1}}])
        code = f"""
import json, socket, openpyxl, pandas
seen = {{}}
for name, path in [("gold", {str(gold)!r}), ("run", {str(run / 'facts.json')!r}),
                   ("secret", {str(home / '.ssh' / 'id_test')!r})]:
    try:
        seen[name] = open(path, "rb").read()[:20].decode("latin-1")
    except Exception as e:
        seen[name] = "blocked"
for name, path in [("write_run", {str(run / 'x')!r}), ("write_home", {str(home / 'x')!r})]:
    try:
        open(path, "w").write("x"); seen[name] = "wrote"
    except Exception:
        seen[name] = "blocked"
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3); seen["net"] = "open"
except Exception:
    seen["net"] = "blocked"
wb = openpyxl.load_workbook(INPUT_PATH)
wb.active["B1"] = json.dumps(seen)
wb.save(OUTPUT_PATH)
"""
        hidden = sandbox.hidden_paths([str(tmp_path / "data"), str(run)])
        ok, err = port.run_generated_code(code, str(source), str(work / "out" / "1_pred.xlsx"),
                                          python=sb_python, wrap=sandbox.wrapper(hidden),
                                          cwd=str(work))
        assert ok, err
        import openpyxl

        seen = json.loads(openpyxl.load_workbook(work / "out" / "1_pred.xlsx").active["B1"].value)
        assert seen == {"gold": "blocked", "run": "blocked", "secret": "blocked",
                        "write_run": "blocked", "write_home": "blocked", "net": "blocked"}
        assert not (run / "x").exists() and not (home / "x").exists()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_sandbox_refuses_to_hide_what_the_code_needs(tmp_path):
    wrap = sandbox.wrapper([str(tmp_path)])
    with pytest.raises(sandbox.SandboxError, match="inside the hidden"):
        wrap(["python", "r.py"], str(tmp_path / "out"), "/elsewhere")


# ── the Task ───────────────────────────────────────────────────────────────

ONE = [{"title": "Sheet1", "cells": {"A1": 1}}]


def _items() -> list[dict[str, Any]]:
    return [
        {"id": "t1", "instruction": "Put 42 in B1.", "instruction_type": "Cell-Level Manipulation",
         "answer_position": "B1", "init": ONE,
         "gold": [{"title": "Sheet1", "cells": {"A1": 1, "B1": 42}}]},
        {"id": "t2", "instruction": "Put 9 in B1.", "instruction_type": "Cell-Level Manipulation",
         "answer_position": "B1", "init": ONE,
         "gold": [{"title": "Sheet1", "cells": {"A1": 1, "B1": 9}}]},
        {"id": "v1", "instruction": "Put 7 in B1.", "instruction_type": "Cell-Level Manipulation",
         "answer_position": "B1", "answer_sheet": "Sheet1", "init": ONE,
         "gold": [{"title": "Sheet1", "cells": {"A1": 1, "B1": 7}}], "exclude": "hint"},
        # 42930's defect: the golden file names another item.
        {"id": "x1", "instruction": "Anything.", "instruction_type": "Sheet-Level Manipulation",
         "answer_position": "A1", "init": ONE, "gold": ONE, "gold_name": "1_x2_golden.xlsx"},
    ]


PARTS = {"train": ["t1", "t2"], "val": ["v1"], "test": ["x1"]}


@pytest.fixture()
def sb_data(tmp_path, monkeypatch):
    manifest = make_dataset(tmp_path / "dataset", _items(), PARTS)
    monkeypatch.setenv(tasks.DATA_ENV, str(tmp_path / "dataset" / "sb400"))
    monkeypatch.setenv(tasks.MANIFEST_ENV, str(manifest))
    monkeypatch.setenv(recalc.GRADER_ENV, "cached")
    return tmp_path / "dataset"


def test_task_items_hide_the_answer(sb_data):
    task = tasks.SpreadsheetBenchTask()
    items = task.instances()
    assert [i.id for i in items] == ["t1", "t2", "v1", "x1"]
    v1 = items[2].metadata
    assert v1["answer_position_eval"] == "Sheet1!B1"
    assert v1["task_type"] == "cell_level" and v1["case_no"] == "1"
    assert v1["input_name"] == "1_v1_init.xlsx"
    visible = task_module.node_item({"id": "v1", **v1}, task)
    assert not {"gold_path", "input_path", "exclude"} & set(visible)
    assert "golden" not in json.dumps(visible)
    assert items[3].metadata["gold_path"] == "" and items[3].metadata["input_name"] == ""


def test_task_refuses_other_data(sb_data, monkeypatch):
    rows = sb_data / "sb400" / "dataset.json"
    rows.write_text(rows.read_text() + " ")
    with pytest.raises(task_module.TaskError, match="sha256"):
        tasks.SpreadsheetBenchTask().instances()
    monkeypatch.setenv(tasks.DATA_ENV, str(sb_data / "absent"))
    with pytest.raises(task_module.TaskError, match="SB_DATA"):
        tasks.SpreadsheetBenchTask().instances()


def test_recalc_needs_libreoffice(sb_data, monkeypatch):
    monkeypatch.setenv(recalc.GRADER_ENV, "recalc")
    monkeypatch.setattr(recalc, "available", lambda: False)
    with pytest.raises(task_module.TaskError, match="LibreOffice"):
        tasks.SpreadsheetBenchTask().instances()


def test_shipped_manifest_is_skillopts_split():
    manifest = tasks.load_manifest()
    assert {k: len(v) for k, v in manifest["parts"].items()} == {"train": 80, "val": 40, "test": 280}
    assert manifest["revision"] == "ab0b742b0fc95b946f212d80ac7771b5531272e4"
    assert {"42930", "56225"} <= set(manifest["parts"]["test"])
    ids = [i for part in manifest["parts"].values() for i in part]
    assert len(ids) == len(set(ids)) == 400


# ── the loop, on the walker ─────────────────────────────────────────────────

def _good(n: int) -> str:
    return ("Here it is.\n```python\nimport openpyxl\nwb = openpyxl.load_workbook(INPUT_PATH)\n"
            f"wb.active['B1'] = {n}\nwb.save(OUTPUT_PATH)\n```")


def _state(**knobs: Any) -> tuple[sb_mode.Solver, dict[str, Any]]:
    from refactory_dsh.nn.edits import configure

    state = sb_mode.Solver().state_dict()
    for name, value in {"data_parallelism": 1, "turn_pause_s": 0, **knobs}.items():
        state = configure(state, name, value)
    return sb_mode.Solver(state=state), state


def _walk(tmp_path: Path, part: str, answers: list[str] | FakeModel, *,
          gold_feedback: bool = False, train: list[str] = ("t1",), **knobs: Any):
    mode, state = _state(**knobs)
    work = tmp_path / f"walk-{part}"
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()
                                       if p != "train"}, train_ids=list(train))
    task_module.write_subset(work, part, 0.0, gold_feedback=gold_feedback)
    model = answers if isinstance(answers, FakeModel) else FakeModel(list(answers))
    result = WalkerEvaluator(model=model, project=Path("."), max_steps=4000).evaluate(mode, state, work)
    assert result.exit_ok, result.error
    return result, model.prompts


@pytest.fixture()
def sb_env(sb_data, sb_python, monkeypatch, tmp_path):
    monkeypatch.setenv(interpreter.ENV, sb_python)
    # The sandbox hides the home directory's secrets by path; a home of the test's own
    # keeps the real one (the DSH home among them) out of every command line.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return sb_data


def test_train_rollout_gets_gold_feedback(tmp_path, sb_env):
    """A prose reply is run as code (a SyntaxError, not NO_CODE), the error is the next
    message; a wrong clean run is told its wrong cell (value hidden); a right one ends
    the item (codegen_agent.py:713-753)."""
    result, prompts = _walk(tmp_path, "train", ["no fence at all", _good(41), _good(42)],
                            gold_feedback=True)
    (record,) = result.item_records
    assert record["passed"] and len(prompts) == 3
    assert "- User: The code raised an error during execution:" in prompts[1]
    assert "SyntaxError" in prompts[1]
    assert "No Python code block" not in prompts[1]
    assert EVAL_FEEDBACK_HEAD in prompts[2]
    assert "Sheet1!B1: your output = 41 (WRONG)" in prompts[2]
    assert "42" not in prompts[2].split(EVAL_FEEDBACK_HEAD)[1]


class _PerItem(FakeModel):
    """Answers each item by its instruction: a wrong value first, then the right one.
    Shared by concurrent items (``thread_safe``)."""

    thread_safe = True

    def __init__(self) -> None:
        import threading

        super().__init__([])
        self.calls: dict[int, int] = {}
        self.lock = threading.Lock()

    def complete(self, prompt: str, **kwargs: Any) -> str:  # type: ignore[override]
        target = 42 if "Put 42 in B1." in prompt else 9
        with self.lock:
            self.prompts.append(prompt)
            seen = self.calls[target] = self.calls.get(target, 0) + 1
        return _good(target if seen > 1 else target + 1)


def test_concurrent_items_get_their_own_gold_feedback(tmp_path, sb_env):
    """Two train items at once (the concurrent data pass): each wrong first answer is told
    its own wrong cell, and each item passes on its second call."""
    model = _PerItem()
    result, prompts = _walk(tmp_path, "train", model, gold_feedback=True, train=["t1", "t2"],
                            data_parallelism=2)
    assert [(r["id"], r["passed"], len(r["exchanges"])) for r in result.item_records] == \
        [("t1", True, 2), ("t2", True, 2)]
    told = [p for p in prompts if EVAL_FEEDBACK_HEAD in p]
    assert len(told) == 2
    assert any("your output = 43 (WRONG)" in p and "Put 42" in p for p in told)
    assert any("your output = 10 (WRONG)" in p and "Put 9" in p for p in told)
    assert "data_parallelism_effective" not in result.facts  # recorded only on a fallback to serial


@pytest.mark.parametrize("part,flag", [("val", False), ("train", False)])
def test_without_gold_feedback_the_first_clean_run_ends_the_item(tmp_path, sb_env, part, flag):
    """Val and test, and a train evaluation that did not ask for gold feedback: a wrong
    clean run ends the item, and nothing about the answer reaches a prompt."""
    result, prompts = _walk(tmp_path, part, ["", _good(8), _good(7)], gold_feedback=flag)
    (record,) = result.item_records
    assert len(prompts) == 2
    assert "- User: No Python code block was found in your response." in prompts[1]
    assert not any(EVAL_FEEDBACK_HEAD in p for p in prompts)
    assert record["passed"] is False
    assert record["fail_reason"].startswith("eval-mismatch: value@Sheet1!B1")


def test_turns_are_capped_and_the_last_code_runs_again(tmp_path, sb_env):
    """Every turn fails: the loop stops at max_turns, the missing prediction gets one more
    run of the last code (rollout.py:799-806), and the item fails with its last line."""
    bad = "```python\nraise ValueError('nope')\n```"
    result, prompts = _walk(tmp_path, "val", [bad] * 5, max_turns=3)
    (record,) = result.item_records
    assert len(prompts) == 3
    assert record["fail_reason"] == "exec-error: ValueError: nope"


def test_graph_caps_reloops_at_skillopts_turns():
    gate = sb_mode.Solver().graph().nodes["turn_gate"]
    assert (gate.max_iterations, gate.on_exhausted) == (sb_mode.MAX_TURNS - 1, "continue")
    assert sb_mode.MAX_TURNS == 30


def test_an_item_with_no_case_scores_zero_without_a_call(tmp_path, sb_env):
    """test 42930: its golden file names another item, so there is no case."""
    result, prompts = _walk(tmp_path, "test", [_good(1)])
    (record,) = result.item_records
    assert prompts == []
    assert (record["passed"], record["fail_reason"]) == (False, "no-test-cases")


def test_an_empty_last_reply_fails_the_item_whatever_ran_before(tmp_path, sb_env):
    """rollout.py:783-786 checks the last code before the prediction: a train item whose
    wrong clean run is followed by an empty reply fails as empty-code-block."""
    result, _ = _walk(tmp_path, "train", [_good(41), ""], gold_feedback=True, max_turns=2)
    assert result.item_records[0]["fail_reason"] == "empty-code-block"


def _ctx(workdir: Path, **knobs: Any) -> RunContext:
    ctx = RunContext(workdir, knobs={"item_budget_s": 3600, "max_turns": 30, **knobs},
                     enforce_writes=False)
    return ctx


def test_turn_gate_stops_with_ten_seconds_left(tmp_path):
    ctx = _ctx(tmp_path, item_budget_s=100)
    ctx.write_json("turn.json", {"start": time.time() - 91, "turn": 1, "status": "exec_error"})
    assert str(ops.turn_gate(ctx)) == "proceed"
    ctx.write_json("turn.json", {"start": time.time() - 80, "turn": 1, "status": "exec_error"})
    assert str(ops.turn_gate(ctx)) == "reloop"
    ctx.write_json("turn.json", {"start": time.time(), "turn": 30, "status": "exec_error"})
    assert str(ops.turn_gate(ctx)) == "proceed"


def test_an_item_past_its_budget_scores_zero(tmp_path, sb_data):
    """The item's turn loop ran past the budget: finalize says task-timeout, and verify
    scores it 0 even with a right prediction (rollout.py:983-1011)."""
    task = tasks.SpreadsheetBenchTask()
    t1 = task.instances()[0]
    ctx = _ctx(tmp_path, item_budget_s=60)
    ctx.write_json("current_item.json", {"id": "t1", **{k: v for k, v in t1.metadata.items()
                                                        if k not in task.grader_fields}})
    ctx.write_json("turn.json", {"start": time.time() - 61, "turn": 1, "status": "ran",
                                 "code": "pass"})
    ctx.write_json("history.json", [{"role": "user", "content": "u"}])
    shutil.copyfile(t1.metadata["gold_path"], tmp_path / tasks.PRED)
    ops.finalize(ctx)
    assert json.loads((tmp_path / tasks.FINAL).read_text())["fail_reason"] == "task-timeout-60s"
    verdict = task.verify(t1, tmp_path)
    assert (verdict.passed, verdict.score, verdict.details) == (False, 0.0, "task-timeout-60s")
    ctx.write_json("turn.json", {"start": time.time(), "turn": 1, "status": "ran", "code": "pass"})
    ops.finalize(ctx)
    assert task.verify(t1, tmp_path).passed


def test_a_missing_prediction_gets_one_more_run(tmp_path, sb_env):
    task = tasks.SpreadsheetBenchTask()
    t1 = task.instances()[0]
    ctx = _ctx(tmp_path)
    task.setup(t1, tmp_path)
    ctx.write_json("current_item.json", task_module.node_item({"id": "t1", **t1.metadata}, task))
    code = port.extract_code(_good(42))
    ctx.write_json("turn.json", {"start": time.time(), "turn": 1, "status": "exec_error",
                                 "code": code})
    ctx.write_json("history.json", [{"role": "user", "content": "u"}])
    ops.finalize(ctx)
    assert json.loads((tmp_path / tasks.FINAL).read_text())["fail_reason"] == ""
    assert task.verify(t1, tmp_path).passed


def test_item_budget_is_one_shared_setting(monkeypatch):
    """SB_ITEM_BUDGET_S, read by the mode and by the lab's SkillOpt wrapper alike; 3600 s
    unless set (SkillOpt ships 600)."""
    monkeypatch.delenv(sb_mode.ITEM_BUDGET_ENV, raising=False)
    assert sb_mode.item_budget_s() == 3600 != sb_mode.SKILLOPT_ITEM_BUDGET_S
    monkeypatch.setenv(sb_mode.ITEM_BUDGET_ENV, "600")
    solver = sb_mode.Solver()
    assert solver.values()["item_budget_s"] == 600
    assert solver.graph().nodes["solve"].timeout_s == 600


# ── train-only gold feedback, in the library ────────────────────────────────

def test_gold_feedback_is_for_train_evaluations_only(tmp_path):
    for part in ("val", "test", "holdout"):
        with pytest.raises(task_module.TaskError, match="train evaluations only"):
            task_module.write_subset(tmp_path, part, 0.0, gold_feedback=True)
    task_module.write_subset(tmp_path, "train", 0.0)
    assert not task_module.gold_feedback_on(tmp_path)
    task_module.write_subset(tmp_path, "train", 0.0, gold_feedback=True)
    assert task_module.gold_feedback_on(tmp_path)
    # A subset file that names another part with the key set by hand is still off.
    (tmp_path / task_module.SUBSET).write_text(json.dumps({"part": "val", "gold_feedback": True}))
    assert not task_module.gold_feedback_on(tmp_path)


def test_the_outer_loop_refuses_gold_feedback_off_train(tmp_path):
    from refactory_dsh.loop import Loop, LoopConfig
    from refactory_dsh.outerloop import EvalRequest, OuterLoopError, Run

    loop = Loop("circle_packing/adaptive", LoopConfig(
        epochs=1, batch_size=1, optimizer="random", seed=3, runs_root=tmp_path / "run",
        max_evaluations=4), project=Path("."))
    loop.init()
    run = Run(_loop=loop)
    with pytest.raises(OuterLoopError, match="train evaluations only"):
        run.evaluate(loop.population[0], part="val", gold_feedback=True)
    with pytest.raises(OuterLoopError, match="train evaluations only"):
        run.evaluate_many([EvalRequest(candidate=loop.population[0], part="test",
                                       gold_feedback=True)])


def _pass_ctx(tmp_path: Path, part: str, flag: bool) -> RunContext:
    from refactory_dsh.items import prepare_pass_dir

    run = tmp_path / f"run-{part}-{flag}"
    task_module.write_subset(run, part, 0.0, gold_feedback=flag)
    pass_dir = prepare_pass_dir(run, "data", "t1", 0, task_ref=sb_mode.TASK)
    ctx = RunContext(pass_dir)
    ctx._declared_facts = ("train_feedback",)  # noqa: SLF001
    return ctx


def test_the_feedback_node_asks_the_task_only_on_gold_feedback_train(tmp_path, sb_data, monkeypatch):
    asked: list[str] = []

    def train_feedback(self, instance, workspace, **_kw):
        asked.append(instance.id)
        return {"passed": False, "feedback": f"from {instance.metadata['gold_path']}"}

    monkeypatch.setattr(tasks.SpreadsheetBenchTask, "train_feedback", train_feedback)
    from refactory_dsh import feedback

    assert feedback.train_feedback(_pass_ctx(tmp_path, "val", False)) == {"train_feedback": None}
    assert feedback.train_feedback(_pass_ctx(tmp_path, "train", False)) == {"train_feedback": None}
    assert asked == []
    on = feedback.train_feedback(_pass_ctx(tmp_path, "train", True))
    assert asked == ["t1"] and "1_t1_golden.xlsx" in on["train_feedback"]["feedback"]
    loose = RunContext(tmp_path / "loose")
    loose._declared_facts = ("train_feedback",)  # noqa: SLF001
    with pytest.raises(feedback.FeedbackError, match="inside a data pass"):
        feedback.train_feedback(loose)
    loose._declared_facts = ("a", "b")  # noqa: SLF001
    with pytest.raises(feedback.FeedbackError, match="exactly one fact"):
        feedback.train_feedback(loose)


def _issues(graph: nn.Graph, protected: tuple[str, ...]) -> list[Any]:
    return [i for i in validate(graph, sb_mode.Solver().values(),
                                declared_knobs=sb_mode.Solver.declare(),
                                grader_side=protected, graders=())
            if i.code == "V17"]


def test_v17_lets_only_a_protected_train_feedback_node_through():
    solver = sb_mode.Solver()
    graph = solver.graph()
    assert _issues(graph, solver.protected_nodes()) == []
    unprotected = _issues(graph, ("data", "finalize"))
    assert [i.node for i in unprotected] == ["gold_check"]
    assert "not protected" in unprotected[0].message
    # The outer loop validates every edit with the mode's grader scope.
    assert solver.grader_scope()["grader_side"] == ("data", "gold_check", "finalize")


def test_v17_holds_a_train_feedback_node_to_its_shape():
    from dataclasses import replace

    solver = sb_mode.Solver()
    graph = solver.graph()
    graph.nodes["gold_check"] = replace(graph.nodes["gold_check"], reads=("turn.json",))
    (issue,) = _issues(graph, solver.protected_nodes())
    assert "declares reads" in issue.message


def test_v17_still_refuses_grade_consumers():
    """The exception is the feedback fact only: a pass node reading the graded records is
    still refused."""
    from dataclasses import replace

    solver = sb_mode.Solver()
    graph = solver.graph()
    graph.nodes["next_turn"] = replace(graph.nodes["next_turn"],
                                       inputs=("train_feedback", "item_scores"))
    issues = {i.node: i.message for i in _issues(graph, solver.protected_nodes())}
    assert "item_scores" in issues["next_turn"]
    # What it writes is then grade-derived, and so is everything made from that.
    assert {"frame", "solve", "run_turn"} <= set(issues)


# ── the analysts' view ──────────────────────────────────────────────────────

def test_skillopt_view_rebuilds_the_rollout(tmp_path, sb_env):
    result, prompts = _walk(tmp_path, "train", ["no fence at all", _good(41), _good(42)],
                            gold_feedback=True)
    (record,) = result.item_records
    view = skillopt_view.rollout_result(record, {"skill": port.INITIAL_SKILL})
    assert view["target_system_prompt"] == FIXTURE["skill_initial_system"]
    assert view["target_user_prompt"].startswith("# Instruction\nPut 42 in B1.")
    assert view["spreadsheet_preview"].startswith("## Sheet: Sheet1  (dim=A1:A1")
    roles = [m["role"] for m in view["conversation"]]
    assert roles == ["assistant", "user", "assistant", "user", "assistant", "system"]
    assert view["conversation"][-1]["content"].startswith(
        "[POST-EXECUTION VERIFICATION]\n\n## Eval Result (case 1): PASS")
    assert view["hard"] == 1 and view["n_turns"] == 3


# ── both runtimes ───────────────────────────────────────────────────────────

NODE = shutil.which("node")
REPO_ROOT = Path(__file__).resolve().parents[2]
DRAIN = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"


def _spine(tmp_path: Path, part: str, answers: list[str], *, gold_feedback: bool) -> tuple[list[dict], list[str]]:
    from refactory_dsh import lowering

    mode, state = _state()
    work = tmp_path / f"spine-{part}"
    work.mkdir(parents=True)
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()
                                       if p != "train"}, train_ids=["t1"])
    task_module.write_subset(work, part, 0.0, gold_feedback=gold_feedback)
    lowering.write_declarations(state, work)
    graph_path, _ = lowering.write_mode_graph(state, tmp_path / f"graphs-{part}", name="sb")
    (work / "responses.json").write_text(json.dumps({"solve": answers}))
    record = tmp_path / f"drain-{part}.jsonl"
    # Knob values reach the spine's ops as they do in a real run (`dsh_runtime`).
    env = {**os.environ, **lowering.knob_environment(state),
           "REFACTORY_WORKDIR": str(work), "DRAIN_RECORD": str(record),
           "PATH": f"{REPO_ROOT / '.venv/bin'}:{os.environ.get('PATH', '')}"}
    proc = subprocess.run(
        [str(NODE), "--import", "tsx/esm", str(DRAIN), str(graph_path), str(work),
         str(REPO_ROOT / ".venv/bin/python"), "120000", str(work / "responses.json")],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600, env=env)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    records = json.loads((work / "facts.json").read_text())["item_scores"]
    prompts = [json.loads(line)["prompt"] for line in record.read_text().splitlines()
               if json.loads(line).get("node") == "solve" and "prompt" in json.loads(line)]
    return records, prompts


@pytest.mark.skipif(NODE is None or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/"
                                         "@deepseek-ai/cordis").exists(),
                    reason="the spine drain needs Node and the installed workspace")
@pytest.mark.parametrize("part,flag,answers", [
    ("train", True, ["no fence at all", _good(41), _good(42)]),
    ("val", False, ["", _good(8), _good(7)]),
])
def test_the_spine_runs_the_loop_as_the_walker_does(tmp_path, sb_env, part, flag, answers):
    """The same scripted replies on both runtimes give the same prompts, byte for byte,
    and the same records; gold feedback reaches the train rollout's prompts and never
    the val rollout's."""
    import re

    walked, walker_prompts = _walk(tmp_path, part, answers, gold_feedback=flag)
    records, spine_prompts = _spine(tmp_path, part, answers, gold_feedback=flag)
    # A traceback names the runner's temporary directory, different on every run.
    temp = re.compile(r"(skillopt-generated|sb-item)-[A-Za-z0-9_]+")
    assert [temp.sub("<tmp>", p) for p in spine_prompts] == \
        [temp.sub("<tmp>", p) for p in walker_prompts]
    assert [(r["id"], r["passed"], r["fail_reason"]) for r in records] == \
        [(r["id"], r["passed"], r["fail_reason"]) for r in walked.item_records]
    feedback_seen = any(EVAL_FEEDBACK_HEAD in p for p in spine_prompts)
    assert feedback_seen is (part == "train")


# ── the real data (lab) ─────────────────────────────────────────────────────

@pytest.mark.skipif(not (os.environ.get("SB_DATA") and os.environ.get("SB_SKILLOPT_RUN")),
                    reason="needs the release data (SB_DATA) and a SkillOpt run (SB_SKILLOPT_RUN)")
def test_parity_on_skillopts_recorded_run():
    """§6.1, 6.4 (cached) and 6.5 on a real SkillOpt run directory."""
    from refactory_packages.spreadsheetbench import parity

    run, data = Path(os.environ["SB_SKILLOPT_RUN"]), Path(os.environ["SB_DATA"])
    assert parity.check_prompts(run, data)["ok"]
    assert parity.check_grader(run, data, "cached", None)["ok"]
    assert parity.check_feedback(run, data, "cached")["ok"]


def test_startskill_takes_its_reference_as_input(tmp_path):
    """§6.7: the reference is SkillOpt's val start-skill pass from grader-check outputs
    (one per seed), never a constant; the port's side is its own passes."""
    from refactory_packages.spreadsheetbench import parity

    refs = []
    for seed, passed in ((42, 33), (43, 31), (44, 32)):
        path = tmp_path / f"s{seed}.json"
        path.write_text(json.dumps({"passes": {"selection_eval_baseline": {
            "graded": 40, "not_graded": 0, "port_pass": passed}}}))
        refs.append(path)
    near, far = tmp_path / "near.json", tmp_path / "far.json"
    near.write_text(json.dumps({"scores": [0.8, 0.8]}))
    far.write_text(json.dumps({"scores": [0.5, 0.52]}))
    assert parity.check_startskill(refs, [near], "selection_eval_baseline", 2.0)["ok"]
    result = parity.check_startskill(refs, [far], "selection_eval_baseline", 2.0)
    assert not result["ok"] and result["reference"] == [33 / 40, 31 / 40, 32 / 40]
