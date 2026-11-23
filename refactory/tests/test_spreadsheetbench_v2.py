"""SpreadsheetBench 2: the prepared data, the grouped split, V2's grader, the agent harness.

Expected grader values come from V2's own code, never from the port:
``fixtures/spreadsheetbench_v2_grader_parity.json`` was recorded by running SpreadsheetBench
2's ``evaluation/evaluation.py`` (``process_single_item``, ``compare_workbooks_with_regression``,
``compare_cell_value``) on workbooks built by `sb2_support` (`sb2_support.record_parity`).

Runtimes: the grader, split and prepare checks are pure. The smoke runs one item end to
end with a stubbed model on the walker (`nn.executor`, the scripted agent of
``test_spreadsheetbench_agent`` acting through the seal it is handed) and on the TS spine
(lowered, its solve step dispatched to the real worker, whose scripted ``dsh`` does the
same). The hidden-path check runs the real DSH command line, bridge and patched Claude
Code adapter with only the Agent SDK's ``query`` faked, on the walker. LibreOffice is a
stand-in that copies the file (`fake_lo`): its recalculation is V2's script, checked
against a real LibreOffice where one is installed (`test_recalculation_with_libreoffice`).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import task as task_module
from refactory_dsh.evaluators import WalkerEvaluator
from refactory_dsh.task import TaskError
from refactory_packages.spreadsheetbench import agent_port, recalc
from refactory_packages.spreadsheetbench import skillopt_port as port
from refactory_packages.spreadsheetbench_v2 import (
    agent_mode, agent_ops, evaluation_port as ev, open_spreadsheet, prepare, tasks)
from sb2_support import build_case, decode, make_workbook
from test_spreadsheetbench import sb_python  # noqa: F401 - fixture
from test_spreadsheetbench_agent import (
    AGENT_ROUTE, PROBE, SOLUTION, AgentStub, _agent_dsh, _claude, _sealed_and_pinned,
    needs_adapter, needs_spine)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PARITY = json.loads((HERE / "fixtures" / "spreadsheetbench_v2_grader_parity.json")
                    .read_text(encoding="utf-8"))


# ── V2's grader, against V2's own results ─────────────────────────────────────

@pytest.mark.parametrize("case", PARITY["cases"], ids=lambda c: c["name"])
def test_the_grade_is_v2s(tmp_path, case):
    """Each case graded by the port as V2's ``process_single_item`` graded it: accuracy,
    both ratios (the 99.8% rule and its rounding included), the cell counts and the
    error message."""
    expected = PARITY["v2"]["cases"][case["name"]]
    inp, gold, out = build_case(tmp_path, case)
    g = ev.grade(str(inp), str(gold), str(out), case["answer_position"])
    assert g.accuracy == expected["accuracy"]
    assert g.regression_accuracy == expected["regression_accuracy"]
    assert g.modification_accuracy == expected["modification_accuracy"]
    assert g.error_message == expected["error_message"]
    if case.get("output") is not None or "regression" in case:
        assert g.regression == expected["regression"]
        assert g.modification == expected["modification"]


def test_the_parity_fixture_covers_v2s_rules():
    """The fixture exercises what V2 is known for: the 99.8% rule on both sides of its
    rounding, 1% tolerance both ways, None==0, an error cell compared by formula, a sheet
    the output lacks, a missing output."""
    results = PARITY["v2"]["cases"]
    assert results["regression_998_pass"]["accuracy"] == 1.0
    assert results["regression_997_fail"]["accuracy"] == 0.0
    assert results["regression_rounding"]["accuracy"] == 1.0
    assert results["tolerance_in"]["accuracy"] == 1.0 and results["tolerance_out"]["accuracy"] == 0.0
    assert results["none_equals_zero"]["accuracy"] == 1.0
    assert results["error_formula_same"]["accuracy"] == 1.0
    assert results["error_formula_differs"]["accuracy"] == 0.0
    assert results["missing_output"]["error_message"] == "output file not exist"
    assert PARITY["v2"]["evaluation_py_sha256"] == prepare.load_manifest()["grader_sha256"]


@pytest.mark.parametrize("pair", PARITY["v2"]["pairs"], ids=lambda p: repr((p["a"], p["b"])))
def test_cell_comparison_is_v2s(pair):
    assert ev.compare_cell_value(decode(pair["a"]), decode(pair["b"])) is pair["value"]
    assert repr(ev.transform_value(decode(pair["a"]))) == pair["transform_a"]


def test_one_workbook_at_a_time_and_formulas_only_for_error_cells(tmp_path, monkeypatch):
    """Each workbook is loaded, read and closed before the next; the formula view only
    when an answer cell holds an Excel error."""
    loads: list[tuple[str, bool]] = []
    real = ev._load

    def counting(path, data_only):
        loads.append((Path(path).name, data_only))
        return real(path, data_only)

    monkeypatch.setattr(ev, "_load", counting)
    case = next(c for c in PARITY["cases"] if c["name"] == "exact")
    inp, gold, out = build_case(tmp_path / "plain", case)
    assert ev.grade(str(inp), str(gold), str(out), case["answer_position"]).passed
    assert loads == [("input.xlsx", True), ("golden.xlsx", True), ("output.xlsx", True)]
    loads.clear()
    case = next(c for c in PARITY["cases"] if c["name"] == "error_formula_same")
    inp, gold, out = build_case(tmp_path / "errors", case)
    assert ev.grade(str(inp), str(gold), str(out), case["answer_position"]).passed
    assert [d for _, d in loads] == [True, True, True, False, False, False]


def test_a_grader_failure_is_an_incident_with_its_message(tmp_path, monkeypatch):
    """V2 turns any exception into a silent 0 (evaluation.py:372-373, :446-448). Here an
    unreadable golden, or memory running out, is a harness incident naming why; an output
    the model wrote that cannot be opened is the model's failure, with its reason."""
    from refactory_dsh.items import error_kind

    case = next(c for c in PARITY["cases"] if c["name"] == "exact")
    inp, gold, out = build_case(tmp_path, case)
    broken = tmp_path / "broken.xlsx"
    broken.write_text("not a workbook")
    with pytest.raises(ev.GraderFailed, match="golden") as caught:
        ev.grade(str(inp), str(broken), str(out), case["answer_position"])
    assert error_kind(caught.value) == "harness_incident"
    with pytest.raises(ev.GraderFailed, match="input"):
        ev.grade(str(broken), str(gold), str(out), case["answer_position"])
    with pytest.raises(ev.OutputUnreadable):
        ev.grade(str(inp), str(gold), str(broken), case["answer_position"])

    def no_memory(path, data_only):
        raise MemoryError()

    monkeypatch.setattr(ev, "_load", no_memory)
    with pytest.raises(ev.GraderFailed, match="out of memory"):
        ev.grade(str(inp), str(gold), str(out), case["answer_position"])


def test_the_modes_of_the_dropped_tasks_are_named_by_v2s_rule():
    """evaluation.py:431-436: "Embedded" grades formulas, "Color" font colours: exactly the
    tasks the manifest drops."""
    assert ev.needs_mode("Embedded Hardcodes_input.xlsx") == "formula"
    assert ev.needs_mode("Inconsistent Color Coding_input.xlsx") == "font_color"
    assert ev.needs_mode("Double Counting_input.xlsx") == ""


# ── the manifest: drops and the grouped split ─────────────────────────────────

MANIFEST = prepare.load_manifest()


def test_the_split_never_puts_a_group_in_two_parts():
    """Goldens are shared within a Debugging workbook and an FM project, and sibling inputs
    hold each other's answers: a whole group goes to one part."""
    tasks_by_id = {t["id"]: t for t in MANIFEST["tasks"]}
    seen: dict[tuple[str, str], str] = {}
    for part, ids in MANIFEST["parts"].items():
        for item in ids:
            key = (tasks_by_id[item]["category"], tasks_by_id[item]["group"])
            assert seen.setdefault(key, part) == part, (key, part, seen[key])
    goldens: dict[str, set[str]] = {}
    part_of = {item: part for part, ids in MANIFEST["parts"].items() for item in ids}
    for task in MANIFEST["tasks"]:
        goldens.setdefault(f"{task['category']}/{task['golden']['source']}", set()).add(part_of[task["id"]])
    assert all(len(parts) == 1 for parts in goldens.values())


def test_the_parts_and_drops():
    """277 tasks in 66 / 38 / 173; 44 dropped, each with its reason."""
    parts = MANIFEST["parts"]
    assert [len(parts[p]) for p in ("train", "val", "test")] == [66, 38, 173]
    ids = [i for p in parts.values() for i in p]
    assert len(ids) == len(set(ids)) == len(MANIFEST["tasks"]) == 277
    category = {t["id"]: t["category"] for t in MANIFEST["tasks"]}
    count = lambda part, cat: sum(category[i] == cat for i in parts[part])  # noqa: E731
    assert [count("train", c) for c in prepare.CATEGORIES] == [16, 25, 25]
    assert [count("val", c) for c in prepare.CATEGORIES] == [8, 15, 15]
    reasons = [d["reason"] for d in MANIFEST["dropped"]]
    assert (reasons.count(prepare.DROP_VISUALIZATION), reasons.count(prepare.DROP_EMBEDDED),
            reasons.count(prepare.DROP_COLOR)) == (24, 10, 10)
    for task in MANIFEST["tasks"]:
        assert ev.needs_mode(os.path.basename(task["input"]["source"])) == ""


def test_the_split_is_the_seeded_draw():
    groups: dict[str, list[str]] = {}
    for task in MANIFEST["tasks"]:
        groups.setdefault(task["category"], [])
        if task["group"] not in groups[task["category"]]:
            groups[task["category"]].append(task["group"])
    assert prepare.split_groups(groups, MANIFEST["split"]["seed"]) == MANIFEST["split"]["groups"]
    assert MANIFEST["split"]["groups"]["Template"]["train"] == prepare.TEMPLATE_GROUPS["train"]


def test_drop_reasons():
    assert prepare.drop_reason("Visualization", {}) == prepare.DROP_VISUALIZATION
    row = lambda name: {"spreadsheet_path": f"spreadsheet/01_Debugging/input_files/{name}"}  # noqa: E731
    assert prepare.drop_reason("Debugging", row("Embedded Hardcode_input.xlsx")) == prepare.DROP_EMBEDDED
    assert prepare.drop_reason("Debugging", row("Inconsistent Color Coding_input.xlsx")) == prepare.DROP_COLOR
    assert prepare.drop_reason("Debugging", row("Unit Mismatch_input.xlsx")) == ""
    assert prepare.drop_reason("Template", row("Color_input.xlsx")) == ""


# ── a synthetic release, prepared ─────────────────────────────────────────────

ONE = [{"title": "Sheet1", "cells": {"A1": 1}}]
#: (category, group, source id, input path in the release, golden path, instruction, n)
RELEASE = [
    ("Debugging", "01_Debugging", "01_01", "spreadsheet/01_Debugging/input_files/Double Counting_input.xlsx",
     "spreadsheet/01_Debugging/01_golden.xlsx", "Put 42 in B1.", 42),
    ("Debugging", "01_Debugging", "01_02", "spreadsheet/01_Debugging/input_files/Unit Mismatch_input.xlsx",
     "spreadsheet/01_Debugging/01_golden.xlsx", "Put 42 in B1.", 42),
    ("Template", "01_bond_accounting", "01_01", "spreadsheet/01_bond_accounting/01_01_input.xlsx",
     "spreadsheet/01_bond_accounting/01_01_golden.xlsx", "Put 7 in B1.", 7),
    ("Template", "02_cash_sweep", "02_01", "spreadsheet/02_cash_sweep/02_01_input.xlsx",
     "spreadsheet/02_cash_sweep/02_01_golden.xlsx", "Put 9 in B1.", 9),
]
IDS = [prepare.item_id(category, sid) for category, _g, sid, *_ in RELEASE]
PARTS = {"train": IDS[:2], "val": [IDS[2]], "test": [IDS[3]]}
VAL = IDS[2]


def _rel(index: int) -> tuple[str, str]:
    """The prepared input and golden of ``RELEASE[index]``, relative to the data folder."""
    category, _group, _sid, _inp, gold, *_ = RELEASE[index]
    return prepare.prepared_paths({"id": IDS[index], "category": category,
                                   "golden": {"source": gold}})


def make_release(root: Path) -> tuple[Path, Path]:
    """A release folder in V2's layout and a manifest for it; returns both."""
    source = root / "spreadsheetbench-v2"
    rows: dict[str, list[dict[str, Any]]] = {c: [] for c in prepare.CATEGORIES}
    entries = []
    for category, group, sid, inp, gold, instruction, n in RELEASE:
        folder = source / category
        make_workbook(folder / inp, ONE)
        if not (folder / gold).exists():  # a shared golden is written once (its checksum)
            make_workbook(folder / gold, [{"title": "Sheet1", "cells": {"A1": 1, "B1": n}}])
        rows[category].append({"id": sid, "instruction": instruction, "spreadsheet_path": inp,
                               "golden_response_path": gold, "answer_position": "'Sheet1'!A1:B2"})
        entries.append({"id": prepare.item_id(category, sid), "category": category, "group": group,
                        "source_id": sid,
                        "input": {"source": inp, "sha256": prepare.sha256(folder / inp)},
                        "golden": {"source": gold, "sha256": prepare.sha256(folder / gold)}})
    for category, items in rows.items():
        (source / category).mkdir(parents=True, exist_ok=True)
        (source / category / "dataset.json").write_text(json.dumps(items))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "dataset_json_sha256": {c: prepare.sha256(source / c / "dataset.json")
                                for c in prepare.CATEGORIES},
        "parts": PARTS, "dropped": [], "tasks": entries}))
    return source, manifest


def _copy(src: str, dst: str) -> tuple[bool, str]:
    shutil.copyfile(src, dst)
    return True, ""


@pytest.fixture()
def fake_lo(tmp_path, monkeypatch):
    """LibreOffice stand-in: ``SB_SOFFICE`` names a ``soffice``, and the ``python`` beside
    it runs V2's helper's part as a copy (no recalculation), in every process."""
    folder = tmp_path / "fake-lo"
    folder.mkdir()
    (folder / "soffice").write_text("#!/bin/sh\necho 'LibreOffice stand-in'\n")
    # The helper's last two arguments are the copy and where to store it (its flags vary).
    (folder / "python").write_text('#!/bin/sh\nfor a; do src=$dst; dst=$a; done\n'
                                   'cp "$src" "$dst" && echo \'{"ok": true}\'\n')
    for name in ("soffice", "python"):
        (folder / name).chmod(0o755)
    monkeypatch.setenv(recalc.SOFFICE_ENV, str(folder / "soffice"))
    return folder


@pytest.fixture()
def sb2_data(tmp_path, monkeypatch, fake_lo):
    source, manifest = make_release(tmp_path / "release")
    monkeypatch.setenv(prepare.MANIFEST_ENV, str(manifest))
    out = tmp_path / "prepared"
    prepare.prepare(source, out)
    prepare.check_grades(out)
    monkeypatch.setenv(tasks.DATA_ENV, str(out))
    return out


def test_prepare_renames_freezes_and_records(tmp_path, monkeypatch, fake_lo):
    """Inputs are renamed to input.xlsx under their task (no planted-error name survives),
    each golden is written once, every file goes through the recalculation, and
    prepared.json records the manifest's checksum and each file's."""
    source, manifest = make_release(tmp_path / "release")
    monkeypatch.setenv(prepare.MANIFEST_ENV, str(manifest))
    seen: list[str] = []

    def recording(src, dst):
        seen.append(src)
        return _copy(src, dst)

    record = prepare.prepare(source, tmp_path / "out", recalculate=recording)
    assert sorted(record["files"]) == sorted({rel for i in range(len(RELEASE)) for rel in _rel(i)})
    assert len(record["files"]) == 7 and _rel(0)[1] == _rel(1)[1]  # the shared golden
    assert len(seen) == 7  # the shared golden once
    assert record["manifest_sha256"] == prepare.sha256(manifest)
    listed = json.loads((tmp_path / "out" / prepare.TASKS_FILE).read_text())
    assert not any("Double" in json.dumps(row) or "Mismatch" in json.dumps(row) for row in listed)
    assert prepare.check(tmp_path / "out")["files"] == record["files"]
    # Resumes: nothing recorded is done again.
    seen.clear()
    prepare.prepare(source, tmp_path / "out", recalculate=recording)
    assert seen == []


def test_prepare_refuses_a_different_release(tmp_path, monkeypatch, fake_lo):
    source, manifest = make_release(tmp_path / "release")
    monkeypatch.setenv(prepare.MANIFEST_ENV, str(manifest))
    make_workbook(source / "Template" / RELEASE[2][3], [{"title": "Sheet1", "cells": {"A1": 2}}])
    with pytest.raises(prepare.PrepareError, match="not the release's file"):
        prepare.prepare(source, tmp_path / "out", recalculate=_copy)


def test_a_failed_recalculation_stops_prepare(tmp_path, monkeypatch, fake_lo):
    source, manifest = make_release(tmp_path / "release")
    monkeypatch.setenv(prepare.MANIFEST_ENV, str(manifest))
    with pytest.raises(prepare.PrepareError, match="could not recalculate"):
        prepare.prepare(source, tmp_path / "out", recalculate=lambda s, d: (False, "boom"))


def test_the_task_refuses_unprepared_or_changed_data(tmp_path, monkeypatch, sb2_data):
    assert [i.id for i in tasks.SpreadsheetBenchV2Task().instances()] == \
        [*PARTS["train"], *PARTS["val"], *PARTS["test"]]
    with pytest.raises(TaskError, match="not prepared"):
        tasks.SpreadsheetBenchV2Task(str(tmp_path / "nothing")).instances()
    golden = sb2_data / _rel(2)[1]
    make_workbook(golden, [{"title": "Sheet1", "cells": {"A1": 1, "B1": 8}}])
    with pytest.raises(TaskError, match="changed since prepare"):
        tasks.SpreadsheetBenchV2Task().instances()
    other = tmp_path / "other-manifest.json"
    other.write_text(Path(os.environ[prepare.MANIFEST_ENV]).read_text() + " ")
    monkeypatch.setenv(prepare.MANIFEST_ENV, str(other))
    with pytest.raises(TaskError, match="another manifest"):
        tasks.SpreadsheetBenchV2Task().instances()


def test_grading_without_libreoffice_is_refused(tmp_path, monkeypatch, sb2_data):
    monkeypatch.setenv(recalc.SOFFICE_ENV, str(tmp_path / "no-soffice"))
    with pytest.raises(TaskError, match="LibreOffice"):
        tasks.SpreadsheetBenchV2Task().instances()


def test_the_output_is_recalculated_before_grading(tmp_path, sb2_data, monkeypatch):
    """V2's pipeline: open_spreadsheet.py on the output, then evaluation.py. The grade reads
    the recalculated copy; the output itself is never changed."""
    task = tasks.SpreadsheetBenchV2Task()
    item = next(i for i in task.instances() if i.id == VAL)
    pred = tmp_path / "pred.xlsx"
    make_workbook(pred, [{"title": "Sheet1", "cells": {"A1": 1, "B1": "=3+4"}}])
    before = pred.read_bytes()
    calls: list[str] = []

    def recalculated(src, dst, **_):
        calls.append(src)
        make_workbook(Path(dst), [{"title": "Sheet1", "cells": {"A1": 1, "B1": {"f": "=3+4", "v": 7}}}])
        return True, ""

    monkeypatch.setattr(open_spreadsheet, "recalculate", recalculated)
    g, note = task.grade(item, pred)
    assert g.passed and note == "" and calls == [str(pred)] and pred.read_bytes() == before
    # Uncalculated, the formula reads as an empty cell, as V2's grader would read it.
    assert not ev.grade(item.metadata["input_path"], item.metadata["gold_path"], str(pred),
                        item.metadata["answer_position"]).passed


def test_an_output_libreoffice_cannot_open_is_graded_as_written(tmp_path, sb2_data, monkeypatch):
    """V2's script leaves a file it cannot open as it was; here so too, when LibreOffice
    works on a known-good workbook. When it does not, the grader is broken: an incident."""
    task = tasks.SpreadsheetBenchV2Task()
    item = next(i for i in task.instances() if i.id == VAL)
    pred = tmp_path / "pred.xlsx"
    make_workbook(pred, [{"title": "Sheet1", "cells": {"A1": 1, "B1": 7}}])
    monkeypatch.setattr(open_spreadsheet, "recalculate", lambda s, d, **_: (False, "cannot open"))
    monkeypatch.setattr(recalc, "soffice_works", lambda: True)
    g, note = task.grade(item, pred)
    assert g.passed and "cannot open" in note
    monkeypatch.setattr(recalc, "soffice_works", lambda: False)
    with pytest.raises(tasks.RecalcFailed):
        task.grade(item, pred)


@pytest.mark.skipif(not os.environ.get("SB2_REAL_SOFFICE") and recalc.soffice() is None,
                    reason="needs LibreOffice (or SB2_REAL_SOFFICE naming one)")
def test_recalculation_with_libreoffice(tmp_path, monkeypatch):
    """With a real LibreOffice, V2's script fills in a formula's value, iterating a
    circular reference."""
    if os.environ.get("SB2_REAL_SOFFICE"):
        monkeypatch.setenv(recalc.SOFFICE_ENV, os.environ["SB2_REAL_SOFFICE"])
    src = make_workbook(tmp_path / "in.xlsx", [{"title": "S", "cells": {
        "A1": 2, "A2": "=A1*3", "B1": "=B2*0.5+1", "B2": "=B1*0.5"}}])
    ok, why = open_spreadsheet.recalculate(str(src), str(tmp_path / "out.xlsx"))
    assert ok, why
    import openpyxl

    ws = openpyxl.load_workbook(tmp_path / "out.xlsx", data_only=True)["S"]
    assert ws["A2"].value == 6
    assert abs(ws["B1"].value - 4 / 3) < 1e-3  # iterative calculation converged


# ── the agent harness on V2: walker and spine ─────────────────────────────────

@pytest.fixture()
def sb2_env(sb2_data, sb_python, monkeypatch, tmp_path):
    from refactory_packages.spreadsheetbench import interpreter

    monkeypatch.setenv(interpreter.ENV, sb_python)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return sb2_data


def _state(**knobs: Any) -> dict[str, Any]:
    from refactory_dsh.nn.edits import configure

    state = agent_mode.Agent().state_dict()
    for name, value in {"data_parallelism": 1, **knobs}.items():
        state = configure(state, name, value)
    return state


def _walk(tmp_path: Path, part: str, model: Any, *, expect_ok: bool = True) -> Any:
    state = _state()
    work = tmp_path / f"walk-{part}"
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()})
    task_module.write_subset(work, part, 0.0)
    result = WalkerEvaluator(model=model, project=Path("."), max_steps=400).evaluate(
        agent_mode.Agent(state=state), state, work)
    assert result.exit_ok or not expect_ok, result.error
    return result


def _gold(data: Path, index: int = 2) -> str:
    return str(data / _rel(index)[1])


def test_one_item_end_to_end_on_the_walker(tmp_path, sb2_env):
    """Walker: the val item; the solve step gets SkillOpt's prompt and workspace with V2's
    task text, works sealed with the pinned interpreter (the golden unreadable), writes
    solution.py, and the item is run and graded by V2's grade."""
    model = AgentStub(_gold(sb2_env))
    (record,) = _walk(tmp_path, "val", model).item_records
    assert (record["id"], record["passed"], record["score"]) == (VAL, True, 1.0), record
    (call,) = model.calls
    assert call["prompt"] == agent_port.SOLVE_PROMPT and call["system"] == ""
    assert sorted(call["files"]) == sorted(agent_mode.sb_agent_mode.WORKSPACE_READS)
    task = call["files"][agent_port.TASK_PATH].decode()
    assert task == agent_ops.build_task("Put 7 in B1.")
    # SWE-agent/config/spreadsheet.yaml:49-60, its two paths the workspace's names.
    assert task.startswith(
        "<instructions>\n\n## Task instructions\nYou need to process a spreadsheet file based on "
        "specific instructions.\n**Instruction:** Put 7 in B1.\n\n## Input and Output\nYou are "
        "provided with an input spreadsheet file and you need to generate an output spreadsheet "
        "file based on the output_path.\n**Input File:** input.xlsx\n**Output Path:** "
        "output.xlsx\n\n</instructions>\n\n## Codex Harness Task\n")
    assert "**Input File:** input.xlsx\n**Output Path:** output.xlsx" in task
    assert task.endswith(agent_port.CODEX_TASK_SECTION)
    assert call["files"][agent_port.SKILL_PATH].decode() == agent_port.build_codex_skill(port.INITIAL_SKILL)
    _sealed_and_pinned(call["seen"])
    assert record["prediction"].startswith("## Eval Result: PASS")


def test_a_wrong_answer_shows_v2s_mismatch_to_both_arms(tmp_path, sb2_env):
    """A wrong value: V2's mismatch line is the record's prediction, which SkillOpt's view
    appends after the final message and the default loop shows as what the item gave; both
    see the input's preview and the instruction."""
    from refactory_dsh.optim import analysts
    from refactory_dsh.task import item_metadata_reader
    from refactory_packages.skillopt.env import SkillOptEnv

    model = AgentStub(_gold(sb2_env), solution=SOLUTION.format(n=8))
    (record,) = _walk(tmp_path, "val", model, expect_ok=False).item_records
    assert not record["passed"]
    assert record["fail_reason"].startswith("eval-mismatch: Modification error at Sheet1!B1")
    assert "Modification error at Sheet1!B1: answer=7, output=8" in record["prediction"]
    preview = port.preview_workbook(str(sb2_env / _rel(2)[0]))
    view = SkillOptEnv.for_mode("spreadsheetbench_v2/agent").rollout_result(
        record, agent_mode.Agent().values())
    assert view["conversation"][-1]["content"] == f"[POST-EXECUTION VERIFICATION]\n\n{record['prediction']}"
    assert view["spreadsheet_preview"] == preview and view["task_type"] == "Template"
    assert view["target_user_prompt"] == agent_ops.build_task("Put 7 in B1.")
    context = item_metadata_reader(agent_mode.Agent().state_dict(), context=True)(record)
    assert context == {"spreadsheet_preview": preview}
    text = analysts.item_text({**record, "item_context": context})
    assert preview.strip() in text and "Put 7 in B1." in text and record["prediction"] in text


def test_an_exec_error_reaches_both_arms(tmp_path, sb2_env):
    from refactory_packages.skillopt.env import SkillOptEnv

    model = AgentStub(_gold(sb2_env), solution="raise RuntimeError('x' * 700)\n")
    (record,) = _walk(tmp_path, "val", model, expect_ok=False).item_records
    assert record["fail_reason"].startswith("exec-error: RuntimeError")
    assert record["prediction"].startswith("## Execution (case 1)\nERROR: ")
    view = SkillOptEnv.for_mode("spreadsheetbench_v2/agent").rollout_result(
        record, agent_mode.Agent().values())
    assert view["conversation"][-1]["content"].endswith(record["prediction"])


def _drain(tmp_path: Path, client: Any, part: str = "val") -> Any:
    """The part's items, lowered with dispatched workers on ``client``'s route and drained by
    the real spine (as `test_spreadsheetbench_agent._drain`, on the V2 mode)."""
    import subprocess

    from refactory_dsh import lowering, workers

    state = _state()
    work = tmp_path / "spine"
    work.mkdir()
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()})
    task_module.write_subset(work, part, 0.0)
    lowering.write_declarations(state, work)

    def dispatched(workflow: Any) -> Any:
        return workers.lower_workflow(workflow, tmp_path / "specs", client.context(),
                                      lowering.node_declarations(state))

    graph_path, _ = lowering.write_mode_graph(state, tmp_path / "graphs", name="sb2a",
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
    return json.loads((work / "facts.json").read_text())["item_scores"]


@needs_spine
def test_one_item_end_to_end_on_the_spine(tmp_path, sb2_env):
    """Spine: the same item, lowered and drained, its solve step dispatched to the real
    worker on SkillOpt's route; the scripted agent works sealed with V2's task text, and
    the item is graded by V2's grade in the spine's item process."""
    from refactory_dsh.model import DshModelClient

    cli, log = _agent_dsh(tmp_path, _gold(sb2_env))
    client = DshModelClient(**AGENT_ROUTE, dsh_cli=str(cli), attempts=1, timeout_s=120)
    (record,) = _drain(tmp_path, client)
    assert (record["id"], record["passed"]) == (VAL, True), record
    (call,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert call["prompt"] == agent_port.SOLVE_PROMPT
    assert call["config"]["env"]["REFACTORY_SEALED_TOOLS"] == "Read,Bash"
    _sealed_and_pinned(call["seen"])


# ── what the agent may not read ───────────────────────────────────────────────

@pytest.fixture
def sealed_sdk_v2(tmp_path, monkeypatch, sb2_env):
    """The sealed fake Agent SDK of the SB agent tests, pointed at V2's data: the item's
    golden, a sibling task's input (it holds the answer), the shared Debugging golden,
    ``tasks.json`` and the prepared directory."""
    from test_claude_code_route import _HOOKS, _REGISTER
    from test_spreadsheetbench_agent import _SEALED_SDK

    hooks = tmp_path / "sdk-hooks"
    hooks.mkdir()
    (hooks / "fake-sdk.mjs").write_text(_SEALED_SDK)
    (hooks / "hooks.mjs").write_text(_HOOKS)
    (hooks / "register.mjs").write_text(_REGISTER)
    record = tmp_path / "sdk-query.json"
    gold = _gold(sb2_env)
    monkeypatch.setenv("NODE_OPTIONS", f"--import {(hooks / 'register.mjs').as_uri()}")
    monkeypatch.setenv("FAKE_SDK_RECORD", str(record))
    monkeypatch.setenv("FAKE_AGENT_WRITE", "cat > solution.py <<'EOF'\n" + SOLUTION.format(n=7) + "EOF")
    monkeypatch.setenv("FAKE_AGENT_PROBE", PROBE.format(gold=gold))
    monkeypatch.setenv("FAKE_AGENT_OUTSIDE", gold)
    monkeypatch.setenv("FAKE_AGENT_FORBIDDEN", json.dumps([
        gold, str(sb2_env / _rel(3)[0]), str(sb2_env / _rel(0)[1]), str(sb2_env / _rel(1)[0]),
        str(sb2_env / prepare.TASKS_FILE), str(sb2_env)]))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    monkeypatch.delenv("REFACTORY_WORKER_CONTEXT", raising=False)
    return record


@needs_adapter
def test_the_agent_cannot_read_a_golden_or_a_sibling_tasks_input(tmp_path, sealed_sdk_v2, sb2_env):
    """From inside the agent's workspace (walker, real adapter): `cat` of the golden, a
    sibling task's input, the shared Debugging golden, tasks.json and the prepared folder
    fails under the seal, and a Read of each is refused by the hook (`tool_hidden`)."""
    state = _state()
    work = tmp_path / "walk"
    task_module.make_manifest(work, **{f"{p}_ids": ids for p, ids in PARTS.items()})
    task_module.write_subset(work, "val", 0.0)
    result = WalkerEvaluator(model=_claude(tmp_path), project=Path("."), max_steps=400) \
        .evaluate(agent_mode.Agent(state=state), state, work)
    assert result.exit_ok, result.error
    sent = json.loads(sealed_sdk_v2.read_text())
    assert sent["forbidden"] and all(v == {"read": "deny", "cat": "BLOCKED"}
                                     for v in sent["forbidden"].values()), sent["forbidden"]
    assert sent["decisions"]["read_outside"] == "deny" and sent["decisions"]["read_link"] == "deny"
    assert "GOLD-HIDDEN" in sent["bash"]["probe"]
    (record,) = result.item_records
    assert record["passed"], record


def test_model_code_is_blind_to_the_prepared_data(tmp_path, sb2_env):
    """The code sandbox hides the prepared directory (`tasks.data_roots`), not SB's."""
    hostile = ("import os\n"
               f"print('SEEN' if os.path.exists({str(sb2_env)!r}) else 'HIDDEN')\n"
               "raise SystemExit(3)\n")
    (record,) = _walk(tmp_path, "val", AgentStub(_gold(sb2_env), solution=hostile),
                      expect_ok=False).item_records
    assert "HIDDEN" in record["prediction"] and "SEEN" not in record["prediction"], record


# ── review fixes: names, hidden lists, tampering, LibreOffice failures, exclusions ──

#: What the release calls these tasks: their source ids, groups and file names (a
#: Debugging source id names its planted error). None may reach the agent or the analysts.
RELEASE_NAMES = ("01_01", "01_02", "02_01", "01_Debugging", "bond_accounting", "cash_sweep",
                 "Double Counting", "Unit Mismatch", "_golden", "_input.xlsx")


class _LookingStub(AgentStub):
    """The agent of `AgentStub`, which also records its working folder, environment and
    what ``pwd``, ``env`` and ``ls`` say through its seal."""

    def complete(self, prompt: str, *, system: str = "", workdir: object = None,
                 cwd: object = None, env: object = None, confine: bool = False,
                 seal: dict[str, Any] | None = None) -> str:
        import subprocess

        reply = super().complete(prompt, system=system, workdir=workdir, cwd=cwd, env=env,
                                 confine=confine, seal=seal)
        looked = subprocess.run([*seal["runner"], "--", "/bin/bash", "-c", "pwd; env; ls -laR"],
                                cwd=str(cwd), capture_output=True, text=True, timeout=120)
        self.calls[-1].update(cwd=str(cwd), env=json.dumps(env or {}), shell=looked.stdout + looked.stderr)
        return reply


def test_the_agent_and_analysts_see_no_release_name(tmp_path, sb2_env):
    """Ids and prepared folders are opaque: no source id, group or original file name in
    the agent's task text, files, folder, environment or shell, nor in the record, either
    arm's analyst view or the item's context."""
    from refactory_dsh.optim import analysts
    from refactory_dsh.task import item_metadata_reader
    from refactory_packages.skillopt.env import SkillOptEnv

    model = _LookingStub(_gold(sb2_env, 0))
    records = _walk(tmp_path, "train", model, expect_ok=False).item_records
    assert len(records) == 2 and len(model.calls) == 2
    read = item_metadata_reader(agent_mode.Agent().state_dict(), context=True)
    seen = []
    for call in model.calls:
        seen += [call["prompt"], call["cwd"], call["env"], call["shell"], call["seen"],
                 *call["files"], *(v.decode(errors="replace") for v in call["files"].values())]
    for record in records:
        context = read(record)
        seen += [json.dumps(record), json.dumps(context),
                 json.dumps(SkillOptEnv.for_mode("spreadsheetbench_v2/agent").rollout_result(
                     record, agent_mode.Agent().values())),
                 analysts.item_text({**record, "item_context": context})]
    text = "\n".join(seen)
    leaked = [name for name in RELEASE_NAMES if name in text]
    assert leaked == [], leaked
    assert IDS[0] in text  # the opaque id is what is shown


def _covered(path: Path, hidden: list[str]) -> bool:
    target = Path(path).resolve()
    return any(target == Path(h).resolve() or Path(h).resolve() in target.parents for h in hidden)


def test_the_hidden_lists_hold_the_dataset_goldens_and_siblings(tmp_path, sb2_data, monkeypatch):
    """[any OS] Model code's hide-list (`agent_ops.finalize`; macOS's only protection,
    `spreadsheetbench.sandbox`) and the sealed steps' (`tasks.tool_hidden`) both cover the
    prepared folder: every golden, every sibling input, tasks.json."""
    from refactory_packages.spreadsheetbench import agent_ops as sb_agent_ops
    from refactory_packages.spreadsheetbench import sandbox

    private = [sb2_data / _rel(2)[1], sb2_data / _rel(0)[1], sb2_data / _rel(1)[0],
               sb2_data / _rel(3)[0], sb2_data / prepare.TASKS_FILE, sb2_data]
    work = tmp_path / "pass"
    work.mkdir()
    sealed = tasks.tool_hidden(work)
    assert all(_covered(p, sealed) for p in private), sealed
    handed: dict[str, Any] = {}
    monkeypatch.setattr(sb_agent_ops, "finalize_item",
                        lambda ctx, roots: handed.setdefault("roots", roots))
    agent_ops.finalize(object())
    model_code = sandbox.hidden_paths(handed["roots"])
    assert all(_covered(p, model_code) for p in private), model_code


def test_an_interpreter_under_the_prepared_data_is_refused(tmp_path, sb2_env, monkeypatch):
    """[both runtimes' `workspace.prepare`] A sealed step's interpreter that runs from inside
    SB2_DATA is refused (`tasks.tool_hidden` reaches the seal); outside, it is given."""
    from test_sealed_tools import _sealable
    from test_tool_steps_item_inputs import _bound, _layout

    from refactory_dsh import workspace as workspaces

    _sealable()
    _run, pass_dir, _workspace = _layout(tmp_path)
    (pass_dir / ".refactory" / "pass.json").write_text(json.dumps({"task": agent_mode.TASK}))
    _bound(monkeypatch, sb2_env / "venv")
    with pytest.raises(workspaces.WorkspaceError, match="may not read"):
        workspaces.prepare(pass_dir, "look", "scratch", route={"worker_backend": "dsh"},
                           tools=("bash",))
    _bound(monkeypatch, tmp_path / "elsewhere" / "venv")
    assert workspaces.prepare(pass_dir, "look", "scratch", route={"worker_backend": "dsh"},
                              tools=("bash",)) is not None


def test_a_changed_tasks_json_is_refused(sb2_data):
    listed = json.loads((sb2_data / prepare.TASKS_FILE).read_text())
    listed[0]["answer_position"] = "'Sheet1'!A1"
    (sb2_data / prepare.TASKS_FILE).write_text(json.dumps(listed))
    with pytest.raises(prepare.PrepareError, match="does not match"):
        prepare.check(sb2_data)
    with pytest.raises(TaskError, match="does not match"):
        tasks.SpreadsheetBenchV2Task().instances()


def test_a_resumed_prepare_redoes_a_file_changed_since(tmp_path, monkeypatch, fake_lo):
    """Resuming skips only files whose checksum still matches the record."""
    source, manifest = make_release(tmp_path / "release")
    monkeypatch.setenv(prepare.MANIFEST_ENV, str(manifest))
    out = tmp_path / "out"
    prepare.prepare(source, out, recalculate=_copy)
    make_workbook(out / _rel(2)[0], [{"title": "Sheet1", "cells": {"A1": "tampered"}}])
    seen: list[str] = []

    def recording(src, dst):
        seen.append(src)
        return _copy(src, dst)

    prepare.prepare(source, out, recalculate=recording)
    assert seen == [str(source / "Template" / RELEASE[2][3])]
    prepare.check(out)


def test_the_task_requires_the_grades_check_and_leaves_failing_goldens_out(tmp_path, monkeypatch,
                                                                         fake_lo):
    source, manifest = make_release(tmp_path / "release")
    monkeypatch.setenv(prepare.MANIFEST_ENV, str(manifest))
    out = tmp_path / "out"
    prepare.prepare(source, out)
    monkeypatch.setenv(tasks.DATA_ENV, str(out))
    with pytest.raises(TaskError, match="check --grades"):
        tasks.SpreadsheetBenchV2Task().instances()
    real = ev.grade

    def no_change_for_val(inp, gold, out_file, position):
        if VAL in inp and out_file == gold:
            return ev.Grade(exact=True, error_message="", regression={"correct": 2, "total": 2},
                            modification={"correct": 0, "total": 0}, regression_accuracy=1.0,
                            modification_accuracy=0.0, accuracy=0.0)
        return real(inp, gold, out_file, position)

    record = prepare.check_grades(out, grade=no_change_for_val)
    assert [row["id"] for row in record["sanity"]["gold_fails"]] == [VAL]
    task = tasks.SpreadsheetBenchV2Task()
    assert [i.id for i in task.instances()] == [*PARTS["train"], *PARTS["test"]]
    assert "changes no answer cell" in task.excluded[VAL]

    def broken(*_):
        raise ev.GraderFailed("out of memory")

    with pytest.raises(prepare.PrepareError, match="grader failed"):
        prepare.check_grades(out, grade=broken)


def _fake_python(folder: Path, body: str) -> None:
    (folder / "python").write_text(f"#!/bin/sh\n{body}\n")
    (folder / "python").chmod(0o755)


@pytest.mark.parametrize("body,document_fault", [
    ("echo '{\"ok\": false, \"stage\": \"open\", \"error\": \"Failed to open document\"}'", True),
    ("echo '{\"ok\": false, \"stage\": \"crash\", \"error\": \"bridge disposed\"}'", False),
    ("echo '{\"ok\": false, \"stage\": \"calc\", \"error\": \"store failed\"}'", False),
    ("kill -9 $$", False),
    ("sleep 30", False),
])
def test_only_a_document_libreoffice_cannot_open_is_its_own_fault(tmp_path, fake_lo, body,
                                                                   document_fault):
    """A document LibreOffice, running, cannot open is graded as written (V2's script
    leaves it); LibreOffice crashing, killed (out of memory), failing to store or past its
    process guard raises `RecalcError`."""
    _fake_python(fake_lo, body)
    src = make_workbook(tmp_path / "in.xlsx", ONE)
    if document_fault:
        assert open_spreadsheet.recalculate(str(src), str(tmp_path / "out.xlsx"), timeout_s=5) == \
            (False, "Failed to open document")
    else:
        with pytest.raises(open_spreadsheet.RecalcError):
            open_spreadsheet.recalculate(str(src), str(tmp_path / "out.xlsx"), timeout_s=2)


def _val_and_pred(tmp_path: Path):
    task = tasks.SpreadsheetBenchV2Task()
    item = next(i for i in task.instances() if i.id == VAL)
    pred = make_workbook(tmp_path / "pred.xlsx", [{"title": "Sheet1", "cells": {"A1": 1, "B1": 7}}])
    return task, item, pred


def test_libreoffice_failing_once_on_a_healthy_machine_is_retried(tmp_path, sb2_data, monkeypatch):
    """LibreOffice healthy: a failed recalculation is tried once more, and its result graded."""
    task, item, pred = _val_and_pred(tmp_path)
    calls: list[str] = []

    def flaky(src, dst, **_):
        calls.append(src)
        if len(calls) == 1:
            raise open_spreadsheet.RecalcError("LibreOffice failed (crash): bridge disposed")
        shutil.copyfile(src, dst)
        return True, ""

    monkeypatch.setattr(open_spreadsheet, "recalculate", flaky)
    monkeypatch.setattr(recalc, "soffice_works", lambda: True)
    g, note = task.grade(item, pred)
    assert g.passed and note == "" and len(calls) == 2


def test_an_output_that_hangs_a_healthy_libreoffice_twice_scores_zero(tmp_path, sb2_data,
                                                                      monkeypatch):
    """The output's doing, not an incident: no item may dodge its 0 by hanging LibreOffice."""
    task, item, pred = _val_and_pred(tmp_path)
    calls: list[str] = []

    def hangs(src, dst, **_):
        calls.append(src)
        raise open_spreadsheet.RecalcError("LibreOffice took more than 900 s (its process guard)")

    monkeypatch.setattr(open_spreadsheet, "recalculate", hangs)
    monkeypatch.setattr(recalc, "soffice_works", lambda: True)
    with pytest.raises(tasks.OutputUnrecalculable):
        task.grade(item, pred)
    assert len(calls) == 2
    work = tmp_path / "pass"
    (work / "item").mkdir(parents=True)
    shutil.copyfile(pred, work / "pred.xlsx")
    (work / "final.json").write_text(json.dumps({"fail_reason": "", "n_turns": 1}))
    verdict = task.verify(item, work)
    assert (verdict.passed, verdict.score) == (False, 0.0)
    assert verdict.details.startswith("recalc-failed: LibreOffice could not recalculate the output")


def test_libreoffice_failing_while_unhealthy_is_a_harness_incident(tmp_path, sb2_data, monkeypatch):
    from refactory_dsh.items import error_kind

    task, item, pred = _val_and_pred(tmp_path)
    health = iter([True, False])

    def hangs(*_a, **_k):
        raise open_spreadsheet.RecalcError("LibreOffice took more than 900 s (its process guard)")

    monkeypatch.setattr(open_spreadsheet, "recalculate", hangs)
    # Healthy before the retry, broken after it; and broken from the start.
    monkeypatch.setattr(recalc, "soffice_works", lambda: next(health))
    with pytest.raises(tasks.RecalcFailed) as caught:
        task.grade(item, pred)
    assert error_kind(caught.value) == "harness_incident"
    monkeypatch.setattr(recalc, "soffice_works", lambda: False)
    with pytest.raises(tasks.RecalcFailed):
        task.grade(item, pred)


def test_a_grades_check_of_an_older_tasks_json_is_refused(sb2_data):
    """A tasks.json changed and re-recorded in prepared.json after `check --grades` leaves
    that check stale: refused until it runs again."""
    listed = json.loads((sb2_data / prepare.TASKS_FILE).read_text())
    listed[0]["answer_position"] = "'Sheet1'!A1"
    (sb2_data / prepare.TASKS_FILE).write_text(json.dumps(listed))
    record = json.loads((sb2_data / prepare.PREPARED_FILE).read_text())
    record["tasks_sha256"] = prepare.sha256(sb2_data / prepare.TASKS_FILE)
    (sb2_data / prepare.PREPARED_FILE).write_text(json.dumps(record))
    prepare.check(sb2_data)  # the checksums match: only the grades check is stale
    with pytest.raises(TaskError, match="check --grades"):
        tasks.SpreadsheetBenchV2Task().instances()


def test_the_health_check_after_a_failure_is_fresh(tmp_path, sb2_data, monkeypatch):
    """A cached "healthy" answer is not trusted after a failure: the real reference check
    runs again (here it fails: the stand-in soffice converts nothing), so the item is an
    incident rather than the output's 0."""
    import time

    task, item, pred = _val_and_pred(tmp_path)

    def hangs(*_a, **_k):
        raise open_spreadsheet.RecalcError("LibreOffice took more than 900 s (its process guard)")

    monkeypatch.setattr(open_spreadsheet, "recalculate", hangs)
    monkeypatch.setitem(recalc._reference, recalc.soffice(), (True, time.monotonic()))
    with pytest.raises(tasks.RecalcFailed):
        task.grade(item, pred)



def _fake_office(tmp_path, monkeypatch, *, bundled: bool):
    """A LibreOffice stand-in whose ``python`` records how it was started (argv, PYTHON*
    variables) and copies the helper's input to its output."""
    from refactory_packages.spreadsheetbench import recalc
    program = tmp_path / "office" / "program"
    program.mkdir(parents=True)
    soffice = program / "soffice"
    soffice.write_text("#!/bin/sh\nexit 0\n")
    soffice.chmod(0o755)
    if bundled:
        log = tmp_path / "started.txt"
        launcher = program / "python"
        launcher.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$@\" > {log}\n"
            f"env | grep '^PYTHON' | sort >> {log}\n"
            "for a; do src=$dst; dst=$a; done\n"
            "cp \"$src\" \"$dst\" && echo '{\"ok\": true}'\n")
        launcher.chmod(0o755)
        (program / "python.bin").write_text("")
    monkeypatch.setattr(recalc, "soffice", lambda: str(soffice))
    return program


def test_the_bundled_libreoffice_python_is_not_run_isolated(tmp_path, monkeypatch):
    """[LibreOffice helper] The bundled program/python finds its stdlib through the
    PYTHONHOME it sets, which -I would make it ignore (the lab's LibreOffice 26.8 failed every
    file so): `recalculate` starts it with -s, PYTHONSAFEPATH=1 and none of the caller's
    PYTHON* variables."""
    from refactory_packages.spreadsheetbench_v2 import open_spreadsheet as lo
    program = _fake_office(tmp_path, monkeypatch, bundled=True)
    monkeypatch.setenv("PYTHONPATH", "/some/venv/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/some/venv")
    monkeypatch.setenv("PYTHONSTARTUP", "/some/startup.py")
    assert lo.lo_python() == str(program / "python") and lo.bundled(lo.lo_python())
    src = tmp_path / "in.xlsx"
    src.write_bytes(b"workbook")
    assert lo.recalculate(str(src), str(tmp_path / "out.xlsx")) == (True, "")
    started = (tmp_path / "started.txt").read_text().splitlines()
    argv = [line for line in started if not line.startswith("PYTHON")]
    python_vars = [line for line in started if line.startswith("PYTHON")]
    assert argv[0] == "-s" and "-I" not in argv and "-E" not in argv, argv
    assert python_vars == ["PYTHONSAFEPATH=1"], python_vars


def test_a_system_python_helper_stays_isolated(tmp_path, monkeypatch):
    from refactory_packages.spreadsheetbench_v2 import open_spreadsheet as lo
    _fake_office(tmp_path, monkeypatch, bundled=False)
    assert lo.lo_python() == "/usr/bin/python3"
    assert not lo.bundled(lo.lo_python()) and lo.python_flags(lo.lo_python()) == ["-I"]


def test_a_python_beside_a_soffice_wrapper_is_not_the_bundled_one(tmp_path, monkeypatch):
    """[LibreOffice helper] A ``python`` in the same directory as ``soffice`` but with no
    ``python.bin`` (a system python beside a wrapper script) keeps -I."""
    from refactory_packages.spreadsheetbench_v2 import open_spreadsheet as lo
    program = _fake_office(tmp_path, monkeypatch, bundled=True)
    (program / "python.bin").unlink()
    assert not lo.bundled(lo.lo_python()) and lo.python_flags(lo.lo_python()) == ["-I"]
