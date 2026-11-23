"""SpreadsheetBench 2 as a Task: the prepared release, V2's grade.

Items are the 277 file-graded tasks of the release (``manifest.json``: Debugging 80,
Financial_Model 100, Template 97; the 24 Visualization and 20 Debugging Embedded-Hardcode
and Color tasks are dropped, each with its reason), in the grouped split's order: train,
val, test. The data is not shipped: ``SB2_DATA`` names a directory written by
``python -m refactory_packages.spreadsheetbench_v2.prepare prepare SOURCE OUT``, whose
``prepared.json`` must match the shipped manifest and whose every file must match its
recorded checksum; anything else is refused.

The input's prepared path and the golden's are grader-only (``grader_fields``): the nodes
see the input as a copy in their pass, the item file ``item/input.xlsx``, and the agent a
copy of that in its sealed workspace, as ``input.xlsx``. The prepared directory itself --
every golden, every other task's input, ``tasks.json`` -- is hidden from model code and
from sealed tool steps (`tool_hidden`).

The grade is V2's pipeline: the output is recalculated by LibreOffice as V2's
``open_spreadsheet.py`` recalculates it (`open_spreadsheet`), then compared with the
prepared golden by V2's ``evaluation.py`` (`evaluation_port`).
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Any

from refactory_dsh.model import HARNESS_INCIDENT_TEXT
from refactory_dsh.task import Task, TaskError, TaskInstance, ToolEnvironment, VerifyResult

from refactory_packages.spreadsheetbench import interpreter, recalc
from refactory_packages.spreadsheetbench import sandbox  # noqa: F401 - SandboxFailed known
from refactory_packages.spreadsheetbench import skillopt_port as port
from refactory_packages.spreadsheetbench.tasks import FINAL, INPUT, PRED

from . import evaluation_port, open_spreadsheet, prepare

DATA_ENV = "SB2_DATA"
#: The input's name in the agent's workspace and in the code sandbox.
INPUT_NAME = "input.xlsx"


def data_roots() -> list[str]:
    """The prepared directory, hidden from model code and sealed tool steps."""
    root = os.environ.get(DATA_ENV, "").strip()
    return [str(Path(root).expanduser())] if root else []


def tool_hidden(workdir: Path) -> list[str]:
    """What no folder of a sealed tool step's environment may hold or lie inside: the run,
    the outer loop's root and the prepared data (`spreadsheetbench.sandbox.hidden_paths`)."""
    from refactory_packages.spreadsheetbench import ops

    return sandbox.hidden_paths([*ops._run_roots(Path(workdir)), *data_roots()])  # noqa: SLF001


class RecalcFailed(recalc.RecalcFailed):
    """LibreOffice failed on a known-good workbook while grading: the grader's failure."""


class SpreadsheetBenchV2Task(Task):
    """SpreadsheetBench 2's 277 file-graded tasks, graded by V2's pipeline."""

    name = "spreadsheetbench_v2"
    graded_files = (PRED, FINAL)
    item_files = MappingProxyType({"input": INPUT})
    tool_environment = ToolEnvironment(summary=interpreter.SUMMARY, python=interpreter.interpreter,
                                       hidden=tool_hidden)
    grader_fields = ("gold_path", "input_path")
    item_metadata = ("category",)
    #: The analysts of both arms see the input's preview as the item's context, as the
    #: agent harness's do (`spreadsheetbench.tasks.SpreadsheetBenchAgentTask`).
    item_context = ("spreadsheet_preview",)

    def __init__(self, data_root: str | None = None) -> None:
        self.data_root = Path(data_root or os.environ.get(DATA_ENV, "")).expanduser()
        self._items: list[TaskInstance] | None = None
        self.excluded: dict[str, str] = {}

    def _load(self) -> list[TaskInstance]:
        if self._items is not None:
            return self._items
        if not str(self.data_root) or not (self.data_root / prepare.PREPARED_FILE).is_file():
            raise TaskError(f"SpreadsheetBench 2 data not prepared: set {DATA_ENV} to a directory "
                            "written by `python -m refactory_packages.spreadsheetbench_v2.prepare "
                            "prepare SOURCE OUT`")
        manifest = prepare.load_manifest()
        try:
            record = prepare.check(self.data_root)
            #: Tasks left out, with why (a golden that fails against itself).
            self.excluded = prepare.excluded_ids(record, manifest)
        except prepare.PrepareError as exc:
            raise TaskError(f"{DATA_ENV}: {exc}") from exc
        if not open_spreadsheet.available():
            raise TaskError(f"grading recalculates the output with LibreOffice: install it or set "
                            f"{recalc.SOFFICE_ENV}")
        rows = {row["id"]: row for row in
                json.loads((self.data_root / prepare.TASKS_FILE).read_text(encoding="utf-8"))}
        items = []
        for part in ("train", "val", "test"):
            for item_id in manifest["parts"][part]:
                if item_id in self.excluded:
                    continue
                if item_id not in rows:
                    raise TaskError(f"{part} id {item_id!r} is not in {prepare.TASKS_FILE}")
                items.append(self._record(rows[item_id]))
        self._items = items
        return items

    def _record(self, row: dict[str, Any]) -> TaskInstance:
        return TaskInstance(id=row["id"], metadata={
            "instruction": row["instruction"],
            "category": row["category"],
            "answer_position": row["answer_position"],
            "input_name": INPUT_NAME,
            # Where the preview is read from (`item_context_values`); the prepared input's
            # own path in the dataset, as the release's spreadsheet_path is in SB.
            "input_rel": row["input"],
            # Grader-only (`grader_fields`).
            "input_path": str(self.data_root / row["input"]),
            "gold_path": str(self.data_root / row["golden"]),
        })

    def instances(self, *, part: str | None = None, seed: int | None = None) -> list[TaskInstance]:
        """Every item, train then val then test; the parts manifest picks the part."""
        return list(self._load())

    def setup(self, instance: TaskInstance, workspace: Path, *, part: str | None = None,
              seed: int | None = None) -> None:
        """Copy the item's prepared input into the pass as its item file."""
        target = Path(workspace) / INPUT
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(instance.metadata["input_path"], target)

    def item_context_values(self, metadata: Any) -> dict[str, Any]:
        rel = str(metadata.get("input_rel") or "")
        if not rel:
            return {}
        return {"spreadsheet_preview": _preview(str(self.data_root / rel))}

    def grade(self, instance: TaskInstance, pred: Path) -> tuple[evaluation_port.Grade, str]:
        """V2's grade of ``pred`` (evaluation/README: recalculate the outputs with
        ``open_spreadsheet.py``, then ``evaluation.py``) and a note on how it was graded.

        When LibreOffice, running, cannot open the output but does recalculate a known-good
        workbook (`recalc.soffice_works`), the output is graded as it is, as V2's script
        leaves a file it could not open in place; when it cannot recalculate that either,
        or LibreOffice itself failed (timed out, killed, crashed: `RecalcError`), the
        grader is broken: `RecalcFailed`, a harness incident."""
        meta = instance.metadata
        note = ""
        with tempfile.TemporaryDirectory(prefix="sb2-grade-") as work:
            graded = str(pred)
            if pred.exists():
                recalculated = os.path.join(work, "output.xlsx")
                try:
                    ok, why = open_spreadsheet.recalculate(str(pred), recalculated)
                except open_spreadsheet.RecalcError as exc:
                    # Timed out, killed, crashed: LibreOffice's failure, never a silent 0.
                    raise RecalcFailed(f"{HARNESS_INCIDENT_TEXT}recalculating the output: "
                                       f"{exc}") from exc
                if ok:
                    graded = recalculated
                elif recalc.soffice_works():
                    note = f"LibreOffice could not recalculate the output ({why}); graded as written"
                else:
                    raise RecalcFailed(f"{HARNESS_INCIDENT_TEXT}LibreOffice could not recalculate "
                                       f"{pred} nor a known-good workbook: {why}")
            g = evaluation_port.grade(meta["input_path"], meta["gold_path"], graded,
                                      meta["answer_position"])
        return g, note

    def verify(self, instance: TaskInstance, workspace: Path, *, part: str | None = None,
               seed: int | None = None) -> VerifyResult:
        """The item's fail reason, or V2's grade. ``prediction`` is what the analysts see:
        the execution error, or V2's numbers and mismatches."""
        final_path = Path(workspace) / FINAL
        final = json.loads(final_path.read_text()) if final_path.exists() else {}
        if final.get("backstop"):
            from refactory_dsh.model import HARNESS_BACKSTOP_TEXT, HarnessBackstop

            raise HarnessBackstop(f"the item's turns {HARNESS_BACKSTOP_TEXT} "
                                  f"({final.get('elapsed_s')} s of {final.get('limit_s')} s)")
        reason = str(final.get("fail_reason") or "")
        if reason:
            error = final.get("exec_error")
            return VerifyResult(passed=False, score=0.0, details=reason, prediction=(
                f"## Execution (case 1)\nERROR: {error}" if error is not None else None))
        try:
            g, note = self.grade(instance, Path(workspace) / PRED)
        except evaluation_port.OutputUnreadable as exc:
            text = f"output-unreadable: {exc}"
            return VerifyResult(passed=False, score=0.0, details=text[:300],
                                prediction=f"## Eval Result: FAIL\n{text[:500]}")
        report = evaluation_port.feedback(g)
        if note:
            report = f"{report}\n({note})"
        return VerifyResult(passed=g.passed, score=g.accuracy,
                            details="" if g.passed else f"eval-mismatch: {g.error_message[:200]}",
                            prediction=report)


@functools.lru_cache(maxsize=512)
def _preview(path: str) -> str:
    """The input's preview (SkillOpt's ``_preview_workbook``), ``(preview failed)`` when it
    cannot be read."""
    try:
        return port.preview_workbook(path)
    except Exception:  # noqa: BLE001 - as SkillOpt's
        return "(preview failed)"
