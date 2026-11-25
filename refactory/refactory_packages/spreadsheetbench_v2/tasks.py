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
import threading
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


#: What the swe mode's sessions find beside ``python3`` (`swe_port`, deviation 8).
SWE_SUMMARY = (f"{interpreter.SUMMARY}; view_xlsx (SpreadsheetBench 2's workbook viewer); "
               "soffice and libreoffice (LibreOffice, headless); a writable /tmp of the step's own")
#: Folders of the machine's LibreOffice may need besides its install, read only where they
#: exist: the font configuration, and a distribution's LibreOffice settings.
SWE_ETC = ("/etc/fonts", "/etc/libreoffice")


def _office_program() -> str:
    """The real ``soffice`` in LibreOffice's program folder (the grader's, `recalc.soffice`)."""
    folder = open_spreadsheet.program_dir()
    if folder is None or not os.access(os.path.join(folder, "soffice"), os.X_OK):
        raise TaskError(f"no LibreOffice for the sessions: install it or set {recalc.SOFFICE_ENV}")
    return os.path.join(folder, "soffice")


def swe_commands() -> dict[str, str]:
    """The swe sessions' commands: upstream's ``view_xlsx``, and LibreOffice as ``soffice``
    and ``libreoffice``. LibreOffice starts a thread pool and a malloc arena per core;
    under a sealed command's per-process limits (128 processes, 8 GiB of address space,
    `seal.LINUX_LIMITS_UNSCOPED`) that fails its xlsx export on a many-core machine, so
    its pool and arenas are capped (neither changes what it computes)."""
    from . import swe_port

    office = _office_program()
    launcher = ("#!/bin/sh\n"
                "MAX_CONCURRENCY=${MAX_CONCURRENCY:-4}; MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}\n"
                "export MAX_CONCURRENCY MALLOC_ARENA_MAX\n"
                f'exec "{office}" "$@"\n')
    return {"view_xlsx": swe_port.VIEW_XLSX, "soffice": launcher, "libreoffice": launcher}


def swe_reads() -> list[str]:
    """LibreOffice's install (the folder above its program folder), and `SWE_ETC`."""
    office = os.path.realpath(_office_program())
    return [os.path.dirname(os.path.dirname(office)),
            *(path for path in SWE_ETC if os.path.isdir(path))]


class RecalcFailed(recalc.RecalcFailed):
    """LibreOffice failed on a known-good workbook while grading: the grader's failure."""


class SpreadsheetBenchV2Task(Task):
    """SpreadsheetBench 2's 277 file-graded tasks, graded by V2's pipeline."""

    name = "spreadsheetbench_v2"
    graded_files = (PRED, FINAL)
    item_files = MappingProxyType({"input": INPUT})
    tool_environment = ToolEnvironment(summary=interpreter.SUMMARY, python=interpreter.interpreter,
                                       hidden=tool_hidden,
                                       packages=tuple((name, interpreter.PINS[name])
                                                      for name in interpreter.NAMED))
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
        the grader is broken: `RecalcFailed`, a harness incident.

        When LibreOffice itself fails on the output (timed out, killed, crashed:
        `RecalcError`), a fresh reference check runs and, if LibreOffice is healthy, the
        recalculation is tried once more. `_RETRY_LOCK` only keeps two retries in one
        process from running together: other items' first attempts (and other processes)
        still run beside a retry, so a retry can still run out of memory next to other
        work. Failing again while LibreOffice stays healthy is the output's doing
        (`OutputUnrecalculable`, scored 0): an output that hangs LibreOffice every time must
        not become an incident that is left out of the score. Only an unhealthy LibreOffice
        is an incident."""
        meta = instance.metadata
        note = ""
        with tempfile.TemporaryDirectory(prefix="sb2-grade-") as work:
            graded = str(pred)
            if pred.exists():
                recalculated = os.path.join(work, "output.xlsx")
                try:
                    ok, why = open_spreadsheet.recalculate(str(pred), recalculated)
                except open_spreadsheet.RecalcError as first:
                    ok, why = _retry_recalculation(pred, recalculated, first)
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
        except OutputUnrecalculable as exc:
            text = f"recalc-failed: {exc}"
            return VerifyResult(passed=False, score=0.0, details=text[:300],
                                prediction=f"## Eval Result: FAIL\n{text[:500]}")
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


class SpreadsheetBenchV2SweTask(SpreadsheetBenchV2Task):
    """The same items and grade, for the swe mode: its sessions' commands also find
    ``view_xlsx`` and LibreOffice, and a /tmp of their own (`swe_port`, deviation 8)."""

    tool_environment = ToolEnvironment(summary=SWE_SUMMARY, python=interpreter.interpreter,
                                       hidden=tool_hidden, commands=swe_commands,
                                       reads=swe_reads, private_tmp=True,
                                       packages=tuple((name, interpreter.PINS[name])
                                                      for name in interpreter.NAMED))

    def verify(self, instance: TaskInstance, workspace: Path, *, part: str | None = None,
               seed: int | None = None) -> VerifyResult:
        """V2's verdict; when the session stopped at its turn cap (`tasks.FINAL`'s
        ``turn_limit``, `swe_ops.finalize`), or the adapter left no facts of it
        (``session_facts: missing``), it says so after the verdict's text, where the
        analysts and reports read it."""
        import dataclasses

        result = super().verify(instance, workspace, part=part, seed=seed)
        final_path = Path(workspace) / FINAL
        final = json.loads(final_path.read_text()) if final_path.exists() else {}
        note = self.session_note(final)
        if not note:
            return result
        return dataclasses.replace(
            result, prediction=f"{result.prediction}\n{note}" if result.prediction else note,
            details=f"{result.details} {note}".strip())


    def session_note(self, final: dict[str, Any]) -> str:
        """What the verdict says of the session after its text: the turn cap reached, or no
        facts from the adapter; ``""`` for neither."""
        if final.get("turn_limit"):
            return turn_limit_note(final.get("n_turns"), final["turn_limit"])
        if final.get("session_facts") == "missing":
            return MISSING_SESSION_NOTE
        return ""


class SpreadsheetBenchV2StepsTask(SpreadsheetBenchV2SweTask):
    """The swe Task for the steps mode (`steps_mode`): the same items, grade and tool
    environment; the verdict names each step stopped at its turn cap (`tasks.FINAL`'s
    ``turn_limits``), each hand-off a step ran without (``missing_inputs``) and each step
    the adapter left no facts of (``session_facts_missing``, `steps_ops.finalize`)."""

    def session_note(self, final: dict[str, Any]) -> str:
        step_turns = final.get("step_turns") or {}
        notes = [f"(the {step} step stopped at its turn cap: "
                 + (f"{step_turns.get(step)} turns, " if step_turns.get(step) is not None else "")
                 + f"cap {cap})"
                 for step, cap in (final.get("turn_limits") or {}).items()]
        for step, names in (final.get("missing_inputs") or {}).items():
            notes.append(f"(the {step} step ran without {', '.join(names)}: no earlier step "
                         "wrote it)" if len(names) == 1 else
                         f"(the {step} step ran without {', '.join(names)}: no earlier step "
                         "wrote them)")
        missing = final.get("session_facts_missing") or []
        if missing:
            many = len(missing) > 1
            notes.append(f"(no session facts from Claude Code's adapter for the "
                         f"{', '.join(missing)} step{'s' if many else ''}: "
                         + ("their turn counts and whether they" if many else
                            "its turn count and whether it")
                         + " reached the turn cap are unknown)")
        return "\n".join(notes)


#: What a record says when Claude Code's adapter left no facts of the session.
MISSING_SESSION_NOTE = ("(no session facts from Claude Code's adapter: the turn count and "
                        "whether the turn cap was reached are unknown)")


def turn_limit_note(turns: Any, cap: Any) -> str:
    """What a record says of a session stopped at its turn cap."""
    used = f"{turns} turns, " if turns is not None else ""
    return f"(the session stopped at its turn cap: {used}cap {cap})"


class OutputUnrecalculable(Exception):
    """LibreOffice failed on the output twice while it recalculates a known-good workbook:
    the output's doing, scored 0 with the reason."""


#: One retry of a failed recalculation at a time in this process. It does not stop other
#: items' first attempts, or other processes, from running LibreOffice beside a retry.
_RETRY_LOCK = threading.Lock()


def _healthy() -> bool:
    """A fresh reference check (`recalc.soffice_works`, its cached answer dropped)."""
    with recalc._lock:  # noqa: SLF001
        recalc._reference.clear()  # noqa: SLF001
    return recalc.soffice_works()


def _retry_recalculation(pred: Path, dst: str, first: Exception) -> tuple[bool, str]:
    """After LibreOffice failed on ``pred``: an incident if LibreOffice is unhealthy; else
    one more try, alone; failing again while healthy, `OutputUnrecalculable`."""
    with _RETRY_LOCK:
        if not _healthy():
            raise RecalcFailed(f"{HARNESS_INCIDENT_TEXT}LibreOffice failed recalculating the "
                               f"output and on a known-good workbook: {first}") from first
        try:
            return open_spreadsheet.recalculate(str(pred), dst)
        except open_spreadsheet.RecalcError as second:
            if not _healthy():
                raise RecalcFailed(f"{HARNESS_INCIDENT_TEXT}LibreOffice failed recalculating "
                                   f"the output and on a known-good workbook: {second}") from second
            raise OutputUnrecalculable(
                f"LibreOffice could not recalculate the output (twice, while it recalculates a "
                f"known-good workbook): {second}") from second


@functools.lru_cache(maxsize=512)
def _preview(path: str) -> str:
    """The input's preview (SkillOpt's ``_preview_workbook``), ``(preview failed)`` when it
    cannot be read."""
    try:
        return port.preview_workbook(path)
    except Exception:  # noqa: BLE001 - as SkillOpt's
        return "(preview failed)"
