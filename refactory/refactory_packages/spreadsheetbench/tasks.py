"""SpreadsheetBench Verified 400 as a Task: SkillOpt's items, SkillOpt's grade.

Items are the rows of the release's ``dataset.json`` (Hugging Face
``KAKA22/SpreadsheetBench``, revision and checksums in ``manifest.json``), in the order
of SkillOpt's id split (80 train, 40 val, 280 test), as the lab's ``materialize.py``
builds them. The data is not shipped: ``SB_DATA`` names the extracted
``spreadsheetbench_verified_400`` directory (``dataset.json`` and ``spreadsheet/``), and
its ``dataset.json`` must match the manifest's checksum.

Two items are known by name, as they are in SkillOpt's runs:

- **test 42930 can never pass.** The release pairs ``1_42930_init.xlsx`` with
  ``1_43930_golden.xlsx`` (an upstream typo), so no test case is found and the item
  scores 0 with ``no-test-cases``, without a model call. The most a test pass can score
  is 279/280.
- **test 56225 passes only with a written value.** The release marks it
  ``exclude: "ignore, initial file already passes verification"``, but the input does
  not pass (its O3 is 0, the gold's 1.8019), and LibreOffice cannot reproduce the gold's
  SUMIFS, so a formula answer fails under ``recalc`` too. SkillOpt reads no ``exclude``
  field, so neither does this Task.

The golden file and the input's path in the dataset are grader-only
(``grader_fields``): the nodes see the input as a copy in their pass, its item file
``item/input.xlsx`` (`Task.item_files`), and model code sees a copy of that in its
sandbox (`sandbox`), under the input's own name.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import MappingProxyType
from typing import Any

from refactory_dsh.task import Task, TaskError, TaskInstance, ToolEnvironment, VerifyResult

from . import interpreter, recalc
# SandboxFailed, like RecalcFailed, must be known as a harness incident wherever a record
# is classified (`model.HARNESS_INCIDENT_CLASSES`), the spine's item command included.
from . import sandbox  # noqa: F401
from . import skillopt_port as port

MANIFEST = Path(__file__).resolve().parent / "manifest.json"
DATA_ENV = "SB_DATA"
#: Another id split and dataset checksum in the manifest's format (a smoke split, a
#: test fixture); the shipped ``manifest.json`` by default.
MANIFEST_ENV = "SB_MANIFEST"
#: Where `setup` lays the item's input in its pass: its item file (`Task.item_files`),
#: under one name for every item, so any step may declare it in its reads.
INPUT = "item/input.xlsx"
#: The pass's copy of the latest prediction (SkillOpt's ``<case>_pred.xlsx``).
PRED = "pred.xlsx"
#: What the pass's last step (`ops.finalize`) concluded, for `verify`.
FINAL = "final.json"
#: The turn loop's state (`ops`).
TURN = "turn.json"


def load_manifest() -> dict[str, Any]:
    override = os.environ.get(MANIFEST_ENV, "").strip()
    return json.loads(Path(override).expanduser().read_text() if override else MANIFEST.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def item_record(row: dict[str, Any], data_root: Path) -> TaskInstance:
    """One dataset row as an instance: SkillOpt's fields, plus the case it is graded on."""
    item_id = str(row["id"])
    sp = row.get("spreadsheet_path", f"spreadsheet/{item_id}")
    task_dir = sp if os.path.isabs(sp) else os.path.join(str(data_root), sp)
    cases = port.find_test_cases(task_dir)
    case_no, input_path, gold_path = cases[0] if cases else ("", "", "")
    metadata: dict[str, Any] = {
        "instruction": row.get("instruction", ""),
        "instruction_type": row.get("instruction_type", ""),
        "answer_position": row.get("answer_position", ""),
        **({"answer_sheet": row["answer_sheet"]} if "answer_sheet" in row else {}),
        **({"data_position": row["data_position"]} if "data_position" in row else {}),
        "spreadsheet_path": sp,
        "answer_position_eval": port.answer_position_eval(row),
        "task_type": port.task_type(row.get("instruction_type", "")),
        "case_no": case_no,
        "input_name": os.path.basename(input_path),
        # Grader-only (`grader_fields`).
        "input_path": input_path,
        "gold_path": gold_path,
        **({"exclude": row["exclude"]} if "exclude" in row else {}),
    }
    return TaskInstance(id=item_id, metadata=metadata)


def _tool_hidden(workdir: Path) -> list[str]:
    """What model code may not see (`ops.tool_hidden`): no folder of a sealed tool step's
    environment may hold or lie inside it."""
    from . import ops

    return ops.tool_hidden(workdir)


class SpreadsheetBenchTask(Task):
    """SpreadsheetBench Verified 400, graded as SkillOpt's runs graded it.

    ``grader``: ``recalc`` (default; the lab's ``SB_GRADER``) or ``cached``
    (`recalc.grader`). `verify` reads what `ops.finalize` concluded.
    """

    name = "spreadsheetbench"
    #: ``verify`` scores the pass's final record and its latest prediction.
    graded_files = (PRED, FINAL)
    #: The item's input workbook, the one item file (all the release's inputs are .xlsx).
    item_files = MappingProxyType({"input": INPUT})
    #: A sealed tool step's commands run the interpreter model code runs under, checked
    #: the same way (`interpreter.interpreter`), as ``python3`` and ``python``.
    tool_environment = ToolEnvironment(summary=interpreter.SUMMARY,
                                       python=interpreter.interpreter,
                                       hidden=_tool_hidden)
    grader_fields = ("gold_path", "input_path", "exclude")
    item_metadata = ("task_type",)

    def __init__(self, data_root: str | None = None, grader: str | None = None) -> None:
        self.data_root = Path(data_root or os.environ.get(DATA_ENV, "")).expanduser()
        self.grader = recalc.grader(grader)
        self._items: list[TaskInstance] | None = None

    def _load(self) -> list[TaskInstance]:
        if self._items is not None:
            return self._items
        manifest = load_manifest()
        dataset = self.data_root / "dataset.json"
        if not str(self.data_root) or not dataset.is_file():
            raise TaskError(f"SpreadsheetBench data not found: set {DATA_ENV} to the extracted "
                            f"spreadsheetbench_verified_400 directory ({manifest['source']} at "
                            f"{manifest['revision']}, {manifest['archive']})")
        if _sha256(dataset) != manifest["dataset_json_sha256"]:
            raise TaskError(f"{dataset} is not the release's dataset.json (sha256 differs "
                            f"from {manifest['dataset_json_sha256']})")
        if self.grader == "recalc" and not recalc.available():
            raise TaskError(f"{recalc.GRADER_ENV}=recalc needs LibreOffice: install it, set "
                            f"{recalc.SOFFICE_ENV}, or set {recalc.GRADER_ENV}=cached")
        rows = {str(row["id"]): row for row in json.loads(dataset.read_text())}
        items = []
        for part in ("train", "val", "test"):
            for item_id in manifest["parts"][part]:
                if item_id not in rows:
                    raise TaskError(f"{part} id {item_id!r} is not in {dataset}")
                items.append(item_record(rows[item_id], self.data_root))
        self._items = items
        return items

    def instances(self, *, part: str | None = None, seed: int | None = None) -> list[TaskInstance]:
        """Every item, train then val then test; the parts manifest picks the part."""
        return list(self._load())

    def setup(self, instance: TaskInstance, workspace: Path, *, part: str | None = None,
              seed: int | None = None) -> None:
        """Copy the item's input into the pass as its item file (nothing for an item with
        no case)."""
        source = instance.metadata.get("input_path") or ""
        if source:
            target = Path(workspace) / INPUT
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def _grade(self, instance: TaskInstance, pred: Path) -> dict[str, Any]:
        meta = instance.metadata
        return recalc.evaluate(str(pred), meta["gold_path"], meta.get("instruction_type", ""),
                               meta.get("answer_position_eval", ""), choice=self.grader)

    def train_feedback(self, instance: TaskInstance, workspace: Path, *, part: str | None = None,
                       seed: int | None = None) -> dict[str, Any] | None:
        """codegen_agent.py:735-750: after a clean run on a train item, grade it; a pass
        ends the item, a miss is told which cells are wrong (values hidden). ``None``
        when the last turn did not run cleanly or the item has no answer position."""
        meta = instance.metadata
        turn = json.loads((Path(workspace) / TURN).read_text())
        if turn.get("status") != "ran" or not meta.get("gold_path") \
                or not meta.get("answer_position_eval"):
            return None
        pred = Path(workspace) / PRED
        try:
            graded = self._grade(instance, pred)
        except recalc.RecalcFailed:
            # The grader failed, not the attempt: nothing to tell it. The item's grade
            # meets the same failure and records a harness incident (`verify`).
            return None
        if graded["ok"]:
            return {"passed": True}
        report = recalc.auto_verify_output(str(pred), meta["gold_path"],
                                           meta["answer_position_eval"], choice=self.grader)
        return {"passed": False, "feedback": port.build_eval_feedback(report)}

    def verify(self, instance: TaskInstance, workspace: Path, *, part: str | None = None,
               seed: int | None = None) -> VerifyResult:
        """rollout.py:779-863: the item's fail reason, or its grade.

        ``prediction`` is SkillOpt's post-execution verification block (the analysts'
        ``[POST-EXECUTION VERIFICATION]``), ``details`` the fail reason."""
        meta = instance.metadata
        final_path = Path(workspace) / FINAL
        final = json.loads(final_path.read_text()) if final_path.exists() else {}
        if final.get("backstop"):
            # The model kept going for the whole limit: its own failure (``backstop``).
            from refactory_dsh.model import HARNESS_BACKSTOP_TEXT, HarnessBackstop

            raise HarnessBackstop(f"the item's turns {HARNESS_BACKSTOP_TEXT} "
                                  f"({final.get('elapsed_s')} s of {final.get('limit_s')} s)")
        reason = str(final.get("fail_reason") or "")
        if not meta.get("gold_path"):
            reason = "no-test-cases"
        if reason:
            # rollout.py:799-812: code that failed to run is shown to the analysts as its
            # first 500 characters of error under ``## Execution (case N)``.
            error = final.get("exec_error")
            return VerifyResult(passed=False, score=0.0, details=reason, prediction=(
                f"## Execution (case {meta.get('case_no') or 1})\nERROR: {error}"
                if error is not None and meta.get("gold_path") else None))
        pred = Path(workspace) / PRED
        ape = meta.get("answer_position_eval", "")
        try:
            ev = self._grade(instance, pred)
        except recalc.RecalcFailed:
            raise  # recorded as a harness incident (`model.HarnessIncident`)
        except Exception as e:  # noqa: BLE001 - rollout.py:820-823
            ev = {"ok": False, "reason": f"eval-exception: {type(e).__name__}: {e}"}
        verification = ""
        if ape:
            report = recalc.auto_verify_output(str(pred), meta["gold_path"], ape,
                                               choice=self.grader)
            verification = (f"## Eval Result (case {meta.get('case_no') or 1}): "
                            f"{'PASS' if ev['ok'] else 'FAIL'}\n"
                            f"{ev.get('reason', '')}\n\n{report}")
        return VerifyResult(
            passed=bool(ev["ok"]), score=1.0 if ev["ok"] else 0.0,
            details="" if ev["ok"] else f"eval-mismatch: {str(ev.get('reason', ''))[:200]}",
            prediction=verification,
        )


class SpreadsheetBenchAgentTask(SpreadsheetBenchTask):
    """The same items and grade, for the agent harness (`agent_mode.Agent`), whose model
    call carries the item only as the files of its workspace: the optimizer is shown the
    input's preview as the item's context (`Task.item_context`), as SkillOpt's analysts
    see it for an exec rollout (rollout.py:697-721: its ``spreadsheet_preview.txt``, also
    inside its target user prompt). The solver's call already carries it in its prompt."""

    item_context = ("spreadsheet_preview",)

    def item_context_values(self, metadata: Any) -> dict[str, Any]:
        sp = str(metadata.get("spreadsheet_path") or "")
        name = str(metadata.get("input_name") or "")
        if not name:
            return {}
        folder = sp if os.path.isabs(sp) else os.path.join(str(self.data_root), sp)
        return {"spreadsheet_preview": _preview(os.path.join(folder, name))}


@functools.lru_cache(maxsize=1024)
def _preview(path: str) -> str:
    """rollout.py:701-704: the input's preview, ``(preview failed)`` when it cannot be read."""
    try:
        return port.preview_workbook(path)
    except Exception:  # noqa: BLE001 - as SkillOpt's
        return "(preview failed)"
