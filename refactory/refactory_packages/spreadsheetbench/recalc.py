"""Formula-aware grading: recalculate with LibreOffice before comparing.

Ported from the lab's ``experiments/spreadsheetbench-20261115/scripts/lo_grader.py``,
the grader SkillOpt's recalc runs used (``SB_GRADER=recalc``). SkillOpt's own grader
reads answer cells with ``openpyxl data_only=True``, i.e. the values Excel cached at
save time, and a workbook openpyxl wrote has none, so a correct formula grades as an
empty cell. Here, when the prediction's answer cells hold formulas, a LibreOffice-
recalculated copy is graded too and the item passes if either matches. The prediction
itself is never changed.

LibreOffice is found at ``SB_SOFFICE``, else ``soffice``/``libreoffice`` on the PATH,
else the macOS application bundle. The lab measured LibreOffice reproducing the stored
answers of 166 of the 177 golden files whose answer cells are formulas (val 17/17).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any

import openpyxl
from openpyxl.worksheet.formula import ArrayFormula

from refactory_dsh.model import HARNESS_INCIDENT_TEXT, HarnessIncident

from . import skillopt_port as port

#: The grader choice, shared with the lab's SkillOpt wrapper: ``recalc`` (default) or
#: ``cached`` (SkillOpt as shipped).
GRADER_ENV = "SB_GRADER"
GRADERS = ("recalc", "cached")
SOFFICE_ENV = "SB_SOFFICE"
_MAC_SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

_ROOT = os.path.join(tempfile.gettempdir(), f"sb-lo-{os.getuid()}")
_local = threading.local()
_cache: dict[tuple, str | None] = {}
_lock = threading.Lock()
# lo_grader.py:31-35: recalculate on load, always.
_XCU = ('<?xml version="1.0" encoding="UTF-8"?>\n<oor:items xmlns:oor="http://openoffice.org/2001/registry" '
        'xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        + "".join(f'<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="{k}" '
                  f'oor:op="fuse"><value>0</value></prop></item>\n' for k in ("OOXMLRecalcMode", "ODFRecalcMode"))
        + "</oor:items>\n")


class RecalcFailed(HarnessIncident):
    """LibreOffice produced no recalculated copy (it failed, or ran past its 180 s wall):
    the grader's failure, not the model's, so the item is a harness incident rather than
    a wrong answer."""


def grader(choice: str | None = None) -> str:
    """The grader in force: ``choice``, else ``SB_GRADER``, else ``recalc``."""
    value = (choice or os.environ.get(GRADER_ENV) or "recalc").strip()
    if value not in GRADERS:
        raise ValueError(f"{GRADER_ENV} must be one of {GRADERS}, not {value!r}")
    return value


def soffice() -> str | None:
    explicit = os.environ.get(SOFFICE_ENV, "").strip()
    if explicit:
        return explicit if os.access(explicit, os.X_OK) else None
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    return _MAC_SOFFICE if os.access(_MAC_SOFFICE, os.X_OK) else None


def available() -> bool:
    return soffice() is not None


def _profile() -> str:
    p = getattr(_local, "profile", None)
    if p is None:
        p = os.path.join(_ROOT, f"profile-{os.getpid()}-{threading.get_ident()}")
        os.makedirs(os.path.join(p, "user"), exist_ok=True)
        with open(os.path.join(p, "user", "registrymodifications.xcu"), "w") as f:
            f.write(_XCU)
        _local.profile = p
    return p


# lo_grader.py:55-65
def _cells(answer_position: str):
    for part in (answer_position or "").split(","):
        part = part.strip()
        if not part:
            continue
        sheet, rng = part.split("!", 1) if "!" in part else (None, part)
        sheet = sheet.strip().strip("'\"") if sheet else None
        yield sheet, list(port.generate_cell_names(rng.strip().strip("'\"")))


def is_formula(value: Any) -> bool:
    """Whether a cell value (as openpyxl loads it) is a formula: a string starting with
    ``=``, or an array formula, which openpyxl loads as an `ArrayFormula` object.
    lo_grader.py counts only the string; an array formula, which Excel computes, would
    then be graded on its cached value (none, in a workbook openpyxl wrote) and always
    fail, so it counts here too (docs/spreadsheetbench.md, "Differences")."""
    return (isinstance(value, str) and value.startswith("=")) or isinstance(value, ArrayFormula)


# lo_grader.py:64-82, with array formulas counted (`is_formula`). The sheet falls back
# to the *prediction's* first sheet when the position names none or one it lacks (the
# evaluator uses gold's); copied as it is.
def has_formula_answer(pred_path: str, answer_position: str) -> bool:
    try:
        wb = openpyxl.load_workbook(pred_path)
    except Exception:
        return False
    try:
        try:
            for sheet, names in _cells(answer_position):
                ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb.worksheets[0]
                if any(is_formula(ws[n].value) for n in names):
                    return True
            return False
        except Exception:
            return any(is_formula(c.value)
                       for ws in wb.worksheets for row in ws.iter_rows() for c in row)
    finally:
        wb.close()


# lo_grader.py:85-110
def recalculated(pred_path: str) -> str | None:
    """A LibreOffice-recalculated copy of ``pred_path`` (cached per file), or ``None``."""
    binary = soffice()
    if binary is None or not os.path.exists(pred_path):
        return None
    st = os.stat(pred_path)
    key = (os.path.abspath(pred_path), st.st_mtime_ns, st.st_size)
    with _lock:
        if key in _cache:
            return _cache[key]
    tag = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
    work = os.path.join(_ROOT, "out", tag)
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "in.xlsx")
    shutil.copyfile(pred_path, src)
    out = os.path.join(work, "out")
    try:
        subprocess.run([binary, f"-env:UserInstallation=file://{_profile()}", "--headless", "--calc",
                        "--convert-to", "xlsx:Calc MS Excel 2007 XML", "--outdir", out, src],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180, check=False)
    except subprocess.TimeoutExpired:
        pass
    res = os.path.join(out, "in.xlsx")
    res = res if os.path.exists(res) else None
    with _lock:
        _cache[key] = res
    return res


#: How long a check that LibreOffice works (`soffice_works`) is trusted, in seconds.
REFERENCE_TTL_S = 300.0
_reference: dict[str, tuple[bool, float]] = {}


def _convert(binary: str, src: str, out: str, timeout_s: float) -> str | None:
    try:
        subprocess.run([binary, f"-env:UserInstallation=file://{_profile()}", "--headless", "--calc",
                        "--convert-to", "xlsx:Calc MS Excel 2007 XML", "--outdir", out, src],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_s,
                       check=False)
    except subprocess.TimeoutExpired:
        pass
    res = os.path.join(out, os.path.basename(src))
    return res if os.path.exists(res) else None


def soffice_works() -> bool:
    """Whether this LibreOffice recalculates a known-good tiny workbook (``=1+1``): when
    it cannot, the grader is broken (`RecalcFailed`); when it can, a prediction it could
    not recalculate is that prediction's own fault. Checked again after
    `REFERENCE_TTL_S`, per binary."""
    binary = soffice()
    if binary is None:
        return False
    with _lock:
        held = _reference.get(binary)
        if held is not None and time.monotonic() - held[1] < REFERENCE_TTL_S:
            return held[0]
    work = tempfile.mkdtemp(prefix="reference-", dir=_ensure_root())
    src = os.path.join(work, "reference.xlsx")
    wb = openpyxl.Workbook()
    wb.active["A1"] = "=1+1"
    wb.save(src)
    ok = _convert(binary, src, os.path.join(work, "out"), 60) is not None
    shutil.rmtree(work, ignore_errors=True)
    with _lock:
        _reference[binary] = (ok, time.monotonic())
    return ok


def _ensure_root() -> str:
    os.makedirs(_ROOT, exist_ok=True)
    return _ROOT


# lo_grader.py:118-128
def evaluate(pred_path: str, gold_path: str, instruction_type: str, answer_position: str,
             *, choice: str | None = None) -> dict[str, Any]:
    """The grade: cached first; under ``recalc``, a failing prediction whose answer
    cells hold formulas is graded again on its recalculated copy. When that copy cannot
    be made, LibreOffice is tried on a known-good workbook (`soffice_works`): if it works
    there, the prediction is at fault and keeps its cached (failing) grade, with
    ``recalc_failed``; if not, the grader is broken: `RecalcFailed`, a harness incident."""
    r = port.evaluate_cached(pred_path, gold_path, instruction_type, answer_position)
    if grader(choice) == "cached":
        return r
    if r.get("ok") or not has_formula_answer(pred_path, answer_position):
        return r
    rc = recalculated(pred_path)
    if not rc:
        if soffice_works():
            # LibreOffice recalculates a known-good workbook, so it is this one it could
            # not: the model's file, graded as its cached values say (failing).
            return {**r, "recalc_failed": True,
                    "reason": f"LibreOffice could not recalculate this workbook; graded on its "
                              f"cached values: {str(r.get('reason', ''))[:200]}"}
        raise RecalcFailed(f"{HARNESS_INCIDENT_TEXT}LibreOffice could not recalculate {pred_path} "
                           f"(cached grade: {str(r.get('reason', ''))[:200]})")
    r2 = port.evaluate_cached(rc, gold_path, instruction_type, answer_position)
    r2["recalculated"] = True
    return r2


# lo_grader.py:130-135
def auto_verify_output(pred_path: str, gold_path: str, answer_position: str,
                       *, choice: str | None = None) -> str:
    """The verification report, read from the recalculated copy under ``recalc`` when
    the answer cells hold formulas; under ``recalc`` its line over the ``=`` cells says
    they hold text (`mode` deviation 7, docs item 8), under ``cached`` it is SkillOpt's."""
    recalculates = grader(choice) == "recalc"
    if recalculates and os.path.exists(pred_path) \
            and has_formula_answer(pred_path, answer_position):
        rc = recalculated(pred_path)
        if rc:
            return port.auto_verify_output(rc, gold_path, answer_position, recalculates=True)
    return port.auto_verify_output(pred_path, gold_path, answer_position,
                                   recalculates=recalculates)
