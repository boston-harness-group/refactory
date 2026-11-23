"""SpreadsheetBench 2's recalculation (``evaluation/open_spreadsheet.py``), one workbook at a time.

V2's README asks for every input and golden file to be recalculated with this script
before any run, and its evaluation recalculates every output with it before grading:
LibreOffice opens the file, iterative calculation is turned on (100 steps, 0.0001), every
formula is recalculated, and the file is stored as "Calc MS Excel 2007 XML", so the grader
(``data_only=True``) reads values LibreOffice computed. `recalculate` does that for one
file through ``lo_open_spreadsheet.py``, run by LibreOffice's Python. LibreOffice is found
as SpreadsheetBench's recalc grader finds it (`spreadsheetbench.recalc.soffice`:
``SB_SOFFICE``, the PATH, the macOS bundle), and its Python as V2 finds it (the bundled
``program/python``, else ``/usr/bin/python3``, open_spreadsheet.py:54-67).

What differs from V2's script, none of it in what LibreOffice computes:

- one file per ``soffice``, started on a private profile and pipe and stopped after, so
  concurrent items never share one (V2 runs one service on port 2002 for a whole batch);
- the result goes to a separate path; the source is never changed (V2 replaces it);
- a file LibreOffice cannot open or store is reported (`recalculate` returns why); V2
  prints the error and leaves the file as it was;
- a wall of `TIMEOUT_S` per file (V2 has none), so a hung LibreOffice cannot hold an item.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from refactory_packages.spreadsheetbench import recalc

HELPER = Path(__file__).resolve().with_name("lo_open_spreadsheet.py")
#: The most one file's recalculation may take, LibreOffice's start included.
TIMEOUT_S = 900


def program_dir() -> str | None:
    """open_spreadsheet.py:26-36: the directory of the real ``soffice``."""
    binary = recalc.soffice()
    return os.path.dirname(os.path.realpath(binary)) if binary else None


def lo_python() -> str | None:
    """open_spreadsheet.py:54-67: the bundled ``program/python``, else ``/usr/bin/python3``."""
    folder = program_dir()
    if folder is None:
        return None
    bundled = os.path.join(folder, "python")
    if os.path.isfile(bundled) and os.access(bundled, os.X_OK):
        return bundled
    return "/usr/bin/python3"


def _environment(folder: str) -> dict[str, str]:
    """open_spreadsheet.py:44-51."""
    env = dict(os.environ)
    env["UNO_PATH"] = folder
    env["URE_BOOTSTRAP"] = f"vnd.sun.star.pathname:{folder}/fundamentalrc"
    return env


def available() -> bool:
    return lo_python() is not None


def version() -> str:
    """``soffice --version``, for the prepared data's record."""
    binary = recalc.soffice()
    if binary is None:
        return ""
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip()


def recalculate(src: str, dst: str, *, timeout_s: float = TIMEOUT_S) -> tuple[bool, str]:
    """Recalculate ``src`` into ``dst`` as V2's ``process_file`` does: ``(True, "")``, or
    ``(False, why)`` when LibreOffice could not (``dst`` is then absent)."""
    folder, python = program_dir(), lo_python()
    binary = recalc.soffice()
    if folder is None or python is None or binary is None:
        return False, "LibreOffice not found"
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).unlink(missing_ok=True)
    profile = f"file://{recalc._profile()}"  # noqa: SLF001 - one profile per thread, shared
    pipe = f"sb2-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    # LibreOffice opens a private copy under a plain name (a release path has spaces, and
    # LibreOffice writes a lock file beside what it opens), and stores beside it.
    with tempfile.TemporaryDirectory(prefix="sb2-lo-") as work:
        copy, stored = os.path.join(work, "in.xlsx"), os.path.join(work, "out.xlsx")
        shutil.copyfile(src, copy)
        try:
            out = subprocess.run([python, "-I", str(HELPER), folder, binary, profile, pipe,
                                  copy, stored],
                                 capture_output=True, text=True, timeout=timeout_s,
                                 env=_environment(folder))
        except subprocess.TimeoutExpired:
            return False, f"LibreOffice took more than {timeout_s:.0f} s"
        except OSError as exc:
            return False, f"LibreOffice's Python could not start: {exc}"
        lines = [line for line in out.stdout.splitlines() if line.startswith("{")]
        try:
            result = json.loads(lines[-1]) if lines else {}
        except ValueError:
            result = {}
        if result.get("ok") and os.path.exists(stored):
            shutil.move(stored, dst)
            return True, ""
    why = result.get("error") or (out.stderr.strip().splitlines() or ["no result"])[-1]
    return False, str(why)[:500]
