"""The pinned interpreter model-written spreadsheet code runs under.

SkillOpt's runs executed generated code with its own venv's Python 3.12.14, openpyxl
3.1.5 and, through ``PYTHONPATH``, pandas 2.3.3 from the lab's ``pylib`` (the venv
lacks it, and the prompt names it). Here that set is one virtual environment with
exactly those versions -- the ones the lab's interpreter reports, numpy and pandas'
own dependencies included -- built once with ``uv``::

    python -m refactory_packages.spreadsheetbench.interpreter

``SB_PYTHON`` names another interpreter; it is checked against the same pins before
any item runs, and a mismatch is refused by name. Python's patch version is not
pinned (``uv`` offers 3.12.14 on some platforms only); the version used is recorded
in each run (`check`).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

#: The interpreter's minor version, and every package SkillOpt's code saw, at the
#: versions the lab's ``~/skillopt/.venv`` (plus ``pylib``) reports.
PYTHON = "3.12"
PINS: dict[str, str] = {
    "openpyxl": "3.1.5",
    "et_xmlfile": "2.0.0",
    "pandas": "2.3.3",
    "numpy": "2.5.4",
    "python-dateutil": "4.0.0",
    "pytz": "2026.6",
    "tzdata": "2026.6",
    "six": "1.17.0",
}
ENV = "SB_PYTHON"
#: The packages a request names when it says what a sealed tool step's ``python3`` is.
NAMED = ("openpyxl", "pandas", "numpy")
#: What it is, as the request says it (`tasks.SpreadsheetBenchTask.tool_environment`).
SUMMARY = (f"python3 (and python) = Python {PYTHON} with "
           + ", ".join(f"{name} {PINS[name]}" for name in NAMED))


class InterpreterError(RuntimeError):
    """The pinned interpreter is missing or holds other versions."""


def default_home() -> Path:
    """Where the default environment lives: the user's cache directory, named by the
    pins, so a change of pins builds a new one. Not under refactory's own directory,
    which the sandbox hides from model code (`sandbox.hidden_paths`)."""
    digest = hashlib.sha256(json.dumps([PYTHON, PINS], sort_keys=True).encode()).hexdigest()[:12]
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
    return cache / "refactory-spreadsheetbench" / f"py{PYTHON.replace('.', '')}-{digest}"


def _venv_python(home: Path) -> Path:
    return home / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def provision(home: Path | None = None) -> Path:
    """Build the pinned environment with ``uv`` (once) and return its interpreter."""
    home = Path(home or default_home())
    python = _venv_python(home)
    if python.exists():
        check(str(python))
        return python
    uv = shutil.which("uv")
    if uv is None:
        raise InterpreterError("building the pinned interpreter needs uv (https://docs.astral.sh/uv/)")
    home.parent.mkdir(parents=True, exist_ok=True)
    partial = home.with_name(home.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    subprocess.run([uv, "venv", "-q", "--python", PYTHON, str(partial)], check=True)
    subprocess.run([uv, "pip", "install", "-q", "--python", str(_venv_python(partial)),
                    *(f"{name}=={version}" for name, version in PINS.items())], check=True)
    partial.rename(home)
    check(str(python))
    return python


def interpreter() -> str:
    """The interpreter model code runs under: ``SB_PYTHON``, else the default
    environment. Checked against the pins (`check`); a missing one is refused with the
    command that builds it."""
    explicit = os.environ.get(ENV, "").strip()
    python = Path(explicit) if explicit else _venv_python(default_home())
    if not python.exists():
        raise InterpreterError(
            f"the pinned interpreter for model code is missing ({python}); build it with "
            f"`{Path(sys.executable).name} -m refactory_packages.spreadsheetbench.interpreter`"
            + (f" or unset {ENV}" if explicit else ""))
    check(str(python))
    return str(python)


@lru_cache(maxsize=None)
def check(python: str) -> dict[str, str]:
    """The interpreter's Python version and each pinned package's version; refused when
    the minor version or any package differs from the pins."""
    probe = ("import importlib.metadata as m, json, sys; "
             f"print(json.dumps({{'python': sys.version.split()[0], "
             f"**{{n: m.version(n) for n in {sorted(PINS)!r}}}}}))")
    try:
        out = subprocess.run([python, "-I", "-c", probe], capture_output=True, text=True,
                             timeout=120, check=True).stdout
        found = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise InterpreterError(f"{python} cannot report its packages: {exc}") from exc
    wrong = [f"{name} {found.get(name)} (pinned {version})"
             for name, version in PINS.items() if found.get(name) != version]
    if not str(found.get("python", "")).startswith(PYTHON + "."):
        wrong.insert(0, f"Python {found.get('python')} (pinned {PYTHON})")
    if wrong:
        raise InterpreterError(f"{python} does not match the pins: " + ", ".join(wrong))
    return found


def main() -> int:
    print(provision())
    return 0


if __name__ == "__main__":
    sys.exit(main())
