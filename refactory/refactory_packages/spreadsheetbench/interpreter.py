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
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
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


#: How long `check`'s probe may take before it is an error (never a mismatch).
PROBE_TIMEOUT_S = 120


class InterpreterError(RuntimeError):
    """The pinned interpreter is missing or holds other versions."""


class InterpreterMismatch(InterpreterError):
    """The interpreter answered `check`'s probe, and what it reported is not the pins:
    another version, a package missing, or one that does not import. Only this says a
    tree is broken; a probe that timed out or could not start says nothing about it."""


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
    """Build the pinned environment with ``uv`` (once) and return its interpreter.

    An environment whose probe reports something other than the pins
    (`InterpreterMismatch`) is built again: a tree whose last move was cut short holds
    the packages' metadata but not all of their files (a directory move on a network or
    FUSE file system is file by file), and nothing else would ever replace it. A probe
    that timed out or could not start is raised, and nothing is replaced: on a loaded
    machine that says nothing about the tree, which another process may be running.
    Each build has a directory of its own, named by host and pid, and is moved into
    place under a lock, so builds that run at once (test workers, parallel runs) never
    write into one another's; one a killed build left behind is removed by the next
    build on its host (`_remove_dead_builds`).
    """
    home = Path(home or default_home())
    python = _venv_python(home)
    if _usable(python):
        return python
    uv = shutil.which("uv")
    if uv is None:
        raise InterpreterError("building the pinned interpreter needs uv (https://docs.astral.sh/uv/)")
    home.parent.mkdir(parents=True, exist_ok=True)
    with _build_lock(home):
        if _usable(python):
            return python  # built by another process while this one waited
        _remove_dead_builds(home)
        build = f"{_host()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        partial = home.with_name(f"{home.name}.partial-{build}")
        stale = home.with_name(f"{home.name}.stale-{build}")
        try:
            subprocess.run([uv, "venv", "-q", "--python", PYTHON, str(partial)], check=True)
            subprocess.run([uv, "pip", "install", "-q", "--python", str(_venv_python(partial)),
                            *(f"{name}=={version}" for name, version in PINS.items())], check=True)
            if _usable(python):
                return python  # another build finished first (no lock on this machine)
            if home.exists():
                home.rename(stale)
            partial.rename(home)
        finally:
            shutil.rmtree(partial, ignore_errors=True)
            shutil.rmtree(stale, ignore_errors=True)
    check.cache_clear()
    check(str(python))
    return python


def _usable(python: Path) -> bool:
    """Whether ``python`` exists and passes `check`; False only for a tree that is
    missing or whose probe reported a mismatch. A probe that timed out or could not
    start raises its `InterpreterError`."""
    if not python.exists():
        return False
    try:
        check(str(python))
    except InterpreterMismatch:
        return False
    return True


def _host() -> str:
    """This machine's name as a build directory carries it: the cache can be a mount
    other machines build into, whose pids say nothing here."""
    import socket

    return "".join(c if c.isalnum() else "_" for c in socket.gethostname()) or "host"


def _remove_dead_builds(home: Path) -> None:
    """Remove the build directories (``.partial-``, ``.stale-``) a build on this host left
    when it was killed before its own cleanup: those whose pid is not running. One whose
    pid runs, or that another host made, is left alone. Called under the build lock."""
    host = _host()
    for kind in ("partial", "stale"):
        for path in home.parent.glob(f"{home.name}.{kind}-{host}-*"):
            pid = path.name[len(f"{home.name}.{kind}-{host}-"):].split("-", 1)[0]
            if not pid.isdigit() or _running(int(pid)):
                continue
            shutil.rmtree(path, ignore_errors=True)


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _build_lock(home: Path) -> Iterator[None]:
    """Hold this machine's lock on building ``home``. The lock file is in the local
    temporary directory, since the cache directory can be a network or FUSE mount that
    refuses locks; without ``fcntl`` (Windows), or when the lock is refused, builds still
    use directories of their own and keep the first one finished."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    key = hashlib.sha256(str(home).encode()).hexdigest()[:16]
    with open(Path(tempfile.gettempdir()) / f"refactory-sb-interpreter-{key}.lock", "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:
            pass
        yield


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
    # Each named package is imported too: its metadata alone is there in a tree whose
    # move was cut short, and the package then fails in model code (`provision`).
    probe = ("import importlib, importlib.metadata as m, json, sys\n"
             "def imported(n):\n"
             "    try:\n"
             "        return getattr(importlib.import_module(n), '__version__', None)\n"
             "    except Exception as error:\n"
             "        return f'{type(error).__name__}: {error}'\n"
             "def version(n):\n"
             "    try:\n"
             "        return m.version(n)\n"
             "    except m.PackageNotFoundError:\n"
             "        return None\n"
             f"print(json.dumps({{'python': sys.version.split()[0], "
             f"'imported': {{n: imported(n) for n in {list(NAMED)!r}}}, "
             f"**{{n: version(n) for n in {sorted(PINS)!r}}}}}))")
    try:
        out = subprocess.run([python, "-I", "-c", probe], capture_output=True, text=True,
                             timeout=PROBE_TIMEOUT_S, check=True).stdout
        found = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise InterpreterError(f"{python} cannot report its packages: {exc}") from exc
    imported = found.pop("imported", None) or {}
    wrong = [f"{name} {found.get(name)} (pinned {version})"
             for name, version in PINS.items() if found.get(name) != version]
    wrong += [f"{name} imports as {imported.get(name)} (pinned {PINS[name]})"
              for name in NAMED if imported.get(name) != PINS[name]]
    if not str(found.get("python", "")).startswith(PYTHON + "."):
        wrong.insert(0, f"Python {found.get('python')} (pinned {PYTHON})")
    if wrong:
        raise InterpreterMismatch(f"{python} does not match the pins: " + ", ".join(wrong))
    return found


def main() -> int:
    print(provision())
    return 0


if __name__ == "__main__":
    sys.exit(main())
