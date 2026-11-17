"""Where model-written spreadsheet code runs: confined, and blind to the answers.

SkillOpt runs generated code as the trainer's user with no sandbox (its executor says
so, executor.py:60-61), with ``INPUT_PATH`` pointing into the dataset directory where
the item's golden file sits beside the input. The lab's runs added a sandbox that hid
secret paths and kept the network off, but left the golden file readable. The port is
stricter, and that can only lower a score that came from reading the answer:

- the code sees a fresh directory outside the run and the dataset, holding a copy of
  the item's input (``in/``), its output directory (``out/``, also ``HOME``) and an
  empty working directory (``cwd/``);
- the dataset directory, the run, the outer loop's root (every candidate's records)
  and the user's secrets (the lab wrapper's list, plus refactory's own directory) are
  hidden;
- there is no network; of the places a user can write (the home directory, ``/tmp``,
  ``/var/tmp``, ``/dev/shm``), only the output directory and the runner's temporary
  directory are writable.

Linux, after the lab's ``sandbox-exec.sh``: in user, mount and network namespaces
(``unshare -rnm``), empty read-only tmpfs over hidden directories and ``/dev/null`` over
hidden files, the writable directories bound to themselves, then the home directory and
the temporary roots remounted read-only around them. The code itself then runs in a
further user, mount and PID namespace with a fresh ``/proc``: the mounts above are locked
there, so it cannot undo them, and it sees no process outside its own (no
``/proc/<pid>/root`` into an unhidden view). macOS: Seatbelt (``sandbox-exec``) allowing
by default and denying the network, reads and writes of the hidden paths, and every write
outside the two writable directories. Where neither can run, code is not run: there is
no unsandboxed fallback (`available`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable

#: Secret paths under the home directory, as the lab's wrapper hides them
#: (``skillopt_sb.py`` ``_HIDE``).
HOME_SECRETS = (".claude", ".claude.json", ".config", ".ssh", ".aws", ".huggingface",
                ".cache/huggingface", ".netrc", ".git-credentials", ".gnupg", ".docker",
                ".kube", ".dsh", ".local/share/keyrings", ".bash_history",
                ".python_history")

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

#: Where a user can write, made read-only around the writable directories on Linux.
TEMP_ROOTS = ("/tmp", "/var/tmp", "/dev/shm")

# sandbox-exec.sh's order: hide, bind the writable directories to themselves, then make
# the roots read-only around them (a bind keeps the flags it is made with, so binding
# after the read-only remount would make the writable ones read-only too). Then a
# nested user, mount and PID namespace locks those mounts and shows a fresh /proc.
def _linux_script(read_only: list[str]) -> str:
    """The setup script, with the read-only roots written into it (quoted)."""
    import shlex

    roots = " ".join(shlex.quote(r) for r in read_only)
    return ("set -e\n"
            'W1=$1; W2=$2; shift 2\n'
            'while [ "$#" -gt 0 ] && [ "$1" != -- ]; do\n'
            '  if [ -d "$1" ]; then mount -n -t tmpfs -o ro,size=16k,mode=000 sbhide "$1"\n'
            '  elif [ -e "$1" ]; then mount -n --bind /dev/null "$1"; fi\n'
            "  shift\n"
            "done\n"
            "shift\n"
            'mount -n --bind "$W1" "$W1"\n'
            'mount -n --bind "$W2" "$W2"\n'
            f"for r in {roots}; do\n"
            '  [ -d "$r" ] || continue\n'
            '  mount -n --rbind "$r" "$r"\n'
            '  mount -n -o remount,bind,ro "$r"\n'
            "done\n"
            'exec unshare -U -r -m -p -f --mount-proc -- "$@"\n')


class SandboxError(RuntimeError):
    """The sandbox cannot run here, or cannot hide what it must."""


def hidden_paths(extra: Iterable[str | os.PathLike] = ()) -> list[str]:
    """What the code must not see: ``extra`` (the dataset and the run) and the secrets."""
    from refactory_dsh.paths import refactory_home

    home = Path.home()
    paths = [str(p) for p in extra if p]
    paths += [str(home / p) for p in HOME_SECRETS]
    paths.append(str(refactory_home()))
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if gac:
        paths.append(os.path.abspath(gac))
    seen: list[str] = []
    for path in paths:
        if os.path.lexists(path) and path not in seen:
            seen.append(path)
    return seen


def _sbpl(path: str) -> str:
    if any(ch in path for ch in ('"', "\\")) or not path.isprintable() or not path.startswith("/"):
        raise SandboxError(f"the sandbox cannot name the path {path!r}")
    return f'"{path}"'


def seatbelt_profile(writable: Iterable[str], hidden: Iterable[str]) -> str:
    """The macOS profile: allow by default; deny the network, every read and write of a
    hidden path, and every write but to ``writable`` and ``/dev/null``."""
    real = lambda p: os.path.realpath(p)  # noqa: E731 - Seatbelt matches resolved paths
    hide = " ".join(f"(subpath {_sbpl(real(p))})" if os.path.isdir(p)
                    else f"(literal {_sbpl(real(p))})" for p in hidden)
    allow = " ".join(f"(subpath {_sbpl(real(p))})" for p in writable)
    return ("(version 1)\n"
            "(allow default)\n"
            "(deny network*)\n"
            "(deny file-write*)\n"
            f'(allow file-write* (literal "/dev/null") {allow})\n'
            + (f"(deny file-read* file-write* {hide})\n" if hide else ""))


def wrapper(hidden: Iterable[str]):
    """``wrap(command, writable_dir, temp_dir)`` for `skillopt_port.run_generated_code`:
    the command confined on this platform, ``writable_dir`` and ``temp_dir`` writable."""
    hidden = list(hidden)

    def wrap(command: list[str], writable_dir: str, temp_dir: str) -> list[str]:
        writable = [writable_dir, temp_dir]
        for path in hidden:
            inside = os.path.realpath(path).rstrip("/") + "/"
            for needed in (*writable, command[0]):
                if os.path.realpath(needed).startswith(inside):
                    raise SandboxError(f"{needed} is inside the hidden {path}")
        if sys.platform == "darwin":
            return [SANDBOX_EXEC, "-p", seatbelt_profile(writable, hidden), *command]
        unshare = shutil.which("unshare") or "unshare"
        read_only = [str(Path.home()), *TEMP_ROOTS]
        return [unshare, "-rnm", "--", "/bin/sh", "-c", _linux_script(read_only), "sh",
                *writable, *hidden, "--", *command]

    return wrap


_PROBE = """
import os, sys
out, tmp, hidden = sys.argv[1:4]
for d in (out, tmp):
    open(os.path.join(d, "probe"), "w").write("x")
try:
    os.listdir(hidden)
except OSError:
    pass
else:
    sys.exit("a hidden directory can be read")
"""


@lru_cache(maxsize=None)
def _probe(python: str) -> tuple[bool, str]:
    """The sandbox runs ``python``, which can write both writable directories and cannot
    list a hidden one."""
    with tempfile.TemporaryDirectory(prefix="sb-probe-") as work:
        out, tmp, hidden = (Path(work) / name for name in ("out", "tmp", "hidden"))
        for d in (out, tmp, hidden):
            d.mkdir()
        command = wrapper([str(hidden)])(
            [python, "-I", "-c", _PROBE, str(out), str(tmp), str(hidden)], str(out), str(tmp))
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).strip()[-500:] or f"exit {proc.returncode}"
    return True, ""


def available(python: str) -> tuple[bool, str]:
    """``(ok, why)``: whether the sandbox runs ``python`` on this machine."""
    if sys.platform == "darwin":
        if not os.path.exists(SANDBOX_EXEC):
            return False, f"{SANDBOX_EXEC} is missing"
    elif sys.platform.startswith("linux"):
        if shutil.which("unshare") is None:
            return False, "the unshare command is not installed"
    else:
        return False, f"no sandbox for {sys.platform}"
    return _probe(python)


def require(python: str) -> None:
    ok, why = available(python)
    if not ok:
        raise SandboxError(f"model code is never run unsandboxed, and the sandbox cannot "
                           f"run here: {why}")
