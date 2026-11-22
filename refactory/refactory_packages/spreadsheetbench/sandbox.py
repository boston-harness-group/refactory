"""Where model-written spreadsheet code runs: confined, and blind to the answers.

SkillOpt runs generated code as the trainer's user with no sandbox (its executor says
so, executor.py:60-61), with ``INPUT_PATH`` pointing into the dataset directory where
the item's golden file sits beside the input. The lab's runs added a sandbox that hid
secret paths and kept the network off, but left the golden file readable. The port is
stricter, and that can only lower a score that came from reading the answer.

Linux: an allow-list. The code runs in a fresh root holding only the system directories
(`SYSTEM_DIRS`, read-only), the interpreter's own directories (read-only), a few device
files, the directories it is given -- a copy of the item's input and its working
directory, read-only -- and its output and temporary directories, the only writable
ones. Nothing else exists there: not the dataset or any copy of it, not the run or the
outer loop's root, not the home directory, ``/tmp``, a model cache or a shared drive,
and there is no network. After the lab's ``sandbox-exec.sh`` and bubblewrap: user,
mount, network and PID namespaces (``unshare -rnmpf``), a tmpfs root built from bind
mounts and a fresh ``/proc``, entered with ``pivot_root`` (the old root detached, so
there is no path back to it), then a further user and mount namespace: the mounts are
locked there, so the code cannot undo them, and it sees no process outside its own.

macOS: a hide-list. Seatbelt (``sandbox-exec``) allowing by default and denying the
network, reads and writes of the hidden paths (`hidden_paths`: the dataset and copies of
the release beside it, the run, the outer loop's root, model caches, the shared drive and
the user's secrets), and every write outside the two writable directories.

Where neither can run, code is not run: there is no unsandboxed fallback (`available`).
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

from refactory_dsh.model import HarnessIncident
# The interpreter's directories, found as a sealed tool step's are (`seal.tool_path`).
from refactory_dsh.seal import _distinct, interpreter_dirs, system_tool  # noqa: F401

#: Secret paths under the home directory, as the lab's wrapper hides them
#: (``skillopt_sb.py`` ``_HIDE``). macOS hides them; Linux shows nothing it does not list.
HOME_SECRETS = (".claude", ".claude.json", ".config", ".ssh", ".aws", ".huggingface",
                ".cache/huggingface", ".netrc", ".git-credentials", ".gnupg", ".docker",
                ".kube", ".dsh", ".local/share/keyrings", ".bash_history",
                ".python_history")

#: Shared drives a copy of the release may sit on, hidden on macOS.
SHARED_DIRS = ("/new_data",)

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

#: The system directories model code sees, read-only (Linux); a symlink among them is
#: recreated as one (``/bin`` -> ``usr/bin``).
SYSTEM_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc")
#: The shell the Linux setup runs in: POSIX only (dash on Ubuntu, bash elsewhere); the
#: tests run it under dash too where dash is installed.
SHELL = "/bin/sh"
#: The device files model code may open.
DEVICES = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")


def _q(path: str) -> str:
    import shlex

    return shlex.quote(path)


def _linux_script(read_only: list[str], writable: list[str], links: dict[str, str],
                  ready_fd: int | None = None) -> str:
    """The setup script, run as PID 1 of a new PID namespace: build a tmpfs root under
    the temporary directory (``$1``) with ``links`` as symlinks, ``read_only`` bound
    read-only, ``writable`` and the devices bound as they are and a fresh ``/proc`` of
    this PID namespace, enter it with ``pivot_root``, detach the old root, then run the
    command (the rest of the arguments) from ``$2`` (its working directory) in a nested
    user and mount namespace, where these mounts are locked. With ``ready_fd``, ``ok``
    is written to it just before the code starts: a setup that failed never writes it."""
    lines = ["set -e",
             'R="$1/.sbroot"; WD=$2; shift 2',
             'mkdir "$R"',
             'mount -n -t tmpfs -o mode=755 sbroot "$R"']
    for link, target in links.items():
        lines.append(f'mkdir -p "$R"{_q(os.path.dirname(link))}; ln -s {_q(target)} "$R"{_q(link)}')
    for path in read_only:
        lines.append(f'mkdir -p "$R"{_q(path)}; mount -n --bind {_q(path)} "$R"{_q(path)}; '
                     f'mount -n -o remount,bind,ro "$R"{_q(path)}')
    for path in writable:
        lines.append(f'mkdir -p "$R"{_q(path)}; mount -n --bind {_q(path)} "$R"{_q(path)}')
    for dev in DEVICES:
        if os.path.exists(dev):
            lines.append(f'mkdir -p "$R/dev"; touch "$R"{_q(dev)}; '
                         f'mount -n --bind {_q(dev)} "$R"{_q(dev)}')
    lines += ['mkdir -p "$R/proc"; mount -n -t proc -o nosuid,nodev,noexec proc "$R/proc"',
              # pivot_root by its path: a service's PATH may lack /usr/sbin (`seal.system_tool`).
              f'mkdir "$R/.old"; cd "$R"; {_q(system_tool("pivot_root"))} . .old; '
              'umount -n -l /.old; rmdir /.old',
              'cd "$WD" 2>/dev/null || cd /',
              ('exec unshare -U -r -m -- "$@"' if ready_fd is None else
               # The last step before the code: say so on the caller's pipe
               # (`skillopt_port.run_generated_code`, which reads only these 2 bytes).
               # POSIX only: dash, Ubuntu's sh, takes no fd number above 9 in a
               # redirection, so the pipe is opened by its /proc path.
               f'exec unshare -U -r -m -- {SHELL} -c '
               f'\'printf ok > /proc/self/fd/{int(ready_fd)} && exec "$@"\' sh "$@"')]
    return "\n".join(lines) + "\n"


class SandboxError(RuntimeError):
    """The sandbox cannot run here, or cannot hide what it must."""


class SandboxFailed(HarnessIncident):
    """The sandbox's setup failed for one call (a mount, a namespace) before the code
    started: the harness's failure, recorded as a harness incident, never the model's
    code error or one of its turns."""


def hidden_paths(extra: Iterable[str | os.PathLike] = ()) -> list[str]:
    """What the code must not see (macOS hides it; on Linux nothing outside the
    allow-list is there, and these are checked against it): ``extra`` (the dataset and
    the run), copies of the release beside the dataset, the model caches, the shared
    drives and the secrets."""
    from refactory_dsh.paths import refactory_home

    home = Path.home()
    paths = [str(p) for p in extra if p]
    paths += [os.path.realpath(p) for p in list(paths)]
    paths += _release_copies([str(p) for p in extra if p])
    paths += [str(home / p) for p in HOME_SECRETS]
    paths.append(str(refactory_home()))
    cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if cache:
        paths.append(os.path.join(cache, "huggingface"))
    for name in ("HF_HOME", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        value = os.environ.get(name, "").strip()
        if value:
            paths.append(os.path.abspath(os.path.expanduser(value)))
    paths += SHARED_DIRS
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if gac:
        paths.append(os.path.abspath(gac))
    seen: list[str] = []
    for path in paths:
        if os.path.lexists(path) and path not in seen:
            seen.append(path)
    return seen


def _release_copies(extra: list[str]) -> list[str]:
    """Beside each directory of ``extra``: the entries named like the release
    (``spreadsheetbench...``: the archive, other extractions of it)."""
    found: list[str] = []
    for path in extra:
        for form in {os.path.abspath(path), os.path.realpath(path)}:
            parent = os.path.dirname(form.rstrip("/"))
            try:
                names = os.listdir(parent)
            except OSError:
                continue
            found += [os.path.join(parent, name) for name in sorted(names)
                      if "spreadsheetbench" in name.lower()]
    return found


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


def wrapper(hidden: Iterable[str], visible: Iterable[str] = ()):
    """``wrap(command, writable_dir, temp_dir)`` for `skillopt_port.run_generated_code`:
    the command confined on this platform, ``writable_dir`` and ``temp_dir`` writable.
    ``visible``: what else the code reads (the input copy, its working directory), and,
    on Linux, the first one that is a directory is where it starts."""
    hidden = list(hidden)
    visible = [os.path.abspath(p) for p in visible]

    def wrap(command: list[str], writable_dir: str, temp_dir: str,
             ready_fd: int | None = None) -> list[str]:
        writable = [os.path.abspath(writable_dir), os.path.abspath(temp_dir)]
        for path in hidden:
            inside = os.path.realpath(path).rstrip("/") + "/"
            for needed in (*writable, *visible, command[0]):
                if os.path.realpath(needed).startswith(inside):
                    raise SandboxError(f"{needed} is inside the hidden {path}")
        if sys.platform == "darwin":
            return [SANDBOX_EXEC, "-p", seatbelt_profile(writable, hidden), *command]
        links = {p: os.readlink(p) for p in SYSTEM_DIRS if os.path.islink(p)}
        read_only = _distinct([*(p for p in SYSTEM_DIRS if p not in links),
                               *interpreter_dirs(command[0]), *visible])
        read_only = [p for p in read_only
                     if not any((p + "/").startswith(w.rstrip("/") + "/") for w in writable)]
        _refuse_covering([*read_only, *writable], [*hidden, str(Path.home())])
        start = next((p for p in visible if os.path.isdir(p)), "/")
        unshare = shutil.which("unshare") or "unshare"
        return [unshare, "-rnmpf", "--", SHELL, "-c",
                _linux_script(read_only, _distinct(writable), links, ready_fd=ready_fd), "sh",
                os.path.abspath(temp_dir), start, *command]

    # Linux says when the code starts (``ready_fd``); Seatbelt has no setup to fail.
    wrap.confirms_start = not sys.platform == "darwin"  # type: ignore[attr-defined]
    return wrap


def _refuse_covering(bound: Iterable[str], secret: Iterable[str]) -> None:
    """The reverse of the hidden check: no directory bound into the code's root may hold
    the home directory, the dataset or another hidden path (an interpreter installed in
    the home directory itself would bring all of it along). Refused, never narrowed."""
    for path in secret:
        real = os.path.realpath(path).rstrip("/") + "/"
        for dir_ in bound:
            if real.startswith(os.path.realpath(dir_).rstrip("/") + "/"):
                raise SandboxError(f"the sandbox would show {path}: it is inside {dir_}, "
                                   "which the code needs")


_PROBE = """
import os, sys
out, tmp, hidden = sys.argv[1:4]
for d in (out, tmp):
    open(os.path.join(d, "probe"), "w").write("x")
try:
    open(os.path.join(hidden, "secret")).read()
except OSError:
    pass
else:
    sys.exit("a file in a hidden directory can be read")
"""


@lru_cache(maxsize=None)
def _probe(python: str) -> tuple[bool, str]:
    """The sandbox runs ``python``, which can write both writable directories and cannot
    read a file in a hidden one (on Linux it is not there at all: nothing outside the
    allow-list is)."""
    with tempfile.TemporaryDirectory(prefix="sb-probe-") as work:
        out, tmp, hidden = (Path(work) / name for name in ("out", "tmp", "hidden"))
        for d in (out, tmp, hidden):
            d.mkdir()
        (hidden / "secret").write_text("x")
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
