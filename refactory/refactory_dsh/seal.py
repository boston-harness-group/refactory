"""The sealed workspace: where the tools of a step an edit adds run.

A step the outer loop adds may list tools (`workspace.SEALED_TOOLS`). It works in a fresh
``scratch`` workspace holding copies of the files it reads, and every command its shell
runs goes through :func:`runner_command`, which DSH's local sandbox provider puts in
front of the command (its ``runnerCommand``). What the command can then see:

- its workspace, read and write, at the same path;
- the operating system's own programs and libraries, read only;
- nothing else: no other item's directory, no run directory, no project, no home
  directory, no grader file or gold answer wherever the run keeps them, no network;
- an environment of its own (`PATH`, its workspace as HOME and TMPDIR, the locale):
  nothing of the DSH process's, so no credential or run variable.

It is the code step's sandbox (`snippet`) with a writable workspace and a shell:

- **Linux**: user, network, mount and PID namespaces (``unshare -Urnmp``), an empty tmpfs
  over every root a run, a person or a service keeps files in (`snippet.HIDDEN_ROOTS`,
  plus the run directory and its parents' tree when it is elsewhere), then the workspace
  bound back at its own path. Its network namespace has only a down loopback.
- **macOS**: Seatbelt, deny by default, allowing reads of the system's programs and
  libraries, reads and writes of the workspace, starting processes from what it may
  read, and nothing on the network.

DSH's in-process tools (read, write, edit, glob, grep, web) run outside any runner, so a
sealed step may list only tools whose effects run through it: the shell. A runner that
cannot set the seal up prints ``refactory-seal:`` and exits 97, and the command does not
run (`FAILURE`). There is no fallback: where the seal cannot run (:func:`available`) a
new step is not offered tools.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from refactory_dsh import snippet

#: What the runner prints when it could not seal a command, as DSH's provider matches it.
FAILURE = "refactory-seal: "

#: What a sealed command's environment holds: nothing of the DSH process's (no
#: credential, no run variable), only these, with HOME and TMPDIR its workspace.
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

#: Runs the command, after DSH's profile arguments and ``--``, with that environment.
_EXEC = ('while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done; '
         '[ "$1" = "--" ] || { echo "refactory-seal: no command" >&2; exit 97; }; shift; '
         'cd "$ws" || exit 97; '
         f'exec /usr/bin/env -i HOME="$ws" TMPDIR="$ws" PATH="{PATH}" '
         f'LANG={snippet.LOCALE} LC_ALL={snippet.LOCALE} "$@"')

#: The shell that seals a command inside the new namespaces (Linux). Arguments: the
#: workspace, the number of directories to hide, those directories, then whatever DSH
#: appends (its bwrap-style profile, ``--``, the command). The workspace is held open
#: before anything is hidden and bound back through that descriptor, at its own path.
_SHELL = (
    'ws="$1"; n="$2"; shift 2; '
    'exec 3<"$ws" || { echo "refactory-seal: cannot open the workspace" >&2; exit 97; }; '
    'i=0; while [ "$i" -lt "$n" ]; do d="$1"; shift; i=$((i+1)); '
    'mount -n -t tmpfs -o size=64m,mode=0755 tmpfs "$d" '
    '|| { echo "refactory-seal: cannot hide $d" >&2; exit 97; }; done; '
    'mkdir -p "$ws" && mount -n --bind /proc/self/fd/3 "$ws" '
    '|| { echo "refactory-seal: cannot bind the workspace" >&2; exit 97; }; '
    'exec 3<&-; ' + _EXEC
)

#: What a sealed macOS command may read besides its workspace: the system's programs and
#: libraries, the command-line tools, Homebrew's, and the files the shell itself opens.
MAC_READS = ("/usr", "/bin", "/sbin", "/System", "/Library/Developer/CommandLineTools",
             "/opt/homebrew", "/private/etc", "/private/var/select", "/dev")

#: The shim that runs the command inside the profile sandbox-exec applies (macOS).
#: Arguments: the workspace, then whatever DSH appends.
_MAC_SHIM = 'ws="$1"; shift; ' + _EXEC


class SealError(RuntimeError):
    """The seal cannot be set up for this workspace; the message says why."""


def _check(workspace: Path) -> str:
    path = os.path.realpath(workspace)
    if not os.path.isdir(path) or path == "/":
        raise SealError(f"a sealed workspace is an existing directory, not {path!r}")
    return path


def mac_profile(workspace: Path) -> str:
    """The deny-by-default Seatbelt profile a sealed macOS command runs under."""
    ws = _check(workspace)
    reads = [path for path in MAC_READS if os.path.exists(path)]
    for path in reads:
        if ws == path or ws.startswith(path.rstrip("/") + "/"):
            raise SealError(f"the workspace {ws} is inside {path}, which every sealed step may read")
    above = sorted({parent for path in (*reads, ws) for parent in snippet._ancestors(path)})
    readable = " ".join(f"(subpath {snippet._sbpl(path)})" for path in reads)
    metadata = " ".join(f"(literal {snippet._sbpl(path)})" for path in above)
    return ("(version 1)\n"
            "(deny default)\n"
            "(allow process-fork)\n"
            f"(allow process-exec (subpath {snippet._sbpl(ws)}) {readable})\n"
            "(allow process-info* (target same-sandbox))\n"
            "(allow signal (target same-sandbox))\n"
            "(allow sysctl-read)\n"
            f'(allow file-read* (literal "/") {readable} (subpath {snippet._sbpl(ws)}))\n'
            + (f"(allow file-read-metadata {metadata})\n" if metadata else "")
            + f'(allow file-write* (literal "/dev/null") (literal "/dev/tty") '
              f'(subpath {snippet._sbpl(ws)}))\n')


def runner_command(workspace: Path, hide: tuple[str, ...] = ()) -> list[str]:
    """The runner argv DSH's local sandbox provider puts in front of each command of a
    sealed step (``runnerCommand``). ``hide`` adds directories to hide on Linux (the run
    directory and the project, when they are outside every hidden root)."""
    ws = _check(workspace)
    if sys.platform == "darwin":
        return [snippet.SANDBOX_EXEC, "-p", mac_profile(workspace), "/bin/sh", "-c", _MAC_SHIM,
                "refactory-seal", ws]
    if not sys.platform.startswith("linux"):
        raise SealError(f"no sealed workspace on {sys.platform}")
    # /proc/sys is hidden after the fresh /proc is mounted, so it goes last.
    hidden = [path for path in snippet._hidden((*hide, os.path.expanduser("~")))
              if path != "/proc/sys"] + ["/proc/sys"]
    return [shutil.which("unshare") or "unshare", "-Urnmp", "--fork", "--kill-child",
            "--mount-proc", "--", snippet.BASH, "-c", _SHELL, "refactory-seal", ws,
            str(len(hidden)), *hidden]


def run(workspace: Path, command: list[str], *, hide: tuple[str, ...] = (),
        timeout_s: float = 60) -> subprocess.CompletedProcess:
    """``command`` run sealed in ``workspace``, as DSH runs a sealed step's command: the
    runner, a stand-in for DSH's profile arguments, ``--``, then the command."""
    argv = [*runner_command(workspace, hide), "--ro-bind", "/", "/", "--", *command]
    return subprocess.run(argv, cwd=os.path.realpath(workspace), capture_output=True,
                          text=True, timeout=timeout_s)


@functools.lru_cache(maxsize=1)
def available() -> tuple[bool, str]:
    """Whether a sealed command runs here, and why not: one probe per process."""
    try:
        with tempfile.TemporaryDirectory() as root:
            result = run(Path(root), ["/bin/sh", "-c", "echo sealed > probe && cat probe"])
    except (SealError, OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0 or result.stdout.strip() != "sealed":
        return False, (result.stderr.strip() or f"exit {result.returncode}")[:300]
    return True, ""
