"""Ends every process a sealed command left behind (`refactory_dsh.seal`).

**macOS** has no PID namespace, so a sealed command can detach a process (``setsid``, a
double fork) that outlives it, still sealed, and keeps writing its workspace. Every
process a sealed command starts inherits its Seatbelt sandbox, and a process cannot
leave one, so the sandbox identifies them: :func:`reap` asks the kernel
(``sandbox_check``) which processes are sandboxed, may read this workspace and may not
read the directory above it -- only this step's -- and kills them. Run as the runner's
first program (stdlib only, ``python -I``): it starts the rest of the runner, waits for
it, then reaps; on SIGTERM or SIGINT it forwards the signal and reaps too.

**Linux**: each command runs in a PID namespace whose init is the runner's child, so
killing the runner (``unshare --kill-child``) or that init ends every process in it.
The runner is started through :func:`exec_runner` (``--exec``), which gives it a fresh
session keyring (none of the person's keys are reachable through it) and has the kernel
kill it when the process that started it (DSH) dies. :func:`reap` finds the runners and
namespace inits still working in a workspace by their arguments and kills them, so a
runner whose DSH was killed outright still ends.

`workspace.collect` reaps before collecting a sealed workspace, on both platforms.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time

#: ``sandbox_check`` filter types (``sandbox/sandbox.h``).
_FILTER_NONE = 0
_FILTER_PATH = 1


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libc.proc_listallpids.restype = ctypes.c_int
    # Declared without its variadic tail: ctypes passes the extra path argument as a
    # variadic one, which arm64 requires (on the stack, not in a register).
    libc.sandbox_check.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    libc.sandbox_check.restype = ctypes.c_int
    return libc


def sealed_pids(workspace: str) -> list[int]:
    """The processes in the sandbox of the sealed command working in ``workspace``."""
    libc = _libc()
    inside = ctypes.c_char_p(os.path.realpath(workspace).encode())
    outside = ctypes.c_char_p(os.path.dirname(os.path.realpath(workspace)).encode())
    count = libc.proc_listallpids(None, 0)
    pids = (ctypes.c_int * (count + 256))()
    count = libc.proc_listallpids(pids, ctypes.sizeof(pids))
    found = []
    for pid in pids[:max(count, 0)]:
        if pid <= 1 or pid == os.getpid():
            continue
        if libc.sandbox_check(pid, None, _FILTER_NONE) != 1:
            continue
        if (libc.sandbox_check(pid, b"file-read-data", _FILTER_PATH, inside) == 0
                and libc.sandbox_check(pid, b"file-read-data", _FILTER_PATH, outside) != 0):
            found.append(pid)
    return found


def _linux_runner_pids(workspace: str) -> list[int]:
    """The Linux runners (and their namespace inits) sealing ``workspace``: this user's
    processes whose arguments are the seal's shell followed by that workspace."""
    marker = b"\0refactory-seal\0" + os.path.realpath(workspace).encode() + b"\0"
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid != os.getuid():
                continue
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                arguments = b"\0" + handle.read()
        except OSError:
            continue
        if marker in arguments:
            found.append(int(entry))
    return found


def _children(pids: list[int]) -> list[int]:
    """The direct children of ``pids`` (a runner's namespace init)."""
    wanted, found = set(pids), []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                parent = int(handle.read().rsplit(b")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if parent in wanted:
            found.append(int(entry))
    return found


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            return handle.read().rsplit(b")", 1)[1].split()[0] != b"Z"
    except (OSError, IndexError):
        return False


def _reap_linux(workspace: str, rounds: int) -> list[int]:
    """Kill each runner sealing ``workspace`` and its namespace init: the kernel then
    kills every process in that PID namespace. Waits until they are gone."""
    killed: list[int] = []
    for _ in range(rounds):
        runners = _linux_runner_pids(workspace)
        if not runners:
            break
        targets = _children(runners) + runners
        for pid in targets:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                pass
        for _ in range(200):
            if not any(_alive(pid) for pid in targets):
                break
            time.sleep(0.01)
    return killed


def reap(workspace: str, rounds: int = 5) -> list[int]:
    """Kill every process sealed in ``workspace``; repeated, since one may fork as it dies."""
    if sys.platform.startswith("linux"):
        return _reap_linux(workspace, rounds)
    if sys.platform != "darwin":
        return []
    killed: list[int] = []
    for _ in range(rounds):
        found = sealed_pids(workspace)
        if not found:
            break
        for pid in found:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                pass
    return killed


#: ``keyctl`` and ``prctl`` (Linux): the system call numbers ``keyctl`` has.
_KEYCTL = {"x86_64": 250, "aarch64": 219}
_KEYCTL_JOIN_SESSION_KEYRING = 1
_PR_SET_PDEATHSIG = 1


def exec_runner(command: list[str]) -> int:
    """Linux: become ``command`` with a fresh, empty session keyring and killed when the
    process that started it dies. Either failing, nothing runs."""
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    number = _KEYCTL.get(os.uname().machine)
    if number is None or libc.syscall(number, _KEYCTL_JOIN_SESSION_KEYRING, None) < 0:
        print(f"refactory-seal: cannot make a session keyring of its own "
              f"(errno {ctypes.get_errno()})", file=sys.stderr)
        return 97
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        print("refactory-seal: cannot tie the runner to its parent", file=sys.stderr)
        return 97
    # Nothing of the caller's beyond stdin, stdout and stderr reaches the runner (the seal
    # closes them again, inside, as the command's last step).
    os.closerange(3, os.sysconf("SC_OPEN_MAX") if os.sysconf("SC_OPEN_MAX") > 0 else 65536)
    try:
        os.execv(command[0], command)
    except OSError as exc:
        print(f"refactory-seal: cannot start the sandbox: {exc}", file=sys.stderr)
    return 97


def main(argv: list[str]) -> int:
    if argv[:1] == ["--exec"]:
        return exec_runner(argv[1:])
    workspace, command = argv[0], argv[1:]
    try:
        child = subprocess.Popen(command)
    except OSError as exc:
        print(f"refactory-seal: cannot start the sandbox: {exc}", file=sys.stderr)
        return 97

    def forward(signum: int, _frame: object) -> None:
        child.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, forward)
    try:
        code = child.wait()
    finally:
        reap(workspace)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
