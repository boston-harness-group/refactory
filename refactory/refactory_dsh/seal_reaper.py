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
import json
import os
import signal
import socket
import subprocess
import sys
import time

#: Beside a sealed workspace (``<workspace>`` + this), where nothing sealed writes: one
#: line per command the runner could not start. `workspace.collect` raises them as a
#: harness incident (`incidents.SealFailed`), so a seal that failed is never a step whose
#: commands quietly found nothing.
INCIDENTS_SUFFIX = ".seal-incidents.jsonl"
#: A sealed command's limits, as ``ulimit`` takes them (`seal`, which re-exports these):
#: CPU seconds and the largest file (KiB), on both platforms.
LIMITS = "-t 3600 -f 4194304"
#: Linux, in a systemd user scope that caps the whole command: a loose per-process bound.
LINUX_LIMITS = "-u 512"
#: Linux without a user scope: a tight per-process bound (128 processes, 8 GiB each).
LINUX_LIMITS_UNSCOPED = "-u 128 -v 8388608"


def incidents_path(workspace: str) -> str:
    """Where the runner records the commands it could not start in ``workspace``."""
    return os.path.realpath(workspace) + INCIDENTS_SUFFIX


#: The descriptor the seal says on that its command is starting (`exec_runner`): high, so
#: no step of the seal (its workspace on 3) meets it.
START_FD = 98
#: Beside a sealed workspace: notes on how its commands ran (one without the whole-step
#: cap, say), which `workspace.collect` puts in the step's dispatch record.
NOTES_SUFFIX = ".seal-notes.jsonl"


def notes_path(workspace: str) -> str:
    """Where the runner notes how the commands of ``workspace`` ran."""
    return os.path.realpath(workspace) + NOTES_SUFFIX


def record_note(workspace: str, note: str) -> None:
    """Note how a command of the sealed step in ``workspace`` ran."""
    try:
        with open(notes_path(workspace), "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": round(time.time(), 3), "note": note}) + "\n")
    except OSError as exc:
        print(f"refactory-seal: cannot record the note: {exc}", file=sys.stderr)


def record_incident(workspace: str, reason: str) -> None:
    """Record that a command of the sealed step in ``workspace`` could not start, and why."""
    try:
        with open(incidents_path(workspace), "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": round(time.time(), 3), "reason": reason}) + "\n")
    except OSError as exc:
        print(f"refactory-seal: cannot record the incident: {exc}", file=sys.stderr)


def _workspace_of(command: list[str]) -> str:
    """The workspace a Linux runner seals: the argument after ``refactory-seal``."""
    if "refactory-seal" in command:
        index = command.index("refactory-seal")
        if index + 1 < len(command):
            return command[index + 1]
    return ""


def scope_bus(command: list[str]) -> str | None:
    """The user bus a runner's systemd scope is made through, or None for an unscoped one."""
    if not any(os.path.basename(arg) == "systemd-run" for arg in command):
        return None
    prefix = "DBUS_SESSION_BUS_ADDRESS=unix:path="
    return next((arg[len(prefix):] for arg in command if arg.startswith(prefix)), None)


def _bus_up(path: str) -> bool:
    """Whether systemd's user bus answers at ``path``."""
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(2)
    try:
        probe.connect(path)
        return True
    except OSError:
        return False
    finally:
        probe.close()


def unscoped(command: list[str]) -> list[str] | None:
    """``command`` without its systemd scope and under the unscoped per-process limits --
    the runner `seal.runner_command` makes where there is no user scope -- or None when
    it is not a scoped Linux runner of that shape."""
    names = [os.path.basename(arg) for arg in command]
    if "systemd-run" not in names or "--" not in command[names.index("systemd-run"):]:
        return None
    rest = command[command.index("--", names.index("systemd-run")) + 1:]
    scoped = f"{LIMITS} {LINUX_LIMITS}"
    if rest.count(scoped) != 1:
        return None
    return [f"{LIMITS} {LINUX_LIMITS_UNSCOPED}" if arg == scoped else arg for arg in rest]

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
    """Linux: run ``command`` (the seal) with a fresh, empty session keyring, tied to the
    process that started it (DSH): this runner dies with it, and the seal with this runner.
    The runner waits on the seal as its child and hands its exit status back. Unless the
    seal says, on `START_FD`, that the command is starting (its last step,
    ``seal_close.sh``, does so after every other descriptor is closed and checked), the
    command never ran: the seal's own failure (an exit 97 such as "cannot bind /etc/group",
    a systemd-run or unshare that failed, a signal during setup) is recorded beside the
    workspace (`record_incident`), so the step is a harness incident, never a step whose
    commands quietly did nothing. Nothing the command does can be recorded so: its start
    is said before it runs, and it never holds that descriptor."""
    parent = os.getppid()
    workspace = _workspace_of(command)

    def refuse(reason: str) -> int:
        # The command does not start: said to its caller, and recorded beside the
        # workspace as the harness's failure (`record_incident`).
        print(f"refactory-seal: {reason}", file=sys.stderr)
        if workspace:
            record_incident(workspace, reason)
        return 97

    bus = scope_bus(command)
    if bus is not None and not _bus_up(bus):
        # systemd's user manager went away since this runner was made (its runtime
        # directory goes with the person's last session): the command runs as it would
        # have had the step started a moment later, unscoped under the tight per-process
        # limits (`seal.linux_scope`), rather than fail in systemd-run before it starts.
        fallback = unscoped(command)
        if fallback is None:
            return refuse(f"systemd's user scope bus {bus} is gone, and the runner has no "
                          "unscoped form")
        command = fallback
        if workspace:
            record_note(workspace, f"systemd's user scope bus {bus} went away during the "
                                   "step: a command ran without the whole-step cap")
    libc = ctypes.CDLL(None, use_errno=True)
    number = _KEYCTL.get(os.uname().machine)
    if number is None or libc.syscall(number, _KEYCTL_JOIN_SESSION_KEYRING, None) < 0:
        return refuse(f"cannot make a session keyring of its own (errno {ctypes.get_errno()})")
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        return refuse("cannot tie the runner to its parent")
    started_r, started_w = os.pipe()
    runner = os.getpid()
    limit = os.sysconf("SC_OPEN_MAX") if os.sysconf("SC_OPEN_MAX") > 0 else 65536
    child = os.fork()
    if child == 0:
        # The seal: killed with this runner, and nothing of the caller's beyond stdin,
        # stdout, stderr and the start descriptor (the seal closes even those above
        # stderr, inside, as the command's last step).
        try:
            if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0 \
                    or os.getppid() != runner:
                os._exit(97)
            os.dup2(started_w, START_FD)
            os.closerange(3, START_FD)
            os.closerange(START_FD + 1, limit)
            os.execv(command[0], command)
        except BaseException as exc:  # noqa: BLE001 - nothing may run in the child after this
            try:
                print(f"refactory-seal: cannot start the sandbox: {exc}", file=sys.stderr)
            finally:
                os._exit(97)
    os.close(started_w)
    os.closerange(3, started_r)
    os.closerange(started_r + 1, limit)

    def forward(signum: int, _frame: object) -> None:
        try:
            os.kill(child, signum)
        except ProcessLookupError:
            pass

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, forward)
    while True:
        try:
            _pid, status = os.waitpid(child, 0)
            break
        except ChildProcessError:
            status = 97 << 8
            break
        except InterruptedError:
            continue
    os.set_blocking(started_r, False)
    try:
        started = bool(os.read(started_r, 1))
    except BlockingIOError:
        started = False
    finally:
        os.close(started_r)
    if os.WIFSIGNALED(status):
        code, how = 128 + os.WTERMSIG(status), f"was killed by signal {os.WTERMSIG(status)}"
    else:
        code = os.WEXITSTATUS(status)
        how = f"exited {code}"
    if workspace and not started:
        record_incident(workspace, f"the seal {how} before the command started")
    return code


def main(argv: list[str]) -> int:
    if argv[:1] == ["--exec"]:
        return exec_runner(argv[1:])
    workspace, command = argv[0], argv[1:]
    try:
        child = subprocess.Popen(command)
    except OSError as exc:
        print(f"refactory-seal: cannot start the sandbox: {exc}", file=sys.stderr)
        record_incident(workspace, f"cannot start the sandbox: {exc}")
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
