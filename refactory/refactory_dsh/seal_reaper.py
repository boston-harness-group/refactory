"""Ends every process a sealed macOS command left behind (`refactory_dsh.seal`).

macOS has no PID namespace, so a sealed command can detach a process (``setsid``, a
double fork) that outlives it, still sealed, and keeps writing its workspace. Every
process a sealed command starts inherits its Seatbelt sandbox, and a process cannot
leave one, so the sandbox identifies them: :func:`reap` asks the kernel
(``sandbox_check``) which processes are sandboxed, may read this workspace and may not
read the directory above it -- only this step's -- and kills them.

Run as the runner's first program (stdlib only, ``python -I``): it starts the rest of
the runner, waits for it, then reaps; on SIGTERM or SIGINT it forwards the signal and
reaps too. `workspace.collect` reaps again before collecting a sealed workspace, so a
runner killed outright still leaves nothing writing it.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys

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


def reap(workspace: str, rounds: int = 5) -> list[int]:
    """Kill every process sealed in ``workspace``; repeated, since one may fork as it dies."""
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


def main(argv: list[str]) -> int:
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
