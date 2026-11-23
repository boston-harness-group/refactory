"""Ends what a timed-out attempt left running before its retry keeps the workspace.

A node whose retry rule keeps its workspace (``keep_workspace``) continues from the
timed-out attempt's edits. The attempt's session is stopped (`owned_process.shutdown`
signals only the CLI), but a command its tools started may still be running there -- a
test run, a build -- and would race the retry's own edits. Before the workspace is kept,
:func:`end` finds those processes, on Linux and macOS alike:

- every process in the attempt's own session (the CLI starts in a session of its own,
  `model.ModelClient._run_completion`, whose id the timeout carries as ``session_id``),
  wherever it now works;
- every process whose working directory is inside the node's workspace, which catches one
  that left the session (``setsid``).

Each is sent SIGTERM, then SIGKILL after a grace period, and waited for until it is gone or
a zombie (a zombie runs nothing and holds no file). The wait is bounded: when a process
survives it, :func:`keeps_workspace` refuses to keep the workspace, and the retry starts
from a fresh one with a warning.
"""

from __future__ import annotations

import functools
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

#: How long a process has after SIGTERM before it is sent SIGKILL.
TERM_GRACE_S = 5.0
#: How long a process has after SIGKILL to be gone before it counts as surviving.
KILL_WAIT_S = 5.0
_POLL_S = 0.05

#: libproc's flavours (``sys/proc_info.h``): the working directory, and its record's size.
_PROC_PIDVNODEPATHINFO = 9
_VNODE_INFO_SIZE = 152
_MAXPATHLEN = 1024
_VNODEPATHINFO_SIZE = 2 * (_VNODE_INFO_SIZE + _MAXPATHLEN)


def _linux() -> bool:
    return Path("/proc/self/stat").exists()


@functools.lru_cache(maxsize=1)
def _libc():
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libc.proc_listallpids.restype = ctypes.c_int
    libc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                  ctypes.c_void_p, ctypes.c_int]
    libc.proc_pidinfo.restype = ctypes.c_int
    return libc


def _pids() -> list[int]:
    """Every process the system lists."""
    if _linux():
        return [int(name) for name in os.listdir("/proc") if name.isdigit()]
    if sys.platform == "darwin":
        import ctypes

        libc = _libc()
        count = libc.proc_listallpids(None, 0)
        pids = (ctypes.c_int * (max(count, 0) + 256))()
        count = libc.proc_listallpids(pids, ctypes.sizeof(pids))
        return [pid for pid in pids[:max(count, 0)] if pid > 0]
    listed = subprocess.run(["ps", "-A", "-o", "pid="], capture_output=True, text=True,
                            timeout=30, check=False).stdout
    return [int(word) for word in listed.split() if word.isdigit()]


def _cwd(pid: int) -> str | None:
    """``pid``'s working directory, or None when it cannot be read (gone, or not ours)."""
    try:
        if _linux():
            return os.readlink(f"/proc/{pid}/cwd")
        if sys.platform == "darwin":
            import ctypes

            buffer = ctypes.create_string_buffer(_VNODEPATHINFO_SIZE)
            size = _libc().proc_pidinfo(pid, _PROC_PIDVNODEPATHINFO, 0, buffer,
                                        _VNODEPATHINFO_SIZE)
            if size < _VNODE_INFO_SIZE + _MAXPATHLEN:
                return None
            raw = buffer.raw[_VNODE_INFO_SIZE:_VNODE_INFO_SIZE + _MAXPATHLEN]
            return raw.split(b"\0", 1)[0].decode(errors="replace") or None
    except OSError:
        return None
    return None


def _state(pid: int) -> str | None:
    """``pid``'s state letter (``Z`` for a zombie), or None when it is gone."""
    if _linux():
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return None
        return text[text.rfind(")") + 2:][:1] or None
    shown = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                           text=True, timeout=30, check=False).stdout.strip()
    return shown[:1] or None


def _gone(pid: int) -> bool:
    """Whether ``pid`` has ended: no longer listed, or a zombie (reaped here if ours)."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    except OSError:
        pass
    state = _state(pid)
    return state is None or state == "Z"


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def find(root: Path, session: int | None = None) -> list[int]:
    """The live processes of the attempt: in its ``session``, or working inside ``root``."""
    root_text = os.path.realpath(root)
    own = {os.getpid(), os.getppid()}
    try:
        own_session = os.getsid(0)
    except OSError:
        own_session = None
    if session is not None and session == own_session:
        session = None  # never this process's own session
    found = []
    for pid in _pids():
        if pid <= 1 or pid in own:
            continue
        member = False
        if session is not None:
            try:
                member = os.getsid(pid) == session
            except OSError:
                member = False
        if not member:
            cwd = _cwd(pid)
            member = cwd is not None and _inside(os.path.realpath(cwd), root_text)
        if member and not _gone(pid):
            found.append(pid)
    return sorted(found)


def _signal(pids: list[int], signum: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, signum)
        except OSError:
            pass


def _wait(pids: list[int], within_s: float) -> list[int]:
    """Those of ``pids`` still alive after waiting up to ``within_s`` for all to end."""
    deadline = time.monotonic() + within_s
    alive = [pid for pid in pids if not _gone(pid)]
    while alive and time.monotonic() < deadline:
        time.sleep(_POLL_S)
        alive = [pid for pid in alive if not _gone(pid)]
    return alive


def end(root: Path, session: int | None = None, *, term_grace_s: float = TERM_GRACE_S,
        kill_wait_s: float = KILL_WAIT_S) -> list[int]:
    """End the attempt's processes (:func:`find`): SIGTERM, SIGKILL after ``term_grace_s``,
    then wait up to ``kill_wait_s``. Returns the ones still alive (none when all ended)."""
    found = find(root, session)
    if not found:
        return []
    print(f"refactory: workspace: ending {len(found)} process(es) the timed-out attempt left "
          f"in {root}: {found}", flush=True)
    _signal(found, signal.SIGTERM)
    alive = _wait(found, term_grace_s)
    # Anything started while they were ending is the attempt's too.
    alive = sorted(set(alive) | set(find(root, session)))
    if alive:
        _signal(alive, signal.SIGKILL)
        alive = _wait(alive, kill_wait_s)
    return alive


def keeps_workspace(workdir: Path, node: str, *, retrying: bool, keep: bool,
                    session: int | None = None) -> bool:
    """Whether this dispatch of ``node`` keeps its workspace: a retry whose rule keeps it,
    once every process the timed-out attempt left there has ended. When one survives, the
    retry starts from a fresh workspace, with a warning naming it."""
    if not (retrying and keep):
        return False
    from refactory_dsh.workspace import ROOT, _slug

    root = Path(workdir) / ROOT / _slug(node)
    survivors = end(root, session)
    if survivors:
        print(f"refactory: workspace: warning: process(es) {survivors} from {node}'s "
              f"timed-out attempt are still running in {root}; the retry starts from a fresh "
              "workspace instead of keeping that attempt's edits", file=sys.stderr, flush=True)
        return False
    return True
