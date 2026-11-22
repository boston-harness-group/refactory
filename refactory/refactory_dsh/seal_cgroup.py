"""A sealed step's whole-step cap through a cgroup delegated to this process (Linux).

Where no systemd user scope can cap a sealed step (`seal.step_cap`: the run is a system
service, say, outside the person's sessions), the cap comes from the run's own cgroup
when systemd delegated it to the run (cgroup v2, a launch with ``-p Delegate=yes``): the
seal makes one child cgroup per sealed workspace, ``refactory-seal-<hash>``, holding
:func:`seal_reaper.cgroup_limits` (the scope's values, one source), and the step's
runner moves itself into it before it starts the seal (`seal_reaper.exec_runner`), so
the step's commands and everything they start count against it together.

cgroup v2 lets a cgroup turn on the memory controller for its children only while no
process sits in it (writing ``+memory`` to its ``cgroup.subtree_control`` fails with
EBUSY otherwise). So, once per delegated cgroup, under a lock on it, every process in it
moves into a leaf beside the steps' cgroups (:data:`LEAF`), and the controllers are then
turned on (:func:`ensure`). A process started from one in the leaf starts in the leaf,
and later workers and runs sharing the unit find the controllers on and change nothing.

Anything that keeps the cap from being made (no cgroup v2, a cgroup not delegated to
this user, no memory or pids controller, a process that cannot be moved, EBUSY) raises
:class:`Unavailable`, whose message says why; the step then runs with per-process
limits only, and its dispatch record says so. It never fails the step.

A step's cgroup is removed once every process sealed in its workspace has been killed
and reaped (`workspace.collect`, the end of its dispatch, the next prepare of the same
node, the run's end: `workspace.release_cgroup`). The kernel refuses to remove a cgroup
that still holds a process, so :func:`remove` waits for it to empty, and ends what is
left (``cgroup.kill``) only as a last resort.

The kernel operations go through :func:`_read`, :func:`_write`, :func:`_mkdir` and
:func:`_rmdir`, so a test can stand in for cgroupfs with a directory tree that keeps the
kernel's rules.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import time
from pathlib import Path

from refactory_dsh import seal_reaper

#: Where cgroup v2 is mounted.
ROOT = "/sys/fs/cgroup"
#: The leaf the delegated cgroup's own processes move into before the controllers are
#: turned on (`ensure`); a process found in it takes the cgroup above as the delegated one.
LEAF = "main"
#: A sealed workspace's cgroup: this, then a digest of the workspace's path (`step_name`).
PREFIX = "refactory-seal-"
#: The controllers a step's cgroup needs: memory (``memory.max``, ``memory.swap.max``) and
#: processes (``pids.max``).
CONTROLLERS = ("memory", "pids")
#: How many times the delegated cgroup's processes are moved into the leaf before turning
#: the controllers on is tried anyway (a process that forks while it moves leaves its
#: child behind, to be moved on the next round).
MOVE_ROUNDS = 50


class Unavailable(Exception):
    """No delegated cgroup can cap a sealed step here; the message says why."""


# ── the kernel's side, which a test stands in for ─────────────────────────────

def _read(path: Path) -> str:
    return Path(path).read_text()


def _write(path: Path, text: str) -> None:
    """One write to a cgroup file, as the kernel takes it: never created, never truncated."""
    fd = os.open(path, os.O_WRONLY | os.O_CLOEXEC)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)


def _mkdir(path: Path) -> None:
    os.mkdir(path)


def _rmdir(path: Path) -> None:
    os.rmdir(path)


def _proc_cgroup() -> str:
    try:
        return Path("/proc/self/cgroup").read_text()
    except OSError:
        return ""


# ── finding and preparing the delegated cgroup ───────────────────────────────

def _why(exc: OSError) -> str:
    return exc.strerror or str(exc)


def own_cgroup(groups: str | None = None) -> str:
    """This process's cgroup v2 path (``0::<path>`` in /proc/self/cgroup); "" without one."""
    for line in (_proc_cgroup() if groups is None else groups).splitlines():
        if line.startswith("0::"):
            return line[3:].strip()
    return ""


def _writable(path: Path) -> bool:
    return os.access(path, os.W_OK)


def delegated_base(groups: str | None = None) -> Path:
    """The cgroup delegated to this process's tree, where the steps' cgroups go: its own,
    or the one above when it already sits in :data:`LEAF`. Delegated means this user may
    make cgroups in it and move processes into them, and may not write the cgroup above
    it (so it is a tree's root someone handed over, never part of a tree this user's
    systemd manager keeps for itself). Raises :class:`Unavailable` naming what is missing."""
    path = own_cgroup(groups)
    if not path.startswith("/"):
        raise Unavailable("this process is in no cgroup v2 hierarchy")
    root = Path(ROOT)
    if not (root / "cgroup.controllers").exists():
        raise Unavailable(f"no cgroup v2 hierarchy is mounted at {ROOT}")
    base = root / path.lstrip("/")
    if base.name == LEAF and _delegated(base.parent):
        base = base.parent
    if not _delegated(base):
        raise Unavailable(f"this run's cgroup {base} is not delegated to it (launch the run "
                          "with -p Delegate=yes)")
    return base


def _delegated(base: Path) -> bool:
    if base == Path(ROOT) or not base.is_dir():
        return False
    return (_writable(base) and _writable(base / "cgroup.procs")
            and _writable(base / "cgroup.subtree_control") and not _writable(base.parent))


def ensure(groups: str | None = None) -> Path:
    """The delegated cgroup (`delegated_base`) with the memory and pids controllers turned
    on for its children: once, under a lock on it, its processes move into :data:`LEAF`
    and the controllers are turned on. Doing it again, from this process or another that
    shares the cgroup, finds them on and changes nothing. Raises :class:`Unavailable`."""
    base = delegated_base(groups)
    try:
        offered = _read(base / "cgroup.controllers").split()
    except OSError as exc:
        raise Unavailable(f"cannot read the controllers of {base}: {_why(exc)}") from None
    missing = [name for name in CONTROLLERS if name not in offered]
    if missing:
        raise Unavailable(f"the delegated cgroup {base} has no {' or '.join(missing)} controller")
    try:
        lock = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as exc:
        raise Unavailable(f"cannot open {base} to lock it: {_why(exc)}") from None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _enabled(base):
            return base
        leaf = base / LEAF
        try:
            _mkdir(leaf)
        except FileExistsError:
            pass
        except OSError as exc:
            raise Unavailable(f"cannot make the leaf {leaf}: {_why(exc)}") from None
        _move_out(base, leaf)
        try:
            _write(base / "cgroup.subtree_control", " ".join(f"+{name}" for name in CONTROLLERS))
        except OSError as exc:
            raise Unavailable(f"cannot turn on the {' and '.join(CONTROLLERS)} controllers in "
                              f"{base}: {_why(exc)}") from None
        if not _enabled(base):
            raise Unavailable(f"the {' and '.join(CONTROLLERS)} controllers did not turn on in {base}")
        return base
    finally:
        os.close(lock)


def _enabled(base: Path) -> bool:
    try:
        enabled = _read(base / "cgroup.subtree_control").split()
    except OSError:
        return False
    return all(name in enabled for name in CONTROLLERS)


def _move_out(base: Path, leaf: Path) -> None:
    """Move every process in ``base`` into ``leaf``, until none is left or the rounds end."""
    for _ in range(MOVE_ROUNDS):
        try:
            pids = _read(base / "cgroup.procs").split()
        except OSError as exc:
            raise Unavailable(f"cannot list the processes in {base}: {_why(exc)}") from None
        if not pids:
            return
        for pid in pids:
            try:
                _write(leaf / "cgroup.procs", pid)
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    continue  # gone since it was listed
                raise Unavailable(f"cannot move process {pid} from {base} into {leaf}: "
                                  f"{_why(exc)}") from None


# ── a sealed step's cgroup ────────────────────────────────────────────────────

def step_name(workspace: str | Path) -> str:
    """The cgroup a sealed workspace's commands run in: one per workspace path."""
    digest = hashlib.sha256(os.path.realpath(workspace).encode()).hexdigest()[:16]
    return PREFIX + digest


def make(workspace: str | Path, groups: str | None = None) -> tuple[str, str]:
    """Make ``workspace``'s cgroup in the delegated cgroup with :func:`seal_reaper.cgroup_limits`;
    its path, and the limits it holds as its dispatch record says them. A leftover of an
    earlier dispatch of the same workspace (a runner that died) has its processes ended
    (`seal_reaper.reap`) and is removed first. Where the kernel keeps no swap account
    (no ``memory.swap.max``) swap is not capped, and the record says so. Raises
    :class:`Unavailable`."""
    base = ensure(groups)
    path = base / step_name(workspace)
    if path.exists():
        seal_reaper.reap(str(workspace))
        left = remove(str(path))
        if left:
            raise Unavailable(left)
    try:
        _mkdir(path)
    except OSError as exc:
        raise Unavailable(f"cannot make {path}: {_why(exc)}") from None
    held = []
    try:
        for name, value in seal_reaper.cgroup_limits():
            if name == "memory.swap.max" and not (path / name).exists():
                held.append(f"{name} not kept by this kernel")
                continue
            _write(path / name, value)
            if _read(path / name).strip() != value:
                raise Unavailable(f"{path / name} did not take {value}")
            held.append(f"{name}={value}")
    except (OSError, Unavailable) as exc:
        remove(str(path))
        why = _why(exc) if isinstance(exc, OSError) else str(exc)
        raise Unavailable(f"cannot cap {path}: {why}") from None
    return str(path), ", ".join(held)


def _populated(path: Path) -> bool:
    try:
        for line in _read(path / "cgroup.events").splitlines():
            if line.startswith("populated "):
                return line.split()[1] != "0"
    except OSError:
        pass
    return True


def remove(path: str, wait_s: float = 5.0) -> str:
    """Remove a sealed step's cgroup, once every process sealed in its workspace has been
    killed and reaped (`seal_reaper.reap`); "" when it is gone, else why it is left. A
    process that is still dying is waited for; one still there after half of ``wait_s``
    is ended through ``cgroup.kill``. Only a step's cgroup is ever removed (:data:`PREFIX`)."""
    if not path:
        return ""
    target = Path(path)
    if not target.name.startswith(PREFIX):
        return f"{path} is not a sealed step's cgroup, so it was left"
    deadline = time.monotonic() + wait_s
    killed = False
    while True:
        try:
            _rmdir(target)
            return ""
        except FileNotFoundError:
            return ""
        except OSError as exc:
            if exc.errno != errno.EBUSY:
                return f"cannot remove {path}: {_why(exc)}"
            now = time.monotonic()
            if not killed and now > deadline - wait_s / 2 and _populated(target):
                killed = True
                try:
                    _write(target / "cgroup.kill", "1")
                except OSError:
                    pass
            if now > deadline:
                return f"cannot remove {path}: {_why(exc)}"
        time.sleep(0.01)
