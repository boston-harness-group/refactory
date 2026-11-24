"""What a watched run counts as work, and how a stopped one ends its node commands.

A smoke's run is stopped after a stretch with no sign of work (`sandbox.smoke`,
`dsh_runtime.run_on_dsh` with ``inactivity``). The run's directory can hold a whole
checkout, so the signs are a fixed set of files each kind of work writes, never a walk
of the directory (`signs`):

- the run's own state: the spine's status, checkpoint and event log, the walker's
  progress record, the facts;
- the watch mark (``MARK``), which every node command of a watched run writes as it
  prints (the node runner's output tee, `refactory_dsh.run`) and as it writes a
  declared file (`nn.RunContext`);
- each worker's activity file, which DSH's worker observer touches on every streamed
  frame (``workers/<receipt>.activity``), and the rest of its receipt;
- each item's pass: its marker, record and completion, and the same files again for
  the pass's own workers and nested passes.

A watched run names its watch directory to its node commands in ``WATCH_ENV``. Each
node command there also records its process group (``PROCESSES``), so a stopped run
ends its node commands however the runtime spawned them (`end_recorded`): DSH spawns
each in a process group, or a systemd scope, of its own, which the run's own group
does not reach.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

#: Set for a watched run's node commands: the run's watch directory (`watch_dir`).
WATCH_ENV = "REFACTORY_RUN_WATCH"
#: Where a run keeps its watch directory, relative to the run directory.
WATCH_DIR = ".refactory/watch"
#: The mark node commands write as they work, in the watch directory.
MARK = "activity"
#: The process records node commands leave, in the watch directory.
PROCESSES = "processes"
#: The least time between two marks from one process.
MARK_INTERVAL_S = 0.2
#: How long recorded process groups get to exit once asked, before they are killed.
GROUP_STOP_GRACE_S = 2.0

#: The run's own state files, relative to the run directory (or a pass directory).
_STATE = (
    ".refactory/events.jsonl", ".refactory/spine-status.json",
    ".refactory/spine-checkpoint.json", ".refactory/progress.json", "facts.json",
)
#: What a pass directory keeps of its item (`items.PASS_MARKER`, `items.PASS_RECORD`).
_PASS = (".refactory/pass.json", ".item_record.json", ".done", "current_item.json")

_last_mark = [0.0]


def watch_dir(run_dir: Path) -> Path:
    return Path(run_dir) / WATCH_DIR


def mark(directory: str | os.PathLike[str] | None = None) -> None:
    """Note work in a watch directory (``WATCH_ENV`` when not given); at most every
    ``MARK_INTERVAL_S`` per process, and nothing at all when there is none."""
    target = directory if directory is not None else os.environ.get(WATCH_ENV)
    if not target:
        return
    now = time.monotonic()
    if now - _last_mark[0] < MARK_INTERVAL_S:
        return
    _last_mark[0] = now
    try:
        path = Path(target) / MARK
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(time.time_ns()))
    except OSError:
        pass


def _stat(path: str, ignored: set[str], out: list[tuple[str, int, int]]) -> None:
    if path in ignored:
        return
    try:
        stat = os.stat(path)
    except OSError:
        return
    out.append((path, stat.st_mtime_ns, stat.st_size))


def _entries(directory: str) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(directory) as found:
            return list(found)
    except OSError:
        return []


def _node_dir(directory: str, ignored: set[str], out: list[tuple[str, int, int]],
              depth: int) -> None:
    for name in _STATE:
        _stat(os.path.join(directory, name), ignored, out)
    for entry in _entries(os.path.join(directory, ".refactory", "workers")):
        if entry.is_file(follow_symlinks=False):
            _stat(entry.path, ignored, out)
    if depth <= 0:
        return
    for node in _entries(os.path.join(directory, "passes")):
        if not node.is_dir(follow_symlinks=False):
            continue
        for item in _entries(node.path):
            if not item.is_dir(follow_symlinks=False):
                continue
            for name in _PASS:
                _stat(os.path.join(item.path, name), ignored, out)
            _node_dir(item.path, ignored, out, depth - 1)


class Watch:
    """Repeated looks at one run's signs of work, with its paths resolved once.

    Each look reads the fixed set of files `signs` describes: no path resolution, no
    walk. The cost depends on the number of items and workers, never on the files a
    node leaves in the run directory (a checkout, a build). ``ignore`` names files that
    are not the run's work -- the watcher's own mark, the budget ledger. Passes are
    followed ``nested_depth`` levels of nesting down.
    """

    def __init__(self, run_dir: Path, *, ignore: tuple[Path | str | None, ...] = (),
                 nested_depth: int = 4) -> None:
        self.root = str(Path(run_dir).resolve())
        self.ignored = {str(Path(path).resolve()) for path in ignore if path}
        self.nested_depth = nested_depth

    def look(self) -> tuple[tuple[str, int, int], ...]:
        """The run's signs of work, as (path, change time, size) for each that exists.
        Two looks that differ mean the run worked in between."""
        out: list[tuple[str, int, int]] = []
        _stat(os.path.join(self.root, WATCH_DIR, MARK), self.ignored, out)
        _node_dir(self.root, self.ignored, out, self.nested_depth)
        return tuple(out)


def signs(run_dir: Path, *, ignore: tuple[Path | str | None, ...] = (),
          nested_depth: int = 4) -> tuple[tuple[str, int, int], ...]:
    """One look at a run's signs of work (`Watch.look`)."""
    return Watch(run_dir, ignore=ignore, nested_depth=nested_depth).look()


def look_interval(limit_s: float, floor_s: float) -> float:
    """How often a watcher looks, for an inactivity limit of ``limit_s``: a twentieth of
    the limit, at least ``floor_s`` and at most 10 s. A one-hour limit is then checked
    every 10 s rather than twice a second, and a stop comes at most that late."""
    return min(max(limit_s / 20.0, floor_s), 10.0)


# ── the node commands a run started ─────────────────────────────────────────

def start_time(pid: int) -> str | None:
    """When process ``pid`` started, as the system reports it, or None when it is gone.

    What tells a process from a later one that reused its pid."""
    if sys.platform.startswith("linux"):
        try:
            raw = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return None
        fields = raw.rsplit(")", 1)[-1].split()
        return fields[19] if len(fields) > 19 else None
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout.strip()
    return text or None


def record_process(directory: str | os.PathLike[str] | None = None, *,
                   pid: int | None = None) -> Path | None:
    """Record a process -- this one, or ``pid`` -- and its process group in a watch
    directory (``WATCH_ENV`` when not given), for `end_recorded`; None when there is
    none or the process cannot be read."""
    target = directory if directory is not None else os.environ.get(WATCH_ENV)
    if not target or os.name != "posix":
        return None
    try:
        pid = os.getpid() if pid is None else int(pid)
        group = os.getpgid(pid)
    except (OSError, TypeError, ValueError):
        return None
    record = {"pid": pid, "start": start_time(pid), "group": group,
              "group_start": start_time(group) if group != pid else None}
    path = Path(target) / PROCESSES / f"{pid}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(record))
        os.replace(pending, path)
    except OSError:
        return None
    return path


def forget_process(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink()
        except OSError:
            pass


def _recorded_groups(directory: Path) -> list[int]:
    """The process groups recorded node commands ran in that still hold the processes
    they recorded: a group whose leader's pid now names another process, or whose
    recorded process is another process, is not one of the run's."""
    own = os.getpgrp()
    groups: list[int] = []
    for path in sorted((Path(directory) / PROCESSES).glob("*.json")):
        try:
            record: dict[str, Any] = json.loads(path.read_text())
            pid, group = int(record["pid"]), int(record["group"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if group in (own, 0, 1) or group in groups:
            continue
        now = start_time(pid)
        if now is not None and now != record.get("start"):
            continue  # the pid names another process now
        if group != pid:
            leader = start_time(group)
            if leader is not None and leader != record.get("group_start"):
                continue  # the group's id names another process's group now
        groups.append(group)
    return groups


def end_groups(groups: list[int]) -> None:
    """Ask each process group to stop, give them ``GROUP_STOP_GRACE_S``, then kill what
    is left. Never this process's own group."""
    if os.name != "posix":
        return
    own = os.getpgrp()

    def _signal(group: int, sig: int) -> bool:
        try:
            os.killpg(group, sig)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    live = [group for group in groups
            if group not in (own, 0, 1) and _signal(group, signal.SIGTERM)]
    deadline = time.monotonic() + GROUP_STOP_GRACE_S
    while live and time.monotonic() < deadline:
        live = [group for group in live if _signal(group, 0)]
        if live:
            time.sleep(0.05)
    for group in live:
        _signal(group, signal.SIGKILL)


def end_recorded(directory: Path | None) -> list[int]:
    """End the process groups a stopped run's node commands recorded in ``directory``
    (`end_groups`). Returns the groups signalled."""
    if directory is None or os.name != "posix":
        return []
    groups = _recorded_groups(Path(directory))
    end_groups(groups)
    return groups


# ── a node command's output, marked as it is written ───────────────────────

#: The tee a watched node command's output goes through (`tee_output`): a process of its
#: own, so it drains every byte even when the node ends abruptly (``os._exit``, a kill),
#: and it outlives the node only as long as something still holds its pipes.
_TEE = r"""
import os, select, sys, time
mark, streams = sys.argv[1], {int(sys.argv[2]): 1, int(sys.argv[3]): 2}
last, dead = 0.0, set()
while streams:
    try:
        ready = select.select(list(streams), [], [])[0]
    except InterruptedError:
        continue
    for fd in ready:
        try:
            chunk = os.read(fd, 65536)
        except InterruptedError:
            continue
        except OSError:
            chunk = b""
        if not chunk:
            os.close(fd)
            del streams[fd]
            continue
        out = streams[fd]
        view = memoryview(chunk)
        while view and out not in dead:
            try:
                view = view[os.write(out, view):]
            except InterruptedError:
                continue
            except OSError:
                dead.add(out)
        now = time.monotonic()
        if now - last >= __INTERVAL__:
            last = now
            try:
                with open(mark, "w") as handle:
                    handle.write(str(time.time_ns()))
            except OSError:
                pass
""".replace("__INTERVAL__", repr(MARK_INTERVAL_S))


def tee_output(directory: str | os.PathLike[str] | None = None) -> bool:
    """Send this process's stdout and stderr through a tee that marks the watch
    directory (``WATCH_ENV`` when not given) as output arrives; do nothing without one.

    At the descriptor level, so what the node's own subprocesses print counts too.
    Every byte reaches where it went before, unchanged and in order per stream, and
    nothing waits for the tee: it ends when the last holder of its pipes does, as the
    original streams would have. Returns whether the output now goes through a tee.
    """
    target = directory if directory is not None else os.environ.get(WATCH_ENV)
    if not target or os.name != "posix":
        return False
    global tee_pid
    sys.stdout.flush()
    sys.stderr.flush()
    out_read, out_write = os.pipe()
    err_read, err_write = os.pipe()
    try:
        Path(target).mkdir(parents=True, exist_ok=True)
        tee = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _TEE, str(Path(target) / MARK),
             str(out_read), str(err_read)],
            pass_fds=(out_read, err_read), stdin=subprocess.DEVNULL,
        )
    except OSError:
        for fd in (out_read, out_write, err_read, err_write):
            os.close(fd)
        return False
    tee_pid = tee.pid
    os.close(out_read)
    os.close(err_read)
    originals = (os.dup(1), os.dup(2))
    os.dup2(out_write, 1)
    os.dup2(err_write, 2)
    os.close(out_write)
    os.close(err_write)
    # Should the tee die, the node's own output -- its verdict included -- still reaches
    # where it went before: the first write that finds the tee gone puts the original
    # streams back and is made again.
    sys.stdout = _Fallback(sys.stdout, 1, originals[0])  # type: ignore[assignment]
    sys.stderr = _Fallback(sys.stderr, 2, originals[1])  # type: ignore[assignment]
    return True


#: The pid of this process's tee (`tee_output`), or None.
tee_pid: int | None = None


class _Fallback:
    """A text stream that goes back to the original descriptor once the tee is gone."""

    def __init__(self, stream: Any, fd: int, original: int) -> None:
        self._stream, self._fd, self._original = stream, fd, original

    def _restore(self) -> None:
        os.dup2(self._original, self._fd)

    def write(self, text: str) -> int:
        try:
            return self._stream.write(text)
        except BrokenPipeError:
            self._restore()
            self._stream.flush()
            return self._stream.write(text)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except BrokenPipeError:
            self._restore()
            self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)
