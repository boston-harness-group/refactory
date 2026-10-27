"""Canonical state classification for a graph run's supervision decisions.

Pure functions that read spine-status.json and spine-checkpoint.json to decide
whether a run is terminal, deliberately paused, or needs automatic supervision.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

TERMINAL = frozenset({'done', 'cancelled', 'failed'})
DELIBERATELY_PAUSED = frozenset({'halted', 'paused', 'waiting_user', 'refused'})


def is_terminal(status: str) -> bool:
    """True when the spine status is done, cancelled, or failed."""
    return status in TERMINAL


def is_deliberately_paused(status: str) -> bool:
    """True when the spine status is halted, paused, waiting_user, or refused."""
    return status in DELIBERATELY_PAUSED


def is_alive(pid: int) -> bool:
    """Whether the process is running.

    ``kill(pid, 0)`` succeeds for a zombie, and a run this process spawned and
    never waited on stays a zombie until it is reaped -- so a finished run would
    read as alive from the very process that started it.
    """
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass  # not our child: the signal probe below is the only answer
    except OSError:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


#: The status file both runtimes write.
STATUS = '.refactory/spine-status.json'


def write_status(root: Path, *, status: str, nodes: list[str], running_nodes: list[dict[str, str]],
                 completed: list[str], error: str | None = None, wait_reason: str | None = None,
                 last_failure: dict[str, str] | None = None, runtime: str = 'walker',
                 owner_pid: int | None = None) -> None:
    """Write the status file in the spine's format (`recordStatus` in workflow-spine).

    The walker writes it through here, so a walked run has the record every reader of a
    run directory reads: where the run stands, what is due, and what completed. Whole
    and atomic, so no reader sees half of it. Like the spine's, it names the runtime and
    the process that owns the run (:func:`walked`, :func:`owner_pid`).
    """
    from datetime import datetime, timezone

    at = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    record: dict[str, Any] = {
        'schema_version': 1, 'run_id': str(Path(root).resolve()), 'runtime': runtime,
        'owner_pid': os.getpid() if owner_pid is None else owner_pid, 'at': at, 'status': status,
        'nodes': list(nodes), 'running_nodes': list(running_nodes), 'completed': list(completed),
    }
    if error is not None:
        record['error'] = error
    if wait_reason is not None:
        record['wait_reason'] = wait_reason
    if last_failure is not None:
        record['last_failure'] = {**last_failure, 'at': at}
    path = Path(root) / STATUS
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    pending.write_text(json.dumps(record, indent=2))
    pending.replace(path)


def _status_record(root: Path) -> dict[str, Any]:
    try:
        data = json.loads((Path(root) / STATUS).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def walked(root: Path) -> bool:
    """Whether the walker drives this run, as its status file says (`runtime`). Whether
    a walked run is still alive is its progress's and its walking process's to say, not
    a spine checkpoint's -- and a status file from before it named its runtime is the
    spine's, the only runtime that wrote one then."""
    return _status_record(root).get('runtime') == 'walker'


def owner_pid(root: Path) -> int | None:
    """The process the status file names as the run's owner, when it names one."""
    pid = _status_record(root).get('owner_pid')
    return pid if isinstance(pid, int) and not isinstance(pid, bool) else None


def spine_status(root: Path) -> str:
    """Read and return the status string from spine-status.json."""
    path = Path(root) / '.refactory/spine-status.json'
    try:
        data = json.loads(path.read_text())
        return str(data.get('status', '')) if isinstance(data, dict) else ''
    except (OSError, json.JSONDecodeError, ValueError):
        return ''


def needs_supervision(root: Path, *, since: float | None = None) -> bool:
    """True when the run needs automatic supervision (owner dead, not terminal/paused).

    When *since* is given (epoch seconds from ``time.time()``), a
    ``stop-requested.json`` whose ``at`` is at or after *since* suppresses
    supervision — the stop was deliberate for this owner.  An older stop
    (from a previous owner) does not block a resumed run's supervision.
    """
    root = Path(root)
    status = spine_status(root)
    if not status:
        return False
    if is_terminal(status):
        return False
    if is_deliberately_paused(status):
        return False
    # A stop requested since the current owner started is deliberate.
    if since is not None:
        stop_path = root / '.refactory/stop-requested.json'
        if stop_path.exists():
            try:
                stop_data = json.loads(stop_path.read_text())
                stop_at = stop_data.get('at')
                if isinstance(stop_at, (int, float)) and stop_at >= since:
                    return False
            except (OSError, json.JSONDecodeError, ValueError):
                pass
    # Check owner liveness
    try:
        checkpoint = json.loads((root / '.refactory/spine-checkpoint.json').read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    pid = checkpoint.get('owner_pid')
    if not isinstance(pid, int):
        return False
    if is_alive(pid):
        return False
    return True


def read_start_failure(root: Path) -> dict | None:
    """Read .refactory/start-failure.json if present."""
    path = Path(root) / '.refactory/start-failure.json'
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def promote_unretried_failure(root: Path) -> None:
    """Promote ready+last_failure to run-level failed when the owner exits.

    dm-95: when the owner exits and the spine's last boundary was a node
    failure that was not retried (status ``ready`` with ``last_failure``),
    the run has ended.  Record it as ``failed`` with the node failure's
    error so the supervisor does not respawn it and ``refactory status``
    reads as failed.  The promotion updates both ``spine-status.json``
    and ``spine-checkpoint.json``.
    """
    status_path = Path(root) / '.refactory/spine-status.json'
    checkpoint_path = Path(root) / '.refactory/spine-checkpoint.json'
    try:
        status_data = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(status_data, dict) or status_data.get('status') != 'ready':
        return
    last_failure = status_data.get('last_failure')
    if not isinstance(last_failure, dict):
        return
    node = last_failure.get('node', 'unknown')
    error = last_failure.get('error', 'node failure')
    status_data['status'] = 'failed'
    status_data['error'] = f'{node}: {error}'
    try:
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status_data, indent=2))
        tmp.replace(status_path)
    except OSError:
        return
    try:
        cp_data = json.loads(checkpoint_path.read_text())
        if isinstance(cp_data, dict):
            cp_data['status'] = 'failed'
            tmp = checkpoint_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(cp_data))
            tmp.replace(checkpoint_path)
    except (OSError, json.JSONDecodeError):
        pass


def read_supervision_history(root: Path) -> list[dict]:
    """Read .refactory/supervision-history.json if present."""
    path = Path(root) / '.refactory/supervision-history.json'
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError, ValueError):
        return []
