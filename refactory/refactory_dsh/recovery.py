"""Explicit recovery of a stopped graph boundary, never of unknown side effects."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def fingerprint(path: Path) -> Any:
    if not path.exists():
        return None
    if path.is_dir():
        return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(path.rglob('*')) if p.is_file() and not p.is_symlink()}
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configuration_fingerprint(path: Path) -> Any:
    """Ignore interpreter caches, but retain executable source and package data."""
    if not path.is_dir():
        return fingerprint(path)
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob('*')) if p.is_file() and not p.is_symlink()
            and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}


def configuration_fingerprints(root: Path, identifier: str, project: Path) -> dict[str, Any]:
    """Pin the prepared graph, declarations and imported mode implementation."""
    import importlib.util
    import inspect
    from refactory_dsh import lowering, registry
    package = registry.where(identifier.split('/', 1)[0], project)
    if package is None:
        raise ValueError('cannot locate the source of the prepared mode')
    manifest, mode = registry.resolve(project, identifier)
    paths = [Path(root) / 'graphs', Path(root) / lowering.DECLARATIONS, package]
    source = inspect.getsourcefile(mode)
    if source:
        paths.append(Path(source))
    # Packages may name library ops instead of containing their implementation
    # (the default fast modes do). Pin those executable modules as well.
    for module in manifest.ops:
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin or not Path(spec.origin).is_file():
            raise ValueError(f'cannot fingerprint declared operation module {module!r}')
        paths.append(Path(spec.origin))
    return {str(path.resolve()): _configuration_fingerprint(path) for path in paths}


#: The checkpoint statuses a resume goes on from without an acknowledgement.
RESUMABLE = frozenset({'ready', 'paused', 'halted', 'done', 'cancelled', 'waiting_user', 'refused'})
#: Where a resume's acknowledgement is kept, bound to the checkpoint it acknowledges.
RESUME_REQUEST = '.refactory/resume-request.json'


def _acknowledged(root: Path, payload: bytes, key: str) -> bool:
    """Whether ``resume-request.json`` acknowledges ``key`` for exactly this checkpoint.

    Bound to the checkpoint's bytes, so a stale request left behind by a crashed owner
    cannot let a later plain resume of a failed or unknown boundary through.
    """
    request = Path(root) / RESUME_REQUEST
    if not request.is_file():
        return False
    try:
        existing = json.loads(request.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return (isinstance(existing, dict) and existing.get(key) is True
            and existing.get('checkpoint_sha256') == hashlib.sha256(payload).hexdigest())


def resume_status_check(status: Any, *, acknowledge_failed: bool, acknowledge_executing: bool,
                        acknowledge_command: str, failed_command: str) -> None:
    """Refuse to resume from ``status`` unless it is a known stopped boundary.

    The one check both runtimes make, with one allowed set and one set of messages: a
    failed boundary needs ``acknowledge_failed``, and a boundary whose last outcome is
    unknown (``executing``) needs ``acknowledge_executing``. The two commands are how
    the runtime that owns the run gives them (dm-160).
    """
    allowed = set(RESUMABLE)
    if acknowledge_failed:
        allowed.add('failed')
    if status == 'executing':
        if not acknowledge_executing:
            raise ValueError(
                'last node outcome is unknown; inspect side effects, then use:\n'
                f'  {acknowledge_command}')
        # With a valid acknowledgement, treat as halted for recovery purposes
        allowed.add('executing')
    if status not in allowed:
        if status == 'failed':
            raise ValueError(
                'checkpoint status is failed; to resume after inspecting side effects, use:\n'
                f'  {failed_command}')
        raise ValueError(
            f'checkpoint status is {status!r}; inspect side effects before any explicit rerun')


def _pending_markers(root: Path) -> list[Path]:
    pending = Path(root) / '.refactory/agent-delivery/pending'
    return sorted(path for path in pending.glob('*.json') if path.is_file()) if pending.is_dir() else []


def _require_settled(root: Path, saved: dict[str, Any], *, acknowledge_command: str) -> None:
    """No dispatched worker's outcome is unknown, and no declared input or output changed
    since the checkpoint recorded it."""
    if _pending_markers(root):
        raise ValueError(
            'a dispatched worker outcome is unknown; set it aside first with:\n'
            f'  {acknowledge_command}')
    for name, expected in (saved.get('file_fingerprints') or {}).items():
        if fingerprint(Path(root) / name) != expected:
            raise ValueError(f'input/output {name!r} changed; dependent results need explicit invalidation')


def check_configuration(root: Path, config: dict[str, Any], *, skip_driver_paths: str | None = None) -> None:
    """Refuse a resume whose prepared configuration -- graph, declarations, the mode's
    source and its operation modules -- changed since the run was prepared."""
    root = Path(root).resolve()
    expected_configuration = config.get('configuration_fingerprints')
    if not isinstance(expected_configuration, dict) or not expected_configuration:
        raise ValueError('saved run has no configuration fingerprints; automatic recovery cannot verify its source')
    # Translate fingerprint paths from recorded checkout to current REPO_ROOT
    # so that a re-exec'd process compares against the pin's files.
    from refactory_dsh.paths import REPO_ROOT
    driver = config.get('driver', {})
    driver_checkout = driver.get('checkout', '')
    repo_root_str = str(REPO_ROOT)
    # When skip_driver_paths is set, driver fingerprints (those whose
    # resolved path is under the driver checkout) are skipped — they are
    # handled by the pin.  Non-driver fingerprints (the run's own
    # graphs, project packages) are still checked so that "prepared
    # configuration … changed" is raised as on main.
    _driver_co_resolved = Path(skip_driver_paths).resolve() if skip_driver_paths else None
    _run_root_resolved = root
    for fp_name, expected in expected_configuration.items():
        actual_name = fp_name
        fp_resolved = Path(fp_name).resolve()
        # Decide if this fingerprint is a driver path.
        _is_driver = False
        if _driver_co_resolved is not None and fp_resolved.is_relative_to(_driver_co_resolved):
            # Paths under the run's own directory are per-run artefacts
            # even when the run is inside the checkout.
            if not fp_resolved.is_relative_to(_run_root_resolved):
                _rel = fp_resolved.relative_to(_driver_co_resolved)
                if not any(part == ".factory" for part in _rel.parts):
                    _is_driver = True
        if _is_driver:
            continue  # skipped: driver fingerprint handled by the pin
        # Path translation for re-exec'd process comparing against the pin.
        # Only translate true driver paths: under the driver checkout by
        # path components and outside the run's own directory and any
        # ``.factory/`` run tree.  A run inside the checkout (the default
        # layout) has its graphs/ here; translating them into the pin
        # would look for files that do not exist there.
        if driver_checkout and fp_name.startswith(driver_checkout) and driver_checkout != repo_root_str:
            _should_translate = True
            if fp_resolved.is_relative_to(_run_root_resolved):
                _should_translate = False
            else:
                try:
                    _trel = fp_resolved.relative_to(Path(driver_checkout).resolve())
                    if any(part == ".factory" for part in _trel.parts):
                        _should_translate = False
                except ValueError:
                    pass
            if _should_translate:
                actual_name = repo_root_str + fp_name[len(driver_checkout):]
        if _configuration_fingerprint(Path(actual_name)) != expected:
            raise ValueError(f'prepared configuration {fp_name!r} changed; dependent results need explicit invalidation')


def prepare(root: Path, name: str, *, resource_limits: dict[str, Any] | None = None,
            acknowledge_failed: bool = False, skip_configuration: bool = False,
            skip_driver_paths: str | None = None) -> Path:
    root = Path(root).resolve()
    checkpoint = root / '.refactory/spine-checkpoint.json'
    payload = checkpoint.read_bytes()
    saved = json.loads(payload)
    if saved.get('run_id') != str(root.resolve()) or saved.get('graph', {}).get('name') != name:
        raise ValueError('checkpoint belongs to a different run or mode')
    pid = saved.get('owner_pid')
    if pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('the graph owner is still alive; use its boundary resume control')
    # When the parent's start() already wrote resume-request.json with
    # acknowledge_failed: true, the child inherits the acknowledgement
    # from the file rather than requiring it as a call argument — but only
    # when the file's checkpoint_sha256 matches the current checkpoint bytes,
    # so a stale request left behind by a crashed child cannot let a later
    # plain resume of a failed run through.
    acknowledge_failed = acknowledge_failed or _acknowledged(root, payload, 'acknowledge_failed')
    # An executing checkpoint is allowed only when a matching acknowledgement exists
    # in the resume-request.json file (written by graph_control.acknowledge).
    resume_status_check(saved.get('status'), acknowledge_failed=acknowledge_failed,
                        acknowledge_executing=_acknowledged(root, payload, 'acknowledge_executing'),
                        acknowledge_command=f'refactory graph-control acknowledge --run {root}',
                        failed_command=f'refactory graph-control resume --acknowledge-failure --run {root}')
    _require_settled(root, saved, acknowledge_command=f'refactory graph-control acknowledge --run {root}')
    if not skip_configuration:
        config = json.loads((root / '.refactory/run-config.json').read_text())
        check_configuration(root, config, skip_driver_paths=skip_driver_paths)
    if resource_limits:
        from refactory_dsh.budgets import ResourceLimits, RunBudget
        RunBudget.from_path(root / '.refactory/resource-usage.json').resume(ResourceLimits.from_dict(resource_limits))
    request = root / '.refactory/resume-request.json'
    temporary = request.with_suffix('.tmp')
    request_data: dict[str, Any] = {'checkpoint_sha256': hashlib.sha256(payload).hexdigest()}
    if acknowledge_failed:
        request_data['acknowledge_failed'] = True
    # Carry the acknowledge_executing flag through so the spine's
    # restoreRequested() also accepts the executing checkpoint.
    if saved.get('status') == 'executing':
        request_data['acknowledge_executing'] = True
    temporary.write_text(json.dumps(request_data))
    temporary.replace(request)
    return root / 'graphs'


def start(root: Path, *, acknowledge_failed: bool = False) -> dict[str, Any]:
    """Resume a known stopped boundary via a detached supervisor process.

    The supervisor (``python -m refactory_dsh.supervisor <root>``) starts the
    owner child, waits on it, and respawns transient exits automatically.  One
    supervisor per run, held by a file lock.
    """
    import subprocess
    import sys
    from refactory_dsh.run_ownership import ownership
    root = Path(root).resolve()
    with ownership(root):
        config = json.loads((root / '.refactory/run-config.json').read_text())
        # Detect drift BEFORE prepare() so a moved driver creates a pin
        # instead of prepare() refusing with "configuration changed".
        from refactory_dsh import pin as _pin
        _drift = _pin.detect_drift(config)
        python_exe = sys.executable
        if _drift is not None:
            _pin_checkout = _pin.resolve_pin(config['driver'], run_dir=root)
            python_exe = str(_pin.pin_python(_pin_checkout))
        # Validate while owning the directory, then release before the child
        # acquires its own nonblocking lock. Otherwise a fast child can reject
        # its parent as the still-active owner. The child repeats validation;
        # an intervening invocation can refuse this queued request safely.
        # When driver drift was detected, skip only the driver's fingerprints
        # so that non-driver changes (run graphs, project packages) still
        # raise the "prepared configuration … changed" refusal.
        _skip_drv = config.get('driver', {}).get('checkout') if _drift is not None else None
        prepare(root, config['mode'].replace('/', '-'),
                acknowledge_failed=acknowledge_failed,
                skip_driver_paths=_skip_drv)
    supervisor_log = root / '.refactory/supervisor.log'
    supervisor_log.parent.mkdir(parents=True, exist_ok=True)
    with supervisor_log.open('ab') as output:
        child = subprocess.Popen([python_exe, '-m', 'refactory_dsh.supervisor', str(root)],
            cwd=config['project'],
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True)
    # Record the supervisor's PID in the console's launch record so that
    # console.stop finds and signals it exactly as it does a launched run's.
    # The supervisor forwards signals to its child.
    from refactory_dsh.invocations import CONSOLE_FILE
    console_path = root / CONSOLE_FILE
    try:
        existing = json.loads(console_path.read_text()) if console_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        existing = {}
    if isinstance(existing, dict):
        existing['pid'] = child.pid
        console_path.write_text(json.dumps(existing, indent=2))
    return {'status': 'queued', 'action': 'resume', 'run_id': str(root), 'pid': child.pid,
            'message': 'Recovery requested for the saved boundary; inspect live state for confirmation.'}


# ── a walked run ───────────────────────────────────────────────────────────

#: The walker's progress record: its checkpoint.
PROGRESS = '.refactory/progress.json'
#: Each walker outcome as the spine's checkpoint status for the same boundary (dm-160).
#: ``unknown`` is a node the walker saw time out mid-call; like ``running`` (a walker
#: that died mid-node), nothing says what that node did.
WALKER_STATUSES = {'pending': 'ready', 'completed': 'done', 'failed': 'failed', 'running': 'executing',
                   'unknown': 'executing', 'refused': 'refused', 'halted': 'halted'}


def walker_status(saved: dict[str, Any]) -> str:
    """The spine's name for the boundary a walker progress record describes."""
    outcome = str(saved.get('outcome') or '')
    return WALKER_STATUSES.get(outcome, outcome)


def _walker_acknowledge_command(config: dict[str, Any], root: Path) -> str:
    """The walked run's own resume, as its main recorded it, acknowledging the boundary."""
    resume = str(config.get('resume_command') or f'<the main that started it> --resume {root}')
    return f'{resume} --acknowledge-failure'


def set_aside(root: Path, *, fingerprints: dict[str, Any], markers: list[Path], status: Any,
              owner_pid: Any, controls: list[Path] = ()) -> tuple[Path, list[dict[str, Any]]]:
    """Move what a stopped boundary cannot vouch for under ``.refactory/set-aside/<time>/``.

    Pending delivery markers, ``controls`` (queued controls the stopped run no longer
    needs), and outputs whose fingerprint differs from the checkpoint's: the file system
    is put back in the state the fingerprints describe, so the resume's fingerprint check
    sees a consistent run. A manifest says what moved where. Both runtimes' acknowledge
    use this (`graph_control.acknowledge`, :func:`acknowledge_walked`).
    """
    import shutil
    from datetime import datetime, timezone

    root = Path(root)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = root / '.refactory/set-aside' / timestamp
    directory.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for kind, paths in (('pending', markers), ('graph-controls', controls)):
        for path in paths:
            dest = directory / kind / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            path.rename(dest)
            files.append({'original_path': str(path.relative_to(root)),
                          'set_aside_path': str(dest.relative_to(root)), 'size': dest.stat().st_size})
    # Completed fork branches' outputs match the recorded fingerprints and stay in
    # place; in-flight branches' partial outputs differ and are set aside.
    for name, expected in fingerprints.items():
        if name == '.refactory/run-config.json':
            # The run's configuration is its identity, not an output: a changed one is
            # refused by name when the run is resumed (`prepare`, `prepare_walked`), so
            # setting it aside here would strand the run and hide the refusal's reason.
            continue
        output = root / name
        if not output.exists() or fingerprint(output) == expected:
            continue
        dest = directory / 'outputs' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if output.is_dir():
            shutil.move(str(output), str(dest))
        else:
            output.rename(dest)
        files.append({'original_path': name, 'set_aside_path': str(dest.relative_to(root)),
                      'size': dest.stat().st_size if dest.is_file() else -1})
    (directory / 'manifest.json').write_text(json.dumps({
        'at': datetime.now(timezone.utc).isoformat(), 'checkpoint_status_before': status,
        'owner_pid': owner_pid, 'files': files}, indent=2))
    return directory, files


def _walked_config(root: Path, mode_id: str) -> dict[str, Any]:
    """The walked run's ``run-config.json``, refused when it is missing or belongs to
    another mode: nothing about a run is acknowledged or resumed by the wrong main."""
    config_path = Path(root) / '.refactory/run-config.json'
    try:
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        config = {}
    if not isinstance(config, dict) or not config:
        raise ValueError('saved run has no configuration fingerprints; automatic recovery cannot verify its source')
    if config.get('mode') != mode_id:
        raise ValueError('checkpoint belongs to a different run or mode')
    return config


def acknowledge_walked(root: Path, mode_id: str) -> dict[str, Any] | None:
    """Acknowledge a walked run's failed or unknown boundary, as `graph-control
    acknowledge` acknowledges the spine's: set aside what it cannot vouch for, then
    record the acknowledgement bound to the progress record in ``resume-request.json``.

    ``None`` when the boundary needs none (a gate stop, a halt, a pause between nodes)
    and no dispatched worker's outcome is unknown.
    """
    from refactory_dsh.live import emit

    root = Path(root).resolve()
    path = root / PROGRESS
    if not path.is_file():
        return None
    # Before anything is moved: an acknowledgement given to the wrong main must leave
    # the run as it was, with no request a later plain resume could use.
    _walked_config(root, mode_id)
    payload = path.read_bytes()
    saved = json.loads(payload)
    status = walker_status(saved)
    markers = _pending_markers(root)
    if status not in ('executing', 'failed') and not markers:
        return None
    from refactory_dsh import run_state

    directory, files = set_aside(root, fingerprints=saved.get('file_fingerprints') or {}, markers=markers,
                                 status=status, owner_pid=run_state.owner_pid(root))
    request: dict[str, Any] = {'checkpoint_sha256': hashlib.sha256(payload).hexdigest()}
    if status == 'failed':
        request['acknowledge_failed'] = True
    elif status == 'executing':
        request['acknowledge_executing'] = True
    pending = root / f'{RESUME_REQUEST}.tmp'
    pending.write_text(json.dumps(request))
    pending.replace(root / RESUME_REQUEST)
    emit(root, 'control_requested', state='acknowledged', summary=f'acknowledge {status} checkpoint',
         set_aside=str(directory.relative_to(root)), files_count=len(files))
    return {'action': 'acknowledge', 'status': 'acknowledged', 'checkpoint_status_before': status,
            'set_aside': str(directory.relative_to(root)), 'files': files}


def prepare_walked(root: Path, mode_id: str, *, acknowledge_failed: bool = False) -> None:
    """Refuse to resume a walked run where :func:`prepare` refuses the spine's (dm-160).

    The walker's progress outcome is mapped to the spine's status first
    (:data:`WALKER_STATUSES`), and then the same checks run, with the same messages:
    the boundary is a known stopped one or acknowledged, no dispatched worker's outcome
    is unknown, no declared input or output changed since the progress record
    fingerprinted it, and the prepared configuration in ``run-config.json`` is
    unchanged, but for the driver checkout's own files: the walker has no pin, and goes
    on with the code checked out now. Only the commands named differ: a walked run's
    resume is its main's ``--resume``, given ``--acknowledge-failure``. The walker's ownership lock, which its
    caller holds, is what refuses a live owner.

    The acknowledgement is consumed here: a later resume of the same boundary needs a
    new one.
    """
    root = Path(root).resolve()
    path = root / PROGRESS
    if not path.is_file():
        return  # nothing to resume; `Mode.resume` says so by name
    payload = path.read_bytes()
    saved = json.loads(payload)
    config = _walked_config(root, mode_id)
    command = _walker_acknowledge_command(config, root)
    resume_status_check(walker_status(saved),
                        acknowledge_failed=acknowledge_failed or _acknowledged(root, payload, 'acknowledge_failed'),
                        acknowledge_executing=_acknowledged(root, payload, 'acknowledge_executing'),
                        acknowledge_command=command, failed_command=command)
    _require_settled(root, saved, acknowledge_command=command)
    # The walker has no pin: it resumes on the checkout as it is now, as it always has.
    # So the driver checkout's own files are not held to their launch fingerprints (the
    # spine skips them too, and re-runs under its pin); the run's graphs, declarations
    # and any project package outside the checkout still are.
    from refactory_dsh.paths import REPO_ROOT
    check_configuration(root, config,
                        skip_driver_paths=(config.get('driver') or {}).get('checkout') or str(REPO_ROOT))
    (root / RESUME_REQUEST).unlink(missing_ok=True)
