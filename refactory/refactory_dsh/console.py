"""Driving a run from a GUI: the catalogue, a live run, and what it produced.

A user watching a search must not need a terminal. This is the surface they act
through, and it is deliberately *not* the loop: it starts runs, tails them, and
reads their artifacts, all through the files the loop already writes. A GUI, a
notebook and a shell all drive the same call, and none of them can see inside the
loop that they could not see on disk.

The loop stays Pythonic. What this adds is a process boundary the loop already
implied — a run is a directory of artifacts, so anything that can read a
directory can watch one.

A run outlives the console: ``start`` launches it in its own process, writes a
``console.json`` naming it, and returns. ``status`` reads that run's progress
file, so a GUI that was restarted mid-run picks the run back up instead of
orphaning it.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from refactory_dsh import live, observe
from refactory_dsh import registry, profiles
from refactory_dsh.adopt import AdoptionError
from refactory_dsh.control import search_finished, search_owner, search_phase
from refactory_dsh.history import find_runs
from refactory_dsh.loop import Loop, LoopConfig, LoopError
from refactory_dsh.model import ModelChoiceError
from refactory_dsh.nn import PackageError
from refactory_dsh.paths import project_root
from refactory_dsh.steering import SteeringError

from refactory_dsh.invocations import CONSOLE_FILE  # canonical in invocations

#: Default place runs live, and the catalogue a GUI lists.
RUNS_ROOT = Path(".factory/loop")

#: The run settings a caller may pass at start, as (name, help, type). One table,
#: so the command line and :func:`start` cannot disagree about what is settable --
#: a setting the loop accepts but the console cannot name is a setting a GUI
#: cannot offer.
SETTINGS: tuple[tuple[str, str, type], ...] = (
    ("epochs", "how many epochs to run", int),
    ("batch_size", "how many candidates per batch", int),
    ("optimizer", "which optimizer proposes the next batch", str),
    ("strategy", "which outer loop drives the search", str),
    ("evaluator", "which runtime runs a candidate: walker or dsh", str),
    ("seed", "seed for the run's randomness, so a run repeats", int),
    ("elite", "how many parents each batch is proposed against", int),
    ("attempts", "retries per batch after a refusal", int),
    ("preset", "which edits the optimizer may use", str),
    ("max_evaluations", "hard ceiling on inner runs", int),
    ("holdout", "the fraction of the Task's items held out of the search", float),
    ("parsimony", "what each node of a candidate's graph costs its score", float),
)

#: The audit a run writes of its winner against the items no candidate saw.
HOLDOUT_FILE = "holdout.json"


class ConsoleError(RuntimeError):
    """The request cannot be served, and why."""


# ── the catalogue ──────────────────────────────────────────────────────────


def catalogue(*, project: Path | None = None) -> list[dict[str, Any]]:
    """Every mode a run can be started from, with the surface it exposes.

    The list is the registry's, not a hand-kept one: a mode added to a package
    appears here without an edit. A package that cannot be imported is listed
    with the reason rather than silently omitted, because "the mode I want is
    missing from the list" is otherwise indistinguishable from a typo.
    """
    project = Path(project or project_root())
    from refactory_dsh.cli import AUTHORING

    grouped = registry.catalogue(project)
    found: list[dict[str, Any]] = []
    for package_name, modes in grouped.items():
        enabled = registry.is_enabled(project, package_name)
        # `pending` while a package a create or edit run registered has not passed its
        # smoke (`refactory list` marks it unverified); reported, not acted on (dm-264).
        smoke = registry.smoke_state(package_name, project)
        for entry in modes:
            identifier = f"{package_name}/{entry['name']}"
            try:
                package, klass = registry.resolve(project, identifier)
            except PackageError as error:
                found.append({
                    "id": identifier,
                    "package": package_name,
                    "mode": entry["name"],
                    "enableable": entry.get("package_enabled", enabled),
                    "enabled": enabled,
                    "startable": False,
                    "smoke": smoke,
                    "reason": str(error),
                })
                continue
            # One `describe()` per mode, not three: the surface a start dialog
            # shows is the surface the optimizer already reads, and a catalogue
            # that reshaped it would be a second answer to the same question.
            mode = klass()
            described = mode.describe()
            found.append({
                "id": identifier,
                "package": package_name,
                "mode": entry["name"],
                "description": entry.get("description", ""),
                "startable": enabled,
                "enabled": enabled,
                "trust": package.trust,
                "smoke": smoke,
                "scored": mode.is_scored(),
                # Which verb runs this mode. An unscored mode is an authoring mode:
                # `start` refuses it, and a caller that has to fail to find that out
                # is a caller that picked the wrong tool from the catalogue. The
                # authoring verbs are `cli.AUTHORING`'s keys, including the fast
                # variants; anything else unscored falls back to `run`.
                "action": (
                    "start" if mode.is_scored()
                    else entry["name"] if entry["name"] in AUTHORING
                    else "run"
                ),
                "guidance": described["guidance"],
                "preferred_edits": described["preferred_edits"],
                "parameters": described["knobs"],
                "nodes": described["nodes"],
                "edges": described["edges"],
                "ports": described["ports"],
                # The second route to every mode: the workflow spine lowers the same
                # mode and hands it over one node at a time, which is how a run that
                # stops at a person's gate gets that person asked. Reported rather
                # than assumed, because a caller that only knows `action` picks the
                # headless route and pre-answers a gate the person could have been
                # asked at (#92, #93).
                "interactive": True,
                # Where the run will stop for a person, before it is opened. `fn` and
                # `agent` gates are answered by the runtime; `user` gates are put to
                # the person, and no verdict may be passed for them.
                "gates": [
                    {"id": node["id"], "user": node.get("gate_kind") == "user"}
                    for node in described["nodes"]
                    if node["kind"] == "gate"
                ],
            })
    from refactory_dsh.discovery import mode_help
    return [mode_help(entry) for entry in sorted(found, key=lambda item: item["id"])]


# ── starting a run ─────────────────────────────────────────────────────────


def start(
    mode: str,
    *,
    project: Path | None = None,
    runs_root: Path | None = None,
    run_id: str | None = None,
    **config: Any,
) -> dict[str, Any]:
    """Begin a run in its own process, and hand back the id to watch it with.

    In a separate process on purpose: a search runs for hours and a GUI request
    does not, and a run that died with its window would be a run nobody could
    start and walk away from.
    """
    project = Path(project or project_root())
    # An authoring mode (`create`, `edit`, `design`) declares no `score()`, so the
    # loop has no signal to search and refuses it. Said here rather than in the run's
    # log, because a caller told "this mode is for authoring; use the create action"
    # picks the right thing, and one told "the run exited 1" does not.
    try:
        _package, mode_class = registry.resolve(project, mode)
    except Exception:
        mode_class = None
    if mode_class is not None and not mode_class().is_scored():
        verb = mode.split("/", 1)[-1]
        action = verb if verb in ("create", "edit", "design") else "the console's authoring verb"
        raise ConsoleError(
            f"{mode} declares no score(), so there is nothing for a run to search; "
            f"it is an authoring mode — run it with the `{action}` action instead of "
            "`start`"
        )
    runs_root = Path(runs_root) if runs_root else project / RUNS_ROOT
    run_id = run_id or _run_id(mode)
    root = runs_root / run_id
    if root.exists():
        raise ConsoleError(f"{root} already exists; pick another run id")
    profile_name = str(config.pop("profile", "") or "")
    profile_version = config.pop("profile_version", None)
    config = {**(config.pop("settings_json", None) or {}), **config}
    snapshot = profiles.resolve(project, mode, name=profile_name, version=profile_version, overrides=config)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_path = root / profiles.SNAPSHOT
    snapshot_path.write_text(json.dumps(snapshot, indent=2))

    console = {
        "run_id": run_id,
        "mode": mode,
        "root": str(root),
        "started": round(time.time(), 3),
        "config": snapshot["settings"],
        "resolved_profile": snapshot,
        "log": f"{run_id}.log",
    }
    (root / CONSOLE_FILE).write_text(json.dumps(console, indent=2))

    process = _spawn(mode, root, {"resolved_config": str(snapshot_path)}, project=project)
    console["pid"] = process.pid
    (root / CONSOLE_FILE).write_text(json.dumps(console, indent=2))
    return console


def _spawn(mode: str, root: Path, config: dict[str, Any], *, project: Path) -> subprocess.Popen:
    """The run, in a process that does not die with this one."""
    argv = [sys.executable, "-m", "refactory_dsh.loop", mode, "--runs-root", str(root)]
    for key, value in sorted(config.items()):
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, (list, tuple)):
            for element in value:
                argv += [flag, str(element)]
        else:
            argv += [flag] if value is True else [flag, str(value)]
    log = (root / f"{root.name}.log").open("wb")
    return subprocess.Popen(
        argv,
        cwd=str(project),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def _run_id(mode: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{mode.replace('/', '-')}-{stamp}"


# ── lowering a mode for a driver that walks it node by node ────────────────


#: The data node's item loader, as the composition mounts it. Spelled here
#: because the console is the Python owner the Web composition reaches, and the
#: graph bundle names the same module in its own `dataCommand`; each uses an
#: absolute interpreter path so they do not depend on PATH.
def _refactory_python() -> str:
    """The Python interpreter for node commands, resolved at call time.

    After a re-exec under a pin, ``venv_bin()`` returns the pin's ``.venv/bin``,
    so the commands point to the pin's Python.
    """
    from refactory_dsh.paths import venv_bin
    return str(venv_bin() / 'python')


def _data_command() -> str:
    return f"{_refactory_python()} -m refactory_dsh.items"


def _agent_command() -> str:
    return f"{_refactory_python()} -m refactory_dsh.agent_delivery"


def _gate_command() -> str:
    return f"{_refactory_python()} -m refactory_dsh.steering"


def _node_environment(
    mode: str,
    *,
    project: Path,
    workdir: Path,
    knobs: list[str] | None = None,
) -> dict[str, str]:
    """The environment a node command of **this** run runs under.

    The values are per run -- the work directory and the mode's knobs -- and a Web
    process serves many runs, so they cannot be mount-time configuration. Both
    owners are reused rather than restated: the lowering owns the knob values
    (:func:`refactory_dsh.lowering.knob_environment`) and the node runner owns the
    names it reads (``refactory_dsh/run.py`` reads ``REFACTORY_WORKDIR``,
    ``REFACTORY_PROJECT`` and ``REFACTORY_KNOBS``, and
    ``refactory_dsh/items.py`` reads ``REFACTORY_WORKDIR``).

    ``PATH`` leads with the run's interpreter so an op node's bare ``python -m
    refactory_dsh.run`` (`lowering.RUNNER`) resolves to the venv that has
    ``refactory_dsh`` installed, exactly as the graph profile's ``commandEnv`` does.
    Agents get the target's own PATH when they are dispatched (`model._environment`).
    """
    from refactory_dsh import interactive, lowering, registry
    from refactory_dsh.paths import venv_bin

    # The same mode with the same knob overrides `interactive.prepare` lowered,
    # through the same owner, so the environment cannot disagree with the graph.
    Mode = registry.resolve(project, mode)[1]
    state = interactive.with_knobs(Mode().state_dict(), list(knobs or []))
    interpreter = str(venv_bin())
    ambient = os.environ.get("PATH", "")
    return lowering.knob_environment(state, extra={
        "REFACTORY_WORKDIR": str(workdir),
        "REFACTORY_PROJECT": str(project),
        "PATH": f"{interpreter}:{ambient}" if ambient else interpreter,
    })


def _open_mode(
    mode: str,
    *,
    project: Path | None = None,
    workdir: Path | str | None = None,
    knobs: list[str] | None = None,
    intent: str = "",
    model_context: dict[str, Any] | None = None,
    resource_limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lower a mode for a caller that will drive its graph one node at a time.

    The web chat session drives a mode through the spine's `wf_*` tools instead
    of the `graph` profile, which lowers its graph once at mount. This verb does
    the lowering -- through :func:`refactory_dsh.interactive.prepare`, the same
    owner the interactive entry point uses -- and prints where the graph landed,
    so the spine can open it after boot.

    The payload's field names are the ones the lowering already uses:

    ``mode``
        the mode asked for, as ``<package>/<name>``.
    ``run``
        the run directory the mode was lowered into, absolute.
    ``name``
        the graph name the lowering chose (``<package>-<mode>``).
    ``graph``
        absolute path of the lowered graph JSON the spine loads.
    ``declarations``
        absolute path of the run's node declarations.
    ``commandEnv``
        the environment this run's node commands run under, an object of
        strings, computed by :func:`_node_environment` from the same lowering.
    ``dataCommand``
        the command that selects and opens a data node's item passes, so a mode
        with a `DataNode` iterates on this run's directory and interpreter.

    ``workdir`` is required: a driven run needs one directory to own, and
    without it there is nowhere to lower the graph or write the run's status.
    A mode that cannot be resolved is refused by name.
    """
    project = Path(project or project_root()).resolve()
    if not workdir:
        raise ConsoleError("open needs a --workdir: the directory this run is lowered into")
    # The interactive entry point owns the lowering (ledger seeding, intent fold,
    # declarations), so this verb delegates to it rather than reimplementing any
    # part of it. Imported here to keep the console's module load light.
    from refactory_dsh import interactive
    from refactory_dsh.lowering import DECLARATIONS, SUFFIX

    root = Path(workdir).expanduser().resolve()
    from refactory_dsh.paths import require_build
    try:
        require_build(gui_mode=bool(os.environ.get("REFACTORY_GUI_MODE")))
    except RuntimeError as exc:
        # A stale build under the GUI must not refuse the open — warn and
        # continue.  Main had no check here; the PR adds one, and refusing
        # every open (including the case where a rebuild was skipped because
        # a run was live) breaks the GUI workflow.
        import logging as _clog
        _clog.getLogger(__name__).warning("%s", exc)
    try:
        name, graphs = interactive.prepare(mode, workdir=root, project=project, knobs=knobs, intent=intent,
                                           **({"model_context": model_context} if model_context else {}),
                                           **({"resource_limits": resource_limits} if resource_limits else {}))
        command_env = _node_environment(mode, project=project, workdir=root, knobs=knobs)
        from refactory_dsh import workers
        context = json.loads((root / '.refactory/run-config.json').read_text())['model_context']
        command_env.update(workers.environment(context))
    except ValueError as error:
        # A malformed identifier is the caller's to fix, so it is a refusal with
        # the reason rather than a traceback.
        raise ConsoleError(str(error)) from error
    return {
        "mode": mode,
        "run": str(root),
        "name": name,
        "graph": str((graphs / f"{name}{SUFFIX}").resolve()),
        "declarations": str((root / DECLARATIONS).resolve()),
        "commandEnv": command_env,
        "dataCommand": _data_command(),
        "agentCommand": _agent_command(),
        "gateCommand": _gate_command(),
    }


# ── watching one ───────────────────────────────────────────────────────────


#: The graph spine's own status file. Its presence is what makes a run directory an
#: authoring run, as opposed to a search: a search writes `progress.jsonl`, and an
#: authoring run writes this.
SPINE_STATUS = Path(".refactory") / "spine-status.json"

#: The spine's states, in the words a reader of `status` already has.
SPINE_STATES = {
    "done": "finished", "ready": "running", "failed": "failed", "refused": "failed",
    "halted": "halted", "paused": "paused", "executing": "running", "waiting_user": "waiting_user",
}

#: How long one `stop` waits for the signalled process to exit before it reports that
#: termination is still unconfirmed. A loop shuts down between boundaries, so this is
#: usually a moment; the record below keeps the answer true when it is longer.
STOP_CONFIRM_SECONDS = 2.0

#: The record a `stop` leaves once it has signalled a live run. The pid alone cannot
#: answer a later poll -- "gone because we stopped it" and "gone because it finished"
#: look identical -- so the request is what a subsequent poll reads.
STOP_REQUESTED = Path(".refactory") / "stop-requested.json"


def _spine(run_root: Path) -> dict[str, Any]:
    """What the graph spine last recorded, or ``{}`` when this is not an authoring run."""
    path = run_root / SPINE_STATUS
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _authoring(run_root: Path, spine: dict[str, Any]) -> dict[str, Any]:
    """An authoring run's own progress: the nodes done, the nodes due, and its facts."""
    facts = _read_json(run_root / "facts.json")
    # Per-step token totals: sum worker receipts by node_id.
    step_tokens: dict[str, int] = {}
    workers_dir = run_root / ".refactory" / "workers"
    if workers_dir.is_dir():
        for path in sorted(workers_dir.glob("*.json")):
            try:
                receipt = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(receipt, dict):
                continue
            node_id = receipt.get("node_id")
            if not node_id or not isinstance(node_id, str):
                continue
            usage = receipt.get("usage") or {}
            tokens = usage.get("tokens") if usage.get("tokens") is not None else usage.get("known_tokens")
            if isinstance(tokens, (int, float)) and tokens >= 0:
                step_tokens[node_id] = step_tokens.get(node_id, 0) + int(tokens)
    total_tokens = sum(step_tokens.values()) if step_tokens else 0
    return {
        "spine": spine.get("status"),
        "completed": spine.get("completed") or [],
        "nodes_due": spine.get("nodes") or [],
        "error": spine.get("error"),
        "name": facts.get("name"),
        "validation": facts.get("validation"),
        "package_path": facts.get("package_path"),
        "files": facts.get("files") or [],
        "before_score": facts.get("before_score"),
        "after_score": facts.get("after_score"),
        "regressed": facts.get("regressed"),
        "revision_note": facts.get("revision_note"),
        "step_tokens": step_tokens if step_tokens else None,
        "total_tokens": total_tokens if step_tokens else None,
        "check_attempts_used": facts.get("check_attempts_used", 0),
        "review_attempts_used": facts.get("review_attempts_used", 0),
        "check_repairs_used": facts.get("check_repairs_used", 0),
        "review_repairs_used": facts.get("review_repairs_used", 0),
    }


def _item_results(run_root: Path) -> dict[str, int] | None:
    """Count recorded Task verdicts without inventing a mode's scoring rule."""
    values = _read_json(run_root / "facts.json").get("item_scores")
    if not isinstance(values, list):
        return None
    rows = [value for value in values if isinstance(value, dict)]
    if not rows:
        return None
    passed = sum(row.get("passed") is True for row in rows)
    failed = sum(row.get("passed") is False for row in rows)
    return {"count": len(rows), "passed": passed, "failed": failed,
            "unclassified": len(rows) - passed - failed}


def _run_kind(run_root: Path, spine: dict[str, Any]) -> str:
    """Offer controls only for a recognized persisted run layout."""
    config = _read_json(run_root / "config.json")
    if isinstance(config.get("mode"), str) and isinstance(config.get("optimizer"), str):
        return "optimizer"
    # A walked run has no spine checkpoint, so no graph controls to offer.
    from refactory_dsh import run_state

    if run_state.walked(run_root):
        return "walker"
    if spine or any((run_root / "graphs").glob("*.graph.json")):
        return "graph"
    return "unknown"


def _activity_page(after: int, limit: int) -> int:
    """Refuse an activity window that cannot advance a cursor; clamp the rest.

    ``limit`` is a page size, so zero and negatives are refused (exit 2) before a
    run directory is read. Above the maximum it is clamped rather than refused: a
    caller asking for more than the console serves still gets a usable page.
    """
    if after < 0:
        raise ConsoleError(f"--activity-after must be nonnegative, got {after}")
    if limit < 1:
        raise ConsoleError(f"--activity-limit must be at least 1, got {limit}; "
                           "a page with no records cannot advance a cursor")
    return min(limit, live.PAGE_MAX)


def status(run_root: Path, *, after: int = 0, activity_after: int = 0,
           activity_limit: int = live.PAGE_DEFAULT) -> dict[str, Any]:
    """Where the run is, and what is new since the reader last looked.

    ``after`` is a line count, not a timestamp: a reader polling this wants the
    events it has not seen, and a clock cannot tell it which those are.
    ``activity_after``/``activity_limit`` page the journal the same way, so the
    activity history can be read a window at a time instead of whole.

    A search run's progress is `progress.jsonl`. An authoring run is driven by the
    graph spine and has none, so its own status file is read instead — without that,
    a creation could only be waited on by waiting for the call to return.
    """
    limit = _activity_page(activity_after, activity_limit)
    run_root = Path(run_root)
    if not run_root.is_dir():
        raise ConsoleError(f"{run_root} is not a run directory")
    events = observe.progress(run_root, after=after)
    console = _console(run_root)
    spine = _spine(run_root)
    halted = spine.get("status") == "halted"
    # Terminal is not the same as successful: polling stops at a halt, while
    # the state still names the halt rather than claiming the work completed.
    finished = search_finished(run_root) or spine.get("status") in {"done", "halted"}
    state = _state(run_root, events, console)
    if spine:
        state = SPINE_STATES.get(str(spine.get("status")), state)
        if finished and not halted:
            state = "finished"
    from refactory_dsh import run_state
    activity = live.snapshot(run_root, after=activity_after, limit=limit)
    # dm-95: surface last_failure info when the graph is retrying a node
    last_failure = spine.get('last_failure')
    if last_failure and isinstance(last_failure, dict) and spine.get('status') == 'ready':
        lf_node = last_failure.get('node', 'unknown')
        activity['last_failure'] = last_failure
        activity['wait_reason'] = f'step {lf_node} is being retried'
    # The nodes the graph's declared retry rule ran again (`node_retry`), on either runtime.
    from refactory_dsh.node_retry import retries as _node_retries
    retried = _node_retries(run_root)
    if retried:
        activity['node_retries'] = retried
    # Report command corrections from facts.json
    facts_data = _read_json(run_root / 'facts.json')
    # dm-127: show design review status when a user gate is due.
    if gates(run_root):
        from refactory_dsh.design_ops import design_review_status as _drs
        drs_text = _drs(facts_data)
        if drs_text:
            activity['design_review_status'] = drs_text
    command_history = facts_data.get('command_history', [])
    if isinstance(command_history, list) and command_history:
        activity['command_corrections'] = len(command_history)
        activity['last_correction'] = command_history[-1]
    # Report queued controls
    from refactory_dsh.graph_control import queued_controls
    qc = queued_controls(run_root)
    if qc:
        activity['queued_controls'] = qc
    # Compute owner liveness before the start-failure check needs it.
    checkpoint = _read_json(run_root / '.refactory/spine-checkpoint.json')
    # A walked run's owner is the walking process its status file names.
    owner = run_state.owner_pid(run_root) if run_state.walked(run_root) else checkpoint.get('owner_pid')
    owner_alive = (True if (isinstance(owner, int) and _alive(owner)) else
                   False if isinstance(owner, int) else None)
    # Supervision-history and start-failure enrich the activity dict.
    sup_history = run_state.read_supervision_history(run_root)
    if sup_history:
        activity['respawn_count'] = len(sup_history)
        activity['last_respawn'] = sup_history[-1].get('at', '')
    start_failure = run_state.read_start_failure(run_root)
    if start_failure and start_failure.get('suggested_command'):
        activity['suggested_command'] = start_failure['suggested_command']
    if start_failure and owner_alive is not True:
        sf_spine = start_failure.get('spine_status_at_exit', '')
        current_spine = str(spine.get('status', ''))
        if current_spine == sf_spine or not current_spine:
            state = 'failed'
            activity['start_failure'] = start_failure.get('reason', '')
    resources = activity.get('resources') or {}
    running = _running(run_root, console) or owner_alive is True
    if resources.get('exhausted') and not finished:
        state = 'failed'
        activity['wait_reason'] = resources.get('reason') or 'resource budget exhausted'
    elif spine and owner and not running and state not in ('finished', 'failed', 'halted', 'paused'):
        state = 'died'
        activity['wait_reason'] = 'owner exited; inspect the saved boundary before explicit recovery'
    return {
        "run_id": run_root.name,
        "root": str(run_root),
        "mode": console.get("mode"),
        "kind": _run_kind(run_root, spine),
        "state": state,
        "events": events,
        "cursor": after + len(events),
        "counters": _counters(observe.progress(run_root)),
        "best_score": _best(run_root, events),
        "finished": finished,
        "running": not finished and state not in ("failed", "died", "paused", "halted") and bool(running),
        "owner_alive": owner_alive,
        "authoring": _authoring(run_root, spine) if spine else None,
        "item_results": _item_results(run_root),
        "activity": activity,
    }


def _state(run_root: Path, events: list[dict[str, Any]], console: dict[str, Any]) -> str:
    """One of ``starting``, ``running``, ``finished``, ``died``.

    A run that left a report is finished whatever its process is doing. A run
    with a console record and no live process did not finish, and saying
    "running" would leave a reader waiting forever for a run that is gone.
    """
    # A start-failure with a dead owner overrides the spine state.
    from refactory_dsh import run_state
    sf = run_state.read_start_failure(run_root)
    if sf:
        checkpoint = _read_json(run_root / '.refactory/spine-checkpoint.json')
        pid = checkpoint.get('owner_pid')
        if isinstance(pid, int) and not _alive(pid):
            sf_spine = sf.get('spine_status_at_exit', '')
            current_spine = _spine(run_root).get('status', '')
            if current_spine == sf_spine or not current_spine:
                return 'failed'
    phase = search_phase(run_root)
    if phase in ("running", "died", "failed"):
        return phase
    if (run_root / "report.json").exists():
        report = _read_json(run_root / "report.json")
        return "failed" if report.get("exit_ok") is False else "finished"
    if _running(run_root, console):
        return "running"
    if console.get("pid"):
        return "died"
    return "running" if events else "starting"


def _counters(events: list[dict[str, Any]]) -> dict[str, Any]:
    latest = events[-1] if events else {}
    return {
        "epoch": latest.get("epoch", 0),
        "generation": latest.get("generation", 0),
        "evaluations": latest.get("evaluations", 0),
        "steps": len(events),
    }


def _best(run_root: Path, events: list[dict[str, Any]]) -> float | None:
    """The current eligible winner, independent of the caller's progress cursor."""
    from refactory_dsh.adopt import best
    from refactory_dsh.history import JOURNAL

    if (run_root / JOURNAL).exists():
        try:
            return float(best(run_root)["score"])
        except AdoptionError:
            return None
    report = _read_json(run_root / "report.json")
    if report.get("exit_ok") is True:
        return report.get("best_score")
    scores = [
        event["score"] for event in events
        if event.get("exit_ok") is True and isinstance(event.get("score"), (int, float))
    ]
    return max(scores) if scores else None


def _console(run_root: Path) -> dict[str, Any]:
    """What the console recorded about a start, or what the run itself did.

    A run started by the loop's own entry point has no console file: nobody
    watched it begin. `run.json` is written by the loop itself and names the mode,
    so a reader learns what a run was optimizing whether or not the console
    started it. The console file still wins, because it also carries the pid and
    the settings the console chose.
    """
    recorded = _read_json(run_root / CONSOLE_FILE)
    owner = search_owner(run_root)
    if owner:
        recorded = {**recorded, "pid": owner.get("pid") if search_phase(run_root) == "running" else None}
    if recorded.get("mode"):
        return recorded
    started = _read_json(run_root / "run.json")
    return {**recorded, "mode": started["mode"]} if isinstance(started.get("mode"), str) else recorded


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _facts_text(values: dict[str, Any]) -> dict[str, str]:
    """Render facts before a JavaScript reader can round large JSON integers."""
    return {
        name: value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, separators=(",", ":"),
        )
        for name, value in values.items() if value is not None and value != ""
    }


def _running(run_root: Path, console: dict[str, Any]) -> bool:
    pid = console.get("pid")
    return bool(pid) and _alive(int(pid))


# Factored to run_state to avoid circular imports (supervisor → console → supervisor).
from refactory_dsh.run_state import is_alive as _alive


def _checkpoint_owner_pid(run_root: Path) -> int | None:
    """The checkpoint's owner PID when it is alive, or None."""
    checkpoint = _read_json(run_root / '.refactory/spine-checkpoint.json')
    pid = checkpoint.get('owner_pid')
    if isinstance(pid, int) and _alive(pid):
        return pid
    return None


def stop(run_root: Path) -> dict[str, Any]:
    """Ask the run to stop, and say whether it did.

    Signalling is not instantaneous: a loop shuts down between boundaries, so the
    process is normally still alive right after ``SIGINT``. Reporting that as "not
    stopped" and then reading the next poll as "not running, nothing to stop" told a
    person nothing was stopped about the very run their stop had just ended. The
    request is recorded before the wait, and that record -- not the pid -- answers a
    later poll.
    """
    run_root = Path(run_root)
    console = _console(run_root)
    pid = console.get("pid")
    if pid:
        pid = int(pid)
    # Fall back to the checkpoint's owner PID — the same source console.status
    # uses for owner_alive — so a resumed run stops like a launched one.
    if not pid or not _alive(pid):
        pid = _checkpoint_owner_pid(run_root)
    marker = run_root / STOP_REQUESTED
    if not pid:
        if marker.exists():
            return {"stopped": True, "requested": True,
                    "reason": "the run stopped"}
        return {"stopped": False, "reason": "the run is not running"}
    if not _alive(pid):
        if marker.exists():
            return {"stopped": True, "requested": True, "pid": pid,
                    "reason": "the run stopped"}
        return {"stopped": False, "reason": "the run is not running"}
    try:
        os.kill(pid, signal.SIGINT)
    except OSError as error:
        return {"stopped": False, "reason": str(error)}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"pid": pid, "at": round(time.time(), 3)}))
    deadline = time.monotonic() + STOP_CONFIRM_SECONDS
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.05)
    if _alive(pid):
        return {"stopped": False, "requested": True, "pid": pid,
                "reason": "cancellation requested; worker termination is not yet confirmed"}
    return {"stopped": True, "requested": True, "pid": pid, "reason": "the run stopped"}


# ── reading one ────────────────────────────────────────────────────────────


def artifacts(run_root: Path, *, contents: bool = False) -> dict[str, Any]:
    """What the run left, and — when asked — what is in it.

    Text is returned inline because that is what a GUI shows and what an optimizer
    reads. A reader that had to fetch each file separately would need to know the
    layout, which is the thing the manifest exists to avoid telling it.
    """
    run_root = Path(run_root)
    manifest = observe.artifact_manifest(run_root)
    if not contents:
        return manifest
    for entry in manifest["artifacts"]:
        if entry["media_type"] == "application/x-ndjson" or entry["path"].endswith("/"):
            continue
        path = run_root / entry["path"]
        if not path.exists():
            continue
        entry["content"] = (
            json.loads(path.read_text())
            if entry["media_type"] == "application/json"
            else path.read_text()
        )
    return manifest


def candidate(run_root: Path, candidate_id: str) -> dict[str, Any]:
    """One candidate, with what it changed against the parent it came from."""
    run_root = Path(run_root)
    diff = observe.candidate_diff(run_root, candidate_id)
    record = _measured(run_root, candidate_id)
    return {**diff, "rendered": observe.render_diff(diff["diff"]), **record}


def comparison(run_root: Path) -> dict[str, Any]:
    """Compare measured and unmeasured states without promoting missing metrics."""
    root = Path(run_root)
    rows = []
    seen = set()
    journal = root / "journal.jsonl"
    for line in journal.read_text().splitlines() if journal.exists() else ():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "candidate" or event.get("id") in seen:
            continue
        seen.add(event["id"])
        row = candidate(root, event["id"])
        row["id"] = event["id"]
        measured = _measured(root, event["id"])
        row["status"] = "unmeasured" if not measured else "measured" if measured.get("exit_ok") is True else "failed"
        row.setdefault("score", None)
        row.setdefault("raw_score", None)
        row.setdefault("duration_s", None)
        row["tokens"] = (measured.get("measurements") or {}).get("tokens")
        row["cost"] = (measured.get("measurements") or {}).get("cost")
        rows.append(row)
    audit = holdout(root)
    return {"candidates": rows, "holdout": audit,
            "resolved_profile": _read_json(root / profiles.SNAPSHOT),
            "selection_rule": profiles.SELECTION_RULE,
            "interpretation": "Scores are observed sample means. No measured improvement is a valid outcome; unavailable cost or token usage is not zero."}


def gates(run_root: Path) -> list[dict[str, Any]]:
    """The `user` gates this run is stopped at, and the verdicts each accepts.

    A run that pauses at a gate a person owns is waiting for an answer, and a
    caller that wants to give it needs two things the ledger cannot supply: which
    gate is due, and which outcome labels route it. Both are the run's own lowered
    graph -- the spine's status names the due node, and that node's outgoing
    conditional edges are the labels the spine accepts (the same derivation the
    spine's own ``acceptedVerdicts`` performs). Reading them here keeps the graph's
    format in the console, the way the ledger's format is, rather than making every
    caller parse a run directory it did not lower.

    Empty for a run with no status (a search, whose graph lives per candidate), for
    a due node that is not a gate, and for a gate a machine answers: only a `user`
    gate is a person's to route.
    """
    from types import SimpleNamespace

    from refactory_dsh.nn.executor import render

    spine = _spine(run_root)
    due = [str(node) for node in (spine.get("nodes") or [])]
    if not due:
        return []
    graphs = Path(run_root) / "graphs"
    if not graphs.is_dir():
        return []
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(graphs.glob("*.graph.json")):
        try:
            loaded = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, dict):
            continue
        nodes = loaded.get("nodes") or {}
        edges = loaded.get("edges") or []
        for node_id in due:
            node = nodes.get(node_id) if isinstance(nodes, dict) else None
            if not isinstance(node, dict):
                continue
            if node.get("_type") != "GateNode" or node.get("evaluator_type") != "user":
                continue
            verdicts = [
                str(edge.get("condition")).strip().lower()
                for edge in edges
                if isinstance(edge, dict)
                and edge.get("source") == node_id
                and edge.get("condition") not in (None, "")
            ]
            found[node_id] = {
                "id": node_id,
                # Filled from the run's facts, as the spine fills the question it puts.
                "prompt": render(str(node.get("gate_prompt") or ""),
                                 SimpleNamespace(knobs={}, facts=_read_json(Path(run_root) / "facts.json"))),
                "verdicts": verdicts,
            }
    # Append CEO-sourced ledger notes to each gate, labelled as the CEO's.
    for node_id, gate_dict in found.items():
        try:
            from refactory_dsh.steering import Ledger
            entries = Ledger(run_root).entries()
            ceo_notes = [
                e["text"] for e in entries
                if e.get("source") == "ceo" and isinstance(e.get("text"), str)
            ]
            gate_dict["ceo_notes"] = "\n\n".join(ceo_notes) if ceo_notes else ""
        except Exception:
            gate_dict["ceo_notes"] = ""
    return [found[node_id] for node_id in due if node_id in found]


def _steer(
    run_root: Path,
    *,
    seed: str | None = None,
    note: str | None = None,
    kind: str = "info",
    scope: str = "product",
    source: str = "user-relayed",
    verdicts: tuple[str, ...] | list[str] = (),
    with_gates: bool = False,
    control: str | None = None,
    target: str | None = None,
    lineage: bool = False,
    as_name: str | None = None,
    project: Path | None = None,
) -> dict[str, Any]:
    """Read the ledger, and optionally add to it.

    Steering is context, not structure: this writes what the person said and nothing
    else. The graph of the running mode is untouched -- that is the outer loop's job,
    and conflating the two is how a run stops being accountable for what it was asked
    to do.

    ``with_gates`` adds :func:`gates` to the payload: the `user` gates the run is
    stopped at, so a caller that only knows this verb can offer the answer the gate
    accepts. It is opt-in because the ledger read is what most callers want, and a
    caller that does not route gates should not pay for reading a graph it will not
    use.
    """
    from refactory_dsh.steering import Ledger, chain, fidelity, outputs_in

    from refactory_dsh.agent_delivery import receipts_in
    from refactory_dsh.finished_control import latest_receipt

    ledger = Ledger(run_root)
    if seed is not None:
        Ledger.seed(run_root, seed)
    effective_kind = kind
    if note is not None:
        if not note.strip():
            raise SteeringError("--note must not be empty")
        if verdicts and kind == "info":
            from refactory_dsh.steering import NOTE_KINDS
            _, _, verdict_value = str(verdicts[-1]).partition("=")
            effective_kind = NOTE_KINDS.get(verdict_value.strip().lower(), kind)
    _effective_kind = effective_kind
    recorded_scopes: dict[str, str] = {}
    for pair in verdicts:
        gate, _, value = str(pair).partition("=")
        entry = ledger.verdict(gate, value)
        recorded_scopes[str(pair)] = entry.get('scope', '')
    if note is not None:
        ledger.note(effective_kind, note, scope=scope, source=source)
    # The chain is checkable only when the run left agent outputs beside the ledger
    # (a single mode run does; an outer-loop run keeps its candidates in subdirs).
    outputs = outputs_in(run_root) if ledger.exists() else {}
    deliveries = receipts_in(run_root)
    # E9: read the spine's completed-nodes list so chain() and fidelity() can
    # distinguish "has not run" from "ran and wrote nothing".
    _completed: set[str] | None = None
    spine_path = run_root / '.refactory' / 'spine-status.json'
    if spine_path.exists():
        try:
            _spine_data = json.loads(spine_path.read_text())
            _cl = _spine_data.get('completed')
            if isinstance(_cl, list):
                _completed = set(_cl)
        except (OSError, json.JSONDecodeError):
            pass
    payload: dict[str, Any] = {
        "run": str(run_root),
        "ledger": str(ledger.path),
        "exists": ledger.exists(),
        "entries": ledger.entries(),
        "constraints": ledger.constraints(),
        "verdicts": ledger.verdicts(),
        "recorded_scopes": recorded_scopes,
        "chain": chain(ledger, outputs, deliveries=deliveries,
                        completed_nodes=_completed) if ledger.exists() else [],
        "problems": fidelity(ledger, outputs, deliveries=deliveries,
                              completed_nodes=_completed),
        "deliveries": deliveries,
        "finished": search_finished(run_root),
        "control": latest_receipt(run_root),
    }
    payload["_effective_kind"] = _effective_kind
    if (note is not None and effective_kind in ("correction", "constraint")
            and scope == "product" and source == "user-relayed"):
        from refactory_dsh.design_ops import late_correction_notice

        notice = late_correction_notice(run_root)
        if notice is not None:
            payload["late_correction"] = notice
    if with_gates:
        payload["gates"] = gates(run_root)
    if control is not None:
        # Control is not prose: the loop polls a small file at a generation boundary,
        # so a pause, a stop, a keep, a reject or an adopt works from here, from the
        # panel, and from a process that is not the run's parent. A request that
        # cannot be honored as written (a keep with no id) is refused here, where the
        # person can still fix it, rather than written for the loop to guess at.
        from refactory_dsh.control import Control, ControlError

        try:
            # A walked run's controls are refused there: they need the spine's checkpoint.
            if _run_kind(run_root, _spine(run_root)) in ("graph", "walker"):
                from refactory_dsh.graph_control import request
                payload["control"] = request(run_root, control)
                return payload
            payload["control"] = Control(run_root).request(
                control, target=target, lineage=lineage, as_name=as_name, project=project
            )
            payload["control"].setdefault("status", "queued")
        except (ControlError, ValueError) as error:
            raise ConsoleError(str(error)) from error
    elif target or lineage or as_name:
        # `--lineage` or `--as` on their own name no act, and a flag that silently
        # does nothing is worse than one that is refused.
        raise ConsoleError(
            "--keep, --reject or --adopt names the act that --lineage and --as "
            "belong to; none of them was given"
        )
    return payload


def _best_payload(run_root: Path, *, keep_state: bool) -> dict[str, Any]:
    """The winner, the state that scored it, and how it did on the held-out items.

    The state is what makes a winner usable and it is the field `leaderboard` and
    `candidate` do not have: one gives a score, the other a diff, and neither can
    be run. It is large, so it is opt-in — but it is available without reading
    `journal.jsonl` by hand, which was the only way before.

    The held-out audit travels with the winner because the score alone cannot say
    whether the harness works on anything but the items it was tuned against.
    """
    from refactory_dsh.adopt import best

    winner = best(run_root)
    if not keep_state:
        winner = {key: value for key, value in winner.items() if key != "state"}
        winner["state_available"] = "pass --state"
    winner["holdout"] = holdout(run_root)
    if (winner["holdout"] or {}).get("overfit"):
        winner["overfit_finding"] = _overfit_finding(winner["holdout"])
    return winner


def holdout(run_root: Path) -> dict[str, Any] | None:
    """The held-out audit a run recorded, or ``None`` when it recorded none."""
    return _read_json(Path(run_root) / HOLDOUT_FILE) or None


def _overfit_finding(audit: dict[str, Any]) -> str:
    """What an overfit audit means for a reader who is about to keep the winner."""
    return (
        f"the winner scores {audit.get('train')} on the items it was searched on and "
        f"{audit.get('holdout')} on the held-out items, a gap of {audit.get('gap')} "
        f"over the threshold {audit.get('threshold')}; it may not work on items it "
        "was not tuned against"
    )


def _measured(run_root: Path, candidate_id: str) -> dict[str, Any]:
    """What the run directory recorded about the candidate's evaluation."""
    path = run_root / "runs" / candidate_id / "candidate.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def leaderboard(run_root: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    """The candidates by score, with each one's diff against its parent.

    The diff is what makes a leaderboard readable: without it the table says which
    number is bigger and nothing about why.
    """
    run_root = Path(run_root)
    report = run_root / "report.json"
    if report.exists():
        try:
            rows = json.loads(report.read_text()).get("leaderboard") or []
        except json.JSONDecodeError:
            rows = []
    else:
        rows = [
            {"id": path.parent.name, **json.loads(path.read_text())}
            for path in sorted((run_root / "runs").glob("*/candidate.json"))
            if _readable(path)
        ]
        rows.sort(key=lambda row: (-(row.get("score") or 0.0), row.get("id") or ""))
    out: list[dict[str, Any]] = []
    # What a person pinned or dismissed, so a reader of a *live* run sees their own
    # acts in the table they are reading. A finished run carries the same two lists
    # in `report.json`'s rows; a live one has them only in `state.json`.
    kept, rejected = _person_lists(run_root)
    eligible = [row for row in rows if row.get("exit_ok") is True]
    for row in eligible[:limit]:
        record = dict(row)
        candidate_id = str(row.get("id") or "")
        record["kept"] = candidate_id in kept
        record["rejected"] = candidate_id in rejected
        try:
            record["diff"] = observe.candidate_diff(run_root, row["id"])
        except FileNotFoundError:
            record["diff"] = None
        out.append(record)
    return out


def _person_lists(run_root: Path) -> tuple[set[str], set[str]]:
    """The candidates a person kept and rejected, from the run's own state.

    Absent state is an empty answer rather than an error: a leaderboard is readable
    before the loop has written anything, and before this existed every run's
    leaderboard is exactly what this returns for one.
    """
    state = _read_json(Path(run_root) / "state.json") or {}
    kept = {str(value) for value in state.get("kept") or ()}
    rejected = {str(value) for value in state.get("rejected") or ()}
    return kept, rejected


def _readable(path: Path) -> bool:
    try:
        json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return True


def runs(
    *,
    project: Path | None = None,
    runs_root: Path | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Every run under the root, newest first, each with where it got to.

    A finished run and a run in flight are both listed: a GUI that hid the one it
    was currently watching would empty its own list. Finished runs come from the
    reader that already bounds them; unfinished ones are found by the console file
    that a start writes before the loop's first artifact exists.
    """
    project = Path(project or project_root())
    root = Path(runs_root) if runs_root else project / RUNS_ROOT
    out: list[dict[str, Any]] = []
    for archive in find_runs(root, limit=limit):
        out.append({
            "run_id": archive.path.name,
            "root": str(archive.path),
            "mode": _console(archive.path).get("mode") or archive.mode,
            "best_score": archive.best_score,
            "finished": True,
        })
    started = sorted(root.glob(f"*/{CONSOLE_FILE}")) if root.is_dir() else []
    for entry in started:
        run_root = entry.parent
        if any(item["run_id"] == run_root.name for item in out):
            continue
        console = _console(run_root)
        out.append({
            "run_id": run_root.name,
            "root": str(run_root),
            "mode": console.get("mode"),
            "best_score": _best(run_root, observe.progress(run_root)),
            "finished": False,
        })
    if runs_root is None:
        for status_file in (project / '.factory' / 'interactive').glob('*/.refactory/spine-status.json'):
            directory = status_file.parent.parent
            frame = status(directory)
            out.append({"run_id": directory.name, "root": str(directory), "mode": frame["mode"],
                        "best_score": frame["best_score"], "finished": frame["finished"],
                        "state": frame["state"], "pending_questions": len(gates(directory))})
    out.sort(key=lambda item: -Path(item["root"]).stat().st_mtime)
    return out[:limit]


# ── the command line a tool shells out to ──────────────────────────────────


def _first_line(cls: Any) -> str:
    """A class's own one-line description, from its docstring."""
    text = (getattr(cls, "__doc__", "") or "").strip()
    return text.split("\n", 1)[0].strip()


def level_two() -> dict[str, Any]:
    """The three level-2 seams by name, each with what it says it is.

    The named choice a GUI needs: a person picks an outer loop, an optimizer or an
    evaluator from a catalogue with descriptions rather than typing a name they have
    to already know. Read from the registries, so a new implementation appears here
    the moment it registers.
    """
    from refactory_dsh import evaluators, optim, strategies

    def entries(registry: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"name": name, "description": _first_line(cls)}
            for name, cls in sorted(registry.items())
        ]

    return {
        "outer_loops": entries(strategies.STRATEGIES),
        "optimizers": entries(optim.REGISTRY),
        "evaluators": entries(evaluators.REGISTRY),
    }


def _design_review(project: Path, name: str) -> str:
    """The design review a creation wrote, so the choice it made is visible.

    A create run surveys approaches, picks one and writes `design.md`; without this
    a caller sees only the package it produced and cannot tell which approaches were
    considered or what the review refused.

    The working directory is shared by every create run, so the review is reported
    only when `brief.json` says it belongs to the package being reported: a run that
    stopped before the design step otherwise reports whichever review happens to be
    on disk — another package's.
    """
    for candidate in (project / ".factory" / "create", Path(".factory/create")):
        design = candidate / "design.md"
        brief = candidate / "brief.json"
        if not (design.exists() and brief.exists()):
            continue
        try:
            if str(json.loads(brief.read_text()).get("name") or "") != name:
                continue
            return design.read_text()[:8000]
        except (OSError, json.JSONDecodeError):
            continue
    return ""


def _parser() -> argparse.ArgumentParser:
    def with_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """The shared options, on every subcommand.

        Repeated per subcommand rather than declared once on the parent, because a
        caller assembling argv naturally appends them: ``console status --run X``
        is the order anyone writes, and a parser that only accepted them before
        the verb would reject the obvious call.
        """
        parser.add_argument("--project", default=None, help="repository root (default: found)")
        parser.add_argument(
            "--runs-root",
            default=None,
            help=(
                "the directory runs live under, one subdirectory per run; this is "
                "not the loop's --runs-root, which is the run directory itself"
            ),
        )
        parser.add_argument("--run", default=None, help="a run directory or run id")
        return parser

    parser = argparse.ArgumentParser(
        prog="python -m refactory_dsh.console",
        description="Drive and read outer-loop runs.",
    )
    sub = parser.add_subparsers(dest="verb", required=True)
    add = with_common

    runtime = add(sub.add_parser("runtime", help="nonsecret nested model context and configured readiness; no model request"))
    runtime.add_argument("--provider", default="", help="provider route to inspect (default: the stored model choice)")
    runtime.add_argument("--model", default="", help="model to inspect (default: the stored model choice)")

    help_parser = add(sub.add_parser("help", help="live capability schemas, interfaces and CEO operating guide"))
    help_parser.add_argument("--topic", default="", help="operation or mode ID; omit for the live index")
    docs_parser = add(sub.add_parser("docs", help="read/search the documentation shipped with this build"))
    docs_parser.add_argument("--page", default="", help="exact page ID from the docs index")
    docs_parser.add_argument("--query", default="", help="words to search in shipped documentation")
    docs_parser.add_argument("--section", default="", help="section ID from the docs index")
    docs_parser.add_argument("--offset", type=int, default=0, help="character offset within page or section")
    docs_parser.add_argument("--limit", type=int, default=6000, help="maximum text characters, 1..12000")
    add(sub.add_parser("workers", help="effective project worker defaults and provenance; read only"))
    add(sub.add_parser("projects", help="the registered projects for target selection"))
    add(sub.add_parser("modes", help="the catalogue of startable modes"))
    add(sub.add_parser("runs", help="every run under the root"))
    add(sub.add_parser("strategies", help="the named outer loops, optimizers and evaluators"))

    begin = add(sub.add_parser("start", help="begin a run in its own process"))
    begin.add_argument("mode")
    begin.add_argument("--run-id", default=None)
    profiles.add_arguments(begin)
    begin.add_argument("--settings-json", type=json.loads, default=None)
    add(sub.add_parser("profiles", help="list saved immutable profile versions"))
    save_profile = add(sub.add_parser("profile-save", help="save a new version of a search profile"))
    save_profile.add_argument("--name", required=True)
    save_profile.add_argument("--spec", required=True, type=json.loads, help="JSON object with settings and contract")
    resolve_profile = add(sub.add_parser("profile-resolve", help="preview effective settings and immutable search contract"))
    resolve_profile.add_argument("mode")
    profiles.add_arguments(resolve_profile)
    resolve_profile.add_argument("--settings-json", type=json.loads, default=None)
    add(sub.add_parser("comparison", help="all candidates including failed and unmeasured trials"))

    # `run` is the CLI's run: use a mode once, with the person answering any gate.
    # `start` is the outer loop (`refactory optimize`); the two are different acts and
    # the tool's actions are named the same way, so neither is mistaken for the other.
    single = add(sub.add_parser("run", help="use one mode once, with you answering any gate"))
    single.add_argument("mode", help="the mode, as <package>/<name>")
    single.add_argument("--intent", default="", help="what this run is for, in your words")
    single.add_argument("--workdir", default="", help="the run directory")
    single.add_argument("--resume", action="store_true")
    single.add_argument(
        "--source", default="", metavar="PATH",
        help="read the items from here instead of the source the mode declares",
    )
    single.add_argument(
        "--source-format", default="", choices=["", "directory", "jsonl", "csv"],
        help="how to read --source (default: the runner's own directory form)",
    )
    single.add_argument(
        "--knob", action="append", default=[], metavar="NAME=VALUE",
        help="set one of the mode's knobs for this run (repeatable)",
    )
    single.add_argument(
        "--verdict", action="append", default=[], metavar="GATE=VALUE",
        help="answer a gate a person owns, as <gate>=<value> (repeatable); a walker "
             "run stops at a user gate and prints the steer command; answer it, then resume",
    )
    single.add_argument("--timeout", type=int, default=0, help="seconds; 0 waits for the person")
    single.add_argument("--provider", default="", help="provider route for this run's model")
    single.add_argument("--model", default="", help="model id for this run")
    for key in ("worker_backend", "effort", "permission_mode"):
        single.add_argument("--" + key.replace("_", "-"), default="")
    for key in ("max_elapsed_s", "max_tokens", "validation_reserve_s", "validation_reserve_tokens"):
        single.add_argument("--" + key.replace("_", "-"), type=int if "tokens" in key else float, default=None)

    for key in ("role_routes", "node_routes"):
        single.add_argument("--" + key.replace("_", "-"), type=json.loads, default=None)

    # `open` lowers a mode for a caller that will drive its graph node by node
    # (a web chat session's spine), rather than running it to completion. It is
    # deliberately not `run`: nothing is executed here, and the caller owns the
    # driving.
    opened = add(sub.add_parser("open", help="lower a mode for a caller that drives it node by node"))
    opened.add_argument("mode", help="the mode to lower, as <package>/<name>")
    for key in ("worker_backend", "provider", "model", "effort", "permission_mode"):
        opened.add_argument("--" + key.replace("_", "-"), default="")
    for key in ("max_elapsed_s", "max_tokens", "validation_reserve_s", "validation_reserve_tokens"):
        opened.add_argument("--" + key.replace("_", "-"), type=int if "tokens" in key else float, default=None)
    for key in ("role_routes", "node_routes", "provider_config"):
        opened.add_argument("--" + key.replace("_", "-"), type=json.loads, default=None)
    opened.add_argument(
        "--model-timeout", type=int, default=None, dest="model_timeout_s",
        help="seconds one model completion may take before it is called hung (default: 1800)",
    )
    opened.add_argument("--intent", default="", help="the user's purpose for this run's ledger")
    opened.add_argument(
        "--workdir", default="",
        help="the run directory this caller drives (required); the graph, the "
             "declarations and the run's status are written here",
    )
    opened.add_argument(
        "--knob", action="append", default=[], metavar="NAME=VALUE",
        help="set one of the mode's knobs for this run (repeatable)",
    )

    watch = add(sub.add_parser("status", help="where a run is"))
    watch.add_argument("--after", type=int, default=0)
    watch.add_argument(
        "--activity-after", type=int, default=0,
        help="journal record index the activity page starts at (default: 0); "
             "serve the next page from the previous answer's activity.cursor",
    )
    watch.add_argument(
        "--activity-limit", type=int, default=200,
        help="journal records one activity page may carry (default: 200, at most 1000); "
             "0 is refused, because a page with no records cannot advance a cursor",
    )
    watch.add_argument("--follow", action="store_true", help="stream durable state changes without launching work")
    watch.add_argument("--interval", type=float, default=1.0, help="reader interval in seconds (minimum 0.1)")

    graph_control = add(sub.add_parser("graph-control", help="preview or queue a graph boundary control"))
    graph_control.add_argument("action", choices=[
        "pause", "resume", "stop", "preview", "rerun",
        "acknowledge", "amend-command", "withdraw", "redeliver",
        "confirm-completion", "skip",
    ])
    graph_control.add_argument("--node", default="")
    graph_control.add_argument("--text", default="")
    graph_control.add_argument("--revision", default="")
    graph_control.add_argument("--check", default="", help="new check command (amend-command)")
    graph_control.add_argument("--setup", default="", help="new setup command (amend-command)")
    graph_control.add_argument("--generate", default="", help="new generate command (amend-command)")
    graph_control.add_argument("--note", default="", help="why the command was changed (amend-command)")
    graph_control.add_argument(
        "--acknowledge-failure", action="store_true", default=False,
        dest="acknowledge_failure",
        help="confirm that side effects of the failed node have been inspected; "
             "required for rerun of a failed run",
    )

    read = add(sub.add_parser("artifacts", help="what a run left"))
    read.add_argument("--contents", action="store_true")

    board = add(sub.add_parser("leaderboard", help="the candidates by score"))
    board.add_argument("--limit", type=int, default=20)

    one = add(sub.add_parser("candidate", help="one candidate and its diff"))
    one.add_argument("candidate")

    steer = add(
        sub.add_parser("steer", help="what this run is for, and what has been said since")
    )
    steer.add_argument("--seed", default=None, help="the request this run is for")
    steer.add_argument("--note", default=None, help="something said while it runs")
    steer.add_argument(
        "--kind", default="info", choices=["info", "correction", "constraint"],
        help="a constraint must be respected; the others are context",
    )
    steer.add_argument(
        "--scope", default="product", choices=["product", "process"],
        help="product (default) notes become auditable requirements; process notes "
             "record run-level directives that are visible but not auditable",
    )
    steer.add_argument(
        "--source", default="user-relayed", choices=["user-relayed", "ceo"],
        help="who said this: user-relayed (default) for the person's words, ceo for "
             "the CEO's own observations; CEO notes are context, not auditable requirements",
    )
    steer.add_argument(
        "--verdict", action="append", default=[],
        help="record a one-use gate answer as <gate>=<value> (repeatable), scoped "
             "to the current declared review inputs; changed inputs require a fresh answer",
    )
    steer.add_argument(
        "--gates", action="store_true",
        help="also report the `user` gates this run is stopped at, and the verdicts "
             "each accepts, so a caller can offer the answer the gate routes on",
    )
    control = steer.add_mutually_exclusive_group()
    control.add_argument(
        "--pause", action="store_true",
        help="hold the run at its next generation boundary until resumed or stopped",
    )
    control.add_argument("--resume", action="store_true", help="release a paused run")
    control.add_argument(
        "--stop", action="store_true",
        help="end the run at its next boundary, recording that a person asked",
    )
    control.add_argument(
        "--keep", metavar="ID",
        help="follow this candidate up: it stays a parent even once a newer batch "
             "outscores it, and a rejection on it is cleared",
    )
    control.add_argument(
        "--reject", metavar="ID",
        help="stop searching around this candidate: it leaves the leaderboard, so it "
             "cannot be proposed from, reported best or adopted",
    )
    control.add_argument(
        "--adopt", action="store_true",
        help="write the run's current winner as a package now, without ending the run",
    )
    steer.add_argument(
        "--lineage", action="store_true",
        help="with --keep or --reject, act on the candidate's descendants too, so a "
             "whole branch is followed up or dropped",
    )
    steer.add_argument(
        "--as", dest="as_name", metavar="NAME",
        help="with --adopt, the package name to write (default: <package>_adopted)",
    )

    champion = add(
        sub.add_parser("best", help="the winner, with the state that scored it")
    )
    champion.add_argument("--state", action="store_true", help="include the full state")

    take = add(
        sub.add_parser("adopt", help="write the winner as a new package, input untouched")
    )
    take.add_argument("--as", dest="as_name", required=True, help="the new package name")
    take.add_argument(
        "--allow-overfit", action="store_true", dest="allow_overfit",
        help="adopt even though the winner's held-out audit says it is overfit; "
             "without this, adopting an overfit winner is refused",
    )

    add(sub.add_parser("stop", help="ask a run to stop"))

    # ── the rest of the system, reachable through the same wire ────────────
    # These forward to the entry points that already own the work, so the Web
    # view reaches package management and the authoring modes without a second
    # implementation. `--args` passes a caller's own flags through verbatim.
    packages = sub.add_parser("registry", help="list, verify, add, import, remove, enable or disable packages")
    packages.add_argument("registry_verb", choices=[
        "list", "show", "verify", "smoke", "tools", "add", "import", "remove", "enable", "disable",
    ])
    packages.add_argument("target", nargs="?", default="")
    packages.add_argument("--args", default="", help="extra registry flags, as a shell-quoted string")

    make = sub.add_parser("create", help="create a package of modes from a brief (model-driven)")
    make.add_argument("--name", default="")
    make.add_argument("--brief", default="")
    make.add_argument("--provider", default="", help="provider route for this authoring run")
    make.add_argument("--model", default="", help="model id for this authoring run")
    make.add_argument("--extends", default="")
    make.add_argument("--attempts", type=int, default=3)
    make.add_argument(
        "--data-node", action="store_true", default=False, dest="data_node",
        help="build the mode as a DataNode that iterates its subgraph per item",
    )
    make.add_argument("--task", default=None, metavar="TASK_REF", dest="task_ref",
                      help="the class of work: a Task the produced harness binds a data node "
                           "to, so the harness runs its subgraph once per item the Task lists")
    make.add_argument(
        "--single", action="store_true", default=False, dest="single_shot",
        help="build a harness for one run of one thing, with no Task binding; one of "
             "--task or --single is required",
    )
    make.add_argument("--workdir", default="", help="the run directory (default: .factory/create/<name>)")
    make.add_argument("--resume", default="", metavar="RUN_DIR", help="continue a stopped run from its directory")
    make.add_argument("--acknowledge-failure", action="store_true",
                       help="with --resume: go on from a failed boundary, or one whose last outcome is unknown, "
                            "once you have inspected it")
    make.add_argument(
        "--sample", action="append", default=[], metavar="NAME=VALUE",
        help="a knob value the validation run should use (repeatable), so the smoke "
             "exercises the input path rather than the built-in fixture",
    )
    make.add_argument(
        "--input", action="append", default=[], metavar="NAME",
        help="an input the produced mode must accept, named the way a knob or a file "
             "would be (repeatable); validation refuses a mode with no way to receive it",
    )
    make.add_argument(
        "--verdict", action="append", default=[], metavar="GATE=VALUE",
        help="answer a gate a person owns, as <gate>=<value> (repeatable); "
             "gate_strategy is one, so a headless create is told the answer",
    )

    revise = sub.add_parser("edit", help="revise a mode from a brief (model-driven)")
    revise.add_argument("--mode", default="")
    revise.add_argument("--name", default="")
    revise.add_argument("--brief", default="")
    revise.add_argument("--provider", default="", help="provider route for this authoring run")
    revise.add_argument("--model", default="", help="model id for this authoring run")
    revise.add_argument("--allow-regression", action="store_true",
                        help="skip original-mode execution and comparison; still validate the revision")
    revise.add_argument("--attempts", type=int, default=3)
    revise.add_argument("--workdir", default="", help="the run directory (default: .factory/edit/<name>)")
    revise.add_argument("--resume", default="", metavar="RUN_DIR", help="continue a stopped run from its directory")
    revise.add_argument("--acknowledge-failure", action="store_true",
                       help="with --resume: go on from a failed boundary, or one whose last outcome is unknown, "
                            "once you have inspected it")
    revise.add_argument(
        "--sample", action="append", default=[], metavar="NAME=VALUE",
        help="a knob value the validation run should use (repeatable)",
    )
    revise.add_argument(
        "--verdict", action="append", default=[], metavar="GATE=VALUE",
        help="answer a gate a person owns, as <gate>=<value> (repeatable)",
    )

    artifact = sub.add_parser("design", help="design and check a software artifact (model-driven)")
    artifact.add_argument("--name", default="")
    artifact.add_argument("--brief", default="")
    artifact.add_argument("--provider", default="", help="provider route for this authoring run")
    artifact.add_argument("--model", default="", help="model id for this authoring run")
    artifact.add_argument("--check", default="", help="the command that proves the artifact works")
    artifact.add_argument("--attempts", type=int, default=3)
    artifact.add_argument("--workdir", default="", help="the run directory (default: .factory/design/<name>)")
    artifact.add_argument("--resume", default="", metavar="RUN_DIR", help="continue a stopped run from its directory")
    artifact.add_argument("--acknowledge-failure", action="store_true",
                       help="with --resume: go on from a failed boundary, or one whose last outcome is unknown, "
                            "once you have inspected it")
    artifact.add_argument("--target", default="",
                          help="the git repository to change, in a worktree of its own (default: a new project)")
    artifact.add_argument("--base", default="", help="the commit, branch or tag the run starts from")
    artifact.add_argument("--setup", default="", help="a command run in the worktree before each check")
    from refactory_dsh import repo_change as _rc
    artifact.add_argument("--generate", default="",
                          help=_rc.knobs()["generate"].description)
    artifact.add_argument("--no-worktree", action="store_true",
                          help="write the artifact under .factory/artifacts instead of a worktree")
    artifact.add_argument("--no-pr", action="store_true",
                          help="keep the work on the run's local branch instead of opening a pull request")
    artifact.add_argument("--refine", default="", metavar="PR",
                          help="an open pull request to follow up (its URL or owner/repo#number); the run "
                               "starts from its branch and updates it")
    artifact.add_argument("--review-comments", action="store_true",
                          help="with --refine: add the pull request's reviews and comments to the steering")
    artifact.add_argument(
        "--verdict", action="append", default=[], metavar="GATE=VALUE",
        help="answer a gate a person owns, as <gate>=<value> (repeatable)",
    )

    return parser


def _resume_argv(args: argparse.Namespace) -> list[str]:
    """An authoring verb's ``--resume``, as the walker's mains take it (dm-160)."""
    forwarded = ["--resume", args.resume]
    if getattr(args, "acknowledge_failure", False):
        forwarded.append("--acknowledge-failure")
    for pair in args.verdict:
        forwarded += ["--verdict", pair]
    if args.provider:
        forwarded += ["--provider", args.provider]
    if args.model:
        forwarded += ["--model", args.model]
    return forwarded


def _forward(module: str, argv: list[str]) -> dict[str, Any]:
    """Run another entry point's ``main`` and return its output as this verb's payload.

    The console's contract is one JSON object on stdout, and the forwarded entry
    points print their own JSON. Capturing and re-parsing keeps that contract for a
    caller that only knows the console, and a non-JSON line is returned as text
    rather than lost — a forwarded command that half-printed before failing is
    exactly the case a caller needs to see.
    """
    # Capture at the process boundary, not just Python's sys.stdout: forwarded
    # commands spawn DSH with inherited file descriptors. redirect_stdout leaves
    # that child's prose and report on the console's machine-readable stdout.
    # The interactive child owns another CLI. Keep interruption on the same
    # graceful ownership chain instead of subprocess.run killing that owner.
    from refactory_dsh.owned_process import run_owned

    runner = run_owned if module == "interactive" else subprocess.run
    completed = runner(
        [sys.executable, "-m", f"refactory_dsh.{module}", *argv],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    code = completed.returncode
    text = completed.stdout.strip()
    stderr = completed.stderr.strip()
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        payload = {"output": text}
    if code != 0 and not (isinstance(payload, dict) and payload.get("error")):
        # The authoring entry points report a handled refusal as its own JSON on
        # *stderr* and print nothing on stdout. Capturing stdout alone made the
        # payload `{"output": ""}`, so the tool could only say "failed (exit 2)".
        payload = {
            "error": _error_from(stderr) or payload.get("output") or f"exited with {code}"
        }
    return {
        "verb": module, "exit_code": code, "ok": code == 0,
        "result": payload, "stderr": stderr,
    }


def _error_from(stderr: str) -> str:
    """The message inside a stderr line an entry point wrote, or the line itself.

    Those entry points print `{"error": "..."}` to stderr. Handing a caller that line
    verbatim shows it a JSON fragment where a sentence belongs, so the message inside
    is preferred; anything else on stderr is passed through as it stands.
    """
    text = stderr.strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(loaded, dict) and isinstance(loaded.get("error"), str):
            return loaded["error"].strip()
    return text


def _forward_error(payload: dict[str, Any]) -> str:
    """Why a forwarded entry point failed, preferring the words it chose."""
    result = payload.get("result")
    if isinstance(result, dict):
        if isinstance(result.get("error"), str) and result["error"].strip():
            return result["error"].strip()
        if isinstance(result.get("output"), str) and result["output"].strip():
            return result["output"].strip()
    if isinstance(payload.get("stderr"), str) and payload["stderr"].strip():
        return payload["stderr"].strip()
    return f"the {payload.get('verb')} entry point failed (exit {payload.get('exit_code')})"


def _root_for(args: argparse.Namespace, project: Path) -> Path:
    if not args.run:
        raise ConsoleError("this verb needs --run <run id or directory>")
    candidate_path = Path(args.run).expanduser()
    base = Path(args.runs_root).expanduser() if args.runs_root else project / RUNS_ROOT
    # Explicit paths name one directory. A bare id can name a cwd-relative run
    # or a run under runs_root; refuse ambiguity instead of steering the wrong one.
    separators = (os.sep,) + ((os.altsep,) if os.altsep else ())
    explicit_path = (
        candidate_path.is_absolute()
        or any(separator in args.run for separator in separators)
        or args.run in (".", "..")
    )
    candidates = [candidate_path] if explicit_path else [base / candidate_path]
    if not explicit_path and not args.runs_root:
        candidates.append(candidate_path)
    found = {path.resolve() for path in candidates if path.is_dir()}
    if len(found) > 1:
        raise ConsoleError(
            f"ambiguous run {args.run!r}: {', '.join(str(path) for path in sorted(found))}. "
            "Pass the absolute run directory returned by start, or --runs-root."
        )
    if not found:
        locations = ', '.join(str(path.resolve()) for path in candidates)
        raise ConsoleError(
            f"run {args.run!r} does not exist (checked {locations}). "
            "Pass the absolute run directory returned by start, or the same --runs-root. "
            "This action does not create a run."
        )
    return found.pop()


#: Where a route came from, in the words `refactory runtime` uses (dm-258).
ROUTE_SOURCES = {
    "explicit-launch": "flag",
    "selected-profile": "saved profile",
    "project-file": "project file",
    "environment": "environment",
    "stored-choice": "stored choice",
    "inherited-context": "inherited run context",
    "default": "none",
}


def model_route(project: Path, *, provider: str = "", model: str = "") -> dict[str, Any]:
    """The model a run in ``project`` would use, and where each part of it came from.

    The run's route comes from a flag, the environment, the project file or the stored
    choice, in that order; ``node_routes`` and ``role_routes`` then override it for the
    nodes and roles they name. Nothing here falls back to a DSH profile's default: a
    route nobody chose says ``none``.
    """
    from refactory_dsh.dsh_profile import stored_model_choice
    from refactory_dsh.worker_config import resolve

    selected = resolve(project, {key: value for key, value in (("provider", provider), ("model", model)) if value})
    settings, fields = selected["settings"], selected["provenance"]["fields"]

    def source(field: str) -> str:
        raw = str(fields.get(field) or "default")
        return ROUTE_SOURCES.get(raw.split(":", 1)[0], raw)

    overrides = [
        {"applies_to": f"{kind} {name}", "source": f"{kind} route",
         **{key: value for key, value in (route or {}).items() if key in ("worker_backend", "provider", "model")}}
        for kind, routes in (("node", settings.get("node_routes")), ("role", settings.get("role_routes")))
        for name, route in sorted((routes or {}).items())
        if isinstance(route, dict) and any(key in route for key in ("worker_backend", "provider", "model"))
    ]
    if settings["worker_backend"] == "openshell":
        # Out of the one choice's scope: the sandbox's only inference path (dm-258).
        sources = {"provider": "openshell gateway (claude-code provider)",
                   "model": "openshell gateway (claude-code provider)"}
    else:
        sources = {"provider": source("provider"), "model": source("model")}
    return {
        "worker_backend": settings["worker_backend"],
        "provider": settings["provider"],
        "model": settings["model"],
        "source": sources,
        "stored_choice": stored_model_choice(),
        "overrides": overrides,
    }


def runtime_context(project: Path, *, provider: str = "", model: str = "") -> dict[str, Any]:
    """Describe the existing nested model configuration without a model request.

    Only selected nonsecret fields cross this boundary. The readiness helper's
    detailed credential reference, parser errors, and launch environment do not.
    """
    from tempfile import TemporaryDirectory
    from refactory_dsh.dsh_profile import NO_MODEL_CHOICE_MESSAGE
    from refactory_dsh.model import DshModelClient
    from refactory_dsh.sandbox import model_readiness

    from refactory_dsh.model import half_route_refusal

    route = model_route(project, provider=provider, model=model)
    chosen = route["worker_backend"] != "dsh" or bool(route["provider"] and route["model"])
    client = DshModelClient(routes_resolved=True, worker_backend=route["worker_backend"],
                            provider=route["provider"], model=route["model"])
    context = client.context()
    refusal = half_route_refusal(route["provider"], route["model"]) if route["worker_backend"] == "dsh" else ""
    if not chosen:
        # Nothing to inspect: a run here would be refused before any model call, with
        # the words `doctor` and the run itself use.
        result: dict[str, Any] = {"ready": False, "status": "no-model-chosen"}
    else:
        try:
            with TemporaryDirectory(prefix="refactory-runtime-") as work:
                result = model_readiness(Path(work), model_context=context, require_ready=False)
        except Exception:
            # A failed inspector is not proof of a bad key. Never relay arbitrary
            # exception text from a custom runtime or secret-bearing configuration.
            result = {"ready": None, "status": "diagnostic-unavailable"}
    messages = {
        "configured": "Local credential configuration is present. Authentication, quota, and network access have not been tested.",
        "missing-credential": "Nested model credentials are missing. Save the provider key through Settings → Models in the DSH home shown before authoring.",
        "invalid-configuration": "Local model configuration is invalid. Inspect refactory doctor and the selected DSH profile/settings before authoring.",
        "native-auth-unverified": "Provider-native authentication has not been verified. Normal execution performs its own checks.",
        "diagnostic-unavailable": "This runtime could not be inspected. Readiness is unverified; normal execution performs its own checks.",
        "no-model-chosen": NO_MODEL_CHOICE_MESSAGE,
    }
    status = result.get("status")
    if status not in messages:
        status = "diagnostic-unavailable"
    credential = result.get("credential")
    from refactory_dsh.model import credential_source
    from refactory_dsh.worker_config import effective
    source = credential_source(context["dsh_home"], project)
    # The probe's refusal wins when it read the store: a missing or invalid credential is
    # a stated fact, not a guess. It does not model a Vertex or base-URL route, so a
    # source of that kind stands on its own.
    route_source = str(source.get("source") or "").startswith(
        ("Vertex", "base-url", "environment ("))
    authenticated = bool(source.get("configured")) and (
        route_source or status not in ("missing-credential", "invalid-configuration"))
    return {
        "worker_configuration": effective(project),
        "model_route": route,
        "project": str(project.resolve()),
        "dsh_home": context["dsh_home"],
        "nested": {
            "provider": route["provider"] or "(none chosen)",
            "model": route["model"] or "(none chosen)",
            "ready": result.get("ready") if status != "diagnostic-unavailable" else None,
            "status": status,
            "credential_configured": credential.get("configured") if isinstance(credential, dict) else None,
            "authenticated": authenticated,
            "authenticated_source": source.get("source"),
            "authenticated_checked": None if authenticated else source.get("checked"),
            "reasoning_effort": None,
            "reasoning_note": "The credential diagnostic does not resolve nested reasoning effort; the current session's effort is not proof of a child's selection.",
            "message": refusal or (
                f"A credential source is configured ({source['source']}); "
                "no authenticated request was made."
                if authenticated and status == "diagnostic-unavailable"
                else messages[status]),
        },
        "inheritance": "This readiness diagnostic describes the selected DSH context, not an executed worker. worker_configuration reports project worker defaults; prepared runs and inherited children retain resolved routes independently of the CEO conversation model.",
    }


def list_projects() -> list[dict[str, Any]]:
    """The registered projects, as the GUI's target picker needs them.

    Calls the same ``projects.load()`` the CLI's ``refactory projects`` uses, so the
    CLI and the GUI list the same projects.
    """
    from refactory_dsh import projects
    return projects.load()["projects"]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # The forwarding verbs carry their own addressing flags, so `--project` is
    # absent on them; a missing one is "the found root", not a crash.
    project = Path(getattr(args, "project", None)) if getattr(args, "project", None) else project_root()
    try:
        if args.verb == "runtime":
            payload: Any = runtime_context(project, provider=args.provider, model=args.model)
        elif args.verb == "help":
            from refactory_dsh.discovery import help_contract
            payload = help_contract(project, args.topic)
        elif args.verb == "docs":
            from refactory_dsh.discovery import docs
            payload = docs(page=args.page, query=args.query, section=args.section, offset=args.offset, limit=args.limit)
        elif args.verb == "workers":
            from refactory_dsh.worker_config import effective
            payload = effective(project)
        elif args.verb == "projects":
            payload = list_projects()
        elif args.verb == "modes":
            payload = catalogue(project=project)
        elif args.verb == "strategies":
            payload = level_two()
        elif args.verb == "runs":
            payload = runs(project=project, runs_root=args.runs_root)
        elif args.verb == "profiles":
            payload = profiles.list_profiles(project)
        elif args.verb == "profile-save":
            payload = profiles.save(project, args.name, args.spec)
        elif args.verb in ("start", "profile-resolve"):
            settings = {key: value for key, value in vars(args).items()
                        if (key in profiles.defaults() or key == "knob") and value is not None}
            settings = {**(args.settings_json or {}), **settings}
            if args.verb == "profile-resolve":
                payload = profiles.resolve(project, args.mode, name=args.profile,
                                           version=args.profile_version, overrides=settings)
            else:
                payload = start(args.mode, project=project, runs_root=args.runs_root,
                                run_id=args.run_id, profile=args.profile,
                                profile_version=args.profile_version, **settings)
        elif args.verb == "comparison":
            payload = comparison(_root_for(args, project))
        elif args.verb == "status" and args.follow:
            root = _root_for(args, project)
            # Each frame is a page, not the whole journal: the cursors the last
            # frame reported are where the next one starts, so a poll costs what
            # was appended since rather than everything ever written.
            progress_cursor = args.after
            activity_cursor = args.activity_after
            previous = None
            try:
                while True:
                    frame = status(root, after=progress_cursor, activity_after=activity_cursor,
                                   activity_limit=args.activity_limit)
                    progress_cursor = frame["cursor"]
                    activity_cursor = frame["activity"]["cursor"]
                    encoded = json.dumps(frame, sort_keys=True)
                    if encoded != previous:
                        print(encoded, flush=True)
                        previous = encoded
                    if frame["finished"] or frame["state"] in ("failed", "died", "halted"):
                        return 0
                    time.sleep(max(0.1, args.interval))
            except KeyboardInterrupt:
                return 0
        elif args.verb == "status":
            payload = status(_root_for(args, project), after=args.after,
                             activity_after=args.activity_after, activity_limit=args.activity_limit)
            from refactory_dsh import readable
            payload['text'] = readable.format_status(payload)
        elif args.verb == "graph-control":
            from refactory_dsh.graph_control import request
            try:
                if args.action == 'amend-command':
                    fields = [(f, getattr(args, f)) for f in ('check', 'setup', 'generate') if getattr(args, f)]
                    if len(fields) != 1:
                        raise ConsoleError('exactly one of --check, --setup, --generate is required')
                    field_name, field_value = fields[0]
                    payload = request(
                        _root_for(args, project), 'amend-command',
                        field=field_name, value=field_value, note=getattr(args, 'note', ''),
                        node=args.node,
                        acknowledge_failure=getattr(args, 'acknowledge_failure', False),
                    )
                else:
                    payload = request(
                        _root_for(args, project), args.action,
                        node=args.node, text=args.text, revision=args.revision,
                        acknowledge_failure=getattr(args, 'acknowledge_failure', False),
                    )
            except ValueError as error:
                raise ConsoleError(str(error)) from error
            from refactory_dsh import readable
            payload['text'] = readable.format_graph_control(payload)
        elif args.verb == "artifacts":
            payload = artifacts(_root_for(args, project), contents=args.contents)
        elif args.verb == "leaderboard":
            payload = leaderboard(_root_for(args, project), limit=args.limit)
        elif args.verb == "candidate":
            payload = candidate(_root_for(args, project), args.candidate)
        elif args.verb == "steer":
            # One flag, one act: the mutually exclusive group already makes "pause and
            # adopt" unwritable, so this only names the act the flags selected.
            asked = (
                "pause" if args.pause else "resume" if args.resume
                else "stop" if args.stop else "keep" if args.keep
                else "reject" if args.reject else "adopt" if args.adopt else None
            )
            payload = _steer(
                _root_for(args, project),
                seed=args.seed, note=args.note, kind=args.kind,
                scope=args.scope, source=args.source,
                verdicts=args.verdict, with_gates=args.gates,
                control=asked,
                target=args.keep or args.reject,
                lineage=args.lineage,
                as_name=args.as_name,
                project=project,
            )
            payload['recorded_verdicts'] = list(args.verdict)
            payload['recorded_note'] = args.note
            payload['recorded_kind'] = payload.pop('_effective_kind', args.kind)
            from refactory_dsh import readable
            payload['text'] = readable.format_steer(payload)
        elif args.verb == "best":
            payload = _best_payload(_root_for(args, project), keep_state=args.state)
        elif args.verb == "adopt":
            from refactory_dsh.adopt import adopt as adopt_winner

            run_root = _root_for(args, project)
            audit = holdout(run_root) or {}
            payload = adopt_winner(
                run_root, as_name=args.as_name, project=project,
                allow_overfit=args.allow_overfit,
            )
            payload["holdout"] = audit or None
        elif args.verb == "stop":
            payload = stop(_root_for(args, project))
        elif args.verb == "registry":
            extra = shlex.split(args.args) if args.args else []
            forwarded = [args.registry_verb, "--json"]
            if args.target:
                forwarded.append(args.target)
            forwarded += extra
            payload = _forward("registry", forwarded)
        elif args.verb == "run":
            forwarded = [args.mode, "--headless"]
            if args.resume:
                forwarded.append("--resume")
            if args.intent:
                forwarded += ["--intent", args.intent]
            if args.workdir:
                forwarded += ["--workdir", args.workdir]
            if args.source:
                forwarded += ["--source", args.source]
            if args.source_format:
                forwarded += ["--source-format", args.source_format]
            for pair in args.knob:
                forwarded += ["--knob", pair]
            for pair in args.verdict:
                forwarded += ["--approve", pair]
            if args.timeout:
                forwarded += ["--timeout", str(args.timeout)]
            if args.provider:
                forwarded += ["--provider", args.provider]
            if args.model:
                forwarded += ["--model", args.model]
            for key in ("worker_backend", "effort", "permission_mode", "max_elapsed_s", "max_tokens", "validation_reserve_s", "validation_reserve_tokens"):
                value = getattr(args, key, None)
                if value is not None and value != "":
                    forwarded += ["--" + key.replace("_", "-"), str(value)]
            for key in ("role_routes", "node_routes"):
                if getattr(args, key, None):
                    forwarded += ["--" + key.replace("_", "-"), json.dumps(getattr(args, key))]
            payload = _forward("interactive", forwarded)
            # The captured child output is diagnostic text; the run directory is
            # the durable structured answer:
            # where it is, how it ended and what it recorded -- and it is the
            # directory the runner itself chose, so this reads the run it made rather
            # than a second guess at where a run would have gone.
            if args.workdir:
                root = Path(args.workdir)
            else:
                from refactory_dsh.interactive import default_workdir

                root = default_workdir(args.mode)
            if root.is_dir():
                recorded_facts = _read_json(root / "facts.json")
                payload["run"] = {
                    **status(root),
                    "mode": args.mode,
                    "facts": recorded_facts,
                    "facts_text": _facts_text(recorded_facts),
                }
        elif args.verb == "open":
            model_timeout_s = getattr(args, 'model_timeout_s', None)
            if model_timeout_s is not None and not (1 <= model_timeout_s <= 7200):
                raise ConsoleError(f"--model-timeout must be between 1 and 7200 seconds, got {model_timeout_s}")
            mc = {key: getattr(args, key) for key in ("worker_backend", "provider", "model", "effort", "permission_mode", "role_routes", "node_routes", "provider_config") if getattr(args, key)}
            if model_timeout_s is not None:
                mc["timeout_s"] = model_timeout_s
            payload = _open_mode(
                args.mode,
                project=project,
                workdir=args.workdir,
                knobs=args.knob,
                intent=args.intent,
                model_context=mc,
                resource_limits={key: getattr(args, key) for key in ("max_elapsed_s", "max_tokens", "validation_reserve_s", "validation_reserve_tokens") if getattr(args, key) is not None},
            )
        elif args.verb == "create":
            forwarded = ["--name", args.name, "--brief", args.brief, "--attempts", str(args.attempts)]
            if args.workdir:
                forwarded += ["--workdir", args.workdir]
            if args.resume:
                forwarded += ["--resume", args.resume]
            for pair in args.sample:
                forwarded += ["--sample", pair]
            for name in args.input:
                forwarded += ["--input", name]
            if args.extends:
                forwarded += ["--extends", args.extends]
            if args.data_node:
                forwarded.append("--data-node")
            if args.task_ref:
                forwarded += ["--task", args.task_ref]
            if args.single_shot:
                forwarded.append("--single")
            for pair in args.verdict:
                forwarded += ["--verdict", pair]
            if args.provider:
                forwarded += ["--provider", args.provider]
            if args.model:
                forwarded += ["--model", args.model]
            if args.resume:
                # A resume keeps what its run was saved with: only what the person gave
                # goes on, and the walker's main refuses anything that would change it.
                forwarded = _resume_argv(args)
            payload = _forward("create", forwarded)
            review = _design_review(project, args.name)
            if review:
                # The approaches it considered, not just the package it produced.
                payload["design_review"] = review
        elif args.verb == "edit":
            forwarded = [
                "--mode", args.mode, "--name", args.name, "--brief", args.brief,
                "--attempts", str(args.attempts),
            ]
            if args.workdir:
                forwarded += ["--workdir", args.workdir]
            if args.resume:
                forwarded += ["--resume", args.resume]
            for pair in args.sample:
                forwarded += ["--sample", pair]
            if args.allow_regression:
                forwarded.append("--allow-regression")
            for pair in args.verdict:
                forwarded += ["--verdict", pair]
            if args.provider:
                forwarded += ["--provider", args.provider]
            if args.model:
                forwarded += ["--model", args.model]
            if args.resume:
                # A resume keeps what its run was saved with: only what the person gave
                # goes on, and the walker's main refuses anything that would change it.
                forwarded = _resume_argv(args)
            payload = _forward("edit", forwarded)
        elif args.verb == "design":
            forwarded = ["--name", args.name, "--brief", args.brief, "--attempts", str(args.attempts)]
            if args.workdir:
                forwarded += ["--workdir", args.workdir]
            if args.resume:
                forwarded += ["--resume", args.resume]
            if args.check:
                forwarded += ["--check", args.check]
            for flag, value in (("--target", args.target), ("--base", args.base), ("--setup", args.setup),
                                ("--generate", args.generate), ("--refine", args.refine)):
                if value:
                    # Absolute, because the run resolves it from its own directory.
                    forwarded += [flag, str(Path(value).expanduser().resolve()) if flag == "--target" else value]
            for flag, on in (("--no-worktree", args.no_worktree), ("--no-pr", args.no_pr),
                             ("--review-comments", args.review_comments)):
                if on:
                    forwarded += [flag]
            for pair in args.verdict:
                forwarded += ["--verdict", pair]
            if args.provider:
                forwarded += ["--provider", args.provider]
            if args.model:
                forwarded += ["--model", args.model]
            if any(pair.partition("=")[0] == "gate_strategy" for pair in args.verdict):
                # The forwarded run's stderr is captured until it ends, so a gate answered
                # up front is announced here, before anything is built (#350).
                from refactory_dsh import delivery

                delivery.announce({"name": args.name, "brief": args.brief, "target": args.target,
                                   "base": args.base, "refine": args.refine, "worktree": not args.no_worktree,
                                   "pr": not args.no_pr}, Path(args.workdir or f".factory/design/{args.name}"))
            if args.resume:
                # A resume keeps what its run was saved with: only what the person gave
                # goes on, and the walker's main refuses anything that would change it.
                forwarded = _resume_argv(args)
            payload = _forward("design", forwarded)
        else:  # pragma: no cover - argparse rejects an unknown verb first
            raise ConsoleError(f"unknown verb {args.verb!r}")
    except (ValueError, ConsoleError, LoopError, PackageError, AdoptionError, SteeringError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stdout)
        return 2
    except ModelChoiceError as error:  # a run with no model to use (dm-258)
        print(json.dumps({"error": str(error)}), file=sys.stdout)
        return 1
    if isinstance(payload, dict) and payload.get("ok") is False:
        # The forwarded call failed and said why; the caller gets the reason at the top
        # level as well, because that is where every console consumer looks for it.
        print(json.dumps({**payload, "error": _forward_error(payload)}, indent=2, default=str))
        return 2
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
