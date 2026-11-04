"""Evaluate a candidate on DSH instead of the Python walker.

The walker is a debug runtime: it makes the outer loop fast, and fast iteration is
what found every bug so far. DSH is the runtime that ships. Both score a mode the
same way, and the equivalence tests are what keep that true.

The boundary is deliberate and is the whole point:

- **DSH produces facts on disk.** The lowered graph's nodes run and leave a
  `facts.json` behind, because the IR's nodes do not have facts.
- **Python turns the facts into a number.** `score()` stays a method on the class,
  out of the graph, so no proposal can reach it.

That is why `fact_produced` is the runtime boundary contract and not only a safety
check: it says "when the run ends, these facts exist", which is exactly what is
needed to score a run that DSH executed.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Callable

from refactory_dsh import lowering
from refactory_dsh.nn import RunResult
from refactory_dsh.paths import REPO_ROOT, dsh_command, require_build, venv_bin
from refactory_dsh.run import FACTS

#: What the CEO is asked to do. A deterministic mode needs the CEO only to advance
#: the cursor; the bundle mounts the spine with `executeCommands: true`, so the
#: nodes themselves run in-process.
TASK = "Read current user steering before advancing; honor requested pause boundaries, then use wf_next to execute the graph."

#: Wall-clock ceiling for one inner run on DSH.
DEFAULT_TIMEOUT_S = 900


@dataclass
class DshRun:
    """What a DSH evaluation produced."""

    facts: dict[str, Any]
    workdir: Path
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    best_score: float | None = None


#: Spine runs in flight in this process, so a stop that reaches the main thread
#: while pool threads run them (``Run.evaluate_many``) can shut them down: they
#: start in their own session, and only a main-thread run installs the SIGTERM
#: handler that would unwind them.
_TRACKED_RUNS: set[subprocess.Popen] = set()
_TRACKED_LOCK = threading.Lock()


def track_run(process: subprocess.Popen) -> None:
    """Track ``process`` as an in-flight spine run (see ``shutdown_tracked_runs``)."""
    with _TRACKED_LOCK:
        _TRACKED_RUNS.add(process)


def untrack_run(process: subprocess.Popen) -> None:
    with _TRACKED_LOCK:
        _TRACKED_RUNS.discard(process)


def shutdown_tracked_runs() -> int:
    """Ask every tracked spine run to shut down and wait for it; return how many.

    Each gets SIGTERM (its own disposal path), then the shutdown grace; one
    still running after it is killed.
    """
    from refactory_dsh.owned_process import SHUTDOWN_GRACE_S
    with _TRACKED_LOCK:
        running = [proc for proc in _TRACKED_RUNS if proc.poll() is None]
    for proc in running:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + SHUTDOWN_GRACE_S
    for proc in running:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    return len(running)


def run_on_dsh(
    state: dict[str, Any],
    workdir: Path,
    *,
    name: str = "mode",
    timeout_s: int = DEFAULT_TIMEOUT_S,
    repo_root: Path | None = None,
    on_output: Callable[[str], None] | None = None,
    model: Any = None,
    item_slots_dir: str | None = None,
    item_slots: int | None = None,
) -> DshRun:
    """Lower ``state``, run it on DSH, and read back the facts it produced.

    ``on_output`` hands each line of the child's output to a watcher as it arrives.
    It is not a replacement for capturing: the same lines are still collected, and
    the runner's final JSON object is still parsed out of them, because
    ``best_score`` comes from there.
    """
    import time

    from refactory_dsh import dsh_profile

    # DSH's session header requires an absolute cwd, and a run root spelled
    # relatively (the CLI's default) would otherwise fail every candidate with
    # "session header cwd must be an absolute path". Resolved here, at the one
    # boundary that hands the path to DSH.
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    context = model.context() if callable(getattr(model, "context", None)) else None
    selected_home = dsh_profile.model_home(context.get("dsh_home") if context else None)
    if context:
        context = {**context, "dsh_home": str(selected_home)}
    _dsh_profile_name = "graph"
    _dsh_pin_checkout: Path | None = None
    try:
        _run_config_path = workdir / ".refactory" / "run-config.json"
        if _run_config_path.exists():
            import refactory_dsh.pin as _pin_mod
            _dsh_profile_name = _pin_mod.driver_profile(workdir)
            if _dsh_profile_name.startswith("graph-pin-"):
                _rc_data = json.loads(_run_config_path.read_text())
                _pco = _rc_data.get("driver", {}).get("pin_checkout")
                if _pco:
                    _dsh_pin_checkout = Path(_pco)
    except Exception:
        pass
    problem = dsh_profile.requirement(name=_dsh_profile_name, home=selected_home, checkout=_dsh_pin_checkout)
    if problem is not None:
        # Checked here rather than left to the child: DSH's own error is about a
        # missing profile, which reads as a packaging bug. The profile is machine
        # state, and this is the one place that knows the command that creates it.
        return DshRun(facts={}, workdir=workdir, returncode=2, stderr=problem)

    require_build(root=Path(context["runtime_root"]) if context else None, check_only=True)

    graphs = workdir / "graphs"
    # The steering layer, on the runtime that ships. A run a person steered has a
    # ledger in its work directory; folding it in here is what makes the CEO's agent
    # nodes carry what was said, and what a resumed run replays.
    from refactory_dsh import steering, workers
    budget = getattr(model, "_budget", None)
    if budget is not None:
        budget.checkpoint()

    lowering.write_mode_graph(
        state,
        graphs,
        name=name,
        transform=lambda workflow: workers.lower_workflow(steering.for_run(workflow, workdir), workdir, context, lowering.node_declarations(state)),
    )
    # The IR cannot carry the facts a node produces, so they travel in the run
    # directory and the runner reads them from there.
    lowering.write_declarations(state, workdir)

    command = [
        *dsh_command(root=Path(context["runtime_root"]) if context else None, cli=context.get("dsh_cli") if context else None),
        "--profile", _dsh_profile_name, "run",
        "--mode", name,
        "--task", TASK,
        "--budget", "1",
        "--graphs", str(graphs),
        "--workdir", str(workdir),
        "--python-bin", str(venv_bin()),
        # The knob environment is already the JSON string `--knobs` wants; dumping
        # it again makes the node runner read a string where it needs an object,
        # and every node falls back to its op's default.
        "--knobs", lowering.knob_environment(state)[lowering.KNOBS_ENV],
    ]
    env = {**os.environ, "DSH_HOME": str(selected_home),
           "REFACTORY_MODEL_DSH_HOME": str(selected_home)}
    if item_slots_dir is not None and item_slots is not None:
        env["REFACTORY_ITEM_SLOTS_DIR"] = str(item_slots_dir)
        env["REFACTORY_ITEM_SLOTS"] = str(item_slots)
    env.update(workers.environment(context))
    telemetry = workers.GraphTelemetry(workdir, context, budget) if context else None
    if telemetry is not None:
        command = telemetry.command(command)
        env.update(telemetry.environment())
    if context:
        if context.get("budget_ledger"):
            env["REFACTORY_BUDGET_LEDGER"] = str(context["budget_ledger"])
        if context.get("worker_backend", "dsh") == "dsh":
            if context.get("provider"):
                command += ["--provider", context["provider"]]
            if context.get("model"):
                command += ["--model", context["model"]]
    else:
        # No client was given, but a CEO that runs still runs on the one choice, never
        # on the graph profile's default; with no choice the graph runner refuses (dm-258).
        stored = dsh_profile.stored_model_choice(selected_home)
        if stored["provider"]:
            command += ["--provider", stored["provider"], "--model", stored["model"]]
    started = time.time()
    from refactory_dsh.owned_process import shutdown, _termination_as_exit
    with _termination_as_exit():
        process = subprocess.Popen(command, cwd=str(repo_root or REPO_ROOT), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=os.name == "posix")
        track_run(process)
        out_lines: list[str] = []
        err_lines: list[str] = []

        def _pump(stream: Any, sink: list[str]) -> None:
            for line in stream:
                sink.append(line)
                if on_output is not None:
                    on_output(line)

        readers = [Thread(target=_pump, args=(process.stdout, out_lines), daemon=True),
                   Thread(target=_pump, args=(process.stderr, err_lines), daemon=True)]
        for reader in readers:
            reader.start()
        try:
            deadline = time.monotonic() + timeout_s
            while True:
                if telemetry is not None:
                    telemetry.checkpoint()
                if budget is not None:
                    budget.checkpoint()
                if callable(getattr(model, "_raise_if_cancelled", None)):
                    model._raise_if_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_s)
                try:
                    returncode = process.wait(timeout=min(remaining, 0.1))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException as error:
            try:
                shutdown(process)
            finally:
                untrack_run(process)
            for reader in readers:
                reader.join(timeout=5)
            if telemetry is not None:
                telemetry.finish("timed_out" if isinstance(error, subprocess.TimeoutExpired) else (
                    "budget_exhausted" if type(error).__name__ == "BudgetExhausted" else "cancelled"), str(error))
            if isinstance(error, subprocess.TimeoutExpired):
                return DshRun(facts={}, workdir=workdir, returncode=-1,
                              stderr=f"dsh timed out after {timeout_s}s; owned runtime shutdown requested",
                              duration_s=time.time() - started)
            raise
        untrack_run(process)
        for reader in readers:
            reader.join()
        stdout, stderr = "".join(out_lines), "".join(err_lines)
        try:
            if telemetry is not None:
                telemetry.checkpoint(final=True)
        except BaseException as error:
            if telemetry is not None:
                telemetry.finish("budget_exhausted", str(error))
            raise
        if telemetry is not None:
            telemetry.finish("completed" if returncode == 0 else "failed")

    payload = _last_json_object(stdout)
    facts_path = workdir / FACTS
    facts: dict[str, Any] = {}
    if facts_path.exists():
        try:
            loaded = json.loads(facts_path.read_text())
            facts = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            facts = {}
    return DshRun(
        facts=facts,
        workdir=workdir,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_s=round(time.time() - started, 4),
        best_score=(
            float(payload["best_score"])
            if isinstance(payload.get("best_score"), (int, float))
            else None
        ),
    )


def _last_json_object(text: str) -> dict[str, Any]:
    """The runner prints one pretty-printed object; take the last one it printed."""
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            loaded = json.loads(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def evaluate_on_dsh(
    mode: Any, state: dict[str, Any], workdir: Path, *, timeout_s: int = DEFAULT_TIMEOUT_S,
    on_output: Callable[[str], None] | None = None,
    model: Any = None,
    item_slots_dir: str | None = None,
    item_slots: int | None = None,
) -> RunResult:
    """One forward pass on DSH, scored by the mode's own `score()`.

    A run that stopped before the graph finished is a **failure naming why**, not a
    score of 0. The distinction matters most at a `user` gate: the runtime refuses to
    route one nobody answered, and without this the refusal would look like a run that
    merely scored badly.
    """
    name = f"{type(mode).__module__.split('.')[-2]}-{mode.name}"
    run = run_on_dsh(state, workdir, name=name, timeout_s=timeout_s, on_output=on_output,
                     **({"model": model} if model is not None else {}),
                     **({"item_slots_dir": item_slots_dir, "item_slots": item_slots}
                        if item_slots_dir is not None else {}))
    # The DSH evaluator's run end (a spine run it launched and waited for): the
    # same cleanups the walker runs when its own run ends.
    from refactory_dsh.nn.executor import _run_end_cleanup
    _run_end_cleanup(Path(run.workdir))
    if run.returncode != 0 or not run.facts:
        detail = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "no output"
        return RunResult(
            facts=run.facts,
            duration_s=run.duration_s,
            exit_ok=False,
            error=f"dsh run failed (exit {run.returncode}): {detail[:300]}",
            workdir=run.workdir,
        )
    status = _spine_status(run.workdir)
    if status.get("status") not in ("", "done"):
        detail = str(status.get("error") or status.get("status"))
        return RunResult(
            facts=run.facts,
            duration_s=run.duration_s,
            exit_ok=False,
            error=f"the graph did not finish: {detail[:300]}",
            workdir=run.workdir,
        )
    result = RunResult(facts=run.facts, duration_s=run.duration_s, workdir=run.workdir)
    try:
        result.score = float(mode.score(result))
    except Exception as exc:
        result.exit_ok = False
        result.error = f"score() raised {type(exc).__name__}: {exc}"
    return result


def _spine_status(workdir: Path) -> dict[str, Any]:
    """What the spine recorded about where the graph got to, if it recorded anything."""
    path = Path(workdir) / ".refactory" / "spine-status.json"
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
