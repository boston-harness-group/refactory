"""Run one mode with a person in the loop, on the runtime that ships.

Level 3. The outer loop *searches* a mode (level 2); this is a person **using** one.
The mode's graph is lowered with the run's intent ledger folded into every agent
node, the CEO drives it on DSH, and a `user` gate is put to the person at the
terminal by the runtime — not by the model, which cannot invent an answer it cannot
give.

    python -m refactory_dsh.interactive review_pack/review \
        --intent "draft a note about circle packing and let me approve it"

The answer is written into the run's ledger, so `console steer --run <workdir>`
shows its history. Each later interactive visit requires a fresh answer. Nothing here changes
the mode: the graph is the author's, plus what the person said.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from refactory_dsh import lowering, registry, steering
from refactory_dsh.dsh_runtime import TASK
from refactory_dsh.paths import REPO_ROOT, dsh_command, require_build, venv_bin, project_root
from refactory_dsh.run_ownership import ownership
from refactory_dsh.owned_process import run_owned

#: Where an interactive run lives when no work directory is named.
DEFAULT_ROOT = Path(".factory/interactive")

#: Wall-clock ceiling is the person's, so there is none: the run waits for them.


def with_knobs(state: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """One state with `NAME=VALUE` knob overrides applied.

    The CLI's spelling for something a person changes about *this* run: which
    package to create, what to check, how many attempts, and the inputs the mode
    operates on. Applied through `nn.edits.configure` rather than the editor's
    `set_knob`: a wrong name is still refused by name and an out-of-range value by
    range, but a *frozen* knob is settable — frozen means the search may not move it,
    not that the person running the mode may not say what it is. Refusing that made
    every input-taking mode unusable from the CLI (`--knob source_root=…` failed with
    "knob 'source_root' is frozen and cannot be moved").
    """
    from refactory_dsh import nn

    for pair in overrides:
        name, separator, value = str(pair).partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not name:
            raise ValueError(f"--knob takes NAME=VALUE, not {pair!r}")
        state = nn.edits.configure(state, name, value)
    return state


def calls_models(state: Any) -> bool:
    """Whether a mode's graph has an agent node anywhere, nested bodies included."""
    if isinstance(state, dict):
        return state.get("kind") == "agent" or any(calls_models(value) for value in state.values())
    if isinstance(state, list):
        return any(calls_models(value) for value in state)
    return False


def require_route(model_context: dict[str, Any], *, interactive: bool = False) -> dict[str, Any]:
    """``model_context`` with a complete DSH route, or ``ModelChoiceError`` (dm-258).

    A route a flag, the environment, the project file or the stored choice completed
    passes unchanged; an empty one gets the visible first-run choice or is refused.
    Another worker backend keeps its native settings.
    """
    if model_context.get("worker_backend", "dsh") != "dsh":
        return model_context
    from refactory_dsh.model import complete_route

    provider, model = complete_route(str(model_context.get("provider") or ""), str(model_context.get("model") or ""),
                                     interactive=interactive, home=model_context.get("dsh_home") or None)
    return {**model_context, "provider": provider, "model": model}


#: What an OpenShell run says about its model, once, at launch (dm-258).
OPENSHELL_MODEL_NOTE = (
    "refactory: an openshell run's model is the OpenShell gateway's verified claude-code "
    "provider, inside the sandbox; the stored model choice does not apply to it"
)


def ceo_route(model_context: dict[str, Any], *, interactive: bool = False, required: bool = True) -> tuple[str, str]:
    """The DSH route the CEO runs on, passed to it as ``--provider``/``--model``.

    A DSH run's own route. With an external worker backend (Codex, Claude Code) the
    workers keep their native settings, but the CEO is still a local DSH agent, so it
    runs on the one choice rather than on the graph profile's default.

    An OpenShell run is the exception, and out of the one choice's scope: its whole
    graph command runs in the sandbox, whose only inference path is the gateway's
    `claude-code` provider (verified at dispatch; credentials never enter the
    sandbox). No route is passed in, and none can fall back silently: the dispatch
    refuses a gateway without that provider, and the graph runner refuses a CEO
    launched with no route.
    """
    backend = model_context.get("worker_backend", "dsh")
    if backend == "openshell":
        return "", ""
    if backend == "dsh":
        return str(model_context.get("provider") or ""), str(model_context.get("model") or "")
    from refactory_dsh.dsh_profile import stored_model_choice
    from refactory_dsh.model import complete_route

    home = model_context.get("dsh_home") or None
    stored = stored_model_choice(home)
    if stored["provider"] or not required:
        return stored["provider"], stored["model"]
    return complete_route("", "", interactive=interactive, home=home)


def prepare(
    identifier: str,
    *,
    workdir: Path,
    intent: str = "",
    project: Path | None = None,
    knobs: list[str] | None = None,
    model_context: dict[str, Any] | None = None,
    resource_limits: dict[str, Any] | None = None,
    resume: bool = False,
    headless: bool = False,
) -> tuple[str, Path]:
    """Lower the mode for a person to use, and leave the run ready to start.

    Returns the graph name the profile loads and the directory it lives in. The
    ledger is seeded **before** the lowering, because the lowering is what folds it
    into the agent prompts; seeding afterwards would produce a run nobody steered.
    """
    workdir = Path(workdir).expanduser().resolve()
    project = Path(project or project_root())
    if not resume:
        if "/" not in identifier:
            raise ValueError(f"{identifier!r} is not a <package>/<mode> identifier")
        Mode = registry.resolve(project, identifier)[1]
        mode = Mode()
        state = with_knobs(mode.state_dict(), list(knobs or []))
        # A required knob's default is only a placeholder. A headless run has nobody to
        # give the value later, so one with no value, or an empty or blank one, is refused
        # here, before ownership creates the run directory. A session a person opens may
        # start without it -- the idea comes in the conversation -- so there only a value
        # given explicitly blank is refused. Every mode that declares one is covered.
        from refactory_dsh.brief_guard import check_required_knobs
        from refactory_dsh.nn.validate import declared_knobs

        values = {name: entry["value"] for name, entry in (state.get("knobs") or {}).items()}
        given = {str(part).partition("=")[0].strip() for part in knobs or [] if "=" in str(part)}
        violations = [(name, message) for name, message in check_required_knobs(declared_knobs(state), values)
                      if headless or name in given]
        if violations:
            raise ValueError(f"refactory: {violations[0][1]}")
        from refactory_dsh.model import DshModelClient
        routing_context = dict(model_context or {})
        if resource_limits:
            routing_context.pop('budget_ledger', None)
        # Persist a concrete policy before lowering, including console/API opens. A run
        # whose workers call a model has its route complete before anything is written:
        # every lowered node then names the provider and model it runs on (dm-258).
        model_context = DshModelClient(project=project, **routing_context).context()
        if calls_models(state):
            model_context = require_route(model_context)
    with ownership(workdir):
        if resume:
            if knobs or intent or model_context:
                raise ValueError('resume retains saved knobs, instructions and route; changes require explicit invalidation')
            from refactory_dsh.recovery import prepare as recover
            name = identifier.replace('/', '-')
            return name, recover(workdir, name, resource_limits=resource_limits)
        package, mode_name = identifier.split("/", 1)

        # Absolute, because this directory becomes DSH's session cwd and DSH refuses a
        # relative one: `interactive <mode>` with no `--workdir` used to fail before the
        # graph started, on the default root that is relative by design.
        workdir.mkdir(parents=True, exist_ok=True)
        from refactory_dsh import invocations, workers
        assert model_context is not None
        if resource_limits:
            from refactory_dsh.budgets import LEDGER, ResourceLimits
            ResourceLimits.from_dict(resource_limits)
            model_context['budget_ledger'] = str(workdir / LEDGER)

        name = f"{package}-{mode_name}"
        from refactory_dsh.run_preparation import prepare_run
        graphs = prepare_run(
            state, workdir,
            invocation_name=name,
            mode_id=identifier,
            intent=intent.strip() or f"Run {identifier}.",
            transform=lambda workflow, stage_dir=None: workers.lower_workflow(
                steering.for_run(workflow, stage_dir or workdir),
                workdir, model_context, lowering.node_declarations(state)),
            runtime="dsh",
            command="refactory ceo",
            resume_command=f"refactory graph-control resume --run {workdir}",
            worker_backend=model_context.get("worker_backend", "dsh"),
            resource_limits=resource_limits,
            project=project,
        )
        import json
        from refactory_dsh.recovery import configuration_fingerprints

        owner_argv = ['-m', 'refactory_dsh.interactive', identifier]
        (workdir / '.refactory/run-config.json').write_text(json.dumps({'mode': identifier,
            'project': str(project.resolve()), 'knobs': knobs or [], 'model_context': model_context,
            'resource_limits': resource_limits,
            'owner_argv': owner_argv,
            'configuration_fingerprints': configuration_fingerprints(workdir, identifier, project)}))
        from refactory_dsh import pin
        pin.record_driver(workdir / '.refactory/run-config.json')
        return name, graphs


def command(
    identifier: str,
    *,
    workdir: Path,
    graphs: Path,
    name: str,
    knobs: list[str] | None = None,
    project: Path | None = None,
    provider: str = "",
    model: str = "",
    headless: bool = False,
) -> list[str]:
    """The DSH run whose stdin is the person's terminal.

    The run is a live session by default: the graph profile is asked for `--chat`, so
    each of the CEO's turns prints as it happens and a line typed between turns reaches
    it. ``headless`` builds exactly the one-shot argv this command has always built —
    the flag is *left out*, not set to false — because `--headless` is a promise to a
    script that the run it gets is the run it got before.
    """
    _, mode_name = identifier.split("/", 1)
    _profile = "graph"
    try:
        from refactory_dsh import pin as _pin_mod
        _profile = _pin_mod.driver_profile(workdir)
    except Exception:
        pass
    argv = [
        *dsh_command(),
        "--profile", _profile, "run",
        "--mode", name,
        "--task", TASK,
        "--budget", "1",
        "--graphs", str(graphs),
        # DSH takes this as the session cwd and refuses a relative one.
        "--workdir", str(Path(workdir).expanduser().resolve()),
        "--python-bin", str(venv_bin()),
        # Already the JSON string `--knobs` wants; see `run_on_dsh`.
        "--knobs", lowering.knob_environment(
            with_knobs(
                registry.resolve(Path(project or "."), identifier)[1]().state_dict(),
                list(knobs or []),
            )
        )[lowering.KNOBS_ENV],
    ]
    # The graph profile takes these for the CEO agent it creates; the mode's own
    # agent nodes run through that same agent, so this is *the* model a run uses.
    if provider:
        argv += ["--provider", provider]
    if model:
        argv += ["--model", model]
    # A live session is the default for a person at a terminal. `--headless` is the
    # script's run, so it must not carry the flag that starts a conversation.
    if not headless:
        argv.append("--chat")
    return argv


def default_workdir(identifier: str) -> Path:
    """The run directory a mode gets when the caller names none.

    One owner for the default, because a reader that has to find the run afterwards —
    the console attaching the run to its JSON, a person looking for the evidence —
    must compute the same directory this run used.
    """
    return (DEFAULT_ROOT / identifier.replace("/", "-")).expanduser().resolve()


def run(
    identifier: str,
    *,
    workdir: Path | None = None,
    source: str = "",
    source_format: str = "directory",
    intent: str = "",
    project: Path | None = None,
    timeout_s: int | None = None,
    model_timeout_s: int | None = None,
    knobs: list[str] | None = None,
    approve: list[str] | None = None,
    provider: str = "",
    model: str = "",
    headless: bool = False,
    worker_backend: str = "",
    effort: str = "",
    permission_mode: str = "",
    role_routes: dict[str, Any] | None = None,
    node_routes: dict[str, Any] | None = None,
    model_context: dict[str, Any] | None = None,
    resource_limits: dict[str, Any] | None = None,
    resume: bool = False,
) -> int:
    """Lower, then run the mode under the CEO with the terminal attached.

    ``approve`` states a gate's answer before the run starts. It explicitly authorizes that gate throughout this run, unlike a one-visit
    terminal answer. It is recorded in the same ledger, so a run nobody is
    sitting at can still be given a person's decision -- explicitly, by name.

    A live session is the default: the CEO's turns print as they happen and a line
    typed between them reaches it. ``headless`` asks for the older one-shot run, which
    is what a script or CI wants.
    """
    if workdir is None:
        workdir = default_workdir(identifier)
    workdir = Path(workdir).expanduser().resolve()
    # Pre-flight: both checks raise on failure before ownership() creates
    # the run directory, so a refused start leaves nothing behind.
    # On resume and respawn, use check_only so a stale build warns instead
    # of rebuilding (the supervisor would respawn repeatedly on a failed build).
    require_build(check_only=resume)
    from refactory_dsh.steer_probe import find_steer_module
    if find_steer_module() is None:
        if not resume:
            raise RuntimeError(
                f"no refactory.steer entry point is installed in {sys.executable}; "
                "run `uv sync` to register it"
            )
        else:
            import logging
            logging.getLogger(__name__).warning(
                "no refactory.steer entry point is installed in %s; "
                "steering will be unavailable — run `uv sync`",
                sys.executable,
            )
    from refactory_dsh import dsh_profile
    _profile_name = 'graph'
    _pin_checkout_for_check: Path | None = None
    if resume and workdir:
        try:
            from refactory_dsh import pin as _pin_mod
            _profile_name = _pin_mod.driver_profile(workdir)
            if _profile_name.startswith('graph-pin-'):
                import json as _json
                _rc = _json.loads((Path(workdir) / '.refactory' / 'run-config.json').read_text())
                _pco = _rc.get('driver', {}).get('pin_checkout')
                if _pco:
                    _pin_checkout_for_check = Path(_pco)
        except Exception:
            pass
    dsh_profile.require(name=_profile_name, checkout=_pin_checkout_for_check)
    # The run's model, settled before the run directory exists (dm-258). A live run
    # talks to the CEO and a run with agent nodes dispatches workers, so either one
    # needs a model: the route a flag, a route map or the stored choice gives, else
    # the visible first-run choice, else a refusal. A headless run of deterministic
    # nodes needs none; if its graph still needs the CEO, the graph runner refuses.
    interactive = not headless and sys.stdin.isatty()
    if not resume:
        route_project = Path(project or project_root())
        if model_context is None:
            from refactory_dsh.model import DshModelClient
            timeout_kwargs = {"timeout_s": model_timeout_s} if model_timeout_s is not None else {}
            model_context = DshModelClient(project=route_project, provider=provider, model=model, worker_backend=worker_backend,
                                           effort=effort, permission_mode=permission_mode,
                                           role_routes=role_routes, node_routes=node_routes, **timeout_kwargs).context()
        needed = not headless or calls_models(with_knobs(registry.resolve(route_project, identifier)[1]().state_dict(),
                                                        list(knobs or [])))
        if needed:
            model_context = require_route(model_context, interactive=interactive)
        provider, model = ceo_route(model_context, interactive=interactive, required=not headless)
        if model_context.get("worker_backend") == "openshell":
            print(OPENSHELL_MODEL_NOTE, file=sys.stderr)
    with ownership(workdir) as owner:
        if resume:
            if source or source_format not in ('', 'directory') or approve or knobs or intent or model_context or provider or model or bool(worker_backend) or effort or permission_mode or role_routes or node_routes or model_timeout_s is not None:
                raise ValueError('resume retains saved inputs, knobs, instructions and routes; changes require explicit invalidation')
            import json

            saved = json.loads((workdir / '.refactory/run-config.json').read_text())
            # ── pin: detect drift and re-exec under the pin ──
            from refactory_dsh import pin as _pin
            _drift = _pin.detect_drift(saved)
            if _drift is not None:
                _pin_checkout = _pin.resolve_pin(saved['driver'], run_dir=workdir)
                if REPO_ROOT.resolve() != Path(_pin_checkout).resolve():
                    _pin_py = str(_pin.pin_python(_pin_checkout))
                    os.execv(_pin_py, [_pin_py, '-m', 'refactory_dsh.interactive', *sys.argv[1:]])
            if saved.get('mode') != identifier:
                raise ValueError('saved run belongs to a different mode')
            # Resource allocation is the only override accepted by recovery.
            effective_limits = resource_limits or saved.get('resource_limits')
            name, graphs = prepare(identifier, workdir=workdir, resource_limits=effective_limits, resume=True)
            project = Path(saved['project'])
            knobs = saved.get('knobs') or []
            model_context = saved.get('model_context')
            if model_context is None:
                from refactory_dsh.model import DshModelClient
                # Old unrouted runs still need a concrete native policy in the
                # child environment, even when no shared budget was configured.
                # They also did not opt into any ambient role or node overrides.
                model_context = DshModelClient(worker_backend='dsh', permission_mode='default',
                                               role_routes={}, node_routes={}).context()
            else:
                model_context = {**model_context, 'permission_mode': model_context.get('permission_mode') or 'default'}
            # A run saved before there was a choice resumes on the stored one, never on the
            # profile default it used to get (dm-258). Without one nothing is guessed: the
            # graph runner and each worker refuse an empty route.
            if (model_context.get('worker_backend', 'dsh') == 'dsh'
                    and not model_context.get('provider') and not model_context.get('model')):
                from refactory_dsh.dsh_profile import stored_model_choice
                model_context = {**model_context, **stored_model_choice(model_context.get('dsh_home') or None)}
            provider, model = ceo_route(model_context, required=False)
            # When pinned, rewrite runtime_root and python_bin so workers
            # use the pin's venv, not the original checkout's.
            _driver_rec = saved.get('driver', {})
            if model_context and _driver_rec.get('pin_checkout'):
                _pin_co = _driver_rec['pin_checkout']
                model_context = {
                    **model_context,
                    'runtime_root': _pin_co,
                    'python_bin': str(Path(_pin_co) / '.venv' / 'bin'),
                }
            resource_limits = effective_limits
        if source:
            from refactory_dsh.items import write_source

            # The harness's input port, fed at run time: the mode keeps the Task it was
            # built for and this run says which items to read instead.
            write_source(workdir, source_path=source, source_format=source_format or "directory")
        # The graph's own argv names the CEO route (`provider`, `model` above), and it is
        # the route the resolved context selected -- from a flag, a project file, a saved
        # run or the stored choice -- not the empty CLI flags a person left unset.
        # Without this the CEO ran the runtime's default provider while every node
        # dispatch honored the project's selection.
        if not resume:
            name, graphs = prepare(
                identifier, workdir=workdir, intent=intent, project=project, knobs=knobs, model_context=model_context, resource_limits=resource_limits, headless=headless,
            )
        for pair in list(approve or []):
            gate, separator, value = str(pair).partition("=")
            if not separator or not gate.strip() or not value.strip():
                raise ValueError(f"--approve takes GATE=VALUE, not {pair!r}")
            # After prepare(), so the intent seed and the verdicts are both in the ledger.
            steering.Ledger(workdir).verdict(gate.strip(), value.strip(), approval=True)
        if identifier == "default_pack/design" and not resume:
            # The approval gate shows delivery.md; one answered up front shows nothing,
            # so the person is told here instead (#350).
            from refactory_dsh import delivery

            approved = any(str(pair).partition("=")[0].strip() == "gate_strategy" for pair in approve or [])
            (delivery.announce if approved else delivery.forget)(
                dict(str(pair).split("=", 1) for pair in knobs or [] if "=" in str(pair)), workdir)
        argv = command(
            identifier, workdir=workdir, graphs=graphs, name=name, knobs=knobs, project=project,
            provider=provider, model=model, headless=headless,
        )
        # stdin is inherited on purpose: the graph profile chats on it -- and its answerer
        # reads a gate's answer from it -- while a captured stdin is not a person.
        # A killed Python parent must not free the directory while its DSH child
        # continues. POSIX inherits this one descriptor, not ambient open files.
        inherited = {"pass_fds": (owner,)} if os.name != "nt" else {}
        env = dict(os.environ)
        from refactory_dsh.workers import environment
        env.update(environment(model_context))
        if project:
            env["REFACTORY_PROJECT"] = str(project)
        budget = None
        if resource_limits:
            from refactory_dsh.budgets import RunBudget
            budget = RunBudget.open(workdir)
            try:
                budget.checkpoint()
            except BaseException:
                budget.pause()
                raise
            env["REFACTORY_BUDGET_LEDGER"] = str(budget.path)
            remaining = budget.remaining_s('final_validation')
            if remaining is not None:
                timeout_s = min(timeout_s, remaining) if timeout_s else remaining
        owner_start = int(time.time() * 1000) / 1000
        # dm-93: for headless runs, provide a stop condition that polls
        # spine-status.json. When the status is terminal, close stdin so the
        # coordinator's final report still gets written, wait a grace period,
        # then end the owner's process group the way run_owned/run_graph_command
        # already end it on a timeout.
        stop_condition = None
        if headless:
            from refactory_dsh import run_state as _rs

            def _terminal_stop() -> bool:
                return _rs.spine_status(workdir) in _rs.TERMINAL
            stop_condition = _terminal_stop
        try:
            if model_context is not None or resource_limits:
                from refactory_dsh.model import DshModelClient
                from refactory_dsh.workers import run_graph_command
                # Older unrouted checkpoints did not record a worker policy.
                # Keep their native policy instead of upgrading them to skip.
                context = model_context or DshModelClient(permission_mode='default').context()
                code = run_graph_command(argv, workdir=workdir, context=context, cwd=str(REPO_ROOT),
                                         timeout=timeout_s, env=env,
                                         stop_condition=stop_condition, **inherited).returncode
            else:
                code = run_owned(argv, cwd=str(REPO_ROOT), timeout=timeout_s, env=env,
                                 stop_condition=stop_condition, **inherited).returncode
        finally:
            if budget is not None:
                budget.pause()
    # ── dm-94: ownership released; inline supervision runs outside ──
    # dm-95: promote ready+last_failure to run-level failed before any reader
    # (status, supervisor) sees an unretried node failure as still active.
    try:
        from refactory_dsh.run_state import promote_unretried_failure
        promote_unretried_failure(workdir)
    except Exception:
        pass  # promotion is best-effort; do not mask the run's exit code
    # The spine run's end, once its DSH child has exited however the run ended:
    # no builder probe's warm clone and no agent's workspace outlives it. One left on
    # disk is named.
    try:
        from refactory_dsh.builder_probe import sweep_warm_clones
        sweep_warm_clones(workdir)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("run-end cleanup under %s: %s", workdir, exc)
    try:
        from refactory_dsh.workspace import sweep as sweep_workspaces
        sweep_workspaces(workdir)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("run-end cleanup under %s: %s", workdir, exc)
    # In headless mode, when the run stopped at a user gate, print the exact
    # commands a person uses to answer it and resume. Printed once, only here.
    if headless and code != 0:
        try:
            from refactory_dsh import console
            due_gates = console.gates(workdir)
            for gate_info in due_gates:
                gate_id = gate_info["id"]
                prompt = gate_info.get("prompt", "")
                verdicts = gate_info.get("verdicts", [])
                verdicts_str = ", ".join(verdicts)
                print(f"refactory: {gate_id} is waiting for a person.", file=sys.stderr)
                if prompt:
                    for line in prompt.splitlines():
                        print(f"  {line}", file=sys.stderr)
                print(f"  Answer it with one of {verdicts_str} (add --note \"...\" to correct the design before it is built):", file=sys.stderr)
                print(f"    refactory steer --run {workdir} --verdict {gate_id}=<verdict>", file=sys.stderr)
                # Read the resume command from run-config.json when available.
                resume_cmd = f"refactory graph-control resume --run {workdir}"
                config_path = workdir / ".refactory/run-config.json"
                if config_path.exists():
                    try:
                        import json as _json
                        _cfg = _json.loads(config_path.read_text())
                        resume_cmd = _cfg.get("resume_command", resume_cmd)
                    except (OSError, ValueError):
                        pass
                if gate_info.get("ceo_notes"):
                    print(f"  CEO observation:", file=sys.stderr)
                    for ceo_line in gate_info["ceo_notes"].splitlines():
                        print(f"    {ceo_line}", file=sys.stderr)
                print(f"  Then continue the run:", file=sys.stderr)
                print(f"    {resume_cmd}", file=sys.stderr)
        except (OSError, ValueError, KeyError, TypeError):
            pass  # gate diagnostic; do not mask the run's exit code
        # A run that stopped short of its end names the nodes it never reached;
        # the gate diagnostic above says nothing for that case.
        try:
            import json
            recorded = json.loads((workdir / ".refactory" / "spine-status.json").read_text())
            if recorded.get("status") == "halted" and recorded.get("error"):
                print(f"refactory: the run stopped early: {recorded['error']}", file=sys.stderr)
        except (OSError, ValueError, AttributeError):
            pass  # diagnostic only; do not mask the run's exit code
    # Inline supervision for headless runs: if the owner exited and the
    # run needs supervision, respawn it within the same process.
    # dm-94: runs outside ownership so a consumed resume can start a new owner.
    if headless and code != 0:
        try:
            from refactory_dsh.supervisor import supervise
            result = supervise(workdir, first_exit_code=code, owner_start_time=owner_start)
            if result is not None:
                code = result
        except Exception:
            pass  # supervision is best-effort; don't mask the run's exit code
    if code == 0:
        summary = run_summary(workdir)
        if summary:
            print(summary)
    # Whatever the exit code: a warning matters most on a run that went wrong.
    warned = run_warnings(workdir)
    if warned:
        print(warned, file=sys.stderr)
    return code


#: The facts worth showing a person who just ran a mode, in the order they read them.
SUMMARY_KEYS = (
    "name", "package_path", "files", "artifact", "check", "validation", "pull_request", "delivery_result",
    "smoke_score", "smoke_reason",
    "revision_of", "before_score", "after_score", "regressed", "revision_note",
)


def run_summary(workdir: Path) -> str:
    """One line of what a finished run recorded, read from the run's own facts.

    DSH's own report is the agent's final word. The facts — where a package went, what
    it scored, whether a revision regressed — are in `facts.json`, and a person should
    not have to open it to find out what they just ran. Only keys that are present are
    shown, so a mode that records none of them prints nothing extra.
    """
    import json

    facts = Path(workdir) / "facts.json"
    if not facts.exists():
        return ""
    try:
        recorded = json.loads(facts.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    shown = {
        key: recorded[key] for key in SUMMARY_KEYS
        if recorded.get(key) not in (None, "", [])
    }
    if not shown:
        return ""
    return "refactory: what the run recorded — " + json.dumps(shown, default=str)


def run_warnings(workdir: Path) -> str:
    """The run warnings this run's steps printed, as the run's journal recorded them.

    The spine hands each one to the CEO, but nothing makes the CEO repeat it, and a
    headless run's turns are read by nobody. A warning such as uncommitted changes
    in the target that the run left out changes what its result means, so the run
    lists them itself once it ends. Empty when no step warned.
    """
    from refactory_dsh import live

    records, _ = live.events(Path(workdir))
    # Once each: a step that reloops repeats the warnings it printed the first time.
    lines = list(dict.fromkeys(
        f"  {record.get('node_id')}: {record['summary']}" for record in records
        if record.get("kind") == "run_warning" and record.get("summary")
    ))
    if not lines:
        return ""
    return "\n".join(["refactory: what the run warned about —", *lines])


def main(argv: list[str] | None = None) -> int:
    """``python -m refactory_dsh.interactive <package>/<mode> --intent <text>``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m refactory_dsh.interactive",
        description="Run one mode with a person in the loop, on DSH.",
    )
    parser.add_argument("mode", help="the mode to run, as <package>/<name>")
    parser.add_argument(
        "--intent", default="",
        help="what the run is for, in the person's words; seeds the ledger the agent "
             "nodes read and a resumed run is judged against",
    )
    parser.add_argument("--workdir", default="", help="the run directory")
    parser.add_argument("--resume", action="store_true", help="recover a stopped known boundary, preserving completed stages")
    parser.add_argument(
        "--source", default="", metavar="PATH",
        help="read the items from here instead of the source the mode declares, so a "
             "harness built for a class of work is pointed at items it has not seen",
    )
    parser.add_argument(
        "--source-format", default="directory", choices=["directory", "jsonl", "csv"],
        help="how to read --source (default: a directory of one item per subdirectory)",
    )
    parser.add_argument(
        "--knob", action="append", default=[], metavar="NAME=VALUE",
        help="set one of the mode's knobs for this run (repeatable)",
    )
    parser.add_argument(
        "--approve", action="append", default=[], metavar="GATE=VALUE",
        help="state a gate's answer before the run, e.g. gate_strategy=proceed "
             "(repeatable); without it the run asks you at the terminal",
    )
    parser.add_argument("--timeout", type=int, default=0, help="seconds; 0 waits for the person")
    parser.add_argument(
        "--provider", default="",
        help="provider route for this run's model (default: the stored choice; see `refactory model set`)",
    )
    parser.add_argument(
        "--model", default="",
        help="model id for this run, e.g. deepseek-v4-pro (default: the stored choice; see `refactory model set`)",
    )
    parser.add_argument(
        "--headless", action="store_true", default=False,
        help="run without a conversation: one task, one report, then exit, as scripts "
             "and CI expect; without it the run is live and a line you type between "
             "turns reaches the CEO",
    )
    parser.add_argument("--worker-backend", choices=["dsh", "codex", "claude_code"], default="")
    parser.add_argument("--effort", default="")
    parser.add_argument("--permission-mode", default="", help="worker permissions: skip (new-run default), default (native policy), or a backend-specific mode")
    parser.add_argument(
        "--model-timeout", type=int, default=None, dest="model_timeout_s",
        help="seconds one model completion may take before it is called hung (default: 1800)",
    )
    import json
    parser.add_argument("--role-routes", type=json.loads, default=None)
    parser.add_argument("--node-routes", type=json.loads, default=None)
    parser.add_argument("--max-elapsed-s", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--validation-reserve-s", type=float, default=0)
    parser.add_argument("--validation-reserve-tokens", type=int, default=0)
    args = parser.parse_args(argv)

    from refactory_dsh.model import ModelChoiceError

    workdir = Path(args.workdir) if args.workdir else None
    try:
        code = run(
            args.mode,
            workdir=workdir,
            source=args.source,
            source_format=args.source_format,
            intent=args.intent,
            knobs=args.knob,
            approve=args.approve,
            timeout_s=args.timeout or None,
            model_timeout_s=getattr(args, 'model_timeout_s', None),
            provider=args.provider,
            model=args.model,
            headless=args.headless, resume=args.resume, worker_backend=args.worker_backend,
            effort=args.effort, permission_mode=args.permission_mode,
            role_routes=args.role_routes, node_routes=args.node_routes,
            resource_limits={"max_elapsed_s": args.max_elapsed_s, "max_tokens": args.max_tokens,
                             "validation_reserve_s": args.validation_reserve_s,
                             "validation_reserve_tokens": args.validation_reserve_tokens}
            if args.max_elapsed_s is not None or args.max_tokens is not None else None,
        )
    except ModelChoiceError as exc:  # no model to run on: the message says how to choose one
        print(f"refactory: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # a failed run must be a message, not a traceback
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
