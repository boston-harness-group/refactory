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
from pathlib import Path
from typing import Any

from refactory_dsh import lowering, registry, steering
from refactory_dsh.dsh_runtime import TASK
from refactory_dsh.paths import REPO_ROOT, dsh_command, venv_bin, project_root
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
) -> tuple[str, Path]:
    """Lower the mode for a person to use, and leave the run ready to start.

    Returns the graph name the profile loads and the directory it lives in. The
    ledger is seeded **before** the lowering, because the lowering is what folds it
    into the agent prompts; seeding afterwards would produce a run nobody steered.
    """
    workdir = Path(workdir).expanduser().resolve()
    with ownership(workdir):
        project = Path(project or project_root())
        if resume:
            if knobs or intent or model_context:
                raise ValueError('resume retains saved knobs, instructions and route; changes require explicit invalidation')
            from refactory_dsh.recovery import prepare as recover
            name = identifier.replace('/', '-')
            return name, recover(workdir, name, resource_limits=resource_limits)
        if "/" not in identifier:
            raise ValueError(f"{identifier!r} is not a <package>/<mode> identifier")
        package, mode_name = identifier.split("/", 1)
        Mode = registry.resolve(project, identifier)[1]
        mode = Mode()
        state = with_knobs(mode.state_dict(), list(knobs or []))

        # Absolute, because this directory becomes DSH's session cwd and DSH refuses a
        # relative one: `interactive <mode>` with no `--workdir` used to fail before the
        # graph started, on the default root that is relative by design.
        workdir.mkdir(parents=True, exist_ok=True)
        if not steering.Ledger(workdir).exists():
            # Selecting a mode is itself an explicit run request. Do not invent a
            # broader purpose when the caller supplied no prose intent.
            steering.Ledger.seed(workdir, intent.strip() or f"Run {identifier}.")

        from refactory_dsh import invocations, workers
        from refactory_dsh.model import DshModelClient
        routing_context = dict(model_context or {})
        if resource_limits:
            routing_context.pop('budget_ledger', None)
        # Persist a concrete policy before lowering, including console/API opens.
        model_context = DshModelClient(project=project, **routing_context).context()
        if resource_limits:
            from refactory_dsh.budgets import LEDGER, ResourceLimits
            ResourceLimits.from_dict(resource_limits)
            model_context['budget_ledger'] = str(workdir / LEDGER)

        name = f"{package}-{mode_name}"
        graphs = invocations.prepare(
            state, workdir, name=name,
            transform=lambda workflow: workers.lower_workflow(steering.for_run(workflow, workdir), workdir, model_context, lowering.node_declarations(state)),
            resource_limits=resource_limits,
        )
        import json
        from refactory_dsh.recovery import configuration_fingerprints

        (workdir / '.refactory/run-config.json').write_text(json.dumps({'mode': identifier,
            'project': str(project.resolve()), 'knobs': knobs or [], 'model_context': model_context,
            'resource_limits': resource_limits,
            'configuration_fingerprints': configuration_fingerprints(workdir, identifier, project)}))
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
    argv = [
        *dsh_command(),
        "--profile", "graph", "run",
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
    if identifier.endswith('/create_fidelity') and approve:
        raise ValueError('Create fidelity strategy approval must be bound to its current review digest; --approve standing answers are refused')
    workdir = Path(workdir).expanduser().resolve()
    with ownership(workdir) as owner:
        if resume:
            if source or source_format not in ('', 'directory') or approve or knobs or intent or model_context or provider or model or bool(worker_backend) or effort or permission_mode or role_routes or node_routes:
                raise ValueError('resume retains saved inputs, knobs, instructions and routes; changes require explicit invalidation')
            import json

            saved = json.loads((workdir / '.refactory/run-config.json').read_text())
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
            resource_limits = effective_limits
        if source:
            from refactory_dsh.items import write_source

            # The harness's input port, fed at run time: the mode keeps the Task it was
            # built for and this run says which items to read instead.
            write_source(workdir, source_path=source, source_format=source_format or "directory")
        if not resume and model_context is None:
            from refactory_dsh.model import DshModelClient
            model_context = DshModelClient(project=Path(project or project_root()), provider=provider, model=model, worker_backend=worker_backend,
                                           effort=effort, permission_mode=permission_mode,
                                           role_routes=role_routes, node_routes=node_routes).context()
        # The graph's own argv names the CEO route, and it must be the route the
        # resolved context selected -- from a project file, a saved run or a
        # caller's model_context -- not the empty CLI flags a person left unset.
        # Without this the CEO ran the runtime's default provider while every
        # node dispatch honored the project's selection.
        if model_context is not None:
            provider = model_context.get('provider', '')
            model = model_context.get('model', '')
            worker_backend = model_context.get('worker_backend', 'dsh')
        if not resume:
            name, graphs = prepare(
                identifier, workdir=workdir, intent=intent, project=project, knobs=knobs, model_context=model_context, resource_limits=resource_limits,
            )
        for pair in list(approve or []):
            gate, separator, value = str(pair).partition("=")
            if not separator or not gate.strip() or not value.strip():
                raise ValueError(f"--approve takes GATE=VALUE, not {pair!r}")
            # After prepare(), so the intent seed and the verdicts are both in the ledger.
            steering.Ledger(workdir).verdict(gate.strip(), value.strip(), approval=True)
        argv = command(
            identifier, workdir=workdir, graphs=graphs, name=name, knobs=knobs, project=project,
            provider=provider if worker_backend == "dsh" else "",
            model=model if worker_backend == "dsh" else "", headless=headless,
        )
        # stdin is inherited on purpose: the graph profile chats on it -- and its answerer
        # reads a gate's answer from it -- while a captured stdin is not a person.
        # A killed Python parent must not free the directory while its DSH child
        # continues. POSIX inherits this one descriptor, not ambient open files.
        inherited = {"pass_fds": (owner,)} if os.name != "nt" else {}
        env = dict(os.environ)
        from refactory_dsh.workers import environment
        env.update(environment(model_context))
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
        try:
            if model_context is not None or resource_limits:
                from refactory_dsh.model import DshModelClient
                from refactory_dsh.workers import run_graph_command
                # Older unrouted checkpoints did not record a worker policy.
                # Keep their native policy instead of upgrading them to skip.
                context = model_context or DshModelClient(permission_mode='default').context()
                code = run_graph_command(argv, workdir=workdir, context=context, cwd=str(REPO_ROOT),
                                         timeout=timeout_s, env=env, **inherited).returncode
            else:
                code = run_owned(argv, cwd=str(REPO_ROOT), timeout=timeout_s, env=env, **inherited).returncode
        finally:
            if budget is not None:
                budget.pause()
        if code == 0:
            summary = run_summary(workdir)
            if summary:
                print(summary)
        return code


#: The facts worth showing a person who just ran a mode, in the order they read them.
SUMMARY_KEYS = (
    "name", "package_path", "files", "artifact", "check", "validation",
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
        help="provider route for this run's model (default: the settings default)",
    )
    parser.add_argument(
        "--model", default="",
        help="model id for this run, e.g. deepseek-v4-pro (default: the settings default)",
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
    import json
    parser.add_argument("--role-routes", type=json.loads, default=None)
    parser.add_argument("--node-routes", type=json.loads, default=None)
    parser.add_argument("--max-elapsed-s", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--validation-reserve-s", type=float, default=0)
    parser.add_argument("--validation-reserve-tokens", type=int, default=0)
    args = parser.parse_args(argv)

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
    except Exception as exc:  # a failed run must be a message, not a traceback
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
