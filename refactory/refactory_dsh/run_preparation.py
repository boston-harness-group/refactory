"""Shared run preparation for both the walker and DSH runtimes.

One function both runtimes call: take ownership, invoke ``invocations.prepare``
(which archives the old invocation, seeds the ledger into the stage, folds it
into the lowering, and installs the result), check the install and profile, and
write ``run-config.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def prepare_run(
    state: dict[str, Any],
    workdir: Path,
    *,
    invocation_name: str,
    mode_id: str,
    intent: str,
    transform: Callable,
    runtime: str,
    command: str,
    resume_command: str,
    fast: bool = False,
    worker_backend: str = "",
    resource_limits: dict[str, Any] | None = None,
    project: Path | None = None,
    knobs: list[str] | None = None,
    model_context: dict[str, Any] | None = None,
    owner_argv: list[str] | None = None,
) -> Path:
    """Prepare a run directory for either runtime.

    Order:
    1. Install and profile check (before anything is archived).
    2. ``invocations.prepare`` — archives the old invocation, seeds the ledger
       into the stage, folds it from there, and installs the result.
    3. Write ``run-config.json``, the same record on both runtimes (dm-160): the mode,
       the project, the run's knobs as its flags gave them, its model route, its
       resource limits, the configuration fingerprints a resume checks, and the
       driver checkout (`pin.record_driver`). Three fields name the runtime that runs
       it: ``runtime`` (``walker`` or ``spine``, as the status file says), the
       ``command`` that started it and the ``resume_command`` that continues it. A
       spine run also records ``owner_argv``, the fragment its supervisor restarts the
       owner with; a walked run has no supervisor and records none.

    Returns the installed graphs directory.
    """
    # 1. Install and profile check — before invocations.prepare moves anything,
    #    so a failure leaves nothing archived without rollback.
    #    DSH has its own checks at interactive.py run() (~253-255); they run on
    #    both launch and resume, so moving them here would drop the resume check.
    #    The walker needs its own check here.
    if runtime == "walker":
        from refactory_dsh.paths import require_build

        require_build()
        if worker_backend == "dsh":
            from refactory_dsh.dsh_profile import model_home, profile_dir

            headless_dir = profile_dir("headless", home=model_home())
            # A missing profile is fine: DSH creates `headless` from its shipped template
            # on first use. Refuse only a profile directory left incomplete.
            if headless_dir.is_dir() and not (headless_dir / "package.json").exists():
                raise ValueError(
                    f"the DSH headless profile at {headless_dir} has no "
                    "package.json; run `dsh profile create headless` and add "
                    "the graph bundle"
                )

    # 2. Archive and stage.
    from refactory_dsh import invocations

    graphs = invocations.prepare(
        state, workdir, name=invocation_name, transform=transform,
        resource_limits=resource_limits, intent=intent,
    )

    # 3. Write run-config.json.
    from refactory_dsh.recovery import configuration_fingerprints

    config: dict[str, Any] = {
        "mode": mode_id,
        "runtime": runtime,
        "command": command,
        "resume_command": resume_command,
        # The supervisor's fragment (`supervisor._build_owner_argv`), on the spine only: a
        # walked run has no supervisor, and its main's `--resume` takes the run directory.
        **({"owner_argv": list(owner_argv)} if owner_argv is not None else {}),
        "fast": fast,
        "project": str(Path(project).resolve()) if project else None,
        "knobs": list(knobs or []),
        "model_context": dict(model_context or {}),
        "resource_limits": resource_limits,
        "configuration_fingerprints": configuration_fingerprints(
            workdir, mode_id, Path(project) if project else Path(".")
        ) if project else {},
    }
    config_path = workdir / ".refactory/run-config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, default=str))
    from refactory_dsh import pin

    pin.record_driver(config_path)
    return graphs


def print_gate_stop(gate_id: str, workdir: Path, *, graph: Any = None) -> None:
    """Print the gate-stop output to stderr: prompt, verdicts, steer and resume.

    The ``{"error": ...}`` JSON line is first so ``console._forward`` keeps
    working.
    """
    import sys

    print(json.dumps({"error": f"gate {gate_id!r} is waiting for a person"}),
          file=sys.stderr)
    # Read the gate's notes (prompt) and edges from the graph.
    prompt = ""
    verdict_labels: list[str] = []
    if graph is not None:
        node = graph.nodes.get(gate_id)
        if node is not None:
            from refactory_dsh.nn.executor import render, RunContext

            ctx = RunContext(workdir)
            # Load facts for rendering.
            facts_path = workdir / "facts.json"
            if facts_path.exists():
                try:
                    ctx.facts = json.loads(facts_path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass
            prompt = render(node.notes, ctx)
            verdict_labels = sorted(
                {e.condition for e in graph.out_edges(gate_id) if e.condition}
            )
    if "halt" not in [v.lower() for v in verdict_labels]:
        verdict_labels.append("halt")
    verdicts_str = ", ".join(verdict_labels)
    print(f"refactory: {gate_id} is waiting for a person.", file=sys.stderr)
    if prompt:
        for line in prompt.splitlines():
            print(f"  {line}", file=sys.stderr)
    print(f"  Accepted verdicts: {verdicts_str}", file=sys.stderr)
    print(f"  Answer it (add --note \"...\" to correct the design before it is built):", file=sys.stderr)
    print(f"    refactory steer --run {workdir} --verdict {gate_id}=<verdict> [--note ...]", file=sys.stderr)
    # Read resume command from run-config.json.
    resume_cmd = ""
    config_path = workdir / ".refactory/run-config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            resume_cmd = cfg.get("resume_command", "")
        except (json.JSONDecodeError, OSError):
            pass
    if resume_cmd:
        print(f"  Then continue the run:", file=sys.stderr)
        print(f"    {resume_cmd}", file=sys.stderr)


# ── the walker's mains ─────────────────────────────────────────────────────

#: What `--acknowledge-failure` says on every walker main.
ACKNOWLEDGE_HELP = ("with --resume: you inspected a failed boundary, or one whose last node's outcome is "
                    "unknown, and want to go on from it; what that node left that the run cannot vouch for "
                    "is set aside first, as `refactory graph-control acknowledge` sets it aside on the spine")
#: What `--resume` says on every walker main.
RESUME_HELP = ("continue a stopped run from its own directory instead of starting a new one; every node "
               "that already finished is kept. A failed run, or one stopped mid-node, needs "
               "--acknowledge-failure as well")


def saved_run_config(workdir: Path) -> dict[str, Any]:
    """The run's ``run-config.json``, or nothing when it has none."""
    try:
        config = json.loads((Path(workdir) / ".refactory/run-config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return config if isinstance(config, dict) else {}


def resume_walked(workdir: Path, family: str, modes: tuple[type, type], *, model: Any,
                  acknowledge_failure: bool = False, resource_limits: dict[str, Any] | None = None,
                  **kwargs: Any) -> tuple[type, Any]:
    """Continue a walked run, validated as the spine validates its own (dm-160).

    ``modes`` are the main's full mode and its ``_fast`` variant; the run's
    ``run-config.json`` says which it was. Under the run's ownership lock: with
    ``acknowledge_failure``, what the stopped boundary cannot vouch for is set aside
    first (`recovery.acknowledge_walked`); then `recovery.prepare_walked` refuses what
    the spine's resume refuses, and the run goes on from its progress record. Returns
    the mode class that ran it and its result.
    """
    from refactory_dsh import recovery
    from refactory_dsh.run_ownership import ownership

    workdir = Path(workdir)
    fast = bool(saved_run_config(workdir).get("fast"))
    full, quick = modes
    mode_class = quick if fast else full
    mode_id = f"default_pack/{family}{'_fast' if fast else ''}"
    with ownership(workdir):
        if acknowledge_failure:
            recovery.acknowledge_walked(workdir, mode_id)
        recovery.prepare_walked(workdir, mode_id, acknowledge_failed=acknowledge_failure)
        run = mode_class.resume(workdir, model=model, resource_limits=resource_limits, enable_recovery=True,
                                acknowledged=acknowledge_failure, **kwargs)
    return mode_class, run


def walker_context(client: Any) -> dict[str, Any]:
    """The model route a walker run records in ``run-config.json``, as the spine records its own."""
    return dict(client.context()) if callable(getattr(client, "context", None)) else {}
