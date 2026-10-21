"""OpenShell sandbox backend for the graph executor.

Runs graph commands inside kernel-confined OpenShell sandboxes instead of
local subprocesses.  The sandbox provides Landlock/seccomp confinement and
gateway-held credentials — inference keys never enter the sandbox.

The composition layer (plan building, CLI argv) is imported from v1's
``factory.contained.openshell`` through the sys.path adapter.  The SDK
(gRPC/protobuf) is deferred to call time; plan composition is pure.

Prerequisites (verified at dispatch time, not at import):

- The OpenShell CLI and SDK (``uv pip install openshell``)
- A gateway (``openshell gateway select``)
- The ``claude-code`` provider registered with the gateway

Credential design: verify-only (Option C).  The gateway and DSH maintain
independent credential registries.  At dispatch time,
``verify_gateway_provider`` checks that the gateway has the expected
provider and that its profile matches the shipped expectation; it fails
with a concrete fix command if not.  No bridging, no automated writes
to gateway state.
"""

from __future__ import annotations

import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import refactory_dsh._v1_compat  # noqa: F401

from factory.contained.errors import ContainedError
from factory.contained.openshell import (
    OpenShellPlan,
    PROVIDER_FIX,
    WORKSPACE_ROOT,
    build_download_argv,
    build_run_launch,
    build_upload_argv,
    connect,
    create_sandbox,
    exec_argv,
    extract_tarball,
    load_provider_profile,
    run_probes,
    sandbox_name,
    start_run,
    validate_sandbox_name,
)
from factory.contained.provenance import Probe, content_probe, provenance_probes
from factory.podman import project_hash

PROVIDER_NAME = "claude-code"

PACK_EXCLUDES = frozenset({
    ".venv", "node_modules", "__pycache__", ".pytest_cache",
    ".ruff_cache", ".mypy_cache", ".factory-worktrees",
})

RUN_ARTIFACTS = frozenset({"run.log", "run.pid", "run.exit"})

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


def _is_secret_key(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_MARKERS)


def verify_gateway_provider(gateway: str | None = None) -> None:
    """Check that the gateway has the expected claude-code provider.

    Fails with a ``ContainedError`` carrying the fix command when the
    provider is missing or its profile diverges from the shipped
    expectation.  This is the verify-only credential design (Option C):
    the gateway and DSH maintain independent registries, and this check
    is the only bridge between them.
    """
    client = connect(gateway)
    try:
        registered = client.get_provider(PROVIDER_NAME)
    except Exception:
        raise ContainedError(
            f"the gateway has no '{PROVIDER_NAME}' provider — register it with:\n"
            f"  {PROVIDER_FIX}"
        )
    expected = load_provider_profile()
    if registered.get("endpoints") != expected.get("endpoints"):
        raise ContainedError(
            f"the gateway's '{PROVIDER_NAME}' provider profile diverges from "
            f"the shipped expectation — update it with:\n  {PROVIDER_FIX}"
        )


def compose_plan(
    workdir: Path,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    image: str | None = None,
    gateway: str | None = None,
    policy_path: Path | None = None,
    name: str | None = None,
) -> OpenShellPlan:
    """Build an OpenShellPlan from v2's run context.

    Pure composition — no gateway contact, no SDK import.
    """
    workdir = Path(workdir).resolve()
    run_id = validate_sandbox_name(name) if name else sandbox_name(workdir)
    project_dir = f"{WORKSPACE_ROOT}/{workdir.name}"

    sandbox_env: dict[str, str] = {}
    warnings: list[str] = []
    for key, value in (env or {}).items():
        if _is_secret_key(key):
            warnings.append(
                f"credential-looking variable {key!r} is being placed in the sandbox "
                "environment; the provider is the supported route for inference credentials"
            )
        sandbox_env[key] = value
    sandbox_env.setdefault("PYTHONUNBUFFERED", "1")
    sandbox_env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")

    inner = " ".join(shlex.quote(token) for token in argv)
    run_command = f"cd {shlex.quote(project_dir)} && {inner}"

    return OpenShellPlan(
        name=run_id,
        image=image or "factory-runtime:latest",
        project_dir=project_dir,
        env=sandbox_env,
        labels={
            "factory.contained": "true",
            "factory.project": project_hash(workdir),
            "factory.name": run_id,
        },
        provider=PROVIDER_NAME,
        policy_path=policy_path,
        run_command=run_command,
        factory_command=inner,
        gateway=gateway,
        warnings=tuple(warnings),
    )


def pack_workspace(workdir: Path, run_id: str) -> Path:
    """Pack the workspace into a tarball for upload to the sandbox."""
    workdir = Path(workdir).resolve()
    staging = workdir / ".refactory" / "openshell" / run_id
    staging.mkdir(parents=True, exist_ok=True)
    tarball = staging / "upload.tar.gz"

    def _filter(entry: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(entry.name).parts
        if set(parts) & PACK_EXCLUDES:
            return None
        if ".factory" in parts[:-1] and parts[-1] in RUN_ARTIFACTS:
            return None
        return entry

    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(str(workdir), arcname=workdir.name, filter=_filter)
    return tarball


def provenance_checks(plan: OpenShellPlan, workdir: Path) -> list[Probe]:
    """Build the provenance probes for the sandbox."""
    return provenance_probes(
        plan.project_dir,
        expect_factory_state=(workdir / ".factory" / "config.json").exists(),
        expect_git=(workdir / ".git").exists(),
        content=content_probe(workdir),
    )


def download_artifacts(
    plan: OpenShellPlan, dest: Path, *, paths: list[str] | None = None,
) -> None:
    """Download workspace artifacts from the sandbox back to the host."""
    for sandbox_path in (paths or [f"{plan.project_dir}/.refactory"]):
        argv = build_download_argv(plan.name, sandbox_path, dest, gateway=plan.gateway)
        result = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise ContainedError(
                f"downloading {sandbox_path} failed: {result.stderr.strip()[:300]}"
            )


def run_in_openshell(
    argv: list[str],
    *,
    workdir: Path,
    context: dict[str, Any],
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    stop_condition: Any | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Run a graph command inside an OpenShell sandbox.

    Follows the same interface as ``run_graph_command`` so callers can
    dispatch to either backend.  The lifecycle:

    1. Compose the plan (pure, no gateway contact)
    2. Pack the workspace into a tarball
    3. Connect to the gateway and create the sandbox
    4. Upload and extract the workspace
    5. Run provenance probes
    6. Start the graph command as a detached process
    7. Poll the run's exit file until it completes or times out
    8. Download artifacts

    Steps 3-8 require the OpenShell SDK and a configured gateway with
    the ``claude-code`` provider.  The verify-only credential design
    checks the gateway's provider at dispatch time and fails with a fix
    command if it is missing or stale.
    """
    import json
    import os
    from refactory_dsh.budgets import RunBudget
    from refactory_dsh.workers import GraphTelemetry

    environment_values = {**os.environ, **(env or {})}
    ledger = context.get('budget_ledger') or environment_values.get('REFACTORY_BUDGET_LEDGER')
    budget = RunBudget.from_path(ledger) if ledger else None
    telemetry = GraphTelemetry(workdir, context, budget)
    environment_values.update(telemetry.environment())

    try:
        telemetry.checkpoint()
    except BaseException as error:
        telemetry.finish('budget_exhausted', str(error))
        raise

    plan = compose_plan(
        workdir, argv, env=environment_values,
        gateway=context.get('openshell_gateway'),
        image=context.get('openshell_image'),
    )
    tarball = pack_workspace(workdir, plan.name)
    probes = provenance_checks(plan, workdir)

    try:
        verify_gateway_provider(plan.gateway)
        from factory.contained.openshell import build_default_policy
        client = connect(plan.gateway)
        policy = build_default_policy()
        session = create_sandbox(client, plan, policy, tarball)
    except ContainedError as exc:
        telemetry.finish('failed', str(exc))
        return subprocess.CompletedProcess(argv, 2)

    try:
        uploaded = subprocess.run(
            build_upload_argv(plan.name, tarball, gateway=plan.gateway),
            capture_output=True, text=True, timeout=600,
        )
        if uploaded.returncode != 0:
            raise ContainedError(
                f"uploading workspace failed: {uploaded.stderr.strip()[:300]}"
            )

        extract_tarball(session, tarball.name)
        run_probes(session, plan, probes)
        start_run(session, plan)

        exit_code = _poll_run(
            session, plan, timeout=timeout,
            stop_condition=stop_condition, telemetry=telemetry,
        )
    except ContainedError as exc:
        telemetry.finish('failed', str(exc))
        return subprocess.CompletedProcess(argv, 1)
    except BaseException as error:
        telemetry.finish('cancelled', str(error))
        raise

    try:
        download_artifacts(plan, workdir)
    except ContainedError:
        pass

    telemetry.finish('completed' if exit_code == 0 else 'failed')
    return subprocess.CompletedProcess(argv, exit_code)


def _poll_run(
    session,
    plan: OpenShellPlan,
    *,
    timeout: float | None,
    stop_condition: Any | None,
    telemetry: Any,
) -> int:
    """Poll the run's exit file until it completes, times out, or is stopped."""
    from factory.contained.openshell import RUN_EXIT, exec_argv

    deadline = time.monotonic() + timeout if timeout else None
    while True:
        telemetry.checkpoint()
        if deadline is not None and time.monotonic() >= deadline:
            telemetry.finish('timed_out', f'timeout after {timeout}s')
            return 1

        if stop_condition is not None:
            try:
                if stop_condition():
                    telemetry.finish('stopped', 'stop condition')
                    return 1
            except Exception:
                pass

        try:
            result = exec_argv(
                session, ["cat", RUN_EXIT],
                workdir=plan.project_dir, timeout=10,
            )
            if result.exit_code == 0 and result.stdout.strip():
                return int(result.stdout.strip())
        except (ContainedError, ValueError):
            pass

        time.sleep(5)
