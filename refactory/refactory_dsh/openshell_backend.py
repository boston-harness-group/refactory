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

import json
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
    build_profile_export_argv,
    build_provider_list_argv,
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
    ".env", ".env.local", ".env.production",
})

RUN_ARTIFACTS = frozenset({"run.log", "run.pid", "run.exit"})

REQUIRED_DOWNLOADS = (
    ".refactory",
    "facts.json",
    "spine-status.json",
)

_ENV_ALLOWLIST = frozenset({
    "PYTHONUNBUFFERED",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "REFACTORY_WORKER_CONTEXT",
    "REFACTORY_BUDGET_LEDGER",
    "REFACTORY_WORKER_RECEIPT",
    "REFACTORY_RUNTIME_ROOT",
    "DSH_HOME",
    "VENV_BIN",
    "VIRTUAL_ENV",
    "LANG",
    "LC_ALL",
    "TZ",
    "TERM",
    "HOME",
})


def _sandbox_env(caller_env: dict[str, str] | None) -> dict[str, str]:
    """Build the sandbox environment from an explicit allowlist.

    The sandbox receives only nonsecret configuration variables.
    Inference credentials are held by the gateway's provider attachment
    and never placed in the sandbox environment.
    """
    source = caller_env or {}
    result: dict[str, str] = {}
    for key, value in source.items():
        if key in _ENV_ALLOWLIST or key.startswith("REFACTORY_"):
            result[key] = value
    result.setdefault("PYTHONUNBUFFERED", "1")
    result.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    return result


def verify_gateway_provider(gateway: str | None = None) -> None:
    """Check that the gateway has the expected claude-code provider.

    Uses the CLI (``openshell provider list`` and ``openshell profile
    export``) rather than the SDK, following v1's verification path.
    Fails with a ``ContainedError`` carrying the fix command when the
    provider is missing or its profile diverges from the shipped
    expectation.
    """
    result = subprocess.run(
        build_provider_list_argv(gateway),
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise ContainedError(
            f"cannot list gateway providers (rc={result.returncode}): "
            f"{result.stderr.strip()[:200]}\n"
            f"register with:\n  {PROVIDER_FIX}"
        )

    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        payload = []

    entries = payload if isinstance(payload, list) else payload.get("providers", [])
    profile_id = None
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("name", "")) == PROVIDER_NAME:
            profile_id = str(entry.get("type", "")) or None
            break

    if profile_id is None:
        raise ContainedError(
            f"the gateway has no '{PROVIDER_NAME}' provider — register it with:\n"
            f"  {PROVIDER_FIX}"
        )

    _verify_profile_shape(gateway, profile_id)


def _verify_profile_shape(gateway: str | None, profile_id: str) -> None:
    """Compare the gateway's profile against the shipped expectation.

    Checks both endpoints (the egress allowlist) and binaries (the
    kernel's process match list).  A divergence in either is a security
    concern, not just a configuration mismatch.
    """
    result = subprocess.run(
        build_profile_export_argv(profile_id, gateway),
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise ContainedError(
            f"provider '{PROVIDER_NAME}' exists but its profile {profile_id!r} "
            f"could not be read back — update with:\n  {PROVIDER_FIX}"
        )

    try:
        live = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        live = {}

    expected = load_provider_profile()
    problems: list[str] = []

    expected_endpoints = {
        (str(e.get("host")), int(e.get("port", 0)))
        for e in expected.get("endpoints", [])
    }
    live_endpoints = {
        (str(e.get("host")), int(e.get("port", 0)))
        for e in live.get("endpoints", [])
    }
    if live_endpoints != expected_endpoints:
        extra = sorted(live_endpoints - expected_endpoints)
        missing = sorted(expected_endpoints - live_endpoints)
        detail = f"profile {profile_id!r} differs from the factory's"
        if extra:
            detail += f" (allows {', '.join(f'{h}:{p}' for h, p in extra)})"
        if missing:
            detail += f" (missing {', '.join(f'{h}:{p}' for h, p in missing)})"
        problems.append(detail)

    expected_binaries = set(expected.get("binaries", []))
    live_binaries = set(live.get("binaries", []))
    if not expected_binaries <= live_binaries:
        problems.append(
            f"profile {profile_id!r} binaries do not include the runtime "
            f"image's claude path ({', '.join(sorted(expected_binaries))})"
        )

    if problems:
        raise ContainedError(
            f"{'; '.join(problems)} — update with:\n  {PROVIDER_FIX}"
        )


def _translate_argv(argv: list[str], workdir: Path, project_dir: str) -> list[str]:
    """Translate host-absolute paths in the command argv to sandbox paths.

    The sandbox mounts the uploaded workspace at ``project_dir``.  Any
    argv token that starts with the host workdir is rewritten to its
    sandbox equivalent.  The host Python interpreter is replaced with
    the image's default.
    """
    workdir_str = str(workdir.resolve())
    translated: list[str] = []
    for token in argv:
        if token.startswith(workdir_str):
            suffix = token[len(workdir_str):]
            translated.append(f"{project_dir}{suffix}")
        elif token.startswith("/") and ("python" in token or "dsh" in token):
            translated.append(Path(token).name)
        else:
            translated.append(token)
    return translated


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

    sandbox_env = _sandbox_env(env)

    translated = _translate_argv(argv, workdir, project_dir)
    inner = " ".join(shlex.quote(token) for token in translated)
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


def _terminate_remote(session: Any, plan: OpenShellPlan) -> None:
    """Kill the detached remote process so it stops consuming resources."""
    try:
        exec_argv(
            session,
            ["sh", "-c", "cat /workspace/.factory/run.pid 2>/dev/null | xargs -r kill -9"],
            workdir=plan.project_dir, timeout=10,
        )
    except (ContainedError, OSError):
        pass


def download_artifacts(
    plan: OpenShellPlan, dest: Path, *, paths: list[str] | None = None,
) -> list[str]:
    """Download workspace artifacts from the sandbox back to the host.

    Returns a list of paths that failed to download.  The caller
    decides which failures are fatal.
    """
    download_paths = paths or [
        f"{plan.project_dir}/{name}" for name in REQUIRED_DOWNLOADS
    ]
    failures: list[str] = []
    for sandbox_path in download_paths:
        argv = build_download_argv(plan.name, sandbox_path, dest, gateway=plan.gateway)
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                failures.append(sandbox_path)
        except (subprocess.TimeoutExpired, OSError):
            failures.append(sandbox_path)
    return failures


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

    On timeout, stop, or cancellation the remote process is terminated
    before returning.

    Steps 3-8 require the OpenShell SDK and a configured gateway with
    the ``claude-code`` provider.  The verify-only credential design
    checks the gateway's provider at dispatch time and fails with a fix
    command if it is missing or stale.
    """
    from refactory_dsh.budgets import RunBudget
    from refactory_dsh.workers import GraphTelemetry

    telemetry_env = GraphTelemetry.env_from_context(context) if hasattr(GraphTelemetry, 'env_from_context') else {}
    caller_env = {**(env or {}), **telemetry_env}

    ledger = context.get('budget_ledger') or caller_env.get('REFACTORY_BUDGET_LEDGER')
    budget = RunBudget.from_path(ledger) if ledger else None
    telemetry = GraphTelemetry(workdir, context, budget)
    caller_env.update(telemetry.environment())

    try:
        telemetry.checkpoint()
    except BaseException as error:
        telemetry.finish('budget_exhausted', str(error))
        raise

    plan = compose_plan(
        workdir, argv, env=caller_env,
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

        exit_code, reason = _poll_run(
            session, plan, timeout=timeout,
            stop_condition=stop_condition, telemetry=telemetry,
        )
    except ContainedError as exc:
        _terminate_remote(session, plan)
        telemetry.finish('failed', str(exc))
        return subprocess.CompletedProcess(argv, 1)
    except BaseException as error:
        _terminate_remote(session, plan)
        telemetry.finish('cancelled', str(error))
        raise

    failures = download_artifacts(plan, workdir)
    required_missing = [
        f for f in failures
        if f.endswith(".refactory") or f.endswith("facts.json")
    ]
    if required_missing:
        telemetry.finish('failed', f"required artifacts missing: {required_missing}")
        return subprocess.CompletedProcess(argv, 1)

    telemetry.finish(reason or ('completed' if exit_code == 0 else 'failed'))
    return subprocess.CompletedProcess(argv, exit_code)


def _poll_run(
    session: Any,
    plan: OpenShellPlan,
    *,
    timeout: float | None,
    stop_condition: Any | None,
    telemetry: Any,
) -> tuple[int, str | None]:
    """Poll the run's exit file until it completes, times out, or is stopped.

    Returns (exit_code, reason).  On timeout or stop the remote process
    is terminated before returning.
    """
    from factory.contained.openshell import RUN_EXIT

    deadline = time.monotonic() + timeout if timeout else None
    while True:
        telemetry.checkpoint()
        if deadline is not None and time.monotonic() >= deadline:
            _terminate_remote(session, plan)
            return 1, 'timed_out'

        if stop_condition is not None:
            try:
                if stop_condition():
                    _terminate_remote(session, plan)
                    return 1, 'stopped'
            except Exception:
                pass

        try:
            result = exec_argv(
                session, ["cat", RUN_EXIT],
                workdir=plan.project_dir, timeout=10,
            )
            if result.exit_code == 0 and result.stdout.strip():
                code = int(result.stdout.strip())
                return code, None
        except (ContainedError, ValueError):
            pass

        time.sleep(5)
