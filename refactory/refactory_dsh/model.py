"""A :class:`ModelClient` backed by a one-shot DSH run.

The profile's own model selection (`--provider`, `--model`) is per invocation, so a
caller that names one gets that model for the completions it makes and nothing else
changes.

The outer loop's optimizers need exactly one thing from a model: text in, text
out. DSH already exposes that as ``dsh --profile headless "<task>"``, which is
the same mechanism the inner loop uses to run a mode, so nothing new is
introduced to reach a model and the credentials come from the same place.

The subprocess runs in a scratch directory to separate temporary output. The
configured worker permission policy determines which other paths its tools can access.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any

from refactory_dsh.paths import SOURCE_CLI_ENTRY, dsh_command, dsh_root, node_bin, venv_bin
from refactory_dsh.owned_process import SHUTDOWN_GRACE_S, shutdown, _termination_as_exit

#: The overlay that keeps a node's completion from backgrounding its own work. Checked
#: in rather than generated, so what a node's model call is configured with is
#: reviewable in the repository.
NODE_PATCH = Path(__file__).resolve().parent / "patches" / "agent-node.yml"

#: Wall-clock ceiling for one completion, before the subprocess is called hung.
#:
#: It is a hang detector rather than a budget: the authoring modes' agent nodes are
#: tool-using tasks that write a package, run it and write up what happened, and one
#: of them (`qa_director`, which builds and runs an adversarial probe) exceeded a
#: five-minute ceiling in a live create run. A caller that knows its own nodes can
#: set `timeout_s`; the default has to cover the slowest node a shipped mode declares.
DEFAULT_TIMEOUT_S = 1800

#: Overhead a node's dispatch reserves beyond its completion attempts, for process
#: startup, sandbox discovery and delivery bookkeeping. `workers.lower_workflow`
#: adds it to the declared budget when reserving the spine's node envelope, and the
#: session inside the dispatch runs under the same wall (#307).
NODE_DISPATCH_OVERHEAD_S = 600

# Stay below common per-argument limits as well as the process-wide ARG_MAX.
# DSH headless explicitly supports '-' as a byte-preserving stdin task.
PROMPT_ARGV_LIMIT_BYTES = 64 * 1024


def node_session_budget_s(attempts: int, timeout_s: int) -> int:
    """The wall a node's dispatch reserves: attempts sessions plus overhead.

    One wall, computed one way, so the spine's `command_timeout_s` and the model
    client's session deadline cannot disagree about when a node's work ends. A
    deadline is never retried — replaying a tool-using session can duplicate real
    work — so the session inside the dispatch gets the whole envelope rather than
    the per-completion `timeout_s` alone.
    """
    return max(1, int(attempts)) * int(timeout_s) + NODE_DISPATCH_OVERHEAD_S



#: How many times one completion may be attempted before it is called failed.
#:
#: A transient transport failure is not the node's fault: it is the network, the
#: provider, or a rate limit. One blip used to discard an entire authoring run — the
#: issue that added this retried nothing and lost thirty-five minutes of finished
#: work — so the client retries, and only a client that keeps failing fails the node.
DEFAULT_ATTEMPTS = 3


#: What the child says when the failure was on the wire rather than in the model.
#:
#: `dsh` wraps provider transport errors with `TRANSPORT:`, and the socket-level
#: spellings below are what the same failure looks like when it happens lower down.
#: Deliberately narrow: a run is retried for a *transport* failure, and anything a
#: model or a key or a prompt caused comes back unchanged, because retrying those
#: wastes the next three attempts and hides the real error.
TRANSPORT_MARKERS = (
    "transport:",
    "econnreset",
    "econnrefused",
    "etimedout",
    "enotfound",
    "eai_again",
    "socket hang up",
    "fetch failed",
    "network error",
    "connection reset",
    "rate limit",
    "429",
    "500 internal",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway",
)


def is_transport_failure(text: str) -> bool:
    """Whether a failed completion was a transport failure rather than a bad answer."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in TRANSPORT_MARKERS)


def failure_cause(stderr: str) -> str:
    """The line that says *why* a completion failed, not the model's last paragraph.

    The child's stderr is mostly the model's own reasoning. Prefer the explicit
    diagnostic `dsh` writes. Reporting the tail of the prose made a transport
    failure read like a code-review defect ("Confirmed non-blocking (design fault).
    Let me write the review now."), which sends a reader looking in the wrong place.

    A runtime bootstrap crash may happen before DSH can report anything. Node ends
    those stacks with its version banner; retain the error header instead of that
    banner or a stack frame. Do not expose the full reasoning/stack transcript.
    """
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("dsh:") or is_transport_failure(line):
            return line
    for line in reversed(lines):
        if re.match(r"^(?:[A-Za-z_$][\w$]*Error|Error)(?: \[[A-Z0-9_]+\])?:\s*\S", line):
            return line
    return lines[-1] if lines else ""


class DshModelClient:
    """One-shot completions through ``dsh --profile headless``.

    Implements ``factory.outer_loop.optimizers.ModelClient``. A completion that fails
    on the wire is retried, bounded, with a backoff; one that fails for any other
    reason is raised straight away with the cause named.
    """

    def __init__(
        self,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        model: str = "",
        provider: str = "",
        attempts: int = DEFAULT_ATTEMPTS,
        backoff_s: float = 1.0,
        dsh_home: str | Path | None = None,
        runtime_root: str | Path | None = None,
        dsh_cli: str | None = None,
        worker_backend: str = "",
        effort: str = "",
        permission_mode: str = "",
        node_timeout_s: int | None = None,
        role_routes: dict[str, Any] | None = None,
        node_routes: dict[str, Any] | None = None,
        provider_config: dict[str, Any] | None = None,
        budget_ledger: str | Path | None = None,
        python_bin: str | Path | None = None,
        project: Path | None = None,
        routes_resolved: bool = False,
        routing_provenance: dict[str, Any] | None = None,
    ) -> None:
        from refactory_dsh.workers import validate_route
        inherited = json.loads(os.environ.get("REFACTORY_WORKER_CONTEXT") or "{}")
        if not isinstance(inherited, dict):
            raise ValueError("REFACTORY_WORKER_CONTEXT must contain an object")
        if not routes_resolved and (project is not None or inherited.get("routes_resolved")):
            from refactory_dsh.worker_config import resolve
            selected = resolve(project or Path("."), {key: value for key, value in {
                "worker_backend": worker_backend, "provider": provider, "model": model,
                "effort": effort, "permission_mode": permission_mode,
                "role_routes": role_routes, "node_routes": node_routes,
                "provider_config": provider_config,
            }.items() if value is not None and value != ""}, inherited=inherited)
            route = selected["settings"]
            worker_backend, provider, model, effort, permission_mode = (route[key] for key in
                ("worker_backend", "provider", "model", "effort", "permission_mode"))
            role_routes, node_routes = route["role_routes"], route["node_routes"]
            provider_config = route.get("provider_config") or None
            routing_provenance = inherited.get("routing_provenance", selected["provenance"])
            routes_resolved = True
        routing_provenance = routing_provenance or inherited.get("routing_provenance")
        inherited_route = inherited if not worker_backend or worker_backend == inherited.get("worker_backend") else {}
        worker_backend = worker_backend or inherited.get("worker_backend", "")
        provider = provider if routes_resolved else provider or inherited_route.get("provider", "")
        model = model if routes_resolved else model or inherited_route.get("model", "")
        effort = effort if routes_resolved else effort or inherited_route.get("effort", "")
        # Shared policies retain their meaning across backends. Native-specific
        # values only inherit within their own product, unlike default/skip.
        inherited_permission = inherited_route.get("permission_mode")
        if inherited.get("permission_mode") in ("default", "skip"):
            inherited_permission = inherited["permission_mode"]
        permission_mode = permission_mode or inherited_permission or "skip"
        role_routes = role_routes if role_routes is not None else inherited.get("role_routes")
        node_routes = node_routes if node_routes is not None else inherited.get("node_routes")
        provider_config = provider_config if provider_config is not None else inherited.get("provider_config")
        runtime_root = runtime_root or inherited.get("runtime_root")
        dsh_cli = dsh_cli if dsh_cli is not None else inherited.get("dsh_cli")
        dsh_home = dsh_home or inherited.get("dsh_home")
        budget_ledger = budget_ledger or inherited.get("budget_ledger")
        python_bin = python_bin or inherited.get("python_bin")
        timeout_s = inherited.get("timeout_s", timeout_s) if timeout_s == DEFAULT_TIMEOUT_S else timeout_s
        attempts = inherited.get("attempts", attempts) if attempts == DEFAULT_ATTEMPTS else attempts
        self._timeout_s = timeout_s
        self._node_timeout_s = node_timeout_s or inherited.get("node_timeout_s")
        self._worker_backend = worker_backend or os.environ.get("REFACTORY_WORKER_BACKEND", "dsh")
        self._effort = effort or (os.environ.get("REFACTORY_MODEL_EFFORT", "") if self._worker_backend == "dsh" and not routes_resolved else "")
        self._permission_mode = permission_mode
        self._role_routes = copy.deepcopy(role_routes or {})
        self._node_routes = copy.deepcopy(node_routes or {})
        self._provider_config = copy.deepcopy(provider_config or {})
        self._routing_provenance = copy.deepcopy(routing_provenance or {})
        self._node_id = ""
        self._role = "worker"
        self._worker_receipt: Path | None = None
        ledger = budget_ledger or os.environ.get("REFACTORY_BUDGET_LEDGER")
        self._budget = None
        if ledger:
            from refactory_dsh.budgets import RunBudget
            self._budget = RunBudget.from_path(ledger)
        self._runtime_root = Path(runtime_root or dsh_root()).expanduser().resolve()
        self._python_bin = Path(python_bin or venv_bin()).expanduser().resolve()
        self._dsh_cli = (dsh_cli if dsh_cli is not None else os.environ.get("DSH_CLI", "")).strip()
        self._model = model or (os.environ.get("REFACTORY_MODEL_ID", "") if self._worker_backend == "dsh" and not routes_resolved else "")
        self._provider = provider or (os.environ.get("REFACTORY_MODEL_PROVIDER", "") if self._worker_backend == "dsh" and not routes_resolved else "")
        # Resolve before containment rewrites HOME. This is a location, never a key.
        from refactory_dsh.dsh_profile import model_home
        self._dsh_home = model_home(dsh_home)
        self._attempts = max(1, int(attempts)) if self._worker_backend == "dsh" else 1
        self._backoff_s = max(0.0, float(backoff_s))
        self._cancellations: tuple[threading.Event, ...] = ()
        self._deadline_observations: tuple[tuple[str, float], ...] = ()
        validate_route(self.route())

    def route(self) -> dict[str, str]:
        return {"worker_backend": self._worker_backend, "provider": self._provider,
                "model": self._model, "effort": self._effort,
                "permission_mode": self._permission_mode}

    def for_node(self, node: str, role: str = "worker") -> DshModelClient:
        from refactory_dsh.workers import resolve_route
        route = resolve_route(self.route(), node=node, role=role,
                              role_routes=self._role_routes, node_routes=self._node_routes)
        client = copy.copy(self)
        client._node_id, client._role = node, role
        for key, value in route.items():
            setattr(client, "_" + key, value)
        return client

    def with_budget(self, budget: Any) -> DshModelClient:
        client = copy.copy(self)
        client._budget = budget
        return client

    def with_identity(self, node: str, role: str = "worker") -> DshModelClient:
        """Retain a resolved route while assigning one execution's receipt ID."""
        client = copy.copy(self)
        client._node_id, client._role = node, role
        return client

    def _checkpoint(self, *, final: bool = False) -> None:
        if self._budget is None:
            return
        if self._worker_receipt is not None and self._worker_receipt.exists():
            receipt = json.loads(self._worker_receipt.read_text())
            from refactory_dsh.workers import account_usage
            account_usage(self._budget, receipt, final=final)
        self._budget.checkpoint()

    def with_cancellation(self, event: threading.Event) -> DshModelClient:
        """Bind an independent client copy to one caller-owned cancellation scope.

        Routing and limits are copied as resolved, retaining inherited scopes without
        changing the original client or serializing process-local events.
        """
        client = copy.copy(self)
        client._cancellations = (*self._cancellations, event)
        return client

    def with_node_timeout(self, timeout_s: int | None) -> DshModelClient:
        """Bind the author's stage wall without multiplying it by retry count."""
        client = copy.copy(self)
        if timeout_s is not None:
            if type(timeout_s) is not int or timeout_s <= 0:
                raise ValueError("node timeout must be a positive integer")
            client._node_timeout_s = timeout_s
        return client

    def with_deadline_observation(self, scope: str, stop_at: float) -> DshModelClient:
        """Carry an already enforced caller timer into dispatch telemetry only.

        This never installs, extends or resets a clock. The caller must retain
        its existing cancellation timer; monotonic timestamps stay process-local
        and are intentionally absent from the serialized worker context.
        """
        import math
        if (not isinstance(scope, str) or not scope or scope in ("stage", "run_budget")
                or type(stop_at) not in (int, float) or not math.isfinite(stop_at)):
            raise ValueError("deadline observation needs a named scope and finite monotonic timestamp")
        client = copy.copy(self)
        client._deadline_observations = tuple((name, value) for name, value in self._deadline_observations
                                             if name != scope) + ((scope, stop_at),)
        return client

    def effective_deadlines(self) -> dict[str, Any]:
        wall = self._node_timeout_s
        if wall is None:
            wall = (node_session_budget_s(self._attempts, self._timeout_s)
                    if self._worker_backend == "dsh" and self._node_id else self._timeout_s)
        return {"completion_timeout_s": self._timeout_s, "stage_timeout_s": wall,
                "declared_stage_timeout_s": self._node_timeout_s,
                "shutdown_grace_s": SHUTDOWN_GRACE_S,
                "run_budget": str(self._budget.path) if self._budget is not None else None}

    def _raise_if_cancelled(self) -> None:
        if any(event.is_set() for event in self._cancellations):
            raise CancelledError("model completion cancelled; not retried")

    def context(self) -> dict[str, Any]:
        """Nonsecret settings safe to carry across the model subprocess boundary."""
        return {
            "routes_resolved": True, "routing_provenance": copy.deepcopy(self._routing_provenance),
            "dsh_home": str(self._dsh_home), "provider": self._provider, "model": self._model,
            "timeout_s": self._timeout_s, "attempts": self._attempts, "backoff_s": self._backoff_s,
            **({"node_timeout_s": self._node_timeout_s} if self._node_timeout_s is not None else {}),
            "runtime_root": str(self._runtime_root), "dsh_cli": self._dsh_cli,
            "python_bin": str(self._python_bin),
            "worker_backend": self._worker_backend, "effort": self._effort,
            "permission_mode": self._permission_mode,
            "role_routes": copy.deepcopy(self._role_routes), "node_routes": copy.deepcopy(self._node_routes),
            "provider_config": copy.deepcopy(self._provider_config),
            "budget_ledger": str(self._budget.path) if self._budget is not None else "",
        }

    def _environment(self) -> dict[str, str]:
        # Never resolve a credential here or copy it into the environment. The DSH
        # credentials service reads its managed store in the selected home.
        result = {**os.environ, "DSH_HOME": str(self._dsh_home),
                  "REFACTORY_MODEL_DSH_HOME": str(self._dsh_home),
                  "REFACTORY_MODEL_PROVIDER": self._provider, "REFACTORY_MODEL_ID": self._model,
                  "REFACTORY_MODEL_EFFORT": self._effort, "REFACTORY_WORKER_BACKEND": self._worker_backend,
                  "REFACTORY_RUNTIME_ROOT": str(self._runtime_root)}
        if self._budget is not None:
            result["REFACTORY_BUDGET_LEDGER"] = str(self._budget.path)
        if self._worker_receipt is not None:
            result["REFACTORY_WORKER_RECEIPT"] = str(self._worker_receipt)
        from refactory_dsh.workers import environment
        result.update(environment(self.context()))
        return result

    def _readiness_unavailable(self, reason: str) -> dict[str, object]:
        return {
            "ready": None, "status": "diagnostic-unavailable", "reason": reason,
            "provider": self._provider or "(runtime default)",
            "model": self._model or "(runtime default)", "authenticated": False,
            "warning": "Credential diagnostic unavailable for the selected runtime/configuration; "
                       "credential readiness is unverified. Normal execution will perform its own checks.",
        }

    def readiness(self, *, workdir: Path, require_ready: bool = True) -> dict[str, object]:
        """Describe local configuration; report known failures only when require_ready is false."""
        if self._worker_backend != "dsh":
            from refactory_dsh.workers import PACKAGES, capabilities
            available = (self._runtime_root / "node_modules" / PACKAGES[self._worker_backend] / "package.json").exists()
            if not available and require_ready:
                raise RuntimeError(f"optional {self._worker_backend} worker provider is not installed")
            return {**self._readiness_unavailable("native-worker-authentication-unverified"),
                    "worker_backend": self._worker_backend, "installed": available,
                    "capabilities": capabilities(self._worker_backend)}
        # Inspect only the runtime this client will actually launch. A custom CLI
        # or source checkout need not use the repository's installed composition.
        if self._dsh_cli:
            return self._readiness_unavailable("external-launcher")
        if (self._runtime_root / SOURCE_CLI_ENTRY).exists():
            return self._readiness_unavailable("source-launcher")
        probe = Path(__file__).with_name("model_readiness.mjs")
        try:
            proc = subprocess.run(
                [node_bin(), str(probe), str(self._runtime_root), self._provider, self._model],
                cwd=str(workdir), env=self._environment(), capture_output=True, text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return self._readiness_unavailable("diagnostic-process-unavailable")
        try:
            result = json.loads(proc.stdout)
        except (ValueError, TypeError):
            # Provider parse errors can quote document values: never relay stderr.
            return self._readiness_unavailable("diagnostic-response-unavailable")
        if not isinstance(result, dict):
            return self._readiness_unavailable("diagnostic-response-unavailable")
        if result.get("status") in ("missing-credential", "invalid-configuration"):
            if require_ready:
                raise RuntimeError(str(result.get("error") or "model credential configuration is invalid"))
            return result
        if proc.returncode or not (
            result.get("ready") is True
            or (result.get("ready") is None and result.get("status") == "diagnostic-unavailable")
        ):
            return self._readiness_unavailable("diagnostic-response-unavailable")
        return result

    def complete(
        self, prompt: str, *, system: str = "", workdir: Path | None = None
    ) -> str:
        from refactory_dsh import workers
        scratch = Path(os.environ.get("REFACTORY_MODEL_SCRATCH", "/tmp")) / "refactory-model"
        run_dir = Path(workdir).expanduser().resolve() if workdir is not None else scratch
        run_dir.mkdir(parents=True, exist_ok=True)
        client = copy.copy(self)
        client._checkpoint()
        if client._budget is not None:
            client._budget.require_usage_capability(self._worker_backend == "dsh")
        client._worker_receipt = workers.begin(run_dir, self.route(), node=self._node_id, role=self._role,
                                               effective_deadlines=client.effective_deadlines())
        try:
            result = client._complete(prompt, system=system, workdir=run_dir)
            client._checkpoint(final=True)
        except BaseException as error:
            state = "failed"
            if "cleanup unconfirmed" in str(error):
                state = "unknown"
            elif isinstance(error, (CancelledError, KeyboardInterrupt, SystemExit)):
                state = "cancelled"
            elif "timed out" in str(error):
                state = "timed_out"
            elif type(error).__name__ == "BudgetExhausted":
                state = "budget_exhausted"
            if client._budget is not None:
                receipt = json.loads(client._worker_receipt.read_text())
                workers.account_usage(client._budget, receipt, final=True)
            workers.finish(client._worker_receipt, state, error=str(error))
            raise
        workers.finish(client._worker_receipt, "completed")
        return result

    def _complete(
        self, prompt: str, *, system: str = "", workdir: Path | None = None
    ) -> str:
        """Return the model's answer to ``prompt``.

        ``workdir`` is where the completion runs, and it is not a detail: the agent
        this spawns has tools and writes files with relative paths, so a completion
        with the checkout as its working directory writes a node's artifacts into the
        repository. The reviewing nodes did exactly that — their `reviews/*.md`
        appeared at the checkout root while the run's own `reviews/` stayed empty, and
        the QA synthesis, reading the run directory, found no reviews at all. A caller
        with no run directory to name gets the scratch directory instead.

        Raises ``RuntimeError`` on a non-zero exit or timeout. The loop survives
        that and records it: a failed round contributes no proposals and lands in
        ``Optimizer.proposal_failures``, so a run whose proposer never answered
        says so in its report instead of looking like a search that found nothing.
        Raising is what makes the failure the caller's to handle -- this is the
        only place a dead key, a wrong model name and an exhausted quota, which all
        present as "the score did not move", are distinguishable from a mode that
        simply has no better version.
        """
        task = f"{system}\n\n{prompt}" if system else prompt
        # The launcher resolves its own entry; the child's working directory stays
        # the node's, not the checkout's.
        cmd = [
            *dsh_command(root=self._runtime_root, cli=self._dsh_cli),
            "--profile",
            "headless",
            # A node is one completion whose answer is the node's output, so a model
            # that backgrounds its subagents and ends its turn has not done the node's
            # work. See the overlay for why the base profile's default is wrong here.
            "--patch",
            str(NODE_PATCH),
        ]
        # The headless profile takes these for one invocation only: nothing is
        # written back to settings, so one optimized run against another model cannot
        # change what the next one uses.
        if self._worker_receipt is not None:
            from refactory_dsh.workers import write_patch
            patch = self._worker_receipt.with_suffix(".patch.yml")
            write_patch(patch, self._runtime_root, self.route(), provider_config=self._provider_config)
            cmd += ["--patch", str(patch)]
        if self._worker_backend == "dsh":
            if self._provider:
                cmd += ["--provider", self._provider]
            if self._model:
                cmd += ["--model", self._model]
        cmd.append(task)

        scratch = Path(os.environ.get("REFACTORY_MODEL_SCRATCH", "/tmp")) / "refactory-model"
        run_dir = Path(workdir).expanduser().resolve() if workdir is not None else scratch
        run_dir.mkdir(parents=True, exist_ok=True)

        last = ""
        # A node's dispatch is one session whose deadline is never retried, so it
        # runs under the whole envelope `workers.lower_workflow` reserved for it —
        # attempts sessions plus overhead — rather than the per-completion
        # `timeout_s` alone (#307). A CEO turn or a console call keeps the plain
        # completion deadline. The wall is absolute across attempts: a transport
        # retry spends whatever remains of the same envelope.
        if self._node_timeout_s is not None:
            budget_s = self._node_timeout_s
            stop_at = time.monotonic() + budget_s
        elif self._worker_backend == "dsh" and self._node_id:
            budget_s = node_session_budget_s(self._attempts, self._timeout_s)
            stop_at = time.monotonic() + budget_s
        else:
            budget_s = self._timeout_s
            stop_at = None
        for attempt in range(1, self._attempts + 1):
            self._raise_if_cancelled()
            self._checkpoint()
            try:
                if self._worker_receipt is not None:
                    from refactory_dsh.workers import dispatch_deadlines
                    dispatch_deadlines(self._worker_receipt, budget=self._budget,
                                       stage_stop_at=stop_at, stage_timeout_s=budget_s,
                                       attempt=attempt, enclosing_deadlines=self._deadline_observations)
                proc = self._run_completion(cmd, run_dir, stop_at, budget_s)
            except subprocess.TimeoutExpired as exc:
                # A tool-using completion may already have changed its workdir.
                # A deadline is not a transport failure: replaying the prompt can
                # duplicate real work even after its subprocesses have shut down.
                raise RuntimeError(
                    f"model completion timed out after {budget_s}s; "
                    "requested runtime shutdown; not retried"
                ) from exc
            else:
                self._raise_if_cancelled()
                if proc.returncode == 0:
                    return proc.stdout
                cause = failure_cause(proc.stderr)
                if not is_transport_failure(proc.stderr):
                    # The model answered wrongly, the key is dead, the prompt is
                    # rejected: retrying spends two more completions on the same
                    # answer. Fail now and keep the diagnosis accurate.
                    raise RuntimeError(
                        f"model completion failed (exit {proc.returncode}): {cause}"
                    )
                last = cause or f"exit {proc.returncode}"
            if attempt < self._attempts:
                delay = self._backoff_s * (2 ** (attempt - 1))
                if not self._cancellations:
                    time.sleep(delay)
                else:
                    deadline = time.monotonic() + delay
                    while True:
                        self._raise_if_cancelled()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        # The innermost event wakes immediately; inherited scopes
                        # are also observed within the bounded polling interval.
                        self._cancellations[-1].wait(min(remaining, 0.1))
        raise RuntimeError(
            f"model completion failed on the wire {self._attempts} time(s); last cause: {last}"
        )

    def _run_completion(
        self, cmd: list[str], run_dir: Path, stop_at: float | None = None, budget_s: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Keep the CLI alive long enough to dispose its managed tool processes.

        ``subprocess.run(timeout=...)`` kills the immediate child before raising,
        bypassing the DSH CLI's SIGTERM handler and stranding detached tool jobs.
        Do not signal process names, inferred descendants, or unrelated groups.
        ``stop_at`` is an absolute monotonic deadline shared across a node's
        attempts; without one, the call is a plain completion bounded by
        ``timeout_s``.
        """
        from contextlib import ExitStack
        import hashlib
        import tempfile

        timeout_s = budget_s if budget_s is not None else self._timeout_s
        with _termination_as_exit(), ExitStack() as transport:
            self._raise_if_cancelled()
            launch = cmd
            stdin = None
            task_bytes = cmd[-1].encode('utf-8')
            transport_kind = 'argv'
            if self._worker_backend == 'dsh' and len(task_bytes) > PROMPT_ARGV_LIMIT_BYTES:
                # Installed dsh-headless resolves '-' by reading all stdin as
                # UTF-8, without trimming task content. File-backed stdin avoids
                # both argv limits and a blocking parent write before polling.
                # Anonymous/private storage remains inside this worker's cwd;
                # ExitStack closes it on success, failed spawn, timeout or stop.
                stdin = transport.enter_context(tempfile.TemporaryFile(dir=run_dir))
                stdin.write(task_bytes)
                stdin.seek(0)
                launch = [*cmd[:-1], '-']
                transport_kind = 'stdin_file'
            if self._worker_receipt is not None:
                from refactory_dsh.workers import _write_receipt
                record = json.loads(self._worker_receipt.read_text())
                record['prompt_transport'] = {'kind': transport_kind, 'encoding': 'utf-8',
                                              'bytes': len(task_bytes),
                                              'sha256': hashlib.sha256(task_bytes).hexdigest()}
                _write_receipt(self._worker_receipt, record)
            environment = self._environment()
            # File preparation, hashing, receipt writes and environment setup
            # spend the existing absolute allowance. Recheck immediately at the
            # spawn boundary so expired/cancelled work never starts a runtime.
            self._checkpoint()
            self._raise_if_cancelled()
            if stop_at is not None and time.monotonic() >= stop_at:
                raise subprocess.TimeoutExpired(launch, timeout_s)
            proc = subprocess.Popen(
                launch, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=str(run_dir), env=environment,
                # A terminal Ctrl-C reaches the Python caller only. Forward one
                # shutdown request: DSH escalates a second signal to immediate exit.
                start_new_session=os.name == "posix",
            )
            try:
                deadline = stop_at if stop_at is not None else time.monotonic() + self._timeout_s
                while True:
                    self._raise_if_cancelled()
                    self._checkpoint()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(cmd, timeout_s)
                    try:
                        stdout, stderr = proc.communicate(
                            timeout=min(remaining, 0.1)
                        )
                        break
                    except subprocess.TimeoutExpired:
                        # communicate retains its output across bounded waits. There
                        # is no stdin input to resend, and the deadline never resets.
                        if time.monotonic() >= deadline:
                            raise
            except BaseException:
                # Direct interruption or the fork's cooperative event both use the
                # runtime-owned cleanup before propagating failure.
                shutdown(proc)
                raise
            return subprocess.CompletedProcess(launch, proc.returncode, stdout, stderr)
