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
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any, Callable, ClassVar

from refactory_dsh.paths import SOURCE_CLI_ENTRY, dsh_command, dsh_root, node_bin, venv_bin
import importlib.metadata

from refactory_dsh.invocations import HOOK_SUPPRESS, WORKTREE_RECORD
from refactory_dsh.owned_process import SHUTDOWN_GRACE_S, shutdown, _termination_as_exit


class CompletionTimeout(RuntimeError):
    """A model session ran out of its time. The client never replays it: a tool-using
    session may already have changed its workdir. A node that declares a retry on
    ``timeout`` gets a fresh session instead (`refactory_dsh.node_retry`)."""


class CompletionFailed(RuntimeError):
    """A model session ended without an answer, and the client did not replay it.

    ``kind`` says what happened (`failure_kind`): ``output_limit`` (the model used its
    whole output allowance without answering), ``transport`` (the wire kept failing
    through every attempt), ``setup`` (a key, a model name or a prompt the provider
    refuses: the next session fails the same way) or ``worker`` (the worker or the
    product failed). In a data pass the runtime gives a crashed agent step a fresh
    session (`refactory_dsh.node_retry`, ``retry.on: ["crash"]``); the message is the
    same ``model completion failed ...`` text as before.
    """

    def __init__(self, message: str, *, kind: str = "worker") -> None:
        super().__init__(message)
        self.kind = kind


class InactivityTimeout(subprocess.TimeoutExpired):
    """The session showed no model output and ran no tool command for ``inactive_s``."""

    def __init__(self, cmd: Any, inactive_s: float) -> None:
        super().__init__(cmd, inactive_s)
        self.inactive_s = inactive_s


def _etime_seconds(text: str) -> float:
    """Seconds in ps's ``[[dd-]hh:]mm:ss`` elapsed-time format."""
    days, _, clock = text.rpartition("-")
    parts = [float(part) for part in clock.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds + (float(days) * 86400 if days else 0.0)


def tool_command_running_since(root_pid: int, since: float) -> bool:
    """Whether a process beneath ``root_pid`` started at or after ``since`` (epoch seconds).

    The session's tool commands are its descendants, and a long test run starts after the
    last model output. A helper the runtime started at the beginning predates the latest
    output and does not count. A failed look reports no activity: the receipt still counts.
    """
    try:
        table = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,etime="], capture_output=True,
                               text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    now = time.time()
    children: dict[int, list[tuple[int, float]]] = {}
    for line in table.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            children.setdefault(int(fields[1]), []).append((int(fields[0]), now - _etime_seconds(fields[2])))
        except ValueError:
            continue
    pending, seen = [root_pid], {root_pid}
    while pending:
        for pid, started in children.get(pending.pop(), []):
            if pid in seen:
                continue
            seen.add(pid)
            # ps counts whole seconds, so a process can look up to a second older.
            if started >= since - 1.0:
                return True
            pending.append(pid)
    return False


class ModelChoiceError(Exception):
    """A run has no model: no flag, no route and no stored choice (dm-258).

    An ordinary exception, not a ``SystemExit``: the walker and the outer loop are
    library code and must never end their process from deep inside. Each
    command-line entry point catches it, prints the message and exits 1.
    """

    def __init__(self, message: str = "") -> None:
        from refactory_dsh.dsh_profile import NO_MODEL_CHOICE_MESSAGE

        super().__init__(message or NO_MODEL_CHOICE_MESSAGE)


def credential_source(home: str | Path | None = None,
                      workdir: str | Path | None = None) -> dict[str, Any]:
    """Where a nested model command finds its credential, naming sources only.

    The model client never reads a key itself: DSH's credentials service reads the
    managed store (and a dotenv beside it) in the selected model home, and the harness's
    providers answer through the environment (Vertex, a base-URL proxy). This reports the
    first source the client would rely on, by name or path, and never a value. Exported
    provider keys are named last because nested workers scrub them, so on their own they
    are not a credential for a worker.
    """
    from refactory_dsh.dsh_profile import model_home

    selected = model_home(home)
    checked: list[str] = []
    for name in ("REFACTORY_MODEL_DSH_HOME", "DSH_HOME"):
        if (os.environ.get(name) or "").strip():
            checked.append(name)
            break
    store = selected / ".credentials.yaml"
    checked.append(str(store))
    if store.is_file():
        return {"configured": True, "source": f"credential store {store}"}
    dotenv = selected / ".env"
    checked.append(str(dotenv))
    if dotenv.is_file():
        return {"configured": True, "source": f"dotenv {dotenv}"}
    if workdir is not None:
        project = Path(workdir) / ".env"
        checked.append(str(project))
        if project.is_file():
            return {"configured": True, "source": f"dotenv {project}"}
    vertex = (os.environ.get("CLAUDE_CODE_USE_VERTEX") or "").strip().lower()
    if vertex not in ("", "0", "false"):
        return {"configured": True, "source": "Vertex (CLAUDE_CODE_USE_VERTEX)"}
    for name in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "REFACTORY_MODEL_BASE_URL"):
        if (os.environ.get(name) or "").strip():
            checked.append(name)
            return {"configured": True, "source": f"base-url proxy ({name})"}
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        if (os.environ.get(name) or "").strip():
            # A worker's own environment reaches DSH, so this runs the top-level worker;
            # the scrubbed nested paths (checks, validation) do not see it.
            checked.append(f"{name} (scrubbed for nested workers)")
            return {"configured": True,
                    "source": f"environment ({name}; scrubbed for nested workers)"}
    return {"configured": False, "source": None, "checked": checked}


def half_route_refusal(provider: str, model: str) -> str:
    """What a route naming only one of provider and model is refused with; ``""`` otherwise.

    One message for the gate, ``doctor`` and ``refactory runtime``, so all three say
    the same thing about the same route.
    """
    if bool(provider) == bool(model):
        return ""
    given, missing = (f"provider {provider}", "model") if provider else (f"model {model}", "provider")
    return (f"No model chosen: the run names {given} but no {missing}; pass both --provider and "
            "--model, or pick one in Settings → Models or with "
            "`refactory model set --provider <provider> --model <model>`.")


def complete_route(provider: str, model: str, *, interactive: bool = False,
                   home: str | Path | None = None) -> tuple[str, str]:
    """A complete DSH route, or :class:`ModelChoiceError`: the gate before any DSH process.

    A route that a flag, a route map or the stored choice completed passes unchanged.
    One with neither a provider nor a model gets the first-run choice
    (:func:`refactory_dsh.dsh_profile.first_run_choice`), which stores it or refuses.
    Half a route is refused: the harness would fill the other half from its own
    default, and a provider named without a model never borrows the stored model of
    another provider (``worker_config.resolve``).
    """
    if provider and model:
        return provider, model
    refusal = half_route_refusal(provider, model)
    if refusal:
        raise ModelChoiceError(refusal)
    from refactory_dsh.dsh_profile import first_run_choice

    choice = first_run_choice(home, interactive=interactive)
    return choice["provider"], choice["model"]


def require_model_choice(client: Any, *, interactive: bool = False) -> Any:
    """``client`` with a complete DSH route (:func:`complete_route`), or ``ModelChoiceError``.

    A client of another backend (Codex, Claude Code, OpenShell) keeps its native
    settings, and an object that is not a DSH client (a test's fake) passes as it is.
    """
    if getattr(client, "_worker_backend", None) != "dsh":
        return client
    route = complete_route(client._provider, client._model, interactive=interactive,
                           home=getattr(client, "_dsh_home", None))
    if route == (client._provider, client._model):
        return client
    chosen = copy.copy(client)
    chosen._provider, chosen._model = route
    return chosen


#: The worker backends that can run a call with no tools at all (tool_policy "none").
TOOL_POLICY_NONE_BACKENDS = ("dsh", "claude_code")

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
#:
#: Overridable via ``REFACTORY_MODEL_TIMEOUT_S`` for workloads whose nodes run much
#: longer than the authoring modes — e.g. a node that builds a container (podman)
#: inside its own session can exceed 30 minutes. The default stays conservative;
#: such a workload exports the env var rather than every consumer inheriting a high
#: ceiling. (Same env-driven pattern as REFACTORY_WORKER_BACKEND / REFACTORY_MODEL_EFFORT.)
DEFAULT_TIMEOUT_S = int(os.environ.get("REFACTORY_MODEL_TIMEOUT_S", "1800"))

#: How long a node's model session may show no sign of life before it is stopped (seconds).
#:
#: A node's declared `timeout_s` is a hard maximum, a backstop; this is the limit that
#: normally ends a stuck session. A session is alive while it produces model output (the
#: worker receipt the DSH observer rewrites on every request and finished response), or
#: while a tool command it started is still running (a process started after the last
#: receipt write, beneath the session). A builder busy on a 6-minute test run is alive; a
#: silent one is stopped after this long. Overridable via ``REFACTORY_MODEL_INACTIVITY_S``,
#: or per client with ``inactivity_s``; 0 turns it off. Applies to node sessions only.
DEFAULT_INACTIVITY_S = int(os.environ.get("REFACTORY_MODEL_INACTIVITY_S", "600"))

#: How often a session's activity is looked at (capped at a quarter of the limit).
ACTIVITY_POLL_S = 2.0

#: Overhead a node's dispatch reserves beyond its completion attempts, for process
#: startup, sandbox discovery and delivery bookkeeping. `workers.lower_workflow`
#: adds it to the declared budget when reserving the spine's node envelope, and the
#: session inside the dispatch runs under the same wall (#307).
NODE_DISPATCH_OVERHEAD_S = 600

_UNSET: object = object()
_environment_overlay: Callable[[Path], dict[str, str] | None] | None | object = _UNSET


def get_environment_overlay() -> Callable[[Path], dict[str, str] | None] | None:
    """Return the registered overlay, discovering it via entry points on first call."""
    global _environment_overlay
    if _environment_overlay is _UNSET:
        eps = importlib.metadata.entry_points(group='refactory.environment_overlay')
        if len(eps) > 1:
            raise RuntimeError(
                f"expected at most one 'refactory.environment_overlay' entry point, "
                f"got {len(eps)}: {[ep.name for ep in eps]}"
            )
        if eps:
            ep = next(iter(eps))
            _environment_overlay = ep.load()
        else:
            _environment_overlay = None
    return _environment_overlay


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

#: How long the waits between those attempts add up to, in seconds, however many attempts
#: there are. A DNS or network blip lasts up to about a minute; retries one and two
#: seconds apart all landed inside one and failed the run (dm-218).
RETRY_WINDOW_S = 60.0


def retry_delays(attempts: int, window_s: float = RETRY_WINDOW_S) -> list[float]:
    """The wait before each retry of ``attempts``: each twice the last, together ``window_s``."""
    retries = max(0, int(attempts) - 1)
    if retries == 0:
        return []
    unit = max(0.0, float(window_s)) / (2 ** retries - 1)
    return [unit * 2 ** index for index in range(retries)]


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
    """Whether a failed completion was a transport failure rather than a bad answer.

    A bare status code counts only as a number of its own: "after 142903 output tokens"
    is not an HTTP 429."""
    lowered = (text or "").lower()
    return any(re.search(rf"(?<!\d){marker}(?!\d)", lowered) if marker.isdigit() else marker in lowered
               for marker in TRANSPORT_MARKERS)


#: What a failure says when the next session would fail the same way: a key, a model
#: name or a prompt the provider refuses. Such a failure is never a crash to retry.
SETUP_FAILURE_MARKERS = (
    "authentication",
    "unauthorized",
    "invalid api key",
    "invalid x-api-key",
    "permission_error",
    "not_found_error",
    "model not found",
    "prompt is too long",
)


#: What a product failure's detail says when the provider, not the model, failed: the
#: same wire failure a `TRANSPORT_MARKERS` line is, reported inside a product result.
PROVIDER_FAILURE_MARKERS = ("overloaded", "connection error", "rate_limit", "api_error",
                            "api error: 5")


def failure_kind(cause: str) -> str:
    """What a failed completion's diagnostic line (`failure_cause`) says happened.

    ``output_limit``: the model used its whole output allowance without answering --
    the DSH runner's ``dsh: max-tokens:`` or the Claude Code adapter's
    ``category: output-limit``. ``setup`` (`SETUP_FAILURE_MARKERS`): the next session
    would be refused the same way. ``transport``: the wire or the provider failed, also
    when a product result reports it (`PROVIDER_FAILURE_MARKERS`). Otherwise ``worker``:
    the worker or the product failed. Only ``output_limit`` and ``worker`` are crashes a
    node's retry answers (`node_retry.CRASH_KINDS`).
    """
    lowered = (cause or "").lower()
    if lowered.startswith("dsh: max-tokens:") or "category: output-limit" in lowered:
        return "output_limit"
    if any(marker in lowered for marker in SETUP_FAILURE_MARKERS):
        return "setup"
    if is_transport_failure(lowered) or any(marker in lowered for marker in PROVIDER_FAILURE_MARKERS):
        return "transport"
    return "worker"


#: A worker failure in the teardown stage after the product CLI itself exited 0: the
#: CLI finished, and DSH's cleanup after it (e.g. waiting for the worker's systemd
#: scope to empty) failed, typically under load. The stage and the clean exit code
#: must be in the same failure group (``(... stage: teardown; ...; exit code: 0)``),
#: so a teardown failure after a nonzero exit, a teardown group with no exit code,
#: or a failure in any other stage is not matched.
_CLEAN_EXIT_TEARDOWN = re.compile(r"stage:\s*teardown;[^)]*exit code:\s*0(?!\d)", re.IGNORECASE)


def is_clean_exit_teardown_failure(cause: str) -> bool:
    """Whether a failure cause -- the one diagnostic line ``failure_cause`` picks, not
    the whole stderr, which is mostly streamed reasoning -- is a worker teardown
    failure after a clean CLI exit: an infrastructure failure, not a bad answer."""
    return bool(_CLEAN_EXIT_TEARDOWN.search(cause or ""))


#: The line the worker bridge (`worker_bridge.mjs`) prints when its cleanup fails after
#: the answer is on stdout; it then exits as the worker did, 0 for a completed answer.
TEARDOWN_WARNING = "dsh: worker teardown failed after its answer:"

#: The live event a kept answer whose worker cleanup failed records (a warning).
TEARDOWN_EVENT = "worker_teardown_failed"
#: The live event every completion the client re-asks records.
RETRY_EVENT = "model_call_retried"
#: The live event a timed-out completion records when its runtime then did not finish
#: shutting down (a warning: the completion is still a timeout).
SHUTDOWN_EVENT = "model_shutdown_unconfirmed"


def teardown_warning(stderr: str) -> str:
    """The bridge's warning that cleanup failed after the answer, or ``""``."""
    return next((line.strip() for line in (stderr or "").splitlines()
                 if line.startswith(TEARDOWN_WARNING)), "")


def answered_before_teardown(stdout: str, stderr: str) -> bool:
    """Whether a failed completion is a finished answer whose cleanup failed: stdout holds
    the answer, the failure is the worker's own ``dsh: worker failed:`` line for a teardown
    after a clean CLI exit (not a turn whose error merely quotes one), and the worker reported
    no other stop (``dsh: worker cancelled: ...``). The bridge itself exits 0 then
    (`TEARDOWN_WARNING`); this keeps the answer from a runtime that does not."""
    cause = failure_cause(stderr)
    if (not (stdout or "").strip() or not cause.startswith("dsh: worker failed:")
            or not is_clean_exit_teardown_failure(cause)):
        return False
    return not any(line.startswith("dsh: worker ") and not is_clean_exit_teardown_failure(line)
                   and not line.startswith(TEARDOWN_WARNING)
                   for line in (stderr or "").splitlines())


def is_retryable_failure(stderr: str) -> bool:
    """Whether ``DshModelClient.complete`` retries a failed completion: a transport
    failure only. A worker that answered and then failed to clean up keeps its answer
    (`answered_before_teardown`), and anything a model, key or prompt caused is not
    retried. Only the diagnostic lines decide (`diagnostic_lines`): the model's streamed
    reasoning may say "429" or "rate limit" about the question it was solving."""
    return is_transport_failure(without_product_detail("\n".join(diagnostic_lines(stderr))))


def diagnostic_lines(stderr: str) -> list[str]:
    """``stderr`` without the model's streamed reasoning: after the ``dsh: reasoning:``
    header, only the lines ``dsh`` itself writes (``dsh: ...``) are diagnostics; every
    other line is the model's own text."""
    lines: list[str] = []
    reasoning = False
    for line in (stderr or "").splitlines():
        stripped = line.strip()
        if stripped == REASONING_HEADER:
            reasoning = True
        elif not reasoning or stripped.startswith("dsh:"):
            lines.append(line)
    return lines


#: A product adapter's failure line: fixed facts in parentheses, then (since the Claude
#: Code adapter names what happened) a free-text detail that may quote the provider.
_PRODUCT_FAILURE = re.compile(r"(Product subagent failure \([^)]*\)):[^\n]*")


def without_product_detail(text: str) -> str:
    """``text`` with each product failure line cut to its fixed facts.

    The detail can quote the provider ("API Error: 429 ... rate_limit_error"). It says
    what happened, but it must never decide whether the client replays a session: a
    session with tools has done real work, and only the fixed facts and the wire's own
    errors decide that (`is_retryable_failure`)."""
    return _PRODUCT_FAILURE.sub(r"\1", text or "")


#: What the headless runner writes before it streams the model's reasoning.
REASONING_HEADER = "dsh: reasoning:"

#: The cause when dsh exited after streaming reasoning and wrote no diagnostic.
NO_DIAGNOSTIC = "dsh: the model turn ended without completing and dsh gave no reason"


def failure_cause(stderr: str) -> str:
    """The line that says *why* a completion failed, not the model's last paragraph.

    The child's stderr is mostly the model's own reasoning. Prefer the explicit
    diagnostic `dsh` writes. Reporting the tail of the prose made a transport
    failure read like a code-review defect ("Confirmed non-blocking (design fault).
    Let me write the review now."), which sends a reader looking in the wrong place.

    A runtime bootstrap crash may happen before DSH can report anything. Node ends
    those stacks with its version banner; retain the error header instead of that
    banner or a stack frame. Do not expose the full reasoning/stack transcript.

    ``dsh: reasoning:`` is not a diagnostic but the header before the streamed
    reasoning. Taking it for one made a completion stopped at its output token
    limit read "dsh: reasoning:". With no diagnostic after it, the cause says so
    rather than quoting the reasoning.
    """
    # The bridge's warning that cleanup failed after an answer is not why a completion
    # failed: a worker that stopped reports its own diagnostic above it.
    lines = [line.strip() for line in (stderr or "").splitlines()
             if line.strip() and not line.strip().startswith(TEARDOWN_WARNING)]
    for line in reversed(lines):
        if line == REASONING_HEADER:
            continue
        if line.startswith("dsh:") or is_transport_failure(line):
            return line
    if REASONING_HEADER in lines:
        return NO_DIAGNOSTIC
    for line in reversed(lines):
        if re.match(r"^(?:[A-Za-z_$][\w$]*Error|Error)(?: \[[A-Z0-9_]+\])?:\s*\S", line):
            return line
    return lines[-1] if lines else ""


class DshModelClient:
    """One-shot completions through ``dsh --profile headless``.

    Implements ``factory.outer_loop.optimizers.ModelClient``. A completion that fails
    on the wire (`is_retryable_failure`) is retried, up to ``attempts`` in all, and each
    retry is recorded (`RETRY_EVENT`); a worker whose cleanup failed after it answered
    keeps its answer, with a warning (`TEARDOWN_EVENT`). Retries come
    with waits that double and add up to ``retry_window_s`` (`retry_delays`); one that
    fails for any other reason is raised straight away with the cause named. Retries
    happen on the ``dsh`` backend, and on any backend for a session without tools; a
    native session with tools makes one attempt (`retry_attempts`), since replaying it
    could repeat real work. ``backoff_s`` is the schedule a saved worker spec
    or run configuration from before ``retry_window_s`` carries: when it is given, the
    first wait is ``backoff_s`` and each next one doubles, as it was then.
    """

    #: One client may serve concurrent callers (a fork's branches, a data node's
    #: parallel items): every completion is its own ``dsh`` subprocess, and every
    #: ``with_*``/``for_node`` binding is a copy. Read by
    #: ``nn.data_parallel.is_shareable_model``; a custom client opts in the same way.
    thread_safe: ClassVar[bool] = True


    def __init__(
        self,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        model: str = "",
        provider: str = "",
        attempts: int = DEFAULT_ATTEMPTS,
        backoff_s: float | None = None,
        retry_window_s: float = RETRY_WINDOW_S,
        dsh_home: str | Path | None = None,
        runtime_root: str | Path | None = None,
        dsh_cli: str | None = None,
        worker_backend: str = "",
        effort: str = "",
        permission_mode: str = "",
        role_routes: dict[str, Any] | None = None,
        node_routes: dict[str, Any] | None = None,
        provider_config: dict[str, Any] | None = None,
        budget_ledger: str | Path | None = None,
        python_bin: str | Path | None = None,
        project: Path | None = None,
        routes_resolved: bool = False,
        routing_provenance: dict[str, Any] | None = None,
        node_wall_s: int | None = None,
        tool_policy: str = "",
        inactivity_s: int | None = None,
        max_completion_tokens: str = "",
        setting_sources: str = "",
        dsh_sessions: str = "",
    ) -> None:
        from refactory_dsh.workers import session_store_choice, validate_route
        inherited = json.loads(os.environ.get("REFACTORY_WORKER_CONTEXT") or "{}")
        if not isinstance(inherited, dict):
            raise ValueError("REFACTORY_WORKER_CONTEXT must contain an object")
        if not routes_resolved and (project is not None or inherited.get("routes_resolved")):
            from refactory_dsh.worker_config import resolve
            selected = resolve(project or Path("."), {key: value for key, value in {
                "worker_backend": worker_backend, "provider": provider, "model": model,
                "effort": effort, "permission_mode": permission_mode,
                "setting_sources": setting_sources,
                "role_routes": role_routes, "node_routes": node_routes,
                "provider_config": provider_config,
            }.items() if value is not None and value != ""}, inherited=inherited)
            route = selected["settings"]
            worker_backend, provider, model, effort, permission_mode = (route[key] for key in
                ("worker_backend", "provider", "model", "effort", "permission_mode"))
            # The run-level settings sources a flag or the project file names; a
            # route map's per-node entry still overrides them (`for_node`).
            setting_sources = route.get("setting_sources", "")
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
        #: Set only when a caller chose it; otherwise `DEFAULT_INACTIVITY_S` applies.
        inactivity_s = inactivity_s if inactivity_s is not None else inherited.get("inactivity_s")
        self._inactivity_s = int(inactivity_s) if inactivity_s is not None else None
        self._worker_backend = worker_backend or os.environ.get("REFACTORY_WORKER_BACKEND", "dsh")
        self._effort = effort or (os.environ.get("REFACTORY_MODEL_EFFORT", "") if self._worker_backend == "dsh" and not routes_resolved else "")
        # Both ride `context()`, so a dispatched worker rebuilt from it (`worker_node`)
        # takes them as arguments, as the walker's client inherits them.
        self._max_completion_tokens = max_completion_tokens or inherited.get("max_completion_tokens", "")
        #: Claude Code's settings sources (`workers.CLAUDE_CODE_SETTING_SOURCES`); empty
        #: keeps the adapter's default.
        #: Inherited only within the same product, as effort is: a DSH client in a
        #: Claude Code worker's child process must not take Claude Code's sources.
        self._setting_sources = setting_sources or inherited_route.get("setting_sources", "")
        #: Where each completion's DSH process keeps its sessions (`workers.session_store`),
        #: carried in `context()` when it is ``shared``, so a worker rebuilt from its spec
        #: keeps it; a context without the field falls back to `workers.SESSIONS_ENV`.
        self._dsh_sessions = session_store_choice(dsh_sessions or inherited.get("dsh_sessions"))
        self._permission_mode = permission_mode
        self._role_routes = copy.deepcopy(role_routes or {})
        self._node_routes = copy.deepcopy(node_routes or {})
        self._provider_config = copy.deepcopy(provider_config or {})
        self._routing_provenance = copy.deepcopy(routing_provenance or {})
        self._node_id = ""
        self._role = "worker"
        #: A node's own bound on its whole model session (`Node.timeout_s`), when
        #: it declares one: the session stops there whatever the route allows.
        self._node_wall_s = int(node_wall_s) if node_wall_s is not None else None
        #: Whether that wall is the session's limit even beyond the route's envelope: a
        #: declared retry's longer session (`with_retry_wall`). Otherwise it only lowers it.
        self._node_wall_exact = False
        self._worker_receipt: Path | None = None
        #: Set per completion on the copy `complete` makes, as the receipt is: where the
        #: agent works when that is not the run directory, what its environment
        #: overrides, and whether its tools are confined there (`workspace.py`).
        self._cwd: Path | None = None
        self._env_overrides: dict[str, str | None] = {}
        self._confine = False
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
        # The one model choice is the bottom layer of every DSH route (dm-258): a route
        # no flag, environment, project file or route map completed — including a saved
        # context from before there was a choice — takes it from the DSH home this client
        # was given (else the person's), never the profile default the harness would
        # otherwise use. Half a route is left for the gate to refuse.
        if self._worker_backend == "dsh" and not self._provider and not self._model:
            from refactory_dsh.dsh_profile import stored_model_choice

            stored = stored_model_choice(dsh_home)
            self._provider, self._model = stored["provider"], stored["model"]
        if tool_policy not in ("", "none"):
            raise ValueError(f"invalid tool_policy {tool_policy!r}")
        self._tool_policy = tool_policy
        self._configured_attempts = max(1, int(attempts))
        self._backoff_s = None if backoff_s is None else max(0.0, float(backoff_s))
        self._retry_window_s = max(0.0, float(retry_window_s))
        self._cancellations: tuple[threading.Event, ...] = ()
        validate_route(self.route())

    def retry_attempts(self) -> int:
        """How many completions one ``complete`` call may make: the configured
        ``attempts`` on the ``dsh`` backend, and on any backend for a session without
        tools (``tool_policy == "none"``), whose replay has no side effects; 1 for a
        native session with tools, whose replay could repeat real work."""
        configured = max(1, int(getattr(self, "_configured_attempts", 1)))
        if (getattr(self, "_worker_backend", "dsh") == "dsh"
                or getattr(self, "_tool_policy", "") == "none"):
            return configured
        return 1

    @property
    def _attempts(self) -> int:
        return self.retry_attempts()

    @_attempts.setter
    def _attempts(self, value: int) -> None:
        self._configured_attempts = max(1, int(value))

    def route(self) -> dict[str, str]:
        result = {"worker_backend": self._worker_backend, "provider": self._provider,
                  "model": self._model, "effort": self._effort,
                  "permission_mode": self._permission_mode}
        if self._max_completion_tokens:
            result["max_completion_tokens"] = self._max_completion_tokens
        if getattr(self, "_setting_sources", ""):
            result["setting_sources"] = self._setting_sources
        return result

    def output_limit(self) -> int | None:
        """The output token limit this client's completions run under, when one is known.

        The smaller of the model's configured ``maxTokens`` and the route's
        ``max_completion_tokens``, as ``worker_observer.mjs`` applies them.
        """
        values: list[Any] = [getattr(self, "_max_completion_tokens", "")]
        provider = (getattr(self, "_provider_config", None) or {}).get(self._provider) or {}
        for entry in provider.get("models") or ():
            if isinstance(entry, dict) and entry.get("id") == self._model:
                values.append(entry.get("maxTokens"))
        known: list[int] = []
        for value in values:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                known.append(number)
        return min(known) if known else None

    def for_node(self, node: str, role: str = "worker") -> DshModelClient:
        from refactory_dsh.workers import resolve_route
        route = resolve_route(self.route(), node=node, role=role,
                              role_routes=self._role_routes, node_routes=self._node_routes)
        if route["worker_backend"] == "dsh" and not route.get("provider") and not route.get("model"):
            # A route map that switched this node back to DSH starts from the one choice too.
            from refactory_dsh.dsh_profile import stored_model_choice

            route.update(stored_model_choice(self._dsh_home))
        client = copy.copy(self)
        client._node_id, client._role = node, role
        for key, value in route.items():
            setattr(client, "_" + key, value)
        return client

    def with_wall(self, seconds: int) -> DshModelClient:
        """A copy whose node session ends after ``seconds``, however the route is set."""
        client = copy.copy(self)
        client._node_wall_s = int(seconds)
        return client

    def with_retry_wall(self, seconds: int) -> DshModelClient:
        """A copy whose node session runs for exactly ``seconds``: a declared retry's
        longer limit (`node_retry.deliver`), which may exceed the route's envelope."""
        client = self.with_wall(seconds)
        client._node_wall_exact = True
        return client

    def with_tool_policy(self, policy: str) -> DshModelClient:
        """A copy whose model call enforces the given tool policy."""
        if policy not in ("", "none"):
            raise ValueError(f"invalid tool_policy {policy!r}")
        client = copy.copy(self)
        client._tool_policy = policy
        return client

    def with_budget(self, budget: Any) -> DshModelClient:
        client = copy.copy(self)
        client._budget = budget
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

    def _raise_if_cancelled(self) -> None:
        if any(event.is_set() for event in self._cancellations):
            raise CancelledError("model completion cancelled; not retried")

    def context(self) -> dict[str, Any]:
        """Nonsecret settings safe to carry across the model subprocess boundary."""
        return {
            "routes_resolved": True, "routing_provenance": copy.deepcopy(self._routing_provenance),
            "dsh_home": str(self._dsh_home), "provider": self._provider, "model": self._model,
            "timeout_s": self._timeout_s,
            # The configured count, so a client rebuilt from this context and given
            # tool_policy "none" keeps its retries; `retry_attempts` applies the rule.
            "attempts": getattr(self, "_configured_attempts", 1), "retry_window_s": self._retry_window_s,
            **({} if self._backoff_s is None else {"backoff_s": self._backoff_s}),
            "runtime_root": str(self._runtime_root), "dsh_cli": self._dsh_cli,
            "python_bin": str(self._python_bin),
            "worker_backend": self._worker_backend, "effort": self._effort,
            "permission_mode": self._permission_mode,
            "role_routes": copy.deepcopy(self._role_routes), "node_routes": copy.deepcopy(self._node_routes),
            "provider_config": copy.deepcopy(self._provider_config),
            "budget_ledger": str(self._budget.path) if self._budget is not None else "",
            # getattr: a client built without __init__ (as a boundary test does) has none.
            **({"node_wall_s": wall} if (wall := getattr(self, "_node_wall_s", None)) is not None
               else {}),
            **({"tool_policy": tp} if (tp := getattr(self, "_tool_policy", ""))
               else {}),
            **({"inactivity_s": ia} if (ia := getattr(self, "_inactivity_s", None)) is not None
               else {}),
            **({"max_completion_tokens": mct} if (mct := getattr(self, "_max_completion_tokens", ""))
               else {}),
            **({"setting_sources": ss} if (ss := getattr(self, "_setting_sources", "")) else {}),
            # Only the non-default choice, so a default context is the bytes it was.
            **({"dsh_sessions": ds} if (ds := getattr(self, "_dsh_sessions", "run")) != "run" else {}),
        }

    def _environment(self, *, target_workdir: "Path | None" = None) -> dict[str, str]:
        # Never resolve a credential here or copy it into the environment. The DSH
        # credentials service reads its managed store in the selected home.
        result = {**os.environ, "DSH_HOME": str(self._dsh_home),
                  "REFACTORY_MODEL_DSH_HOME": str(self._dsh_home),
                  "REFACTORY_MODEL_PROVIDER": self._provider, "REFACTORY_MODEL_ID": self._model,
                  "REFACTORY_MODEL_EFFORT": self._effort, "REFACTORY_WORKER_BACKEND": self._worker_backend,
                  "REFACTORY_RUNTIME_ROOT": str(self._runtime_root)}
        # Clear inherited values because this client owns the current call.
        result.pop("REFACTORY_SYSTEM_PROMPT", None)
        # Tool policy: set when this node has one, clear for ordinary nodes so a
        # previous no-tools dispatch's env var does not leak to the next node.
        if getattr(self, '_tool_policy', '') == 'none':
            result["REFACTORY_TOOL_POLICY"] = "none"
        else:
            result.pop("REFACTORY_TOOL_POLICY", None)
        # Only DSH's own system-prompt plugin reads this. An external worker gets the
        # text in its adapter config (`workers.write_patch`), so a large prompt never
        # meets the kernel's 128 KiB limit on one environment value.
        if getattr(self, '_system_text', '') and getattr(self, '_worker_backend', 'dsh') in ('dsh', 'openshell'):
            result["REFACTORY_SYSTEM_PROMPT"] = self._system_text
        if self._budget is not None:
            result["REFACTORY_BUDGET_LEDGER"] = str(self._budget.path)
        if self._worker_receipt is not None:
            result["REFACTORY_WORKER_RECEIPT"] = str(self._worker_receipt)
        from refactory_dsh.workers import environment
        result.update(environment(self.context()))
        # Late-bind the target's environment: when the run's worktree exists,
        # replace whatever PATH the subprocess inherited (which may still lead
        # with refactory's venv from startup.ts or _node_environment) with the
        # target's check_environment PATH.  Without a worktree (research,
        # strategy), the inherited PATH passes through.
        _workdir = str(target_workdir) if target_workdir is not None else os.environ.get('REFACTORY_WORKDIR', '')
        if _workdir:
            _overlay = get_environment_overlay()
            if _overlay is not None:
                _target_env = _overlay(Path(_workdir))
                if _target_env is not None:
                    result['PATH'] = _target_env['PATH']
                    if 'VIRTUAL_ENV' in _target_env:
                        result['VIRTUAL_ENV'] = _target_env['VIRTUAL_ENV']
            elif (Path(_workdir) / WORKTREE_RECORD).exists():
                raise RuntimeError(
                    "the worktree record exists but no environment overlay is installed; "
                    "re-run `uv sync` or `./install.sh`"
                )
        # An agent's own shell can run install commands that would rewrite the
        # target's hooks; suppress hook managers there too (#347).
        result.update(HOOK_SUPPRESS)
        # Last: an agent workspace's own environment (its PATH, HOME and virtual
        # environment) replaces the run worktree's, and None removes a name.
        for name, value in (getattr(self, "_env_overrides", None) or {}).items():
            if value is None:
                result.pop(name, None)
            else:
                result[name] = value
        # Owned policy must win over arbitrary caller/workspace environment.
        if getattr(self, '_tool_policy', '') == 'none':
            result['REFACTORY_TOOL_POLICY'] = 'none'
        else:
            result.pop('REFACTORY_TOOL_POLICY', None)
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
        self, prompt: str, *, system: str = "", workdir: Path | None = None,
        cwd: Path | None = None, env: dict[str, str | None] | None = None,
        confine: bool = False,
    ) -> str:
        """One completion. ``workdir`` is the run directory: the worker's receipt and
        patch are written there, and the run's worktree environment is found from it.
        ``cwd`` is where the agent works, the run directory unless given (an agent
        workspace, `workspace.py`); ``env`` overrides its environment last, a None value
        removing the name; ``confine`` makes DSH's sandbox policy workspace-write, rooted
        at ``cwd`` (`workers.permission_patch`).
        """
        from refactory_dsh import workers
        scratch = Path(os.environ.get("REFACTORY_MODEL_SCRATCH", "/tmp")) / "refactory-model"
        original_workdir = workdir
        run_dir = Path(workdir).expanduser().resolve() if workdir is not None else scratch
        run_dir.mkdir(parents=True, exist_ok=True)
        client = copy.copy(self)
        client._cwd = Path(cwd).expanduser().resolve() if cwd is not None else None
        client._env_overrides = dict(env or {})
        client._confine = bool(confine and cwd is not None)
        client._checkpoint()
        if client._budget is not None:
            client._budget.require_usage_capability(self._worker_backend == "dsh")
        client._worker_receipt = workers.begin(run_dir, self.route(), node=self._node_id, role=self._role,
                                               tool_policy=getattr(self, '_tool_policy', ''))
        try:
            result = client._complete(prompt, system=system, workdir=run_dir, target_workdir=original_workdir)
            client._checkpoint(final=True)
        except BaseException as error:
            state = "failed"
            if isinstance(error, (CancelledError, KeyboardInterrupt, SystemExit)):
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
        finally:
            if original_workdir is None:
                # The scratch directory is every caller's, so it keeps no session store;
                # a run directory keeps its own beside the receipt (`workers.session_store`).
                store = workers.session_store(client._worker_receipt, getattr(client, "_dsh_sessions", None))
                if store is not None:
                    shutil.rmtree(store, ignore_errors=True)
        workers.finish(client._worker_receipt, "completed")
        return result

    def _complete(
        self, prompt: str, *, system: str = "", workdir: Path | None = None,
        target_workdir: "Path | None" = None,
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
        # The last line of defence: whatever path built this client, no DSH process
        # starts without a complete route, so the headless profile's default never answers.
        if self._worker_backend == "dsh" and not (self._provider and self._model):
            raise ModelChoiceError()
        if getattr(self, '_tool_policy', '') == 'none' and self._worker_backend not in TOOL_POLICY_NONE_BACKENDS:
            raise ValueError(f"{self._worker_backend} cannot enforce tool_policy='none'")
        self._system_text = system
        task = prompt
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
            cwd = getattr(self, "_cwd", None)
            from refactory_dsh.workspace import tool_dsh_home
            from refactory_dsh.workers import session_store
            write_patch(patch, self._runtime_root, self.route(), provider_config=self._provider_config,
                        workspace_root=cwd if getattr(self, "_confine", False) else None,
                        tool_home=tool_dsh_home(cwd) if cwd is not None else None,
                        system_prompt=system,
                        tool_policy=getattr(self, '_tool_policy', ''),
                        session_store=session_store(self._worker_receipt,
                                                    getattr(self, "_dsh_sessions", None)))
            cmd += ["--patch", str(patch)]
        if self._worker_backend == "dsh":
            if self._provider:
                cmd += ["--provider", self._provider]
            if self._model:
                cmd += ["--model", self._model]
        cmd.append("-")

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
        #
        # ``cap_s`` is the longest one attempt may run; ``total_s`` the node's limit
        # across its attempts, when it has one. A node with a wall has that one limit on
        # every backend: an attempt runs for at most ``cap_s`` and never past the wall,
        # so a retry spends what remains of the wall, not a fresh `timeout_s`, and gets
        # no less than an attempt's own time while the wall allows it.
        dsh_node = self._worker_backend == "dsh" and bool(self._node_id)
        cap_s = node_session_budget_s(self._attempts, self._timeout_s) if dsh_node else self._timeout_s
        total_s: int | None = cap_s if dsh_node else None
        wall = getattr(self, "_node_wall_s", None)
        if wall is not None:
            exact = getattr(self, "_node_wall_exact", False)
            if dsh_node:
                if wall < cap_s or exact:
                    cap_s = total_s = wall
            else:
                total_s = wall
                if wall < cap_s or exact:
                    cap_s = wall
        stop_at = time.monotonic() + total_s if total_s is not None else None
        delays = (retry_delays(self._attempts, self._retry_window_s) if self._backoff_s is None
                  else [self._backoff_s * 2 ** index for index in range(self._attempts - 1)])
        for attempt in range(1, self._attempts + 1):
            self._raise_if_cancelled()
            self._checkpoint()
            now = time.monotonic()
            if stop_at is not None and attempt > 1 and stop_at - now <= 0:
                # The wait before this retry used up the node's time: an attempt now
                # would time out at once. It failed on the wire, and says so -- not a
                # timeout, which a node's timeout retry would run again.
                raise CompletionFailed(
                    f"model completion failed on the wire {attempt - 1} time(s) and the node's "
                    f"time ran out before it could ask again; last cause: {last}",
                    kind="transport")
            attempt_stop = now + cap_s if stop_at is None else min(stop_at, now + cap_s)
            budget_s = max(1, int(round(attempt_stop - now)))
            try:
                proc = self._run_completion(cmd, run_dir, attempt_stop, budget_s, task=task,
                                            target_workdir=target_workdir)
            except subprocess.TimeoutExpired as exc:
                # The deadline passed first, so this is a timeout even when the runtime
                # then failed to shut down: that failure is the timeout's detail and a
                # recorded warning, and the error still says it timed out.
                unconfirmed = getattr(exc, "shutdown_failure", "")
                if unconfirmed:
                    who = self._node_id or "a model call"
                    self._emit(run_dir, SHUTDOWN_EVENT, state="warning", error=unconfirmed[:500],
                               summary=f"{who}: timed out, and its runtime did not finish shutting "
                                       "down; child cleanup unconfirmed")
                detail = f"; {unconfirmed}" if unconfirmed else ""
                if isinstance(exc, InactivityTimeout):
                    error = CompletionTimeout(
                        f"model completion timed out after {int(exc.inactive_s)}s of inactivity "
                        "(no model output and no running tool command); requested runtime "
                        f"shutdown; the client does not replay a deadline{detail}")
                else:
                    # A tool-using completion may already have changed its workdir.
                    # A deadline is not a transport failure: replaying the prompt can
                    # duplicate real work even after its subprocesses have shut down.
                    error = CompletionTimeout(
                        f"model completion timed out after {budget_s}s; "
                        f"requested runtime shutdown; the client does not replay a deadline{detail}")
                error.cleanup_unconfirmed = bool(unconfirmed)  # type: ignore[attr-defined]
                raise error from exc
            else:
                self._raise_if_cancelled()
                if proc.returncode == 0:
                    if warning := teardown_warning(proc.stderr):
                        self._record_teardown(run_dir, warning)
                    return proc.stdout
                cause = failure_cause(proc.stderr)
                if answered_before_teardown(proc.stdout, proc.stderr):
                    # The worker answered and only its cleanup failed: the answer is
                    # the node's. Re-asking would discard it, and repeat a tool
                    # session's work.
                    self._record_teardown(run_dir, cause)
                    return proc.stdout
                if cause.startswith("dsh: max-tokens:") and (limit := self.output_limit()):
                    # Name the limit: a reader, or an optimizer, cannot weigh a
                    # shorter answer against a larger cap without the number.
                    cause = f"{cause} ({limit} tokens)"
                if not is_retryable_failure(proc.stderr) or failure_kind(cause) != "transport":
                    # The model answered wrongly, the key is dead, the prompt is
                    # rejected: retrying spends two more completions on the same
                    # answer. Fail now and keep the diagnosis accurate. A transport
                    # failure is infrastructure and is retried within the same
                    # budget. A crashed session (a worker failure, an output limit) is
                    # not replayed here either: a replay would run outside the node's
                    # delivery bracket, unrecorded, with the same prompt. A node's retry
                    # rule runs it again as a fresh, recorded session (`node_retry`).
                    # The cause itself must be the wire's: a session that stopped for
                    # any other reason is not called a wire failure, whatever else its
                    # diagnostics mention.
                    raise CompletionFailed(
                        f"model completion failed (exit {proc.returncode}): {cause}",
                        kind=failure_kind(cause))
                last = cause or f"exit {proc.returncode}"
            if attempt < self._attempts:
                self._record_retry(run_dir, attempt, last)
                delay = delays[attempt - 1]
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
        raise CompletionFailed(
            f"model completion failed on the wire {self._attempts} time(s); last cause: {last}",
            kind="transport")

    def _receipt(self) -> dict[str, Any]:
        try:
            return json.loads(self._worker_receipt.read_text()) if self._worker_receipt else {}
        except (OSError, ValueError):
            return {}

    def _emit(self, run_dir: Path, kind: str, **fields: Any) -> None:
        """Record one live event in the run directory; a record that cannot be written
        never fails the completion it describes."""
        from refactory_dsh.live import emit

        try:
            emit(run_dir, kind, node_id=self._node_id or "",
                 worker_id=self._receipt().get("worker_id"), **fields)
        except OSError:
            pass

    def _record_teardown(self, run_dir: Path, detail: str) -> None:
        who = self._node_id or "a model call"
        tools = getattr(self, "_tool_policy", "") != "none"
        # With tools, the session's tool processes may outlive the failed cleanup.
        self._emit(run_dir, TEARDOWN_EVENT, state="warning", error=detail[:500],
                   tools=tools, cleanup="unconfirmed",
                   summary=f"{who}: the worker's cleanup failed after it answered; the answer is kept"
                           + ("; its tool processes may still be running" if tools else ""))

    def _record_retry(self, run_dir: Path, attempt: int, cause: str) -> None:
        """A completion the client is about to ask again: why, which attempt failed, and
        what the discarded attempt cost (the worker observer's ``session_usage``)."""
        receipt = self._receipt()
        # The session's own usage once the observer writes it (none when it died before
        # reporting); a receipt from before any session_usage has only its total.
        usage = receipt["session_usage"] if "session_usage" in receipt else receipt.get("usage")
        who = self._node_id or "a model call"
        self._emit(run_dir, RETRY_EVENT, state="retrying", cause=cause[:500], attempt=attempt,
                   attempts=self._attempts, usage=usage,
                   summary=f"{who}: attempt {attempt} of {self._attempts} failed on the wire; "
                           "asking again")

    def _inactivity_limit(self) -> float:
        """The inactivity limit for this session: node sessions on DSH only, else 0 (off)."""
        if self._worker_backend != "dsh" or not self._node_id:
            return 0
        limit = self._inactivity_s if self._inactivity_s is not None else DEFAULT_INACTIVITY_S
        return max(0, limit)

    def _last_activity(self, pid: int, started_epoch: float, now: float) -> float:
        """When (monotonic) the session last showed life: model output, or a tool running."""
        receipt = getattr(self, "_worker_receipt", None)
        written = started_epoch
        try:
            if receipt is not None:
                written = max(written, receipt.stat().st_mtime)
        except OSError:
            pass
        if tool_command_running_since(pid, written):
            return now
        return now - max(0.0, time.time() - written)

    def _run_completion(
        self, cmd: list[str], run_dir: Path, stop_at: float | None = None, budget_s: int | None = None,
        task: str | None = None, target_workdir: "Path | None" = None,
    ) -> subprocess.CompletedProcess[str]:
        """Keep the CLI alive long enough to dispose its managed tool processes.

        ``subprocess.run(timeout=...)`` kills the immediate child before raising,
        bypassing the DSH CLI's SIGTERM handler and stranding detached tool jobs.
        Do not signal process names, inferred descendants, or unrelated groups.
        ``stop_at`` is an absolute monotonic deadline shared across a node's
        attempts; without one, the call is a plain completion bounded by
        ``timeout_s``.

        When ``task`` is not ``None`` the prompt is delivered on the child's stdin
        via a seekable temporary file, keeping large prompts off the command line
        where they would exceed ``MAX_ARG_STRLEN``.
        """
        timeout_s = budget_s if budget_s is not None else self._timeout_s
        inactivity_s = self._inactivity_limit()
        with _termination_as_exit():
            self._raise_if_cancelled()
            task_file = None
            try:
                if task is not None:
                    task_file = tempfile.TemporaryFile("w+", encoding="utf-8")
                    task_file.write(task)
                    task_file.seek(0)
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    stdin=task_file if task_file is not None else None,
                    cwd=str(getattr(self, "_cwd", None) or run_dir),
                    env=self._environment(target_workdir=target_workdir),
                    # A terminal Ctrl-C reaches the Python caller only. Forward one
                    # shutdown request: DSH escalates a second signal to immediate exit.
                    start_new_session=os.name == "posix",
                )
                try:
                    deadline = stop_at if stop_at is not None else time.monotonic() + self._timeout_s
                    last_activity = time.monotonic()
                    started_epoch = time.time()
                    next_look = last_activity
                    look_every = min(ACTIVITY_POLL_S, inactivity_s / 4) if inactivity_s else 0.0
                    while True:
                        self._raise_if_cancelled()
                        self._checkpoint()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(cmd, timeout_s)
                        if inactivity_s:
                            now = time.monotonic()
                            if now >= next_look:
                                next_look = now + look_every
                                last_activity = max(last_activity, self._last_activity(
                                    proc.pid, started_epoch, now))
                            if now - last_activity >= inactivity_s:
                                raise InactivityTimeout(cmd, inactivity_s)
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
                except BaseException as error:
                    # Direct interruption or the fork's cooperative event both use the
                    # runtime-owned cleanup before propagating failure.
                    try:
                        shutdown(proc)
                    except RuntimeError as unconfirmed:
                        if not isinstance(error, subprocess.TimeoutExpired):
                            raise
                        # The deadline had already passed: the call timed out, and a
                        # shutdown that then hangs does not change that. The timeout
                        # goes on, carrying the shutdown failure (`_complete`).
                        error.shutdown_failure = str(unconfirmed)  # type: ignore[attr-defined]
                    raise
                return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
            finally:
                if task_file is not None:
                    task_file.close()
