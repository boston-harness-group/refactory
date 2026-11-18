"""Recovery for agent nodes: a timed-out or crashed session gets a fresh one.

The rule lives in the graph (`Node.retry`, canonicalized by
`nn.graph.retry_declaration`, or the default an agent step in a data pass gets:
`nn.graph.Graph.retry_rule`), never in a CEO's judgment, so an automatic headless run
recovers the same way a supervised one does. Both runtimes run it through
:func:`deliver`: the walker from `Executor._visit_agent`, the spine from the worker every
dispatched agent node runs (`worker_node.execute`). The spine reserves the node's whole
envelope for it when it lowers the graph (`workers.lower_workflow`, with
:func:`node_allowance_s`), so the retry runs inside the one command the spine started.

A retry on ``timeout`` answers a session that went silent past the inactivity limit, ran
out of a per-completion limit, or reached the node's own declared wall
(`model.DeclaredTimeout`: the wall and its retry are both the graph's choice; its item is
still recorded ``declared_timeout``), never one the harness backstop stopped
(`model.HarnessBackstop`). It needs the node's own `timeout_s`, which each retry's limit
multiplies. A retry on ``crash`` runs under the node's limit, or the route's when it has
none. A retry after the output limit tells the model its last attempt ran out of room
(`OUTPUT_LIMIT_NOTE`).

An attempt goes through the delivery bracket like any other (`agent_delivery.begin` and
`finish`, which carry the target guard). A timed-out attempt is settled before the next
begins (:func:`settle_for_retry`): if it changed the target, the guard's refusal stands
and nothing is retried; otherwise its delivery is superseded, so the next `begin` sets
its partial outputs aside and the retry writes them afresh. Only a node whose outputs are
its own declared report files may declare a retry (`Node.outputs_only`), which is what
makes setting them aside a reset; a timed-out attempt during which the run's worktree
changed is not retried.

Each retry is checked against the run's budget first, its limit is clamped to what is
left of the node's envelope and of the run's wall, and it is recorded as a
``node_retried`` event that `console.status` counts.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

#: The event a retry is recorded as, in the run's `.refactory/events.jsonl`.
RETRY_EVENT = "node_retried"

#: The shortest session a retry is worth starting. When the node's envelope or the run's
#: wall leaves less than this (or than the retry's own limit, when that is shorter), the
#: timeout stands.
MIN_RETRY_WALL_S = 30

#: How much of the dispatch overhead the node's own attempts may use. The spine reserves
#: every attempt's limit plus `model.NODE_DISPATCH_OVERHEAD_S` (`node_allowance_s`); a
#: session runs a little past its limit while it shuts down, so the attempts may spend
#: half the overhead and still leave the dispatch the other half.
ENVELOPE_SLACK_S = 300


#: What a retry after the output limit adds to the node's prompt: the model used its
#: whole output allowance without answering, and a fresh session with the same prompt
#: tends to do so again.
OUTPUT_LIMIT_NOTE = ("Note: your previous attempt at this step ran out of output room "
                     "before it gave an answer. Keep your working short and commit to an "
                     "answer well before the limit.")


#: Where a step whose retries were spent records why, beside its
#: ``reviews/<step>-missing.txt``: ``<step>.json`` with its cause and session count
#: (`record_spent`, read by `items.missing_steps`).
MISSING_DIR = ".refactory/missing"

#: The runtime behaviour a score depends on beyond the mode's own code: since this
#: marker, a crashed agent step in a data pass is retried and a spent one leaves its item
#: without an answer. Scores and seed caches from before it are not comparable
#: (`profiles.resolve`, `seed_cache.score_context`).
SEMANTICS = "pass-crash-retry/1"


class RetriesExhausted(RuntimeError):
    """A node's session timed out or crashed, and its retries are spent or refused.

    ``kind`` is the `nn.graph.RETRY_ON` failure of the last attempt; ``cause`` says
    more (``output_limit`` for a crash at the output limit); ``sessions`` is how many
    sessions ran."""

    def __init__(self, message: str, *, kind: str = "timeout", cause: str | None = None,
                 sessions: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.cause = cause or kind
        self.sessions = sessions


def leaves_no_answer(error: BaseException) -> bool:
    """Whether ``error`` leaves a data-pass step missing rather than failing its item:
    its retries are spent (`RetriesExhausted`), or the model used its whole output
    allowance without answering, which is the model failing the question and is not
    retried unless the step opts in (``retry.on: ["output_limit"]``)."""
    return isinstance(error, RetriesExhausted) or failure_kind(error) == "output_limit"


#: How the walker's op child hands back a `subprocess.TimeoutExpired` it cannot rebuild
#: from its message (`nn.executor.Executor._reraise_child_error`): a RuntimeError whose
#: text starts with the class name.
_OWN_TIMEOUT_TEXT = "TimeoutExpired: "


def own_timeout(error: BaseException) -> bool:
    """Whether ``error`` is a step running over its own time limit: an op's own
    `subprocess.TimeoutExpired` (not a model session's `model.InactivityTimeout`) or a code
    step's `snippet.SnippetTimeout`, by type or as the walker's op child hands it back
    (``"TimeoutExpired: "``, `items.SNIPPET_TIMEOUT_LINE`). The one classifier both
    runtimes use: recorded ``step_timeout`` (`missing_cause`), never a fallback
    (`falls_back_on`)."""
    from refactory_dsh.items import SNIPPET_TIMEOUT_LINE
    from refactory_dsh.model import InactivityTimeout
    from refactory_dsh.snippet import SnippetTimeout

    if isinstance(error, SnippetTimeout) or (
            isinstance(error, subprocess.TimeoutExpired)
            and not isinstance(error, InactivityTimeout)):
        return True
    return str(error).startswith(_OWN_TIMEOUT_TEXT) or bool(
        SNIPPET_TIMEOUT_LINE.search(str(error)))


def missing_cause(error: BaseException) -> str | None:
    """Why a session's failure left its step without an answer, by the error's type, when
    it ended for time: ``harness_incident`` (`model.HarnessBackstop`), ``declared_timeout``
    (`model.DeclaredTimeout`), ``step_timeout`` (the step's own limit: an op's own
    `subprocess.TimeoutExpired`, a code step's `snippet.SnippetTimeout` -- its fault, never
    a harness timeout) or ``timeout`` (`failure_kind`); else None."""
    from refactory_dsh.model import DeclaredTimeout, HarnessBackstop

    if own_timeout(error):
        return "step_timeout"
    if isinstance(error, HarnessBackstop):
        return "harness_incident"
    if isinstance(error, DeclaredTimeout):
        return "declared_timeout"
    return "timeout" if failure_kind(error) == "timeout" else None


#: The causes and error kinds that are a step running out of time or the harness failing,
#: never the step failing: what `missing_cause` and `items.error_kind` say for one. A
#: fallback never answers one (`falls_back_on`).
TIMEOUT_KINDS = frozenset({"timeout", "declared_timeout", "harness_incident", "step_timeout"})


def falls_back_on(error: BaseException) -> bool:
    """Whether ``error`` is a genuine failure of a step with a fallback (`nn.graph.Graph.
    falls_back`), which leaves it missing so its fallback answers. Never a failure of time
    or of the harness, on either runtime: any timeout (`items.error_kind` in
    `TIMEOUT_KINDS`: a model call the client ended, the backstop, a declared wall, a code
    step's `snippet.WALL_S` (`SnippetTimeout`), an unstarted sandbox (`SandboxUnavailable`),
    and a spent retry carrying one of them as its cause; or a `failure_kind` timeout such
    as an op's own `subprocess.TimeoutExpired`), an op past its per-op timeout
    (`OpTimeoutError`) or whose child vanished (`OpOutcomeUnknown`), an interrupted model
    call (`items.interrupted_cause`: a wire that kept failing, a killed or cancelled
    session), a request the provider refuses for its key, permission or model name
    (``setup``: configuration, the fallback would be refused alike), nor the run's budget
    or a build-loop stop, which end the run as before. A code step past its own wall
    (`SnippetTimeout`) is its code's fault but still time, so it never falls back either.
    The output limit and a prompt too long are the model's or the step's own failure, so
    they do."""
    from refactory_dsh.budgets import BudgetExhausted
    from refactory_dsh.items import error_kind, interrupted_cause
    from refactory_dsh.model import CompletionFailed
    from refactory_dsh.nn.executor import OpOutcomeUnknown, OpTimeoutError
    from refactory_dsh.run import BuildLoopStop

    if isinstance(error, (BudgetExhausted, BuildLoopStop, OpTimeoutError, OpOutcomeUnknown)):
        return False
    if isinstance(error, CompletionFailed) and (
            error.kind == "transport"
            or (error.kind == "setup" and "prompt is too long" not in str(error).lower())):
        return False
    if own_timeout(error):
        return False
    if failure_kind(error) == "timeout" or interrupted_cause(str(error)) is not None:
        return False
    return error_kind(error) not in TIMEOUT_KINDS


#: A declared output a step's visit found absent (`snapshot_outputs`).
_ABSENT = object()


def snapshot_outputs(workdir: Path, writes: Any) -> dict[str, Any]:
    """What a step with a fallback's declared outputs hold before its visit, so a fall back
    restores them (`restore_outputs`): a file's bytes, a directory's whole tree (each
    subdirectory, each file's bytes), or that it was absent."""
    found: dict[str, Any] = {}
    for name in writes or ():
        path = Path(workdir) / str(name)
        if path.is_dir() and not path.is_symlink():
            found[str(name)] = (
                sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_dir()),
                {str(p.relative_to(path)): p.read_bytes()
                 for p in path.rglob("*") if p.is_file()})
        elif path.is_file():
            found[str(name)] = path.read_bytes()
        else:
            found[str(name)] = _ABSENT
    return found


def restore_outputs(workdir: Path, snapshot: dict[str, Any]) -> None:
    """Put a failed step's declared outputs back as its visit found them
    (`snapshot_outputs`) before its fallback runs: what it made goes, file or directory;
    what it only rewrote is as it was; so its fallback sees what it saw and nothing
    half-made is graded."""
    import shutil

    for name, content in snapshot.items():
        path = Path(workdir) / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
        if content is _ABSENT:
            continue
        if isinstance(content, bytes):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            continue
        dirs, files = content
        path.mkdir(parents=True, exist_ok=True)
        for sub in dirs:
            (path / sub).mkdir(parents=True, exist_ok=True)
        for rel, data in files.items():
            (path / rel).parent.mkdir(parents=True, exist_ok=True)
            (path / rel).write_bytes(data)


def record_spent(workdir: Path, node: str, error: BaseException) -> None:
    """Record why a step left without an answer has none (`MISSING_DIR`): its retries
    were spent, it ran out of output, or (an optional branch) it ended for time
    (`missing_cause`). Any other failure records nothing (an earlier visit's record goes),
    and its step's cause reads ``failed`` (`items.missing_steps`)."""
    if isinstance(error, RetriesExhausted):
        cause: str | None = error.cause
    elif failure_kind(error) == "output_limit":
        cause = "output_limit"
    else:
        cause = missing_cause(error)
    target = Path(workdir) / MISSING_DIR / f"{node}.json"
    if cause is None:
        # Only this visit's cause counts: an earlier visit's record must not outlive it.
        target.unlink(missing_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"cause": cause, "sessions": getattr(error, "sessions", None),
                                  "error": str(error)[:2000]}) + "\n")


def clear_spent(workdir: Path, node: str) -> None:
    """A step that answered after it went missing (a gate sent it back) is not missing: so
    a fallback gate reads only its latest visit (`nn.operators.step_failed`). Every step
    that answers clears it, agent, op or code, on both runtimes (`deliver`,
    `nn.executor.Executor._visit`, `run.run_node`)."""
    (Path(workdir) / "reviews" / f"{node}-missing.txt").unlink(missing_ok=True)
    (Path(workdir) / MISSING_DIR / f"{node}.json").unlink(missing_ok=True)


def failure_kind(error: BaseException) -> str | None:
    """The `nn.graph.RETRY_ON` failure ``error`` is, or None when it is none of them.

    A timeout is the model client's deadline (`model.CompletionTimeout`: inactivity, or
    a per-completion limit), a timeout another client reports the same way, or a session
    stopped at its node's declared wall (`model.DeclaredTimeout`): the wall and the retry
    rule a node declares are both the graph's own choice, so a node that declares a retry
    on timeout gets it at its wall (its item is still recorded ``declared_timeout``, the
    pipeline's limit, by `items.error_kind`). A session stopped at the harness backstop
    (`model.HarnessBackstop`) is not one. A crash is a session the worker or product
    failed (`model.CompletionFailed` of kind ``worker``: an invalid result, a worker
    error). ``output_limit`` is the model using its whole output allowance without
    answering: the model's own failure on the question, retried only when a step opts in.
    A wire that kept failing (``transport``) and a key, model name or prompt the provider
    refuses (``setup``) are none of them: the next session would fail the same way.
    Cancellation, budget exhaustion, a refusal and a bad answer are never retried, and
    neither is a session whose runtime did not finish shutting down.
    """
    from refactory_dsh.model import CompletionFailed, CompletionTimeout, DeclaredTimeout

    if getattr(error, "cleanup_unconfirmed", False) or "child cleanup unconfirmed" in str(error):
        # A session whose runtime did not finish shutting down (a timeout that then
        # hung): its tools may still be running, so a fresh session could collide.
        return None
    if isinstance(error, (CompletionTimeout, DeclaredTimeout, subprocess.TimeoutExpired)) \
            or str(error).startswith(_OWN_TIMEOUT_TEXT):
        # The last: an op's own subprocess timeout as the walker's op child hands it back
        # (`own_timeout`), so both runtimes read it alike.
        return "timeout"
    if isinstance(error, RuntimeError) and str(error).startswith("model completion timed out"):
        return "timeout"
    if isinstance(error, CompletionFailed):
        # Only a product failure: a wire that kept failing (a proxy outage) or a request
        # the provider refuses would fail the next session the same way.
        return {"worker": "crash", "output_limit": "output_limit"}.get(error.kind)
    return None


def _cause(error: BaseException, kind: str) -> str:
    """What a `RetriesExhausted` says its last attempt ended on: ``output_limit``, or
    ``declared_timeout`` for a session its node's declared wall stopped
    (`model.DeclaredTimeout`, the pipeline's limit, though its declared retry answered it),
    else the retry kind."""
    from refactory_dsh.model import DeclaredTimeout

    if _output_limit(error):
        return "output_limit"
    if isinstance(error, DeclaredTimeout):
        return "declared_timeout"
    return kind


def _output_limit(error: BaseException) -> bool:
    return getattr(error, "kind", None) == "output_limit"


def attempt_walls(base_s: int, rule: dict[str, Any]) -> list[int]:
    """Each attempt's session limit: the base, then each retry ``timeout_factor`` longer."""
    factor = float(rule["timeout_factor"])
    return [int(math.ceil(base_s * factor ** index)) for index in range(int(rule["times"]) + 1)]


def node_allowance_s(base_s: int, rule: dict[str, Any] | None) -> int:
    """The model time a node's dispatch reserves for all its attempts, before overhead."""
    return sum(attempt_walls(base_s, rule)) if rule else int(base_s)


def _times(count: int) -> str:
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


def _context(client: Any) -> dict[str, Any] | None:
    return client.context() if callable(getattr(client, "context", None)) else None


def _with_wall(client: Any, seconds: int) -> Any:
    bind = getattr(client, "with_retry_wall", None) or getattr(client, "with_wall", None)
    return bind(int(seconds)) if callable(bind) else client


def _worktree_state(workdir: Path) -> dict[str, Any] | None:
    """The run's watched worktree as it is now (the baseline's ``also`` paths), or None
    when the run records no baseline."""
    from refactory_dsh import target_guard

    baseline = target_guard._read(workdir)  # noqa: SLF001
    if baseline is None:
        return None
    return {path: target_guard._also_state(path, Path(workdir))  # noqa: SLF001
            for path in baseline.get("also", [])}


def settle_for_retry(workdir: Path, node: str, reason: str, *,
                     worktree: dict[str, Any] | None = None, failed: str = "timed out") -> None:
    """Close a timed-out attempt's delivery bracket so the next attempt starts clean.

    The target guard judges the attempt first: one that changed the target is refused
    by `agent_delivery.finish`, which restores the target and raises, and that refusal is
    the node's failure -- a retry never runs on top of it. Nor does one whose run's
    worktree changed against ``worktree``, its state from before the attempt (raising
    :class:`RetriesExhausted`). Otherwise the delivery is superseded (`agent_delivery.invalidate`), and the next `begin` sets the attempt's
    outputs aside (`_set_aside_stale_outputs`), so the retry must write every one afresh.
    """
    from refactory_dsh import agent_delivery, target_guard

    pending = agent_delivery._pending(Path(workdir).resolve(), node)  # noqa: SLF001
    if not pending.exists():
        return
    record = json.loads(pending.read_text())
    fail_changes, _warn = target_guard.strayed(Path(workdir).resolve(), record.get("target"))
    if fail_changes:
        agent_delivery.finish(workdir, node)
    if worktree:
        # Only a timed-out attempt is checked, and only to decide whether to retry: a
        # retry would start from whatever the session left there. A sibling branch may
        # be the writer, so this refuses the retry and never the node's delivery.
        now = _worktree_state(workdir) or {}
        changed = [phrase for path, before in worktree.items()
                   for phrase in target_guard._compare(before, now.get(path, before))]  # noqa: SLF001
        if changed:
            raise RetriesExhausted(f"{node} {failed} and was not retried: the run's worktree "
                                   f"changed while it ran ({'; '.join(changed)})",
                                   kind="timeout" if failed == "timed out" else "crash")
    agent_delivery.invalidate(Path(workdir).resolve(), [node], reason)


def _record_retry(workdir: Path, node: str, *, attempt: int, tries: int, waited_s: int,
                  wall_s: int | None, error: BaseException, kind: str = "timeout") -> None:
    from refactory_dsh.live import emit

    if kind == "timeout":
        what = f"timed out after {waited_s}s"
    elif _output_limit(error):
        what = f"reached its output limit without answering after {waited_s}s"
    else:
        what = f"crashed after {waited_s}s"
    limit = f" and {wall_s}s" if wall_s is not None else ""
    emit(workdir, RETRY_EVENT, node_id=node, state="retrying", reason=kind,
         attempt=attempt, attempts=tries, after_s=waited_s,
         **({"wall_s": wall_s} if wall_s is not None else {}),
         error=str(error)[:500],
         summary=f"{node} {what}; retrying it with a fresh session{limit} "
                 f"(attempt {attempt} of {tries})")


def retries(workdir: Path) -> dict[str, int]:
    """How many times each node of the run was retried, from its events."""
    from refactory_dsh.live import events

    counts: dict[str, int] = {}
    for event in events(Path(workdir))[0]:
        if event.get("kind") == RETRY_EVENT and event.get("node_id"):
            counts[str(event["node_id"])] = counts.get(str(event["node_id"]), 0) + 1
    return counts


def deliver(
    workdir: Path,
    node: str,
    *,
    prompt: str,
    writes: list[str],
    reads: list[str],
    client: Any,
    ask: Callable[[Any, str, Any], str],
    capture_final: bool,
    rule: dict[str, Any] | None,
    timeout_s: int | None,
    budget: Any = None,
    workspace: Callable[[bool], AbstractContextManager[Any]] | None = None,
    tool_policy: str = "",
) -> tuple[str, dict[str, Any]]:
    """Run one agent node's model work through the delivery bracket, honouring ``rule``.

    ``workspace(retrying)`` prepares and locks a workspace afresh for each attempt; without
    it the node works in the run directory. ``ask(client, prompt, place)`` makes one
    completion, where ``place`` is that workspace or None. Collection happens before
    the delivery bracket closes. ``timeout_s`` is the node's own bound,
    which a rule multiplies; a rule that answers only crashes needs none, and its
    retries run under the client's own limit. Returns the answer and the delivery
    receipt. With no rule, or a failure the rule does not answer, the first failure is
    raised unchanged. When the last allowed attempt fails too,
    :class:`RetriesExhausted` says how long each session ran.
    """
    from refactory_dsh import workspace as workspaces
    from refactory_dsh.agent_delivery import begin, finish

    workdir = Path(workdir)
    if rule and int(rule["times"]) <= 0:
        rule = None
    if rule and timeout_s is None and "timeout" in rule["on"]:
        raise ValueError(f"{node} declares a retry but no timeout_s for it to multiply")
    tries = 1 + (int(rule["times"]) if rule else 0)
    walls = attempt_walls(int(timeout_s), rule) if rule and timeout_s is not None else []
    #: How long each failed session actually ran: what the failure reports, and what
    #: the node's envelope is charged with.
    ran: list[int] = []
    attempt_client = client
    attempt_prompt = prompt
    for index in range(tries):
        worktree = _worktree_state(workdir) if rule else None
        started = None
        try:
            with workspace(index > 0) if workspace is not None else nullcontext(None) as place:
                delivery = begin(workdir, node, attempt_prompt, list(writes), reads=list(reads),
                                 model_context=_context(attempt_client),
                                 **({"workspace": place.describe()} if place is not None else {}),
                                 tool_policy=tool_policy)
                started = time.monotonic()
                output = ask(attempt_client, delivery["prompt"], place)
                if place is not None:
                    workspaces.collect(place, workdir, writes, final_response=output)
        except Exception as error:
            kind = failure_kind(error)
            if started is not None and isinstance(error, RuntimeError):
                # How many sessions ran, for the step's no-answer record (`record_spent`).
                try:
                    error.sessions = index + 1  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            if started is None or rule is None or kind not in rule["on"]:
                raise
            ran.append(int(round(time.monotonic() - started)))
            spent = ", then ".join(f"{seconds}s" for seconds in ran)
            failed = {"timeout": "timed out", "output_limit": "ran out of output"}.get(kind, "crashed")
            its = "its declared retry" if kind == "timeout" else "its retry"
            if index + 1 >= tries:
                raise RetriesExhausted(
                    f"{node} {failed} {_times(len(ran))} (after {spent}); {its} "
                    f"is spent. Last error: {error}", kind=kind, cause=_cause(error, kind),
                    sessions=len(ran)) from error
            limit: int | None = None
            if walls:
                # The retry's own limit, within what is left of the node's envelope (every
                # attempt's limit plus some of the dispatch overhead) and of the run's wall.
                declared = walls[index + 1]
                bounds = [declared, sum(walls) + ENVELOPE_SLACK_S - sum(ran)]
                remaining = budget.remaining_s() if budget is not None else None
                if remaining is not None:
                    bounds.append(remaining)
                limit = int(min(bounds))
                if limit < min(MIN_RETRY_WALL_S, declared):
                    raise RetriesExhausted(
                        f"{node} {failed} (after {spent}) and {its} was not "
                        f"started: {max(0, limit)}s is left of the node's envelope and the "
                        f"run's wall. Last error: {error}", kind=kind,
                        cause=_cause(error, kind), sessions=len(ran)) from error
            if budget is not None:
                budget.checkpoint()
            settle_for_retry(workdir, node, f"{failed} after {ran[-1]}s; retried by the "
                                            "node's retry rule", worktree=worktree, failed=failed)
            _record_retry(workdir, node, attempt=index + 2, tries=tries, waited_s=ran[-1],
                          wall_s=limit, error=error, kind=kind)
            attempt_client = _with_wall(client, limit) if limit is not None else client
            # A session that used its whole output allowance tends to do so again on the
            # same prompt; the retry is told, and the note is not carried further.
            attempt_prompt = f"{prompt}\n\n{OUTPUT_LIMIT_NOTE}" if _output_limit(error) else prompt
            continue
        receipt = finish(workdir, node, final_response=output, capture_final=capture_final)
        # A step a gate sent back after it went missing has answered now: it is no
        # longer missing (`items.missing_steps`).
        clear_spent(workdir, node)
        return output, receipt
    raise AssertionError("unreachable: every attempt returns or raises")  # pragma: no cover
