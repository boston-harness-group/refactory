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
#: The event an answer that named no verdict is recorded as (`settle_verdict`).
VERDICT_REASK_EVENT = "verdict_reasked"

#: The clock `deliver` reads for its timing: `time.monotonic` by default, monkeypatched
#: to a fake clock in tests that must be deterministic under load.
_clock: Callable[[], float] = time.monotonic

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

#: What the retry of a step with no tools (``tool_policy`` ``none``) adds to its prompt
#: when its reply wrote out tool calls or their results (`agent_json.invented_tool_calls`):
#: nothing ran, so any result in it was made up.
NO_TOOLS_NOTE = ("Note: your previous reply to this step was refused because it wrote out "
                 "tool calls or tool results. This step has no tools, so nothing was run and "
                 "those results were invented. Answer directly from the prompt and your own "
                 "reasoning, without writing tool calls or their results.")

#: What a retry after a timeout with ``keep_workspace`` adds to the node's prompt when its
#: workspace was kept (`KEPT_SYNCS`): it holds the previous attempt's unfinished edits; the
#: builder is told how to see them.
KEEP_WORKSPACE_NOTE = ("Your workspace holds the previous attempt's unfinished edits. "
                       "Start with `git status` and `git diff --stat` against the base "
                       "to see them, then continue from there.")

#: How a workspace was synced (`workspace.Workspace.synced`) when it still holds the last
#: attempt's work: a clone reset, re-cloned or made afresh does not.
KEPT_SYNCS = frozenset({"kept", "verified"})

#: The ``reason`` of the `RETRY_EVENT` for that retry.
INVENTED_TOOL_CALLS = "invented_tool_calls"


#: Where a step whose retries were spent records why, beside its
#: ``reviews/<step>-missing.txt``: ``<step>.json`` with its cause and session count
#: (`record_spent`, read by `items.missing_steps`).
MISSING_DIR = ".refactory/missing"

#: The runtime behaviour a score depends on beyond the mode's own code: since
#: ``pass-crash-retry/1``, a crashed agent step in a data pass is retried and a spent one
#: leaves its item without an answer; since ``invented-tool-calls/1``, a step with no
#: tools whose reply writes out tool calls is asked once more and then left without an
#: answer (`refuses_tool_markup`). Scores and seed caches from before it are not
#: comparable (`profiles.resolve`, `seed_cache.score_context`).
SEMANTICS = "invented-tool-calls/1"


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


class InventedToolCalls(RuntimeError):
    """A step with no tools answered with tool calls or results it wrote itself, again
    after it was told it has none (or with no time left to ask again): its reply is
    refused, never taken as its output. In a data pass it leaves the step without an
    answer, as the output limit does (`leaves_no_answer`). ``sessions`` is how many ran."""

    def __init__(self, message: str, *, sessions: int | None = None) -> None:
        super().__init__(message)
        self.sessions = sessions


def refuses_tool_markup(tool_policy: Any, keeps_tool_markup: Any = False) -> bool:
    """Whether a step's reply that writes out tool calls is refused and the step asked once
    more (`deliver`): a step with no tools (``tool_policy`` ``none``) that does not keep
    such a reply (`nn.graph.Node.keeps_tool_markup`). The spine reserves that extra session
    in the step's envelope by the same test (`workers.lower_workflow`)."""
    return tool_policy == "none" and not keeps_tool_markup


def leaves_no_answer(error: BaseException) -> bool:
    """Whether ``error`` leaves a data-pass step missing rather than failing its item:
    its retries are spent (`RetriesExhausted`), or the model used its whole output
    allowance without answering, which is the model failing the question and is not
    retried unless the step opts in (``retry.on: ["output_limit"]``), or a step with no
    tools wrote out tool calls again after it was told (`InventedToolCalls`)."""
    return (isinstance(error, (RetriesExhausted, InventedToolCalls))
            or failure_kind(error) == "output_limit")


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
    it ended for time: ``backstop`` (`model.HarnessBackstop`: the model still writing, its
    own failure), ``declared_timeout`` (`model.DeclaredTimeout`), ``step_timeout`` (the
    step's own limit: an op's own `subprocess.TimeoutExpired`, a code step's
    `snippet.SnippetTimeout` -- its fault, never a harness timeout) or ``timeout``
    (`failure_kind`); else None."""
    from refactory_dsh.model import DeclaredTimeout, HarnessBackstop

    if own_timeout(error):
        return "step_timeout"
    if isinstance(error, HarnessBackstop):
        return "backstop"
    if isinstance(error, DeclaredTimeout):
        return "declared_timeout"
    return "timeout" if failure_kind(error) == "timeout" else None


#: The causes and error kinds that are a step running out of time or the harness failing,
#: never the step failing: what `missing_cause` and `items.error_kind` say for one. A
#: fallback never answers one (`falls_back_on`). Not ``backstop``: a session still writing
#: at the backstop is the model's own failure, like the output limit.
TIMEOUT_KINDS = frozenset({"timeout", "declared_timeout", "harness_incident", "step_timeout"})


def falls_back_on(error: BaseException) -> bool:
    """Whether ``error`` is a genuine failure of a step with a fallback (`nn.graph.Graph.
    falls_back`), which leaves it missing so its fallback answers. Never a failure of time
    or of the harness, on either runtime: any timeout (`items.error_kind` in
    `TIMEOUT_KINDS`: a model call the client ended, a declared wall, a code
    step's `snippet.WALL_S` (`SnippetTimeout`), an unstarted sandbox (`SandboxUnavailable`),
    and a spent retry carrying one of them as its cause; or a `failure_kind` timeout such
    as an op's own `subprocess.TimeoutExpired`), an op past its per-op timeout
    (`OpTimeoutError`) or whose child vanished (`OpOutcomeUnknown`), an interrupted model
    call (`items.interrupted_cause`: a wire that kept failing, a killed or cancelled
    session), a request the provider refuses for its key, permission or model name
    (``setup``: configuration, the fallback would be refused alike), nor the run's budget
    or a build-loop stop, which end the run as before. A code step past its own wall
    (`SnippetTimeout`) is its code's fault but still time, so it never falls back either.
    The output limit, a session still writing at the backstop (`model.HarnessBackstop`,
    ``backstop``) and a prompt too long are the model's or the step's own failure, so
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
    elif isinstance(error, InventedToolCalls):
        cause = INVENTED_TOOL_CALLS
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


def _session_of(error: BaseException) -> int | None:
    """The session the failed attempt's CLI ran in (``session_id``, set by
    `model.ModelClient._run_completion`), from ``error`` or what it was raised from."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        session = getattr(current, "session_id", None)
        if isinstance(session, int):
            return session
        current = current.__cause__ or current.__context__
    return None


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


def node_allowance_s(base_s: int, rule: dict[str, Any] | None, *,
                     markup_retry: bool = False) -> int:
    """The model time a node's dispatch reserves for all its attempts, before overhead:
    with ``markup_retry`` (`refuses_tool_markup`), one more session as long as its longest
    attempt, which a reply that writes out tool calls may take after any of them."""
    walls = attempt_walls(base_s, rule) if rule else [int(base_s)]
    return sum(walls) + (max(walls) if markup_retry else 0)


def _retry_limit(declared: int, envelope_left: int, budget: Any) -> int:
    """A retry session's limit: its declared wall, within what is left of the node's
    envelope (with `ENVELOPE_SLACK_S` of the dispatch overhead) and of the run's wall."""
    bounds = [declared, envelope_left + ENVELOPE_SLACK_S]
    remaining = budget.remaining_s() if budget is not None else None
    if remaining is not None:
        bounds.append(remaining)
    return int(min(bounds))


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
    elif kind == INVENTED_TOOL_CALLS:
        what = f"has no tools but wrote out tool calls after {waited_s}s"
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


#: The counts :func:`retries` derived from the journal, with the bytes they cover and
#: what identifies that journal, so a status poll reads only what was appended since.
RETRY_COUNTS = Path(".refactory/node-retries.json")


def retries(workdir: Path) -> dict[str, int]:
    """How many times each node of the run was retried, from its events.

    `console.status` asks on every poll, and a poll must not read the whole history
    (`live.snapshot`): the counts are kept in :data:`RETRY_COUNTS` with the journal bytes
    they cover, and only the journal after those bytes is read. A journal that is not the
    one they were counted from (a new invocation moved it away: another file, or another
    first record) is counted again from its start, as is a journal shorter than the
    bytes they cover, and one whose last covered record is no longer where the counts
    left it (the file cut short in place and grown again past those bytes).
    """
    from refactory_dsh import live

    root = Path(workdir).resolve()
    journal = root / live.EVENTS
    try:
        size = journal.stat().st_size
    except FileNotFoundError:
        return {}
    link = live._link(journal)
    kept = _kept_counts(root)
    trusted = (kept is not None and kept["link"] == (list(link) if link else None)
               and kept["size"] <= size
               and live._anchor(journal, kept["first_end"]) == kept["first"]
               and _digest(journal, kept["last_start"], kept["size"]) == kept["last"])
    counts: dict[str, int] = dict(kept["counts"]) if trusted and kept else {}
    covered = kept["size"] if trusted and kept else 0
    first, first_end = (kept["first"], kept["first_end"]) if trusted and kept else (None, 0)
    last, last_start = (kept["last"], kept["last_start"]) if trusted and kept else ("", 0)
    if covered == size and trusted:
        return counts
    import hashlib

    end = covered
    with journal.open("rb") as handle:
        handle.seek(covered)
        while line := handle.readline():
            if not line.endswith(b"\n"):
                break  # a write still in flight: counted once it is complete
            last_start, end = end, handle.tell()
            last = hashlib.sha256(line).hexdigest()
            if not line.strip():
                continue
            if first is None:
                first, first_end = hashlib.sha256(line).hexdigest(), end
            if RETRY_EVENT.encode() not in line:
                continue
            event = live._record(root, line)
            if event.get("kind") == RETRY_EVENT and event.get("node_id"):
                counts[str(event["node_id"])] = counts.get(str(event["node_id"]), 0) + 1
    if first is not None and end != covered:
        _keep_counts(root, {"link": list(link) if link else None, "size": end, "first": first,
                            "first_end": first_end, "last": last, "last_start": last_start,
                            "counts": counts})
    return counts


def _kept_counts(root: Path) -> dict[str, Any] | None:
    try:
        kept = json.loads((root / RETRY_COUNTS).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(kept, dict) or not isinstance(kept.get("size"), int)
            or not isinstance(kept.get("first_end"), int) or not isinstance(kept.get("first"), str)
            or not isinstance(kept.get("last_start"), int) or not isinstance(kept.get("last"), str)
            or not isinstance(kept.get("counts"), dict)
            or not all(isinstance(n, int) for n in kept["counts"].values())):
        return None
    return kept


def _digest(journal: Path, start: int, end: int) -> str | None:
    """The digest of the journal's bytes from ``start`` to ``end``: the last line the
    counts cover, read in one bounded go."""
    import hashlib

    try:
        with journal.open("rb") as handle:
            handle.seek(start)
            data = handle.read(end - start)
    except (OSError, ValueError):
        return None
    return hashlib.sha256(data).hexdigest() if len(data) == end - start else None


def _keep_counts(root: Path, kept: dict[str, Any]) -> None:
    """Publish the counts whole, through a temporary of this writer's own; a run directory
    that cannot be written keeps none, and the next poll counts again."""
    import os
    import uuid

    temp = root / f"{RETRY_COUNTS}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        temp.write_text(json.dumps(kept, sort_keys=True))
        os.replace(temp, root / RETRY_COUNTS)
    except OSError:
        pass
    finally:
        temp.unlink(missing_ok=True)


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
    workspace: Callable[..., AbstractContextManager[Any]] | None = None,
    tool_policy: str = "",
    keeps_tool_markup: bool = False,
    verdict: dict[str, Any] | None = None,
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

    A step with no tools (``tool_policy`` ``none``) whose reply writes out tool calls
    (`agent_json.invented_tool_calls`) is asked once more, told it has none
    (`NO_TOOLS_NOTE`), whatever its rule, unless it keeps such a reply
    (``keeps_tool_markup``, `refuses_tool_markup`). That session runs under the wall of the
    attempt it replaces, within what is left of the node's envelope -- which counts it
    (`node_allowance_s`) -- and of the run's wall, as a rule's retry does; with no time
    left, or a second such reply, the step fails (:class:`InventedToolCalls`). A step with
    tools is never judged so. Both runtimes come here, so both refuse such a reply alike.

    ``verdict`` (`lowering.judge_verdict`) marks a step whose answer a verdict gate reads:
    an answer that names no verdict is settled before the gate reads it
    (:func:`settle_verdict`).
    """
    from refactory_dsh import workspace as workspaces
    from refactory_dsh.agent_delivery import begin, finish
    from refactory_dsh.agent_json import invented_tool_calls

    workdir = Path(workdir)
    if rule and int(rule["times"]) <= 0:
        rule = None
    if rule and timeout_s is None and "timeout" in rule["on"]:
        raise ValueError(f"{node} declares a retry but no timeout_s for it to multiply")
    tries = 1 + (int(rule["times"]) if rule else 0)
    # A step with no tools that writes out tool calls gets one more session, told it has
    # none, whatever its rule (`refuses_tool_markup`); ``told`` once it has had it.
    markup = refuses_tool_markup(tool_policy, keeps_tool_markup)
    #: Each attempt's wall, and the time the node's envelope holds for all its sessions
    #: (`node_allowance_s`, as `workers.lower_workflow` reserves it), when it has a bound.
    walls = ((attempt_walls(int(timeout_s), rule) if rule else [int(timeout_s)])
             if timeout_s is not None else [])
    allowance = (node_allowance_s(int(timeout_s), rule, markup_retry=markup)
                 if timeout_s is not None else 0)
    #: How long each failed session actually ran: what the failure reports.
    ran: list[int] = []
    #: How long every session that delivered nothing ran: what the envelope is charged with.
    spent_s: list[int] = []
    #: Every session the node may run, for the retry events' "attempt N of M".
    most = tries + markup
    keep = bool(rule and rule.get("keep_workspace"))
    attempt_client = client
    attempt_prompt = prompt
    told = False
    index = 0
    #: Whether this attempt keeps the workspace: only a retry after a timeout, under a rule
    #: that keeps it; and the timed-out attempt's session, whose processes end first.
    keep_now = False
    stale_session: int | None = None
    while True:
        worktree = _worktree_state(workdir) if rule else None
        started = None
        invented: list[str] = []
        try:
            opened = (workspace(index > 0 or told, keep=keep_now, session=stale_session)
                      if workspace is not None else nullcontext(None))
            with opened as place:
                if keep_now and place is not None and place.synced in KEPT_SYNCS:
                    attempt_prompt = f"{attempt_prompt}\n\n{KEEP_WORKSPACE_NOTE}"
                keep_now, stale_session = False, None
                delivery = begin(workdir, node, attempt_prompt, list(writes), reads=list(reads),
                                 model_context=_context(attempt_client),
                                 **({"workspace": place.describe()} if place is not None else {}),
                                 tool_policy=tool_policy)
                started = _clock()
                output = ask(attempt_client, delivery["prompt"], place)
                invented = invented_tool_calls(output) if markup else []
                if place is not None and not invented:
                    workspaces.collect(place, workdir, writes, final_response=output)
        except Exception as error:
            kind = failure_kind(error)
            if started is not None and isinstance(error, RuntimeError):
                # How many sessions ran, for the step's no-answer record (`record_spent`).
                try:
                    error.sessions = index + 1 + told  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            if started is None or rule is None or kind not in rule["on"]:
                raise
            ran.append(int(round(_clock() - started)))
            spent_s.append(ran[-1])
            spent = ", then ".join(f"{seconds}s" for seconds in ran)
            failed = {"timeout": "timed out", "output_limit": "ran out of output"}.get(kind, "crashed")
            its = "its declared retry" if kind == "timeout" else "its retry"
            if index + 1 >= tries:
                raise RetriesExhausted(
                    f"{node} {failed} {_times(len(ran))} (after {spent}); {its} "
                    f"is spent. Last error: {error}", kind=kind, cause=_cause(error, kind),
                    sessions=len(ran) + told) from error
            limit: int | None = None
            if walls:
                # The retry's own limit, within what is left of the node's envelope (every
                # attempt's limit plus some of the dispatch overhead) and of the run's wall.
                declared = walls[index + 1]
                limit = _retry_limit(declared, allowance - sum(spent_s), budget)
                if limit < min(MIN_RETRY_WALL_S, declared):
                    raise RetriesExhausted(
                        f"{node} {failed} (after {spent}) and {its} was not "
                        f"started: {max(0, limit)}s is left of the node's envelope and the "
                        f"run's wall. Last error: {error}", kind=kind,
                        cause=_cause(error, kind), sessions=len(ran) + told) from error
            if budget is not None:
                budget.checkpoint()
            settle_for_retry(workdir, node, f"{failed} after {ran[-1]}s; retried by the "
                                            "node's retry rule", worktree=worktree, failed=failed)
            _record_retry(workdir, node, attempt=index + 2 + told, tries=most, waited_s=ran[-1],
                          wall_s=limit, error=error, kind=kind)
            attempt_client = _with_wall(client, limit) if limit is not None else client
            # A session that used its whole output allowance tends to do so again on the
            # same prompt; the retry is told, and the note is not carried further. A step
            # told it has no tools stays told.
            attempt_prompt = (f"{prompt}\n\n{OUTPUT_LIMIT_NOTE}" if _output_limit(error) else prompt) \
                + (f"\n\n{NO_TOOLS_NOTE}" if told else "")
            keep_now = keep and kind == "timeout" and workspace is not None
            stale_session = _session_of(error) if keep_now else None
            index += 1
            continue
        if invented:
            # The reply "ran" what this step cannot run, and reports what it made up: never
            # its output. Its bracket is settled as a timed-out attempt's is, so the retry
            # writes the output afresh and nothing of this reply is delivered.
            tags = ", ".join(f"<{tag}>" for tag in dict.fromkeys(invented))
            waited = int(round(_clock() - started)) if started is not None else 0
            spent_s.append(waited)
            wrote = (f"{node} has no tools, but its reply wrote out tool calls or results "
                     f"({tags})")
            if told:
                raise InventedToolCalls(f"{wrote} again after it was told so; the reply is "
                                        "refused, not taken as its output",
                                        sessions=index + 1 + told)
            limit = None
            if walls:
                # The wall of the attempt it replaces, within what is left of the node's
                # envelope and of the run's wall: the deadline a rule's retry keeps.
                declared = walls[index]
                limit = _retry_limit(declared, allowance - sum(spent_s), budget)
                if limit < min(MIN_RETRY_WALL_S, declared):
                    raise InventedToolCalls(
                        f"{wrote}, and its retry was not started: {max(0, limit)}s is left "
                        "of the node's envelope and the run's wall; the reply is refused, "
                        "not taken as its output", sessions=index + 1)
            if budget is not None:
                budget.checkpoint()
            settle_for_retry(workdir, node, f"its reply wrote out tool calls ({tags}) though "
                                            "it has no tools; retried once, told so",
                             failed="wrote out tool calls")
            _record_retry(workdir, node, attempt=index + 2, tries=most, waited_s=waited,
                          wall_s=limit, error=InventedToolCalls(f"its reply wrote {tags}"),
                          kind=INVENTED_TOOL_CALLS)
            told = True
            if limit is not None:
                attempt_client = _with_wall(client, limit)
            attempt_prompt = f"{attempt_prompt}\n\n{NO_TOOLS_NOTE}"
            continue
        receipt = finish(workdir, node, final_response=output, capture_final=capture_final)
        # A step a gate sent back after it went missing has answered now: it is no
        # longer missing (`items.missing_steps`).
        clear_spent(workdir, node)
        if verdict and writes:
            settled = settle_verdict(workdir, node, writes[0], verdict, client=client, ask=ask,
                                     budget=budget)
            if settled in ("reasked", "unreadable"):
                receipt = _resettled(workdir, receipt, writes[0], settled)
        return output, receipt


def settle_verdict(workdir: Path, node: str, name: str, verdict: dict[str, Any], *,
                   client: Any, ask: Callable[[Any, str, Any], str], budget: Any = None) -> str:
    """Settle the answer ``name`` an agent wrote for a verdict gate (``verdict``:
    `lowering.judge_verdict`) before the gate reads it; what happened, as a word.

    An answer that names a verdict (`snippet.parse_verdict`) is left as it is
    (``named``), and so is one that names two (``two``), which the gate refuses as before.
    One that names none (a judge that wrote "Let me inspect ..." or "I found ..." and
    never put the word first or alone last) is asked again once, in a fresh call with no
    tools and its own answer in the prompt (`snippet.reask_prompt`), for the word alone:
    the word it answers is written first in the file, with a line that says it was asked
    (`snippet.REASKED_NOTE`), above the answer, which stays the reason (``reasked``).
    When that names none either (or the call fails), the gate's declared way on is taken
    when it has one -- its ``proceed``, as when its loop cannot settle -- and the note says
    so (`snippet.UNREADABLE_NOTE`, ``unreadable``); else the file is left as it is and the
    gate fails as it did (``failed``). Never silent: each case but ``named`` is recorded as
    a `VERDICT_REASK_EVENT` event, and the re-ask as one of the item's model calls."""
    from refactory_dsh import snippet
    from refactory_dsh.agent_delivery import _path
    from refactory_dsh.live import emit

    target = _path(workdir, name)
    try:
        text = target.read_text()
    except (OSError, UnicodeDecodeError):
        return "named"  # nothing to read here; the gate says what is wrong with it
    labels = list(verdict.get("labels") or ())
    named = snippet.names_verdict(text, snippet.answers({"labels": labels}))
    if named is None:
        emit(workdir, VERDICT_REASK_EVENT, node_id=node, state="two", gate=verdict.get("gate"),
             summary=f"{node}'s answer names two verdicts; gate {verdict.get('gate')!r} "
                     "refuses it")
        return "two"
    if named or not labels:
        return "named"
    prompt = snippet.reask_prompt(text, labels)
    reply, why = "", ""
    # The re-ask runs in the run directory, outside any sealed workspace, so it must run
    # without tools; a client that cannot promise that is not asked at all.
    quiet = (client.with_tool_policy("none")
             if callable(getattr(client, "with_tool_policy", None)) else None)
    try:
        if quiet is None:
            raise RuntimeError("its model client cannot run a call without tools")
        if budget is not None:
            budget.checkpoint()
        reply = str(ask(quiet, prompt, None) or "")
    except Exception as error:  # noqa: BLE001 - a failed re-ask is an unreadable verdict, said
        from refactory_dsh.budgets import BudgetExhausted

        if isinstance(error, BudgetExhausted):
            raise
        why = f"the call failed: {type(error).__name__}: {str(error)[:200]}"
    if quiet is not None:
        _record_reask(workdir, node, prompt, reply)
    label = None
    if not why:
        try:
            parsed = snippet.parse_verdict(reply, labels)
        except snippet.SnippetError:
            parsed = None
        label = parsed[0] if parsed is not None else None
        if label is None:
            why = f"it answered {' '.join(reply.split())[:120]!r}"
    gate = str(verdict.get("gate") or "")
    if label is not None:
        note = snippet.REASKED_NOTE.format(node=node, label=label)
        outcome = "reasked"
    elif verdict.get("unreadable"):
        label = str(verdict["unreadable"])
        note = snippet.UNREADABLE_NOTE.format(node=node, why=why, gate=gate, label=label)
        outcome = "unreadable"
    else:
        emit(workdir, VERDICT_REASK_EVENT, node_id=node, state="failed", gate=gate,
             reply=reply[:500], summary=f"{node}'s answer named no verdict, also when asked "
                                        f"again ({why}); gate {gate!r} has no way on to take")
        return "failed"
    target.write_text(f"{label}\n{note}\n\n{text}")
    emit(workdir, VERDICT_REASK_EVENT, node_id=node, state=outcome, gate=gate, verdict=label,
         reply=reply[:500], summary=note.strip("[]").removeprefix("refactory: "))
    return outcome


def _resettled(workdir: Path, receipt: dict[str, Any], name: str, settled: str
               ) -> dict[str, Any]:
    """The delivery receipt with the settled answer's fingerprint and how it was settled
    (``verdict_settled``), written back, so a reader that checks a delivery is current
    (`agent_delivery.receipts_in`) sees the file as the step's delivery."""
    from refactory_dsh.agent_delivery import ROOT, _fingerprint

    receipt = {**receipt, "artifacts": {**(receipt.get("artifacts") or {}),
                                        name: _fingerprint(workdir, name)},
               "verdict_settled": settled}
    path = Path(workdir) / ROOT / str(receipt.get("id") or "") / "receipt.json"
    if receipt.get("id") and path.is_file():
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _record_reask(workdir: Path, node: str, prompt: str, reply: str) -> None:
    """The re-ask as one of the item's model calls, on the spine's per-item sidecar
    (`items.record_exchange`, read only on the spine; the walker's ``ask`` keeps its
    exchanges itself), inside a data pass only."""
    from refactory_dsh.nn.executor import CURRENT_ITEM

    if (Path(workdir) / CURRENT_ITEM).exists():
        from refactory_dsh.items import record_exchange

        record_exchange(workdir, node, prompt, reply)
