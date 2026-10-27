"""Declared recovery for agent nodes: a timed-out session gets one fresh, longer one.

The rule lives in the graph (`Node.retry`, canonicalized by
`nn.graph.retry_declaration`), never in a CEO's judgment, so an automatic headless run
recovers the same way a supervised one does. Both runtimes run it through
:func:`deliver`: the walker from `Executor._visit_agent`, the spine from the worker every
dispatched agent node runs (`worker_node.execute`). The spine reserves the node's whole
envelope for it when it lowers the graph (`workers.lower_workflow`, with
:func:`node_allowance_s`), so the retry runs inside the one command the spine started.

A retry needs the node's own `timeout_s`, which each retry's limit multiplies.

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


class RetriesExhausted(RuntimeError):
    """A node's session timed out, and its declared retries are spent or refused."""


def failure_kind(error: BaseException) -> str | None:
    """The `nn.graph.RETRY_ON` failure ``error`` is, or None when it is none of them.

    A timeout is the model client's deadline (`model.CompletionTimeout`), or a timeout
    another client reports the same way. Cancellation, budget exhaustion, a refusal and a
    bad answer are not timeouts, and are never retried.
    """
    from refactory_dsh.model import CompletionTimeout

    if isinstance(error, (CompletionTimeout, subprocess.TimeoutExpired)):
        return "timeout"
    if isinstance(error, RuntimeError) and str(error).startswith("model completion timed out"):
        return "timeout"
    return None


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
                     worktree: dict[str, Any] | None = None) -> None:
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
            raise RetriesExhausted(f"{node} timed out and was not retried: the run's worktree "
                                   f"changed while it ran ({'; '.join(changed)})")
    agent_delivery.invalidate(Path(workdir).resolve(), [node], reason)


def _record_retry(workdir: Path, node: str, *, attempt: int, tries: int, waited_s: int,
                  wall_s: int, error: BaseException) -> None:
    from refactory_dsh.live import emit

    limit = f"{wall_s}s"
    emit(workdir, RETRY_EVENT, node_id=node, state="retrying", reason="timeout",
         attempt=attempt, attempts=tries, after_s=waited_s, wall_s=wall_s,
         error=str(error)[:500],
         summary=f"{node} timed out after {waited_s}s; retrying it with a fresh session "
                 f"and {limit} (attempt {attempt} of {tries})")


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
) -> tuple[str, dict[str, Any]]:
    """Run one agent node's model work through the delivery bracket, honouring ``rule``.

    ``workspace(retrying)`` prepares and locks a workspace afresh for each attempt; without
    it the node works in the run directory. ``ask(client, prompt, place)`` makes one
    completion, where ``place`` is that workspace or None. Collection happens before
    the delivery bracket closes. ``timeout_s`` is the node's own bound,
    which a rule multiplies. Returns the answer and the delivery receipt. With no rule,
    or a failure the rule does not answer, the first failure is raised unchanged. When
    the last allowed attempt times out too, :class:`RetriesExhausted` says how long each
    session ran.
    """
    from refactory_dsh import workspace as workspaces
    from refactory_dsh.agent_delivery import begin, finish

    workdir = Path(workdir)
    if rule and timeout_s is None:
        raise ValueError(f"{node} declares a retry but no timeout_s for it to multiply")
    tries = 1 + (int(rule["times"]) if rule else 0)
    walls = attempt_walls(int(timeout_s), rule) if rule else []
    #: How long each timed-out session actually ran: what the failure reports, and what
    #: the node's envelope is charged with.
    ran: list[int] = []
    attempt_client = client
    for index in range(tries):
        worktree = _worktree_state(workdir) if rule else None
        started = None
        try:
            with workspace(index > 0) if workspace is not None else nullcontext(None) as place:
                delivery = begin(workdir, node, prompt, list(writes), reads=list(reads),
                                 model_context=_context(attempt_client),
                                 **({"workspace": place.describe()} if place is not None else {}))
                started = time.monotonic()
                output = ask(attempt_client, delivery["prompt"], place)
                if place is not None:
                    workspaces.collect(place, workdir, writes, final_response=output)
        except Exception as error:
            if started is None or rule is None or failure_kind(error) not in rule["on"]:
                raise
            ran.append(int(round(time.monotonic() - started)))
            spent = ", then ".join(f"{seconds}s" for seconds in ran)
            if index + 1 >= tries:
                raise RetriesExhausted(
                    f"{node} timed out {_times(len(ran))} (after {spent}); its declared "
                    f"retry is spent. Last error: {error}") from error
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
                    f"{node} timed out (after {spent}) and its declared retry was not "
                    f"started: {max(0, limit)}s is left of the node's envelope and the "
                    f"run's wall. Last error: {error}") from error
            if budget is not None:
                budget.checkpoint()
            settle_for_retry(workdir, node, f"timed out after {ran[-1]}s; retried by the "
                                            "node's declared rule", worktree=worktree)
            _record_retry(workdir, node, attempt=index + 2, tries=tries, waited_s=ran[-1],
                          wall_s=limit, error=error)
            attempt_client = _with_wall(client, limit)
            continue
        receipt = finish(workdir, node, final_response=output, capture_final=capture_final)
        return output, receipt
    raise AssertionError("unreachable: every attempt returns or raises")  # pragma: no cover
