"""Append-only JSONL trace recorder for SkillOpt.

Each line is a JSON object with ``type`` (``model_call`` or ``decision``),
``timestamp``, ``epoch``, ``step_type``, and an optional ``call_key`` (the
stable 4-tuple ``[step_type, epoch, group_index, call_order]`` that names the
call site). A ``model_call`` made by :func:`ask` also has ``attempt`` (from 1):
an optimizer call is retried as SkillOpt retries it, one line per attempt; an
attempt that waited out a network outage also has ``network_waits`` (one entry per
outage: ``started_at``, ``ended_at``, ``waited_s``, ``error``, ``outcome``).

The call key is never sent to the model client: an optimizer call uses the
client protocol the walker uses, ``complete(prompt, *, system=...)``, so any
client works (refactory's live ``DshModelClient`` takes no other keyword;
#493 G30). While a call is in flight its key is readable through
:func:`current_call_key`, for clients that observe calls (the F20 replay picks
an optimizer call's attempts by it; test doubles key canned replies by it).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

#: The key of the optimizer call :func:`ask` has in flight, ``None`` outside one.
_CALL_KEY: ContextVar[tuple[str, int, int, int] | None] = ContextVar(
    "skillopt_call_key", default=None,
)


def current_call_key() -> tuple[str, int, int, int] | None:
    """The call key of the optimizer call in flight (``None`` outside :func:`ask`).

    Read-only context for clients that observe calls; it is not part of the
    client protocol, and a solver call (the walker's) never sees one.
    """
    return _CALL_KEY.get()


@contextmanager
def optimizer_call(call_key: tuple[str, int, int, int] | list) -> Iterator[None]:
    """Mark ``call_key`` as the optimizer call in flight (:func:`current_call_key`)."""
    token = _CALL_KEY.set(tuple(call_key))
    try:
        yield
    finally:
        _CALL_KEY.reset(token)


class TraceRecorder:
    """Append-only JSONL trace file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._epoch = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        #: Optimizer calls made through :func:`ask`, and those that failed.
        self.optimizer_calls = 0
        self.optimizer_failures: list[str] = []
        #: Called with each call's error (``None`` on success), when set.
        self.listener: Any = None

    def count_call(self, error: str | None) -> None:
        self.optimizer_calls += 1
        if error is not None:
            self.optimizer_failures.append(error)
        if self.listener is not None:
            self.listener(error)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def record_model_call(
        self,
        step_type: str,
        system: str,
        user: str,
        output: str | None,
        *,
        call_key: tuple[str, int, int, int] | list | None = None,
        error: str | None = None,
        attempt: int | None = None,
        network_waits: list[dict[str, Any]] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "type": "model_call",
            "timestamp": time.time(),
            "epoch": self._epoch,
            "step_type": step_type,
            "system_prompt": system,
            "user_prompt": user,
            "raw_output": output,
        }
        if call_key is not None:
            entry["call_key"] = list(call_key)
        if attempt is not None:
            entry["attempt"] = attempt
        if error is not None:
            entry["error"] = error
        if network_waits:
            entry["network_waits"] = network_waits
        self._write(entry)

    def record_decision(self, step_type: str, **fields: Any) -> None:
        entry: dict[str, Any] = {
            "type": "decision",
            "timestamp": time.time(),
            "epoch": self._epoch,
            "step_type": step_type,
        }
        entry.update(fields)
        self._write(entry)

    def _write(self, entry: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            f.flush()


#: Failures that are not a bad reply but the run itself ending or misconfigured:
#: refactory's own rule re-raises them (``Loop.propose`` re-raises
#: ``BudgetExhausted``; the model client raises ``CancelledError`` and
#: ``ModelChoiceError`` to stop, not to be retried around).
def _fatal() -> tuple[type[BaseException], ...]:
    from concurrent.futures import CancelledError

    from refactory_dsh.budgets import BudgetExhausted
    from refactory_dsh.model import ModelChoiceError

    return (BudgetExhausted, CancelledError, ModelChoiceError)


#: SkillOpt's attempts per optimizer call: every analyst, merge, rank, slow-update
#: and meta-skill call asks with ``retries=3`` (reflect.py:359, :456;
#: aggregate.py:50, :248; clip.py:77; slow_update.py:431; meta_skill.py:67), and
#: its Claude backend makes that many attempts (claude_backend.py:314-324).
OPTIMIZER_ATTEMPTS = 3


def retry_delay(attempt: int) -> float:
    """Seconds SkillOpt's Claude backend waits after failed attempt ``attempt``
    (0-based): ``min(2 ** attempt, 15)`` (claude_backend.py:323), so 1, 2, 4."""
    return min(2 ** attempt, 15)


#: How :func:`ask` waits between attempts (tests replace it).
_sleep = time.sleep


@contextmanager
def no_retry_waits() -> Iterator[None]:
    """Skip :func:`ask`'s waits between attempts while inside: a replay serves
    recorded attempts, so waiting for a provider to recover buys nothing."""
    global _sleep
    saved, _sleep = _sleep, (lambda seconds: None)
    try:
        yield
    finally:
        _sleep = saved


def ask(
    client: Any,
    trace: TraceRecorder,
    step_type: str,
    user: str,
    *,
    system: str,
    call_key: tuple[str, int, int, int],
) -> str | None:
    """One optimizer call, retried as SkillOpt retries it, every attempt recorded
    in the trace; ``None`` when every attempt fails.

    SkillOpt's Claude backend makes up to ``retries=3`` attempts
    (:data:`OPTIMIZER_ATTEMPTS`): any exception from an attempt counts as a
    failed attempt, it waits ``min(2 ** attempt, 15)`` seconds after each one --
    the last included -- and then raises (claude_backend.py:314-324). A reply
    is never retried, however unusable: parsing it is the caller's
    (reflect.py:362, aggregate.py:53).

    SkillOpt catches that final failure and carries on as if the response were
    unusable: the analysts drop the group's patch (reflect.py:383-384,
    :467-468), the merges fall back (aggregate.py:60-64, :261-265), the ranking
    truncates (clip.py:101-102), and the slow update and meta skill produce
    nothing (slow_update.py:440-441, meta_skill.py:76-77). Every caller here
    parses ``None`` the same way it parses an unusable reply.

    Budget exhaustion, cancellation and a missing model choice are re-raised at
    once, never retried (:func:`_fatal`): they end the run rather than read as
    a bad reply. Every attempt is its own ``client.complete`` (so its own paid
    worker), and every attempt, and every failure, goes to the trace (with its
    ``attempt`` number, from 1) and to ``trace.listener`` (the loop counts them
    into the run report's proposer health).

    The client is called as the walker calls it, ``complete(user, system=...)``;
    ``call_key`` goes to the trace and to :func:`current_call_key` only.

    A network, DNS or credential-refresh outage is not one of these attempts: the
    model client waits it out inside one ``complete`` and asks again
    (``DshModelClient._wait_for_network``), so an attempt here fails only for a reason
    the model or the provider gave, or after the client's wait bound. The outages an
    attempt waited out go on its trace line as ``network_waits``.
    """
    from refactory_dsh.model import observe_network_waits

    for attempt in range(1, OPTIMIZER_ATTEMPTS + 1):
        with observe_network_waits() as waits:
            try:
                with optimizer_call(call_key):
                    output = client.complete(user, system=system)
            except _fatal() as exc:
                error = f"{type(exc).__name__}: {exc}"
                trace.record_model_call(step_type, system, user, None, call_key=call_key,
                                        error=error, attempt=attempt, network_waits=waits)
                trace.count_call(error)
                raise
            except Exception as exc:  # noqa: BLE001 - SkillOpt's own handling, cited above
                error = f"{type(exc).__name__}: {exc}"
                trace.record_model_call(step_type, system, user, None, call_key=call_key,
                                        error=error, attempt=attempt, network_waits=waits)
                trace.count_call(error)
                _sleep(retry_delay(attempt - 1))
                continue
        trace.record_model_call(step_type, system, user, output, call_key=call_key,
                                attempt=attempt, network_waits=waits)
        trace.count_call(None)
        return output
    return None
