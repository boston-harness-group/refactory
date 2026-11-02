"""Append-only JSONL trace recorder for SkillOpt.

Each line is a JSON object with ``type`` (``model_call`` or ``decision``),
``timestamp``, ``epoch``, ``step_type``, and an optional ``call_key`` (the
stable 4-tuple ``[step_type, epoch, group_index, call_order]`` that names the
call site).

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
        if error is not None:
            entry["error"] = error
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


def ask(
    client: Any,
    trace: TraceRecorder,
    step_type: str,
    user: str,
    *,
    system: str,
    call_key: tuple[str, int, int, int],
) -> str | None:
    """One optimizer call, recorded in the trace; ``None`` when the call fails.

    SkillOpt catches every optimizer-call failure and carries on as if the
    response were unusable: the analysts drop the group's patch
    (reflect.py:383-384, :467-468), the merges fall back (aggregate.py:60-64,
    :261-265), the ranking truncates (clip.py:101-102), and the slow update and
    meta skill produce nothing (slow_update.py:440-441, meta_skill.py:76-77).
    Every caller here parses ``None`` the same way it parses an unusable reply.

    Budget exhaustion, cancellation and a missing model choice are re-raised
    (:func:`_fatal`): they end the run rather than read as a bad reply. Every
    call, and every failure, goes to the trace and to ``trace.listener`` (the
    loop counts them into the run report's proposer health).

    The client is called as the walker calls it, ``complete(user, system=...)``;
    ``call_key`` goes to the trace and to :func:`current_call_key` only.
    """
    try:
        with optimizer_call(call_key):
            output = client.complete(user, system=system)
    except _fatal() as exc:
        error = f"{type(exc).__name__}: {exc}"
        trace.record_model_call(step_type, system, user, None, call_key=call_key, error=error)
        trace.count_call(error)
        raise
    except Exception as exc:  # noqa: BLE001 - SkillOpt's own handling, cited above
        error = f"{type(exc).__name__}: {exc}"
        trace.record_model_call(step_type, system, user, None, call_key=call_key, error=error)
        trace.count_call(error)
        return None
    trace.record_model_call(step_type, system, user, output, call_key=call_key)
    trace.count_call(None)
    return output
