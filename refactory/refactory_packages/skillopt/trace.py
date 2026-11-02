"""Append-only JSONL trace recorder for SkillOpt.

Each line is a JSON object with ``type`` (``model_call`` or ``decision``),
``timestamp``, ``epoch``, ``step_type``, and an optional ``call_key`` (the
stable 4-tuple ``[step_type, epoch, group_index, call_order]`` that the F20
replay uses to match recorded outputs to call sites).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TraceRecorder:
    """Append-only JSONL trace file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._epoch = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

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
    """
    try:
        output = client.complete(user, system=system, call_key=call_key)
    except Exception as exc:  # noqa: BLE001 - SkillOpt's own handling, cited above
        trace.record_model_call(
            step_type, system, user, None, call_key=call_key,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None
    trace.record_model_call(step_type, system, user, output, call_key=call_key)
    return output
