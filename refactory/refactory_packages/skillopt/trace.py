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
        output: str,
        *,
        call_key: tuple[str, int, int, int] | list | None = None,
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
