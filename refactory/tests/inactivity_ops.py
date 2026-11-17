"""A two-item task for `test_inactivity_only`, each item graded from its ``answer.txt``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from refactory_dsh.task import Task, TaskInstance, VerifyResult


class TwoItemTask(Task):
    """Items ``0`` (the session under test) and ``1`` (a quick one, so a pass whose item 0
    fails is not a run whose every item failed), each passed when its ``answer.txt`` says
    ``right``."""

    name = "two_item_task"

    def instances(self, **kw: Any) -> list[TaskInstance]:
        return [TaskInstance(id="0", metadata={}), TaskInstance(id="1", metadata={})]

    def setup(self, instance: TaskInstance, workspace: Any, **kw: Any) -> None:
        pass

    def verify(self, instance: TaskInstance, workspace: Any, **kw: Any) -> VerifyResult:
        answer = Path(workspace) / "answer.txt"
        if not answer.exists():
            return VerifyResult(passed=False, score=0.0, details="no answer")
        right = answer.read_text().strip() == "right"
        return VerifyResult(passed=right, score=1.0 if right else 0.0)
