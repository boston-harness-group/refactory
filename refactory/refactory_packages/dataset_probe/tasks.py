from __future__ import annotations

from pathlib import Path

from refactory_dsh.task import Task, TaskInstance, VerifyResult


class CountingTask(Task):
    """Two items, each judged by whether the pass wrote its own record."""

    name = "counting"
    #: ``verify`` reads the pass's seen.json.
    graded_files = ("seen.json",)

    def instances(self):
        return [TaskInstance(id="alpha"), TaskInstance(id="beta")]

    def verify(self, instance, workspace: Path) -> VerifyResult:
        """Whether this pass did its share of the work the run asked for.

        A pass counts only if the item was processed *and* the whole run has processed
        at least `min_items`: the run's own knob is part of the judgement, so the knob
        is a real search axis rather than a declared number nothing reads.
        """
        import json

        seen = workspace / "seen.json"
        if not seen.exists():
            return VerifyResult(passed=False, score=0.0)
        try:
            recorded = json.loads(seen.read_text())
        except json.JSONDecodeError:
            return VerifyResult(passed=False, score=0.0)
        ids = [str(entry) for entry in recorded.get("ids", [])]
        enough = len(ids) >= int(recorded.get("min_items", 1))
        passed = enough and instance.id in ids
        score = 1.0 if instance.id == "alpha" and passed else 0.5 if passed else 0.0
        return VerifyResult(passed=passed, score=score)
