"""The op and Task the GEPA strategy tests run (``tests/test_gepa.py``).

The planted task: odd items are category A, even ones category B. An A item passes
when the ``alpha_rule`` knob (or, in the one-knob mode, ``instruction``) says ALPHA
and does not say BETA; a B item when ``beta_rule`` (or ``instruction``) says BETA
and not ALPHA. A failed item's details say what it needed, the feedback GEPA reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from refactory_dsh.task import Task, TaskInstance, VerifyResult

TRAIN_IDS = [f"t{i}" for i in range(1, 9)]
VAL_IDS = [f"v{i}" for i in range(1, 7)]
KNOBS = ("instruction", "alpha_rule", "beta_rule")


def write_knobs(ctx: Any) -> dict[str, Any]:
    """Leave the knobs' values in the item's workspace for ``verify``."""
    values = {name: ctx.knob(name, None) for name in KNOBS}
    ctx.write_json("knobs.json", values)
    return {"knobs_seen": True}


def category(item_id: str) -> str:
    return "A" if int(item_id[1:]) % 2 else "B"


def judge(item_id: str, knobs: dict[str, Any]) -> tuple[bool, str]:
    cat = category(item_id)
    text = knobs.get("instruction")
    if text is None:
        text = knobs.get("alpha_rule" if cat == "A" else "beta_rule") or ""
    want, avoid = ("ALPHA", "BETA") if cat == "A" else ("BETA", "ALPHA")
    if want in text and avoid not in text:
        return True, ""
    return False, f"a category {cat} item: the instruction must say {want} and never {avoid}"


class PlantedTask(Task):
    name = "gepa_planted_task"

    def instances(self, **kw: Any) -> list[TaskInstance]:
        return [TaskInstance(id=i, metadata={}) for i in TRAIN_IDS + VAL_IDS]

    def setup(self, instance: TaskInstance, workspace: Any, **kw: Any) -> None:
        pass

    def verify(self, instance: TaskInstance, workspace: Any, **kw: Any) -> VerifyResult:
        path = Path(workspace) / "knobs.json"
        knobs = json.loads(path.read_text()) if path.exists() else {}
        passed, why = judge(instance.id, knobs)
        return VerifyResult(passed=passed, score=1.0 if passed else 0.0, details=why)
