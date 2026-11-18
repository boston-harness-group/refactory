"""Train-only gold feedback: the one way grader-derived data may reach a pass.

Some benchmarks' reference loops tell the solver, during a training rollout, how its
attempt compares with the answer -- SkillOpt's SpreadsheetBench rollouts list the cells
a training attempt got wrong and give it another turn. Rule V17 forbids exactly that
anywhere else: nothing outside the grader side may consume grader-derived data. This
module is the declared exception, and the only one:

- a Task says what an attempt is told (``Task.train_feedback``), from the whole item;
- a mode puts one op node calling :data:`CALL` in its data pass, declares it protected
  (`Mode.protected_nodes`), gives it no reads or writes and exactly one fact, and lets
  the pass's later nodes read that fact;
- the op asks the Task only when the run directory's ``subset.json`` names the train
  part and turns gold feedback on (`task.gold_feedback_on`), which only the outer loop
  writes (`task.write_subset`) and which it refuses for val and test. Otherwise its fact
  is ``None``.

V17 lets that fact through (`nn.validate.feedback_channel_issues` says what a channel
must look like); it cannot know which part a run will evaluate, and does not need to:
off the train part the fact holds nothing. The op runs the same on both runtimes
(the spine's op process and the walker), since both mark each pass with its data
node's Task (`items.PASS_MARKER`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: The call a train-feedback channel node names.
CALL = "refactory_dsh.feedback:train_feedback"


class FeedbackError(RuntimeError):
    """A channel node is not inside a data pass, or its item cannot be found."""


def _marker(workdir: Path) -> dict[str, Any]:
    from refactory_dsh.items import PASS_MARKER

    path = Path(workdir) / PASS_MARKER
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FeedbackError(f"a train-feedback node runs inside a data pass, and {workdir} "
                            f"is not one ({PASS_MARKER}: {exc})") from exc
    if not isinstance(record, dict):
        raise FeedbackError(f"{path} does not describe a pass")
    return record


def train_feedback(ctx: Any) -> dict[str, Any]:
    """The channel op: its one declared fact is the Task's ``train_feedback`` for this
    pass's item on a gold-feedback train evaluation, else ``None``."""
    from refactory_dsh.task import (
        gold_feedback_on, load_task, part_seed_kwargs, subset_part_seed,
    )

    facts = tuple(getattr(ctx, "_declared_facts", ()) or ())
    if len(facts) != 1:
        raise FeedbackError(f"a train-feedback node declares exactly one fact; "
                            f"{getattr(ctx, 'node_id', '?')!r} declares {list(facts)}")
    workdir = Path(ctx.workdir)
    marker = _marker(workdir)
    if not gold_feedback_on(workdir):
        return {facts[0]: None}
    task_ref = str(marker.get("task") or "")
    if not task_ref:
        raise FeedbackError(f"the data pass at {workdir} names no Task")
    task = load_task(task_ref)
    part, seed = subset_part_seed(workdir)
    item_id = str(marker.get("item") or "")
    found = [instance for instance in
             task.instances(**part_seed_kwargs(task.instances, part=part, seed=seed))
             if instance.id == item_id]
    if len(found) != 1:
        raise FeedbackError(f"Task {task_ref!r} has {len(found)} items with id {item_id!r}")
    value = task.train_feedback(found[0], workdir,
                                **part_seed_kwargs(task.train_feedback, part=part, seed=seed))
    return {facts[0]: value}
