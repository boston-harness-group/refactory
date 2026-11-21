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

**A train item is scored on its first attempt** (:data:`SCORING`). Feedback that says an
attempt is wrong (a value that is a mapping whose ``passed`` is ``False``) is counted in
the pass's :data:`TOLD_WRONG` record, and an item told wrong scores 0 however its later
turns end (`items.make_item_record`): the turns the answer's feedback buys show in its
record and reach the optimizer's analysts, but never lift the score that drives the
search. Without this a gold-feedback train evaluation scores close to 100% for any
skill, and the search has nothing to learn from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: The call a train-feedback channel node names.
CALL = "refactory_dsh.feedback:train_feedback"

#: How a gold-feedback train evaluation scores an item, in every seed cache's score
#: context (`seed_cache.score_context`), so a seed scored another way is never reused.
SCORING = "first-attempt"

#: The pass's count of feedback that said its attempt was wrong. Written only by
#: `train_feedback`, cleared when the item's pass starts (`items.prepare_pass_dir`), read
#: when its record is made (`told_wrong`).
TOLD_WRONG = ".refactory/gold_feedback.json"

#: How an item told it was wrong begins its record's ``fail_reason``.
FIRST_ATTEMPT_WRONG = "first attempt wrong"


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


def says_wrong(value: Any) -> bool:
    """Whether a Task's feedback value says the attempt is wrong (`Task.train_feedback`)."""
    return isinstance(value, dict) and value.get("passed") is False


def told_wrong(pass_dir: Path) -> int:
    """How many times the pass at ``pass_dir`` was told its attempt was wrong: 0 with no
    record. A record that is there but cannot be read counts as told wrong once (fail
    closed): a damaged count never gives an item back its grade after the feedback."""
    path = Path(pass_dir) / TOLD_WRONG
    if not path.exists():
        return 0
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return 1
    count = record.get("told_wrong") if isinstance(record, dict) else None
    return count if isinstance(count, int) and count > 0 else 1


def clear_told_wrong(pass_dir: Path) -> None:
    """Forget an earlier attempt's count: the item's pass starts again."""
    (Path(pass_dir) / TOLD_WRONG).unlink(missing_ok=True)


def _count_told_wrong(pass_dir: Path) -> None:
    path = Path(pass_dir) / TOLD_WRONG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"told_wrong": told_wrong(pass_dir) + 1}) + "\n")


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
    if says_wrong(value):
        _count_told_wrong(workdir)
    return {facts[0]: value}
