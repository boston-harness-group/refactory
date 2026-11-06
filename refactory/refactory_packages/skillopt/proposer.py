"""SkillOpt's proposer as a reusable part: typed edits to one named text knob.

Everything SkillOpt does to turn rollouts into a change of the text it optimizes,
taken out of the strategy so another strategy (a population search, a graph with
several text knobs) can call it:

- the train minibatch seed (``batch_seed``) and the slow update's sample;
- analysts in groups of ``group_size``, failures and successes apart
  (``analysts.reflect``);
- the failure-first hierarchical merge and the ranked edit budget
  (``merge.hierarchical_merge``, ``merge.rank``, ``schedule.edit_budget``);
- the patch step, the slow update and the meta skill.

Its output is typed edits: a ``set_knob`` with a ``patch`` or a ``value``, addressed
to ``knob`` (``"skill"`` by default), handed to ``Run.attempt`` like any proposal.
It never runs or keeps anything; the strategy decides what is rolled out and kept.
The prompts and decisions are SkillOpt's, unchanged.

The proposer's own state -- the meta skill, the edit-budget step and the last slow
update's comparison pairs -- is ``state_dict``/``load_state_dict``, journaled with
the strategy's.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

from refactory_dsh.nn.edits import SKILLOPT_PATCH_CONFIG
from refactory_dsh.optim import Candidate, Proposal

from refactory_packages.skillopt import analysts, merge, prompts, schedule
from refactory_packages.skillopt.env import SkillOptEnv
from refactory_packages.skillopt.json_utils import extract_json
from refactory_packages.skillopt.trace import TraceRecorder, ask


class SkillOptProposer:
    """SkillOpt's optimizer steps over one text knob of a candidate.

    ``client`` returns the model client each call goes through (called per step,
    so a rebound model is used). ``trace`` and ``env`` are set by the strategy
    once its run starts. ``rollout_order`` and ``patch_order`` are the offline
    replay's hooks (see ``SkillOptLoop.configure``).
    """

    def __init__(
        self,
        *,
        knob: str = "skill",
        client: Callable[[], Any] = lambda: None,
        seed: int = 42,
        max_epochs: int = 4,
        group_size: int = 8,
        merge_batch_size: int = 8,
        analyst_edit_budget: int = 4,
        edit_budget_max: int = 4,
        edit_budget_min: int = 2,
        slow_update_samples: int = 20,
        rollout_order: Callable[[int, list[dict]], Any] | None = None,
        patch_order: Callable[[int, str, list[dict]], Any] | None = None,
    ) -> None:
        self.knob = knob
        self.client = client
        self.seed = seed
        self.max_epochs = max_epochs
        self.group_size = group_size
        self.merge_batch_size = merge_batch_size
        self.analyst_edit_budget = analyst_edit_budget
        self.edit_budget_max = edit_budget_max
        self.edit_budget_min = edit_budget_min
        self.slow_update_samples = slow_update_samples
        self.rollout_order = rollout_order
        self.patch_order = patch_order
        self.trace: TraceRecorder | None = None
        self.env: SkillOptEnv = SkillOptEnv()
        # State carried across epochs.
        self.meta_skill: str = ""
        self.budget_step: int = 0
        self.comparison_pairs: list[dict] = []

    # ── reading and writing the knob ───────────────────────────────────────

    def text(self, candidate: Candidate) -> str:
        """The text the proposer optimizes: ``candidate``'s ``knob`` value."""
        entry = (candidate.state.get("knobs") or {}).get(self.knob) or {}
        return str(entry.get("value", "")) if isinstance(entry, dict) else str(entry)

    def set_text(self, text: str, parent: Candidate, rationale: str) -> Proposal:
        """A ``set_knob`` replacing the knob's whole value."""
        return Proposal(
            edit={"op": "set_knob", "name": self.knob, "value": text},
            rationale=rationale, parent=parent.id,
        )

    def patch(self, ranked: list[dict[str, Any]], parent: Candidate, rationale: str) -> Proposal:
        """The patch step: the ranked edits as a ``set_knob`` patch (trainer.py:1507)."""
        return Proposal(
            edit={
                "op": "set_knob",
                "name": self.knob,
                "patch": [_op_to_kind(edit) for edit in ranked],
                "config": dict(SKILLOPT_PATCH_CONFIG),
            },
            rationale=rationale, parent=parent.id,
        )

    def placeholder(self, parent: Candidate) -> Proposal:
        """Epoch 1's slow update: empty SLOW_UPDATE markers (trainer.py:1809-1824)."""
        return self.set_text(
            prompts.inject_empty_slow_update_field(self.text(parent)), parent,
            "slow update placeholder",
        )

    # ── minibatches ────────────────────────────────────────────────────────

    def batch_seed(self, epoch: int, batch_idx: int = 0) -> int:
        """The epoch's train-batch seed; see :func:`batch_seed`."""
        return batch_seed(self.seed, epoch, batch_idx)

    def slow_update_sample(self, train_ids: list[str], epoch: int) -> tuple[int, list[str]]:
        """The slow update's seed and sample: the first ``slow_update_samples`` train
        ids shuffled by ``seed + epoch*2000`` (trainer.py:1843-1844;
        dataloader.py:278-283 rolls them out with the same seed)."""
        slow_seed = self.seed + epoch * 2000
        n_samples = min(self.slow_update_samples, len(train_ids))
        shuffled = list(train_ids)
        random.Random(slow_seed).shuffle(shuffled)
        return slow_seed, shuffled[:max(n_samples, 0)]

    # ── the step: analysts, merge, rank ────────────────────────────────────

    def reflect(
        self, epoch: int, records: list[dict], candidate: Candidate,
    ) -> tuple[list[dict], list[dict]]:
        """The analysts over ``candidate``'s train records, in the merge's order."""
        failure, success = analysts.reflect(
            client=self.client(),
            records=self._analyst_records(epoch, records),
            skill_text=self.text(candidate),
            edit_budget=self.analyst_edit_budget,
            meta_skill=self.meta_skill,
            # The analysts shuffle with the batch's own seed
            # (trainer.py:1242, :1283 pass batch.seed as random_seed).
            seed=self.batch_seed(epoch),
            group_size=self.group_size,
            epoch=epoch,
            trace=self.trace,
            env=self.env,
            knobs=knob_values(candidate),
        )
        return self._merge_order(epoch, "failure", failure), self._merge_order(epoch, "success", success)

    def merge(self, epoch: int, text: str, failure: list[dict], success: list[dict]) -> dict:
        """The failure-first hierarchical merge of the analysts' patches."""
        merged = merge.hierarchical_merge(
            client=self.client(),
            skill_text=text,
            failure_patches=failure,
            success_patches=success,
            meta_skill=self.meta_skill,
            epoch=epoch,
            trace=self.trace,
            merge_batch_size=self.merge_batch_size,
        )
        # SkillOpt writes the merged patch to steps/step_NNNN/merged_patch.json
        # (trainer.py:1376); recording the dict here lets the F20 replay compare it
        # structurally rather than by count. Observability only.
        self.trace.record_decision(
            "merged_patch", patch=merged, n_edits=len(merge.patch_edits(merged)),
        )
        return merged

    def rank(self, epoch: int, text: str, merged: dict) -> list[dict]:
        """The ranked edits within this step's edit budget (the cosine schedule)."""
        budget = schedule.edit_budget(
            self.budget_step + 1, self.max_epochs, self.edit_budget_max, self.edit_budget_min,
        )
        self.budget_step += 1
        ranked = merge.rank(
            client=self.client(),
            skill_text=text,
            merged_edits=merge.patch_edits(merged),
            budget=budget,
            meta_skill=self.meta_skill,
            epoch=epoch,
            trace=self.trace,
        )
        # ``edits`` mirrors SkillOpt's steps/step_NNNN/ranked_edits.json
        # (trainer.py:1396, :1427), so the F20 replay compares the selected edits
        # themselves. Observability only.
        self.trace.record_decision(
            "ranked_edits", budget_step=self.budget_step, edit_budget=budget,
            n_edits=len(ranked), edits=ranked,
        )
        return ranked

    # ── end of epoch: slow update and meta skill ───────────────────────────

    def comparison(
        self, prev: tuple[list[dict], dict], curr: tuple[list[dict], dict], sample_ids: list[str],
    ) -> list[dict]:
        """The slow update's comparison pairs from two rollouts of the sample, each
        given as ``(records, knob values)``; kept for the meta skill."""
        self.comparison_pairs = comparison_pairs(
            prev[0], curr[0], sample_ids, env=self.env, prev_knobs=prev[1], curr_knobs=curr[1],
        )
        return self.comparison_pairs

    def slow_update(self, epoch: int, prev_text: str, curr_text: str) -> tuple[str, dict | None]:
        """The slow update's guidance for ``curr_text`` (slow_update.py:434-439),
        empty when the model gave none, and the parsed response."""
        prev_guidance = prompts.extract_slow_update_field(curr_text)  # trainer.py:1916
        user = prompts.format_slow_update_user(
            prev_text, curr_text, prev_guidance, prompts.format_comparison_text(self.comparison_pairs),
        )
        output = ask(
            self.client(), self.trace, "slow_update", user,
            system=prompts.SLOW_UPDATE_SYSTEM, call_key=("slow_update", epoch, 0, 0),
        )
        result = extract_json(output) if isinstance(output, str) else None
        guidance = ""
        if isinstance(result, dict) and result.get("slow_update_content"):
            guidance = str(result["slow_update_content"]).strip()
        return guidance, result

    def meta_skill_step(self, epoch: int, prev_text: str, last_step_text: str) -> None:
        """Mechanism 9, from epoch 2 (trainer.py:2083-2198): compare the previous
        epoch's text with this epoch's last-step text and write the meta skill; a
        response without content leaves none (trainer.py:384-397, :2189-2194)."""
        user = prompts.format_meta_skill_user(
            prev_text, last_step_text, self.meta_skill,
            prompts.format_comparison_text(self.comparison_pairs),
        )
        output = ask(
            self.client(), self.trace, "meta_skill", user,
            system=prompts.META_SKILL_SYSTEM, call_key=("meta_skill", epoch, 0, 0),
        )
        result = extract_json(output) if isinstance(output, str) else None  # meta_skill.py:70-75
        content = ""
        if isinstance(result, dict) and result.get("meta_skill_content"):
            content = str(result["meta_skill_content"]).strip()
        self.meta_skill = content
        self.trace.record_decision(
            "meta_skill",
            action="write_meta_skill" if content else "no_content",
            meta_skill_length=len(content),
            # meta_skill_result.json's own fields (meta_skill.py:70-75), compared
            # by the replay; SkillOpt writes them only for a skill it got.
            **({
                "reasoning": str(result.get("reasoning", "")).strip(),
                "meta_skill_content": content,
            } if content else {}),
        )

    # ── state ──────────────────────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        return {
            "meta_skill_text": self.meta_skill,
            "edit_budget_step": self.budget_step,
            "comparison_pairs": self.comparison_pairs,
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.meta_skill = data.get("meta_skill_text", "")
        self.budget_step = data.get("edit_budget_step", 0)
        self.comparison_pairs = data.get("comparison_pairs", [])

    # ── replay ordering hooks ──────────────────────────────────────────────

    def _analyst_records(self, epoch: int, records: list[dict]) -> list[dict]:
        """The epoch's records in the order the analysts split and shuffle them.

        The Run's order unless ``rollout_order`` is set; then its answer, which
        must be the same records in another order.
        """
        if self.rollout_order is None:
            return records
        ordered = list(self.rollout_order(epoch, list(records)))
        by_id = {str(r.get("id", "")): r for r in records}
        ids = [str(r.get("id", "")) for r in ordered]
        if len(by_id) != len(records) or sorted(ids) != sorted(by_id):
            raise ValueError(
                "skillopt: rollout_order must return a permutation of the epoch's "
                "records; it changed which records there are"
            )
        return [by_id[item_id] for item_id in ids]

    def _merge_order(self, epoch: int, kind: str, patches: list[dict]) -> list[dict]:
        """The epoch's ``kind`` patches in the order the merge receives them.

        The analysts' group order unless ``patch_order`` is set; then its answer,
        taken only as a permutation of these same patch objects.
        """
        if self.patch_order is None or len(patches) < 2:
            return patches
        ordered = list(self.patch_order(epoch, kind, list(patches)))
        remaining = list(range(len(patches)))
        picked: list[dict] = []
        for patch in ordered:
            index = next((i for i in remaining if patches[i] is patch), None)
            if index is None:
                raise ValueError(
                    "skillopt: patch_order must return a permutation of the epoch's "
                    "patches; it changed which patches there are"
                )
            remaining.remove(index)
            picked.append(patches[index])
        if remaining:
            raise ValueError(
                "skillopt: patch_order must return a permutation of the epoch's "
                "patches; it dropped some"
            )
        return picked


def batch_seed(seed: int, epoch: int, batch_idx: int = 0) -> int:
    """SkillOpt's train-batch seed for LiveMath: ``seed + epoch*1000 + batch_idx + 1``.

    LiveMath has a dataloader, so the trainer takes each batch's own seed from
    ``plan_train_epoch`` (trainer.py:1170-1177, :1242; dataloader.py:278) and
    hands it to the analysts as ``random_seed`` (trainer.py:1283). The
    ``Random(seed + epoch*1000)`` shuffle of base seeds (datasets/base.py:98-109)
    is the dataloader-less path and does not apply.
    """
    return seed + epoch * 1000 + batch_idx + 1


def knob_values(candidate: Candidate) -> dict[str, Any]:
    """The candidate's knob values by name, as its rollout used them."""
    knobs = candidate.state.get("knobs", {}) or {}
    return {
        name: entry.get("value") if isinstance(entry, dict) else entry
        for name, entry in knobs.items()
    }


def comparison_pairs(
    prev_records: list[dict],
    curr_records: list[dict],
    sample_ids: list[str],
    *,
    env: SkillOptEnv | None = None,
    prev_knobs: dict[str, Any] | None = None,
    curr_knobs: dict[str, Any] | None = None,
) -> list[dict]:
    """Per-sample comparison entries — slow_update.py:206-269 ``build_comparison_pairs``.

    SkillOpt builds them from the two rollouts' result dicts and reads each
    trajectory from the rollout's ``conversation.json``; here both come from
    the env's rendering of the F6 records (``SkillOptEnv.rollout_result``),
    each side with the knobs it was rolled out with.
    """
    env = env if env is not None else SkillOptEnv()
    prev_by_id = {
        str(r.get("id", "")): env.rollout_result(r, dict(prev_knobs or {}))
        for r in prev_records
    }
    curr_by_id = {
        str(r.get("id", "")): env.rollout_result(r, dict(curr_knobs or {}))
        for r in curr_records
    }
    pairs: list[dict] = []
    for item_id in sample_ids:
        prev = prev_by_id.get(item_id, {})
        curr = curr_by_id.get(item_id, {})
        prev_ok = _is_result_success(prev)
        curr_ok = _is_result_success(curr)
        if not prev_ok and curr_ok:
            category = "improved"
        elif prev_ok and not curr_ok:
            category = "regressed"
        elif not prev_ok and not curr_ok:
            category = "persistent_fail"
        else:
            category = "stable_success"
        # slow_update.py:242: SkillOpt reads the task from the sampled item;
        # an env's result carries the item's question (e.g.
        # livemathematicianbench/rollout.py:131).
        item = curr or prev
        pairs.append({
            "id": item_id,
            "task": item.get("question", item.get(
                "task_description", item.get("instruction", item_id))),
            "category": category,
            "prev": _outcome(prev, prev_ok),
            "curr": _outcome(curr, curr_ok),
            "prev_trajectory": _read_trajectory(prev),
            "curr_trajectory": _read_trajectory(curr),
        })
    return pairs


def hard_score(records: list[dict]) -> float:
    """SkillOpt's ``compute_score`` hard half over a rollout (utils/scoring.py:7-23)."""
    if not records:
        return 0.0
    return sum(1.0 if record.get("passed") else 0.0 for record in records) / len(records)


def _is_result_success(res: dict) -> bool:
    """slow_update.py:182-195: ``hard`` must be a finite number in [0, 1], >= 0.5."""
    if not res or not isinstance(res, dict):
        return False
    hard_val = res.get("hard")
    if isinstance(hard_val, (bool, int, float)):
        if math.isfinite(hard_val) and 0.0 <= hard_val <= 1.0:
            return float(hard_val) >= 0.5
    return False


def _get_soft_score(res: dict) -> float:
    """slow_update.py:198-211: ``soft`` when a finite number in [0, 1], else 0.0."""
    if not res or not isinstance(res, dict):
        return 0.0
    soft_val = res.get("soft")
    if isinstance(soft_val, (bool, int, float)):
        if math.isfinite(soft_val) and 0.0 <= soft_val <= 1.0:
            return float(soft_val)
    return 0.0


def _outcome(result: dict, passed: bool) -> dict[str, Any]:
    """One side of a comparison pair — slow_update.py:246-257."""
    return {
        "hard": int(passed),
        "soft": _get_soft_score(result),
        "predicted_answer": result.get("predicted_answer", result.get("answer", "N/A")),
        "fail_reason": result.get("fail_reason", ""),
    }


def _read_trajectory(result: dict) -> str:
    """slow_update.py:116-168 ``_read_trajectory``, over a rendered result.

    A result without a conversation is SkillOpt's missing ``conversation.json``.
    Unlike ``fmt_trajectory``, entries that are not dicts are skipped.
    """
    conversation = result.get("conversation") if result else None
    if conversation is None:
        return "(trajectory not available)"
    if not conversation:
        return "(empty trajectory)"
    return prompts.fmt_trajectory([e for e in conversation if isinstance(e, dict)])


def _op_to_kind(edit: dict[str, Any]) -> dict[str, Any]:
    """Convert SkillOpt's ``"op"`` key to a set_knob patch's ``"kind"`` key.

    SkillOpt prompts produce edits with ``"op": "append"``; a refactory
    ``set_knob`` patch reads ``"kind": "append"``.
    """
    out = dict(edit)
    if "op" in out and "kind" not in out:
        out["kind"] = out.pop("op")
    return out
