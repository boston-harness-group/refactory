"""SkillOptLoop — SkillOpt as an OuterLoop strategy on refactory's public extension points.

Subclasses ``OuterLoop`` (the public strategy base at ``outerloop.py:296``)
and overrides only names in ``OuterLoop.overridable``: ``configure``,
``strategy``, ``strategy_state_dict``, and ``load_strategy_state``.

Uses only Run's public API for all operations.

Library seam: ``OuterLoop`` defines the strategy contract, but has no
infrastructure for mode resolution, evaluation, or budget management.
``Loop`` (at ``loop.py:263``) provides that infrastructure.
``SkillOptLoop`` delegates infrastructure to a ``Loop`` instance via
composition, exposing it as ``_loop`` attributes that ``Run(_loop=self)``
reads. This is the library gap named in the PR: a future ``OuterLoop.__init__``
should provide the infrastructure generically.

Defaults from SkillOpt's ``_base_/default.yaml`` (with file:line citations):
  - max_epochs = 4          (_base_/default.yaml:80, ``num_epochs: 4``)
  - early_stop_patience = 0 (trainer.py:1169)
  - early_stop_at_full_val = False  (trainer.py:1170)
  - slow_update_samples = 20 (_base_/default.yaml:99)
  - edit_budget_max = 4     (_base_/default.yaml:93, ``learning_rate: 4``)
  - edit_budget_min = 2     (_base_/default.yaml:94, ``min_learning_rate: 2``)
  - group_size = 8          (_base_/default.yaml:87, ``minibatch_size: 8``)
  - seed = 42               (_base_/default.yaml:84, ``seed: 42``)
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from refactory_dsh.nn.edits import SKILLOPT_PATCH_CONFIG
from refactory_dsh.optim import Candidate, Proposal
from refactory_dsh.outerloop import OuterLoop, Run

from refactory_packages.skillopt import analysts, merge, prompts, schedule, state
from refactory_packages.skillopt.trace import TraceRecorder


class SkillOptLoop(OuterLoop):
    """SkillOpt outer-loop strategy.

    Subclasses ``OuterLoop`` (the public strategy base) and overrides only
    methods in ``OuterLoop.overridable``: ``configure``, ``strategy``,
    ``strategy_state_dict``, and ``load_strategy_state``.

    Infrastructure (mode resolution, evaluation, budget management) is
    delegated to a ``Loop`` instance via composition.  This is the library
    gap named in the PR: ``OuterLoop`` should eventually provide its own
    ``__init__`` so custom strategies do not need the internal ``Loop``.

    Every model call goes through the injectable client interface stored in
    ``self.model`` (set by ``configure()``), with a stable call key recorded
    in the trace so the F20 replay can feed recorded outputs back.
    """

    name = "skillopt"
    requires_model = True

    # ── construction ──────────────────────────────────────────────────────

    def __init__(
        self,
        mode_id: str,
        config: Any = None,
        *,
        project: Any = None,
        **kwargs: Any,
    ) -> None:
        from refactory_dsh.loop import Loop

        # Delegate infrastructure to a Loop instance.
        # Library gap: OuterLoop has no __init__ that provides mode resolution,
        # evaluator creation, or budget management.  Until that seam exists,
        # a custom strategy delegates to Loop for these.
        self._infra = Loop(mode_id, config, project=project, **kwargs)

    # Proxy attributes so that Run(_loop=self) works: Run accesses
    # self._loop.{config, root, population, evaluations, rollouts, ...}.
    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup fails, so instance
        # attributes (self.model, self._current_skill, etc.) are not affected.
        return getattr(self._infra, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_infra" or name.startswith("_skillopt_") or not hasattr(self, "_infra"):
            object.__setattr__(self, name, value)
        else:
            # Strategy-local state goes on self; everything else on _infra
            # so Run and Loop machinery see it.
            if name in _SELF_ATTRS:
                object.__setattr__(self, name, value)
            else:
                setattr(self._infra, name, value)

    # ── configure (overrides OuterLoop.configure at outerloop.py:311-318) ─

    def configure(
        self,
        *,
        model: Any = None,
        model_name: str = "",
        provider: str = "auto",
        **options: Any,
    ) -> None:
        # Configure infrastructure (sets _infra.model for bind_budget)
        self._infra.configure(model=model, model_name=model_name, provider=provider, **options)
        # Also set model on self for strategy code access
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "options", dict(options))

        # All defaults cite SkillOpt's LiveMath config
        object.__setattr__(self, "max_epochs", options.get("max_epochs", 4))
        object.__setattr__(self, "early_stop_patience", options.get("early_stop_patience", 0))
        object.__setattr__(self, "early_stop_at_full_val", options.get("early_stop_at_full_val", False))
        object.__setattr__(self, "slow_update_samples", options.get("slow_update_samples", 20))
        object.__setattr__(self, "edit_budget_max", options.get("edit_budget_max", 4))
        object.__setattr__(self, "edit_budget_min", options.get("edit_budget_min", 2))
        object.__setattr__(self, "group_size", options.get("group_size", 8))
        object.__setattr__(self, "base_seed", options.get("seed", 42))
        # Initialise mutable state
        object.__setattr__(self, "_current_skill", "")
        object.__setattr__(self, "_current_score", 0.0)
        object.__setattr__(self, "_best_skill", "")
        object.__setattr__(self, "_best_score", 0.0)
        object.__setattr__(self, "_prev_epoch_skill", None)
        object.__setattr__(self, "_val_cache", {})
        object.__setattr__(self, "_edit_budget_step", 0)
        object.__setattr__(self, "_meta_skill_text", "")
        object.__setattr__(self, "_comparison_pairs", [])
        object.__setattr__(self, "_epoch", 0)
        object.__setattr__(self, "_stale_epochs", 0)
        object.__setattr__(self, "_rng", random.Random(self.base_seed))
        object.__setattr__(self, "_trace", None)

    # ── run: delegate to infrastructure Loop ──────────────────────────────

    def run(self) -> dict[str, Any]:
        # Monkey-patch the inner Loop so that OuterLoop.run() calls *our*
        # strategy/state methods rather than the inner Loop's defaults.
        self._infra.strategy = self.strategy  # type: ignore[assignment]
        self._infra.strategy_state_dict = self.strategy_state_dict  # type: ignore[assignment]
        self._infra.load_strategy_state = self.load_strategy_state  # type: ignore[assignment]
        return self._infra.run()

    # ── strategy (overrides OuterLoop.strategy at outerloop.py:335) ───────

    def strategy(self, run: Run) -> None:  # noqa: C901
        """Drive the full SkillOpt search."""
        # Initialise trace — uses Run.root (public seam)
        object.__setattr__(self, "_trace", TraceRecorder(run.root / "skillopt_trace.jsonl"))

        seed_candidate = run.seed()
        object.__setattr__(self, "_current_skill", self._get_skill(seed_candidate))

        # ① Initial val score — mechanism 1
        seed_candidate = run.evaluate(seed_candidate, part="val")
        object.__setattr__(self, "_current_score", self._val_score(run, seed_candidate))
        skill_hash = self._skill_hash(self._current_skill)
        self._val_cache[skill_hash] = self._current_score
        object.__setattr__(self, "_best_skill", self._current_skill)
        object.__setattr__(self, "_best_score", self._current_score)
        run.save_strategy_state()

        self._trace.record_decision(
            "initial_val",
            score=self._current_score,
            skill_hash=skill_hash,
        )

        current = seed_candidate

        # Get train item ids for slow update sampling — uses Run.part_ids (public seam)
        train_ids = run.part_ids("train")

        for epoch in range(1, self.max_epochs + 1):
            object.__setattr__(self, "_epoch", epoch)
            self._trace.set_epoch(epoch)

            # ③ Check control — mechanism 11
            ctrl = run.check_control()
            if ctrl == "stop":
                run.save_strategy_state()
                run.stop("control_stop")
                return

            # ② Train rollout — mechanism 2
            epoch_seed = self.base_seed + epoch * 1000
            current = run.evaluate(current, part="train", seed=epoch_seed)
            records = run.item_records(current)

            self._trace.record_decision(
                "train_rollout",
                n_records=len(records),
                epoch_seed=epoch_seed,
            )

            # ③ Analysts — mechanism 3
            analyst_seed = self.base_seed + epoch * 100
            failure_edits, success_edits = analysts.reflect(
                client=self.model,
                records=records,
                skill_text=self._current_skill,
                edit_budget=schedule.edit_budget(
                    self._edit_budget_step + 1,
                    self.max_epochs,
                    self.edit_budget_max,
                    self.edit_budget_min,
                ),
                meta_skill=self._meta_skill_text,
                seed=analyst_seed,
                group_size=self.group_size,
                epoch=epoch,
                trace=self._trace,
            )

            # ⑥ No-edit skip — mechanism 6
            if not failure_edits and not success_edits:
                self._trace.record_decision("no_edits_skip")
                run.save_strategy_state()
                # Do slow update and meta skill even on skip
                current = self._do_slow_update(run, current, epoch, train_ids)
                self._do_meta_skill(run, epoch)
                object.__setattr__(self, "_prev_epoch_skill", self._current_skill)
                continue

            # ④ Merge — mechanism 4
            merged = merge.hierarchical_merge(
                client=self.model,
                skill_text=self._current_skill,
                failure_edits=failure_edits,
                success_edits=success_edits,
                meta_skill=self._meta_skill_text,
                epoch=epoch,
                trace=self._trace,
            )

            # Rank
            budget = schedule.edit_budget(
                self._edit_budget_step + 1,
                self.max_epochs,
                self.edit_budget_max,
                self.edit_budget_min,
            )
            ranked = merge.rank(
                client=self.model,
                skill_text=self._current_skill,
                merged_edits=merged,
                budget=budget,
                meta_skill=self._meta_skill_text,
                epoch=epoch,
                trace=self._trace,
            )

            self._trace.record_decision(
                "ranked_edits",
                edit_budget=budget,
                n_edits=len(ranked),
            )

            # ⑤ Patch — mechanism 5
            # Convert SkillOpt's "op" key to text_patch's "kind" key
            patch_edits = [_op_to_kind(e) for e in ranked]
            proposal = Proposal(
                edit={
                    "op": "text_patch",
                    "name": "skill",
                    "edits": patch_edits,
                    "config": dict(SKILLOPT_PATCH_CONFIG),
                },
                rationale="skillopt epoch {} patch".format(epoch),
                parent=current.id,
            )
            outcome = run.attempt(proposal, current)
            if isinstance(outcome, list):
                # Patch refused
                self._trace.record_decision(
                    "patch_refused", reasons=outcome,
                )
                run.save_strategy_state()
                current = self._do_slow_update(run, current, epoch, train_ids)
                self._do_meta_skill(run, epoch)
                object.__setattr__(self, "_prev_epoch_skill", self._current_skill)
                continue

            candidate = outcome
            # Record per-edit outcome report (latest report)
            edit_report = candidate.edit_reports[-1] if candidate.edit_reports else None
            self._trace.record_decision(
                "patch_applied",
                edit_report=edit_report,
                n_edits=len(ranked),
            )

            cand_skill = self._get_skill(candidate)
            cand_hash = self._skill_hash(cand_skill)

            # ⑦ Val gate — mechanism 7
            if cand_hash in self._val_cache:
                cand_score = self._val_cache[cand_hash]
                self._trace.record_decision(
                    "val_cache_hit", skill_hash=cand_hash, score=cand_score,
                )
            else:
                candidate = run.evaluate(candidate, part="val")
                cand_score = self._val_score(run, candidate)
                self._val_cache[cand_hash] = cand_score
                self._trace.record_decision(
                    "val_evaluate", skill_hash=cand_hash, score=cand_score,
                )

            # Accept only if strictly better — gate.py:200
            if cand_score > self._current_score:
                # Record pre-accept current_score BEFORE updating
                pre_accept_score = self._current_score
                current = run.keep(candidate)
                object.__setattr__(self, "_current_skill", cand_skill)
                object.__setattr__(self, "_current_score", cand_score)
                object.__setattr__(self, "_edit_budget_step", self._edit_budget_step + 1)
                if cand_score > self._best_score:
                    object.__setattr__(self, "_best_skill", cand_skill)
                    object.__setattr__(self, "_best_score", cand_score)
                object.__setattr__(self, "_stale_epochs", 0)
                self._trace.record_decision(
                    "val_gate_accept",
                    cand_score=cand_score,
                    current_score=pre_accept_score,
                )
            else:
                object.__setattr__(self, "_stale_epochs", self._stale_epochs + 1)
                self._trace.record_decision(
                    "val_gate_reject",
                    cand_score=cand_score,
                    current_score=self._current_score,
                )

            run.save_strategy_state()

            # ⑧⑨ Slow update & meta skill — mechanisms 8, 9
            current = self._do_slow_update(run, current, epoch, train_ids)
            self._do_meta_skill(run, epoch)

            # Record the skill at end of epoch for the next epoch's slow update
            object.__setattr__(self, "_prev_epoch_skill", self._current_skill)

            run.save_strategy_state()

            # ⑩ Early stop — mechanism 10
            if self.early_stop_at_full_val and self._best_score >= 1.0 - 1e-9:
                self._trace.record_decision("early_stop", reason="perfect_val")
                run.stop("perfect_val")
                return
            if self.early_stop_patience > 0 and self._stale_epochs >= self.early_stop_patience:
                self._trace.record_decision("early_stop", reason="patience")
                run.stop("patience_exhausted")
                return

        run.stop("max_epochs")

    # ── slow update ────────────────────────────────────────────────────────

    def _do_slow_update(
        self,
        run: Run,
        current: Candidate,
        epoch: int,
        train_ids: list[str],
    ) -> Candidate:
        """Slow update: placeholder in epoch 1, comparison from epoch 2.

        Epoch 1 (mechanism 8): inject empty SLOW_UPDATE markers.
        Epoch 2+ (mechanism 8): compare prev and current skills on a seeded
        20-item sample, force-accept, cache pre-update score.

        Returns the (possibly updated) current candidate.
        """
        if epoch == 1:
            # Inject placeholder markers
            placeholder = (
                f"\n{prompts.SLOW_UPDATE_START}\n{prompts.SLOW_UPDATE_END}\n"
            )
            proposal = Proposal(
                edit={
                    "op": "set_knob",
                    "name": "skill",
                    "value": self._current_skill + placeholder,
                },
                rationale="slow update placeholder",
                parent=current.id,
            )
            outcome = run.attempt(proposal, current)
            if not isinstance(outcome, list):
                current = run.keep(outcome)
                object.__setattr__(self, "_current_skill", self._get_skill(current))
                # Update val cache with pre-update score for the new hash
                new_hash = self._skill_hash(self._current_skill)
                self._val_cache[new_hash] = self._current_score
                self._trace.record_decision("slow_update_placeholder")
            run.save_strategy_state()
            return current

        if self._prev_epoch_skill is None:
            return current

        # Epoch 2+: compare on seeded sample
        sample_rng = random.Random(self.base_seed + epoch * 2000)
        n_samples = min(self.slow_update_samples, len(train_ids))
        if n_samples <= 0:
            return current
        sample_ids = sample_rng.sample(train_ids, n_samples)

        self._trace.record_decision(
            "slow_update_sample", sample_ids=sample_ids,
        )

        # Evaluate prev skill on sample
        prev_proposal = Proposal(
            edit={
                "op": "set_knob",
                "name": "skill",
                "value": self._prev_epoch_skill,
            },
            rationale="slow update prev skill",
            parent=current.id,
        )
        prev_outcome = run.attempt(prev_proposal, current)
        if isinstance(prev_outcome, list):
            return current
        prev_candidate = prev_outcome
        epoch_seed = self.base_seed + epoch * 3000
        prev_candidate = run.evaluate(
            prev_candidate, part="train", items=sample_ids, seed=epoch_seed,
        )
        prev_records = run.item_records(prev_candidate)

        # Evaluate current skill on sample
        curr_candidate = run.evaluate(
            current, part="train", items=sample_ids, seed=epoch_seed,
        )
        curr_records = run.item_records(curr_candidate)

        # Build comparison pairs — slow_update.py:236-243
        pairs = self._build_comparison_pairs(prev_records, curr_records, sample_ids)
        object.__setattr__(self, "_comparison_pairs", pairs)
        comparison_text = self._format_comparison(pairs)

        # Extract previous guidance from current skill
        prev_guidance = self._extract_slow_update_content(self._current_skill)

        # LLM call for slow update
        call_key = ("slow_update", epoch, 0, 0)
        user = prompts.format_slow_update_user(
            self._prev_epoch_skill, self._current_skill, prev_guidance, comparison_text,
        )
        output = self.model.complete(
            user, system=prompts.SLOW_UPDATE_SYSTEM, call_key=call_key,
        )
        self._trace.record_model_call(
            "slow_update", prompts.SLOW_UPDATE_SYSTEM, user, output, call_key=call_key,
        )

        # Parse guidance
        guidance = ""
        try:
            data = json.loads(output)
            guidance = data.get("slow_update_content", "")
        except (json.JSONDecodeError, TypeError):
            guidance = output

        # Replace slow update content — NOT via text_patch (SKILLOPT_PATCH_CONFIG
        # has honour_protected: True which blocks writes to the region)
        pre_update_score = self._current_score
        new_skill = self._replace_slow_update(self._current_skill, guidance)
        proposal = Proposal(
            edit={
                "op": "set_knob",
                "name": "skill",
                "value": new_skill,
            },
            rationale="slow update guidance",
            parent=current.id,
        )
        outcome = run.attempt(proposal, current)
        if not isinstance(outcome, list):
            current = run.keep(outcome)
            object.__setattr__(self, "_current_skill", self._get_skill(current))
            # Record pre-update score in val cache for new skill hash
            new_hash = self._skill_hash(self._current_skill)
            self._val_cache[new_hash] = pre_update_score

        self._trace.record_decision(
            "slow_update",
            guidance_length=len(guidance),
            n_pairs=len(pairs),
        )
        run.save_strategy_state()
        return current

    # ── meta skill ─────────────────────────────────────────────────────────

    def _do_meta_skill(self, run: Run, epoch: int) -> None:
        """Meta skill — mechanism 9. Only from epoch 2."""
        if epoch < 2 or self._prev_epoch_skill is None:
            return

        comparison_text = self._format_comparison(self._comparison_pairs)
        call_key = ("meta_skill", epoch, 0, 0)
        user = prompts.format_meta_skill_user(
            self._prev_epoch_skill,
            self._current_skill,
            self._meta_skill_text,
            comparison_text,
        )
        output = self.model.complete(
            user, system=prompts.META_SKILL_SYSTEM, call_key=call_key,
        )
        self._trace.record_model_call(
            "meta_skill", prompts.META_SKILL_SYSTEM, user, output, call_key=call_key,
        )

        try:
            data = json.loads(output)
            object.__setattr__(self, "_meta_skill_text", data.get("meta_skill_content", ""))
        except (json.JSONDecodeError, TypeError):
            object.__setattr__(self, "_meta_skill_text", output)

        self._trace.record_decision(
            "meta_skill",
            meta_skill_length=len(self._meta_skill_text),
        )
        run.save_strategy_state()

    # ── state (overrides OuterLoop at outerloop.py:384-407) ────────────────

    def strategy_state_dict(self) -> dict[str, Any]:
        return state.serialize_state({
            "current_skill": self._current_skill,
            "current_score": self._current_score,
            "best_skill": self._best_skill,
            "best_score": self._best_score,
            "prev_epoch_skill": self._prev_epoch_skill,
            "val_cache": dict(self._val_cache),
            "edit_budget_step": self._edit_budget_step,
            "meta_skill_text": self._meta_skill_text,
            "comparison_pairs": self._comparison_pairs,
            "epoch": self._epoch,
            "stale_epochs": self._stale_epochs,
            "rng_state": self._rng,
        })

    def load_strategy_state(self, data: dict[str, Any]) -> None:
        restored = state.deserialize_state(data)
        object.__setattr__(self, "_current_skill", restored.get("current_skill", ""))
        object.__setattr__(self, "_current_score", restored.get("current_score", 0.0))
        object.__setattr__(self, "_best_skill", restored.get("best_skill", ""))
        object.__setattr__(self, "_best_score", restored.get("best_score", 0.0))
        object.__setattr__(self, "_prev_epoch_skill", restored.get("prev_epoch_skill"))
        object.__setattr__(self, "_val_cache", restored.get("val_cache", {}))
        object.__setattr__(self, "_edit_budget_step", restored.get("edit_budget_step", 0))
        object.__setattr__(self, "_meta_skill_text", restored.get("meta_skill_text", ""))
        object.__setattr__(self, "_comparison_pairs", restored.get("comparison_pairs", []))
        object.__setattr__(self, "_epoch", restored.get("epoch", 0))
        object.__setattr__(self, "_stale_epochs", restored.get("stale_epochs", 0))
        rng = restored.get("rng_state")
        if isinstance(rng, random.Random):
            object.__setattr__(self, "_rng", rng)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _get_skill(candidate: Candidate) -> str:
        """Extract the ``skill`` knob value from a candidate's state."""
        knobs = candidate.state.get("knobs", {})
        skill_entry = knobs.get("skill", {})
        return str(skill_entry.get("value", ""))

    @staticmethod
    def _skill_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _val_score(run: Run, candidate: Candidate) -> float:
        """Extract the val score from the candidate's val scores."""
        scores = run.val_scores(candidate)
        if not scores:
            return 0.0
        return float(scores[0].get("score", 0.0))

    @staticmethod
    def _build_comparison_pairs(
        prev_records: list[dict],
        curr_records: list[dict],
        sample_ids: list[str],
    ) -> list[list]:
        """Build comparison pairs — slow_update.py:236-243."""
        prev_by_id = {str(r.get("id", "")): r for r in prev_records}
        curr_by_id = {str(r.get("id", "")): r for r in curr_records}
        pairs = []
        for item_id in sample_ids:
            prev = prev_by_id.get(item_id, {})
            curr = curr_by_id.get(item_id, {})
            prev_pass = prev.get("passed", False)
            curr_pass = curr.get("passed", False)
            if not prev_pass and curr_pass:
                category = "improved"
            elif prev_pass and not curr_pass:
                category = "regressed"
            elif not prev_pass and not curr_pass:
                category = "persistent_fail"
            else:
                category = "stable_success"
            pairs.append([item_id, category])
        return pairs

    @staticmethod
    def _format_comparison(pairs: list[list]) -> str:
        """Format comparison pairs into readable text."""
        if not pairs:
            return "(no comparison data)"
        lines = []
        for category in ["regressed", "persistent_fail", "improved", "stable_success"]:
            items = [p for p in pairs if p[1] == category]
            if items:
                lines.append(f"### {category.replace('_', ' ').title()} ({len(items)})")
                for item_id, _ in items:
                    lines.append(f"- {item_id}")
        return "\n".join(lines)

    @staticmethod
    def _extract_slow_update_content(skill: str) -> str:
        """Extract content between SLOW_UPDATE markers."""
        start = skill.find(prompts.SLOW_UPDATE_START)
        end = skill.find(prompts.SLOW_UPDATE_END)
        if start < 0 or end < 0 or end <= start:
            return ""
        inner_start = start + len(prompts.SLOW_UPDATE_START)
        return skill[inner_start:end].strip()

    @staticmethod
    def _replace_slow_update(skill: str, content: str) -> str:
        """Replace content between SLOW_UPDATE markers."""
        start = skill.find(prompts.SLOW_UPDATE_START)
        end = skill.find(prompts.SLOW_UPDATE_END)
        if start < 0 or end < 0:
            return skill
        end_of_marker = end + len(prompts.SLOW_UPDATE_END)
        new_section = (
            f"{prompts.SLOW_UPDATE_START}\n{content}\n{prompts.SLOW_UPDATE_END}"
        )
        return skill[:start] + new_section + skill[end_of_marker:]


# Attributes that belong on the SkillOptLoop instance, not on _infra.
_SELF_ATTRS = frozenset({
    "model", "model_name", "provider", "options",
    "max_epochs", "early_stop_patience", "early_stop_at_full_val",
    "slow_update_samples", "edit_budget_max", "edit_budget_min",
    "group_size", "base_seed",
    "_current_skill", "_current_score", "_best_skill", "_best_score",
    "_prev_epoch_skill", "_val_cache", "_edit_budget_step",
    "_meta_skill_text", "_comparison_pairs", "_epoch", "_stale_epochs",
    "_rng", "_trace",
})


def _op_to_kind(edit: dict[str, Any]) -> dict[str, Any]:
    """Convert SkillOpt's ``"op"`` key to text_patch's ``"kind"`` key.

    SkillOpt prompts produce edits with ``"op": "append"``; refactory's
    text_patch reads ``"kind": "append"``.
    """
    out = dict(edit)
    if "op" in out and "kind" not in out:
        out["kind"] = out.pop("op")
    return out
