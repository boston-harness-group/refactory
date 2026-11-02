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
  - merge_batch_size = 8    (_base_/default.yaml:88, ``merge_batch_size: 8``)
  - analyst_edit_budget = 4 (envs/base.py:265 — the analysts' L is the adapter's
                             constant ``edit_budget``, not the schedule)
  - seed = 42               (_base_/default.yaml:84, ``seed: 42``)
"""

from __future__ import annotations

import hashlib
import random
import warnings
from typing import Any

from refactory_dsh.nn.edits import SKILLOPT_PATCH_CONFIG
from refactory_dsh.optim import Candidate, Proposal
from refactory_dsh.outerloop import OuterLoop, Run

from refactory_packages.skillopt import analysts, merge, prompts, schedule, state
from refactory_packages.skillopt.json_utils import extract_json
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
        object.__setattr__(self, "merge_batch_size", options.get("merge_batch_size", 8))
        object.__setattr__(
            self, "analyst_edit_budget",
            options.get("analyst_edit_budget", self.edit_budget_max),
        )
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

        # Train item ids for slow update sampling — uses Run.part_ids (public
        # seam). Checked before any evaluation so a run that cannot complete
        # the algorithm spends nothing.
        train_ids = run.part_ids("train")
        if self.max_epochs >= 2 and not train_ids:
            # The slow update samples the train set from epoch 2
            # (trainer.py:1843-1860); without train ids it cannot run, and
            # skipping it silently would make this a different algorithm.
            raise ValueError(
                "skillopt: the slow update needs the parts manifest's train ids "
                "(run.part_ids('train') returned none); give the run a parts "
                "manifest with a 'train' part or set max_epochs=1"
            )

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

            # ③ Analysts — mechanism 3. The shuffle seed is the step's batch
            # seed (trainer.py:1244, :1283) and L is the adapter's constant
            # edit budget (envs/base.py:265), not the scheduled one.
            analyst_seed = batch_seed(self.base_seed, epoch)
            failure_patches, success_patches = analysts.reflect(
                client=self.model,
                records=records,
                skill_text=self._current_skill,
                edit_budget=self.analyst_edit_budget,
                meta_skill=self._meta_skill_text,
                seed=analyst_seed,
                group_size=self.group_size,
                epoch=epoch,
                trace=self._trace,
            )

            # ⑥ No-edit skip — mechanism 6 (trainer.py:1336-1360: before the
            # scheduler steps, so the budget does not advance)
            if not failure_patches and not success_patches:
                self._trace.record_decision("no_edits_skip")
                run.save_strategy_state()
                current = self._end_epoch(run, current, epoch, train_ids)
                continue

            # ④ Merge — mechanism 4
            merged = merge.hierarchical_merge(
                client=self.model,
                skill_text=self._current_skill,
                failure_patches=failure_patches,
                success_patches=success_patches,
                meta_skill=self._meta_skill_text,
                epoch=epoch,
                trace=self._trace,
                merge_batch_size=self.merge_batch_size,
            )

            # Rank. The scheduler steps every time a step reaches ranking,
            # whatever the gate later decides — trainer.py:1415, before the
            # gate at :1583.
            budget = schedule.edit_budget(
                self._edit_budget_step + 1,
                self.max_epochs,
                self.edit_budget_max,
                self.edit_budget_min,
            )
            object.__setattr__(self, "_edit_budget_step", self._edit_budget_step + 1)
            ranked = merge.rank(
                client=self.model,
                skill_text=self._current_skill,
                merged_edits=merge.patch_edits(merged),
                budget=budget,
                meta_skill=self._meta_skill_text,
                epoch=epoch,
                trace=self._trace,
            )

            self._trace.record_decision(
                "ranked_edits",
                budget_step=self._edit_budget_step,
                edit_budget=budget,
                n_edits=len(ranked),
            )

            # ⑤ Patch — mechanism 5
            # Convert SkillOpt's "op" key to text_patch's "kind" key
            text_edits = [_op_to_kind(e) for e in ranked]
            proposal = Proposal(
                edit={
                    "op": "text_patch",
                    "name": "skill",
                    "edits": text_edits,
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
                current = self._end_epoch(run, current, epoch, train_ids)
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
            current = self._end_epoch(run, current, epoch, train_ids)

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

    # ── end of epoch: slow update then meta skill ─────────────────────────

    def _end_epoch(
        self,
        run: Run,
        current: Candidate,
        epoch: int,
        train_ids: list[str],
    ) -> Candidate:
        """Slow update, then meta skill, then remember this epoch's skill.

        The meta skill compares against the epoch's last-step skill, taken
        before this epoch's slow update rewrites it (trainer.py:1763, :2176).
        """
        epoch_last_step_skill = self._current_skill
        current = self._do_slow_update(run, current, epoch, train_ids)
        self._do_meta_skill(run, epoch, epoch_last_step_skill)
        # The previous epoch's skill for the next epoch is the saved skill of
        # this epoch's last step, which already carries the slow update
        # (trainer.py:1838-1839 load it; :2059 saved it after the update).
        object.__setattr__(self, "_prev_epoch_skill", self._current_skill)
        return current

    def _do_slow_update(
        self,
        run: Run,
        current: Candidate,
        epoch: int,
        train_ids: list[str],
    ) -> Candidate:
        """Slow update: placeholder in epoch 1, comparison from epoch 2.

        Epoch 1 (trainer.py:1809-1824): inject empty SLOW_UPDATE markers; the
        selection cache is not touched.
        Epoch 2+ (trainer.py:1825-2066): compare prev and current skills on a
        seeded sample, force-accept the guidance, cache the pre-update score.
        A response without guidance changes nothing (trainer.py:2040-2047).

        Returns the (possibly updated) current candidate.
        """
        if epoch == 1:
            proposal = Proposal(
                edit={
                    "op": "set_knob",
                    "name": "skill",
                    "value": prompts.inject_empty_slow_update_field(self._current_skill),
                },
                rationale="slow update placeholder",
                parent=current.id,
            )
            outcome = run.attempt(proposal, current)
            if not isinstance(outcome, list):
                current = run.keep(outcome)
                object.__setattr__(self, "_current_skill", self._get_skill(current))
                self._trace.record_decision("slow_update_placeholder")
            run.save_strategy_state()
            return current

        if self._prev_epoch_skill is None:
            return current
        object.__setattr__(self, "_comparison_pairs", [])

        # Epoch 2+: compare on a seeded sample — trainer.py:1843-1844
        sample_rng = random.Random(self.base_seed + epoch * 2000)
        n_samples = min(self.slow_update_samples, len(train_ids))
        if n_samples <= 0:
            warnings.warn(
                f"skillopt: slow update skipped in epoch {epoch} — no train items to sample "
                f"(slow_update_samples={self.slow_update_samples}, "
                f"train ids={len(train_ids)})",
                RuntimeWarning,
                stacklevel=2,
            )
            self._trace.record_decision("slow_update_skipped", reason="no_samples")
            return current
        sample_ids = sample_rng.sample(train_ids, n_samples)

        self._trace.record_decision(
            "slow_update_sample", sample_ids=sample_ids,
        )

        # Evaluate prev skill on sample. SkillOpt rolls out both skills even
        # when the epoch changed nothing (trainer.py:1866-1867); the same
        # skill cannot be re-proposed here (E-noop), so roll out `current`.
        if self._prev_epoch_skill == self._current_skill:
            prev_candidate = current
        else:
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
                self._trace.record_decision(
                    "slow_update_skipped", reason="prev_skill_refused", reasons=prev_outcome,
                )
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

        # Build comparison pairs — slow_update.py:206-269
        pairs = self._build_comparison_pairs(prev_records, curr_records, sample_ids)
        object.__setattr__(self, "_comparison_pairs", pairs)
        comparison_text = prompts.format_comparison_text(pairs)

        # Previous guidance from the current skill — trainer.py:1916
        prev_guidance = prompts.extract_slow_update_field(self._current_skill)

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

        # Parse guidance — slow_update.py:434-439
        result = extract_json(output) if isinstance(output, str) else None
        guidance = ""
        if isinstance(result, dict) and result.get("slow_update_content"):
            guidance = str(result["slow_update_content"]).strip()
        if not guidance:
            # No guidance: the skill is unchanged — trainer.py:2040-2047
            self._trace.record_decision("slow_update", action="no_content", n_pairs=len(pairs))
            run.save_strategy_state()
            return current

        # Force-accept (trainer.py:2042-2055). Not via text_patch:
        # SKILLOPT_PATCH_CONFIG honours the protected region, which blocks it.
        pre_update_score = self._current_score
        new_skill = prompts.replace_slow_update_field(self._current_skill, guidance)
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
            # Cache the pre-update score for the new hash — trainer.py:2052-2053
            new_hash = self._skill_hash(self._current_skill)
            self._val_cache[new_hash] = pre_update_score

        self._trace.record_decision(
            "slow_update",
            action="force_accept" if not isinstance(outcome, list) else "refused",
            guidance_length=len(guidance),
            n_pairs=len(pairs),
        )
        run.save_strategy_state()
        return current

    # ── meta skill ─────────────────────────────────────────────────────────

    def _do_meta_skill(self, run: Run, epoch: int, epoch_last_step_skill: str) -> None:
        """Meta skill — mechanism 9, from epoch 2 (trainer.py:2083-2198).

        Compares the previous epoch's skill with this epoch's last-step skill
        (before the slow update; trainer.py:2176). A response without content
        leaves no meta skill for the next epoch: SkillOpt then loads an empty
        one (trainer.py:384-397, :2189-2194).
        """
        if epoch < 2 or self._prev_epoch_skill is None:
            return

        comparison_text = prompts.format_comparison_text(self._comparison_pairs)
        call_key = ("meta_skill", epoch, 0, 0)
        user = prompts.format_meta_skill_user(
            self._prev_epoch_skill,
            epoch_last_step_skill,
            self._meta_skill_text,
            comparison_text,
        )
        output = self.model.complete(
            user, system=prompts.META_SKILL_SYSTEM, call_key=call_key,
        )
        self._trace.record_model_call(
            "meta_skill", prompts.META_SKILL_SYSTEM, user, output, call_key=call_key,
        )

        # meta_skill.py:70-75
        result = extract_json(output) if isinstance(output, str) else None
        content = ""
        if isinstance(result, dict) and result.get("meta_skill_content"):
            content = str(result["meta_skill_content"]).strip()
        object.__setattr__(self, "_meta_skill_text", content)

        self._trace.record_decision(
            "meta_skill",
            action="write_meta_skill" if content else "no_content",
            meta_skill_length=len(content),
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
    ) -> list[dict]:
        """Per-sample comparison entries — slow_update.py:206-269.

        Each entry carries the task, both outcomes (pass, score, answer, fail
        reason) and both trajectories, rendered from the F6 records.
        """
        prev_by_id = {str(r.get("id", "")): r for r in prev_records}
        curr_by_id = {str(r.get("id", "")): r for r in curr_records}
        pairs: list[dict] = []
        for item_id in sample_ids:
            prev = prev_by_id.get(item_id, {})
            curr = curr_by_id.get(item_id, {})
            prev_ok = bool(prev.get("passed", False))
            curr_ok = bool(curr.get("passed", False))
            if not prev_ok and curr_ok:
                category = "improved"
            elif prev_ok and not curr_ok:
                category = "regressed"
            elif not prev_ok and not curr_ok:
                category = "persistent_fail"
            else:
                category = "stable_success"
            pairs.append({
                "id": item_id,
                "task": prompts.record_task(curr or prev) or item_id,
                "category": category,
                "prev": _outcome(prev, prev_ok),
                "curr": _outcome(curr, curr_ok),
                "prev_trajectory": _trajectory(prev),
                "curr_trajectory": _trajectory(curr),
            })
        return pairs


def batch_seed(seed: int, epoch: int, batches_per_epoch: int = 1) -> int:
    """The seed of an epoch's first batch — SkillOpt's train-batch seed.

    datasets/base.py:98-109 (and trainer.py:966, :1181-1183): the base seeds
    are ``seed + i + 1`` for each batch of an epoch, shuffled by
    ``Random(seed + epoch * 1000)``. This strategy takes one step of one batch
    per epoch, so the step's batch seed is the first shuffled seed.
    """
    base_seeds = [seed + i + 1 for i in range(batches_per_epoch)]
    random.Random(seed + epoch * 1000).shuffle(base_seeds)
    return base_seeds[0]


def _outcome(record: dict, passed: bool) -> dict[str, Any]:
    """One side of a comparison pair — slow_update.py:249-260."""
    score = record.get("score")
    try:
        soft = float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        soft = 0.0
    prediction = record.get("prediction")
    return {
        "hard": int(passed),
        "soft": soft,
        "predicted_answer": prediction if prediction is not None else "N/A",
        "fail_reason": prompts.record_fail_reason(record) if record else "",
    }


def _trajectory(record: dict) -> str:
    """slow_update.py:116-168 ``_read_trajectory``, from an F6 record."""
    if not record:
        return "(trajectory not available)"
    return prompts.format_trajectory(record)


# Attributes that belong on the SkillOptLoop instance, not on _infra.
_SELF_ATTRS = frozenset({
    "model", "model_name", "provider", "options",
    "max_epochs", "early_stop_patience", "early_stop_at_full_val",
    "slow_update_samples", "edit_budget_max", "edit_budget_min",
    "group_size", "merge_batch_size", "analyst_edit_budget", "base_seed",
    "_current_skill", "_current_score", "_best_skill", "_best_score",
    "_prev_epoch_skill", "_val_cache", "_edit_budget_step",
    "_meta_skill_text", "_comparison_pairs", "_epoch", "_stale_epochs",
    "_trace",
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
