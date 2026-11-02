"""SkillOptLoop — SkillOpt as an OuterLoop strategy on refactory's public extension points.

Subclasses ``OuterLoop`` (the public strategy base at ``outerloop.py:296``)
and overrides only names in ``OuterLoop.overridable``: ``configure``,
``strategy``, ``strategy_state_dict``, and ``load_strategy_state``.

The loop machinery (``Loop``) is the caller: it constructs the strategy
object, calls ``configure(model=…, **strategy_options)`` after construction,
creates a ``Run`` and passes it to ``strategy(run)``.  The strategy keeps
its own state as plain instance attributes and interacts with the machinery
**only** through ``Run``'s public API — no ``__getattr__`` proxy, no
monkey-patching, no ``_infra`` attribute.  This is the "custom strategy"
seam named in the PR.

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

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.edits import SKILLOPT_PATCH_CONFIG
from refactory_dsh.optim import Candidate, Proposal
from refactory_dsh.outerloop import OuterLoop, Run

from refactory_packages.skillopt import analysts, merge, prompts, schedule, state
from refactory_packages.skillopt.json_utils import extract_json
from refactory_packages.skillopt.trace import TraceRecorder


class SkillOptLoop(OuterLoop):
    """SkillOpt outer-loop strategy — a pure custom strategy.

    Subclasses ``OuterLoop`` (the public strategy base) and overrides only
    names in ``OuterLoop.overridable``: ``configure``, ``strategy``,
    ``strategy_state_dict``, and ``load_strategy_state``.

    The loop machinery (``Loop``) is the caller: it constructs this object,
    calls ``configure(model=…, **strategy_options)``, creates a ``Run`` and
    passes it to ``strategy(run)``.  This strategy keeps its own state as
    plain instance attributes and interacts with the machinery **only**
    through ``Run``'s public API — no ``__getattr__`` proxy, no
    monkey-patching, no ``_infra`` attribute.

    Every model call goes through the injectable client interface stored in
    ``self.model`` (set by ``configure()``), with a stable call key recorded
    in the trace so the F20 replay can feed recorded outputs back.
    """

    name = "skillopt"
    requires_model = True

    # ── configure (overrides OuterLoop.configure) ─────────────────────────

    def configure(
        self,
        *,
        model: Any = None,
        model_name: str = "",
        provider: str = "auto",
        **options: Any,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.provider = provider
        self.options = dict(options)

        # All defaults cite SkillOpt's LiveMath config
        self.max_epochs = int(options.get("max_epochs", 4))
        self.early_stop_patience = int(options.get("early_stop_patience", 0))
        self.early_stop_at_full_val = bool(options.get("early_stop_at_full_val", False))
        self.slow_update_samples = int(options.get("slow_update_samples", 20))
        self.edit_budget_max = int(options.get("edit_budget_max", 4))
        self.edit_budget_min = int(options.get("edit_budget_min", 2))
        self.group_size = int(options.get("group_size", 8))
        self.merge_batch_size = int(options.get("merge_batch_size", 8))
        self.analyst_edit_budget = int(
            options.get("analyst_edit_budget", self.edit_budget_max),
        )
        self.base_seed = int(options.get("seed", 42))
        # Initialise mutable state
        self._current_skill: str = ""
        self._current_score: float = 0.0
        self._best_skill: str = ""
        self._best_score: float = 0.0
        self._prev_epoch_skill: str | None = None
        # The epoch's last-step skill, taken before its slow update; kept in
        # the state so a run resumed between the slow update and the meta
        # skill still compares the right skill (trainer.py:2176).
        self._epoch_last_step_skill: str = ""
        self._val_cache: dict[str, float] = {}
        self._edit_budget_step: int = 0
        self._meta_skill_text: str = ""
        self._comparison_pairs: list[dict] = []
        self._epoch: int = 0
        self._stale_epochs: int = 0
        self._trace: TraceRecorder | None = None
        # Per-phase marker for resume (ask 3)
        self._phase: str = ""
        # Cached intermediate outputs for resume (so we don't re-run model
        # calls for already-completed phases within the resume epoch).
        self._cached_failure_patches: list[dict] = []
        self._cached_success_patches: list[dict] = []
        self._cached_merged: dict = {}
        self._cached_ranked: list[dict] = []

    # ── strategy (overrides OuterLoop.strategy) ─────────────────────────────

    # Phase names for per-phase resume markers (ask 3).
    _PHASES = (
        "", "initial_val", "train", "analysts", "merge", "rank", "patch",
        "val_gate", "slow_update", "meta_skill", "epoch_done",
    )

    def _past_phase(self, resume_epoch: int, resume_phase: str, current_epoch: int, current_phase: str) -> bool:
        """Return True if the current phase was already completed on resume."""
        if current_epoch > resume_epoch:
            return False
        if current_epoch < resume_epoch:
            return True
        # Same epoch: compare phase order
        try:
            resume_idx = self._PHASES.index(resume_phase)
            current_idx = self._PHASES.index(current_phase)
        except ValueError:
            return False
        return current_idx <= resume_idx

    def strategy(self, run: Run) -> None:  # noqa: C901
        """Drive the full SkillOpt search."""
        # Initialise trace — uses Run.root (public seam)
        self._trace = TraceRecorder(run.root / "skillopt_trace.jsonl")

        # Train item ids for slow update sampling — uses Run.part_ids (public
        # seam). Checked before any evaluation so a run that cannot complete
        # the algorithm spends nothing.
        train_ids = run.part_ids("train")
        if self.max_epochs >= 2 and not train_ids:
            raise ValueError(
                "skillopt: the slow update needs the parts manifest's train ids "
                "(run.part_ids('train') returned none); give the run a parts "
                "manifest with a 'train' part or set max_epochs=1"
            )

        # Resume state: saved epoch and phase from a prior run
        resume_epoch = self._epoch
        resume_phase = self._phase
        resumed = bool(resume_epoch or resume_phase)

        seed_candidate = run.seed()
        if resumed:
            # Continue from the restored skill, never the seed's: the state the
            # journal replayed is where the interrupted run stood.
            current = self._candidate_for_skill(run, self._current_skill)
        else:
            self._current_skill = self._get_skill(seed_candidate)
            current = seed_candidate

        # ① Initial val score — mechanism 1 (skip on resume if progressed)
        if not self._past_phase(resume_epoch, resume_phase, 0, "initial_val"):
            self._phase = "initial_val"
            # SkillOpt evaluates the selection set with the base seed
            # (trainer.py:1133-1136; dataloader.py:291-301 shuffles with it).
            seed_candidate = run.evaluate(seed_candidate, part="val", seed=self.base_seed)
            current = seed_candidate
            self._current_score = self._val_score(run, seed_candidate)
            skill_hash = self._skill_hash(self._current_skill)
            self._val_cache[skill_hash] = self._current_score
            self._best_skill = self._current_skill
            self._best_score = self._current_score
            run.save_strategy_state()

            self._trace.record_decision(
                "initial_val",
                score=self._current_score,
                skill_hash=skill_hash,
            )

        for epoch in range(1, self.max_epochs + 1):
            self._epoch = epoch
            self._trace.set_epoch(epoch)

            # Skip completed epochs on resume.  An epoch whose phase is
            # "epoch_done" is fully complete — skip it just like earlier ones.
            if epoch < resume_epoch:
                continue
            if epoch == resume_epoch and resume_phase == "epoch_done":
                continue

            # ③ Check control — mechanism 11
            ctrl = run.check_control()
            if ctrl == "stop":
                run.save_strategy_state()
                run.stop("control_stop")
                return

            # ② Train rollout — mechanism 2
            # The batch seed (dataloader.py:278): seed + epoch*1000 + batch_idx + 1,
            # with one batch per epoch (batch_idx=0).
            epoch_seed = batch_seed(self.base_seed, epoch)
            if not self._past_phase(resume_epoch, resume_phase, epoch, "train"):
                self._phase = "train"
                current = run.evaluate(current, part="train", seed=epoch_seed)
                run.save_strategy_state()
            records = run.item_records(current)

            self._trace.record_decision(
                "train_rollout",
                n_records=len(records),
                epoch_seed=epoch_seed,
            )

            # ③ Analysts — mechanism 3
            if not self._past_phase(resume_epoch, resume_phase, epoch, "analysts"):
                self._phase = "analysts"
                # The analysts shuffle with the batch's own seed
                # (trainer.py:1242, :1283 pass batch.seed as random_seed).
                analyst_seed = epoch_seed
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
                # Cache for resume
                self._cached_failure_patches = failure_patches
                self._cached_success_patches = success_patches
                run.save_strategy_state()
            else:
                # Resume past analysts: restore from cached state (no model calls)
                failure_patches = self._cached_failure_patches
                success_patches = self._cached_success_patches

            # ⑥ No-edit skip — mechanism 6
            if not failure_patches and not success_patches:
                self._trace.record_decision("no_edits_skip")
                run.save_strategy_state()
                current = self._finish_epoch(
                    run, current, epoch, train_ids, resume_epoch, resume_phase,
                )
                self._phase = "epoch_done"
                run.save_strategy_state()
                continue

            # ④ Merge — mechanism 4
            if not self._past_phase(resume_epoch, resume_phase, epoch, "merge"):
                self._phase = "merge"
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
                # Cache for resume
                self._cached_merged = merged
                run.save_strategy_state()
            else:
                # Resume past merge: restore from cached state (no model calls)
                merged = self._cached_merged

            # Rank
            if not self._past_phase(resume_epoch, resume_phase, epoch, "rank"):
                self._phase = "rank"
                budget = schedule.edit_budget(
                    self._edit_budget_step + 1,
                    self.max_epochs,
                    self.edit_budget_max,
                    self.edit_budget_min,
                )
                self._edit_budget_step = self._edit_budget_step + 1
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
                # Cache for resume
                self._cached_ranked = ranked
                run.save_strategy_state()
            else:
                # Resume past rank: restore from cached state (no model calls,
                # no double-increment of _edit_budget_step).
                ranked = self._cached_ranked

            # ⑤ Patch — mechanism 5
            if not self._past_phase(resume_epoch, resume_phase, epoch, "patch"):
                self._phase = "patch"
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
                    self._trace.record_decision(
                        "patch_refused", reasons=outcome,
                    )
                    run.save_strategy_state()
                    current = self._finish_epoch(
                        run, current, epoch, train_ids, resume_epoch, resume_phase,
                    )
                    self._phase = "epoch_done"
                    run.save_strategy_state()
                    continue

                candidate = outcome
                edit_report = candidate.edit_reports[-1] if candidate.edit_reports else None
                self._trace.record_decision(
                    "patch_applied",
                    edit_report=edit_report,
                    n_edits=len(ranked),
                )

                cand_skill = self._get_skill(candidate)
                cand_hash = self._skill_hash(cand_skill)
                run.save_strategy_state()
            else:
                # Resume past patch: the candidate is the current (patch was
                # already applied and accepted/rejected in the original run).
                candidate = current
                cand_skill = self._current_skill
                cand_hash = self._skill_hash(cand_skill)

            # ⑦ Val gate — mechanism 7
            if not self._past_phase(resume_epoch, resume_phase, epoch, "val_gate"):
                self._phase = "val_gate"
                if cand_hash in self._val_cache:
                    cand_score = self._val_cache[cand_hash]
                    self._trace.record_decision(
                        "val_cache_hit", skill_hash=cand_hash, score=cand_score,
                    )
                else:
                    candidate = run.evaluate(candidate, part="val", seed=self.base_seed)
                    cand_score = self._val_score(run, candidate)
                    self._val_cache[cand_hash] = cand_score
                    self._trace.record_decision(
                        "val_evaluate", skill_hash=cand_hash, score=cand_score,
                    )

                if cand_score > self._current_score:
                    pre_accept_score = self._current_score
                    current = run.keep(candidate)
                    self._current_skill = cand_skill
                    self._current_score = cand_score
                    if cand_score > self._best_score:
                        self._best_skill = cand_skill
                        self._best_score = cand_score
                    self._stale_epochs = 0
                    self._trace.record_decision(
                        "val_gate_accept",
                        cand_score=cand_score,
                        current_score=pre_accept_score,
                    )
                else:
                    self._stale_epochs = self._stale_epochs + 1
                    self._trace.record_decision(
                        "val_gate_reject",
                        cand_score=cand_score,
                        current_score=self._current_score,
                    )

                run.save_strategy_state()

            # ⑧⑨ Slow update & meta skill — mechanisms 8, 9
            current = self._finish_epoch(
                run, current, epoch, train_ids, resume_epoch, resume_phase,
            )

            self._phase = "epoch_done"
            run.save_strategy_state()

            # Clear resume markers after the resume epoch so subsequent
            # epochs run normally.
            if epoch == resume_epoch:
                resume_epoch = 0
                resume_phase = ""

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

    def _finish_epoch(
        self,
        run: Run,
        current: Candidate,
        epoch: int,
        train_ids: list[str],
        resume_epoch: int,
        resume_phase: str,
    ) -> Candidate:
        """Slow update, then meta skill, then remember this epoch's skill.

        Each sub-step saves its own phase, so a run resumed after the slow update
        makes the meta skill rather than skipping it. The meta skill compares
        against the epoch's last-step skill, taken before this epoch's slow
        update rewrites it (trainer.py:1763, :2176).
        """
        if not self._past_phase(resume_epoch, resume_phase, epoch, "slow_update"):
            self._epoch_last_step_skill = self._current_skill
            self._phase = "slow_update"
            current = self._do_slow_update(run, current, epoch, train_ids)
            run.save_strategy_state()
        if not self._past_phase(resume_epoch, resume_phase, epoch, "meta_skill"):
            self._phase = "meta_skill"
            self._do_meta_skill(run, epoch, self._epoch_last_step_skill)
            # The previous epoch's skill for the next epoch is the saved skill of
            # this epoch's last step, which already carries the slow update
            # (trainer.py:1838-1839 load it; :2059 saved it after the update).
            self._prev_epoch_skill = self._current_skill
            run.save_strategy_state()
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
                self._current_skill = self._get_skill(current)
                self._trace.record_decision("slow_update_placeholder")
            run.save_strategy_state()
            return current

        if self._prev_epoch_skill is None:
            return current
        self._comparison_pairs = []

        # Epoch 2+: compare on a seeded sample — trainer.py:1843-1844; the
        # sample is the first n of the train items shuffled by that seed, rolled
        # out with the same seed (dataloader.py:278-283 build_train_batch).
        slow_seed = self.base_seed + epoch * 2000
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
        shuffled = list(train_ids)
        random.Random(slow_seed).shuffle(shuffled)
        sample_ids = shuffled[:n_samples]

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
        prev_candidate = run.evaluate(
            prev_candidate, part="train", items=sample_ids, seed=slow_seed,
        )
        prev_records = run.item_records(prev_candidate)

        # Evaluate current skill on sample
        curr_candidate = run.evaluate(
            current, part="train", items=sample_ids, seed=slow_seed,
        )
        curr_records = run.item_records(curr_candidate)

        # Build comparison pairs — slow_update.py:206-269
        pairs = self._build_comparison_pairs(prev_records, curr_records, sample_ids)
        self._comparison_pairs = pairs
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
            self._current_skill = self._get_skill(current)
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
        self._meta_skill_text = content

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
            "epoch_last_step_skill": self._epoch_last_step_skill,
            "val_cache": dict(self._val_cache),
            "edit_budget_step": self._edit_budget_step,
            "meta_skill_text": self._meta_skill_text,
            "comparison_pairs": self._comparison_pairs,
            "epoch": self._epoch,
            "stale_epochs": self._stale_epochs,
            "phase": self._phase,
            # Cached intermediate outputs for mid-epoch resume.
            "cached_failure_patches": self._cached_failure_patches,
            "cached_success_patches": self._cached_success_patches,
            "cached_merged": self._cached_merged,
            "cached_ranked": self._cached_ranked,
        })

    def load_strategy_state(self, data: dict[str, Any]) -> None:
        restored = state.deserialize_state(data)
        self._current_skill = restored.get("current_skill", "")
        self._current_score = restored.get("current_score", 0.0)
        self._best_skill = restored.get("best_skill", "")
        self._best_score = restored.get("best_score", 0.0)
        self._prev_epoch_skill = restored.get("prev_epoch_skill")
        self._epoch_last_step_skill = restored.get("epoch_last_step_skill", "")
        self._val_cache = restored.get("val_cache", {})
        self._edit_budget_step = restored.get("edit_budget_step", 0)
        self._meta_skill_text = restored.get("meta_skill_text", "")
        self._comparison_pairs = restored.get("comparison_pairs", [])
        self._epoch = restored.get("epoch", 0)
        self._stale_epochs = restored.get("stale_epochs", 0)
        self._phase = restored.get("phase", "")
        # Cached intermediate outputs for mid-epoch resume.
        self._cached_failure_patches = restored.get("cached_failure_patches", [])
        self._cached_success_patches = restored.get("cached_success_patches", [])
        self._cached_merged = restored.get("cached_merged", {})
        self._cached_ranked = restored.get("cached_ranked", [])

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _get_skill(candidate: Candidate) -> str:
        """Extract the ``skill`` knob value from a candidate's state."""
        knobs = candidate.state.get("knobs", {})
        skill_entry = knobs.get("skill", {})
        return str(skill_entry.get("value", ""))

    @classmethod
    def _candidate_for_skill(cls, run: Run, skill: str) -> Candidate:
        """The candidate carrying ``skill``: the seed with that skill set.

        A SkillOpt search edits only the ``skill`` knob, and a candidate's id is
        its state's, so this is the very candidate the interrupted run held --
        the evaluated one from the run's history when it was evaluated.
        """
        seed = run.seed()
        if cls._get_skill(seed) == skill:
            return seed
        built = Candidate.from_state(
            nn_edits.configure(seed.state, "skill", skill), parent=seed.id,
        )
        for candidate in run.history():
            if candidate.id == built.id:
                return candidate
        return built

    @staticmethod
    def _skill_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _val_score(run: Run, candidate: Candidate) -> float:
        """Extract the val score from the candidate's val scores."""
        scores = run.val_scores(candidate)
        if not scores:
            return 0.0
        raw = scores[0].get("score")
        return float(raw) if raw is not None else 0.0

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


def batch_seed(seed: int, epoch: int, batch_idx: int = 0) -> int:
    """SkillOpt's train-batch seed for LiveMath: ``seed + epoch*1000 + batch_idx + 1``.

    LiveMath has a dataloader, so the trainer takes each batch's own seed from
    ``plan_train_epoch`` (trainer.py:1170-1177, :1242; dataloader.py:278) and
    hands it to the analysts as ``random_seed`` (trainer.py:1283). The
    ``Random(seed + epoch*1000)`` shuffle of base seeds (datasets/base.py:98-109)
    is the dataloader-less path and does not apply.
    """
    return seed + epoch * 1000 + batch_idx + 1


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



def _op_to_kind(edit: dict[str, Any]) -> dict[str, Any]:
    """Convert SkillOpt's ``"op"`` key to text_patch's ``"kind"`` key.

    SkillOpt prompts produce edits with ``"op": "append"``; refactory's
    text_patch reads ``"kind": "append"``.
    """
    out = dict(edit)
    if "op" in out and "kind" not in out:
        out["kind"] = out.pop("op")
    return out
