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
  - eval_test = on when the parts manifest has a test part (_base_/default.yaml:111
                ``eval_test: true``; trainer.py:2241). After training: the final
                skill's val check and promotion, then the initial, best and final
                skills on the test part with the run seed (trainer.py:2241-2441),
                reported as A's summary.json names them (trainer.py:2480-2515).
                ``eval_test=false`` skips it, and the report says so.
"""

from __future__ import annotations

import hashlib
import math
import random
import warnings
from typing import Any

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.edits import SKILLOPT_PATCH_CONFIG
from refactory_dsh.optim import Candidate, Proposal
from refactory_dsh.outerloop import EvalRequest, OuterLoop, OuterLoopError, Run

from refactory_packages.skillopt import analysts, merge, prompts, schedule, state
from refactory_packages.skillopt.env import SkillOptEnv
from refactory_packages.skillopt.json_utils import extract_json
from refactory_packages.skillopt.trace import TraceRecorder, ask


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
    #: The patch step is a ``text_patch``; the slow update and the previous
    #: skill's rollout are ``set_knob``.
    required_ops = ("text_patch", "set_knob")

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
        # SkillOpt's post-training evaluation (trainer.py:2241 ``if cfg["eval_test"]``).
        # ``None`` (unset) runs it when the parts manifest has a test part and says
        # why not otherwise; ``True`` requires a test part; ``False`` skips it.
        eval_test = options.get("eval_test")
        if eval_test is not None and not isinstance(eval_test, bool):
            raise TypeError(
                "skillopt: eval_test must be true or false (or unset: on when the parts "
                f"manifest has a test part), not {eval_test!r}"
            )
        self.eval_test = eval_test
        # The order the analysts receive an epoch's train records in, before
        # the seeded shuffle. ``None`` (every live run) keeps the Run's own
        # deterministic item order. SkillOpt's order is the rollout's
        # completion order (livemathematicianbench/rollout.py:383-436 appends
        # results as futures finish; reflect.py:558-561 splits and shuffles
        # that list), which no code can reproduce; an offline replay passes a
        # callable ``(epoch, records) -> records`` that restores the recorded
        # order. Only a callable is accepted, so a run configured from the
        # command line or a file (strings, numbers) cannot set it, and
        # ``_analyst_records`` refuses anything but a permutation.
        rollout_order = options.get("rollout_order")
        if rollout_order is not None and not callable(rollout_order):
            raise TypeError(
                "skillopt: rollout_order must be a callable (epoch, records) -> records "
                f"or None, not {type(rollout_order).__name__}"
            )
        self.rollout_order = rollout_order
        # The order of an epoch's analyst patches before the merge. ``None``
        # (every live run) keeps the analysts' group order. SkillOpt's order is
        # the analysts' completion order (reflect.py:643-650 appends each patch
        # as its future completes; trainer.py:157-186 keeps it per source), which
        # no code can reproduce; it shows in the merge prompts and in a merge's
        # fallback concatenation (aggregate.py:70-76). An offline replay passes
        # a callable ``(epoch, kind, patches) -> patches`` that restores the
        # recorded order; like ``rollout_order`` only a callable is accepted and
        # only a permutation is taken.
        patch_order = options.get("patch_order")
        if patch_order is not None and not callable(patch_order):
            raise TypeError(
                "skillopt: patch_order must be a callable (epoch, kind, patches) -> "
                f"patches or None, not {type(patch_order).__name__}"
            )
        self.patch_order = patch_order
        # Initialise mutable state
        self._current_skill: str = ""
        self._current_score: float = 0.0
        self._best_skill: str = ""
        self._best_score: float = 0.0
        self._best_candidate_id: str | None = None
        # Why the run stopped as a failure (an epoch whose every optimizer call
        # failed), for the report; empty while the run is healthy.
        self._run_error: str = ""
        self._prev_epoch_skill: str | None = None
        # The epoch's last-step skill, taken before its slow update; kept in
        # the state so a run resumed between the slow update and the meta
        # skill still compares the right skill (trainer.py:2176).
        self._epoch_last_step_skill: str = ""
        # The skill the epoch's patch produced, saved with the "patch" phase so
        # a run resumed before the val gate gates the patched candidate.
        self._patched_skill: str = ""
        self._val_cache: dict[str, float] = {}
        self._edit_budget_step: int = 0
        self._meta_skill_text: str = ""
        self._comparison_pairs: list[dict] = []
        self._epoch: int = 0
        self._stale_epochs: int = 0
        # The best score the early stop measures improvement against.
        self._early_stop_best: float = 0.0
        self._trace: TraceRecorder | None = None
        # The mode's package as SkillOpt's env adapter; set when the search starts.
        self._env: SkillOptEnv = SkillOptEnv()
        # Per-phase marker for resume (ask 3)
        self._phase: str = ""
        # Cached intermediate outputs for resume (so we don't re-run model
        # calls for already-completed phases within the resume epoch).
        self._cached_failure_patches: list[dict] = []
        self._cached_success_patches: list[dict] = []
        self._cached_merged: dict = {}
        self._cached_ranked: list[dict] = []
        # Why training ended (``run.stop``'s reason), kept so a run resumed in the
        # post-training evaluation stops for the same reason.
        self._train_stop: str = ""
        # The post-training evaluation's outcome, under A's summary.json names.
        self._final_eval: dict[str, Any] = {}
        # Completed post-training phases for set-based resume (correction 2).
        self._completed_phases: set[str] = set()

    # ── strategy (overrides OuterLoop.strategy) ─────────────────────────────

    # Phase names for per-phase resume markers (ask 3).
    _PHASES = (
        "", "initial_val", "train", "analysts", "merge", "rank", "patch",
        "val_gate", "slow_update", "meta_skill", "epoch_done",
    )
    #: The post-training evaluation's phases, in A's order (trainer.py:2241-2441):
    #: the final skill's val check, then the initial, best and final skills on test.
    _FINAL_PHASES = ("final_selection", "baseline_test", "best_test", "final_test", "final_done")
    #: The bounds training ends on normally; after any other (a person's stop, an
    #: epoch whose every optimizer call failed) there is no post-training evaluation.
    _TRAINED = ("max_epochs", "perfect_val", "patience_exhausted")

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

    def strategy(self, run: Run) -> None:
        """Drive the full SkillOpt search: training, then SkillOpt's post-training
        evaluation (trainer.py:2241-2441) when training ended normally."""
        # Initialise trace — uses Run.root (public seam)
        self._trace = TraceRecorder(run.root / "skillopt_trace.jsonl")
        # Every optimizer call counts as a proposal round in the run report, so a
        # run whose optimizer never answered says so (Loop._proposer_health).
        self._trace.listener = run.record_proposal_round
        # SkillOpt's env is the adapter the run's env names (envs/base.py:288-296
        # derives it from the adapter's module path); here it is the package that
        # holds the run's mode -- its prompt overrides and rollout rendering.
        self._env = SkillOptEnv.for_mode(str(run.state().get("mode", "")))
        # A run told to evaluate on test that cannot is refused before it spends.
        if self.eval_test and not run.has_part("test"):
            raise ValueError(
                "skillopt: eval_test=true needs a parts manifest with a 'test' part; "
                "give the run one, or set eval_test=false"
            )

        if self._phase in self._FINAL_PHASES:
            # Resumed after training: stop for the reason training ended on.
            run.stop(self._train_stop)
        else:
            self._train(run)
            self._train_stop = run.stopped or ""
        if run.stopped in self._TRAINED and not self._run_error:
            self._final_evaluation(run)
        else:
            self._final_eval = {
                "status": "skipped",
                "reason": f"training did not finish: it ended on {run.stopped!r}"
                          + (f" ({self._run_error})" if self._run_error else ""),
            }
            run.save_strategy_state()

    def _train(self, run: Run) -> None:  # noqa: C901
        """SkillOpt's training: the initial val score, then one step per epoch."""
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
            self._best_candidate_id = seed_candidate.id
            self._early_stop_best = self._best_score
            run.save_strategy_state()

            self._trace.record_decision(
                "initial_val",
                score=self._current_score,
                skill_hash=skill_hash,
            )

        for epoch in range(1, self.max_epochs + 1):
            self._epoch = epoch
            self._epoch_calls_mark = (
                self._trace.optimizer_calls, len(self._trace.optimizer_failures),
            )
            self._trace.set_epoch(epoch)

            # Skip completed epochs on resume.  An epoch whose phase is
            # "epoch_done" is fully complete — skip it just like earlier ones.
            if epoch < resume_epoch:
                continue
            if epoch == resume_epoch and resume_phase == "epoch_done":
                if self._early_stop(run):
                    return
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

            # ``rollout_ids`` and ``rollout_hard`` are the Run's own per-item
            # records reduced to the two figures SkillOpt writes per step
            # (history.json ``rollout_hard``, steps/step_NNNN/rollout/results.jsonl
            # ``id``): recorded so the trace alone identifies what this epoch
            # rolled out. Observability only — nothing reads them back.
            self._trace.record_decision(
                "train_rollout",
                n_records=len(records),
                epoch_seed=epoch_seed,
                rollout_ids=[str(record.get("id", "")) for record in records],
                rollout_hard=(
                    sum(1 for record in records if record.get("passed", False)) / len(records)
                    if records else None
                ),
            )

            # ③ Analysts — mechanism 3
            if not self._past_phase(resume_epoch, resume_phase, epoch, "analysts"):
                self._phase = "analysts"
                # The analysts shuffle with the batch's own seed
                # (trainer.py:1242, :1283 pass batch.seed as random_seed).
                analyst_seed = epoch_seed
                failure_patches, success_patches = analysts.reflect(
                    client=self.model,
                    records=self._analyst_records(epoch, records),
                    skill_text=self._current_skill,
                    edit_budget=self.analyst_edit_budget,
                    meta_skill=self._meta_skill_text,
                    seed=analyst_seed,
                    group_size=self.group_size,
                    epoch=epoch,
                    trace=self._trace,
                    env=self._env,
                    knobs=_knob_values(current),
                )
                failure_patches = self._merge_order(epoch, "failure", failure_patches)
                success_patches = self._merge_order(epoch, "success", success_patches)
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
                self._count_stale_epoch()
                if self._record_epoch_calls(run, epoch):
                    self._phase = "epoch_done"
                    run.save_strategy_state()
                    return
                self._phase = "epoch_done"
                run.save_strategy_state()
                if self._early_stop(run):
                    return
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
                # SkillOpt writes the merged patch to steps/step_NNNN/merged_patch.json
                # (trainer.py:1376); recording the dict here
                # is what lets the F20 replay compare it structurally rather than by
                # count. Observability only — nothing reads it back.
                self._trace.record_decision(
                    "merged_patch",
                    patch=merged,
                    n_edits=len(merge.patch_edits(merged)),
                )
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

                # ``edits`` mirrors SkillOpt's steps/step_NNNN/ranked_edits.json
                # (trainer.py:1396, :1427) so
                # the F20 replay compares the selected edits themselves, not their
                # count. Observability only — nothing reads it back.
                self._trace.record_decision(
                    "ranked_edits",
                    budget_step=self._edit_budget_step,
                    edit_budget=budget,
                    n_edits=len(ranked),
                    edits=ranked,
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
                if isinstance(outcome, list) and _only_noop(outcome):
                    # The edits changed nothing: SkillOpt still gates the
                    # unchanged skill (trainer.py:1507, :1514), its selection
                    # cache answers with the score it holds for that hash
                    # (:1567-1572), and the gate does not accept an equal score.
                    candidate = current
                    # Its per-edit outcomes, as SkillOpt writes edit_apply_report.json
                    # for it (trainer.py:1510-1512). Observability only.
                    noop_reports: list[dict[str, Any]] = []
                    nn_edits.apply(proposal.edit, current.state, edit_reports=noop_reports)
                    self._trace.record_decision(
                        "patch_noop", reasons=outcome, n_edits=len(ranked),
                        edit_report=noop_reports[0] if noop_reports else None,
                    )
                elif isinstance(outcome, list):
                    self._trace.record_decision(
                        "patch_refused", reasons=outcome,
                    )
                    run.save_strategy_state()
                    current = self._finish_epoch(
                        run, current, epoch, train_ids, resume_epoch, resume_phase,
                    )
                    self._count_stale_epoch()
                    if self._record_epoch_calls(run, epoch):
                        self._phase = "epoch_done"
                        run.save_strategy_state()
                        return
                    self._phase = "epoch_done"
                    run.save_strategy_state()
                    if self._early_stop(run):
                        return
                    continue
                else:
                    candidate = outcome
                    edit_report = candidate.edit_reports[-1] if candidate.edit_reports else None
                    self._trace.record_decision(
                        "patch_applied",
                        edit_report=edit_report,
                        n_edits=len(ranked),
                    )

                cand_skill = self._get_skill(candidate)
                cand_hash = self._skill_hash(cand_skill)
                self._patched_skill = cand_skill
                run.save_strategy_state()
            else:
                # Resume past patch: the patched candidate is the one the saved
                # "patch" phase recorded, rebuilt from its skill (a candidate's
                # id is its state's, so it is the very candidate patched).
                cand_skill = self._patched_skill or self._current_skill
                candidate = self._candidate_for_skill(run, cand_skill)
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
                        self._best_candidate_id = candidate.id
                    self._trace.record_decision(
                        "val_gate_accept",
                        cand_score=cand_score,
                        current_score=pre_accept_score,
                    )
                else:
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

            self._count_stale_epoch()
            if self._record_epoch_calls(run, epoch):
                self._phase = "epoch_done"
                run.save_strategy_state()
                return
            self._phase = "epoch_done"
            run.save_strategy_state()

            # Clear resume markers after the resume epoch so subsequent
            # epochs run normally.
            if epoch == resume_epoch:
                resume_epoch = 0
                resume_phase = ""

            # ⑩ Early stop — mechanism 10
            if self._early_stop(run):
                return

        run.stop("max_epochs")

    def _analyst_records(self, epoch: int, records: list[dict]) -> list[dict]:
        """The epoch's records in the order the analysts split and shuffle them.

        The Run's order unless ``rollout_order`` is set (see ``configure``);
        then its answer, which must be the same records in another order.
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

    # ── after training: trainer.py:2241-2441 ──────────────────────────────

    def _final_evaluation(self, run: Run) -> None:
        """SkillOpt's evaluation after training, in its order (trainer.py:2241-2441).

        1. The final skill's val check (trainer.py:2244-2302): the best's score when
           the final skill is the best (:2253-2258), else the final skill rolled out
           on val; a score above the best's promotes the final skill to best
           (:2282-2298).
        2. The initial skill on test -- the baseline (trainer.py:2304-2337).
        3. The best (val-selected) skill on test (trainer.py:2339-2372).
        4. The final skill on test (trainer.py:2374-2441); the best's score when the
           two skills are the same, as A reuses it (:2383-2402).

        Every rollout uses the run seed, as A's ``_build_eval_env(split=...,
        seed=seed)`` does, so a mode that shuffles per item and seed (LiveMath's
        choices) shows each item as A's did. Each step saves its phase, so a
        resumed run continues where it stopped; the scores are A's summary.json
        names (trainer.py:2480-2515), in the strategy state and its report.

        The three test evaluations (baseline, best, final) run concurrently via
        ``evaluate_many`` (correction 2).  Trace decisions are written after
        ``evaluate_many`` returns, in A's canonical order, not completion order.
        """
        skip = self._final_eval_skip(run)
        if skip:
            self._final_eval = {"status": "skipped", "reason": skip}
            self._trace.record_decision("test_eval", action="skipped", reason=skip)
            run.log(f"skillopt: no post-training test evaluation: {skip}")
            run.save_strategy_state()
            return
        # Use _completed_phases for set-based resume (correction 2).
        if not self._completed_phases:
            # A fresh evaluation (or backward-compat from _phase-based resume).
            done = self._phase if self._phase in self._FINAL_PHASES else ""
            if done:
                # Backward compat: populate from linear _phase
                idx = self._FINAL_PHASES.index(done)
                self._completed_phases = set(self._FINAL_PHASES[:idx + 1])
            else:
                self._final_eval = {}
        result = self._final_eval
        result["status"] = "running"

        def pending(phase: str) -> bool:
            return phase not in self._completed_phases

        def finished(phase: str) -> None:
            self._completed_phases.add(phase)
            self._phase = phase
            run.save_strategy_state()

        try:
            # Step 1: final_selection runs first (sequential, as its result
            # can promote the best skill).
            if pending("final_selection"):
                if self._is_best(self._current_skill):
                    result["final_selection_hard"] = self._best_score
                    self._trace.record_decision(
                        "final_selection_eval", action="reused", score=self._best_score)
                else:
                    final = self._candidate_for_skill(run, self._current_skill)
                    run.evaluate(final, part="val", seed=self.base_seed)
                    score = self._latest_score(run, final, "val")
                    result["final_selection_hard"] = score
                    promoted = score is not None and score > self._best_score
                    if promoted:
                        self._best_skill = self._current_skill
                        self._best_score = score
                        self._best_candidate_id = final.id
                    self._trace.record_decision(
                        "final_selection_eval", action="promoted" if promoted else "evaluated",
                        score=score, best_score=self._best_score,
                    )
                finished("final_selection")

            # Step 2-4: collect pending test phases and run concurrently.
            # Each entry: (phase, result_key, candidate, is_reuse)
            test_plan: list[tuple[str, str, Candidate, bool]] = []
            final_is_best = self._is_best(self._current_skill)

            if pending("baseline_test"):
                test_plan.append(("baseline_test", "baseline_test_hard", run.seed(), False))
            if pending("best_test"):
                test_plan.append(("best_test", "test_hard",
                                  self._candidate_for_skill(run, self._best_skill), False))
            if pending("final_test"):
                if final_is_best:
                    # Reuse best_test score — no evaluation needed
                    test_plan.append(("final_test", "final_test_hard", None, True))
                else:
                    test_plan.append(("final_test", "final_test_hard",
                                      self._candidate_for_skill(run, self._current_skill), False))

            # Build EvalRequests for phases that need actual evaluation
            requests: list[EvalRequest] = []
            request_plan: list[tuple[str, str]] = []  # (phase, result_key) per request
            reuse_phases: list[tuple[str, str]] = []  # phases that reuse a score

            for phase, key, cand, is_reuse in test_plan:
                if is_reuse:
                    reuse_phases.append((phase, key))
                else:
                    requests.append(EvalRequest(candidate=cand, part="test", seed=self.base_seed))
                    request_plan.append((phase, key))

            # Collected scores from concurrent evaluations
            eval_scores: dict[str, float | None] = {}
            eval_candidates: dict[str, str] = {}

            if requests:
                def _on_result(index: int, candidate: Candidate) -> None:
                    phase, key = request_plan[index]
                    score = self._latest_score(run, candidate, "test")
                    eval_scores[phase] = score
                    eval_candidates[phase] = candidate.id
                    result[key] = score
                    finished(phase)

                run.evaluate_many(requests, on_result=_on_result)

            # Handle reuse phases (final_test when final == best)
            for phase, key in reuse_phases:
                result[key] = result.get("test_hard")
                eval_scores[phase] = result[key]
                finished(phase)

            # Write trace decisions in A's canonical order AFTER evaluate_many
            # returns (correction 2: order must not depend on completion order).
            _test_order = [
                ("baseline_test", "baseline_test_hard"),
                ("best_test", "test_hard"),
            ]
            for phase, key in _test_order:
                if phase in eval_scores:
                    self._trace.record_decision(
                        "test_eval", action=phase,
                        score=eval_scores[phase],
                        candidate=eval_candidates.get(phase, ""))
            # final_test trace
            if "final_test" in eval_scores:
                if final_is_best:
                    self._trace.record_decision(
                        "test_eval", action="final_test_reused",
                        score=eval_scores["final_test"])
                else:
                    self._trace.record_decision(
                        "test_eval", action="final_test",
                        score=eval_scores["final_test"],
                        candidate=eval_candidates.get("final_test", ""))

        except OuterLoopError as error:
            # A's evaluation tolerates a failed step (trainer.py:2299-2302, :2439-2441);
            # here the harness refused one (a spent rollout budget): say so and stop.
            result["status"] = "incomplete"
            result["reason"] = str(error)
            self._trace.record_decision("test_eval", action="incomplete", reason=str(error))
            run.log(f"skillopt: post-training evaluation stopped: {error}")
            run.save_strategy_state()
            return
        # trainer.py:2503-2512
        for delta, score in (("test_delta_hard", "test_hard"), ("final_test_delta_hard", "final_test_hard")):
            result[delta] = (
                result[score] - (result.get("baseline_test_hard") or 0)
                if result.get(score) is not None else None
            )
        result["status"] = "done"
        finished("final_done")

    def _final_eval_skip(self, run: Run) -> str:
        """Why the post-training evaluation does not run; empty when it does."""
        if self.eval_test is False:
            return "the eval_test strategy option is false"
        if not run.has_part("test"):
            return "the parts manifest has no 'test' part"
        return ""

    def _is_best(self, skill: str) -> bool:
        """SkillOpt's ``skill_hash(current_skill) == skill_hash(best_skill)``."""
        return self._skill_hash(skill) == self._skill_hash(self._best_skill)

    @staticmethod
    def _latest_score(run: Run, candidate: Candidate, part: str) -> float | None:
        """The candidate's score on its latest ``part`` evaluation; ``None`` if it failed."""
        scores = run.part_scores(candidate, part)
        raw = scores[-1].get("score") if scores else None
        return float(raw) if raw is not None else None

    def report(self, run: Run) -> dict[str, Any]:
        """The strategy's own figures, under A's summary.json names (trainer.py:2480-2515).

        ``test_eval`` says whether the post-training evaluation ran (``done``), was
        ``skipped`` or stopped ``incomplete``, and ``test_eval_reason`` why not.
        """
        result = self._final_eval
        report: dict[str, Any] = {
            "best_selection_hard": self._best_score if self._best_candidate_id else None,
            "test_eval": result.get("status") or "not_run",
        }
        if result.get("reason"):
            report["test_eval_reason"] = result["reason"]
        for key in ("final_selection_hard", "baseline_test_hard", "test_hard",
                    "final_test_hard", "test_delta_hard", "final_test_delta_hard"):
            report[key] = result.get(key)
        return report

    # ── early stop: arm A's trainer patch (skillopt_local.diff, trainer.py) ──
    #
    # Arm A's SkillOpt carries a local early-stop patch (refactory-dsh#493): at
    # the end of EVERY epoch -- one that skipped for want of edits included --
    # an epoch whose best selection score did not beat the best seen so far
    # counts as stale; the run ends once ``early_stop_at_full_val`` sees a full
    # best score, or ``early_stop_patience`` stale epochs in a row. The best is
    # the score at the start of training until an epoch improves it.

    def _record_epoch_calls(self, run: Run, epoch: int) -> bool:
        """The epoch's optimizer calls and failures, in the trace; an epoch whose
        every call failed is flagged and logged, as the run report flags a run
        whose every proposal round failed (Loop._proposer_health).

        Returns ``True`` when every call failed, so the caller can stop the run.
        """
        calls_before, failed_before = getattr(self, "_epoch_calls_mark", (0, 0))
        calls = self._trace.optimizer_calls - calls_before
        failures = self._trace.optimizer_failures[failed_before:]
        all_failed = bool(calls) and len(failures) >= calls
        self._trace.record_decision(
            "optimizer_calls", calls=calls, failed=len(failures),
            failures=failures[:5], all_failed=all_failed,
        )
        if all_failed:
            run.log(
                f"skillopt epoch {epoch}: every one of its {calls} optimizer call(s) "
                f"failed ({failures[0]}); the epoch proposed nothing"
            )
            self._run_error = (
                f"all {calls} optimizer call(s) in epoch {epoch} failed: "
                f"{failures[0]}"
            )
            run.stop(self._run_error)
        return all_failed

    def _count_stale_epoch(self) -> None:
        if not (self.early_stop_patience or self.early_stop_at_full_val):
            return
        if self._best_score > self._early_stop_best + 1e-9:
            self._early_stop_best = self._best_score
            self._stale_epochs = 0
        else:
            self._stale_epochs += 1

    def _early_stop(self, run: Run) -> bool:
        if self.early_stop_at_full_val and self._best_score >= 1.0 - 1e-9:
            self._trace.record_decision("early_stop", reason="perfect_val")
            run.stop("perfect_val")
            return True
        if self.early_stop_patience > 0 and self._stale_epochs >= self.early_stop_patience:
            self._trace.record_decision("early_stop", reason="patience")
            run.stop("patience_exhausted")
            return True
        return False

    def _merge_order(self, epoch: int, kind: str, patches: list[dict]) -> list[dict]:
        """The epoch's ``kind`` patches in the order the merge receives them.

        The analysts' group order unless ``patch_order`` is set (see
        ``configure``); then its answer, taken only as a permutation of these
        same patch objects.
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

        # Evaluate prev skill on sample. SkillOpt rolls out both skills on the
        # sample whatever they are (trainer.py:1866-1867). A skill the search
        # already holds cannot be proposed again (E-noop for the current one,
        # E-duplicate for one evaluated earlier, e.g. the skill the previous
        # epoch ended on), so roll out the candidate that already carries it.
        prev_candidate = self._known_candidate(run, current, self._prev_epoch_skill)
        if prev_candidate is None:
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
        if prev_candidate.id == current.id:
            # The epoch's candidate was rejected, so prev and current are one
            # candidate. Train records are kept per candidate, so the second
            # rollout would overwrite the first: run them one after the other
            # and read each rollout's records before the next starts
            # (evaluate_many refuses two train runs of one candidate).
            prev_candidate = run.evaluate(
                prev_candidate, part="train", items=sample_ids, seed=slow_seed,
            )
            prev_records = run.item_records(prev_candidate)
            curr_candidate = run.evaluate(
                current, part="train", items=sample_ids, seed=slow_seed,
            )
            curr_records = run.item_records(curr_candidate)
        else:
            # Distinct candidates: roll both out concurrently.
            slow_requests = [
                EvalRequest(candidate=prev_candidate, part="train", items=sample_ids, seed=slow_seed),
                EvalRequest(candidate=current, part="train", items=sample_ids, seed=slow_seed),
            ]
            prev_candidate, curr_candidate = run.evaluate_many(slow_requests)
            prev_records = run.item_records(prev_candidate)
            curr_records = run.item_records(curr_candidate)

        # Build comparison pairs — slow_update.py:206-269
        pairs = self._build_comparison_pairs(
            prev_records, curr_records, sample_ids,
            env=self._env,
            prev_knobs=_knob_values(prev_candidate),
            curr_knobs=_knob_values(curr_candidate),
        )
        self._comparison_pairs = pairs
        comparison_text = prompts.format_comparison_text(pairs)

        # Previous guidance from the current skill — trainer.py:1916
        prev_guidance = prompts.extract_slow_update_field(self._current_skill)

        call_key = ("slow_update", epoch, 0, 0)
        user = prompts.format_slow_update_user(
            self._prev_epoch_skill, self._current_skill, prev_guidance, comparison_text,
        )
        output = ask(
            self.model, self._trace, "slow_update", user,
            system=prompts.SLOW_UPDATE_SYSTEM, call_key=call_key,
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

        # SkillOpt's slow_result.json for a guidance it got (trainer.py:1936-1947,
        # slow_update.py:434-439): the reasoning and content, both stripped, the
        # two rollouts' hard scores (compute_score, utils/scoring.py:7-23) and the
        # candidate's short hash (utils/scoring.py:26-28). The replay compares them.
        self._trace.record_decision(
            "slow_update",
            action="force_accept" if not isinstance(outcome, list) else "refused",
            guidance_length=len(guidance),
            n_pairs=len(pairs),
            reasoning=str(result.get("reasoning", "")).strip(),
            slow_update_content=guidance,
            prev_hard=_hard_score(prev_records),
            curr_hard=_hard_score(curr_records),
            candidate_hash=hashlib.sha256(new_skill.encode()).hexdigest()[:16],
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
            # SkillOpt still writes meta_skill/epoch_NN/meta_skill_result.json with
            # ``action: skip_first_epoch`` for the first epoch (trainer.py:2103),
            # so record the skip rather than leaving the epoch silent in the trace.
            self._trace.record_decision(
                "meta_skill",
                action="skip_first_epoch" if epoch < 2 else "skip_no_prev_skill",
            )
            return

        comparison_text = prompts.format_comparison_text(self._comparison_pairs)
        call_key = ("meta_skill", epoch, 0, 0)
        user = prompts.format_meta_skill_user(
            self._prev_epoch_skill,
            epoch_last_step_skill,
            self._meta_skill_text,
            comparison_text,
        )
        output = ask(
            self.model, self._trace, "meta_skill", user,
            system=prompts.META_SKILL_SYSTEM, call_key=call_key,
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
            # meta_skill_result.json's own fields (meta_skill.py:70-75), compared
            # by the replay; SkillOpt writes them only for a skill it got.
            **({
                "reasoning": str(result.get("reasoning", "")).strip(),
                "meta_skill_content": content,
            } if content else {}),
        )
        run.save_strategy_state()

    # ── state (overrides OuterLoop at outerloop.py:384-407) ────────────────

    def strategy_state_dict(self) -> dict[str, Any]:
        return state.serialize_state({
            "current_skill": self._current_skill,
            "current_score": self._current_score,
            "best_skill": self._best_skill,
            "best_score": self._best_score,
            "best_candidate_id": self._best_candidate_id,
            "run_error": self._run_error,
            "prev_epoch_skill": self._prev_epoch_skill,
            "epoch_last_step_skill": self._epoch_last_step_skill,
            "patched_skill": self._patched_skill,
            "val_cache": dict(self._val_cache),
            "edit_budget_step": self._edit_budget_step,
            "meta_skill_text": self._meta_skill_text,
            "comparison_pairs": self._comparison_pairs,
            "epoch": self._epoch,
            "stale_epochs": self._stale_epochs,
            "early_stop_best": self._early_stop_best,
            "phase": self._phase,
            "completed_phases": sorted(self._completed_phases),
            # Cached intermediate outputs for mid-epoch resume.
            "cached_failure_patches": self._cached_failure_patches,
            "cached_success_patches": self._cached_success_patches,
            "cached_merged": self._cached_merged,
            "cached_ranked": self._cached_ranked,
            "train_stop": self._train_stop,
            "final_eval": dict(self._final_eval),
        })

    def load_strategy_state(self, data: dict[str, Any]) -> None:
        restored = state.deserialize_state(data)
        self._current_skill = restored.get("current_skill", "")
        self._current_score = restored.get("current_score", 0.0)
        self._best_skill = restored.get("best_skill", "")
        self._best_score = restored.get("best_score", 0.0)
        self._best_candidate_id = restored.get("best_candidate_id")
        self._run_error = restored.get("run_error", "")
        self._prev_epoch_skill = restored.get("prev_epoch_skill")
        self._epoch_last_step_skill = restored.get("epoch_last_step_skill", "")
        self._patched_skill = restored.get("patched_skill", "")
        self._val_cache = restored.get("val_cache", {})
        self._edit_budget_step = restored.get("edit_budget_step", 0)
        self._meta_skill_text = restored.get("meta_skill_text", "")
        self._comparison_pairs = restored.get("comparison_pairs", [])
        self._epoch = restored.get("epoch", 0)
        self._stale_epochs = restored.get("stale_epochs", 0)
        self._early_stop_best = restored.get("early_stop_best", self._best_score)
        self._phase = restored.get("phase", "")
        self._completed_phases = set(restored.get("completed_phases", []))
        # Backward compat: if completed_phases is empty and _phase is a final
        # phase, populate the set from all phases up to and including _phase.
        if not self._completed_phases and self._phase in self._FINAL_PHASES:
            idx = self._FINAL_PHASES.index(self._phase)
            self._completed_phases = set(self._FINAL_PHASES[:idx + 1])
        # Cached intermediate outputs for mid-epoch resume.
        self._cached_failure_patches = restored.get("cached_failure_patches", [])
        self._cached_success_patches = restored.get("cached_success_patches", [])
        self._cached_merged = restored.get("cached_merged", {})
        self._cached_ranked = restored.get("cached_ranked", [])
        self._train_stop = restored.get("train_stop", "")
        self._final_eval = dict(restored.get("final_eval") or {})

    def run_error(self) -> str | None:
        return self._run_error or None

    def best_candidate(self, population: "list[Candidate]") -> "Candidate | None":
        """Return the val-gated best candidate by id from the population.

        If the strategy has a best id that is not in the population, raise
        ValueError so the caller reports exit_ok=False with a clear message.
        Never falls back silently to the leaderboard: a best SkillOpt recorded
        without its id (state saved before the id was kept) is an error too.
        ``None`` only when no val score was ever recorded (the seed's initial
        val never ran), so there is no SkillOpt best to report.
        """
        if not self._best_candidate_id:
            if self._best_skill or self._val_cache:
                raise ValueError(
                    "skillopt recorded a best skill (val score "
                    f"{self._best_score}) but not its candidate id, so its winner "
                    "cannot be named; the strategy state predates best_candidate_id"
                )
            return None
        for candidate in population:
            if candidate.id == self._best_candidate_id:
                return candidate
        raise ValueError(
            f"strategy's best candidate {self._best_candidate_id!r} "
            f"is not in the population of {len(population)} candidates"
        )

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _get_skill(candidate: Candidate) -> str:
        """Extract the ``skill`` knob value from a candidate's state."""
        knobs = candidate.state.get("knobs", {})
        skill_entry = knobs.get("skill", {})
        return str(skill_entry.get("value", ""))

    @classmethod
    def _known_candidate(cls, run: Run, current: Candidate, skill: str) -> Candidate | None:
        """The candidate in the search that carries ``skill``, if there is one.

        ``current`` first, then the seed, then the evaluated history: a SkillOpt
        search edits only the ``skill`` knob, so the seed with that skill set
        has the same state id as the candidate that holds it.
        """
        if cls._get_skill(current) == skill:
            return current
        seed = run.seed()
        if cls._get_skill(seed) == skill:
            return seed
        built = Candidate.from_state(
            nn_edits.configure(seed.state, "skill", skill), parent=seed.id,
        )
        for candidate in run.history():
            if candidate.id == built.id:
                return candidate
        return None

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


def batch_seed(seed: int, epoch: int, batch_idx: int = 0) -> int:
    """SkillOpt's train-batch seed for LiveMath: ``seed + epoch*1000 + batch_idx + 1``.

    LiveMath has a dataloader, so the trainer takes each batch's own seed from
    ``plan_train_epoch`` (trainer.py:1170-1177, :1242; dataloader.py:278) and
    hands it to the analysts as ``random_seed`` (trainer.py:1283). The
    ``Random(seed + epoch*1000)`` shuffle of base seeds (datasets/base.py:98-109)
    is the dataloader-less path and does not apply.
    """
    return seed + epoch * 1000 + batch_idx + 1


def _hard_score(records: list[dict]) -> float:
    """SkillOpt's ``compute_score`` hard half over a rollout (utils/scoring.py:7-23)."""
    if not records:
        return 0.0
    return sum(1.0 if record.get("passed") else 0.0 for record in records) / len(records)


def _only_noop(reasons: list[str]) -> bool:
    """Whether every refusal reason is ``E-noop`` (the edit changed nothing)."""
    return bool(reasons) and all(str(reason).startswith("E-noop") for reason in reasons)


def _knob_values(candidate: Candidate) -> dict[str, Any]:
    """The candidate's knob values by name, as its rollout used them."""
    knobs = candidate.state.get("knobs", {}) or {}
    return {
        name: entry.get("value") if isinstance(entry, dict) else entry
        for name, entry in knobs.items()
    }


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
    """Convert SkillOpt's ``"op"`` key to text_patch's ``"kind"`` key.

    SkillOpt prompts produce edits with ``"op": "append"``; refactory's
    text_patch reads ``"kind": "append"``.
    """
    out = dict(edit)
    if "op" in out and "kind" not in out:
        out["kind"] = out.pop("op")
    return out
