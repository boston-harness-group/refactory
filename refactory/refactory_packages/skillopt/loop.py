"""SkillOptLoop — SkillOpt as an OuterLoop strategy on refactory's public extension points.

Subclasses ``OuterLoop`` and overrides only names in ``OuterLoop.overridable``:
``configure``, ``strategy``, ``strategy_state_dict``, ``load_strategy_state``,
``best_candidate``, ``run_error`` and ``report``. The loop machinery (``Loop``)
constructs it, calls ``configure(model=…, **strategy_options)``, and passes a
``Run`` to ``strategy(run)``; the strategy works **only** through ``Run``'s public
API.

It is built from three shared parts, and keeps only SkillOpt's control flow (the
epoch, its resume phases, the early stop):

- ``SkillOptProposer`` (``proposer.py``): the analysts, the merge, the ranked edit
  budget, the patch step, the slow update and the meta skill, as typed edits to the
  open text knob(s) it is given (the ``knob`` option, ``skill`` by default, or the
  ``knobs`` option: several knobs SkillOpt sees as one sectioned document);
- ``outerloop.ValidationGate``: the val gate (strict improvement over the current
  candidate, val scores cached per whole candidate) and the best on val;
- ``outerloop.FinalEvaluation``: the post-training block.

Every candidate it builds goes through ``run.attempt``; a candidate it saved by id
comes back from ``run.candidate`` after a resume.

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
  - claude_backend_wrapper = False (not a SkillOpt setting: True frames every
                optimizer call's user message as SkillOpt's Claude backend does,
                claude_backend.py; SkillOpt's other backends send it bare)
  - eval_test = on when the parts manifest has a test part (_base_/default.yaml:111
                ``eval_test: true``; trainer.py:2241). After training: the final
                skill's val check and promotion, then the initial, best and final
                skills on the test part with the run seed (trainer.py:2241-2441),
                reported as A's summary.json names them (trainer.py:2480-2515).
                ``eval_test=false`` skips it, and the report says so.
  - knob = "skill"          (not a SkillOpt setting: the text knob it optimizes)
  - knobs = (knob,)         (not a SkillOpt setting: several open text knobs, as a
                             comma-separated list or a list, that SkillOpt sees as
                             one document of sections -- see ``proposer.py``)
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Any

from refactory_dsh.budgets import RESERVE_RETAINED
from refactory_dsh.optim import Candidate
from refactory_dsh.outerloop import (
    BudgetStopped, EvalRequest, FinalEvaluation, OuterLoop, OuterLoopError, Run, ValidationGate,
)

from refactory_packages.skillopt import state
from refactory_packages.skillopt.claude_backend import ClaudeBackendFraming
from refactory_packages.skillopt.env import SkillOptEnv
from refactory_packages.skillopt.proposer import (
    SkillOptProposer, only_noop, hard_score, knob_values,
)
from refactory_packages.skillopt.trace import TraceRecorder

#: The report's figures under A's summary.json names (trainer.py:2480-2515), and
#: the ``FinalEvaluation`` result each comes from.
A_SUMMARY_NAMES = (
    ("final_selection_hard", "final_selection"),
    ("baseline_test_hard", "baseline_test"),
    ("test_hard", "best_test"),
    ("final_test_hard", "final_test"),
    ("test_delta_hard", "best_test_delta"),
    ("final_test_delta_hard", "final_test_delta"),
)


class SkillOptLoop(OuterLoop):
    """SkillOpt outer-loop strategy — a pure custom strategy (see the module docstring).

    Every model call goes through the injectable client in ``self.model`` (set by
    ``configure()``), with a stable call key recorded in the trace so the F20
    replay can feed recorded outputs back.
    """

    name = "skillopt"
    requires_model = True
    #: The patch step is a ``set_knob`` with a ``patch``; the slow update is a
    #: ``set_knob`` with a ``value``.
    required_ops = ("set_knob",)
    #: The strategy options ``configure`` reads; any other is refused
    #: (``strategies.check_options``), so a misspelled one is not silently ignored.
    accepted_options = (
        "analyst_edit_budget", "claude_backend_wrapper", "early_stop_at_full_val",
        "early_stop_patience", "edit_budget_max", "edit_budget_min", "eval_test",
        "group_size", "knob", "knobs", "max_epochs", "merge_batch_size", "patch_order",
        "rollout_order", "seed", "slow_update_samples",
    )

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
        # command line or a file (strings, numbers) cannot set it, and the
        # proposer refuses anything but a permutation.
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
        # Frame every optimizer call's user message as arm A's Claude backend does
        # (``claude_backend.claude_backend_prompt``). Off by default: SkillOpt frames
        # only on that backend, so a mode or backend compared with anything else
        # keeps the bare message.
        wrapper = options.get("claude_backend_wrapper", False)
        if not isinstance(wrapper, bool):
            raise TypeError(
                f"skillopt: claude_backend_wrapper must be true or false, not {wrapper!r}"
            )
        self.claude_backend_wrapper = wrapper
        knob = options.get("knob", "skill")
        if not isinstance(knob, str) or not knob:
            raise TypeError(f"skillopt: knob must name a text knob, not {knob!r}")
        knobs = _knob_list(options.get("knobs"))
        if knobs is not None and "knob" in options and knobs != (knob,):
            raise TypeError(
                f"skillopt: give the knob option or the knobs option, not both "
                f"(knob={knob!r}, knobs={list(knobs)!r})"
            )

        # The three shared parts (see the module docstring).
        self.proposer = SkillOptProposer(
            knob=knob,
            knobs=knobs,
            client=self._optimizer_client,
            seed=self.base_seed,
            max_epochs=self.max_epochs,
            group_size=self.group_size,
            merge_batch_size=self.merge_batch_size,
            analyst_edit_budget=self.analyst_edit_budget,
            edit_budget_max=self.edit_budget_max,
            edit_budget_min=self.edit_budget_min,
            slow_update_samples=self.slow_update_samples,
            rollout_order=rollout_order,
            patch_order=patch_order,
        )
        # SkillOpt evaluates the selection set with the base seed
        # (trainer.py:1133-1136; dataloader.py:291-301 shuffles with it).
        self.gate = ValidationGate(seed=self.base_seed)
        self.final = FinalEvaluation(label="skillopt")

        # Why the run stopped as a failure (an epoch whose every optimizer call
        # failed), for the report; empty while the run is healthy.
        self._run_error: str = ""
        # Candidates by id (``Candidate.id``, the whole state): the previous
        # epoch's last candidate, this epoch's last-step candidate (taken before
        # its slow update, trainer.py:2176) and the candidate this epoch's patch
        # produced (so a run resumed before the val gate gates it).
        self._prev_epoch_id: str | None = None
        self._epoch_last_step_id: str | None = None
        self._patched_id: str | None = None
        self._epoch: int = 0
        self._stale_epochs: int = 0
        # The best score the early stop measures improvement against.
        self._early_stop_best: float = 0.0
        self._trace: TraceRecorder | None = None
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

    def _optimizer_client(self) -> Any:
        """The client every optimizer call goes through: ``self.model``, framed as arm
        A's Claude backend frames it when ``claude_backend_wrapper`` is on."""
        if self.claude_backend_wrapper and self.model is not None:
            return ClaudeBackendFraming(self.model)
        return self.model

    # ── strategy (overrides OuterLoop.strategy) ─────────────────────────────

    # Phase names for per-phase resume markers (ask 3).
    _PHASES = (
        "", "initial_val", "train", "analysts", "merge", "rank", "patch",
        "val_gate", "slow_update", "meta_skill", "epoch_done",
    )
    #: The post-training evaluation's phases, in A's order (trainer.py:2241-2441).
    _FINAL_PHASES = FinalEvaluation.PHASES
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
        self.proposer.trace = self._trace
        self.final.record = self._trace.record_decision
        # SkillOpt's env is the adapter the run's env names (envs/base.py:288-296
        # derives it from the adapter's module path); here it is the package that
        # holds the run's mode -- its prompt overrides and rollout rendering.
        self.proposer.env = SkillOptEnv.for_mode(str(run.state().get("mode", "")))
        # A run that cannot do what it was told is refused before it spends.
        self.proposer.check_knobs(run.seed().state)
        if self.eval_test and not run.has_part("test"):
            raise ValueError(
                "skillopt: eval_test=true needs a parts manifest with a 'test' part; "
                "give the run one, or set eval_test=false"
            )

        if self._phase in self._FINAL_PHASES:
            # Resumed after training: stop for the reason training ended on.
            run.stop(self._train_stop)
        else:
            try:
                self._train(run)
            except BudgetStopped as exc:
                # A time or token budget stopped an evaluation: training ends on it.
                run.stop(exc.reason)
            self._train_stop = run.stopped or ""
        # Stopped at the validation reserve's line: the reserve is kept for this.
        reserved = str(run.stopped or "").endswith(RESERVE_RETAINED)
        if (run.stopped in self._TRAINED or reserved) and not self._run_error:
            self._final_evaluation(run)
        else:
            self.final.result = {
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
        # Continue from the candidate the interrupted run held, never the seed's.
        current = (_held(run, self.gate.current_id) if resumed and self.gate.current_id
                   else run.seed())

        # ① Initial val score — mechanism 1 (skip on resume if progressed)
        if not self._past_phase(resume_epoch, resume_phase, 0, "initial_val"):
            self._phase = "initial_val"
            current = run.seed()
            score = self.gate.start(run, current)
            self._early_stop_best = self.gate.best_score
            run.save_strategy_state()

            self._trace.record_decision(
                "initial_val", score=score, skill_hash=self._text_hash(current),
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

            # ② Train rollout — mechanism 2, with the batch seed (dataloader.py:278).
            epoch_seed = self.proposer.batch_seed(epoch)
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
                failure_patches, success_patches = self.proposer.reflect(epoch, records, current)
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
                if self._end_epoch(run, epoch):
                    return
                continue

            # ④ Merge — mechanism 4
            if not self._past_phase(resume_epoch, resume_phase, epoch, "merge"):
                self._phase = "merge"
                merged = self.proposer.merge(
                    epoch, self.proposer.text(current), failure_patches, success_patches,
                )
                self._cached_merged = merged
                run.save_strategy_state()
            else:
                # Resume past merge: restore from cached state (no model calls)
                merged = self._cached_merged

            # Rank
            if not self._past_phase(resume_epoch, resume_phase, epoch, "rank"):
                self._phase = "rank"
                ranked = self.proposer.rank(epoch, self.proposer.text(current), merged,
                                            parent=current)
                self._cached_ranked = ranked
                run.save_strategy_state()
            else:
                # Resume past rank: restore from cached state (no model calls,
                # no second step of the edit-budget schedule).
                ranked = self._cached_ranked

            # ⑤ Patch — mechanism 5
            if not self._past_phase(resume_epoch, resume_phase, epoch, "patch"):
                self._phase = "patch"
                outcome, edit_report = self.proposer.attempt_patch(
                    run, ranked, current, f"skillopt epoch {epoch} patch",
                )
                if isinstance(outcome, list) and only_noop(outcome):
                    # The edits changed nothing: SkillOpt still gates the
                    # unchanged skill (trainer.py:1507, :1514), its selection
                    # cache answers with the score it holds for it
                    # (:1567-1572), and the gate does not accept an equal score.
                    # ``edit_report`` is its per-edit outcomes, as SkillOpt writes
                    # edit_apply_report.json for it (trainer.py:1510-1512).
                    # Observability only.
                    candidate = current
                    self._trace.record_decision(
                        "patch_noop", reasons=outcome, n_edits=len(ranked),
                        edit_report=edit_report,
                    )
                elif isinstance(outcome, list):
                    self._trace.record_decision(
                        "patch_refused", reasons=outcome,
                    )
                    run.save_strategy_state()
                    current = self._finish_epoch(
                        run, current, epoch, train_ids, resume_epoch, resume_phase,
                    )
                    if self._end_epoch(run, epoch):
                        return
                    continue
                else:
                    candidate = outcome
                    self._trace.record_decision(
                        "patch_applied",
                        edit_report=edit_report,
                        n_edits=len(ranked),
                    )
                self._patched_id = candidate.id
                run.save_strategy_state()
            else:
                candidate = None  # resolved below if the val gate is still to run

            # ⑦ Val gate — mechanism 7
            if not self._past_phase(resume_epoch, resume_phase, epoch, "val_gate"):
                self._phase = "val_gate"
                if candidate is None:
                    candidate = self._patched_candidate(run, current, epoch, ranked)
                verdict = self.gate.judge(run, candidate)
                self._trace.record_decision(
                    "val_cache_hit" if verdict.cached else "val_evaluate",
                    skill_hash=self._text_hash(candidate), score=verdict.score,
                )
                if verdict.accepted:
                    current = run.keep(candidate)
                    self._trace.record_decision(
                        "val_gate_accept",
                        cand_score=verdict.score,
                        current_score=verdict.incumbent,
                    )
                else:
                    self._trace.record_decision(
                        "val_gate_reject",
                        cand_score=verdict.score,
                        current_score=self.gate.current_score,
                    )

                run.save_strategy_state()

            # ⑧⑨ Slow update & meta skill — mechanisms 8, 9
            current = self._finish_epoch(
                run, current, epoch, train_ids, resume_epoch, resume_phase,
            )
            if self._end_epoch(run, epoch):
                return

            # Clear resume markers after the resume epoch so subsequent
            # epochs run normally.
            if epoch == resume_epoch:
                resume_epoch = 0
                resume_phase = ""

        run.stop("max_epochs")

    def _end_epoch(self, run: Run, epoch: int) -> bool:
        """Close the epoch: count a stale epoch, record its optimizer calls, mark it
        done, and check the early stop (⑩). ``True`` when the run stops here."""
        self._count_stale_epoch()
        all_failed = self._record_epoch_calls(run, epoch)
        self._phase = "epoch_done"
        run.save_strategy_state()
        return all_failed or self._early_stop(run)

    def _patched_candidate(self, run: Run, current: Candidate, epoch: int, ranked: list[dict]) -> Candidate:
        """The candidate the saved "patch" phase recorded, for a run resumed past it.

        The current candidate when the patch changed nothing; else the held one,
        or the same patch applied again through ``run.attempt`` (the same parent
        and edits give the same state, so the same id).
        """
        if self._patched_id is None or self._patched_id == current.id:
            return current
        held = run.candidate(self._patched_id)
        if held is not None:
            return held
        outcome, _ = self.proposer.attempt_patch(
            run, ranked, current, f"skillopt epoch {epoch} patch",
        )
        if isinstance(outcome, list) or outcome.id != self._patched_id:
            raise OuterLoopError(
                f"skillopt: cannot rebuild the epoch-{epoch} patched candidate "
                f"{self._patched_id!r} on resume ({outcome if isinstance(outcome, list) else outcome.id})"
            )
        return outcome

    # ── after training: trainer.py:2241-2441 ──────────────────────────────

    def _final_evaluation(self, run: Run) -> None:
        """SkillOpt's evaluation after training, in its order (trainer.py:2241-2441),
        through ``outerloop.FinalEvaluation``: the final skill's val check
        (:2244-2302), then the initial (:2304-2337), best (:2339-2372) and final
        (:2374-2441) skills on test with the run seed, as A's
        ``_build_eval_env(split=..., seed=seed)`` does."""
        skip = self._final_eval_skip(run)
        if skip:
            self.final.skip(run, skip, run.save_strategy_state)
            return

        def save() -> None:
            # The furthest completed phase in A's order, whatever order the
            # concurrent test phases finish in.
            if self.final.completed:
                self._phase = self.final.phase
            run.save_strategy_state()

        self.final.run(run, self.gate, seed=self.base_seed, save=save)

    def _final_eval_skip(self, run: Run) -> str:
        """Why the post-training evaluation does not run; empty when it does."""
        if self.eval_test is False:
            return "the eval_test strategy option is false"
        if not run.has_part("test"):
            return "the parts manifest has no 'test' part"
        return ""

    def report(self, run: Run) -> dict[str, Any]:
        """The strategy's own figures, under A's summary.json names (trainer.py:2480-2515).

        ``test_eval`` says whether the post-training evaluation ran (``done``), was
        ``skipped`` or stopped ``incomplete``, and ``test_eval_reason`` why not.
        """
        result = self.final.result
        report: dict[str, Any] = {
            "best_selection_hard": self.gate.best_score if self.gate.best_id else None,
            "test_eval": result.get("status") or "not_run",
        }
        if result.get("reason"):
            report["test_eval_reason"] = result["reason"]
        for name, key in A_SUMMARY_NAMES:
            report[name] = result.get(key)
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
        if self.gate.best_score > self._early_stop_best + 1e-9:
            self._early_stop_best = self.gate.best_score
            self._stale_epochs = 0
        else:
            self._stale_epochs += 1

    def _early_stop(self, run: Run) -> bool:
        if self.early_stop_at_full_val and self.gate.best_score >= 1.0 - 1e-9:
            self._trace.record_decision("early_stop", reason="perfect_val")
            run.stop("perfect_val")
            return True
        if self.early_stop_patience > 0 and self._stale_epochs >= self.early_stop_patience:
            self._trace.record_decision("early_stop", reason="patience")
            run.stop("patience_exhausted")
            return True
        return False

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
        """Slow update, then meta skill, then remember this epoch's candidate.

        Each sub-step saves its own phase, so a run resumed after the slow update
        makes the meta skill rather than skipping it. The meta skill compares
        against the epoch's last-step skill, taken before this epoch's slow
        update rewrites it (trainer.py:1763, :2176).
        """
        if not self._past_phase(resume_epoch, resume_phase, epoch, "slow_update"):
            self._epoch_last_step_id = current.id
            self._phase = "slow_update"
            current = self._do_slow_update(run, current, epoch, train_ids)
            run.save_strategy_state()
        if not self._past_phase(resume_epoch, resume_phase, epoch, "meta_skill"):
            self._phase = "meta_skill"
            self._do_meta_skill(run, epoch)
            # The previous epoch's skill for the next epoch is the saved skill of
            # this epoch's last step, which already carries the slow update
            # (trainer.py:1838-1839 load it; :2059 saved it after the update).
            self._prev_epoch_id = current.id
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
            outcome = run.attempt(self.proposer.placeholder(current), current)
            if not isinstance(outcome, list):
                current = run.keep(outcome)
                self.gate.carry(current)
                self._trace.record_decision("slow_update_placeholder")
            run.save_strategy_state()
            return current

        if self._prev_epoch_id is None:
            return current
        self.proposer.comparison_pairs = []

        slow_seed, sample_ids = self.proposer.slow_update_sample(train_ids, epoch)
        if not sample_ids:
            warnings.warn(
                f"skillopt: slow update skipped in epoch {epoch} — no train items to sample "
                f"(slow_update_samples={self.slow_update_samples}, "
                f"train ids={len(train_ids)})",
                RuntimeWarning,
                stacklevel=2,
            )
            self._trace.record_decision("slow_update_skipped", reason="no_samples")
            return current

        self._trace.record_decision(
            "slow_update_sample", sample_ids=sample_ids,
        )

        # SkillOpt rolls out both skills on the sample whatever they are
        # (trainer.py:1866-1867): the previous epoch's candidate is held by id.
        prev_candidate = current if self._prev_epoch_id == current.id else _held(run, self._prev_epoch_id)
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
        pairs = self.proposer.comparison(
            (prev_records, knob_values(prev_candidate)),
            (curr_records, knob_values(curr_candidate)),
            sample_ids,
        )
        current_text = self.proposer.text(current)
        guidance, result = self.proposer.slow_update(
            epoch, self.proposer.text(prev_candidate), current_text,
        )
        if not guidance:
            # No guidance: the skill is unchanged — trainer.py:2040-2047
            self._trace.record_decision("slow_update", action="no_content", n_pairs=len(pairs))
            run.save_strategy_state()
            return current

        # Force-accept (trainer.py:2042-2055) at the pre-update score, which is
        # cached for the new candidate (:2052-2053). Not via a patch:
        # SKILLOPT_PATCH_CONFIG honours the protected region, which blocks it.
        proposal, new_text = self.proposer.slow_update_proposal(guidance, current)
        outcome = run.attempt(proposal, current)
        if not isinstance(outcome, list):
            current = run.keep(outcome)
            self.gate.carry(current, remember=True)

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
            prev_hard=hard_score(prev_records),
            curr_hard=hard_score(curr_records),
            candidate_hash=hashlib.sha256(new_text.encode()).hexdigest()[:16],
        )
        run.save_strategy_state()
        return current

    def _do_meta_skill(self, run: Run, epoch: int) -> None:
        """Meta skill — mechanism 9, from epoch 2 (trainer.py:2083-2198): the
        previous epoch's skill against this epoch's last-step skill."""
        if epoch < 2 or self._prev_epoch_id is None:
            # SkillOpt still writes meta_skill/epoch_NN/meta_skill_result.json with
            # ``action: skip_first_epoch`` for the first epoch (trainer.py:2103),
            # so record the skip rather than leaving the epoch silent in the trace.
            self._trace.record_decision(
                "meta_skill",
                action="skip_first_epoch" if epoch < 2 else "skip_no_prev_skill",
            )
            return
        self.proposer.meta_skill_step(
            epoch,
            self.proposer.text(_held(run, self._prev_epoch_id)),
            self.proposer.text(_held(run, self._epoch_last_step_id)),
        )
        run.save_strategy_state()

    # ── state ──────────────────────────────────────────────────────────────

    def strategy_state_dict(self) -> dict[str, Any]:
        return state.serialize_state({
            **self.gate.state_dict(),
            **self.proposer.state_dict(),
            **self.final.state_dict(),
            "run_error": self._run_error,
            "prev_epoch_candidate_id": self._prev_epoch_id,
            "epoch_last_step_candidate_id": self._epoch_last_step_id,
            "patched_candidate_id": self._patched_id,
            "epoch": self._epoch,
            "stale_epochs": self._stale_epochs,
            "early_stop_best": self._early_stop_best,
            "phase": self._phase,
            # Cached intermediate outputs for mid-epoch resume.
            "cached_failure_patches": self._cached_failure_patches,
            "cached_success_patches": self._cached_success_patches,
            "cached_merged": self._cached_merged,
            "cached_ranked": self._cached_ranked,
            "train_stop": self._train_stop,
        })

    def load_strategy_state(self, data: dict[str, Any]) -> None:
        restored = state.deserialize_state(data)
        if "current_skill" in restored and "current_candidate_id" not in restored:
            raise ValueError(
                "skillopt: this run's strategy state keys its candidates by skill text, "
                "as SkillOpt runs did before whole-candidate ids; this version cannot "
                "resume it -- start a new run"
            )
        self.gate.load_state_dict(restored)
        self.proposer.load_state_dict(restored)
        self.final.load_state_dict(restored)
        self._run_error = restored.get("run_error", "")
        self._prev_epoch_id = restored.get("prev_epoch_candidate_id")
        self._epoch_last_step_id = restored.get("epoch_last_step_candidate_id")
        self._patched_id = restored.get("patched_candidate_id")
        self._epoch = restored.get("epoch", 0)
        self._stale_epochs = restored.get("stale_epochs", 0)
        self._early_stop_best = restored.get("early_stop_best", self.gate.best_score)
        self._phase = restored.get("phase", "")
        self._cached_failure_patches = restored.get("cached_failure_patches", [])
        self._cached_success_patches = restored.get("cached_success_patches", [])
        self._cached_merged = restored.get("cached_merged", {})
        self._cached_ranked = restored.get("cached_ranked", [])
        self._train_stop = restored.get("train_stop", "")

    def run_error(self) -> str | None:
        return self._run_error or None

    def best_candidate(self, population: "list[Candidate]") -> "Candidate | None":
        """The val-gated best, by id; ``None`` when no val score was recorded.

        Raises ``ValueError`` (the report then says the run failed and why) when
        the population does not hold it: never a silent fall back to the
        leaderboard.
        """
        return self.gate.best_in(population)

    # ── helpers ────────────────────────────────────────────────────────────

    def _text_hash(self, candidate: Candidate) -> str:
        """SkillOpt's skill hash of the candidate's text, for the trace (the F20
        replay compares it with A's); the cache is keyed by candidate id."""
        return hashlib.sha256(self.proposer.text(candidate).encode()).hexdigest()


def _held(run: Run, candidate_id: str | None) -> Candidate:
    """The candidate the search holds under ``candidate_id``; an error if none."""
    candidate = run.candidate(candidate_id) if candidate_id else None
    if candidate is None:
        raise OuterLoopError(f"skillopt: the search does not hold candidate {candidate_id!r}")
    return candidate


def _knob_list(value: Any) -> tuple[str, ...] | None:
    """The ``knobs`` option as a tuple of names: a comma-separated string (the
    command line's ``--strategy-option knobs=a,b``) or a list; ``None`` when unset."""
    if value is None:
        return None
    if isinstance(value, str):
        names = [name.strip() for name in value.split(",")]
    elif isinstance(value, (list, tuple)):
        names = [name.strip() if isinstance(name, str) else name for name in value]
    else:
        raise TypeError(f"skillopt: knobs must be a comma-separated list of knob names, not {value!r}")
    if not names or not all(isinstance(name, str) and name for name in names):
        raise TypeError(f"skillopt: knobs must be a comma-separated list of knob names, not {value!r}")
    if len(set(names)) != len(names):
        raise TypeError(f"skillopt: knobs names a knob twice: {value!r}")
    return tuple(names)
