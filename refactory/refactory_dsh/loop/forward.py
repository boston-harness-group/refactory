
from __future__ import annotations

import dataclasses
import functools
import json
import math
import random
import re
import shutil
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from refactory_dsh.nn.mode import RunResult

from refactory_dsh import history as run_history
from refactory_dsh import events
from refactory_dsh import observe as progress_file
from refactory_dsh import task as task_module
from refactory_dsh.evaluators import create as create_evaluator
from refactory_dsh import registry
from refactory_dsh.nn import Knob, PackageError
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import validate_state
from refactory_dsh.nn import pipeline_view
from refactory_dsh.optim import (
    Candidate,
    Optimizer,
    Proposal,
    ProposalContext,
    Rejection,
    RandomOptimizer,
)
from refactory_dsh.optim import create as create_optimizer
from refactory_dsh.optim.base import (
    MAX_EDITS_PER_PROPOSAL, NEW_STEP_TOOLS_OFF, RunStopped, available_before, edit_level, proposal_refusal,
    refused_entries,
    repair_losses,
    summarise_reasons,
)
from refactory_dsh.optim.memory import report as memory_report
from refactory_dsh.optim.memory import report_lines as memory_report_lines
from refactory_dsh.outerloop import (
    BudgetStopped, EvalRequest, FinalEvaluation, OuterLoop, OuterLoopError, Run, ValidationGate, train_batch_seed,
)
from refactory_dsh.paths import project_for_nodes, project_root
from refactory_dsh.workers import StepRoutes

from refactory_dsh.loop.config import INCIDENT_RERUNS



class _ForwardMixin:

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, *, epoch: int | None = None) -> int:
        """Evaluate every candidate in the batch that has not run yet.

        ``epoch`` (default: the current one) picks the train seed under ``val_gate``.
        The time and token budgets are checked before each wave of candidates: once one
        is spent the rest stay unscored (the report names them) and the spent budget is
        the run's bound, rather than a measurement refused by it being recorded as a
        failure.

        The candidates are independent, so up to ``max_concurrent_evaluations`` of them
        run at once (``Run.evaluate_many``, under the run-wide item limit). A wave holds
        only candidates the one-after-another order would also have started: the
        evaluation budget and, with ``max_rollouts``, the most rollouts each can add
        decide its size, so the same candidates are scored, with the same items and
        seed. ``max_concurrent_evaluations`` 1 runs them one at a time, as before.

        Under an elapsed-time or token budget the budget is checked before each wave,
        so the last wave may finish up to one wave of evaluations past the limit (one
        at a time overshoots by one evaluation). A candidate whose evaluation the
        budget refused or cut (``BudgetStopped``) stays unevaluated, for a resume.
        """
        from refactory_dsh.budgets import BudgetExhausted

        batch = [candidate for candidate in self.population if not candidate.evaluated]
        at = self.epoch if epoch is None else epoch
        seed = self._train_seed(at)
        # Fresh evidence: the round's parent is scored on the round's sample beside its
        # children (`_round_parent`), so each child pairs with it on the same seed. Not
        # when scoring a run's last proposals (an explicit ``epoch``): no proposal reads
        # those comparisons.
        parent = self._round_parent(batch, at) if epoch is None and batch else None
        gold = self._train_gold_feedback()
        # Under ``checkpoint`` the round's one candidate is rolled out on val beside its
        # train run (`_val_beside`).
        with_val = self._val_beside(batch, at)
        index = 0
        while index < len(batch):
            if self.evaluations >= self.config.max_evaluations:
                break
            if self.config.max_rollouts is not None and self.rollouts >= self.config.max_rollouts:
                break
            if self.budget:
                try:
                    self.budget.checkpoint("work" if self._owns_resource_budget else None)
                except BudgetExhausted as exc:
                    self.bound = str(exc)
                    break
            items = self._minibatch(at)
            each = len(items) if items is not None else len(self._train_ids())
            remaining = len(batch) - index
            if parent is not None and not self._room_for_rescore(remaining, each):
                # The children keep priority: the re-score goes only where the
                # evaluation and rollout budgets hold every child of the batch and it.
                parent = None
            size = self._wave_size(min(remaining,
                                       self.config.max_evaluations - self.evaluations), each)
            wave = batch[index:index + size]
            # The children keep the slots too: the re-score joins the round's last wave
            # when that wave leaves a concurrent slot free (the live shape: two children,
            # three slots), else it runs after the children (`_rescore_after`).
            beside = None
            if (parent is not None and index + len(wave) == len(batch)
                    and len(wave) < int(self.config.max_concurrent_evaluations or 1)):
                beside, parent = parent, None
            try:
                if beside is not None:
                    # One batch: the re-score shares the children's reservation, the
                    # run-wide item gate and the stop signal (`Run.evaluate_many`).
                    start = index
                    Run(_loop=self).evaluate_many(
                        [*(EvalRequest(candidate=candidate, part=task_module.TRAIN, items=items,
                                       seed=seed, gold_feedback=gold) for candidate in wave),
                         EvalRequest(candidate=beside, part=task_module.TRAIN, items=items,
                                     seed=seed, rescore_epoch=at)],
                        on_result=lambda position, candidate: (
                            self._forwarded(candidate, start + position, len(batch), at, items)
                            if position < len(wave) else None),
                    )
                elif with_val is not None and wave == [with_val]:
                    self._train_beside_val(with_val, seed=seed, items=items, gold=gold,
                                           of=len(batch), epoch=at)
                elif len(wave) == 1:
                    self._evaluate(wave[0], seed=seed, items=items, gold_feedback=gold)
                    self._forwarded(wave[0], index, len(batch), at, items)
                else:
                    start = index
                    Run(_loop=self).evaluate_many(
                        [EvalRequest(candidate=candidate, part=task_module.TRAIN, items=items, seed=seed,
                                     gold_feedback=gold)
                         for candidate in wave],
                        on_result=lambda position, candidate: self._forwarded(
                            candidate, start + position, len(batch), at, items),
                    )
            except BudgetStopped as exc:
                self.bound = self.bound or str(exc)
                break
            index += len(wave)
            if self.budget and self.budget.snapshot().get("exhausted"):
                break
        else:
            if parent is not None:
                self._rescore_after(parent, at, seed)
        return len(batch)

    def _val_beside(self, batch: Sequence[Candidate], epoch: int) -> Candidate | None:
        """Under ``checkpoint``, the candidate ``forward`` rolls out on val beside its
        train run, as one ``Run.evaluate_many`` batch, so a round costs the longer of the
        two runs instead of both; ``None`` when there is none (the order before).

        The round's only candidate, when the gate's next decision rolls only it out on
        val whatever its train score: the seed (``ValidationGate.start``), or a round's
        one child (applied unless its train run fails). Not with several children, as
        their train scores pick the one rolled out; not when another candidate waits
        for a decision; not when its val result is already held (a resume); not under
        a time or token budget (``max_elapsed_s``, ``max_tokens``), whose spend the val
        run would share; not with a model client that writes no worker receipts (any
        but ``DshModelClient``), whose spend could not be split between the two runs
        (`_trial_usage`); and not with ``max_concurrent_evaluations`` 1 or a parts
        manifest without val. The two share the run-wide item limit
        (``max_items_in_flight``), so the round takes about the slower of the two only
        when the limit fits both. With
        ``max_rollouts``, only where neither run, nor their harness-incident reruns,
        can meet the cap in either order, so both are run exactly as one after the
        other would run them. The val rollout is the one the gate would make, with the
        gate's seed; the gate then reads it from the journal (`_journaled_val`). A
        child whose train run fails has been rolled out on val for nothing: its val
        result reaches no decision or request."""
        if (self.selection_mode() != "checkpoint" or len(batch) != 1
                or int(self.config.max_concurrent_evaluations or 1) < 2
                or task_module.VAL not in (self._parts_manifest or {})):
            return None
        model = getattr(self.evaluator, "model", None)
        if model is not None:
            from refactory_dsh.model import DshModelClient

            if not isinstance(model, DshModelClient):
                # A custom client records its usage without a worker receipt, so the
                # two runs' tokens and cost could not be told apart (`_trial_usage`).
                return None
        limits = self.budget.limits if self.budget else None
        if limits is not None and (limits.max_elapsed_s is not None
                                   or limits.max_tokens is not None):
            # A time or token budget: the val run would spend beside the train run what
            # one after the other leaves to it, and could stop a train run that would
            # have finished.
            return None
        candidate = batch[0]
        gate = self._gate()
        judged = set(self._gate_accepted) | set(self._gate_rejected)
        if any(c.evaluated and c.id not in judged for c in self.population):
            return None
        held = [s for s in (self._part_results.get(candidate.id) or {}).get(task_module.VAL)
                or () if not s.get("budget_stopped")]
        if candidate.id in gate.cache or held:
            return None
        cap = self.config.max_rollouts
        if cap is not None:
            # The minibatch's size, without sampling (and journaling) it before forward's
            # budget checks: ``_minibatch`` samples only a part larger than it.
            ids = len(self._train_ids())
            size = int(self.config.train_minibatch or 0)
            train = (len(self._minibatches[epoch]) if epoch in self._minibatches
                     else size if size and ids > size else ids)
            val = len((self._parts_manifest or {}).get(task_module.VAL) or ())
            most = ((max(train, 1) + max(val, 1)) * max(int(self.config.repeats or 1), 1)
                    * (1 + INCIDENT_RERUNS))
            if self.rollouts + most > cap:
                return None
        return candidate

    def _train_beside_val(self, candidate: Candidate, *, seed: int, items: list[str] | None,
                          gold: bool, of: int, epoch: int) -> None:
        """``candidate``'s train run and its val rollout at once (`_val_beside`): one
        ``Run.evaluate_many`` batch, under the run-wide item limit and stop signal. What
        the train run records is ``forward``'s; the val result is journaled as the
        gate's own rollout would be, and a val rollout a budget stopped is kept as the
        gate's refused rollout (``_val_stopped_beside``)."""
        val = task_module.VAL
        before = len((self._part_results.get(candidate.id) or {}).get(val) or ())
        try:
            Run(_loop=self).evaluate_many(
                [EvalRequest(candidate=candidate, part=task_module.TRAIN, items=items,
                             seed=seed, gold_feedback=gold),
                 EvalRequest(candidate=candidate, part=val, seed=self._gate().seed,
                             beside=True)],
                on_result=lambda position, done: (
                    self._forwarded(done, 0, of, epoch, items) if position == 0 else None),
            )
        finally:
            for entry in ((self._part_results.get(candidate.id) or {}).get(val) or [])[before:]:
                if entry.get("budget_stopped"):
                    self._val_stopped_beside[candidate.id] = (
                        f"the budget stopped {candidate.id}'s {val} rollout: {entry.get('error')}")

    def _rescore_after(self, parent: Candidate, epoch: int, seed: int) -> None:
        """The round's parent re-score when its children filled every concurrent slot:
        alone, after them, as a batch of its own (the item gate and stop signal of
        `Run.evaluate_many`). ``forward`` kept it only where the evaluation and rollout
        budgets hold every child and it (`_room_for_rescore`). A time or token budget the
        wave spent refuses it in `_measure` before any trial runs: the reservation comes
        back and the bound is set (``BudgetStopped``)."""
        items = self._minibatch(epoch)
        try:
            Run(_loop=self).evaluate_many([EvalRequest(
                candidate=parent, part=task_module.TRAIN, items=items, seed=seed,
                rescore_epoch=epoch)])
        except BudgetStopped as exc:
            self.bound = self.bound or str(exc)

    def _room_for_rescore(self, children: int, items_each: int) -> bool:
        """Whether the round's parent may be re-scored beside its ``children`` still to
        score: the evaluation budget holds all of them and the re-score, and with
        ``max_rollouts`` so does the rollout budget -- each child's train and, under
        ``val_gate``, val rollouts, then the re-score's train rollouts (train at its
        items times ``repeats``). The children keep priority: a re-score never takes
        the rollouts a child needs to be scored or judged."""
        if self.evaluations + children + 1 > self.config.max_evaluations:
            return False
        cap = self.config.max_rollouts
        if cap is None:
            return True
        train = max(int(items_each), 1) * max(int(self.config.repeats or 1), 1)
        val = (len(task_module.manifest_part_ids(self._parts_manifest, task_module.VAL) or ())
               if self.config.val_gate and self._parts_manifest is not None else 0)
        return self.rollouts + children * (train + val) + train <= cap

    def _wave_size(self, wanted: int, items_each: int) -> int:
        """How many of the next ``wanted`` evaluations may run at once (at least one).

        At most ``max_concurrent_evaluations``. With ``max_rollouts`` set, only as many
        as the one-after-another order would also start: each is counted at the most
        rollouts it can add (its items times ``repeats``), and the wave also fits
        ``evaluate_many``'s own rollout check; a single evaluation runs alone, as before.
        """
        size = max(1, min(wanted, int(self.config.max_concurrent_evaluations or 1)))
        cap = self.config.max_rollouts
        if cap is None:
            return size
        each = max(int(items_each), 1)
        count = 0
        while (count < size and self.rollouts + count * each * self.config.repeats < cap
               and self.rollouts + (count + 1) * each <= cap):
            count += 1
        return max(count, 1)

    def _forwarded(self, candidate: Candidate, index: int, of: int, epoch: int,
                   items: list[str] | None) -> None:
        """What ``forward`` records after one candidate's train evaluation."""
        if items is not None or self._fresh_evidence():
            # Which epoch's sample and seed its records come from, so the proposal
            # step re-scores it only when they are another epoch's.
            with self._state_lock:
                self._records_epoch[candidate.id] = epoch
                self._records_by_epoch.setdefault(candidate.id, {})[epoch] = list(
                    self._train_item_records.get(candidate.id, []))
                self._journal("train_records_epoch", id=candidate.id, epoch=epoch)
        self._progress(
            "forward",
            candidate=candidate.id,
            index=index,
            of=of,
            score=candidate.score,
            exit_ok=candidate.exit_ok,
            error=candidate.error,
            duration_s=candidate.duration_s,
        )

    def _train_seed(self, epoch: int) -> int:
        """The seed a train rollout in ``epoch`` uses.

        A Task that shuffles per (item, seed) -- LiveMath's answer choices -- meets
        that seed. Without ``val_gate`` it is the run's seed, so every candidate of
        the run meets the same shuffle; with it, SkillOpt's per-epoch train seed
        (``train_batch_seed``), as SkillOpt's train rollouts use; under ``checkpoint``
        the first epoch's for the whole run (``_epoch_seeds``), so a child and its
        parent meet the same shuffle without a re-score.
        """
        if self._epoch_seeds():
            return train_batch_seed(self.config.seed, epoch)
        if self.config.val_gate:
            return train_batch_seed(self.config.seed, 1)
        return self.config.seed

    def _epoch_seeds(self) -> bool:
        """Whether train rollouts take a seed per epoch: under ``val_gate``, but not under
        ``checkpoint``, whose fixed seed pairs every child with its parent's own run."""
        return bool(self.config.val_gate and self.selection_mode() != "checkpoint")

    def _train_ids(self) -> list[str]:
        """The train part's item ids: the manifest's, else the Task's train split."""
        manifest = self._parts_manifest
        if manifest and task_module.TRAIN in manifest:
            return sorted(str(item) for item in manifest[task_module.TRAIN])
        state = self.population[0].state if self.population else self.mode.state_dict()
        return sorted(self._expected_item_ids(state, task_module.TRAIN)[0])

    def _minibatch(self, epoch: int) -> list[str] | None:
        """The train ids every candidate of ``epoch`` is scored on, or ``None`` for the
        whole train part. Sampled once per epoch with ``train_batch_seed`` (so a resumed
        run samples the same ids) and journaled."""
        size = self.config.train_minibatch
        if not size:
            return None
        if epoch in self._minibatches:
            return self._minibatches[epoch]
        ids = self._train_ids()
        if len(ids) <= size:
            return None
        sample = sorted(random.Random(train_batch_seed(self.config.seed, epoch)).sample(ids, size))
        self._minibatches[epoch] = sample
        self._journal("train_minibatch", epoch=epoch, ids=sample)
        return sample

    def _fresh_evidence(self) -> bool:
        """Whether the edited candidate is re-scored each epoch on the whole train part:
        under ``val_gate``, the default loop's own gate, but not under ``checkpoint``
        (one train seed: a candidate's own run is its evidence)."""
        return bool(self.config.val_gate and self._custom_strategy is None
                    and self.selection_mode() != "checkpoint")

    def _refresh_edited_records(self) -> None:
        """Fresh evidence: score the edited candidate on this epoch's train sample with
        this epoch's seed when its train records come from another epoch, so the
        proposal evidence (and every reader's input) is this epoch's rollout.

        On under ``val_gate`` (the whole train part, or the
        ``train_minibatch`` sample), and with a minibatch in any case. A candidate
        created and scored this epoch already has this epoch's records and is not
        scored again; a resumed run reads the epoch from the journal, so it does not
        either. Its score is left alone: only the records the optimizer reads change.
        """
        if self.selection_mode() == "checkpoint":
            # No re-score: a candidate's own train run is its evidence, on a minibatch
            # too (comparisons then cover the items both runs scored).
            return
        batch = self._minibatch(self.epoch)
        if not batch and not self._fresh_evidence():
            return
        edited = self.strategy_decision.get("parent") or self._editing_parent
        if not edited or self._records_epoch.get(str(edited)) == self.epoch:
            return
        parent = next((c for c in self.population if c.id == edited and c.evaluated), None)
        if parent is None or (parent.id, self.epoch) in self._rescore_tried:
            return
        self._rescore(parent, self.epoch)

    def _round_parent(self, batch: Sequence[Candidate], epoch: int) -> Candidate | None:
        """The parent ``forward`` re-scores beside its children (fresh evidence): the
        candidate the last proposal step edited (else the one parent every child of the
        batch shares), when a child of the batch has it as parent, it was evaluated, and
        it has no records from ``epoch`` yet. Scoring it on the round's train sample with
        the round's seed pairs every child with its parent on the same seed, a round
        that keeps a child included; in a round that keeps none it is the re-score the
        proposal step would have made, which then finds it done."""
        if not self._fresh_evidence():
            return None
        parents = {c.parent for c in batch if c.parent}
        edited = self.strategy_decision.get("parent") or self._editing_parent
        chosen = (str(edited) if edited in parents
                  else next(iter(parents)) if len(parents) == 1 else None)
        if (chosen is None or self._records_epoch.get(chosen) == epoch
                or (chosen, epoch) in self._rescore_tried):
            return None
        return next((c for c in self.population if c.id == chosen and c.evaluated), None)

    def _rescore(self, parent: Candidate, epoch: int, *, reserved: bool = False,
                 item_gate: Any = None, item_slots: bool = False,
                 stop_event: threading.Event | None = None) -> bool:
        """Score ``parent`` on ``epoch``'s train sample with ``epoch``'s seed as fresh
        evidence: its records (the optimizer's evidence, and the ones its children are
        compared with) are replaced and kept by epoch; its id and scores stay. Counted,
        journaled and shown as a re-score. Whether it was scored.

        ``reserved``: run as a request of ``Run.evaluate_many`` (``forward``'s batch),
        which reserved its evaluation, checked its rollouts and hands over its item gate
        and stop signal; the evaluation is then not counted again here, and a re-score
        a budget stopped before any trial ran raises ``BudgetStopped`` so the
        reservation is returned."""
        self._rescore_tried.add((parent.id, epoch))
        counted = 0 if reserved else 1
        if not reserved:
            if self.config.max_rollouts is not None and self.rollouts >= self.config.max_rollouts:
                return False
            if self.evaluations >= self.config.max_evaluations:
                return False
        if self.budget and not reserved:
            # The time and token budgets, checked as every scoring path checks them: a
            # spent budget stops the re-score here, before anything runs.
            from refactory_dsh.budgets import BudgetExhausted
            try:
                self.budget.checkpoint("work" if self._owns_resource_budget else None)
            except BudgetExhausted as exc:
                self.bound = str(exc)
                return False
        batch = self._minibatch(epoch)
        workdir = self._rescore_dir(parent.id, epoch)
        # A re-score of this epoch that did not count (it failed, or the run died in
        # it) left its passes here: kept in the archive, not overwritten.
        self._archive_unfinished(parent.id, workdir)

        def prepare() -> None:
            if batch:
                task_module.make_manifest(workdir, train_ids=batch)
            elif self._parts_manifest is not None and (self.root / task_module.MANIFEST).exists():
                # The whole train part, as the candidate's first evaluation read it.
                workdir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.root / task_module.MANIFEST, workdir / task_module.MANIFEST)

        result, measurements = self._settled_measure(
            parent, workdir, task_module.TRAIN, prepare, rescore_epoch=epoch,
            seed=self._train_seed(epoch), items=batch, item_gate=item_gate,
            item_slots=item_slots, stop_event=stop_event,
            gold_feedback=self._train_gold_feedback())
        if not measurements.get("count"):
            # The measurement's own budget checkpoint stopped it before a trial ran:
            # nothing was scored, so nothing is counted and the records stay.
            if reserved:
                raise BudgetStopped(f"the re-score of {parent.id} was stopped before it ran")
            return False
        expected = len(batch) if batch else len(self._train_ids())
        scored = {str(record.get("id")) for record in result.item_records}
        if not (result.exit_ok and measurements.get("complete") and len(scored) >= expected):
            # A failed or partial re-score is no evidence: the previous records stay, the
            # epoch is not marked (a resume in this epoch tries again), and what it spent
            # is counted and journaled like any evaluation.
            with self._state_lock:
                self.evaluations += counted
                if not measurements.get("cached_from"):
                    self.rollouts += len(result.item_records) or 1
                self._journal("rescore_failed", id=parent.id, epoch=epoch,
                              seed=self._train_seed(epoch), exit_ok=result.exit_ok,
                              error=result.error, records=len(scored), expected=expected,
                              measurements=measurements)
                self._journal("rollouts", rollouts=self.rollouts)
            self._progress("rescore", candidate=parent.id, epoch=epoch, failed=True,
                           error=result.error, records=len(scored), expected=expected,
                           duration_s=result.duration_s)
            return False
        with self._state_lock:
            # An evaluation like any other: it counts toward the evaluation budget (the
            # time and token budgets are charged inside the measurement), and it is
            # journaled so a resumed run counts it too.
            self.evaluations += counted
            self._journal("rescore", id=parent.id, epoch=epoch,
                          seed=self._train_seed(epoch), items=len(batch or ()) or None,
                          score=result.score if result.exit_ok else None,
                          exit_ok=result.exit_ok, measurements=measurements)
            self._train_item_records[parent.id] = list(result.item_records)
            self._records_epoch[parent.id] = epoch
            self._records_by_epoch.setdefault(parent.id, {})[epoch] = list(result.item_records)
            if not measurements.get("cached_from"):
                self.rollouts += len(result.item_records) or 1
            self._journal("train_records", id=parent.id, records=list(result.item_records))
            self._journal("train_records_epoch", id=parent.id, epoch=epoch)
            self._journal("rollouts", rollouts=self.rollouts)
        self._progress("rescore", candidate=parent.id, epoch=epoch,
                       score=result.score if result.exit_ok else None,
                       records=len(result.item_records), duration_s=result.duration_s)
        return True

    def _carry_ledger(self, workdir: Path) -> None:
        """Put the run's ledger where the candidate's own nodes look for it.

        The run's ledger lives at the run root; a candidate is evaluated in
        ``runs/<id>/``; and the steering transform folds in the ledger of the work
        directory it is handed. Without this the ledger a person seeded with
        ``--intent``, or appended to with ``refactory steer --note`` while the search
        was going, was written, reported by ``steer``, and read by no node — in the one
        place steering is most wanted, because a search is what runs for hours.

        Copied rather than searched for up the tree, so each candidate directory stays
        a self-contained record of what the person had said when it was evaluated. A
        note added while a candidate is running reaches the next candidate, exactly as
        it reaches the next node of an interactive run.
        """
        from refactory_dsh.steering import LEDGER

        source = self.root / LEDGER
        if not source.exists():
            return
        workdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, workdir / LEDGER)

    def _write_subset(self, workdir: Path, part: str, *, seed: int | None = None,
                      gold_feedback: bool = False) -> None:
        """Name the part of the items this candidate is evaluated on.

        Written into the candidate's own run directory before the evaluator starts,
        and read by the item loader of either runtime. It is not a knob: a search
        that could set it would choose its own exam, and there would be no item left
        for the audit to hold out. ``gold_feedback`` lets a train evaluation's passes be
        told how their attempt compares with the answer (`refactory_dsh.feedback`).
        """
        task_module.write_subset(workdir, part, self.config.holdout, seed=seed,
                                 gold_feedback=gold_feedback)
