
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

from refactory_dsh.loop.config import EvaluationInterrupted, INTERRUPTED_RETRY_WAITS_S, INTERRUPTION_CAP, LoopError, SMOKE_CODE, opaque_item_key
from refactory_dsh.loop.report import _trim, node_count


def _wiring_warnings(state: Mapping[str, Any] | None,
                     baseline: Mapping[str, Any] | None = None) -> list[str]:
    """A state's wiring warnings (`nn.validate.wiring_warnings`), its steps judged against
    ``baseline``'s (new or changed ones only) when given; none for a state whose graph
    does not parse (the validator refuses it on its own)."""
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.validate import wiring_warnings

    try:
        before = (Graph.from_state((baseline or {}).get("graph") or {})
                  if baseline is not None else None)
        return wiring_warnings(Graph.from_state((state or {}).get("graph") or {}), before)
    except Exception:  # noqa: BLE001 - a broken graph is V0's to report, not this note's
        return []

def _new_wiring_warnings(state: Mapping[str, Any] | None,
                         parent_state: Mapping[str, Any] | None) -> list[str]:
    """The wiring warnings an edit added: ``state``'s, its new or changed steps judged
    against ``parent_state``'s, less those the parent already had (`_wiring_warnings`).
    What the ledger says of an edit, and what the proposer is shown before its candidate
    runs (`Loop._wiring_check`)."""
    before = set(_wiring_warnings(parent_state, parent_state))
    return [w for w in _wiring_warnings(state, parent_state) if w not in before]

def _other_value(knob: Knob, current: Any, rng: random.Random) -> Any:
    """A value a knob accepts that is not the one it currently holds.

    One sampler for the whole system, so the starting sweep and the random
    optimizer agree about what is reachable.
    """
    return knob.sample(rng, current)


class _EvaluateMixin:

    def _evaluate(
        self, candidate: Candidate, *, seed: int | None = None, items: list[str] | None = None,
        gold_feedback: bool = False,
        _budget_reserved: bool = False,
        item_gate: threading.BoundedSemaphore | None = None,
        item_slots: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Run one candidate through the evaluator, and record what it scored.

        This is the loop's forward pass and the evaluator is its loss: what the
        candidate scored is whatever the evaluator says it scored. Which evaluator
        ran it is not this method's business -- see :mod:`refactory_dsh.evaluators`
        for the two that ship and what makes them agree.

        The evaluator runs the train part of the item selection, and the loop takes
        the size cost off afterwards. ``raw_score`` keeps the evaluator's number, so
        the report can say whether a candidate won on merit or on size.

        When ``_budget_reserved`` is True, the caller has already pre-incremented
        ``self.evaluations``; this method skips that increment.
        """
        workdir = self.root / "runs" / candidate.id
        mode = self.mode_class(state=candidate.state)
        if not candidate.evaluated:
            # An earlier attempt that never finished (the run was killed or its
            # evaluation interrupted, and this is the resume) left its passes and
            # per-item receipts here; the evaluation starts clean, so keep them.
            self._archive_unfinished(candidate.id, workdir, keep=("occasions",))

        def prepare() -> None:
            # Copy the loop-level manifest so select_subset finds it.
            if self._parts_manifest is not None:
                workdir.mkdir(parents=True, exist_ok=True)
                manifest_src = self.root / task_module.MANIFEST
                if manifest_src.exists():
                    shutil.copyfile(manifest_src, workdir / task_module.MANIFEST)
            # When explicit items are given, write a per-candidate manifest.
            if items is not None:
                workdir.mkdir(parents=True, exist_ok=True)
                task_module.make_manifest(workdir, train_ids=items)

        result, measurements = self._settled_measure(
            candidate, workdir, task_module.TRAIN, prepare, keep=("occasions",),
            seed=seed, items=items, item_gate=item_gate, item_slots=item_slots,
            stop_event=stop_event, gold_feedback=gold_feedback,
        )
        if measurements.get("budget_stopped"):
            self._budget_stopped(candidate, task_module.TRAIN, result, measurements)
        nodes = node_count(candidate.state)
        candidate.raw_score = result.score if result.exit_ok else None
        item_scores = [float(r["score"]) for r in result.item_records
                       if isinstance(r.get("score"), (int, float)) and not isinstance(r.get("score"), bool)]
        if result.exit_ok and item_scores:
            # A mode's score may be any metric of the run; it is a per-item mean only
            # when it equals the mean of the item scores, every time.
            mean = math.isclose(sum(item_scores) / len(item_scores), float(result.score),
                                rel_tol=1e-9, abs_tol=1e-9)
            with self._state_lock:
                self._score_is_item_mean = mean and self._score_is_item_mean is not False
        candidate.score = result.score - self.config.parsimony * nodes if result.exit_ok else None
        candidate.mean_score = candidate.raw_score
        candidate.measurements = measurements
        candidate.facts = _trim(result.facts)
        candidate.exit_ok = result.exit_ok
        candidate.error = result.error
        candidate.duration_s = result.duration_s
        candidate.model_calls = result.model_calls
        candidate.evaluated = True
        with self._state_lock:
            self._train_item_records[candidate.id] = list(result.item_records)
            if not _budget_reserved:
                self.evaluations += 1
            if not measurements.get("cached_from"):
                self.rollouts += len(result.item_records) or 1
            self._journal("train_records", id=candidate.id,
                          records=list(result.item_records))
            self._journal("rollouts", rollouts=self.rollouts)
            self._journal(
                "score",
                id=candidate.id,
                score=candidate.score,
                raw_score=candidate.raw_score,
                nodes=nodes,
                parsimony=self.config.parsimony,
                # The rollout seed, so an export can say what its records were measured
                # with (``seed_cache``).
                seed=seed,
                exit_ok=candidate.exit_ok,
                error=candidate.error,
                duration_s=candidate.duration_s,
                model_calls=candidate.model_calls,
                facts=candidate.facts,
                measurements=measurements,
            )
            self._write_candidate(candidate, workdir, mode=mode)

    def _budget_stopped(self, candidate: Candidate, part: str, result: Any,
                        measurements: dict[str, Any]) -> None:
        """A train evaluation a time or token budget refused or cut: what it ran counts
        toward rollouts and is journaled (``budget_stopped``), but the candidate stays
        unevaluated, so a resume with more budget scores it. Raises ``BudgetStopped``."""
        with self._state_lock:
            if not measurements.get("cached_from"):
                self.rollouts += len(result.item_records)
            self._journal("budget_stopped", id=candidate.id, part=part, error=result.error,
                          measurements=measurements)
            self._journal("rollouts", rollouts=self.rollouts)
        raise BudgetStopped(f"the budget stopped {candidate.id}'s {part} evaluation: {result.error}",
                            reason=self.bound or str(result.error or ""))

    def _settled_measure(
        self, candidate: Candidate, workdir: Path, part: str, prepare: Any, *,
        keep: tuple[str, ...] = (), rescore_epoch: int | None = None,
        stop_event: threading.Event | None = None, **measure: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """``_measure`` after ``prepare``, run again while it is interrupted and an
        in-run retry is left (`_settle_interruption`), each earlier attempt archived
        first (``keep`` stays). A measurement a budget stopped is returned as it is: the
        budget, not an interruption, is why it ended."""
        attempt = 0
        while True:
            prepare()
            result, measurements = self._measure(candidate, workdir, part,
                                                 stop_event=stop_event, **measure)
            if measurements.get("budget_stopped") or not self._settle_interruption(
                    candidate, part, result, workdir, attempt,
                    rescore_epoch=rescore_epoch, stop_event=stop_event):
                return result, measurements
            attempt += 1
            self._archive_unfinished(candidate.id, workdir, keep=keep)

    @staticmethod
    def _interruption(result: Any) -> dict[str, int]:
        """Why ``result`` was interrupted rather than measured, as ``{cause: items}``, or
        ``{}``: any of its items, or (with none of them) the run itself outside any item,
        failed because a process of the run died (`items.interrupted_item`)."""
        from collections import Counter

        from refactory_dsh.items import interrupted_cause, interrupted_item

        causes = Counter(cause for cause in map(interrupted_item, result.item_records or ())
                         if cause)
        if not causes and not result.exit_ok and (cause := interrupted_cause(result.error)):
            causes[cause] = 0  # the run's own error, outside any item
        return dict(causes)

    def _settle_interruption(self, candidate: Candidate, part: str, result: Any, workdir: Path,
                             attempt: int, *, rescore_epoch: int | None = None,
                             stop_event: threading.Event | None = None) -> bool:
        """What to do with ``candidate``'s ``part`` evaluation when it was interrupted
        (`_interruption`); False when it was not, so it is the candidate's result.

        Interrupted, it is journaled (``evaluation_interrupted``, with how many items died
        and why, and the outcome) and shown. While an in-run retry is left
        (`INTERRUPTED_RETRY_WAITS_S`) the loop waits and returns True: run it again.
        Otherwise it is one stop of this evaluation: below `INTERRUPTION_CAP` stops the
        run stops (`EvaluationInterrupted`), and nothing of the evaluation is recorded --
        no score, records, val occasion or rollouts; at the cap ``result`` becomes a
        failed evaluation ("interrupted N times; recorded as failed") and False is
        returned, so it is recorded as one. A model's own failure -- a timeout, a wrong
        or garbled answer, an output limit -- is never interrupted."""
        causes = self._interruption(result)
        if not causes:
            return False
        key = f"{candidate.id}:{part}" + (f"@{rescore_epoch}" if rescore_epoch is not None else "")
        waits = INTERRUPTED_RETRY_WAITS_S
        stops = None
        if attempt < len(waits):
            outcome = "retrying"
        else:
            with self._state_lock:
                stops = self._interrupted_stops.get(key, 0) + 1
                self._interrupted_stops[key] = stops
            outcome = "recorded_failed" if stops >= INTERRUPTION_CAP else "stopped"
        records = len(result.item_records or ())
        fields = {
            "part": part, "attempt": attempt + 1, "outcome": outcome,
            "interrupted": sum(causes.values()), "items": records, "causes": causes,
            "error": result.error,
            **({"retry_in_s": waits[attempt]} if outcome == "retrying" else {"stops": stops}),
            **({"rescore_epoch": rescore_epoch} if rescore_epoch is not None else {}),
        }
        with self._state_lock:
            self._journal("evaluation_interrupted", id=candidate.id, key=key,
                          workdir=str(workdir.relative_to(self.root)), **fields)
        self._progress("evaluation_interrupted", candidate=candidate.id, **fields)
        said = ", ".join(f"{cause} {count}" for cause, count in sorted(causes.items()))
        if outcome == "retrying":
            if stop_event is not None:
                if stop_event.wait(waits[attempt]):
                    raise EvaluationInterrupted(
                        f"{candidate.id}'s {part} evaluation was interrupted ({said}) and the "
                        "run was stopped before its retry; resume the run (--resume)")
            else:
                time.sleep(waits[attempt])
            return True
        if outcome == "recorded_failed":
            result.exit_ok = False
            result.score = 0.0
            result.error = (f"{part} evaluation interrupted {stops} times; recorded as failed "
                            f"(last: {sum(causes.values())} of {records} item(s) died: {said})")
            return False
        raise EvaluationInterrupted(
            f"{candidate.id}'s {part} evaluation was interrupted: {sum(causes.values())} of "
            f"{records} item(s) died with the run's own processes ({said}) in "
            f"{attempt + 1} attempt(s), so it is not a result; resume the run (--resume) to "
            f"evaluate it again (stop {stops} of {INTERRUPTION_CAP - 1} before it is "
            "recorded as failed)")

    def _archive_unfinished(self, candidate_id: str, workdir: Path, *,
                            keep: tuple[str, ...] = ()) -> Path | None:
        """Move what an evaluation that never finished left in ``workdir`` -- its
        per-item passes and receipts -- to the run's archive
        (``.refactory/archive/<UTC time>/<workdir relative to the run>``), so the
        evaluation that replaces it starts clean and nothing of the earlier one is
        deleted. ``keep`` names entries that are not the earlier attempt's (a
        candidate's ``occasions``). Journaled (``evaluation_archived``); returns the
        archive folder, or None when there was nothing to move."""
        from datetime import datetime, timezone

        from refactory_dsh.run_artifacts import archive_root

        if not workdir.is_dir():
            return None
        leftovers = sorted(path for path in workdir.iterdir() if path.name not in keep)
        if not leftovers:
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        relative = workdir.relative_to(self.root)
        target = archive_root(self.root, "archive") / stamp / relative
        target.mkdir(parents=True)
        for path in leftovers:
            path.rename(target / path.name)
        with self._state_lock:
            self._journal("evaluation_archived", id=candidate_id, workdir=str(relative),
                          archive=str(target.relative_to(self.root)),
                          entries=[path.name for path in leftovers])
        return target

    def _evaluate_part(
        self, candidate: Candidate, part: str,
        items: list[str] | None, seed: int | None,
        *,
        item_gate: threading.BoundedSemaphore | None = None,
        item_slots: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Evaluate on val or test in a fresh occasion directory (F18)."""
        with self._state_lock:
            counters = self._occasion_counters.setdefault(candidate.id, {})
            counter = counters.get(part, 0)
            counters[part] = counter + 1
            self._journal("occasion_counter", id=candidate.id, part=part,
                          counter=counters[part])
        workdir = self.root / "runs" / candidate.id / "occasions" / f"{part}_{counter}"

        def prepare() -> None:
            workdir.mkdir(parents=True, exist_ok=True)
            task_module.write_subset(workdir, part, self.config.holdout, seed=seed)
            # Copy or build the manifest so select_subset finds it.
            if items is not None:
                task_module.make_manifest(workdir, train_ids=items if part == "train" else [],
                                         val_ids=items if part == "val" else None,
                                         test_ids=items if part == "test" else None)
            elif self._parts_manifest is not None:
                manifest_src = self.root / task_module.MANIFEST
                if manifest_src.exists():
                    shutil.copyfile(manifest_src, workdir / task_module.MANIFEST)
            self._carry_ledger(workdir)

        result, measurements = self._settled_measure(
            candidate, workdir, part, prepare, seed=seed, items=items,
            item_gate=item_gate, item_slots=item_slots, stop_event=stop_event,
        )
        # Score-only occasion record for optimizer-facing storage (no records,
        # no workdir).
        occasion_public = {
            "part": part, "occasion": counter,
            "score": result.score if result.exit_ok else None,
            "exit_ok": result.exit_ok, "error": result.error,
            "duration_s": result.duration_s,
            # A rollout a time or token budget refused or cut: no score of the
            # candidate's (``ValidationGate.val_score`` skips it).
            **({"budget_stopped": True} if measurements.get("budget_stopped") else {}),
            # Items a harness fault kept from an outcome, left out of the score
            # (`_settle_incidents`), and each scored item's score, so a val comparison
            # of two candidates can use the items both were scored on
            # (`shared_scores`): ``[id, score, repeat]``. Neither is shown to a strategy
            # (`Run.part_scores`).
            **({"harness_incidents": measurements["harness_incidents"]}
               if measurements.get("harness_incidents") else {}),
            "item_scores": [[str(r.get("id")), float(r["score"]), int(r.get("repeat") or 1)]
                            for r in result.item_records or ()
                            if isinstance(r.get("score"), (int, float))
                            and not isinstance(r.get("score"), bool)],
        }
        with self._state_lock:
            self._part_results.setdefault(candidate.id, {}).setdefault(part, []).append(
                occasion_public
            )
            self._journal("part_result", id=candidate.id, part=part, seed=seed,
                          result=occasion_public)
            if not measurements.get("cached_from"):
                self.rollouts += len(result.item_records) or 1
            self._journal("rollouts", rollouts=self.rollouts)
        if result.item_records:
            (workdir / "item_records.json").write_text(
                json.dumps(result.item_records, indent=2, default=str))
        if measurements.get("budget_stopped"):
            raise BudgetStopped(f"the budget stopped {candidate.id}'s {part} rollout: {result.error}",
                                reason=self.bound or str(result.error or ""))

    def _measure(
        self, candidate: Candidate, workdir: Path, part: str, *,
        seed: int | None = None, items: list[str] | None = None,
        gold_feedback: bool = False,
        item_gate: threading.BoundedSemaphore | None = None,
        item_slots: bool = False,
        stop_event: threading.Event | None = None,
        repeats: int | None = None,
        traces: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        """Independent bounded trials; any failed contract makes a candidate ineligible.

        ``repeats`` (default ``LoopConfig.repeats``) is how many trials; ``traces`` False
        writes no trace files (the smoke check's runs, `_smoke_check`)."""
        from refactory_dsh.nn.mode import RunResult
        from refactory_dsh.budgets import BudgetExhausted

        # Loop-owned item channel: expected ids computed from the seed's state
        # and the loop root, not from the candidate's trial directory.
        seed_state = self.population[0].state if self.population else candidate.state
        expected_ids, seed_declares_item_scores = self._expected_item_ids(seed_state, part)
        # When an explicit items subset was requested, narrow expected_ids to
        # just those items so _verify_item_set does not fail on the "missing"
        # items that were intentionally excluded.
        if items is not None:
            expected_ids = set(str(i) for i in items)

        cached = self._cached_measure(candidate, workdir, part, expected_ids, seed,
                                      gold_feedback=gold_feedback)
        if cached is not None:
            if traces:
                self._write_traces(candidate, workdir, part, seed, cached[0].item_records)
            return cached

        wanted = self.config.repeats if repeats is None else max(1, int(repeats))
        trials = []
        budget_cut = False  # a time or token budget refused or cut a trial
        all_item_records: list[dict[str, Any]] = []  # accumulate across repeats
        last = RunResult(exit_ok=False, error="no measurement completed")
        required = (self.config.resolved_profile.get("contract") or {}).get("required_artifacts", [])
        for index in range(wanted):
            trial_dir = workdir if index == 0 else workdir / "trials" / str(index + 1)
            self._carry_ledger(trial_dir)
            self._write_subset(trial_dir, part, seed=seed, gold_feedback=gold_feedback)
            # Copy the manifest to trial sub-directories so select_subset
            # finds it there too (repeats > 1 creates separate directories).
            manifest = workdir / task_module.MANIFEST
            if trial_dir != workdir and manifest.exists():
                trial_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(manifest, trial_dir / task_module.MANIFEST)
            # Noted as running before the ledger is read, so a trial that starts and ends
            # in between is one beside it (`_trial_usage`); ended however the trial ends.
            running = self._trial_started(trial_dir)
            try:
                before = self.budget.snapshot() if self.budget else {}
                try:
                    if self.budget:
                        final = part == task_module.HOLDOUT or getattr(self, "_final_scoring", False)
                        self.budget.checkpoint(("final_validation" if final else "work")
                                               if self._owns_resource_budget else None)
                    with self.budget.retain_validation_reserve() if self.budget else nullcontext():
                        eval_kwargs: dict[str, Any] = {}
                        if stop_event is not None and self.evaluator.out_of_process:
                            eval_kwargs["stop_event"] = stop_event
                        if item_gate is not None:
                            eval_kwargs["item_gate"] = item_gate
                        elif (item_slots and self.evaluator.out_of_process
                              and hasattr(self, '_resolved_max_items_in_flight')):
                            # Spine, under evaluate_many only: the shared slot
                            # directory. A plain Run.evaluate launches as before.
                            slot_dir = self.root / ".refactory" / "item-slots"
                            eval_kwargs["item_slots_dir"] = str(slot_dir)
                            eval_kwargs["item_slots"] = self._resolved_max_items_in_flight
                        result = self.evaluator.evaluate(
                            self.mode_class(state=candidate.state), candidate.state,
                            trial_dir, **eval_kwargs,
                        )
                except BudgetExhausted as exc:
                    last = RunResult(exit_ok=False, error=str(exc))
                    self.bound = str(exc)
                    budget_cut = True
                    break
                budget_exhausted_after_trial = False
                if self.budget:
                    try:
                        self.budget.checkpoint()
                    except BudgetExhausted as exc:
                        self.bound = str(exc)
                        budget_exhausted_after_trial = True
                # Post-trial item-ownership verification (works with both evaluators).
                self._verify_item_set(expected_ids, seed_declares_item_scores, trial_dir, part, result)
                incidents, budget_hit = self._settle_incidents(
                    candidate, trial_dir, part, seed, result, eval_kwargs,
                    rerun=not budget_exhausted_after_trial,
                    pending=len(all_item_records) + len(result.item_records or ()))
                budget_exhausted_after_trial = budget_exhausted_after_trial or budget_hit
                # After the incident reruns, so their tokens and cost are the trial's.
                after = self.budget.snapshot() if self.budget else {}
            finally:
                beside = self._trial_ended(running)
            tokens, cost = self._trial_usage(before, after, trial_dir, beside)
            missing = [path for path in required if not (trial_dir / path).exists()]
            valid_score = isinstance(result.score, (int, float)) and math.isfinite(result.score)
            if missing or not valid_score:
                result.exit_ok = False
                result.error = result.error or ("required artifacts missing: " + ", ".join(missing) if missing else "score is unmeasured or nonfinite")
            trial = {"trial": index + 1, "score": result.score if result.exit_ok else None,
                     "exit_ok": result.exit_ok, "error": result.error, "duration_s": result.duration_s,
                     "model_calls": result.model_calls, "required_contract": {"passed": result.exit_ok, "missing_artifacts": missing},
                     "tokens": tokens, "cost": cost,
                     "workdir": str(trial_dir),
                     **({"harness_incidents": incidents} if incidents else {})}
            trial["workers"] = []
            for receipt_path in sorted(trial_dir.glob("**/.refactory/workers/*.json")):
                if receipt_path.relative_to(trial_dir).parts[0] == "occasions":
                    # The candidate's other evaluations (`occasions/`: val_<n>, test_<n>,
                    # smoke_<n>, train_epoch_<n>, memory_epoch_<n>_<key>; under
                    # ``checkpoint`` a val run beside this one): not this trial's workers.
                    continue
                try:
                    receipt = json.loads(receipt_path.read_text())
                    trial["workers"].append({key: receipt.get(key) for key in ("worker_id", "node_id", "role", "state", "actual_route", "runtime", "usage")})
                except (OSError, json.JSONDecodeError):
                    continue
            trials.append(trial)
            with self._state_lock:
                self.measurements += 1
            # Each record says which repeat graded it: joined across repeats, the
            # same item id appears once per repeat, and without the index two
            # gradings of one item read as a duplicate.
            all_item_records.extend({**record, "repeat": index + 1} for record in result.item_records)
            last = result
            if budget_exhausted_after_trial and not result.exit_ok:
                budget_cut = True  # the budget ran out inside the trial and ended it
            if not result.exit_ok:
                break
            if budget_exhausted_after_trial:
                budget_cut = budget_cut or len(trials) < wanted
                break
        scores = [t["score"] for t in trials if t["score"] is not None]
        complete = len(trials) == wanted and all(t["exit_ok"] for t in trials)
        summary = {"requested": wanted, "count": len(trials), "successful": len(scores),
                   "mean": statistics.mean(scores) if scores else None,
                   "stdev": statistics.stdev(scores) if len(scores) > 1 else None,
                   "min": min(scores) if scores else None, "max": max(scores) if scores else None,
                   "complete": complete, "trials": trials, "context_hash": self.config.resolved_profile.get("context_hash"),
                   "tokens": sum(t["tokens"] for t in trials) if trials and all(isinstance(t["tokens"], (int, float)) for t in trials) else None,
                   "cost": sum(t["cost"] for t in trials) if trials and all(isinstance(t["cost"], (int, float)) for t in trials) else None,
                   "interpretation": "observed sample mean; improvement is not a statistical proof"}
        if budget_cut and not complete:
            summary["budget_stopped"] = True
        incident_ids = list(dict.fromkeys(i for t in trials for i in t.get("harness_incidents") or ()))
        if incident_ids:
            summary["harness_incidents"] = incident_ids
        last.score = float(summary["mean"] or 0) if complete else 0.0
        last.exit_ok = complete
        last.duration_s = sum(t["duration_s"] for t in trials)
        last.model_calls = sum(t["model_calls"] for t in trials)
        # Every trial's item records count toward rollouts and are kept
        # per occasion (brief ask 1: "every trial's items count toward
        # rollouts and the records of every repeat are kept per occasion").
        last.item_records = all_item_records
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "measurements.json").write_text(json.dumps(summary, indent=2))
        if traces:
            self._write_traces(candidate, workdir, part, seed, all_item_records)
        return last, summary

    def _trial_started(self, trial_dir: Path) -> dict[str, Any]:
        """Note a trial as running, with every other trial running beside it, now or
        later (``Run.evaluate_many``): ``_trial_usage`` leaves their spend out of its.
        Ended by ``_trial_ended``."""
        entry: dict[str, Any] = {"dir": trial_dir, "met": set()}
        with self._state_lock:
            running = self.__dict__.setdefault("_running_trials", [])
            for other in running:
                other["met"].add(trial_dir)
                entry["met"].add(other["dir"])
            running.append(entry)
        return entry

    def _trial_ended(self, entry: dict[str, Any]) -> set[Path]:
        """The trial is no longer running; the folders of the trials that ran beside it."""
        with self._state_lock:
            running = self.__dict__.get("_running_trials") or []
            running[:] = [other for other in running if other is not entry]
        return set(entry["met"])

    def _trial_usage(self, before: dict[str, Any], after: dict[str, Any], trial_dir: Path,
                     beside: set[Path]) -> tuple[int | None, float | None]:
        """A trial's tokens and cost: what the run's ledger counted from ``before`` to
        ``after``, less what the workers of the trials that ran ``beside`` it spent then.
        Those are the workers whose receipts (``.refactory/workers/*.json``) lie in a
        folder of theirs and not in the trial's own, as an agent node's and a model
        call's do on both runtimes. A trial no other ran beside gets the ledger's
        difference as it is. A worker that records usage with no receipt (a custom
        client's own accounting) cannot be told apart and stays in every trial it ran
        beside."""
        tokens = (after.get("known_tokens", 0) - before.get("known_tokens", 0)
                  if after.get("token_status") in ("measured", "estimated") else None)
        cost = (after["estimated_cost_usd"] - (before.get("estimated_cost_usd") or 0)
                if after.get("estimated_cost_usd") is not None else None)
        if not beside:
            return tokens, cost
        own = trial_dir.resolve()
        foreign: set[str] = set()
        for folder in beside:
            holds_own = own.is_relative_to(Path(folder).resolve())
            for receipt in Path(folder).glob("**/.refactory/workers/*.json"):
                if holds_own and receipt.resolve().is_relative_to(own):
                    continue  # the trial's own worker, in a folder that holds its own
                try:
                    foreign.add(str(json.loads(receipt.read_text()).get("worker_id")
                                    or receipt.stem))
                except (OSError, ValueError):
                    foreign.add(receipt.stem)
        was = before.get("workers") or {}
        for key, entry in (after.get("workers") or {}).items():
            # A worker's own entry, or one it records beside it (``<id>:...``).
            if key not in foreign and key.split(":", 1)[0] not in foreign:
                continue
            old = was.get(key) or {}
            if tokens is not None:
                tokens -= int(entry.get("tokens") or 0) - int(old.get("tokens") or 0)
            if cost is not None:
                cost -= (float(entry.get("estimated_cost_usd") or 0)
                         - float(old.get("estimated_cost_usd") or 0))
        return tokens, cost

    def shared_scores(self, candidate_ids: Sequence[str], part: str = "val", *,
                      latest: bool = False) -> tuple[dict[str, float], list[str]] | None:
        """Candidates' scores on ``part`` over the same items, when a harness incident left
        an item out of any one's rollout (`_settle_incidents`): ``({id: score}, the items
        left out)``. Each score is its rollout's mean over the items every candidate was
        scored on in every repeat, none of them left out; with repeats, the mean of each
        repeat's mean, as the rollout's own score is (`_measure`). The rollout is each
        candidate's first on ``part`` not stopped by a budget (the val score the gate
        reads), or with ``latest`` its latest (the test score a final evaluation reads).

        ``None`` when no candidate left an item out: their scores already cover the same
        items and compare as they are. Also ``None``, with a warning (once per reason),
        when they cannot be put on the same items -- a candidate has no finished rollout,
        a score is not its items' mean (another metric cannot be recomputed on fewer
        items), or no item is left that all were scored on -- so the scores compare as
        they are and the run says so."""
        ids = list(dict.fromkeys(str(c) for c in candidate_ids))
        rollouts: dict[str, dict[str, Any]] = {}
        for cid in ids:
            done = [o for o in self._part_results.get(cid, {}).get(part, [])
                    if not o.get("budget_stopped")]
            if done:
                rollouts[cid] = done[-1] if latest else done[0]
        left_out = sorted({str(i) for o in rollouts.values() for i in o.get("harness_incidents") or ()})
        if not left_out:
            return None

        def unmatched(reason: str) -> None:
            note = (f"{part} scores of {', '.join(ids)} compared as they are, not on the same "
                    f"items, though a harness incident left {', '.join(left_out)} out: {reason}")
            with self._state_lock:
                warned = self.__dict__.setdefault("_shared_warned", set())
                if reason in warned:
                    return
                warned.add(reason)
                self._journal("shared_scores_unmatched", part=part, candidates=ids,
                              left_out=left_out, reason=reason)
            self._progress("warning", warning=note)

        if len(rollouts) < len(ids) or any(o.get("score") is None for o in rollouts.values()):
            unmatched("a candidate has no finished rollout")
            return None
        by_repeat: dict[str, dict[int, dict[str, float]]] = {}
        for cid, occasion in rollouts.items():
            repeats: dict[int, dict[str, float]] = {}
            for entry in occasion.get("item_scores") or ():
                item, value = str(entry[0]), float(entry[1])
                repeats.setdefault(int(entry[2]) if len(entry) > 2 else 1, {})[item] = value
            if not repeats or not math.isclose(
                    statistics.mean(statistics.mean(r.values()) for r in repeats.values()),
                    float(occasion["score"]), rel_tol=1e-9, abs_tol=1e-9):
                unmatched("a score is not its items' mean")
                return None
            by_repeat[cid] = repeats
        common = set.intersection(*(set(r) for repeats in by_repeat.values()
                                    for r in repeats.values())) - set(left_out)
        if not common:
            unmatched("no item is left that every candidate was scored on")
            return None
        return ({cid: statistics.mean(statistics.mean(r[i] for i in common)
                                      for r in repeats.values())
                 for cid, repeats in by_repeat.items()}, left_out)

    @property
    def item_key_secret(self) -> str:
        """The run's secret for `opaque_item_key`: made on first use, journaled, and
        restored on resume, so a key names the same item for the whole run."""
        if self._item_key_secret is None:
            import secrets

            self._item_key_secret = secrets.token_hex(16)
            self._journal("item_key_secret", secret=self._item_key_secret)
        return self._item_key_secret

    def part_item_scores(self, candidate_id: str, part: str = "val") -> dict[str, float]:
        """Each item's score in the candidate's first finished ``part`` rollout (the one
        the val gate reads), averaged over repeats, keyed by `opaque_item_key`: a
        per-instance view for a strategy that keeps a per-item frontier on val (GEPA),
        which names no held-out item. In the parts manifest's order, so a strategy that
        walks it is deterministic. Items a harness incident left out have no score and
        are absent; empty before a finished rollout."""
        done = [o for o in self._part_results.get(candidate_id, {}).get(part, [])
                if not o.get("budget_stopped") and o.get("score") is not None]
        if not done:
            return {}
        values: dict[str, list[float]] = {}
        for entry in done[0].get("item_scores") or ():
            values.setdefault(str(entry[0]), []).append(float(entry[1]))
        manifest = [str(i) for i in ((self._parts_manifest or {}).get(part) or ())]
        rank = {item: n for n, item in enumerate(manifest)}
        ordered = sorted(values, key=lambda item: (rank.get(item, len(rank)), item))
        secret = self.item_key_secret
        return {opaque_item_key(item, secret): statistics.mean(values[item]) for item in ordered}

    def part_incidents(self, candidate_id: str, part: str = "val") -> list[str]:
        """The items a harness incident left out of the candidate's first finished
        ``part`` rollout (`_settle_incidents`); empty when none."""
        done = [o for o in self._part_results.get(candidate_id, {}).get(part, [])
                if not o.get("budget_stopped")]
        return list(done[0].get("harness_incidents") or ()) if done else []

    def _write_traces(self, candidate: Candidate, workdir: Path, part: str,
                      seed: int | None, records: Sequence[dict[str, Any]] | None) -> None:
        """Write each record of this evaluation as files (`refactory_dsh.traces`), for
        people and a later trace-reading analyst; nothing in the run reads them back.

        Every evaluation passes through ``_measure`` on both runtimes (train, re-score,
        memory rollout, val, test, holdout), and its records here are the evaluator's
        own, whole. The occasion is the evaluation directory's name (``train_epoch_3``,
        ``val_0``, ``memory_epoch_2_<key>``), empty for a candidate's own first
        evaluation and the holdout audit.

        The number is one past the evaluations the journal already records for this
        folder (`_note_recorded`): a second train rollout of a candidate at one seed
        gets a folder of its own, while an attempt the journal has not recorded (an
        in-run retry, or a resume running an interrupted or budget-stopped evaluation
        again) gets the same number and rewrites its own folder. Adds no journal
        event; never raises."""
        from refactory_dsh import traces

        occasion = "" if workdir.name == candidate.id else workdir.name
        with self._state_lock:
            number = self._recorded_evaluations.get(self._relative(workdir), 0) + 1
        traces.write_evaluation(self.root, candidate.id, part, occasion, seed, records or (),
                                number=number)

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def _rescore_dir(self, candidate_id: str, epoch: int) -> Path:
        """Where ``candidate_id``'s fresh-evidence re-score on ``epoch`` runs."""
        return self.root / "runs" / candidate_id / "occasions" / f"train_epoch_{epoch}"

    def _memory_dir(self, candidate_id: str, for_epoch: int, key: str) -> Path:
        """Where ``candidate_id``'s memory rollout ``key`` for ``for_epoch`` runs."""
        return (self.root / "runs" / candidate_id / "occasions"
                / f"memory_epoch_{for_epoch}_{key}")

    def _note_recorded(self, event: Any, record: dict[str, Any]) -> None:
        """Count a journal event that records a finished evaluation against the folder
        it ran in: ``score`` (a candidate's train evaluation), ``rescore`` and
        ``rescore_failed``, ``memory_rollout``. A val or test rollout runs in a fresh
        occasion folder every time, and the holdout once, so they need no count. Called
        for every event written (`_journal`) and replayed, so a resume counts alike."""
        if event == "bound":
            # The bound the journal holds last, so the run's end does not write it again.
            self.__dict__["_journaled_bound"] = record.get("bound")
        cid = record.get("id")
        if event == "train_records" and cid is not None:
            # Each train run's outcomes for the item history (`_with_item_history`): an
            # item a harness incident or timeout kept from an outcome has none.
            from refactory_dsh.optim.base import harness_timeout

            self.__dict__.setdefault("_train_runs", []).append((str(cid), [
                (str(r.get("id")), bool(r.get("passed"))) for r in record.get("records") or ()
                if not r.get("not_scored") and not harness_timeout(r)]))
        if cid is None or event not in ("score", "rescore", "rescore_failed", "memory_rollout"):
            return
        cid = str(cid)
        if event == "score":
            folder = self.root / "runs" / cid
        elif event == "memory_rollout":
            epoch = int(record.get("epoch") or 0)
            folder = self._memory_dir(cid, int(record.get("for_epoch") or 0),
                                      str(record.get("key") or f"e{epoch}"))
        else:
            folder = self._rescore_dir(cid, int(record.get("epoch") or 0))
        counts = self.__dict__.setdefault("_recorded_evaluations", {})
        key = self._relative(folder)
        counts[key] = counts.get(key, 0) + 1

    def _cached_measure(
        self, candidate: Candidate, workdir: Path, part: str, expected_ids: set[str],
        seed: int | None = None, gold_feedback: bool = False,
    ) -> tuple[Any, dict[str, Any]] | None:
        """The ``--seed-results`` entry for this evaluation, if one applies.

        Only the cache's own candidate, only on the exact item set it recorded and
        with the rollout seed it was measured with (a Task may shuffle an item by it:
        a gated run's train rollouts use SkillOpt's per-epoch seed, an ungated run's
        the run's own), with the same gold feedback (an entry without the flag had none),
        and only once per part: a later occasion is a fresh measurement. The
        recorded run directory is copied in, so an optimizer that reads a
        candidate's pass files finds them as if the evaluation had run here.
        """
        from refactory_dsh.nn.mode import RunResult

        cache = self._seed_cache
        if cache is None or candidate.id != cache.get("candidate"):
            return None
        entry = (cache.get("parts") or {}).get(part)
        if entry is None:
            return None
        recorded = entry["result"]
        # Every trial the cache holds; a file written before trials were exported
        # holds one, with its records on the result.
        trials = entry.get("trials") or [{"score": recorded["score"], "duration_s": recorded.get("duration_s"),
                                          "model_calls": recorded.get("model_calls"),
                                          "workdir": entry.get("workdir"),
                                          "item_records": recorded.get("item_records") or []}]
        other_seed = entry.get("seed") != seed
        other_feedback = bool(entry.get("gold_feedback")) != bool(gold_feedback)
        # A cached item that died with its run's processes, or that a harness fault
        # stopped, is no result to replay (`items.interrupted_item`,
        # `items.harness_incident`): the part is measured here instead.
        from refactory_dsh.items import harness_incident, interrupted_item
        cached_records = [record for trial in trials for record in trial.get("item_records") or ()]
        killed = any(interrupted_item(record) for record in cached_records)
        died = killed or any(harness_incident(record) for record in cached_records)
        if (set(entry.get("items") or ()) != expected_ids or other_seed or other_feedback
                or len(trials) != self.config.repeats or died):
            # The cache's own candidate on a part it covers, measured on other items
            # (a train minibatch), with another rollout seed, or with items that died:
            # said once per part, so the run shows why the cache was not used.
            with self._state_lock:
                if part not in self._seed_cache_missed and part not in self._seed_cache_used:
                    self._seed_cache_missed.add(part)
                    reason = ("other items" if set(entry.get("items") or ()) != expected_ids
                              else f"rollout seed {seed}, cached {entry.get('seed')}" if other_seed
                              else (f"gold feedback {bool(gold_feedback)}, cached "
                                    f"{bool(entry.get('gold_feedback'))}") if other_feedback
                              else f"{len(trials)} cached trial(s), {self.config.repeats} requested"
                              if len(trials) != self.config.repeats
                              else "cached items died with their run's processes" if killed
                              else "cached items had a harness incident")
                    self._journal("seed_cache_miss", id=candidate.id, part=part, reason=reason,
                                  cached_items=len(entry.get("items") or ()), items=len(expected_ids))
            return None
        # Claim the part under the state lock: concurrent evaluations
        # (``Run.evaluate_many``) must not both replay the same entry.
        with self._state_lock:
            if part in self._seed_cache_used:
                return None
            self._seed_cache_used.add(part)
        ignore = shutil.ignore_patterns(
            "occasions", "trials", task_module.MANIFEST, "subset.json", "measurements.json")
        source = Path(entry.get("workdir") or "")
        if source.is_dir():
            shutil.copytree(source, workdir, dirs_exist_ok=True, ignore=ignore)
            for number in range(2, len(trials) + 1):
                # Each later trial's run directory, where a live run keeps it.
                if (source / "trials" / str(number)).is_dir():
                    shutil.copytree(source / "trials" / str(number), workdir / "trials" / str(number),
                                    dirs_exist_ok=True, ignore=ignore)
        records = [{**record, "repeat": number}
                   for number, trial in enumerate(trials, start=1) for record in trial["item_records"]]
        result = RunResult(
            exit_ok=True, score=float(recorded["score"]), error=None,
            duration_s=float(recorded.get("duration_s") or 0.0),
            model_calls=int(recorded.get("model_calls") or 0),
            facts=dict(recorded.get("facts") or {}),
            item_records=records,
            workdir=workdir,
        )
        origin = f"{cache.get('source')}#{part}"
        scores = [float(t["score"]) if t.get("score") is not None else result.score for t in trials]
        summary = {"requested": self.config.repeats, "count": len(trials), "successful": len(trials),
                   "mean": result.score, "stdev": statistics.stdev(scores) if len(scores) > 1 else None,
                   "min": min(scores), "max": max(scores), "complete": True,
                   "trials": [{"trial": number, "score": score, "exit_ok": True, "error": None,
                               "duration_s": float(trial.get("duration_s") or 0.0),
                               "model_calls": int(trial.get("model_calls") or 0),
                               "required_contract": {"passed": True, "missing_artifacts": []},
                               "tokens": None, "cost": None,
                               "workdir": str(workdir if number == 1 else workdir / "trials" / str(number)),
                               "workers": [], "cached_from": origin}
                              for number, (trial, score) in enumerate(zip(trials, scores), start=1)],
                   "context_hash": self.config.resolved_profile.get("context_hash"),
                   "tokens": None, "cost": None, "cached_from": origin,
                   "interpretation": "recorded by an earlier run under the same score context"}
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "measurements.json").write_text(json.dumps(summary, indent=2))
        with self._state_lock:
            self._journal("seed_cache_hit", id=candidate.id, part=part, source=origin,
                          items=len(result.item_records), trials=len(trials), score=result.score)
        return result, summary

    def _write_candidate(
        self, candidate: Candidate, workdir: Path, *, mode: Any = None
    ) -> None:
        workdir.mkdir(parents=True, exist_ok=True)
        nodes = node_count(candidate.state)
        (workdir / "candidate.json").write_text(
            json.dumps(
                {
                    "id": candidate.id,
                    "parent": candidate.parent,
                    # Both numbers, so a reader of one candidate directory can tell
                    # what the evaluator measured from what the search compared.
                    "score": candidate.score,
                    "raw_score": candidate.raw_score,
                    "nodes": nodes,
                    "parsimony": self.config.parsimony,
                    "size_cost": round(self.config.parsimony * nodes, 6),
                    "exit_ok": candidate.exit_ok,
                    "error": candidate.error,
                    "duration_s": candidate.duration_s,
                    "edits": candidate.edits,
                    "facts": candidate.facts,
                    "model_calls": candidate.model_calls,
                    "measurements": candidate.measurements,
                    # What the run published, against what the mode declared: a port
                    # that is missing is a promise the mode did not keep. Written for
                    # every candidate whichever evaluator ran it, because a reader
                    # comparing two runs should not have to know how they were run.
                    "ports": {
                        port.name: (workdir / port.path).exists()
                        for port in (mode.port_declarations() if mode is not None else ())
                    },
                },
                indent=2,
                default=str,
            )
        )

    # ── loss ───────────────────────────────────────────────────────────────
    def _seed_score(self) -> float | None:
        """The seed's score, the point the search started from; None until it ran and scored."""
        seed = self.population[0] if self.population else None
        return seed.score if seed is not None and seed.evaluated and seed.exit_ok else None

    # ── the integrity judge (``LoopConfig.integrity_judge``; `optim.integrity`) ──
    def _integrity_setting(self) -> dict[str, Any] | None:
        from refactory_dsh.optim.integrity import parse_setting

        return parse_setting(self.config.integrity_judge)

    def _integrity_on(self) -> bool:
        return self._integrity_setting() is not None

    def _integrity_policy(self) -> tuple[dict[str, str], dict[str, str]]:
        """Each category's policy (`optim.integrity.resolve_policy`: the run's setting,
        else the Tasks' declaration, else the run's default) and the allowances the
        judge is told (the Tasks', for the categories that are off). Read once."""
        cached = self.__dict__.get("_integrity_resolved")
        if cached is None:
            from refactory_dsh.optim import integrity
            from refactory_dsh.task import integrity_declaration

            from refactory_dsh.task import TaskError

            try:
                declared = integrity_declaration(self.mode.state_dict())
            except TaskError as exc:
                raise LoopError(f"integrity judge: {exc}") from exc
            policy = integrity.resolve_policy(self._integrity_setting(),
                                              declared=declared["policy"],
                                              allowances=declared["allowances"])
            cached = (policy, integrity.told(declared["allowances"], policy))
            self.__dict__["_integrity_resolved"] = cached
            self.__dict__["_integrity_declared"] = declared
        return cached

    def _journal_integrity_policy(self) -> None:
        """Journal the judge's setting, the Tasks' declaration and the policy they give
        (``integrity_policy``) before the first judgement that uses them, and again only
        when they change (a resume with another setting): what the offline replay needs
        to rebuild the run's prompts (`optim.integrity.proposals_from_run`)."""
        policy, _told = self._integrity_policy()
        declared = self.__dict__["_integrity_declared"]
        record = {"setting": self.config.integrity_judge, "task_policy": declared["policy"],
                  "allowances": declared["allowances"], "policy": policy}
        if json.dumps(record, sort_keys=True, default=str) != json.dumps(
                self.__dict__.get("_integrity_policy_journaled"), sort_keys=True, default=str):
            self.__dict__["_integrity_policy_journaled"] = record
            self._journal("integrity_policy", **record)

    def _judge_client(self) -> Any:
        """The judge's model, without tools: ``integrity_client`` when a caller set one,
        else the optimizer's route the budget bound (``bind_budget``); ``None`` when the
        run has no model."""
        from refactory_dsh.optim.random_optimizer import text_only

        model = self.integrity_client if self.integrity_client is not None else self._judge_model
        if model is None and self.model is not None:
            # No budget bound the route (the graph-run loop's node without one): the
            # optimizer's route of the run's own model.
            model = (self.model.for_node("optimizer", role="optimizer")
                     if callable(getattr(self.model, "for_node", None)) else self.model)
        return text_only(model) if model is not None else None

    def _judge(self, scope: str, prompt: str, **fields: Any) -> dict[str, Any]:
        """One judgement of ``prompt`` (`optim.integrity.ask`): ``{"verdict",
        "categories", "evidence", "reason", "error", "key"}``. A prompt judged before in
        this run is answered from the cache. A call or reply that fails is recorded in
        the optimizer trace and returned with ``error`` and no verdict -- never raised,
        except a spent budget during a proposal's judgement (as a step review raises it).
        Each call is recorded in the optimizer trace (kind ``integrity_<scope>``) and,
        through the optimizer's route, charged to the run's budget."""
        from refactory_dsh.budgets import BudgetExhausted
        from refactory_dsh.optim import integrity

        key = integrity.key(prompt)
        if key in self._integrity_cache:
            return {**self._integrity_cache[key], "error": None, "key": key}
        failed = {"verdict": None, "categories": [], "evidence": [], "reason": "", "key": key}
        categories = integrity.asked(self._integrity_policy()[0])
        client = self._judge_client()
        if client is None:
            raise LoopError("integrity_judge is on but this run has no model to judge with "
                            "(the judge uses the optimizer's model route)")
        import inspect
        options: dict[str, Any] = {}
        try:
            if "workdir" in inspect.signature(client.complete).parameters:
                options["workdir"] = self.root / "proposer"
        except (TypeError, ValueError):
            pass
        call = {"kind": f"integrity_{scope}", "epoch": self.epoch, "generation": self.generation,
                **fields, "system": integrity.SYSTEM, "prompt": prompt}
        reply = None
        try:
            found, reply = integrity.ask(client, prompt, categories, **options)
        except Exception as exc:  # a failed judgement is recorded and shown, never fatal
            error = f"{type(exc).__name__}: {exc}"
            self.optimizer.record_call(**call, reply=reply, error=error)
            if isinstance(exc, BudgetExhausted) and scope == "proposal":
                raise
            self._say_judge_error(error)
            return {**failed, "error": error}
        self.optimizer.record_call(**call, reply=reply, error=None, verdict=found["verdict"],
                                   categories=found["categories"])
        self._integrity_cache[key] = found
        return {**found, "error": None, "key": key}

    def _say_judge_error(self, error: str) -> None:
        """A failed judgement on stderr, the first time in this process (each one is in
        the trace, the progress stream and the report), so a judge that fails every
        time is never silent."""
        if not self._judge_error_said:
            self._judge_error_said = True
            print(f"refactory: the integrity judge failed ({error[:300]}); the change is "
                  "evaluated without a verdict. Later failures are in the report.",
                  file=sys.stderr)

    def _remember_integrity(self, record: dict[str, Any]) -> None:
        """Keep one verdict record, and its judgement by prompt when it has one (every
        parsed field, ``VERDICT_FIELDS``, so a cached salvaged or inherited verdict stays
        one)."""
        from refactory_dsh.optim.integrity import VERDICT_FIELDS

        self._integrity.append(record)
        if record.get("verdict") and record.get("key"):
            cached = {name: record[name] for name in VERDICT_FIELDS if name in record}
            if "judged_source" in record:
                # The judge's own label: the quote backstop is made again from the states.
                cached["source"] = record["judged_source"]
            self._integrity_cache[str(record["key"])] = cached

    def _integrity_check(self, proposal: Proposal, parent: Candidate,
                         candidate: Candidate) -> Candidate | list[str]:
        """Judge a proposal that became a valid candidate (``Run.attempt``, so every
        strategy's proposals): ``candidate``, or an ``E-integrity`` refusal when the
        verdict is ``exploit`` or ``suspect`` about what the change adds and names a
        category whose policy is ``veto`` (`optim.integrity.vetoes`); an inherited
        finding (unless one of its quotes is text this change added,
        `optim.integrity.backstop`) or a failed judgement never is.
        The verdict is journaled
        (``integrity_verdict``), kept for the ledger, the next request and the report.
        Off, nothing is called or recorded. A candidate an earlier judgement let through
        (a strategy rebuilding it on resume) is not judged again."""
        if not self._integrity_on():
            return candidate
        from refactory_dsh.optim import integrity

        if any(r.get("candidate") == candidate.id and r.get("scope") == "proposal"
               and not r.get("vetoed") for r in self._integrity):
            return candidate
        policy, allowances = self._integrity_policy()
        if not integrity.asked(policy):
            # Every category is off: nothing to ask about.
            return candidate
        self._journal_integrity_policy()
        diagnosis = (str(getattr(self.optimizer, "last_diagnosis", "") or "")
                     if self._custom_strategy is None else "")
        # What the judge reads, journaled with its verdict so a past run's proposals,
        # vetoed ones included, can be judged again offline with the same prompt.
        given = {"edits": candidate.own_edits(), "rationale": str(proposal.rationale or ""),
                 "prediction": str(proposal.prediction or ""), "diagnosis": diagnosis}
        if not any(c.id == parent.id for c in self.population):
            # A parent the journal does not hold (a strategy's in-between candidate,
            # SkillOpt's per-knob patches): its state goes with the record.
            given["parent_state"] = parent.state
        try:
            prompt = integrity.proposal_prompt(
                edits=given["edits"], parent_state=parent.state, state=candidate.state,
                rationale=given["rationale"], prediction=given["prediction"],
                diagnosis=diagnosis,
                # The records the proposer read: the parent's train records, never val or test.
                records=list(self._train_item_records.get(parent.id) or ()),
                mode_name=self.mode_id, parent_id=parent.id, candidate_id=candidate.id,
                categories=integrity.asked(policy), allowances=allowances,
                seed_state=self._seed_state())
        except Exception as exc:  # a prompt that cannot be built is a failed judgement
            error = f"{type(exc).__name__}: {exc}"
            self._say_judge_error(error)
            found = {"verdict": None, "categories": [], "evidence": [], "reason": "",
                     "key": None, "error": f"the judge's prompt could not be built: {error}"}
        else:
            found = self._judge("proposal", prompt, candidate=candidate.id, parent=parent.id)
        # A finding labelled inherited whose quote this change added is about what it adds
        # (`optim.integrity.backstop`): the label alone does not spare it.
        found = integrity.backstop(found, policy, parent.state, candidate.state)
        # Only an exploit or suspect verdict about what this change adds, naming a category
        # whose policy is veto; an inherited finding or a failed judgement is shown, never
        # vetoed.
        vetoed = integrity.vetoes(found, policy)
        record = {"scope": "proposal", "candidate": candidate.id, "parent": parent.id,
                  "epoch": self.epoch, "generation": self.generation,
                  "vetoed": vetoed, **found, "input": given}
        with self._state_lock:
            self._remember_integrity(record)
            self._journal("integrity_verdict", **record)
        self._progress("integrity", candidate=candidate.id, verdict=found.get("verdict"),
                       categories=found.get("categories"), vetoed=vetoed,
                       error=found.get("error"))
        return [integrity.refusal(found, policy)] if vetoed else candidate

    # ── the wiring review (``Run.attempt``, every strategy's proposals) ──────────
    def _wiring_check(self, run: Any, proposal: Proposal, parent: Candidate,
                      candidate: Candidate) -> Candidate:
        """Show the proposer the wiring warnings its edit adds (`_new_wiring_warnings`:
        ``W-unread-output``, ``W-reloop-target``) before the candidate is judged, smoked
        or evaluated, and let it act on them: one slim call (`Optimizer.review_wiring`)
        answers with fixed edits, or with none to keep the change as it is, saying why.
        A warning is information, never a refusal (a Python op may read a file it does
        not declare): the candidate is kept unless the proposer's fix passes every check
        a proposal must (``Run.mechanics``), and then the fix is the candidate.

        Said either way: a note on the candidate (shown with it in later requests), the
        optimizer trace (``wiring_review``) and the journal (``wiring_review``, so a
        resumed run asks no candidate twice). An optimizer that makes no model call
        (`Optimizer.review_wiring` returns ``None``) keeps the change, with the note.
        A text-only edit changes no step's wiring, so it is never asked about."""
        held = self._wiring_reviews.get(candidate.id)
        if held is not None:
            # Reviewed before (a resumed run): the outcome the journal holds, with no new
            # call -- a fix is rebuilt from its journaled edits, as it was accepted.
            if (held.get("outcome") == "fixed" and held.get("id") == candidate.id
                    and held.get("fixed_edits")):
                again = run.mechanics(Proposal(edits=tuple(held["fixed_edits"]),
                                               rationale=proposal.rationale,
                                               prediction=proposal.prediction,
                                               parent=proposal.parent), parent)
                if isinstance(again, Candidate):
                    again.notes = [*again.notes, "wiring warnings shown before it ran: the "
                                   "proposer changed the wiring before it ran (journaled)"]
                    return again
            return candidate
        warnings = _new_wiring_warnings(candidate.state, parent.state)
        if not warnings:
            return candidate
        review = getattr(self.optimizer, "review_wiring", None)
        fixed, why = None, ""
        if callable(review):
            fixed = review(self._repair_context(), proposal, warnings, parent)
            why = str(getattr(self.optimizer, "last_wiring_answer", "") or "")
        outcome, refused, result = "kept", [], candidate
        fixed_edits: list[Any] = []
        if fixed is not None:
            fixed_edits = list(fixed.steps)
            again = run.mechanics(fixed, parent)
            if isinstance(again, Candidate):
                outcome, result = "fixed", again
            else:
                outcome, refused = "fix_refused", list(again)
        codes = ", ".join(dict.fromkeys(w.split(" ", 1)[0] for w in warnings))
        said = {"fixed": "the proposer changed the wiring before it ran",
                "fix_refused": "the proposer's fix was refused, so the change runs as "
                               f"proposed ({'; '.join(refused)[:300]})",
                "kept": "the proposer kept the change as it is"}[outcome]
        result.notes = [*result.notes, f"wiring warnings shown before it ran ({codes}): {said}"
                        + (f"; it said: {why[:300]}" if why and outcome != "fix_refused" else "")]
        remaining = _new_wiring_warnings(result.state, parent.state)
        record = {"id": candidate.id, "became": result.id if result is not candidate else None,
                  "parent": parent.id, "epoch": self.epoch, "warnings": warnings,
                  "outcome": outcome, "why": why, "refused": refused, "remaining": remaining,
                  **({"fixed_edits": fixed_edits} if outcome == "fixed" else {})}
        with self._state_lock:
            self._wiring_reviews[candidate.id] = record
            if result is not candidate:
                self._wiring_reviews[result.id] = record
            self._journal("wiring_review", **record)
        self.optimizer.record_call(kind="wiring_outcome", epoch=self.epoch, parent=parent.id,
                                   candidate=candidate.id, became=record["became"],
                                   outcome=outcome, warnings=warnings, remaining=remaining,
                                   refused=refused)
        self._progress("wiring", candidate=candidate.id, parent=parent.id, outcome=outcome,
                       warnings=len(warnings), remaining=len(remaining))
        return result

    # ── the smoke check (``Run.attempt``, the default loop's own proposals) ───────
    def _smoke_items(self, parent: Candidate) -> list[str]:
        """The train items a child of ``parent`` is smoked on: the first (by item id) its
        parent answered wrong (no error, not a harness timeout or incident) and the first it
        passed, from the parent's latest train records; fewer when it has neither. By id,
        so the pick does not follow the order a seed shuffled the items into."""
        from refactory_dsh.optim.base import harness_timeout

        records = sorted((r for r in self._train_item_records.get(parent.id) or ()
                          if not r.get("not_scored") and not harness_timeout(r)),
                         key=lambda r: (str(r.get("id")), int(r.get("repeat") or 1)))
        wrong = next((r for r in records if not r.get("passed") and not r.get("error_text")),
                     None)
        right = next((r for r in records if r.get("passed")), None)
        return list(dict.fromkeys(str(r.get("id")) for r in (wrong, right) if r is not None))

    def _smoke_check(self, parent: Candidate, candidate: Candidate, *,
                     item_gate: threading.BoundedSemaphore | None = None,
                     stop_event: threading.Event | None = None,
                     ) -> Candidate | list[str]:
        """Run a new candidate on up to two train items (`_smoke_items`) before it is
        evaluated: ``candidate``, or an ``E-smoke`` refusal when an item failed with an
        error (`optim.base.pipeline_item_errors`) or the run failed before any item. A
        mechanics refusal: the slim repair call may fix it, and its fix is smoked again.

        One trial, gold feedback off, the seed the candidate's train run will use, in
        ``runs/<id>/occasions/smoke_<n>``. An interrupted run, a harness timeout or
        incident, and a spent budget are inconclusive and pass; with no evaluation left
        (``max_evaluations``), or no room in ``max_rollouts``, it does not run. Its records
        are no train records (no item history, no traces); its items count as rollouts,
        not as an evaluation. Journaled (``smoke``), so a resumed run never smokes a graph twice.
        The default loop's proposals only: a custom strategy (SkillOpt) is not smoked.

        With ``batch_size`` > 1, ``install`` defers the check (``_defer_smoke``) and
        runs the batch through ``_smoke_check_batch``, so a round's smoke checks overlap
        under the run-wide item limit.

        ``item_gate`` and ``stop_event`` are forwarded to ``_measure`` when the check
        runs under ``_smoke_check_batch``.

        The ``max_rollouts`` check and reservation are atomic under ``_state_lock``,
        so concurrent smoke threads cannot overshoot the cap."""
        if self._custom_strategy is not None:
            return candidate
        if self._defer_smoke:
            return candidate
        held = self._smoke_verdicts.get(candidate.id)
        if held is not None:
            return [self._smoke_refusal(held)] if held.get("outcome") == "failed" else candidate
        items = self._smoke_items(parent)
        if not items or self.evaluations >= self.config.max_evaluations:
            return candidate
        with self._state_lock:
            cap = self.config.max_rollouts
            if cap is not None and self.rollouts + len(items) > cap:
                return candidate
            self.rollouts += len(items)
            number = self._smoke_runs.get(candidate.id, 0) + 1
        seed = self._train_seed(self.epoch + 1)
        workdir = self.root / "runs" / candidate.id / "occasions" / f"smoke_{number}"
        self._archive_unfinished(candidate.id, workdir)
        task_module.make_manifest(workdir, train_ids=items)
        measure_kw: dict[str, Any] = {}
        if item_gate is not None:
            measure_kw["item_gate"] = item_gate
        if stop_event is not None:
            measure_kw["stop_event"] = stop_event
        result, measurements = self._measure(candidate, workdir, task_module.TRAIN, seed=seed,
                                             items=items, repeats=1, traces=False, **measure_kw)
        outcome, errors = self._smoke_outcome(result, measurements)
        rolled = ((len(result.item_records or ()) or 1)
                  if measurements.get("count") and not measurements.get("cached_from") else 0)
        record = {"id": candidate.id, "parent": parent.id, "epoch": self.epoch,
                  "items": items, "seed": seed, "outcome": outcome, "errors": errors,
                  **({"error": result.error} if outcome == "failed" and not errors else {}),
                  "rollouts": rolled, "measurements": measurements}
        with self._state_lock:
            self._smoke_runs[candidate.id] = number
            self._smoke_verdicts[candidate.id] = record
            self.rollouts += rolled - len(items)
            self._journal("smoke", **record)
            self._journal("rollouts", rollouts=self.rollouts)
            self._progress("smoke", candidate=candidate.id, parent=parent.id, items=len(items),
                           outcome=outcome, errors=errors, error=record.get("error"))
        return [self._smoke_refusal(record)] if outcome == "failed" else candidate

    def _smoke_outcome(self, result: Any, measurements: Mapping[str, Any]
                       ) -> tuple[str, list[dict[str, Any]]]:
        """``("failed", errors)``, ``("passed", [])`` or ``("inconclusive", [])`` for a
        smoke run (`_smoke_check`)."""
        from refactory_dsh.optim.base import pipeline_item_errors

        if (measurements.get("budget_stopped") or not measurements.get("count")
                or self._interruption(result)):
            return "inconclusive", []
        errors = pipeline_item_errors(result.item_records or ())
        if errors:
            return "failed", errors
        if not result.exit_ok and not result.item_records:
            return "failed", []
        return "passed", []

    @staticmethod
    def _smoke_refusal(record: Mapping[str, Any]) -> str:
        """The ``E-smoke`` reason: what failed, where, on which item."""
        from refactory_dsh.optim.base import _clip

        items = len(record.get("items") or ())
        said = "; ".join(
            f"{e.get('kind')}" + (f" at node {e['node']}" if e.get("node") else "")
            + f" (item {e.get('item')}): {e.get('first')}"
            for e in list(record.get("errors") or ())[:2])
        return (f"{SMOKE_CODE}: a smoke run on {items} train item(s) the parent ran without an "
                "error failed: " + (said or f"before any item: {_clip(record.get('error'), 300)}"))

    def _smoke_check_batch(
        self,
        run: Run,
        candidates: list[Candidate],
        smoke_meta: dict[str, tuple[Proposal, Candidate]],
    ) -> list[Candidate]:
        """Smoke-check a batch of candidates concurrently, under the run-wide item limit.

        Called by ``install`` when ``batch_size`` > 1 after every candidate passed
        validation and integrity without being smoked. Each candidate's check runs
        in its own thread through the same ``ThreadPoolExecutor`` machinery
        ``Run.evaluate_many`` uses, sharing the item gate and stop event.

        Each thread's ``_smoke_check`` does its own atomic rollout check
        (check + increment under ``_state_lock``), so concurrent threads
        cannot overshoot ``max_rollouts``.

        A candidate that fails is refused ``E-smoke``: its rejection is recorded and
        the slim repair call runs with the error (``_repair``); a repaired candidate is
        smoked individually (``_defer_smoke`` is off by then).  The returned list keeps
        the input order, with failed-and-not-repaired candidates removed.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from refactory_dsh.owned_process import StoppableGate

        if not candidates:
            return candidates
        to_smoke = [(c, smoke_meta[c.id]) for c in candidates if c.id in smoke_meta]
        if not to_smoke:
            return candidates

        seed_candidate = self.population[0] if self.population else candidates[0]
        self._init_item_gate(seed_candidate)
        gate = getattr(self, '_item_gate', None)
        stop_event = threading.Event()
        stoppable_gate = StoppableGate(gate, stop_event) if gate is not None else None

        results: list[Candidate | list[str] | None] = [None] * len(to_smoke)
        errors: list[BaseException | None] = [None] * len(to_smoke)

        def _run_one(index: int) -> None:
            cand, (_, parent) = to_smoke[index]
            if stop_event.is_set():
                results[index] = cand
                return
            try:
                results[index] = self._smoke_check(
                    parent, cand, item_gate=stoppable_gate, stop_event=stop_event)
            except BaseException as exc:
                errors[index] = exc

        max_workers = int(self.config.max_concurrent_evaluations or 4)
        if len(to_smoke) == 1:
            _run_one(0)
        else:
            from refactory_dsh.dsh_runtime import shutdown_tracked_runs
            from refactory_dsh.owned_process import _termination_as_exit

            with _termination_as_exit():
                pool = ThreadPoolExecutor(
                    max_workers=min(max_workers, len(to_smoke)))
                futures = [pool.submit(_run_one, i) for i in range(len(to_smoke))]
                try:
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except BaseException:
                            pass
                except BaseException:
                    stop_event.set()
                    for f in futures:
                        f.cancel()
                    shutdown_tracked_runs()
                    raise
                finally:
                    pool.shutdown(wait=True)

        first_error = next((e for e in errors if e is not None), None)
        if first_error is not None:
            raise first_error

        passed_ids: set[str] = set()
        repaired: dict[str, Candidate] = {}
        for i, (cand, (proposal, parent)) in enumerate(to_smoke):
            outcome = results[i]
            if outcome is cand or isinstance(outcome, Candidate):
                passed_ids.add(cand.id)
            elif isinstance(outcome, list):
                self.rejections.append(
                    Rejection(attempt=1, edit=proposal.edit, reasons=outcome,
                              rationale=proposal.rationale))
                fixed = self._repair(run, proposal, parent, outcome)
                if isinstance(fixed, Candidate):
                    repaired[cand.id] = fixed
                    notes = self._proposal_notes.get(cand.id) or {}
                    self._proposal_notes[fixed.id] = {
                        **notes, "repaired_from": list(outcome)}

        final: list[Candidate] = []
        for cand in candidates:
            if cand.id in passed_ids:
                final.append(cand)
            elif cand.id in repaired:
                final.append(repaired[cand.id])
            elif cand.id not in smoke_meta:
                final.append(cand)
        return final

    def _integrity_of(self, candidate_id: str) -> dict[str, Any] | None:
        """The latest proposal verdict on ``candidate_id`` (the change ledger's column);
        ``None`` with the judge off or no verdict."""
        if not self._integrity_on():
            return None
        # The verdict that let it through to evaluation (an earlier proposal of the same
        # state may have been vetoed).
        mine = [r for r in self._integrity
                if r.get("scope") == "proposal" and r.get("candidate") == candidate_id]
        return next((r for r in mine if not r.get("vetoed")), mine[-1] if mine else None)

    def _seed_state(self) -> dict[str, Any] | None:
        """The pipeline the search started from (the population's first candidate)."""
        seed = next((c for c in self.population if not c.parent), None)
        return seed.state if seed is not None else None

    def _integrity_feedback(self) -> list[dict[str, Any]]:
        """The run's proposal verdicts, for "Integrity review of your proposals"; none
        with the judge off."""
        if not self._integrity_on():
            return []
        return [r for r in self._integrity if r.get("scope") == "proposal"]

    def _integrity_audit(self, best: Candidate | None) -> dict[str, Any]:
        """The winner's combined change from the seed, judged once (an exploit can be
        assembled across several edits that each passed): the seed-to-winner state
        diff, the accepted changes in order with their rationales, and the winner's
        train records. Journaled (scope ``final``). Called while the run executes; a
        report-only replay reads the journaled verdict or says it was not judged."""
        from refactory_dsh.optim import integrity

        if best is None:
            return {"note": "no winner to audit"}
        by_id = {c.id: c for c in self.population}
        chain: list[Candidate] = []
        node: Candidate | None = best
        while node is not None and node.id not in {c.id for c in chain}:
            chain.append(node)
            node = by_id.get(node.parent or "")
        chain.reverse()
        seed = chain[0]
        if seed.id == best.id:
            return {"candidate": best.id, "note": "the winner is the seed: nothing changed"}
        policy, allowances = self._integrity_policy()
        if not integrity.asked(policy):
            return {"candidate": best.id, "note": "every category is off: nothing to judge"}
        steps = [{"id": c.id, "edits": c.own_edits(),
                  "rationale": str((self.proposals.get(c.id) or {}).get("rationale") or "")}
                 for c in chain[1:]]
        prompt = integrity.cumulative_prompt(
            seed_state=seed.state, state=best.state, steps=steps,
            records=list(self._train_item_records.get(best.id) or ()),
            mode_name=self.mode_id, candidate_id=best.id,
            categories=integrity.asked(policy), allowances=allowances)
        done = next((r for r in reversed(self._integrity) if r.get("scope") == "final"
                     and r.get("candidate") == best.id and r.get("key") == integrity.key(prompt)
                     and r.get("verdict")), None)
        if done is not None:
            return done
        if not self.audit_on_report:
            return {"candidate": best.id, "verdict": None,
                    "error": "not judged: a report-only replay makes no model call"}
        self._journal_integrity_policy()
        found = self._judge("final", prompt, candidate=best.id, seed=seed.id)
        # Reported, never vetoed: the search is over (a person reads it before adopting).
        record = {"scope": "final", "candidate": best.id, "seed": seed.id,
                  "steps": len(steps), "epoch": self.epoch, "vetoed": False, **found}
        with self._state_lock:
            self._remember_integrity(record)
            self._journal("integrity_verdict", **record)
        return record

    def _integrity_report(self, best: Candidate | None) -> dict[str, Any]:
        """The report's ``integrity``: the proposal verdicts by kind, the flagged ones,
        and the winner's combined-change audit (``_integrity_audit``)."""
        proposals = [r for r in self._integrity if r.get("scope") == "proposal"]
        brief = ("candidate", "parent", "epoch", "verdict", "categories", "evidence",
                 "reason", "vetoed", "error")
        policy, allowances = self._integrity_policy()
        setting = self._integrity_setting() or {}
        return {
            "mode": json.dumps(self.config.integrity_judge) if isinstance(
                self.config.integrity_judge, dict) else str(self.config.integrity_judge),
            "default": setting.get("default"),
            "policy": policy,
            "allowances": allowances,
            "proposals": {
                "reviewed": len(proposals),
                **{verdict: sum(1 for r in proposals if r.get("verdict") == verdict)
                   for verdict in ("clean", "suspect", "exploit")},
                "errors": sum(1 for r in proposals if not r.get("verdict")),
                "vetoed": sum(1 for r in proposals if r.get("vetoed")),
            },
            "flagged": [{name: r.get(name) for name in brief} for r in proposals
                        if r.get("verdict") in ("suspect", "exploit") or r.get("error")],
            "final_audit": self._integrity_audit(best),
        }

    def _judged(self) -> list[Candidate]:
        """The evaluated candidates the search made (every one with a parent), in
        population order; under ``val_gate`` only those the gate has judged."""
        out = []
        for candidate in self.population:
            if not candidate.parent or not candidate.evaluated:
                continue
            if self._gate_decision(candidate.id) == "not gated":
                continue
            out.append(candidate)
        return out
