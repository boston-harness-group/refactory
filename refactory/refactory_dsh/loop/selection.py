
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

from refactory_dsh.loop.config import EpochSummary, NOT_APPLIED, NO_VALIDATION, _item_sources, _train_key
from refactory_dsh.loop.report import _codes, _edit_record, _yield



class _SelectionMixin:

    def loss_step(self) -> dict[str, Any]:
        evaluated = self._evaluated()
        scored = [c for c in evaluated if c.exit_ok and c.score is not None]
        scores = [c.score for c in scored]
        raw_scores = [c.raw_score for c in scored if c.raw_score is not None]
        best = max(scores) if scores else None
        mean = (sum(scores) / len(scores)) if scores else None
        # What this epoch is compared with: the last epoch's best, or for the first epoch
        # the seed's score, the point the search started from (dm-265).
        previous = self.loss.get("best") if self.loss else self._seed_score()
        self.loss = {
            "epoch": self.epoch,
            "batch": len(evaluated),
            # The scores the search compared, and the evaluator's own numbers beside
            # them, so an epoch's numbers can be read either way.
            "scores": scores,
            "raw_scores": raw_scores,
            "mean": mean,
            "best": best,
            "raw_best": max(raw_scores) if raw_scores else None,
            "parsimony": self.config.parsimony,
            "best_ever": max([s for s in (previous, best) if s is not None], default=None),
            "improved": bool(best is not None and previous is not None and best > previous),
            "run_failures": [c.id for c in self.population if c.evaluated and not c.exit_ok],
        }
        self._write("loss.json", self.loss)
        self._journal("loss", **self.loss)
        self._progress("loss", **self.loss)
        return self.loss

    # ── backward ───────────────────────────────────────────────────────────
    def backward(self) -> dict[str, Any]:
        """Turn the traces into the gradient the optimizer reads."""
        evaluated = [c for c in self._evaluated() if c.exit_ok and c.score is not None]
        self.gradient = {
            "epoch": self.epoch,
            "scoring": self.scoring(),
            "knob_attribution": self._knob_attribution(evaluated),
            "cost": self._cost(),
            "measurement_summary": self._measurement_summary(),
            "run_failures": self.loss.get("run_failures", []),
            "rejection_codes": self._rejection_codes(),
            "rejection_reasons": [r for rej in self.rejections[-8:] for r in rej.reasons],
            "accepted_op_mix": self._op_mix(),
            "structural": self._structural_stats(),
            "leaderboard": [self._row(c) for c in self.leaderboard(8)],
        }
        self._write("gradient.json", self.gradient)
        self._progress(
            "backward",
            knobs=len(self.gradient["knob_attribution"]),
            failures=len(self.gradient["run_failures"]),
        )
        return self.gradient

    def _knob_attribution(self, evaluated: Sequence[Candidate]) -> dict[str, Any]:
        """What each parameter value was worth, over everything evaluated so far.

        The score attributed here is the one the search compares -- the evaluator's
        number minus the run's size cost -- so a value is credited with what choosing
        it did to the search's objective rather than to the raw measurement.

        ``untried`` is part of the gradient, not decoration: an optimizer reading
        only the values that have run has no way to tell a value that failed from
        one nobody has tried, and will keep re-proposing what it already knows.
        """
        buckets: dict[str, dict[str, list[float]]] = {}
        # With the validation gate, what it decided about the candidates behind each
        # value, so a value whose candidates were all rejected is not read as a gain.
        decisions: dict[str, dict[str, dict[str, int]]] = {}
        # Each knob's declaration as the candidates carry it (the latest wins), so a
        # knob an edit added is enumerated like one the mode declares.
        specs: dict[str, dict[str, Any]] = {}
        # A frozen knob (the mode's run settings, or one the operator froze) is not the
        # optimizer's to move, so its values are not attributed, nor its untried ones offered.
        # The candidate each value was first evaluated in, which names a long text value
        # in a request (`optim.base._attribution_key`).
        first: dict[str, dict[str, str]] = {}
        frozen = set(self.config.frozen) | {
            name for name, knob in self.parameters.items() if getattr(knob, "frozen", False)}
        # Under ``checkpoint`` a change is not rejected: it is applied or not.
        refused = NOT_APPLIED if self.selection_mode() == "checkpoint" else "rejected"
        for candidate in evaluated:
            decision = self._gate_decision(candidate.id)
            for name, entry in (candidate.state.get("knobs") or {}).items():
                if isinstance(entry.get("spec"), dict):
                    specs[name] = entry["spec"]
                    if entry["spec"].get("frozen"):
                        frozen.add(name)
                first.setdefault(name, {}).setdefault(str(entry["value"]), candidate.id)
                buckets.setdefault(name, {}).setdefault(str(entry["value"]), []).append(
                    float(candidate.score)
                )
                if decision is not None:
                    counts = decisions.setdefault(name, {}).setdefault(
                        str(entry["value"]), {"kept": 0, refused: 0, "not gated": 0})
                    counts["kept" if decision in ("kept", "current") else decision] += 1
        attribution: dict[str, Any] = {}
        for name, values in sorted(buckets.items()):
            if name in frozen:
                continue
            attribution[name] = {
                "tried": {
                    value: {
                        "n": len(scores),
                        "mean": round(sum(scores) / len(scores), 6),
                        "max": round(max(scores), 6),
                        "first": first[name][value],
                        **({"gate": decisions[name][value]} if name in decisions else {}),
                    }
                    for value, scores in sorted(values.items())
                },
                "untried": self._untried(name, values, specs.get(name)),
            }
        return attribution

    def _untried(self, name: str, tried: dict[str, list[float]],
                 spec: Mapping[str, Any] | None = None) -> list[Any]:
        """The declared values of a knob that have not been evaluated.

        A knob with a finite set -- an enum, or a ``str`` with declared values --
        enumerates it. An ``int`` with both bounds declared is just as finite and
        just as readable, so it enumerates its range too. An open-ended numeric
        knob has no set to enumerate and answers empty. ``spec`` is the declaration
        the candidates carry, read before the mode's: a knob an edit added has no
        declaration in the mode.
        """
        knob = Knob.from_spec(dict(spec)) if spec else self.parameters.get(name)
        if knob is None:
            return []
        finite = knob.accepted()
        if finite:
            return [member for member in finite if str(member) not in tried]
        if knob.kind == "int" and knob.lo is not None and knob.hi is not None:
            return [
                value
                for value in range(int(knob.lo), int(knob.hi) + 1)
                if str(value) not in tried
            ]
        return []

    def _efficiency(self) -> dict[str, Any]:
        """How many attempts it took per candidate that could actually run."""
        proposed = len(self.proposals)
        attempts = sum(self.attempts_used)
        return {
            "rounds": attempts,
            "batches": len(self.attempts_used),
            "effective_k": sorted(set(self.attempts_used)),
            "candidates": proposed,
            "attempts_per_candidate": (
                round(attempts / proposed, 3) if proposed else None
            ),
            "why": "k starts at the configured attempts and rises by one when the "
                   "previous batch was refused more than it was accepted, because "
                   "that is the signal that the optimizer is close to something "
                   "legal rather than out of ideas",
        }

    def _proposer_health(self) -> dict[str, Any]:
        """Whether the proposer answered at all, said where a reader will see it.

        A failed round is caught and the run continues, which is right: a model
        that is down should not kill a run. But the run then looks exactly like one
        where the model answered and had nothing useful, because both end with a
        score that did not move and no rejections. That reads as a finding about
        the mode when it is a fact about the machine, so the count is reported and
        a run that never got an answer says so in one line.
        """
        failures = list(getattr(self.optimizer, "proposal_failures", []))
        rounds = int(getattr(self.optimizer, "proposal_rounds", 0))
        all_failed = bool(rounds) and len(failures) >= rounds
        if all_failed:
            finding = (
                f"every one of the {rounds} proposal round(s) failed, so this run is "
                "not a search result: nothing was proposed, and the score did not "
                "move because the proposer never answered."
            )
        elif failures:
            finding = (
                f"{len(failures)} of {rounds} proposal round(s) failed; the rest "
                "were proposed and evaluated normally."
            )
        else:
            finding = ""
        return {
            "optimizer": self.optimizer.name,
            "rounds": rounds,
            "failures": failures[:5],
            "failure_count": len(failures),
            "all_failed": all_failed,
            "finding": finding,
        }

    def _dominant_refusal(self) -> dict[str, Any]:
        """The commonest refusal, and what it says about the catalogue.

        A rule that refuses most of what the optimizer proposes is a finding about
        the ops it was offered, not about the optimizer. Counting the codes and
        stopping there leaves that unsaid.
        """
        counts = self._rejection_codes()
        if not counts:
            return {"code": None, "count": 0, "share": 0.0, "finding": ""}
        code, count = max(counts.items(), key=lambda item: item[1])
        total = sum(counts.values())
        share = round(count / total, 3)
        finding = ""
        if share >= 0.5 and code.startswith("E-"):
            finding = (
                f"{share:.0%} of refusals were {code}. That is a statement about the "
                "operations the optimizer was offered, not about the optimizer: the "
                "catalogue asks for something these proposals cannot satisfy, so the "
                "field descriptions or the preconditions are the thing to fix."
            )
        elif share >= 0.5 and code.startswith("V"):
            finding = (
                f"{share:.0%} of refusals were {code}. The optimizer is proposing edits "
                "that break a rule it does not have to guess at -- the rules are in its "
                "prompt -- so either the mode's graph invites that edit or the rule needs "
                "stating more plainly."
            )
        return {"code": code, "count": count, "share": share, "finding": finding}

    def _convergence(self) -> dict[str, Any]:
        """Whether the optimizer has run out of ideas within the surface it sees.

        A no-op or a duplicate means the same thing: every edit it can think of
        lands on a point that is already the case. That is a convergence symptom,
        not one more refusal.
        """
        counts = self._rejection_codes()
        spent = {code: counts.get(code, 0) for code in ("E-noop", "E-duplicate")}
        total = sum(spent.values())
        if not total:
            return {"reason": None, "detail": ""}
        accepted = sum(1 for c in self.population if c.edits)
        if total > accepted:
            return {
                "reason": "surface_exhausted",
                "detail": (
                    f"{total} proposal(s) landed on a state that already exists and "
                    f"{accepted} produced something new, so the optimizer is out of "
                    "ideas within the surface it can see; the surface needs widening, "
                    "not more attempts"
                ),
            }
        return {"reason": None, "detail": "", "noops": total}

    def _cost(self) -> dict[str, Any]:
        durations = [c.duration_s for c in self._evaluated()]
        if not durations:
            return {"n": 0}
        calls = sum(c.model_calls for c in self._evaluated())
        return {
            "n": len(durations),
            "mean_s": round(statistics.fmean(durations), 4),
            "max_s": round(max(durations), 4),
            "total_s": round(sum(durations), 4),
            "inner_model_calls": calls,
        }

    def _measurement_summary(self) -> dict[str, Any]:
        """The evaluations' trials in a few numbers, for the optimizer: how many ran,
        how many failed, and the failures by cause. No receipts and no paths."""
        trials = [trial for c in self._evaluated()
                  for trial in (c.measurements or {}).get("trials") or ()]
        causes: dict[str, int] = {}
        for trial in trials:
            if not trial.get("exit_ok"):
                cause = " ".join(str(trial.get("error") or "unknown").split())[:120]
                causes[cause] = causes.get(cause, 0) + 1
        return {
            "trial_runs": len(trials),
            "failed": sum(causes.values()),
            "timeouts": sum(n for cause, n in causes.items() if "timeout" in cause.lower()
                            or "timed out" in cause.lower()),
            "failures_by_cause": dict(sorted(causes.items(), key=lambda kv: -kv[1])[:6]),
        }

    def _rejection_codes(self) -> dict[str, int]:
        """Each rule's refusals that stood: a refusal a slim repair fixed is counted in
        `_repaired_codes` instead."""
        return _codes([r for r in self.rejections if not r.repaired])

    def _repaired_codes(self) -> dict[str, int]:
        """Each rule's refusals a slim repair call fixed (`optim.repair`)."""
        return _codes([r for r in self.rejections if r.repaired])

    def _op_mix(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.population:
            for edit in candidate.edits:
                op = str(edit.get("op", "?"))
                counts[op] = counts.get(op, 0) + 1
        return dict(sorted(counts.items()))

    def _structural_stats(self) -> dict[str, Any]:
        structural = [c for c in self.population if c.is_structural()]
        knob_only = [c for c in self.population if not c.is_structural()]
        structural_rejections = [
            r for r in self.rejections
            if not r.repaired and any(isinstance(step, dict) and nn_edits.is_macro(step.get("op", ""))
                   for step in Proposal(edit=r.edit).steps)
        ]
        return {
            "structural_candidates": len(structural),
            "knob_candidates": len(knob_only),
            "structural_refusals": len(structural_rejections),
            "structural_yield": _yield(len(structural), len(structural_rejections)),
            "accepted_ops": self._op_mix(),
        }

    # ── optimizer ──────────────────────────────────────────────────────────
    def optimizer_step(
        self, run: Run | None = None
    ) -> tuple[list[Candidate], list[Rejection]]:
        """Propose a batch and turn it into candidates, through the overridable steps.

        Going through ``propose`` and ``install`` rather than straight to the
        optimizer is what lets a strategy change one of them and keep the other.
        """
        run = run or Run(_loop=self)
        # The pair a proposer is handed, and why it is the whole population rather
        # than the elite: `describe()` carries a score and the edits that produced
        # it, not the state, so a candidate the loop does not hand over cannot be
        # bred from. Passing only the elite made parent selection impossible inside
        # an optimizer, which is why every SRF method that selects parents had to be
        # a loop instead. Selection is part of "how to search", so the optimizer gets
        # the pool and chooses from it (issue #32).
        parents = [candidate for candidate in self.population if candidate.evaluated]
        if self.config.val_gate:
            parents = self._gated_parents()
        if not parents:
            parents = self.select_parents(self.config.elite)
        if not parents:
            return [], []
        # The parent this step edits, so ProposalContext.item_records carries its
        # records, not board[0]'s (brief ask 5). Recomputed every step: the leader
        # moves between steps, and a parent fixed at the first step would hand every
        # later proposal a stale candidate's records. A strategy may name the parent
        # in strategy_decision; otherwise select_parents(1) -- which respects kept
        # candidates and is overridable -- decides. Kept out of strategy_decision,
        # which reaches the proposal as the strategy's own choice.
        explicit = self.strategy_decision.get("parent")
        # An optimizer that picks its own parent (a parent picker paired with a
        # proposer) picks it here, before the request is built, so the request
        # describes the candidate its proposals are applied to.
        # When the round will be split into per-slot calls, skip the pick here:
        # _propose_per_slot picks per slot, and an extra pick here would waste
        # a picker call and advance the picker's RNG/explore state.
        n = self.batch_size(run)
        will_split = self._split_proposal_calls(n)
        choose = getattr(self.optimizer, "choose_parent", None)
        picked = (None if explicit or will_split or not callable(choose)
                  else choose(
            ProposalContext(epoch=self.epoch, generation=self.generation,
                            mode_name=self.mode_id, elite=int(self.config.elite),
                            val_scores=self._val_scores(),
                            val_item_scores=self._val_item_scores()), parents))
        if explicit:
            self._editing_parent = str(explicit)
        elif picked is not None:
            self._editing_parent = picked.id
        else:
            # A default for the round; _propose_per_slot overrides it per slot.
            selected = self.select_parents(1)
            self._editing_parent = selected[0].id if selected else None
        self._refresh_edited_records()
        # A retry's wait inside an optimizer call ends early on a person's stop.
        try:
            self.optimizer.stop_check = self._stop_requested
        except AttributeError:  # an optimizer that takes no attribute keeps plain waits
            pass
        self._memory_pass()
        before = len(self.rejections)
        self._levels_wanted, self._levels_done = self._round_levels(), []
        self._proposal_attempt = 0
        proposals = self.propose(run, parents, n)
        candidates = self.install(run, proposals, parents)
        rejections = self.rejections[before:]
        self._round_refusals.setdefault(self.epoch, []).extend(rejections)
        for rejection in rejections:
            self._journal(
                "rejection",
                attempt=rejection.attempt,
                edit=rejection.edit,
                reasons=rejection.reasons,
                epoch=self.epoch,
                **({"repaired": rejection.repaired} if rejection.repaired else {}),
                **({"dropped": rejection.dropped} if rejection.dropped else {}),
            )
        known = {c.id: c for c in (*self.population, *parents)}
        for candidate in candidates:
            self._journal_candidate(candidate)
            notes = self._proposal_notes.get(candidate.id) or {}
            parent = known.get(candidate.parent or "")
            record = self.proposals.setdefault(candidate.id, {
                "id": candidate.id,
                "parent": candidate.parent,
                # Every edit its proposal applied: one, or a compound proposal's list.
                "edits": candidate.own_edits(),
                "rationale": str(notes.get("rationale") or ""),
                "prediction": str(notes.get("prediction") or ""),
                "confidence": float(notes.get("confidence") or 0.0),
                # A structural proposal's reasoning (`REASONING_FIELDS`), and which of it
                # was left out (marked, never refused).
                **({"addresses": str(notes["addresses"])} if notes.get("addresses") else {}),
                **({"flow_change": str(notes["flow_change"])} if notes.get("flow_change")
                   else {}),
                **({"reasoning_missing": list(notes["reasoning_missing"])}
                   if notes.get("reasoning_missing") else {}),
                "origin": "optimizer",
                # Its edits counted as the per-proposal cap counts them (`optim.edit_count`).
                "edit_count": self._edit_count(candidate),
                # The mechanical fixes the attempt path made, and what it could not check.
                **({"repairs": list(candidate.repairs)} if candidate.repairs else {}),
                **({"checks": list(candidate.notes)} if candidate.notes else {}),
                # What the edit built: its pipeline view's new and changed lines against
                # its parent's (`nn.pipeline_view`), whatever the prose says.
                **({"wiring": built} if parent is not None
                   and (built := pipeline_view.changed_lines(
                       parent.state, candidate.state, self.protected, self.graders)) else {}),
                **({"repaired_from": list(notes["repaired_from"])}
                   if notes.get("repaired_from") else {}),
                **({"level": notes["level"]} if notes.get("level") else {}),
                "diagnosis": str(notes.get("diagnosis")
                                 or getattr(self.optimizer, "last_diagnosis", "") or ""),
                # The kinds of change its own answer weighed, when it said (optional);
                # never the optimizer's latest, which may be a later retry's.
                **({"considered": str(notes["considered"])} if notes.get("considered")
                   else {}),
                "epoch": self.epoch,
            })
            # Journaled, so a resumed run still shows the optimizer its own reasoning.
            self._journal("proposal", **record)
        self._write_analysis()
        self._append_rejections(rejections)
        # A proposal a slim repair fixed was refused, then accepted as its fix: counted
        # once, among the accepted, and said as repaired (`optim.repair`).
        repaired = [r for r in rejections if r.repaired]
        still = [r for r in rejections if not r.repaired]
        self._progress(
            "optimizer",
            # This epoch's proposals, every round `install` asked for: each one was either
            # accepted for evaluation or refused, so the three counts add up (dm-265).
            proposed=len(candidates) + len(still),
            accepted=len(candidates),
            refused=len(still),
            repaired=len(repaired),
            repair_calls=self.optimizer.repairs_used(self.epoch)
            if hasattr(self.optimizer, "repairs_used") else 0,
            fixed=sum(1 for c in candidates if c.repairs),
            # Why this epoch's were refused, beside the count it explains.
            epoch_refusal_codes=_codes(still),
            **({"repaired_codes": _codes(repaired)} if repaired else {}),
            # The running tally, so a watcher sees which rule is dominating without
            # opening the run directory. Same derivation as the report's.
            refusal_codes=self._rejection_codes(),
            children=[c.id for c in candidates],
        )
        return candidates, rejections

    def _portfolio_report(self) -> dict[str, Any]:
        """What ``portfolio`` cost and produced, from the journaled proposal records.

        ``by_level``: candidates proposed, evaluated and kept by the validation gate
        (``kept``: accepted, under ``pool`` as the incumbent; with ``in_pool`` under
        ``pool``: in the pool now), and the evaluations they cost (a train run, plus
        each val or test occasion).
        ``rounds``: each round's candidates in the order they were proposed, with the
        gate's decision (``_gate_decision``) and whether it was ``accepted``; a
        candidate the gate did not accept although its val score beat its parent's,
        because a sibling of the same round scored higher on val (or tied and won the
        tie), is ``shadowed``.
        ``extra_evaluations``: what the candidates beyond ``batch_size`` in each round
        cost."""
        from refactory_dsh.optim.base import EDIT_LEVELS

        by_id = {c.id: c for c in self.population}
        order = {c.id: index for index, c in enumerate(self.population)}
        records = sorted((r for r in self.proposals.values() if r.get("level")),
                         key=lambda r: order.get(r["id"], len(order)))
        cache = self._gate().cache if self.config.val_gate and self._custom_strategy is None else {}

        def cost(candidate: Candidate | None) -> int:
            if candidate is None or not candidate.evaluated:
                return 0
            return 1 + sum((self._occasion_counters.get(candidate.id) or {}).values())

        pool = self.selection_mode() == "pool"
        accepted = set(self._gate_accepted)
        fields = ("proposed", "evaluated", "kept", *(("in_pool",) if pool else ()), "evaluations")
        by_level = {level: dict.fromkeys(fields, 0) for level in EDIT_LEVELS}
        rounds: dict[Any, list[dict[str, Any]]] = {}
        for record in records:
            candidate = by_id.get(record["id"])
            entry = by_level.setdefault(str(record["level"]), dict.fromkeys(fields, 0))
            entry["proposed"] += 1
            decision = self._gate_decision(record["id"], detail=True)
            if candidate is not None and candidate.evaluated:
                entry["evaluated"] += 1
            entry["evaluations"] += cost(candidate)
            # Accepted by the gate: under single, kept; under pool, made the incumbent
            # (a pool member that was never the incumbent was not accepted).
            kept = record["id"] in accepted
            if kept:
                entry["kept"] += 1
            if pool and decision in ("current", "in pool"):
                entry["in_pool"] += 1
            siblings = rounds.setdefault(record.get("epoch"), [])
            val, parent_val = cache.get(record["id"]), cache.get(record.get("parent") or "")
            shadowed = (record["id"] in self._gate_rejected and val is not None
                        and parent_val is not None and val > parent_val and any(
                            r["id"] in accepted
                            for r in records if r.get("epoch") == record.get("epoch")
                            and r["id"] != record["id"]))
            siblings.append({"id": record["id"], "level": record["level"],
                             "evaluated": bool(candidate is not None and candidate.evaluated),
                             "gate": decision, "kept": kept, "shadowed": shadowed,
                             "val": val, "parent_val": parent_val,
                             "evaluations": cost(candidate)})
        base = int(self.config.batch_size)
        return {"setting": self.config.portfolio, "batch_size": base,
                "selection": self.selection_mode(),
                "rounds": len(rounds), "by_level": by_level,
                "per_round": [{"epoch": epoch, "candidates": entries}
                              for epoch, entries in rounds.items()],
                "extra_evaluations": sum(entry["evaluations"] for entries in rounds.values()
                                         for entry in entries[base:])}

    def _selection_report(self) -> dict[str, Any]:
        """How this run selected, what its pool holds, and what the search cost.

        ``setting``: ``checkpoint``, ``single`` or ``pool`` under ``val_gate``, ``train``
        without it. ``pool``: the candidates the next round would choose parents from (the
        pool; under ``single`` and ``checkpoint`` the current candidate and what a person
        kept; without the gate the ``pool_size`` best by train score).
        ``lines_of_descent``: how many distinct lines the pool holds, a line being the
        seed's child a candidate descends from
        (the seed is a line of its own), with ``largest_line_share``. ``edit_kinds``:
        the pool's candidates by the level of the edit that made each one (``seed``,
        ``prompt_knob``, ``pipeline``). ``parents_from_incumbent``: of the candidates
        the run added, the share whose parent was the incumbent when they were added.
        ``budget``: train scorings (evaluations, re-scores included), val and test
        scorings, tokens and wall-clock, so arms are compared at equal cost rather
        than equal rounds.
        """
        from refactory_dsh.optim.base import edit_level

        setting = self.selection_mode() or ("train" if self._custom_strategy is None else "")
        # Without the gate a built-in strategy's run has no val items: it selects on train.
        warning = NO_VALIDATION if setting == "train" else ""
        if setting == "pool":
            members = self._judged_ranked() if self._population_policy() else self._pool()
        elif setting in ("single", "checkpoint"):
            members = self._gated_parents()
        else:
            members = self.leaderboard(int(self.config.pool_size))
        by_id = {c.id: c for c in self.population}

        def line(candidate: Candidate) -> str:
            seen: set[str] = set()
            while candidate.parent and candidate.parent in by_id and candidate.id not in seen:
                seen.add(candidate.id)
                parent = by_id[candidate.parent]
                if not parent.parent:
                    return candidate.id
                candidate = parent
            return candidate.id

        lines: dict[str, int] = {}
        kinds: dict[str, int] = {}
        for candidate in members:
            lines[line(candidate)] = lines.get(line(candidate), 0) + 1
            kind = edit_level(candidate.own_edits()) if candidate.parent else "seed"
            kinds[kind] = kinds.get(kind, 0) + 1
        choices = dict(self._strategy_state.get("parent_from_incumbent") or {})
        counters = [counts for counts in self._occasion_counters.values()]
        snapshot = self.budget.snapshot() if self.budget else {}
        return {
            "setting": setting,
            **({"warning": warning} if warning else {}),
            "val_gate": bool(self.config.val_gate),
            "val_gate_basis": self._val_gate_basis,
            "pool_size": int(self.config.pool_size) if setting == "pool" else None,
            # What ``pool`` lists: the top pool_size, or every judged candidate for a
            # population optimizer or picker.
            "parent_set": ("every judged candidate" if setting == "pool" and self._population_policy()
                           else setting),
            "pool": [c.id for c in members],
            "lines_of_descent": len(lines),
            "largest_line_share": (round(max(lines.values()) / len(members), 3)
                                   if members else None),
            "edit_kinds": dict(sorted(kinds.items())),
            "parents_from_incumbent": (round(sum(1 for v in choices.values() if v) / len(choices), 3)
                                       if choices else None),
            "children": len(choices),
            "budget": {
                "train_scorings": self.evaluations,
                "val_scorings": sum(int(c.get(task_module.VAL, 0)) for c in counters),
                "test_scorings": sum(int(c.get(task_module.TEST, 0)) for c in counters),
                "rollouts": self.rollouts,
                "tokens": snapshot.get("tokens"),
                "known_tokens": snapshot.get("known_tokens"),
                "elapsed_s": snapshot.get("elapsed_s"),
            },
        }

    def _write_analysis(self) -> None:
        diagnosis = str(getattr(self.optimizer, "last_diagnosis", "") or "")
        if not diagnosis:
            return
        considered = str(getattr(self.optimizer, "last_considered", "") or "")
        path = self.root / "analysis" / f"generation-{self.generation}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# Generation {self.generation} — {self.optimizer.name}\n\n{diagnosis}\n"
            + (f"\n## Considered\n\n{considered}\n" if considered else "")
        )

    def _append_rejections(self, rejections: Sequence[Rejection]) -> None:
        if not rejections:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "rejections.jsonl").open("a") as handle:
            for rejection in rejections:
                handle.write(
                    json.dumps(
                        {
                            "attempt": rejection.attempt,
                            "edit": rejection.edit,
                            "rationale": rejection.rationale,
                            "reasons": rejection.reasons,
                            "codes": rejection.codes(),
                            # The candidate a slim repair call made of it, or empty.
                            "repaired": rejection.repaired,
                        },
                        default=str,
                    )
                    + "\n"
                )

    def _exam_moved(self, candidate: Candidate) -> str | None:
        """Whether a candidate changed which items it would be graded on.

        The seed names the class of work. A candidate that rebinds a data node's Task
        or source chooses a different exam, and the held-out split is derived from the
        items a candidate actually has, so it would be graded on its own choice. A
        person's route to other items is a run's ``--source``.
        """
        if not self.population:
            return None
        before = _item_sources(self.population[0].state)
        after = _item_sources(candidate.state)
        changed = sorted(
            node for node in set(before) | set(after) if before.get(node) != after.get(node)
        )
        if not changed:
            return None
        return (
            f"E-exam: the candidate changed which items it would be graded on "
            f"({', '.join(changed)}); the class of work is the seed's, and a run points "
            f"a harness at other items with --source"
        )

    # ── item-ownership verification ──────────────────────────────────────

    def _expected_item_ids(self, state: dict[str, Any], part: str) -> tuple[set[str], bool]:
        """The item ids the loop expects a trial to grade, and whether item_scores is declared.

        Computed from the loop root and the seed's state — the loop-owned channel
        the candidate cannot write. Falls back silently when the mode cannot build
        or has no data nodes with items.

        When a loop-level manifest exists, val/test ids are read from it instead
        of calling ``partition().get(part)`` (which is always empty for val/test).
        """
        # When a manifest exists, read the part's ids directly from it (the
        # holdout audit's are its val ids: ``task.manifest_part_ids``).
        listed = (task_module.manifest_part_ids(self._parts_manifest, part)
                  if self._parts_manifest is not None else None)
        if listed is not None:
            manifest_ids = set(listed)
            # Still need to determine seed_declares_item_scores
            declares = False
            if state:
                try:
                    graph = self.mode_class(state=state).build()
                    for node in graph.nodes.values():
                        if node.kind == "data" and "item_scores" in node.produces:
                            declares = True
                            break
                except Exception:
                    pass
            return manifest_ids, declares

        if not state:
            return set(), False
        try:
            graph = self.mode_class(state=state).build()
        except Exception:
            return set(), False

        from refactory_dsh.nn.executor import full_items

        ids: set[str] = set()
        seed_declares_item_scores = False
        nodes = sorted(
            (node for node in graph.nodes.values() if node.kind == "data"),
            key=lambda node: node.id,
        )
        for node in nodes:
            if "item_scores" in node.produces:
                seed_declares_item_scores = True
            try:
                items = full_items(node, self.root)
            except Exception:
                continue
            if not items:
                continue
            selected = task_module.partition(items, self.config.holdout).get(part, [])
            for i, item in enumerate(selected):
                ids.add(str(item.get("id") or i))
        return ids, seed_declares_item_scores

    def _verify_item_set(
        self,
        expected_ids: set[str],
        seed_declares_item_scores: bool,
        trial_dir: Path,
        part: str,
        result: "RunResult",
    ) -> None:
        """Post-trial check: verify the actually-graded ids match the expected set.

        Three sub-checks:
        1. Subset integrity (backstop): the trial-dir subset.json must match the
           loop's part and holdout.
        2. Skip-loophole closure: when the seed declares item_scores, a trial that
           produces none is refused as E-exam.
        3. Item-set comparison: extra ids are refused; missing ids are refused only
           when they have no recorded per-item failure.
        """
        # 1. Subset integrity backstop
        subset = task_module.read_subset(trial_dir)
        expected_subset = {"part": part, "holdout": self.config.holdout}
        if subset and (subset.get("part") != expected_subset["part"]
                       or subset.get("holdout") != expected_subset["holdout"]):
            result.exit_ok = False
            result.error = (
                f"E-exam: subset.json was tampered with: expected {expected_subset}, "
                f"got {{'part': {subset.get('part')!r}, 'holdout': {subset.get('holdout')!r}}}"
            )
            return

        # 2. Skip-loophole closure
        has_item_scores = result.facts is not None and "item_scores" in result.facts
        item_scores = result.facts.get("item_scores") if has_item_scores else None
        if seed_declares_item_scores and not has_item_scores:
            result.exit_ok = False
            result.error = (
                "E-exam: the seed declares item_scores but the trial produced none; "
                "a trial may not avoid grading by producing no item_scores"
            )
            return

        # 3. Item-set comparison
        if not result.exit_ok or item_scores is None or not expected_ids:
            return

        graded_ids = {str(s["id"]) for s in item_scores if isinstance(s, dict) and "id" in s}

        extra = graded_ids - expected_ids
        if extra:
            result.exit_ok = False
            result.error = (
                f"E-exam: the trial graded ids outside the allowed set: "
                f"{sorted(extra)}"
            )
            return

        missing = expected_ids - graded_ids
        if missing:
            # A missing item that has a recorded per-item failure is accepted
            # (consistent with F4: failed items score 0 and keep their error).
            # A missing item with no failure record is refused.
            if result.error is not None:
                # The run had an error; missing items are attributed to it.
                return
            result.exit_ok = False
            result.error = (
                f"E-exam: expected ids were not graded and have no recorded failure: "
                f"{sorted(missing)}"
            )

    # ── step ───────────────────────────────────────────────────────────────
    def step(
        self, candidates: Sequence[Candidate], run: Run | None = None
    ) -> list[Candidate]:
        """Add offspring that are genuinely new points in the search."""
        known = {candidate.id for candidate in self.population}
        added: list[Candidate] = []
        for candidate in candidates:
            moved = self._exam_moved(candidate)
            if moved is not None:
                self.rejections.append(
                    Rejection(
                        attempt=0,
                        edit=_edit_record(candidate),
                        reasons=[moved],
                    )
                )
                continue
            if candidate.id in known:
                # The optimizer's loop refuses these now, so reaching here means a
                # strategy bypassed it. Record it rather than losing the slot in
                # silence: a batch that tried fewer points than it claimed is a
                # result nobody can trust.
                self.rejections.append(
                    Rejection(
                        attempt=0,
                        edit=_edit_record(candidate),
                        reasons=[
                            f"E-duplicate: the candidate's state already exists as "
                            f"candidate {candidate.id}, so its batch slot was lost"
                        ],
                    )
                )
                continue
            known.add(candidate.id)
            if candidate.parent and self._custom_strategy is None:
                # Whether its parent was the incumbent when it was added, for the
                # report's pool diversity; in the strategy state, so a resume keeps it.
                self._strategy_state.setdefault("parent_from_incumbent", {})[candidate.id] = (
                    candidate.parent == self._incumbent_id())
            self.population.append(candidate)
            added.append(candidate)
        self.generation += 1
        self._journal(
            "step",
            generation=self.generation,
            # Journaled, not derived: the adaptive k depends on it, so a resumed
            # run that guessed it would propose something else.
            refusal_pressure=round(self._refusal_pressure, 6),
        )
        self._write_state()
        self._progress("step", added=len(added), population=len(self.population))
        return added

    # ── scheduler ──────────────────────────────────────────────────────────
    def scheduler(self, *, evaluated: int = 0, proposed: int = 0, accepted: int = 0,
                  refused: int = 0) -> bool:
        """Close the epoch, and say whether there is another.

        The epoch is opened by the loop, not here: a resume has to know which
        iteration it is in, and an epoch that is only counted once it is finished
        cannot say that.
        """
        batch = [c for c in self._evaluated() if c.score is not None]
        scores = [c.score for c in batch if c.exit_ok and c.score is not None]
        summary = EpochSummary(
            epoch=self.epoch,
            evaluated=evaluated,
            best=max(scores) if scores else None,
            mean=(sum(scores) / len(scores)) if scores else None,
            improved=bool(self.loss.get("improved")),
            proposed=proposed,
            accepted=accepted,
            refused=refused,
            population=len(self.population),
            refused_total=sum(1 for r in self.rejections if not r.repaired),
            structural_in_population=sum(1 for c in self.population if c.lineage_is_structural()),
        )
        self.epochs.append(summary)
        self._write_state()

        if self.config.val_gate and self.bound:
            # A budget a measurement found spent is why the run stops; with the
            # validation gate that reason decides whether the run is scored further.
            pass
        elif self.epoch >= self.config.epochs:
            self.bound = "epochs"
        elif self.evaluations >= self.config.max_evaluations:
            self.bound = "evaluations"
        elif self.config.max_rollouts is not None and self.rollouts >= self.config.max_rollouts:
            self.bound = "rollouts"
        if self.bound:
            # Journaled, not derived: the report states what bound the run, and a
            # run whose steps are separate processes reads that from the log.
            self._journal("bound", bound=self.bound)
        gate_fields: dict[str, Any] = {}
        if self.config.val_gate and self._custom_strategy is None:
            # A person reads this line: say what the gate decided, with its val scores,
            # rather than a train-score "improved" the gate may have rejected.
            gate = self._gate()
            gate_fields = {"gate": dict(getattr(self, "_epoch_gate", {}) or {}),
                           "val_best": gate.best_score if gate.best_id else None,
                           "val_current": gate.current_score if gate.current_id else None}
        self._progress(
            "scheduler",
            epoch=self.epoch,
            best=summary.best,
            mean=summary.mean,
            improved=summary.improved,
            proposed=proposed,
            accepted=accepted,
            bound=self.bound,
            another=not self.bound,
            **gate_fields,
        )
        return not self.bound

    # ── the loop ───────────────────────────────────────────────────────────
    # ── the steps a strategy may override ──────────────────────────────────
    def select_parents(self, n: int, *, by: str = "score") -> list[Candidate]:
        """The candidates a proposal is made against. Overridable, not fixed.

        Falls back to the population when nothing has been scored yet: a strategy
        that asks for parents before the first epoch must get the seed, not an empty
        list, or the only way to write a strategy is to know the order of the loop.

        A kept candidate is always among them, whether or not it is still in the top
        `n`. That is the whole of `keep`: without it, elite selection is a cull and a
        person's "follow this one up" would be silently dropped the moment a newer
        batch outscores it.

        Under ``checkpoint`` the current candidate comes first (``_gated_parents``): it is
        the one being trained, while the leaderboard's first, the best checkpoint, is
        only the winner.
        """
        chosen = (self._gated_parents()[:n]
                  if self.config.val_gate and self.selection_mode() == "checkpoint"
                  else self.leaderboard(n))
        pinned = [
            candidate
            for candidate in self.leaderboard()
            if candidate.id in self.kept and candidate not in chosen
        ]
        if chosen or pinned:
            return chosen + pinned
        return [c for c in self.population[:n] if c.id not in self.rejected]

    def _split_proposal_calls(self, n: int) -> bool:
        """Whether this round makes N independent single-proposal calls instead of one
        call asking for N.  True when the optimizer uses a model and ``n > 1``, unless
        the strategy signalled a deliberate multi-sample call (BestOfN, Autoscientists)
        via ``strategy_decision["samples"]``."""
        return (n > 1
                and getattr(self.optimizer, "uses_model", False)
                and not self.strategy_decision.get("samples"))

    def propose(self, run: Run, parents: Sequence[Candidate], n: int) -> list[Proposal]:
        """One round of proposals. The default delegates to the optimizer.

        A failed round is survived and recorded rather than raised. The run is the
        only place the evaluations so far are reported, so letting a model
        exception escape loses the whole run to a fact about the machine -- and a
        run that ends with no report looks, from outside, exactly like one that
        never started. What was refused and why is the optimizer's business;
        whether the optimizer answered at all is the loop's, and it is recorded in
        ``Optimizer.proposal_failures`` so the report can say which happened.

        When :meth:`_split_proposal_calls` is true, makes *n* sequential single-
        proposal calls instead of one call asking for *n*.  Each call gets its own
        parent (via the picker, when present), its own per-parent context, and the
        earlier proposals of the round as facts.  The analysts run once and are
        shared.
        """
        if self._split_proposal_calls(n):
            return self._propose_per_slot(run, parents, n)
        return self._propose_single(run, parents, n)

    def _propose_single(self, run: Run, parents: Sequence[Candidate], n: int) -> list[Proposal]:
        """One optimizer call asking for ``n`` proposals (the original path)."""
        from refactory_dsh.budgets import BudgetExhausted

        self.optimizer.proposal_rounds += 1
        # Which attempt of this generation's proposal round this is: a retry after a
        # refusal reads another sample of the parent's trajectories.
        self._proposal_attempt += 1
        try:
            context = self._context()
            context.attempt = self._proposal_attempt
            if hasattr(self.optimizer, "last_analysis"):
                self.optimizer.last_analysis = None
            try:
                proposals = self.optimizer.propose(context, n, parents)
            finally:
                # Journaled whether or not the proposal call succeeded: the analysts'
                # calls were paid either way.
                self._keep_analysis()
            return proposals
        except BudgetExhausted as exc:
            self.optimizer.proposal_failures.append(f"{type(exc).__name__}: {exc}")
            raise
        except RunStopped:
            # A person's stop during a retry wait is a stop, not a failed round (nor a
            # round counted): it is made again on resume (the epoch stays open).
            self.optimizer.proposal_rounds -= 1
            raise
        except Exception as exc:  # a proposer that is down is not a crash
            detail = f"{type(exc).__name__}: {exc}"
            self.optimizer.proposal_failures.append(detail)
            run.log(f"{self.optimizer.name} could not propose: {detail}")
            # In the progress stream too, where a person watching the run looks: a
            # round that never reached the model otherwise reads as one that did and
            # proposed nothing.
            self._progress("proposal_failed", optimizer=self.optimizer.name,
                           parent=self.strategy_decision.get("parent") or self._editing_parent,
                           attempt=self._proposal_attempt,
                           error=type(exc).__name__, detail=str(exc)[:500])
            return []

    def _propose_per_slot(self, run: Run, parents: Sequence[Candidate], n: int) -> list[Proposal]:
        """N sequential single-proposal calls, one per candidate slot.

        Each call gets its own parent (via the picker, when present), its own per-
        parent context (pipeline view, knobs, trajectories), and a list of the
        earlier proposals of the round as facts.  The analysts and memory run once
        and are shared across slots.
        """
        from refactory_dsh.budgets import BudgetExhausted

        choose = getattr(self.optimizer, "choose_parent", None)
        all_proposals: list[Proposal] = []
        shared_analysis: dict[str, Any] | None = None
        original_parent = self._editing_parent
        slot_levels: list[str] = []

        for slot in range(n):
            # --- pick the parent for this slot ---
            if callable(choose) and not self.strategy_decision.get("parent"):
                picked = choose(
                    ProposalContext(epoch=self.epoch, generation=self.generation,
                                   mode_name=self.mode_id, elite=int(self.config.elite),
                                   val_scores=self._val_scores(),
                                   val_item_scores=self._val_item_scores()), parents)
                if picked is not None:
                    self._editing_parent = picked.id
            self._refresh_edited_records()

            self.optimizer.proposal_rounds += 1
            self._proposal_attempt += 1

            try:
                context = self._context()
                context.attempt = self._proposal_attempt
                # The first slot runs the analysts; later slots reuse its analysis.
                if shared_analysis is not None:
                    context = dataclasses.replace(
                        context, analysis=shared_analysis, analysts={})
                    if hasattr(self.optimizer, "last_analysis"):
                        self.optimizer.last_analysis = shared_analysis
                elif hasattr(self.optimizer, "last_analysis"):
                    self.optimizer.last_analysis = None
                # Facts about what this round already holds.
                if all_proposals:
                    context = dataclasses.replace(context, prior_proposals=[
                        {"edits": list(p.steps), "rationale": p.rationale,
                         "parent": p.parent or ""}
                        for p in all_proposals])
                # Under portfolio, narrow the edit levels to what this slot should propose.
                # Use slot_levels (assigned this round) alongside _levels_done (installed
                # in prior rounds) so each slot gets a different level.
                if self._levels_wanted:
                    assigned = set(self._levels_done) | set(slot_levels)
                    missing = [l for l in self._levels_wanted if l not in assigned]
                    if missing:
                        slot_levels.append(missing[0])
                        context = dataclasses.replace(context, edit_levels=[missing[0]])
                try:
                    proposals = self.optimizer.propose(context, 1, parents)
                finally:
                    self._keep_analysis()
                    # Capture the first slot's analysis for sharing.
                    if slot == 0 and shared_analysis is None:
                        shared_analysis = getattr(self.optimizer, "last_analysis", None)
                all_proposals.extend(proposals or [])
            except BudgetExhausted as exc:
                self.optimizer.proposal_failures.append(f"{type(exc).__name__}: {exc}")
                raise
            except RunStopped:
                self.optimizer.proposal_rounds -= 1
                raise
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                self.optimizer.proposal_failures.append(detail)
                run.log(f"{self.optimizer.name} could not propose slot {slot + 1}/{n}: {detail}")
                self._progress("proposal_failed", optimizer=self.optimizer.name,
                               parent=self.strategy_decision.get("parent") or self._editing_parent,
                               attempt=self._proposal_attempt, slot=slot + 1,
                               error=type(exc).__name__, detail=str(exc)[:500])
                # A failed slot stays empty; continue to the next.
                continue

        self._editing_parent = original_parent
        return all_proposals

    def batch_size(self, run: Run) -> int:
        levels = self._round_levels()
        return len(levels) if levels else int(self.config.batch_size)

    def _round_levels(self) -> list[str]:
        """The edit levels this round asks for under ``portfolio``: every level, or the
        one a strategy decided for the round; none without a portfolio."""
        from refactory_dsh.optim.base import EDIT_LEVELS

        if not self.config.portfolio:
            return []
        kind = self.strategy_decision.get("edit_kind")
        return [kind] if kind in EDIT_LEVELS else list(EDIT_LEVELS)

    def install(
        self, run: Run, proposals: Sequence[Proposal], parents: Sequence[Candidate]
    ) -> list[Candidate]:
        """Turn proposals into candidates, asking again with the reasons.

        This is the validating loop the whole design rests on: apply, refuse with a
        coded reason, feed the reasons back, and only ever hand a candidate to
        evaluation. Overriding it is how a strategy changes how many times it asks.

        With ``batch_size`` > 1 (and no custom strategy), the smoke check is deferred:
        every candidate passes validation and integrity first, then the batch is smoked
        concurrently through ``_smoke_check_batch``.
        """
        accepted: list[Candidate] = []
        accepted_ids: set[str] = set()
        wanted = self.batch_size(run)
        defer_smoke = wanted > 1 and self._custom_strategy is None
        smoke_meta: dict[str, tuple[Proposal, Candidate]] = {}
        self._defer_smoke = defer_smoke
        # `k` adapts: one more round when the last batch was refused more than it was
        # accepted, because that is the signal that the optimizer is close to
        # something legal rather than out of ideas.
        cap = max(1, self.config.attempts) + (1 if self._refusal_pressure >= 0.5 else 0)
        rounds = 0
        before_all = len(self.rejections)
        for _attempt in range(1, cap + 1):
            rounds += 1
            if len(accepted) >= wanted:
                break
            before = len(self.rejections)
            for proposal in proposals:
                if len(accepted) >= wanted:
                    break
                # An op that is not in the catalogue at all is a different fact from
                # one that exists but is outside this run's permitted edits. Calling
                # both E-policy hides which fix the refusal is asking for. Each edit of
                # a compound proposal is checked, and a refusal names the one.
                parent = next((p for p in parents if p.id == proposal.parent), parents[0])
                refused = proposal_refusal(proposal, permitted=self.permitted_ops(),
                                           max_edits=MAX_EDITS_PER_PROPOSAL,
                                           parent_state=parent.state)
                level = edit_level(proposal.steps)
                missing = [l for l in self._levels_wanted if l not in self._levels_done]
                if not refused and self._levels_wanted and level not in missing:
                    refused = [f"E-level: this round asks for one candidate per edit level "
                               f"({', '.join(self._levels_wanted)}); this proposal is a {level} "
                               f"change, which is "
                               + ("already covered" if level in self._levels_done
                                  else "not asked for")
                               + (f"; still missing: {', '.join(missing)}" if missing else "")]
                # A refused proposal goes to the slim repair call (`_repair`); its fix,
                # when accepted, is the candidate, and the refusal says so.
                repaired_from: list[str] = []
                if refused:
                    self.rejections.append(Rejection(rounds, proposal.edit, refused))
                    repaired_from = list(refused)
                    outcome = self._repair(run, proposal, parent, refused)
                else:
                    outcome = run.attempt(proposal, parent)
                    if isinstance(outcome, list):
                        repaired_from = list(outcome)
                        outcome = self._repair(run, proposal, parent, outcome)
                if isinstance(outcome, Candidate):
                    if outcome.id in accepted_ids:
                        # Two proposals in one batch that reach the same state. The
                        # population does not hold the first one yet, so the attempt
                        # above could not see it, and `step` would drop the second
                        # without saying why. Refused here, where the optimizer is
                        # still listening.
                        self.rejections.append(
                            Rejection(
                                attempt=rounds,
                                edit=proposal.edit,
                                reasons=[
                                    f"E-duplicate: this state is candidate {outcome.id}, "
                                    "already accepted from the same batch, so evaluating it "
                                    "again would spend a batch slot on a point already scored"
                                ],
                                rationale=proposal.rationale,
                            )
                        )
                        continue
                    accepted_ids.add(outcome.id)
                    accepted.append(outcome)
                    if defer_smoke:
                        smoke_meta[outcome.id] = (proposal, parent)
                    if repaired_from and getattr(self, "_repaired_level", ""):
                        level = self._repaired_level  # the fix's level, as E-level judged it
                    if self._levels_wanted:
                        self._levels_done.append(level)
                    # Its own words, for "Your earlier reasoning" in later requests.
                    # The diagnosis of the call that produced this proposal: a later
                    # retry in the same round replaces the optimizer's last one.
                    self._proposal_notes[outcome.id] = {
                        "rationale": proposal.rationale, "prediction": proposal.prediction,
                        "confidence": proposal.confidence,
                        "addresses": proposal.addresses, "flow_change": proposal.flow_change,
                        "reasoning_missing": proposal.missing_reasoning(),
                        "diagnosis": str(getattr(self.optimizer, "last_diagnosis", "") or ""),
                        "considered": str(getattr(self.optimizer, "last_considered", "") or ""),
                        **({"level": level} if self._levels_wanted else {}),
                        **({"repaired_from": repaired_from} if repaired_from else {})}
            # Another response is useful only if a validation round remains.
            if len(accepted) >= wanted or len(self.rejections) == before or _attempt >= cap:
                break
            proposals = self.propose(run, parents, wanted - len(accepted))
            if not proposals:
                break
        if defer_smoke and accepted:
            self._defer_smoke = False
            accepted = self._smoke_check_batch(run, accepted, smoke_meta)
        self._defer_smoke = False
        refused = sum(1 for r in self.rejections[before_all:] if not r.repaired)
        self._refusal_pressure = refused / max(1, refused + len(accepted))
        self.attempts_used.append(rounds)
        return accepted

    def _repair(self, run: Run, proposal: Proposal, parent: Candidate,
                reasons: list[str]) -> Candidate | None:
        """A refused proposal sent to the optimizer's slim repair call
        (`Optimizer.repair_refused`), each fix validated as a proposal is (this round's
        refusals first, then ``Run.attempt``): the candidate, or ``None``.

        On success the proposal's rejection, just recorded, names the candidate it became,
        and ``_repaired_level`` the fix's edit level. When the fixes are refused, the last
        refusal is added to the rejection's reasons (marked as after a slim repair), so a
        refusal of the fix (an integrity verdict, say) reaches later requests too. Each
        epoch's call count is journaled (``repair_calls``), so a resumed run keeps the
        bound."""
        index = len(self.rejections) - 1
        if index < 0:
            return None
        self._repaired_level = ""

        def attempt(fixed: Proposal) -> Candidate | list[str]:
            refused = proposal_refusal(fixed, permitted=self.permitted_ops(),
                                       max_edits=MAX_EDITS_PER_PROPOSAL,
                                       parent_state=parent.state)
            level = edit_level(fixed.steps)
            missing = [l for l in self._levels_wanted if l not in self._levels_done]
            if not refused and self._levels_wanted and level not in missing:
                refused = [f"E-level: this round asks for a {', '.join(missing) or 'different'} "
                           f"change; the repaired proposal is a {level} change"]
            outcome = refused or run.attempt(fixed, parent, record=False)
            if isinstance(outcome, Candidate):
                self._repaired_level = level
            return outcome

        before = self.optimizer.repairs_used(self.epoch) if hasattr(self.optimizer, "repairs_used") else 0
        fixed = self.optimizer.repair_refused(self._repair_context(), proposal, reasons, parent,
                                              attempt)
        used = self.optimizer.repairs_used(self.epoch) if hasattr(self.optimizer, "repairs_used") else 0
        if used != before:
            self._journal("repair_calls", epoch=self.epoch, calls=used)
        if fixed is not None:
            self.rejections[index] = dataclasses.replace(
                self.rejections[index], repaired=fixed.id,
                dropped=str(getattr(self.optimizer, "last_repair_lost", "") or ""))
        else:
            after = list(getattr(self.optimizer, "last_repair_refusal", None) or ())
            if after:
                self.rejections[index] = dataclasses.replace(
                    self.rejections[index],
                    reasons=[*self.rejections[index].reasons,
                             *(f"{reason} [the proposal as a slim repair call fixed it]"
                               for reason in after)])
        return fixed

    def _repair_context(self) -> ProposalContext:
        """What a slim repair call is written from: the epoch, the operations, whether code
        nodes are offered, the protected nodes and the change cap. Not a whole request."""
        return ProposalContext(
            epoch=self.epoch, generation=self.generation, mode_name=self.mode_id,
            catalogue=[op for op in nn_edits.catalogue(self.config.edit_preset)
                       if not self.config.allowed_ops or op["op"] in self.config.allowed_ops],
            protected=list(self.protected), preset=self.config.edit_preset,
            code_nodes=self._code_nodes_offered(self._sandbox_answer()),
            person_gates=False, max_edits=MAX_EDITS_PER_PROPOSAL)

    def strategy(self, run: Run) -> None:
        """The default control flow: repeated epochs, each one step of the loop.

        `should_stop` is consulted first so a strategy can end for a reason of its
        own -- a plateau, a convergence -- without replacing the loop.

        When a custom strategy is attached (``_custom_strategy``), delegates
        to it — the **custom strategy seam**.
        """
        if self._custom_strategy is not None:
            return self._custom_strategy.strategy(run)
        self._search(run)
    #: How this strategy's search ends normally, on its own epoch limit: the run then
    #: scores the batch its last epoch proposed (``finish_strategy``). A strategy whose
    #: schedule is its own (a curriculum) adds the reason it stops with.
    normal_ends: tuple[str, ...] = ("epochs",)

    def _ended(self, run: Run | None) -> str:
        """Why the search ended: the bound the loop recorded, else the strategy's stop
        reason (a strategy that drives its own epochs stops with ``"epochs"`` whatever
        bound ended them)."""
        return str(self.bound or (run.stopped if run is not None else "") or "")

    def finish_strategy(self, run: Run) -> None:
        """After any built-in strategy: score what the last epoch proposed, when the
        search ended normally. Under ``val_gate`` that is ``_finish_gated_search`` (the
        last proposals scored and gated, then the post-training evaluation); without
        it, ``score_last_batch``. A custom strategy keeps its own ending."""
        if self._custom_strategy is not None:
            return
        if self.config.val_gate:
            self._finish_gated_search(run)
        else:
            self.score_last_batch(run)

    def score_last_batch(self, run: Run | None = None) -> None:
        """Without ``val_gate``: score the last epoch's proposals once the search ends
        on its epoch bound, as ``_finish_gated_search`` does under the gate.

        Each epoch scores the batch the one before proposed and then proposes the
        next, so a run of N epochs makes N proposal rounds; without this the last
        round was paid for and never scored. ``forward`` keeps the evaluation and
        rollout caps and charges what it runs. After any other bound (a person's
        stop, a spent evaluation, rollout, time or token budget) the proposals stay
        unscored and the report names them.

        Public because the loop is also a graph (``loop_mode``): its report node runs
        this, so the graph scores what the loop scores.
        """
        if self.config.val_gate:
            return
        if self._ended(run) in self.normal_ends and self._unevaluated():
            self.forward(epoch=self.epoch + 1)

    def _search(self, run: Run) -> None:
        """Epochs until a bound, a stop reason or a person's stop."""
        while True:
            if self._apply_control() == "stop":
                run.stop("a person asked the run to stop")
                return
            reason = self.should_stop(run)
            if reason:
                run.stop(reason)
                return
            if not self.epoch_step(run):
                return

    def _apply_control(self) -> str | None:
        """Honor what another process asked for, at the boundary.

        A pause blocks here until the request changes, so a person can hold a search
        between generations without killing it. The poll interval is configuration
        only so a test does not have to wait a second for each step.

        The population acts -- keep, reject, adopt -- are honored here too and then
        consumed, because this is the only place a request is safe (never inside an
        evaluation) and because one act is one act: a boundary that adopted the same
        winner every epoch would be a bug that looks like enthusiasm.
        """
        from refactory_dsh.control import Control, ONE_SHOT

        control = Control(self.root)
        paused_budget = None
        while True:
            request = control.read_request()
            action = str(request["action"]) if request is not None else ""
            if action != "pause" and paused_budget is not None:
                paused_budget.resume()
                paused_budget = None
            if request is None:
                return None
            if action == "pause":
                # Only the allocation's owner can stop its clock: a nested search
                # may share resources with enclosing work that is still running.
                if self.budget and self._owns_resource_budget and paused_budget is None:
                    self.budget.pause()
                    paused_budget = self.budget
                self._progress("paused")
                time.sleep(max(0.01, float(self.config.control_poll_s)))
                continue
            if action == "resume":
                # The pause it released is over; clearing keeps "absent means run"
                # the only reading of the file.
                control.clear()
                return None
            if action == "stop":
                self._progress("control", action="stop")
                return "stop"
            outcome = self._honor(request)
            if action in ONE_SHOT:
                control.clear()
            self._progress("control", **outcome)
            return None

    def _honor(self, request: dict[str, Any]) -> dict[str, Any]:
        """Carry out one population act, and say what it did.

        Every outcome is a fact rather than an exception. A run that dies because a
        person asked it to keep a candidate they mistyped has turned their control
        into a way to lose the search, and the file the request travels in has no
        way to say "that id was wrong" other than the journal.
        """
        action = str(request.get("action") or "")
        target = str(request.get("target") or "")
        lineage = bool(request.get("lineage"))
        known = {candidate.id for candidate in self.population}
        from refactory_dsh.control import request_identity

        outcome: dict[str, Any] = {"action": action, "request": request_identity(request)}
        if request.get("recovered"):
            outcome.update(recovered=True, requested_at=request.get("at"), status="applied")
        if target:
            outcome["target"] = target
        if lineage:
            outcome["lineage"] = True

        if action not in ("keep", "reject"):
            return self._adopt_now(str(request.get("as") or ""), outcome)

        targets = self._lineage_ids(target) if lineage else [target]
        outcome["targets"] = targets
        unknown = [candidate_id for candidate_id in targets if candidate_id not in known]
        applied: list[str] = []
        if action == "reject":
            for candidate_id in targets:
                if candidate_id in known and candidate_id not in self.rejected:
                    self.rejected.append(candidate_id)
                    applied.append(candidate_id)
                if candidate_id in self.kept:
                    self.kept.remove(candidate_id)
        else:
            # A keep is also the undo of a reject: a candidate is in and protected,
            # or out, and two flags saying both would be a state no reader can act on.
            for candidate_id in targets:
                if candidate_id in self.rejected:
                    self.rejected.remove(candidate_id)
                if candidate_id in known and candidate_id not in self.kept:
                    self.kept.append(candidate_id)
                    applied.append(candidate_id)

        outcome["applied"] = applied
        if unknown:
            outcome["unknown"] = unknown
        self._journal("control", **outcome, kept=list(self.kept), rejected=list(self.rejected))
        self._write_state()
        return outcome

    def _lineage_ids(self, target: str) -> list[str]:
        """The candidate and every descendant of it, in population order.

        "Stop spending budget around this branch" means the branch: rejecting one
        point while its descendants keep proposing from it is not what a person who
        read the lineage asked for.
        """
        children: dict[str, list[str]] = {}
        for candidate in self.population:
            if candidate.parent:
                children.setdefault(candidate.parent, []).append(candidate.id)
        found = [target]
        queue = [target]
        while queue:
            for child in children.get(queue.pop(0), ()):
                if child not in found:
                    found.append(child)
                    queue.append(child)
        return found

    def _adopt_now(self, as_name: str, outcome: dict[str, Any]) -> dict[str, Any]:
        """Write the current winner as a package, without ending the search.

        The winner is read from the journal the run is writing right now, which is
        the only artifact that carries a candidate's *state*, so this works while the
        run is live. An explicit live adoption audits that winner first; ordinary
        search evaluations remain training-only. A refusal is recorded as an event:
        adopting never overwrites, and
        a second adopt is a person's mistake rather than a reason to lose the search.
        """
        from refactory_dsh.adopt import adopt as adopt_winner

        # The identifier's package part, not `self.package`: that is the loaded
        # `Package` object, and its `str()` is a dataclass repr -- which became a
        # default name the filesystem refused with "File name too long" the first
        # time the panel adopted without naming a package.
        name = as_name or f"{self.mode_id.split('/', 1)[0]}_adopted"
        outcome["as"] = name
        try:
            self.audit_holdout()
            written = adopt_winner(
                self.root, as_name=name, project=self.project, mode=self.mode_id
            )
            outcome.update(
                package=written["package"],
                mode=written["mode"],
                candidate=written["candidate"],
                score=written["score"],
                package_root=written["package_root"],
            )
        except Exception as exc:  # a refusal is a fact, not the end of the run
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        self.adoptions.append(outcome)
        self._journal("control", **outcome, kept=list(self.kept), rejected=list(self.rejected))
        self._write_state()
        return outcome

    def should_stop(self, run: Run) -> str | None:
        """A reason to stop before the bound, or ``None``. The bound is the default."""
        if run.stopped:
            return run.stopped
        if self.budget:
            from refactory_dsh.budgets import BudgetExhausted
            try:
                self.budget.checkpoint("work" if self._owns_resource_budget else None)
            except BudgetExhausted as exc:
                return str(exc)
        # A resumed completed epoch must not buy another proposal round. An
        # interrupted open epoch may still finish its bookkeeping at this limit.
        if not self._epoch_open and self.config.val_gate:
            early = self._early_stop_reason()
            if early:
                return early
        if not self._epoch_open and self.epoch >= self.config.epochs:
            return "epochs"
        if not self._epoch_open and self.evaluations >= self.config.max_evaluations:
            return "max_evaluations"
        return None

    def accepts(self, run: Run, candidate: Candidate) -> bool:
        """Whether a validated candidate is kept. A loser can still be archived."""
        return True

    def update(self, run: Run, candidate: Candidate) -> None:
        """Record a kept candidate. This is where an archive lives."""
        self.archive.add(candidate) if hasattr(self, "archive") else None

    def on_epoch_end(self, run: Run) -> None:
        """Called once per iteration, after the scheduler."""

    def open_epoch(self) -> None:
        """Open the epoch the following steps belong to. Idempotent.

        Public because the eight steps are also the nodes of a graph: an executor
        that runs one node per process has to open the epoch in the node that
        begins it, and reaching for the flag would make the graph depend on private
        state instead of on this method.
        """
        if self._epoch_open:
            return
        self.epoch += 1
        self._journal("epoch_open", epoch=self.epoch)
        self._epoch_open = True

    def close_epoch(self) -> None:
        """Close the epoch, so the next step opens the next one."""
        self._epoch_open = False
        self._journal("epoch_end", epoch=self.epoch)

    def strategy_state_dict(self) -> dict[str, Any]:
        if self._custom_strategy is not None:
            return self._custom_strategy.strategy_state_dict()
        return super().strategy_state_dict()

    def load_strategy_state(self, state: dict[str, Any]) -> None:
        if self._custom_strategy is not None:
            return self._custom_strategy.load_strategy_state(state)
        super().load_strategy_state(state)
        self._val_gate = None  # rebuilt from the restored state by _gate()

    def _gate(self) -> ValidationGate:
        """The default loop's validation gate, restored from the strategy state.

        Its state, with the ids it accepted and rejected, is the ``val_gate`` entry of
        the loop's strategy state, journaled after every decision.
        """
        if self._val_gate is None:
            saved = dict(self._strategy_state.get("val_gate") or {})
            gate = ValidationGate(seed=self.config.seed)
            gate.load_state_dict(saved)
            self._gate_accepted = list(saved.get("accepted") or [])
            self._gate_rejected = list(saved.get("rejected") or [])
            self._gate_rounds = [list(r) for r in saved.get("rounds") or []]
            self._val_gate = gate
        return self._val_gate

    def _gate_population(self, run: Run) -> None:
        """Judge each candidate evaluated on train that the gate has not judged.

        The seed is rolled out on val and accepted. Every later candidate the gate has
        not judged is one round (``ValidationGate.judge_round``): each is judged against
        the same bar, the gate's current candidate as the round starts, so the order
        siblings were proposed in decides nothing. Of those whose val score strictly
        beats it, the highest is accepted and becomes current; a tie goes to the higher
        train score, then to the one proposed first. The rest are rejected, one that
        beat the bar but lost to a sibling included (the report marks it shadowed). A
        candidate whose train run failed is rolled out on val too and judged the same
        way. A refused rollout (a spent rollout budget) judges what was scored before
        it and leaves the rest unjudged; an unjudged candidate is not a parent or a
        winner. SkillOpt gates one candidate per epoch, where both rules agree.

        The val rollouts are rolled out first, up to ``max_concurrent_evaluations`` at
        once (``_prefetch_val``), and the decisions are then taken from the gate's cache.

        Under ``checkpoint`` the seed starts the gate the same way, and every later round
        is applied instead (``_apply_round``): only the applied candidate is rolled out.
        """
        from refactory_dsh.outerloop import RoundStopped

        gate = self._gate()
        judged = set(self._gate_accepted) | set(self._gate_rejected)
        pending = [c for c in self.population if c.evaluated and c.id not in judged]
        checkpoint = self.selection_mode() == "checkpoint"
        cached = len(gate.cache)
        if not checkpoint:
            self._prefetch_val(run, gate, pending)
        if len(gate.cache) != cached:
            # Journaled now, so a run killed before the decisions rolls none again.
            self._store_gate_state()
            self._save_strategy_state()
        if pending and gate.best_id is None:
            try:
                if checkpoint:
                    self._journaled_val(run, gate, pending[0])
                gate.start(run, pending[0])
            except OuterLoopError as error:
                run.log(f"val gate stopped: {error}")
                return
            self._strategy_state["gate_start_epoch"] = self.epoch
            self._gate_accepted.append(pending[0].id)
            self._gate_rounds.append([pending[0].id])
            self._store_gate_state()
            self._save_strategy_state()
            pending = pending[1:]
        if not pending:
            return
        if checkpoint:
            self._apply_round(run, pending)
            return
        try:
            accepted, rejected = gate.judge_round(run, pending)
        except RoundStopped as stop:
            accepted, rejected = stop.accepted, stop.rejected
            run.log(f"val gate stopped: {stop.error}")
        decided = {c.id for c in accepted} | {c.id for c in rejected}
        for candidate in pending:
            if candidate.id in decided:
                (self._gate_accepted if candidate in accepted else self._gate_rejected).append(
                    candidate.id)
        if decided:
            self._gate_rounds.append([c.id for c in pending if c.id in decided])
        self._store_gate_state()
        self._save_strategy_state()

    def _apply_round(self, run: Run, pending: Sequence[Candidate]) -> None:
        """One ``checkpoint`` round: apply the change. Of the candidates whose train run
        succeeded, the one with the highest train score (the earlier on a tie) is
        applied: it alone is rolled out on val, once, becomes current, and becomes the
        best when its val score beats the best's (``ValidationGate.offer_best``: a tie
        keeps the earlier). The rest are not applied, a failed train run included, and
        get no val run. A val rollout the harness refuses (a spent budget) leaves the
        round unjudged, as under ``single``. Journaled (``checkpoint``)."""
        gate = self._gate()
        parent = gate.current_id
        order = {c.id: index for index, c in enumerate(pending)}
        trained = [c for c in pending if c.exit_ok and c.score is not None]
        chosen = (max(trained, key=lambda c: (float(c.score or 0.0), -order[c.id]))
                  if trained else None)
        best = False
        if chosen is not None:
            try:
                self._journaled_val(run, gate, chosen)
                score, _cached = gate.score(run, chosen)
            except OuterLoopError as error:
                run.log(f"val gate stopped: {error}")
                return
            gate.current_id, gate.current_score = chosen.id, score
            best = gate.offer_best(chosen.id, score, run=run)
        applied = [chosen] if chosen is not None else []
        others = [c for c in pending if c is not chosen]
        self._gate_accepted.extend(c.id for c in applied)
        self._gate_rejected.extend(c.id for c in others)
        self._gate_rounds.append([c.id for c in pending])
        self._journal("checkpoint", epoch=self.epoch, parent=parent,
                      applied=chosen.id if chosen is not None else None,
                      not_applied=[c.id for c in others],
                      val=gate.cache.get(chosen.id) if chosen is not None else None,
                      new_best=best, best=gate.best_id, best_val=gate.best_score)
        self._progress("checkpoint", applied=chosen.id if chosen is not None else None,
                       not_applied=len(others),
                       val=gate.cache.get(chosen.id) if chosen is not None else None,
                       new_best=best, best_val=gate.best_score)
        self._store_gate_state()
        self._save_strategy_state()

    def _journaled_val(self, run: Run, gate: ValidationGate, candidate: Candidate) -> None:
        """Under ``checkpoint``, a val score the journal already holds for ``candidate``
        (rolled out beside its train run, or a run killed after the rollout and before
        the decision) goes into the gate's cache, so it is not rolled out again. A
        rollout beside the train run that a budget stopped in this process is the
        refused rollout: ``BudgetStopped``, as the gate's own rollout would raise."""
        if candidate.id in gate.cache:
            return
        stopped = self._val_stopped_beside.pop(candidate.id, None)
        if stopped is not None:
            raise BudgetStopped(stopped, reason=self.bound or stopped)
        held = [s for s in (self._part_results.get(candidate.id) or {}).get(task_module.VAL) or ()
                if not s.get("budget_stopped")]
        if held:
            gate.cache[candidate.id] = gate.val_score(run, candidate)

    def _prefetch_val(self, run: Run, gate: ValidationGate, pending: Sequence[Candidate]) -> None:
        """Roll the candidates the gate will judge out on val, in waves of up to
        ``max_concurrent_evaluations`` (``ValidationGate.prefetch``), so the in-order
        decisions read cached scores. Each val rollout depends only on its candidate
        and the run's seed, so the scores are the ones the decisions would have rolled.

        A wave holds only rollouts the in-order gating would also have made (with
        ``max_rollouts``, see ``_wave_size``); whatever it leaves, or a wave that did
        not score every candidate, is rolled out by the decisions, one at a time.
        """
        manifest = self._parts_manifest or {}
        each = len(manifest.get(task_module.VAL) or ())
        todo = [c for c in pending if c.id not in gate.cache]
        while len(todo) > 1:
            size = self._wave_size(len(todo), each)
            if size < 2 or (self.config.max_rollouts is not None
                            and self.rollouts >= self.config.max_rollouts):
                return
            wave, todo = todo[:size], todo[size:]
            gate.prefetch(run, wave)
            if any(c.id not in gate.cache for c in wave):
                return

    def best_score_fields(self, winner: Candidate) -> dict[str, Any]:
        """The report's ``best_score`` for ``winner``: with ``val_gate`` its val score,
        said so, with its train score (none when its train run failed) beside it;
        otherwise its train score, as before."""
        if not self.config.val_gate or self._custom_strategy is not None:
            return {"best_score": winner.score}
        gate = self._gate()
        val = gate.cache.get(winner.id, gate.best_score if winner.id == gate.best_id else None)
        return {"best_score": val, "best_score_basis": "val: the validation gate's score",
                "best_train_score": winner.score}

    def _gated_parents(self) -> list[Candidate]:
        """The parents under ``val_gate``, for every optimizer and strategy.

        ``single`` (SkillOpt's rule): the gate's current candidate (the last one
        accepted, which is also the best on val), whether or not its train run
        succeeded, with any accepted candidate a person kept. ``pool``: the pool
        (``_pool``, the incumbent first), with any judged candidate a person kept; a
        population optimizer or parent picker (``archive``, ``evolutionary``,
        ``islands``, ``map_elites``: ``ParentSelector.population``) gets every judged
        candidate instead (``_judged_ranked``) and applies its own rule to them (niche
        champions, island bests, its archive's own limit, tournaments), so a new
        structure ranked below the pool keeps its niche. The optimizer chooses by val
        (``ProposalContext.val_scores``), so which candidate is edited never depends on
        a train score from another round's seed or sample.
        """
        gate = self._gate()
        if self.selection_mode() == "pool":
            pool = self._judged_ranked() if self._population_policy() else self._pool()
            judged = set(self._gate_accepted) | set(self._gate_rejected)
            return pool + [c for c in self._evaluated() if c.id in self.kept and c.id in judged
                           and c not in pool and c.id not in self.rejected]
        board = self.leaderboard()
        current = next((c for c in board if c.id == gate.current_id), None)
        first = [current] if current is not None else board[:1]
        return first + [c for c in board if c.id in self.kept and c not in first]

    def _population_policy(self) -> bool:
        """Whether this run's parents are picked by a population policy (the optimizer
        itself, or its parent picker), which gets every judged candidate."""
        policy = getattr(self.optimizer, "picker", self.optimizer)
        return bool(getattr(policy, "population", False))

    def selection_mode(self) -> str:
        """How the validation average selects in this run (``LoopConfig.selection``):
        ``checkpoint``, ``single`` or ``pool`` under the default loop's ``val_gate``; empty
        without it (train scores select) or with a custom strategy (which keeps its own gate).
        An explicit setting wins; then the one the run's gate state recorded (so a
        reader that rebuilds a finished run as a plain ``Loop`` reads the run's own
        rule); then ``default_selection``."""
        if not self.config.val_gate or self._custom_strategy is not None:
            return ""
        if self.config.selection:
            return self.config.selection
        stored = (self._strategy_state.get("val_gate") or {}).get("selection")
        if stored in ("single", "pool", "checkpoint"):
            return str(stored)
        return self._default_selection

    def _pool(self) -> list[Candidate]:
        """The pool under ``selection`` ``pool``: the judged candidates with the
        ``pool_size`` best val scores, plus any tied with the last of them, minus what a
        person rejected. Ranked by val score, a tie by the higher train score (a failed
        train run lowest), then the earlier candidate; the incumbent (the gate's current
        candidate) is first, since it changes only on a strictly higher val score.

        Rebuilt from the gate's cache each time it is read: a val score never changes,
        so the pool is the same set a pool updated round by round would hold, and a
        resumed run holds it again without storing it.
        """
        gate = self._gate()
        ranked = self._top(self._judged_ranked())
        incumbent = next((c for c in ranked if c.id == gate.current_id), None)
        if incumbent is not None:
            ranked = [incumbent] + [c for c in ranked if c is not incumbent]
        return ranked

    def _top(self, ranked: list[Candidate]) -> list[Candidate]:
        """``ranked`` (best val first) cut to the pool: the ``pool_size`` best val
        scores, plus any tied with the last of them."""
        gate = self._gate()
        size = max(1, int(self.config.pool_size))
        if len(ranked) > size:
            edge = gate.cache[ranked[size - 1].id]
            ranked = [c for c in ranked if gate.cache[c.id] >= edge]
        return ranked

    def _pool_entered(self) -> set[str]:
        """The judged candidates that were in the pool after the gating that judged them.

        A val score never changes and later gatings only add candidates, so a candidate
        out of the pool now either entered it then and was pushed out since (``left
        pool``) or never entered it. Rebuilt gating by gating from the gate's cache and
        the ids each gating judged (``rounds`` in the gate state). Every judged
        candidate counts, one a person rejected later included: it held its place when
        the others were judged, so a later rejection does not turn them into ``left
        pool``. A run recorded before ``rounds`` existed groups by proposal epoch.
        """
        gate = self._gate()
        key = (len(self.population), len(self._gate_accepted), len(self._gate_rejected),
               len(gate.cache), len(self._gate_rounds), int(self.config.pool_size))
        cached = getattr(self, "_pool_entered_memo", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        judged = set(self._gate_accepted) | set(self._gate_rejected)
        by_id = {c.id: c for c in self.population}
        order = {c.id: index for index, c in enumerate(self.population)}
        rounds = [[cid for cid in r if cid in by_id and cid in gate.cache]
                  for r in self._gate_rounds]
        if not self._gate_rounds:
            members = [c for c in self._evaluated() if c.id in judged and c.id in gate.cache]
            rounds, last = [], object()
            for candidate in members:
                epoch = (self.proposals.get(candidate.id) or {}).get("epoch")
                if rounds and epoch is not None and epoch == last:
                    rounds[-1].append(candidate.id)
                else:
                    rounds.append([candidate.id])
                last = epoch
        entered: set[str] = set()
        seen: list[Candidate] = []
        for ids in rounds:
            seen.extend(by_id[cid] for cid in ids)
            top = {c.id for c in self._top(sorted(
                seen, key=lambda c: (-gate.cache[c.id], -_train_key(c), order[c.id])))}
            entered |= {cid for cid in ids if cid in top}
        self._pool_entered_memo = (key, entered)
        return entered

    def _judged_ranked(self) -> list[Candidate]:
        """Every judged candidate a person has not rejected (each has a val score),
        ranked as the pool ranks: val score, a tie by train score (a failed train run
        lowest), then the earlier; the incumbent first."""
        gate = self._gate()
        judged = set(self._gate_accepted) | set(self._gate_rejected)
        order = {c.id: index for index, c in enumerate(self.population)}
        ranked = sorted(
            (c for c in self._evaluated()
             if c.id in judged and c.id in gate.cache and c.id not in self.rejected),
            key=lambda c: (-gate.cache[c.id], -_train_key(c), order[c.id]))
        incumbent = next((c for c in ranked if c.id == gate.current_id), None)
        if incumbent is not None:
            ranked = [incumbent] + [c for c in ranked if c is not incumbent]
        return ranked

    def _incumbent_id(self) -> str | None:
        """The candidate a parent counts as "the incumbent" for (``selection`` report):
        the gate's current candidate under ``val_gate``, else the train leader."""
        if self.config.val_gate and self._custom_strategy is None:
            return self._gate().current_id
        board = self.leaderboard(1)
        return board[0].id if board else None

    def _store_gate_state(self) -> None:
        """Put the gate's state and decisions in the strategy state (not journaled)."""
        self._strategy_state["val_gate"] = {
            **self._gate().state_dict(),
            "accepted": list(self._gate_accepted), "rejected": list(self._gate_rejected),
            "rounds": [list(r) for r in self._gate_rounds],
            # Which rule ranked the winner, so adopt names the one the report names.
            "selection": self.selection_mode(), "pool_size": int(self.config.pool_size),
        }
    # ── val_gate: SkillOpt's early stop and post-training evaluation ────────
    #: The bounds a search ends on normally, as SkillOpt's ``_TRAINED``. After any
    #: other (a person's stop, a spent evaluation, rollout, time or token budget)
    #: the last proposals are not scored and the post-training evaluation is
    #: skipped, with the reason recorded.
    _TRAINED = ("epochs", "perfect_val", "patience_exhausted")

    def _count_stale_epoch(self) -> None:
        """SkillOpt's stale-epoch count, once per epoch after its gating: an epoch
        whose gating did not raise the best val score above the best seen so far is
        stale; one that did resets the count.

        The epoch that started the gate gated the starting population, which is
        SkillOpt's initial val score: it sets the best the count measures from and
        is not counted. Each later epoch gates the batch the one before proposed,
        so it is counted as SkillOpt counts the epoch that proposed that batch.
        """
        if not (self.config.early_stop_patience or self.config.early_stop_at_full_val):
            return
        state = self._strategy_state
        if state.get("stale_counted_epoch") == self.epoch:
            return
        gate = self._gate()
        state["stale_counted_epoch"] = self.epoch
        if state.get("gate_start_epoch", self.epoch) == self.epoch:
            state["early_stop_best"] = gate.best_score
            return
        best = float(state.get("early_stop_best", gate.best_score))
        if gate.best_score > best + 1e-9:
            state["early_stop_best"] = gate.best_score
            state["stale_epochs"] = 0
        else:
            state["stale_epochs"] = int(state.get("stale_epochs", 0)) + 1

    def _early_stop_reason(self) -> str:
        """SkillOpt's early stop: ``perfect_val`` once the best val score is 1.0 (with
        ``early_stop_at_full_val``), ``patience_exhausted`` after
        ``early_stop_patience`` stale epochs in a row; empty otherwise.

        Checked only once an epoch that gated a proposed batch has been counted, as
        SkillOpt checks at an epoch's end: a seed already at 1.0 on val still gets
        one batch proposed and gated."""
        state = self._strategy_state
        counted = state.get("stale_counted_epoch")
        if counted is None or counted == state.get("gate_start_epoch"):
            return ""
        if self.config.early_stop_at_full_val and self._gate().best_score >= 1.0 - 1e-9:
            return "perfect_val"
        patience = self.config.early_stop_patience
        if patience > 0 and int(state.get("stale_epochs", 0)) >= patience:
            return "patience_exhausted"
        return ""

    def _final(self) -> FinalEvaluation:
        """The post-training evaluation, restored from the strategy state."""
        final = FinalEvaluation(
            record=lambda step_type, **fields: self._journal(
                "final_evaluation", step_type=step_type, **fields),
            label="default loop",
        )
        final.load_state_dict(self._strategy_state.get("final_evaluation") or {})
        return final

    def _finish_gated_search(self, run: Run) -> None:
        """What SkillOpt does once training ends normally, under ``val_gate``.

        The last epoch's proposals are evaluated on train and gated, so a run of N
        epochs gates every batch it proposed. Then, with a test part,
        ``FinalEvaluation`` runs: the final candidate's val check, then the seed,
        the best and the final candidate on test with the run seed. Each phase is
        saved, so a resumed run continues where it stopped; a search resumed with
        more epochs evaluates again. A search stopped by a time or token budget that
        held a validation reserve back (``validation_reserve_s``/``_tokens``) leaves
        its last proposals unscored and runs the final evaluation on the reserve. A
        search that ended any other way is not scored further, and the evaluation
        records why it was skipped.
        """
        from refactory_dsh.budgets import RESERVE_RETAINED

        ended = self._ended(run)
        reserved = ended.endswith(RESERVE_RETAINED)
        state = self._strategy_state
        if (state.get("final_evaluation") or {}).get("search_epoch") != self.epoch:
            state.pop("final_evaluation", None)
        final = self._final()

        def save() -> None:
            self._store_gate_state()
            state["final_evaluation"] = {**final.state_dict(), "search_epoch": self.epoch}
            self._save_strategy_state()

        if final.result.get("status") == "done":
            return
        if ended not in self._TRAINED + self.normal_ends and not reserved:
            final.result = {"status": "skipped",
                            "reason": f"the search did not finish: it ended on {ended!r}"}
            save()
            return
        if self._unevaluated() and not reserved:
            self.forward(epoch=self.epoch + 1)
            self._gate_population(run)
            # The last batch's scores and gate decisions, which no later epoch's
            # scheduler writes: without this state.json shows them "not gated".
            self._write_state()
        if not run.has_part(task_module.TEST):
            final.skip(run, run.empty_part_reason(task_module.TEST), save)
            return
        self._reuse_final_selection(final, save)
        final.run(run, self._gate(), seed=self.config.seed, save=save)

    def _reuse_final_selection(self, final: FinalEvaluation, save: Any) -> None:
        """Under ``checkpoint`` the final candidate is a checkpoint, already scored on
        val once: ``final_selection`` takes that score from the gate's cache rather than
        a second rollout (``FinalEvaluation`` would roll it again when it is not the
        best)."""
        gate = self._gate()
        if (self.selection_mode() != "checkpoint" or final.completed
                or gate.current_id not in gate.cache):
            return
        score = gate.cache[gate.current_id]
        final.result = {"status": "running", "final_selection": score}
        final.record("final_selection_eval", action="reused", score=score,
                     best_score=gate.best_score)
        final.completed.add("final_selection")
        save()

    def final_evaluation(self) -> dict[str, Any]:
        """The post-training evaluation's figures, for the report (``val_gate``)."""
        result = self._final().result
        if not result and self.bound:
            # The search stopped before the gated search could finish (a spent budget
            # raised out of a measurement).
            result = {"status": "skipped",
                      "reason": f"the search did not finish: it ended on {self.bound!r}"}
        report: dict[str, Any] = {
            "best_selection": self._gate().best_score if self._gate().best_id else None,
            "test_eval": result.get("status") or "not_run",
        }
        if result.get("reason"):
            report["test_eval_reason"] = result["reason"]
        for key in ("final_selection", "baseline_test", "best_test", "final_test",
                    "best_test_delta", "final_test_delta"):
            report[key] = result.get(key)
        # A delta taken on the items both were scored on (a harness incident left one
        # out): the raw delta and the items left out, beside it.
        for key in ("best_test_delta", "final_test_delta"):
            for extra in (f"{key}_raw", f"{key}_left_out"):
                if extra in result:
                    report[extra] = result[extra]
        return report

    def _save_strategy_state(self) -> None:
        """Journal the strategy's current state unconditionally.

        Called after close_epoch and by run.save_strategy_state(). Journals
        even when empty, because skipping empty dicts would let a resume
        restore a stale non-empty state from an earlier epoch.
        """
        strategy_state = self.strategy_state_dict()
        self._journal("strategy_state", state=strategy_state)

    def epoch_step(self, run: Run | None = None) -> bool:
        """One iteration, and whether there is another.

        The epoch is opened here and closed by the scheduler, and both are journaled.
        A run that died inside an iteration therefore resumes *into* it rather than
        opening it again, which would count the same epoch twice.
        """
        self.open_epoch()
        evaluated = self.forward()
        early = ""
        before = (len(self._gate_accepted), len(self._gate_rejected))
        if self.config.val_gate:
            self._gate_population(run or Run(_loop=self))
            # What this epoch's gating decided, for the epoch's progress line.
            self._epoch_gate = {"kept": len(self._gate_accepted) - before[0],
                                "rejected": len(self._gate_rejected) - before[1],
                                **({"selection": "checkpoint"}
                                   if self.selection_mode() == "checkpoint" else {})}
            # SkillOpt's early stop, checked right after the epoch's gating so a
            # stopping run proposes nothing it will not score.
            self._count_stale_epoch()
            early = self._early_stop_reason()
        self.loss_step()
        self.backward()
        candidates: list[Candidate] = []
        rejections: list[Rejection] = []
        if not early:
            candidates, rejections = self.optimizer_step(run)
            kept = self.step([c for c in candidates if self.accepts(run, c)], run=run)
            for candidate in kept:
                self.update(run, candidate)
        refused = [r for r in rejections if not r.repaired]
        another = self.scheduler(
            evaluated=evaluated, proposed=len(candidates) + len(refused),
            accepted=len(candidates), refused=len(refused),
        )
        if early:
            # A budget a measurement found spent stays the reason (the scheduler kept
            # it); an early stop replaces only the epoch bound.
            if self.bound in ("", "epochs"):
                self.bound = early
                self._journal("bound", bound=early)
            another = False
        self.on_epoch_end(run)
        self.close_epoch()
        self._save_strategy_state()
        return another
