
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

from refactory_dsh.loop.config import INCIDENT_RERUNS, INCIDENT_RERUN_DIR



class _IncidentsMixin:

    def _settle_incidents(
        self, candidate: Candidate, trial_dir: Path, part: str, seed: int | None,
        result: Any, eval_kwargs: dict[str, Any], *, rerun: bool = True, pending: int = 0,
    ) -> tuple[list[str], bool]:
        """Run a trial's harness-incident items again, then leave out of its score those
        still incidents; ``(the ids left out, whether a budget stopped a rerun)``.

        An item a harness fault failed (`items.harness_incident`) has no outcome of
        the candidate's. Each is run again on the same candidate, part and rollout seed,
        alone, in ``<trial>/incident_reruns/<n>``, up to `INCIDENT_RERUNS` times; a rerun's
        record takes the item's place in the trial's records and in its ``item_scores``
        fact. An item still an incident after them keeps its record, with ``score``
        ``None`` and ``not_scored: "harness_incident"``, and leaves ``item_scores``, so
        the mode's own ``score()`` -- run again here on the trial's facts -- counts it in
        neither numerator nor denominator, on every mode and runtime. Each step is
        journaled (``harness_incident``) and shown. Each rerun item counts as a rollout
        (none past ``max_rollouts``), and its tokens and cost are the trial's. A trial that
        did not finish is left as it is: its score is no measurement either way. A mode
        with no ``item_scores`` fact keeps its incidents, as its score cannot leave them
        out: the run warns once (``harness_incident_kept``). With no item left to score
        the trial fails ("every item was a harness incident")."""
        from refactory_dsh.budgets import BudgetExhausted
        from refactory_dsh.items import NOT_SCORED_INCIDENT, SCORES_FACT, harness_incident

        records = list(result.item_records or ())
        ids = list(dict.fromkeys(str(r.get("id")) for r in records if harness_incident(r)))
        if not ids or not result.exit_ok:
            return [], False
        first = list(ids)
        facts = dict(result.facts or {})
        declared = isinstance(facts.get(SCORES_FACT), list)
        if not declared:
            # The mode's score() reads no item_scores fact, so nothing can take the item
            # out of it: the records stay as they are, and the run says so once.
            note = (f"{self.mode_id} has no {SCORES_FACT} fact, so its score cannot leave out "
                    f"the items a harness incident stopped ({', '.join(ids)}): they stay in it")
            with self._state_lock:
                warned = self.__dict__.setdefault("_shared_warned", set())
                if SCORES_FACT not in warned:
                    warned.add(SCORES_FACT)
                    self._journal("harness_incident_kept", id=candidate.id, part=part, items=ids)
                    self._progress("warning", candidate=candidate.id, warning=note)
            # Said as such to the optimizer (`base._item_evidence`), not as a timeout.
            result.item_records = [{**r, "incident_kept": True} if harness_incident(r) else r
                                   for r in records]
            return [], False
        fact_entries = list(facts.get(SCORES_FACT) or [])
        budget_hit = False
        reran = 0
        max_rollouts = getattr(self.config, "max_rollouts", None)
        while ids and rerun and reran < INCIDENT_RERUNS:
            # This evaluation's own items count first: the caller adds them to
            # ``rollouts`` only after it returns (``pending``).
            if max_rollouts is not None and self.rollouts + pending + len(ids) > max_rollouts:
                # A rerun is a rollout of each item it runs: none past the budget.
                break
            reran += 1
            rerun_dir = trial_dir / INCIDENT_RERUN_DIR / str(reran)
            if rerun_dir.exists():
                shutil.rmtree(rerun_dir)
            rerun_dir.mkdir(parents=True)
            self._carry_ledger(rerun_dir)
            task_module.write_subset(rerun_dir, part, self.config.holdout, seed=seed)
            (rerun_dir / task_module.MANIFEST).write_text(json.dumps({part: ids}, indent=2))
            self._progress("harness_incident", candidate=candidate.id, part=part, items=ids,
                           outcome="rerun", attempt=reran, of=INCIDENT_RERUNS)
            try:
                if self.budget:
                    final = part == task_module.HOLDOUT or getattr(self, "_final_scoring", False)
                    self.budget.checkpoint(("final_validation" if final else "work")
                                           if self._owns_resource_budget else None)
                with self.budget.retain_validation_reserve() if self.budget else nullcontext():
                    again = self.evaluator.evaluate(
                        self.mode_class(state=candidate.state), candidate.state, rerun_dir,
                        **eval_kwargs)
            except BudgetExhausted as exc:
                self.bound = str(exc)
                budget_hit = True
                break
            with self._state_lock:
                # Each rerun item is a rollout, toward ``max_rollouts`` like any other.
                self.rollouts += len(again.item_records or ()) or len(ids)
                self._journal("rollouts", rollouts=self.rollouts)
            fresh = {str(r.get("id")): r for r in again.item_records or () if str(r.get("id")) in ids}
            fresh_facts = {str(e.get("id")): e for e in (again.facts or {}).get(SCORES_FACT) or ()
                           if isinstance(e, dict) and str(e.get("id")) in fresh}
            records = [{**fresh[str(r.get("id"))], "incident_reruns": reran}
                       if str(r.get("id")) in fresh else r for r in records]
            fact_entries = [fresh_facts.get(str(e.get("id")), e) if isinstance(e, dict) else e
                            for e in fact_entries]
            ids = [i for i in ids if i not in fresh or harness_incident(fresh[i])]
        left_out = set(ids)
        if left_out:
            records = [{**r, "score": None, "not_scored": NOT_SCORED_INCIDENT,
                        "incident_reruns": reran}
                       if str(r.get("id")) in left_out else r for r in records]
            fact_entries = [e for e in fact_entries
                            if not (isinstance(e, dict) and str(e.get("id")) in left_out)]
        facts[SCORES_FACT] = fact_entries
        result.facts = facts
        result.item_records = records
        if not fact_entries:
            result.exit_ok = False
            result.error = (f"every item was a harness incident ({', '.join(ids)}); "
                            "nothing was scored")
        else:
            try:
                result.score = float(self.mode_class(state=candidate.state).score(result))
            except Exception as exc:
                result.exit_ok = False
                result.error = f"score() raised {type(exc).__name__}: {exc}"
        if records:
            (trial_dir / "item_records.json").write_text(
                json.dumps(records, indent=2, default=str))
        fields = {"part": part, "items": first, "reran": reran,
                  "recovered": [i for i in first if i not in left_out], "not_scored": ids}
        with self._state_lock:
            self._note_incident({"candidate": candidate.id, **fields,
                                 "workdir": self._relative(trial_dir)})
            self._journal("harness_incident", id=candidate.id,
                          workdir=self._relative(trial_dir), **fields)
        self._progress("harness_incident", candidate=candidate.id, part=part, items=first,
                       outcome="not_scored" if ids else "recovered", not_scored=ids,
                       reran=reran)
        return ids, budget_hit

    def _note_incident(self, entry: dict[str, Any]) -> None:
        """Keep one evaluation's harness incidents for the report: an evaluation run again
        in the same folder (an interrupted one's retry, a resume) replaces its earlier
        attempt's entry rather than adding to it."""
        key = (entry.get("candidate"), entry.get("part"), entry.get("workdir"))
        self._harness_incidents = [
            e for e in self._harness_incidents
            if (e.get("candidate"), e.get("part"), e.get("workdir")) != key] + [entry]
