
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

from refactory_dsh.loop.config import LoopError, OVERFIT_THRESHOLD, SEED_WINNER_NOTE, gap_standard_error



class _PartsMixin:

    def _load_parts_manifest(self) -> None:
        """Load and install the loop-level parts manifest.

        If ``config.parts_manifest`` names a file, copy it to the run root as
        ``parts.json`` and load it.  If the run root already has a ``parts.json``
        (e.g. from a previous run being resumed), load that.
        """
        dest = self.root / task_module.MANIFEST
        if self.config.parts_manifest:
            source = Path(self.config.parts_manifest)
            if not source.exists():
                raise LoopError(
                    f"parts_manifest {self.config.parts_manifest!r} does not exist"
                )
            self.root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        if not dest.exists():
            return
        # A manifest that cannot be read is refused rather than ignored: falling back
        # to the fraction split would grade the run on items nobody chose, and
        # overlapping parts would let val/test items into the optimizer's train set.
        try:
            data = json.loads(dest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise LoopError(f"parts manifest {dest} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LoopError(f"parts manifest {dest} must be a JSON object of id lists")
        parts = {
            k: [str(x) for x in v]
            for k, v in data.items()
            if isinstance(v, list)
        }
        names = [name for name in (task_module.TRAIN, task_module.VAL, task_module.TEST) if name in parts]
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                shared = set(parts[first]) & set(parts[second])
                if shared:
                    raise LoopError(
                        f"parts manifest {dest}: {first} and {second} share items "
                        f"{sorted(shared)}; train, val and test must be disjoint"
                    )
        self._parts_manifest = parts

    def audit_parts(self, winner: Candidate) -> dict[str, Any] | None:
        """Overfit check comparing val vs test scores (like with like).

        When the winner has been evaluated on both 'val' and 'test' parts,
        the overfit check compares the latest val score vs the latest test
        score.  Both are unseen by the optimizer, so the comparison is
        *like with like*.

        Returns ``None`` when the winner does not have both val and test.
        """
        candidate_results = self._part_results.get(winner.id, {})
        val_parts = candidate_results.get("val", [])
        test_parts = candidate_results.get("test", [])
        if not val_parts or not test_parts:
            return None

        val_score = val_parts[-1].get("score")
        test_score = test_parts[-1].get("score")

        if val_score is None or test_score is None:
            return {
                "status": "failed",
                "val": val_score,
                "test": test_score,
                "winner": winner.id,
                "note": self._missing_part_note(
                    [(part, parts[-1]) for part, parts, score in
                     (("val", val_parts, val_score), ("test", test_parts, test_score))
                     if score is None]),
            }

        # The gap the optimization added: the winner's val-test gap minus the seed's on
        # the same parts (the seed's val score and its test score, e.g. FinalEvaluation's
        # baseline_test). A val part easier than the test part shows in the seed too.
        gap = val_score - test_score
        seed = self.population[0] if self.population else None
        seed_results = self._part_results.get(seed.id, {}) if seed is not None else {}
        seed_val = (seed_results.get("val") or [{}])[-1].get("score")
        seed_test = (seed_results.get("test") or [{}])[-1].get("score")
        seed_gap = (seed_val - seed_test if seed_val is not None and seed_test is not None
                    else None)
        manifest = self._parts_manifest or {}
        se = gap_standard_error(val_score, len(manifest.get(task_module.VAL) or ()),
                                test_score, len(manifest.get(task_module.TEST) or ()))
        audit: dict[str, Any] = {
            "status": "measured",
            "val": val_score,
            "test": test_score,
            "gap": gap,
            "seed_gap": seed_gap,
            "threshold": OVERFIT_THRESHOLD,
            "standard_error": se,
            "threshold_basis": (f"fixed {OVERFIT_THRESHOLD} on the gap the search added; "
                                "standard_error (from the val and test part sizes) is for "
                                "information"),
            "winner": winner.id,
            "split": task_module.TEST, "independent": True,
            "split_note": "the test part, which nothing selected on",
        }
        if seed is not None and winner.id == seed.id:
            audit.update(added_gap=0.0, overfit=False, note=SEED_WINNER_NOTE)
            return audit
        if seed_gap is None:
            # Without the seed's own gap the search's part of it cannot be told apart:
            # neither a pass nor a failure.
            audit.update(status="unmeasured", added_gap=None, overfit=None,
                         note=("unmeasured: the seed's "
                               + ("test" if seed_val is not None else "val")
                               + " score is missing, so the gap the search added is unknown"))
            return audit
        added = gap - seed_gap
        audit.update(added_gap=added, overfit=added > OVERFIT_THRESHOLD)
        return audit

    def _audit_split(self) -> dict[str, Any]:
        """Which items the holdout audit runs, and whether they are independent of the
        selection. With a parts manifest the audit's held-out items are its val part
        (``task.manifest_part_ids``): under ``val_gate`` that is the part the search
        selected on, so the audit is not independent, and the report says so rather
        than present it as a check on unseen items. Without a manifest they are the
        Task's ``holdout`` fraction, which nothing selects on."""
        if self._parts_manifest is None:
            return {"split": task_module.HOLDOUT, "independent": True,
                    "split_note": "the holdout fraction of the Task's items, which nothing "
                                  "selected on"}
        if self._custom_strategy is not None:
            return {"split": task_module.VAL, "independent": None,
                    "split_note": f"the val part; the {self._custom_strategy.name} strategy "
                                  "keeps its own selection, which may read it"}
        if self.config.val_gate:
            return {"split": task_module.VAL, "independent": False,
                    "split_note": "the val part, which the search selected on, so this is "
                                  "not an independent check; add a test part for one"}
        return {"split": task_module.VAL, "independent": True,
                "split_note": "the val part, which nothing selected on (train scores selected)"}

    def _missing_part_note(self, missing: Sequence[tuple[str, dict[str, Any]]]) -> str:
        """Why the winner's val or test score is missing: the part and its cause, and,
        when a time or token budget ran out during it, which budget and the reserve
        that keeps the end-of-run scoring its time."""
        causes = []
        for part, occasion in missing:
            error = str(occasion.get("error") or "")
            if "elapsed-time budget exhausted" in error:
                cause = (f"the elapsed-time budget (max_elapsed_s {self.config.max_elapsed_s:g}s) "
                         f"ran out during the winner's {part} scoring"
                         if self.config.max_elapsed_s else
                         f"the elapsed-time budget ran out during the winner's {part} scoring")
                if not self.config.validation_reserve_s:
                    cause += "; validation_reserve_s keeps time for the end-of-run scoring"
            elif "token budget exhausted" in error:
                cause = f"the token budget ran out during the winner's {part} scoring"
                if not self.config.validation_reserve_tokens:
                    cause += "; validation_reserve_tokens keeps tokens for the end-of-run scoring"
            else:
                cause = f"the winner's {part} scoring failed" + (f": {error}" if error else "")
            causes.append(cause)
        parts = " and ".join(part for part, _ in missing)
        return f"{parts} score is missing: " + "; ".join(causes)
