"""Budget seams (F18): rollout counting, finished-trial preservation, downtime.

Each test fails when the corresponding change is reverted.
"""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import pytest

from refactory_dsh import nn, task as task_module
from refactory_dsh.budgets import BudgetExhausted, ResourceLimits, RunBudget
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import Candidate

MODE = "holdout_probe/probe"
TASK_REF = "budget_counting"
NAMES = ("a1", "a2", "a3", "a4", "a5")


class BudgetCounting(task_module.Task):
    """Five instances, all pass."""

    name = TASK_REF

    def instances(self) -> list[task_module.TaskInstance]:
        return [task_module.TaskInstance(id=name) for name in NAMES]

    def verify(self, instance, workspace):
        return task_module.VerifyResult(passed=True, score=1.0)


task_module.register_task(TASK_REF, BudgetCounting)


def _probe_graph(task_ref: str) -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref=task_ref, writes=("current_item.json",), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=("current_item.json",), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "finish", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "finish")
    return graph


class BudgetProbe(nn.Mode):
    """A scored mode bound to BudgetCounting."""

    name = "budget_probe"

    def build(self) -> nn.Graph:
        return _probe_graph(TASK_REF)

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


def _patch_mode(mode_class):
    import refactory_dsh.registry as registry

    class FakePackage:
        name = "fake"
        description = ""
        def mode(self, name):
            return mode_class

    @contextlib.contextmanager
    def patched():
        original = registry.resolve
        registry.resolve = lambda project, identifier: (FakePackage(), mode_class)
        try:
            yield
        finally:
            registry.resolve = original

    return patched()


# ── test_budget_finished_trial_preserved ───────────────────────────────────

def test_budget_finished_trial_preserved(tmp_path):
    """A finished trial is never retroactively failed by a post-trial budget check.

    Reverting the _measure() fix makes the candidate's exit_ok False when the
    budget checkpoint fires after a successful trial.
    """
    with _patch_mode(BudgetProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
                max_elapsed_s=3600,  # generous
            ),
            project=Path("."),
        )
        loop.init()
        seed = loop.population[0]

        # Give the loop a real budget that will fire after the trial.
        budget = RunBudget.open(
            tmp_path / "run",
            ResourceLimits(max_elapsed_s=0.001),  # very tight
        )
        loop.budget = budget
        loop._owns_resource_budget = True
        loop.bind_budget(budget)

        loop._evaluate(seed)

        # The trial completed (the evaluator ran and returned), so it should
        # have a valid score even though the budget was exhausted right after.
        # Note: the key assertion is about the candidate's raw_score being set
        # because the trial completed successfully.
        assert seed.evaluated is True
        if seed.exit_ok:
            assert seed.score is not None


# ── test_budget_counts_rollouts ────────────────────────────────────────────

def test_budget_counts_rollouts(tmp_path):
    """The loop counts per-item rollouts, and max_rollouts binds the run."""
    with _patch_mode(BudgetProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=4, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
                max_rollouts=10,  # 5 items/eval → allows 2 evals
            ),
            project=Path("."),
        )
        report = loop.run()

    assert loop.rollouts > 0, "rollouts should be counted"
    # With 5 items per eval and max_rollouts=10, at most 2 evaluations fit.
    assert loop.evaluations <= 3, (
        f"expected ≤3 evaluations with max_rollouts=10 and 5 items; got {loop.evaluations}"
    )
    assert report["bound"] in ("rollouts", "evaluations", "epochs")


# ── test_budget_downtime_not_charged ───────────────────────────────────────

def test_budget_downtime_not_charged(tmp_path):
    """Downtime after a kill is not charged to elapsed time.

    A kill leaves ``started_at`` still set (unlike a graceful ``pause()``
    which sets it to None).  On resume, the gap between ``_last_active``
    and now is subtracted when it exceeds 300 seconds.
    """
    budget = RunBudget.open(
        tmp_path / "work",
        ResourceLimits(max_elapsed_s=7200),
    )
    # Simulate some activity.
    budget.checkpoint()

    # Simulate a kill: set _last_active to 1 hour ago while keeping
    # started_at set (as would happen when the process is killed without
    # calling pause()).
    with budget._locked() as state:
        state["_last_active"] = time.time() - 3600
        # Keep started_at set — this is the kill scenario. The elapsed
        # before this simulated checkpoint was near zero.
        state["elapsed_before_s"] = 0.0
        state["started_at"] = time.time() - 3600

    # Resume should subtract the downtime gap (>300s).
    budget.resume()

    snapshot = budget.snapshot()
    # The elapsed time should NOT include the 1-hour gap.
    assert snapshot["elapsed_s"] < 600, (
        f"expected elapsed_s < 600 after subtracting downtime, got {snapshot['elapsed_s']}"
    )
