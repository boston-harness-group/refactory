"""Budget seams (F18): rollout counting, finished-trial preservation, downtime.

Each test fails when the corresponding change is reverted.
"""
from __future__ import annotations

import contextlib
import copy
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

    Reverting the _measure() fix (putting ``result.exit_ok = False`` back in the
    post-trial BudgetExhausted handler) makes exit_ok False even though the trial
    completed successfully.  This test uses a mock budget that allows the pre-trial
    checkpoint but fires on the post-trial checkpoint, so the evaluator actually
    runs and the post-trial path is exercised.
    """
    class _PostTrialBudget:
        """Budget mock that fires only on the post-trial checkpoint."""
        def __init__(self):
            self._pre_trial_passed = False

        def checkpoint(self, label=None):
            if label is not None:
                # Pre-trial checkpoint with label: allow.
                self._pre_trial_passed = True
                return
            if self._pre_trial_passed:
                # Post-trial checkpoint (no label): fire.
                raise BudgetExhausted("budget exhausted after trial completed")

        def snapshot(self):
            return {"elapsed_s": 0, "known_tokens": 0, "token_status": "measured",
                    "estimated_cost_usd": 0}

        def retain_validation_reserve(self):
            return contextlib.nullcontext()

    with _patch_mode(BudgetProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
            ),
            project=Path("."),
        )
        loop.init()
        seed = loop.population[0]

        mock_budget = _PostTrialBudget()
        loop.budget = mock_budget
        loop._owns_resource_budget = True

        loop._evaluate(seed)

        # The trial completed successfully; the post-trial budget fired but the
        # finished trial's result must be preserved.
        assert seed.evaluated is True
        assert seed.exit_ok is True, (
            "a finished trial must not be retroactively failed by the budget"
        )
        assert seed.score is not None, (
            "a finished trial must preserve its score"
        )


# ── test_budget_counts_rollouts ────────────────────────────────────────────

def test_budget_counts_rollouts(tmp_path):
    """The loop counts per-item rollouts, and max_rollouts binds the run.

    The batch holds three candidates; the cap admits two evaluations' worth of
    items, so the forward pass must stop at the cap with the third unevaluated,
    and the run must report that the rollout cap is what bound it.
    """
    with _patch_mode(BudgetProbe):
        loop = Loop(
            MODE, LoopConfig(
                # epochs=8 so the loop has enough budget to hit the rollout cap
                # before epochs run out; F14 removed ops so the random sampler
                # produces fewer distinct candidates per batch.
                epochs=8, batch_size=3, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
                max_rollouts=10,
            ),
            project=Path("."),
        )


        def three_wide(seed):
            # The probe has no knobs to sweep, so fill the batch by hand: two more
            # candidates, all unevaluated when forward starts.
            for value in (1, 2):
                child = Candidate.from_state(copy.deepcopy(seed.state), parent=seed.id)
                child.id = f"wide{value:08d}"
                loop.population.append(child)

        loop._seed_population = three_wide
        report = loop.run()

    seed = loop.population[0]
    assert len(loop.population) >= 3, "the batch must outlast the cap"
    per_eval = len(loop._train_item_records[seed.id])
    assert per_eval > 0, "rollouts should be counted per item"
    expected_evals = -(-10 // per_eval)  # evaluations until the cap is reached
    assert loop.evaluations == expected_evals, (
        f"the batch loop must stop at the rollout cap: {loop.evaluations} evaluations, "
        f"expected {expected_evals} at {per_eval} items each"
    )
    assert loop.rollouts == expected_evals * per_eval
    assert loop.rollouts >= 10
    assert report["bound"] == "rollouts"


def test_run_evaluate_refuses_past_rollout_cap(tmp_path):
    """Run.evaluate refuses once max_rollouts is spent, on every part."""
    from refactory_dsh.outerloop import OuterLoopError, Run

    with _patch_mode(BudgetProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run", max_rollouts=1,
            ),
            project=Path("."),
        )
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        run.evaluate(seed)
        assert loop.rollouts >= 1
        assert loop.evaluations < loop.config.max_evaluations
        before = loop.rollouts
        with pytest.raises(OuterLoopError, match="rollout budget is spent"):
            run.evaluate(seed)
        with pytest.raises(OuterLoopError, match="rollout budget is spent"):
            run.evaluate(seed, part="val")
        assert loop.rollouts == before


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


# ── test_downtime_clean_stop ──────────────────────────────────────────────

def test_downtime_clean_stop(tmp_path):
    """Clean stop (pause then resume) charges no gap.

    pause() freezes elapsed_before_s and clears started_at.  record_stop
    writes stop_kind="clean" so resume() knows the stop was graceful.
    After resume, elapsed should be only the 10 s of work.
    """
    budget = RunBudget.open(
        tmp_path / "work",
        ResourceLimits(max_elapsed_s=7200),
    )
    budget.checkpoint()

    # Record some work time.
    with budget._locked() as state:
        state["started_at"] = time.time() - 10  # 10 s of work
        state["elapsed_before_s"] = 0.0

    # Graceful pause — calls record_stop("clean").
    budget.pause()

    # Verify stop_kind was recorded.
    with budget._locked() as state:
        assert state.get("stop_kind") == "clean", (
            f"pause() should record stop_kind='clean', got {state.get('stop_kind')!r}"
        )
        assert state.get("started_at") is None, (
            "pause() should clear started_at"
        )
        pre_elapsed = state.get("elapsed_before_s", 0.0)
        assert pre_elapsed < 20, (
            f"elapsed_before_s should be ~10s of work, got {pre_elapsed}"
        )

    # Resume after a gap.
    budget.resume()

    # Verify stop_kind was cleared on resume.
    with budget._locked() as state:
        assert state.get("stop_kind") is None, (
            "resume() should clear stop_kind"
        )

    snapshot = budget.snapshot()
    # The elapsed time should be approximately the 10s of work.
    assert snapshot["elapsed_s"] < 60, (
        f"expected elapsed_s < 60 after clean stop, got {snapshot['elapsed_s']}"
    )


# ── test_downtime_kill_charges_to_heartbeat ──────────────────────────────

def test_downtime_kill_charges_to_heartbeat(tmp_path):
    """Kill charges time only up to the last heartbeat.

    A kill leaves started_at set (pause was not called).  On resume,
    elapsed is charged up to _last_active and not the gap after it.
    Replaces the 300 s heuristic.

    The test uses a gap UNDER 300 seconds so the old 300 s heuristic
    would NOT subtract the gap (it only fires on gaps > 300 s), while
    the new heartbeat-based code always charges up to _last_active.
    """
    budget = RunBudget.open(
        tmp_path / "work",
        ResourceLimits(max_elapsed_s=7200),
    )
    budget.checkpoint()

    now = time.time()
    # Simulate a kill with a gap of 100 s (under the old 300 s threshold).
    # 300 s of work, heartbeat 100 s ago.
    with budget._locked() as state:
        state["started_at"] = now - 300    # started 300 s ago
        state["_last_active"] = now - 100  # last heartbeat 100 s ago
        state["elapsed_before_s"] = 0.0
        state.pop("stop_kind", None)
        state.pop("stop_at", None)

    budget.resume()

    snapshot = budget.snapshot()
    # The new code should charge ~200 s (from started_at to _last_active).
    # The old 300 s heuristic would charge ~300 s (gap < 300 → no adjustment),
    # so asserting < 250 catches a revert to the old heuristic.
    assert snapshot["elapsed_s"] < 250, (
        f"expected elapsed_s < 250 (200s of work, not 300s), got {snapshot['elapsed_s']}"
    )


# ── test_rollout_counting_mutation ───────────────────────────────────────

def test_rollout_counting_mutation(tmp_path):
    """Rollout counting is guarded: loop.rollouts > 0 after evaluation.

    This test would fail if the rollouts += line were removed.
    """
    with _patch_mode(BudgetProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
            ),
            project=Path("."),
        )
        loop.init()
        seed = loop.population[0]
        loop._evaluate(seed)

        assert loop.rollouts > 0, (
            "rollouts must be > 0 after evaluation; "
            "removing the rollouts += line would break this"
        )
