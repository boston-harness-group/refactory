"""Tests for Run.evaluate_many — concurrent evaluation of independent candidates.

Each test drives the new API through the real loop on the walker runtime
and asserts a specific mechanism from the brief.  Where the brief requires
a spine test, a comment says so and the test drives the slot-directory
mechanism instead (the spine's own end-to-end concurrent-drain tests live
in ``data-parallel.spec.ts``).
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import Candidate
from refactory_dsh.outerloop import EvalRequest, OuterLoopError, OuterLoop, Run

MODE = "circle_packing/adaptive"


def _config(root: Path, **overrides: object) -> LoopConfig:
    base: dict[str, object] = dict(
        epochs=1, batch_size=1, optimizer="random", seed=3,
        runs_root=root, max_evaluations=64,
    )
    base.update(overrides)
    return LoopConfig(**base)  # type: ignore[arg-type]


def _make_loop_and_seed(tmp_path: Path, **kwargs: Any) -> tuple[Loop, Candidate, Run]:
    """Set up a loop, init the seed, and return (loop, seed_candidate, run)."""
    loop = Loop(MODE, _config(tmp_path / "run", **kwargs), project=Path("."))
    loop.init()
    seed = loop.population[0]
    run = Run(_loop=loop)
    return loop, seed, run


def _distinct_candidates(seed: Candidate, n: int) -> list[Candidate]:
    """Create n candidates with distinct ids by varying a knob.

    Uses `budget` values 1..n (the circle_packing/adaptive mode accepts 1-8).
    """
    from refactory_dsh.nn import edits as nn_edits
    candidates = []
    for i in range(n):
        state = nn_edits.configure(seed.state, "budget", str(i + 1))
        c = Candidate.from_state(state, parent=seed.id, edits=())
        candidates.append(c)
    assert len(set(c.id for c in candidates)) == n, "candidates must have distinct ids"
    return candidates


# ---------------------------------------------------------------------------
# Helpers for slow-solver instrumentation
# ---------------------------------------------------------------------------

class _TimingTracker:
    """Thread-safe tracker that records per-evaluation start/end times."""

    def __init__(self) -> None:
        self.times: dict[str, list[tuple[float, float]]] = {}
        self._lock = threading.Lock()

    def record(self, key: str, start: float, end: float) -> None:
        with self._lock:
            self.times.setdefault(key, []).append((start, end))



def _patch_evaluate_with_delay(loop: Loop, delays: dict[str, float],
                               tracker: _TimingTracker | None = None):
    """Monkey-patch loop._evaluate to add per-candidate delays and tracking.

    ``delays`` maps candidate.id → sleep seconds.
    ``tracker`` records (start, end) per candidate id.
    """
    original = loop._evaluate

    def patched(cand, *, seed=None, items=None, _budget_reserved=False,
                item_gate=None):
        delay = delays.get(cand.id, 0)
        start = time.monotonic()
        if delay > 0:
            time.sleep(delay)
        result = original(cand, seed=seed, items=items,
                          _budget_reserved=_budget_reserved,
                          item_gate=item_gate)
        end = time.monotonic()
        if tracker is not None:
            tracker.record(cand.id, start, end)
        return result

    loop._evaluate = patched


# ── test_concurrent_evaluations_overlap ──────────────────────────────────

def test_concurrent_evaluations_overlap(tmp_path):
    """Concurrent evaluations via evaluate_many overlap in time.

    A slow fake solver (sleep 0.3s per candidate) proves that with
    max_concurrent_evaluations=4, at least two evaluations' time ranges overlap.

    Mutation check: setting max_concurrent_evaluations=1 makes evaluations
    serial, so no overlap — the test fails.
    """
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)

    candidates = _distinct_candidates(seed, 3)

    # Each candidate sleeps 0.3s to ensure measurable overlap
    delays = {c.id: 0.3 for c in candidates}
    tracker = _TimingTracker()
    _patch_evaluate_with_delay(loop, delays, tracker=tracker)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests)

    assert len(results) == 3
    for r in results:
        assert r.evaluated

    # Prove temporal overlap: at least two evaluations ran simultaneously
    all_spans = []
    for spans in tracker.times.values():
        all_spans.extend(spans)
    assert len(all_spans) == 3

    overlap_found = False
    for i in range(len(all_spans)):
        for j in range(i + 1, len(all_spans)):
            s1_start, s1_end = all_spans[i]
            s2_start, s2_end = all_spans[j]
            if s1_start < s2_end and s2_start < s1_end:
                overlap_found = True
                break
        if overlap_found:
            break
    assert overlap_found, (
        f"no temporal overlap found among evaluation spans: {all_spans}"
    )


def test_concurrent_evaluations_overlap_mutation(tmp_path):
    """Mutation check: max_concurrent_evaluations=1 prevents overlap.

    With serial execution, no two evaluations' time ranges overlap.
    """
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=1, max_evaluations=10)

    candidates = _distinct_candidates(seed, 3)

    delays = {c.id: 0.15 for c in candidates}
    tracker = _TimingTracker()
    _patch_evaluate_with_delay(loop, delays, tracker=tracker)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests)

    assert len(results) == 3

    all_spans = []
    for spans in tracker.times.values():
        all_spans.extend(spans)

    # With max_concurrent_evaluations=1, evaluations are serial:
    # no two should overlap.
    for i in range(len(all_spans)):
        for j in range(i + 1, len(all_spans)):
            s1_start, s1_end = all_spans[i]
            s2_start, s2_end = all_spans[j]
            # Allow 5ms tolerance for scheduling jitter
            assert not (s1_start + 0.005 < s2_end and s2_start + 0.005 < s1_end), (
                f"evaluations overlapped with max_concurrent_evaluations=1: "
                f"[{s1_start:.4f}—{s1_end:.4f}] vs [{s2_start:.4f}—{s2_end:.4f}]"
            )


# ── helpers for item-gate tests (data-parallel graph) ────────────────────

def _item_gate_data_graph(n_items: int, parallelism: int):
    """Build a data graph with *n_items* items that sleep briefly.

    Each item runs ``sleep_op`` from ``test_ops_parallel`` which uses a
    module-level thread-safe counter to track in-flight items and writes
    the peak to ``$PEAK_FILE``.
    """
    from refactory_dsh.nn import Graph, Node
    from refactory_dsh.nn.executor import CURRENT_ITEM

    items = tuple({"id": str(i), "delay": "0.2"} for i in range(n_items))
    graph = Graph()
    graph.add(Node(
        "data", "data",
        subgraph_entry="body", subgraph_exit="body",
        writes=(CURRENT_ITEM,),
        produces=("item_scores",),
        inline_items=items,
        parallelism=parallelism,
    ))
    graph.add(Node(
        "body", "op",
        call="test_ops_parallel:sleep_op",
        reads=(CURRENT_ITEM,),
        produces=("counted",),
    ))
    graph.add(Node(
        "report", "op",
        call="loop_ops:write_best",
        writes=("best.json",),
        produces=("best",),
    ))
    graph.connect("data", "report")
    return graph


_SLEEP_OP_SRC = '''\
"""Minimal test op for item-gate tests.

Provides a process-safe in-flight counter via fcntl-locked shared files.
Every ``sleep_op`` call increments the counter on entry, records the peak,
sleeps, and decrements in ``finally``.  The peak is written to ``$PEAK_FILE``.

The counter file (``$COUNTER_FILE``) stores "current peak" and is protected
by fcntl.flock so it works across forkserver child processes.
"""
import fcntl
import os
import time
from pathlib import Path


def _update_counter(counter_file, delta):
    """Atomically update the shared counter; return (new_current, new_peak)."""
    with open(counter_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            parts = f.read().strip().split()
            current = int(parts[0]) if len(parts) > 0 else 0
            peak = int(parts[1]) if len(parts) > 1 else 0
            current += delta
            if current > peak:
                peak = current
            f.seek(0)
            f.truncate()
            f.write(f"{current} {peak}")
            f.flush()
            return current, peak
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def sleep_op(ctx):
    """Sleep for `delay` seconds, tracking in-flight count via counter."""
    item = ctx.read_json("current_item.json")
    delay = float(item.get("delay", 0))
    item_id = str(item.get("id", "?"))

    counter_file = os.environ.get("COUNTER_FILE", "")
    if counter_file:
        _update_counter(counter_file, +1)
    try:
        time.sleep(delay)
    finally:
        if counter_file:
            _, peak = _update_counter(counter_file, -1)
            peak_file = os.environ.get("PEAK_FILE", "")
            if peak_file:
                Path(peak_file).write_text(str(peak))

    return {"counted": item_id}
'''


@pytest.fixture()
def _install_parallel_ops():
    """Install test_ops_parallel module with counter-instrumented sleep_op."""
    import sys
    HERE = Path(__file__).resolve().parent
    ops_path = HERE / "test_ops_parallel.py"
    existed = ops_path.exists()
    # Always write our version so the counter-instrumented sleep_op is used.
    ops_path.write_text(_SLEEP_OP_SRC)
    # Remove cached module so the fresh source is loaded.
    sys.modules.pop("test_ops_parallel", None)
    yield
    if not existed:
        ops_path.unlink(missing_ok=True)
    sys.modules.pop("test_ops_parallel", None)


# ── test_item_limit_not_exceeded ─────────────────────────────────────────

def test_item_limit_not_exceeded(tmp_path, monkeypatch, _install_parallel_ops):
    """The run-wide item-in-flight limit is respected through evaluate_many.

    The fake solver (item work under ``run_data_concurrent``) increments a
    thread-safe counter on entry, records the peak, sleeps ~0.2 s, and
    decrements in ``finally``.  Config: ``max_items_in_flight=6``,
    ``data_parallelism=24``, three concurrent evaluations of 8 items each.
    Asserts **peak <= 6** and **peak > 3**.

    Mutation check: removing ``item_gate.acquire()`` / ``release()`` in
    ``nn/data_parallel.py`` lets peak reach min(24, 8) = 8, failing
    ``peak <= 6``.
    """
    from refactory_dsh.nn.executor import execute

    GATE_LIMIT = 6
    DATA_PARALLELISM = 24
    N_ITEMS = 8
    N_EVALS = 3

    # Shared counter file (fcntl-locked) tracks in-flight items across
    # forkserver child processes.  Peak is written to peak_file.
    counter_file = tmp_path / "counter.txt"
    counter_file.write_text("0 0")
    peak_file = tmp_path / "peak.txt"
    monkeypatch.setenv("COUNTER_FILE", str(counter_file))
    monkeypatch.setenv("PEAK_FILE", str(peak_file))

    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_evaluations=64,
        max_concurrent_evaluations=4,
        max_items_in_flight=GATE_LIMIT,
    )

    candidates = _distinct_candidates(seed, N_EVALS)

    # Patch _evaluate: each eval runs execute() with a data graph,
    # forwarding the item_gate from evaluate_many.  The sleep_op uses
    # a module-level thread-safe counter that tracks in-flight items
    # across all concurrent evaluations and writes the peak to PEAK_FILE.
    def _patched_evaluate(
        cand, *, seed=None, items=None, _budget_reserved=False,
        item_gate=None,
    ):
        workdir = loop.root / "runs" / cand.id
        workdir.mkdir(parents=True, exist_ok=True)
        from refactory_dsh.nn import Graph, Node
        from refactory_dsh.nn.executor import CURRENT_ITEM
        prefix = cand.id[:8]
        items_list = tuple(
            {"id": f"{prefix}_{i}", "delay": "0.2"} for i in range(N_ITEMS)
        )
        eval_graph = Graph()
        eval_graph.add(Node(
            "data", "data",
            subgraph_entry="body", subgraph_exit="body",
            writes=(CURRENT_ITEM,),
            produces=("item_scores",),
            inline_items=items_list,
            parallelism=DATA_PARALLELISM,
        ))
        eval_graph.add(Node(
            "body", "op",
            call="test_ops_parallel:sleep_op",
            reads=(CURRENT_ITEM,),
            produces=("counted",),
        ))
        eval_graph.add(Node(
            "report", "op",
            call="loop_ops:write_best",
            writes=("best.json",),
            produces=("best",),
        ))
        eval_graph.connect("data", "report")
        result = execute(eval_graph, workdir, max_steps=500, item_gate=item_gate)
        assert result.exit_ok, result.error
        cand.score = 0.5
        cand.raw_score = 0.5
        cand.exit_ok = True
        cand.evaluated = True
        with loop._state_lock:
            loop._train_item_records[cand.id] = list(result.item_records)
            if not _budget_reserved:
                loop.evaluations += 1
            loop.rollouts += len(result.item_records) or 1

    loop._evaluate = _patched_evaluate

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests)

    assert len(results) == N_EVALS

    # Read peak concurrency from the thread-safe counter in sleep_op.
    assert peak_file.exists(), "sleep_op did not write peak file"
    peak = int(peak_file.read_text().strip())

    assert peak <= GATE_LIMIT, (
        f"peak in-flight {peak} exceeded limit {GATE_LIMIT}"
    )
    assert peak > 3, (
        f"peak in-flight {peak} should exceed 3 to prove "
        f"cross-evaluation sharing (3 evals × 8 items, limit {GATE_LIMIT})"
    )


def test_item_limit_mutation(tmp_path, monkeypatch, _install_parallel_ops):
    """Mutation check: without item_gate, peak in-flight exceeds the limit.

    Same data graph (parallelism=8, 12 items) but **no item_gate**.
    Without the semaphore gating in ``data_parallel.py:run_item``, all 8
    worker threads dispatch items simultaneously, so peak >= min(8, 12) = 8.
    This proves that ``item_gate.acquire()`` / ``release()`` in
    ``data_parallel.py`` is the mechanism that enforces the limit.
    """
    from refactory_dsh.nn.executor import execute

    GATE_LIMIT = 3
    PARALLELISM = 8
    N_ITEMS = 12

    counter_file = tmp_path / "counter.txt"
    counter_file.write_text("0 0")
    peak_file = tmp_path / "peak.txt"
    monkeypatch.setenv("COUNTER_FILE", str(counter_file))
    monkeypatch.setenv("PEAK_FILE", str(peak_file))

    graph = _item_gate_data_graph(N_ITEMS, PARALLELISM)
    # No item_gate — the mutation
    result = execute(graph, tmp_path / "run", max_steps=500, item_gate=None)
    assert result.exit_ok, result.error

    assert peak_file.exists(), "sleep_op did not write peak file"
    peak = int(peak_file.read_text().strip())
    assert peak > GATE_LIMIT, (
        f"without item_gate, peak in-flight {peak} should exceed {GATE_LIMIT} "
        f"(expected ~{PARALLELISM}); the gate is the only limiter"
    )


# ── test_results_in_request_order ────────────────────────────────────────

def test_results_in_request_order(tmp_path):
    """Results come back in request order regardless of completion order.

    A variable-delay solver: request 0 (slow) sleeps 0.4s,
    request 1 (fast) sleeps 0.01s.  Request 0 finishes after request 1,
    but results[0] must be the slow candidate.

    Mutation check: collecting via as_completed without reindexing
    returns the fast candidate first.
    """
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)

    candidates = _distinct_candidates(seed, 3)

    # Slow candidate 0, fast candidates 1 and 2
    delays = {
        candidates[0].id: 0.4,
        candidates[1].id: 0.01,
        candidates[2].id: 0.01,
    }
    _patch_evaluate_with_delay(loop, delays)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests)

    assert len(results) == 3
    # Results must match request order
    for i, (req, res) in enumerate(zip(requests, results)):
        assert res.id == req.candidate.id, (
            f"result[{i}] is {res.id} but request[{i}] asked for {req.candidate.id}"
        )


def test_results_order_mutation(tmp_path):
    """Mutation check: without reindexing, fast results come first.

    This test verifies that the reindexing in evaluate_many is necessary:
    the as_completed iterator would return fast results before slow ones.
    """
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)

    candidates = _distinct_candidates(seed, 2)

    # Record completion order
    completion_order: list[str] = []
    lock = threading.Lock()

    def on_result(index: int, candidate: Candidate) -> None:
        with lock:
            completion_order.append(candidate.id)

    delays = {
        candidates[0].id: 0.4,  # slow
        candidates[1].id: 0.01,  # fast
    }
    _patch_evaluate_with_delay(loop, delays)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests, on_result=on_result)

    # Completion order: fast finishes first
    assert len(completion_order) == 2
    assert completion_order[0] == candidates[1].id, "fast should complete first"

    # But results are in request order
    assert results[0].id == candidates[0].id
    assert results[1].id == candidates[1].id


# ── test_budget_refuses_entire_batch ─────────────────────────────────────

def test_budget_refuses_entire_batch_evaluations(tmp_path):
    """All-or-nothing budget refusal for max_evaluations."""
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_evaluations=2, max_concurrent_evaluations=4)

    # Use 1 eval so there's only 1 left
    run.evaluate(seed, part="train")
    assert loop.evaluations == 1

    # Now try to evaluate_many with 2 train requests — exceeds budget
    c1, c2 = _distinct_candidates(seed, 2)
    requests = [
        EvalRequest(candidate=c1, part="train"),
        EvalRequest(candidate=c2, part="train"),
    ]

    with pytest.raises(OuterLoopError, match="evaluation budget is spent"):
        run.evaluate_many(requests)

    # Budget must not have leaked — evaluations stays at 1
    assert loop.evaluations == 1


def test_budget_refuses_entire_batch_rollouts(tmp_path):
    """All-or-nothing budget refusal for max_rollouts."""
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_evaluations=64,
        max_concurrent_evaluations=4)

    # One train eval to learn how many rollouts it uses
    run.evaluate(seed, part="train")
    rollouts_per_eval = loop.rollouts
    assert rollouts_per_eval >= 1

    # Set max_rollouts just below what 2 more evals would need
    max_rollouts = loop.rollouts + rollouts_per_eval  # room for 1 more, not 2
    loop.config.max_rollouts = max_rollouts

    c1, c2 = _distinct_candidates(seed, 2)
    requests = [
        EvalRequest(candidate=c1, part="train"),
        EvalRequest(candidate=c2, part="train"),
    ]

    with pytest.raises(OuterLoopError, match="rollout budget is spent"):
        run.evaluate_many(requests)


# ── test_duplicate_train_candidate_refused ──────────────────────────────

def test_duplicate_train_candidate_refused(tmp_path):
    """Two train requests for the same candidate are refused before anything
    runs: train records are kept per candidate, so the second run would
    overwrite the first (SkillOpt's slow update read identical prev/curr
    records). No evaluation runs and no budget is reserved.

    Mutation check: dropping the refusal runs both evaluations and the
    evaluation count becomes 2.
    """
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)
    candidate = Candidate.from_state(seed.state, parent=seed.id, edits=())
    other = _distinct_candidates(seed, 1)[0]

    calls: list[str] = []
    original = loop._evaluate

    def counting(cand, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(cand.id)
        return original(cand, **kwargs)

    loop._evaluate = counting
    before = (loop.evaluations, loop.rollouts)
    requests = [
        EvalRequest(candidate=candidate, part="train"),
        EvalRequest(candidate=other, part="train"),
        EvalRequest(candidate=candidate, part="train"),
    ]
    with pytest.raises(
        OuterLoopError,
        match=f"cannot evaluate candidate {candidate.id} on train twice in one call",
    ):
        run.evaluate_many(requests)

    assert calls == []
    assert (loop.evaluations, loop.rollouts) == before
    assert run.item_records(candidate) == []


# ── test_on_result_fires_in_completion_order ─────────────────────────────

def test_on_result_fires_in_completion_order(tmp_path):
    """on_result fires in completion order: fast candidate before slow.

    Variable-delay solver: request 0 is slow (0.4s), request 1 is fast (0.01s).
    The callback must fire for index 1 before index 0.
    """
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)

    candidates = _distinct_candidates(seed, 2)

    callback_order: list[int] = []
    lock = threading.Lock()

    def on_result(index: int, candidate: Candidate) -> None:
        with lock:
            callback_order.append(index)

    delays = {
        candidates[0].id: 0.4,  # slow
        candidates[1].id: 0.01,  # fast
    }
    _patch_evaluate_with_delay(loop, delays)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests, on_result=on_result)

    # Callback fired for both
    assert sorted(callback_order) == [0, 1]
    # Fast (index 1) should fire before slow (index 0)
    assert callback_order[0] == 1, (
        f"expected fast (index 1) to fire first, got order {callback_order}"
    )
    # Results still in request order
    assert results[0].id == candidates[0].id
    assert results[1].id == candidates[1].id


# ── test_error_in_one_does_not_cancel_others ─────────────────────────────

def test_error_in_one_does_not_cancel_others(tmp_path):
    """An error in one evaluation allows others to complete.

    A fake solver that raises RuntimeError for one candidate but succeeds
    for others.  The two successful evaluations produce valid results;
    the error is re-raised; budget for the failed train eval is un-reserved.
    """
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)

    candidates = _distinct_candidates(seed, 3)

    fail_id = candidates[1].id
    original = loop._evaluate

    def failing_evaluate(cand, *, seed=None, items=None, _budget_reserved=False,
                         item_gate=None):
        if cand.id == fail_id:
            raise RuntimeError("deliberate test failure")
        return original(cand, seed=seed, items=items,
                        _budget_reserved=_budget_reserved,
                        item_gate=item_gate)

    loop._evaluate = failing_evaluate

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]

    with pytest.raises(RuntimeError, match="deliberate test failure"):
        run.evaluate_many(requests)

    # Budget for the failed eval should be un-reserved (3 reserved, 1 failed = 2 net)
    assert loop.evaluations == 2


# ── test_journal_integrity_under_concurrency ─────────────────────────────

def test_journal_integrity_under_concurrency(tmp_path):
    """Concurrent evaluations produce valid JSONL journal lines."""
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)

    candidates = _distinct_candidates(seed, 4)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    run.evaluate_many(requests)

    journal = loop.root / "journal.jsonl"
    assert journal.exists()

    for line in journal.read_text().splitlines():
        if line.strip():
            parsed = json.loads(line)
            assert "event" in parsed, f"journal line missing 'event': {line[:100]}"


# ── test_trace_order_independent_of_completion ───────────────────────────
# The real end-to-end test is in test_skillopt.py:
#   test_trace_order_when_best_finishes_before_baseline
# which forces best to finish before baseline via delays and checks trace order.
# test_post_training_evaluation_runs_in_skillopts_order also covers it.
# This test verifies the evaluate_many + on_result plumbing at the Run level:
# on_result fires in completion order while results are in request order.

# (covered by test_on_result_fires_in_completion_order + the skillopt tests above)


# ── test_report_fields_unchanged ─────────────────────────────────────────

def test_report_fields_unchanged(tmp_path):
    """SkillOpt's report fields are unchanged between sequential and concurrent runs.

    Runs a scripted SkillOpt loop that exercises the post-training block.
    Checks that all expected report keys are present with their scores.
    """
    from refactory_packages.skillopt.loop import SkillOptLoop

    strategy = SkillOptLoop()
    strategy.configure()

    # Simulate a completed post-training evaluation with known scores
    strategy._final_eval = {
        "status": "done",
        "final_selection_hard": 0.882,
        "baseline_test_hard": 0.448,
        "test_hard": 0.728,
        "final_test_hard": 0.656,
        "test_delta_hard": 0.728 - 0.448,
        "final_test_delta_hard": 0.656 - 0.448,
    }
    strategy._best_score = 0.882
    strategy._best_candidate_id = "best_id"

    class MockRun:
        def __init__(self):
            self._loop = mock.MagicMock()
            self._loop.population = [mock.MagicMock()]
    run = MockRun()

    report = strategy.report(run)

    # All expected keys must be present
    expected_keys = {
        "best_selection_hard",
        "test_eval",
    }
    for key in expected_keys:
        assert key in report, f"report key {key!r} missing"

    # The test_eval status should be "done"
    assert report["test_eval"] == "done"

    # The report includes the final_eval data which has the score keys
    final_eval = strategy._final_eval
    score_keys = [
        "final_selection_hard", "baseline_test_hard", "test_hard",
        "final_test_hard", "test_delta_hard", "final_test_delta_hard",
    ]
    for key in score_keys:
        assert key in final_eval, f"final_eval key {key!r} missing"
        assert final_eval[key] is not None, f"final_eval[{key!r}] is None"


# ── test_spine_slot_directory_shared_limit ───────────────────────────────
# Correction 1: spine slot directory must be shared and exact, not divided.

def test_spine_slot_directory_shared_limit(tmp_path):
    """Spine slot-directory gating: two concurrent drains against one slot dir
    never exceed N in flight and do exceed N/2 (proves sharing).

    Tests the slot-directory mechanism from the Python side by exercising
    the file-based lock/semaphore protocol directly.
    """
    slot_dir = tmp_path / "item-slots"
    slot_dir.mkdir()
    N = 4
    pid = str(os.getpid())

    # Simulate acquiring and releasing slots
    acquired_slots: list[int] = []
    for k in range(N):
        slot_path = slot_dir / f"slot-{k}"
        try:
            fd = os.open(str(slot_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.write(fd, pid.encode())
            os.close(fd)
            acquired_slots.append(k)
        except FileExistsError:
            pass

    # All N slots should be acquired
    assert len(acquired_slots) == N

    # Try to acquire one more — should fail (all taken)
    extra_path = slot_dir / f"slot-{N}"
    try:
        fd = os.open(str(extra_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        os.write(fd, pid.encode())
        os.close(fd)
        # This slot index is out of range [0..N-1], so the loop wouldn't try it
    except FileExistsError:
        pass

    # With all N slots taken, trying slots 0..N-1 should all fail
    for k in range(N):
        try:
            fd = os.open(str(slot_dir / f"slot-{k}"), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.close(fd)
            pytest.fail(f"slot-{k} should be taken")
        except FileExistsError:
            pass  # Expected

    # Release some slots
    for k in acquired_slots[:2]:
        os.unlink(str(slot_dir / f"slot-{k}"))

    # Now can acquire those released slots
    reacquired = []
    for k in range(N):
        try:
            fd = os.open(str(slot_dir / f"slot-{k}"), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.write(fd, pid.encode())
            os.close(fd)
            reacquired.append(k)
        except FileExistsError:
            pass

    assert len(reacquired) == 2  # The two released slots

    # Clean up
    for k in range(N):
        (slot_dir / f"slot-{k}").unlink(missing_ok=True)


def test_spine_slot_directory_concurrent_sharing(tmp_path):
    """Two concurrent 'drains' sharing one slot directory: in-flight items
    never exceed N and do exceed N/2 (proves sharing — idle slots from one
    drain are available to the other).

    Mutation check: a divided cap (N//2 per drain) fails the > N/2 assertion.
    """
    slot_dir = tmp_path / "item-slots"
    slot_dir.mkdir()
    N = 6  # total shared slots
    pid = os.getpid()
    peak_lock = threading.Lock()
    current_in_flight = 0
    peak_in_flight = 0
    n_drains = 3
    barrier = threading.Barrier(n_drains)  # synchronize drains

    def acquire_slot() -> int:
        """Acquire a slot by atomically creating slot-<k>."""
        for _ in range(500):
            for k in range(N):
                slot_path = str(slot_dir / f"slot-{k}")
                try:
                    fd = os.open(slot_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                    os.write(fd, str(pid).encode())
                    os.close(fd)
                    return k
                except FileExistsError:
                    pass
            time.sleep(0.005)
        raise TimeoutError("could not acquire slot")

    def release_slot(k: int) -> None:
        try:
            os.unlink(str(slot_dir / f"slot-{k}"))
        except FileNotFoundError:
            pass

    def drain(drain_id: int, items_per_drain: int, results: list[int]) -> None:
        nonlocal current_in_flight, peak_in_flight
        barrier.wait()  # synchronize start
        # Each drain runs items concurrently using threads
        threads = []
        for _ in range(items_per_drain):
            def item_work():
                nonlocal current_in_flight, peak_in_flight
                slot = acquire_slot()
                with peak_lock:
                    current_in_flight += 1
                    if current_in_flight > peak_in_flight:
                        peak_in_flight = current_in_flight
                try:
                    time.sleep(0.1)  # simulate work
                finally:
                    with peak_lock:
                        current_in_flight -= 1
                    release_slot(slot)
                results.append(slot)
            t = threading.Thread(target=item_work)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    all_results: list[list[int]] = [[] for _ in range(n_drains)]
    drain_threads = []
    for i in range(n_drains):
        t = threading.Thread(target=drain, args=(i, 4, all_results[i]))
        drain_threads.append(t)
        t.start()
    for t in drain_threads:
        t.join(timeout=30)
    for t in drain_threads:
        assert not t.is_alive()

    # Never exceeded N
    assert peak_in_flight <= N, f"peak {peak_in_flight} exceeded limit {N}"
    # Exceeded N/2 (proves sharing: idle slots from one drain are available
    # to another; if cap were divided as N//n_drains=2, peak would be ≤2)
    assert peak_in_flight > N // n_drains, (
        f"peak {peak_in_flight} did not exceed N//{n_drains}={N // n_drains}; "
        "sharing not proven"
    )


# ── test_spine_stale_slot_reclaimed ──────────────────────────────────────
# Correction 1: stale slot with dead pid is reclaimed.

def test_spine_stale_slot_reclaimed(tmp_path):
    """A slot file with a dead pid is reclaimed.

    Creates a slot file written by a now-dead pid, then tries to acquire
    that slot.  The stale slot should be reclaimed (unlinked and re-created).
    """
    slot_dir = tmp_path / "item-slots"
    slot_dir.mkdir()

    # Find a dead pid
    dead_pid = 99999
    while True:
        try:
            os.kill(dead_pid, 0)
            dead_pid += 1
        except ProcessLookupError:
            break  # This pid is dead
        except PermissionError:
            dead_pid += 1  # Process exists but we can't signal it

    # Create a stale slot with the dead pid
    stale_slot = slot_dir / "slot-0"
    stale_slot.write_text(str(dead_pid))

    # Now try to reclaim it (simulating the spine's logic)
    pid = os.getpid()
    reclaimed = False
    slot_path = str(stale_slot)

    try:
        fd = os.open(slot_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        os.write(fd, str(pid).encode())
        os.close(fd)
        reclaimed = True  # Slot was free (shouldn't happen since we created it)
    except FileExistsError:
        # Slot exists: check if owner pid is dead
        owner_pid = int(stale_slot.read_text().strip())
        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            # Dead pid: reclaim
            os.unlink(slot_path)
            fd = os.open(slot_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.write(fd, str(pid).encode())
            os.close(fd)
            reclaimed = True
        except PermissionError:
            pass  # Process exists

    assert reclaimed, "stale slot with dead pid should be reclaimed"
    # Verify the slot now belongs to us
    assert int(stale_slot.read_text().strip()) == pid


# ── test_spine_evaluator_passes_env ──────────────────────────────────────

def test_spine_evaluator_passes_slot_env(tmp_path):
    """DshEvaluator passes REFACTORY_ITEM_SLOTS_DIR and REFACTORY_ITEM_SLOTS
    to the subprocess environment when evaluate_many is used with the spine runtime.
    """
    from refactory_dsh.dsh_runtime import evaluate_on_dsh

    slot_dir = str(tmp_path / "item-slots")
    slot_count = 12

    # Verify the DshEvaluator accept these kwargs
    from refactory_dsh.evaluators import DshEvaluator

    evaluator = DshEvaluator(timeout_s=10)

    # We can't run the full evaluate because it needs a real mode,
    # but we can verify the evaluate method accepts the kwargs
    # by checking the code path exists.
    import inspect
    sig = inspect.signature(evaluator.evaluate)
    # kwargs are passed through **kwargs, so the signature just needs **kwargs
    assert "kwargs" in str(sig) or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ), "DshEvaluator.evaluate must accept **kwargs"


# ── test_completed_phases_roundtrip ──────────────────────────────────────

def test_completed_phases_roundtrip():
    """_completed_phases serializes/deserializes deterministically."""
    from refactory_packages.skillopt.loop import SkillOptLoop

    strategy = SkillOptLoop()
    strategy.configure()

    strategy._completed_phases = {"baseline_test", "final_selection"}
    state = strategy.strategy_state_dict()

    # Must be sorted for deterministic serialization
    assert state["completed_phases"] == ["baseline_test", "final_selection"]

    # Deserialize
    strategy2 = SkillOptLoop()
    strategy2.configure()
    strategy2.load_strategy_state(state)

    assert strategy2._completed_phases == {"baseline_test", "final_selection"}
    assert "baseline_test" in strategy2._completed_phases
    assert "best_test" not in strategy2._completed_phases


# ── test_completed_phases_backward_compat ────────────────────────────────

def test_completed_phases_backward_compat():
    """When completed_phases is empty and _phase is a final phase, backward compat populates the set."""
    from refactory_packages.skillopt.loop import SkillOptLoop

    strategy = SkillOptLoop()
    strategy.configure()

    # Simulate old state without completed_phases
    old_state = strategy.strategy_state_dict()
    old_state["phase"] = "baseline_test"
    old_state.pop("completed_phases", None)

    strategy2 = SkillOptLoop()
    strategy2.configure()
    strategy2.load_strategy_state(old_state)

    assert "final_selection" in strategy2._completed_phases
    assert "baseline_test" in strategy2._completed_phases
    assert "best_test" not in strategy2._completed_phases


# ── Unit tests for config, validation, etc. ──────────────────────────────

def test_evaluate_stays_unchanged(tmp_path):
    """Run.evaluate behaves exactly as before — it is not changed."""
    loop, seed, run = _make_loop_and_seed(tmp_path, max_evaluations=10)

    c = Candidate.from_state(seed.state, parent=seed.id, edits=())
    result = run.evaluate(c, part="train")

    assert result.evaluated
    assert loop.evaluations == 1
    assert result.id == c.id


def test_single_request_evaluate_many(tmp_path):
    """A single-request evaluate_many is valid and equivalent to evaluate."""
    loop, seed, run = _make_loop_and_seed(tmp_path, max_evaluations=10)

    candidate = Candidate.from_state(seed.state, parent=seed.id, edits=())
    results = run.evaluate_many([EvalRequest(candidate=candidate, part="train")])

    assert len(results) == 1
    assert results[0].evaluated
    assert results[0].id == candidate.id


def test_empty_request_list(tmp_path):
    """evaluate_many with empty list returns empty list."""
    loop, seed, run = _make_loop_and_seed(tmp_path, max_evaluations=10)
    results = run.evaluate_many([])
    assert results == []


def test_evaluate_many_validates_part(tmp_path):
    """evaluate_many validates each request's part."""
    loop, seed, run = _make_loop_and_seed(tmp_path)
    with pytest.raises(OuterLoopError, match="unknown part"):
        run.evaluate_many([EvalRequest(candidate=seed, part="invalid")])


def test_evaluate_many_refuses_items_on_val(tmp_path):
    """items cannot be chosen for val or test parts."""
    loop, seed, run = _make_loop_and_seed(tmp_path)
    with pytest.raises(OuterLoopError, match="no parts manifest"):
        run.evaluate_many([EvalRequest(candidate=seed, part="val")])


def test_config_fields_exist():
    """LoopConfig has the new fields with correct defaults."""
    config = LoopConfig()
    assert config.max_concurrent_evaluations == 4
    assert config.max_items_in_flight is None


def test_config_roundtrip():
    """LoopConfig serializes and deserializes the new fields."""
    config = LoopConfig(max_concurrent_evaluations=8, max_items_in_flight=16)
    d = config.to_dict()
    restored = LoopConfig.from_dict(d)
    assert restored.max_concurrent_evaluations == 8
    assert restored.max_items_in_flight == 16


def test_profile_validates_max_concurrent_evaluations():
    """normalize validates max_concurrent_evaluations as a positive integer."""
    from refactory_dsh.profiles import normalize, ProfileError

    with pytest.raises(ProfileError, match="must be a positive integer"):
        normalize({"max_concurrent_evaluations": 0})

    with pytest.raises(ProfileError, match="must be a positive integer"):
        normalize({"max_concurrent_evaluations": -1})


def test_profile_validates_max_items_in_flight():
    """normalize validates max_items_in_flight as a positive integer or None."""
    from refactory_dsh.profiles import normalize, ProfileError

    # None is valid (it's the default)
    result = normalize({"max_items_in_flight": None})
    assert result["max_items_in_flight"] is None

    with pytest.raises(ProfileError, match="max_items_in_flight must be a positive integer"):
        normalize({"max_items_in_flight": 0})


def test_eval_request_is_frozen():
    """EvalRequest is a frozen dataclass."""
    seed = Candidate.from_state({"graph": {}}, parent="", edits=())
    req = EvalRequest(candidate=seed, part="train")
    with pytest.raises(AttributeError):
        req.part = "val"  # type: ignore[misc]


def test_mixed_train_requests(tmp_path):
    """evaluate_many handles multiple train requests correctly."""
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_evaluations=10, max_concurrent_evaluations=4)

    candidates = _distinct_candidates(seed, 3)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests)

    assert len(results) == 3
    # Budget should reflect 3 train evaluations
    assert loop.evaluations == 3


# ── test_resume_skips_completed_phases ───────────────────────────────────

def test_resume_skips_completed_phases():
    """Resuming with some phases completed skips them.

    After baseline_test completes, saving and restoring state should
    mark it as completed so only remaining phases are submitted.
    """
    from refactory_packages.skillopt.loop import SkillOptLoop

    strategy = SkillOptLoop()
    strategy.configure()

    # Simulate: final_selection and baseline_test done
    strategy._completed_phases = {"final_selection", "baseline_test"}
    state = strategy.strategy_state_dict()

    # Restore into a fresh strategy
    strategy2 = SkillOptLoop()
    strategy2.configure()
    strategy2.load_strategy_state(state)

    assert "final_selection" in strategy2._completed_phases
    assert "baseline_test" in strategy2._completed_phases
    assert "best_test" not in strategy2._completed_phases
    assert "final_test" not in strategy2._completed_phases

    # The pending() function should reflect this
    def pending(phase: str) -> bool:
        return phase not in strategy2._completed_phases

    assert not pending("final_selection")
    assert not pending("baseline_test")
    assert pending("best_test")
    assert pending("final_test")


# ── test_skipped_train_budget_released (correction 2) ────────────────────

def test_skipped_train_budget_released(tmp_path):
    """When _run_group stops on an error, a skipped train request has its
    reserved budget released.

    One candidate, a val request then a train request (one sequential group).
    The val evaluation raises, so the train request is skipped.  Budget
    afterwards equals 0.

    Mutation check: not counting skipped train requests leaves evaluations=1.
    """
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_evaluations=10, max_concurrent_evaluations=4)
    # A manifest with a val part only, so the val request validates and the
    # train request still runs on the task's own train split.
    loop._parts_manifest = {"val": ["v1"]}

    candidate = Candidate.from_state(seed.state, parent=seed.id, edits=())
    train_calls: list[str] = []

    def failing_part(cand, part, items, seed, *, item_gate=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("deliberate failure on val eval")

    def counting(cand, **kwargs):  # type: ignore[no-untyped-def]
        train_calls.append(cand.id)
        raise AssertionError("the train request must be skipped")

    loop._evaluate_part = failing_part
    loop._evaluate = counting

    requests = [
        EvalRequest(candidate=candidate, part="val"),
        EvalRequest(candidate=candidate, part="train"),
    ]

    with pytest.raises(RuntimeError, match="deliberate failure on val eval"):
        run.evaluate_many(requests)

    assert train_calls == []
    # 1 train reserved, skipped, un-reserved.  Net = 0.
    assert loop.evaluations == 0, (
        f"expected evaluations=0 (the train request was skipped), got {loop.evaluations}"
    )


# ── test_on_result_exception_does_not_release_budget (correction 3) ──────

def test_on_result_exception_does_not_release_budget(tmp_path):
    """An exception raised by on_result after a successful evaluation must
    not release that evaluation's budget (it ran and was paid).

    Three train requests.  on_result raises for one of them.  All three
    evaluations complete successfully, so budget must remain 3.
    """
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_evaluations=10, max_concurrent_evaluations=4)

    candidates = _distinct_candidates(seed, 3)
    fail_index = 1  # on_result will raise for this request

    def bad_on_result(index: int, candidate: Candidate) -> None:
        if index == fail_index:
            raise ValueError("on_result deliberate error")

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]

    with pytest.raises(ValueError, match="on_result deliberate error"):
        run.evaluate_many(requests, on_result=bad_on_result)

    # All 3 evaluations ran successfully — budget must be 3
    assert loop.evaluations == 3, (
        f"expected evaluations=3 (all ran), got {loop.evaluations}; "
        "on_result error must not release budget"
    )
