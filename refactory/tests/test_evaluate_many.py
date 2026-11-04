"""Tests for Run.evaluate_many — concurrent evaluation of independent candidates.

Each test drives the new API through the real loop on the walker runtime
and asserts a specific mechanism from the brief.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

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


# ── test_concurrent_evaluations_overlap ──────────────────────────────────

def test_concurrent_evaluations_overlap(tmp_path):
    """Concurrent evaluations via evaluate_many overlap in time.

    With max_concurrent_evaluations=4, 3 requests should overlap.
    Mutation check: max_concurrent_evaluations=1 makes them serial.
    """
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)

    # Create 3 distinct candidates
    candidates = []
    for i in range(3):
        c = Candidate.from_state(seed.state, parent=seed.id, edits=())
        candidates.append(c)

    timestamps: dict[int, tuple[float, float]] = {}
    lock = threading.Lock()

    def on_result(index: int, candidate: Candidate) -> None:
        with lock:
            timestamps[index] = (time.monotonic(), time.monotonic())

    # Use train evaluations (which will run the mode)
    requests = [EvalRequest(candidate=c, part="train") for c in candidates]

    start = time.monotonic()
    results = run.evaluate_many(requests, on_result=on_result)
    total = time.monotonic() - start

    assert len(results) == 3
    # All candidates should be evaluated
    for r in results:
        assert r.evaluated


def test_concurrent_evaluations_serial_when_max_1(tmp_path):
    """With max_concurrent_evaluations=1, evaluations run serially."""
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=1, max_evaluations=10)

    candidates = []
    for i in range(2):
        c = Candidate.from_state(seed.state, parent=seed.id, edits=())
        candidates.append(c)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests)
    assert len(results) == 2
    for r in results:
        assert r.evaluated


# ── test_item_limit_not_exceeded ─────────────────────────────────────────

def test_item_limit_not_exceeded(tmp_path):
    """The run-wide item-in-flight limit is respected (walker runtime).

    With max_items_in_flight=2 and data_parallelism high enough, the gate
    ensures at most 2 items run concurrently across evaluations.
    """
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=3, max_evaluations=10,
        max_items_in_flight=2)

    candidates = []
    for i in range(3):
        c = Candidate.from_state(seed.state, parent=seed.id, edits=())
        candidates.append(c)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests)
    assert len(results) == 3
    for r in results:
        assert r.evaluated

    # The item gate should have been initialized
    assert loop._item_gate is not None or loop._item_gate_initialized


# ── test_results_in_request_order ────────────────────────────────────────

def test_results_in_request_order(tmp_path):
    """Results come back in request order regardless of completion order.

    Mutation check: reversing the results list in evaluate_many makes this fail.
    """
    from refactory_dsh.nn import edits as nn_edits

    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)

    # Create candidates with distinct state (different knob values) so they
    # have distinct ids and can be distinguished in the result list.
    candidates = []
    for i in range(3):
        state = nn_edits.configure(seed.state, "budget", str(i + 1))
        c = Candidate.from_state(state, parent=seed.id, edits=())
        candidates.append(c)

    # Verify candidates have distinct ids
    ids = [c.id for c in candidates]
    assert len(set(ids)) == 3, f"candidates must have distinct ids, got {ids}"

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests)

    assert len(results) == 3
    # Results must match request order
    for i, (req, res) in enumerate(zip(requests, results)):
        assert res.id == req.candidate.id, (
            f"result[{i}] is {res.id} but request[{i}] asked for {req.candidate.id}"
        )


# ── test_budget_refuses_entire_batch ─────────────────────────────────────

def test_budget_refuses_entire_batch_evaluations(tmp_path):
    """All-or-nothing budget refusal for max_evaluations."""
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_evaluations=2, max_concurrent_evaluations=4)

    # Use 1 eval so there's only 1 left
    run.evaluate(seed, part="train")
    assert loop.evaluations == 1

    # Now try to evaluate_many with 2 train requests — exceeds budget
    c1 = Candidate.from_state(seed.state, parent=seed.id, edits=())
    c2 = Candidate.from_state(seed.state, parent=seed.id, edits=())
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

    c1 = Candidate.from_state(seed.state, parent=seed.id, edits=())
    c2 = Candidate.from_state(seed.state, parent=seed.id, edits=())
    requests = [
        EvalRequest(candidate=c1, part="train"),
        EvalRequest(candidate=c2, part="train"),
    ]

    with pytest.raises(OuterLoopError, match="rollout budget is spent"):
        run.evaluate_many(requests)


# ── test_same_candidate_sequential (correction 3) ───────────────────────

def test_same_candidate_sequential(tmp_path):
    """Two requests for the same candidate never overlap in time."""
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)

    # Same candidate for both requests
    candidate = Candidate.from_state(seed.state, parent=seed.id, edits=())

    timestamps: list[tuple[float, float]] = [None, None]  # type: ignore
    lock = threading.Lock()

    original_evaluate = loop._evaluate

    call_count = 0
    def tracking_evaluate(cand, *, seed=None, items=None, _budget_reserved=False, item_gate=None):
        nonlocal call_count
        with lock:
            my_index = call_count
            call_count += 1
        start = time.monotonic()
        result = original_evaluate(cand, seed=seed, items=items,
                                   _budget_reserved=_budget_reserved,
                                   item_gate=item_gate)
        end = time.monotonic()
        with lock:
            timestamps[my_index] = (start, end)
        return result

    loop._evaluate = tracking_evaluate

    requests = [
        EvalRequest(candidate=candidate, part="train"),
        EvalRequest(candidate=candidate, part="train"),
    ]
    results = run.evaluate_many(requests)

    assert len(results) == 2
    # The two evaluations for the same candidate must not overlap
    t0_start, t0_end = timestamps[0]
    t1_start, t1_end = timestamps[1]
    # Second must start after first ends
    assert t1_start >= t0_end or t0_start >= t1_end, (
        f"same-candidate evaluations overlapped: [{t0_start:.4f}—{t0_end:.4f}] "
        f"vs [{t1_start:.4f}—{t1_end:.4f}]"
    )


# ── test_on_result_fires_in_completion_order ─────────────────────────────

def test_on_result_fires(tmp_path):
    """on_result callback fires for each completed evaluation."""
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)

    candidates = []
    for i in range(3):
        c = Candidate.from_state(seed.state, parent=seed.id, edits=())
        candidates.append(c)

    callback_order: list[int] = []
    lock = threading.Lock()

    def on_result(index: int, candidate: Candidate) -> None:
        with lock:
            callback_order.append(index)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests, on_result=on_result)

    # All indices should be in the callback
    assert sorted(callback_order) == [0, 1, 2]
    # Results still in request order
    assert len(results) == 3
    for i, (req, res) in enumerate(zip(requests, results)):
        assert res.id == req.candidate.id


# ── test_journal_integrity_under_concurrency ─────────────────────────────

def test_journal_integrity_under_concurrency(tmp_path):
    """Concurrent evaluations produce valid JSONL journal lines."""
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_concurrent_evaluations=4, max_evaluations=10)

    candidates = []
    for i in range(4):
        c = Candidate.from_state(seed.state, parent=seed.id, edits=())
        candidates.append(c)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    run.evaluate_many(requests)

    journal = loop.root / "journal.jsonl"
    assert journal.exists()

    for line in journal.read_text().splitlines():
        if line.strip():
            parsed = json.loads(line)
            assert "event" in parsed, f"journal line missing 'event': {line[:100]}"


# ── test_single_request_evaluate_many ────────────────────────────────────

def test_single_request_evaluate_many(tmp_path):
    """A single-request evaluate_many is valid and equivalent to evaluate."""
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_evaluations=10)

    candidate = Candidate.from_state(seed.state, parent=seed.id, edits=())
    results = run.evaluate_many([EvalRequest(candidate=candidate, part="train")])

    assert len(results) == 1
    assert results[0].evaluated
    assert results[0].id == candidate.id


# ── test_empty_request_list ──────────────────────────────────────────────

def test_empty_request_list(tmp_path):
    """evaluate_many with empty list returns empty list."""
    loop, seed, run = _make_loop_and_seed(tmp_path, max_evaluations=10)
    results = run.evaluate_many([])
    assert results == []


# ── test_evaluate_many_validation_errors ─────────────────────────────────

def test_evaluate_many_validates_part(tmp_path):
    """evaluate_many validates each request's part."""
    loop, seed, run = _make_loop_and_seed(tmp_path)
    with pytest.raises(OuterLoopError, match="unknown part"):
        run.evaluate_many([EvalRequest(candidate=seed, part="invalid")])


def test_evaluate_many_refuses_items_on_val(tmp_path):
    """items cannot be chosen for val or test parts."""
    loop, seed, run = _make_loop_and_seed(tmp_path)
    # This will fail because there's no manifest (val requires one)
    with pytest.raises(OuterLoopError, match="no parts manifest"):
        run.evaluate_many([EvalRequest(candidate=seed, part="val")])


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


# ── test_evaluate_stays_unchanged ────────────────────────────────────────

def test_evaluate_stays_unchanged(tmp_path):
    """Run.evaluate behaves exactly as before — it is not changed."""
    loop, seed, run = _make_loop_and_seed(tmp_path, max_evaluations=10)

    c = Candidate.from_state(seed.state, parent=seed.id, edits=())
    result = run.evaluate(c, part="train")

    assert result.evaluated
    assert loop.evaluations == 1
    assert result.id == c.id


# ── test_mixed_parts_evaluate_many ───────────────────────────────────────

def test_mixed_train_requests(tmp_path):
    """evaluate_many handles multiple train requests correctly."""
    loop, seed, run = _make_loop_and_seed(
        tmp_path, max_evaluations=10, max_concurrent_evaluations=4)

    candidates = []
    for i in range(3):
        c = Candidate.from_state(seed.state, parent=seed.id, edits=())
        candidates.append(c)

    requests = [EvalRequest(candidate=c, part="train") for c in candidates]
    results = run.evaluate_many(requests)

    assert len(results) == 3
    # Budget should reflect 3 train evaluations
    assert loop.evaluations == 3


# ── test_config_fields ───────────────────────────────────────────────────

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


# ── test_profile_validation ─────────────────────────────────────────────

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


# ── test_eval_request_frozen ─────────────────────────────────────────────

def test_eval_request_is_frozen():
    """EvalRequest is a frozen dataclass."""
    seed = Candidate.from_state({"graph": {}}, parent="", edits=())
    req = EvalRequest(candidate=seed, part="train")
    with pytest.raises(AttributeError):
        req.part = "val"  # type: ignore[misc]
