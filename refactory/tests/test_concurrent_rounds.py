"""The default loop's rounds run their independent evaluations at once.

A round's train evaluations (``Loop.forward``) and its val-gate rollouts
(``Loop._gate_population``) go through ``Run.evaluate_many``, up to
``max_concurrent_evaluations`` at once; the gate still decides in population order,
so a run scores and decides exactly as one evaluation at a time would. The
end-of-run val/test scoring may spend the budget's validation reserve, and a
missing val or test score names its cause. Each test fails when the behaviour it
names is reverted.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.budgets import RESERVE_RETAINED, RunBudget
from refactory_dsh.loop import Loop
from refactory_dsh.optim import Candidate
from refactory_dsh.outerloop import ValidationGate

from test_skillopt import TRAIN_IDS, VAL_IDS, _patch_mode
from test_val_gate import (
    LevelProbe, _cand, _gated, _journal_events, _level, _loop, _record_test_rollouts,
)


def _meet_at(monkeypatch: Any, method: str, levels: set[int], *, part: str | None = None,
             timeout: float = 5.0) -> list[int]:
    """Make ``Loop.<method>`` wait at a barrier for the candidates at ``levels``
    (on ``part``, for ``_evaluate_part``); returns the levels that met there, which
    is all of them only when their evaluations ran at the same time."""
    barrier = threading.Barrier(len(levels), timeout=timeout)
    met: list[int] = []
    original = getattr(Loop, method)

    def wrapped(self, candidate, *args, **kwargs):  # type: ignore[no-untyped-def]
        if _level(candidate) in levels and (part is None or args[0] == part):
            try:
                barrier.wait()
                met.append(_level(candidate))
            except threading.BrokenBarrierError:
                pass
        return original(self, candidate, *args, **kwargs)

    monkeypatch.setattr(Loop, method, wrapped)
    return met


@pytest.fixture
def one_seed(monkeypatch):  # type: ignore[no-untyped-def]
    """The scripted plans start from the seed alone."""
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)


def test_a_rounds_train_evaluations_run_at_once(tmp_path, monkeypatch, one_seed):
    """The two children of a round are scored on train at the same time."""
    met = _meet_at(monkeypatch, "_evaluate", {6, 7})
    loop, _, _ = _gated(tmp_path, monkeypatch, [[(5, 7), (5, 6)], []], epochs=2, batch_size=2)
    assert sorted(met) == [6, 7], "the round's train evaluations did not overlap"
    assert all(c.evaluated for c in loop.population)


def test_one_concurrent_evaluation_runs_a_round_one_at_a_time(tmp_path, monkeypatch, one_seed):
    """max_concurrent_evaluations 1 keeps the old order: nothing overlaps."""
    met = _meet_at(monkeypatch, "_evaluate", {6, 7}, timeout=0.5)
    vals = _meet_at(monkeypatch, "_evaluate_part", {6, 7}, part="val", timeout=0.5)
    _gated(tmp_path, monkeypatch, [[(5, 7), (5, 6)], []], epochs=2, batch_size=2,
           max_concurrent_evaluations=1)
    assert met == [] and vals == []


def test_a_rounds_val_gate_rollouts_run_at_once(tmp_path, monkeypatch, one_seed):
    """The gate rolls the round's candidates out on val at the same time, and still
    decides in population order: 7 is accepted, then 6 does not beat it."""
    met = _meet_at(monkeypatch, "_evaluate_part", {6, 7}, part="val")
    loop, _, _ = _gated(tmp_path, monkeypatch, [[(5, 7), (5, 6)], []], epochs=2, batch_size=2)
    assert sorted(met) == [6, 7], "the round's val rollouts did not overlap"
    by_id = {c.id: c for c in loop.population}
    assert [_level(by_id[c]) for c in loop._gate_accepted] == [5, 7]
    assert [_level(by_id[c]) for c in loop._gate_rejected] == [6]
    assert len(_journal_events(loop, "part_result")) == 3 + 2, "one val rollout each, then 2 test"


def _normalised(loop: Loop) -> dict[str, Any]:
    """What a run scored and decided, without timings or paths."""
    def events(name: str, *keys: str) -> list[tuple[Any, ...]]:
        return sorted((tuple(str(r.get(k)) for k in keys) for r in _journal_events(loop, name)))

    records = sorted((r["id"], tuple(sorted((str(x.get("id")), x.get("score"), x.get("repeat"))
                                            for x in r["records"])))
                     for r in _journal_events(loop, "train_records"))
    parts = sorted((r["id"], r["part"], r["seed"], r["result"]["occasion"], r["result"]["score"],
                    r["result"]["exit_ok"]) for r in _journal_events(loop, "part_result"))
    final = [(r.get("step_type"), r.get("action"), r.get("score"), r.get("candidate"))
             for r in _journal_events(loop, "final_evaluation")]
    return {
        "population": [(c.id, c.parent, c.score, c.raw_score, c.exit_ok) for c in loop.population],
        "accepted": list(loop._gate_accepted), "rejected": list(loop._gate_rejected),
        "val_cache": dict(loop._gate().cache),
        "scores": events("score", "id", "score", "raw_score", "seed", "exit_ok", "error"),
        "records": records, "records_epoch": events("train_records_epoch", "id", "epoch"),
        "rescores": events("rescore", "id", "epoch", "seed", "score"),
        "parts": parts, "final": final,
        "evaluations": loop.evaluations, "rollouts": loop.rollouts,
        "forward": sorted((p.get("candidate"), p.get("index"), p.get("of"))
                          for p in _progress(loop, "forward")),
    }


def _progress(loop: Loop, step: str) -> list[dict[str, Any]]:
    import json

    from refactory_dsh import observe
    path = loop.root / observe.PROGRESS
    lines = path.read_text().splitlines() if path.exists() else []
    return [r for r in map(json.loads, lines) if r.get("step") == step]


def _slow_and_val_by_level(monkeypatch: Any) -> None:
    """A higher level finishes its train evaluation first; val scores level / 10."""
    original = Loop._evaluate

    def slow(self, candidate, **kwargs):  # type: ignore[no-untyped-def]
        time.sleep(0.05 * (9 - _level(candidate)))
        return original(self, candidate, **kwargs)

    monkeypatch.setattr(Loop, "_evaluate", slow)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))


def _same_as_one_at_a_time(runs: dict[int, tuple[Loop, dict[str, Any]]]) -> dict[str, Any]:
    serial = _normalised(runs[1][0])
    assert serial["forward"], "setup: the runs write progress"
    order = {w: [r["id"] for r in _journal_events(loop, "score")] for w, (loop, _r) in runs.items()}
    for width, (loop, report) in runs.items():
        assert _normalised(loop) == serial, f"{width} at once"
        assert report["best"]["id"] == runs[1][1]["best"]["id"]
        assert report["strategy_report"] == runs[1][1]["strategy_report"]
        if width > 1:
            assert order[width] != order[1], "setup: the concurrent run finished in another order"
    return serial


def test_concurrent_rounds_score_and_decide_as_one_at_a_time(tmp_path, monkeypatch, one_seed):
    """A round of four children, all rejected, so the current candidate is re-scored
    (fresh evidence), then the final test evaluation: with the evaluations finishing
    in reverse order, two at once (two waves) or four (one), the run scores the same
    candidates on the same items and seeds, journals the same records and progress,
    and the gate takes the same decisions, as the one-at-a-time run."""
    _slow_and_val_by_level(monkeypatch)
    plan = [[(5, 6), (5, 7)], [(7, 1), (7, 2), (7, 3), (7, 4)], []]
    runs = {}
    for width in (1, 2, 4):
        loop, _, report = _gated(tmp_path / str(width), monkeypatch, plan, epochs=3,
                                 batch_size=4, max_concurrent_evaluations=width)
        runs[width] = (loop, report)
    serial = _same_as_one_at_a_time(runs)
    assert serial["rescores"], "setup: the current candidate is re-scored"
    assert len(serial["rejected"]) == 4 and max(of for _c, _i, of in serial["forward"]) == 4


def test_the_seeds_first_round_scores_and_decides_as_one_at_a_time(tmp_path, monkeypatch):
    """hill_climb from the seed and its neighbours, whose val rollouts include the
    gate's start: the same run whichever the width."""
    _slow_and_val_by_level(monkeypatch)
    runs = {}
    for width in (1, 4):
        with _patch_mode(LevelProbe):
            loop = _loop(tmp_path / str(width), optimizer="hill_climb", val_gate=True,
                         epochs=3, batch_size=3, max_concurrent_evaluations=width)
            report = loop.run()
        runs[width] = (loop, report)
    serial = _same_as_one_at_a_time(runs)
    assert len(serial["accepted"]) > 1


class _TwoAtOnce(LevelProbe):
    """LevelProbe whose data node runs two items at once."""

    def build(self):  # type: ignore[no-untyped-def]
        import dataclasses
        graph = super().build()
        graph.nodes["data"] = dataclasses.replace(graph.nodes["data"], parallelism=2)
        return graph


def test_the_speedup_needs_room_in_the_item_limit(tmp_path, monkeypatch, one_seed):
    """Wall clock of a round of four children (8 train items each, 50 ms an item, a
    data node two items wide). One at a time, and four at once under the default
    item limit (the seed's parallelism, 2: a run-wide cap, #580's), take about as
    long; with max_items_in_flight 8 (2 x 4) the four run side by side, ~4x faster."""
    from test_skillopt import SkillTask
    from test_val_gate import ScriptedLevels

    verify = SkillTask.verify

    def slow(self, instance, workspace, *, part=None, seed=None):  # type: ignore[no-untyped-def]
        if part == "train":
            time.sleep(0.05)
        return verify(self, instance, workspace, part=part, seed=seed)

    monkeypatch.setattr(SkillTask, "verify", slow)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    forward = Loop.forward
    took: dict[str, float] = {}

    def timed(self, **kwargs):  # type: ignore[no-untyped-def]
        start = time.monotonic()
        count = forward(self, **kwargs)
        if count == 4:
            took[label] = time.monotonic() - start
        return count

    monkeypatch.setattr(Loop, "forward", timed)
    for label, extra in (("serial", {"max_concurrent_evaluations": 1}),
                         ("default limit", {"max_concurrent_evaluations": 4}),
                         ("raised limit", {"max_concurrent_evaluations": 4, "max_items_in_flight": 8})):
        with _patch_mode(_TwoAtOnce):
            loop = _loop(tmp_path / label.replace(" ", "-"), val_gate=True, epochs=2,
                         batch_size=4, **extra)
            loop.optimizer = ScriptedLevels([[(5, 1), (5, 2), (5, 3), (5, 4)], []])
            loop.run()
    print("round wall clock (s):", {k: round(v, 2) for k, v in took.items()})
    assert took["default limit"] > 0.7 * took["serial"], took
    assert took["raised limit"] < 0.5 * took["serial"], took


def test_a_rollout_cap_starts_only_what_one_at_a_time_would(tmp_path, monkeypatch, one_seed):
    """With max_rollouts, a round starts only the evaluations the one-at-a-time
    order would start: here two of three children (the cap falls inside the second),
    in both widths, with the same rollout count."""
    cap = len(TRAIN_IDS) + len(VAL_IDS) + len(TRAIN_IDS) + 1
    plan = [[(5, 7), (5, 6), (5, 8)], []]
    results = {}
    for width in (1, 4):
        loop, _, _ = _gated(tmp_path / str(width), monkeypatch, plan, epochs=2, batch_size=3,
                            max_rollouts=cap, max_concurrent_evaluations=width)
        results[width] = (sorted(_level(c) for c in loop.population if c.evaluated), loop.rollouts)
    assert results[4] == results[1]
    assert results[1][0] == [5, 6, 7]


def test_the_wave_size_counts_repeats_and_the_cap():
    """A wave holds only evaluations that would start one after another (each at its
    most rollouts, items times repeats) and that evaluate_many accepts together."""
    class Config:
        max_concurrent_evaluations = 4
        max_rollouts: int | None = None
        repeats = 1

    class Fake:
        config = Config()
        rollouts = 0
        _wave_size = Loop._wave_size

    fake = Fake()
    assert fake._wave_size(3, 8) == 3 and fake._wave_size(9, 8) == 4
    fake.config.max_rollouts = 20
    assert fake._wave_size(4, 8) == 2, "16 fits, 24 does not"
    fake.config.repeats = 2
    assert fake._wave_size(4, 8) == 2, "the second starts below 20 after 16"
    fake.config.repeats = 3
    assert fake._wave_size(4, 8) == 1, "after 24 the second would not start"
    fake.rollouts = 19
    assert fake._wave_size(4, 8) == 1, "one alone runs, as before"


class _ManyRun:
    """A run whose evaluate_many rolls every request out on val."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.rolled: list[str] = []
        self.batches: list[list[str]] = []

    def evaluate(self, candidate: Candidate, *, part: str = "train", seed: int | None = None) -> Candidate:
        self.rolled.append(candidate.id)
        return candidate

    def evaluate_many(self, requests, *, on_result=None):  # type: ignore[no-untyped-def]
        self.batches.append([r.candidate.id for r in requests])
        for index, request in enumerate(requests):
            self.rolled.append(request.candidate.id)
            on_result(index, request.candidate)
        return [r.candidate for r in requests]

    def val_scores(self, candidate: Candidate) -> list[dict[str, Any]]:
        return [{"score": self.scores[candidate.id]}] if candidate.id in self.rolled else []


def test_prefetched_val_scores_are_reused_by_start_and_judge():
    """prefetch rolls the uncached candidates out together, once; start and judge
    then read the cache and roll nothing again."""
    seed, child, other = _cand("s"), _cand("c"), _cand("o")
    run = _ManyRun({seed.id: 0.5, child.id: 0.7, other.id: 0.6})
    gate = ValidationGate(seed=3)
    gate.prefetch(run, [seed, child, other, child])
    assert run.batches == [[seed.id, child.id, other.id]]
    assert gate.start(run, seed) == 0.5
    assert gate.judge(run, child).accepted and not gate.judge(run, other).accepted
    assert run.rolled == [seed.id, child.id, other.id], "nothing was rolled out twice"
    gate.prefetch(run, [seed, _cand("alone")])
    assert len(run.batches) == 1, "one candidate to roll is left to the in-order calls"


def test_a_missing_test_score_names_the_spent_time_budget(tmp_path):
    """The audit says which score is missing and why: a spent elapsed-time budget,
    with the reserve that keeps time for the end-of-run scoring; another failure
    gives its own error."""
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, val_gate=True, max_elapsed_s=43200)
        loop.init()
    winner = loop.population[0]
    loop._part_results[winner.id] = {
        "val": [{"score": 0.6, "error": None}],
        "test": [{"score": None, "error": "finish: BudgetExhausted: elapsed-time budget exhausted"}]}
    note = loop.audit_parts(winner)["note"]
    assert note.startswith("test score is missing: the elapsed-time budget (max_elapsed_s 43200s)")
    assert "validation_reserve_s" in note
    loop._part_results[winner.id]["test"] = [{"score": None, "error": "the item grader crashed"}]
    assert loop.audit_parts(winner)["note"] == (
        "test score is missing: the winner's test scoring failed: the item grader crashed")


def _record_phases(monkeypatch: Any) -> list[tuple[str, str | None]]:
    """(part, phase) of every budget checkpoint a measurement makes."""
    seen: list[tuple[str, str | None]] = []
    local = threading.local()
    measure, checkpoint = Loop._measure, RunBudget.checkpoint

    def measuring(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        local.part = part
        try:
            return measure(self, candidate, workdir, part, **kwargs)
        finally:
            local.part = None

    def checking(self, phase=None):  # type: ignore[no-untyped-def]
        if getattr(local, "part", None) and phase is not None:
            seen.append((local.part, phase))
        return checkpoint(self, phase)

    monkeypatch.setattr(Loop, "_measure", measuring)
    monkeypatch.setattr(RunBudget, "checkpoint", checking)
    return seen


def test_the_end_of_run_scoring_may_spend_the_validation_reserve(tmp_path, monkeypatch, one_seed):
    """The search's measurements are work; the final evaluation's val and test
    measurements are final validation, the phase the reserve is kept for."""
    phases = _record_phases(monkeypatch)
    loop, _, report = _gated(tmp_path, monkeypatch, [[(5, 7)], []], epochs=2,
                             max_elapsed_s=3600, validation_reserve_s=60)
    assert report["strategy_report"]["test_eval"] == "done"
    assert {phase for part, phase in phases if part == "test"} == {"final_validation"}
    assert {phase for part, phase in phases if part == "train"} == {"work"}
    assert not loop._final_scoring


def test_a_search_stopped_at_the_reserve_still_runs_the_final_evaluation(tmp_path, monkeypatch, one_seed):
    """A time budget that stops the search at the reserve leaves the last proposals
    unscored and runs the final evaluation on the reserve; without a reserve the
    final evaluation is skipped, as before (test_val_gate_budget_endings_skip_...)."""
    tests = _record_test_rollouts(monkeypatch)
    spent = "elapsed-time budget exhausted" + RESERVE_RETAINED
    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        if part == "train" and _level(candidate) == 7:
            self.bound = spent
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)
    loop, _, report = _gated(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)]], epochs=2)
    assert report["bound"] == spent
    assert report["strategy_report"]["test_eval"] == "done"
    assert [_level(c) for c in loop.population if not c.evaluated] == [8]
    assert sorted(level for level, _seed in tests) == [5, 7]


@pytest.mark.integration
def test_the_spine_runs_a_rounds_train_evaluations_at_once_with_the_same_scores(tmp_path, monkeypatch):
    """On the spine (``evaluator=dsh``): the seed and its neighbours, scored on train
    in one round, run as overlapping spine processes under the shared item-slot
    directory, and score what the one-at-a-time round scores."""
    from refactory_dsh import dsh_runtime
    from refactory_dsh.loop import LoopConfig

    spans: dict[int, list[tuple[float, float, Any]]] = {}
    original = dsh_runtime.run_on_dsh

    def timed(state, workdir, **kwargs):  # type: ignore[no-untyped-def]
        start = time.monotonic()
        try:
            return original(state, workdir, **kwargs)
        finally:
            spans.setdefault(width, []).append((start, time.monotonic(), kwargs.get("item_slots_dir")))

    monkeypatch.setattr(dsh_runtime, "run_on_dsh", timed)
    scores = {}
    for width in (1, 4):
        loop = Loop("circle_packing/adaptive", LoopConfig(
            epochs=1, batch_size=3, optimizer="random", seed=3, evaluator="dsh",
            runs_root=tmp_path / str(width), max_evaluations=64,
            max_concurrent_evaluations=width), project=Path(__file__).resolve().parents[2])
        loop.init()
        loop.open_epoch()
        assert loop.forward() == 3
        scores[width] = {c.id: (c.score, c.exit_ok) for c in loop.population}
        assert all(ok for _score, ok in scores[width].values()), scores[width]
    assert scores[4] == scores[1]
    wide = sorted(spans[4])
    assert all(slots for _s, _e, slots in wide), "every concurrent spine run shares the slot directory"
    assert any(later[0] < earlier[1] for earlier, later in zip(wide, wide[1:])), "the spine runs overlapped"
    assert not any(slots for _s, _e, slots in spans[1])
