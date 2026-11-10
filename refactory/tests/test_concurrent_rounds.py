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
    """Every wave of train evaluations finishes in reverse request order (each waits
    for the ones after it, by events, not sleeps); val scores level / 10."""
    from refactory_dsh.outerloop import Run

    waves: dict[str, list[str]] = {}
    done: dict[str, threading.Event] = {}
    lock = threading.Lock()
    original_many, original = Run.evaluate_many, Loop._evaluate

    def many(self, requests, *, on_result=None):  # type: ignore[no-untyped-def]
        ids = [r.candidate.id for r in requests if r.part == "train"]
        with lock:
            for cid in ids:
                waves[cid], done[cid] = ids, threading.Event()
        return original_many(self, requests, on_result=on_result)

    def evaluate(self, candidate, **kwargs):  # type: ignore[no-untyped-def]
        with lock:
            wave = waves.pop(candidate.id, None)
        for later in (wave or [])[wave.index(candidate.id) + 1:] if wave else []:
            assert done[later].wait(timeout=60), "a later evaluation of the wave never finished"
        try:
            return original(self, candidate, **kwargs)
        finally:
            if wave:
                done[candidate.id].set()

    monkeypatch.setattr(Run, "evaluate_many", many)
    monkeypatch.setattr(Loop, "_evaluate", evaluate)
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
    """A round of four children (8 train items each, a data node two items wide).
    One at a time, and four at once under the default item limit (the seed's
    parallelism, 2: a run-wide cap, #580's), never have more than 2 items in flight,
    so four at once is no faster; with max_items_in_flight 8 (2 x 4) the four
    children's items run side by side: 8 of them meet at once. Counted, not timed,
    so the result does not depend on the machine's speed."""
    from test_skillopt import SkillTask
    from test_val_gate import ScriptedLevels

    verify = SkillTask.verify
    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0, "arrived": 0}
    meet = threading.Barrier(8, timeout=20)
    label = ""

    def counted(self, instance, workspace, *, part=None, seed=None):  # type: ignore[no-untyped-def]
        with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
            state["arrived"] += 1
            first_eight = state["arrived"] <= 8
        try:
            if part == "train" and label == "raised limit" and first_eight and round_two["on"]:
                meet.wait()  # eight items of four children in flight together
            else:
                time.sleep(0.01)
            return verify(self, instance, workspace, part=part, seed=seed)
        finally:
            with lock:
                state["in_flight"] -= 1

    round_two = {"on": False}
    forward = Loop.forward

    def marked(self, **kwargs):  # type: ignore[no-untyped-def]
        # Count only the round of four children.
        children = len([c for c in self.population if not c.evaluated]) == 4
        with lock:
            state.update(in_flight=0, peak=0, arrived=0)
        round_two["on"] = children
        count = forward(self, **kwargs)
        if children:
            peaks[label] = state["peak"]
        round_two["on"] = False
        return count

    monkeypatch.setattr(SkillTask, "verify", counted)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    monkeypatch.setattr(Loop, "forward", marked)
    peaks: dict[str, int] = {}
    for label, extra in (("serial", {"max_concurrent_evaluations": 1}),
                         ("default limit", {"max_concurrent_evaluations": 4}),
                         ("raised limit", {"max_concurrent_evaluations": 4, "max_items_in_flight": 8})):
        with _patch_mode(_TwoAtOnce):
            loop = _loop(tmp_path / label.replace(" ", "-"), val_gate=True, epochs=2,
                         batch_size=4, **extra)
            loop.optimizer = ScriptedLevels([[(5, 1), (5, 2), (5, 3), (5, 4)], []])
            loop.run()
    assert peaks == {"serial": 2, "default limit": 2, "raised limit": 8}, peaks


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


def test_an_evaluation_budget_spent_mid_round_scores_what_one_at_a_time_does(tmp_path, monkeypatch, one_seed):
    """max_evaluations runs out inside a round of four: four at once scores the same
    two children as one at a time, and the run ends on the evaluation budget."""
    plan = [[(5, 1), (5, 2), (5, 3), (5, 4)], []]
    results = {}
    for width in (1, 4):
        loop, _, report = _gated(tmp_path / str(width), monkeypatch, plan, epochs=2, batch_size=4,
                                 max_evaluations=3, max_concurrent_evaluations=width)
        results[width] = (sorted(_level(c) for c in loop.population if c.evaluated),
                          loop.evaluations, report["bound"])
    assert results[4] == results[1]
    assert results[1][0] == [1, 2, 5]


def test_a_rollout_cap_inside_the_val_prefetch_decides_as_one_at_a_time(tmp_path, monkeypatch, one_seed):
    """repeats 2 (8 train and 4 val items: 16 and 8 rollouts each): the cap falls
    where evaluate_many would take all three val rollouts at once but one at a time
    starts only two. Four at once rolls out and judges the same two."""
    n_train, n_val, repeats = len(TRAIN_IDS), len(VAL_IDS), 2
    before_val = (n_train + n_val) * repeats + 3 * n_train * repeats  # seed, then 3 children on train
    cap = before_val + 3 * n_val + 2  # three fit evaluate_many's count; the third would not start
    plan = [[(5, 1), (5, 2), (5, 3)], []]
    results = {}
    for width in (1, 4):
        loop, _, _ = _gated(tmp_path / str(width), monkeypatch, plan, epochs=2, batch_size=3,
                            repeats=repeats, max_rollouts=cap, max_concurrent_evaluations=width)
        vals = sorted(r["id"] for r in _journal_events(loop, "part_result") if r["part"] == "val")
        results[width] = (vals, list(loop._gate_accepted), list(loop._gate_rejected), loop.rollouts)
    assert results[4] == results[1]
    assert len(results[1][1]) + len(results[1][2]) == 3, "setup: the seed and two children judged"


@pytest.mark.parametrize("cut", ["val", "train"])
def test_a_budget_stopped_evaluation_leaves_the_candidate_for_a_resume(tmp_path, monkeypatch, one_seed, cut):
    """The review's probe: max_elapsed_s 2 and four children whose train runs take
    3 s, four at once. ``val``: the train runs finish past the limit (the last wave
    may overshoot) and their val rollouts are refused by the spent budget.
    ``train``: the budget cuts the train runs themselves. Either way no child is
    recorded as scored or gate-rejected from a refused evaluation, and a resume with
    more time judges all four on their real val scores."""
    from refactory_dsh.evaluators import WalkerEvaluator
    from test_val_gate import ScriptedLevels

    clock = _Clock(monkeypatch)
    walked = threading.Barrier(4, timeout=60)
    original = WalkerEvaluator.evaluate
    slow = {"on": True}

    def evaluate(self, mode, state, workdir, **kwargs):  # type: ignore[no-untyped-def]
        child = int(state["knobs"]["level"]["value"]) != 5 and "occasions" not in str(workdir)
        if slow["on"] and child and cut == "train":
            clock.advance(3)  # before the walk: its nodes meet the spent budget
        result = original(self, mode, state, workdir, **kwargs)
        if slow["on"] and child and cut == "val":
            walked.wait()  # every child's walk has finished, inside the limit
            clock.advance(3)  # ...and each run ends past it
        return result

    monkeypatch.setattr(WalkerEvaluator, "evaluate", evaluate)
    settings = dict(val_gate=True, batch_size=4, max_concurrent_evaluations=4)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=2, max_elapsed_s=2, **settings)
        loop.optimizer = ScriptedLevels([[(5, 1), (5, 2), (5, 3), (5, 4)], []])
        loop.run()
    children = [c for c in loop.population if _level(c) != 5]
    ids = {c.id for c in children}
    assert all(c.evaluated for c in children) == (cut == "val")
    assert loop._gate_rejected == [] and loop._gate_accepted == [loop.population[0].id]
    stopped = {r["id"] for r in _journal_events(loop, "budget_stopped")}
    stopped |= {r["id"] for r in _journal_events(loop, "part_result")
                if r["result"].get("budget_stopped")}
    assert stopped == ids
    if cut == "train":
        assert not ids & {r["id"] for r in _journal_events(loop, "score")}
        assert loop.evaluations == 1
    saved = [r["state"]["val_gate"] for r in _journal_events(loop, "strategy_state")
             if "val_gate" in r.get("state", {})][-1]
    assert saved["rejected"] == [], "no budget-refused candidate is journaled as rejected"

    slow["on"] = False
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=3, max_elapsed_s=600, resume=True, **settings)
        resumed.optimizer = ScriptedLevels([[]])
        resumed.run()
    gate = resumed._gate()
    assert set(resumed._gate_accepted + resumed._gate_rejected) >= ids
    assert {_level(c): gate.cache[c.id] for c in children} == pytest.approx(
        {level: 1 - level / 10 for level in (1, 2, 3, 4)}), "judged on real val scores, not 0"


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


class _Clock:
    """The budget's clock (``refactory_dsh.budgets.time``): it moves only when a test
    advances it, so a time budget runs out where the test says, whatever the
    machine's speed."""

    def __init__(self, monkeypatch: Any) -> None:
        import types

        from refactory_dsh import budgets

        self.now = 1_000_000.0
        self._lock = threading.Lock()
        monkeypatch.setattr(budgets, "time", types.SimpleNamespace(time=lambda: self.now))

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


def _slow_train(monkeypatch: Any, seconds: float) -> dict[str, bool]:
    """Every train run (not a val/test occasion) takes ``seconds`` more on the
    budget's clock (``_Clock``), after its walk, so it finishes past a short time
    limit; returns a switch."""
    from refactory_dsh.evaluators import WalkerEvaluator

    clock = _Clock(monkeypatch)
    original = WalkerEvaluator.evaluate
    switch = {"on": True}

    def evaluate(self, mode, state, workdir, **kwargs):  # type: ignore[no-untyped-def]
        result = original(self, mode, state, workdir, **kwargs)
        if switch["on"] and "occasions" not in str(workdir):
            clock.advance(seconds)
        return result

    monkeypatch.setattr(WalkerEvaluator, "evaluate", evaluate)
    return switch


def test_budget_left_is_zero_once_the_time_budget_is_spent(tmp_path, monkeypatch):
    """A strategy that loops on budget_left() and skips refused evaluations stops
    once the time budget is spent: a refused evaluation counts nothing, so the time
    budget has to show in budget_left()."""
    from refactory_dsh.outerloop import OuterLoop, OuterLoopError

    class Spinner(OuterLoop):
        name = "spinner"
        required_ops = ("set_knob",)

        def configure(self, **options: Any) -> None:
            self.attempts = 0

        def strategy(self, run):  # type: ignore[no-untyped-def]
            while run.budget_left() > 0 and self.attempts < 40:
                self.attempts += 1
                try:
                    run.evaluate(run.seed())
                except OuterLoopError:
                    continue

    _slow_train(monkeypatch, 1.2)
    spinner = Spinner()
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, strategy=spinner, max_elapsed_s=1, max_evaluations=5)
        report = loop.run()
    assert spinner.attempts == 1, spinner.attempts
    assert "elapsed-time budget exhausted" in str(report["bound"])


@pytest.mark.parametrize("reserve", [0, 4])
def test_skillopt_stopped_by_the_time_budget_says_so_or_scores_on_the_reserve(tmp_path, monkeypatch, reserve):
    """SkillOpt whose training a time budget stops: without a reserve the
    post-training evaluation is skipped with the budget as its reason; stopped at
    the reserve's line, it still runs on the reserve."""
    from test_skillopt import _run_loop, _write_parts

    _slow_train(monkeypatch, 0.6)
    parts = _write_parts(tmp_path / "parts", test=True)
    overrides = {"parts_manifest": str(parts), "max_elapsed_s": 6 if reserve else 1.5,
                 "validation_reserve_s": reserve}
    _strategy, report = _run_loop(tmp_path, config_overrides=overrides)
    final = report["strategy_report"]
    assert "elapsed-time budget exhausted" in str(report["bound"])
    if reserve:
        assert final["test_eval"] == "done", final
    else:
        assert final["test_eval"] == "skipped", final
        assert "elapsed-time budget exhausted" in final["test_eval_reason"]


def test_the_prefetched_val_scores_survive_a_kill_before_the_decisions(tmp_path, monkeypatch, one_seed):
    """A run killed after the round's val rollouts and before the gate's decisions
    has their scores journaled, so the resume rolls none of them out again."""
    from test_val_gate import ScriptedLevels

    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    judge = ValidationGate.judge

    def killed(self, run, candidate, **kwargs):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt("killed before the decisions")

    monkeypatch.setattr(ValidationGate, "judge", killed)
    settings = dict(val_gate=True, batch_size=2)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=2, **settings)
        loop.optimizer = ScriptedLevels([[(5, 6), (5, 7)], []])
        with pytest.raises(KeyboardInterrupt):
            loop.run()
    rolled = [r["id"] for r in _journal_events(loop, "part_result") if r["part"] == "val"]
    assert len(rolled) == 3, "setup: the seed's and both children's val rollouts"

    monkeypatch.setattr(ValidationGate, "judge", judge)
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=2, resume=True, **settings)
        resumed.optimizer = ScriptedLevels([[]])
        resumed.run()
    again = [r["id"] for r in _journal_events(resumed, "part_result") if r["part"] == "val"]
    assert again[:3] == rolled and sorted(again[3:]) == [], "no val rollout again"
    assert len(resumed._gate_accepted) == 3


def test_a_partial_prefetch_wave_leaves_the_rest_to_the_in_order_gate(tmp_path, monkeypatch, one_seed):
    """Two at once, four children whose val rollouts the budget refuses: the first
    wave comes back partial, so no further wave starts; the in-order gate tries the
    first child once more, is refused, and stops. Nothing is judged."""
    from refactory_dsh.outerloop import BudgetStopped

    original = Loop._evaluate_part
    refused: list[int] = []

    def evaluate_part(self, candidate, part, *args, **kwargs):  # type: ignore[no-untyped-def]
        if part == "val" and _level(candidate) != 5:
            refused.append(_level(candidate))
            raise BudgetStopped("refused", reason="elapsed-time budget exhausted")
        return original(self, candidate, part, *args, **kwargs)

    monkeypatch.setattr(Loop, "_evaluate_part", evaluate_part)
    loop, _, _ = _gated(tmp_path, monkeypatch, [[(5, 1), (5, 2), (5, 3), (5, 4)], []],
                        epochs=2, batch_size=4, max_concurrent_evaluations=2)
    assert sorted(refused) == [1, 1, 2], refused
    assert loop._gate_rejected == []


@pytest.mark.parametrize("reserve", [False, True])
def test_skillopt_whose_proposer_meets_the_spent_budget_ends_with_a_reason(tmp_path, reserve):
    """The budget line crossed during the proposer's model call (a plain
    BudgetExhausted, not a stopped evaluation): training ends on it, and the
    post-training evaluation runs on the reserve when the line was the reserve's,
    else is recorded as skipped with that reason."""
    from refactory_dsh.budgets import BudgetExhausted
    from test_skillopt import _Ending, _run_loop, _write_parts

    reason = "elapsed-time budget exhausted" + (RESERVE_RETAINED if reserve else "")
    overrides = {"parts_manifest": str(_write_parts(tmp_path / "parts", test=True)),
                 "max_elapsed_s": 3600, "validation_reserve_s": 60}
    _strategy, report = _run_loop(tmp_path, client=_Ending(BudgetExhausted(reason)),
                                  config_overrides=overrides, skillopt_options={"max_epochs": 3})
    final = report["strategy_report"]
    assert report["bound"] == reason
    if reserve:
        assert final["test_eval"] == "done", final
    else:
        assert final["test_eval"] == "skipped" and reason in final["test_eval_reason"], final


def test_a_crash_beside_a_budget_stop_is_raised_as_the_crash(tmp_path, monkeypatch, one_seed):
    """evaluate_many with one evaluation the budget stopped and one that crashed
    raises the crash, whichever came first."""
    from refactory_dsh.outerloop import BudgetStopped, EvalRequest, Run

    def evaluate(self, candidate, **kwargs):  # type: ignore[no-untyped-def]
        if _level(candidate) == 1:
            raise BudgetStopped("stopped", reason="elapsed-time budget exhausted")
        time.sleep(0.2)
        raise ZeroDivisionError("a real failure")

    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, max_concurrent_evaluations=2)
        loop.init()
        seed = loop.population[0]
        children = [Candidate.from_state({**seed.state, "knobs": {**seed.state["knobs"],
                                          "level": {**seed.state["knobs"]["level"], "value": level}}})
                    for level in (1, 2)]
        monkeypatch.setattr(Loop, "_evaluate", evaluate)
        with pytest.raises(ZeroDivisionError):
            Run(_loop=loop).evaluate_many([EvalRequest(candidate=c) for c in children])


def test_a_budget_stopped_strategy_still_runs_its_own_ending(tmp_path, monkeypatch):
    """A built-in-style strategy (a Loop subclass) whose evaluation the time budget
    stops: the search ends on the budget and its finish_strategy runs, once."""
    from refactory_dsh.loop import LoopConfig
    from test_skillopt import MODE_ID, _write_parts

    class Ending(Loop):
        finished = 0

        def strategy(self, run):  # type: ignore[no-untyped-def]
            for _ in range(5):
                run.evaluate(run.seed())

        def finish_strategy(self, run):  # type: ignore[no-untyped-def]
            type(self).finished += 1

    _slow_train(monkeypatch, 1.2)
    config = LoopConfig(epochs=2, batch_size=1, optimizer="random", seed=7, runs_root=tmp_path / "run",
                        allowed_ops=("set_knob",), max_evaluations=60, holdout=0.0, max_elapsed_s=1,
                        parts_manifest=str(_write_parts(tmp_path, test=True)))
    with _patch_mode(LevelProbe):
        loop = Ending(MODE_ID, config, project=Path("."))
        report = loop.run()
    assert Ending.finished == 1
    assert "elapsed-time budget exhausted" in str(report["bound"])
