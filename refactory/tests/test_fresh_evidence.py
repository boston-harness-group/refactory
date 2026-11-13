"""Fresh evidence: under ``val_gate`` the candidate the optimizer edits is re-scored
on each epoch's train sample with that epoch's seed, so its evidence is fresh.

As SkillOpt rolls out its current candidate on the train part every epoch with the
epoch's seed (``skillopt/loop.py``), the default loop re-scores the gate's current
candidate at the start of each epoch's proposal step -- unless its records already
come from this epoch (it was just created and scored). Only the evidence changes: the
candidate's id, train score and gate val score stay. Each test fails when the
behaviour it names is reverted.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from refactory_dsh import events
from refactory_dsh.loop import Loop
from refactory_dsh.optim import AutoresearchOptimizer
from refactory_dsh.outerloop import ValidationGate, train_batch_seed

from test_skillopt import TRAIN_IDS, _patch_mode
from test_train_minibatch import Watching, _events
from test_val_gate import LevelProbe, _level, _loop

SEED = 7


def _tag_seeds(monkeypatch: Any) -> list[tuple[int, str, int | None]]:
    """Each train record says which seed rolled it out (as its prediction), and each
    train measurement is listed as (epoch, candidate, seed)."""
    measured: list[tuple[int, str, int | None]] = []
    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        if part == "train":
            measured.append((self.epoch, candidate.id, kwargs.get("seed")))
            result.item_records = [{**r, "prediction": f"seed {kwargs.get('seed')}"}
                                   for r in result.item_records]
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)
    return measured


class SeedWatching(Watching):
    """Records, per proposal round, the epoch and the seeds its evidence came from."""

    def __init__(self, plan):  # type: ignore[no-untyped-def]
        super().__init__(plan)
        self.seeds: list[tuple[int, set[str]]] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.seeds.append((ctx.epoch, {str(r.get("prediction")) for r in ctx.item_records}))
        return super().propose(ctx, n, parents)


def _run(tmp_path: Path, monkeypatch: Any, plan, *, proposer=None, **overrides: Any):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    proposer = proposer or SeedWatching(plan)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"epochs": 3, "batch_size": 1, "val_gate": True, **overrides})
        loop.optimizer = proposer
        loop.run()
    return loop, proposer


def test_an_unchanged_current_candidate_is_rescored_each_epoch_with_its_seed(tmp_path, monkeypatch):
    """Every child is rejected (a lower level is a lower val score), so the seed stays
    current; each later epoch re-scores it on the whole train part with that epoch's
    seed, and the optimizer reads those records."""
    measured = _tag_seeds(monkeypatch)
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 3)], [(5, 2)], [(5, 1)]])
    seed_id = loop.population[0].id
    rescores = _events(loop, "rescore")
    assert [(r["id"], r["epoch"], r["seed"]) for r in rescores] == [
        (seed_id, 2, train_batch_seed(SEED, 2)), (seed_id, 3, train_batch_seed(SEED, 3))]
    for epoch, seeds in proposer.seeds:
        assert seeds == {f"seed {train_batch_seed(SEED, epoch)}"}, (epoch, seeds)
    assert all(ids == sorted(TRAIN_IDS) for _e, ids in proposer.evidence), "the whole train part"
    assert (seed_id, train_batch_seed(SEED, 3)) in {(c, s) for _e, c, s in measured}
    # A watcher sees each re-score: it costs a whole train evaluation.
    progress = [json.loads(line) for line in (loop.root / "progress.jsonl").read_text().splitlines()]
    shown = [events.render(r) for r in progress if r.get("step") == "rescore"]
    assert len(shown) == 2 and all("fresh evidence: re-scored" in line for line in shown), shown


def test_a_candidate_created_this_epoch_is_not_scored_twice(tmp_path, monkeypatch):
    """Each child is accepted and becomes current in the epoch that scored it, so its
    records are already that epoch's: it is not re-scored. Its parent is, beside it, on
    the round's sample (so the two pair on one seed) -- one re-score per round, the
    run's last proposals excepted; no candidate is scored twice in one epoch."""
    measured = _tag_seeds(monkeypatch)
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], [(8, 9)]])
    levels = {c.id: _level(c) for c in loop.population}
    assert [(levels[r["id"]], r["epoch"]) for r in _events(loop, "rescore")] == [(5, 2), (7, 3)]
    pairs = [(e, c) for e, c, _s in measured]
    assert len(pairs) == len(set(pairs)), "no candidate scored twice in one epoch"
    assert loop.evaluations == sum(1 for c in loop.population if c.evaluated) + 2
    for epoch, seeds in proposer.seeds:
        assert seeds == {f"seed {train_batch_seed(SEED, epoch)}"}, (epoch, seeds)


def test_the_score_identity_and_gate_score_stay(tmp_path, monkeypatch):
    """The re-score replaces the evidence only: the candidate keeps its id, its first
    train score and its val score."""
    loop, _ = _run(tmp_path, monkeypatch, [[(5, 3)], [(5, 2)], []])
    seed = loop.population[0]
    assert _events(loop, "rescore")
    assert seed.score == next(r["score"] for r in _events(loop, "score") if r["id"] == seed.id)
    assert loop._gate().cache[seed.id] == 0.5


def test_a_rescore_counts_and_is_not_repeated_after_a_resume(tmp_path, monkeypatch):
    """A re-score is an evaluation, a measurement and rollouts like any other, it is
    journaled, and a run that dies after it and resumes in the same epoch does not
    re-score again."""
    loop, _ = _run(tmp_path, monkeypatch, [[(5, 3)], [(5, 2)], []])
    rescores = _events(loop, "rescore")
    assert len(rescores) == 2
    evaluated = sum(1 for c in loop.population if c.evaluated)
    assert loop.evaluations == evaluated + 2
    assert loop.measurements >= evaluated + 2
    train_rollouts = (evaluated + 2) * len(TRAIN_IDS)
    assert loop.rollouts >= train_rollouts

    class Died(BaseException):
        pass

    class DieInEpochTwo(SeedWatching):
        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            if ctx.epoch == 2:
                raise Died()
            return super().propose(ctx, n, parents)

    root = tmp_path / "dies"
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(LevelProbe):
        first = _loop(root, epochs=3, batch_size=1, val_gate=True)
        first.optimizer = DieInEpochTwo([[(5, 3)], [], []])
        try:
            first.run()
        except Died:
            pass
    assert [r["epoch"] for r in _events(first, "rescore")] == [2]
    with _patch_mode(LevelProbe):
        resumed = _loop(root, epochs=3, batch_size=1, val_gate=True, resume=True)
        resumed.optimizer = SeedWatching([[], []])
        resumed.run()
    assert [r["epoch"] for r in _events(resumed, "rescore")] == [2, 3], "epoch 2 once"
    assert resumed.evaluations == sum(1 for c in resumed.population if c.evaluated) + 2


def test_without_the_gate_nothing_is_rescored(tmp_path, monkeypatch):
    measured = _tag_seeds(monkeypatch)
    loop, _ = _run(tmp_path / "nogate", monkeypatch, [[(5, 3)], [(5, 2)], []], val_gate=False)
    assert not _events(loop, "rescore")
    assert {s for _e, _c, s in measured} == {SEED}, "without the gate: the run's seed"


class _Client:
    """A model that answers readers with one pattern and proposes nothing."""

    def __init__(self) -> None:
        self.readers: list[tuple[int, str]] = []
        self.epoch = 0

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        if prompt.startswith("# Reading trajectories"):
            self.readers.append((self.epoch, prompt))
            return json.dumps({"failure_patterns": [], "success_patterns": [
                {"pattern": "ok", "step": "work", "items": ["t1"], "evidence": ["x"]}]})
        return json.dumps({"diagnosis": "nothing to change", "proposals": []})


def test_readers_read_the_fresh_records_every_epoch(tmp_path, monkeypatch):
    """With evidence too large for one call, readers go through it. Their cache is
    keyed by the records' text, so the fresh records of each epoch are read again,
    and each reader sees that epoch's rollout."""
    _tag_seeds(monkeypatch)
    client = _Client()
    original = Loop._context

    def small_budget(self):  # type: ignore[no-untyped-def]
        client.epoch = self.epoch
        return replace(original(self), evidence_budget=400)

    monkeypatch.setattr(Loop, "_context", small_budget)
    loop, _ = _run(tmp_path, monkeypatch, [], proposer=AutoresearchOptimizer(client))
    epochs = sorted({epoch for epoch, _prompt in client.readers})
    assert epochs == [1, 2, 3], epochs
    for epoch, prompt in client.readers:
        assert f"Predicted: seed {train_batch_seed(SEED, epoch)}" in prompt, epoch
    assert len(_events(loop, "rescore")) == 2


def _failing_rescore(monkeypatch: Any, how: str) -> None:
    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        if "train_epoch_" in str(workdir):
            if how == "failed":
                result.exit_ok, result.error, result.item_records = False, "the model is down", []
            else:
                result.item_records = result.item_records[:2]
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)


import pytest  # noqa: E402


@pytest.mark.parametrize("how", ["failed", "partial"])
def test_a_failed_or_partial_rescore_keeps_the_previous_evidence(tmp_path, monkeypatch, how):
    _failing_rescore(monkeypatch, how)
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 3)], [(5, 2)], []])
    seed_id = loop.population[0].id
    assert not _events(loop, "rescore")
    failed = _events(loop, "rescore_failed")
    assert [r["epoch"] for r in failed] == [2, 3]
    assert all(ids == sorted(TRAIN_IDS) for _e, ids in proposer.evidence), "the old records stay"
    assert [r["epoch"] for r in _events(loop, "train_records_epoch") if r["id"] == seed_id] == [1]
    assert loop.evaluations == sum(1 for c in loop.population if c.evaluated) + 2
    progress = [json.loads(line) for line in (loop.root / "progress.jsonl").read_text().splitlines()]
    shown = [events.render(r) for r in progress if r.get("step") == "rescore"]
    assert shown and all("previous records stay" in line for line in shown), shown


class PairWatching(SeedWatching):
    def __init__(self, plan):  # type: ignore[no-untyped-def]
        super().__init__(plan)
        self.pairs: list[tuple[int, list[dict[str, Any]]]] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.pairs.append((ctx.epoch, list(ctx.edit_pairs)))
        return super().propose(ctx, n, parents)


def test_a_child_is_compared_with_its_parent_scored_with_the_same_seed(tmp_path, monkeypatch):
    """The rejected child of epoch e (scored in epoch e+1) meets its parent's epoch e+1
    records."""
    _tag_seeds(monkeypatch)
    proposer = PairWatching([[(5, 3)], [(5, 2)], []])
    _run(tmp_path, monkeypatch, [], proposer=proposer)
    epoch, pairs = proposer.pairs[1]
    pair = next(p for p in pairs if p.get("parent"))
    seeds = {r.get("prediction") for r in pair["parent_records"]}
    child = {r.get("prediction") for r in pair["child_records"]}
    assert pair["same_seed"] is True and seeds == child == {f"seed {train_batch_seed(SEED, epoch)}"}


def test_an_older_child_is_compared_with_its_parents_records_from_its_own_epoch(tmp_path, monkeypatch):
    """In epoch 3 the parent has been re-scored twice; the child scored in epoch 2 is
    still compared with the parent's epoch 2 records (per-epoch records), not its latest."""
    _tag_seeds(monkeypatch)

    class ReasoningWatching(SeedWatching):
        def __init__(self, plan):  # type: ignore[no-untyped-def]
            super().__init__(plan)
            self.reasoning: list[tuple[int, list[dict[str, Any]]]] = []

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            self.reasoning.append((ctx.epoch, list(ctx.reasoning)))
            return super().propose(ctx, n, parents)

    proposer = ReasoningWatching([[(5, 3)], [(5, 2)], []])
    loop, _ = _run(tmp_path, monkeypatch, [], proposer=proposer)
    level3 = next(c.id for c in loop.population if _level(c) == 3)
    epoch, entries = proposer.reasoning[2]
    entry = next(e for e in entries if e["id"] == level3)
    assert epoch == 3 and entry["same_seed"] is True and entry["fixed"] is not None, entry
