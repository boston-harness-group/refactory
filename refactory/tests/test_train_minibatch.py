"""``train_minibatch``: each epoch's candidates are scored on a sampled train minibatch.

0 (the default) is the whole train part, as before. N > 0 with more train items than
N samples N ids per epoch with ``train_batch_seed`` -- the same within an epoch,
different across epochs, the same after a resume -- and the proposal evidence is that
epoch's minibatch. Val and test are unaffected. Each test fails when the behaviour it
names is reverted.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import profiles
from refactory_dsh.loop import Loop, LoopConfig, LoopError
from refactory_dsh.outerloop import Run, ValidationGate, train_batch_seed

from test_skillopt import TRAIN_IDS, VAL_IDS, _patch_mode
from test_val_gate import LevelProbe, ScriptedLevels, _level, _loop


class Watching(ScriptedLevels):
    def __init__(self, plan: list[list[tuple[int, int]]]) -> None:
        super().__init__(plan)
        self.evidence: list[tuple[int, list[str]]] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.evidence.append((ctx.epoch, sorted(str(r["id"]) for r in ctx.item_records)))
        return super().propose(ctx, n, parents)


def _scored(monkeypatch: Any) -> list[tuple[int, int, list[str] | None]]:
    calls: list[tuple[int, int, list[str] | None]] = []
    original = Loop._evaluate

    def evaluate(self, candidate, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((self.epoch, _level(candidate), kwargs.get("items")))
        return original(self, candidate, **kwargs)

    monkeypatch.setattr(Loop, "_evaluate", evaluate)
    return calls


def _expected(seed: int, epoch: int, size: int) -> list[str]:
    return sorted(random.Random(train_batch_seed(seed, epoch)).sample(sorted(TRAIN_IDS), size))


def _run(tmp_path: Path, monkeypatch: Any, plan: list[list[tuple[int, int]]], **overrides: Any):
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    proposer = Watching(plan)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"epochs": 3, "batch_size": 2, **overrides})
        loop.optimizer = proposer
        loop.run()
    return loop, proposer


def _events(loop: Loop, name: str) -> list[dict[str, Any]]:
    return [r for r in map(json.loads, (loop.root / "journal.jsonl").read_text().splitlines())
            if r.get("event") == name]


def test_the_default_scores_the_whole_train_part(tmp_path, monkeypatch):
    calls = _scored(monkeypatch)
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], []])
    assert calls and all(items is None for _e, _l, items in calls)
    assert not _events(loop, "train_minibatch")
    assert all(ids == sorted(TRAIN_IDS) for _e, ids in proposer.evidence)


def test_each_epoch_scores_its_candidates_on_one_seeded_sample(tmp_path, monkeypatch):
    """N=3 of 8 train items: every candidate of an epoch on the same 3 ids, each
    epoch its own sample from train_batch_seed, journaled."""
    calls = _scored(monkeypatch)
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    loop, _ = _run(tmp_path, monkeypatch, [[(5, 7), (5, 3)], [(7, 8), (7, 9)], []],
                   train_minibatch=3, val_gate=True)
    by_epoch: dict[int, set[tuple[str, ...]]] = {}
    for epoch, _level_, items in calls:
        if epoch <= 3:  # the last epoch's proposals are scored after it, on epoch 4's sample
            by_epoch.setdefault(epoch, set()).add(tuple(items))
    assert all(len(samples) == 1 for samples in by_epoch.values()), by_epoch
    samples = {epoch: list(next(iter(s))) for epoch, s in by_epoch.items()}
    for epoch, sample in samples.items():
        assert sample == _expected(7, epoch, 3), epoch
    assert len({tuple(s) for s in samples.values()}) > 1, "the sample varies by epoch"
    journaled = {r["epoch"]: r["ids"] for r in _events(loop, "train_minibatch")}
    assert all(journaled[epoch] == sample for epoch, sample in samples.items())


def test_the_evidence_is_the_epochs_minibatch(tmp_path, monkeypatch):
    """The edited candidate's records shown to the optimizer are that epoch's
    sample, re-scored when the candidate was scored in an earlier epoch."""
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 3)], [(7, 2)]],
                          train_minibatch=3, val_gate=True, batch_size=1)
    for epoch, ids in proposer.evidence:
        assert ids == _expected(7, epoch, 3), (epoch, ids)


def test_a_train_part_no_larger_than_n_is_scored_whole(tmp_path, monkeypatch):
    calls = _scored(monkeypatch)
    _run(tmp_path, monkeypatch, [[(5, 7)], [], []], train_minibatch=len(TRAIN_IDS))
    assert all(items is None for _e, _l, items in calls)


def test_val_rollouts_are_unaffected(tmp_path, monkeypatch):
    seen: list[int] = []
    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        if part == "val":
            seen.append(len(result.item_records))
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)
    _run(tmp_path, monkeypatch, [[(5, 7)], [], []], train_minibatch=3, val_gate=True,
         batch_size=1)
    assert seen and set(seen) == {len(VAL_IDS)}


def test_a_resume_samples_the_same_ids(tmp_path, monkeypatch):
    """A run stopped after one epoch and resumed scores the same minibatches as one
    that ran straight through."""
    straight, _ = _run(tmp_path / "straight", monkeypatch, [[(5, 7)], [(7, 8)], []],
                       train_minibatch=3, batch_size=1, val_gate=True)
    _run(tmp_path / "resumed", monkeypatch, [[(5, 7)]], train_minibatch=3, batch_size=1, epochs=1,
         val_gate=True)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path / "resumed", epochs=3, batch_size=1, train_minibatch=3, resume=True,
                        val_gate=True)
        resumed.optimizer = Watching([[(7, 8)], []])
        resumed.run()
    def batches(loop: Loop) -> dict[int, list[str]]:
        return {r["epoch"]: r["ids"] for r in _events(loop, "train_minibatch")}
    assert batches(resumed) == batches(straight)


def test_a_negative_size_is_refused(tmp_path):
    with pytest.raises(profiles.ProfileError, match="train_minibatch"):
        profiles.normalize({"train_minibatch": -1})
    with _patch_mode(LevelProbe), pytest.raises(LoopError, match="train_minibatch"):
        _loop(tmp_path, train_minibatch=-2)
    import argparse
    parser = argparse.ArgumentParser()
    profiles.add_arguments(parser)
    args = parser.parse_args(["--train-minibatch", "5"])
    assert profiles.normalize({"train_minibatch": args.train_minibatch})["train_minibatch"] == 5



def test_a_minibatch_without_the_gate_is_refused_before_any_rollout(tmp_path):
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, train_minibatch=3, val_gate=False)
        with pytest.raises(LoopError, match="give the run val items"):
            loop.run()
    assert loop.rollouts == 0 and not (loop.root / "runs").exists()
    with _patch_mode(LevelProbe):
        whole = _loop(tmp_path / "whole", train_minibatch=len(TRAIN_IDS), val_gate=False)
        whole.run()  # a "minibatch" as large as the train part is the whole part: allowed


def test_a_rescore_is_an_evaluation_and_survives_a_resume(tmp_path, monkeypatch):
    """The edited candidate's re-score on a new epoch's minibatch counts toward the
    evaluation budget, is journaled, and is still counted after a resume."""
    loop, _ = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 3)], [(7, 2)]],
                   train_minibatch=3, val_gate=True, batch_size=1)
    rescores = _events(loop, "rescore")
    assert rescores, "setup: the edited candidate was re-scored"
    assert loop.evaluations == sum(1 for c in loop.population if c.evaluated) + len(rescores)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=3, batch_size=1, train_minibatch=3, val_gate=True,
                        resume=True)
        resumed.optimizer = Watching([[]])
        resumed.run()
    assert resumed.evaluations >= loop.evaluations


def test_a_resume_with_another_size_is_refused(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, [[(5, 7)]], train_minibatch=3, val_gate=True, batch_size=1,
         epochs=1)
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=2, batch_size=1, train_minibatch=4, val_gate=True,
                        resume=True)
        with pytest.raises(LoopError, match="resume it with train_minibatch 3"):
            resumed.run()


class _SpentBudget:
    """A budget whose time or tokens are spent: every checkpoint refuses."""

    def __init__(self, *, refuse_first: bool = True) -> None:
        self.refuse_first = refuse_first
        self.calls = 0

    def checkpoint(self, *args: Any, **kwargs: Any) -> None:
        from refactory_dsh.budgets import BudgetExhausted

        self.calls += 1
        if self.refuse_first or self.calls > 1:
            raise BudgetExhausted("the time budget is spent")

    def snapshot(self) -> dict[str, Any]:
        return {"exhausted": True}

    def retain_validation_reserve(self):  # type: ignore[no-untyped-def]
        import contextlib
        return contextlib.nullcontext()


def _ready_for_rescore(tmp_path: Path, monkeypatch: Any) -> Loop:
    """A finished gated minibatch run, set up as the next epoch about to re-score its
    current candidate."""
    loop, _ = _run(tmp_path, monkeypatch, [[(5, 7)], []], train_minibatch=3, val_gate=True,
                   batch_size=1, epochs=2)
    loop.epoch += 1
    loop._epoch_open = True
    loop._editing_parent = loop._gate().current_id
    return loop


def test_a_spent_time_or_token_budget_stops_the_rescore_before_it_runs(tmp_path, monkeypatch):
    loop = _ready_for_rescore(tmp_path, monkeypatch)
    measured: list[str] = []
    monkeypatch.setattr(Loop, "_measure", lambda self, *a, **k: measured.append("m") or (None, {}))
    before = (loop.evaluations, list(loop._train_item_records[loop._editing_parent]))
    rescores = len(_events(loop, "rescore"))
    loop.budget = _SpentBudget()
    loop._refresh_edited_records()
    assert measured == [], "nothing is measured on a spent budget"
    assert (loop.evaluations, loop._train_item_records[loop._editing_parent]) == (
        before[0], before[1])
    assert len(_events(loop, "rescore")) == rescores and "spent" in loop.bound


def test_a_rescore_the_measurement_stopped_counts_nothing(tmp_path, monkeypatch):
    """The budget runs out between the loop's check and the measurement's own: no
    trial ran, so no evaluation is counted and the records stay."""
    loop = _ready_for_rescore(tmp_path, monkeypatch)
    edited = loop._editing_parent
    before = (loop.evaluations, list(loop._train_item_records[edited]))
    rescores = len(_events(loop, "rescore"))
    loop.budget = _SpentBudget(refuse_first=False)
    loop._owns_resource_budget = True
    loop._refresh_edited_records()
    assert (loop.evaluations, loop._train_item_records[edited]) == before
    assert len(_events(loop, "rescore")) == rescores
