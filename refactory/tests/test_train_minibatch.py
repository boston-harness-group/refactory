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
    loop, _ = _run(tmp_path, monkeypatch, [[(5, 7), (5, 3)], [(7, 8), (7, 9)], []],
                   train_minibatch=3)
    by_epoch: dict[int, set[tuple[str, ...]]] = {}
    for epoch, _level_, items in calls:
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
                       train_minibatch=3, batch_size=1)
    _run(tmp_path / "resumed", monkeypatch, [[(5, 7)]], train_minibatch=3, batch_size=1, epochs=1)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path / "resumed", epochs=3, batch_size=1, train_minibatch=3, resume=True)
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
