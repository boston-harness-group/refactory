"""The val-vs-test overfit audit counts only the gap the search added.

overfit = (winner_val - winner_test) - (seed_val - seed_test) > 0.15 (fixed); the gap's
standard error from the part sizes is reported beside it for information. A seed
winner is never overfit; without the seed's test score the audit is unmeasured.
Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refactory_dsh.loop import Loop, LoopConfig, gap_standard_error
from refactory_dsh.optim import Candidate

from test_skillopt import _patch_mode
from test_val_gate import LevelProbe, _loop


def _audit(tmp_path: Path, *, seed: tuple[float, float] | None, winner: tuple[float, float] | None,
           val_items: int, test_items: int):
    """``audit_parts`` on a loop holding a seed and (unless ``winner`` is None) one
    optimized candidate, with the given (val, test) scores and part sizes."""
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path)
    seed_c = Candidate.from_state({"graph": {"nodes": {}}, "knobs": {"level": {"value": 5}}})
    child = Candidate.from_state({"graph": {"nodes": {}}, "knobs": {"level": {"value": 7}}},
                                 parent=seed_c.id)
    loop.population = [seed_c, child]
    loop._parts_manifest = {"train": ["t1"], "val": [f"v{i}" for i in range(val_items)],
                            "test": [f"x{i}" for i in range(test_items)]}

    def put(candidate: Candidate, scores: tuple[float, float]) -> None:
        loop._part_results[candidate.id] = {"val": [{"score": scores[0]}],
                                            "test": [{"score": scores[1]}]}

    if seed is not None:
        put(seed_c, seed)
    if winner is not None:
        put(child, winner)
        return loop.audit_parts(child)
    return loop.audit_parts(seed_c)


def test_a_seed_winner_is_never_overfit(tmp_path):
    """The live finding: the seed won (nothing was accepted), val 0.6471 on 17 items,
    test 0.4960 on 125. Its gap is the parts' difference, not overfitting."""
    audit = _audit(tmp_path, seed=(0.6471, 0.4960), winner=None, val_items=17, test_items=125)
    assert audit["status"] == "measured" and audit["overfit"] is False
    assert audit["added_gap"] == 0.0 and audit["gap"] == pytest.approx(0.1511)
    assert "the winner is the seed" in audit["note"]
    # A seed with an even larger raw gap is still not overfit.
    big = _audit(tmp_path / "big", seed=(0.95, 0.30), winner=None, val_items=200, test_items=200)
    assert big["overfit"] is False


def test_an_optimized_winner_whose_gap_grew_is_overfit(tmp_path):
    """Seed gap 0.05, winner gap 0.45 on 100-item parts: the search added 0.40,
    beyond the threshold, so the winner overfit -- real overfitting still fails."""
    audit = _audit(tmp_path, seed=(0.60, 0.55), winner=(0.95, 0.50), val_items=100, test_items=100)
    assert audit["seed_gap"] == pytest.approx(0.05) and audit["added_gap"] == pytest.approx(0.40)
    assert audit["threshold"] == pytest.approx(0.15) and audit["overfit"] is True


def test_a_gap_the_seed_already_had_is_not_overfitting(tmp_path):
    audit = _audit(tmp_path, seed=(0.70, 0.40), winner=(0.80, 0.50), val_items=100, test_items=100)
    assert audit["gap"] == pytest.approx(0.30) and audit["added_gap"] == pytest.approx(0.0)
    assert audit["overfit"] is False


def test_the_standard_error_is_reported_and_does_not_move_the_threshold(tmp_path):
    """An added gap of 0.40 on 5-item parts is overfitting: the threshold stays 0.15;
    the standard error is there for a reader."""
    small = _audit(tmp_path / "s", seed=(0.5, 0.5), winner=(0.9, 0.5), val_items=5, test_items=5)
    assert small["standard_error"] == pytest.approx(0.260768, abs=1e-5)
    assert small["threshold"] == 0.15 and small["overfit"] is True
    assert gap_standard_error(0.6471, 17, 0.496, 125) == pytest.approx(0.124229, abs=1e-5)


def test_without_the_seeds_test_score_the_audit_is_unmeasured(tmp_path):
    """The seed has a val score but no test score (a skipped or incomplete final
    evaluation): neither overfit nor clean."""
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path)
    seed_c = Candidate.from_state({"graph": {"nodes": {}}, "knobs": {"level": {"value": 5}}})
    child = Candidate.from_state({"graph": {"nodes": {}}, "knobs": {"level": {"value": 7}}},
                                 parent=seed_c.id)
    loop.population = [seed_c, child]
    loop._parts_manifest = {"train": ["t1"], "val": ["v1"], "test": ["x1"]}
    loop._part_results[seed_c.id] = {"val": [{"score": 0.5}]}
    loop._part_results[child.id] = {"val": [{"score": 0.9}], "test": [{"score": 0.4}]}
    audit = loop.audit_parts(child)
    assert audit["status"] == "unmeasured" and audit["overfit"] is None
    assert audit["note"].startswith("unmeasured: the seed's test score is missing")


def test_a_candidate_that_games_its_scorer_on_a_small_val_part_is_overfit(tmp_path, monkeypatch):
    """The #76 kind of failure, run for real: the gate accepts a candidate that scores
    high on the 4-item val part and low on test, while the seed scores the same on
    both. The final evaluation measures it, the audit calls it overfit and the run
    fails."""
    from refactory_dsh.outerloop import ValidationGate
    from test_val_gate import ScriptedLevels, _level

    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        gaming = _level(candidate) == 9
        if part == "val":
            result.score = 1.0 if gaming else 0.5
        elif part == "test":
            result.score = 0.2 if gaming else 0.5
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(
        lambda run, cand: 1.0 if _level(cand) == 9 else 0.5))
    proposer = ScriptedLevels([[(5, 9)], []])
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=2, val_gate=True)
        loop.optimizer = proposer
        report = loop.run()
    audit = report["holdout"]
    assert _level(next(c for c in loop.population if c.id == report["best"]["id"])) == 9
    assert audit["seed_gap"] == pytest.approx(0.0) and audit["added_gap"] == pytest.approx(0.8)
    assert audit["overfit"] is True and report["exit_ok"] is False
    assert "overfit" in report["error"]
