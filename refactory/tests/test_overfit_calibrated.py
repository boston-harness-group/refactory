"""The val-vs-test overfit audit counts only the gap the search added, against a
threshold that allows for small parts.

overfit = (winner_val - winner_test) - (seed_val - seed_test) > max(0.15, 2 x SE),
SE from the scores and the val and test part sizes; a seed winner is never overfit.
Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refactory_dsh.loop import Loop, LoopConfig, overfit_threshold
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


def test_small_parts_widen_the_threshold(tmp_path):
    """The same added gap of 0.40 is chance on 5-item parts and overfitting on
    200-item parts; the standard error is reported beside the threshold."""
    small = _audit(tmp_path / "s", seed=(0.5, 0.5), winner=(0.9, 0.5), val_items=5, test_items=5)
    large = _audit(tmp_path / "l", seed=(0.5, 0.5), winner=(0.9, 0.5), val_items=200, test_items=200)
    assert small["standard_error"] == pytest.approx(0.260768, abs=1e-5)
    assert small["threshold"] > 0.5 and small["overfit"] is False
    assert large["threshold"] == pytest.approx(0.15) and large["overfit"] is True
    assert overfit_threshold(0.6471, 17, 0.496, 125)[0] == pytest.approx(0.2485, abs=1e-4)
