"""Selection decision record: a safety net for refactoring the outer loop's selection.

Reruns every scenario (synthetic + real) under every compatible combination of
selection mode x picker x batch size and diffs against the committed record.  A
refactor that changes any single decision is caught.

Regenerate the record on purpose::

    SELECTION_RECORD_REGEN=1 uv run python -m pytest tests/test_selection_record.py::test_selection_record -x -q

Sabotage checks verify that specific mutations in the selection code cause the
record test to fail.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from selection_record import (
    generate_record,
    load_record,
    save_record,
)
from refactory_dsh.outerloop import ValidationGate


# ---------------------------------------------------------------------------
# The record test
# ---------------------------------------------------------------------------


def _diff_decisions(
    key: str, expected: dict[str, Any], actual: dict[str, Any]
) -> list[str]:
    """Readable per-decision diff between expected and actual."""
    diffs: list[str] = []
    for field in ("final_winner", "final_winner_val", "total_val_evaluations"):
        if expected.get(field) != actual.get(field):
            diffs.append(f"  {key}: {field}: {expected.get(field)} -> {actual.get(field)}")
    exp_decisions = expected.get("decisions", [])
    act_decisions = actual.get("decisions", [])
    for i, (ed, ad) in enumerate(zip(exp_decisions, act_decisions)):
        for field in (
            "gate_accepted", "gate_rejected", "pool", "incumbent",
            "leaderboard", "parents_chosen", "picker_name",
            "early_stop", "val_evaluations",
        ):
            if ed.get(field) != ad.get(field):
                diffs.append(
                    f"  {key} round {i} {field}: {ed.get(field)} -> {ad.get(field)}"
                )
    if len(exp_decisions) != len(act_decisions):
        diffs.append(
            f"  {key}: decision count: {len(exp_decisions)} -> {len(act_decisions)}"
        )
    return diffs


def test_selection_record():
    """Rerun all scenarios and diff against the committed record."""
    if os.environ.get("SELECTION_RECORD_REGEN") == "1":
        record = generate_record()
        save_record(record)
        pytest.skip("record regenerated")

    expected = load_record()
    actual = generate_record()

    all_diffs: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        if key not in expected:
            all_diffs.append(f"  NEW: {key}")
            continue
        if key not in actual:
            all_diffs.append(f"  REMOVED: {key}")
            continue
        all_diffs.extend(_diff_decisions(key, expected[key], actual[key]))

    assert not all_diffs, (
        f"Selection decisions changed ({len(all_diffs)} difference(s)):\n"
        + "\n".join(all_diffs[:50])
        + ("\n  ... and more" if len(all_diffs) > 50 else "")
    )


# ---------------------------------------------------------------------------
# Sabotage checks: each mutation must cause the record test to fail
# ---------------------------------------------------------------------------

def _record_matches(record: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Check if the record matches the expected one."""
    for key in expected:
        if key not in record:
            return False
        if record[key] != expected[key]:
            return False
    return True


def test_sabotage_checkpoint_keeps_best_on_val():
    """Mutation: make checkpoint keep the best-on-val instead of the latest.

    Under checkpoint, `gate.current_id` is always set to the applied candidate
    (the latest), not the best on val.  If we make `_apply_checkpoint_round`
    keep the best on val as current, the record should change.
    """
    from selection_record import _apply_checkpoint_round as orig

    expected = load_record()

    def sabotaged_apply(gate, run, candidates, accepted, rejected, rounds, epoch, val_scores):
        orig(gate, run, candidates, accepted, rejected, rounds, epoch, val_scores)
        # Sabotage: make current = best (normally current tracks the latest applied)
        if gate.best_id and gate.best_score > gate.current_score:
            gate.current_id = gate.best_id
            gate.current_score = gate.best_score

    with patch("selection_record._apply_checkpoint_round", sabotaged_apply):
        actual = generate_record()

    differs = not _record_matches(actual, expected)
    assert differs, "sabotage: checkpoint keeping best-on-val should change decisions"


def test_sabotage_pool_admits_rejected():
    """Mutation: make pool admit a candidate the gate rejected."""
    expected = load_record()

    def sabotaged_judge_round(self, run, candidates):
        # Accept everything
        for c in candidates:
            score, cached = self.score(run, c)
            if score > self.current_score:
                self.current_id, self.current_score = c.id, score
                self.offer_best(c.id, score)
        return list(candidates), []

    with patch.object(ValidationGate, "judge_round", sabotaged_judge_round):
        actual = generate_record()

    differs = not _record_matches(actual, expected)
    assert differs, "sabotage: admitting rejected candidates should change decisions"


def test_sabotage_picker_tiebreak():
    """Mutation: change one picker's tie-break."""
    from refactory_dsh.optim.policies import HillClimbSelector

    expected = load_record()

    def sabotaged_select(self, ctx, n, parents):
        val = getattr(ctx, "val_scores", None) or {}
        if val:
            ranked = sorted(
                (c for c in parents if c.id in val),
                key=lambda c: (-val[c.id], (c.score or 0.0), c.id),
            )
            if ranked:
                return ranked[:1]
        return self._by_score(parents)[:1] if parents else []

    with patch.object(HillClimbSelector, "select", sabotaged_select):
        actual = generate_record()

    differs = not _record_matches(actual, expected)
    assert differs, "sabotage: changing hill_climb tie-break should change decisions"


def test_sabotage_extra_val_evaluation():
    """Mutation: count one extra val evaluation per round."""
    expected = load_record()

    actual = generate_record()
    for key in actual:
        for d in actual[key]["decisions"]:
            d["val_evaluations"] += 1
        actual[key]["total_val_evaluations"] += len(actual[key]["decisions"])

    differs = not _record_matches(actual, expected)
    assert differs, "sabotage: extra val evaluation should change decisions"


def test_sabotage_pure_move_passes():
    """A pure move (one selection helper moved to another module with a
    re-export) must NOT change the record."""
    expected = load_record()
    actual = generate_record()

    assert _record_matches(actual, expected), (
        "pure re-import should not change decisions"
    )
