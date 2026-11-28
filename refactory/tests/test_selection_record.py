"""Selection decision record: a safety net for refactoring the outer loop's selection.

Reruns every scenario (synthetic + real) under every compatible combination of
selection mode x picker x batch size and diffs against the committed record.  A
refactor that changes any single decision is caught.

The harness drives REAL Loop objects — the same ``Loop._gate_population``,
``Loop.select_parents``, ``Loop._apply_round``, ``Loop.leaderboard``, and
``Loop._early_stop_reason`` that production runs use.  The only stubs are the
evaluator (scripted train scores via ``ScenarioProbe``) and
``ValidationGate.val_score`` (scripted val scores by level).

Regenerate the record on purpose::

    SELECTION_RECORD_REGEN=1 uv run python -m pytest tests/test_selection_record.py::test_selection_record -x -q

Sabotage checks verify that specific mutations in the REAL selection code
cause the record test to fail.
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
from refactory_dsh.loop import Loop


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


def test_sabotage_gate_population_noop():
    """Monkeypatch ``Loop._gate_population`` to a no-op: no candidates are
    gated, so every selection decision changes."""
    expected = load_record()

    with patch.object(Loop, "_gate_population", lambda self, run: None):
        actual = generate_record()

    assert not _record_matches(actual, expected), (
        "sabotage: a no-op _gate_population must change the record"
    )


def test_sabotage_select_parents_reversed():
    """Monkeypatch ``Loop.select_parents`` to reverse its result: the
    parents_chosen field changes for every scenario with more than one
    candidate."""
    expected = load_record()
    original = Loop.select_parents

    def reversed_parents(self, n, *, by="score"):  # type: ignore[no-untyped-def]
        result = original(self, n, by=by)
        return list(reversed(result))

    with patch.object(Loop, "select_parents", reversed_parents):
        actual = generate_record()

    assert not _record_matches(actual, expected), (
        "sabotage: reversed select_parents must change the record"
    )


def test_sabotage_checkpoint_apply_keeps_best():
    """Monkeypatch ``Loop._apply_round`` so that checkpoint keeps the best on
    val as current instead of the latest applied candidate."""
    expected = load_record()
    original = Loop._apply_round

    def sabotaged_apply(self, run, pending):  # type: ignore[no-untyped-def]
        original(self, run, pending)
        gate = self._gate()
        if gate.best_id and gate.best_score > gate.current_score:
            gate.current_id = gate.best_id
            gate.current_score = gate.best_score

    with patch.object(Loop, "_apply_round", sabotaged_apply):
        actual = generate_record()

    assert not _record_matches(actual, expected), (
        "sabotage: checkpoint keeping best-on-val must change the record"
    )


def test_sabotage_pool_ranking():
    """Monkeypatch ``Loop._pool`` to reverse its ranking: the pool and
    leaderboard fields change for every pool-mode scenario."""
    expected = load_record()
    original = Loop._pool

    def reversed_pool(self):  # type: ignore[no-untyped-def]
        return list(reversed(original(self)))

    with patch.object(Loop, "_pool", reversed_pool):
        actual = generate_record()

    assert not _record_matches(actual, expected), (
        "sabotage: reversed pool ranking must change the record"
    )


def test_sabotage_extra_val_evaluation():
    """Mutation: count one extra val evaluation per round."""
    expected = load_record()

    actual = generate_record()
    for key in actual:
        for d in actual[key]["decisions"]:
            d["val_evaluations"] += 1
        actual[key]["total_val_evaluations"] += len(actual[key]["decisions"])

    assert not _record_matches(actual, expected), (
        "sabotage: extra val evaluation must change the record"
    )


def test_sabotage_pure_move_passes():
    """A pure move (one selection helper moved to another module with a
    re-export) must NOT change the record."""
    expected = load_record()
    actual = generate_record()

    assert _record_matches(actual, expected), (
        "pure re-import should not change decisions"
    )
