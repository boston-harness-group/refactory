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
cause the record test to fail.  Each sabotage test generates only the 1–2
combos its mutation is known to change, compared against the corresponding
slice of the committed record.
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
            "leaderboard", "parents_chosen", "picker_parents", "picker_name",
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


def _slice_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Check that every key in *actual* matches the corresponding entry in *expected*."""
    for key in actual:
        if key not in expected:
            return False
        if actual[key] != expected[key]:
            return False
    return True


def test_sabotage_gate_population_noop():
    """Monkeypatch ``Loop._gate_population`` to a no-op: no candidates are
    gated, so decisions disappear."""
    expected = load_record()
    target = {"monotone|checkpoint|none|batch1"}

    with patch.object(Loop, "_gate_population", lambda self, run: None):
        actual = generate_record(keys=target)

    assert not _slice_matches(actual, expected), (
        "sabotage: a no-op _gate_population must change the record"
    )


def test_sabotage_select_parents_reversed():
    """Monkeypatch ``Loop.select_parents`` to reverse its result: the
    parents_chosen field changes for a scenario with >1 parent."""
    expected = load_record()
    target = {"structural_mix|single|none|batch3"}
    original = Loop.select_parents

    def reversed_parents(self, n, *, by="score"):  # type: ignore[no-untyped-def]
        result = original(self, n, by=by)
        return list(reversed(result))

    with patch.object(Loop, "select_parents", reversed_parents):
        actual = generate_record(keys=target)

    assert not _slice_matches(actual, expected), (
        "sabotage: reversed select_parents must change the record"
    )


def test_sabotage_checkpoint_apply_keeps_best():
    """Monkeypatch ``Loop._apply_round`` so that checkpoint keeps the best on
    val as current instead of the latest applied candidate."""
    expected = load_record()
    target = {"val_drop|checkpoint|none|batch1"}
    original = Loop._apply_round

    def sabotaged_apply(self, run, pending):  # type: ignore[no-untyped-def]
        original(self, run, pending)
        gate = self._gate()
        if gate.best_id and gate.best_score > gate.current_score:
            gate.current_id = gate.best_id
            gate.current_score = gate.best_score

    with patch.object(Loop, "_apply_round", sabotaged_apply):
        actual = generate_record(keys=target)

    assert not _slice_matches(actual, expected), (
        "sabotage: checkpoint keeping best-on-val must change the record"
    )


def test_sabotage_pool_ranking():
    """Monkeypatch ``Loop._pool`` to reverse its ranking: the pool and
    leaderboard fields change for pool-mode scenarios."""
    expected = load_record()
    target = {"monotone|pool|none|batch1"}
    original = Loop._pool

    def reversed_pool(self):  # type: ignore[no-untyped-def]
        return list(reversed(original(self)))

    with patch.object(Loop, "_pool", reversed_pool):
        actual = generate_record(keys=target)

    assert not _slice_matches(actual, expected), (
        "sabotage: reversed pool ranking must change the record"
    )


def test_sabotage_extra_val_evaluation():
    """Mutation: count one extra val evaluation per round."""
    expected = load_record()
    key = "monotone|checkpoint|none|batch1"
    actual = generate_record(keys={key})

    for d in actual[key]["decisions"]:
        d["val_evaluations"] += 1
    actual[key]["total_val_evaluations"] += len(actual[key]["decisions"])

    assert not _slice_matches(actual, expected), (
        "sabotage: extra val evaluation must change the record"
    )


def test_sabotage_pure_move_passes():
    """A pure move (one selection helper moved to another module with a
    re-export) must NOT change the record."""
    expected = load_record()
    actual = generate_record(keys={"monotone|checkpoint|none|batch1"})

    assert _slice_matches(actual, expected), (
        "pure re-import should not change decisions"
    )


def test_sabotage_map_elites_fitness_reversed():
    """Reverse the fitness comparison in ``MapElitesSelector.add``: the worst
    candidate becomes each cell's champion, changing picker_parents."""
    from refactory_dsh.optim.policies import MapElitesSelector

    expected = load_record()
    target = {"monotone|pool|map_elites|batch1"}

    def reversed_add(self, candidate, fitness=None):  # type: ignore[no-untyped-def]
        fitness = fitness or (lambda c: (c.score or 0.0))
        key = self.features(candidate)
        held = self.cells.get(key)
        if held is None or fitness(candidate) < fitness(held):
            self.cells[key] = candidate

    with patch.object(MapElitesSelector, "add", reversed_add):
        actual = generate_record(keys=target)

    assert not _slice_matches(actual, expected), (
        "sabotage: reversed map_elites fitness must change picker_parents"
    )


def test_sabotage_islands_max_min():
    """Replace ``max`` with ``min`` in ``IslandsSelector.select``: the worst
    member of each island is picked instead of the best."""
    from refactory_dsh.optim.policies import IslandsSelector

    expected = load_record()
    target = {"real_real3-OL-s42|pool|islands|batch1"}

    def min_select(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        islands: dict[int, list] = {}
        for candidate in parents:
            islands.setdefault(self.island_of(candidate), []).append(candidate)
        populated = sorted(islands)
        if not populated:
            return []
        fitness = self.fitness(ctx)
        chosen = []
        for offset in range(n):
            island = populated[(self._turn + offset) % len(populated)]
            members = islands[island]
            chosen.append(min(members, key=fitness))
        self._turn += 1
        return chosen

    with patch.object(IslandsSelector, "select", min_select):
        actual = generate_record(keys=target)

    assert not _slice_matches(actual, expected), (
        "sabotage: islands min-instead-of-max must change picker_parents"
    )


def test_sabotage_exit_ok_guard_removed():
    """Remove the ``exit_ok`` guard in ``_apply_round``: a partial-failure
    candidate (exit_ok=False, non-None score) gets applied."""
    expected = load_record()
    target = {"partial_fail|checkpoint|none|batch1"}
    original = Loop._apply_round

    def sabotaged_apply(self, run, pending):  # type: ignore[no-untyped-def]
        for c in pending:
            c.exit_ok = True
        original(self, run, pending)

    with patch.object(Loop, "_apply_round", sabotaged_apply):
        actual = generate_record(keys=target)

    assert not _slice_matches(actual, expected), (
        "sabotage: removing exit_ok guard must let the partial failure through"
    )
