"""With ``val_gate``, the optimizer reads what the gate decided, never a val score.

Every row the optimizer sees (recent results, the history, the gradient's
leaderboard) and the per-knob attribution carry the gate's decision -- current, kept,
rejected or not gated -- and the request says the current candidate is what it edits
and a rejected one did not generalise. A duplicate proposal is told which candidate it
repeats. Without ``val_gate`` nothing changes. Each test fails when the behaviour it
names is reverted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from refactory_dsh.outerloop import ValidationGate

from test_skillopt import _patch_mode
from test_val_gate import LevelProbe, ScriptedLevels, _level, _loop


class Recording(ScriptedLevels):
    def __init__(self, plan: list[list[tuple[int, int]]]) -> None:
        super().__init__(plan)
        self.contexts: list[Any] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.contexts.append(ctx)
        return super().propose(ctx, n, parents)


#: Val scores no train score or knob value could print as.
def _val(run: Any, cand: Any) -> float:
    return 0.1234 + _level(cand) / 1000


def _run(tmp_path: Path, monkeypatch: Any, plan: list[list[tuple[int, int]]], **overrides: Any):
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    proposer = Recording(plan)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"epochs": 3, **overrides})
        loop.optimizer = proposer
        loop.run()
    return loop, proposer


def _by_level(rows: list[dict[str, Any]], loop: Any) -> dict[int, dict[str, Any]]:
    levels = {c.id: _level(c) for c in loop.population}
    return {levels[row["id"]]: row for row in rows}


def test_rows_and_the_request_carry_the_gates_decision_and_no_val_score(tmp_path, monkeypatch):
    """Seed 5, then 7 (kept, current), then 6 (rejected): the third request shows
    each decision, says what they mean, and prints no val score."""
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 6)], []], val_gate=True)
    ctx = proposer.contexts[2]
    history = _by_level(ctx.history, loop)
    assert {level: row["gate"] for level, row in history.items()} == {
        5: "kept", 7: "current", 6: "rejected"}
    assert {row["gate"] for row in ctx.leaderboard} <= {"current", "kept"}
    text = ctx.render()
    assert "You edit the candidate marked `current`" in text
    assert "A `rejected` candidate did not generalise" in text
    assert "| id | parent | score | structural | ok | edits | gate |" in text
    assert "| id | parent | score | structural | edits | gate |" in text
    row6 = next(line for line in text.splitlines() if history[6]["id"] in line and "rejected" in line)
    assert row6.rstrip().endswith("| rejected |")
    assert "best kept by the validation gate" in text
    for score in loop._gate().cache.values():
        assert f"{score:.6g}" not in text and str(score) not in text, score
    assert not any("val" in key for row in ctx.history for key in row)


def test_the_knob_attribution_says_what_the_gate_decided(tmp_path, monkeypatch):
    loop, _ = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 6)], []], val_gate=True)
    tried = loop.gradient["knob_attribution"]["level"]["tried"]
    assert tried["6"]["gate"] == {"kept": 0, "rejected": 1, "not gated": 0}
    assert tried["7"]["gate"] == {"kept": 1, "rejected": 0, "not gated": 0}
    rows = _by_level(loop.gradient["leaderboard"], loop)
    assert rows[7]["gate"] == "current"


def test_without_the_gate_nothing_changes(tmp_path, monkeypatch):
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 6)], []])
    ctx = proposer.contexts[2]
    assert not any("gate" in row for row in ctx.history + ctx.leaderboard)
    text = ctx.render()
    assert "validation gate" not in text and "| gate |" not in text
    assert "gate" not in loop.gradient["knob_attribution"]["level"]["tried"]["7"]


def test_a_duplicate_is_told_which_candidate_it_repeats(tmp_path, monkeypatch):
    """Proposing the seed's state again: the refusal names the seed's id, and the
    retry reads it."""
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 5)], [], []], val_gate=True,
                          attempts=2)
    seed = loop.population[0].id
    reasons = [r for rejection in loop.rejections for r in rejection.reasons]
    assert any(r.startswith("E-duplicate:") and f"as candidate {seed}" in r for r in reasons)
    assert any(f"as candidate {seed}" in "\n".join(ctx.rejections) for ctx in proposer.contexts[2:])


def _epoch_lines(loop: Any) -> list[str]:
    import json

    from refactory_dsh import events

    records = [json.loads(line) for line in (loop.root / "progress.jsonl").read_text().splitlines()]
    return [events.render(r) for r in records if r.get("step") == "scheduler"]


def test_the_epoch_line_says_what_the_gate_decided(tmp_path, monkeypatch):
    """Epoch 3 gates 6, which the gate rejects though its train score is the best:
    the line says kept 0, rejected 1, with the val best and current, and does not
    call it an improvement."""
    loop, _ = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 6)], []], val_gate=True)
    lines = _epoch_lines(loop)
    assert lines[1].startswith("epoch 2 done · validation gate kept 1, rejected 0")
    assert lines[2].startswith("epoch 3 done · validation gate kept 0, rejected 1")
    assert "val best 0.130 (current 0.130)" in lines[2]  # 0.1234 + 7/1000
    assert "train best" in lines[2] and "improved" not in lines[2]


def test_without_the_gate_the_epoch_line_is_unchanged(tmp_path, monkeypatch):
    loop, _ = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 6)], []])
    lines = _epoch_lines(loop)
    assert all(" · improved " in line and "validation gate" not in line for line in lines)
