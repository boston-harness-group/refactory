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
    row6 = next(line for line in text.splitlines()
                if line.startswith("| ") and history[6]["id"] in line and "rejected" in line)
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



# ── what the last edit changed, and the optimizer's own reasoning ──────────


def _passes_by_level(monkeypatch: Any) -> None:
    """Train item tN passes when N <= level - 2 (so 7 passes t1-t5, 6 passes t1-t4);
    graded scores rise with the level. Val and test records are left alone."""
    from refactory_dsh.loop import Loop

    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        if part == "train":
            level = _level(candidate)
            for record in result.item_records:
                n = int(str(record["id"]).lstrip("t"))
                record["passed"] = n <= level - 2
                record["score"] = round(level / 10 + n / 100, 3)
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)


class Reasoning(Recording):
    """Proposals that state a rationale and a prediction, and a diagnosis."""

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.last_diagnosis = f"DIAGNOSIS-{len(self.contexts) + 1}"
        from dataclasses import replace

        return [replace(proposal, rationale=f"RATIONALE-{len(self.contexts)}",
                        prediction=f"PREDICTION-{len(self.contexts)}")
                for proposal in super().propose(ctx, n, parents)]


def _run_reasoning(tmp_path: Path, monkeypatch: Any, **overrides: Any):
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    _passes_by_level(monkeypatch)
    proposer = Reasoning([[(5, 7)], [(7, 6)], [], []])
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"epochs": 3, "val_gate": True, **overrides})
        loop.optimizer = proposer
        loop.run()
    return loop, proposer


def test_the_last_edit_is_compared_item_by_item_with_its_parent(tmp_path, monkeypatch):
    """Epoch 3 edits 7, whose child 6 the gate rejected: 6 against 7 on their shared
    train items, t5 broken, with both trajectories, the edit, the gate's decision
    and the noise warning; no val or test item appears."""
    loop, proposer = _run_reasoning(tmp_path, monkeypatch)
    text = proposer.contexts[2].render()
    section = text[text.index("## What the last edit changed"):text.index("\n## ", text.index("## What the last edit changed") + 5)]
    ids = {_level(c): c.id for c in loop.population}
    assert f"### Child {ids[6]} of {ids[7]} -- gate: rejected" in section
    assert "On 8 shared train item(s): 0 fixed (parent wrong, child right), 1 broken" in section
    assert "#### Item t5 -- broken\nParent:\n### Item t5 — passed" in section
    assert "Child:\n### Item t5 — answered wrong" in section
    assert '"value": 6' in section and "different rounds with different seeds" in section
    assert not any(f"Item {v}" in section for v in ("v1", "v2", "v3", "v4", "x1", "x2", "x3"))


def test_without_a_parent_or_shared_items_the_section_says_so(tmp_path, monkeypatch):
    from refactory_dsh.optim.base import _edit_change_lines, edit_change

    loop, proposer = _run_reasoning(tmp_path, monkeypatch)
    first = proposer.contexts[0].render()
    assert "it is the seed), so there is nothing to compare" in first
    lines = "\n".join(_edit_change_lines([{"child": "c", "parent": "p", "edit": [],
                                            "child_records": [{"id": "a", "passed": True}],
                                            "parent_records": [{"id": "b", "passed": False}]}]))
    assert "share no scored train item" in lines
    partial = edit_change([{"id": "a", "passed": False, "score": 0.1},
                           {"id": "b", "passed": True, "score": 0.9},
                           {"id": "c", "passed": True, "score": 0.5}],
                          [{"id": "a", "passed": True, "score": 0.8},
                           {"id": "c", "passed": True, "score": 0.2},
                           {"id": "d", "passed": False, "score": 0.0}])
    assert partial["shared"] == 2 and partial["fixed"] == ["a"] and partial["broken"] == []
    assert partial["down"] == ["c"] and partial["deltas"]["c"] == -0.3


def test_graded_changes_list_the_largest_moves_each_way():
    from refactory_dsh.optim.base import edit_change

    parent = [{"id": f"i{k}", "passed": False, "score": 0.5} for k in range(8)]
    child = [{"id": f"i{k}", "passed": False, "score": 0.5 + (k - 4) / 10} for k in range(8)]
    change = edit_change(parent, child)
    assert change["up"] == ["i7", "i6", "i5"] and change["down"] == ["i0", "i1", "i2"]


def test_the_comparison_stays_within_its_budget():
    from refactory_dsh.optim.base import EDIT_CHANGE_BUDGET, _edit_change_lines

    def big(item: str, passed: bool) -> dict[str, Any]:
        return {"id": item, "passed": passed, "score": float(passed),
                "exchanges": [{"node": "s", "prompt": "p", "response": "z" * 9000}]}

    pair = {"child": "c", "parent": "p", "edit": [],
            "parent_records": [big(f"i{k}", True) for k in range(20)],
            "child_records": [big(f"i{k}", False) for k in range(20)]}
    text = "\n".join(_edit_change_lines([pair]))
    assert len(text) <= EDIT_CHANGE_BUDGET + 3000
    assert "The other changed items (their trajectories did not fit)" in text
    assert "| i19 | broken |" in text


def test_the_optimizer_sees_its_earlier_reasoning_beside_what_happened(tmp_path, monkeypatch):
    """The line to the edited candidate and the latest rejected child, each with
    the rationale, prediction and diagnosis it was proposed with, the gate's
    decision, train score and fixed/broken counts; kept across a resume."""
    loop, proposer = _run_reasoning(tmp_path, monkeypatch)
    text = proposer.contexts[2].render()
    section = text[text.index("## Your earlier reasoning"):]
    ids = {_level(c): c.id for c in loop.population}
    assert f"### {ids[7]} (proposed in epoch 1) -- gate current" in section
    assert f"### {ids[6]} (proposed in epoch 2) -- gate rejected" in section
    for marker in ("RATIONALE-1", "PREDICTION-1", "DIAGNOSIS-1", "RATIONALE-2", "DIAGNOSIS-2"):
        assert marker in section, marker
    assert "1 broken vs its parent" in section and "0 fixed" in section

    again = Reasoning([[]])
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=4, val_gate=True, resume=True)
        resumed.optimizer = again
        resumed.run()
    after = again.contexts[0].render()
    assert "RATIONALE-1" in after and "DIAGNOSIS-2" in after, "kept across a resume"


def test_a_prior_gated_runs_val_score_never_reaches_the_prompt(tmp_path, monkeypatch):
    """A sibling run with the validation gate reports its winner's val score as
    best_score; the next run's "Previous runs" shows that run's train score instead."""
    import json

    first, _ = _run(tmp_path, monkeypatch, [[(5, 7)], [], []], val_gate=True,
                    runs_root=tmp_path / "runs" / "a")
    report = json.loads((first.root / "report.json").read_text())
    assert report["best_score_basis"].startswith("val"), "setup: a gated run"
    val_best, train_best = report["best_score"], report["best_train_score"]
    assert val_best != train_best
    second, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [], []],
                            val_gate=True, epochs=1, runs_root=tmp_path / "runs" / "b")
    text = proposer.contexts[0].render()
    previous = text[text.index("## Previous runs of this mode"):]
    previous = previous[:previous.index("\n## ", 5)]
    assert str(val_best) not in previous and f"{val_best:.6g}" not in previous
    assert str(train_best) in previous


def test_each_proposal_keeps_the_diagnosis_of_the_call_that_made_it(tmp_path, monkeypatch):
    """The first call's proposal is accepted; a retry for the rest of the batch says
    something else. The accepted candidate keeps the first call's diagnosis."""
    from refactory_dsh.loop import Loop
    from refactory_dsh.optim import Optimizer, Proposal

    class TwoCalls(Optimizer):
        name = "two_calls"

        def __init__(self) -> None:
            super().__init__(k=1)
            self.calls = 0

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            self.calls += 1
            self.last_diagnosis = f"DIAGNOSIS-{self.calls}"
            if self.calls == 1:
                return [Proposal(edit={"op": "set_knob", "name": "level", "value": 7},
                                 parent=parents[0].id),
                        Proposal(edit={"op": "no_such_op"}, parent=parents[0].id)]
            return [Proposal(edit={"op": "no_such_op_again"}, parent=parents[0].id)]

    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    proposer = TwoCalls()
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=1, batch_size=2, attempts=2)
        loop.optimizer = proposer
        loop.run()
    assert proposer.calls >= 2, "setup: a retry happened"
    seven = next(c for c in loop.population if _level(c) == 7)
    assert loop.proposals[seven.id]["diagnosis"] == "DIAGNOSIS-1"


def test_the_comparison_heading_says_children(tmp_path, monkeypatch):
    _loop_, proposer = _run_reasoning(tmp_path, monkeypatch)
    assert "The latest children of the candidate you edit (its last batch)" in proposer.contexts[2].render()
