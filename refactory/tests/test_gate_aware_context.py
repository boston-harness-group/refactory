"""With ``val_gate``, the optimizer reads what the gate decided, never a val score.

Every row the optimizer sees (recent results, the history, the gradient's
leaderboard) and the per-knob attribution carry the gate's decision -- current, kept,
rejected or not gated -- and the request says the current candidate is what it edits
and a rejected one was not shown better on validation. A duplicate proposal is told
which candidate it repeats. Without ``val_gate`` nothing changes. Each test fails when the behaviour it
names is reverted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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
        # Without val_gate given, the run selects on train (the manifest has val items,
        # which would otherwise turn the gate on).
        loop = _loop(tmp_path, **{"epochs": 3, "val_gate": False, **overrides})
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
    assert "`rejected` candidate was not shown better on validation" in text
    assert "did not generalise" not in text
    # What this run has tried: one line per attempt, ending with the gate's decision.
    tried = text[text.index("## What this run has already tried"):]
    row6 = next(line for line in tried.splitlines() if line.startswith(f"- `{history[6]['id']}`"))
    assert row6.rstrip().endswith("| gate rejected")
    # The tried-line gives the current candidate's train score, labelled as train.
    assert f"train score of the validation gate's current candidate {history[7]['score']:.6g}" in text
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
    train items, t5 broken in one line with both outcomes, the edit, the gate's decision
    and the noise warning; no val or test item appears."""
    loop, proposer = _run_reasoning(tmp_path, monkeypatch)
    text = proposer.contexts[2].render()
    section = text[text.index("## What the last edit changed"):text.index("\n## ", text.index("## What the last edit changed") + 5)]
    ids = {_level(c): c.id for c in loop.population}
    assert f"### Child {ids[6]} of {ids[7]} -- gate: rejected" in section
    assert "On 8 shared train item(s): 0 fixed (parent wrong, child right), 1 broken" in section
    line = next(line for line in section.splitlines()
                if line.startswith(f"- Item t5 -- broken (child {ids[6]})"))
    assert "parent passed" in line and "child answered wrong" in line, line
    assert 'name="level" -- value: 6' in section and "different seeds" in section
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


def test_every_changed_item_is_named_in_one_line_and_no_reply_is_repeated():
    """Twenty broken items with long replies: every one is named in one line, with its
    answers and the step whose reply changed; no reply is shown here (the items' traces
    are under "Trajectories")."""
    from refactory_dsh.optim.base import _edit_change_lines

    def rec(item: str, passed: bool, reply: str) -> dict[str, Any]:
        return {"id": item, "passed": passed, "score": float(passed),
                "metadata": {"question": f"QUESTION {item}"},
                "exchanges": [{"node": "s", "prompt": "p", "response": f"KEPT {item}"},
                              {"node": "t", "prompt": "p", "response": reply}]}

    pair = {"child": "c", "parent": "p", "edit": [],
            "parent_records": [rec(f"i{k}", True, f"OLD-{k} " + "z" * 9000) for k in range(20)],
            "child_records": [rec(f"i{k}", False, f"NEW-{k} " + "y" * 9000) for k in range(20)]}
    text = "\n".join(_edit_change_lines([pair]))
    for k in range(20):
        assert (f"- Item i{k} -- broken (child c), score 1 -> 0: parent passed; child "
                "answered wrong; `t` 1 -> 1 run(s), reply changed.") in text
        assert f"OLD-{k}" not in text and f"NEW-{k}" not in text and f"KEPT i{k}" not in text
    assert "#### Item" not in text and len(text) < 5000

def test_the_optimizer_sees_its_earlier_reasoning_beside_what_happened(tmp_path, monkeypatch):
    """The latest round's proposals in full -- the rationale, prediction and diagnosis
    each was proposed with, the gate's decision, train score and fixed/broken counts --
    and an older round's as one line with its rationale (memory summarises the rest);
    kept across a resume."""
    loop, proposer = _run_reasoning(tmp_path, monkeypatch)
    text = proposer.contexts[2].render()
    section = text[text.index("## Your earlier reasoning"):]
    ids = {_level(c): c.id for c in loop.population}
    assert f"### {ids[6]} (proposed in epoch 2) -- gate rejected" in section
    assert f"### {ids[7]} " not in section, "an older round is one line, not a block"
    for marker in ("RATIONALE-2", "DIAGNOSIS-2", "RATIONALE-1"):
        assert marker in section, marker
    # A child of the last round has its prediction under "What the last edit changed",
    # once; the reasoning points there.
    assert "PREDICTION-2" in text
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
    assert "The latest children of the candidate you edit (every child of its last round, siblings included)" in proposer.contexts[2].render()


def test_the_gate_note_states_its_resolution_and_when_a_rejected_change_may_be_built_on():
    """A rejection is "not shown better on validation", with what one validation item
    is worth when the val size is known; a rejected change is never re-proposed
    unchanged, but one with strongly positive train fixed/broken counts may be built
    on or combined with edits to other steps of the same failure path."""
    from refactory_dsh.optim import ProposalContext

    rows = [{"id": "a", "parent": None, "score": 0.5, "gate": "current"},
            {"id": "b", "parent": "a", "score": 0.6, "gate": "rejected"}]
    known = ProposalContext(history=rows, leaderboard=rows[:1], val_items=25).render()
    assert "`rejected` candidate was not shown better on validation" in known
    assert ("the mean over 25 validation items, so one item is 1/25 = 0.04 of it, and a "
            "rejection can come down to a single item") in known
    assert "must not be proposed again unchanged" in known
    # Building on a rejected change is about a fixable mechanism, not its net count.
    assert ("A rejected change can hold the right idea with a fixable mechanism, whatever its "
            "net count") in known
    assert "strongly positive" not in known
    assert "did not generalise" not in known and "do not build on" not in known
    unknown = ProposalContext(history=rows, leaderboard=rows[:1]).render()
    assert "not shown better on validation" in unknown and "validation items, so one" not in unknown
    assert "not shown better" not in ProposalContext(history=[{"id": "a", "score": 0.5}]).render()


@pytest.mark.parametrize("mean", [False, True])
def test_the_loop_hands_the_val_size_when_the_score_is_the_items_mean(tmp_path, monkeypatch, mean):
    """Under ``val_gate`` the request gets the val part's size, but only when every
    score is the items' mean; otherwise (and without ``val_gate``) none."""
    from refactory_dsh.loop import Loop
    from test_skillopt import VAL_IDS

    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        scores = [r["score"] for r in result.item_records]
        result.score = sum(scores) / len(scores) + (0.0 if mean else 0.05)
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)
    loop, proposer = _run(tmp_path / "gated", monkeypatch, [[(5, 7)], []], val_gate=True, epochs=2)
    ctx = proposer.contexts[-1]
    assert ctx.val_items == (len(VAL_IDS) if mean else 0)
    assert (f"1/{len(VAL_IDS)}" in ctx.render()) is mean
    loop, proposer = _run(tmp_path / "plain", monkeypatch, [[(5, 7)], []], epochs=2)
    assert proposer.contexts[-1].val_items == 0


def test_the_loop_hands_the_seeds_knob_values(tmp_path, monkeypatch):
    """The history's set_knob diffs start from the seed's values, which the loop hands
    over from the population's root candidate."""
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], []], epochs=2)
    root = next(c for c in loop.population if not c.parent)
    ctx = proposer.contexts[-1]
    assert ctx.seed_knob_values and ctx.seed_knob_values == {
        name: entry["value"] for name, entry in root.state["knobs"].items()}


def test_the_latest_rejected_candidate_is_shown_in_full_whatever_its_net():
    """Under "Your earlier reasoning", the full edits of the most recent rejected
    candidate with train counts, whatever its net (a change that broke more than it
    fixed can hold a fixable mechanism); not those of an older one or a kept one. Its
    texts are whole: a new step's instructions are never shortened."""
    from refactory_dsh.optim import ProposalContext
    from refactory_dsh.optim.base import REJECTED_EDITS_CHARS

    def entry(cid, fixed, broken, value, gate="rejected"):
        return {"id": cid, "epoch": 1, "gate": gate, "score": 0.5, "fixed": fixed,
                "broken": broken, "rationale": "r", "prediction": "p", "diagnosis": "d",
                "edit": [{"op": "set_knob", "name": "k", "value": value}]}

    long = "HEAD " + "y" * 1500 + " TAIL"
    entries = [entry("old", 5, 1, "OLDVALUE " + "o" * 900), entry("even", 2, 2, "E" * 900),
               entry("mine", 1, 7, long),
               entry("kept", 9, 0, "KEPTVALUE " + "k" * 900, gate="kept")]
    text = ProposalContext(reasoning=entries).render()
    section = text[text.index("## Your earlier reasoning"):]
    full = section[section.index("### Full edits of"):]
    assert full.startswith("### Full edits of mine, the latest rejected candidate "
                           "(1 fixed / 7 broken)")
    # Its edits are whole once, in its own block, which the full-edits heading points to.
    assert text.count(long) == 1 and long not in full and "Whole in its block above" in full
    assert "OLDVALUE" not in full and "KEPTVALUE" not in full
    huge = [entry("big", 3, 0, "Z" * 9000)]
    big = ProposalContext(reasoning=huge).render()
    assert big.count("Z" * 9000) == 1, "whole once, never cut"
    # One not shown in a block of its own keeps the bounded full edits.
    hidden = [{**entry("hid", 3, 0, "Z" * 9000), "epoch": 1},
              *({**entry(f"n{i}", 0, 0, "v", gate="kept"), "epoch": 2} for i in range(2))]
    block = ProposalContext(reasoning=hidden).render()
    block = block[block.index("### Full edits of hid"):].split("\n## ", 1)[0]
    assert "Z" * 9000 in block, "a new step's or knob's text is whole, not shortened"
    assert "its middle is cut" not in block
    # Past the generous bound the middle is cut, and said.
    vast = [{**entry("vast", 3, 0, "W" * (REJECTED_EDITS_CHARS + 5000)), "epoch": 1},
            *({**entry(f"n{i}", 0, 0, "v", gate="kept"), "epoch": 2} for i in range(2))]
    cut = ProposalContext(reasoning=vast).render()
    cut = cut[cut.index("### Full edits of vast"):].split("\n## ", 1)[0]
    assert "have their middle cut, each cut said in it; none is left out" in cut
    assert "characters omitted" in cut
    assert len(cut) < REJECTED_EDITS_CHARS + 1200
    none = [entry("y", None, None, "w")]
    assert "### Full edits of" not in ProposalContext(reasoning=none).render()


def test_the_latest_rejected_change_says_what_went_wrong_in_it():
    """A rejected structural change's block says how it moved the items it changed
    (`optim.edit_effect`), the mechanism to fix, and points to its step's review."""
    from refactory_dsh.optim import ProposalContext

    from test_edit_effect import _tried

    # The real S1-s45 loop's records (`fixtures/edit_effect`): 8 broken, 1 fixed.
    effect = _tried("s1_s45_loop.json")[0]["effect"]
    rows = [{"id": "loopy", "epoch": 1, "gate": "rejected", "score": 0.4, "fixed": 1,
             "broken": 7, "rationale": "r", "prediction": "p", "diagnosis": "d",
             "edit": [{"op": "loop", "step": "solve", "judge": "agent",
                       "text": "CHECK-CRITERIA", "max_iterations": 1}],
             "effect": effect, "added": True},
            *({"id": f"n{i}", "epoch": 2, "gate": "kept", "score": 0.5, "fixed": 0,
               "broken": 0, "edit": [{"op": "set_knob", "name": "k", "value": "v"}]}
              for i in range(2))]
    text = ProposalContext(reasoning=rows, added_steps=[{"candidate": "loopy", "step": "j"}]
                           ).render()
    block = text[text.index("### Full edits of loopy"):].split("\n## ", 1)[0]
    assert "CHECK-CRITERIA" in block
    assert "What went wrong in it (how it moved the items it changed):" in block
    assert "gate solve_check gave" in block and "first reply was already right" in block
    assert 'reviewed under "Did the added steps contribute?"' in block
    # Pointed to only when that section reviews this candidate's step.
    other = ProposalContext(reasoning=rows, added_steps=[{"candidate": "else", "step": "j"}]
                            ).render()
    other = other[other.index("### Full edits of loopy"):].split("\n## ", 1)[0]
    assert "Did the added steps contribute?" not in other
    # A text-only rejected change says what there is: its counts, and that no step-level
    # account is available.
    knob = [{**rows[0], "id": "words", "effect": {}, "added": False,
             "edit": [{"op": "set_knob", "name": "k", "value": "NEW"}]}, *rows[1:]]
    words = ProposalContext(reasoning=knob).render()
    words = words[words.index("### Full edits of words"):].split("\n## ", 1)[0]
    assert ("What went wrong in it: it broke 7 train item(s) and fixed 1 against its parent; "
            "no step-by-step account") in words


#: A child whose structural edit made items fail mechanically (LiveMath B-s42, round 2):
#: a verdict gate that could not read its reviewer's answer, wrapped as the record has it.
_SNIPPET_ERROR = ("meta_gate: SnippetError: verdict gate 'meta_gate': the first word of file "
                  "'meta_review.txt' ('Looking at this carefully:') is none of its answers "
                  "['proceed', 'reloop', 'halt'] (its out-edge conditions, or halt) " + "x" * 400)


def _errored(item: str, text: str, kind: str = "RuntimeError") -> dict[str, Any]:
    return {"id": item, "passed": False, "score": 0.0, "error_text": text, "error_kind": kind}


def test_item_errors_count_each_kind_at_its_node_with_its_first_message():
    from refactory_dsh.optim.base import ITEM_ERROR_CHARS, item_errors

    records = [_errored("t1", _SNIPPET_ERROR), {"id": "t2", "passed": True, "score": 1.0},
               _errored("t3", _SNIPPET_ERROR.replace("Looking", "Reading")),
               _errored("t4", "dsh run failed (exit 1): boom", kind="RuntimeError"),
               _errored("t5", "decide: CompletionTimeout: model completion timed out", "timeout"),
               {"id": "t6", "passed": False, "score": 0.0, "fail_reason": "wrong letter"}]
    errors = item_errors(records)
    assert [(e["kind"], e["node"], e["count"], e["item"]) for e in errors] == [
        ("SnippetError", "meta_gate", 2, "t1"), ("RuntimeError", "", 1, "t4"),
        ("CompletionTimeout", "decide", 1, "t5")]
    first = errors[0]["first"]
    assert first.startswith("verdict gate 'meta_gate': the first word") and "Looking" in first
    assert len(first) <= ITEM_ERROR_CHARS
    assert errors[1]["first"] == "dsh run failed (exit 1): boom"
    assert item_errors([{"id": "a", "passed": False, "score": 0.0}]) == []


@pytest.mark.parametrize("text, origin", [
    # The walker's shape, and a fork's nesting of its branch's error.
    ("meta_gate: SnippetError: verdict gate 'meta_gate': x",
     ("SnippetError", "meta_gate", "verdict gate 'meta_gate': x")),
    ("fork1: RuntimeError: inner: SnippetError: y", ("SnippetError", "inner", "y")),
    ("fork1: RuntimeError: branch_b: ValueError: deliberate branch failure",
     ("ValueError", "branch_b", "deliberate branch failure")),
    # A message that starts with its kind names no node, whatever colons follow.
    ("ValueError: Expected: 3 got 4", ("ValueError", "", "Expected: 3 got 4")),
    ("RuntimeError: branch_b: ValueError: x", ("ValueError", "branch_b", "x")),
    # A node's failure with no exception kind: the node, and the record's kind.
    ("decide: worker: model completion timed out", ("timeout", "decide", "worker: model completion timed out")),
    ("dsh run failed (exit 1): boom", ("timeout", "", "dsh run failed (exit 1): boom")),
    # The spine's record (`workflow-spine` `nodeErrorText`): the attribution line, then
    # the node's stderr tail as before -- log lines, the error, the runner's marker and
    # its account. Only the first line is read.
    ("g: SnippetError: verdict gate 'g': bad\nloading notes.txt\n--- refactory failure ---\n"
     "SnippetError: verdict gate 'g': bad",
     ("SnippetError", "g", "verdict gate 'g': bad")),
    ("decide: CompletionTimeout: model completion timed out after 60s\nworker: model "
     "completion timed out after 60s", ("CompletionTimeout", "decide", "model completion timed out after 60s")),
    # An older spine record: the raw tail, read from the runner's account after its marker.
    ("WARNING: slow\nSnippetError: verdict gate 'g': bad\n--- refactory failure ---\n"
     "SnippetError: verdict gate 'g': bad", ("SnippetError", "", "verdict gate 'g': bad")),
    # A multi-line message: the summary is its first line.
    ("check: SnippetError: first line\nsecond line", ("SnippetError", "check", "first line")),
])
def test_an_item_errors_node_and_kind_are_its_innermost(text, origin):
    from refactory_dsh.optim.base import item_errors

    (error,) = item_errors([_errored("t1", text, kind="timeout")])
    assert (error["kind"], error["node"], error["first"]) == origin


def test_a_failed_items_record_keeps_the_spines_error_line_and_its_whole_tail():
    """The spine sends the line naming the failed node, then the node's 4096-character
    stderr tail as before (`nodeErrorText`); the record keeps all of it."""
    from refactory_dsh import items

    tail = "loading notes.txt\n" + "y" * 4096
    text = "work: SnippetError: verdict gate check: no answer\n" + tail[-4096:]
    record = items.make_item_record("i1", None, RuntimeError(text))
    assert record["error_text"] == text
    assert record["error_text"].endswith("y" * 4096)


def test_the_last_edit_section_shows_a_childs_item_errors():
    """A child whose errored items were wrong in the parent too is neither fixed nor
    broken, so before this its errors never reached the request."""
    from refactory_dsh.optim.base import _edit_change_lines

    parent = [{"id": f"t{k}", "passed": k == 0, "score": float(k == 0)} for k in range(4)]
    child = [{"id": "t0", "passed": True, "score": 1.0},
             *[_errored(f"t{k}", _SNIPPET_ERROR) for k in range(1, 4)]]
    text = "\n".join(_edit_change_lines([{"child": "c", "parent": "p", "edit": [],
                                           "parent_records": parent, "child_records": child}]))
    assert "0 fixed (parent wrong, child right), 0 broken" in text, "setup: no item changed"
    assert "Item errors: 3 of 4 train item(s) failed with an error" in text
    assert "- 3 x SnippetError at node meta_gate; first (item t1): verdict gate 'meta_gate'" in text
    assert text.count("SnippetError") == 1, "one line per kind, not one per item"
    clean = "\n".join(_edit_change_lines([{"child": "c", "parent": "p", "edit": [],
                                            "parent_records": parent, "child_records": parent}]))
    assert "Item errors" not in clean


def test_the_reasoning_section_shows_a_rejected_childs_item_errors(tmp_path, monkeypatch):
    """The rejected child 6 had items fail with a SnippetError: its entry in "Your
    earlier reasoning" counts them by kind at their node, with the first message, next
    to its train score and fixed/broken counts; the current candidate 7 shows none."""
    from refactory_dsh.optim.base import _reasoning_lines

    loop, _proposer = _run_reasoning(tmp_path, monkeypatch)
    ids = {_level(c): c.id for c in loop.population}
    records = loop._train_item_records[ids[6]]
    records[1:3] = [_errored(r["id"], _SNIPPET_ERROR) for r in records[1:3]]
    seven = next(c for c in loop.population if c.id == ids[7])
    loop._editing_parent = ids[7]
    entries = {e["id"]: e for e in loop._reasoning_entries([seven])}
    assert [(e["kind"], e["node"], e["count"]) for e in entries[ids[6]]["errors"]] == [
        ("SnippetError", "meta_gate", 2)]
    assert entries[ids[7]]["errors"] == []
    text = "\n".join(_reasoning_lines(list(entries.values())))
    block = text[text.index(f"### {ids[6]} "):]
    assert block.index("Item errors: 2 of") < block.index("Edits:")
    assert "- 2 x SnippetError at node meta_gate; first (item " in block
    assert text.count("Item errors") == 1, "the current candidate 7 shows none"


def test_a_rationale_shown_in_full_is_not_repeated_in_every_change_so_far(tmp_path, monkeypatch):
    """The latest round's rationale is in full under "Your earlier reasoning"; its row in
    "Every change so far" points there; an older round's row keeps its own."""
    _loop_, proposer = _run_reasoning(tmp_path, monkeypatch)
    text = proposer.contexts[2].render()
    assert text.count("RATIONALE-2") == 1
    ledger = text[text.index("## Every change so far"):].split("\n## ", 1)[0]
    assert '(in full under "Your earlier reasoning")' in ledger and "RATIONALE-1" in ledger


