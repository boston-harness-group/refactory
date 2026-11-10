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
    assert "`rejected` candidate was not shown better on validation" in text
    assert "did not generalise" not in text
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
    assert ("When its train fixed/broken counts against its parent (when shown below) were "
            "strongly positive, its direction may still be right: you may build on its change "
            "or combine it with edits to other steps in the same failure path") in known
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


def test_the_latest_train_positive_rejected_candidate_is_shown_in_full():
    """Under "Your earlier reasoning", the full edits of the most recent rejected
    candidate with more fixed than broken train items, within the cap; not those of an
    older one, nor of one that broke as many as it fixed."""
    from refactory_dsh.optim import ProposalContext
    from refactory_dsh.optim.base import REJECTED_EDITS_CHARS

    def entry(cid, fixed, broken, value, gate="rejected"):
        return {"id": cid, "epoch": 1, "gate": gate, "score": 0.5, "fixed": fixed,
                "broken": broken, "rationale": "r", "prediction": "p", "diagnosis": "d",
                "edit": [{"op": "set_knob", "name": "k", "value": value}]}

    long = "HEAD " + "y" * 1500 + " TAIL"
    entries = [entry("old", 5, 1, "OLDVALUE " + "o" * 900), entry("mine", 8, 1, long),
               entry("even", 2, 2, "EVENVALUE " + "e" * 900),
               entry("kept", 9, 0, "KEPTVALUE " + "k" * 900, gate="kept")]
    text = ProposalContext(reasoning=entries).render()
    section = text[text.index("## Your earlier reasoning"):]
    full = section[section.index("### Full edits of"):]
    assert full.startswith("### Full edits of mine, the latest rejected candidate that fixed "
                           "more train items than it broke (8 fixed / 1 broken)")
    assert long in full and "OLDVALUE" not in full and "EVENVALUE" not in full
    assert "KEPTVALUE" not in full
    huge = [entry("big", 3, 0, "Z" * 9000)]
    big = ProposalContext(reasoning=huge).render()
    block = big[big.index("### Full edits of big"):].split("\n## ", 1)[0]
    assert "Z" * 9000 not in block and "more chars)" in block
    assert len(block) < REJECTED_EDITS_CHARS + 600
    none = [entry("x", 1, 3, "v"), entry("y", None, None, "w")]
    assert "### Full edits of" not in ProposalContext(reasoning=none).render()


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
    # A raw stderr tail from the node runner (`refactory_dsh.run.main`): the text after
    # its marker, once; the warning and the first copy before it are dropped.
    ("WARNING: slow\nSnippetError: verdict gate 'g': bad\n--- refactory failure ---\n"
     "SnippetError: verdict gate 'g': bad", ("SnippetError", "", "verdict gate 'g': bad")),
    ("g: SnippetError: verdict gate 'g': bad\n--- refactory failure ---\n", ("SnippetError", "g", "verdict gate 'g': bad")),
])
def test_an_item_errors_node_and_kind_are_its_innermost(text, origin):
    from refactory_dsh.optim.base import item_errors

    (error,) = item_errors([_errored("t1", text, kind="timeout")])
    assert (error["kind"], error["node"], error["first"]) == origin


def test_the_spine_runners_failure_text_is_attributed_to_its_node(tmp_path, monkeypatch, capsys):
    """The spine records a failed node's error as its id, then the node runner's text
    after its failure marker (`workflow-spine` `nodeErrorText`). Made here from the real
    runner's stderr for a verdict gate that cannot read its answer."""
    import json as json_mod

    from refactory_dsh import lowering, run as run_mod, snippet
    from refactory_dsh.optim.base import _FAILURE_MARKER, item_errors
    from refactory_dsh.run import FAILURE_MARKER
    from test_code_nodes import CRITIC_LOOP, _accepted

    assert _FAILURE_MARKER == FAILURE_MARKER
    lowering.write_declarations(_accepted(CRITIC_LOOP), tmp_path)
    (tmp_path / "critic.txt").write_text("Looking at this carefully: fine.\nNo obvious omission.")
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json_mod.dumps({"id": "check", "type": "gate"}))
    monkeypatch.setenv("REFACTORY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    assert run_mod.main([snippet.VERDICT]) == 1
    stderr = capsys.readouterr().err
    assert FAILURE_MARKER in stderr and stderr.count("SnippetError") == 2, "setup: the runner's shape"
    spine_text = "check: " + stderr.rpartition(FAILURE_MARKER)[2].strip()
    for text in (spine_text, stderr):
        (error,) = item_errors([_errored("t1", text)])
        assert error["kind"] == "SnippetError"
        assert error["first"].startswith("verdict gate 'check': neither the first word"), error
        assert FAILURE_MARKER not in error["first"]
    assert item_errors([_errored("t1", spine_text)])[0]["node"] == "check"


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
    seven_block = text[text.index(f"### {ids[7]} "):text.index(f"### {ids[6]} ")]
    assert "Item errors" not in seven_block
