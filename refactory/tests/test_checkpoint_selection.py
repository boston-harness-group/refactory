"""``selection: checkpoint``: the outer loop trained as an ML model is (``LoopConfig.selection``).

Each round's change is applied unless its graph is broken; only the applied candidate is
scored on val, once (a checkpoint); the winner is the best checkpoint on val; no parent
is re-scored, as train rollouts use one seed for the run. The smoke check (``E-smoke``)
and the item history run under every default-loop selection. Each test fails when the
behaviour it names is reverted. Every test runs on the walker but the one marked
[spine], which runs the smoke check on the TS spine through the lowering.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import adopt, nn, strategies
from refactory_dsh.loop import Loop, LoopConfig, LoopError, default_selection
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.optim import analysts, base
from refactory_dsh.optim.evidence import item_history
from refactory_dsh.outerloop import OuterLoop, ValidationGate, train_batch_seed

from test_skillopt import MODE_ID, PASS_IDS_ENV, _patch_mode, _write_parts
from test_validation_selection import FitProbe, Scripted, _level, _scores

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
NODE = shutil.which("node")
DRAIN_AGENT = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
PYTHON_BIN = str(REPO_ROOT / ".venv/bin")
SEED = 7


def _config(tmp_path: Path, *, test: bool = False, **overrides: Any) -> LoopConfig:
    base_settings: dict[str, Any] = dict(
        epochs=2, batch_size=1, optimizer="random", seed=SEED, runs_root=tmp_path / "run",
        allowed_ops=("set_knob",), max_evaluations=200, holdout=0.0,
        parts_manifest=str(_write_parts(tmp_path, test=test)))
    base_settings.update(overrides)
    return LoopConfig(**base_settings)


def _run(tmp_path: Path, monkeypatch: Any, plan: list[list[tuple[int, int]]], *,
         model: bool = False, test: bool = False, proposer: Any = None, **overrides: Any):
    """The default loop (selection left to its default) on the toy mode, one scripted
    round per epoch."""
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    proposer = proposer or Scripted(plan, model=model)
    with _patch_mode(FitProbe):
        loop = Loop(MODE_ID, _config(tmp_path, test=test, **overrides), project=Path("."))
        loop.optimizer = proposer
        report = loop.run()
    return loop, proposer, report


def _events(loop: Loop, name: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (loop.root / "journal.jsonl").read_text().splitlines()
            if line.strip()]
    return [row for row in rows if row.get("event") == name]


def _by_level(loop: Loop) -> dict[int, str]:
    return {_level(c): c.id for c in loop.population}


def _levels(loop: Loop, ids: Any) -> list[int]:
    by_id = {c.id: c for c in loop.population}
    return [_level(by_id[cid]) for cid in ids]


def _vals(loop: Loop) -> Counter[int]:
    """Val rollouts per candidate level."""
    return Counter(_levels(loop, [r["id"] for r in _events(loop, "part_result")
                                  if r["part"] == "val"]))


# ── always apply, one val run per checkpoint, the winner ─────────────────


class Watching(Scripted):
    """Records the candidate each request is about (``ProposalContext.parent``)."""

    def __init__(self, plan):  # type: ignore[no-untyped-def]
        super().__init__(plan)
        self.edited: list[str | None] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.edited.append(ctx.parent)
        return super().propose(ctx, n, parents)


def test_checkpoint_is_the_default_and_applies_a_child_with_a_lower_val_score(
        tmp_path, monkeypatch):
    """The seed scores 0.7 on val, its child 0.5: the child is applied anyway (it becomes
    current, the next parent and the candidate the request is about), and so is its child
    (0.6). The best checkpoint, the seed, stays the winner: the report, the leaderboard
    (the best first, then by val) and the gate agree."""
    _scores(monkeypatch, {0: 0.5, 1: 0.4, 2: 0.6}, {0: 0.7, 1: 0.5, 2: 0.6})
    loop, proposer, report = _run(tmp_path, monkeypatch, [],
                                  proposer=Watching([[(0, 1)], [(1, 2)]]))
    assert loop.config.selection == "checkpoint" and loop.selection_mode() == "checkpoint"
    assert proposer.handed == [[0], [1]], "the applied child is the next round's parent"
    assert _levels(loop, proposer.edited) == [0, 1], "the request is about the current one"
    assert _levels(loop, loop._gate_accepted) == [0, 1, 2] and not loop._gate_rejected
    gate = loop._gate()
    assert _levels(loop, [gate.current_id, gate.best_id]) == [2, 0]
    assert report["best"]["id"] == gate.best_id == loop.population[0].id
    assert _levels(loop, [c.id for c in loop.leaderboard()]) == [0, 2, 1]
    assert report["best_score"] == pytest.approx(0.7)
    assert [e["applied"] for e in _events(loop, "checkpoint")] == [
        _by_level(loop)[1], _by_level(loop)[2]]


def test_each_graph_is_scored_once_and_only_the_applied_child_gets_a_val_run(
        tmp_path, monkeypatch):
    """Two children a round: the one with the higher train score is applied (1, then 3),
    even when its sibling scores higher on val (2 has 0.9); only it is rolled out on val.
    No graph is scored twice: one train score each, no re-score, one val run per
    checkpoint; the sibling is `not applied`."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.4, 3: 0.7, 4: 0.2},
            {0: 0.5, 1: 0.6, 2: 0.9, 3: 0.4, 4: 0.8})
    loop, _, _ = _run(tmp_path, monkeypatch, [[(0, 1), (0, 2)], [(1, 3), (1, 4)]],
                      batch_size=2)
    assert Counter(_levels(loop, [r["id"] for r in _events(loop, "score")])) == {
        0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    assert not _events(loop, "rescore") and not _events(loop, "rescore_failed")
    assert _vals(loop) == {0: 1, 1: 1, 3: 1}
    assert _levels(loop, loop._gate_accepted) == [0, 1, 3]
    assert _levels(loop, loop._gate_rejected) == [2, 4]
    words = {level: loop._gate_decision(cid) for level, cid in _by_level(loop).items()}
    assert words == {0: "kept", 1: "kept", 2: "not applied", 3: "current", 4: "not applied"}
    assert _levels(loop, [loop._gate().best_id]) == [1], "the best checkpoint: 0.6"


def _rollouts_per_epoch(loop: Loop) -> list[int]:
    """The rollouts each epoch added, then those after the last epoch's end."""
    totals, latest = [], 0
    for line in (loop.root / "journal.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "rollouts":
            latest = int(row["rollouts"])
        elif row.get("event") == "epoch_end":
            totals.append(latest)
    totals.append(loop.rollouts)
    return [b - a for a, b in zip([0, *totals], totals)]


def test_a_round_costs_its_childs_train_and_val_runs_and_two_smoke_items(
        tmp_path, monkeypatch):
    """8 train and 4 val items, one child a round, the smoke check on 2 train items (one
    the parent answered wrong, one it passed). A checkpoint round is 8 + 4 + 2 = 14 runs;
    under single the parent's re-score adds 8. The last proposals cost 8 + 4."""
    monkeypatch.setenv(PASS_IDS_ENV, "t2")
    _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.8}, {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.8})
    plan = [[(0, 1)], [(1, 2)], [(2, 3)]]
    checkpoint, _, _ = _run(tmp_path / "checkpoint", monkeypatch, plan, epochs=3)
    single, _, _ = _run(tmp_path / "single", monkeypatch, plan, epochs=3, selection="single")
    assert _rollouts_per_epoch(checkpoint) == [14, 14, 14, 12]
    assert _rollouts_per_epoch(single) == [14, 22, 22, 12]
    assert [len(e["items"]) for e in _events(checkpoint, "smoke")] == [2, 2, 2]


def test_one_train_seed_pairs_every_child_with_its_parent(tmp_path, monkeypatch):
    """Every train rollout uses the first epoch's seed, so each child's fixed/broken
    counts against its parent's own run are known (never "n/a, different seed"), in the
    change ledger and in "Your earlier reasoning"."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.7}, {0: 0.5, 1: 0.6, 2: 0.7})
    loop, _, _ = _run(tmp_path, monkeypatch, [[(0, 1)], [(1, 2)]])
    assert {r["seed"] for r in _events(loop, "score")} == {train_batch_seed(SEED, 1)}
    children = [e for e in loop._ledger() if e.get("parent")]
    assert children and all(isinstance(e["fixed"], int) and isinstance(e["broken"], int)
                            and e["same_seed"] for e in children), children
    reasoning = [e for e in loop._reasoning_entries(loop.leaderboard()) if e.get("same_seed")]
    assert reasoning and all(isinstance(e["fixed"], int) for e in reasoning)


@pytest.mark.parametrize("stop", ["patience", "perfect_val"])
def test_the_early_stops_count_rounds_without_a_new_best(tmp_path, monkeypatch, stop):
    """Patience 2: two applied children below the seed's val score and the run stops
    (`patience_exhausted`). A child at 1.0 on val stops it at once (`perfect_val`)."""
    if stop == "patience":
        _scores(monkeypatch, {level: 0.5 for level in range(9)},
                {0: 0.7, 1: 0.6, 2: 0.5, 3: 0.4, 4: 0.3})
        loop, _, report = _run(tmp_path, monkeypatch, [[(n, n + 1)] for n in range(6)],
                               epochs=8, early_stop_patience=2)
        assert report["bound"] == "patience_exhausted"
        assert _levels(loop, loop._gate_accepted) == [0, 1, 2]
    else:
        _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.7}, {0: 0.5, 1: 1.0, 2: 0.9})
        loop, _, report = _run(tmp_path, monkeypatch, [[(0, 1)], [(1, 2)]], epochs=5,
                               early_stop_at_full_val=True)
        assert report["bound"] == "perfect_val"
        assert _levels(loop, loop._gate_accepted) == [0, 1]


def test_broken_vetoed_and_failed_children_are_not_applied(tmp_path, monkeypatch):
    """One round of four: a knob value the validator refuses never becomes a candidate,
    nor does one the integrity judge vetoes; a child whose train run fails is evaluated and
    not applied (no val run), so its sibling is, whatever its train score."""
    _scores(monkeypatch, {0: 0.5, 1: 0.3, 2: 0.9, 3: 0.9}, {0: 0.5, 1: 0.6})
    score = FitProbe.score

    def failing(self, run):  # type: ignore[no-untyped-def]
        if int(self.values()["level"]) == 2:
            raise RuntimeError("no score for level 2")
        return score(self, run)

    monkeypatch.setattr(FitProbe, "score", failing)
    check = Loop._integrity_check

    def judge(self, proposal, parent, candidate):  # type: ignore[no-untyped-def]
        if _level(candidate) == 3:
            return ["E-integrity: exploit (grader): scripted veto"]
        return check(self, proposal, parent, candidate)

    monkeypatch.setattr(Loop, "_integrity_check", judge)
    loop, _, _ = _run(tmp_path, monkeypatch, [[(0, 1), (0, 2), (0, 3), (0, 150)]],
                      epochs=1, batch_size=4)
    assert sorted(_by_level(loop)) == [0, 1, 2], "the refused and the vetoed never ran"
    reasons = " | ".join(r for rejection in loop.rejections for r in rejection.reasons)
    assert "E-integrity" in reasons and "150" in reasons
    two = next(c for c in loop.population if _level(c) == 2)
    assert two.evaluated and not two.exit_ok
    assert loop._gate_decision(two.id) == "not applied"
    assert _levels(loop, loop._gate_accepted) == [0, 1]
    assert _vals(loop) == {0: 1, 1: 1}


# ── resume ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("where", ["before the apply", "after the val run"])
def test_a_resume_after_a_kill_rolls_no_val_or_smoke_again(tmp_path, monkeypatch, where):
    """Killed after the child's train run and before its apply, or after its val
    rollout and before the decision: the resumed run applies it with one val run in all,
    and smokes nothing it smoked."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.7}, {0: 0.5, 1: 0.6, 2: 0.7})
    if where == "before the apply":
        target, name = Loop, "_apply_round"
    else:
        target, name = ValidationGate, "offer_best"
    original = getattr(target, name)

    def killed(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt("killed")

    monkeypatch.setattr(target, name, killed)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, monkeypatch, [[(0, 1)]], epochs=2)
    monkeypatch.setattr(target, name, original)
    loop, _, report = _run(tmp_path, monkeypatch, [[(1, 2)]], epochs=2, resume=True)
    assert _vals(loop) == {0: 1, 1: 1, 2: 1}
    assert Counter(_levels(loop, [e["id"] for e in _events(loop, "smoke")])) == {1: 1, 2: 1}
    assert _levels(loop, loop._gate_accepted) == [0, 1, 2]
    assert report["bound"] == "epochs"


def test_a_resumed_run_never_smokes_a_graph_twice(tmp_path, monkeypatch):
    """Killed after the child's smoke check and before it was journaled as a candidate:
    the resumed round proposes it again, and its smoke verdict comes from the journal."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6}, {0: 0.5, 1: 0.6})
    original = Loop._journal_candidate

    def killed(self, candidate):  # type: ignore[no-untyped-def]
        if candidate.parent:
            raise KeyboardInterrupt("killed")
        return original(self, candidate)

    monkeypatch.setattr(Loop, "_journal_candidate", killed)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, monkeypatch, [[(0, 1)]], epochs=1)
    monkeypatch.setattr(Loop, "_journal_candidate", original)
    loop, _, _ = _run(tmp_path, monkeypatch, [[(0, 1)]], epochs=1, resume=True)
    assert 1 in _by_level(loop)
    assert [e["id"] for e in _events(loop, "smoke")] == [_by_level(loop)[1]]


def test_an_old_run_without_a_recorded_selection_resumes_as_single(tmp_path, monkeypatch):
    """A run recorded before checkpoint (its config.json has the gate on and no
    selection) selected by the strict gate: it resumes as single."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6}, {0: 0.5, 1: 0.6})
    loop, _, _ = _run(tmp_path, monkeypatch, [[(0, 1)]], epochs=1, selection="single")
    saved = loop.root / "config.json"
    recorded = json.loads(saved.read_text())
    recorded.pop("selection")
    saved.write_text(json.dumps(recorded))
    resumed, _, report = _run(tmp_path, monkeypatch, [[]], epochs=2, resume=True)
    assert resumed.config.selection == "single" and report["selection"]["setting"] == "single"


# ── the winner and the final evaluation ───────────────────────────────────


@pytest.mark.parametrize("first,second", [(1, 2), (2, 1)])
def test_val_ties_go_to_the_earlier_checkpoint_everywhere(tmp_path, monkeypatch, first,
                                                          second):
    """Two checkpoints tie on val (0.7): the best is the earlier, for the gate, the
    leaderboard, the report and adopt, whichever of the two ids sorts first (both orders
    run). A person rejects it: both fall to the later one."""
    _scores(monkeypatch, {n: 0.5 for n in range(4)}, {0: 0.5, 1: 0.7, 2: 0.7, 3: 0.6})
    loop, _, report = _run(tmp_path, monkeypatch,
                           [[(0, first)], [(first, second)], [(second, 3)]], epochs=3)
    earlier, later = _by_level(loop)[first], _by_level(loop)[second]
    assert loop._gate().best_id == earlier == loop.leaderboard()[0].id == report["best"]["id"]
    assert adopt.best(loop.root)["candidate"] == earlier
    with (loop.root / "journal.jsonl").open("a") as journal:
        journal.write(json.dumps({"event": "control", "kept": [], "rejected": [earlier]})
                      + "\n")
    loop.rejected = [earlier]
    assert loop.leaderboard()[0].id == later == adopt.best(loop.root)["candidate"]


def test_the_final_val_check_reuses_the_checkpoints_val_score(tmp_path, monkeypatch):
    """The final candidate (2, val 0.6) is not the best (1, 0.8): the post-training
    evaluation's final selection is its checkpoint's score, not a second val run."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.7}, {0: 0.5, 1: 0.8, 2: 0.6})
    loop, _, report = _run(tmp_path, monkeypatch, [[(0, 1)], [(1, 2)]], test=True)
    assert _vals(loop) == {0: 1, 1: 1, 2: 1}
    final = [e for e in _events(loop, "final_evaluation")
             if e.get("step_type") == "final_selection_eval"]
    assert [(e["action"], e["score"]) for e in final] == [("reused", 0.6)]
    figures = report["strategy_report"]
    assert figures["test_eval"] == "done" and figures["final_selection"] == pytest.approx(0.6)
    assert report["best"]["id"] == _by_level(loop)[1]


# ── memory, gate words and the request ─────────────────────────────────────


class Remembering(Scripted):
    """A model proposer that writes a memory note, recording each pass's request."""

    def __init__(self, plan):  # type: ignore[no-untyped-def]
        super().__init__(plan, model=True)
        self.builds: list[Any] = []

    def write_memory(self, *, epoch, generation, build):  # type: ignore[no-untyped-def]
        self.builds.append(build)
        return {"note": {"text": f"note of epoch {epoch}", "reasoning": ""}}


def test_the_memory_pass_reads_each_sides_own_train_run(tmp_path, monkeypatch):
    """Epoch 2 compares the seed with the applied child 1 from their own train runs, and
    rolls nothing out. In epoch 3 nothing was applied (2's train run failed) and no
    change moved an item: no call, the note marked not updated."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6, 3: 0.7}, {0: 0.5, 1: 0.6, 3: 0.7})
    score = FitProbe.score

    def failing(self, run):  # type: ignore[no-untyped-def]
        if int(self.values()["level"]) == 2:
            raise RuntimeError("no score for level 2")
        return score(self, run)

    monkeypatch.setattr(FitProbe, "score", failing)
    proposer = Remembering([[(0, 1)], [(1, 2)], [(1, 3)]])
    loop, _, _ = _run(tmp_path, monkeypatch, [], proposer=proposer, epochs=3)
    assert not _events(loop, "memory_rollout")
    passes = _events(loop, "optimizer_memory")
    assert [(p["epoch"], p["updated"], p["evaluated"]) for p in passes] == [
        (2, True, 0), (3, False, 0)]
    assert len(proposer.builds) == 1
    assert proposer.builds[0].prev["rollout"] == "its own train run"
    assert passes[0]["pairs"] == 8, "every train item both ran"


def test_the_request_says_applied_not_rejected(tmp_path, monkeypatch):
    """Round 1's sibling 2 lost on train: the next request's gate column says `current`,
    `kept` and `not applied`, its note says what they mean, and the score section gives
    the checkpoint rule; nothing says a candidate was rejected on validation."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.4, 3: 0.7}, {0: 0.5, 1: 0.6, 2: 0.9, 3: 0.7})
    loop, proposer, _ = _run(tmp_path, monkeypatch, [[(0, 1), (0, 2)], [(1, 3)]],
                             model=True, batch_size=2)
    prompt = proposer.prompts[1]
    assert "`not applied` was not" in prompt and "`current` is the last one applied" in prompt
    assert "Each round's change is applied unless its train run fails" in prompt
    assert "not applied" in prompt.split("## What this run has already tried", 1)[1]
    assert "was not shown better on validation" not in prompt
    assert "keeps a candidate only when that score beats" not in prompt


def test_defaults_and_refusals(tmp_path):
    """Checkpoint is the default for the default loop with a single-parent optimizer; it
    is refused with a population optimizer, a population parent picker or another
    strategy."""
    assert default_selection("default", "random") == "checkpoint"
    assert default_selection("default", "autoresearch", "hill_climb") == "checkpoint"
    assert default_selection("default", "archive") == "pool"
    with _patch_mode(FitProbe):
        with pytest.raises(LoopError, match="checkpoint .* single-parent optimizer"):
            Loop(MODE_ID, _config(tmp_path, optimizer="archive", selection="checkpoint"))
        with pytest.raises(LoopError, match="the islands parent picker"):
            Loop(MODE_ID, _config(tmp_path, parent_picker="islands", selection="checkpoint"))
        with pytest.raises(LoopError, match="the reflective strategy"):
            strategies.create("reflective", MODE_ID, _config(tmp_path, selection="checkpoint"),
                              project=Path("."))


def test_a_custom_strategy_is_not_smoked(tmp_path, monkeypatch):
    """A custom strategy's attempts (the SkillOpt port's) are not smoked; the default
    loop's same attempt is."""
    class OneChild(OuterLoop):
        name = "one_child"

        def strategy(self, run):  # type: ignore[no-untyped-def]
            seed = run.seed()
            run.evaluate(seed)
            child = run.attempt(Proposal(edit={"op": "set_knob", "name": "level", "value": 1},
                                         parent=seed.id), seed)
            assert isinstance(child, Candidate), child
            run.stop("done")

    _scores(monkeypatch, {0: 0.5, 1: 0.6}, {})
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(FitProbe):
        loop = Loop(MODE_ID, _config(tmp_path), project=Path("."))
        strategy = OneChild()
        strategy.configure()
        loop.set_custom_strategy(strategy)
        loop.run()
    assert not _events(loop, "smoke")
    plain, _, _ = _run(tmp_path / "plain", monkeypatch, [[(0, 1)]], epochs=1)
    assert len(_events(plain, "smoke")) == 1


# ── item history ───────────────────────────────────────────────────────────


def test_item_history_format():
    assert item_history([("a", False)]) == "", "one outcome is the one shown"
    assert item_history([("a", False), ("b", False)]) == "history: solved 0/2 in 2 graphs"
    assert item_history([("a", False), ("b", True)]) == "history: solved 1/2 in 2 graphs"
    # Flipped twice in scoring order, or one graph both ways: flaky.
    assert item_history([("a", False), ("b", True), ("c", False)]).endswith(", flaky")
    assert item_history([("a", True), ("a", False)]) == "history: solved 1/2 in 1 graph, flaky"
    longest = item_history([("g%d" % n, n % 2 == 0) for n in range(99)])
    assert len(" — " + longest) <= 45, longest


def _three_rounds(tmp_path: Path, monkeypatch: Any) -> tuple[Loop, Scripted]:
    monkeypatch.setenv(PASS_IDS_ENV, "t2")
    _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.8}, {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.8})
    loop, proposer, _ = _run(tmp_path, monkeypatch, [[(0, 1)], [(1, 2)], [(2, 3)]],
                             model=True, epochs=3)
    return loop, proposer


def _stripped(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "item_history"} for r in records]


def test_the_history_heads_each_failing_item_in_the_request_and_for_the_analysts(
        tmp_path, monkeypatch):
    """By epoch 3 the edited candidate's failing items were answered wrong by all three
    graphs: their headings say so, in the proposal request and in what an analyst reads;
    the passing item t2 says nothing; the request grows by the heading suffix only."""
    loop, proposer = _three_rounds(tmp_path, monkeypatch)
    prompt = proposer.prompts[2]
    said = "history: solved 0/3 in 3 graphs"
    assert re.search(rf"Item t1 — answered wrong \(score [^)]*\) — {said}", prompt)
    assert re.search(r"Item t2 — passed \(score [^)]*\)\n", prompt)
    assert "history:" not in prompt.split("Item t2 —", 1)[1].split("\n", 1)[0]
    ctx = loop._context()
    t1 = next(r for r in ctx.item_records if r["id"] == "t1")
    assert t1["item_history"] in analysts.item_text(t1).splitlines()[0]
    plain = dataclasses.replace(ctx, item_records=_stripped(ctx.item_records),
                                parent_item_records=_stripped(ctx.parent_item_records))
    failing = sum(1 for r in ctx.item_records if not r.get("passed"))
    grown = len(ctx.render()) - len(plain.render())
    assert 0 < grown <= 45 * failing


def test_the_history_changes_no_item_shown_or_analysed(tmp_path, monkeypatch):
    """Information only: the same items are shown whole, picked for the table's full
    view and split for the analysts, with and without it."""
    loop, _ = _three_rounds(tmp_path, monkeypatch)
    records = loop._context().item_records
    assert any(r.get("item_history") for r in records)
    plain = _stripped(records)

    def ids(rows: list[dict[str, Any]]) -> list[str]:
        return [str(r.get("id")) for r in rows]

    assert ids(base.trajectories_whole(records)) == ids(base.trajectories_whole(plain))
    assert ids(base.trajectories_whole(records, budget=50)) == ids(
        base.trajectories_whole(plain, budget=50))
    assert [ids(part) for part in analysts.split(records, SEED)] == [
        ids(part) for part in analysts.split(plain, SEED)]
    with_history = "\n".join(base._trajectories(records, budget=50))
    without = "\n".join(base._trajectories(plain, budget=50))
    assert re.findall(r"Item (\S+) —", with_history) == re.findall(r"Item (\S+) —", without)
    # In the table of every item: the history column, "(below)" for one shown in full.
    table = with_history.split("### Every item", 1)[1].split("###", 1)[0]
    assert "| history |" in table and "solved 0/4 in 4 graphs" in table
    assert re.search(r"^\| t1 \|.*\| \(below\) \|$", table, re.M), table
    assert "history" not in without.split("### Every item", 1)[1].split("###", 1)[0]


def test_the_history_counts_train_runs_only(tmp_path, monkeypatch):
    """Smoke runs and val rollouts are no part of an item's history: with one train run
    per graph it counts exactly the graphs."""
    loop, _ = _three_rounds(tmp_path, monkeypatch)
    assert _events(loop, "smoke") and _vals(loop)
    t1 = next(r for r in loop._context().item_records if r["id"] == "t1")
    assert t1["item_history"] == "history: solved 0/4 in 4 graphs"


# ── the smoke check: walker and spine ─────────────────────────────────────


class SmokeProbe(nn.Mode):
    """The smoke check's mode: `loop_ops.SmokeTask`'s items through a step that fails
    on every item while ``level`` is 13 (`loop_ops.break_at_level`)."""

    name = "smoke_probe"

    @classmethod
    def declare(cls):  # type: ignore[no-untyped-def]
        return {"level": nn.Knob(default=0, kind="int", lo=0, hi=99, description="a level")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="work", subgraph_exit="work",
                          task_ref="loop_ops:SmokeTask", writes=("current_item.json",),
                          produces=("item_scores",)))
        graph.add(nn.Node("work", "op", call="loop_ops:break_at_level",
                          reads=("current_item.json",), writes=("seen.json",),
                          produces=("seen",), knobs=("level",)))
        graph.add(nn.Node("finish", "op", call="loop_ops:write_best", writes=("best.json",),
                          produces=("best",)))
        graph.connect("data", "finish")
        return graph

    def score(self, run):  # type: ignore[no-untyped-def]
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


class Fixer(Optimizer):
    """Proposes level 13 (a broken step) once; its repair answers level 14."""

    name = "fixer"

    def __init__(self) -> None:
        super().__init__(k=1)
        self.repaired: list[list[str]] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        if ctx.epoch != 1 or self.repaired:
            return []
        return [Proposal(edit={"op": "set_knob", "name": "level", "value": 13},
                         parent=parents[0].id)]

    def repair(self, ctx, proposal, reasons, parent):  # type: ignore[no-untyped-def]
        self.repaired.append(list(reasons))
        return Proposal(edit={"op": "set_knob", "name": "level", "value": 14}, parent=parent.id)


def _drain(graphs: Path):  # type: ignore[no-untyped-def]
    """``dsh_runtime.run_on_dsh`` through the real TS spine drain: the state lowered
    into the run directory, its knobs in every command's environment, the drain driven
    to the end, its facts read back."""
    from refactory_dsh import lowering
    from refactory_dsh.dsh_runtime import FACTS, DshRun

    runs: list[Path] = []

    def run_on_dsh(state, workdir, *, name="mode", timeout_s=900, **_kwargs):  # type: ignore[no-untyped-def]
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        runs.append(workdir)
        lowering.write_declarations(state, workdir)
        graph_path, _ = lowering.write_mode_graph(state, graphs / str(len(runs)), name="smoke")
        responses = graphs / f"responses-{len(runs)}.json"
        responses.write_text("{}")
        # The knobs as run_on_dsh hands them to the spine (``--knobs``): every command's env.
        env = {**os.environ, "REFACTORY_WORKDIR": str(workdir), "PYTHONPATH": str(HERE),
               **lowering.knob_environment(state)}
        env["PATH"] = f"{PYTHON_BIN}:{env.get('PATH', '')}"
        done = subprocess.run(
            [str(NODE), "--import", "tsx/esm", str(DRAIN_AGENT), str(graph_path), str(workdir),
             f"{PYTHON_BIN}/python", "30000", str(responses)],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180, env=env)
        facts_path = workdir / FACTS
        facts = json.loads(facts_path.read_text()) if facts_path.exists() else {}
        return DshRun(facts=facts, workdir=workdir, returncode=done.returncode,
                      stdout=done.stdout, stderr=done.stderr)

    return run_on_dsh, runs


@pytest.mark.parametrize("runtime", [
    "walker",
    pytest.param("spine", marks=pytest.mark.skipif(
        NODE is None or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/"
                             "@deepseek-ai/cordis").exists(),
        reason="the spine drain needs Node and the installed workspace (pnpm install)"))])
def test_a_smoke_failure_is_refused_repaired_and_smoked_again(tmp_path, monkeypatch, runtime):
    """[walker] / [spine] The proposal sets level 13, whose step fails on every item: its
    smoke run on the parent's first wrong item (s1) and first passed one (s2) fails, so
    it is refused E-smoke, naming the error and the step; the repair call fixes it (14),
    the fix is smoked again and passes, and only the fix is evaluated. The smoke runs
    count as rollouts and never as train records or evaluations."""
    from refactory_dsh import dsh_runtime

    if runtime == "spine":
        drain, runs = _drain(tmp_path / "graphs")
        monkeypatch.setattr(dsh_runtime, "run_on_dsh", drain)
        # The spine evaluator names its run after a package mode's dotted module.
        dotted = f"tests.{__name__}"
        monkeypatch.setitem(sys.modules, dotted, sys.modules[__name__])
        monkeypatch.setattr(SmokeProbe, "__module__", dotted)
    parts = tmp_path / "smoke-parts.json"
    parts.write_text(json.dumps({"train": ["s1", "s2", "s3", "s4"], "val": ["v1", "v2"]}))
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(SmokeProbe):
        loop = Loop(MODE_ID, LoopConfig(
            epochs=1, batch_size=1, optimizer="random", seed=SEED, runs_root=tmp_path / "run",
            allowed_ops=("set_knob",), max_evaluations=20, holdout=0.0,
            evaluator="dsh" if runtime == "spine" else "walker", parts_manifest=str(parts)),
            project=Path("."))
        loop.optimizer = fixer = Fixer()
        loop.run()
    smokes = _events(loop, "smoke")
    assert [(e["items"], e["outcome"]) for e in smokes] == [
        (["s1", "s2"], "failed"), (["s1", "s2"], "passed")]
    [error] = smokes[0]["errors"]
    assert (error["kind"], error["node"], error["item"]) == ("KeyError", "work", "s1")
    # The spine's data node stops recording once every item failed: it keeps s1's record.
    assert error["count"] == (2 if runtime == "walker" else 1)
    [reasons] = fixer.repaired
    assert reasons[0].startswith("E-smoke: a smoke run on 2 train item(s)")
    assert "KeyError at node work (item s1)" in reasons[0]
    levels = _by_level(loop)
    assert sorted(levels) == [0, 14], "the broken graph never ran on train"
    assert smokes[1]["id"] == levels[14]
    refusal = next(r for r in loop.rejections if r.reasons[0].startswith("E-smoke"))
    assert refusal.repaired == levels[14]
    assert [e["id"] for e in _events(loop, "train_records")] == [levels[0], levels[14]]
    assert loop.evaluations == 2, "the seed and the fix; the smoke runs are no evaluation"
    # 4 train + 2 val for each, and the smoke items: 2, and the failed run's records.
    assert loop.rollouts == 6 + 6 + 2 + error["count"]
    smoke_dir = loop.root / "runs" / smokes[0]["id"] / "occasions" / "smoke_1"
    assert (smoke_dir / "measurements.json").exists()
    if runtime == "spine":
        assert smoke_dir in runs, "the smoke run went through the spine"


def test_a_person_reads_what_was_applied_and_a_failed_smoke_check():
    """The progress lines: each round's checkpoint (applied, its val score, a new best),
    the epoch's applied/not-applied counts, and a smoke check only when it failed."""
    from refactory_dsh import events

    line = events.render({"step": "checkpoint", "epoch": 2, "applied": "abc", "val": 0.6,
                          "new_best": True, "not_applied": 1, "best_val": 0.6})
    assert line == "epoch 2 · checkpoint: applied abc · val 0.600 · new best · 1 not applied"
    done = events.render({"step": "scheduler", "epoch": 2, "best": 0.5, "bound": "",
                          "gate": {"kept": 1, "rejected": 1, "selection": "checkpoint"},
                          "val_best": 0.7, "val_current": 0.6})
    assert done is not None and "applied 1, not applied 1" in done and "rejected" not in done
    failed = {"step": "smoke", "epoch": 1, "candidate": "abc", "items": 2, "outcome": "failed",
              "errors": [{"kind": "KeyError", "node": "work"}]}
    assert events.render(failed) == ("epoch 1 · smoke check of abc on 2 train item(s): failed "
                                     "(KeyError at work)")
    assert events.render({**failed, "outcome": "passed", "errors": []}) is None
