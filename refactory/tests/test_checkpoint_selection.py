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


def test_a_flaky_pass_shows_its_history_and_a_solid_pass_none():
    """A passing item that failed before in the run (a flaky pass) gets its history in
    its heading and the table of every item; a pass that never failed gets none; a
    failing item's is as before."""
    from types import SimpleNamespace

    runs = SimpleNamespace(_train_runs=[
        ("g1", [("a", True), ("b", True), ("c", False), ("d", False)]),
        ("g2", [("a", False), ("b", True), ("c", False), ("d", True)]),
        ("g3", [("a", True), ("b", True), ("c", False)])])
    records = [{"id": item, "passed": passed, "score": float(passed)}
               for item, passed in (("a", True), ("b", True), ("c", False), ("d", True))]
    shown = {r["id"]: r for r in Loop._with_item_history(runs, records)}  # type: ignore[arg-type]
    expected = {"a": "history: solved 2/3 in 3 graphs, flaky", "b": "",
                "c": "history: solved 0/3 in 3 graphs", "d": "history: solved 1/2 in 2 graphs"}
    assert {item: r.get("item_history", "") for item, r in shown.items()} == expected
    for item, said in expected.items():
        heading = analysts.item_text(shown[item]).splitlines()[0]
        assert (heading.endswith(f" — {said}") if said else "history" not in heading), heading
    text = "\n".join(base._trajectories(list(shown.values()), budget=50))
    table = text.split("### Every item", 1)[1].split("\n###", 1)[0]
    rows = dict(re.findall(r"^\| (\w) \|(.*)$", table, re.M))
    assert set(rows) == set(expected), table
    for item, said in expected.items():
        if said:
            assert said.removeprefix("history: ") in rows[item] or "(below)" in rows[item]
        else:
            assert "solved" not in rows[item] and "(below)" not in rows[item], rows[item]


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


def test_the_request_room_counts_the_history_and_the_checkpoint_sections(
        tmp_path, monkeypatch):
    """#709's room (``ProposalContext.request_room``) holds a checkpoint request whole: it
    is measured on the rendered request, the item histories (in the table and in each
    heading of an item shown in full) and the checkpoint rule included, so they leave
    less room for the items shown whole, and only those give way (named, still in the
    table)."""
    loop, _ = _three_rounds(tmp_path, monkeypatch)
    ctx = loop._context()
    # The passing item gives a long answer, so leaving it out frees room; a small evidence
    # budget, so the rule picks the items shown whole.
    records = [{**r, "prediction": "x" * 4000} if r.get("passed") else r
               for r in ctx.item_records]
    ctx = dataclasses.replace(ctx, item_records=records, evidence_budget=200, request_room=0)
    full = ctx.render()
    history = re.compile(r"### Item t\d — answered wrong [^\n]* — history: solved 0/\d+ in "
                         r"\d+ graphs")
    rule = "Each round's change is applied unless its train run fails"
    assert history.search(full) and "| history |" in full and rule in full
    assert "### Item t2 — passed" in full
    room = len(full) - 1000
    fitted = dataclasses.replace(ctx, request_room=room).render()
    assert len(fitted) <= room, (len(fitted), room)
    assert "### Item t2 — passed" not in fitted and "left out so this request fits" in fitted
    assert history.search(fitted) and "| history |" in fitted and rule in fitted


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

    import threading

    runs: list[Path] = []
    lock = threading.Lock()  # a train run and its val run start drains at once

    def run_on_dsh(state, workdir, *, name="mode", timeout_s=900, **_kwargs):  # type: ignore[no-untyped-def]
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        with lock:
            runs.append(workdir)
            number = len(runs)
        lowering.write_declarations(state, workdir)
        graph_path, _ = lowering.write_mode_graph(state, graphs / str(number), name="smoke")
        responses = graphs / f"responses-{number}.json"
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


#: The walker, and the TS spine through the lowering where Node and the workspace are there.
RUNTIMES = [
    "walker",
    pytest.param("spine", marks=pytest.mark.skipif(
        NODE is None or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/"
                             "@deepseek-ai/cordis").exists(),
        reason="the spine drain needs Node and the installed workspace (pnpm install)"))]


def _smoke_run(tmp_path: Path, monkeypatch: Any, runtime: str):  # type: ignore[no-untyped-def]
    """One epoch of SmokeProbe with the Fixer on ``runtime``: (loop, fixer, the run
    directories the spine ran, empty on the walker)."""
    from refactory_dsh import dsh_runtime

    runs: list[Path] = []
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
    return loop, fixer, runs


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_smoke_failure_is_refused_repaired_and_smoked_again(tmp_path, monkeypatch, runtime):
    """[walker] / [spine] The proposal sets level 13, whose step fails on every item: its
    smoke run on the parent's first wrong item (s1) and first passed one (s2) fails, so
    it is refused E-smoke, naming the error and the step; the repair call fixes it (14),
    the fix is smoked again and passes, and only the fix is evaluated. The smoke runs
    count as rollouts and never as train records or evaluations."""
    loop, fixer, runs = _smoke_run(tmp_path, monkeypatch, runtime)
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


# ── what the requests may say: train information only ─────────────────────


class Telling(Scripted):
    """A model proposer recording every proposal request and memory request it is sent."""

    def __init__(self, plan):  # type: ignore[no-untyped-def]
        super().__init__(plan, model=True)
        self.memory_prompts: list[str] = []

    def write_memory(self, *, epoch, generation, build):  # type: ignore[no-untyped-def]
        self.memory_prompts.append(build()[0])
        return {"note": {"text": f"note of epoch {epoch}", "reasoning": ""}}


#: The seed is the best checkpoint on val by far; every later one is applied anyway.
CHECKPOINT_VAL = {0: 0.913771, **{n: round(0.5 + n * 0.0101 + 0.000771, 6) for n in range(1, 11)}}
CHECKPOINT_TRAIN = {n: round(0.3 + n * 0.0137, 4) for n in range(11)}


def test_a_checkpoint_request_says_no_val_score_and_does_not_mark_the_val_best(
        tmp_path, monkeypatch):
    """Ten checkpoints, the seed the best on val. No proposal or memory request holds a
    val score; the header gives the current candidate's train score (never the val
    best's) and names no validation gate; "Recent results" lists the current one and the
    newest checkpoints, so the val-best seed drops out of it as later ones arrive."""
    from refactory_dsh.optim.base import fmt_score

    _scores(monkeypatch, CHECKPOINT_TRAIN, CHECKPOINT_VAL)
    proposer = Telling([[(n, n + 1)] for n in range(9)])
    loop, _, report = _run(tmp_path, monkeypatch, [], proposer=proposer, epochs=10)
    seed = loop.population[0].id
    assert loop._gate().best_id == seed == report["best"]["id"], "setup: the seed wins on val"
    assert len(proposer.prompts) == 10 and proposer.memory_prompts
    for prompt in [*proposer.prompts, *proposer.memory_prompts]:
        # Wall-clock seconds (the cost block's ``"total_s": 0.521``) are not scores, and
        # one can read like a val score to three places.
        scores = re.sub(r'"\w+_s": [0-9.]+', "", prompt)
        for value in CHECKPOINT_VAL.values():
            assert fmt_score(value) not in scores and str(value) not in prompt, value
        assert "validation gate" not in prompt
    for number, prompt in enumerate(proposer.prompts):
        header = prompt.split("\n## ", 1)[0]
        # Round n's current candidate is level n (each round applies its child).
        assert (f"Train score of the current candidate, the one being trained: "
                f"{fmt_score(CHECKPOINT_TRAIN[number] - 0.01 * 3)}") in header, header
        recent = prompt.split("## Recent results", 1)[1].split("\n## ", 1)[0]
        rows = re.findall(r"^\| ([0-9a-f]{12}) \|", recent, re.M)
        assert _levels(loop, rows) == [number, *range(number - 1, -1, -1)][:8]
    last = proposer.prompts[-1]
    recent = last.split("## Recent results", 1)[1].split("\n## ", 1)[0]
    assert seed not in recent, "the val-best seed is no longer among the newest checkpoints"
    assert not any("best" in line for line in last.splitlines() if seed in line)


# ── the built-in strategies and population optimizers get both ────────────


@pytest.mark.parametrize("strategy,optimizer,picker", [
    *((name, "random", "") for name in strategies.names() if name != "default"),
    ("default", "archive", ""), ("default", "random", "islands")])
def test_every_built_in_strategy_is_smoked_and_shows_item_history(
        tmp_path, monkeypatch, strategy, optimizer, picker):
    """Every built-in strategy, and the default loop with a population optimizer or
    picker, under its default selection (pool): each new candidate is smoked on its
    parent's train items, and the request every strategy builds (``Loop._context``)
    heads a failing item with its history once it has more than one train run."""
    from refactory_dsh.loop import Loop as _Loop

    monkeypatch.setenv(PASS_IDS_ENV, "t2")
    monkeypatch.setattr(ValidationGate, "val_score",
                        staticmethod(lambda run, cand: 0.5 + _level(cand) / 1000))
    monkeypatch.setattr(FitProbe, "TRAIN", {level: 0.2 + level / 1000 for level in range(100)})
    monkeypatch.setattr(_Loop, "_seed_population", lambda self, seed: None)

    class Climber(Scripted):
        """One child of the pipeline it is shown, a level up, per proposal asked for."""

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            parent = next((p for p in parents if p.id == ctx.parent), parents[0])
            out = []
            for _ in range(max(1, n)):
                self.level = max(getattr(self, "level", 0), _level(parent)) + 1
                out.append(Proposal(edit={"op": "set_knob", "name": "level",
                                          "value": self.level}, parent=parent.id))
            return out

    with _patch_mode(FitProbe):
        loop = strategies.create(strategy, MODE_ID, _config(
            tmp_path, test=True, epochs=3, batch_size=2, optimizer=optimizer,
            parent_picker=picker), project=Path("."))
        climber = Climber([], model=True)
        if picker:
            from refactory_dsh.optim import PickedParent

            loop.optimizer = PickedParent(loop.optimizer.picker, climber)
        else:
            loop.optimizer = climber
        if hasattr(loop, "per_stage"):
            loop.per_stage = 1
        loop.run()
    assert loop.selection_mode() == "pool"
    smoked = _events(loop, "smoke")
    children = [c for c in loop.population if c.parent]
    assert children and {e["id"] for e in smoked} >= {c.id for c in children}
    assert all(e["items"] == ["t1", "t2"] for e in smoked)
    failing = [r for r in loop._context().item_records if not r.get("passed")]
    assert failing and all(r["item_history"].startswith("history: solved 0/")
                           for r in failing), failing[:1]


# ── a graph's train and val runs together ─────────────────────────────────


def _overlaps(monkeypatch: Any, wait_s: float = 10.0) -> dict[str, list[str]]:
    """Make each candidate's own train run (``runs/<id>``) and its val runs
    (``occasions/val_<n>``) wait for each other inside ``Loop._measure``: the result
    says, per candidate id, which of them met the other (``"together"``) and which
    waited ``wait_s`` in vain (``"alone"``). Smoke runs pass straight through."""
    import threading

    original = Loop._measure
    barriers: dict[str, threading.Barrier] = {}
    lock = threading.Lock()
    met: dict[str, list[str]] = {}

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        own = workdir.name == candidate.id and part == "train"
        val = part == "val" and workdir.name.startswith("val_")
        if own or val:
            with lock:
                barrier = barriers.setdefault(candidate.id, threading.Barrier(2))
            try:
                barrier.wait(timeout=wait_s)
                outcome = "together"
            except threading.BrokenBarrierError:
                outcome = "alone"
            with lock:
                met.setdefault(candidate.id, []).append(f"{part} {outcome}")
            if val:
                # A worker receipt of the val run's, as an agent node leaves one.
                receipt = workdir / ".refactory" / "workers" / "val-worker.json"
                receipt.parent.mkdir(parents=True, exist_ok=True)
                receipt.write_text(json.dumps({"worker_id": "val-worker", "role": "val"}))
        return original(self, candidate, workdir, part, **kwargs)

    monkeypatch.setattr(Loop, "_measure", measure)
    return met


def _batches(monkeypatch: Any) -> list[list[tuple[str, str]]]:
    """Every ``Run.evaluate_many`` call, as its requests' ``(candidate id, part)``."""
    from refactory_dsh.outerloop import Run

    calls: list[list[tuple[str, str]]] = []
    original = Run.evaluate_many

    def evaluate_many(self, requests, **kwargs):  # type: ignore[no-untyped-def]
        calls.append([(r.candidate.id, r.part) for r in requests])
        return original(self, requests, **kwargs)

    monkeypatch.setattr(Run, "evaluate_many", evaluate_many)
    return calls


def test_a_rounds_train_and_val_runs_run_together(tmp_path, monkeypatch):
    """The seed's and each applied child's train run and val checkpoint are one
    ``evaluate_many`` batch, run at once: each waits inside its measurement until the other
    has started, so a round costs the longer of the two, not both. The train run's record
    lists none of the val run's workers."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.7}, {0: 0.5, 1: 0.6, 2: 0.7})
    met = _overlaps(monkeypatch)
    calls = _batches(monkeypatch)
    loop, _, _ = _run(tmp_path / "together", monkeypatch, [[(0, 1)], [(1, 2)]])
    ids = _by_level(loop)
    assert {_levels(loop, [cid])[0]: sorted(m) for cid, m in met.items()} == {
        level: ["train together", "val together"] for level in (0, 1, 2)}
    assert [call for call in calls if len(call) == 2] == [
        [(ids[level], "train"), (ids[level], "val")] for level in (0, 1, 2)]
    assert all(len(call) == 2 for call in calls), "no other batch: each round is one"
    workers = [w for e in _events(loop, "score") for t in e["measurements"]["trials"]
               for w in t["workers"]]
    assert not [w for w in workers if w["worker_id"] == "val-worker"], "not the train run's"


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_train_and_val_run_together_on_both_runtimes(tmp_path, monkeypatch, runtime):
    """[walker] / [spine] The seed's and the applied fix's train and val runs meet inside
    their measurements on either runtime (on the spine, as two drains of the lowered
    graph)."""
    met = _overlaps(monkeypatch)
    loop, _, runs = _smoke_run(tmp_path, monkeypatch, runtime)
    levels = _by_level(loop)
    assert {cid: sorted(m) for cid, m in met.items()} == {
        levels[level]: ["train together", "val together"] for level in (0, 14)}
    assert [e["applied"] for e in _events(loop, "checkpoint")] == [levels[14]]
    if runtime == "spine":
        for level in (0, 14):
            root = loop.root / "runs" / levels[level]
            assert root in runs and root / "occasions" / "val_0" in runs


#: The fields that hold a time, not a result; ``resources``: the walker's copy of the
#: run-wide ledger as a run ends (`nn.executor`), whose totals depend on what else ran;
#: ``context_hash``: the settings' hash, which the two runs' concurrency setting changes.
TIMINGS = {"duration_s", "ts", "time", "at", "_last_active", "elapsed_s", "started_at",
           "finished_at", "resources", "context_hash"}


def _normalized(loop: Loop) -> dict[str, Any]:
    """What a run decided and recorded, without timings."""
    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            # A cost is a sum of floats, taken in another order when a run beside it
            # is left out: equal to a billionth of a dollar.
            return {k: round(v, 9) if k == "cost" and isinstance(v, float) else strip(v)
                    for k, v in value.items() if k not in TIMINGS}
        if isinstance(value, list):
            return [strip(v) for v in value]
        # The two runs' folders differ only in their root.
        return value.replace(str(loop.root), "<root>") if isinstance(value, str) else value

    keep = ("score", "train_records", "part_result", "checkpoint", "occasion_counter",
            "final_evaluation")
    def ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # The final evaluation's test runs finish in either order (`FinalEvaluation`'s
        # own concurrent batch, whatever the rounds do): sorted, the rest as written.
        test = sorted((r for r in rows if r.get("part") == "test"), key=json.dumps)
        return [r for r in rows if r.get("part") != "test"] + test

    return {
        **{name: ordered([strip(e) for e in _events(loop, name)]) for name in keep},
        "rollouts_per_epoch": _rollouts_per_epoch(loop),
        "accepted": list(loop._gate_accepted), "rejected": list(loop._gate_rejected),
        "best": loop._gate().best_id, "cache": dict(loop._gate().cache),
        "evaluations": loop.evaluations, "rollouts": loop.rollouts,
    }


def _spending(monkeypatch: Any) -> None:
    """Every walker evaluation spends tokens as an agent node does: a worker receipt in
    its folder and the worker's usage in the run's ledger, 100000 + level tokens on train
    and 10 + level on val (a millionth of a dollar each). A train run spends late, so a
    val run beside it has spent by then."""
    import hashlib
    import time

    from refactory_dsh.evaluators import WalkerEvaluator

    evaluate = WalkerEvaluator.evaluate

    def spending(self, mode, state, workdir, **kwargs):  # type: ignore[no-untyped-def]
        where = str(workdir).split("/runs/", 1)[-1]
        val = "/occasions/val_" in where
        if not val:
            time.sleep(0.3)
        tokens = (10 if val else 100000) + int(state["knobs"]["level"]["value"])
        worker = "worker-" + hashlib.sha1(where.encode()).hexdigest()[:16]
        receipt = Path(workdir) / ".refactory" / "workers" / f"{worker}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"worker_id": worker, "role": "fake",
                                       "usage": {"tokens": tokens, "status": "measured"}}))
        self.budget.record_usage(worker, tokens, estimated_cost_usd=tokens / 1e6)
        return evaluate(self, mode, state, workdir, **kwargs)

    monkeypatch.setattr(WalkerEvaluator, "evaluate", spending)


def test_together_decides_and_records_what_one_after_the_other_does(tmp_path, monkeypatch):
    """The same run with the train and val runs together and one after the other
    (``max_concurrent_evaluations`` 1, the order before): the same scores, seeds, val
    results, checkpoints, early stop, winner and test scores, and the same measurements,
    tokens and cost of each run among them (each run's own, not the other's spent beside
    it); only timings differ."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.4, 3: 0.7, 4: 0.2},
            {0: 0.7, 1: 0.6, 2: 0.8, 3: 0.5, 4: 0.4})
    _spending(monkeypatch)
    plan = [[(0, 1)], [(1, 2)], [(2, 3)], [(3, 4)], [(4, 5)]]
    together, _, report = _run(tmp_path / "together", monkeypatch, plan, epochs=6,
                               test=True, early_stop_patience=2)
    apart, _, report_apart = _run(tmp_path / "apart", monkeypatch, plan, epochs=6,
                                  test=True, early_stop_patience=2,
                                  max_concurrent_evaluations=1)
    assert report["bound"] == report_apart["bound"] == "patience_exhausted"
    mine, theirs = _normalized(together), _normalized(apart)
    for key in mine:
        assert mine[key] == theirs[key], key
    assert report["best"]["id"] == report_apart["best"]["id"]
    final = together.final_evaluation()
    assert final == apart.final_evaluation() and final["test_eval"] == "done"
    assert report["best_score"] == report_apart["best_score"]
    assert {r["seed"] for r in _events(together, "score")} == {train_batch_seed(SEED, 1)}
    assert {r["seed"] for r in _events(together, "part_result") if r["part"] == "val"} == {SEED}
    spent = {(_levels(together, [e["id"]])[0], t["tokens"])
             for e in _events(together, "score") for t in e["measurements"]["trials"]}
    assert spent == {(level, 100000 + level) for level in range(5)}, spent


def test_a_trials_usage_is_the_ledgers_difference_unless_another_ran_beside_it(tmp_path):
    """A trial's tokens and cost are the ledger's difference over the trial, exactly the
    figures before, when no other trial ran beside it (whatever the snapshots hold); with
    one beside it, that one's workers (their receipts in its folder) are left out, and
    a worker of the trial's own in a folder holding it (a val run's, under its train
    run's) is not."""
    def snap(workers):  # type: ignore[no-untyped-def]
        from refactory_dsh.budgets import RunBudget

        return RunBudget._summary({"workers": workers, "started_at": None})

    def old(before, after):  # type: ignore[no-untyped-def]
        return (after.get("known_tokens", 0) - before.get("known_tokens", 0)
                if after.get("token_status") in ("measured", "estimated") else None,
                after["estimated_cost_usd"] - (before.get("estimated_cost_usd") or 0)
                if after.get("estimated_cost_usd") is not None else None)

    def worker(tokens, cost=None, status="measured"):  # type: ignore[no-untyped-def]
        return {"tokens": tokens, "status": status if tokens is not None else "unavailable",
                "estimated_cost_usd": cost}

    loop = Loop.__new__(Loop)
    pairs = [
        ({}, {"a": worker(5, 0.5)}),
        ({"a": worker(5, 0.5)}, {"a": worker(9, 0.9), "b": worker(3, 0.1)}),
        ({"a": worker(5, None)}, {"a": worker(9, 0.9)}),
        ({"a": worker(5, 0.5)}, {"a": worker(9, 0.9), "b": worker(None)}),
        ({}, {}),
        ({"a": worker(4, 0.4, "estimated")}, {"a": worker(7, 0.7, "estimated")}),
    ]
    for before, after in pairs:
        assert loop._trial_usage(snap(before), snap(after), tmp_path / "t", set()) == old(
            snap(before), snap(after))
    train, other = tmp_path / "train", tmp_path / "o"
    val = train / "occasions" / "val_0"
    for folder, name in ((train, "w-train"), (val, "w-val"), (other, "w-other")):
        (folder / ".refactory" / "workers").mkdir(parents=True)
        (folder / ".refactory" / "workers" / f"{name}.json").write_text(
            json.dumps({"worker_id": name}))
    before = snap({"w-train": worker(1, 0.1), "w-val": worker(1, 0.1), "w-other": worker(1, 0.1)})
    after = snap({"w-train": worker(11, 1.1), "w-val": worker(3, 0.3), "w-other": worker(101, 10.1),
                  "w-other:unmetered-descendants": worker(5, 0.0)})
    tokens, cost = loop._trial_usage(before, after, train, {val, other})
    assert (tokens, cost) == (10, pytest.approx(1.0)), "train: neither the val run's nor o's"
    tokens, cost = loop._trial_usage(before, after, val, {train})
    assert (tokens, cost) == (2 + 100 + 5, pytest.approx(0.2 + 10.0)), "val: not train's own"


@pytest.mark.parametrize("limit", [{"max_elapsed_s": 3600}, {"max_tokens": 10 ** 9}])
def test_under_a_time_or_token_budget_val_runs_after_train(tmp_path, monkeypatch, limit):
    """A time or token budget: the val run would spend beside the train run what one
    after the other leaves to it, and could stop a train run that would have finished,
    so it runs after train, in a batch of its own, and decides as before."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.7}, {0: 0.5, 1: 0.6, 2: 0.7})
    calls = _batches(monkeypatch)
    loop, _, report = _run(tmp_path, monkeypatch, [[(0, 1)], [(1, 2)]], **limit)
    assert not [call for call in calls if len(call) > 1], calls
    assert _vals(loop) == {0: 1, 1: 1, 2: 1}
    assert _levels(loop, loop._gate_accepted) == [0, 1, 2] and report["bound"] == "epochs"


@pytest.mark.parametrize("killed", ["train", "val"])
def test_a_run_killed_mid_round_resumes_with_one_val_run(tmp_path, monkeypatch, killed):
    """Killed in the child's round while its train and val runs are running: its train
    run after its val run finished, or its val run after its train run finished. The
    resumed run scores the unfinished one only: one train score and one val result per
    graph, the finished run's result read back from the journal, and the child applied."""
    import time

    _scores(monkeypatch, {0: 0.5, 1: 0.6, 2: 0.7}, {0: 0.5, 1: 0.6, 2: 0.7})
    evaluate, part = Loop._evaluate, Loop._evaluate_part
    saw: list[bool] = []

    def wait_for(done) -> bool:  # type: ignore[no-untyped-def]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if done():
                return True
            time.sleep(0.02)
        return False

    def train(self, candidate, **kwargs):  # type: ignore[no-untyped-def]
        if _level(candidate) == 1 and killed == "train":
            saw.append(wait_for(lambda: bool(
                (self._part_results.get(candidate.id) or {}).get("val"))))
            raise KeyboardInterrupt("killed")
        return evaluate(self, candidate, **kwargs)

    def held_out(self, candidate, which, *args, **kwargs):  # type: ignore[no-untyped-def]
        if _level(candidate) == 1 and killed == "val":
            saw.append(wait_for(lambda: candidate.evaluated))
            raise KeyboardInterrupt("killed")
        return part(self, candidate, which, *args, **kwargs)

    monkeypatch.setattr(Loop, "_evaluate", train)
    monkeypatch.setattr(Loop, "_evaluate_part", held_out)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, monkeypatch, [[(0, 1)]], epochs=2)
    assert saw == [True], f"the other run finished while the {killed} run ran"
    monkeypatch.setattr(Loop, "_evaluate", evaluate)
    monkeypatch.setattr(Loop, "_evaluate_part", part)
    loop, _, report = _run(tmp_path, monkeypatch, [[(1, 2)]], epochs=2, resume=True)
    assert _vals(loop) == {0: 1, 1: 1, 2: 1}
    assert Counter(_levels(loop, [e["id"] for e in _events(loop, "score")])) == {
        0: 1, 1: 1, 2: 1}
    assert _levels(loop, loop._gate_accepted) == [0, 1, 2]
    assert [e["applied"] for e in _events(loop, "checkpoint")] == [
        _by_level(loop)[1], _by_level(loop)[2]]
    assert report["bound"] == "epochs"


class TrainFails(FitProbe):
    """FitProbe whose level 1 fails on train items and scores ``VAL_OF_ONE`` on val."""

    name = "train_fails"

    def score(self, run):  # type: ignore[no-untyped-def]
        ids = [str(e.get("id")) for e in run.facts.get("item_scores", []) if isinstance(e, dict)]
        if int(self.values()["level"]) == 1:
            if any(i.startswith("t") for i in ids):
                raise RuntimeError("no train score for level 1")
            return VAL_OF_ONE
        return super().score(run)


VAL_OF_ONE = 0.913771


def test_a_failed_childs_val_run_reaches_no_decision_or_request(tmp_path, monkeypatch):
    """The child's val run ran beside its train run, which failed: the child is not
    applied, its val result is in no decision (not the gate's cache, not the val scores a
    parent policy reads) and in no proposal or memory request, and the next round edits
    the seed."""
    from refactory_dsh.optim.base import fmt_score

    monkeypatch.setattr(FitProbe, "TRAIN", {0: 0.5, 1: 0.6, 2: 0.7})
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    proposer = Telling([[(0, 1)], [(0, 2)]])
    with _patch_mode(TrainFails):
        loop = Loop(MODE_ID, _config(tmp_path, epochs=3), project=Path("."))
        loop.optimizer = proposer
        loop.run()
    one = _by_level(loop)[1]
    results = [r["result"] for r in _events(loop, "part_result")
               if r["id"] == one and r["part"] == "val"]
    assert [r["score"] for r in results] == [pytest.approx(VAL_OF_ONE)], "setup: rolled out"
    assert loop._gate_decision(one) == "not applied"
    assert one not in loop._gate().cache and one not in loop._val_scores()
    assert proposer.handed == [[0], [0], [2]]
    assert proposer.prompts and proposer.memory_prompts
    for prompt in [*proposer.prompts, *proposer.memory_prompts]:
        assert fmt_score(VAL_OF_ONE) not in prompt and str(VAL_OF_ONE) not in prompt


@pytest.mark.parametrize("concurrent", [4, 1])
def test_a_budget_stopped_val_run_leaves_the_round_unjudged_once(tmp_path, monkeypatch,
                                                                 concurrent):
    """A time or token budget stops the child's val run: no val score, the child is
    neither applied nor refused, and its val run is not tried again in the round, with
    the runs together or (``max_concurrent_evaluations`` 1) one after the other."""
    from refactory_dsh.nn.mode import RunResult

    _scores(monkeypatch, {0: 0.5, 1: 0.6}, {0: 0.5, 1: 0.6})
    measure = Loop._measure

    def stopped(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        if part == "val" and _level(candidate) == 1:
            self.bound = "elapsed-time budget exhausted"
            return (RunResult(exit_ok=False, error="elapsed-time budget exhausted"),
                    {"budget_stopped": True})
        return measure(self, candidate, workdir, part, **kwargs)

    monkeypatch.setattr(Loop, "_measure", stopped)
    loop, _, _ = _run(tmp_path, monkeypatch, [[(0, 1)]], epochs=2,
                      max_concurrent_evaluations=concurrent)
    one = _by_level(loop)[1]
    vals = [r["result"] for r in _events(loop, "part_result")
            if r["id"] == one and r["part"] == "val"]
    assert [v.get("budget_stopped") for v in vals] == [True]
    assert one not in loop._gate_accepted and one not in loop._gate_rejected
    assert one not in loop._gate().cache
    assert not [e for e in _events(loop, "checkpoint") if e["applied"] == one]


class _OwnAccounting:
    """A custom model client: it would record its usage itself, with no worker receipt.
    Never called here (the toy mode asks no model)."""

    def complete(self, prompt, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("not called")


@pytest.mark.parametrize("client", ["none", "dsh", "custom"])
def test_val_runs_beside_train_only_where_each_runs_spend_can_be_told_apart(
        tmp_path, monkeypatch, client):
    """No model client, or ``DshModelClient`` (every call leaves a worker receipt in the
    run's folder): the train and val runs go together. A custom client, whose usage has
    no receipt, would leave the two runs' tokens and cost mixed: val runs after train."""
    from refactory_dsh.model import DshModelClient

    _scores(monkeypatch, {0: 0.5, 1: 0.6}, {0: 0.5, 1: 0.6})
    calls = _batches(monkeypatch)
    bind = Loop.bind_budget

    def bound(self, budget):  # type: ignore[no-untyped-def]
        bind(self, budget)
        if client != "none":
            self.evaluator.model = DshModelClient() if client == "dsh" else _OwnAccounting()

    monkeypatch.setattr(Loop, "bind_budget", bound)
    loop, _, _ = _run(tmp_path, monkeypatch, [[(0, 1)]], epochs=2)
    together = [call for call in calls if len(call) == 2]
    assert len(together) == (0 if client == "custom" else 2), calls
    assert _vals(loop) == {0: 1, 1: 1} and _levels(loop, loop._gate_accepted) == [0, 1]


def test_the_headline_compares_val_with_val_and_train_with_train(tmp_path, monkeypatch):
    """Under ``checkpoint`` ``best_score`` is the winner's val checkpoint: the report and
    the console set it beside the seed's val checkpoint (``seed_val``), and the seed's
    train score beside the winner's. ``seed_score`` stays the seed's train score. Under
    ``single`` the headline and the fields are as before."""
    from refactory_dsh import events

    _scores(monkeypatch, {0: 0.5, 1: 0.4, 2: 0.6}, {0: 0.5, 1: 0.8, 2: 0.6})
    loop, _, report = _run(tmp_path / "checkpoint", monkeypatch, [[(0, 1)], [(1, 2)]])
    assert report["best"]["id"] == _by_level(loop)[1]
    assert report["seed_val"] == pytest.approx(0.5) and report["best_score"] == pytest.approx(0.8)
    assert report["seed_score"] == pytest.approx(0.47)
    assert report["best_train_score"] == pytest.approx(0.37)
    text = (loop.root / "report.md").read_text()
    assert "Val checkpoint: seed **0.5** → best **0.8**" in text
    assert "Train: seed **0.47** → winner **0.37** (raw 0.5 → 0.4)" in text
    assert "Seed score **" not in text
    steps = [json.loads(line) for line in (loop.root / "progress.jsonl").read_text().splitlines()]
    [done] = [s for s in steps if s.get("step") == "report"]
    line = events.render(done)
    assert line.startswith("done · best val 0.800 (seed val 0.500) · train 0.370 (seed 0.470, "
                           "raw 0.400)"), line
    single, _, plain = _run(tmp_path / "single", monkeypatch, [[(0, 1)], [(1, 2)]],
                            selection="single")
    assert "seed_val" not in plain
    assert "Seed score **0.47** → best **" in (single.root / "report.md").read_text()
    steps = [json.loads(line) for line in (single.root / "progress.jsonl").read_text().splitlines()]
    [done] = [s for s in steps if s.get("step") == "report"]
    assert "seed_val" not in done and events.render(done).startswith("done · best ")
    assert "best val" not in events.render(done)


def test_the_bound_is_journaled_once(tmp_path, monkeypatch):
    """The run's bound is written once at its end, under any selection: by the early stop
    or the scheduler, not again by the run."""
    _scores(monkeypatch, {level: 0.5 for level in range(9)},
            {0: 0.7, 1: 0.6, 2: 0.5, 3: 0.4, 4: 0.3})
    loop, _, _ = _run(tmp_path / "patience", monkeypatch, [[(n, n + 1)] for n in range(6)],
                      epochs=8, early_stop_patience=2)
    assert [e["bound"] for e in _events(loop, "bound")] == ["patience_exhausted"]
    single, _, _ = _run(tmp_path / "single", monkeypatch, [[(0, 1)]], epochs=2,
                        selection="single")
    assert [e["bound"] for e in _events(single, "bound")] == ["epochs"]


def test_analyst_findings_cannot_break_the_report():
    """A finding's text with line breaks, a code fence and quotes renders as one line of
    the report, with no fence left to open a code block."""
    from refactory_dsh.loop import _analysts_section

    finding = {"source": "failure", "items": ["t1"], "finding": "the step answers\n```\n"
               "\"proceed\"\n```", "where": {},
               "evidence": ['he said:\n```python\nx = 1', " | ", "~~~~\ny = 2\n~~~"]}
    report = {"rounds": 1, "calls": 1, "cached": 0, "group_size": 4, "merge_batch_size": 4,
              "latest": {"epoch": 2, "generation": 1, "candidate": "abc", "failed_items": 1,
                         "passed_items": 0, "findings": [finding]}}
    lines = _analysts_section(report)
    [line] = [text for text in lines if text.startswith("1. ")]
    assert "```" not in "\n".join(lines) and "~~~" not in "\n".join(lines)
    assert "\n" not in line and '"proceed"' in line and "x = 1" in line and "y = 2" in line


def test_a_trial_that_raises_after_its_run_is_no_longer_running(tmp_path, monkeypatch):
    """A trial is noted as running (`Loop._trial_started`) until it ends, however it ends:
    one whose incident settling raises is not left among the running trials, where it
    would be taken as running beside every later one."""
    _scores(monkeypatch, {0: 0.5}, {0: 0.5})
    seen: list[Loop] = []

    def raising(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(self)
        raise RuntimeError("settling failed")

    monkeypatch.setattr(Loop, "_settle_incidents", raising)
    with pytest.raises(RuntimeError, match="settling failed"):
        _run(tmp_path, monkeypatch, [], epochs=1)
    assert seen and seen[0].__dict__.get("_running_trials") == []


def test_a_trial_is_noted_as_running_before_the_ledger_is_read(tmp_path, monkeypatch):
    """Each trial is noted as running before its measurement reads the ledger, so a trial
    that starts and ends in between counts as one beside it: per trial, start, then the
    ledger before and after."""
    import inspect

    from refactory_dsh.budgets import RunBudget

    _scores(monkeypatch, {0: 0.5, 1: 0.6}, {0: 0.5, 1: 0.6})
    order: list[str] = []
    started, snapshot = Loop._trial_started, RunBudget.snapshot

    def start(self, trial_dir):  # type: ignore[no-untyped-def]
        order.append("start")
        return started(self, trial_dir)

    def snap(self):  # type: ignore[no-untyped-def]
        caller = inspect.currentframe().f_back  # type: ignore[union-attr]
        if caller is not None and caller.f_code.co_name == "_measure":
            order.append("ledger")
        return snapshot(self)

    monkeypatch.setattr(Loop, "_trial_started", start)
    monkeypatch.setattr(RunBudget, "snapshot", snap)
    _run(tmp_path, monkeypatch, [[(0, 1)]], epochs=1, max_concurrent_evaluations=1)
    assert order and order == ["start", "ledger", "ledger"] * (len(order) // 3), order


def test_the_headline_says_n_a_and_different_train_samples(tmp_path, monkeypatch):
    """The seed with no val checkpoint reads n/a; with ``train_minibatch`` the seed and the
    winner were scored on different epochs' samples, and the headline and the console
    say so."""
    from refactory_dsh import events
    from refactory_dsh.loop import _headline

    line = _headline({"seed_val": None, "best_score": 0.8, "seed_score": 0.5,
                      "best_train_score": 0.6, "seed_raw_score": 0.5, "raw_best_score": 0.6,
                      "scoring": {"parsimony": 0.0}})[0]
    assert line.startswith("Val checkpoint: seed **n/a** → best **0.8**") and "Train: " in line
    _scores(monkeypatch, {0: 0.5, 1: 0.4, 2: 0.6}, {0: 0.5, 1: 0.8, 2: 0.6})
    loop, _, report = _run(tmp_path, monkeypatch, [[(0, 1)], [(1, 2)]], train_minibatch=4)
    assert report["best"]["id"] == _by_level(loop)[1]
    assert loop._minibatches[1] != loop._minibatches[2], "setup: two samples"
    assert report["train_samples_differ"] is True
    text = (loop.root / "report.md").read_text()
    assert "Train, on different samples (each epoch's own `train_minibatch` items): seed" in text
    steps = [json.loads(line) for line in (loop.root / "progress.jsonl").read_text().splitlines()]
    [done] = [step for step in steps if step.get("step") == "report"]
    assert "· train (different samples) " in events.render(done)


def test_a_train_trial_lists_none_of_its_candidates_smoke_workers(tmp_path, monkeypatch):
    """A child's smoke run (``occasions/smoke_<n>``) runs before its train run, in a folder
    under the train run's: its workers are not the train trial's."""
    _scores(monkeypatch, {0: 0.5, 1: 0.6}, {0: 0.5, 1: 0.6})
    measure = Loop._measure

    def smoke_worker(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        if workdir.name.startswith("smoke_"):
            receipt = workdir / ".refactory" / "workers" / "smoke-worker.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({"worker_id": "smoke-worker", "role": "smoke"}))
        return measure(self, candidate, workdir, part, **kwargs)

    monkeypatch.setattr(Loop, "_measure", smoke_worker)
    loop, _, _ = _run(tmp_path, monkeypatch, [[(0, 1)]], epochs=2)
    child = _by_level(loop)[1]
    assert list((loop.root / "runs" / child / "occasions").glob(
        "smoke_*/.refactory/workers/smoke-worker.json")), "setup: the smoke run's receipt"
    workers = [w for e in _events(loop, "score") for t in e["measurements"]["trials"]
               for w in t["workers"]]
    assert not [w for w in workers if w["worker_id"] == "smoke-worker"]
