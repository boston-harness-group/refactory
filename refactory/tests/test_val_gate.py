"""The search pieces the library offers every strategy (``outerloop.py``).

``ValidationGate`` (accept on a strict val improvement, val scores cached per whole
candidate, best on val), ``FinalEvaluation`` (the post-training block), the default
loop's ``val_gate`` with a population optimizer, and ``Run.keep``/``Run.candidate``
across a resume. Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn
from refactory_dsh.loop import Loop, LoopConfig, LoopError
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.outerloop import (
    FinalEvaluation, OuterLoop, Run, ValidationGate,
)

from test_skillopt import (  # the toy Task and mode SkillOpt's tests run on
    INITIAL_SKILL, MODE_ID, TASK_REF, TEST_IDS, TRAIN_IDS, SkillProbe, SkillTask, _patch_mode,
    _write_parts,
)


# ── ValidationGate, on a recording run ─────────────────────────────────────


class _RecordingRun:
    """The two Run methods the gate calls: ``evaluate`` and ``val_scores``."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.rolled: list[tuple[str, str, int | None]] = []
        self.occasions: dict[str, list[dict[str, Any]]] = {}

    def evaluate(self, candidate: Candidate, *, part: str = "train", seed: int | None = None) -> Candidate:
        self.rolled.append((candidate.id, part, seed))
        self.occasions.setdefault(candidate.id, []).append({"score": self.scores[candidate.id]})
        return candidate

    def val_scores(self, candidate: Candidate) -> list[dict[str, Any]]:
        return self.occasions.get(candidate.id, [])


def _cand(skill: str, other: int = 0) -> Candidate:
    return Candidate.from_state({"graph": {"nodes": {}}, "knobs": {
        "skill": {"value": skill}, "other": {"value": other}}})


def test_gate_accepts_only_a_strict_improvement_and_keeps_the_best():
    seed, equal, better, worse = _cand("s"), _cand("eq"), _cand("up"), _cand("down")
    run = _RecordingRun({seed.id: 0.5, equal.id: 0.5, better.id: 0.7, worse.id: 0.6})
    gate = ValidationGate(seed=9)
    assert gate.start(run, seed) == 0.5

    verdict = gate.judge(run, equal)
    assert not verdict.accepted and verdict.incumbent == 0.5, "an equal score is rejected"
    assert gate.current_id == seed.id

    verdict = gate.judge(run, better)
    assert verdict.accepted and verdict.best and verdict.incumbent == 0.5
    assert (gate.current_id, gate.best_id, gate.best_score) == (better.id, better.id, 0.7)

    assert not gate.judge(run, worse).accepted, "0.6 does not beat the current 0.7"
    # Against a named incumbent (a population search's parent) 0.6 beats 0.5.
    assert gate.judge(run, worse, against=seed).accepted
    assert gate.best_id == better.id, "the best on val stays the 0.7 candidate"
    # Every rollout was on val with the gate's seed, and none twice.
    assert {(part, s) for _cid, part, s in run.rolled} == {("val", 9)}
    assert len(run.rolled) == len({cid for cid, _p, _s in run.rolled}) == 4


def test_gate_caches_by_the_whole_candidate_not_one_knob():
    """Two candidates with the same skill text but another knob apart are two
    candidates: each is rolled out once, and a cached one is never rolled again."""
    first, same_skill = _cand("text", other=1), _cand("text", other=2)
    run = _RecordingRun({first.id: 0.4, same_skill.id: 0.8})
    gate = ValidationGate()
    assert gate.score(run, first) == (0.4, False)
    assert gate.score(run, same_skill) == (0.8, False), "a different candidate is not a cache hit"
    assert gate.score(run, first) == (0.4, True)
    assert [cid for cid, _p, _s in run.rolled] == [first.id, same_skill.id]


def test_gate_state_round_trips_and_carry_keeps_the_score():
    seed, derived = _cand("s"), _cand("s+")
    run = _RecordingRun({seed.id: 0.5})
    gate = ValidationGate()
    gate.start(run, seed)
    gate.carry(derived, remember=True)
    assert (gate.current_id, gate.current_score, gate.cache[derived.id]) == (derived.id, 0.5, 0.5)
    restored = ValidationGate()
    restored.load_state_dict(json.loads(json.dumps(gate.state_dict())))
    assert restored.state_dict() == gate.state_dict()
    with pytest.raises(ValueError, match="not in the population"):
        restored.best_in([derived])
    assert restored.best_in([derived, seed]) is seed


# ── a strategy that is not SkillOpt, on the real loop ──────────────────────


def _write_parts_without_val(root: Path) -> Path:
    """A parts.json with train and test items and no val part: the run selects on train."""
    path = root / "parts-noval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train": TRAIN_IDS, "test": TEST_IDS}))
    return path


def _loop(tmp_path: Path, strategy: OuterLoop | None = None, **overrides: Any) -> Loop:
    """A loop on the toy mode. ``val_gate=False`` (with no ``parts_manifest`` given) runs
    it without the gate, on a manifest with no val part: the gate is decided by whether
    the run has val items, never set."""
    without_gate = overrides.get("val_gate") is False and "parts_manifest" not in overrides
    base: dict[str, Any] = dict(
        epochs=2, batch_size=1, optimizer="random", seed=7, runs_root=tmp_path / "run",
        allowed_ops=("set_knob",), max_evaluations=60, holdout=0.0,
        parts_manifest=str(_write_parts_without_val(tmp_path) if without_gate
                           else _write_parts(tmp_path, test=True)),
    )
    base.update(overrides)
    loop = Loop(MODE_ID, LoopConfig(**base), project=Path("."))
    if strategy is not None:
        strategy.configure()
        loop.set_custom_strategy(strategy)
    return loop


def _set_skill(run: Run, parent: Candidate, text: str) -> Candidate:
    outcome = run.attempt(Proposal(edit={"op": "set_knob", "name": "skill", "value": text},
                                   parent=parent.id), parent)
    assert isinstance(outcome, Candidate), outcome
    return outcome


class GatedToy(OuterLoop):
    """Gate two proposals, carry the current score to a derived candidate, then run
    the post-training block: the shared pieces with no SkillOpt in them."""

    name = "gated_toy"
    required_ops = ("set_knob",)

    def configure(self, **options: Any) -> None:
        self.gate = ValidationGate(seed=7)
        self.decisions: list[tuple[str, dict[str, Any]]] = []
        self.final = FinalEvaluation(record=lambda step, **fields: self.decisions.append((step, fields)))

    def strategy(self, run: Run) -> None:
        seed = run.seed()
        self.gate.start(run, seed)
        for text in ("A better", "B"):
            candidate = _set_skill(run, seed, text)
            if self.gate.judge(run, candidate).accepted:
                run.keep(candidate)
        derived = run.keep(_set_skill(run, run.candidate(self.gate.current_id), "A better, revised"))
        self.gate.carry(derived)
        self.final.run(run, self.gate, seed=7, save=run.save_strategy_state)
        run.stop("done")

    def strategy_state_dict(self) -> dict[str, Any]:
        return {**self.gate.state_dict(), **self.final.state_dict()}

    def load_strategy_state(self, state: dict[str, Any]) -> None:
        self.gate.load_state_dict(state)
        self.final.load_state_dict(state)

    def best_candidate(self, population: list[Candidate]) -> Candidate | None:
        return self.gate.best_in(population)


def _skill_of(candidate: Candidate) -> str:
    return str(candidate.state["knobs"]["skill"]["value"])


def test_any_strategy_gets_the_gate_and_the_post_training_block(tmp_path, monkeypatch):
    """The final candidate's val check, then the seed, the best and the final
    candidate on test, all with the strategy's seed, in that order."""
    scores = {INITIAL_SKILL: 0.5, "A better": 0.9, "B": 0.1}
    monkeypatch.setattr(ValidationGate, "val_score",
                        staticmethod(lambda run, cand: scores.get(_skill_of(cand), 0.0)))
    tests: list[tuple[str, int | None]] = []
    original_many = Run.evaluate_many

    def evaluate_many(self, requests, *, on_result=None):  # type: ignore[no-untyped-def]
        tests.extend((_skill_of(r.candidate), r.seed) for r in requests if r.part == "test")
        return original_many(self, requests, on_result=on_result)

    monkeypatch.setattr(Run, "evaluate_many", evaluate_many)
    SkillTask.seen = []
    strategy = GatedToy()
    with _patch_mode(SkillProbe):
        report = _loop(tmp_path, strategy).run()

    best = report["best"]["id"]
    assert best == strategy.gate.best_id and best != strategy.gate.current_id
    assert sorted(tests) == sorted([(INITIAL_SKILL, 7), ("A better", 7), ("A better, revised", 7)])
    # The seed reached the Task on every test item, not only on the request.
    assert {seed for part, seed in SkillTask.seen if part == "test"} == {7}
    result = strategy.final.result
    assert result["status"] == "done"
    assert [k for k in result if k in FinalEvaluation.PHASES] == [
        "final_selection", "baseline_test", "best_test", "final_test"]
    assert result["best_test_delta"] == pytest.approx(result["best_test"] - result["baseline_test"])
    steps = [(step, fields.get("action")) for step, fields in strategy.decisions]
    assert steps == [("final_selection_eval", "evaluated"), ("test_eval", "baseline_test"),
                     ("test_eval", "best_test"), ("test_eval", "final_test")]


# ── Run.keep and Run.candidate across a resume ─────────────────────────────


class ProcessDied(BaseException):
    pass


class KeepThenDie(OuterLoop):
    """Keep one candidate unevaluated and one evaluated before it was kept, then
    die; resumed, look both up by id."""

    name = "keep_then_die"
    required_ops = ("set_knob",)

    def configure(self, **options: Any) -> None:
        self.ids: list[str] = []
        self.found: list[Candidate | None] = []

    def strategy(self, run: Run) -> None:
        if self.ids:
            self.found = [run.candidate(i) for i in self.ids]
            run.stop("done")
            return
        seed = run.seed()
        unevaluated = run.keep(_set_skill(run, seed, "kept unevaluated"))
        evaluated = _set_skill(run, seed, "kept evaluated")
        run.keep(run.evaluate(evaluated, part="train"))
        self.ids = [unevaluated.id, evaluated.id]
        run.save_strategy_state()
        raise ProcessDied()

    def strategy_state_dict(self) -> dict[str, Any]:
        return {"ids": self.ids}

    def load_strategy_state(self, state: dict[str, Any]) -> None:
        self.ids = list(state.get("ids") or [])


class KeepsOneUnscored(OuterLoop):
    """Score the seed, keep a child it never scores, and stop on its epochs."""

    name = "keeps_one_unscored"
    required_ops = ("set_knob",)

    def strategy(self, run: Run) -> None:
        seed = run.seed()
        run.evaluate(seed, part="train")
        run.keep(_set_skill(run, seed, "kept, never scored"))
        run.stop("epochs")


def test_a_custom_strategy_keeps_its_own_ending(tmp_path):
    """The built-in strategies score their last batch (``Loop.finish_strategy``); a
    custom strategy (SkillOpt's seam) owns its protocol, so nothing is scored for it."""
    with _patch_mode(SkillProbe):
        loop = _loop(tmp_path, KeepsOneUnscored())
        report = loop.run()
    assert report["bound"] == "epochs"
    assert len(report["unevaluated"]) == 1 and report["evaluations"] == 1


def test_a_kept_candidate_is_held_again_after_a_resume(tmp_path):
    with _patch_mode(SkillProbe):
        with pytest.raises(ProcessDied):
            _loop(tmp_path, KeepThenDie()).run()
        resumed = KeepThenDie()
        _loop(tmp_path, resumed, resume=True).run()
    unevaluated, evaluated = resumed.found
    assert unevaluated is not None and not unevaluated.evaluated
    assert evaluated is not None and evaluated.evaluated and evaluated.score is not None, (
        "a candidate scored before it was kept keeps its score"
    )
    assert _skill_of(evaluated) == "kept evaluated"


# ── the default loop's val_gate, with a population optimizer ───────────────


class LevelProbe(nn.Mode):
    """The toy Task's data node, and one integer knob a sampler can move; its
    train score falls as the level rises."""

    name = "level_probe"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"level": nn.Knob(default=5, kind="int", lo=0, hi=9, description="a level")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref=TASK_REF, writes=("current_item.json",), produces=("item_scores",),
        ))
        graph.add(nn.Node(
            "work", "op", call="loop_ops:record_item", reads=("current_item.json",),
            writes=("seen.json",), produces=("seen",), knobs=("level",),
        ))
        graph.add(nn.Node("finish", "op", call="loop_ops:write_best",
                          writes=("best.json",), produces=("best",)))
        graph.connect("data", "finish")
        return graph

    def score(self, run: nn.RunResult) -> float:
        # Train ranks the levels the other way round from val (``level / 10`` in
        # the tests), so the train leaderboard's first is never the val best.
        return 1.0 - int(self.values()["level"]) / 10


def _level(candidate: Candidate) -> int:
    return int(candidate.state["knobs"]["level"]["value"])


def test_hill_climb_runs_with_the_default_loops_val_gate(tmp_path, monkeypatch):
    """--optimizer hill_climb with the default loop and val_gate on: every
    candidate evaluated on train is rolled out on val once, each accepted one beats
    the current candidate before it, only accepted candidates are parents, and the
    winner is the best on val. A resumed run keeps the gate's decisions and rolls
    nothing again."""
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    val_rollouts: list[str] = []
    original = Loop._evaluate_part

    # Every val rollout, whichever Run call made it (a round's are rolled out
    # together through evaluate_many).
    def evaluate_part(self, candidate, part, items, seed, **kwargs):  # type: ignore[no-untyped-def]
        if part == "val":
            assert seed == 7
            val_rollouts.append(candidate.id)
        return original(self, candidate, part, items, seed, **kwargs)

    monkeypatch.setattr(Loop, "_evaluate_part", evaluate_part)
    overrides = dict(optimizer="hill_climb", val_gate=True, epochs=3, batch_size=3)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **overrides)
        report = loop.run()
    assert report["optimizer"] == "hill_climb"
    by_id = {c.id: c for c in loop.population}
    accepted, rejected = loop._gate_accepted, loop._gate_rejected
    assert accepted[0] == loop.population[0].id, "the seed is the gate's start"
    judged = accepted[1:] + rejected
    assert accepted[1:] and rejected, "setup: the search both improves and regresses"
    levels = [_level(by_id[cid]) for cid in accepted]
    assert levels == sorted(set(levels)), "each accepted candidate beats the current one"
    for cid in judged:
        assert by_id[cid].parent in accepted, "only an accepted candidate is a parent"
    assert sorted(val_rollouts) == sorted(accepted + rejected), "one val rollout per judged candidate"
    assert {c.id for c in loop.leaderboard()} <= set(accepted)
    best = max(accepted, key=lambda cid: _level(by_id[cid]))
    assert report["best"]["id"] == best == loop._gate().best_id

    before = list(val_rollouts)
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, **{**overrides, "epochs": 4, "resume": True})
        resumed.run()
    assert resumed._gate_accepted[:len(accepted)] == accepted
    assert resumed._gate_rejected[:len(rejected)] == rejected
    again = val_rollouts[len(before):]
    assert not set(again) & set(before), "a judged candidate is never rolled out on val again"


# Population optimizers, pickers and strategies run under the gate with ``selection``
# ``pool``: see test_validation_selection.py.


@pytest.mark.parametrize("optimizer", ["random", "autoresearch", "hill_climb"])
def test_val_gate_is_not_a_setting(optimizer):
    """No flag, no profile key: a saved profile's ``val_gate`` (``false`` was the default
    before #627) is dropped on load, and the run decides."""
    import argparse

    from refactory_dsh import profiles

    for saved in (True, False):
        assert "val_gate" not in profiles.normalize({"val_gate": saved, "optimizer": optimizer})
    parser = argparse.ArgumentParser()
    profiles.add_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--val-gate", "0"])


def test_val_gate_is_decided_by_the_val_part_and_the_strategy(tmp_path):
    """On with val items for a built-in strategy; off without them, and off for a custom
    strategy (it keeps its own gate). A value given in code that differs is refused."""
    no_val = tmp_path / "noval.json"
    no_val.write_text(json.dumps({"train": ["t1", "t2"]}))
    with _patch_mode(LevelProbe):
        gated = _loop(tmp_path / "gated")
        gated.init()
        assert gated.config.val_gate is True
        ungated = _loop(tmp_path / "ungated", parts_manifest=str(no_val))
        ungated.init()
        assert ungated.config.val_gate is False
        with pytest.raises(LoopError, match="val_gate is not a setting"):
            _loop(tmp_path / "on", val_gate=True, parts_manifest=str(no_val)).run()
        with pytest.raises(LoopError, match="val_gate is not a setting"):
            _loop(tmp_path / "off", val_gate=False, parts_manifest=_write_parts(tmp_path)).run()
        with pytest.raises(LoopError, match="keeps its own"):
            _loop(tmp_path / "custom", KeepThenDie(), val_gate=True).run()


def test_resuming_a_run_recorded_with_the_gate_off_and_val_items_is_refused(tmp_path):
    """A run started with ``--val-gate 0`` although it had val items (the default before
    #627) selected on train scores; resuming it would keep doing so, so the resume is
    refused with a clear message instead."""
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=1)
        loop.run()
        config = json.loads((loop.root / "config.json").read_text())
        assert config["val_gate"] is True
        (loop.root / "config.json").write_text(json.dumps({**config, "val_gate": False}))
        with pytest.raises(LoopError, match="no longer supported; start a new run"):
            _loop(tmp_path, epochs=2, resume=True).init()


class ScriptedLevels(Optimizer):
    """A scripted Type 1 proposer: each round, ``(parent level, child level)`` moves."""

    name = "scripted_levels"

    def __init__(self, plan: list[list[tuple[int, int]]]) -> None:
        super().__init__(k=1)
        self.plan = list(plan)
        self.handed: list[set[int]] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.handed.append({_level(p) for p in parents})
        moves = self.plan.pop(0) if self.plan else []
        return [
            Proposal(edit={"op": "set_knob", "name": "level", "value": child},
                     parent=next(p for p in parents if _level(p) == parent).id)
            for parent, child in moves
        ]


def test_the_default_gate_compares_a_child_with_its_current_candidate(tmp_path, monkeypatch):
    """SkillOpt's gate: a batch of two children of the seed 5, 7 then 6. 7 is
    accepted and becomes current; 6 beats its parent 5 but not the current 7, so
    it is rejected. Then 7 → 8 is accepted."""
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    loop, proposer, report = _gated(
        tmp_path, monkeypatch, [[(5, 7), (5, 6)], [(7, 8)], []], epochs=3, batch_size=2)
    by_id = {c.id: c for c in loop.population}
    assert [_level(by_id[cid]) for cid in loop._gate_accepted] == [5, 7, 8]
    assert [_level(by_id[cid]) for cid in loop._gate_rejected] == [6]
    assert _level(by_id[report["best"]["id"]]) == 8


# ── val_gate: SkillOpt's evaluation protocol ───────────────────────────────


def _gated(tmp_path: Path, monkeypatch: Any, plan: list[list[tuple[int, int]]], *,
           val: Any = None, **overrides: Any) -> tuple[Loop, ScriptedLevels, dict[str, Any]]:
    """A val_gate run of the scripted proposer; val scores ``level / 10`` unless ``val``."""
    monkeypatch.setattr(ValidationGate, "val_score",
                        staticmethod(val or (lambda run, cand: _level(cand) / 10)))
    proposer = ScriptedLevels(plan)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"val_gate": True, **overrides})
        loop.optimizer = proposer
        report = loop.run()
    return loop, proposer, report


def _record_train_seeds(monkeypatch: Any) -> list[tuple[int, int | None]]:
    seeds: list[tuple[int, int | None]] = []
    original = Loop._evaluate

    def evaluate(self, candidate, **kwargs):  # type: ignore[no-untyped-def]
        seeds.append((_level(candidate), kwargs.get("seed")))
        return original(self, candidate, **kwargs)

    monkeypatch.setattr(Loop, "_evaluate", evaluate)
    return seeds


def _record_test_rollouts(monkeypatch: Any) -> list[tuple[int, int | None]]:
    tests: list[tuple[int, int | None]] = []
    original_many = Run.evaluate_many

    def evaluate_many(self, requests, *, on_result=None):  # type: ignore[no-untyped-def]
        tests.extend((_level(r.candidate), r.seed) for r in requests if r.part == "test")
        return original_many(self, requests, on_result=on_result)

    monkeypatch.setattr(Run, "evaluate_many", evaluate_many)
    return tests


def _journal_events(loop: Loop, event: str) -> list[dict[str, Any]]:
    lines = (loop.root / "journal.jsonl").read_text().splitlines()
    return [record for record in map(json.loads, lines) if record.get("event") == event]


def test_val_gate_train_rollouts_use_skillopts_epoch_seed(tmp_path, monkeypatch):
    """Epoch e's train rollouts use seed + e*1000 + 1, as SkillOpt's; the last
    epoch's proposals, scored after it, get the next epoch's. Val and test keep the
    run's seed."""
    seeds = _record_train_seeds(monkeypatch)
    SkillTask.seen = []
    _gated(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)]], epochs=2)
    assert seeds == [(5, 1008), (7, 2008), (8, 3008)]
    assert {seed for part, seed in SkillTask.seen if part == "train"} == {1008, 2008, 3008}
    assert {seed for part, seed in SkillTask.seen if part in ("val", "test")} == {7}


def test_without_val_gate_the_default_loop_is_unchanged(tmp_path, monkeypatch):
    """No val_gate: every train rollout uses the run's seed, the last epoch's
    proposals are scored before the run ends (as the gate's are), and there is no
    post-training evaluation."""
    seeds = _record_train_seeds(monkeypatch)
    proposer = ScriptedLevels([[(5, 7)], [(7, 8)]])
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=2, val_gate=False)
        loop.optimizer = proposer
        report = loop.run()
    assert seeds == [(5, 7), (7, 7), (8, 7)]
    assert [_level(c) for c in loop.population if not c.evaluated] == []
    assert report["unevaluated"] == [] and report["evaluations"] == 3
    assert report["strategy_report"] == {}
    assert not _journal_events(loop, "final_evaluation")
    assert report["bound"] == "epochs"


def test_val_gate_still_gates_a_candidate_whose_train_run_failed(tmp_path, monkeypatch):
    """A candidate whose train run failed is rolled out on val and the gate
    decides; here it beats the current candidate, so it is accepted and wins."""
    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        if part == "train" and _level(candidate) == 8:
            result.exit_ok, result.error = False, "the train run failed"
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)
    val_rollouts: list[int] = []
    original_evaluate = Run.evaluate

    def evaluate(self, candidate, *, part="train", items=None, seed=None):  # type: ignore[no-untyped-def]
        if part == "val":
            val_rollouts.append(_level(candidate))
        return original_evaluate(self, candidate, part=part, items=items, seed=seed)

    monkeypatch.setattr(Run, "evaluate", evaluate)
    loop, _, report = _gated(tmp_path, monkeypatch, [[(5, 8)], []], epochs=2)
    eight = next(c for c in loop.population if _level(c) == 8)
    assert not eight.exit_ok, "setup: its train run failed"
    assert 8 in val_rollouts
    assert eight.id in loop._gate_accepted
    assert report["best"]["id"] == eight.id


def _fail_train_at(monkeypatch: Any, level: int) -> None:
    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        if part == "train" and _level(candidate) == level:
            result.exit_ok, result.error = False, "the train run failed"
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)


def test_val_gate_parent_is_the_current_candidate_for_any_optimizer(tmp_path, monkeypatch):
    """The gate's current candidate is the parent, though its train run failed and
    an earlier accepted candidate has a train score: with a scripted proposer and
    with the random optimizer (which ranks parents by train score)."""
    _fail_train_at(monkeypatch, 8)
    loop, proposer, _ = _gated(tmp_path / "scripted", monkeypatch, [[(5, 8)], [(8, 9)]], epochs=2)
    assert proposer.handed == [{5}, {8}]
    nine = next(c for c in loop.population if _level(c) == 9)
    assert _level(next(c for c in loop.population if c.id == nine.parent)) == 8
    # Every accepted candidate, the failed one included, ranked by val (train ranks
    # them 5, 9 and 8 last).
    assert [_level(c) for c in loop.leaderboard()] == [9, 8, 5]

    first = ScriptedLevels([[(5, 8)]])
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path / "random", val_gate=True, epochs=1, edit_preset="knob_only")
        loop.optimizer = first
        loop.run()
        eight = next(c for c in loop.population if _level(c) == 8)
        resumed = _loop(tmp_path / "random", val_gate=True, epochs=2, resume=True,
                        optimizer="random", batch_size=3, edit_preset="knob_only", attempts=10)
        resumed.run()
    assert eight.id in resumed._gate_accepted and not eight.exit_ok
    epoch2 = [c for c in resumed.population if c.id not in {x.id for x in loop.population}]
    assert epoch2 and {c.parent for c in epoch2} == {eight.id}


def test_val_gate_report_and_adopt_name_the_same_winner(tmp_path, monkeypatch):
    """A candidate whose train run failed wins on val: the report, the held-out
    audit and adopt all name it, the report gives its val score as best_score, and
    adopt's winner passes adopt's held-out check. (The toy package cannot be
    written out as a package, so adopt's selection and check are called directly.)"""
    from refactory_dsh.adopt import _check_holdout, best

    _fail_train_at(monkeypatch, 8)
    loop, _, report = _gated(tmp_path, monkeypatch, [[(5, 8)], [(8, 3)]], epochs=2)
    eight = next(c for c in loop.population if _level(c) == 8)
    assert report["best"]["id"] == eight.id and report["exit_ok"] is True, report["error"]
    assert report["best_score"] == pytest.approx(0.8)
    assert report["best_score_basis"].startswith("val")
    assert report["best_train_score"] is None
    holdout = json.loads((loop.root / "holdout.json").read_text())
    assert holdout["winner"] == eight.id and holdout["status"] == "measured"
    winner = best(loop.root)
    assert winner["candidate"] == eight.id and winner["score"] == pytest.approx(0.8)
    assert winner["score_basis"] == "val" and "train run failed" in winner["score_note"]
    _check_holdout(loop.root, winner["candidate"], allow_overfit=False)


def test_val_gate_winner_skips_a_candidate_a_person_rejected(tmp_path, monkeypatch):
    """A person rejects the gate's best (8): the report, holdout.json and adopt all
    name the best on val among what is left (7), with its val score as best_score;
    a reject recovered after the search reports the same best_score basis."""
    from refactory_dsh.adopt import _check_holdout, best
    from refactory_dsh.control import Control
    from refactory_dsh.finished_control import recover_selection

    loop, _, _ = _gated(tmp_path / "live", monkeypatch, [[(5, 7)], [(7, 8)]], epochs=2)
    by_level = {_level(c): c for c in loop.population}
    with _patch_mode(LevelProbe):
        loop._honor({"action": "reject", "target": by_level[8].id})
        report = loop.report()
    assert report["best"]["id"] == by_level[7].id and report["exit_ok"] is True, report["error"]
    assert report["best_score"] == pytest.approx(0.7) and report["best_score_basis"].startswith("val")
    assert json.loads((loop.root / "holdout.json").read_text())["winner"] == by_level[7].id
    winner = best(loop.root)
    assert winner["candidate"] == by_level[7].id and winner["score"] == pytest.approx(0.7)
    _check_holdout(loop.root, winner["candidate"], allow_overfit=False)

    loop, _, _ = _gated(tmp_path / "finished", monkeypatch, [[(5, 7)], [(7, 8)]], epochs=2)
    by_level = {_level(c): c for c in loop.population}
    Control(loop.root).path.write_text(json.dumps(
        {"action": "reject", "target": by_level[8].id, "at": 1.0}))
    with _patch_mode(LevelProbe):
        assert recover_selection(loop.root, project=Path(".")) is not None
    recovered = json.loads((loop.root / "report.json").read_text())
    assert recovered["best"]["id"] == by_level[7].id
    assert recovered["best_score"] == pytest.approx(0.7)
    assert recovered["best_score_basis"].startswith("val")
    assert recovered["best_train_score"] == pytest.approx(by_level[7].score)
    assert best(loop.root)["candidate"] == by_level[7].id


def test_val_gate_scores_the_last_epochs_proposals(tmp_path, monkeypatch):
    """Two epochs gate both proposed batches: the one proposed in the last epoch is
    evaluated and gated before the run ends, and can win."""
    loop, _, report = _gated(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)]], epochs=2)
    assert report["unevaluated"] == []
    eight = next(c for c in loop.population if _level(c) == 8)
    assert eight.evaluated and eight.id in loop._gate_accepted
    assert report["best"]["id"] == eight.id


def test_val_gate_early_stop_after_patience_stale_epochs(tmp_path, monkeypatch):
    """SkillOpt's patience: an epoch whose gating does not raise the best val
    score is stale; two in a row stop the run before it proposes again. A resumed
    run stops for the same reason without another epoch."""
    plan = [[(5, 7)], [(7, 6)], [(7, 4)], [(7, 3)], [(7, 2)]]
    loop, proposer, report = _gated(tmp_path, monkeypatch, plan, epochs=10, early_stop_patience=2)
    # Epoch 1 gates the seed; 2 gates 7 (best rises); 3 and 4 gate 6 and 4 (stale).
    assert report["bound"] == "patience_exhausted"
    assert loop.epoch == 4
    assert len(proposer.handed) == 3, "the stopping epoch proposes nothing"
    assert report["unevaluated"] == []

    again = ScriptedLevels([[(7, 1)]])
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, val_gate=True, epochs=10, early_stop_patience=2, resume=True)
        resumed.optimizer = again
        report = resumed.run()
    assert report["bound"] == "patience_exhausted"
    assert again.handed == [] and resumed.epoch == 4


def test_val_gate_early_stop_at_full_val(tmp_path, monkeypatch):
    """early_stop_at_full_val stops the run once the best val score is 1.0."""
    plan = [[(5, 9)], [(9, 8)], [(9, 7)]]
    loop, proposer, report = _gated(
        tmp_path, monkeypatch, plan, epochs=10, early_stop_at_full_val=True,
        val=lambda run, cand: 1.0 if _level(cand) == 9 else _level(cand) / 10)
    assert report["bound"] == "perfect_val"
    assert loop.epoch == 2 and len(proposer.handed) == 1


def test_full_val_seed_still_gets_one_batch_gated(tmp_path, monkeypatch):
    """SkillOpt checks the early stop at an epoch's end: a seed already at 1.0 on
    val still gets one batch proposed and gated before perfect_val stops the run."""
    loop, proposer, report = _gated(
        tmp_path, monkeypatch, [[(5, 6)], [(5, 4)]], epochs=10, early_stop_at_full_val=True,
        val=lambda run, cand: 1.0 if _level(cand) == 5 else _level(cand) / 10)
    assert report["bound"] == "perfect_val"
    assert len(proposer.handed) == 1 and loop.epoch == 2
    six = next(c for c in loop.population if _level(c) == 6)
    assert six.id in loop._gate_rejected


def test_val_gate_budget_endings_skip_the_final_scoring(tmp_path, monkeypatch):
    """As SkillOpt: a spent evaluation budget, or a time/token budget a measurement
    found spent, ends the search without scoring the last proposals or running the
    final evaluation; the report keeps the real reason and says why it skipped."""
    tests = _record_test_rollouts(monkeypatch)
    loop, _, report = _gated(tmp_path / "evals", monkeypatch, [[(5, 7)], [(7, 8)]],
                             epochs=5, max_evaluations=2)
    assert report["bound"] == "evaluations"
    assert report["strategy_report"]["test_eval"] == "skipped"
    assert "'evaluations'" in report["strategy_report"]["test_eval_reason"]
    assert [_level(c) for c in loop.population if not c.evaluated] == [8]
    assert tests == []

    spent = "the time budget is spent"
    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        if part == "train" and _level(candidate) == 7:
            self.bound = spent
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)
    loop, _, report = _gated(tmp_path / "time", monkeypatch, [[(5, 7)], [(7, 8)]], epochs=2)
    assert report["bound"] == spent
    assert report["strategy_report"]["test_eval"] == "skipped"
    assert spent in report["strategy_report"]["test_eval_reason"]
    assert [_level(c) for c in loop.population if not c.evaluated] == [8]
    assert tests == []


def test_a_budget_stop_wins_over_an_early_stop_in_the_same_epoch(tmp_path, monkeypatch):
    """The epoch whose gating exhausts the patience also finds the time budget
    spent: the budget is the stop reason, and nothing more is scored."""
    tests = _record_test_rollouts(monkeypatch)
    spent = "the time budget is spent"
    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        if part == "train" and _level(candidate) == 4:
            self.bound = spent
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)
    loop, _, report = _gated(tmp_path, monkeypatch, [[(5, 7)], [(7, 6)], [(7, 4)], [(7, 3)]],
                             epochs=10, early_stop_patience=2)
    assert loop.epoch == 4, "setup: the patience is exhausted in the epoch that spent the budget"
    assert report["bound"] == spent
    assert report["strategy_report"]["test_eval"] == "skipped"
    assert tests == []


def test_early_stop_needs_the_val_gate(tmp_path):
    with _patch_mode(LevelProbe):
        with pytest.raises(LoopError, match="has no validation gate"):
            _loop(tmp_path, early_stop_patience=2, val_gate=False).run()


def test_early_stop_settings_come_from_the_command_line_and_are_saved(tmp_path, monkeypatch):
    """--early-stop-patience N --early-stop-at-full-val 1; the run's config.json keeps
    them, and a resume reads its settings from there."""
    import argparse

    from refactory_dsh import profiles

    parser = argparse.ArgumentParser()
    profiles.add_arguments(parser)
    args = parser.parse_args(["--early-stop-patience", "3", "--early-stop-at-full-val", "1"])
    settings = profiles.normalize({key: value for key, value in vars(args).items()
                                   if key in profiles.defaults() and value is not None})
    assert settings["early_stop_patience"] == 3
    assert settings["early_stop_at_full_val"] is True
    for bad in ({"early_stop_patience": -1}, {"early_stop_patience": True},
                {"early_stop_at_full_val": 2}):
        with pytest.raises(profiles.ProfileError, match="early_stop"):
            profiles.normalize(bad)
    loop, _, _ = _gated(tmp_path, monkeypatch, [[(5, 7)], []], epochs=2,
                        early_stop_patience=3, early_stop_at_full_val=True)
    restored = LoopConfig.from_dict(json.loads((loop.root / "config.json").read_text()))
    assert restored.early_stop_patience == 3 and restored.early_stop_at_full_val is True


def test_val_gate_runs_the_final_evaluation_and_the_audit_sees_test(tmp_path, monkeypatch):
    """With a test part: the final selection, then the seed and the best on test
    with the run seed, recorded in the report as SkillOpt records them; the audit
    compares the winner's val and test, so a --holdout 0 run is measured and passes."""
    tests = _record_test_rollouts(monkeypatch)
    loop, _, report = _gated(tmp_path, monkeypatch, [[(5, 7)], [(7, 6)]], epochs=2)
    figures = report["strategy_report"]
    assert figures["test_eval"] == "done"
    assert sorted(tests) == [(5, 7), (7, 7)], "the final candidate is the best: its score is reused"
    # LevelProbe scores 1 - level/10 on every part; the gate saw level/10.
    assert figures["final_selection"] == pytest.approx(0.7)
    assert figures["baseline_test"] == pytest.approx(0.5)
    assert figures["best_test"] == pytest.approx(0.3) == figures["final_test"]
    assert figures["best_test_delta"] == pytest.approx(-0.2) == figures["final_test_delta"]
    assert report["holdout"]["status"] == "measured"
    assert report["holdout"]["test"] == pytest.approx(0.3)
    assert report["exit_ok"] is True, report["error"]
    steps = [(e["step_type"], e.get("action")) for e in _journal_events(loop, "final_evaluation")]
    assert steps == [("final_selection_eval", "reused"), ("test_eval", "baseline_test"),
                     ("test_eval", "best_test"), ("test_eval", "final_test_reused")]


def test_val_gate_without_a_test_part_records_why_it_skipped(tmp_path, monkeypatch):
    tests = _record_test_rollouts(monkeypatch)
    _, _, report = _gated(tmp_path, monkeypatch, [[(5, 7)], []], epochs=2,
                          parts_manifest=str(_write_parts(tmp_path / "no-test")))
    figures = report["strategy_report"]
    assert figures["test_eval"] == "skipped"
    assert figures["test_eval_reason"] == "the parts manifest has no 'test' part"
    assert tests == []


def test_a_resumed_run_continues_its_final_evaluation(tmp_path, monkeypatch):
    """A run that dies after the final selection resumes into the test phases
    without repeating it."""
    original = Loop._save_strategy_state
    died: list[bool] = []

    def save(self):  # type: ignore[no-untyped-def]
        original(self)
        completed = (self._strategy_state.get("final_evaluation") or {}).get("completed_phases") or []
        if "final_selection" in completed and not died:
            died.append(True)
            raise ProcessDied()

    monkeypatch.setattr(Loop, "_save_strategy_state", save)
    with pytest.raises(ProcessDied):
        _gated(tmp_path, monkeypatch, [[(5, 7)], []], epochs=2)
    tests = _record_test_rollouts(monkeypatch)
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, val_gate=True, epochs=2, resume=True)
        resumed.optimizer = ScriptedLevels([])
        report = resumed.run()
    assert report["strategy_report"]["test_eval"] == "done"
    assert sorted(tests) == [(5, 7), (7, 7)]
    steps = [e["step_type"] for e in _journal_events(resumed, "final_evaluation")]
    assert steps.count("final_selection_eval") == 1
