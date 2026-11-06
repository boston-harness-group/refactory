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
    INITIAL_SKILL, MODE_ID, TASK_REF, SkillProbe, SkillTask, _patch_mode, _write_parts,
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


def _loop(tmp_path: Path, strategy: OuterLoop | None = None, **overrides: Any) -> Loop:
    base: dict[str, Any] = dict(
        epochs=2, batch_size=1, optimizer="random", seed=7, runs_root=tmp_path / "run",
        allowed_ops=("set_knob",), max_evaluations=60, holdout=0.0,
        parts_manifest=str(_write_parts(tmp_path, test=True)),
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


def test_map_elites_runs_with_the_default_loops_val_gate(tmp_path, monkeypatch):
    """--optimizer map_elites with the default loop and val_gate on: every
    candidate evaluated on train is rolled out on val once, a child stays a parent
    only when its val score strictly beats its parent's, and the winner is the
    best on val. A resumed run keeps the gate's decisions and rolls nothing again."""
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    val_rollouts: list[str] = []
    original = Run.evaluate

    def evaluate(self, candidate, *, part="train", items=None, seed=None):  # type: ignore[no-untyped-def]
        if part == "val":
            assert seed == 7
            val_rollouts.append(candidate.id)
        return original(self, candidate, part=part, items=items, seed=seed)

    monkeypatch.setattr(Run, "evaluate", evaluate)
    overrides = dict(optimizer="map_elites", val_gate=True, epochs=3, batch_size=3)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **overrides)
        report = loop.run()
    assert report["optimizer"] == "map_elites"
    by_id = {c.id: c for c in loop.population}
    accepted, rejected = loop._gate_accepted, loop._gate_rejected
    assert accepted[0] == loop.population[0].id, "the seed is the gate's start"
    judged = accepted[1:] + rejected
    assert accepted[1:] and rejected, "setup: the search both improves and regresses"
    for cid in judged:
        child, parent = by_id[cid], by_id[by_id[cid].parent]
        assert (cid in accepted) == (_level(child) > _level(parent)), (cid, _level(child), _level(parent))
        assert parent.id in accepted, "only an accepted candidate is a parent"
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


def test_val_gate_needs_a_val_part_and_the_default_loop(tmp_path):
    no_val = tmp_path / "noval.json"
    no_val.write_text(json.dumps({"train": ["t1", "t2"]}))
    with _patch_mode(LevelProbe):
        with pytest.raises(LoopError, match="'val' part"):
            _loop(tmp_path, val_gate=True, parts_manifest=str(no_val)).run()
        with pytest.raises(LoopError, match="keeps its own"):
            _loop(tmp_path / "custom", KeepThenDie(), val_gate=True).run()


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


def test_the_default_gate_compares_a_child_with_its_parent(tmp_path, monkeypatch):
    """Seed 5 → 7 (accepted, the best), then 5 → 6: it beats its parent 5 though
    not the best 7, so it is accepted; then 7 → 2 is rejected and is never handed
    to the proposer as a parent."""
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: _level(cand) / 10))
    proposer = ScriptedLevels([[(5, 7)], [(5, 6)], [(7, 2)], []])
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=4, batch_size=1)
        loop.optimizer = proposer
        loop.config.val_gate = True
        report = loop.run()
    by_id = {c.id: c for c in loop.population}
    assert [_level(by_id[cid]) for cid in loop._gate_accepted] == [5, 7, 6]
    assert [_level(by_id[cid]) for cid in loop._gate_rejected] == [2]
    assert proposer.handed[-1] == {5, 7, 6}, "the rejected candidate is not a parent"
    assert _level(by_id[report["best"]["id"]]) == 7
