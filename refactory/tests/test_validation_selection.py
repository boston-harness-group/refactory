"""Validation as the fitness, for every strategy (``LoopConfig.selection``).

Train is the proposer's feedback, validation decides every choice (parents, the pool,
the winner) and test is scored once at the end. ``single`` is SkillOpt's line and
reproduces the recorded gate decisions; ``pool`` keeps the best candidates on
validation as parents. Each test fails when the rule it names is reverted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import profiles, strategies
from refactory_dsh.loop import NO_VALIDATION, Loop, LoopConfig, LoopError, default_selection
from refactory_dsh.optim import Candidate, Optimizer, Proposal, ProposalContext
from refactory_dsh.optim.policies import (
    ArchiveSelector, EvolutionarySelector, IslandsSelector, MapElitesSelector,
)
from refactory_dsh.optim.random_optimizer import _elite
from refactory_dsh.outerloop import Run, ValidationGate

from test_skillopt import MODE_ID, _patch_mode, _write_parts
from test_val_gate import LevelProbe

FIXTURE = Path(__file__).parent / "fixtures" / "livemath_gate_decisions.json"


class FitProbe(LevelProbe):
    """LevelProbe's graph with a level knob wide enough to name every scripted
    candidate; its train score is ``TRAIN[level]`` (0.0 when not listed)."""

    name = "fit_probe"
    TRAIN: dict[int, float] = {}

    @classmethod
    def declare(cls):  # type: ignore[no-untyped-def]
        from refactory_dsh import nn

        return {"level": nn.Knob(default=0, kind="int", lo=0, hi=99, description="a level")}

    def score(self, run):  # type: ignore[no-untyped-def]
        return float(FitProbe.TRAIN.get(int(self.values()["level"]), 0.0))


def _level(candidate: Candidate) -> int:
    return int(candidate.state["knobs"]["level"]["value"])


class Scripted(Optimizer):
    """Each round, ``(parent level, child level)`` moves. A parent the loop did not hand
    over fails the round (StopIteration), so the plan also checks the parent set."""

    name = "scripted"

    def __init__(self, plan: list[list[tuple[int, int]]], *, model: bool = False) -> None:
        super().__init__(k=1)
        self.plan = list(plan)
        self.handed: list[list[int]] = []
        self.prompts: list[str] = []
        self.uses_model = model

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.handed.append([_level(p) for p in parents])
        if self.uses_model:
            self.prompts.append(ctx.render())
        moves = self.plan.pop(0) if self.plan else []
        return [Proposal(edit={"op": "set_knob", "name": "level", "value": child},
                         parent=next(p for p in parents if _level(p) == parent).id)
                for parent, child in moves]


def _config(tmp_path: Path, **overrides: Any) -> LoopConfig:
    base: dict[str, Any] = dict(
        epochs=2, batch_size=1, optimizer="random", seed=7, runs_root=tmp_path / "run",
        allowed_ops=("set_knob",), max_evaluations=200, holdout=0.0, val_gate=True,
        parts_manifest=str(_write_parts(tmp_path, test=True)),
    )
    base.update(overrides)
    return LoopConfig(**base)


def _scores(monkeypatch: Any, train: dict[int, float], val: dict[int, float]) -> None:
    monkeypatch.setattr(FitProbe, "TRAIN", dict(train))
    monkeypatch.setattr(ValidationGate, "val_score",
                        staticmethod(lambda run, cand: float(val.get(_level(cand), 0.0))))


def _run(tmp_path: Path, monkeypatch: Any, plan: list[list[tuple[int, int]]], *,
         strategy: str = "default", model: bool = False, **overrides: Any):
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    proposer = Scripted(plan, model=model)
    with _patch_mode(FitProbe):
        loop = strategies.create(strategy, MODE_ID, _config(tmp_path, **overrides),
                                 project=Path("."))
        loop.optimizer = proposer
        report = loop.run()
    return loop, proposer, report


def _levels(loop: Loop, ids: Any) -> list[int]:
    by_id = {c.id: c for c in loop.population}
    return [_level(by_id[cid]) for cid in ids]


# ── pool: entry, ties, the incumbent and the final pick ───────────────────

#: val and train per level, for the pool scenario: 1 and 2 tie on val above the seed
#: (2 wins on train), 4 ties the incumbent on val with the best train score (and must
#: not replace it), 5 is strictly higher.
POOL_VAL = {0: 0.5, 1: 0.6, 2: 0.6, 3: 0.4, 4: 0.6, 5: 0.7}
POOL_TRAIN = {0: 0.5, 1: 0.3, 2: 0.4, 3: 0.95, 4: 0.9, 5: 0.1}
POOL_PLAN = [[(0, 1), (0, 2), (0, 3)], [(1, 4)], [(4, 5)]]


def test_pool_keeps_the_top_k_on_val_with_ties_and_the_incumbent_moves_only_strictly(
        tmp_path, monkeypatch):
    """pool_size 2. Round 1: 1 and 2 tie on val above the seed; 2 has the higher train
    score, so it is the incumbent; the pool is {2, 1} (the seed and 3 are out). Round 2
    breeds from 1, a pool member that is not the incumbent: its child 4 ties the
    incumbent on val with a far better train score, so the incumbent stays 2 and the
    pool holds 2, 4 and 1 (three: ties at the edge are kept). Round 3: 5 is strictly
    higher and becomes the incumbent and the winner."""
    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    loop, proposer, report = _run(tmp_path, monkeypatch, POOL_PLAN, epochs=3, batch_size=3,
                                  selection="pool", pool_size=2)
    assert proposer.handed == [[0], [2, 1], [2, 4, 1]]
    assert _levels(loop, loop._gate_accepted) == [0, 2, 5], "the incumbent's line"
    assert _levels(loop, [c.id for c in loop.leaderboard()]) == [5, 4, 2, 1]
    assert _level(next(c for c in loop.population if c.id == report["best"]["id"])) == 5
    assert report["best_score"] == pytest.approx(0.7)
    assert report["strategy_report"]["test_eval"] == "done"


def test_pool_winner_is_the_incumbent_not_a_val_tie_with_a_better_train_score(
        tmp_path, monkeypatch):
    """Stopped after round 2: 4 ties the incumbent 2 on val and beats it on train, but
    the incumbent changes only on a strictly higher val score, so 2 is the winner, the
    leaderboard's first, the final evaluation's best and adopt's choice."""
    from refactory_dsh.adopt import best

    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    loop, _, report = _run(tmp_path, monkeypatch, POOL_PLAN[:2], epochs=2, batch_size=3,
                           selection="pool", pool_size=2)
    two = next(c for c in loop.population if _level(c) == 2)
    assert report["best"]["id"] == two.id == loop.leaderboard()[0].id
    assert loop._gate().best_id == two.id
    assert _levels(loop, [c.id for c in loop.leaderboard()]) == [2, 4, 1]
    assert best(loop.root)["candidate"] == two.id
    # A person rejects the incumbent: the report's leaderboard and adopt both fall to
    # the next of the pool, 4 (val tie, the higher train score), which was never accepted.
    with (loop.root / "journal.jsonl").open("a") as journal:
        journal.write(json.dumps({"event": "control", "kept": [], "rejected": [two.id]}) + "\n")
    loop.rejected = [two.id]
    four = next(c for c in loop.population if _level(c) == 4)
    assert loop.leaderboard()[0].id == four.id == best(loop.root)["candidate"]


def test_single_keeps_skillopts_line_on_the_same_scores(tmp_path, monkeypatch):
    """The same scores under single: the incumbent is the only parent, so a plan that
    breeds from 1 (a near miss) finds no such parent and its round proposes nothing;
    breeding from the incumbent gives the same incumbent line as pool."""
    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    loop, proposer, report = _run(tmp_path / "a", monkeypatch, POOL_PLAN, epochs=3,
                                  batch_size=3, selection="single")
    assert proposer.handed[:2] == [[0], [2]]
    assert 4 not in {_level(c) for c in loop.population}
    assert report["proposer"]["failure_count"] >= 1
    loop, proposer, report = _run(tmp_path / "b", monkeypatch,
                                  [[(0, 1), (0, 2), (0, 3)], [(2, 4)], [(2, 5)]],
                                  epochs=3, batch_size=3, selection="single")
    assert proposer.handed == [[0], [2], [2]]
    assert _levels(loop, loop._gate_accepted) == [0, 2, 5]
    assert _levels(loop, [c.id for c in loop.leaderboard()]) == [5, 2, 0]


def test_the_report_gives_pool_diversity_and_the_budget(tmp_path, monkeypatch):
    """The pool scenario's report: the pool's lines of descent (the seed's children 1
    and 2 start two lines; 4 and 5 descend from 1), its edit kinds, the share of
    children bred from the incumbent (1, 2 and 3 of 5), and the scorings."""
    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    loop, _, report = _run(tmp_path, monkeypatch, POOL_PLAN, epochs=3, batch_size=3,
                           selection="pool", pool_size=2)
    selection = report["selection"]
    assert selection["setting"] == "pool" and selection["pool_size"] == 2
    assert _levels(loop, selection["pool"]) == [5, 4, 2, 1]
    assert selection["lines_of_descent"] == 2
    assert selection["largest_line_share"] == pytest.approx(0.75)
    assert selection["edit_kinds"] == {"prompt_knob": 4}
    assert selection["children"] == 5
    assert selection["parents_from_incumbent"] == pytest.approx(0.6)
    budget = selection["budget"]
    assert budget["val_scorings"] == 6, "each candidate on val once"
    assert budget["test_scorings"] == 2, "the seed and the winner, at the end"
    assert budget["train_scorings"] == report["evaluations"]
    assert {"tokens", "elapsed_s", "rollouts"} <= set(budget)
    assert "## Selection" in (loop.root / "report.md").read_text()


# ── which setting a run gets ──────────────────────────────────────────────


def _manifest(tmp_path: Path, **parts: list[str]) -> str:
    path = tmp_path / "parts-custom.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(parts))
    return str(path)


def test_validation_selection_is_on_by_default_when_the_run_has_val_items(tmp_path):
    """No --val-gate: a parts manifest with val items turns the gate on (single for
    the default loop), config.json records it, adopt and the report agree, and the
    audit says its items are the val part the search selected on when there is no
    test part."""
    with _patch_mode(FitProbe):
        loop = Loop(MODE_ID, _config(tmp_path, val_gate=None, parts_manifest=_manifest(
            tmp_path, train=["t1", "t2", "t3"], val=["v1", "v2"])), project=Path("."))
        report = loop.run()
    assert loop.config.val_gate is True and loop.selection_mode() == "single"
    assert json.loads((loop.root / "config.json").read_text())["val_gate"] is True
    assert report["selection"]["val_gate_basis"].startswith("default")
    assert "warning" not in report["selection"]
    assert report["best_score_basis"].startswith("val")
    holdout = report["holdout"]
    assert holdout["split"] == "val" and holdout["independent"] is False
    assert "not an independent check" in (loop.root / "report.md").read_text()


def test_val_gate_0_turns_validation_selection_off(tmp_path):
    with _patch_mode(FitProbe):
        loop = Loop(MODE_ID, _config(tmp_path, val_gate=False), project=Path("."))
        report = loop.run()
    assert loop.config.val_gate is False and loop.selection_mode() == ""
    assert report["selection"]["setting"] == "train"
    assert report["selection"]["warning"].startswith("val_gate 0")
    assert report["holdout"]["independent"] is True, "train selected: val is unseen"


@pytest.mark.parametrize("where", ["manifest without val", "no manifest"])
def test_a_run_without_val_items_selects_on_train_and_warns(tmp_path, where):
    """No val items (a manifest with none, or only the --holdout fraction): train
    selection, with the warning in the run's events, its strategy log and its report.
    The --holdout fraction stays the audit's independent split."""
    parts = (_manifest(tmp_path, train=["t1", "t2", "t3"], test=["x1"])
             if where == "manifest without val" else None)
    with _patch_mode(FitProbe):
        loop = Loop(MODE_ID, _config(tmp_path, val_gate=None, parts_manifest=parts,
                                     holdout=0.3), project=Path("."))
        report = loop.run()
    assert loop.config.val_gate is False
    assert report["selection"]["warning"] == NO_VALIDATION
    assert NO_VALIDATION in (loop.root / "strategy.log").read_text()
    assert NO_VALIDATION in (loop.root / "report.md").read_text()
    steps = [json.loads(line) for line in (loop.root / "progress.jsonl").read_text().splitlines()]
    assert any(s.get("step") == "init" and s.get("warning") == NO_VALIDATION for s in steps)
    if where == "no manifest":
        assert report["holdout"]["split"] == "holdout" and report["holdout"]["independent"]


def test_explicit_selection_without_val_items_is_refused(tmp_path):
    with _patch_mode(FitProbe):
        with pytest.raises(LoopError, match="has none"):
            Loop(MODE_ID, _config(tmp_path, val_gate=None, selection="pool", parts_manifest=_manifest(
                tmp_path, train=["t1", "t2"])), project=Path(".")).run()


def test_a_custom_strategy_keeps_its_own_gate_by_default(tmp_path):
    """SkillOpt (a custom strategy) with val_gate unset is not refused: the default
    leaves the loop's gate off for it, as an explicit 1 is refused."""
    from refactory_dsh.outerloop import OuterLoop

    class Nothing(OuterLoop):
        name = "nothing"

        def strategy(self, run):  # type: ignore[no-untyped-def]
            run.stop("done")

    with _patch_mode(FitProbe):
        loop = Loop(MODE_ID, _config(tmp_path, val_gate=None), project=Path("."))
        strategy = Nothing()
        strategy.configure()
        loop.set_custom_strategy(strategy)
        loop.run()
    assert loop.config.val_gate is False and loop.selection_warning == ""


def test_profiles_and_the_command_line_leave_val_gate_unset_unless_given():
    import argparse

    assert profiles.defaults()["val_gate"] is None
    assert profiles.normalize({})["val_gate"] is None
    assert profiles.normalize({"val_gate": 0})["val_gate"] is False
    assert profiles.normalize({"val_gate": 1})["val_gate"] is True
    parser = argparse.ArgumentParser()
    profiles.add_arguments(parser)
    assert "val_gate" not in profiles.cli_overrides(parser.parse_args([]))
    assert profiles.cli_overrides(parser.parse_args(["--val-gate", "0"]))["val_gate"] == 0


def test_defaults_single_for_the_skillopt_matched_loop_and_pool_for_everything_else(tmp_path):
    assert default_selection("default", "autoresearch") == "single"
    assert default_selection("default", "random") == "single"
    assert default_selection("default", "hill_climb") == "single"
    assert default_selection("default", "autoresearch", "hill_climb") == "single"
    for optimizer in ("archive", "evolutionary", "islands", "map_elites"):
        assert default_selection("default", optimizer) == "pool"
        assert default_selection("default", "autoresearch", optimizer) == "pool"
    for name in strategies.names():
        if name != "default":
            assert default_selection(name, "autoresearch") == "pool", name
    with _patch_mode(FitProbe):
        assert Loop(MODE_ID, _config(tmp_path)).selection_mode() == "single"
        assert Loop(MODE_ID, _config(tmp_path, selection="pool")).selection_mode() == "pool"
        assert strategies.create("reflective", MODE_ID, _config(tmp_path),
                                 project=Path(".")).selection_mode() == "pool"
        assert strategies.create("tree_search", MODE_ID, _config(tmp_path, selection="single"),
                                 project=Path(".")).selection_mode() == "single"
        assert Loop(MODE_ID, _config(tmp_path, val_gate=False)).selection_mode() == ""


def test_selection_needs_the_gate_and_a_known_value(tmp_path):
    with pytest.raises(profiles.ProfileError, match="val_gate"):
        profiles.normalize({"selection": "pool", "val_gate": False})
    assert profiles.normalize({"selection": "pool"})["val_gate"] is None, "unset: the run decides"
    with pytest.raises(profiles.ProfileError, match="unknown selection"):
        profiles.normalize({"val_gate": True, "selection": "pareto"})
    with pytest.raises(profiles.ProfileError, match="pool_size"):
        profiles.normalize({"val_gate": True, "pool_size": 0})
    with _patch_mode(FitProbe):
        with pytest.raises(LoopError, match="unknown selection"):
            Loop(MODE_ID, _config(tmp_path, selection="pareto"))
        with pytest.raises(LoopError, match="needs|only the validation gate"):
            Loop(MODE_ID, _config(tmp_path, val_gate=False, selection="pool")).run()


@pytest.mark.parametrize("optimizer", ["archive", "evolutionary", "islands", "map_elites"])
def test_population_optimizers_and_pickers_run_with_the_gate(tmp_path, optimizer):
    """The refusal is gone: a population optimizer, or the same policy as a parent
    picker, runs with val_gate (pool by default) through the final evaluation."""
    assert profiles.normalize({"val_gate": True, "optimizer": optimizer})["optimizer"] == optimizer
    assert profiles.normalize({"val_gate": True, "parent_picker": optimizer})["val_gate"] is True
    with _patch_mode(FitProbe):
        loop = Loop(MODE_ID, _config(tmp_path, optimizer=optimizer, batch_size=2,
                                     edit_preset="knob_only"), project=Path("."))
        report = loop.run()
    assert loop.selection_mode() == "pool"
    assert report["strategy_report"]["test_eval"] == "done"
    assert report["best"]["id"] == loop._gate().current_id


# ── the policies rank by val ──────────────────────────────────────────────


def _cands(*levels: int, train: dict[int, float | None]) -> list[Candidate]:
    out = []
    for level in levels:
        candidate = Candidate.from_state({"graph": {"nodes": {}}, "knobs": {
            "level": {"value": level}}})
        candidate.score, candidate.evaluated, candidate.exit_ok = train.get(level), True, True
        out.append(candidate)
    return out


def _ctx(val: dict[str, float], **fields: Any) -> ProposalContext:
    return ProposalContext(epoch=1, generation=1, mode_name="m", val_scores=val, **fields)


#: Train ranks 1 > 2 > 3; val ranks 3 > 2 > 1.
TRAIN_UP = {1: 0.9, 2: 0.5, 3: 0.1}


def _val(cands: list[Candidate]) -> dict[str, float]:
    return {c.id: _level(c) / 10 for c in cands}


def test_archive_ranks_by_val_and_rebuilds_from_the_parents_it_is_handed():
    one, two, three = cands = _cands(1, 2, 3, train=TRAIN_UP)
    policy = ArchiveSelector(seed=0)
    assert [_level(c) for c in policy.select(_ctx(_val(cands)), 3, cands)] == [3, 2, 1]
    # The next round's pool no longer holds 3: the archive does not keep it.
    assert [_level(c) for c in policy.select(_ctx(_val(cands)), 3, [one, two])] == [2, 1]
    assert three.id not in policy.members
    # Without the gate, train as before.
    assert [_level(c) for c in policy.select(_ctx({}), 3, cands)] == [1, 2, 3]


def test_evolutionary_tournament_is_won_on_val():
    cands = _cands(1, 2, 3, train=TRAIN_UP)
    policy = EvolutionarySelector(seed=0, tournament=3)
    assert {_level(c) for c in policy.select(_ctx(_val(cands)), 4, cands)} == {3}
    assert {_level(c) for c in policy.select(_ctx({}), 4, cands)} == {1}


def test_islands_breed_the_best_of_each_island_on_val():
    cands = _cands(1, 2, 3, train=TRAIN_UP)
    policy = IslandsSelector(seed=0, count=1)
    assert [_level(c) for c in policy.select(_ctx(_val(cands)), 2, cands)] == [3, 3]
    assert [_level(c) for c in IslandsSelector(seed=0, count=1).select(_ctx({}), 1, cands)] == [1]


def test_map_elites_champions_are_chosen_on_val_and_cells_rebuilt_from_the_pool():
    one, two, three = cands = _cands(1, 2, 3, train=TRAIN_UP)
    policy = MapElitesSelector(seed=0)
    assert {_level(c) for c in policy.select(_ctx(_val(cands)), 3, cands)} == {3}
    # 3 left the pool: the cell's champion is the best of what is handed over now.
    assert {_level(c) for c in policy.select(_ctx(_val(cands)), 3, [one, two])} == {2}
    assert {_level(c) for c in policy.select(_ctx({}), 3, cands)} == {1}


def test_random_elite_ranks_by_val_and_keeps_an_incumbent_whose_train_run_failed():
    failed, two, three = cands = _cands(9, 2, 3, train={9: None, 2: 0.9, 3: 0.1})
    val = {failed.id: 0.9, two.id: 0.2, three.id: 0.3}
    assert [_level(c) for c in _elite(cands, _ctx(val, elite=2))] == [9, 3]
    assert [_level(c) for c in _elite(cands, _ctx({}, elite=2))] == [2, 3]


# ── strategies ────────────────────────────────────────────────────────────


def _settings_children(train_up_val_down: bool) -> tuple[list[Candidate], dict[str, float]]:
    """A seed and three set_knob children in one line, each with a higher train score
    than its parent and a lower (or equal) val score."""
    seed = Candidate.from_state({"graph": {"nodes": {}}, "knobs": {"level": {"value": 0}}})
    seed.score, seed.evaluated, seed.exit_ok = 0.1, True, True
    line, val = [seed], {seed.id: 0.9}
    for level in (1, 2, 3):
        parent = line[-1]
        child = Candidate.from_state(
            {"graph": {"nodes": {}}, "knobs": {"level": {"value": level}}}, parent=parent.id,
            edits=[*parent.edits, {"op": "set_knob", "name": "level", "value": level}])
        child.score, child.evaluated, child.exit_ok = 0.1 + level / 10, True, True
        val[child.id] = 0.9 - level / 100 if train_up_val_down else 0.9 + level / 100
        line.append(child)
    return line, val


def test_reflective_edit_kind_reads_val_gains_not_train():
    """Three prompt/knob children that each raised train but lowered val: on val the
    settings have stalled, so structure is next; on train they still pay. The decision's
    text, which the proposer reads, carries no val score."""
    from refactory_dsh.strategies import edit_kind_decision

    population, val = _settings_children(train_up_val_down=True)
    on_val = edit_kind_decision(population, scores=val)
    assert on_val["edit_kind"] == "pipeline"
    assert edit_kind_decision(population)["edit_kind"] == "prompt_knob"
    text = json.dumps(on_val)
    assert not any(f"{v:.4g}" in text or repr(v) in text for v in val.values())
    _, rising = _settings_children(train_up_val_down=False)
    settings = edit_kind_decision(population, scores=rising)
    assert settings["edit_kind"] == "prompt_knob"
    # This branch states the top and bottom means on train; on val it says they are hidden.
    text = json.dumps(settings)
    assert "mean 0." not in text and "not shown" in text


def test_reflective_chooses_the_edit_kind_on_the_gates_val_scores(tmp_path):
    """A line of prompt/knob children, each better on train and worse on val than its
    parent: under the gate reflective asks for structure (settings stopped paying on
    val); without the gate it would read train and ask for settings again."""
    levels = {0: (0.1, 0.9), 1: (0.2, 0.89), 2: (0.3, 0.88), 3: (0.4, 0.87)}
    loop = _judged_loop(tmp_path, "reflective", levels, current=0, chain=True)
    assert loop._edit_kind()["edit_kind"] == "pipeline"
    assert edit_kind_train(loop.population)["edit_kind"] == "prompt_knob"


def edit_kind_train(population: list[Candidate]) -> dict[str, Any]:
    from refactory_dsh.strategies import edit_kind_decision

    return edit_kind_decision(population)


def _judged_loop(tmp_path: Path, name: str, levels: dict[int, tuple[float | None, float]],
                 *, current: int, accepted: tuple[int, ...] = (), chain: bool = False,
                 **overrides: Any) -> Loop:
    """A strategy's loop holding judged candidates (level: (train, val)): children of
    the seed (level 0), or with ``chain`` a line of set_knob children each of the one
    before; the gate's incumbent is ``current``, accepted with the seed and ``accepted``."""
    with _patch_mode(FitProbe):
        loop = strategies.create(name, MODE_ID, _config(tmp_path, **overrides), project=Path("."))
    seed_state = loop.mode.state_dict()
    gate = loop._gate()
    for level, (train, val) in levels.items():
        state = json.loads(json.dumps(seed_state))
        state["knobs"]["level"]["value"] = level
        parent = (None if level == 0 else
                  loop.population[-1] if chain else loop.population[0])
        candidate = Candidate.from_state(
            state, parent=parent.id if parent is not None else None,
            edits=[] if parent is None else
            [*parent.edits, {"op": "set_knob", "name": "level", "value": level}])
        candidate.score, candidate.evaluated = train, True
        candidate.exit_ok = train is not None
        loop.population.append(candidate)
        gate.cache[candidate.id] = val
        (loop._gate_accepted if level in (0, current, *accepted)
         else loop._gate_rejected).append(candidate.id)
    incumbent = next(c for c in loop.population if _level(c) == current)
    gate.current_id = gate.best_id = incumbent.id
    gate.current_score = gate.best_score = gate.cache[incumbent.id]
    return loop


def test_reflective_merges_the_two_best_on_val_from_the_pool(tmp_path):
    """Under pool the merge pair is the two best on val (train ranks them last), and a
    merge round is a merge; under single the same run never merges."""
    levels = {0: (0.9, 0.1), 1: (0.8, 0.2), 2: (0.2, 0.6), 3: (0.1, 0.7)}
    loop = _judged_loop(tmp_path, "reflective", levels, current=3)
    loop.mutations = 2
    run = Run(_loop=loop)
    assert loop._merge_now(run)
    loop.strategy_decision = {"branch": "merge"}
    proposals = loop.propose(run, loop._gated_parents(), 1)
    left = next(c for c in loop.population if _level(c) == 3)
    right = next(c for c in loop.population if _level(c) == 2)
    assert proposals[0].parent == left.id
    assert proposals[0].rationale == f"merge {left.id} with {right.id}"
    single = _judged_loop(tmp_path / "single", "reflective", levels, current=3,
                          accepted=(1, 2), selection="single")
    single.mutations = 2
    assert len(single.leaderboard()) == 4, "setup: four accepted candidates"
    assert not single._merge_now(Run(_loop=single)), "a single line has nothing to merge"


def test_best_of_n_sends_its_samples_through_the_gate_and_has_a_winner(tmp_path, monkeypatch):
    """The samples are judged on val against the seed: the best on val (whose train
    score is the lowest) is the winner, and the final evaluation runs."""
    monkeypatch.setattr(ValidationGate, "val_score",
                        staticmethod(lambda run, cand: _level(cand) / 100))
    monkeypatch.setattr(FitProbe, "TRAIN", {level: 1 - level / 100 for level in range(100)})
    with _patch_mode(FitProbe):
        loop = strategies.create("best_of_n", MODE_ID, _config(tmp_path), project=Path("."))
        loop.width = 4
        report = loop.run()
    samples = [c for c in loop.population if c.parent]
    assert len(samples) >= 2 and loop.population[0].evaluated
    assert loop._gate_accepted[0] == loop.population[0].id, "the seed starts the gate"
    best = max([loop.population[0], *samples], key=_level)
    assert report["bound"] == "samples_exhausted"
    assert report["best"]["id"] == best.id == loop._gate().current_id
    assert report["strategy_report"]["test_eval"] == "done"
    assert report["unevaluated"] == []


def test_autoscientists_merges_onto_the_best_branch_on_val(tmp_path, monkeypatch):
    monkeypatch.setattr(ValidationGate, "val_score",
                        staticmethod(lambda run, cand: _level(cand) / 100))
    monkeypatch.setattr(FitProbe, "TRAIN", {level: 1 - level / 100 for level in range(100)})
    bases: list[Candidate] = []
    monkeypatch.setattr(strategies.Autoscientists, "_merge",
                        lambda self, produced, best: bases.append(best))
    with _patch_mode(FitProbe):
        loop = strategies.create("autoscientists", MODE_ID, _config(tmp_path), project=Path("."))
        loop.width = 3
        report = loop.run()
    branches = [c for c in loop.population if c.parent]
    assert bases and bases[0] is max(branches, key=_level)
    assert report["bound"] == "merged" and report["strategy_report"]["test_eval"] == "done"


def test_adaptive_meta_measures_the_stall_on_the_best_val(tmp_path):
    """Train keeps rising while the best val score does not: the meta branch fires,
    and its note to the proposer carries no val score."""
    levels = {0: (0.1, 0.5), 1: (0.2, 0.7), 2: (0.3, 0.6)}
    loop = _judged_loop(tmp_path, "adaptive_meta", levels, current=1)
    run = Run(_loop=loop)
    assert not loop._meta_now(run), "the first look sets the bar"
    better_train = next(c for c in loop.population if _level(c) == 2)
    better_train.score = 0.99
    assert loop._meta_now(run), "val has not risen, though train has"


def test_tree_search_debugs_a_failed_candidate_outside_the_pool(tmp_path):
    """A failed candidate the gate rejected (it is not in a pool of one) is still
    debugged, and the proposals go to it; improving names no val score."""
    levels = {0: (0.5, 0.5), 1: (None, 0.1), 2: (0.4, 0.8)}
    loop = _judged_loop(tmp_path, "tree_search", levels, current=2, pool_size=1)
    failed = next(c for c in loop.population if _level(c) == 1)
    assert failed not in loop._gated_parents()
    node = loop._decide(loop._gated_parents())
    assert loop.strategy_decision["action"] == "debug" and node is failed
    installed = loop.install(Run(_loop=loop), [Proposal(
        edit={"op": "set_knob", "name": "level", "value": 7}, parent=failed.id)],
        loop._gated_parents())
    assert installed and installed[0].parent == failed.id
    failed.exit_ok, failed.score = True, 0.3  # repaired: no failed leaf now
    loop._decide(loop._gated_parents())
    assert loop.strategy_decision["action"] == "improve"
    assert "0.8" not in loop.strategy_decision["why"]


@pytest.mark.parametrize("strategy", [
    "default", "reflective", "reflective_sampler", "adaptive_meta", "tree_search",
    "staged_curriculum", "staged_v1", "research_then_optimize", "best_of_n", "autoscientists"])
def test_every_strategy_selects_on_val_scores_test_only_at_the_end_and_hides_val(
        tmp_path, monkeypatch, strategy):
    """Every built-in strategy under the gate with pool: it ends normally, the final
    evaluation runs, its winner is the incumbent, no test rollout happens before the
    final scoring, and no proposer prompt holds a val score."""
    # Every val score ends in ...137 (six decimals), so it is told from a timing (four
    # decimals) or a train score in the prompt.
    val = {level: round(0.600137 + level * 0.003, 6) for level in range(100)}
    monkeypatch.setattr(ValidationGate, "val_score",
                        staticmethod(lambda run, cand: val[_level(cand)]))
    monkeypatch.setattr(FitProbe, "TRAIN", {level: 0.2 + level / 1000 for level in range(100)})
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    early_tests: list[str] = []
    original = Loop._evaluate_part

    def evaluate_part(self, candidate, part, items, seed, **kwargs):  # type: ignore[no-untyped-def]
        if part == "test" and not self._final_scoring:
            early_tests.append(candidate.id)
        return original(self, candidate, part, items, seed, **kwargs)

    monkeypatch.setattr(Loop, "_evaluate_part", evaluate_part)

    class Climber(Scripted):
        """A model proposer: one child of the pipeline it is shown, a level up."""

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            self.handed.append([_level(p) for p in parents])
            self.prompts.append(ctx.render())
            parent = next((p for p in parents if p.id == ctx.parent), parents[0])
            used = {_level(c) for c in parents} | set(self.used)
            out = []
            for _ in range(max(1, n)):
                child = next(level for level in range(_level(parent) + 1, 100)
                             if level not in used)
                used.add(child)
                self.used.append(child)
                out.append(Proposal(edit={"op": "set_knob", "name": "level", "value": child},
                                    parent=parent.id))
            return out

    proposer = Climber([], model=True)
    proposer.used = []
    with _patch_mode(FitProbe):
        loop = strategies.create(strategy, MODE_ID, _config(tmp_path, epochs=3, batch_size=2),
                                 project=Path("."))
        loop.optimizer = proposer
        if hasattr(loop, "per_stage"):
            loop.per_stage = 1
        report = loop.run()
    assert loop.selection_mode() == ("single" if strategy == "default" else "pool")
    assert report["strategy_report"]["test_eval"] == "done", report["strategy_report"]
    assert report["best"]["id"] == loop._gate().current_id
    assert not early_tests, "test is scored only in the final evaluation"
    assert proposer.prompts, "setup: the proposer was asked"
    assert "Train score of the candidate the validation gate ranks first" in proposer.prompts[-1]
    shown = [form for v in val.values() for form in (f"{v}", f"{v:.6g}", f"{v:.5g}")]
    for prompt in proposer.prompts:
        leaked = [s for s in shown if s in prompt]
        assert not leaked, (f"a val score reached the proposer: {leaked[:3]}: "
                            + " | ".join(prompt[max(0, prompt.index(s) - 300):prompt.index(s) + 80]
                                         for s in leaked[:1]))


# ── single reproduces the recorded gate decisions ──────────────────────────


def _recorded() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())["runs"]


@pytest.mark.parametrize("run_name", sorted(_recorded()))
@pytest.mark.parametrize("selection", ["", "single", "pool"])
def test_recorded_livemath_runs_replay_to_the_same_gate_decisions(
        tmp_path, monkeypatch, run_name, selection):
    """Each recorded arm A/B run, replayed on its recorded parents, train and val scores:
    the default setting and ``single`` reproduce its accepted and rejected candidates, its
    incumbent, its winner and its stop exactly. ``pool`` keeps the same incumbent line
    (the reported best moves by the same strict rule), with more parents on offer."""
    run = _recorded()[run_name]
    ids = [c["id"] for c in run["candidates"]]
    level = {cid: index for index, cid in enumerate(ids)}
    train = {level[c["id"]]: c["score"] for c in run["candidates"]}
    val = {level[c["id"]]: c["val"] for c in run["candidates"] if c["val"] is not None}
    rounds = max(c["epoch"] for c in run["candidates"])
    plan = [[(level[c["parent"]], level[c["id"]]) for c in run["candidates"] if c["epoch"] == e]
            for e in range(1, rounds + 1)]
    _scores(monkeypatch, train, val)
    config = run["config"]
    # The empty setting is the whole default: val_gate unset too, on for a val part.
    loop, proposer, report = _run(
        tmp_path, monkeypatch, plan, epochs=config["epochs"],
        batch_size=2 if config["portfolio"] else 1,
        early_stop_patience=config["early_stop_patience"],
        early_stop_at_full_val=config["early_stop_at_full_val"], selection=selection,
        **({"val_gate": None} if not selection else {}))
    assert loop.config.val_gate is True
    by_level = {_level(c): c.id for c in loop.population}

    def recorded(cids: list[str]) -> list[str]:
        return [by_level[level[cid]] for cid in cids]

    assert loop._gate_accepted == recorded(run["accepted"])
    assert loop._gate().current_id == recorded([run["current"]])[0]
    assert report["best"]["id"] == recorded([run["best"]])[0]
    if selection != "pool":
        assert loop._gate_rejected == recorded(run["rejected"])
        assert report["bound"] == run["bound"]
        assert all(len(handed) == 1 for handed in proposer.handed), "one parent: the incumbent"
