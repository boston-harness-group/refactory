"""Validation as the fitness, for every strategy (``LoopConfig.selection``).

Train is the proposer's feedback, validation decides every choice (parents, the pool,
the winner) and test is scored once at the end. ``single`` is SkillOpt's line and
reproduces the recorded gate decisions; ``pool`` keeps the best candidates on
validation as parents. Each test fails when the rule it names is reverted.
"""

from __future__ import annotations

import json
import re
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
        loop = Loop(MODE_ID, _config(tmp_path, val_gate=None, holdout=0.3, parts_manifest=_manifest(
            tmp_path, train=["t1", "t2", "t3"], val=["v1", "v2"])), project=Path("."))
        report = loop.run()
    assert loop.config.val_gate is True and loop.selection_mode() == "single"
    assert json.loads((loop.root / "config.json").read_text())["val_gate"] is True
    assert report["selection"]["val_gate_basis"].startswith("default")
    assert "warning" not in report["selection"]
    assert report["best_score_basis"].startswith("val")
    holdout = report["holdout"]
    assert holdout["split"] == "val" and holdout["independent"] is False
    # No verdict on the part the search selected on, and no test part to compare with.
    assert holdout["status"] == "not_applicable" and holdout["overfit"] is None
    assert holdout["gap"] is None and "no independent check" in holdout["note"]
    assert "the run has no test part" in holdout["note"] and report["exit_ok"] is True
    rendered = (loop.root / "report.md").read_text()
    assert "not an independent check" in rendered
    assert "holds up" not in rendered and "— overfit" not in rendered


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
                 new_shape: tuple[int, ...] = (), **overrides: Any) -> Loop:
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
        if level in new_shape:  # another agent step: a pipeline shape of its own
            state["graph"]["nodes"]["extra_agent"] = {"id": "extra_agent", "kind": "agent"}
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


def test_map_elites_keeps_the_niche_of_a_new_shape_ranked_below_the_pool(tmp_path):
    """Nine judged candidates, pool_size 8: the one new shape (an extra agent step) is
    9th on val. MAP-Elites is handed every judged candidate, so its cell holds it and
    it can be picked; the pool alone would have dropped it."""
    levels = {level: (0.5, 0.9 - level / 100) for level in range(8)}
    levels[8] = (0.5, 0.1)
    loop = _judged_loop(tmp_path, "default", levels, current=0, new_shape=(8,),
                        optimizer="map_elites", selection="pool", pool_size=8)
    new = next(c for c in loop.population if _level(c) == 8)
    assert new not in loop._pool(), "setup: the new shape is outside the pool"
    parents = loop._gated_parents()
    assert new in parents
    picked = loop.optimizer.select(_ctx(loop._val_scores()), 40, parents)
    assert loop.optimizer.cells[loop.optimizer.features(new)] is new
    assert new in picked
    report_pool = loop._selection_report()
    assert report_pool["parent_set"] == "every judged candidate" and new.id in report_pool["pool"]


def test_an_island_whose_best_is_outside_the_top_k_still_breeds(tmp_path):
    """Twenty judged candidates over three islands; every member of one island ranks
    below the top 8. Islands still has that island and breeds its best."""
    levels = {level: (0.5, 0.5) for level in range(20)}
    loop = _judged_loop(tmp_path, "default", levels, current=0, optimizer="islands",
                        selection="pool", pool_size=8)
    policy = loop.optimizer
    islands: dict[int, list[Candidate]] = {}
    for candidate in loop.population:
        islands.setdefault(policy.island_of(candidate), []).append(candidate)
    weak = min(islands, key=lambda island: len(islands[island]))
    others = [c for c in loop.population if policy.island_of(c) != weak]
    gate = loop._gate()
    for rank, candidate in enumerate(others):
        gate.cache[candidate.id] = 0.9 - rank / 100
    for rank, candidate in enumerate(islands[weak]):
        gate.cache[candidate.id] = 0.2 - rank / 100
    top = others[0]
    loop._gate_accepted = [loop.population[0].id, top.id]
    loop._gate_rejected = [c.id for c in loop.population if c.id not in loop._gate_accepted]
    gate.current_id = gate.best_id = top.id
    assert len(others) >= 8 and not {c.id for c in islands[weak]} & {c.id for c in loop._pool()}, "setup"
    chosen = policy.select(_ctx(loop._val_scores()), len(islands), loop._gated_parents())
    assert islands[weak][0] in chosen, "the weak island's best is bred"


def test_archive_keeps_up_to_its_own_limit_under_the_pool(tmp_path):
    levels = {level: (0.5, 0.9 - level / 100) for level in range(14)}
    loop = _judged_loop(tmp_path, "default", levels, current=0, optimizer="archive",
                        selection="pool", pool_size=8)
    kept = loop.optimizer.select(_ctx(loop._val_scores()), 14, loop._gated_parents())
    assert len(kept) == loop.optimizer.limit == 12
    assert [_level(c) for c in kept] == list(range(12)), "the best 12 on val"


def test_edit_kind_survives_a_child_whose_parent_has_no_val_score():
    """Under the gate a parent may have no val score (never judged): its children are
    not compared with it, rather than the decision failing."""
    from refactory_dsh.strategies import edit_kind_decision

    population, val = _settings_children(train_up_val_down=True)
    del val[population[0].id]  # the seed was never judged
    decision = edit_kind_decision(population, scores=val)
    assert decision["edit_kind"] == "prompt_knob", "two compared children: no stall yet"


def test_a_resumed_run_keeps_its_recorded_val_gate(tmp_path, monkeypatch):
    """A run started with val_gate 0 and resumed with the setting left unset stays
    train-selected (the manifest has val items, which would turn the gate on), and
    names the winner an uninterrupted train-selected run names."""
    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    plan = [[(0, 1), (0, 2), (0, 3)], [(3, 4)]]
    straight, _, expected = _run(tmp_path / "straight", monkeypatch, plan, epochs=2,
                                 batch_size=3, val_gate=False)
    _run(tmp_path / "stopped", monkeypatch, plan[:1], epochs=1, batch_size=3, val_gate=False)
    loop, _, report = _run(tmp_path / "stopped", monkeypatch, plan[1:], epochs=2,
                           batch_size=3, val_gate=None, resume=True)
    assert loop.config.val_gate is False and report["selection"]["setting"] == "train"
    assert report["selection"]["val_gate_basis"].startswith("resumed")
    winner = _level(next(c for c in loop.population if c.id == report["best"]["id"]))
    assert winner == _level(next(c for c in straight.population
                                 if c.id == expected["best"]["id"])) == 3


def test_a_finished_pool_run_recovers_a_reject_with_its_own_selection(tmp_path, monkeypatch):
    """A finished reflective run (pool) whose incumbent a person rejects afterwards:
    finished_control rebuilds the run as a plain Loop, and still reads the run's pool
    rule, so report.json and adopt both name 4, the next of the pool."""
    from refactory_dsh.adopt import best
    from refactory_dsh.control import Control
    from refactory_dsh.finished_control import recover_selection

    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    loop, _, report = _run(tmp_path, monkeypatch, POOL_PLAN[:2], strategy="reflective",
                           epochs=2, batch_size=3, pool_size=2)
    by_level = {_level(c): c for c in loop.population}
    assert loop.selection_mode() == "pool" and report["best"]["id"] == by_level[2].id
    Control(loop.root).path.write_text(json.dumps(
        {"action": "reject", "target": by_level[2].id, "at": 1.0}))
    with _patch_mode(FitProbe):
        assert recover_selection(loop.root, project=Path(".")) is not None
    recovered = json.loads((loop.root / "report.json").read_text())
    assert recovered["best"]["id"] == by_level[4].id == best(loop.root)["candidate"]


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
    judged = list(loop._gate().cache.values())
    for prompt in proposer.prompts:
        leaked = _leaked(prompt, judged, items=4)
        assert not leaked, f"a val score reached the proposer: {leaked[:3]}"


def _leaked(text: str, values: list[float], *, items: int) -> list[str]:
    """Each way ``text`` could show one of ``values`` (val scores), at any rounding: the
    full number, 2 to 6 decimals, a percentage (0 or 1 decimal), and k/items when the
    score is a whole number of the val part's items. A form must stand alone (no digit
    or point around it), so 0.82 does not match inside 0.8213, a timing."""
    found: list[str] = []
    for value in values:
        forms = {repr(value), *(f"{value:.{d}f}" for d in range(2, 7)),
                 f"{value * 100:.0f}%", f"{value * 100:.1f}%", f"{value * 100:.0f} %"}
        whole = value * items
        if items and abs(whole - round(whole)) < 1e-9:
            forms.add(f"{round(whole)}/{items}")
        for form in forms:
            match = re.search(rf"(?<![\d.]){re.escape(form)}(?![\d])", text)
            if match:
                found.append(f"{form!r} in ...{text[max(0, match.start() - 120):match.end() + 40]}...")
    return found


def test_the_leak_check_catches_a_val_score_at_any_rounding():
    """The checker the prompt test uses: 14/17 = 0.8235... is caught as 0.82, 0.824,
    82%, 82.4% and 14/17, and not inside an unrelated 0.8213."""
    score = 14 / 17
    for shown in ("best 0.82 so far", "best 0.824", "82% of items", "82.4%", "14/17 right",
                  f"score {score}"):
        assert _leaked(shown, [score], items=17), shown
    assert not _leaked("took 0.8213 s, 182 items, 0.825", [score], items=17)


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


# ── pool follow-ups: what the proposer sees, resume, labels, the report ───


def _recent_rows(prompt: str) -> list[str]:
    """The candidate ids of a request's "Recent results" table, in its order."""
    section = prompt.split("## Recent results", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^\| ([0-9a-f]{12}) \|", section, re.M)


def test_the_requests_rows_do_not_follow_the_val_ranking(tmp_path, monkeypatch):
    """Round 1's children 1, 2 and 3 tie the seed on train and differ on val (2 > 3 > 1
    > the seed). The next request lists the current candidate (2) first and the rest in
    the order they were made (the train tie), not in val order (3, then 1, then the
    seed), which would hand the proposer the val ranking."""
    _scores(monkeypatch, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
            {0: 0.5, 1: 0.6, 2: 0.8, 3: 0.7, 4: 0.1})
    loop, proposer, _ = _run(tmp_path, monkeypatch, [[(0, 1), (0, 2), (0, 3)], [(2, 4)]],
                             model=True, epochs=2, batch_size=3, selection="pool")
    rows = _levels(loop, _recent_rows(proposer.prompts[1]))
    assert rows == [2, 0, 1, 3], rows
    val_order = sorted(rows, key=lambda level: -{0: 0.5, 1: 0.6, 2: 0.8, 3: 0.7}[level])
    assert rows != val_order


def test_the_requests_rows_follow_train_score_after_the_current_candidate(
        tmp_path, monkeypatch):
    """With train scores that differ, the rows after the current candidate are by train
    score, highest first, whatever their val scores."""
    _scores(monkeypatch, {0: 0.5, 1: 0.9, 2: 0.2, 3: 0.4, 4: 0.5},
            {0: 0.5, 1: 0.6, 2: 0.8, 3: 0.7, 4: 0.1})
    loop, proposer, _ = _run(tmp_path, monkeypatch, [[(0, 1), (0, 2), (0, 3)], [(2, 4)]],
                             model=True, epochs=2, batch_size=3, selection="pool")
    assert _levels(loop, _recent_rows(proposer.prompts[1])) == [2, 1, 0, 3]


def test_pool_labels_say_in_pool_left_pool_and_rejected(tmp_path, monkeypatch):
    """The pool scenario (pool_size 2), in the report's labels: 5 is the incumbent; 4,
    2 and 1 are in the pool (2 was the incumbent before 5); the seed was in the pool
    until round 1 pushed it out; 3 never entered it. The optimizer's labels are
    ``single``'s: kept (accepted as the incumbent when judged) or rejected."""
    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    loop, _, report = _run(tmp_path / "pool", monkeypatch, POOL_PLAN, epochs=3,
                           batch_size=3, selection="pool", pool_size=2)
    labels = {_level(c): loop._gate_decision(c.id, detail=True) for c in loop.population}
    assert labels == {5: "current", 4: "in pool", 2: "in pool", 1: "in pool",
                      0: "left pool", 3: "rejected"}
    proposer_view = {_level(c): loop._gate_decision(c.id) for c in loop.population}
    assert proposer_view == {5: "current", 4: "rejected", 2: "kept", 1: "rejected",
                             0: "kept", 3: "rejected"}
    assert {row["gate"] for row in report["leaderboard"]} == {"current", "in pool"}


def test_the_proposer_never_sees_a_pool_label(tmp_path, monkeypatch):
    """Under pool, every request (rows, history, reasoning, gate note) uses single's
    labels only; the pool labels are the report's."""
    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    loop, proposer, _ = _run(tmp_path, monkeypatch, POOL_PLAN + [[(5, 6)]], model=True,
                             epochs=4, batch_size=3, selection="pool", pool_size=2)
    assert loop._gate_decision(loop.population[0].id, detail=True) == "left pool", "setup"
    assert len(proposer.prompts) == 4
    for prompt in proposer.prompts:
        for label in ("in pool", "left pool", "below pool", "pushed out"):
            assert label not in prompt, label
        cells = set(re.findall(r"\| (current|kept|rejected|not gated|[a-z ]+pool) \|$",
                               prompt, re.M))
        assert cells <= {"current", "kept", "rejected", "not gated"}, cells
    assert "left pool" in (loop.root / "state.json").read_text()


def test_ties_at_the_pool_edge_enter_the_pool(tmp_path, monkeypatch):
    """pool_size 2. Round 1: 1 (0.7) and 2 and 3 tied on 0.6 at the edge: all three
    enter. Round 2: 4 (0.8) and 5 (0.75) push the edge up to 0.7, so 2 and 3 leave the
    pool: ``left pool``, not ``rejected``, as is the seed."""
    val = {0: 0.5, 1: 0.7, 2: 0.6, 3: 0.6, 4: 0.8, 5: 0.75}
    _scores(monkeypatch, {level: 0.5 for level in val}, val)
    loop, _, _ = _run(tmp_path, monkeypatch, [[(0, 1), (0, 2), (0, 3)], [(1, 4), (1, 5)]],
                      epochs=2, batch_size=3, selection="pool", pool_size=2)
    labels = {_level(c): loop._gate_decision(c.id, detail=True) for c in loop.population}
    assert labels == {4: "current", 5: "in pool", 1: "left pool", 2: "left pool",
                      3: "left pool", 0: "left pool"}
    # Entry is judged per gating, as recorded: had one gating judged them all, 2 and 3
    # would never have entered.
    ids = {_level(c): c.id for c in loop.population}
    loop._gate_rounds = [[ids[0]], [ids[level] for level in (1, 2, 3, 4, 5)]]
    assert loop._gate_decision(ids[2], detail=True) == "rejected"


def test_a_person_rejection_does_not_make_others_leave_the_pool(tmp_path, monkeypatch):
    """pool_size 2. Round 1: 1 (0.9) and 2 (0.7) enter, 3 (0.6) does not. Round 2: 4
    (0.95). A person then rejects 2: 3 still never entered the pool (2 held its place
    when 3 was judged), so it is ``rejected``, not ``left pool``; 2 is rejected."""
    val = {0: 0.5, 1: 0.9, 2: 0.7, 3: 0.6, 4: 0.95}
    _scores(monkeypatch, {level: 0.5 for level in val}, val)
    loop, _, _ = _run(tmp_path, monkeypatch, [[(0, 1), (0, 2), (0, 3)], [(1, 4)]],
                      epochs=2, batch_size=3, selection="pool", pool_size=2)
    by_level = {_level(c): c.id for c in loop.population}
    loop.rejected = [by_level[2]]
    loop._pool_entered_memo = None  # recomputed with the rejection in place
    labels = {level: loop._gate_decision(cid, detail=True) for level, cid in by_level.items()}
    assert labels == {4: "current", 1: "in pool", 2: "rejected", 3: "rejected",
                      0: "left pool"}


class _PickedScript(Optimizer):
    """A proposer for a real parent picker: each round, the child levels it is given,
    each a child of whatever parent the picker handed over."""

    name = "picked_script"

    def __init__(self, rounds: list[list[int]]) -> None:
        super().__init__(k=1)
        self.rounds = list(rounds)
        self.parents: list[int] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        parent = parents[0]
        self.parents.append(_level(parent))
        levels = self.rounds.pop(0) if self.rounds else []
        return [Proposal(edit={"op": "set_knob", "name": "level", "value": level},
                         parent=parent.id) for level in levels]


def test_below_pool_with_a_real_population_picker(tmp_path, monkeypatch):
    """The evolutionary picker (a population policy) breeds from every judged candidate,
    so 3, which never entered the pool of 2, is ``below pool`` in the report, and the
    selection resolves to pool by default and is recorded so."""
    from refactory_dsh.optim.compose import PickedParent

    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(FitProbe):
        loop = strategies.create("default", MODE_ID, _config(
            tmp_path, epochs=3, batch_size=3, pool_size=2, parent_picker="evolutionary"),
            project=Path("."))
        assert loop._population_policy()
        picker = loop.optimizer.picker
        loop.optimizer = PickedParent(picker, _PickedScript([[1, 2, 3], [4], [5]]))
        loop.run()
    assert loop.selection_mode() == "pool"
    assert json.loads((loop.root / "config.json").read_text())["selection"] == "pool"
    by_level = {_level(c): c.id for c in loop.population}
    assert loop._gate_decision(by_level[3], detail=True) == "below pool"
    assert loop._gate_decision(by_level[3]) == "rejected"


def test_the_portfolio_report_under_pool_counts_incumbents_and_pool_members(
        tmp_path, monkeypatch):
    """The pool scenario read as a portfolio: "kept" counts the candidates accepted as
    the incumbent (2 and 5), "in pool" those in the pool now (5, 4, 2, 1), and each
    round's outcome says both (1 beat its parent on val but lost to its sibling 2)."""
    from refactory_dsh.loop import _portfolio_section

    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    loop, _, _ = _run(tmp_path, monkeypatch, POOL_PLAN, epochs=3, batch_size=3,
                      selection="pool", pool_size=2)
    for record in loop.proposals.values():
        record["level"] = "prompt_knob"
    loop.config.portfolio = "prompt+pipeline"
    report = loop._portfolio_report()
    entry = report["by_level"]["prompt_knob"]
    assert (entry["proposed"], entry["kept"], entry["in_pool"]) == (5, 2, 4)
    text = "\n".join(_portfolio_section(report))
    by_level = {_level(c): c.id for c in loop.population}
    outcomes = {level: re.search(rf"\| {by_level[level]} \| ([^|]+) \|", text).group(1).strip()
                for level in (1, 2, 3, 4, 5)}
    assert outcomes == {1: "shadowed, in pool", 2: "accepted, in pool", 3: "rejected",
                        4: "in pool", 5: "current"}
    assert "accepted as incumbent" in text and "in pool now" in text


def test_the_tried_line_gives_the_current_candidates_train_score(tmp_path, monkeypatch):
    """Round 3's request, pool {2 (current, train 0.4), 4 (train 0.9), 1}: the line above
    "What this run has already tried" gives the current candidate's train score, said to
    be a train score, not the best train score in the pool under a gate label."""
    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    loop, proposer, _ = _run(tmp_path, monkeypatch, POOL_PLAN, model=True, epochs=3,
                             batch_size=3, selection="pool", pool_size=2)
    score = {_level(c): c.score for c in loop.population}
    tried = proposer.prompts[2].split("## What this run has already tried", 1)[1]
    first = tried.strip().splitlines()[0]
    assert f"train score of the validation gate's current candidate {score[2]:.6g}" in first, first
    assert f"{score[4]:.6g}" not in first and "best kept" not in first


def test_structural_wins_under_the_gate_are_judged_on_val():
    """Under the gate a structural child beats its parent on the gate's val score, not
    on train: a train loss with a val gain is a win, a train gain with a val loss is not."""
    from types import SimpleNamespace

    edit = [{"op": "add_node", "node": {}, "position": {"start": True}}]
    parent = Candidate(id="p", parent=None, state={}, edits=[], score=0.9, evaluated=True)
    up = Candidate(id="up", parent="p", state={}, edits=edit, step_edits=1, score=0.5,
                   evaluated=True)
    down = Candidate(id="down", parent="p", state={}, edits=edit, step_edits=1, score=0.95,
                     evaluated=True)
    fake = SimpleNamespace(
        population=[parent, up, down], _row=lambda c, **_: {"id": c.id},
        config=SimpleNamespace(val_gate=True), _custom_strategy=None,
        _gate=lambda: SimpleNamespace(cache={"p": 0.5, "up": 0.75, "down": 0.25}))
    wins = Loop._structural_wins(fake, [up, down, parent])
    assert wins == [{"id": "up", "val": 0.75, "parent_val": 0.5}]


def test_state_json_shows_the_last_batchs_gate_decision(tmp_path, monkeypatch):
    """The last batch is scored and gated after the last epoch; state.json is rewritten
    then, so its incumbent (5) reads ``current``, not ``not gated``."""
    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    loop, _, _ = _run(tmp_path, monkeypatch, POOL_PLAN, epochs=3, batch_size=3,
                      selection="pool", pool_size=2)
    rows = {row["id"]: row for row in json.loads((loop.root / "state.json").read_text())["population"]}
    five = next(c.id for c in loop.population if _level(c) == 5)
    assert rows[five]["gate"] == "current"


def test_a_seed_that_wins_is_scored_on_test_once_and_its_delta_is_zero(tmp_path, monkeypatch):
    """No child beats the seed on val, so the seed is the winner: its test score is
    the baseline's (one test rollout, not two), and best_test_delta is exactly 0."""
    tests: list[str] = []
    original = Loop._evaluate_part

    def evaluate_part(self, candidate, part, items, seed, **kwargs):  # type: ignore[no-untyped-def]
        if part == "test":
            tests.append(candidate.id)
        return original(self, candidate, part, items, seed, **kwargs)

    monkeypatch.setattr(Loop, "_evaluate_part", evaluate_part)
    _scores(monkeypatch, {0: 0.5, 1: 0.9, 2: 0.9}, {0: 0.7, 1: 0.5, 2: 0.6})
    loop, _, report = _run(tmp_path, monkeypatch, [[(0, 1)], [(0, 2)]], epochs=2)
    seed = loop.population[0]
    assert report["best"]["id"] == seed.id
    figures = report["strategy_report"]
    assert figures["test_eval"] == "done"
    assert tests == [seed.id], "the seed is rolled out on test once"
    assert figures["best_test"] == figures["baseline_test"] and figures["best_test_delta"] == 0


# ── resume keeps the run's selection ───────────────────────────────────────


def _cli(tmp_path: Path, *argv: str, profile: bool = True) -> int:
    from refactory_dsh import loop as loop_module

    with _patch_mode(FitProbe):
        return loop_module.main([MODE_ID, "--project", str(tmp_path), "--runs-root",
                                 str(tmp_path / "run"),
                                 *(["--profile", "pool-probe"] if profile else []), *argv])


@pytest.fixture
def pool_profile(tmp_path, monkeypatch):
    """A saved profile (selection left to its default, single) and a run launched from
    it with ``--selection pool --pool-size 3`` for one epoch."""
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    _scores(monkeypatch, {}, {})
    profiles.save(tmp_path, "pool-probe", {"settings": {
        "optimizer": "random", "allowed_ops": ["set_knob"], "val_gate": True, "seed": 7,
        "holdout": 0.0, "batch_size": 1, "max_evaluations": 50,
        "parts_manifest": str(_write_parts(tmp_path, test=True))}})
    assert _cli(tmp_path, "--epochs", "1", "--selection", "pool", "--pool-size", "3") == 0
    return tmp_path / "run"


def test_a_resume_keeps_the_selection_and_pool_size_its_launch_gave(tmp_path, pool_profile):
    """Resumed with the profile and ``--resume`` only: the run keeps the selection and
    pool size it was launched with (the profile says single and 8), in its config, its
    gate state and its report."""
    assert _cli(tmp_path, "--epochs", "2", "--resume") == 0
    config = json.loads((pool_profile / "config.json").read_text())
    assert (config["selection"], config["pool_size"]) == ("pool", 3)
    report = json.loads((pool_profile / "report.json").read_text())
    assert report["selection"]["setting"] == "pool" and report["selection"]["pool_size"] == 3


@pytest.mark.parametrize("profile", [True, False])
@pytest.mark.parametrize("flag,named", [
    (("--pool-size", "5"), "pool_size 3 and this resume asks for pool_size 5"),
    (("--selection", "single"), "selection pool and this resume asks for selection single"),
    (("--val-gate", "0"), "val_gate 1 and this resume asks for val_gate 0"),
])
def test_a_resume_that_names_another_selection_is_refused(
        tmp_path, pool_profile, capsys, flag, named, profile):
    """A resume command, with its profile or without one, that gives a different
    selection, pool size or val_gate explicitly is refused, as a different
    train_minibatch is, naming both values; the recorded run is left as it was."""
    before = (pool_profile / "config.json").read_text()
    assert _cli(tmp_path, "--epochs", "2", "--resume", *flag, profile=profile) == 2
    assert named in capsys.readouterr().err
    assert (pool_profile / "config.json").read_text() == before


def test_a_launch_records_the_selection_it_resolved(tmp_path, monkeypatch):
    """Launched with no --selection, the run records the rule it resolved (single for
    the default loop here), so a resume compares real values."""
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    _scores(monkeypatch, {}, {})
    profiles.save(tmp_path, "pool-probe", {"settings": {
        "optimizer": "random", "allowed_ops": ["set_knob"], "val_gate": True, "seed": 7,
        "holdout": 0.0, "batch_size": 1, "max_evaluations": 50,
        "parts_manifest": str(_write_parts(tmp_path, test=True))}})
    assert _cli(tmp_path, "--epochs", "1") == 0
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert (config["selection"], config["pool_size"]) == ("single", 8)


def test_a_loop_built_in_code_resumes_with_its_recorded_selection(tmp_path, monkeypatch):
    """Without a launch snapshot (a loop built in code), settings left at their
    defaults take the recorded ones: the resumed pool run keeps pool_size 2 and its
    pool, and names the winner an uninterrupted run names."""
    _scores(monkeypatch, POOL_TRAIN, POOL_VAL)
    _, _, expected = _run(tmp_path / "straight", monkeypatch, POOL_PLAN, epochs=3,
                          batch_size=3, selection="pool", pool_size=2)
    _run(tmp_path / "stopped", monkeypatch, POOL_PLAN[:1], epochs=1, batch_size=3,
         selection="pool", pool_size=2)
    loop, proposer, report = _run(tmp_path / "stopped", monkeypatch, POOL_PLAN[1:], epochs=3,
                                  batch_size=3, resume=True)
    assert (loop.config.selection, loop.config.pool_size) == ("pool", 2)
    assert proposer.handed[0] == [2, 1], "round 2 breeds from the pool of 2"
    assert _levels(loop, report["selection"]["pool"]) == [5, 4, 2, 1]
    assert _levels(loop, [report["best"]["id"]]) == [5]


def test_a_final_candidate_that_is_the_seed_reuses_the_seeds_test_score():
    """The final (current) candidate is the seed and the best another: the seed is
    rolled out on test once, as the baseline; the final's score is that one and its
    delta exactly 0. Only the best gets its own rollout."""
    import contextlib
    from types import SimpleNamespace

    from refactory_dsh.outerloop import FinalEvaluation

    seed, best = SimpleNamespace(id="s"), SimpleNamespace(id="b")
    rolled: list[str] = []
    scores = {"s": 0.5, "b": 0.7}

    class FakeRun:
        def final_scoring(self):  # type: ignore[no-untyped-def]
            return contextlib.nullcontext()

        def seed(self):  # type: ignore[no-untyped-def]
            return seed

        def candidate(self, cid):  # type: ignore[no-untyped-def]
            return {"s": seed, "b": best}.get(cid)

        def evaluate(self, candidate, *, part, seed=None):  # type: ignore[no-untyped-def]
            assert part == "val"

        def part_scores(self, candidate, part):  # type: ignore[no-untyped-def]
            return [{"score": 0.4 if part == "val" else scores[candidate.id]}]

        def evaluate_many(self, requests, *, on_result=None):  # type: ignore[no-untyped-def]
            for index, request in enumerate(requests):
                rolled.append(request.candidate.id)
                on_result(index, request.candidate)

        def log(self, text):  # type: ignore[no-untyped-def]
            pass

    gate = SimpleNamespace(current_id="s", best_id="b", best_score=0.6,
                           offer_best=lambda cid, score: False)
    final = FinalEvaluation()
    final.run(FakeRun(), gate, seed=7, save=lambda: None)  # type: ignore[arg-type]
    assert sorted(rolled) == ["b", "s"], "the seed once, the best once"
    result = final.result
    assert result["status"] == "done"
    assert result["final_test"] == result["baseline_test"] == 0.5
    assert result["final_test_delta"] == 0 and result["best_test_delta"] == pytest.approx(0.2)
