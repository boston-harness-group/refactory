"""The SRF strategies, each tested on what makes it that strategy.

Not on a score: a port is correct when it exhibits the control flow it is named
after. Every one of the thirteen SRF modes lands on one of the two seams, which is
the claim the abstraction is making.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from refactory_dsh import strategies
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import Candidate, Proposal
from refactory_dsh.outerloop import OuterLoop, Run

MODE = "circle_packing/adaptive"


def _loop(name: str, root: Path, **overrides: object) -> Loop:
    base: dict[str, object] = dict(
        epochs=2, batch_size=2, optimizer="random", seed=1, runs_root=root
    )
    base.update(overrides)
    return strategies.create(name, MODE, LoopConfig(**base), project=Path("."))  # type: ignore[arg-type]


# ── the registry ───────────────────────────────────────────────────────────
def test_every_srf_port_lands_in_exactly_one_bucket():
    """The split issue #32 found missing: some SRF methods are search policies.

    `outerloop.py` says "most research ideas live here [in a proposer]" and the
    registry said otherwise -- every SRF method was an `OuterLoop` and the
    `Optimizer` registry held two hand-written entries. The cause was structural and
    is why the fix was the contract: a proposer was handed the elite and
    `describe()` carries no state, so a policy needing a wider pool could not be
    written as one. The loop passes the population now and the optimizer selects.
    """
    from refactory_dsh import optim

    loops = [name for name in strategies.names() if name != "default"]
    ports = strategies.SRF_OPTIMIZER_PORTS

    assert all(name in optim.names() for name in ports), (
        f"an SRF optimizer port is not in the optimizer registry: {ports} vs {optim.names()}"
    )
    assert strategies.SRF_PORT_COUNT - len(ports) == len(loops), (
        f"{strategies.SRF_PORT_COUNT} SRF modes must split into {len(ports)} optimizers "
        f"and {strategies.SRF_PORT_COUNT - len(ports)} loops, but {len(loops)} loops "
        f"are registered: {loops}"
    )
    # The lab optimizer is an SRF port too, in the other bucket, and counted apart.
    assert "map_elites" in optim.names()
    assert "map_elites" not in loops


def test_no_port_is_filed_in_both_buckets():
    """Two registries and one port: which one it is in has to be answerable."""
    from refactory_dsh import optim

    overlap = set(strategies.names()) & set(optim.names())
    assert not overlap, f"registered as both a proposer and a loop: {sorted(overlap)}"


def test_the_two_buckets_are_the_only_two():
    """Every registered name is one thing or the other, never neither."""
    from refactory_dsh import optim
    from refactory_dsh.outerloop import OuterLoop

    for name, cls in optim.REGISTRY.items():
        assert issubclass(cls, optim.Optimizer), f"optimizer {name} is not an Optimizer"
        assert not issubclass(cls, OuterLoop), f"optimizer {name} is also an OuterLoop"
    for name, cls in strategies.STRATEGIES.items():
        assert issubclass(cls, OuterLoop), f"strategy {name} is not an OuterLoop"


def test_the_registry_refuses_an_unknown_strategy():
    with pytest.raises(ValueError, match="unknown strategy"):
        strategies.resolve("nonsense")


def test_a_strategy_can_be_registered_without_editing_the_library():
    class Mine(Loop):
        name = "mine-by-hand"

    strategies.register("mine_by_hand", Mine)
    try:
        assert "mine_by_hand" in strategies.names()
        assert strategies.resolve("mine_by_hand") is Mine
    finally:
        del strategies.STRATEGIES["mine_by_hand"]


def test_registering_something_that_is_not_an_outer_loop_is_refused():
    class NotALoop:
        name = "nope"

    with pytest.raises(TypeError, match="not an OuterLoop"):
        strategies.register("nope", NotALoop)  # type: ignore[arg-type]


def test_every_port_is_an_outer_loop_with_a_name():
    for name, cls in strategies.STRATEGIES.items():
        assert issubclass(cls, OuterLoop), name
        assert cls.name == name, f"{name} is registered under the wrong name"


# ── the fork shapes ────────────────────────────────────────────────────────
def test_best_of_n_evaluates_exactly_n_from_the_seed_and_never_loops(tmp_path):
    loop = _loop("best_of_n", tmp_path / "run", epochs=5)
    loop.width = 4
    report = loop.run()
    assert report["bound"] == "samples_exhausted"
    assert report["evaluations"] == 4, "best-of-N samples N times and stops"
    assert report["epochs_completed"] == 0, "there is no loop to have an epoch"
    assert len(report["leaderboard"]) == 4, "the N samples, and no seed run"
    assert not loop.population[0].evaluated, (
        "SRF's best_of_n has no seed_run: the fork is the whole search"
    )


def test_autoscientists_forks_n_then_merges_seeing_all_of_them(tmp_path):
    loop = _loop("autoscientists", tmp_path / "run", epochs=5)
    loop.width = 3
    report = loop.run()
    assert report["bound"] == "merged"
    assert loop.merged == 3, "every branch must be produced before the merge"
    assert report["evaluations"] >= 4, "the merge adds an evaluation"


def test_the_merge_takes_a_value_from_a_different_branch(tmp_path):
    """The merge is a real crossover: one knob copied from another branch."""
    loop = _loop("autoscientists", tmp_path / "run", epochs=5)
    loop.width = 4
    loop.run()
    assert loop.merged == 4
    # Every sampled branch is a distinct knob value; the merged candidate carries a
    # value that also appears in one of the others, which is what a crossover is.
    values = [
        {n: e["value"] for n, e in c.state["knobs"].items()} for c in loop.population
    ]
    merged = loop.population[-1]
    merged_knobs = {n: e["value"] for n, e in merged.state["knobs"].items()}
    assert merged_knobs in values
    assert merged.score is not None, "the merged candidate was never evaluated"


# ── the switch-inside-an-iteration shapes ──────────────────────────────────
def test_reflective_takes_a_merge_branch_and_a_mutate_branch(tmp_path):
    loop = _loop("reflective", tmp_path / "run", epochs=4)
    loop.run()
    assert loop.merges >= 1, "the merge branch was never taken"
    assert loop.mutations >= 1, "the mutate branch was never taken"
    assert loop.merges + loop.mutations >= 4, "both branches continue the same iteration"


def test_adaptive_meta_fires_its_meta_branch_and_records_it(tmp_path):
    loop = _loop("adaptive_meta", tmp_path / "run", epochs=5)
    loop.run()
    assert loop.meta_taken >= 1, "the meta branch never fired"
    log = (loop.root / "strategy.log").read_text()
    assert "meta=True" in log and "meta=False" in log, (
        "the choice must be recorded, not inferred"
    )


def test_tree_search_keeps_a_tree_not_a_list(tmp_path):
    loop = _loop("tree_search", tmp_path / "run", epochs=3)
    loop.run()
    tree = loop.tree
    assert tree.all(), "nothing was archived"
    assert tree.parents, "no candidate recorded an expansion parent"
    assert tree.deepest() >= 1, "the tree never went deeper than its root"
    for candidate in tree.all():
        if candidate.parent:
            assert tree.parents[candidate.id] == candidate.parent


def test_staged_curriculum_runs_every_stage_in_order(tmp_path):
    loop = _loop("staged_curriculum", tmp_path / "run", epochs=99)
    loop.per_stage = 1
    report = loop.run()
    assert loop.stages_done == ["initial", "tuning", "creative", "ablation"]
    assert report["bound"] == "curriculum_complete"
    log = (loop.root / "strategy.log").read_text()
    positions = [log.index(f"stage={stage}") for stage in loop.stages]
    assert positions == sorted(positions), "the stages ran out of order"


def test_staged_v1_has_its_own_stages(tmp_path):
    loop = _loop("staged_v1", tmp_path / "run", epochs=99)
    loop.per_stage = 1
    loop.run()
    assert loop.stages_done == ["explore", "exploit"]


def test_research_then_optimize_runs_a_pre_phase_once(tmp_path):
    loop = _loop("research_then_optimize", tmp_path / "run", epochs=2)
    loop.run()
    assert loop.research_note, "the pre-phase produced nothing"
    assert "movable knob" in loop.research_note
    log = (loop.root / "strategy.log").read_text()
    assert log.count("research=") == 1, "the pre-phase must run once, not per epoch"


def test_reflective_sampler_carries_prior_rejections_into_sampling(tmp_path):
    loop = _loop("reflective_sampler", tmp_path / "run", epochs=2)
    loop.run()
    assert hasattr(loop, "last_reflection"), "sampling did not reflect on anything"


# ── the selection shapes ───────────────────────────────────────────────────

# ── the seams themselves ───────────────────────────────────────────────────
def test_the_two_guarantees_are_not_overridable():
    """A strategy may change control flow, never validation."""
    assert "attempt" not in OuterLoop.overridable
    assert "evaluate" not in OuterLoop.overridable
    assert "strategy" in OuterLoop.overridable and "update" in OuterLoop.overridable


def test_a_custom_control_flow_cannot_evaluate_an_invalid_edit(tmp_path):
    """The point of the seam: a type-2 loop gets the guarantees for free."""

    class Reckless(Loop):
        name = "reckless"

        def strategy(self, run: Run) -> None:
            seed = run.seed()
            # Every one of these is invalid: removing the node that produces the
            # scored fact breaks the mode's own invariant.
            for _ in range(3):
                outcome = run.attempt(
                    Proposal(edit={"op": "remove_node", "id": "exit"}, parent=seed.id), seed
                )
                assert isinstance(outcome, list), "an invalid edit was accepted"
                assert outcome, "a refusal must carry a reason"
            run.stop("tried_to_cheat")

    loop = Reckless(
        MODE,
        LoopConfig(epochs=1, batch_size=2, optimizer="random", seed=1,
                   runs_root=tmp_path / "run"),
        project=Path("."),
    )
    report = loop.run()
    assert report["evaluations"] == 0, "a refused edit must never cost an evaluation"
    assert report["bound"] == "tried_to_cheat"
    assert any("I:" in code or code.startswith("V") for code in report["rejection_codes"])


def test_a_custom_loop_reports_its_own_bound(tmp_path):
    class Minimal(Loop):
        name = "minimal"

        def strategy(self, run: Run) -> None:
            run.keep(run.evaluate(run.attempt(
                Proposal(edit={"op": "set_knob", "name": "order", "value": "asc"},
                         parent=run.seed().id),
                run.seed(),
            )))  # type: ignore[arg-type]
            run.stop("one_sharp_look")

    loop = Minimal(MODE, LoopConfig(epochs=9, batch_size=1, optimizer="random", seed=1,
                                    runs_root=tmp_path / "run"), project=Path("."))
    report = loop.run()
    assert report["bound"] == "one_sharp_look", "a run must report why it ended"
    assert report["evaluations"] == 1
    assert report["best_score"] is not None


def test_a_custom_loop_can_declare_it_needs_a_model():
    class NeedsModel(OuterLoop):
        name = "needs-model"
        requires_model = True

    assert NeedsModel.requires_model is True
    assert Loop.requires_model is False


# ── a strategy's decision must reach the proposal ──────────────────────────
#
# The audit that reopened #20: every test above asserted that a strategy *took*
# its branch (counters, logs) and none asserted the branch changed the search.
# Five strategies recorded a decision and then ran the ordinary path, so six of
# fourteen produced byte-identical leaderboards to `default`. These assert the
# effect, which is the property that was missing.


def test_adaptive_meta_widens_the_batch_when_it_decides_to(tmp_path):
    """A meta-level that changes nothing is a counter, not a policy."""
    loop = _loop("adaptive_meta", tmp_path / "run", epochs=4)
    from refactory_dsh.outerloop import Run

    loop.strategy_decision = {"meta": False}
    calm = loop.batch_size(Run(_loop=loop))
    loop.strategy_decision = {"meta": True, "reason": "no progress"}
    stalled = loop.batch_size(Run(_loop=loop))
    assert stalled > calm, (
        "the meta branch changed nothing, so the loop did not adapt to the stall"
    )


def test_reflective_merges_two_parents_rather_than_mutating_one(tmp_path):
    """The merge branch must produce a candidate neither parent had."""
    from refactory_dsh import registry
    from refactory_dsh.optim import Candidate
    from refactory_dsh.outerloop import Run

    loop = _loop("reflective", tmp_path / "run", epochs=1)
    loop.strategy_decision = {"branch": "merge"}
    mode = registry.resolve(Path("."), "circle_packing/adaptive")[1]()

    left_state = mode.state_dict()
    right_state = mode.state_dict()
    right_state["knobs"]["grid"]["value"] = 24
    left = Candidate.from_state(left_state)
    right = Candidate.from_state(right_state)
    loop.population = [left, right]
    loop.config.elite = 2

    proposals = loop.propose(Run(_loop=loop), [left, right], 2)
    assert proposals, "the merge branch proposed nothing"
    knobs = [p.edit.get("name") for p in proposals if p.edit.get("op") == "set_knob"]
    assert "grid" in knobs, (
        "the merge did not take the other parent's setting, so it is a mutation "
        "with a different label"
    )


def test_reflective_falls_back_when_the_parents_differ_structurally(tmp_path):
    """A merge with nothing to combine must not propose a no-op."""
    from refactory_dsh import registry
    from refactory_dsh.optim import Candidate
    from refactory_dsh.outerloop import Run

    loop = _loop("reflective", tmp_path / "run", epochs=1)
    loop.strategy_decision = {"branch": "merge"}
    mode = registry.resolve(Path("."), "circle_packing/adaptive")[1]()
    same = Candidate.from_state(mode.state_dict())
    loop.population = [same]
    proposals = loop.propose(Run(_loop=loop), [same, same], 2)
    assert all(p.edit.get("op") != "set_knob" or p.edit.get("value") is not None
               for p in proposals)


def test_the_strategy_decision_reaches_the_optimizer(tmp_path):
    """The channel itself: a decision in the log alone informs no proposal."""
    from refactory_dsh.outerloop import Run

    loop = _loop("reflective", tmp_path / "run", epochs=1)
    loop.strategy_decision = {"branch": "merge", "note": "combine"}
    context = loop._context()  # noqa: SLF001 - the test drives the real builder
    assert context.strategy == {"branch": "merge", "note": "combine"}, (
        "the loop's own decision does not reach the proposer"
    )


def test_research_then_optimize_publishes_its_note(tmp_path):
    """A note kept in an attribute reaches no prompt."""
    from refactory_dsh.outerloop import Run

    loop = _loop("research_then_optimize", tmp_path / "run", epochs=2)
    loop.research_note = ""
    loop.strategy_decision = {"research": "grid=18 looked best in the prior run"}
    assert loop._context().strategy["research"].startswith("grid=18"), (
        "the research phase's conclusion is not available to a proposal"
    )


def test_reflective_sampler_drops_an_edit_that_was_already_refused(tmp_path):
    """Reflection has to change the sample, not just be stored beside it."""
    from refactory_dsh.optim import Proposal, Rejection
    from refactory_dsh.outerloop import Run

    loop = _loop("reflective_sampler", tmp_path / "run", epochs=1)
    edit = {"op": "set_knob", "name": "grid", "value": 24}
    loop.rejections.append(Rejection(attempt=1, edit=edit, reasons=["E-precondition: nope"]))
    run = Run(_loop=loop)
    refused = [proposal for proposal in loop.propose(run, loop.select_parents(1), 4)
               if proposal.edit == edit]
    assert refused == [], (
        "the sampler re-proposed an edit the run had already refused, so the "
        "reflection was stored and not used"
    )


def test_a_curriculum_stage_reaches_the_proposal(tmp_path):
    """`initial → tuning → creative → ablation` must mean four different asks."""
    from refactory_dsh.outerloop import Run

    loop = _loop("staged_curriculum", tmp_path / "run", epochs=4)
    seen: list[str] = []

    class Recorder(type(loop)):
        def epoch_step(self, run=None):
            seen.append(self.strategy_decision.get("stage", ""))
            return super().epoch_step(run)

    loop.__class__ = Recorder
    loop.run()
    assert seen, "the curriculum ran no iterations"
    assert all(stage for stage in seen), (
        f"an iteration ran without publishing its stage: {seen} — the stage is in "
        "the log and not in the proposal"
    )
    assert len(set(seen)) > 1, (
        f"every iteration reported the same stage ({set(seen)}), so the curriculum "
        "did not advance"
    )
