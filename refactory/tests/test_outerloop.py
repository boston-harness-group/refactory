"""The two extension points: a proposer plugs in, an outer loop overrides.

Type 1 subclasses `Optimizer` and implements `propose`. Type 2 subclasses
`OuterLoop` and overrides its methods. Both get the same two guarantees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.outerloop import OuterLoop, OuterLoopError, Run

MODE = "circle_packing/adaptive"


def _config(root: Path, **overrides: object) -> LoopConfig:
    base: dict[str, object] = dict(
        epochs=2, batch_size=2, optimizer="random", seed=3, runs_root=root
    )
    base.update(overrides)
    return LoopConfig(**base)  # type: ignore[arg-type]


class Sweep(Optimizer):
    """A type-1 proposer: move one knob at a time, to values nobody has tried."""

    name = "sweep"

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        parent = parents[0]
        tried = set()
        for row in ctx.history:
            for edit in row.get("edits") or ():
                if edit.get("op") == "set_knob":
                    tried.add((edit["name"], str(edit["value"])))
        out: list[Proposal] = []
        for knob in ctx.knobs:
            if knob["frozen"] or not knob.get("reachable", True):
                continue
            for value in knob.get("values") or ():
                if (knob["name"], str(value)) in tried:
                    continue
                out.append(Proposal(
                    edit={"op": "set_knob", "name": knob["name"], "value": value},
                    parent=parent.id,
                    rationale=f"{knob['name']}={value} untried",
                ))
                if len(out) >= n:
                    return out
        return out


# ── type 1: a proposer plugs into the default loop ─────────────────────────
def test_a_proposer_plugs_into_the_default_loop_unchanged(tmp_path):
    """A type-1 proposer needs no changes to the loop, and its work shows.

    A sweep is naive: it walks the knobs in order, so it needs enough budget to
    reach the one that matters. That is a fact about the strategy, not the seam.
    """
    loop = Loop(
        MODE, _config(tmp_path / "run", epochs=4, batch_size=4),
        project=Path("."), optimizer=Sweep(k=1),
    )
    report = loop.run()
    assert report["best_score"] > report["seed_score"], (
        "a sweep over untried values should improve on the seed"
    )
    assert report["optimizer"] == "sweep"
    assert report["evaluations"] >= 9, (
        "the sweep should have run to its budget; it stops early only when it runs "
        "out of untried values"
    )


def test_a_proposer_needs_nothing_but_propose():
    """Everything else is the base class's, which is what makes type 1 cheap."""
    for method in ("attempt", "propose_valid", "can_move"):
        assert hasattr(Sweep, method)
    assert Sweep.__dict__["propose"] is not Optimizer.propose


def test_the_loop_refuses_an_unknown_optimizer(tmp_path):
    with pytest.raises(ValueError, match="unknown optimizer"):
        Loop(MODE, _config(tmp_path / "run", optimizer="nonsense"), project=Path("."))


# ── the Run handle ─────────────────────────────────────────────────────────
def test_the_handle_offers_the_primitives_a_strategy_is_built_from(tmp_path):
    loop = Loop(MODE, _config(tmp_path / "run"), project=Path("."))
    loop.init()
    run = Run(_loop=loop)
    assert run.seed() is loop.population[0]
    assert run.history() == [], "nothing has been scored"
    assert run.budget_left() == loop.config.max_evaluations
    assert run.parents(2), "a strategy asking before the first epoch must get the seed"
    assert isinstance(run.state(), dict)
    assert run.knob_surface()
    assert run.rejections() == []


def test_evaluating_past_the_budget_is_refused(tmp_path):
    loop = Loop(MODE, _config(tmp_path / "run", max_evaluations=1), project=Path("."))
    loop.init()
    run = Run(_loop=loop)
    candidate = Candidate.from_state(loop.mode.state_dict())
    run.evaluate(candidate)
    assert run.budget_left() == 0
    with pytest.raises(OuterLoopError, match="budget is spent"):
        run.evaluate(Candidate.from_state(loop.mode.state_dict()))


def test_the_two_primitives_are_distinct(tmp_path):
    """`attempt` is cheap and spends nothing; `evaluate` is expensive."""
    loop = Loop(MODE, _config(tmp_path / "run"), project=Path("."))
    loop.init()
    run = Run(_loop=loop)
    before = loop.evaluations
    run.attempt(Proposal(edit={"op": "remove_node", "id": "exit"},
                         parent=run.seed().id), run.seed())
    assert loop.evaluations == before, "a refusal must not spend budget"
    assert run.rejections(), "a refusal must be recorded"


def test_logging_reaches_a_file(tmp_path):
    loop = Loop(MODE, _config(tmp_path / "run"), project=Path("."))
    loop.init()
    run = Run(_loop=loop)
    run.log("hello")
    assert "hello" in (loop.root / "strategy.log").read_text()


# ── type 2: overriding one step ────────────────────────────────────────────
def test_a_strategy_can_override_parent_selection(tmp_path):
    class BestOnly(Loop):
        name = "best-only"

        def select_parents(self, n: int, *, by: str = "score") -> list[Candidate]:
            return super().select_parents(1)

    loop = BestOnly(MODE, _config(tmp_path / "run"), project=Path("."))
    report = loop.run()
    assert report["evaluations"] >= 2
    assert report["bound"] == "epochs"


def test_a_strategy_can_override_how_often_it_asks(tmp_path):
    class Once(Loop):
        name = "once"

        def install(self, run, proposals, parents):  # type: ignore[no-untyped-def]
            accepted = []
            for proposal in proposals:
                outcome = run.attempt(proposal, parents[0])
                if isinstance(outcome, Candidate):
                    accepted.append(outcome)
            return accepted

    asked = {"count": 0}
    original = Once.batch_size

    def batch_size(self, run):  # type: ignore[no-untyped-def]
        asked["count"] += 1
        return original(self, run)

    Once.batch_size = batch_size  # type: ignore[assignment]
    loop = Once(MODE, _config(tmp_path / "run", epochs=3), project=Path("."))
    loop.run()
    assert asked["count"] == 3, "one ask per epoch, with no retry round"


def test_a_strategy_can_reject_a_candidate_and_still_archive_it(tmp_path):
    class Picky(Loop):
        name = "picky"

        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(*args, **kwargs)
            self.seen = 0

        def accepts(self, run, candidate):  # type: ignore[no-untyped-def]
            self.seen += 1
            return False  # nothing is kept

        def update(self, run, candidate):  # type: ignore[no-untyped-def]
            pass  # ... but it is still seen

    loop = Picky(MODE, _config(tmp_path / "run"), project=Path("."))
    loop.run()
    assert loop.seen >= 1, "accepts was never consulted"
    # `accepts` gates what the optimizer produced, not the starting population: the
    # seed batch is evaluated before any proposal exists.
    assert len(loop.population) <= loop.config.batch_size, (
        "a candidate the strategy rejected was added to the population"
    )


def test_a_strategy_can_stop_for_its_own_reason(tmp_path):
    class Impatient(Loop):
        name = "impatient"

        def should_stop(self, run):  # type: ignore[no-untyped-def]
            if len(run.history()) >= 2:
                return "seen_enough"
            return None

    loop = Impatient(MODE, _config(tmp_path / "run", epochs=9), project=Path("."))
    report = loop.run()
    assert report["bound"] == "seen_enough"
    assert report["evaluations"] < 9


def test_a_strategy_that_overrides_nothing_is_the_default(tmp_path):
    """The default loop is a strategy like any other, not a special case."""
    plain = Loop(MODE, _config(tmp_path / "a"), project=Path(".")).run()
    assert plain["bound"] == "epochs"
    assert plain["best_score"] is not None


def test_keep_is_separate_from_evaluate(tmp_path):
    """A fork whose branches are all discarded is a legitimate search."""

    class Discard(Loop):
        name = "discard"

        def strategy(self, run: Run) -> None:
            seed = run.seed()
            outcome = run.attempt(
                Proposal(edit={"op": "set_knob", "name": "order", "value": "asc"},
                         parent=seed.id),
                seed,
            )
            assert isinstance(outcome, Candidate)
            run.evaluate(outcome)  # evaluated, deliberately not kept
            run.stop("discarded")

    loop = Discard(MODE, _config(tmp_path / "run"), project=Path("."))
    report = loop.run()
    assert report["evaluations"] == 1
    assert report["leaderboard"] == [], "an unkept candidate is not in the population"
    assert report["bound"] == "discarded"
