"""The optimizers whose choice is *which parent* to breed from.

Issue #32 found two extension points that were not what the code claimed:
`outerloop.py` says "most research ideas live here [in a proposer]", and the
registry disagreed -- every SRF-derived method was an `OuterLoop`, and the
`Optimizer` registry held two hand-written entries.

The cause was structural. `Loop.optimizer_step` handed a proposer
`select_parents(self.config.elite)`, and `Candidate.describe()` carries a score and
the edits that produced it but **not the state**, so a proposer only ever saw the
elite's states. Island selection, a niche grid and a tournament over an archive all
need candidates the loop was not handing over, which is why those five had to be
loops. The loop passes the evaluated population now.

These assert the policies, not the machinery: each one is here because it chooses
differently, so each test names the difference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refactory_dsh import optim, registry
from refactory_dsh.optim.base import ProposalContext
from refactory_dsh.optim.policies import (
    ArchiveSelector,
    EvolutionarySelector,
    HillClimbSelector,
    IslandsSelector,
    MapElitesSelector,
)

ROOT = Path(__file__).resolve().parent.parent.parent
MODE = "circle_packing/adaptive"


def _parents() -> list:
    """A pool with contrast in it: three distinct scores and two structures."""
    from refactory_dsh.optim import Candidate

    mode = registry.resolve(ROOT, MODE)[1]()
    base = mode.state_dict()
    made = []
    for index in range(6):
        state = registry.resolve(ROOT, MODE)[1]().state_dict()
        state["knobs"]["grid"]["value"] = 8 + index * 2
        if index % 2:  # a second structure: one more agent step in the graph itself
            state["graph"]["nodes"]["extra"] = {"id": "extra", "kind": "agent",
                                                "prompt_template": "Look again."}
        candidate = Candidate.from_state(state, edits=[{"op": "add_node"}] if index % 2 else [])
        candidate.score = float(index)
        candidate.evaluated = True
        made.append(candidate)
    assert len({c.id for c in made}) == len(made), "the fixtures collide"
    return made


def _ctx(**overrides: object) -> ProposalContext:
    return ProposalContext(**overrides)  # type: ignore[arg-type]


# ── every policy is registered, and is an optimizer ────────────────────────
@pytest.mark.parametrize(
    "name", ["archive", "evolutionary", "islands", "hill_climb", "map_elites"]
)
def test_each_policy_is_reachable_by_name(name):
    assert name in optim.names(), f"{name} is not selectable"
    built = optim.create(name, seed=0)
    assert isinstance(built, optim.Optimizer)


# ── the differences, one test each ─────────────────────────────────────────
def test_archive_breeds_from_the_best_of_everything_seen():
    optimizer = ArchiveSelector(seed=0)
    pool = _parents()
    chosen = optimizer.select(_ctx(), 3, pool)
    assert chosen, "the archive selected nothing"
    assert chosen[0].score == max(c.score for c in pool), (
        "an archive's best is the best it has seen, whichever batch it came from"
    )


def test_the_archive_keeps_more_than_the_elite():
    """Otherwise it is a hill climb with a different name."""
    optimizer = ArchiveSelector(seed=0, limit=12)
    optimizer.select(_ctx(), 1, _parents())
    assert len(optimizer.members) > 1, "an archive that keeps one point is not an archive"


def test_evolutionary_plays_a_tournament_so_a_weaker_member_can_breed():
    """The one way it differs from `archive`, and the reason both are ported."""
    pool = _parents()
    winners = set()
    for seed in range(12):
        chosen = EvolutionarySelector(seed=seed, tournament=3).select(_ctx(), 1, pool)
        winners.add(chosen[0].id)
    assert len(winners) > 1, (
        "every tournament picked the same winner, so this is not a tournament"
    )
    assert max(c.score for c in pool) in {c.score for c in pool if c.id in winners}


def test_islands_place_a_candidate_stably():
    """`hash()` is salted per process, so it cannot decide which island a lineage is.

    The original `IslandArchive` derived the island from `hash(candidate.id)`, which
    means the same candidate sat in a different island on a resumed run than it did
    when it was evaluated. It is invisible until a run is resumed, or a test is run
    twice, which is why it survived.
    """
    optimizer = IslandsSelector(seed=0, count=3)
    pool = _parents()
    first = [optimizer.island_of(c) for c in pool]
    assert first == [optimizer.island_of(c) for c in pool], "the island moved"
    assert first == [IslandsSelector(seed=0, count=3).island_of(c) for c in pool], (
        "a fresh optimizer places the same candidate differently"
    )
    assert all(0 <= island < 3 for island in first)


def test_islands_select_from_every_populated_island():
    """Round-robin across the islands that have members, not best-overall."""
    optimizer = IslandsSelector(seed=0, count=3)
    pool = _parents()
    populated = {optimizer.island_of(c) for c in pool}
    chosen = optimizer.select(_ctx(), 6, pool)

    assert chosen, "the islands selected nothing"
    assert {optimizer.island_of(c) for c in chosen} == populated, (
        "selection skipped an island that has members, so a lineage was starved"
    )
    # Every pick is the best its island has, which is what keeps a weak island alive.
    for island in populated:
        members = [c for c in pool if optimizer.island_of(c) == island]
        best = max(members, key=lambda c: (c.score or 0.0))
        assert best in chosen or len(chosen) < len(populated)


def test_hill_climb_breeds_from_one_point():
    chosen = HillClimbSelector(seed=0).select(_ctx(), 5, _parents())
    assert len(chosen) == 1, "a hill climb walks from one point, not from a population"
    assert chosen[0].score == 5.0


def _shaped(nodes: int, *, structural: bool, score: float):
    """A candidate whose niche is known, so a test can aim at one."""
    from refactory_dsh.optim import Candidate

    state = registry.resolve(ROOT, MODE)[1]().state_dict()
    # A distinct node count is a distinct niche on the size axis.
    template = next(iter(state["graph"]["nodes"].values()))
    state["graph"]["nodes"] = {f"n{i}": dict(template, id=f"n{i}") for i in range(nodes)}
    candidate = Candidate.from_state(
        state, edits=[{"op": "add_node"}] if structural else []
    )
    candidate.score = score
    candidate.evaluated = True
    return candidate


def test_map_elites_keeps_a_weak_candidate_beside_a_strong_one():
    """The property that makes it MAP-Elites rather than an archive.

    A weaker candidate in an occupied niche does not displace its champion; the same
    candidate in an *empty* niche is kept. An archive comparing every candidate to
    the whole population would drop it in both cases, and that difference is the
    whole reason the method is ported.
    """
    optimizer = MapElitesSelector(seed=0)
    strong = _shaped(4, structural=False, score=0.9)
    weak_same = _shaped(4, structural=False, score=0.1)
    weak_elsewhere = _shaped(14, structural=False, score=0.05)

    key = optimizer.features(strong)
    assert optimizer.features(weak_same) == key, "the fixture is not in the same niche"
    assert optimizer.features(weak_elsewhere) != key, "the fixture is not elsewhere"

    optimizer.add(strong)
    assert optimizer.cells[key] is strong

    optimizer.add(weak_same)
    assert optimizer.cells[key] is strong, (
        "a weaker candidate in an occupied niche displaced its champion"
    )

    optimizer.add(weak_elsewhere)
    other = optimizer.features(weak_elsewhere)
    assert optimizer.cells[other] is weak_elsewhere, (
        "a candidate in an empty niche was dropped for being weak, so the search "
        "keeps only the best frontier and the map never fills"
    )
    assert len(optimizer.cells) == 2


def test_map_elites_reads_the_nodes_out_of_the_graph():
    """A niche axis that always reads zero makes a grid of one cell.

    `state["nodes"]` does not exist -- the nodes are at `state["graph"]["nodes"]` --
    and `.get` returns None rather than raising, so a wrong path silently collapses
    every candidate into one niche and the map looks like it is working.
    """
    optimizer = MapElitesSelector(seed=0)
    small = _shaped(3, structural=False, score=0.5)
    large = _shaped(14, structural=False, score=0.5)
    assert optimizer.features(small) != optimizer.features(large), (
        "two graphs of very different size share a niche, so the size axis is dead"
    )


def test_map_elites_draws_parents_across_niches():
    """Selection has to use the map, or the grid is bookkeeping."""
    optimizer = MapElitesSelector(seed=0)
    pool = _parents()
    chosen = optimizer.select(_ctx(), 24, pool)
    niches = {optimizer.features(c) for c in chosen}
    assert len(niches) >= 2, f"every parent came from one niche ({niches})"
    scores = [c.score for c in chosen]
    assert min(scores) < max(scores), (
        "selection only ever returned the best candidate, which is every other "
        "optimizer here"
    )


def test_map_elites_reports_its_coverage():
    optimizer = MapElitesSelector(seed=0)
    optimizer.select(_ctx(), 1, _parents())
    assert 0.0 < optimizer.coverage() <= 1.0


# ── the contract that made them possible ───────────────────────────────────
def test_a_proposer_is_handed_the_population_not_only_the_elite():
    """The change that let these five leave the loop bucket.

    `describe()` carries no state, so the elite alone cannot support a policy that
    needs a wider pool. If this regresses, the five quietly become loops again.
    """
    import inspect

    from refactory_dsh.loop import Loop

    source = inspect.getsource(Loop.optimizer_step)
    assert "for candidate in self.population if candidate.evaluated" in source, (
        "`optimizer_step` no longer hands over the evaluated population, so no "
        "optimizer can select a parent outside the elite"
    )


def test_an_optimizer_with_no_parent_policy_still_gets_the_elite():
    """Widening the pool must not change what an existing optimizer proposes."""
    from refactory_dsh.optim.random_optimizer import RandomOptimizer

    pool = _parents()
    ctx = _ctx(elite=2)
    proposals = RandomOptimizer(seed=0).propose(ctx, 4, pool)
    elite_ids = {c.id for c in sorted(pool, key=lambda c: c.score, reverse=True)[:2]}
    assert proposals, "the elite pass proposed nothing"
    assert {p.parent for p in proposals} <= elite_ids, (
        "an optimizer with no parent policy bred from outside the elite, which "
        "silently changes every run that uses it"
    )


def test_policies_propose_edits_against_a_named_parent():
    """A policy selects; the sampling of an edit is the same for all of them."""
    pool = _parents()
    for optimizer in (ArchiveSelector(seed=0), IslandsSelector(seed=0),
                      HillClimbSelector(seed=0), MapElitesSelector(seed=0),
                      EvolutionarySelector(seed=0)):
        proposals = optimizer.propose(_ctx(), 2, pool)
        assert proposals, f"{optimizer.name} proposed nothing"
        for proposal in proposals:
            assert proposal.parent in {c.id for c in pool}
            assert proposal.edit.get("op")
            assert proposal.rationale == optimizer.policy
