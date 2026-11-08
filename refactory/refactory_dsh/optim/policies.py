"""Optimizers whose distinguishing choice is *which parent* to breed from.

The audit behind issue #32 found the two extension points were not what the code
claimed. `outerloop.py` says "most research ideas live here [in a proposer]", and
the registry said otherwise: every one of the SRF-derived methods was an
`OuterLoop`, and the `Optimizer` registry held two hand-written entries.

The reason was structural, not an oversight. `Loop.optimizer_step` passed a
proposer `select_parents(self.config.elite)` -- the top few candidates -- and
`Candidate.describe()` carries a score and the edits that produced it, **not the
state**. A proposer therefore only ever saw the elite's states, so a policy that
needs a wider pool could not be written as one. Island selection, a niche grid and
a tournament over an archive all need candidates the loop was not handing over.

**The fix is the contract, not the filing.** The loop now passes the evaluated
population and the optimizer selects; selection is part of "how to search", which is
what an optimizer owns. The five methods that were loops only because they selected
parents are proposers now.

The base class here is shared so the five differ only in the choice they make. Each
returns parents; the sampling of an edit against a chosen parent is the same for all
of them and is written once. Run alone (``--optimizer map_elites``) a policy samples
random edits; paired with another proposer (``--parent-picker map_elites``, see
:mod:`refactory_dsh.optim.compose`) it only picks, and the proposer writes.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Sequence

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim.base import Candidate, Optimizer, Proposal, ProposalContext


def pipeline_shape(state: dict[str, Any] | None) -> dict[str, Any]:
    """What a candidate's graph is made of, independent of its prompts and settings.

    ``agents`` (how many model steps), ``gate`` (a gate node or a conditional edge,
    i.e. a verdict loop or a branch), ``parallel`` (a fork), ``code`` (a node whose
    Python an edit wrote, `refactory_dsh.snippet.is_code_node`) and ``nodes``. Read
    from ``state["graph"]``; a state without a graph has every count at zero.
    """
    from refactory_dsh.snippet import is_code_node

    graph = (state or {}).get("graph") or {}
    nodes = graph.get("nodes") or {}
    if not isinstance(nodes, dict):  # a hand-written state, not a mode's
        nodes = {}
    members = [node for node in nodes.values() if isinstance(node, dict)]
    edges = [edge for edge in (graph.get("edges") or ()) if isinstance(edge, dict)]
    return {
        "nodes": len(members),
        "agents": sum(1 for node in members if node.get("kind") == "agent"),
        "gate": any(node.get("kind") == "gate" for node in members)
        or any(edge.get("condition") for edge in edges),
        "parallel": any(node.get("kind") == "fork" for node in members),
        "code": any(is_code_node(node) for node in members),
    }


def shape_label(shape: dict[str, Any]) -> str:
    """A shape in a few words, for a request or a report."""
    parts = [f"{shape.get('agents', 0)} agent step(s)"]
    parts += [name for name in ("gate", "parallel", "code") if shape.get(name)]
    return ", ".join(parts) + f"; {shape.get('nodes', 0)} nodes"


class ParentSelector(Optimizer):
    """An optimizer that chooses parents by a policy, then samples an edit for each.

    A subclass implements :meth:`select` and nothing else. `propose` is the same for
    all of them, which is the point: these differ in one decision, and a base class
    that made them repeat the sampling would hide which decision that is.
    """

    #: Index of the choice, for a reproducible stream and a readable rationale.
    policy: str = ""

    def __init__(self, *, k: int = 3, seed: int = 0) -> None:
        super().__init__(k=k)
        self.seed = seed

    def rng(self, ctx: ProposalContext, n: int) -> random.Random:
        """Derived from the seed and how far the run has got, never carried.

        A pure function of the configuration and the step, so a resumed run proposes
        what an uninterrupted one would have.
        """
        return random.Random(f"{self.name}:{self.seed}:{ctx.generation}:{ctx.epoch}:{n}")

    def select(
        self, ctx: ProposalContext, n: int, parents: Sequence[Candidate]
    ) -> list[Candidate]:
        """The parents to breed from. The one method a subclass writes."""
        raise NotImplementedError

    def propose(
        self, ctx: ProposalContext, n: int, parents: Sequence[Candidate]
    ) -> list[Proposal]:
        if not parents:
            return []
        rng = self.rng(ctx, n)
        wanted = max(1, n)
        chosen = self.select(ctx, wanted, parents)
        if not chosen:
            return []
        proposals: list[Proposal] = []
        for index in range(wanted):
            parent = chosen[index % len(chosen)]
            edit = nn_edits.random_edit(
                parent.state,
                preset=ctx.preset,
                insertable=ctx.insertable,
                rng=rng,
            )
            if edit is None:
                continue
            proposals.append(
                Proposal(edit=edit, parent=parent.id, rationale=self.policy)
            )
        return proposals

    def _by_score(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        return sorted(candidates, key=lambda c: (c.score or 0.0), reverse=True)

    def reason(self, candidate: Candidate) -> str:
        """Why this policy chose ``candidate``, in a few words, for the request."""
        return self.policy


class ArchiveSelector(ParentSelector):
    """SRF `scs`: breed from the best of everything seen, not just this epoch.

    The archive is what makes it not a hill climb: a candidate that scored well two
    epochs ago and has since been overtaken is still a parent, so the search does not
    depend on the current batch being the good one.
    """

    name = "archive"
    policy = "bred from the archive's best"
    limit = 12

    def __init__(self, *, k: int = 3, seed: int = 0, limit: int = 12) -> None:
        super().__init__(k=k, seed=seed)
        self.limit = limit
        self.members: dict[str, Candidate] = {}

    def select(self, ctx, n, parents):
        for candidate in parents:
            self.members.setdefault(candidate.id, candidate)
        kept = self._by_score(list(self.members.values()))[: self.limit]
        self.members = {candidate.id: candidate for candidate in kept}
        return kept[:n] if kept else []


class EvolutionarySelector(ParentSelector):
    """SRF `evox`: a tournament, so a weaker member can still breed.

    Differs from `archive` in exactly one way, and it is the reason both are ported:
    taking the top n converges on one lineage, and a tournament does not.
    """

    name = "evolutionary"
    policy = "won a tournament"
    tournament = 3

    def __init__(self, *, k: int = 3, seed: int = 0, tournament: int = 3) -> None:
        super().__init__(k=k, seed=seed)
        self.tournament = max(2, tournament)

    def select(self, ctx, n, parents):
        rng = self.rng(ctx, n)
        pool = self._by_score(parents)[: max(self.tournament * 4, n)]
        if len(pool) < self.tournament:
            return pool[:n]
        chosen: list[Candidate] = []
        for _ in range(n):
            chosen.append(max(rng.sample(pool, self.tournament), key=lambda c: (c.score or 0.0)))
        return chosen


class IslandsSelector(ParentSelector):
    """SRF `openevolve`: sub-populations whose lineages never cross.

    Selection is round-robin across the populated islands rather than best-overall,
    so the best candidate from a weak island is still bred while the strongest island
    is not allowed to take every slot.
    """

    name = "islands"
    policy = "bred within its island"
    count = 3

    def __init__(self, *, k: int = 3, seed: int = 0, count: int = 3) -> None:
        super().__init__(k=k, seed=seed)
        self.count = max(1, count)
        self._turn = 0

    def island_of(self, candidate: Candidate) -> int:
        """Which island a candidate belongs to, stably.

        A digest, not ``hash()``. Python salts string hashing per process, so
        ``hash(id) % count`` puts the same candidate in a different island on a
        resumed run than it was in when it was evaluated -- and which island a
        lineage belongs to is the entire content of this policy. The original
        `IslandArchive` had this, and it is only visible when a run is resumed or a
        test happens to be run twice.
        """
        digest = hashlib.sha256(candidate.id.encode()).digest()
        return int.from_bytes(digest[:4], "big") % self.count

    def select(self, ctx, n, parents):
        islands: dict[int, list[Candidate]] = {}
        for candidate in parents:
            islands.setdefault(self.island_of(candidate), []).append(candidate)
        populated = sorted(islands)
        if not populated:
            return []
        chosen: list[Candidate] = []
        for offset in range(n):
            island = populated[(self._turn + offset) % len(populated)]
            members = islands[island]
            chosen.append(max(members, key=lambda c: (c.score or 0.0)))
        self._turn += 1
        return chosen


class HillClimbSelector(ParentSelector):
    """SRF `karpathy`: one candidate per round, from the best so far."""

    name = "hill_climb"
    policy = "the best so far"

    def select(self, ctx, n, parents):
        ranked = self._by_score(parents)
        return ranked[:1] if ranked else []


class MapElitesSelector(ParentSelector):
    """SRF's own optimizer (`srf/lab/map_elites.py`): a champion per niche.

    The property that makes it MAP-Elites rather than an archive: a candidate
    competes only against the occupant of its own cell, so a weak candidate in an
    empty niche survives beside a strong one in a crowded niche, and selection can
    return something that is not the best.

    The niches are pipeline shapes (:func:`pipeline_shape`): a structurally new
    candidate -- another agent step, a verdict loop, a parallel fork, a code node --
    is the best of its own group until something of the same shape beats it, so a
    new structure is not lost to a prompt-tuned lineage that already scores higher.
    """

    name = "map_elites"
    policy = "a champion of its niche"
    #: Cells per axis of :meth:`features`: size, agent steps, gate, parallel, code,
    #: step budget.
    dims: tuple[int, ...] = (3, 4, 2, 2, 2, 3)

    def __init__(self, *, k: int = 3, seed: int = 0, explore: float = 0.3) -> None:
        super().__init__(k=k, seed=seed)
        self.explore = explore
        self.cells: dict[tuple[int, ...], Candidate] = {}

    def features(self, candidate: Candidate) -> tuple[int, ...]:
        """Which niche a candidate falls in: its pipeline's shape, size and step budget.

        The shape axes (agent steps 0 / 1 / 2 / 3+, a gate or verdict loop, a parallel
        fork, a code node) are what makes two pipelines different answers at the
        same score. Size and step budget split a shape further.

        The nodes are at ``state["graph"]["nodes"]``, not ``state["nodes"]``. Reading
        the wrong path does not raise -- ``.get`` returns ``None`` and every
        candidate lands in one cell -- which is exactly the failure a niche grid
        hides: it looks like it is working and it is a single bucket.
        """
        state = candidate.state or {}
        shape = pipeline_shape(state)
        nodes = (state.get("graph") or {}).get("nodes") or {}
        if not isinstance(nodes, dict):
            nodes = {}
        steps = 0
        for node in nodes.values():
            if not isinstance(node, dict):
                continue
            budget = (node.get("fields") or {}).get("max_iterations")
            if isinstance(budget, int):
                steps = max(steps, budget)
        return (
            sum(1 for edge in (6, 10) if shape["nodes"] > edge),
            sum(1 for edge in (0, 1, 2) if shape["agents"] > edge),
            int(shape["gate"]),
            int(shape["parallel"]),
            int(shape["code"]),
            sum(1 for edge in (1, 3) if steps > edge),
        )

    def add(self, candidate: Candidate) -> None:
        key = self.features(candidate)
        held = self.cells.get(key)
        if held is None or (candidate.score or 0.0) > (held.score or 0.0):
            self.cells[key] = candidate

    def coverage(self) -> float:
        total = 1
        for dim in self.dims:
            total *= dim
        return len(self.cells) / total

    def reason(self, candidate: Candidate) -> str:
        shape = shape_label(pipeline_shape(candidate.state))
        return (f"{self.policy}: the best-scoring candidate of shape [{shape}], one of "
                f"{len(self.cells)} shape group(s) held")

    def select(self, ctx, n, parents):
        for candidate in parents:
            self.add(candidate)
        occupied = sorted(self.cells)
        if not occupied:
            return []
        rng = self.rng(ctx, n)
        return [self.cells[rng.choice(occupied)] for _ in range(n)]
