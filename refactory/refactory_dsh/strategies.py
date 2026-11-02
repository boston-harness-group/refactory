"""The SRF strategies, as outer loops.

SRF ships thirteen modes that do different things with the same input. What varies
between them is **control flow**, not policy: a switch inside an iteration, a fork
with no loop at all, a staged curriculum, an island model. That is why the seam is
an overridable outer loop and not a set of hooks on a fixed one.

Each class here is the same strategy operating at the outer level: it searches over
*mode states* rather than over solution programs, and is scored by the mode's own
`evaluate`. What ports is the control flow. What differs from the SRF original is
recorded in each docstring, because "the same algorithm" is a claim worth stating
precisely.

Counterpart names are provisional and deliberately recognisable.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from refactory_dsh.loop import Loop
from refactory_dsh.optim import Candidate, Proposal
from refactory_dsh.outerloop import OuterLoop, Run

# ── the archives a strategy keeps ──────────────────────────────────────────


class Archive:
    """A set of candidates a strategy keeps, and how it picks from them."""

    def add(self, candidate: Candidate) -> None:
        raise NotImplementedError

    def all(self) -> list[Candidate]:
        raise NotImplementedError

    def best(self) -> Candidate | None:
        kept = self.all()
        return max(kept, key=lambda c: (c.score or 0.0)) if kept else None


@dataclass
class FlatArchive(Archive):
    """Keeps everything. The default loop's behaviour."""

    members: list[Candidate] = field(default_factory=list)

    def add(self, candidate: Candidate) -> None:
        if not any(existing.id == candidate.id for existing in self.members):
            self.members.append(candidate)

    def all(self) -> list[Candidate]:
        return list(self.members)



@dataclass
class TreeArchive(Archive):
    """A tree. A node's parent is the node it was expanded from."""

    members: list[Candidate] = field(default_factory=list)
    parents: dict[str, str] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)

    def add(self, candidate: Candidate) -> None:
        if any(existing.id == candidate.id for existing in self.members):
            return
        self.members.append(candidate)
        if candidate.parent:
            self.parents[candidate.id] = candidate.parent
            self.children.setdefault(candidate.parent, []).append(candidate.id)

    def all(self) -> list[Candidate]:
        return list(self.members)

    def depth(self, candidate_id: str) -> int:
        seen: set[str] = set()
        depth = 0
        while candidate_id in self.parents and candidate_id not in seen:
            seen.add(candidate_id)
            candidate_id = self.parents[candidate_id]
            depth += 1
        return depth

    def deepest(self) -> int:
        return max((self.depth(c.id) for c in self.members), default=0)


# ── the strategies ─────────────────────────────────────────────────────────


def _bin(value: int, edges: tuple[int, ...]) -> int:
    """Which band a number falls in, for a feature axis."""
    return sum(1 for edge in edges if value > edge)


class BestOfN(Loop):
    """SRF `best_of_n`: `Fork(N samples) → Join → select_best`.

    Differs from the SRF original in that the N samples are N mode states rather
    than N solution programs, so the "fork" is sequential here and the DSH runtime
    is what would make it concurrent.
    """

    name = "best_of_n"
    width = 4

    def strategy(self, run: Run) -> None:
        seed = run.seed()
        for index in range(self.width):
            if run.budget_left() <= 0:
                break
            edit = self._variant(index, seed)
            if edit is None:
                continue
            outcome = run.attempt(Proposal(edit=edit, parent=seed.id), seed)
            if isinstance(outcome, list):
                run.log(f"variant {index} refused: {outcome[0]}")
                continue
            run.keep(run.evaluate(outcome))
        run.stop("samples_exhausted")

    def _variant(self, index: int, seed: Candidate) -> dict[str, Any] | None:
        from refactory_dsh.nn import Knob, edits

        entries = seed.state.get("knobs") or {}
        movable = [
            name
            for name, entry in entries.items()
            if not entry["spec"].get("frozen", False)
        ]
        if not movable:
            return None
        name = movable[index % len(movable)]
        knob = Knob.from_spec(entries[name]["spec"])
        rng = random.Random(f"best_of_n:{self.config.seed}:{index}")
        value = knob.sample(rng, entries[name]["value"])
        if value is None:
            return None
        return {"op": "set_knob", "name": name, "value": value}

    def update(self, run: Run, candidate: Candidate) -> None:
        """No archive: best-of-N is a single round, so keeping is enough."""



class Reflective(Loop):
    """SRF `gepa`: reflective evolution, with a merge branch.

    `merge_decision` is a switch, not a rewind: both outcomes continue the same
    iteration down different pipelines. That is the shape a policy hook cannot
    express, and it is why this is an outer loop.
    """

    name = "reflective"
    requires_model = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.merges = 0
        self.mutations = 0

    def strategy(self, run: Run) -> None:
        while True:
            reason = self.should_stop(run)
            if reason:
                run.stop(reason)
                return
            branch = "merge" if self._merge_now(run) else "mutate"
            if not self.epoch_step(run, branch=branch):
                run.stop("epochs")

    def _merge_now(self, run: Run) -> bool:
        """Merge once there are enough parents to have something to combine."""
        return len(self.select_parents(3)) >= 3 and (self.merges + self.mutations) % 3 == 2

    def epoch_step(self, run: Run | None = None, *, branch: str = "mutate") -> bool:
        """Record the branch **and publish it**, so the proposal can act on it.

        Counting the branch is not making the decision: a merge that produces the
        same proposals as a mutation is a mutation with a counter. The branch goes
        into the proposal context, and :meth:`propose` reads it.
        """
        if branch == "merge":
            self.merges += 1
        else:
            self.mutations += 1
        run = run or Run(_loop=self)
        self.strategy_decision = {"branch": branch}
        run.log(f"branch={branch}")
        return super().epoch_step(run)

    def propose(
        self, run: Run, parents: Sequence[Candidate], n: int
    ) -> list[Proposal]:
        """On a merge, take a candidate whose knobs come from two parents.

        This is what GEPA's merge branch means, and it is the difference between a
        merge and a mutation: a mutation moves one parent's values, a merge takes
        one parent's structure and another's settings. Without this the branch is
        indistinguishable from the mutation it was chosen over.
        """
        if self.strategy_decision.get("branch") != "merge" or len(parents) < 2:
            return super().propose(run, parents, n)
        left, right = parents[0], parents[1]
        edits = [
            {"op": "set_knob", "name": name, "value": entry["value"]}
            for name, entry in sorted((right.state.get("knobs") or {}).items())
            if name in (left.state.get("knobs") or {})
            and (left.state["knobs"][name] or {}).get("value") != entry.get("value")
        ]
        if not edits:
            # Nothing to combine: the two parents differ structurally, not in their
            # settings. Fall back rather than propose an edit that changes nothing,
            # which the harness would refuse as `E-noop` and the batch slot would be
            # lost to a branch that could not do its job.
            return super().propose(run, parents, n)
        return [
            Proposal(
                edit=edits[index % len(edits)],
                rationale=f"merge {left.id} with {right.id}",
                parent=left.id,
            )
            for index in range(max(1, n))
        ]


class ReflectiveSampler(Loop):
    """SRF `shinka`: reflection inside sampling.

    Every proposer already receives the recent refusals: the loop puts a summary
    of them in `ProposalContext.rejections`, so "the prompt carries prior
    rejections" is the base behaviour and not this strategy's distinction. What
    this adds is acting on them — a proposal that repeats a refused edit is
    dropped before it is attempted, rather than refused a second time.
    """

    name = "reflective_sampler"
    requires_model = True

    def propose(self, run: Run, parents: Sequence[Candidate], n: int) -> list[Proposal]:
        proposals = super().propose(run, parents, n)
        # `Run.rejections()` is the summarised *reasons*, which is what a prompt
        # wants; the edits themselves live on the loop, which is what this needs.
        refused = self.rejections
        self.last_reflection = run.rejections()
        # The refusals this run has already paid for, as comparable edits. An
        # attempt that failed once fails again for the same reason, so re-proposing
        # it spends a batch slot to learn what the run already knows.
        seen = {
            json.dumps(rejection.edit, sort_keys=True, default=str)
            for rejection in refused
        }
        kept = [
            proposal
            for proposal in proposals
            if json.dumps(proposal.edit, sort_keys=True, default=str) not in seen
        ]
        self.reflected_away = len(proposals) - len(kept)
        return kept or proposals


class AdaptiveMeta(Loop):
    """SRF `adaevolve`: an adaptive meta-level.

    `meta_gate` routes an iteration to either the normal pipeline or the meta one,
    on a criterion of its own, and the choice is recorded.
    """

    name = "adaptive_meta"
    requires_model = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.meta_taken = 0
        self._best: float | None = None

    def strategy(self, run: Run) -> None:
        while True:
            reason = self.should_stop(run)
            if reason:
                run.stop(reason)
                return
            meta = self._meta_now(run)
            if meta:
                self.meta_taken += 1
            # A meta-level that only logs has not adapted to anything. On a stall
            # the loop widens its own batch: the evidence says the current width is
            # not producing contrast, and asking for the same width again is the
            # definition of not adapting.
            self.strategy_decision = ({"meta": True, "reason": "no progress since the last epoch"}
                                      if meta else {"meta": False})
            run.log(f"meta={meta}")
            if not self.epoch_step(run):
                run.stop("epochs")

    def batch_size(self, run: Run) -> int:
        """Widen on a stall, so the meta decision costs and buys something.

        Two more candidates is a real budget decision, which is the point: a
        meta-level that changes nothing is a counter, not a policy.
        """
        base = super().batch_size(run)
        return base + 2 if self.strategy_decision.get("meta") else base

    def _meta_now(self, run: Run) -> bool:
        """Fire the meta branch when progress has stalled."""
        scored = [c.score for c in run.history() if c.score is not None]
        if len(scored) < 3:
            return False
        previous, self._best = self._best, max(scored)
        return previous is not None and self._best <= previous


class TreeSearch(Loop):
    """SRF `aide`: `policy → action_gate(draft|improve|debug) → tree_update`.

    The archive is a tree, so a candidate's parent is the node it expanded from and
    the search can say how deep it has gone.
    """

    name = "tree_search"
    requires_model = True
    actions = ("draft", "improve", "debug")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tree = TreeArchive()

    def update(self, run: Run, candidate: Candidate) -> None:
        self.tree.add(candidate)

    def propose(self, run: Run, parents: Sequence[Candidate], n: int) -> list[Proposal]:
        parents = self.select_parents(1) or list(parents)
        self.last_action = self.actions[len(self.tree.all()) % len(self.actions)]
        return super().propose(run, parents, n)


class StagedCurriculum(Loop):
    """SRF `ai_sci_v2`: a staged curriculum, not a loop.

    `initial → tuning → creative → ablation`, each with its own action gate. Every
    stage runs, in order, before the next.
    """

    name = "staged_curriculum"
    requires_model = True
    stages = ("initial", "tuning", "creative", "ablation")
    per_stage = 2

    def strategy(self, run: Run) -> None:
        """Run each stage in order, and **publish which stage is running**.

        A stage is an instruction to the proposer — "initial", "tuning",
        "creative", "ablation" are four different things to ask for. Logging the
        name and running the ordinary step makes the curriculum a schedule with no
        content: every stage proposes the same way, so the stages differ only in
        the log that says they happened.
        """
        self.stages_done = []
        for index, stage in enumerate(self.stages):
            self.stages_done.append(stage)
            self.strategy_decision = {
                "stage": stage,
                "stage_index": index,
                "stages": list(self.stages),
            }
            run.log(f"stage={stage}")
            for _ in range(self.per_stage):
                if not self.epoch_step(run):
                    run.stop("epochs")
                    return
        run.stop("curriculum_complete")


class StagedV1(StagedCurriculum):
    """SRF `ai_sci_v1`: the earlier staged variant."""

    name = "staged_v1"
    stages = ("explore", "exploit")


class ResearchThenOptimize(Loop):
    """SRF `autoresearch`: a pre-phase, then the loop.

    `research_phase → research_llm` runs once, before the search starts, and what it
    concludes is available to every proposal after it.
    """

    name = "research_then_optimize"
    requires_model = True

    def strategy(self, run: Run) -> None:
        self.research_note = self._research(run)
        # Published, not merely kept: a note in an attribute reaches no prompt and
        # so informs no proposal. This is what puts it in front of a proposer.
        self.strategy_decision = {"research": self.research_note}
        run.log(f"research={self.research_note!r}")
        if not self.epoch_step(run):
            run.stop("epochs")
            return
        super().strategy(run)

    def _research(self, run: Run) -> str:
        """What is known before searching: the surface, and what has been tried."""
        live = [k["name"] for k in run.knob_surface() if not k["frozen"]]
        return f"{len(live)} movable knob(s): {', '.join(sorted(live))}"


class Autoscientists(Loop):
    """SRF `autoscientists`: `Fork(N scientists) → Join → merge`.

    N independent branches, then a step that sees all of them.
    """

    name = "autoscientists"
    width = 3
    requires_model = True

    def strategy(self, run: Run) -> None:
        seed = run.seed()
        produced: list[Candidate] = []
        for index in range(self.width):
            if run.budget_left() <= 0:
                break
            edit = BestOfN._variant(self, index, seed)  # the same N-sample shape
            if edit is None:
                continue
            outcome = run.attempt(Proposal(edit=edit, parent=seed.id), seed)
            if isinstance(outcome, list):
                continue
            produced.append(run.keep(run.evaluate(outcome)))
        self.merged = len(produced)
        if len(produced) >= 2 and run.budget_left() > 0:
            best = max(produced, key=lambda c: (c.score or 0.0))
            merge = self._merge(produced, best)
            if merge is not None:
                outcome = run.attempt(Proposal(edit=merge, parent=best.id), best)
                if not isinstance(outcome, list):
                    run.keep(run.evaluate(outcome))
        run.stop("merged")

    def _merge(self, produced: Sequence[Candidate], best: Candidate) -> dict[str, Any] | None:
        """Combine the branches: take one knob value from a different scientist.

        A merge that sees every branch, which is the whole reason the SRF mode forks
        before it merges rather than sampling N times.
        """
        donors = [c for c in produced if c.id != best.id]
        if not donors:
            return None
        donor = donors[0]
        mine = best.state.get("knobs") or {}
        theirs = donor.state.get("knobs") or {}
        for name in sorted(mine):
            if mine[name]["value"] != theirs[name]["value"]:
                return {"op": "set_knob", "name": name, "value": theirs[name]["value"]}
        return None


# ── selection ──────────────────────────────────────────────────────────────

#: The strategies, by name. `default` is the harness's own loop.
STRATEGIES: dict[str, type[Loop]] = {
    "default": Loop,
    BestOfN.name: BestOfN,
    Reflective.name: Reflective,
    ReflectiveSampler.name: ReflectiveSampler,
    AdaptiveMeta.name: AdaptiveMeta,
    TreeSearch.name: TreeSearch,
    StagedCurriculum.name: StagedCurriculum,
    StagedV1.name: StagedV1,
    ResearchThenOptimize.name: ResearchThenOptimize,
    Autoscientists.name: Autoscientists,
}

#: Every SRF mode ported into this repository, counted once. Five of the thirteen are
#: optimizers -- they choose a parent and propose, which is a search policy -- and the
#: rest are outer loops, which change what an iteration is. Asserted by the tests, so
#: a port can be neither quietly dropped nor filed in the wrong bucket.
SRF_PORT_COUNT = 13

#: SRF keeps an optimizer of its own in ``srf/lab/``, beside its thirteen modes:
#: ``map_elites.py``, a multi-emitter MAP-Elites over mode and knob space. It is not
#: one of the modes, so it is counted separately, and it is an optimizer for the same
#: reason the five are: it selects.
SRF_LAB_PORT_COUNT = 1

#: SRF's *modes* that are optimizers rather than loops, by the name they register
#: under. Four of the thirteen select a parent and propose, which is a search policy;
#: the other nine change what an iteration is. `map_elites` is the fifth optimizer but
#: it is not one of the thirteen -- it lives in `srf/lab/`, which is why it is counted
#: by `SRF_LAB_PORT_COUNT` and not here.
SRF_OPTIMIZER_PORTS: tuple[str, ...] = (
    "archive",        # scs
    "evolutionary",   # evox
    "islands",        # openevolve
    "hill_climb",     # karpathy
)


def names() -> list[str]:
    return sorted(STRATEGIES)


def resolve(name: str) -> type[Loop]:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; known: {names()}")
    return STRATEGIES[name]


def create(name: str, mode_id: str, config: Any = None, **kwargs: Any) -> Loop:
    """Build a strategy by name, failing loud on an unknown one."""
    return resolve(name)(mode_id, config, **kwargs)


def register(name: str, cls: type[Loop]) -> None:
    """Add a strategy without editing the library."""
    if not issubclass(cls, OuterLoop):
        raise TypeError(f"{cls.__name__} is not an OuterLoop")
    STRATEGIES[name] = cls
