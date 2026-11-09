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
from pathlib import Path
from typing import Any, Sequence

from refactory_dsh.loop import Loop
from refactory_dsh.optim import Candidate, Proposal, pipeline_shape
from refactory_dsh.optim.policies import shape_label
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


def _shape_key(candidate: Candidate) -> tuple[Any, ...]:
    shape = pipeline_shape(candidate.state)
    return (shape["agents"], shape["gate"], shape["parallel"], shape["code"])


def _shape_text(candidates: Sequence[Candidate]) -> str:
    seen: dict[tuple[Any, ...], str] = {}
    for candidate in candidates:
        shape = pipeline_shape(candidate.state)
        seen.setdefault(_shape_key(candidate), shape_label(shape).split(";")[0])
    return " | ".join(seen.values())


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _is_settings_only(candidate: Candidate) -> bool:
    own = candidate.own_edits()
    return bool(own) and all(edit.get("op") == "set_knob" for edit in own)


#: How many prompt/knob-level children in a row must fail to beat their parents,
#: with every candidate of one shape, before the evidence says to try structure.
#: An evidence threshold, not a schedule: nothing turns structure on by the clock.
SETTINGS_STALL = 3


def edit_kind_decision(population: Sequence[Candidate],
                       *, stall: int = SETTINGS_STALL) -> dict[str, Any]:
    """What kind of edit the evidence points to: ``pipeline`` or ``prompt_knob``.

    Compares the top third of the scored candidates with the bottom third by
    pipeline shape (`optim.policies.pipeline_shape`) and score:

    - winners and losers have **different shapes** -> a pipeline-level change, the
      score follows structure;
    - **one shape everywhere**, and the last ``stall`` prompt/knob children all
      failed to beat their parents -> a pipeline-level change, settings have
      stopped paying and structure is untested;
    - otherwise -> a prompt/knob-level change: candidates of the same shape span
      the range, so settings explain the differences.

    Empty when fewer than two candidates are scored: no evidence, no decision.
    """
    scored = sorted((c for c in population if c.evaluated and c.score is not None),
                    key=lambda c: (-(c.score or 0.0), c.id))
    if len(scored) < 2:
        return {}
    k = max(1, len(scored) // 3)
    top, bottom = scored[:k], scored[-k:]
    top_keys = {_shape_key(c) for c in top}
    bottom_keys = {_shape_key(c) for c in bottom}
    means = (f"top {k} mean {_mean([c.score or 0.0 for c in top]):.4g}, "
             f"bottom {k} mean {_mean([c.score or 0.0 for c in bottom]):.4g}")
    pipeline = {"edit_kind": "pipeline",
                "purpose": "this round: a pipeline-level change -- add, remove, rewire, "
                           "gate or parallelise steps, or (where the run allows code steps) "
                           "add a code step that checks or extracts something "
                           "deterministically -- rather than another prompt or setting "
                           "change"}
    settings = {"edit_kind": "prompt_knob",
                "purpose": "this round: a prompt/knob-level change -- rewrite or patch a "
                           "prompt, move a setting -- on the current pipeline"}
    if not top_keys & bottom_keys:
        return {**pipeline, "why": (
            f"the best candidates have shape [{_shape_text(top)}] and the worst have "
            f"[{_shape_text(bottom)}] ({means}): the score follows the pipeline's shape")}
    if len({_shape_key(c) for c in scored}) == 1:
        by_id = {c.id: c for c in population}
        recent = [c for c in population if c.evaluated and c.score is not None
                  and c.parent in by_id and _is_settings_only(c)][-stall:]
        stalled = len(recent) >= stall and all(
            (c.score or 0.0) <= (by_id[c.parent].score or 0.0) for c in recent)
        if stalled:
            return {**pipeline, "why": (
                f"the last {len(recent)} prompt/knob edits did not beat their parents, and "
                f"every candidate so far has one shape [{_shape_text(scored)}], so the "
                "scores say nothing yet about a different structure")}
        return {**settings, "why": (
            f"every candidate so far has one shape [{_shape_text(scored)}] and prompt/knob "
            f"edits still move the score ({means})")}
    return {**settings, "why": (
        f"the best and the worst candidates share shape(s) [{_shape_text(top)}] "
        f"({means}): candidates of one shape span the range, so prompts and settings "
        "explain the differences")}


class BestOfN(Loop):
    """SRF `best_of_n`: `Fork(N samples) → Join → select_best`.

    Differs from the SRF original in that the N samples are N mode states rather
    than N solution programs, so the "fork" is sequential here and the DSH runtime
    is what would make it concurrent.
    """

    name = "best_of_n"
    width = 4

    def batch_size(self, run: Run) -> int:
        return int(self.width) if self.strategy_decision.get("samples") else super().batch_size(run)

    def strategy(self, run: Run) -> None:
        seed = run.seed()
        if getattr(self.optimizer, "uses_model", False):
            # A model proposer writes the N samples: one request, N proposals, all
            # against the seed. The knob sampler below is the random proposer's way.
            for candidate in _sample_from_optimizer(self, run, seed, self.width, (
                    f"{self.width} independent variants of the seed; each is evaluated "
                    "once and the best is kept, so make them different from each other")):
                run.keep(run.evaluate(candidate))
            run.stop("samples_exhausted")
            return
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


def _sample_from_optimizer(loop: Loop, run: Run, seed: Candidate, width: int,
                           purpose: str) -> list[Candidate]:
    """``width`` candidates the loop's optimizer writes against ``seed``, refused ones
    asked again with their reasons (``Loop.install``); the decision says what for."""
    loop.strategy_decision = {"parent": seed.id, "samples": width, "purpose": purpose}
    loop._editing_parent = seed.id
    budget = max(0, min(width, run.budget_left()))
    if not budget:
        return []
    proposals = loop.propose(run, [seed], budget)
    return loop.install(run, proposals, [seed])[:budget]



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
        if branch == "merge":
            self.strategy_decision = {
                "branch": "merge",
                "purpose": "combine two good parents: one parent's pipeline with the "
                           "other's settings"}
        else:
            # Reflection on the evidence: what kind of edit the scores point to.
            self.strategy_decision = {"branch": "mutate",
                                      **edit_kind_decision(self.population)}
        run.log(f"branch={branch}"
                + (f" edit_kind={self.strategy_decision['edit_kind']}"
                   if self.strategy_decision.get("edit_kind") else ""))
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
        # The two best parents. ``parents`` is the population in the order it was
        # made, so its first two were the seed and the first sweep candidate, whatever
        # they scored.
        left, right = sorted(parents, key=lambda c: (-(c.score or 0.0), c.id))[:2]
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
        # The merge writes settings only, so the round is a prompt/knob-level one; under
        # the portfolio it then asks for that level alone instead of refusing the merge's
        # second candidate (E-level) until the attempts run out.
        self.strategy_decision = {**self.strategy_decision, "edit_kind": "prompt_knob"}
        if self._levels_wanted:
            self._levels_wanted = ["prompt_knob"]
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
            self.strategy_decision = ({
                "meta": True, "reason": "no progress since the last epoch",
                "purpose": "widen the search: this round evaluates more candidates, so "
                           "make them differ from each other and from what was tried",
                "why": f"the best score ({self._best}) has not improved since the last epoch",
            } if meta else {"meta": False})
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

    def epoch_step(self, run: Run | None = None) -> bool:
        """Decide the action before the step, so the proposal reads it.

        The action gate was a counter (``len(tree) % 3``) kept in an attribute, so
        no proposal saw it and the strategy ran as ``default``. It decides from the
        tree now, as AIDE's does, and publishes the action and the node it expands.
        """
        self._decided = False
        # Last step's node is not this step's: the loop would refresh its records.
        self.strategy_decision = {}
        return super().epoch_step(run)

    def _decide(self, parents: Sequence[Candidate]) -> Candidate | None:
        """``debug`` the newest failed leaf, ``draft`` while nothing beats the seed,
        else ``improve`` the leader."""
        by_id = {c.id: c for c in self.population}
        expanded = {c.parent for c in self.population if c.parent}
        failed = [c for c in parents if c.evaluated and not c.exit_ok and c.id not in expanded]
        leader = self.select_parents(1)
        root = next((c for c in self.population if not c.parent), None)
        if failed:
            node, action = failed[-1], "debug"
            why = (f"candidate {node.id} failed ({str(node.error or 'no score')[:200]}) and "
                   "has not been expanded; fix what broke it")
        elif root is not None and (not leader or leader[0].id == root.id):
            node, action = root, "draft"
            why = "nothing has beaten the seed yet; write a different first version of it"
        elif leader:
            node, action = leader[0], "improve"
            why = (f"candidate {leader[0].id} leads with score {leader[0].score} at depth "
                   f"{self.tree.depth(leader[0].id)} of the tree")
        else:
            return None
        self.last_action = action
        self.strategy_decision = {"action": action, "parent": node.id, "why": why,
                                  "purpose": {"debug": "repair a failed candidate",
                                              "draft": "a new draft from the seed",
                                              "improve": "improve the best node"}[action]}
        if node.id in by_id and node.id != self._editing_parent:
            self._editing_parent = node.id
            self._refresh_edited_records()
        return node

    def propose(self, run: Run, parents: Sequence[Candidate], n: int) -> list[Proposal]:
        pool = self._gated_parents() if self.config.val_gate else list(parents)
        if not getattr(self, "_decided", False):
            # First round of the step: decide, and rebuild the request around the node.
            self._decided = True
            node = self._decide(pool)
        else:
            node = next((c for c in self.population
                         if c.id == self.strategy_decision.get("parent")), None)
        parents = [node] if node is not None else (self.select_parents(1) or list(parents))
        return super().propose(run, parents, n)


class StagedCurriculum(Loop):
    """SRF `ai_sci_v2`: a staged curriculum, not a loop.

    `initial → tuning → creative → ablation`, each with its own action gate. Every
    stage runs, in order, before the next.
    """

    name = "staged_curriculum"
    requires_model = True
    #: The curriculum is the schedule: finishing it is a normal end, and the batch its
    #: last stage proposed is scored (``Loop.finish_strategy``).
    normal_ends = ("epochs", "curriculum_complete")
    stages = ("initial", "tuning", "creative", "ablation")
    per_stage = 2
    #: What each stage asks of a proposal; a stage not listed is asked by its name.
    purposes = {
        "initial": "establish a working first improvement on the seed",
        "tuning": "tune prompts and settings of the current pipeline",
        "creative": "try something different: a new step, a different pipeline shape",
        "ablation": "remove or simplify a part, to learn which parts carry the score",
        "explore": "explore: try varied changes, structural ones included",
        "exploit": "exploit: refine what scores best so far",
    }

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
                "purpose": f"stage {index + 1} of {len(self.stages)} ({stage}): "
                           + self.purposes.get(stage, stage),
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
        self.strategy_decision = {
            "research": self.research_note,
            "purpose": "use what the research phase found before the search started"}
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

    def batch_size(self, run: Run) -> int:
        return int(self.width) if self.strategy_decision.get("samples") else super().batch_size(run)

    def strategy(self, run: Run) -> None:
        seed = run.seed()
        produced: list[Candidate] = []
        if getattr(self.optimizer, "uses_model", False):
            # A model proposer writes the N branches; the merge below sees them all.
            produced = [run.keep(run.evaluate(candidate)) for candidate in
                        _sample_from_optimizer(self, run, seed, self.width, (
                            f"{self.width} independent lines of attack on the seed, one per "
                            "scientist; a merge then combines the branches, so make each "
                            "branch a different idea"))]
        for index in range(0 if produced or getattr(self.optimizer, "uses_model", False)
                           else self.width):
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
            # A knob only one branch has (a structural branch's new step) is not one
            # to cross over.
            if name in theirs and mine[name]["value"] != theirs[name]["value"]:
                return {"op": "set_knob", "name": name, "value": theirs[name]["value"]}
        return None


# ── selection ──────────────────────────────────────────────────────────────

#: The strategies, by name. `default` is the harness's own loop.
#: Values may be ``Loop`` subclasses (built-in strategies) or pure
#: ``OuterLoop`` subclasses (custom strategies using the custom strategy seam).
STRATEGIES: dict[str, type[OuterLoop]] = {
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


def package_strategies(project: Path | None = None) -> dict[str, type[OuterLoop]]:
    """Strategies the discoverable packages contribute, by name.

    A package lists them in ``Package.strategies``, so a strategy shipped as a
    package is selectable by name (``--strategy``) the way a mode is, without a
    registration call and without joining the library's own ports in
    ``STRATEGIES``.
    """
    from refactory_dsh import registry

    found: dict[str, type[OuterLoop]] = {}
    for package in registry.discover(project).values():
        for cls in package.strategies:
            if not (isinstance(cls, type) and issubclass(cls, OuterLoop)):
                raise TypeError(
                    f"package {package.name!r} lists {cls!r}, which is not an OuterLoop"
                )
            found[cls.name] = cls
    return found


def available(project: Path | None = None) -> list[str]:
    """Every selectable strategy: the library's own and the packages'."""
    return sorted(set(STRATEGIES) | set(package_strategies(project)))


def resolve(name: str, project: Path | None = None) -> type[OuterLoop]:
    if name in STRATEGIES:
        return STRATEGIES[name]
    contributed = package_strategies(project)
    if name in contributed:
        return contributed[name]
    known = sorted(set(STRATEGIES) | set(contributed))
    raise ValueError(f"unknown strategy {name!r}; known: {known}")


def check_options(name: str, options: dict[str, Any] | None, project: Path | None = None) -> None:
    """Refuse strategy options the named strategy does not read.

    A strategy lists the options it reads in its own ``accepted_options``. One that
    declares none itself, whatever its base class declares, is not checked, since
    nothing says what it reads.
    """
    accepted = vars(resolve(name, project)).get("accepted_options")
    if accepted is None or not options:
        return
    unknown = sorted(set(options) - set(accepted))
    if unknown:
        raise ValueError(
            f"the {name} strategy has no option {', '.join(unknown)}; "
            + (f"it reads {', '.join(sorted(accepted))}" if accepted else "it takes no strategy options"))


def create(
    name: str,
    mode_id: str,
    config: Any = None,
    *,
    strategy_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Loop:
    """Build a strategy by name, failing loud on an unknown one.

    For strategies that subclass ``Loop`` directly, constructs and returns
    them.  For pure ``OuterLoop`` subclasses (the custom strategy seam),
    constructs a ``Loop`` for the infrastructure and attaches the strategy
    via ``set_custom_strategy()``.

    ``strategy_options`` are forwarded to the strategy's ``configure()``
    method.  The ``model`` kwarg is extracted and passed to ``configure()``
    as well.
    """
    project = kwargs.get("project")
    cls = resolve(name, Path(project) if project is not None else None)
    check_options(name, strategy_options, Path(project) if project is not None else None)
    model = kwargs.pop("model", None)
    if issubclass(cls, Loop):
        # A Loop subclass: construct directly, configure with options
        loop = cls(mode_id, config, model=model, **kwargs)
        if strategy_options:
            loop.configure(model=model, **strategy_options)
        return loop
    # A pure OuterLoop subclass: create a Loop for infrastructure, attach
    # the strategy object.
    loop = Loop(mode_id, config, model=model, **kwargs)
    strategy = cls()
    strategy.configure(model=model, **(strategy_options or {}))
    loop.set_custom_strategy(strategy)
    return loop


def register(name: str, cls: type[OuterLoop]) -> None:
    """Add a strategy without editing the library."""
    if not issubclass(cls, OuterLoop):
        raise TypeError(f"{cls.__name__} is not an OuterLoop")
    STRATEGIES[name] = cls
