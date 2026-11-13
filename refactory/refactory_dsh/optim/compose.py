"""A parent picker paired with any proposer: where to search, and what to write.

The five parent policies (:mod:`refactory_dsh.optim.policies`) choose which candidate
to breed from; run alone they write random edits. :class:`PickedParent` keeps the
choice and hands the writing to another optimizer -- the model proposer, the random
sampler, or any :class:`~refactory_dsh.optim.base.Optimizer`:

    PickedParent(MapElitesSelector(seed=0), AutoresearchOptimizer(client))

Selection stays inside an optimizer, as issue #32 settled: the pair *is* one
optimizer, and the loop hands it the evaluated population as it hands any other.

One parent per optimizer step. The loop asks :meth:`PickedParent.choose_parent`
before it builds the request, so "The pipeline you may edit", the knobs (a
structural child's own included) and the trajectories are the picked candidate's,
and every proposal of the step -- a retry after a refusal included -- is applied to
it. A model proposer describes one pipeline per request, so picking several parents
per step would mean several requests; across steps the picker still moves between
niches, islands or tournament winners. The picker's reason reaches the request as
part of the round's decision (``ProposalContext.strategy["parent_picker"]``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from .base import Candidate, Optimizer, Proposal, ProposalContext
from .policies import ParentSelector

#: Attributes the loop sets on its optimizer that the proposer is the one to honour.
_FORWARDED = ("trace_path", "max_edits", "allow_code_nodes", "grader_secrets",
              # so the proposer's trace records which round each call belongs to
              "proposal_rounds")


class PickedParent(Optimizer):
    """``picker`` chooses the parent each step; ``proposer`` writes the edits."""

    def __init__(self, picker: ParentSelector, proposer: Optimizer) -> None:
        if not isinstance(picker, ParentSelector):
            raise ValueError(f"{picker!r} is not a parent picker (a ParentSelector)")
        if isinstance(proposer, (ParentSelector, PickedParent)):
            raise ValueError(
                f"the {proposer.name} optimizer picks parents itself; pair a parent "
                "picker with a proposer (random, autoresearch, ...)")
        object.__setattr__(self, "picker", picker)
        object.__setattr__(self, "proposer", proposer)
        super().__init__(k=proposer.k)
        self.name = f"{proposer.name}+{picker.name}"
        self.uses_model = bool(getattr(proposer, "uses_model", False))
        self._picked: tuple[tuple[int, int], str] | None = None

    # The loop sets these on its optimizer; the proposer is what reads them.
    def __setattr__(self, key: str, value: Any) -> None:
        if key in _FORWARDED and "proposer" in self.__dict__:
            setattr(self.proposer, key, value)
        object.__setattr__(self, key, value)

    @property
    def last_diagnosis(self) -> str:  # type: ignore[override]
        return str(getattr(self.proposer, "last_diagnosis", "") or "")

    @last_diagnosis.setter
    def last_diagnosis(self, value: str) -> None:
        if "proposer" in self.__dict__:
            self.proposer.last_diagnosis = value

    @property
    def last_considered(self) -> str:  # type: ignore[override]
        return str(getattr(self.proposer, "last_considered", "") or "")

    @last_considered.setter
    def last_considered(self, value: str) -> None:
        if "proposer" in self.__dict__:
            self.proposer.last_considered = value

    @property
    def last_lessons(self) -> str:
        return str(getattr(self.proposer, "last_lessons", "") or "")

    @last_lessons.setter
    def last_lessons(self, value: str) -> None:
        if "proposer" in self.__dict__ and hasattr(self.proposer, "last_lessons"):
            self.proposer.last_lessons = value

    @property
    def last_reply_problem(self) -> str:
        return str(getattr(self.proposer, "last_reply_problem", "") or "")

    @last_reply_problem.setter
    def last_reply_problem(self, value: str) -> None:
        if "proposer" in self.__dict__ and hasattr(self.proposer, "last_reply_problem"):
            self.proposer.last_reply_problem = value

    @property
    def last_analysis(self) -> dict[str, Any] | None:
        return getattr(self.proposer, "last_analysis", None)

    @last_analysis.setter
    def last_analysis(self, value: dict[str, Any] | None) -> None:
        if "proposer" in self.__dict__ and hasattr(self.proposer, "last_analysis"):
            self.proposer.last_analysis = value

    def bind_model(self, model: Any, *, workdir: Any = None) -> None:
        self.proposer.bind_model(model, workdir=workdir)

    def can_move(self, knob: Any, current: Any) -> bool:
        return self.proposer.can_move(knob, current)

    def choose_parent(
        self, ctx: ProposalContext, parents: Sequence[Candidate]
    ) -> Candidate | None:
        """The picker's choice for this step, remembered for the step's rounds."""
        if not parents:
            return None
        chosen = self.picker.select(ctx, 1, parents)
        if not chosen:
            return None
        self._picked = ((ctx.epoch, ctx.generation), chosen[0].id)
        return chosen[0]

    def propose(
        self, ctx: ProposalContext, n: int, parents: Sequence[Candidate]
    ) -> list[Proposal]:
        if not parents:
            return []
        by_id = {parent.id: parent for parent in parents}
        named = str(ctx.strategy.get("parent") or "")
        if named in by_id:
            # The strategy chose this round's node itself (tree_search's debug, draft or
            # improve): its choice stands, and the request credits it, not the picker.
            parent = by_id[named]
            decision = {**ctx.strategy,
                        "parent_picker": f"not used this round: the strategy chose "
                                         f"candidate {parent.id}, so {self.picker.name} did not"}
            ctx = replace(ctx, parent=parent.id, strategy=decision)
            proposals = self.proposer.propose(ctx, n, [parent]) or []
            return [replace(proposal, parent=parent.id) for proposal in proposals]
        parent = by_id.get(ctx.parent or "")
        if parent is None and self._picked and self._picked[0] == (ctx.epoch, ctx.generation):
            parent = by_id.get(self._picked[1])
        if parent is None:
            # Called without the loop's step (a library caller, a strategy that
            # narrowed the parents): pick now, from what was handed over.
            parent = self.choose_parent(ctx, parents)
        if parent is None:
            return []
        decision = {**ctx.strategy,
                    "parent_picker": f"{self.picker.name} chose candidate {parent.id}: "
                                     f"{self.picker.reason(parent)}"}
        ctx = replace(ctx, parent=parent.id, strategy=decision)
        proposals = self.proposer.propose(ctx, n, [parent]) or []
        return [replace(proposal, parent=parent.id) for proposal in proposals]
