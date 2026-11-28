"""Core data types: Proposal, Candidate, Rejection, RunStopped, candidate_id."""


from __future__ import annotations

import hashlib
import json
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from typing import Any, Iterable

from refactory_dsh.nn import edits as nn_edits

#: Edit fields that take a whole node object (``into`` takes a list of two).
NODE_OBJECT_FIELDS = ("node", "into", "gate", "branches")


def candidate_id(state: dict[str, Any]) -> str:
    """A stable id for a state, so two candidates that are the same state agree.

    Derived from the canonical state rather than from the edit that produced it:
    the same architecture reached two ways is the same candidate, and evaluating
    it twice is exactly the waste this loop exists to avoid.
    """
    payload = json.dumps(
        {"graph": state.get("graph"), "knobs": state.get("knobs")}, sort_keys=True, default=str
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


class RunStopped(CancelledError):
    """A person asked the run to stop (``refactory stop``) while an optimizer call waited
    to try again (`random_optimizer.AutoresearchOptimizer._wait`). The search ends there
    as a stop at an epoch boundary ends it (``OuterLoop.run``): the stop is the bound,
    the report is written, and the open epoch's step is made again on resume."""

    reason = "a person asked the run to stop"


#: How many changes one proposal may bundle: SkillOpt's edit budget, so a coordinated
#: fix across a pipeline's steps can be one candidate rather than a chain of single
#: edits the gate judges one by one. Counted by logical change (`optim.edit_count`):
#: a structural edit and the edits that only connect it are one. A constant, not a run
#: setting.
MAX_EDITS_PER_PROPOSAL = 4


#: The most edits one proposal may carry however they count (three times the cap): a
#: guard against runaway bundles, not the cap itself.
MAX_RAW_EDITS_PER_PROPOSAL = 12


#: How the cap counts, as a refusal and the bundling instructions say it.
COUNTING_RULE = ("The limit counts changes, not edits: a structural edit plus its own "
                 "prompt and settings, edges to it, and set_knob edits of steps that read "
                 "its output or name it count as one change (e.g. a new branch and the "
                 "tuning of the step that reads it). Every other edit counts as one more. "
                 f"At most {MAX_RAW_EDITS_PER_PROPOSAL} edits in all.")


@dataclass(frozen=True)
class Proposal:
    """One proposed update, as typed data.

    Usually one ``edit``. A *compound* proposal carries an ordered list in ``edits``,
    applied together to the parent as one candidate (each edit to the state the one
    before it left) and judged once. A one-item ``edits`` is the same as ``edit``.
    For a compound proposal ``edit`` is ``{"edits": [...]}``, the form refusals and
    records show; ``steps`` is the list to apply, whichever form was given.
    """

    edit: Any = field(default_factory=dict)
    rationale: str = ""
    prediction: str = ""
    confidence: float = 0.0
    parent: str | None = None
    edits: tuple[Any, ...] = ()
    #: For a structural proposal (`REASONING_FIELDS`): the failure it targets, citing the
    #: per-step evidence by step and item id ...
    addresses: str = ""
    #: ... and how it changes the information flow or the way the work is done.
    flow_change: str = ""

    def __post_init__(self) -> None:
        steps = tuple(self.edits or ())
        if (not steps and isinstance(self.edit, dict) and "op" not in self.edit
                and isinstance(self.edit.get("edits"), list)):
            # The compound form itself, as a refusal or a record shows it.
            steps = tuple(self.edit["edits"])
        if not steps:
            return
        if len(steps) == 1:
            object.__setattr__(self, "edit", steps[0])
            object.__setattr__(self, "edits", ())
            return
        object.__setattr__(self, "edits", steps)
        object.__setattr__(self, "edit", {"edits": list(steps)})

    @property
    def steps(self) -> list[Any]:
        """The edits to apply, in order: ``edits`` for a compound proposal, else ``[edit]``."""
        return list(self.edits) if self.edits else [self.edit]

    def missing_reasoning(self) -> list[str]:
        """For a structural proposal, the reasoning fields it left empty
        (`REASONING_FIELDS`); none for one that changes settings only. Not a refusal: the
        trace marks it and a replay counts it."""
        if not any(_structural_edit(step) for step in self.steps):
            return []
        return [name for name in REASONING_FIELDS if not str(getattr(self, name) or "").strip()]

    def summary(self) -> str:
        ops = "+".join(str(step.get("op") if isinstance(step, dict) else step)
                       for step in self.steps)
        return f"{ops}: {self.rationale or self.prediction}".strip()


#: What a structural proposal says beside its edit lines, so the design gets the thought
#: and the edit stays one line: the failure it targets (citing per-step evidence by step
#: and item id), how it changes the information flow, and a prediction that can be checked.
REASONING_FIELDS = ("addresses", "flow_change", "prediction")


@dataclass(frozen=True)
class Rejection:
    """A proposal that did not become a candidate, and why.

    The reasons are the feedback the loop hands back, and their distribution is
    the gradient on the edit catalogue itself: one reason dominating means a rule
    the optimizer cannot see.
    """

    attempt: int
    edit: dict[str, Any]
    reasons: list[str]
    rationale: str = ""
    #: The candidate a slim repair call made of this refused proposal (`optim.repair`),
    #: or empty when it was not repaired.
    repaired: str = ""
    #: What that repair dropped or changed of the proposal's text (`repair.lost_content`),
    #: or empty: said in later requests (`repair_loss_lines`).
    dropped: str = ""

    def codes(self) -> list[str]:
        """The rules that were broken, which is how a refusal histogram is built."""
        return sorted(
            {reason.split(" ", 1)[0].rstrip(":") for reason in self.reasons if reason}
        )


def _structural_edit(edit: Any) -> bool:
    """Whether one edit changes the graph's structure: a macro or a node or edge edit."""
    return isinstance(edit, dict) and (nn_edits.is_macro(edit.get("op", "")) or edit.get("op") in (
        "add_node", "remove_node", "add_edge", "remove_edge"))


@dataclass
class Candidate:
    """One point in the search: a state, where it came from, and how it scored."""

    id: str
    parent: str | None
    state: dict[str, Any]
    edits: list[dict[str, Any]] = field(default_factory=list)
    #: Per-edit reports, parallel to *edits*.  Only a ``set_knob`` with a
    #: ``patch`` produces one today; for other edits the corresponding slot is ``None``.  Stored on the
    #: candidate (not on the edit dict) so it is journaled, survives stop+resume,
    #: and does not leak into optimizer prompts.
    edit_reports: list[dict[str, Any] | None] = field(default_factory=list)
    #: How many of the last ``edits`` the proposal that made this candidate added: 1
    #: for a single edit, the list's length for a compound proposal (``Proposal.edits``),
    #: 0 for the seed. ``own_edits()`` is that tail.
    step_edits: int = 1
    #: The score the search compares: the evaluator's number minus the run's size
    #: cost. Set by the loop, which is the only place the cost is known.
    score: float | None = None
    #: What the evaluator scored, before the size cost. Kept so a reader can tell a
    #: candidate that won on merit from one that won on size.
    raw_score: float | None = None
    mean_score: float | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    exit_ok: bool = True
    error: str | None = None
    duration_s: float = 0.0
    evaluated: bool = False
    measurements: dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    #: The mechanical fixes the attempt path made to the proposal that made this
    #: candidate (`optim.repair`), each said in a line; its ``edits`` are the fixed ones.
    repairs: list[str] = field(default_factory=list)
    #: What the attempt path could not check, and why (a code dry run skipped).
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        parent: str | None = None,
        edits: Iterable[dict[str, Any]] = (),
        edit_reports: Iterable[dict[str, Any] | None] = (),
        step_edits: int = 1,
    ) -> "Candidate":
        return cls(
            id=candidate_id(state), parent=parent, state=state,
            edits=list(edits), edit_reports=list(edit_reports), step_edits=int(step_edits),
        )

    def own_edits(self) -> list[dict[str, Any]]:
        """The edits the proposal that made this candidate applied to its parent, in
        order: one for a single edit, all of a compound proposal's."""
        count = min(len(self.edits), max(0, int(self.step_edits)))
        return list(self.edits[len(self.edits) - count:]) if count else []

    def is_structural(self) -> bool:
        """Whether the edits that made this candidate from its parent (``own_edits``)
        change the graph's structure. A prompt-only child of a structural parent is not
        structural: the label says what this step did, not what its lineage did."""
        return any(_structural_edit(edit) for edit in self.own_edits())

    def lineage_is_structural(self) -> bool:
        """Whether any edit from the seed to this candidate changed the structure."""
        return any(_structural_edit(edit) for edit in self.edits)

    def describe(self, *, include_reports: bool = False) -> dict[str, Any]:
        """Summarise for prompt rendering.

        Reports are omitted by default so they never leak into optimizer
        prompts.  Pass ``include_reports=True`` when an optimizer explicitly
        asks for the per-edit outcomes.
        """
        d: dict[str, Any] = {
            "id": self.id,
            "parent": self.parent,
            "score": self.score,
            "raw_score": self.raw_score,
            "edits": self.edits,
            "structural": self.is_structural(),
            "exit_ok": self.exit_ok,
            "error": self.error,
            "duration_s": self.duration_s,
            "measurements": self.measurements,
            **({"repairs": list(self.repairs)} if self.repairs else {}),
            **({"notes": list(self.notes)} if self.notes else {}),
        }
        if include_reports:
            d["edit_reports"] = self.edit_reports
        return d
