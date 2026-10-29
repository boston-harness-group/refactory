"""Optimizers propose updates to a mode's state.

An optimizer is one method — ``propose`` — plus a loop the base class owns. The
loop is where the evaluation budget is protected: a proposal is expanded and
validated for microseconds before it can cost an inner run, and a proposal that
does not survive is regenerated with the machine-readable reason why. That turns
wasted *budget* into wasted *tokens*, which is the trade worth making, because an
invalid candidate cannot run and therefore teaches the search nothing.

The loop regenerates; it never repairs. An edit that was fixed up before running
would be scored as something the optimizer never proposed, and the gradient would
be a description of the repair rather than of the idea.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import validate_state
from refactory_dsh.nn.knob import Knob
from refactory_dsh.nn.validate import Invariant


#: How the refusal summary is built. A list truncated by count drops whatever came
#: last, and what comes last is often the one reason that explains *why* an edit was
#: wrong rather than which edges it broke.
REASON_LIMIT = 12
REASONS_PER_CODE = 2


def reason_code(reason: str) -> str:
    """The rule a refusal names, from the `CODE [node]: message` form."""
    return reason.split(" ", 1)[0].rstrip(":") or "?"


def summarise_reasons(
    reasons: Sequence[str], *, limit: int = REASON_LIMIT, per_code: int = REASONS_PER_CODE
) -> list[str]:
    """The refusals, grouped by rule, with the ones that explain intent first.

    One removal can produce a dozen mechanical V6 complaints and one invariant
    violation. Ordered by arrival, the invariant is the one that gets cut. Ordered by
    rule -- intent first, then universal rules, then edit refusals -- every rule keeps
    a representative and the count is stated rather than implied.
    """
    grouped: dict[str, list[str]] = {}
    for reason in reasons:
        grouped.setdefault(reason_code(reason), []).append(reason)
    out: list[str] = []
    for code in sorted(grouped, key=_reason_priority):
        entries = grouped[code]
        out.extend(entries[:per_code])
        if len(entries) > per_code:
            out.append(f"({len(entries) - per_code} more {code} refusal(s) of the same kind)")
        if len(out) >= limit:
            break
    return out[:limit]


def _reason_priority(code: str) -> tuple[int, str]:
    """Mode-declared rules first: they are the only ones that state intent."""
    if code.startswith("I:"):
        return (0, code)
    if code.startswith("V"):
        return (1, code)
    return (2, code)


def _tool_policy_escalated(seed_state: dict[str, Any], candidate_state: dict[str, Any]) -> list[str]:
    """Check whether any node in the candidate escalates a seed's no-tools policy.

    Anchored to the actual persisted original seed: a seed node with
    ``tool_policy='none'`` must keep that policy in every candidate, and no
    candidate may add tools to such a node. New node IDs (present only in the
    candidate, not in the seed) are allowed tools -- they represent genuinely
    new architecture. Returns a list of rejection reasons, or an empty list.
    """
    seed_nodes = (seed_state.get("graph") or {}).get("nodes") or {}
    candidate_nodes = (candidate_state.get("graph") or {}).get("nodes") or {}
    reasons: list[str] = []
    for node_id, seed_node in seed_nodes.items():
        seed_policy = seed_node.get("tool_policy")
        if seed_policy != "none":
            continue
        candidate_node = candidate_nodes.get(node_id)
        if candidate_node is None:
            # The seed node was deleted -- that is allowed (the node is gone).
            # But if it was re-added under a different policy, we catch that below.
            continue
        candidate_policy = candidate_node.get("tool_policy")
        if candidate_policy != "none":
            reasons.append(
                f"E-tool-policy-escalation: node {node_id!r} seed declares "
                f"tool_policy='none'; candidate removes it or changes it"
            )
        candidate_tools = candidate_node.get("tools") or []
        if candidate_tools:
            reasons.append(
                f"E-tool-policy-escalation: node {node_id!r} seed declares "
                f"tool_policy='none'; candidate adds tools {candidate_tools!r}"
            )
    # Check for delete/re-add escape: a seed node that was deleted and re-added
    # without the policy. We already handle existing nodes above; here we check
    # that no NEW node with the same ID as a deleted seed no-tools node drops the policy.
    for node_id, seed_node in seed_nodes.items():
        if seed_node.get("tool_policy") != "none":
            continue
        if node_id not in candidate_nodes:
            # Node was deleted -- fine, no escalation
            continue
    return reasons


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


@dataclass(frozen=True)
class Proposal:
    """One proposed update, as typed data."""

    edit: dict[str, Any]
    rationale: str = ""
    prediction: str = ""
    confidence: float = 0.0
    parent: str | None = None

    def summary(self) -> str:
        return f"{self.edit.get('op')}: {self.rationale or self.prediction}".strip()


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

    def codes(self) -> list[str]:
        """The rules that were broken, which is how a refusal histogram is built."""
        return sorted(
            {reason.split(" ", 1)[0].rstrip(":") for reason in self.reasons if reason}
        )


@dataclass
class Candidate:
    """One point in the search: a state, where it came from, and how it scored."""

    id: str
    parent: str | None
    state: dict[str, Any]
    edits: list[dict[str, Any]] = field(default_factory=list)
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

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        parent: str | None = None,
        edits: Iterable[dict[str, Any]] = (),
    ) -> "Candidate":
        return cls(id=candidate_id(state), parent=parent, state=state, edits=list(edits))

    def is_structural(self) -> bool:
        return any(nn_edits.is_macro(edit.get("op", "")) or edit.get("op") in (
            "add_node", "remove_node", "replace_node", "add_edge", "remove_edge",
            "set_node_field",
        ) for edit in self.edits)

    def describe(self) -> dict[str, Any]:
        return {
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
        }


@dataclass
class ProposalContext:
    """The gradient: everything the optimizer is allowed to know.

    It carries the surface it may write to, the catalogue of edits it may use, the
    blocks it may insert, the run's numbers, and the reasons recent proposals were
    refused. Nothing here reaches ``score()`` or ``invariants()``.
    """

    epoch: int = 0
    generation: int = 0
    mode_name: str = ""
    best_score: float | None = None
    mean_score: float | None = None
    #: The run's intent ledger, rendered, read fresh before each generation. A note a
    #: person adds while the search runs reaches the next proposal, so a search is
    #: steerable and not only watchable.
    steering: str = ""
    knobs: list[dict[str, Any]] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    catalogue: list[dict[str, Any]] = field(default_factory=list)
    insertable: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    #: Every evaluated candidate this run, bounded. An optimizer that can only see
    #: the top few cannot ask "have I tried this?", and the base class cannot refuse
    #: a duplicate without knowing what already exists.
    history: list[dict[str, Any]] = field(default_factory=list)
    #: The state ids that already exist, so a proposal recreating one can be refused
    #: instead of silently consuming a batch slot.
    tried: list[str] = field(default_factory=list)
    #: How many parents the loop would have offered before the population was handed
    #: over. An optimizer that has no parent policy of its own selects this many and
    #: behaves as it did; one that has a policy ignores it and chooses from the pool.
    #: Without it, widening the pool silently changes every existing optimizer.
    elite: int = 3
    history_truncated: bool = False
    #: Summaries of previous runs, so a strategy can carry a prior across runs.
    prior_runs: list[dict[str, Any]] = field(default_factory=list)
    #: What the author says is worth doing, which the traces would only show later.
    guidance: str = ""
    preferred_edits: list[str] = field(default_factory=list)
    #: Knobs the operator has frozen for this run, on top of the ones the mode
    #: itself declares frozen. Freezing is a property of the run, not of the mode,
    #: so it cannot be edited into the definition by a candidate.
    frozen: list[str] = field(default_factory=list)
    preset: str = "broad"
    #: What the outer loop decided for *this* iteration, as data.
    #:
    #: An outer loop that only logs its decision has not made one: the proposal
    #: cannot see it, so the log is the only place it exists and the search is the
    #: default loop with extra bookkeeping. A strategy that branches puts the
    #: branch here, and a proposer that can act on it does.
    strategy: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """The request, as text, for a model optimizer."""
        sections = [
            f"# Update request — generation {self.generation}, epoch {self.epoch}",
            "",
            f"Mode: `{self.mode_name}`. Best score so far: {self.best_score}. "
            f"Batch mean: {self.mean_score}.",
            "",
            *([self.steering, ""] if self.steering.strip() else []),
            "## Parameters you may move",
            _table(["name", "value", "kind", "values", "range", "frozen", "reachable",
                    "description"],
                   [[k["name"], k["value"], k["kind"], k.get("values"), k.get("range"),
                     k["frozen"], k.get("reachable", True), k.get("description", "")]
                    for k in self.knobs]),
            "",
            "## The pipeline you may edit",
            "Nodes:",
            _table(["id", "kind", "call", "knobs", "notes"],
                   [[n["id"], n["kind"], n.get("call", ""), n.get("knobs", []),
                     n.get("notes", "")] for n in self.nodes]),
            "Edges:",
            _table(["source", "target", "condition"],
                   [[e["source"], e["target"], e.get("condition")] for e in self.edges]),
            "",
            "## What the author says is worth doing",
            self.guidance or "(the mode's author offered no guidance)",
            (
                "Operations the author suggests trying first: "
                + ", ".join(f"`{name}`" for name in self.preferred_edits)
                if self.preferred_edits
                else ""
            ),
            "",
            "`score_effect: none` on a block means inserting it cannot change the "
            "score. It is available and it may be worth having, but it is not a way "
            "to improve the number.",
            "",
            "## The rules a proposal must satisfy",
            "A proposal that breaks one of these is refused before anything runs, so it "
            "costs nothing but the answer. They are the same rules the validator "
            "enforces, generated from it.",
            _rules_text(),
            "",
            "## The operations you may use",
            _table(["op", "kind", "macro", "fields", "description"],
                   [[c["op"], c["kind"], c["macro"], c["fields"], c["description"]]
                    for c in self.catalogue]),
            "",
            "## Blocks you may insert",
            "Use these node objects verbatim in an `insert_after`/`insert_before`/"
            "`wrap`/`split` edit, changing nothing but their `id`:",
            "```json",
            json.dumps(self.insertable, indent=2, default=str),
            "```",
            "",
            "## Recent results",
            _table(["id", "parent", "score", "structural", "edits"],
                   [[c["id"], c.get("parent"), c.get("score"), c.get("structural"),
                     json.dumps(c.get("edits"), default=str)] for c in self.leaderboard]),
            "",
            "## Evidence from the traces",
            _table(["what", "value"],
                   [[str(k), json.dumps(v, default=str)] for k, v in _flatten(self.traces)]),
        ]
        sections += [
            "",
            "## What this run has already tried",
            (
                f"{len(self.history)} candidate(s), best "
                f"{max((c.get('score') or 0.0) for c in self.history):.6g}"
                if self.history
                else "(nothing evaluated yet)"
            )
            + (
                " — older candidates omitted, so do not assume the list is complete"
                if self.history_truncated
                else ""
            ),
            _table(
                ["id", "parent", "score", "structural", "ok", "edits"],
                [
                    [c.get("id"), c.get("parent"), c.get("score"), c.get("structural"),
                     c.get("exit_ok"), json.dumps(c.get("edits"), default=str)]
                    for c in self.history[-16:]
                ],
            ),
        ]
        if self.prior_runs:
            sections += [
                "",
                "## Previous runs of this mode",
                "What earlier searches already settled, so it does not have to be "
                "rediscovered:",
                _table(
                    ["path", "optimizer", "seed", "best", "bound", "best edits"],
                    [
                        [r.get("path"), r.get("optimizer"), r.get("seed_score"),
                         r.get("best_score"), r.get("bound"),
                         json.dumps(r.get("best_edits"), default=str)]
                        for r in self.prior_runs
                    ],
                ),
            ]
        if self.rejections:
            sections += [
                "",
                "## Why your last proposals were refused",
                "A proposal that breaks one of these is not evaluated at all. It costs "
                "nothing but the answer, so fix the stated reason rather than guessing.",
                _table(["reasons"], [[r] for r in self.rejections[-12:]]),
            ]
        sections += ["", "## Answer", ANSWER_INSTRUCTIONS]
        return "\n".join(sections)


ANSWER_INSTRUCTIONS = """\
Write one JSON object and nothing else:

{"diagnosis": "<what the evidence shows is limiting the score>",
 "proposals": [{"edit": {"op": "<operation>", ...},
                "rationale": "<why, citing the evidence>",
                "prediction": "<expected effect on the score>",
                "confidence": 0.0}]}

Rules:
- A parameter move is {"op": "set_knob", "name": "<parameter>", "value": <value>}.
- A structural move is one of the operations above, using the fields it lists and
  the block objects verbatim.
- Few well-argued changes, not many guesses. Prefer an edit whose effect you can
  state, and vary what you touch.
- Your prediction is recorded and later checked against the outcome, so say what
  you expect and how you could be wrong.
- If the evidence justifies no change, return an empty proposal list and say why.
"""


def _rules_text() -> str:
    from refactory_dsh.nn.validate import rules_text

    return rules_text()


def _flatten(rows: Sequence[dict[str, Any]]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for row in rows:
        for key, value in row.items():
            out.append((key, value))
    return out


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "(none)"
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = [
        "| " + " | ".join(_cell(value) for value in row) + " |" for row in rows
    ]
    return "\n".join([head, rule, *body])


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text.replace("|", "\\|").replace("\n", " ")


class Optimizer:
    """Base class: one ``propose``, and the validating loop around it."""

    name: str = "optimizer"

    def bind_model(self, model: Any, *, workdir: Any = None) -> None:
        """Model-using optimizers override this to inherit the run's resources."""

    def __init__(self, *, k: int = 3) -> None:
        #: How many times the loop may ask again after a refusal.
        self.k = max(1, int(k))
        self.rejections: list[Rejection] = []
        self.last_diagnosis: str = ""
        #: Every round that failed to propose, with the reason. A failure here is
        #: not a crash -- a run should survive a model that is down -- but it is
        #: also not a search result, and the difference between "the model did not
        #: answer" and "the model answered and nothing improved" is the difference
        #: between a claim about the machine and a claim about the mode.
        self.proposal_failures: list[str] = []
        #: Rounds that were asked for a proposal, however they ended. Paired with
        #: ``proposal_failures`` to tell a wholly-failed run from a partly-failed one.
        self.proposal_rounds: int = 0

    # ── what this optimizer can reach ──────────────────────────────────────
    def can_move(self, knob: Knob, current: Any) -> bool:
        """Whether this optimizer can produce a different value for ``knob``.

        A run reports the knobs its optimizer cannot reach, because the failure to
        avoid is a search that looks like it covers the declared surface and does
        not. Override to declare a narrower capability than the kinds allow.
        """
        if knob.kind == "float":
            low, high = knob.range()
            return (high if high is not None else 0.0) > (low if low is not None else 0.0)
        return bool(knob.alternatives(current))

    # ── the part a subclass writes ─────────────────────────────────────────
    def propose(
        self, ctx: ProposalContext, n: int, parents: Sequence["Candidate"]
    ) -> list[Proposal]:
        """One round of proposals. Unvalidated and possibly unusable.

        ``parents`` are the candidates the edits are made against, so a proposal
        can name the one it belongs to and the caller can apply it to that state.
        """
        raise NotImplementedError

    # ── the loop ───────────────────────────────────────────────────────────
    def propose_valid(
        self,
        ctx: ProposalContext,
        n: int,
        *,
        parents: Sequence[Candidate],
        invariants: tuple[Invariant, ...] = (),
        preset: str = "broad",
        resolve_calls: bool = True,
        seen: Sequence[str] | None = None,
    ) -> tuple[list[Candidate], list[Rejection]]:
        """Propose until ``n`` valid candidates exist, or the attempts run out.

        Returns the offspring and every refusal. A refusal is never silent: it is
        recorded here, fed back into the next attempt, and reported at the end.
        """
        self.rejections = []
        accepted: list[Candidate] = []
        feedback: list[str] = []
        if not parents:
            return accepted, self.rejections

        for attempt in range(1, self.k + 1):
            if len(accepted) >= n:
                break
            round_ctx = replace(
                ctx,
                rejections=[*ctx.rejections, *summarise_reasons(feedback)],
                preset=preset,
            )
            try:
                self.proposal_rounds += 1
                proposals = self.propose(round_ctx, n - len(accepted), parents) or []
            except Exception as exc:  # a failed optimizer round is not a crash
                from refactory_dsh.budgets import BudgetExhausted

                if isinstance(exc, BudgetExhausted):
                    self.proposal_failures.append(f"{type(exc).__name__}: {exc}")
                    raise
                detail = f"{type(exc).__name__}: {exc}"
                self.proposal_failures.append(detail)
                feedback = [f"{self.name} could not propose: {detail}"]
                continue
            if not proposals:
                if feedback:
                    continue
                break

            progressed = False
            for proposal in proposals:
                if len(accepted) >= n:
                    break
                parent = self._pick_parent(proposal, parents)
                outcome = self.attempt(
                    proposal, parent, invariants, resolve_calls, ctx.frozen, seen
                )
                if isinstance(outcome, Candidate):
                    accepted.append(outcome)
                    progressed = True
                else:
                    rejection = Rejection(
                        attempt=attempt,
                        edit=proposal.edit,
                        reasons=outcome,
                        rationale=proposal.rationale,
                    )
                    self.rejections.append(rejection)
                    feedback.extend(outcome)
            if not progressed and not feedback:
                break
        return accepted, self.rejections

    def _pick_parent(
        self, proposal: Proposal, parents: Sequence[Candidate]
    ) -> Candidate:
        if proposal.parent:
            for parent in parents:
                if parent.id == proposal.parent:
                    return parent
        return parents[0]

    def attempt(
        self,
        proposal: Proposal,
        parent: Candidate,
        invariants: tuple[Invariant, ...],
        resolve_calls: bool,
        frozen: Sequence[str] | None = None,
        seen: Sequence[str] | None = None,
        seed_state: dict[str, Any] | None = None,
    ) -> Candidate | list[str]:
        """Apply and validate one proposal: a candidate, or the reasons it failed.

        Public because seeding a population and proposing an update have to be the
        same test: a candidate that could not be proposed could not be seeded
        either, and two code paths would eventually disagree about that.
        """
        frozen = set(frozen or ())
        if proposal.edit.get("op") == "set_knob" and proposal.edit.get("name") in frozen:
            return [
                f"E-frozen: knob {proposal.edit.get('name')!r} is frozen for this run"
            ]
        # An operation that is not in the catalogue is not a failed precondition: the
        # two are told apart because they point at different fixes, and calling every
        # refusal a precondition failure hides which one the catalogue is causing.
        op = proposal.edit.get("op")
        if op not in nn_edits.names():
            return [f"E-op: unknown edit {op!r}; known: {nn_edits.names()}"]
        reason = nn_edits.precheck(proposal.edit, parent.state)
        if reason:
            return [f"E-precondition: {reason}"]
        try:
            state = nn_edits.apply(proposal.edit, parent.state)
        except nn_edits.EditError as exc:
            return [f"{exc.code}: {exc}"]
        except Exception as exc:  # an operator bug is reported, not swallowed
            return [f"E-internal: {type(exc).__name__}: {exc}"]

        # ── tool-policy escalation check (anchored to the original seed) ──
        if seed_state is not None:
            escalation = _tool_policy_escalated(seed_state, state)
            if escalation:
                return escalation

        fixed = frozen | {name for name, value in parent.state.get("knobs", {}).items()
                          if (value.get("spec") or {}).get("frozen")}
        for name in fixed:
            if state.get("knobs", {}).get(name) != parent.state.get("knobs", {}).get(name):
                return [f"E-frozen: knob {name!r} must retain its value and declaration"]

        if candidate_id(state) == parent.id:
            # The edit is legal and changes nothing: setting a knob to the value it
            # already holds, or re-adding a node that is already there. Accepting it
            # would spend a batch slot re-evaluating a point already scored, and the
            # search would look productive while making no progress.
            return [
                f"E-noop: {proposal.edit.get('op')} leaves the state identical to its "
                "parent, so it would re-evaluate a candidate already scored"
            ]

        if candidate_id(state) in set(seen or ()):
            # Without this the proposal passes every gate and is then dropped
            # silently by the loop, which loses the batch slot and makes the run
            # report a batch it did not actually try.
            return [
                "E-duplicate: this state already exists in the search, so evaluating it "
                "again would spend a batch slot on a point already scored"
            ]

        issues = validate_state(state, invariants=invariants, resolve_calls=resolve_calls)
        if issues:
            return [str(issue) for issue in issues]
        return Candidate.from_state(
            state,
            parent=parent.id,
            edits=[*parent.edits, proposal.edit],
        )
