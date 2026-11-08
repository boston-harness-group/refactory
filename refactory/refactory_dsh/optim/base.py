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
from typing import Any, Iterable, Mapping, Sequence

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import validate_state
from refactory_dsh.nn.knob import Knob
from refactory_dsh.nn.validate import Invariant


#: How the refusal summary is built. A list truncated by count drops whatever came
#: last, and what comes last is often the one reason that explains *why* an edit was
#: wrong rather than which edges it broke.
REASON_LIMIT = 12
REASONS_PER_CODE = 2

#: How the edited candidate's train trajectories reach the optimizer
#: (``ProposalContext``): every train item, every proposal, at full detail -- its input,
#: every step's output in the order the pass ran them, the expected answer and the
#: prediction -- so a proposal can name the failure pattern and see what already works.
#: One call's evidence budget: when every item fits it in full, they all go to the
#: proposal call inline; when not, readers go through them in groups that each fit it,
#: and the proposal call gets their findings, a table of every item and some in full.
TRAJECTORY_BUDGET = 120000
#: How many reader calls run at once.
READER_CONCURRENCY = 4
#: A reader's findings as the proposal call shows them: at most this many patterns
#: per reader, each pattern's text and quotes clipped, at most this many quotes and
#: item ids per pattern, and the whole findings block within a share of the budget.
READER_PATTERNS = 12
READER_PATTERN_CHARS = 400
READER_QUOTES = 3
READER_QUOTE_CHARS = 240
READER_ITEM_IDS = 40
READER_FINDINGS_SHARE = 0.4
#: The metadata keys an item's input is read from, first found; else the first
#: step's prompt stands in for it.
TRAJECTORY_INPUT_KEYS = ("question", "problem", "task_description", "instruction", "input", "prompt")

#: How the edited graph's nodes are shown in full (``ProposalContext.node_states``):
#: each long text field keeps its head and tail within this many characters, and the
#: whole section (each node's state plus the generated node schemas) fits this cap.
NODE_SPEC_TEXT_CHARS = 1200
NODE_SPEC_TEXT_FLOOR = 200
NODE_SPEC_SECTION_CHARS = 30000
#: Edit fields that take a whole node object (``into`` takes a list of two).
NODE_OBJECT_FIELDS = ("node", "into", "gate", "branches")
#: The fields among them that hold a list of node objects.
NODE_LIST_FIELDS = ("into", "branches")


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
    return reasons


def _lowering_refusal(parent_state: dict[str, Any], state: dict[str, Any]) -> str | None:
    """``E-lowering`` when the edit makes a state the spine cannot be given.

    The validator states the rules both runtimes share; lowering is the spine's own last
    word (`lowering.to_workflow`). A candidate whose parent lowers and which does not
    would score on the walker and fail on the spine, so it is refused here, with
    lowering's reason. A parent that does not lower itself (a walker-only mode) is not
    held to it.
    """
    from refactory_dsh.lowering import to_workflow

    try:
        to_workflow(state)
    except Exception as exc:  # noqa: BLE001 - any failure to lower is the refusal
        try:
            to_workflow(parent_state)
        except Exception:  # noqa: BLE001 - the parent never lowered: not this edit's doing
            return None
        return f"E-lowering: the edited state does not lower to the spine's workflow: {exc}"
    return None


def _new_node_tools(seed_state: dict[str, Any], candidate_state: dict[str, Any],
                    allowed: bool) -> list[str]:
    """``E-new-node-tools`` for each agent node the search added that would run with tools.

    A node an edit adds gets ``tool_policy: none`` by default (`nn.edits`): with tools
    it could read anything in the run directory or beyond it. Asking for tools
    (``tool_policy: null`` or a ``tools`` list) needs the run to allow it
    (``LoopConfig.allow_new_node_tools``). A node whose id the seed has is held to the
    seed's own policy instead (`_tool_policy_escalated`).
    """
    if allowed:
        return []
    seed_nodes = (seed_state.get("graph") or {}).get("nodes") or {}
    reasons = []
    for node_id, node in sorted(((candidate_state.get("graph") or {}).get("nodes") or {}).items()):
        if node_id in seed_nodes or node.get("kind") != "agent":
            continue
        if node.get("tool_policy") != "none" or node.get("tools"):
            reasons.append(
                f"E-new-node-tools: new agent node {node_id!r} would run with tools; a node "
                "an edit adds runs without tools (omit tool_policy, or set it to 'none') "
                "unless the run allows tools for new nodes (allow_new_node_tools)")
    return reasons


def _code_signatures(state: dict[str, Any]) -> dict[str, tuple[str, str, str]]:
    """Each code node of a state (`refactory_dsh.snippet`) by id: its kind, the callable
    it runs and the code it runs (its bound knob's value, as both runtimes read it)."""
    from refactory_dsh import snippet
    from refactory_dsh.nn.graph import Graph

    try:
        graph = Graph.from_state(state.get("graph") or {})
    except Exception:  # noqa: BLE001 - an unreadable graph is the validator's to report
        return {}
    values = {name: (entry or {}).get("value") for name, entry in (state.get("knobs") or {}).items()}
    out = {}
    for node_id, node in graph.nodes.items():
        if snippet.is_code_node(node):
            try:
                code = str(node.bound("code", values) or "")
            except Exception:  # noqa: BLE001 - a bad binding is V14's to report
                code = node.code
            out[node_id] = (node.kind, node.call or node.evaluator, code)
    return out


def _code_node_refusals(seed_state: dict[str, Any], candidate_state: dict[str, Any],
                        allowed: bool) -> list[str]:
    """``E-code-nodes`` for each code node of the candidate (`refactory_dsh.snippet`) that
    the seed does not have as the same code node -- same id, kind, callable and code --
    when the run does not allow code nodes (``LoopConfig.allow_code_nodes``) or this
    machine cannot sandbox them (`snippet.available`). Compared by content, not by id: a
    proposal that removes a node and adds a code node under its id, or rewrites a seed
    code node's code, is a new code node. The seed's own code nodes, unchanged, are the
    author's and are not held to it."""
    from refactory_dsh import snippet

    seed = _code_signatures(seed_state)
    added = sorted(node_id for node_id, signature in _code_signatures(candidate_state).items()
                   if seed.get(node_id) != signature)
    if not added:
        return []
    if not allowed:
        why = "this run does not allow code nodes (allow_code_nodes is off)"
    else:
        ok, reason = snippet.available()
        if ok:
            return []
        why = f"this machine cannot sandbox code nodes ({reason})"
    return [f"E-code-nodes: code node {node_id!r} is new or changed and is refused: {why}"
            for node_id in added]


def _secret_refusals(candidate_state: dict[str, Any], secrets: Any) -> list[str]:
    """``E-secret`` for each code node whose code holds a value only the grader may see
    (a train item's grader-only field, e.g. its answer): code that carried the answers
    would score them without solving anything. The value itself is never echoed."""
    from refactory_dsh import snippet
    from refactory_dsh.nn.graph import Graph

    try:
        graph = Graph.from_state(candidate_state.get("graph") or {})
    except Exception:  # noqa: BLE001 - an unreadable graph is the validator's to report
        return []
    values = {name: (entry or {}).get("value") for name, entry in (candidate_state.get("knobs") or {}).items()}
    reasons = []
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if not snippet.is_code_node(node) and not node.code:
            continue
        if callable(secrets):
            # Loaded only once a candidate carries code: reading a Task's items costs.
            secrets = secrets()
        if not secrets:
            return []
        texts = [node.code, str(values.get(node.bound_knob("code")) or "")]
        if any(snippet.secret_in(text, secrets) for text in texts if text):
            reasons.append(
                f"E-secret: the code of node {node_id!r} contains a value only the grader may "
                "see (a train item's grader-only field, such as its answer); code may compute "
                "from a node's inputs, never carry the answers")
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


#: How many edits one proposal may bundle by default (``LoopConfig.max_edits_per_proposal``):
#: SkillOpt's edit budget, so a coordinated fix across a pipeline's steps can be one
#: candidate rather than a chain of single edits the gate judges one by one.
DEFAULT_MAX_EDITS = 4


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

    def summary(self) -> str:
        ops = "+".join(str(step.get("op") if isinstance(step, dict) else step)
                       for step in self.steps)
        return f"{ops}: {self.rationale or self.prediction}".strip()


def _at(reason: str, index: int, steps: Sequence[Any]) -> str:
    """``reason``, naming which edit of a compound proposal it is about.

    A single edit's reason is unchanged. The rule's code stays the first word, so a
    refusal histogram counts a compound proposal's refusals with everyone else's.
    """
    if len(steps) <= 1:
        return reason
    step = steps[index]
    op = step.get("op") if isinstance(step, dict) else type(step).__name__
    return f"{reason} [edit {index + 1} of {len(steps)}: {op}]"


def _grader_side_knobs(state: dict[str, Any], nodes: Sequence[str]) -> dict[str, str]:
    """The knobs the given (protected or grading) nodes read or bind, each with the
    first node that does."""
    out: dict[str, str] = {}
    graph_nodes = (state.get("graph") or {}).get("nodes") or {}
    for node_id in sorted(set(nodes)):
        node = graph_nodes.get(node_id) or {}
        names = list(node.get("knobs") or ())
        names += [pair[1] for pair in node.get("field_bindings") or () if len(pair) == 2]
        for name in names:
            out.setdefault(str(name), node_id)
    return out


def _dead_edit(steps: Sequence[dict[str, Any]], start: dict[str, Any], result: dict[str, Any],
               protected: Sequence[str] = ()) -> int | None:
    """The index of a compound proposal's first edit the result does not depend on: the
    other edits, applied without it, reach the same state (a later edit overwrote it or
    undid it). ``None`` for a single edit, or when every edit counts."""
    if len(steps) < 2:
        return None
    target = candidate_id(result)
    for index in range(len(steps)):
        state = start
        try:
            for other, edit in enumerate(steps):
                if other != index:
                    state = nn_edits.apply(edit, state, protected=protected)
        except Exception:  # noqa: BLE001 - the others need this one: it counts
            continue
        if candidate_id(state) == target:
            return index
    return None


def proposal_refusal(proposal: Proposal, *, permitted: Sequence[str] | None = None,
                     max_edits: int | None = None) -> list[str]:
    """Why a proposal is refused before any edit is applied; empty when it is not.

    Too many edits (``E-too-many-edits``), an edit that is not an object (``E-op``),
    an op outside the catalogue, and, given ``permitted``, an op outside this run's
    permitted edits (``E-policy``). Each refusal of a compound proposal names the
    edit (its index, from 1) it is about.
    """
    steps = proposal.steps
    if max_edits is not None and len(steps) > max(1, int(max_edits)):
        return [f"E-too-many-edits: the proposal bundles {len(steps)} edits and this run "
                f"allows at most {max(1, int(max_edits))} per proposal "
                "(max_edits_per_proposal); split it, or keep the edits the fix needs together"]
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return [_at(f"E-op: an edit is a JSON object naming its op, not "
                        f"{type(step).__name__} {step!r}"[:300], index, steps)]
        refused = nn_edits.unknown(step.get("op"))
        if refused is not None:
            return [_at(refused, index, steps)]
        if permitted is not None and step.get("op") not in permitted:
            return [_at("E-policy: operation is outside this run's permitted edits",
                        index, steps)]
    return []


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
        return any(nn_edits.is_macro(edit.get("op", "")) or edit.get("op") in (
            "add_node", "remove_node", "add_edge", "remove_edge",
        ) for edit in self.edits)

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
        }
        if include_reports:
            d["edit_reports"] = self.edit_reports
        return d


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
    #: The mode's protected nodes (``Mode.protected_nodes``): no edit may remove,
    #: rewire or replace them, and one that does is refused with ``E-protected``.
    protected: list[str] = field(default_factory=list)
    preset: str = "broad"
    #: What the outer loop decided for *this* iteration, as data.
    #:
    #: An outer loop that only logs its decision has not made one: the proposal
    #: cannot see it, so the log is the only place it exists and the search is the
    #: default loop with extra bookkeeping. A strategy that branches puts the
    #: branch here, and a proposer that can act on it does.
    strategy: dict[str, Any] = field(default_factory=dict)
    #: Per-item records for the parent/current candidate the optimizer is editing (F6).
    item_records: list[dict[str, Any]] = field(default_factory=list)
    #: Alias for the parent candidate's train records.
    parent_item_records: list[dict[str, Any]] = field(default_factory=list)
    #: Why the pipeline shown is not the parent's, when it is not (empty when it is).
    graph_note: str = ""
    #: The edited graph's nodes in node-state form, as ``add_node`` and
    #: ``insert_after`` take a node, so a model can copy one and change it.
    node_states: list[dict[str, Any]] = field(default_factory=list)
    node_spec_text_chars: int = NODE_SPEC_TEXT_CHARS
    node_spec_section_chars: int = NODE_SPEC_SECTION_CHARS
    #: One call's evidence budget (``TRAJECTORY_BUDGET``).
    evidence_budget: int = TRAJECTORY_BUDGET
    #: The latest children of the edited candidate with their parents' train records
    #: (``Loop._edit_pairs``), for "What the last edit changed".
    edit_pairs: list[dict[str, Any]] = field(default_factory=list)
    #: The optimizer's earlier reasoning beside outcomes (``Loop._reasoning_entries``).
    reasoning: list[dict[str, Any]] = field(default_factory=list)
    #: What readers found in the items, group by group, when they did not fit one
    #: call (``AutoresearchOptimizer``); ``None`` when no reader went through them.
    reader_findings: list[dict[str, Any]] | None = None
    #: The groups readers could not read, with why.
    reader_failures: list[dict[str, Any]] = field(default_factory=list)
    #: The proposal attempt within this generation (``Loop.propose`` and
    #: ``Optimizer.propose_valid`` count from 1).
    attempt: int = 0
    #: How many edits one proposal may bundle (``Proposal.edits``;
    #: ``LoopConfig.max_edits_per_proposal``). 1 asks for single edits only, and the
    #: request does not mention bundling.
    max_edits: int = 1
    #: The run's size cost per node (``LoopConfig.parsimony``), stated in the answer
    #: rules with what it means in score units; ``None`` says nothing about it.
    parsimony: float | None = None
    #: Whether the score is the mean of the items' scores (``Loop`` checks every train
    #: evaluation); only then does the size cost line compare a node with one item.
    score_is_item_mean: bool = False
    #: How many proposals this request asks for (``AutoresearchOptimizer`` sets it to
    #: its ``n``); 0 when not said. With 1, only one candidate is evaluated, so the
    #: bundling advice asks for one proposal covering the whole failure path instead
    #: of separate proposals that would never run.
    proposals: int = 0
    #: How many validation items the gate's score is the mean of (``Loop``: the val
    #: part's size, under ``val_gate`` when the score is the items' mean); 0 when not
    #: known. The gate note then says what one validation item is worth.
    val_items: int = 0
    #: The seed's knob values (``Loop``: the population's root candidate), so the
    #: history can show a ``set_knob`` as a diff against the value it replaced; empty
    #: when not known (the first change to a text knob then shows its length and head).
    seed_knob_values: dict[str, Any] = field(default_factory=dict)
    #: Whether this run's edits may add code nodes (``LoopConfig.allow_code_nodes`` and a
    #: machine that can sandbox them, `refactory_dsh.snippet.available`); only then does
    #: the request describe them.
    code_nodes: bool = False
    #: The candidate whose pipeline "The pipeline you may edit" describes, and so the
    #: one a proposal that names no parent is applied to (``Loop._edited_id``); ``None``
    #: when not known.
    parent: str | None = None
    #: The outer loop's name (``Loop.name``), said beside its decision; empty for none.
    strategy_name: str = ""
    #: The edit levels this request asks for, one proposal each (``EDIT_LEVELS``;
    #: ``LoopConfig.portfolio``, or one level a strategy decided); empty asks for none
    #: in particular.
    edit_levels: list[str] = field(default_factory=list)
    #: The validation gate's score of each candidate it judged, by id, under the
    #: default loop's ``val_gate``; empty otherwise. For a parent policy to rank by
    #: what the gate ranks by (``HillClimbSelector``). Never rendered: the model's
    #: request shows gate decisions, not val scores.
    val_scores: dict[str, float] = field(default_factory=dict)

    def render(self) -> str:
        """The request, as text, for a model optimizer."""
        sections = [
            f"# Update request — generation {self.generation}, epoch {self.epoch}",
            "",
            f"Mode: `{self.mode_name}`. Best score so far: {self.best_score}. "
            f"Batch mean: {self.mean_score}.",
            "",
            *([self.steering, ""] if self.steering.strip() else []),
            *_strategy_section(self.strategy, self.strategy_name),
            "## Parameters you may move",
            _table(["name", "value", "kind", "values", "range", "frozen", "reachable",
                    "description"],
                   [[k["name"], k["value"], k["kind"], k.get("values"), k.get("range"),
                     k["frozen"], k.get("reachable", True), k.get("description", "")]
                    for k in self.knobs]),
            "",
            "## The pipeline you may edit",
            *([f"Note: {self.graph_note}"] if self.graph_note else []),
            *([f"Protected nodes: {', '.join(f'`{n}`' for n in self.protected)}. An edit may "
               "not remove them, change their declarations or out-edges, move a knob they "
               "read, or let a run finish without them; a new node may go in front of one."]
              if self.protected else []),
            "Nodes:",
            _table(["id", "kind", "call", "knobs", "notes"],
                   [[n["id"], n["kind"], n.get("call", ""), n.get("knobs", []),
                     n.get("notes", "")] for n in self.nodes]),
            "Edges:",
            _table(["source", "target", "condition"],
                   [[e["source"], e["target"], e.get("condition")] for e in self.edges]),
            "",
            *_node_specs(self.node_states, text_chars=self.node_spec_text_chars,
                         section_chars=self.node_spec_section_chars,
                         knob_values={k.get("name"): k.get("value") for k in self.knobs}),
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
                   [[c["op"], c["kind"], c["macro"], _edit_fields(c["fields"]), c["description"]]
                    for c in self.catalogue]),
            "",
            *_blocks_section(self.insertable, self.catalogue),
            *composition_section(self.catalogue, self.node_states, self.edges,
                                 protected=self.protected, code_nodes=self.code_nodes),
            "## Recent results",
            *_gate_note(self.leaderboard, self.history, val_items=self.val_items),
            _table(["id", "parent", "score", "structural", "edits", *_gate_column(self.leaderboard)],
                   [[c["id"], c.get("parent"), c.get("score"), c.get("structural"),
                     compact_edits(c.get("edits"), self.seed_knob_values),
                     *([c.get("gate", "")] if _gate_column(self.leaderboard) else [])]
                    for c in self.leaderboard]),
            "",
            "## Evidence from the traces",
            _table(["what", "value"],
                   [[str(k), json.dumps(v, default=str)] for k, v in _trace_rows(self.traces)]),
        ]
        if self.edit_pairs:
            sections += ["", *_edit_change_lines(self.edit_pairs)]
        if self.reasoning:
            sections += ["", *_reasoning_lines(self.reasoning)]
        if self.item_records:
            sections += ["", *_item_evidence(self.item_records)]
            sections += ["", *_trajectories(
                self.item_records, budget=self.evidence_budget,
                findings=self.reader_findings, failures=self.reader_failures)]
        sections += [
            "",
            "## What this run has already tried",
            (
                f"{len(self.history)} candidate(s), best "
                f"{max((c.get('score') or 0.0) for c in self.history):.6g}"
                if self.history and not _gate_column(self.history)
                else f"{len(self.history)} candidate(s), best kept by the validation gate "
                f"{max((c.get('score') or 0.0) for c in self.history if c.get('gate') in ('kept', 'current')):.6g}"
                if any(c.get("gate") in ("kept", "current") for c in self.history)
                else f"{len(self.history)} candidate(s), none kept by the validation gate yet"
                if self.history
                else "(nothing evaluated yet)"
            )
            + (
                " — older candidates omitted, so do not assume the list is complete"
                if self.history_truncated
                else ""
            ),
            _table(
                ["id", "parent", "score", "structural", "ok", "edits", *_gate_column(self.history)],
                [
                    [c.get("id"), c.get("parent"), c.get("score"), c.get("structural"),
                     c.get("exit_ok"), compact_edits(c.get("edits"), self.seed_knob_values),
                     *([c.get("gate", "")] if _gate_column(self.history) else [])]
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
        sections += ["", "## Answer", answer_instructions(
            self.catalogue, parsimony=self.parsimony,
            items=_item_count(self.item_records) if self.score_is_item_mean else 0)]
        if self.max_edits > 1:
            sections += ["", bundle_instructions(self.max_edits, self.catalogue,
                                                 proposals=self.proposals)]
        if self.edit_levels:
            sections += ["", *levels_instructions(self.edit_levels)]
        return "\n".join(sections)


#: The edit levels a portfolio round asks for, one proposal each (``edit_level``).
EDIT_LEVELS = ("prompt_knob", "pipeline")
_LEVEL_TEXT = {
    "prompt_knob": "a prompt/knob-level change: only `set_knob` edits (rewrite or patch a "
                   "prompt, move a setting) on the current pipeline",
    "pipeline": "a pipeline-level change: at least one structural operation (add, remove, "
                "rewire, gate, loop or parallelise steps), with any knob moves it needs",
}


def edit_level(steps: Sequence[Any]) -> str:
    """``prompt_knob`` when every edit is a ``set_knob``, else ``pipeline``."""
    ops = [step.get("op") if isinstance(step, dict) else None for step in steps]
    return "prompt_knob" if ops and all(op == "set_knob" for op in ops) else "pipeline"


def levels_instructions(levels: Sequence[str]) -> list[str]:
    """What the request asks for when it names edit levels."""
    if len(levels) == 1:
        return ["## The kind of change this round",
                f"Propose exactly one update: {_LEVEL_TEXT.get(levels[0], levels[0])}."]
    return ["## The kind of change this round",
            f"Propose exactly {len(levels)} updates, one per level, each your best of its "
            "kind:",
            *[f"- {_LEVEL_TEXT.get(level, level)}" for level in levels],
            "The search evaluates every one of them on validation and keeps what wins; do "
            "not choose between them, and do not leave one out."]


#: Keys of ``ProposalContext.strategy`` that are not a decision of the outer loop: the
#: run's resolved search contract rides in the same dict and is not rendered here.
_NOT_DECISIONS = ("search_contract",)
#: Decision keys said first, as sentences, in this order.
_DECISION_LEAD = ("purpose", "why")


def _strategy_section(decision: Mapping[str, Any] | None, strategy: str = "") -> list[str]:
    """What the outer loop decided for this round, as data: its purpose and reason
    first, then every other key it published. Empty when it decided nothing (the
    default loop), so a request without a decision is unchanged."""
    shown = {key: value for key, value in (decision or {}).items()
             if key not in _NOT_DECISIONS and value not in (None, "", [], {})}
    if not shown:
        return []
    who = f"The search strategy (`{strategy}`)" if strategy else "The search strategy"
    lines = ["## What this round is for",
             f"{who} decided the following for this round. It decides where the search "
             "goes; you decide what to write. Take it into account, and say in your "
             "rationale if the evidence argues against it."]
    for key in (*[k for k in _DECISION_LEAD if k in shown],
                *[k for k in shown if k not in _DECISION_LEAD]):
        value = shown[key]
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        lines.append(f"- {key}: {_bounded(text, 600)}")
    return [*lines, ""]


#: The catalogue kinds that change the pipeline's structure (``nn.edits.catalogue``).
STRUCTURE_KINDS = ("node", "edge", "topology")
#: Catalogue fields an example leaves out: optional for every op that has them.
_OPTIONAL_EDIT_FIELDS = ("condition", "new_id", "patch", "config")
#: Placeholders for the catalogue fields that name an existing step.
_STEP_FIELDS = ("target", "id", "source", "first", "second")


def structural_ops(catalogue: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The permitted operations that change the pipeline's structure."""
    return [op for op in catalogue if op.get("kind") in STRUCTURE_KINDS]


def example_node() -> dict[str, Any]:
    """A new agent node for an example, from the generated node schema: its required
    fields with placeholders, its own system prompt, and the one write a node without
    tools declares (its reply; rule V15)."""
    from refactory_dsh.nn.node_schema import node_schema

    fill = {"id": "<new step id>", "kind": "agent", "prompt_template": "<prompt text>"}
    node = {name: fill.get(name, f"<{name}>") for name in node_schema()["agent"]["required"]}
    node["system_prompt_template"] = "<system prompt text>"
    node["writes"] = ["<new step id>.txt"]
    return node


def _node_op(catalogue: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The structural op an example uses: one that puts a new node after an existing
    step (``target`` and ``node``), else any that takes a node, else any structural op."""
    ops = structural_ops(catalogue)
    for wanted in (lambda f: set(f) == {"target", "node"}, lambda f: "node" in f, lambda f: True):
        for op in ops:
            if wanted(op.get("fields") or []):
                return op
    return None


def structural_example(catalogue: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """An example edit for a structural op this run permits, built from its catalogue
    fields with placeholders; ``None`` when none is permitted."""
    op = _node_op(catalogue)
    if op is None:
        return None
    edit: dict[str, Any] = {"op": op["op"]}
    for name in op.get("fields") or []:
        if name in _OPTIONAL_EDIT_FIELDS:
            continue
        if name == "node":
            edit[name] = example_node()
        elif name == "position":
            edit[name] = {"after": "<existing step id>"}
        elif name in _STEP_FIELDS:
            edit[name] = "<existing step id>"
        else:
            edit[name] = f"<{name}>"
    return edit


def _item_count(records: Sequence[dict[str, Any]]) -> int:
    return len({str(record.get("id")) for record in records or ()})


def _blocks_section(insertable: Sequence[dict[str, Any]],
                    catalogue: Sequence[dict[str, Any]]) -> list[str]:
    """Where a new node comes from: the mode's blocks, verbatim, when it declares any;
    otherwise the node schema. Nothing when the run permits no structural op."""
    if not structural_ops(catalogue):
        return []
    if not insertable:
        return ["## New nodes",
                "This mode declares no ready-made blocks. A new node is written from the node "
                "schema above (or copied from a node's JSON under \"Each node in full\", with a "
                "new id).", ""]
    return ["## Blocks you may insert",
            "Use these node objects verbatim in an edit that takes a node, changing nothing "
            "but their `id`, or write a node from the node schema above:",
            "```json", json.dumps(list(insertable), indent=2, default=str), "```", ""]


def _composition_anchor(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                        protected: Sequence[str]) -> tuple[dict[str, Any], str] | None:
    """An agent of this graph with one plain out-edge to a step that is not a gate,
    and that out-edge's target: where the composition examples attach. ``None`` when
    the graph has none."""
    by_id = {str(state.get("id")): state for state in node_states}
    for node_id in sorted(by_id):
        state = by_id[node_id]
        if state.get("kind") != "agent" or node_id in protected or not state.get("writes"):
            continue
        out = [edge for edge in edges if edge.get("source") == node_id]
        if len(out) != 1 or out[0].get("condition") is not None:
            continue
        target = by_id.get(str(out[0].get("target"))) or {}
        if target.get("kind") in ("agent", "op"):
            return state, str(out[0].get("target"))
    return None


def composition_section(catalogue: Sequence[dict[str, Any]], node_states: Sequence[dict[str, Any]],
                        edges: Sequence[dict[str, Any]], *, protected: Sequence[str] = (),
                        code_nodes: bool = False) -> list[str]:
    """The language's primitives and how they compose, with one short example per
    composition the run permits (a new agent, a code op, a verdict loop, a parallel
    fork), built from this graph's own ids, so no one shape reads as the way to grow
    the pipeline. Nothing when the run permits no structural op."""
    from refactory_dsh import snippet

    if not structural_ops(catalogue):
        return []
    ops = {op.get("op") for op in catalogue}
    lines = [
        "## How nodes compose",
        "The pipeline is made of four kinds of step, and any of them may be put anywhere the "
        "rules above allow:",
        "- **agent**: one model call. Its prompt fills each `{{name}}` from a fact (or knob) "
        "produced upstream; its reply is written to its first `writes` file and, when it "
        "declares `produces`, also becomes that fact. A new agent runs without tools, so it "
        "sees only its prompt: hand it what it needs as `{{facts}}`.",
    ]
    if code_nodes:
        lines += [
            f"- **code op**: `\"call\": \"{snippet.RUN}\"` with `code`, Python that defines "
            "`main(inp)`. `inp` is `{\"files\": {each declared read that exists: its text}, "
            "\"facts\": {each name in its `inputs`: its value}}`; `main` returns "
            "`{\"facts\": {...}, \"files\": {...}}`, each one declared in its `produces` / "
            "`writes`. It costs no model call. Limits: imports only these modules: "
            f"{snippet.ALLOWED_IMPORTS_TEXT}; at most {snippet.MAX_CODE_BYTES} "
            f"bytes, {snippet.WALL_S} s, no files, network, open/exec/eval or dunders (rule "
            "V18). The code goes into the node's own `<id>__code` knob, which a later "
            "`set_knob` rewrites.",
        ]
    lines += [
        "- **gate**: routes by a verdict, to the out-edge whose `condition` is that verdict. "
        "Give each out-edge a single-word condition (`proceed`, `reloop`, `revise` ...). "
        f"Any declared upstream output can be the verdict: with `\"evaluator\": "
        f"\"{snippet.VERDICT}\"` the gate reads the first word of its first `required_reads` "
        "file (or, with none, of its first `inputs` fact) -- whatever an agent or a code op "
        "upstream wrote there; leading markdown or quotes are skipped -- and the rest of the "
        "text is the reason. That first word must be one of the conditions (any case) or "
        "`halt`, which ends the whole run; anything else fails the run, so tell the writer "
        "to start with the word.",
    ]
    if code_nodes:
        lines.append(
            f"  With `\"evaluator\": \"{snippet.GATE}\"` and `code`, `main(inp)` returns "
            "`{\"verdict\": <condition or halt>, \"reason\": <text>}`.")
    lines += [
        "- **loop**: a gate's out-edge back to an earlier step. The `loop` operation makes "
        "one around a step (step -> gate; gate -> step on `reloop`, gate -> the step's old "
        "successor on `proceed`); a reloop hands the gate's reason to the step it goes back "
        "to as feedback, and the gate's `max_iterations` bounds how often it may. When a gate "
        "would reloop once more than that, its `on_exhausted` decides: `continue` (the "
        "default inside the per-item pass) goes on along `proceed` with the last attempt, "
        "which is graded as it stands; `fail_item` fails that item only; `halt` ends the "
        "whole run (no later item runs). A gate with no forward `proceed` edge defaults to "
        "`fail_item`. The last step of the pass can be looped too: the "
        "loop then adds a step after the gate to end the pass.",
    ]
    if "parallel" in ops:
        lines.append(
            "- **fork**: the `parallel` operation runs 2-8 new steps at once between a fork "
            "and a join after a step; branches write distinct files and facts. Nothing reads "
            "them until you say so: insert a node after the join that reads the branches' "
            "files or facts, or use a branch's fact as `{{name}}` in a later step's prompt "
            "knob.")
    anchor = _composition_anchor(node_states, edges, protected)
    if anchor is None or not ({"loop", "insert_before", "insert_after"} & ops):
        return lines + [""]
    agent, successor = anchor
    step = str(agent["id"])
    output = str((agent.get("writes") or ["output.txt"])[0])
    examples: list[tuple[str, list[dict[str, Any]]]] = []
    if "insert_after" in ops:
        extra = f"{step}_second"
        examples.append((
            f"A new agent step after `{step}`, reading its output as a fact:",
            [{"op": "insert_after", "target": step,
              "node": {"id": extra, "kind": "agent", "writes": [f"{extra}.txt"],
                       "prompt_template": "<what this step does; put each fact it needs in double braces>"}}]))
    if "parallel" in ops:
        examples.append((
            f"Two steps at once after `{step}` (a fork and a join):",
            [{"op": "parallel", "target": step, "branches": [
                {"id": f"{step}_{side}", "kind": "agent", "writes": [f"{step}_{side}.txt"],
                 "prompt_template": f"<angle {side}>"} for side in ("a", "b")]}]))
    if {"loop", "insert_before"} <= ops:
        review = f"{step}_review"
        examples.append((
            f"An agent judges `{step}`'s work and sends it back with a reason (a verdict gate "
            f"reading the reviewer's file; `{review}`'s prompt needs `{step}`'s output as a "
            "fact -- a code op can turn the file into one):",
            [{"op": "loop", "target": step,
              "gate": {"id": f"{step}_check", "kind": "gate", "evaluator": snippet.VERDICT,
                       "reads": [f"{review}.txt"], "required_reads": [f"{review}.txt"],
                       "max_iterations": 2, "on_exhausted": "continue"}},
             {"op": "insert_before", "target": f"{step}_check",
              "node": {"id": review, "kind": "agent", "writes": [f"{review}.txt"],
                       "prompt_template": "<what to check, with the work as fact slots; "
                                          "answer 'proceed' or 'reloop' first, then why>"}}]))
    if code_nodes and "insert_after" in ops:
        fact = f"{step}_digest"
        examples.append((
            f"A code op computes a fact from `{step}`'s file for `{successor}` or any later "
            f"step to use as `{{{{{fact}}}}}`:",
            [{"op": "insert_after", "target": step,
              "node": {"id": fact, "kind": "op", "call": snippet.RUN, "reads": [output],
                       "produces": [fact],
                       "code": ("def main(inp):\n"
                                f"    text = inp['files'].get({output!r}, '')\n"
                                f"    return {{'facts': {{{fact!r}: text.strip()[-2000:]}}}}\n")}}]))
    lines += ["", "Examples, one per composition (built from this graph; none is the "
                  "preferred shape -- compose your own the same way):"]
    for title, bundle in examples:
        lines += [f"- {title}", "  " + json.dumps({"edits": bundle} if len(bundle) > 1 else bundle[0])]
    return lines + [""]


def answer_instructions(catalogue: Sequence[dict[str, Any]], *, parsimony: float | None = None,
                        items: int = 0) -> str:
    """The answer format and its rules, with examples for what this run permits.

    ``items`` is the number of train items when the raw score is known to be their
    mean (0 otherwise): only then does the size cost line compare a node with one item.
    """
    structural = structural_example(catalogue)
    lines = [
        "Write one JSON object and nothing else:",
        "",
        '{"diagnosis": "<what the evidence shows is limiting the score>",',
        ' "proposals": [{"edit": {"op": "<operation>", ...},',
        '                "rationale": "<why, citing the evidence>",',
        '                "prediction": "<expected effect on the score>",',
        '                "confidence": 0.0}]}',
        "",
        "Rules:",
        '- A parameter move is {"op": "set_knob", "name": "<parameter>", "value": <value>}.',
    ]
    if any(op.get("op") == "set_knob" for op in catalogue):
        lines.append("- A str parameter may be patched instead of rewritten: "
                     + nn_edits.PATCH_FORMAT + ".")
    if structural is not None:
        lines += [
            "- A structural move is one of the structural operations above, with the fields it "
            "lists. A `node` is a node object written from the node schema (or a block, or a "
            "copy of a node's JSON, with a new id). For example:",
            "  " + json.dumps(structural),
            "  A new agent node gets its own `<new step id>__system_prompt` knob holding its "
            "system prompt.",
        ]
    else:
        lines.append("- This run permits parameter moves only.")
    if parsimony:
        size = f"- Size cost: {parsimony:g} per node (score = raw score - {parsimony:g} x nodes)"
        if items:
            # Said only when the raw score is the mean of the items' scores.
            size += (f"; the raw score is the mean over the {items} train items, so one item "
                     f"is worth 1/{items} = {1 / items:.3g}")
        else:
            size += "; compare it with the size of the score differences between candidates"
        lines.append(size + ".")
    lines += [
        "- Few well-argued changes, not many guesses. Prefer an edit whose effect you can",
        "  state, and vary what you touch.",
        "- Your prediction is recorded and later checked against the outcome, so say what",
        "  you expect and how you could be wrong.",
        "- If the evidence justifies no change, return an empty proposal list and say why.",
    ]
    return "\n".join(lines) + "\n"


#: The answer format for a run that permits every operation, as the request shows it.
ANSWER_INSTRUCTIONS = answer_instructions(
    [{"op": "insert_after", "kind": "topology", "fields": ["target", "node"]}])


def bundle_examples(catalogue: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Bundles for the example, neither kind first by privilege: two steps' prompt knobs,
    then -- when the run also permits a structural op that takes a node and
    ``set_knob`` -- a structural edit that adds a node and the new node's own prompt
    knob."""
    out = [[{"op": "set_knob", "name": "<step A's prompt knob>", "value": "<text>"},
            {"op": "set_knob", "name": "<step B's prompt knob>", "value": "<text>"}]]
    structural = structural_example(catalogue)
    ops = {op.get("op") for op in catalogue}
    if structural is not None and "node" in structural and "set_knob" in ops:
        # The node leaves its system prompt to the knob the bundle sets.
        node = {k: v for k, v in structural["node"].items() if k != "system_prompt_template"}
        out.append([{**structural, "node": node},
                    {"op": "set_knob", "name": "<new step id>__system_prompt",
                     "value": "<system prompt text>"}])
    return out


def bundle_instructions(max_edits: int, catalogue: Sequence[dict[str, Any]] = (),
                        proposals: int = 0) -> str:
    """How a proposal bundles edits, said only when the run allows more than one.

    When the request asks for a single proposal (``proposals == 1``) only one
    candidate is evaluated, so an idea put in a separate proposal is never run, and
    fixes that only work together would be split across rounds and each judged
    alone: the advice is then to cover every step of the diagnosed failure path in
    the one proposal. Otherwise separate ideas go in separate proposals."""
    example = json.dumps({"proposals": [{"edits": bundle,
                                         "rationale": "<why these changes only work together>",
                                         "prediction": "<expected effect on the score>",
                                         "confidence": 0.0} for bundle in bundle_examples(catalogue)]})
    if proposals == 1:
        scope = f"""\
Only one proposal is asked for, so only one candidate is evaluated: an idea left for a
separate proposal is not run this round. When your diagnosis traces a failure through
several steps, make this one proposal's edits cover every step it names on that
failure path (up to {max_edits} edits), so the fix is evaluated as a whole; changes that
only work together each fail when tried alone."""
    else:
        scope = """\
Bundle only what belongs to one idea: separate ideas go in separate proposals, so each
is judged on its own."""
    return f"""\
Bundling: when one fix needs coordinated changes -- several steps' prompts that have
to agree, or a new node and the knob that feeds it -- one proposal may carry up to
{max_edits} edits as an ordered list under "edits" instead of "edit". They are applied in
order to the same parent, each to the state the one before it left, and the result is
evaluated and judged once as one candidate; if any edit is refused, the whole proposal
is, and the refusal names the edit.

{scope}

Example:

{example}"""


#: How many error causes and wrong answers the item summary lists, and how much of
#: each it quotes: enough to see a pattern, bounded so a large part stays a section.
ITEM_EVIDENCE_ROWS = 6
ITEM_EVIDENCE_CHARS = 240


def _clip(text: Any, limit: int = ITEM_EVIDENCE_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _item_evidence(records: Sequence[dict[str, Any]]) -> list[str]:
    """How the candidate being edited did item by item, on its train part.

    A score is a mean over items, and a mean cannot say *why* it is low: an item
    answered wrong and an item whose completion failed (a model call that stopped
    at its output token limit, a timeout) cost the same. The loop already hands the
    proposer these records (``item_records``); this is where it reads them. Train
    records only, as ``OuterLoop.item_records`` enforces.
    """
    errored = [r for r in records if r.get("error_text")]
    passed = [r for r in records if not r.get("error_text") and r.get("passed")]
    wrong = [r for r in records if not r.get("error_text") and not r.get("passed")]
    scores = [float(r["score"]) for r in records if isinstance(r.get("score"), (int, float))]
    mean = f", mean score {sum(scores) / len(scores):.3g}" if scores else ""
    lines = [
        "## Item results of the candidate being edited",
        f"{len(records)} train item(s): {len(passed)} passed, {len(wrong)} answered "
        f"wrong, {len(errored)} failed with an error{mean}. An error is the item's "
        "run failing (its cause is below), not a wrong answer.",
    ]
    if errored:
        causes: dict[str, list[str]] = {}
        for record in errored:
            causes.setdefault(_clip(record["error_text"]), []).append(str(record.get("id", "")))
        ranked = sorted(causes.items(), key=lambda kv: -len(kv[1]))
        lines += [
            "",
            "Errors, by cause:",
            _table(["items", "cause", "e.g."],
                   [[len(ids), cause, ", ".join(ids[:3])]
                    for cause, ids in ranked[:ITEM_EVIDENCE_ROWS]]),
        ]
        if len(ranked) > ITEM_EVIDENCE_ROWS:
            lines.append(f"({len(ranked) - ITEM_EVIDENCE_ROWS} more cause(s) not shown)")
    reasons = [r for r in wrong if r.get("fail_reason")]
    if reasons:
        lines += [
            "",
            "Wrong answers, with the grader's reason:",
            _table(["item", "score", "reason"],
                   [[r.get("id"), r.get("score"), _clip(r["fail_reason"])]
                    for r in reasons[:ITEM_EVIDENCE_ROWS]]),
        ]
        if len(reasons) > ITEM_EVIDENCE_ROWS:
            lines.append(f"({len(reasons) - ITEM_EVIDENCE_ROWS} more not shown)")
    return lines


def _clip_middle(text: Any, limit: int) -> str:
    """``text`` whole when it fits ``limit``, else its head and tail with the cut said.

    ``None`` is empty; any other value (a prediction of ``0`` included) is shown."""
    text = ("" if text is None else str(text)).strip()
    if len(text) <= limit:
        return text
    keep = max(0, limit // 2)
    return f"{text[:keep]}\n[... {len(text) - 2 * keep} characters omitted ...]\n{text[-keep:]}"


def _choice_lines(choices: Any) -> list[str]:
    lines: list[str] = []
    for index, choice in enumerate(choices if isinstance(choices, list) else []):
        if isinstance(choice, dict):
            label = choice.get("label", chr(ord("A") + index))
            lines.append(f"{label}. {choice.get('text', '')}")
        else:
            lines.append(f"{chr(ord('A') + index)}. {choice}")
    return lines


#: A text kept whole (``_trajectory`` clips only to fit one item into the budget).
_WHOLE = 10 ** 9


def _trajectory(record: dict[str, Any], text_chars: int = _WHOLE) -> str:
    """One train item in full: input, each step's output, the answer.

    The steps are in the order the record holds them, which is the order the pass
    ran them (a fork merges its branches in branch order), never sorted by name.
    Texts are whole unless ``text_chars`` clips them (head and tail).
    """
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    exchanges = [e for e in record.get("exchanges") or () if isinstance(e, dict)]
    lines = [f"### Item {record.get('id')} — {_outcome(record)} (score {record.get('score')})"]
    key = next((k for k in TRAJECTORY_INPUT_KEYS if meta.get(k)), None)
    if key is not None:
        lines += [f"Input ({key}):", _clip_middle(meta[key], text_chars)]
    elif exchanges:
        lines += ["Input (the first step's prompt):",
                  _clip_middle(exchanges[0].get("prompt"), text_chars)]
    choices = _choice_lines(meta.get("choices"))
    if choices:
        lines += ["Choices:", *(_clip_middle(line, text_chars) for line in choices)]
    for exchange in exchanges:
        lines += [f"Step `{exchange.get('node') or 'agent'}` output:",
                  _clip_middle(exchange.get("response"), text_chars)]
    if not exchanges:
        lines.append("(no step output was recorded)")
    expected = _expected(record)
    if expected:
        lines.append(f"Expected: {_clip_middle(expected, text_chars)}")
    if record.get("prediction") is not None:
        lines.append(f"Predicted: {_clip_middle(record.get('prediction'), text_chars)}")
    if record.get("fail_reason") and not record.get("passed"):
        lines.append(f"Grader: {_clip_middle(record['fail_reason'], text_chars)}")
    if record.get("error_text"):
        lines.append(f"Error: {_clip_middle(record['error_text'], text_chars)}")
    return "\n".join(lines)


def _outcome(record: dict[str, Any]) -> str:
    return ("passed" if record.get("passed") else
            "failed with an error" if record.get("error_text") else "answered wrong")


def _expected(record: dict[str, Any]) -> str:
    label, label_text = record.get("label"), record.get("label_text")
    return " — ".join(str(part) for part in (label, label_text) if part not in (None, ""))


_CLIPPED_ITEM_NOTE = ("(This item alone is longer than one call's evidence budget, so its "
                      "texts are clipped head and tail.)")


def _item_block(record: dict[str, Any], budget: int) -> str:
    """An item in full; clipped (head and tail, with a note) only when the item alone
    is longer than ``budget``."""
    whole = _trajectory(record)
    if len(whole) <= budget:
        return whole
    low, high = 1, max(1, budget)
    while low < high:  # the largest per-text limit that fits
        middle = (low + high + 1) // 2
        if len(_trajectory(record, middle)) + len(_CLIPPED_ITEM_NOTE) + 1 <= budget:
            low = middle
        else:
            high = middle - 1
    return _trajectory(record, low) + "\n" + _CLIPPED_ITEM_NOTE


def _ordered_items(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every train item once (its first repeat): wrong, then failed with an error,
    then passed, each group by id."""
    first: dict[str, dict[str, Any]] = {}
    for record in records:
        item = str(record.get("id", ""))
        held = first.get(item)
        if held is None or int(record.get("repeat") or 1) < int(held.get("repeat") or 1):
            first[item] = record
    ordered = [first[item] for item in sorted(first)]
    return ([r for r in ordered if not r.get("passed") and not r.get("error_text")]
            + [r for r in ordered if not r.get("passed") and r.get("error_text")]
            + [r for r in ordered if r.get("passed")])


def trajectory_groups(records: Sequence[dict[str, Any]],
                      budget: int = TRAJECTORY_BUDGET) -> list[list[tuple[str, str]]]:
    """The items as ``(id, full block)`` groups that each fit ``budget``.

    One group when everything fits; otherwise as few balanced groups as fit, in the
    stable order of ``_ordered_items``, each item in exactly one group.
    """
    import math

    blocks = [(str(r.get("id")), _item_block(r, budget)) for r in _ordered_items(records)]
    if not blocks:
        return []
    total = sum(len(block) + 2 for _id, block in blocks)
    if total <= budget:
        return [blocks]
    target = total / math.ceil(total / budget)
    groups: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    size = 0
    for item, block in blocks:
        step = len(block) + 2
        if current and (size + step > budget or size >= target):
            groups.append(current)
            current, size = [], 0
        current.append((item, block))
        size += step
    groups.append(current)
    return groups


#: A cell of the every-item table, at most this long.
_TABLE_CELL_CHARS = 200


def _reader_patterns(findings: Sequence[dict[str, Any]] | None) -> list[tuple[str, dict[str, Any]]]:
    """The readers' patterns the request shows, as ``(kind, pattern)`` in their order:
    reader by reader, failure patterns then success patterns, at most
    ``READER_PATTERNS`` per reader."""
    out: list[tuple[str, dict[str, Any]]] = []
    for found in findings or ():
        shown = 0
        for kind in ("failure_patterns", "success_patterns"):
            for pattern in found.get(kind) or ():
                if not isinstance(pattern, dict) or shown >= READER_PATTERNS:
                    continue
                shown += 1
                out.append((kind, pattern))
    return out


def _cited_items(items: Sequence[dict[str, Any]],
                 findings: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """The items the readers cited, in the order they are shown in full: one item per
    pattern in turn (failure patterns before success patterns), so every pattern has
    an item shown before any pattern has a second. Each item once; ids no item has
    are skipped. Empty without findings."""
    by_id = {str(record.get("id")): record for record in items}
    patterns = sorted(_reader_patterns(findings), key=lambda pair: pair[0] != "failure_patterns")
    queues = [iter([item for item in dict.fromkeys(str(i) for i in pattern.get("items") or ())
                    if item in by_id]) for _kind, pattern in patterns]
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    while queues:
        still = []
        for queue in queues:
            for item in queue:
                if item not in seen:
                    seen.add(item)
                    chosen.append(by_id[item])
                    still.append(queue)
                    break
        queues = still
    return chosen


def _trajectories(records: Sequence[dict[str, Any]], *, budget: int = TRAJECTORY_BUDGET,
                  findings: Sequence[dict[str, Any]] | None = None,
                  failures: Sequence[dict[str, Any]] = ()) -> list[str]:
    """Every train item of the candidate being edited.

    When all of them fit ``budget`` in full, every item is shown in full here. When
    not, readers went through them in groups (``AutoresearchOptimizer``): their
    findings come first, then a table of every item, then as many items in full as
    the rest of the budget holds: the items the readers cited first, one per pattern
    in turn (``_cited_items``), then failures and passes alternating. Train records
    only.
    """
    items = _ordered_items(records)
    if not items:
        return []
    wrong = sum(1 for r in items if _outcome(r) == "answered wrong")
    errored = sum(1 for r in items if _outcome(r) == "failed with an error")
    header = ["## Trajectories of the candidate being edited",
              f"All {len(items)} train item(s): {wrong} answered wrong, {errored} failed with "
              f"an error, {len(items) - wrong - errored} passed."]
    groups = trajectory_groups(records, budget)
    if len(groups) == 1:
        return header + ["Every item in full. Read what each step produced and where it "
                         "went wrong before proposing.",
                         *(line for _id, block in groups[0] for line in ("", block))]
    lines = header + [
        f"They do not fit one request in full, so readers went through them in "
        f"{len(groups)} groups of whole items. Their findings come first, then a table of "
        "every item, then as many items in full as fit."]
    if findings is None:
        lines.append("(No reader went through these items for this request: read the "
                     "table and the items in full below.)")
    else:
        lines += ["", "### What the readers found"]
        found_lines: list[str] = []
        for kind, pattern in _reader_patterns(findings):
            ids = [str(i) for i in pattern.get("items") or ()]
            id_text = ", ".join(ids[:READER_ITEM_IDS]) + (
                f" and {len(ids) - READER_ITEM_IDS} more" if len(ids) > READER_ITEM_IDS else "")
            quotes = [f'"{_clip(quote, READER_QUOTE_CHARS)}"'
                      for quote in list(pattern.get("evidence") or ())[:READER_QUOTES]]
            found_lines.append(
                f"- [{kind.split('_')[0]}] {_clip(pattern.get('pattern', ''), READER_PATTERN_CHARS)} "
                f"(step `{_clip(pattern.get('step', 'unclear'), 60)}`; items {id_text}) "
                "-- evidence: " + " | ".join(quotes))
        cap = int(budget * READER_FINDINGS_SHARE)
        used_findings = 0
        for index, line in enumerate(found_lines):
            if used_findings + len(line) + 1 > cap:
                lines.append(f"({len(found_lines) - index} more pattern(s) not shown: the "
                             f"findings are kept within {cap} characters)")
                break
            lines.append(line)
            used_findings += len(line) + 1
        for failure in failures:
            lines.append(f"(Group {failure.get('group')} -- items "
                         f"{', '.join(str(i) for i in failure.get('items') or ())} -- could "
                         f"not be read: {_clip(failure.get('error'), 300)}; those items are in "
                         "the table.)")
    lines += ["", "### Every item",
              _table(["item", "outcome", "expected", "predicted", "grader", "error"],
                     [[r.get("id"), _outcome(r), _clip(_expected(r), _TABLE_CELL_CHARS),
                       _clip(r.get("prediction"), _TABLE_CELL_CHARS),
                       _clip(r.get("fail_reason") if not r.get("passed") else "", _TABLE_CELL_CHARS),
                       _clip(r.get("error_text"), _TABLE_CELL_CHARS)] for r in items])]
    used = len("\n".join(lines))
    examples = ["", "### Some items in full"]
    failing = [r for r in items if not r.get("passed")]
    passing = [r for r in items if r.get("passed")]
    interleaved = [r for pair in zip(failing, passing) for r in pair]
    interleaved += failing[len(passing):] + passing[len(failing):]
    # The items behind the readers' patterns first, so the proposal call sees each
    # pattern in full; then the rest by the rule above.
    cited = _cited_items(items, findings)
    if cited:
        examples.append("The items the readers cited come first, one per pattern in turn, "
                        "then the others.")
    picked = {id(record) for record in cited}
    shown = 0
    for record in cited + [r for r in interleaved if id(r) not in picked]:
        block = _item_block(record, budget)
        if used + len("\n".join(examples)) + len(block) + 2 > budget:
            continue
        examples += ["", block]
        shown += 1
    return lines + (examples if shown else [])


def _edit_fields(fields: Sequence[str]) -> str:
    """An operation's fields, saying which take a whole node object."""
    described = []
    for name in fields:
        if name == "into":
            described.append("into: a list of two full node objects (see the node schemas above)")
        elif name == "branches":
            described.append("branches: a list of 2-8 full node objects (see the node schemas above)")
        elif name in NODE_OBJECT_FIELDS:
            described.append(f"{name}: a full node object (see the node schemas above)")
        else:
            described.append(name)
    return ", ".join(described)


def _node_spec(state: dict[str, Any], limit: int,
               knob_values: Mapping[str, Any] | None = None) -> str:
    """One node in full, as a node object to copy; long texts clipped, and said.

    A field bound to a knob shows the knob's current value (the stored field is not
    what runs), and says to edit the knob. The bindings themselves, and the bound
    knobs in ``knobs``, are left out: a copy must not bind the original's knob.
    """
    knob_values = knob_values or {}
    bound = {str(pair[0]): str(pair[1]) for pair in state.get("field_bindings") or ()
             if isinstance(pair, (list, tuple)) and len(pair) == 2}
    shown: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in state.items():
        if key == "field_bindings":
            continue
        if key == "knobs":
            value = [name for name in value or () if name not in bound.values()]
        if key in bound:
            value = knob_values.get(bound[key], value)
            notes.append(f"`{key}` is bound to the knob `{bound[key]}` and shows its current "
                         "value: edit the knob, not this field.")
        if isinstance(value, str) and len(value) > limit:
            shown[key] = _clip_middle(value, limit)
            notes.append(f"`{key}` is clipped here ({len(value)} characters in full); do not "
                         "copy it.")
        else:
            shown[key] = value
    lines = [f"#### `{state.get('id')}` ({state.get('kind')})", "```json",
             json.dumps(shown, default=str), "```"]
    return "\n".join(lines + notes)


def _node_specs(states: Sequence[dict[str, Any]], *, text_chars: int = NODE_SPEC_TEXT_CHARS,
                section_chars: int = NODE_SPEC_SECTION_CHARS,
                knob_values: Mapping[str, Any] | None = None) -> list[str]:
    """Each node of the edited graph in full, then the generated node schemas.

    A structure edit (``add_node``, ``insert_after``, ``split``, ...) needs a whole
    node object; with these a model copies a node, changes its id and the fields it
    means to change, and uses it verbatim. Fits ``section_chars`` exactly: the text
    limit shrinks (to ``NODE_SPEC_TEXT_FLOOR``) before nodes are left out, last first,
    with a note naming them.
    """
    from refactory_dsh.nn.node_schema import render_schema

    if not states:
        return []
    schema = render_schema()
    ordered = sorted(states, key=lambda s: str(s.get("id")))

    def section(chosen: Sequence[dict[str, Any]], limit: int) -> list[str]:
        lines = ["### Each node in full",
                 "Copy a node object, give it a new id, change what you mean to change "
                 "(prompt, reads, writes, produces, knobs), and use it as the `node` of an "
                 "`add_node`, `insert_after`, `insert_before` or `wrap` edit. Knob bindings "
                 "are left out of these copies: a new agent node gets its own "
                 "`<id>__system_prompt` knob holding its `system_prompt_template`, and a "
                 "new node may not bind another node's knob. A clipped text (marked "
                 "`[... N characters omitted ...]`) must not be copied: write your own, or "
                 "edit the knob; an edit carrying the marker is refused."]
        lines += [_node_spec(state, limit, knob_values) for state in chosen]
        left = [str(s.get("id")) for s in ordered[len(chosen):]]
        if left:
            lines.append(f"({len(left)} node(s) not shown in full to stay within "
                         f"{section_chars} characters: {', '.join(left)})")
        return lines + [""] + schema

    def size(lines: list[str]) -> int:
        return len("\n".join(lines))

    chosen = list(ordered)
    floor = min(NODE_SPEC_TEXT_FLOOR, text_chars)
    while True:
        if size(section(chosen, floor)) <= section_chars:
            low, high = floor, max(floor, text_chars)
            while low < high:
                middle = (low + high + 1) // 2
                if size(section(chosen, middle)) <= section_chars:
                    low = middle
                else:
                    high = middle - 1
            return section(chosen, low)
        if not chosen:
            return schema if size(schema) <= section_chars else []
        chosen = chosen[:-1]


def _gate_column(rows: Sequence[dict[str, Any]]) -> list[str]:
    """``["gate"]`` when the rows carry the validation gate's decisions, else nothing."""
    return ["gate"] if any("gate" in row for row in rows) else []


def _gate_note(*tables: Sequence[dict[str, Any]], val_items: int = 0) -> list[str]:
    """What the gate column means, said once, when there is one.

    A rejection says only that the candidate was not shown better on validation; with
    ``val_items`` (the validation items the gate's score is the mean of) the note says
    what one item is worth, so a one-item loss is read as the gate's resolution."""
    if not any(_gate_column(rows) for rows in tables):
        return []
    resolution = (f"The gate's score is the mean over {val_items} validation items, so one "
                  f"item is 1/{val_items} = {1 / val_items:.3g} of it, and a rejection can come "
                  "down to a single item. " if val_items > 0 else "")
    return ["This run has a validation gate: each candidate was also run on held-out "
            "items, and the gate kept it only if it scored strictly higher there than the "
            "current candidate (an equal score is rejected). Scores here are train scores; "
            "held-out scores are never shown. You edit the candidate marked `current`. A "
            "`rejected` candidate was not shown better on validation; that is all a "
            "rejection says. " + resolution + "A rejected candidate is never kept, and its "
            "change must not be proposed again unchanged. When its train fixed/broken counts "
            "against its parent (when shown below) were strongly positive, its direction may "
            "still be right: "
            "you may build on its change or combine it with edits to other steps in the "
            "same failure path."]


#: The section comparing the last edit's child with its parent fits this many
#: characters: changed items in full while they fit, the rest in a table.
EDIT_CHANGE_BUDGET = 60000
#: The most score changes in each direction listed for graded items.
EDIT_CHANGE_GRADED = 3
#: Entries in "Your earlier reasoning", and how much of each text it keeps.
REASONING_ENTRIES = 6
REASONING_TEXT_CHARS = 600
#: The full edits of the latest rejected candidate that fixed more train items than
#: it broke, shown under "Your earlier reasoning" so it can be built on, at most this
#: many characters.
REJECTED_EDITS_CHARS = 4000


def _by_item(records: Sequence[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records or ():
        item = str(record.get("id", ""))
        held = out.get(item)
        if held is None or int(record.get("repeat") or 1) < int(held.get("repeat") or 1):
            out[item] = record
    return out


def _score_of(record: dict[str, Any]) -> float | None:
    value = record.get("score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def edit_change(parent_records: Sequence[dict[str, Any]] | None,
                child_records: Sequence[dict[str, Any]] | None) -> dict[str, Any]:
    """How a child did against its parent, item by item, on the train items both
    were scored on: fixed (parent wrong, child right), broken (the reverse) and the
    largest score changes either way among the rest. Uses only each record's
    ``passed`` and ``score``."""
    parent, child = _by_item(parent_records), _by_item(child_records)
    shared = sorted(set(parent) & set(child))
    fixed = [i for i in shared if not parent[i].get("passed") and child[i].get("passed")]
    broken = [i for i in shared if parent[i].get("passed") and not child[i].get("passed")]
    moved = []
    for item in shared:
        if item in fixed or item in broken:
            continue
        before, after = _score_of(parent[item]), _score_of(child[item])
        if before is not None and after is not None and after != before:
            moved.append((after - before, item))
    up = [item for delta, item in sorted(moved, reverse=True) if delta > 0][:EDIT_CHANGE_GRADED]
    down = [item for delta, item in sorted(moved) if delta < 0][:EDIT_CHANGE_GRADED]
    return {"shared": len(shared), "fixed": fixed, "broken": broken, "up": up, "down": down,
            "deltas": {item: round(delta, 6) for delta, item in moved}}


#: How much of each long field of an edit (a value, a patch, a node) a listing of a
#: candidate's edits keeps; the op and what it names are always shown whole.
EDIT_FIELD_CHARS = 400
#: The fields that say what an edit touches, shown whole before any value.
EDIT_NAMING_FIELDS = ("op", "name", "target", "id", "source", "first", "second", "new_id",
                      "position", "condition")


def describe_edits(edits: Any, field_chars: int = EDIT_FIELD_CHARS) -> str:
    """A candidate's edits as a reader needs them: one line per edit, its op and what it
    names first, whole, then each other field clipped, so no edit of a compound
    proposal is cut away by a long value before it."""
    steps = list(edits) if isinstance(edits, (list, tuple)) else [edits]
    lines = []
    for index, edit in enumerate(steps, 1):
        if not isinstance(edit, dict):
            lines.append(f"{index}. {_clip(json.dumps(edit, default=str), field_chars)}")
            continue
        head = " ".join(f"{key}={json.dumps(edit[key], default=str)}"
                        for key in EDIT_NAMING_FIELDS if key in edit)
        rest = "; ".join(f"{key}: {_clip(json.dumps(value, default=str), field_chars)}"
                         for key, value in edit.items() if key not in EDIT_NAMING_FIELDS)
        lines.append(f"{index}. {head}" + (f" -- {rest}" if rest else ""))
    return "\n".join(lines) if lines else "(none)"


#: How much of one edit "What this run has already tried" shows: a ``set_knob``'s
#: diff against the value it replaced, a patch's ops, a structural edit's fields.
HISTORY_EDIT_CHARS = 600
#: How much of a node's prompt a structural edit in that table shows.
HISTORY_NODE_PROMPT_CHARS = 200
_UNKNOWN = object()


def _bounded(text: str, limit: int = HISTORY_EDIT_CHARS) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text) - limit} more chars)"


def _node_summary(node: Any) -> Any:
    """A node object as the history shows it: id, kind and the head of its prompt."""
    if not isinstance(node, dict):
        return node
    out = {key: node[key] for key in ("id", "kind") if key in node}
    prompt = node.get("prompt_template") or node.get("system_prompt_template")
    if prompt:
        out["prompt"] = _bounded(str(prompt), HISTORY_NODE_PROMPT_CHARS)
    return out


#: Joins a diff's lines in a table cell, where a newline would become a space.
DIFF_LINE_SEPARATOR = " ⏎ "


def _text_diff(old: str, new: str) -> str:
    """The changed lines of a text, unified-diff style without the two file header
    lines, joined with ``DIFF_LINE_SEPARATOR`` so line boundaries survive a table."""
    import difflib

    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                      lineterm="", n=0))[2:]
    return DIFF_LINE_SEPARATOR.join(lines) or "(no change)"


def compact_edits(edits: Any, before: Mapping[str, Any] | None = None) -> str:
    """A candidate's edits (its whole lineage, oldest first) for the history table,
    without whole values: each edit's op and what it names; a text ``set_knob`` as
    old -> new length and a bounded diff against the value it replaced (the previous
    ``set_knob`` of that knob in the lineage, else ``before``, the seed's values); a
    patch's ops; a structural edit's fields with node objects summarised. Each edit is
    at most ``HISTORY_EDIT_CHARS`` characters plus a "(N more chars)" marker. The
    "Parameters you may move" table already shows the current values in full."""
    values: dict[str, Any] = dict(before or {})
    parts: list[str] = []
    for edit in list(edits) if isinstance(edits, (list, tuple)) else [edits]:
        if not isinstance(edit, dict):
            parts.append(_bounded(json.dumps(edit, default=str)))
            continue
        op, name = edit.get("op"), edit.get("name")
        if op == "set_knob" and "value" in edit:
            new, old = edit["value"], values.get(name, _UNKNOWN)
            values[name] = new
            if isinstance(new, str) and isinstance(old, str):
                parts.append(f"set_knob {name}: {len(old)} -> {len(new)} chars; diff: "
                             + _bounded(_text_diff(old, new)))
            elif isinstance(new, str):
                parts.append(f"set_knob {name} = ({len(new)} chars) "
                             + _bounded(json.dumps(new)))
            else:
                parts.append(f"set_knob {name} = " + _bounded(json.dumps(new, default=str)))
        elif op == "set_knob" and "patch" in edit:
            values[name] = _UNKNOWN  # the patched value is not tracked here
            parts.append(f"set_knob {name} patch: "
                         + _bounded(json.dumps(edit["patch"], default=str)))
        else:
            shown = {key: ([_node_summary(n) for n in value] if key in NODE_LIST_FIELDS
                           and isinstance(value, list) else _node_summary(value)
                           if key in NODE_OBJECT_FIELDS else value)
                     for key, value in edit.items()}
            parts.append(_bounded(json.dumps(shown, default=str)))
    return "; ".join(parts) if parts else "(none)"


def _edit_change_lines(pairs: Sequence[dict[str, Any]], budget: int = EDIT_CHANGE_BUDGET) -> list[str]:
    """'What the last edit changed': each recent child of the edited candidate
    against its parent, on train items only."""
    if not pairs:
        return []
    lines = ["## What the last edit changed",
             "The latest children of the candidate you edit (its last batch), or that "
             "candidate itself when it has none yet, each compared with its parent item by "
             "item on the train items both were scored on. The parent's records are from the "
             "child's epoch, so the same seed, when it was scored then (the candidate you edit "
             "is re-scored each epoch); a pair scored with different seeds is marked, and some "
             "of its changes may be run-to-run noise."]
    blocks: list[str] = []
    rest: list[list[Any]] = []
    for pair in pairs:
        head = (f"### Child {pair.get('child')} of {pair.get('parent') or '(none)'}"
                + (f" -- gate: {pair['gate']}" if pair.get("gate") else ""))
        lines += ["", head, "Edits:", describe_edits(pair.get("edit"))]
        if pair.get("same_seed") is False:
            lines.append("Parent and child were scored in different epochs, with different "
                         "seeds: some changes below may be run-to-run noise.")
        if not pair.get("parent"):
            lines.append("This candidate has no parent (it is the seed), so there is nothing "
                         "to compare.")
            continue
        change = edit_change(pair.get("parent_records"), pair.get("child_records"))
        if not change["shared"]:
            lines.append("The two share no scored train item, so there is nothing to compare.")
            continue
        lines.append(f"On {change['shared']} shared train item(s): {len(change['fixed'])} fixed "
                     f"(parent wrong, child right), {len(change['broken'])} broken (parent right, "
                     f"child wrong); largest score rises {change['up'] or 'none'}, largest drops "
                     f"{change['down'] or 'none'}.")
        parent, child = _by_item(pair.get("parent_records")), _by_item(pair.get("child_records"))
        for kind, items in (("fixed", change["fixed"]), ("broken", change["broken"]),
                            ("score rose", change["up"]), ("score dropped", change["down"])):
            for item in items:
                block = (f"#### Item {item} -- {kind}\nParent:\n"
                         + _item_block(parent[item], budget // 2)
                         + "\nChild:\n" + _item_block(child[item], budget // 2))
                blocks.append(block)
                rest.append([item, kind, _outcome(parent[item]), parent[item].get("score"),
                             _outcome(child[item]), child[item].get("score")])
    used = len("\n".join(lines))
    shown = 0
    for block in blocks:
        if used + len(block) + 2 > budget:
            break
        lines += ["", block]
        used += len(block) + 2
        shown += 1
    if shown < len(blocks):
        lines += ["", "The other changed items (their trajectories did not fit):",
                  _table(["item", "change", "parent", "parent score", "child", "child score"],
                         rest[shown:])]
    return lines


def _reasoning_lines(entries: Sequence[dict[str, Any]]) -> list[str]:
    """'Your earlier reasoning': what the optimizer said about recent candidates,
    beside what happened to them."""
    if not entries:
        return []
    lines = ["## Your earlier reasoning and what happened",
             "Recent candidates on the line you edit, and recently refused ones, with what you "
             "predicted when you proposed them and what came of it."]
    for entry in list(entries)[-REASONING_ENTRIES:]:
        outcome = ", ".join(part for part in (
            f"gate {entry['gate']}" if entry.get("gate") else "",
            f"train score {entry.get('score')}",
            (f"{entry['fixed']} fixed / {entry['broken']} broken vs its parent"
             + (" (scored with different seeds)" if entry.get("same_seed") is False else "")
             if entry.get("fixed") is not None else "")) if part)
        lines += ["", f"### {entry.get('id')} (proposed in epoch {entry.get('epoch', '?')}) -- {outcome}",
                  "Edits:", describe_edits(entry.get("edit"), REASONING_TEXT_CHARS // 2),
                  f"Rationale: {_clip(entry.get('rationale'), REASONING_TEXT_CHARS) or '(none)'}",
                  f"Prediction: {_clip(entry.get('prediction'), REASONING_TEXT_CHARS) or '(none)'}",
                  f"Diagnosis that round: {_clip(entry.get('diagnosis'), REASONING_TEXT_CHARS) or '(none)'}"]
    promising = [entry for entry in entries if entry.get("gate") == "rejected"
                 and isinstance(entry.get("fixed"), int) and isinstance(entry.get("broken"), int)
                 and entry["fixed"] > entry["broken"]]
    if promising:
        entry = promising[-1]
        lines += ["", f"### Full edits of {entry.get('id')}, the latest rejected candidate that "
                  f"fixed more train items than it broke ({entry['fixed']} fixed / "
                  f"{entry['broken']} broken)",
                  f"Shown whole up to {REJECTED_EDITS_CHARS} characters, so it can be built on "
                  "or combined with edits to other steps; not to be re-proposed unchanged.",
                  _bounded(describe_edits(entry.get("edit"), REJECTED_EDITS_CHARS),
                           REJECTED_EDITS_CHARS)]
    return lines


def _rules_text() -> str:
    from refactory_dsh.nn.validate import rules_text

    return rules_text()


#: Trace keys the request leaves out: the leaderboard is "Recent results" already,
#: and measurements carry per-trial worker receipts and run-directory paths -- noise to
#: an optimizer, which reads their aggregates (``measurement_summary``) instead.
TRACE_OMITTED_KEYS = ("leaderboard", "measurements", "workers", "workdir", "trials")
#: The longest key or text a trace row shows; a text knob's value used as a key
#: (``knob_attribution``) is shown by its head and length.
TRACE_TEXT_CHARS = 160


def _trace_rows(traces: Sequence[dict[str, Any]]) -> list[tuple[str, Any]]:
    """The traces as the request shows them: no receipts, no paths, short texts."""

    def short(text: str) -> str:
        return text if len(text) <= TRACE_TEXT_CHARS else (
            f"{text[:TRACE_TEXT_CHARS - 40]}… ({len(text)} characters)")

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {short(str(k)): clean(v) for k, v in value.items()
                    if k not in TRACE_OMITTED_KEYS}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            return short(value)
        return value

    return [(key, clean(value)) for key, value in _flatten(traces)
            if key not in TRACE_OMITTED_KEYS]


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

    #: Where an optimizer that calls a model records each call (the loop sets it to
    #: ``<run root>/optimizer_trace.jsonl``); ``None`` records nothing.
    trace_path: Any = None

    #: The most edits one proposal may bundle (``Proposal.edits``); more is refused
    #: with ``E-too-many-edits``. The loop sets it from ``max_edits_per_proposal``.
    max_edits: int = DEFAULT_MAX_EDITS

    #: Whether an edit may add a code node (`refactory_dsh.snippet`); one that does
    #: otherwise is refused with ``E-code-nodes``. The loop sets it from
    #: ``allow_code_nodes``.
    allow_code_nodes: bool = True

    #: Values only the grader may see (the train items' grader-only fields), which no
    #: code node's code may contain (``E-secret``), or a function returning them (called
    #: only for a candidate with code). The loop sets it from its Task.
    grader_secrets: Any = frozenset()

    #: Whether ``propose`` asks a model, so its calls must run on a route without tools
    #: (``Loop._check_optimizer_route``) and a strategy may hand it the round's sampling.
    uses_model: bool = False

    def choose_parent(
        self, ctx: "ProposalContext", parents: Sequence["Candidate"]
    ) -> "Candidate | None":
        """The candidate this step edits, when the optimizer picks it itself.

        Asked once per optimizer step, before the request is built, so the request
        describes the candidate the proposals are applied to. ``None`` (the default)
        leaves the choice to the loop (``Loop.select_parents``).
        """
        return None

    def bind_model(self, model: Any, *, workdir: Any = None) -> None:
        """Model-using optimizers override this to inherit the run's resources."""

    def record_call(self, **fields: Any) -> None:
        """Append one model call to ``trace_path`` as a JSON line, as it happens.

        Appended, never rewritten, so a resumed run keeps what it recorded before.
        """
        if self.trace_path is None:
            return
        import threading
        import time
        import warnings
        from pathlib import Path

        lock = self.__dict__.setdefault("_trace_lock", threading.Lock())
        try:
            path = Path(self.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with lock, path.open("a") as handle:
                handle.write(json.dumps({"at": time.time(), "optimizer": self.name,
                                         "round": self.proposal_rounds, **fields},
                                        default=str) + "\n")
        except Exception as exc:  # a trace that cannot be written never costs a proposal
            warnings.warn(f"{self.name}: could not record the optimizer call in "
                          f"{self.trace_path}: {type(exc).__name__}: {exc}",
                          RuntimeWarning, stacklevel=2)

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
                attempt=attempt,
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
                        edit=proposal.edit,  # {"edits": [...]} for a compound proposal
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
        protected: Sequence[str] = (),
        allow_new_node_tools: bool = False,
        graders: Sequence[str] = (),
    ) -> Candidate | list[str]:
        """Apply and validate one proposal: a candidate, or the reasons it failed.

        Public because seeding a population and proposing an update have to be the
        same test: a candidate that could not be proposed could not be seeded
        either, and two code paths would eventually disagree about that.

        ``protected`` is the mode's ``protected_nodes()``: an edit that removes,
        rewires or replaces one is refused with ``E-protected``. An agent node the
        search added may run with tools only with ``allow_new_node_tools``
        (``E-new-node-tools``). ``graders`` is the mode's ``grader_nodes()``: with the
        protected nodes, the only nodes besides the data nodes that may consume
        grader-derived data (V17).
        """
        steps = proposal.steps
        refusal = proposal_refusal(proposal, max_edits=self.max_edits)
        if refusal:
            return refusal
        frozen = set(frozen or ())
        # Knobs a protected or grader node reads or binds hold for every edit: a search
        # that could rewrite them could rewrite the grader side's behaviour (a protected
        # agent's prompt copying answers into the scored artifact).
        guarded = _grader_side_knobs(parent.state, (*protected, *graders))
        # A compound proposal's edits go through the same checks one by one, each
        # against the state the edits before it left; the combined state is then held
        # to every rule below once, as one candidate.
        state = parent.state
        edit_reports: list[dict[str, Any] | None] = []
        for index, edit in enumerate(steps):
            if edit.get("op") == "set_knob" and edit.get("name") in frozen:
                return [_at(f"E-frozen: knob {edit.get('name')!r} is frozen for this run",
                            index, steps)]
            if edit.get("op") in ("set_knob", "remove_knob") and edit.get("name") in guarded:
                return [_at(f"E-frozen: knob {edit.get('name')!r} is read by "
                            f"{guarded[edit.get('name')]!r}, a protected or grading node, so no "
                            "edit may change it", index, steps)]
            # An operation that is not in the catalogue is not a failed precondition
            # (`proposal_refusal` refuses it): the two point at different fixes.
            before_step = state
            reason = nn_edits.precheck(edit, state)
            if reason:
                return [_at(f"E-precondition: {reason}", index, steps)]
            reports: list[dict[str, Any]] = []
            try:
                state = nn_edits.apply(edit, state, edit_reports=reports, protected=protected)
            except nn_edits.EditError as exc:
                return [_at(f"{exc.code}: {exc}", index, steps)]
            except Exception as exc:  # an operator bug is reported, not swallowed
                return [_at(f"E-internal: {type(exc).__name__}: {exc}", index, steps)]
            if len(steps) > 1 and candidate_id(state) == candidate_id(before_step):
                # An edit of a bundle that changes nothing would be recorded as part of
                # the idea while doing nothing; said back rather than kept.
                return [_at(f"E-noop: {edit.get('op')} leaves the state as the edits before "
                            "it left it; drop it or change it", index, steps)]
            edit_reports.append(reports[0] if reports else None)

        # A bundle that changes nothing at all is refused below as one no-op.
        dead = (_dead_edit(steps, parent.state, state, protected)
                if candidate_id(state) != parent.id else None)
        if dead is not None:
            return [_at(f"E-noop: {steps[dead].get('op')} has no effect on the result: the "
                        "edits after it undo or overwrite it; drop it", dead, steps)]

        # ── tool-policy escalation check (anchored to the original seed) ──
        escalation = _tool_policy_escalated(seed_state if seed_state is not None else parent.state, state)
        if escalation:
            return escalation
        tooled = _new_node_tools(seed_state if seed_state is not None else parent.state, state,
                                 allow_new_node_tools)
        if tooled:
            return tooled
        coded = _code_node_refusals(seed_state if seed_state is not None else parent.state, state,
                                    bool(getattr(self, "allow_code_nodes", True)))
        if coded:
            return coded
        leaked = _secret_refusals(state, getattr(self, "grader_secrets", ()))
        if leaked:
            return leaked

        fixed = frozen | set(guarded) | {name for name, value in parent.state.get("knobs", {}).items()
                                         if (value.get("spec") or {}).get("frozen")}
        for name in fixed:
            if state.get("knobs", {}).get(name) != parent.state.get("knobs", {}).get(name):
                return [f"E-frozen: knob {name!r} must retain its value and declaration"]

        if candidate_id(state) == parent.id:
            # The edit is legal and changes nothing: setting a knob to the value it
            # already holds, or re-adding a node that is already there. Accepting it
            # would spend a batch slot re-evaluating a point already scored, and the
            # search would look productive while making no progress.
            what = (str(steps[0].get("op")) if len(steps) == 1
                    else f"the {len(steps)} edits together")
            return [
                f"E-noop: {what} leaves the state identical to its "
                "parent, so it would re-evaluate a candidate already scored"
            ]

        if candidate_id(state) in set(seen or ()):
            # Without this the proposal passes every gate and is then dropped
            # silently by the loop, which loses the batch slot and makes the run
            # report a batch it did not actually try.
            return [
                f"E-duplicate: this state already exists in the search as candidate "
                f"{candidate_id(state)}, so evaluating it again would spend a batch slot on "
                "a point already scored; propose a different change"
            ]

        issues = validate_state(state, invariants=invariants, resolve_calls=resolve_calls,
                                grader_side=(*protected, *graders), graders=tuple(graders))
        if issues:
            return [str(issue) for issue in issues]
        unlowerable = _lowering_refusal(parent.state, state)
        if unlowerable:
            return [unlowerable]
        # Build the per-edit report list parallel to edits.  Parent reports
        # are carried forward; one report (or None) per applied edit is appended.
        parent_reports = list(parent.edit_reports) if parent.edit_reports else [
            None for _ in parent.edits
        ]
        return Candidate.from_state(
            state,
            parent=parent.id,
            edits=[*parent.edits, *steps],
            edit_reports=[*parent_reports, *edit_reports],
            step_edits=len(steps),
        )
