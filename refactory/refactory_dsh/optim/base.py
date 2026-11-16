"""Optimizers propose updates to a mode's state.

An optimizer is one method — ``propose`` — plus a loop the base class owns. The
loop is where the evaluation budget is protected: a proposal is expanded and
validated for microseconds before it can cost an inner run, and a proposal that
does not survive is regenerated with the machine-readable reason why. That turns
wasted *budget* into wasted *tokens*, which is the trade worth making, because an
invalid candidate cannot run and therefore teaches the search nothing.

The loop regenerates; it repairs only mechanics. An edit fixed up silently would be
scored as something the optimizer never proposed, and the gradient would be a
description of the repair rather than of the idea. So a fix is made only where what
the proposal meant is unambiguous, and it is said: the candidate's edits are the
repaired ones and its ``repairs`` name each fix (`optim.repair`). A refusal about how a
proposal is written may also go back to the model in a slim repair call that shows it
the refused proposal; one about the idea never does.
"""

from __future__ import annotations

from concurrent.futures import CancelledError

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import validate_state
from . import checks, repair
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
    (``LoopConfig.allow_new_node_tools``), except for the tools a step of the seed already
    runs with, in a fresh workspace of the new node's own (a copy of a tool step, as the
    core operators make one). A node whose id the seed has is held to the seed's own policy
    instead (`_tool_policy_escalated`).
    """
    if allowed:
        return []
    seed_nodes = (seed_state.get("graph") or {}).get("nodes") or {}
    # The tools a step of the seed already runs with: a copy of such a step may keep them
    # when it works in a fresh workspace of its own (the user's decision, recorded in
    # `nn.coverage.DECISIONS`), so it never works where the step it copies works.
    seed_tools = {(node.get("tool_policy"), tuple(node.get("tools") or ()))
                  for node in seed_nodes.values() if node.get("kind") == "agent"
                  and (node.get("tool_policy") != "none" or node.get("tools"))}
    reasons = []
    for node_id, node in sorted(((candidate_state.get("graph") or {}).get("nodes") or {}).items()):
        if node_id in seed_nodes or node.get("kind") != "agent":
            continue
        if (node.get("tool_policy"), tuple(node.get("tools") or ())) in seed_tools \
                and _fresh_workspace(node):
            continue
        if node.get("tool_policy") != "none" or node.get("tools"):
            reasons.append(
                f"E-new-node-tools: new agent node {node_id!r} would run with tools; a node "
                "an edit adds runs without tools (omit tool_policy, or set it to 'none') "
                "unless the run allows tools for new nodes (allow_new_node_tools)")
    return reasons


def _fresh_workspace(node: Mapping[str, Any]) -> bool:
    """Whether a node state works in a workspace of its own, made fresh for each dispatch
    (``scratch``, or a ``clone`` of the run's base)."""
    from refactory_dsh.nn.graph import workspace_declaration

    try:
        workspace = workspace_declaration(node.get("workspace"))
    except ValueError:
        return False
    return workspace is not None and (workspace["kind"] == "scratch"
                                      or workspace.get("of") == "base")


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
                        allowed: bool, sandbox: tuple[bool, str] | None = None) -> list[str]:
    """``E-code-nodes`` for each code node of the candidate (`refactory_dsh.snippet`) that
    the seed does not have as the same code node -- same id, kind, callable and code --
    when the run does not allow code nodes (``LoopConfig.allow_code_nodes``) or this
    machine cannot sandbox them (`snippet.available`). Compared by content, not by id: a
    proposal that removes a node and adds a code node under its id, or rewrites a seed
    code node's code, is a new code node. The seed's own code nodes, unchanged, are the
    author's and are not held to it. ``sandbox`` is the machine's answer the request
    was written with (``ProposalContext.sandbox``), so the request's offer and this
    refusal never disagree; asked afresh only when there is none."""
    from refactory_dsh import snippet

    seed = _code_signatures(seed_state)
    added = sorted(node_id for node_id, signature in _code_signatures(candidate_state).items()
                   if seed.get(node_id) != signature)
    if not added:
        return []
    if not allowed:
        why = "this run does not allow code nodes (allow_code_nodes is off)"
    else:
        ok, reason = sandbox if sandbox is not None else snippet.available()
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
#: How the cap counts, as a refusal and the bundling instructions say it.
COUNTING_RULE = ("A structural edit and the edits that only connect it count as one change: "
                 "the new step's own prompt and settings, its direct neighbours' prompt "
                 "lines that name it or what it writes, and edges between it and them. "
                 "Every other edit counts as one more.")


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


def _dead_edit(steps: Sequence[dict[str, Any]], start: dict[str, Any],
               result: dict[str, Any]) -> int | None:
    """The index of a compound proposal's first edit the result does not depend on: the
    other edits, applied without it, reach the same state (a later edit overwrote it or
    undid it). ``None`` for a single edit, or when every edit counts. Protected nodes need
    no check here: a subset reaching the result's id reaches the result's graph and
    knobs, which already passed it."""
    if len(steps) < 2:
        return None
    target = candidate_id(result)
    for index in range(len(steps)):
        state = start
        try:
            for other, edit in enumerate(steps):
                if other != index:
                    state = nn_edits.apply(edit, state)
        except Exception:  # noqa: BLE001 - the others need this one: it counts
            continue
        if candidate_id(state) == target:
            return index
    return None


def _protected_culprit(steps: Sequence[dict[str, Any]], start: dict[str, Any],
                       prefixes: Sequence[dict[str, Any]], protected: Sequence[str]) -> str:
    """For a bundle whose end state breaks a protected node: the edit from which every
    later state breaks it too (the one that did it and nothing after repaired it), named
    beside the bundle. Empty for a single edit."""
    if len(steps) < 2:
        return ""
    culprit = len(prefixes) - 1
    while culprit > 0 and nn_edits.protected_violation(start, prefixes[culprit - 1], protected):
        culprit -= 1
    op = steps[culprit].get("op") if isinstance(steps[culprit], dict) else "?"
    return (f" [the {len(steps)} edits together; from edit {culprit + 1} ({op}) on, "
            "no later edit restores it]")


#: The key of the stand-in edit an answer's unreadable proposal entry becomes
#: (``unreadable_proposal``): its value says why, and ``proposal_refusal`` refuses it
#: with ``E-proposal-shape``, so the entry is counted and shown back, never dropped.
UNREADABLE_PROPOSAL = "unreadable_proposal"


def unreadable_proposal(raw: Any, why: str) -> Proposal:
    """A proposal entry of a model's answer that carries no usable edit, as a proposal
    the loop refuses with ``why`` (``E-proposal-shape``) like any other refusal."""
    entry = _clip(json.dumps(raw, default=str), 300)
    rationale = str(raw.get("rationale", "") or "") if isinstance(raw, dict) else ""
    return Proposal(edit={UNREADABLE_PROPOSAL: f"{why}; the entry was {entry}"},
                    rationale=rationale)


def proposal_refusal(proposal: Proposal, *, permitted: Sequence[str] | None = None,
                     max_edits: int | None = None,
                     parent_state: dict[str, Any] | None = None) -> list[str]:
    """Why a proposal is refused before it becomes a candidate; empty when it is not.

    An answer's entry that carried no edit (``E-proposal-shape``), too many changes
    (``E-too-many-edits``), an edit that is not an object (``E-op``), an op outside
    the catalogue, and, given ``permitted``, an op outside this run's permitted edits
    (``E-policy``). Each refusal of a compound proposal names the edit (its index,
    from 1) it is about.

    Given the ``parent_state`` the proposal applies to, its edits are counted by
    logical change (`optim.edit_count`): a structural edit and the edits that only
    connect it count as one. Without it every edit counts.
    """
    steps = proposal.steps
    if (len(steps) == 1 and isinstance(steps[0], dict) and "op" not in steps[0]
            and UNREADABLE_PROPOSAL in steps[0]):
        return [f"E-proposal-shape: {steps[0][UNREADABLE_PROPOSAL]}. Write each proposal "
                'as {"edit": {"op": ...}} or {"edits": [{"op": ...}, ...]}, with '
                "rationale, prediction and confidence beside it"]
    if max_edits is not None and len(steps) > max(1, int(max_edits)):
        limit = max(1, int(max_edits))
        # A limit of one asks for single edits only; a bundle counts by logical change.
        if (limit == 1 or parent_state is None
                or not all(isinstance(step, dict) for step in steps)):
            return [f"E-too-many-edits: the proposal bundles {len(steps)} edits and the limit "
                    f"is {limit} edits per proposal; split it, or "
                    "keep the edits the fix needs together"]
        from .edit_count import count

        made = count(steps, parent_state)["logical"]
        if made > limit:
            return [f"E-too-many-edits: the proposal makes {made} changes ({len(steps)} edits) "
                    f"and the limit is {limit} changes per proposal. {COUNTING_RULE} Split "
                    "it, or keep the edits the fix needs together"]
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


def _patch_targets_missing(edit: dict[str, Any], reports: Sequence[dict[str, Any]]) -> str:
    """``E-patch`` when a ``set_knob`` patch entry's target text is not in the knob, or is
    empty where its action needs one, and the patch did not say what to do then
    (``config.on_missing``): skipping it would score a knob the proposal did not write.
    Empty otherwise."""
    if edit.get("op") != "set_knob" or "patch" not in edit or not reports:
        return ""
    if "on_missing" in dict(edit.get("config") or {}):
        return ""
    for outcome in (reports[0] or {}).get("outcomes") or ():
        if (str(outcome.get("status", "")).startswith("skipped") and not outcome.get("target")
                and outcome.get("kind") in ("replace", "delete", "insert_after", "insert_before")):
            return (f"E-patch: knob {edit.get('name')!r}: patch entry {int(outcome.get('index', 0)) + 1} "
                    f"({outcome.get('kind')}) has no target text; give the exact text it acts on "
                    "(only append needs none)")
        if str(outcome.get("status", "")).endswith("_target_not_found"):
            return (f"E-patch: knob {edit.get('name')!r}: patch entry {int(outcome.get('index', 0)) + 1} "
                    f"({outcome.get('kind')}) has a target that is not in the knob's text: "
                    f"{str(outcome.get('target') or '')[:200]!r}; quote the target exactly as the "
                    "knob holds it now")
    return ""


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

    def codes(self) -> list[str]:
        """The rules that were broken, which is how a refusal histogram is built."""
        return sorted(
            {reason.split(" ", 1)[0].rstrip(":") for reason in self.reasons if reason}
        )


#: How many refused proposals a request shows as written (``ProposalContext.refused``).
REFUSED_SHOWN = 2
#: The characters of free text (prompts, code) one shown refused proposal keeps; its
#: wiring is always whole (`describe_edits`).
REFUSED_EDITS_CHARS = 4000


def refused_lines(refused: Sequence[Mapping[str, Any]]) -> list[str]:
    """The latest refused proposals as written, each with its reasons, for a retry."""
    shown = [entry for entry in refused if entry.get("edit")][-REFUSED_SHOWN:]
    if not shown:
        return []
    lines = ["", "The latest refused proposal(s) as written, so the fix keeps what each "
                 "meant (wiring whole, long texts shortened):"]
    for entry in shown:
        edit = entry.get("edit")
        steps = edit.get("edits") if isinstance(edit, dict) and "op" not in edit else edit
        lines += ["", "Refused for: " + " | ".join(str(r) for r in entry.get("reasons") or ()),
                  describe_edits(steps, text_budget=REFUSED_EDITS_CHARS)]
    return lines


def refused_entries(rejections: Sequence["Rejection"]) -> list[dict[str, Any]]:
    """The latest rejections a slim repair did not fix, as ``ProposalContext.refused``
    entries (an unreadable answer has no edit to show and is left out)."""
    out = [{"edit": r.edit, "reasons": list(r.reasons)} for r in rejections
           if not r.repaired and isinstance(r.edit, dict) and UNREADABLE_PROPOSAL not in r.edit]
    return out[-REFUSED_SHOWN:]


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
    #: The mean score of every candidate scored so far (``Loop.loss_step``), over
    #: ``mean_over`` of them; not one batch's mean.
    mean_score: float | None = None
    mean_over: int = 0
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
    #: The mode's grader nodes (``Mode.grader_nodes``), which the pipeline view reads as
    #: it does in the loop's records.
    graders: list[str] = field(default_factory=list)
    #: The loop's verdict on one edit to the edited candidate (`Run.mechanics`: a
    #: candidate, or its refusals), for the request's operator table; ``None`` leaves
    #: the table out.
    edit_check: Callable[[dict[str, Any]], Any] | None = field(default=None, repr=False,
                                                                 compare=False)
    #: What the last ``render`` left out of an item's texts (each: item, step, chars), for
    #: the run's own record; the request says only "[... N chars omitted]".
    omissions: list[dict[str, Any]] = field(default_factory=list, repr=False, compare=False)
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
    #: ``MAX_EDITS_PER_PROPOSAL`` in a run). 1 asks for single edits only, and the
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
    #: How the validation gate selects (``Loop.selection_mode``): ``single``, ``pool``,
    #: or empty without the gate. Under ``pool`` the gate note says what the pool is.
    selection: str = ""
    #: Under ``pool``, whether a population parent picker (``archive``, ``evolutionary``,
    #: ``islands``, ``map_elites``) chooses the parent from every judged candidate
    #: rather than from the pool; the gate note says which (``_gate_note``).
    population_parents: bool = False
    #: What is there before each node of the edited graph, on every path that reaches
    #: it (`available_before`): ``{node: {"facts": [...], "files": [...], "made_by":
    #: {name: [step ids]}, "task_inputs": [...]}}``. Rendered under "The pipeline you may
    #: edit", so a prompt's ``{{name}}`` names a fact that exists there (rule V15) rather
    #: than a file name, made by the step meant; empty says nothing.
    available: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Wiring of the edited graph that runs but is likely not what was meant
    #: (`nn.validate.wiring_warnings`), said beside its edges; never a refusal.
    wiring_warnings: list[str] = field(default_factory=list)
    #: The machine's code-node sandbox answer (`snippet.available`) this request was
    #: written with: ``code_nodes`` offers by it and the refusal of a code node
    #: (``E-code-nodes``) refuses by it, so the two agree within a request; ``None``
    #: when not taken (the run does not allow code nodes). Never rendered.
    sandbox: tuple[bool, str] | None = None
    #: Whether a person answers gates while this run's candidates run. An outer loop's
    #: candidates run headless, so the default leaves the person-answered gate kind out
    #: of the node schemas (`node_schema.render_schema`).
    person_gates: bool = False
    #: Every evaluated candidate of the run with the kind of change it made, its train
    #: fixed/broken counts against its parent and the gate's word (``Loop._ledger``,
    #: ``LoopConfig.run_ledger``), for "Every change so far, by kind"; empty says nothing.
    ledger: list[dict[str, Any]] = field(default_factory=list)
    #: The loop's memory note (`optim.memory`; ``LoopConfig.lessons``): ``{"text",
    #: "epoch", ...}`` and, when the epoch's pass failed, ``not_updated``; for
    #: "Optimizer memory" here and in every analyst and merge call. Empty says nothing.
    memory: dict[str, Any] = field(default_factory=dict)
    #: The steps the last round's candidates added, with their train evidence
    #: (``feedback.step_evidence``; ``LoopConfig.step_review``) and, once a model
    #: optimizer has reviewed them, each step's ``review`` or ``review_error``, for "Did
    #: the added steps contribute?".
    added_steps: list[dict[str, Any]] = field(default_factory=list)
    #: Whether "How nodes compose" says what makes a new step pay off
    #: (``LoopConfig.structure_guidance``; ``feedback.STRUCTURE_GUIDANCE``).
    structure_guidance: bool = False
    #: The analyst stage's settings (``LoopConfig.analysts``; `optim.analysts.settings`):
    #: ``{"group_size", "merge_batch_size", "max_batches"}``; empty is off. A model
    #: proposer then runs failure and success analysts over ``item_records`` and merges
    #: their findings before it writes the proposal. Never rendered itself.
    analysts: dict[str, Any] = field(default_factory=dict)
    #: The round's analysis (`AutoresearchOptimizer._analyse`): the batches, the merges
    #: and the ranked ``findings``, for "What the analysts found, ranked"; ``None`` when
    #: no analyst ran.
    analysis: dict[str, Any] | None = None
    #: The integrity judge's verdicts on this run's proposals (``Loop._integrity``,
    #: ``LoopConfig.integrity_judge``), for "Integrity review of your proposals"; empty
    #: (the judge off) says nothing.
    integrity: list[dict[str, Any]] = field(default_factory=list)
    #: The latest refused proposals as written (``{"edit", "reasons"}``, at most
    #: ``REFUSED_SHOWN``), shown under "Why your last proposals were refused" so a retry
    #: fixes the proposal it wrote rather than writing one from scratch.
    refused: list[dict[str, Any]] = field(default_factory=list)
    #: Whether a new step may use tools in this run (``LoopConfig.allow_new_node_tools``);
    #: only then do the compositions show a tool step.
    new_node_tools: bool = False
    #: Time, in numbers, for "Time per item" (`time_lines`): ``step_limit_s`` (how long one
    #: model step may run), ``per_item_s`` (the edited candidate's typical time per item),
    #: ``items`` (distinct train items), ``runs`` (their runs, counting repeats),
    #: ``timeouts`` (runs that timed out) and ``parallelism``; empty says nothing.
    time_facts: dict[str, Any] = field(default_factory=dict)

    def _tried_lines(self) -> list[str]:
        """What this run has tried and how each attempt did: the recent results, the
        traces' numbers, the last edit's children and the reasoning behind recent
        candidates (each with what its edit built), every candidate so far, the run-wide
        feedback, earlier runs and the latest refusals."""
        lines = [
            "## Recent results",
            *_gate_note(self.leaderboard, self.history, val_items=self.val_items,
                        selection=self.selection, population=self.population_parents),
            *_cost_note(self.parsimony),
            _table(["id", "parent", "score", "structural", "edits", *_gate_column(self.leaderboard),
                    *(["cost per item"] if self.parsimony else [])],
                   [[c["id"], c.get("parent"), c.get("score"), c.get("structural"),
                     compact_edits(c.get("edits"), self.seed_knob_values),
                     *([c.get("gate", "")] if _gate_column(self.leaderboard) else []),
                     *([_cost_cell(c.get("cost_per_item"))] if self.parsimony else [])]
                    for c in self.leaderboard]),
            "",
            "## What this run has already tried",
            (
                f"{len(self.history)} candidate(s), best "
                f"{max((c.get('score') or 0.0) for c in self.history):.6g}"
                if self.history and not _gate_column(self.history)
                else f"{len(self.history)} candidate(s); {_current_line(self.history)}"
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
            "",
            "## Evidence from the traces",
            _table(["what", "value"], _evidence_rows(_trace_rows(
                self.traces, omit=UNSCORED_TRACE_KEYS
                if self.parsimony == 0 and self.score_is_item_mean else ()),
                [] if self.history_truncated else self.history)),
        ]
        if self.edit_pairs:
            lines += ["", *_edit_change_lines(self.edit_pairs, omissions=self.omissions)]
        if self.reasoning:
            above = [str(p.get("child")) for p in self.edit_pairs or ()]
            lines += ["", *_reasoning_lines(self.reasoning, above)]
        lines += _feedback_sections(self)
        if self.prior_runs:
            lines += [
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
            lines += [
                "",
                "## Why your last proposals were refused",
                "A proposal that breaks one of these is not evaluated at all. It costs "
                "nothing but the answer, so fix the stated reason rather than guessing.",
                _table(["reasons"], [[r] for r in self.rejections[-12:]]),
                *refused_lines(self.refused),
            ]
        return lines

    def render(self) -> str:
        """The request, as text, for a model optimizer: the score, the pipeline (once, as
        the view), what this run has tried, what the analysts found, the edits it may
        make, the items, then the reference, and the answer's form last. Fills
        ``omissions`` with any text it had to leave out of an item."""
        self.omissions.clear()
        sections = [
            f"# Update request — generation {self.generation}, epoch {self.epoch}",
            "",
            f"Mode: `{self.mode_name}`. "
            + (f"Best score so far: {self.best_score}. " if not self.selection else
               f"Train score of the candidate the validation gate ranks first: "
               f"{self.best_score}. ")
            + (f"Mean over the {self.mean_over} candidate(s) scored so far: {self.mean_score}."
             if self.mean_over else f"Mean score so far: {self.mean_score}."),
            "",
            *objective_lines(self.parsimony, item_mean=self.score_is_item_mean,
                             gated=bool(self.selection)),
            *time_lines(self.time_facts),
            *([self.steering, ""] if self.steering.strip() else []),
            *_strategy_section(self.strategy, self.strategy_name),
            # 1-2: the score (above), then the pipeline once, in the view's lines.
            "## The pipeline you may edit",
            _levers_line(self.catalogue),
            *([f"Note: {self.graph_note}"] if self.graph_note else []),
            *([f"Protected nodes: {', '.join(f'`{n}`' for n in self.protected)}. An edit may "
               "not remove them, change their declarations or out-edges, move a knob they "
               "read, or let a run finish without them; a new node may go in front of one."]
              if self.protected else []),
            *pipeline_view_lines(self.node_states, self.edges, self.knobs, self.protected,
                                 self.graders),
            *(["Wiring warnings (the graph runs, but likely not as meant; not refusals):",
                *(f"- {w}" for w in self.wiring_warnings)] if self.wiring_warnings else []),
            REFERENCE_POINTER,
            "",
            # 3: what this run has tried, and how each attempt did.
            *self._tried_lines(),
        ]
        # 4: what the analysts found, failures and successes, and where it fails by step.
        if self.analysis is not None:
            from .analysts import section_lines
            sections += ["", *section_lines(self.analysis)]
        sections += ["", *_step_map(self)]
        # 5: the edits it may make, with which apply where, and the steps' settings.
        sections += [
            "## The operations you may use",
            _table(["op", "kind", "macro", "fields", "description"],
                   [[c["op"], c["kind"], c["macro"], _edit_fields(c["fields"]), c["description"]]
                    for c in self.catalogue]),
            "",
            *composition_section(self.catalogue, self.node_states, self.edges,
                                 protected=self.protected, code_nodes=self.code_nodes,
                                 text_knobs=[k.get("name") for k in self.knobs
                                             if k.get("kind") == "str"],
                                 structure_guidance=self.structure_guidance,
                                 tools=self.new_node_tools, rotation=self.generation,
                                 knob_values={k.get("name"): k.get("value") for k in self.knobs}),
            *operator_fit_section(self.catalogue, self.node_states, self.edges, self.knobs,
                                  self.edit_check),
            *_settings_lines(self.knobs),
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
            *_blocks_section(self.insertable, self.catalogue),
        ]
        # 6: the items themselves, chosen rather than clipped.
        if self.item_records:
            sections += ["", *_item_evidence(self.item_records)]
            sections += ["", *_trajectories(
                self.item_records, budget=self.evidence_budget,
                findings=self.reader_findings, failures=self.reader_failures,
                analysed=(None if self.analysis is None
                          else list(self.analysis.get("findings") or ())),
                omissions=self.omissions)]
        # 7: the reference, last: every node in full, the schemas, the validator's rules.
        sections += [
            "",
            REFERENCE_HEADING,
            "The pipeline's nodes and edges as the graph declares them, what each node has "
            "before it runs, each step in full with the node schema, and the rules the "
            "validator checks.",
            "Nodes:",
            _table(["id", "kind", "call", "knobs", "notes"],
                   [[n["id"], n["kind"], n.get("call", ""), n.get("knobs", []),
                     n.get("notes", "")] for n in self.nodes]),
            "Edges:",
            _table(["source", "target", "condition"],
                   [[e["source"], e["target"], e.get("condition")] for e in self.edges]),
            *_available_section(self.available, code_nodes=self.code_nodes),
            "",
            *_node_reference(self),
            "## The rules the validator checks",
            "Every proposal is checked against these rules before anything runs; they are "
            "the validator's own, generated from it. One that breaks a rule comes back with "
            "the rule's code and the reason, so it can be fixed in a later answer, and it "
            "costs nothing but the answer.",
            _rules_text(),
        ]
        sections += ["", "## Answer", answer_instructions(
            self.catalogue, parsimony=self.parsimony,
            items=_item_count(self.item_records) if self.score_is_item_mean else 0,
            code_nodes=self.code_nodes)]
        if self.max_edits > 1:
            sections += ["", bundle_instructions(self.max_edits, self.catalogue,
                                                 proposals=self.proposals)]
        if self.edit_levels:
            sections += ["", *levels_instructions(self.edit_levels, code_nodes=self.code_nodes)]
        return "\n".join(sections)


def _levers_line(catalogue: Sequence[dict[str, Any]]) -> str:
    """What the request lets the optimizer change, said before anything else about the
    pipeline: its structure and its steps' settings, as equal levers when the run permits
    structural operations; the settings only when it does not."""
    if not structural_ops(catalogue):
        return ("This run permits changes to the steps' settings only (their prompts and "
                "parameters, listed after the operations); the structure below stays as it is.")
    return ("You may change the pipeline's structure -- its steps and how they connect, "
            "with the operations below -- and each step's settings -- its prompts and "
            "parameters, listed after the operations. Both are levers of the same standing: "
            "read where the pipeline fails first, then choose by the cause the evidence shows.")


def _exact_in_table(value: Any) -> bool:
    """Whether a table cell shows ``value`` as it is: a cell puts a text on one line and
    escapes ``|`` (`_cell`), so a text with either is shown whole below the table."""
    return not (isinstance(value, str) and ("\n" in value or "|" in value))


def _fence(text: str) -> str:
    """A code fence longer than any run of backticks in ``text``."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _settings_lines(knobs: Sequence[dict[str, Any]]) -> list[str]:
    """"Each step's settings": every knob in a table; a text the table cannot show as it
    is (a line break or a ``|``) is shown there by reference and whole below it, fenced,
    so a patch can quote its target text exactly."""
    whole = [k for k in knobs if not _exact_in_table(k.get("value"))]
    lines = ["## Each step's settings",
             "The prompts and parameters the steps read, which `set_knob` moves (a new step "
             "brings its own). A patch quotes its target text exactly: a value in the table "
             "is exact as shown"
             + ("; a text with line breaks or `|` is shown whole after the table." if whole
                else "."),
             _table(["name", "value", "kind", "values", "range", "frozen", "reachable",
                     "description"],
                    [[k["name"], k["value"] if _exact_in_table(k.get("value"))
                      else f"(shown whole below, {len(k['value'])} characters)",
                      k["kind"], k.get("values"), k.get("range"),
                      k["frozen"], k.get("reachable", True), k.get("description", "")]
                     for k in knobs])]
    for knob in whole:
        fence = _fence(knob["value"])
        lines += [f"`{knob['name']}`:", fence, knob["value"], fence]
    return [*lines, ""]


def _step_map(ctx: "ProposalContext") -> list[str]:
    """"Where the pipeline fails, step by step" (`analysts.step_map_lines`): from the
    round's analysis, or, without one, from the readers' failure patterns (where each
    starts); nothing when neither ran."""
    from .analysts import SHOWN_FINDINGS, reader_failures, step_map_lines

    structural = bool(structural_ops(ctx.catalogue))
    if ctx.analysis:
        return step_map_lines(list(ctx.analysis.get("findings") or ()), ctx.nodes, ctx.edges,
                              source="analysts", shown=SHOWN_FINDINGS,
                              code_nodes=ctx.code_nodes, structural=structural)
    if ctx.reader_findings:
        return step_map_lines(reader_failures(ctx.reader_findings), ctx.nodes, ctx.edges,
                              source="readers", code_nodes=ctx.code_nodes,
                              structural=structural)
    return []


def _node_reference(ctx: "ProposalContext") -> list[str]:
    """Each node of the edited graph in full and the node schemas (`_node_specs`), under
    their own heading; nothing when the request carries no node states."""
    specs = _node_specs(ctx.node_states, text_chars=ctx.node_spec_text_chars,
                        section_chars=ctx.node_spec_section_chars,
                        knob_values={k.get("name"): k.get("value") for k in ctx.knobs},
                        person_gates=ctx.person_gates)
    return ["## Each step in full, and the node schema", *specs, ""] if specs else []


def _count(value: Any) -> int:
    """A count from a trace, 0 when it is not a whole number."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def structural_text(stats: Mapping[str, Any]) -> str:
    """The run's structural changes so far (``Loop._structural_stats``) in plain words:
    how many candidates made one, how many did not, how many structural proposals the
    validator refused, and the operations used -- and, when none was tried, that none
    has been (a fact about this run, not a judgement of structure)."""
    structural = _count(stats.get("structural_candidates"))
    other = _count(stats.get("knob_candidates"))
    refused = _count(stats.get("structural_refusals"))
    if structural:
        text = (f"{structural} candidate(s) so far made a structural change and {other} "
                "did not")
    elif refused:
        text = ("No structural change has been evaluated in this run yet; every candidate "
                f"so far ({other}) keeps the starting pipeline's structure")
    else:
        text = ("No structural change has been tried in this run yet; every candidate so "
                f"far ({other}) keeps the starting pipeline's structure")
    if refused:
        text += (f". {refused} structural proposal(s) were refused by the validator before "
                 "running")
    ops = stats.get("accepted_ops")
    if isinstance(ops, Mapping) and ops:
        text += ". Operations in the candidates so far: " + ", ".join(
            f"`{op}` x{_count(n)}" for op, n in ops.items())
    return text + "."


def _evidence_rows(rows: Sequence[tuple[str, Any]],
                   history: Sequence[Mapping[str, Any]] = ()) -> list[list[str]]:
    """"Evidence from the traces" rows: what the run has done with structure first, in
    plain words (`structural_text`), then every other trace row as data. Before the
    loop's first gradient (a strategy's first request, the traces still empty) the
    structure row is counted from ``history``, every evaluated candidate, when it is
    whole (the caller passes none when it is truncated)."""
    out = [["structural changes", structural_text(value)] for key, value in rows
           if key == "structural" and isinstance(value, Mapping)]
    if not out and history:
        flags = [bool(c.get("structural")) for c in history if isinstance(c, Mapping)]
        if flags:
            out = [["structural changes", structural_text(
                {"structural_candidates": sum(flags),
                 "knob_candidates": len(flags) - sum(flags)})]]
    return out + [[str(key), json.dumps(value, default=str)] for key, value in rows
                  if not (key == "structural" and isinstance(value, Mapping))]


def _feedback_sections(ctx: "ProposalContext") -> list[str]:
    """The run-wide feedback sections (`optim.feedback`): every change by kind, whether
    the last round's added steps contributed, the loop's memory note (`optim.memory`),
    and the integrity judge's verdicts on its proposals (`optim.integrity`); each only
    when its field holds something."""
    from . import feedback, integrity, memory

    out: list[str] = []
    for lines in (feedback.ledger_lines(ctx.ledger),
                  feedback.step_review_lines(ctx.added_steps),
                  memory.request_lines(ctx.memory),
                  integrity.feedback_lines(ctx.integrity)):
        if lines:
            out += ["", *lines]
    return out


#: The edit levels a portfolio round asks for, one proposal each (``edit_level``).
EDIT_LEVELS = ("prompt_knob", "pipeline")
_LEVEL_TEXT = {
    "prompt_knob": "a prompt/knob-level change: only `set_knob` edits (patch or rewrite a "
                   "prompt, move a setting) on the current pipeline",
    "pipeline": "a pipeline-level change: at least one structural operation (add, remove, "
                "rewire, gate, loop or parallelise steps{code}), with any knob moves it needs",
}


def edit_level(steps: Sequence[Any]) -> str:
    """``prompt_knob`` when every edit is a ``set_knob``, else ``pipeline``."""
    ops = [step.get("op") if isinstance(step, dict) else None for step in steps]
    return "prompt_knob" if ops and all(op == "set_knob" for op in ops) else "pipeline"


#: What a pipeline-level ask adds when the run allows code steps (``code_nodes``).
_CODE_STEP_TEXT = (", or add a code step that checks or extracts something "
                   "deterministically, at no model call")


def _level_text(level: str, code_nodes: bool) -> str:
    return _LEVEL_TEXT.get(level, level).replace("{code}", _CODE_STEP_TEXT if code_nodes else "")


def levels_instructions(levels: Sequence[str], *, code_nodes: bool = False) -> list[str]:
    """What the request asks for when it names edit levels; the pipeline level names code
    steps among its moves when the run allows them (``code_nodes``)."""
    if len(levels) == 1:
        return ["## The kind of change this round",
                f"Propose exactly one update: {_level_text(levels[0], code_nodes)}."]
    return ["## The kind of change this round",
            f"Propose exactly {len(levels)} updates, one per level, each your best of its "
            "kind:",
            *[f"- {_level_text(level, code_nodes)}" for level in levels],
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
    """An example edit for a structural op this run permits: the one-line ``insert`` when
    the run permits it, else one built from another op's catalogue fields with
    placeholders; ``None`` when none is permitted."""
    if any(op.get("op") == "insert" for op in catalogue):
        return {"op": "insert", "after": "<existing step id>", "kind": "agent",
                "text": "<what the new step does>"}
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


def _fact_filled_knobs(node_states: Sequence[dict[str, Any]],
                       text_knobs: Iterable[str] | None = None) -> list[str]:
    """The knobs an op that fills facts reads (`nn.fills_facts`): beside an agent's
    templates, the only places a `{{fact}}` is filled in. Only text knobs, given
    ``text_knobs`` (a flag the same op reads holds no slot)."""
    from refactory_dsh.nn.graph import Node
    from refactory_dsh.nn.validate import filled_knobs

    names: set[str] = set()
    for state in node_states:
        try:
            names.update(filled_knobs(Node.from_state(dict(state))))
        except Exception:  # a node that does not read back is the validator's to report
            continue
    if text_knobs is not None:
        names &= set(text_knobs)
    return sorted(names)


def composition_section(catalogue: Sequence[dict[str, Any]], node_states: Sequence[dict[str, Any]],
                        edges: Sequence[dict[str, Any]], *, protected: Sequence[str] = (),
                        code_nodes: bool = False,
                        text_knobs: Iterable[str] | None = None,
                        structure_guidance: bool = False, tools: bool = False,
                        rotation: int = 0,
                        knob_values: Mapping[str, Any] | None = None) -> list[str]:
    """The language's primitives and how they compose; then the core operators
    (`nn.core_ops`), one line and one example each, and compositions built only from them
    (several attempts and a choice, check and send back, an independent attempt, ...),
    from this graph's own ids, rotated by ``rotation`` so no one shape reads as the way to
    grow the pipeline (`operator_lines`, `composition_examples`). With
    ``structure_guidance``, what makes a new step pay off (`feedback.STRUCTURE_GUIDANCE`).
    Nothing when the run permits no structural op."""
    from refactory_dsh import snippet

    if not structural_ops(catalogue):
        return []
    ops = {op.get("op") for op in catalogue}
    reader = ("a code op that reads the file and `produces` the fact" if code_nodes else
              "an op that reads the file and `produces` the fact")
    filled = _fact_filled_knobs(node_states, text_knobs)
    lines = [
        "## How nodes compose",
        "The pipeline is made of four kinds of step, and any of them may be put anywhere the "
        "validator's rules (listed below) allow:",
        "- **agent**: one model call. Its prompt fills each `{{name}}` from a fact (or knob) "
        "produced upstream; its reply is written to its first `writes` file and, when it "
        "declares `produces`, also becomes that fact. A new agent runs without tools, so it "
        "sees only its prompt and the files it lists in `reads` (pasted in after the prompt): "
        "hand it anything else as `{{facts}}`. "
        f"A file's text becomes a fact through {reader}, or when the agent that writes the "
        "file declares `produces`.",
        "- **where `{{fact}}` is filled in**: only in an agent's `prompt_template` (its "
        "`<id>__task` knob) and its system prompt (its `<id>__system_prompt` knob)"
        + (f", and in {', '.join(f'`{name}`' for name in filled)}, which an op that fills "
           "facts reads" if filled else "")
        + ". Any other knob is read as written, so a `{{fact}}` there would never arrive "
        "and is refused (V20).",
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
            "`set_knob` patches or rewrites.",
        ]
    lines += [
        "- **gate**: routes by a verdict, to the out-edge whose `condition` is that verdict. "
        "Give each out-edge a single-word condition (`proceed`, `reloop`, `revise` ...). "
        f"Any declared upstream output can be the verdict: with `\"evaluator\": "
        f"\"{snippet.VERDICT}\"` the gate reads the verdict from its first `required_reads` "
        "file (or, with none, its first `inputs` fact) -- whatever an agent or a code op "
        "upstream wrote there: its first word, or else its last line when that is the word "
        "alone or the word then `:`, `—`, `–`, `-` or `.` and more (leading markdown or "
        "quotes are skipped; a list item or a code fence is not read) -- and the rest of "
        "the text is the reason. The word must be one of the conditions (any case) or "
        "`halt`, which ends the whole run. Anything else fails that item's run, and so does "
        "a first word and a last line that name different verdicts, so tell the writer to "
        "start with the word or put it alone on the last line.",
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
        "- **running out of output**: a model call that uses its whole output allowance "
        "without answering is the model failing that question. It is not retried: the step "
        "has no answer, and the item goes on without it (a gate after it sees no answer and "
        "may reloop; a fork's join sees that branch without one; a code op's `inp['files']` "
        "holds only the files that exist; an item left with nothing to grade scores 0). "
        "Address it in the graph: a prompt that commits to an answer early and keeps its "
        "working short, shorter attempts plus a check that sends a step back, or several "
        "attempts and a vote.",
        "- **crashes**: when the worker or product fails (not the model's own output), an "
        "agent step in the per-item pass that is safe to rerun (no tools, its own report "
        "files only, or a workspace) is run again in a fresh session, up to 2 more times; "
        "if every session crashes it has no answer, as above. A timeout and a failure of the "
        "connection to the model are not retried and still fail the item. Retried sessions "
        "count toward the item's time and cost.",
    ]
    if {"parallel", "parallelize"} & ops:
        lines.append(
            "- **fork**: steps run at once between a fork and a join; branches write distinct "
            "files and facts. `parallelize` and `branch` wire what the branches make into the "
            "step after the join; the written-out `parallel` operation leaves that to you.")
    if structure_guidance:
        from .feedback import STRUCTURE_GUIDANCE
        lines.append(STRUCTURE_GUIDANCE)
    lines += operator_lines(catalogue, node_states, edges, protected=protected,
                            knob_values=knob_values)
    examples = composition_examples(catalogue, node_states, edges, protected=protected,
                                    code_nodes=code_nodes, tools=tools,
                                    knob_values=knob_values)
    if examples:
        turn = rotation % len(examples)
        lines += ["", "Compositions built only from these operators -- none is the preferred "
                      "shape; compose your own:"]
        for title, bundle in examples[turn:] + examples[:turn]:
            lines += [f"- {title}:",
                      "  " + json.dumps({"edits": bundle} if len(bundle) > 1 else bundle[0])]
    return lines + [""]


#: The core operators a request explains, in its order (`nn.core_ops.CORE_OPS`).
CORE_OPERATORS = nn_edits.CORE_OPS


def _example_ids(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                 protected: Sequence[str]) -> dict[str, str]:
    """Ids of this graph for the operators' examples: a step (an agent with one plain
    out-edge to a step) and its next step, and a fork, where the graph has them."""
    ids: dict[str, str] = {}
    anchor = _composition_anchor(node_states, edges, protected)
    if anchor is not None:
        ids["step"], ids["next step"] = str(anchor[0]["id"]), anchor[1]
    fork = next((str(n.get("id")) for n in sorted(node_states, key=lambda n: str(n.get("id")))
                 if n.get("kind") == "fork"), None)
    if fork:
        ids["fork"] = fork
    return ids


def _example_state(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                   knob_values: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """The request's graph as a state the operators' preconditions can read (its start is
    the step nothing leads to), or ``None`` when it does not read back."""
    from refactory_dsh.nn.graph import Graph

    entered = {str(e.get("target")) for e in edges} | {
        str(t) for n in node_states for t in (n.get("targets") or ())} | {
        str(n.get("subgraph_entry")) for n in node_states if n.get("subgraph_entry")}
    start = next((str(n["id"]) for n in sorted(node_states, key=lambda n: str(n.get("id")))
                  if str(n["id"]) not in entered), "")
    try:
        graph = Graph.from_state({"start": start,
                                  "nodes": {str(n["id"]): dict(n) for n in node_states},
                                  "edges": list(edges)})
    except Exception:  # noqa: BLE001 - a graph that does not read back gets no examples
        return None
    return {"graph": graph.to_state(),
            "knobs": {k: {"value": v, "spec": {"kind": "str"}}
                      for k, v in (knob_values or {}).items()}}


def _removable_step(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                    protected: Sequence[str], state: Mapping[str, Any] | None) -> str | None:
    """A step ``remove`` can take out of this graph as an example: an agent (else an op) with
    a next step, not one a gate's verdict leads to, not after a protected or data node, and
    making nothing a protected node reads; ``None`` when there is none."""
    if state is None:
        return None
    kinds = {str(n.get("id")): n.get("kind") for n in node_states}
    guarded = {name for other in node_states if str(other.get("id")) in protected
               for key in ("reads", "required_reads", "inputs") for name in other.get(key) or ()}
    sources = {str(e.get("source")) for e in edges}
    ranked = sorted(node_states, key=lambda n: (n.get("kind") != "agent", str(n.get("id"))))
    for node in ranked:
        node_id = str(node.get("id"))
        into = [e for e in edges if e.get("target") == node_id]
        made = {*(node.get("writes") or ()), *(node.get("produces") or ())}
        if node.get("kind") not in ("agent", "op") or node_id in protected or made & guarded \
                or node_id not in sources or any(
                    e.get("condition") or e.get("source") in protected
                    or kinds.get(str(e.get("source"))) == "data" for e in into):
            continue
        if nn_edits.precheck({"op": "remove", "step": node_id}, dict(state)) is None:
            return node_id
    return None


def _fitting_ids(name: str, ids: Mapping[str, str], node_states: Sequence[dict[str, Any]],
                 edges: Sequence[dict[str, Any]], protected: Sequence[str],
                 state: Mapping[str, Any] | None) -> dict[str, str]:
    """The example ids for operator ``name``: ``ids`` when its example passes its
    precondition on this graph, else the first other step (agents first, each with its one
    plain next step) for which it does, else placeholders for the step ids."""
    from refactory_dsh.nn import core_ops

    placeholders = {**ids, "step": "<step>", "next step": "<next step>"}
    if state is None:
        return placeholders
    out_of = {}
    for edge in edges:
        out_of.setdefault(str(edge.get("source")), []).append(edge)
    options = [dict(ids)] if "step" in ids else []
    # A data pass's first and last steps are named by its data node, which an edit there
    # would change: not an example to copy.
    ends = {str(n.get(key)) for n in node_states if n.get("kind") == "data"
            for key in ("subgraph_entry", "subgraph_exit")}
    for node in sorted(node_states, key=lambda n: (n.get("kind") != "agent", str(n.get("id")))):
        node_id = str(node.get("id"))
        out = out_of.get(node_id) or []
        if node.get("kind") in ("agent", "op") and node_id not in protected and len(out) == 1 \
                and not out[0].get("condition") and node_id not in ends:
            options.append({**ids, "step": node_id, "next step": str(out[0].get("target"))})
    for option in options:
        example = core_ops.example(name, option)
        named = [example.get(k) for k in ("step", "steps", "after", "before", "fork")]
        if "<" in json.dumps(named) or "<" in json.dumps(list(example.get("to") or {}.values())):
            continue
        if nn_edits.precheck(example, dict(state)) is None:
            return option
    return placeholders


#: How a core operator's refusal names the edit that replaces the step (`core_ops`).
_REPLACE_HINT = "replace it with "


def checked_hint(reason: str, check: Callable[[dict[str, Any]], Any]) -> str:
    """``reason`` with the replacing edit it names checked by ``check`` (an attempt's
    verdict: a candidate, or its refusals): kept when it would be accepted, otherwise
    turned into "replacing it is refused too (<why>)". The operators' own precheck knows
    only the graph; an attempt also refuses on protected steps, frozen and guarded knobs,
    wiring and validation, so the hint is held to that."""
    start = reason.find(_REPLACE_HINT)
    if start < 0:
        return reason
    try:
        edit, end = json.JSONDecoder().raw_decode(reason, start + len(_REPLACE_HINT))
    except ValueError:
        return reason
    verdict = check(edit)
    if not isinstance(verdict, list):
        code = any(isinstance(step, dict) and step.get("kind") == "code"
                   for step in edit.get("into") or ())
        # Accepted without running the new code on recorded inputs (`attempt` skips that
        # for a hint: a process per request), which an attempt of the edit itself does.
        return (f"{reason[:end]} (not dry-run){reason[end:]}" if code else reason)
    from refactory_dsh.nn.core_ops import CODE_HINT_NOTE

    why = "; ".join(str(r).removeprefix("E-precondition: ") for r in verdict)
    rest = reason[end:].removeprefix(CODE_HINT_NOTE)
    return f"{reason[:start]}replacing it is refused too ({why}){rest}"


#: The most characters the "what each step carries" section takes: past it, each refusal
#: is cut to its first clause, then rows are left out, last first, and named.
OPERATOR_FIT_CHARS = 7000


def pipeline_view_lines(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                        knobs: Sequence[Mapping[str, Any]], protected: Sequence[str],
                        graders: Sequence[str] = ()) -> list[str]:
    """The pipeline in one line per step (`nn.pipeline_view`), in the operators' own words;
    nothing when the graph does not read back."""
    from refactory_dsh.nn.pipeline_view import render

    state = _fit_state(node_states, edges, knobs)
    text = (render(state, protected, graders) if state is not None
            else "pipeline view unavailable: the graph does not read back")
    return [PIPELINE_VIEW_LEAD, "```text", text, "```"]


#: The reference at the end of the request, and the line that points to it.
REFERENCE_HEADING = "## Reference: every node in full, the schemas and the rules"
REFERENCE_POINTER = ("Every node in full (its declarations, what it has before it runs), the "
                     "node schemas and the validator's rules are in the reference at the end.")

#: What the pipeline view's words mean.
def _view_lead() -> str:
    from refactory_dsh.nn.pipeline_view import FORMS

    return (
        "The pipeline, one line per step, generated from the graph in the operators' own "
        "words (full node detail is in the reference at the end): `sees` lists a step's "
        "inputs as `\"sees\"` names them, `name@maker` (`@task`: no step makes it); a plain "
        "op that only assembles a model step's input is folded in as `name (via op: its "
        "inputs)`; an agent with tools sees the run dir (or its own workspace); `told` is "
        "each knob whose text reaches the step, its opening words (`*` frozen; `?` read by "
        "an op with several outputs, for all of them), or a code step's code; `->` what it "
        "writes; " + "; ".join(f"`{form}`" for form in FORMS.values())
        + "; `[optional]`; `[protected]`.")


PIPELINE_VIEW_LEAD = _view_lead()


def _fit_edits(node: Any, graph: Any, ops: set[str]) -> list[tuple[str, dict[str, Any]]]:
    """Each core operator the run permits, aimed at one step with its default arguments:
    a split into one step that sees only the task stands for "replace it"."""
    from refactory_dsh import snippet

    node_id = node.id
    plain_next = [e.target for e in graph.out_edges(node_id) if e.condition is None
                  and graph.nodes[e.target].kind in ("agent", "op")]
    model = node.kind == "agent" or snippet.is_code_node(node)
    if node.kind == "fork":
        edits = [("branch", {"op": "branch", "fork": node_id, "kind": "agent"}),
                 ("serialize", {"op": "serialize", "fork": node_id})]
    else:
        edits = [("insert", {"op": "insert", "after": node_id, "kind": "agent"}),
                 ("remove", {"op": "remove", "step": node_id}),
                 *([("rewire", {"op": "rewire", "step": node_id, "sees": ["task"]})] if model
                   else []),
                 ("loop", {"op": "loop", "step": node_id, "judge": "agent"}),
                 ("route", {"op": "route", "after": node_id, "judge": "agent"}),
                 ("split", {"op": "split", "step": node_id}),
                 ("split (replace)", {"op": "split", "step": node_id,
                                      "into": [{"kind": "agent", "sees": ["task"]}]}),
                 *([("parallelize", {"op": "parallelize", "steps": [node_id], "copies": 1})]
                   if node.kind == "agent" else [])]
        if len(plain_next) == 1:
            edits.append(("merge", {"op": "merge", "steps": [node_id, plain_next[0]]}))
    return [(label, edit) for label, edit in edits if edit["op"] in ops]


def _fit_state(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
               knobs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The request's graph as a state, its knobs with their kind and frozen mark (the
    operators' refusals read them)."""
    state = _example_state(node_states, edges, {k.get("name"): k.get("value") for k in knobs})
    if state is None:
        return None
    for knob in knobs:
        entry = state["knobs"].get(knob.get("name"))
        if entry is not None:
            from refactory_dsh.nn.knob import _SPEC_FIELDS, KINDS

            spec = {k: knob[k] for k in _SPEC_FIELDS if knob.get(k) is not None}
            spec.setdefault("default", knob.get("value"))
            if spec.get("kind") not in KINDS or (spec.get("kind") == "enum"
                                                 and not spec.get("enum")):
                spec["kind"] = "str"
            entry["spec"] = spec
    return state


#: How many plain ops back a step's input is followed before the table stops ("not
#: followed further").
CARRIES_DEPTH = 4


def _carries(node: Any, graph: Any, knob_values: Mapping[str, Any]) -> str:
    """What a model step's prompt and reads carry, each with what made it: a plain op's
    output is followed back through the plain ops that built it to its sources (model
    steps, the task's own inputs, knobs), at most `CARRIES_DEPTH` ops deep. The knobs an
    op with several outputs reads are the op's: no op says which output a knob feeds."""
    from refactory_dsh import snippet
    from refactory_dsh.nn.operators import made_by
    from refactory_dsh.nn.validate import consumed

    makers = made_by(graph)

    def described(kind: str, name: str, depth: int, seen: frozenset[str]) -> str:
        made = makers.get((kind, name)) or []
        plain = [m for m in made if graph.nodes[m].kind == "op"
                 and not snippet.is_code_node(graph.nodes[m]) and m not in seen]
        if not plain:
            return f"{name} ({', '.join(made) or 'task input'})"
        if depth >= CARRIES_DEPTH:
            return f"{name} (op {', '.join(plain)}, not followed further)"
        parts = []
        for op_id in plain:
            op = graph.nodes[op_id]
            sources = [described(k, n, depth + 1, seen | {op_id})
                       for k, n in sorted(consumed(op, knob_values))]
            knobs = list(dict.fromkeys([*op.knobs, *(k for _f, k in op.field_bindings)]))
            outputs = len(op.produces) + len(op.writes)
            if knobs and outputs > 1:
                sources.append(f"knob(s) {', '.join(knobs)} (the op's, for all {outputs} "
                               "of its outputs)")
            else:
                sources += [f"knob {knob}" for knob in knobs]
            parts.append(f"op {op_id}, which reads {', '.join(sources) or 'nothing'}")
        return f"{name} ({'; '.join(parts)})"

    return "; ".join(described(k, n, 0, frozenset())
                     for k, n in sorted(consumed(node, knob_values))) or "-"


def _fork_inputs(fork_id: str, state: dict[str, Any]) -> str:
    """What a new branch of ``fork_id`` can name in ``sees``: each fact or file there before
    the fork as the pipeline view writes it (``name@maker``), one built by a plain op that
    reads a text knob marked with it (it may carry that knob's instructions), and what
    ``"task"`` gives there. Read from the graph; "-" when it does not read back."""
    from refactory_dsh import snippet
    from refactory_dsh.nn import core_ops
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.operators import made_by

    try:
        graph = Graph.from_state(state["graph"])
        there = core_ops._readable(graph, state, set(core_ops._available(graph).get(fork_id, set())))
        task = core_ops._task_inputs(graph, state) & there
    except Exception as exc:  # noqa: BLE001 - a fork that does not read back gets no line
        import logging

        logging.getLogger(__name__).debug("no inputs for fork %s: %s", fork_id, exc)
        return "-"
    makers = made_by(graph)
    values = {k: (e or {}).get("value") for k, e in (state.get("knobs") or {}).items()}

    def named(kind: str, name: str) -> str:
        made = makers.get((kind, name)) or []
        texts = [k for m in made if graph.nodes[m].kind == "op"
                 and not snippet.is_code_node(graph.nodes[m])
                 for k in graph.nodes[m].knobs if isinstance(values.get(k), str)]
        return (f"{name}@{','.join(made) or 'task'}"
                + (f" (+knob {', '.join(dict.fromkeys(texts))})" if texts else ""))

    order = sorted(there, key=lambda key: (key[0] != "fact", key[1]))
    return (f"a new branch here can see: {', '.join(named(*key) for key in order) or 'nothing'}"
            f"; \"task\" gives {', '.join(n for _k, n in sorted(task)) or 'nothing'}")


def operator_fit_section(catalogue: Sequence[dict[str, Any]],
                         node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                         knobs: Sequence[Mapping[str, Any]],
                         check: Callable[[dict[str, Any]], Any] | None, *,
                         limit: int = OPERATOR_FIT_CHARS) -> list[str]:
    """"What each step carries, and which operators apply to it": for each model step,
    what its prompt carries (a plain op's output followed back to what that op reads),
    and for each step which core operators, with their default arguments, apply as written
    and which are refused, with the refusal's own words. Each verdict is an attempt's:
    ``check`` is the loop's (`Run.mechanics` on the edited candidate: every check an
    attempt makes but the model-judged integrity one). Information only; nothing without
    a check (a request built outside a run, whose verdicts would be guesses) or when the
    run permits no core operator."""
    from refactory_dsh.nn.graph import Graph

    ops = {op.get("op") for op in catalogue} & set(CORE_OPERATORS)
    state = _fit_state(node_states, edges, knobs) if ops and check is not None else None
    if state is None or check is None:
        return []
    graph = Graph.from_state(state["graph"])
    knob_values = {k.get("name"): k.get("value") for k in knobs}
    code_before = _code_signatures(state)
    rows: list[list[str]] = []
    full_refusals: list[list[str]] = []
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if node.kind not in ("agent", "op", "fork"):
            continue
        carries = (_carries(node, graph, knob_values) if node.kind == "agent"
                   else _fork_inputs(node_id, state) if node.kind == "fork" else "-")
        applies: list[str] = []
        reasons: dict[str, list[str]] = {}
        for label, edit in _fit_edits(node, graph, ops):
            verdict = check(edit)
            if not isinstance(verdict, list):
                # Its new or changed code was not run on recorded inputs (the loop's check
                # skips that); proposing the edit runs it, as any attempt does.
                changed = any(code_before.get(node_id) != signature for node_id, signature
                              in _code_signatures(verdict.state).items())
                applies.append(f"{label} (not dry-run)" if changed else label)
                continue
            reason = "; ".join(str(r).removeprefix("E-precondition: ") for r in verdict)
            reasons.setdefault(reason, []).append(label)
        refused = [f"{', '.join(labels)}: {reason}" for reason, labels in reasons.items()]
        rows.append([node_id, carries, ", ".join(applies) or "-", " | ".join(refused) or "-"])
        full_refusals.append(refused)
    if not rows:
        return []
    from refactory_dsh.nn.core_ops import TASK, _readable, _task_inputs

    task = sorted(_readable(graph, state, _task_inputs(graph, state)),
                  key=lambda key: (key[0] != "fact", key[1]))
    task_line = (f" `\"sees\": [\"{TASK}\"]` gives a new step these, each where it is there "
                 f"before the step: {', '.join(name for _k, name in task)}: what no model "
                 "step's output reaches, as the steps declare their inputs. A fact a plain op "
                 "builds from a knob carries that knob's text (another step's instructions, "
                 "say)." if task else "") + (
        " A fork's row lists every name a new branch there can put in `sees` for that input "
        "alone (`name@maker`; `(+knob k)`: built by an op that reads text knob k, so it may "
        "carry that text); a new step after a step can name what that step sees (its line in "
        "the pipeline view) or what it writes.")
    lead = ("## What each step carries, and which operators apply to it\n"
            "Worked out on the current pipeline, before you choose: what each model step's "
            "prompt and reads carry, each with the step that makes it (`(op X, which reads "
            "...)`: a fact a plain op builds, and everything that op reads), and which core "
            "operators an attempt accepts for each step as written with their default "
            "arguments (`split (replace)`: a split into one step of your own text that sees "
            "the task) and which it refuses, in the refusal's own words. A step that "
            "alone reads a frozen knob is not removed, and is split or merged only when the "
            "new step's prompt names each such knob as `{{knob}}` (never one named after the "
            "step, `<step>__...`); otherwise insert a step beside it." + task_line)
    header = ["step", "its prompt and reads carry", "applies as written", "refused (why)"]

    def first_clause(refusal: str) -> str:
        labels, why = refusal.split(": ", 1)
        return f"{labels}: {why.split(';')[0].split('. ')[0].strip()}"

    short = [row[:3] + [" | ".join(map(first_clause, refused)) or "-"]
             for row, refused in zip(rows, full_refusals)]
    for shown in (rows, short):
        text = _table(header, shown)
        if len(lead) + len(text) <= limit:
            return [lead, text, ""]
    listed = len(short)
    while short:
        short = short[:-1]
        text = _table(header, short)
        note = (f"({listed - len(short)} of {listed} steps not listed, to stay within "
                f"{limit} characters)")
        if len(lead) + len(text) + len(note) <= limit:
            return [lead, text, note, ""]
    return []


def operator_lines(catalogue: Sequence[dict[str, Any]], node_states: Sequence[dict[str, Any]],
                   edges: Sequence[dict[str, Any]], *, protected: Sequence[str] = (),
                   knob_values: Mapping[str, Any] | None = None) -> list[str]:
    """"Structural operators": what the system wires for every one, then each core operator
    the run permits in one line with one example (`nn.core_ops.OPERATORS`), its ids taken
    from this graph where it has them. Nothing when the run permits none."""
    from refactory_dsh.nn import core_ops

    ops = {op.get("op") for op in catalogue}
    chosen = [name for name in CORE_OPERATORS if name in ops]
    if not chosen:
        return []
    ids = _example_ids(node_states, edges, protected)
    removable = _removable_step(node_states, edges, protected,
                                _example_state(node_states, edges, knob_values))
    lines = [
        "",
        "## Structural operators",
        "Each is one line: what kind of step (`agent`, `code` or `op`), where (a step id), and "
        "its text (a prompt, or code). The system does the wiring: a new step sees what its "
        "neighbour sees plus the neighbour's outputs (`\"sees\"` names others: `\"task\"` "
        "for the task's own inputs -- facts and files no model step made -- step ids for "
        "their outputs, or single fact or file names, as the pipeline view writes them "
        "before `@`, for that input alone); files reach a model step through a collect step that says a missing "
        "file instead of failing; its output (a file and a fact named after its id) goes into "
        "the prompt of the next model step after it (`\"to\"` names another step or prompt "
        "knob); ids, files and facts are generated; each text goes into a knob of its own "
        "(`<id>__task`, `<id>__code`) for a later `set_knob`; and the candidate says what was "
        "wired. Every text has a default (the neighbour's, or a neutral line), and each "
        "operator counts as one change. Steps after the change keep the names they read. The "
        "written-out operations in the table above stay for what these do not express.",
    ]
    state = _example_state(node_states, edges, knob_values)
    for name in chosen:
        spec = core_ops.OPERATORS[name]
        # Ids this graph's own steps fill where the operator applies to them; a step this
        # graph can lose for `remove`; placeholders otherwise, never a step it refuses.
        own = {**ids, "step": removable or "<step>"} if name == "remove" else _fitting_ids(
            name, ids, node_states, edges, protected, state)
        lines.append(f"- `{name}`: {spec['line']}. E.g. "
                     + json.dumps(core_ops.example(name, own)))
    return lines


def composition_examples(catalogue: Sequence[dict[str, Any]],
                         node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                         *, protected: Sequence[str] = (), code_nodes: bool = False,
                         tools: bool = False, knob_values: Mapping[str, Any] | None = None
                         ) -> list[tuple[str, list[dict[str, Any]]]]:
    """Purpose-level shapes built only from the core operators, with this graph's ids: none
    is preferred, each is one way to compose them. Each bundle applies as written to the
    graph it was built from (``tests/test_core_operators.py`` holds it to that). None when
    the graph has no step to anchor them or the run permits no core operator."""
    from refactory_dsh.nn import core_ops
    from refactory_dsh.nn.graph import Graph

    ops = {op.get("op") for op in catalogue}
    if not set(CORE_OPERATORS) & ops:
        return []
    anchor = _composition_anchor(node_states, edges, protected)
    if anchor is None:
        return []
    state = _example_state(node_states, edges, knob_values)
    if state is None:
        return []
    graph = Graph.from_state(state["graph"])
    step = str(anchor[0]["id"])
    out: list[tuple[str, list[dict[str, Any]]]] = []
    if {"parallelize", "insert"} <= ops:
        join = core_ops.parallel_join_id(graph, step)
        copies = [f"{step}_copy{n}" for n in (1, 2)]
        stems = [str(anchor[0]["writes"][0]).rsplit(".", 1)[0], *copies]
        choose = f"{step}_choose"
        out.append(("Several attempts and a choice", [
            {"op": "parallelize", "steps": [step], "copies": 2},
            {"op": "insert", "after": join, "kind": "agent", "id": choose,
             "text": "<compare the attempts against the task>; start your answer with the "
                     f"best one's name: {', '.join(stems)}"},
            {"op": "insert", "after": choose, "kind": "op", "call": core_ops.select_call()}]))
    if "loop" in ops:
        out.append(("Check and send back", [
            {"op": "loop", "step": step, "judge": "agent", "text": "<what to check in the work>"}]))
    fork = next((str(n.get("id")) for n in node_states if n.get("kind") == "fork"), None)
    if fork and "branch" in ops:
        out.append(("An independent attempt that sees only the task", [
            {"op": "branch", "fork": fork, "kind": "agent", "text": "<solve it your own way>",
             "sees": ["task"]}]))
    elif "parallelize" in ops:
        out.append(("An independent attempt that sees only the task", [
            {"op": "parallelize", "steps": [step], "copies": 1, "sees": ["task"]}]))
    if code_nodes and "insert" in ops:
        sim = f"{step}_sim"
        out.append(("A simulation step, then an analysis of its results", [
            {"op": "insert", "after": step, "kind": "code", "id": sim,
             "text": "def main(inp):\n    result = len(inp['facts'])  # <the simulation>\n"
                     f"    return {{'facts': {{{sim!r}: str(result)}}}}\n"},
            {"op": "insert", "after": sim, "kind": "agent",
             "text": "<analyse the simulation's results>"}]))
    if "parallelize" in ops:
        beside = [{"kind": "agent", "text": "<a background pass: what is already known about "
                                            "this>"},
                  {"kind": "agent", "text": "<the experiment>"}]
        out.append(("A background pass beside an experiment", [
            {"op": "parallelize", "after": step, "new": beside}]))
    if code_nodes and "loop" in ops:
        out.append(("Iterative fitting: repeat a step until a code check passes", [
            {"op": "loop", "step": step, "judge": "code",
             "text": "def main(inp):\n    residual = 1.0  # <measure it from inp['facts'] and "
                     "inp['files']>\n    ok = residual < 0.1\n    return {'verdict': 'proceed' "
                     "if ok else 'reloop', 'reason': 'residual ' + str(residual)}\n"}]))
    if tools and "insert" in ops:
        out.append(("A tool step before reasoning", [
            {"op": "insert", "before": step, "kind": "agent", "tools": True,
             "text": "<use your tools to gather what the next step needs>"}]))
    removable = _removable_step(node_states, edges, protected, state) if "remove" in ops else None
    if removable is not None:
        out.append(("Dropping a step that does not help", [{"op": "remove", "step": removable}]))
    return out


def time_lines(facts: Mapping[str, Any] | None) -> list[str]:
    """"Time per item", in numbers, so time is reasoned about from them: how long one model
    step may run (the run's limit; a step's own ``timeout_s`` overrides it), the edited
    candidate's typical time per item, and how many of its items timed out. Nothing when
    none is known."""
    facts = dict(facts or {})
    limit, per_item = facts.get("step_limit_s"), facts.get("per_item_s")
    items, timeouts = int(facts.get("items") or 0), int(facts.get("timeouts") or 0)
    runs = int(facts.get("runs") or items)
    counted = f"{items} items" if runs == items else f"{runs} runs of {items} items"
    if not (limit or per_item or items):
        return []
    parts = []
    if limit:
        parts.append(f"One model step may run for up to {int(limit)} s (the run's limit per "
                     "step; a step's own `timeout_s` overrides it); a step past its limit fails "
                     "that item, as a timeout.")
    if per_item:
        parallel = int(facts.get("parallelism") or 1)
        parts.append(f"The candidate you edit took about {float(per_item):.0f} s per item on "
                     "its train evaluation (its wall time"
                     + (f" x {parallel} items at once" if parallel > 1 else "")
                     + (f" / {items} items)." if runs == items else f" / {runs} runs)."))
    if items:
        parts.append(f"{timeouts} of its {counted.replace('items', 'train items')} timed out.")
    return ["## Time per item", " ".join(parts), ""]


def _cost_note(parsimony: float | None) -> list[str]:
    """With a cost term in the objective, what the cost column of the results says."""
    if not parsimony:
        return []
    return ["The cost per item column is each candidate's measured model cost (USD) over its "
            "train items, so a change's cost can be weighed against the size cost the score "
            "carries; - where it was not measured."]


def _cost_cell(value: Any) -> str:
    return f"{float(value):.4g}" if isinstance(value, (int, float)) else "-"


#: What a run's score does not count unless the task's own score measures it.
_UNSCORED = "inference cost, latency, and the number of steps or model calls"
#: How the validation gate decides (``ValidationGate.judge_round``), said beside the
#: objective in every request, since the analyst, merge and memory requests do not carry
#: the proposal request's gate note.
_GATE_RULE = ("The validation gate decides on the task's score on the validation items: "
              "it keeps a candidate only when that score beats the current candidate's{none}; "
              "a train score{train} only breaks a tie between candidates that beat the "
              "current one by the same validation score; a tie with the current candidate "
              "is rejected.")


def objective_text(parsimony: float | None, *, item_mean: bool = False,
                   gated: bool = False) -> str:
    """What the score measures, said from the run's own settings (``LoopConfig.parsimony``,
    whether the score is the items' mean and whether a validation gate decides), so the
    optimizer does not trade task score for a cost the run does not count. Empty when
    ``parsimony`` is not known (``None``). Domain-free: it names no task, and it says
    that cost is outside the score only as far as the loop knows: a task's own score
    may measure it.

    With parsimony 0 the score is the task's own score and nothing else. Above 0 the
    train score carries ``parsimony`` per node of the graph, as `Loop` computes it. Under
    the validation gate (either way) the gate's own comparison is the task's validation
    score, which carries no size cost; a train score only breaks a tie between
    candidates that beat the current one by the same validation score, and a tie with
    the current one is rejected (``ValidationGate.judge_round``)."""
    if parsimony is None:
        return ""
    task = "the task's own score" + (" (the mean of the items' scores)" if item_mean else "")
    unless = "unless the task's own score measures them"
    if not parsimony:
        text = (f"The score is {task} and nothing else. This run adds no cost term to it "
                f"(parsimony is 0): {_UNSCORED} are not part of the score {unless}. A "
                "change is judged only by the task score it reaches.")
        return text + (" " + _GATE_RULE.format(none="", train="") if gated else "")
    size = (f"a size cost of {parsimony:g} per node of the pipeline's graph (score = raw "
            f"score - {parsimony:g} x nodes)")
    rest = ("Apart from that size cost, inference cost, latency and the number of model "
            f"calls are not part of the score {unless}.")
    if gated:
        return (f"The train scores shown are {task} minus {size}. "
                + _GATE_RULE.format(none=", and that score carries no size cost",
                                    train=", size cost included,")
                + f" {rest}")
    return f"The score is {task} minus {size}. {rest}"


def objective_lines(parsimony: float | None, *, item_mean: bool = False,
                    gated: bool = False) -> list[str]:
    """"What the score measures" (`objective_text`) as request lines; none when not
    known."""
    text = objective_text(parsimony, item_mean=item_mean, gated=gated)
    policy = compute_policy(parsimony, item_mean=item_mean)
    return ["## What the score measures", text, *([policy] if policy else []), ""] if text else []


def compute_policy(parsimony: float | None, *, item_mean: bool = False) -> str:
    """The run's compute policy, from its scoring settings, for the proposal request: with
    parsimony 0, extra steps and model calls per item are not penalised, so a change that
    adds them is weighed like any other. Empty with a size cost (the size cost lines say
    what a node costs) or when the settings are not known."""
    if parsimony is None or parsimony:
        return ""
    if item_mean:
        return ("Compute policy: this run does not penalise extra steps or model calls per "
                "item. Per-item cost is reported separately and is not part of the score, so "
                "a change that adds steps or model calls is weighed only by the score it "
                "should reach, as any other change is. Cost is not a reason for or against a "
                "change in this run.")
    return ("Compute policy: the loop does not penalise extra steps or model calls per item "
            "(parsimony is 0), so a change that adds them is weighed by the score it should "
            "reach, as any other change is; cost counts only as far as the task's own score "
            "measures it. Cost is not a reason for or against a change in this run.")


def answer_instructions(catalogue: Sequence[dict[str, Any]], *, parsimony: float | None = None,
                        items: int = 0, code_nodes: bool = False) -> str:
    """The answer format and its rules, with examples for what this run permits.

    ``items`` is the number of train items when the raw score is known to be their
    mean (0 otherwise): only then does the size cost line compare a node with one item.
    """
    structural = structural_example(catalogue)
    ops = {op.get("op") for op in catalogue}
    kinds = ["a prompt or setting"] + ([] if structural is None else [
        *(["a new agent step"] if ops & {"insert", "insert_after", "insert_before", "add_node"}
          else []),
        *(["a code step"] if code_nodes else []),
        *(["a gate or loop"] if "loop" in ops else []),
        *(["parallel steps"] if ops & {"parallel", "parallelize"} else []),
        *(["removing, rewiring or merging steps"] if ops & {"remove", "rewire", "merge"}
          else [])])
    lines = [
        "Write one JSON object and nothing else:",
        "",
        '{"diagnosis": "<what the evidence shows is limiting the score>",',
        ' "considered": "<optional, short: the kinds of change you weighed for this '
        f'diagnosis ({", ".join(kinds)}) and why you chose yours>",',
        ' "proposals": [{"edit": {"op": "<operation>", ...},',
        '                "rationale": "<why, citing the evidence>",',
        *(['                "addresses": "<structural: the failure it targets, citing the '
           'steps and items in the evidence by id>",',
           '                "flow_change": "<structural: which step sees what, or what runs '
           'beside or after what, once it is made>",'] if structural is not None else []),
        '                "prediction": "<expected effect: which steps\' outputs or which '
        'items change, and how>",',
        '                "confidence": 0.0}]}',
        "",
        "Rules:",
        '- A parameter move is {"op": "set_knob", "name": "<parameter>", "value": <value>} or, '
        'for a text parameter, the same with "patch" in place of "value" (below).',
    ]
    if any(op.get("op") == "set_knob" for op in catalogue):
        # SkillOpt's patch mode edits its skill only by local append/insert/replace/delete
        # patches; a whole new value stays available for what a patch cannot express.
        lines += ["- To change a text parameter such as a prompt, patch it: "
                  + nn_edits.PATCH_FORMAT + ".",
                  "  Patch the parts the evidence says to change and keep the rest. A whole new",
                  "  `value` is for a setting that is not text, a new step's own prompt, or a",
                  "  prompt that has to be replaced outright."]
    if structural is not None and structural.get("op") == "insert":
        lines += [
            "- A structural move is one line: a structural operator above (or a written-out "
            "operation, with the fields it lists). For example:",
            "  " + json.dumps(structural),
            "  The new step gets its own `<id>__task` knob holding its text, so a later round "
            "can patch or rewrite it with `set_knob`.",
            "- A structural proposal also says what it `addresses` (the failure, citing steps "
            "and items in the evidence by id), its `flow_change` (which step sees what, or "
            "what runs beside or after what) and a `prediction` that can be checked (which "
            "steps' outputs or which items change, and how). Its edit line stays one line; "
            "the thought goes here. A proposal without them is still judged, and marked.",
        ]
    elif structural is not None:
        lines += [
            "- A structural move is one of the structural operations above, with the fields it "
            "lists. A `node` is a node object written from the node schema (or a block, or a "
            "copy of a node's JSON, with a new id). For example:",
            "  " + json.dumps(structural),
            "  A new agent node gets its own `<new step id>__system_prompt` knob holding its "
            "system prompt and `<new step id>__task` knob holding its `prompt_template`, so a "
            "later round can patch or rewrite either with `set_knob`.",
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
        "- Ground each change in the evidence: a setting, a prompt or a structural change, whose",
        "  effect you can state -- the failure it addresses (the step where it starts), how it",
        "  changes what a step sees or how the work is done, and a prediction that can be",
        "  checked. Choose the kind of change by the cause you diagnosed; neither kind is the",
        "  safer default. Vary what you touch.",
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
    patch = [{"kind": "replace", "target": "<exact text in it>", "content": "<new text>"}]
    out = [[{"op": "set_knob", "name": "<step A's prompt knob>", "patch": patch},
            {"op": "set_knob", "name": "<step B's prompt knob>", "patch": patch}]]
    structural = structural_example(catalogue)
    ops = {op.get("op") for op in catalogue}
    if structural is not None and "node" in structural and "set_knob" in ops:
        # The node leaves its system prompt to the knob the bundle sets.
        node = {k: v for k, v in structural["node"].items() if k != "system_prompt_template"}
        out.append([{**structural, "node": node},
                    {"op": "set_knob", "name": "<new step id>__system_prompt",
                     "value": "<system prompt text>"}])
    elif structural is not None and "set_knob" in ops:
        # A new step, and how a later step is told to use what it makes.
        out.append([structural, {"op": "set_knob", "name": "<a later step's prompt knob>",
                                 "patch": [{"kind": "append",
                                            "content": "<how to use the new step's output>"}]}])
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
failure path (up to {max_edits} changes), so the fix is evaluated as a whole; changes that
only work together each fail when tried alone."""
    else:
        scope = """\
Bundle only what belongs to one idea: separate ideas go in separate proposals, so each
is judged on its own."""
    return f"""\
Bundling: when one fix needs coordinated changes -- several steps' prompts that have
to agree, or a new node and the knob that feeds it -- one proposal may carry up to
{max_edits} changes as an ordered list of edits under "edits" instead of "edit". {COUNTING_RULE}
They are applied in order to the same parent, each to the state the one before it left,
and the result is evaluated and judged once as one candidate; if any edit is refused,
the whole proposal is, and the refusal names the edit.

{scope}

Example:

{example}"""


#: How many error causes and wrong answers the item summary lists, and how much of
#: each it quotes: enough to see a pattern, bounded so a large part stays a section.
ITEM_EVIDENCE_ROWS = 6
ITEM_EVIDENCE_CHARS = 240


#: The most of an answer's ``considered`` kept in the journal, the trace and reports.
CONSIDERED_CHARS = 1500


def considered_text(value: Any) -> str:
    """An answer's optional ``considered`` as one bounded text: a string as it is, a list
    one entry per line, an object as JSON; empty when it is missing or empty. Never an
    error: the field is optional, so a missing or odd one costs the proposal nothing."""
    if value is None or value == "":
        return ""
    if isinstance(value, (list, tuple)):
        text = "\n".join(str(entry if not isinstance(entry, (dict, list)) else
                              json.dumps(entry, default=str)) for entry in value)
    elif isinstance(value, dict):
        text = json.dumps(value, default=str)
    else:
        text = str(value)
    text = text.strip()
    return text if len(text) <= CONSIDERED_CHARS else text[: CONSIDERED_CHARS - 1] + "…"


def _clip(text: Any, limit: int = ITEM_EVIDENCE_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


#: How the optimizer is told why a step gave no answer (`items.missing_steps` causes).
NO_ANSWER_CAUSES = {
    "output_limit": "ran out of output without answering",
    "crash": "crashed in every session",
    "timeout": "timed out",
    "failed": "failed (optional branch)",
}


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
    unanswered = [r for r in records if not r.get("error_text") and not r.get("passed")
                  and r.get("missing_steps")]
    wrong = [r for r in records if not r.get("error_text") and not r.get("passed")
             and not r.get("missing_steps")]
    scores = [float(r["score"]) for r in records if isinstance(r.get("score"), (int, float))]
    mean = f", mean score {sum(scores) / len(scores):.3g}" if scores else ""
    lines = [
        "## Item results of the candidate being edited",
        f"{len(records)} train item(s): {len(passed)} passed, {len(wrong)} answered "
        f"wrong, " + (f"{len(unanswered)} had no answer (a step ran out of output or "
                      "crashed), " if unanswered else "")
        + f"{len(errored)} failed with an error{mean}. An error is the item's run failing "
        "(its cause is below), not a wrong answer.",
    ]
    if unanswered:
        rows = [[r.get("id"), step, NO_ANSWER_CAUSES.get(str(info.get("cause")), info.get("cause")),
                 info.get("sessions")]
                for r in unanswered for step, info in sorted((r.get("missing_steps") or {}).items())
                if isinstance(info, dict)]
        lines += [
            "",
            "No answer -- not a wrong answer: a step gave none, so the item was graded "
            "without it. Running out of output is the model's own failure on the question "
            "and is not retried; a crash is the worker's, retried in fresh sessions first:",
            _table(["item", "step", "cause", "sessions"], rows[:ITEM_EVIDENCE_ROWS]),
        ]
        if len(rows) > ITEM_EVIDENCE_ROWS:
            lines.append(f"({len(rows) - ITEM_EVIDENCE_ROWS} more not shown)")
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


#: What an item's text clipped in a request says in its place: the count only, never a
#: path (the proposer has no tools); the full record goes to ``ProposalContext.omissions``.
OMITTED = "[… {chars:,} chars omitted]"


def _marked(text: Any, limit: int, *, item: Any = None, step: str = "",
            omissions: list[dict[str, Any]] | None = None) -> str:
    """``text`` whole when it fits ``limit``, else its head and tail around an
    `OMITTED` mark; each cut is added to ``omissions`` (item, step, chars)."""
    text = ("" if text is None else str(text)).strip()
    if len(text) <= limit:
        return text
    keep = max(0, limit // 2)
    chars = len(text) - 2 * keep
    if omissions is not None:
        omissions.append({"item": item, "step": step, "chars": chars})
    return f"{text[:keep]}\n{OMITTED.format(chars=chars)}\n{text[-keep:]}"


def _demoted(text: str) -> str:
    """``text`` with each Markdown heading line pushed below the request's own levels, so
    an output's "## Answer" cannot read as one of the request's sections."""
    return re.sub(r"(?m)^(#{1,3})(?=\s)", lambda m: "#" * (len(m.group(1)) + 3), text)


def _fenced(text: str) -> list[str]:
    """``text`` (headings demoted) in a code fence longer than any run of backticks in it."""
    body = _demoted(text)
    fence = _fence(body)
    return [fence, body, fence]


def _trajectory(record: dict[str, Any], text_chars: int = _WHOLE,
                omissions: list[dict[str, Any]] | None = None) -> str:
    """One train item: its verdict, what was expected and what it gave, then its input and
    each step's output, each fenced (headings demoted) so nothing in them reads as a
    section of the request.

    The steps are in the order the record holds them, which is the order the pass ran them
    (a fork merges its branches in branch order), never sorted by name. Texts are whole
    unless ``text_chars`` clips them (head and tail, marked; each cut in ``omissions``).
    """
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    exchanges = [e for e in record.get("exchanges") or () if isinstance(e, dict)]
    from .analysts import metadata_text

    item = record.get("id")

    def cut(value: Any, step: str) -> str:
        return _marked(value, text_chars, item=item, step=step, omissions=omissions)

    declared = metadata_text(record)
    lines = [f"### Item {item} — {_outcome(record)} (score {record.get('score')})"
             + (f" — {declared}" if declared else "")]
    expected = _expected(record)
    said = [part for part in (
        f"Expected: {cut(expected, 'expected')}" if expected else "",
        f"Given: {cut(record.get('prediction'), 'prediction')}"
        if record.get("prediction") is not None else "") if part]
    if said:
        lines.append("; ".join(said))
    if record.get("fail_reason") and not record.get("passed"):
        lines.append(f"Grader: {cut(record['fail_reason'], 'grader')}")
    if record.get("error_text"):
        lines.append(f"Error: {cut(record['error_text'], 'error')}")
    key = next((k for k in TRAJECTORY_INPUT_KEYS if meta.get(k)), None)
    choices = _choice_lines(meta.get("choices"))
    if key is not None:
        text = cut(meta[key], "input")
        if choices:
            text += "\n\nChoices:\n" + "\n".join(cut(line, "choices") for line in choices)
        lines += [f"Input ({key}):", *_fenced(text)]
    elif exchanges:
        lines += ["Input (the first step's prompt):",
                  *_fenced(cut(exchanges[0].get("prompt"), "input"))]
    for exchange in exchanges:
        node = exchange.get("node") or "agent"
        lines += [f"Step `{node}` output:", *_fenced(cut(exchange.get("response"), str(node)))]
    if not exchanges:
        lines.append("(no step output was recorded)")
    return "\n".join(lines)


def _outcome(record: dict[str, Any]) -> str:
    return ("passed" if record.get("passed") else
            "failed with an error" if record.get("error_text") else "answered wrong")


def _expected(record: dict[str, Any]) -> str:
    label, label_text = record.get("label"), record.get("label_text")
    return " — ".join(str(part) for part in (label, label_text) if part not in (None, ""))


def _item_block(record: dict[str, Any], budget: int,
                omissions: list[dict[str, Any]] | None = None) -> str:
    """An item in full; clipped (each long text head and tail, marked) only when the item
    alone is longer than ``budget``, each cut added to ``omissions``."""
    whole = _trajectory(record)
    if len(whole) <= budget:
        return whole
    low, high = 1, max(1, budget)
    while low < high:  # the largest per-text limit that fits
        middle = (low + high + 1) // 2
        if len(_trajectory(record, middle)) <= budget:
            low = middle
        else:
            high = middle - 1
    return _trajectory(record, low, omissions)


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


#: Per reader, the success patterns kept whatever its failure patterns (at most this
#: many); failure patterns fill the rest of ``READER_PATTERNS``.
READER_SUCCESS_PATTERNS = 4
#: The share of the findings' room held for success patterns when a reader found some.
READER_SUCCESS_SHARE = 0.25


def _reader_patterns(findings: Sequence[dict[str, Any]] | None) -> list[tuple[str, dict[str, Any]]]:
    """The readers' patterns the request shows, as ``(kind, pattern)`` in their order:
    reader by reader, failure patterns then success patterns, at most
    ``READER_PATTERNS`` per reader, of which up to ``READER_SUCCESS_PATTERNS`` success
    patterns are kept however many failure patterns the reader found."""
    out: list[tuple[str, dict[str, Any]]] = []
    for found in findings or ():
        kept = {kind: [p for p in found.get(kind) or () if isinstance(p, dict)]
                for kind in ("failure_patterns", "success_patterns")}
        successes = kept["success_patterns"][:READER_SUCCESS_PATTERNS]
        failures = kept["failure_patterns"][:READER_PATTERNS - len(successes)]
        out += [("failure_patterns", p) for p in failures]
        out += [("success_patterns", p) for p in successes]
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


#: Whether every train item is shown in full or the rule (`_shown_items`) picks them:
#: every item when all of them together are under this many characters. It only decides
#: that switch; it never cuts an item.
SHOW_ALL_BELOW = 120000
#: A single text of an item shown in full (a step's reply, the input) is cut only past
#: this many characters -- one text too large to send whole; the items themselves are
#: never cut to fit a size.
ITEM_TEXT_MAX = 60000


def _contrast_items(items: Sequence[dict[str, Any]], shown: Sequence[dict[str, Any]],
                    findings: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Passing items to show beside the failures: the success findings' items first, then
    passes of the same declared kind as a failure shown (a near miss), then the rest."""
    from .analysts import metadata_text

    passing = [r for r in items if r.get("passed")]
    cited = [r for r in _cited_items(passing, [{"success_patterns": [
        p for f in findings or () for p in f.get("success_patterns") or ()]}])] if findings else []
    kinds = {metadata_text(r) for r in shown if metadata_text(r)}
    near = [r for r in passing if metadata_text(r) in kinds]
    out: list[dict[str, Any]] = []
    for record in [*cited, *near, *passing]:
        if all(record is not r for r in out):
            out.append(record)
    return out


def _failure_kind(record: dict[str, Any]) -> str:
    """What kind of failure an item is, for choosing items without findings: its outcome
    and its declared kind (`analysts.metadata_text`), when it has one."""
    from .analysts import metadata_text

    return f"{_outcome(record)} {metadata_text(record)}".strip()


def _shown_items(items: Sequence[dict[str, Any]], findings: Sequence[dict[str, Any]] | None
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """The items shown whole when not all are, by rule, never by size: one failure per
    failure finding that cites a known item, in rank order (an item already chosen is not
    chosen twice); then one contrasting success per success finding -- or, with none, one
    per kind of failure shown (`_failure_kind`), the same declared kind first. Without
    findings, one failure per kind of failure. The last value says whether the failures
    were chosen by findings."""
    known = {str(r.get("id")) for r in items}
    by_id = {str(r.get("id")): r for r in items}
    patterns = [(kind, p) for kind, p in _reader_patterns(findings)
                if known & {str(i) for i in p.get("items") or ()}]
    failing: list[dict[str, Any]] = []
    for kind, pattern in patterns:
        if kind != "failure_patterns":
            continue
        pick = next((by_id[str(i)] for i in pattern.get("items") or ()
                     if str(i) in by_id and not by_id[str(i)].get("passed")
                     and all(by_id[str(i)] is not r for r in failing)), None)
        if pick is not None:
            failing.append(pick)
    by_findings = bool(failing)
    if not failing:
        kinds: dict[str, dict[str, Any]] = {}
        for record in items:
            if not record.get("passed"):
                kinds.setdefault(_failure_kind(record), record)
        failing = list(kinds.values())
    contrast = _contrast_items(items, failing, findings)
    n_pass = sum(1 for kind, _p in patterns if kind == "success_patterns")
    if n_pass:
        passing = contrast[:n_pass]
    else:
        from .analysts import metadata_text

        passing = []
        for kind in dict.fromkeys(metadata_text(r) for r in failing):
            pick = next((r for r in contrast if metadata_text(r) == kind
                         and all(r is not p for p in passing)), None) or next(
                (r for r in contrast if all(r is not p for p in passing)), None)
            if pick is not None:
                passing.append(pick)
    return failing, passing, by_findings


def _trajectories(records: Sequence[dict[str, Any]], *, budget: int = TRAJECTORY_BUDGET,
                  findings: Sequence[dict[str, Any]] | None = None,
                  failures: Sequence[dict[str, Any]] = (),
                  analysed: Sequence[dict[str, Any]] | None = None,
                  omissions: list[dict[str, Any]] | None = None) -> list[str]:
    """Every train item of the candidate being edited.

    When all of them together are under ``SHOW_ALL_BELOW`` characters (or the call's
    smaller ``budget``), every item is shown in full here. When not, readers went through them in groups of ``budget``
    (``AutoresearchOptimizer``): their findings come first, then a table of every item,
    then the items the rule picks (`_shown_items`), each whole. Train records only.

    ``analysed``: the analysts' ranked findings (`optim.analysts`), shown before this
    section, which take the readers' place: no reader findings are shown, and the
    items they cite are shown in full first (one per finding in turn, failure findings
    first, each kind in rank order).
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
    whole = [_trajectory(r) for r in items]
    if sum(len(block) + 2 for block in whole) <= min(budget, SHOW_ALL_BELOW):
        return header + ["Every item in full. Read what each step produced and where it "
                         "went wrong before proposing.",
                         *(line for r in items
                           for line in ("", _trajectory(r, ITEM_TEXT_MAX, omissions)))]
    if analysed is not None:
        # The analysts' findings are their own section, before this one; the items they
        # cite are shown in full first (``_cited_items``).
        lines = header + [
            "They do not fit one request in full. The analysts' ranked findings above "
            "cover them; here is a table of every item, then as many items in full as fit."]
        findings = [{"failure_patterns": [f for f in analysed if f.get("source") != "success"],
                     "success_patterns": [f for f in analysed if f.get("source") == "success"]}]
    else:
        lines = header + [
            f"They do not fit one request in full, so readers went through them in "
            f"{len(groups)} groups of whole items. Their findings come first, then a table "
            "of every item, then as many items in full as fit."]
    if analysed is None and findings is None:
        lines.append("(No reader went through these items for this request: read the "
                     "table and the items in full below.)")
    elif analysed is None:
        lines += ["", "### What the readers found"]
        found_lines: list[str] = []
        success_lines: list[str] = []
        for kind, pattern in _reader_patterns(findings):
            ids = [str(i) for i in pattern.get("items") or ()]
            id_text = (", ".join(ids[:READER_ITEM_IDS]) or "none in its group") + (
                f" and {len(ids) - READER_ITEM_IDS} more" if len(ids) > READER_ITEM_IDS else "")
            stray = [str(i) for i in pattern.get("not_in_group") or ()]
            if stray:
                id_text += (f"; it also named {', '.join(stray[:READER_ITEM_IDS])}, not an item "
                            "of its group, so dropped")
            quotes = [f'"{_clip(quote, READER_QUOTE_CHARS)}"'
                      for quote in list(pattern.get("evidence") or ())[:READER_QUOTES]]
            (success_lines if kind == "success_patterns" else found_lines).append(
                f"- [{kind.split('_')[0]}] {_clip(pattern.get('pattern', ''), READER_PATTERN_CHARS)} "
                f"(step `{_clip(pattern.get('step', 'unclear'), 60)}`; items {id_text}) "
                "-- evidence: " + " | ".join(quotes))
        cap = int(budget * READER_FINDINGS_SHARE)
        # What the passing items do right gets its own heading after the failures, and
        # room the failures cannot take: up to READER_SUCCESS_SHARE of the cap.
        held = min(sum(len(line) + 1 for line in success_lines), int(cap * READER_SUCCESS_SHARE))
        used_findings = 0
        for heading, kind_lines, room in (
                ("", found_lines, cap - held),
                ("### What the passing items do right (the readers' success patterns)",
                 success_lines, cap)):
            if heading and kind_lines:
                lines += ["", heading]
            for index, line in enumerate(kind_lines):
                if used_findings + len(line) + 1 > room:
                    lines.append(f"({len(kind_lines) - index} more pattern(s) not shown: the "
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
    # Items in full, chosen by rule (`_shown_items`) rather than cut, each whole.
    failing, passing, cited = _shown_items(items, findings)
    chosen = [*failing, *passing]
    if not chosen:
        return lines
    examples = ["", "### Some items in full",
                f"{len(failing)} failing item(s)"
                + (f" the {'analysts' if analysed is not None else 'readers'} cited, "
                   "one per finding in rank order" if cited else ", one per kind of failure")
                + (f", then {len(passing)} passing item(s) for contrast" if passing else "")
                + f", each whole (of {len(items)}; the table above has every one)."]
    for record in chosen:
        if record.get("passed"):
            examples += ["", "(A success, for contrast.)"]
        examples += ["", _trajectory(record, ITEM_TEXT_MAX, omissions)]
    return lines + examples


def _edit_fields(fields: Sequence[str]) -> str:
    """An operation's fields, saying which take a whole node object."""
    described = []
    for name in fields:
        if name == "into":
            described.append("into: a list of two full node objects (see the node schemas below)")
        elif name == "branches":
            described.append("branches: a list of 2-8 full node objects (see the node schemas below)")
        elif name in NODE_OBJECT_FIELDS:
            described.append(f"{name}: a full node object (see the node schemas below)")
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


def available_before(graph_state: Mapping[str, Any],
                     knob_values: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """For each node a run can reach, the facts and files there before it starts on every
    path that reaches it (`nn.validate.available_inputs`, the sets rule V15 checks
    against), with who makes each one (``made_by``: name -> step ids, `nn.operators.made_by`)
    and which carry no model output (``task_inputs``, `nn.operators.model_free_inputs`;
    ``knob_values`` fills the slots of bound prompts); empty when the graph does not parse."""
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.operators import made_by, model_free_inputs
    from refactory_dsh.nn.validate import available_inputs

    try:
        graph = Graph.from_state(dict(graph_state))
    except Exception:  # a broken graph is the validator's to report
        return {}
    have = available_inputs(graph)
    makers = made_by(graph)
    try:
        free = model_free_inputs(graph, knob_values)
    except Exception:  # a node that cannot be read is the validator's to report
        free = set()
    out: dict[str, dict[str, Any]] = {}
    for node_id in sorted(graph.reachable()):
        entries = have.get(node_id, ())
        out[node_id] = {
            "facts": sorted(n for k, n in entries if k == "fact"),
            "files": sorted(n for k, n in entries if k == "file"),
            "made_by": {n: makers.get((k, n), []) for k, n in sorted(entries)},
            "task_inputs": sorted(n for k, n in entries if (k, n) in free),
        }
    return out


#: The most characters the "what each node has" table takes: past it, each cell lists
#: fewer names (and says how many more), then rows are left out, last first, and named.
AVAILABLE_SECTION_CHARS = 8000


def _available_section(available: Mapping[str, Mapping[str, Any]], *,
                       code_nodes: bool = False,
                       limit: int = AVAILABLE_SECTION_CHARS) -> list[str]:
    """What each node can use: the facts a `{{name}}` may name there, and the files already
    written, each with the step that makes it, the ones that carry no model output marked
    "(task input)", and how a file's text reaches a prompt. Fits ``limit``. Nothing when
    not known."""
    from refactory_dsh.nn.operators import COLLECT_CALL, COLLECT_TEXT_CHARS, FILE_TEXT_CALL

    if not available:
        return []
    how = ("a code op (`\"call\": \"refactory_dsh.snippet:run\"`) that reads the file and "
           "`produces` the fact" if code_nodes else "an op that reads the file and `produces` "
           "the fact")
    lead = (
        "What each node has before it runs (on every path that reaches it), each with the "
        "step that makes it in parentheses. \"(task input)\" marks one that no model step's "
        "output reaches: the task's own inputs, and what plain ops make from them alone, so a "
        "step given only these sees the task and none of the other steps' work. A step's "
        "input is not its output: name the fact or file the step makes. A prompt's "
        "`{{name}}` may name only a fact listed for its node (or a knob): a file's name is "
        "not a fact, and `{{notes}}` does not read `notes.txt`. An agent without tools sees "
        "its prompt and the files it lists in `reads`, pasted in after the prompt. To hand "
        "files to a prompt as one `{{name}}`, insert a collect op before the node "
        f"(`\"call\": \"{COLLECT_CALL}\"`, the files in `reads`, the fact in `produces`: "
        "each file under its own header, a missing one said to be missing; one file over "
        f"{COLLECT_TEXT_CHARS} characters is shown as its path and size), or a file_text op "
        f"for one file's whole text (`\"call\": \"{FILE_TEXT_CALL}\"`), or "
        f"{how}, or have the agent that writes the file declare `produces`, which makes its "
        "reply that fact too.")

    def named(entry: Mapping[str, Any], name: str) -> str:
        if name in (entry.get("task_inputs") or ()):
            return f"{name} (task input)"
        makers = (entry.get("made_by") or {}).get(name) or ()
        return f"{name} ({', '.join(makers)})" if makers else name

    def cell(entry: Mapping[str, Any], key: str, keep: int | None) -> str:
        names = [named(entry, name) for name in entry.get(key) or ()]
        if not names:
            return "-"
        if keep is None or len(names) <= keep:
            return ", ".join(names)
        return ", ".join(names[:keep]) + f" (+{len(names) - keep} more)"

    def table(rows: Sequence[tuple[str, Mapping[str, Any]]], keep: int | None) -> str:
        return _table(["node", "facts available (made by)", "files written before it (made by)"],
                      [[node_id, cell(entry, "facts", keep), cell(entry, "files", keep)]
                       for node_id, entry in rows])

    rows = sorted(available.items())
    room = max(0, limit - len(lead))
    for keep in (None, 12, 6, 3, 1, 0):
        text = table(rows, keep)
        if len(text) <= room:
            return [lead, text]
    while rows:
        left = len(available) - len(rows)
        note = f"({left} node(s) not listed to stay within {limit} characters)" if left else ""
        text = table(rows, 0)
        if len(text) + len(note) + 1 <= room:
            return [lead, text, *([note] if note else [])]
        rows = rows[:-1]
    return [lead, f"({len(available)} node(s) not listed to stay within {limit} characters)"]


def _node_specs(states: Sequence[dict[str, Any]], *, text_chars: int = NODE_SPEC_TEXT_CHARS,
                section_chars: int = NODE_SPEC_SECTION_CHARS,
                knob_values: Mapping[str, Any] | None = None,
                person_gates: bool = True) -> list[str]:
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
    schema = render_schema(person_gates=person_gates)
    ordered = sorted(states, key=lambda s: str(s.get("id")))

    def section(chosen: Sequence[dict[str, Any]], limit: int) -> list[str]:
        lines = ["### Each node in full",
                 "Copy a node object, give it a new id, change what you mean to change "
                 "(prompt, reads, writes, produces, knobs), and use it as the `node` of an "
                 "`add_node`, `insert_after`, `insert_before` or `wrap` edit. Knob bindings "
                 "are left out of these copies: a new agent node gets its own "
                 "`<id>__system_prompt` knob holding its `system_prompt_template` and its own "
                 "`<id>__task` knob holding its `prompt_template`, and a "
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


def _current_line(history: Sequence[dict[str, Any]]) -> str:
    """What "What this run has already tried" says of the validation gate's choice: the
    train score of the candidate marked ``current`` (a train score, labelled so; the
    val score that made it current is never shown)."""
    current = next((c for c in history if c.get("gate") == "current"), None)
    if current is None:
        return ("none judged by the validation gate yet"
                if all(c.get("gate") in (None, "not gated") for c in history)
                else "the validation gate's current candidate is older than this list")
    score = current.get("score")
    return ("train score of the validation gate's current candidate "
            + (f"{score:.6g}" if isinstance(score, (int, float)) else "n/a (its train run failed)"))


def _gate_note(*tables: Sequence[dict[str, Any]], val_items: int = 0,
               selection: str = "", population: bool = False) -> list[str]:
    """What the gate column means, said once, when there is one.

    A rejection says only that the candidate was not shown better on validation; with
    ``val_items`` (the validation items the gate's score is the mean of) the note says
    what one item is worth, so a one-item loss is read as the gate's resolution. Under
    ``selection`` ``pool`` it says what the pool is instead of SkillOpt's single line."""
    if not any(_gate_column(rows) for rows in tables):
        return []
    resolution = (f"The gate's score is the mean over {val_items} validation items, so one "
                  f"item is 1/{val_items} = {1 / val_items:.3g} of it, and a rejection can come "
                  "down to a single item. " if val_items > 0 else "")
    if selection == "pool":
        # The same labels as ``single`` (the pool's own membership follows the val
        # ranking, so it is never shown): what changes is that a parent may be any
        # candidate that scored well there (with a population picker, any judged
        # candidate), a rejected one included.
        return ["This run has a validation gate with a pool of parents: each candidate "
                "was also run on held-out items, and the gate kept it only if it scored "
                "strictly higher there than the current candidate (an equal score is "
                "rejected); the best so far is marked `current`. "
                + ("A parent may be any candidate judged so far, a `rejected` one included, "
                   "as the parent picker chooses. " if population else
                   "A parent may be any of the candidates that scored well there, a "
                   "`rejected` one included. ")
                + "Rows are not in order of the held-out scores. Scores here are train "
                "scores; held-out scores are never shown. You edit the pipeline shown below. A `rejected` "
                "candidate was not shown better than the current one on validation; that is "
                "all a rejection says. " + resolution + "A change that was rejected must not "
                "be proposed again unchanged, but when its train fixed/broken counts against "
                "its parent (when shown below) were strongly positive, its direction may still "
                "be right: you may build on it or combine it with edits to other steps in the "
                "same failure path."]
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


#: How much of the first error of each kind a summary of a candidate's item errors keeps.
ITEM_ERROR_CHARS = 300
#: One ``<name>: `` step of an item error's ``<node>: <Kind>: ...`` chain.
_ERROR_STEP = re.compile(r"([A-Za-z_][\w.-]*): ")
#: A name in that chain that is an exception class, not a node id.
_ERROR_KIND = re.compile(r"[A-Z][A-Za-z0-9_]*(?:Error|Exception|Timeout|Expired|Exit|Interrupt"
                         r"|Stop|Failure|Iteration|Warning)")
#: What the node runner (``refactory_dsh.run.FAILURE_MARKER``) prints before its own
#: account of a failure, after the same text on stderr.
_FAILURE_MARKER = "--- refactory failure ---"


def _error_origin(text: str) -> tuple[str, str, str]:
    """``(node, kind, message)`` of an item error, read from its first line, where both
    runtimes say what failed: ``<node>: <Kind>: <message>`` (the walker's
    ``nn.executor``; the spine's ``nodeErrorText`` puts that line before the node's
    stderr tail). A fork nests its branch's error (``fork1: RuntimeError: inner:
    SnippetError: y``), so the innermost node and kind are the origin. A message that
    starts with a kind has no node (``ValueError: Expected: 3`` is a ValueError); one
    that starts with a name and no kind is that node's (``decide: worker failed``). A
    raw stderr tail with the node runner's failure marker and no such line (an older
    spine record) is read from the runner's own account after the marker. Empty node or
    kind when the text names none.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    node, kind, message = _error_line(lines[0] if lines else "")
    if not kind and _FAILURE_MARKER in lines:
        account = lines[len(lines) - 1 - lines[::-1].index(_FAILURE_MARKER) + 1:]
        if account:
            node, kind, message = _error_line(account[0])
    return node, kind, message


def _error_line(text: str) -> tuple[str, str, str]:
    """``(node, kind, message)`` of one ``<node>: <Kind>: <message>`` line."""
    node = kind = ""
    pos = 0
    while step := _ERROR_STEP.match(text, pos):
        name = step.group(1)
        if _ERROR_KIND.fullmatch(name):
            kind, pos = name, step.end()
            continue
        then = _ERROR_STEP.match(text, step.end())
        if then and _ERROR_KIND.fullmatch(then.group(1)):
            node, kind, pos = name, then.group(1), then.end()
        elif pos == 0:
            node, pos = name, step.end()
        else:
            break
    return node, kind, text[pos:]


def item_errors(records: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """A candidate's item errors -- items whose run failed, not wrong answers -- counted
    by kind and the node that raised them, most frequent first: ``{"kind", "node",
    "count", "item", "first"}``, with the first such item and its message (clipped to
    ``ITEM_ERROR_CHARS``). Kind and node come from the error's ``<node>: <Error>:
    <message>`` text when it has that shape (so a node's ``SnippetError`` wrapped in a
    ``RuntimeError`` counts as a ``SnippetError`` at that node; see :func:`_error_origin`),
    else the record's ``error_kind``."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records or ():
        text = str(record.get("error_text") or "").strip()
        if not text:
            continue
        node, kind, message = _error_origin(text)
        kind = kind or str(record.get("error_kind") or "error")
        group = groups.setdefault((kind, node), {
            "kind": kind, "node": node, "count": 0, "item": str(record.get("id", "")),
            "first": _clip(message, ITEM_ERROR_CHARS)})
        group["count"] += 1
    return sorted(groups.values(), key=lambda group: -group["count"])


def _item_error_lines(errors: Sequence[dict[str, Any]], items: int) -> list[str]:
    """A candidate's item errors (``item_errors``) as a request shows them; none when
    it had none."""
    if not errors:
        return []
    total = sum(int(error["count"]) for error in errors)
    lines = [f"Item errors: {total} of {items} train item(s) failed with an error -- the "
             "item's run failed, so it scored 0; not a wrong answer:"]
    for error in errors:
        where = f" at node {error['node']}" if error["node"] else ""
        lines.append(f"- {error['count']} x {error['kind']}{where}; first (item "
                     f"{error['item']}): {error['first']}")
    return lines


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
#: The fields of an edit, and of a node it carries, that say how it is wired: shown
#: whole wherever past edits are listed, however long. Only the other text (prompts,
#: code, knob values, descriptions) is shortened, so a reader never mistakes a step's
#: cut-off ``reads`` for a step that reads nothing. They are wiring only at an edit's
#: top level and inside the node objects it carries (``EDIT_NODE_FIELDS``): under a
#: ``value``, ``patch``, ``config``, ``spec`` or ``code`` every string is free text (a
#: patch entry's ``target`` is the text it replaces).
EDIT_WIRING_FIELDS = frozenset((
    *EDIT_NAMING_FIELDS, "ids", "kind", "reads", "writes", "produces", "inputs",
    "required_reads", "optional_targets", "call", "evaluator", "gate_kind",
    "max_iterations", "on_exhausted", "targets", "sources", "edges", "start", "knobs",
    "parallelism", "parallelism_knob", "feedback_keys", "task_ref", "tools", "blocking",
    "subgraph_entry", "subgraph_exit", "path", "role", "source_path"))
#: The fields of an edit or node that hold node objects (one, or a list).
EDIT_NODE_FIELDS = frozenset((*NODE_OBJECT_FIELDS, "nodes"))
#: A text the budget would cut to fewer characters than this is shown as its length
#: marker alone ("… (N chars)").
EDIT_TEXT_MIN = 20


def _map_text(value: Any, fn: Any, structured: bool = True) -> Any:
    """``value`` with ``fn`` applied to each free-text string. In an edit or a node
    (``structured``) the wiring fields stay as they are and node fields are walked as
    nodes; everything else, at any depth, is free text."""
    if isinstance(value, (list, tuple)):
        return [_map_text(item, fn, structured) for item in value]
    if isinstance(value, str):
        return fn(value)
    if not isinstance(value, dict):
        return value
    if not structured:
        return {key: _map_text(item, fn, False) for key, item in value.items()}
    return {key: item if key in EDIT_WIRING_FIELDS
            else _map_text(item, fn, key in EDIT_NODE_FIELDS)
            for key, item in value.items()}


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit < EDIT_TEXT_MIN:
        return f"… ({len(text)} chars)"
    return f"{text[:limit]}… ({len(text) - limit} more chars)"


def _text_limit(lengths: Sequence[int], budget: int) -> int:
    """The longest any one text may stay so all of them together fit ``budget``: short
    texts stay whole and the long ones share what is left equally, however little
    that is (``_cut`` shows a share under ``EDIT_TEXT_MIN`` as a marker alone)."""
    remaining = max(0, budget)
    ordered = sorted(lengths)
    for index, length in enumerate(ordered):
        share = remaining // (len(ordered) - index)
        if length > share:
            return share
        remaining -= length
    return max(ordered, default=0)


def fit_edit_text(value: Any, budget: int, cap: int | None = None) -> Any:
    """An edit (or a list of edits) with its free text shortened to fit ``budget``
    characters in all (and no one text past ``cap``), and its structure and wiring
    kept whole however long they are."""
    lengths: list[int] = []
    _map_text(value, lambda text: lengths.append(len(text)) or text)
    limit = _text_limit(lengths, budget)
    limit = limit if cap is None else min(cap, limit)
    return _map_text(value, lambda text: _cut(text, limit))


def describe_edits(edits: Any, field_chars: int = EDIT_FIELD_CHARS,
                   text_budget: int | None = None) -> str:
    """A candidate's edits as a reader needs them: one line per edit, its op and what it
    names first, whole, then each other field with its wiring whole (every node's id,
    kind, reads, writes, ...; ``EDIT_WIRING_FIELDS``) and only its free text shortened,
    so no edit of a compound proposal, and no step of a split, is cut away by a long
    prompt before it. The free text of all the edits shares ``text_budget`` characters
    (by default ``field_chars`` per field)."""
    steps = list(edits) if isinstance(edits, (list, tuple)) else [edits]
    rests = [{key: value for key, value in edit.items() if key not in EDIT_NAMING_FIELDS}
             if isinstance(edit, dict) else edit for edit in steps]
    if text_budget is None:
        text_budget = field_chars * sum(max(1, len(rest)) if isinstance(rest, dict) else 1
                                        for rest in rests)
    shown = fit_edit_text(rests, text_budget)
    lines = []
    for index, (edit, rest) in enumerate(zip(steps, shown), 1):
        if not isinstance(edit, dict):
            lines.append(f"{index}. {_clip(json.dumps(edit, default=str), field_chars)}")
            continue
        head = " ".join(f"{key}={json.dumps(edit[key], default=str)}"
                        for key in EDIT_NAMING_FIELDS if key in edit)
        tail = "; ".join(f"{key}: {json.dumps(value, default=str, ensure_ascii=False)}"
                         for key, value in rest.items())
        lines.append(f"{index}. {head}" + (f" -- {tail}" if tail else ""))
    return "\n".join(lines) if lines else "(none)"


#: How much of one edit "What this run has already tried" shows: a ``set_knob``'s
#: diff against the value it replaced, a patch's ops, a structural edit's free text
#: (its wiring is always whole).
HISTORY_EDIT_CHARS = 600
#: How much of any one text (a node's prompt) a structural edit in that table shows.
HISTORY_NODE_PROMPT_CHARS = 200
_UNKNOWN = object()


def _bounded(text: str, limit: int = HISTORY_EDIT_CHARS) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text) - limit} more chars)"


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
    patch's ops; a structural edit's fields with every node's wiring whole and its free
    text (prompts, code) sharing ``HISTORY_EDIT_CHARS`` characters (``fit_edit_text``).
    A diff, value or patch is at most ``HISTORY_EDIT_CHARS`` characters plus a "(N more
    chars)" marker. The "Each step's settings" table already shows the current
    values in full."""
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
            parts.append(json.dumps(fit_edit_text(edit, HISTORY_EDIT_CHARS,
                                                  cap=HISTORY_NODE_PROMPT_CHARS),
                                    default=str, ensure_ascii=False))
    return "; ".join(parts) if parts else "(none)"


#: How many named items and steps a prediction check lists.
PREDICTION_NAMED = 8


def _named(text: str, names: Iterable[str]) -> list[str]:
    """The ``names`` a text mentions as whole words, in the order given."""
    return [name for name in names if name and re.search(
        r"(?<![\w.-])" + re.escape(str(name)) + r"(?![\w-])", text)]


def _step_replies(record: Mapping[str, Any]) -> dict[str, str]:
    return {str(e.get("node")): str(e.get("response") or "") for e in record.get("exchanges") or ()
            if isinstance(e, Mapping) and e.get("node")}


def prediction_check(prediction: str, parent_records: Sequence[dict[str, Any]] | None,
                     child_records: Sequence[dict[str, Any]] | None,
                     steps: Sequence[str] = ()) -> list[str]:
    """What happened to the items and steps a prediction names, measured on the train items
    parent and child share: each named item's outcome (fixed, broken, score up or down,
    unchanged), and for each named step how many items its reply changed on (or ran on, for
    a step only one of them has). Mechanical: whether the prediction held is the reader's
    call. Empty when the prediction names neither."""
    text = str(prediction or "")
    parent, child = _by_item(parent_records), _by_item(child_records)
    shared = sorted(set(parent) & set(child))
    if not text.strip() or not shared:
        return []
    lines: list[str] = []
    items = _named(text, shared)[:PREDICTION_NAMED]
    if items:
        said = []
        for item in items:
            before, after = parent[item], child[item]
            if not before.get("passed") and after.get("passed"):
                said.append(f"{item} fixed")
            elif before.get("passed") and not after.get("passed"):
                said.append(f"{item} broken")
            else:
                delta = (_score_of(after) or 0.0) - (_score_of(before) or 0.0)
                said.append(f"{item} {'score up' if delta > 0 else 'score down' if delta < 0 else 'unchanged'}")
        lines.append("Named items: " + ", ".join(said) + ".")
    named_steps = _named(text, steps)[:PREDICTION_NAMED]
    if named_steps:
        said = []
        for step in named_steps:
            ran_before = sum(1 for i in shared if step in _step_replies(parent[i]))
            ran_after = sum(1 for i in shared if step in _step_replies(child[i]))
            if not ran_before and not ran_after:
                said.append(f"{step}: no model reply recorded on either")
            elif not ran_before:
                said.append(f"{step}: new, replied on {ran_after} of {len(shared)} item(s)")
            elif not ran_after:
                said.append(f"{step}: gone (replied on {ran_before} before)")
            else:
                ran = [i for i in shared if step in _step_replies(parent[i])
                       or step in _step_replies(child[i])]
                changed = sum(1 for i in ran if _step_replies(parent[i]).get(step)
                              != _step_replies(child[i]).get(step))
                said.append(f"{step}: reply changed on {changed} of the {len(ran)} item(s) "
                            "it ran on")
        lines.append("Named steps: " + "; ".join(said) + ".")
    return lines


#: The most lines of one candidate's "As built" a request shows; the rest are counted.
AS_BUILT_LINES = 12


def _as_built_lines(entry: Mapping[str, Any]) -> list[str]:
    """The proposal's own flow change beside what its edit built: the candidate's
    pipeline-view lines that are new (``+``) or changed (``~``) against its parent's, and
    the steps gone (`nn.pipeline_view.changed_lines`), so a description that says
    otherwise is seen against what ran, in the view's own words."""
    built = list(entry.get("wiring") or ())
    if len(built) > AS_BUILT_LINES:
        built = [*built[:AS_BUILT_LINES],
                 f"(and {len(built) - AS_BUILT_LINES} more line(s))"]
    said = ([f"Flow change, as proposed: {_clip(entry['flow_change'], REASONING_TEXT_CHARS)}"]
            if entry.get("flow_change") else [])
    if not built:
        return said
    return [*said, "As built (its pipeline lines, new + or changed ~, read from the "
            "candidate's graph; where the proposal's words say otherwise, this is what ran):",
            "```text", *built, "```"]


def _edit_change_lines(pairs: Sequence[dict[str, Any]], budget: int = EDIT_CHANGE_BUDGET,
                       omissions: list[dict[str, Any]] | None = None) -> list[str]:
    """'What the last edit changed': each recent child of the edited candidate
    against its parent, on train items only. Every child is listed with its counts; the
    changed items' trajectories share ``budget`` child by child in turn, so the first
    child's cannot crowd out a sibling's."""
    if not pairs:
        return []
    lines = ["## What the last edit changed",
             "The latest children of the candidate you edit (every child of its last round, "
             "siblings included), or that "
             "candidate itself when it has none yet, each compared with its parent item by "
             "item on the train items both were scored on. The parent's records are from the "
             "child's epoch, so the same seed, when it was scored then (the candidate you edit "
             "is re-scored each epoch); a pair scored with different seeds is marked, and some "
             "of its changes may be run-to-run noise."]
    per_child: list[list[tuple[str, list[Any]]]] = []
    for pair in pairs:
        own: list[tuple[str, list[Any]]] = []
        per_child.append(own)
        head = (f"### Child {pair.get('child')} of {pair.get('parent') or '(none)'}"
                + (f" -- gate: {pair['gate']}" if pair.get("gate") else ""))
        lines += ["", head, "Edits:", describe_edits(pair.get("edit"))]
        lines += [f"Wired by the system: {note}" for note in pair.get("notes") or ()]
        lines += _as_built_lines(pair)
        if pair.get("same_seed") is False:
            lines.append("Parent and child were scored in different epochs, with different "
                         "seeds: some changes below may be run-to-run noise.")
        child_records = list(pair.get("child_records") or ())
        lines += _item_error_lines(item_errors(child_records), len(child_records))
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
        if pair.get("prediction"):
            checked = prediction_check(pair["prediction"], pair.get("parent_records"),
                                       pair.get("child_records"), pair.get("steps") or ())
            lines += [f"Its prediction: {_clip(pair['prediction'], REASONING_TEXT_CHARS)}",
                      *([f"Prediction check -- {line}" for line in checked] or [
                          "Prediction check: it names no shared item or step, so only the "
                          "counts above can say whether it held."])]
        parent, child = _by_item(pair.get("parent_records")), _by_item(pair.get("child_records"))
        for kind, items in (("fixed", change["fixed"]), ("broken", change["broken"]),
                            ("score rose", change["up"]), ("score dropped", change["down"])):
            for item in items:
                block = (f"#### Item {item} -- {kind} (child {pair.get('child')})\nParent:\n"
                         + _item_block(parent[item], budget // 2, omissions)
                         + "\nChild:\n" + _item_block(child[item], budget // 2, omissions))
                own.append((block, [pair.get("child"), item, kind, _outcome(parent[item]),
                                    parent[item].get("score"), _outcome(child[item]),
                                    child[item].get("score")]))
    # Child by child in turn: each child's first changed item, then each one's second.
    turns = [entry for depth in range(max((len(own) for own in per_child), default=0))
             for own in per_child if depth < len(own) for entry in [own[depth]]]
    used = len("\n".join(lines))
    rest: list[list[Any]] = []
    for block, row in turns:
        # A block that does not fit goes to the table; a later, shorter one that does
        # still gets its trajectories.
        if used + len(block) + 2 > budget:
            rest.append(row)
            continue
        lines += ["", block]
        used += len(block) + 2
    if rest:
        lines += ["", "The other changed items (their trajectories did not fit):",
                  _table(["child id", "item", "change", "parent", "parent score", "child",
                          "child score"], rest)]
    return lines


def _reasoning_lines(entries: Sequence[dict[str, Any]], above: Sequence[str] = ()) -> list[str]:
    """'Your earlier reasoning': what the optimizer said about recent candidates,
    beside what happened to them, with each one's item errors (``item_errors``). A
    candidate in ``above`` ("What the last edit changed" shows its edits, its flow change
    and what they built) points there instead of repeating them."""
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
                  *_item_error_lines(entry.get("errors") or (), int(entry.get("items") or 0)),
                  *[f"Wiring warning this edit added: {w}" for w in entry.get("warnings") or ()],
                  *(["Its edits, flow change and what they built: under \"What the last edit "
                     "changed\" above."] if str(entry.get("id")) in above else [
                      "Edits:", describe_edits(entry.get("edit"), REASONING_TEXT_CHARS // 2),
                      *[f"Wired by the system: {note}" for note in entry.get("notes") or ()]]),
                  f"Rationale: {_clip(entry.get('rationale'), REASONING_TEXT_CHARS) or '(none)'}",
                  *([f"Addresses: {_clip(entry['addresses'], REASONING_TEXT_CHARS)}"]
                    if entry.get("addresses") else []),
                  *([] if str(entry.get("id")) in above else _as_built_lines(entry)),
                  f"Prediction: {_clip(entry.get('prediction'), REASONING_TEXT_CHARS) or '(none)'}",
                  f"Diagnosis that round: {_clip(entry.get('diagnosis'), REASONING_TEXT_CHARS) or '(none)'}",
                  *([f"Considered that round: {_clip(entry['considered'], REASONING_TEXT_CHARS)}"]
                    if entry.get("considered") else [])]
    promising = [entry for entry in entries if entry.get("gate") == "rejected"
                 and isinstance(entry.get("fixed"), int) and isinstance(entry.get("broken"), int)
                 and entry["fixed"] > entry["broken"]]
    if promising:
        entry = promising[-1]
        lines += ["", f"### Full edits of {entry.get('id')}, the latest rejected candidate that "
                  f"fixed more train items than it broke ({entry['fixed']} fixed / "
                  f"{entry['broken']} broken)",
                  f"Every step's wiring whole, its text shortened to fit {REJECTED_EDITS_CHARS} "
                  "characters, so it can be built on "
                  "or combined with edits to other steps; not to be re-proposed unchanged.",
                  describe_edits(entry.get("edit"), text_budget=REJECTED_EDITS_CHARS)]
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


#: The trace rows about what running a pipeline cost (seconds, inner model calls). Left
#: out of the request when the loop adds no cost term (parsimony 0) and the score is the
#: items' mean (so no cost is in it): the run's gradient.json, journal and report keep
#: them.
UNSCORED_TRACE_KEYS = ("cost",)


def _trace_rows(traces: Sequence[dict[str, Any]],
                omit: Sequence[str] = ()) -> list[tuple[str, Any]]:
    """The traces as the request shows them: no receipts, no paths, short texts, and
    none of the top-level rows ``omit`` names."""

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
            if key not in TRACE_OMITTED_KEYS and key not in omit]


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
    #: with ``E-too-many-edits``.
    max_edits: int = MAX_EDITS_PER_PROPOSAL

    #: Whether an edit may add a code node (`refactory_dsh.snippet`); one that does
    #: otherwise is refused with ``E-code-nodes``. The loop sets it from
    #: ``allow_code_nodes``.
    allow_code_nodes: bool = True
    #: The sandbox answer of the request being validated (``ProposalContext.sandbox``,
    #: set by ``propose_valid``); ``None`` asks `snippet.available` afresh.
    sandbox: tuple[bool, str] | None = None

    #: Values only the grader may see (the train items' grader-only fields), which no
    #: code node's code may contain (``E-secret``), or a function returning them (called
    #: only for a candidate with code). The loop sets it from its Task.
    grader_secrets: Any = frozenset()

    #: Whether ``propose`` asks a model, so its calls must run on a route without tools
    #: (``Loop._check_optimizer_route``) and a strategy may hand it the round's sampling.
    uses_model: bool = False

    #: Whether someone answers gates while this run's candidates run. An outer loop's
    #: candidates run headless, so an edit adding a gate a person or an agent answers is
    #: refused ``E-gate-kind`` (`optim.checks`).
    answered_gates: bool = False
    #: A candidate's recorded run inputs by id (`optim.checks.recorded_inputs`), for the
    #: code dry run of new or changed code nodes; ``None`` skips the dry run, said in the
    #: candidate's ``notes``. The loop sets it to its run directory's.
    recorded_inputs: Any = None

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
        #: The last refusal of a slim repair's fix (`repair_refused`), when none was taken.
        self.last_repair_refusal: list[str] = []
        #: The answer's optional ``considered``: the kinds of change it weighed and why it
        #: chose its own, recorded beside the diagnosis so each round's choice can be read.
        self.last_considered: str = ""
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
        # The request's sandbox answer: what it offered is what the refusals judge by.
        self.sandbox = ctx.sandbox
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
                refused=[*ctx.refused, *refused_entries(self.rejections)][-REFUSED_SHOWN:],
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
                    fixed = self.repair_refused(
                        round_ctx, proposal, outcome, parent,
                        lambda again, parent=parent: self.attempt(
                            again, parent, invariants, resolve_calls, ctx.frozen, seen))
                    after = [] if fixed is not None else list(self.last_repair_refusal)
                    rejection = Rejection(
                        attempt=attempt,
                        edit=proposal.edit,  # {"edits": [...]} for a compound proposal
                        reasons=[*outcome, *(f"{reason} [the proposal as a slim repair call "
                                             "fixed it]" for reason in after)],
                        rationale=proposal.rationale,
                        repaired=fixed.id if fixed is not None else "",
                    )
                    self.rejections.append(rejection)
                    if fixed is not None:
                        accepted.append(fixed)
                        progressed = True
                    else:
                        feedback.extend(rejection.reasons)
            if not progressed and not feedback:
                break
        return accepted, self.rejections

    # ── repairing a refused proposal (`optim.repair`) ─────────────────────
    def repair(self, ctx: ProposalContext, proposal: Proposal, reasons: Sequence[str],
               parent: Candidate) -> Proposal | None:
        """A fixed version of a proposal refused for its mechanics, or ``None``. The
        default makes none: an optimizer that asks a model overrides it with the slim
        repair call (`AutoresearchOptimizer.repair`)."""
        return None

    def repair_refused(self, ctx: ProposalContext, proposal: Proposal, reasons: Sequence[str],
                       parent: Candidate, attempt: Any) -> Candidate | None:
        """Ask `repair` to fix a refused proposal and validate each fix with ``attempt``
        (the caller's attempt path, so a fix passes every check the proposal had to):
        the candidate, or ``None``. Only a refusal about mechanics (`repair.repairable`)
        is sent, at most `repair.REPAIR_CALLS_PER_PROPOSAL` times and
        `repair.REPAIR_CALLS_PER_EPOCH` times in ``ctx.epoch``; each outcome is traced
        (kind ``repair_outcome``). When no fix is accepted, ``last_repair_refusal`` holds
        the last fix's refusal (empty when no fix was tried), for the caller to keep."""
        self.last_repair_refusal = []
        if type(self).repair is Optimizer.repair:
            return None
        used: dict[int, int] = self.__dict__.setdefault("_repair_calls", {})
        current, why = proposal, list(reasons)
        for _ in range(repair.REPAIR_CALLS_PER_PROPOSAL):
            if not repair.repairable(why) or used.get(ctx.epoch, 0) >= repair.REPAIR_CALLS_PER_EPOCH:
                return None
            used[ctx.epoch] = used.get(ctx.epoch, 0) + 1
            fixed = self.repair(ctx, current, why, parent)
            if fixed is None:
                return None
            outcome = attempt(fixed)
            self.record_call(kind="repair_outcome", epoch=ctx.epoch, parent=parent.id,
                             accepted=isinstance(outcome, Candidate),
                             candidate=outcome.id if isinstance(outcome, Candidate) else None,
                             reasons=[] if isinstance(outcome, Candidate) else list(outcome))
            if isinstance(outcome, Candidate):
                return outcome
            current, why = fixed, list(outcome)
            # The fix's own refusal, for the caller to keep with the proposal's.
            self.last_repair_refusal = list(outcome)
        return None

    def repairs_used(self, epoch: int) -> int:
        """How many slim repair calls `repair_refused` made in ``epoch``."""
        return int(self.__dict__.get("_repair_calls", {}).get(epoch, 0))

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
        record: bool = True,
        check_hints: bool = True,
        dry_run_code: bool = True,
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

        ``record`` False records no call (the request's operator table asks this for
        every operator it lists). A precondition refusal that names a replacing edit
        has that edit checked here the same way (``check_hints``): when it would be
        refused too, the refusal says so instead of naming it. That check skips the code
        dry run (``dry_run_code``), a process per request, and the hint says so.
        """
        steps = proposal.steps

        def hinted(reason: str) -> str:
            if not check_hints:
                return reason
            return checked_hint(reason, lambda edit: self.attempt(
                Proposal(edit=edit), parent, invariants, resolve_calls, frozen, seen,
                seed_state=seed_state, protected=protected,
                allow_new_node_tools=allow_new_node_tools, graders=graders, record=False,
                check_hints=False, dry_run_code=False))
        refusal = proposal_refusal(proposal, max_edits=self.max_edits, parent_state=parent.state)
        if refusal:
            return refusal
        frozen = set(frozen or ())
        # Knobs a protected or grader node reads or binds hold for every edit: a search
        # that could rewrite them could rewrite the grader side's behaviour (a protected
        # agent's prompt copying answers into the scored artifact).
        guarded = _grader_side_knobs(parent.state, (*protected, *graders))
        # A compound proposal's edits go through the same checks one by one, each
        # against the state the edits before it left; the combined state is then held
        # to every rule below once, as one candidate. Protected nodes are judged on that
        # combined state too: "remove a step, then add one in its place" passes through
        # a state where the run skips a protected node, and only the end state runs.
        state = parent.state
        edit_reports: list[dict[str, Any] | None] = []
        prefixes: list[dict[str, Any]] = []  # the state after each edit, for E-protected
        # The edits as applied: each with the fixes whose meaning is unambiguous
        # (`optim.repair`), each fix said in ``repairs``; the candidate records these.
        applied: list[dict[str, Any]] = []
        repairs: list[str] = []
        wired_notes: list[str] = []
        for index, edit in enumerate(steps):
            edit, fixed, ambiguous = repair.fix_edit(edit, state, steps, index)
            if ambiguous:
                return [_at(f"E-node: {ambiguous}", index, steps)]
            repairs += fixed
            if len(steps) > 1 and repair.noop_knob(edit, state):
                repairs.append(f"edit {index + 1} (set_knob) sets knob {edit.get('name')!r} to "
                               "the value it already holds, so it was dropped")
                continue
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
                return [_at(f"E-precondition: {hinted(reason)}", index, steps)]
            reports: list[dict[str, Any]] = []
            try:
                state = nn_edits.apply(edit, state, edit_reports=reports)
            except nn_edits.EditError as exc:
                return [_at(f"{exc.code}: {exc}", index, steps)]
            except Exception as exc:  # an operator bug is reported, not swallowed
                return [_at(f"E-internal: {type(exc).__name__}: {exc}", index, steps)]
            missed = _patch_targets_missing(edit, reports)
            if missed:
                return [_at(missed, index, steps)]
            if len(steps) > 1 and candidate_id(state) == candidate_id(before_step):
                # An edit of a bundle that changes nothing would be recorded as part of
                # the idea while doing nothing; said back rather than kept.
                return [_at(f"E-noop: {edit.get('op')} leaves the state as the edits before "
                            "it left it; drop it or change it", index, steps)]
            edit_reports.append(reports[0] if reports else None)
            prefixes.append(state)
            applied.append(edit)
            # What a core operator wired beside the steps it names, said with the candidate.
            wired_notes += nn_edits.core_wiring_notes(edit, before_step, state)

        violation = nn_edits.protected_violation(parent.state, state, protected)
        if violation:
            return [f"E-protected: {violation}" + _protected_culprit(applied, parent.state,
                                                                      prefixes, protected)]

        # A bundle that changes nothing at all is refused below as one no-op.
        dead = (_dead_edit(applied, parent.state, state)
                if candidate_id(state) != parent.id else None)
        if dead is not None:
            return [_at(f"E-noop: {applied[dead].get('op')} has no effect on the result: the "
                        "edits after it undo or overwrite it; drop it", dead, applied)]
        # A `{{name}}` an added or changed step fills from a file's name, not a fact: a
        # file_text op hands it the file's whole text (`repair.wire_file_slots`), and says so.
        # The wired state is held to the protected nodes again: a file_text op put in front
        # of a step must not rewire a protected one (as the same edit written by hand
        # would be refused). Then it is not made, and the validator says why (V15).
        wired_state, wiring, wired = repair.wire_file_slots(parent.state, state)
        if wiring and not nn_edits.protected_violation(parent.state, wired_state, protected):
            state = wired_state
            applied += wiring
            edit_reports += [None] * len(wiring)
            repairs += wired

        # ── tool-policy escalation check (anchored to the original seed) ──
        escalation = _tool_policy_escalated(seed_state if seed_state is not None else parent.state, state)
        if escalation:
            return escalation
        tooled = _new_node_tools(seed_state if seed_state is not None else parent.state, state,
                                 allow_new_node_tools)
        if tooled:
            return tooled
        coded = _code_node_refusals(seed_state if seed_state is not None else parent.state, state,
                                    bool(getattr(self, "allow_code_nodes", True)),
                                    getattr(self, "sandbox", None))
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
                                grader_side=(*protected, *graders), graders=tuple(graders),
                                baseline=parent.state)
        if issues:
            return [str(issue) for issue in issues]
        unlowerable = _lowering_refusal(parent.state, state)
        if unlowerable:
            return [unlowerable]
        # Before scoring (`optim.checks`): a gate nobody will answer fails every item.
        if not getattr(self, "answered_gates", False):
            unanswered = checks.answered_gate_refusals(parent.state, state)
            if unanswered:
                return unanswered
        # New or changed code runs on the parent's recorded inputs before any item is
        # scored with it (`optim.checks.code_dry_run`); skipped, and said, when it cannot.
        before_code = _code_signatures(parent.state)
        changed_code = sorted(node_id for node_id, signature in _code_signatures(state).items()
                              if before_code.get(node_id) != signature)
        crashed, notes = checks.code_dry_run(parent.id, parent.state, state, changed_code,
                                             getattr(self, "recorded_inputs", None)
                                             if dry_run_code else None,
                                             getattr(self, "sandbox", None))
        if crashed:
            return crashed
        # Build the per-edit report list parallel to edits.  Parent reports
        # are carried forward; one report (or None) per applied edit is appended.
        parent_reports = list(parent.edit_reports) if parent.edit_reports else [
            None for _ in parent.edits
        ]
        candidate = Candidate.from_state(
            state,
            parent=parent.id,
            edits=[*parent.edits, *applied],
            edit_reports=[*parent_reports, *edit_reports],
            step_edits=len(applied),
        )
        candidate.repairs, candidate.notes = list(repairs), [*wired_notes, *notes]
        if repairs and record:
            self.record_call(kind="repair", how="deterministic", candidate=candidate.id,
                             parent=parent.id, fixes=repairs)
        return candidate
