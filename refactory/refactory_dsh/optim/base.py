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
import itertools
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.core_ops import STEP_KINDS
from refactory_dsh.nn import validate_state
from . import checks, evidence, repair
from .evidence import CLAIMS
from refactory_dsh.nn.knob import Knob
from refactory_dsh.nn.validate import Invariant
from refactory_dsh.tool_policy import no_effect


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
#: The whole proposal request's room in characters, tied to the optimizer model's context
#: as ``analysts.ANALYST_ROOM`` is (same size, same reasoning: Opus 4.6 reads 200k
#: tokens, Claude Code's own system prompt and the reserved reply take 30-40k, and traces
#: run 2.5-3.5 characters a token). Only the items' texts give way to it, whatever their
#: sizes (`ProposalContext.render`): "Some items in full" and the blocks of "What the last
#: edit changed" share what the rest of the request leaves, items whole while they fit
#: and in rank order, the rest named; an item or block that alone passes its share has
#: its longest texts cut to fit (marked), or is named. Nothing else is cut for it. A
#: request the provider still refuses as too long is asked again once at half its length
#: (``AutoresearchOptimizer.propose``), which these then always make shorter.
PROPOSAL_ROOM = 400000
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
    ``tool_policy='none'`` or ``'refused'`` must keep a policy whose model can change
    nothing in every candidate, and no candidate may add tools to such a node. New node IDs (present only in the
    candidate, not in the seed) are allowed tools -- they represent genuinely
    new architecture. Returns a list of rejection reasons, or an empty list.
    """
    seed_nodes = (seed_state.get("graph") or {}).get("nodes") or {}
    candidate_nodes = (candidate_state.get("graph") or {}).get("nodes") or {}
    reasons: list[str] = []
    for node_id, seed_node in seed_nodes.items():
        seed_policy = seed_node.get("tool_policy")
        if not no_effect(seed_policy):
            continue
        candidate_node = candidate_nodes.get(node_id)
        if candidate_node is None:
            # The seed node was deleted -- that is allowed (the node is gone).
            # But if it was re-added under a different policy, we catch that below.
            continue
        candidate_policy = candidate_node.get("tool_policy")
        if not no_effect(candidate_policy):
            reasons.append(
                f"E-tool-policy-escalation: node {node_id!r} seed declares "
                f"tool_policy={seed_policy!r}; candidate removes it or changes it"
            )
        candidate_tools = candidate_node.get("tools") or []
        if candidate_tools:
            reasons.append(
                f"E-tool-policy-escalation: node {node_id!r} seed declares "
                f"tool_policy={seed_policy!r}; candidate adds tools {candidate_tools!r}"
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


def _step_route_refusals(parent_state: dict[str, Any], candidate_state: dict[str, Any],
                         routes: Any) -> list[str]:
    """``E-route`` for each agent step whose own route (`Node.route`) the proposal sets or
    changes to a model, effort or cap that no route of the run uses, or that the step runs
    on already (`workers.StepRoutes.refusal`); every one when the run has no routes to
    choose from (a scripted model)."""
    before = (parent_state.get("graph") or {}).get("nodes") or {}
    reasons = []
    for node_id, node in sorted(((candidate_state.get("graph") or {}).get("nodes") or {}).items()):
        declared = node.get("route")
        if not declared or (before.get(node_id) or {}).get("route") == declared:
            continue
        reason = (routes.refusal(node_id, str(node.get("role") or "ceo"), declared)
                  if routes is not None else
                  f"step {node_id!r} sets a route of its own, and this run has none to choose from")
        if reason:
            reasons.append(f"E-route: {reason}")
    return reasons


def _route_binding_refusals(seed_state: dict[str, Any], candidate_state: dict[str, Any]
                            ) -> list[str]:
    """``E-route`` for each agent step whose route setting is bound to a knob the mode did
    not freeze (``route.*`` field bindings, `nn.graph.Node.bound_route`): a step's output
    cap and thinking budget are the run's settings, which a search never chooses. A
    binding is kept only to a knob the seed declares frozen, at the seed's value, or to a
    frozen copy of one the seed binds to the same setting, its spec and value the seed
    knob's (a ``duplicate`` of the step: its copy of the knob stays frozen, so the copy
    runs on the same budget)."""
    seed_knobs = seed_state.get("knobs") or {}
    knobs = candidate_state.get("knobs") or {}
    seed_bound = {(str(pair[0]), str(pair[1]))
                  for node in ((seed_state.get("graph") or {}).get("nodes") or {}).values()
                  for pair in node.get("field_bindings") or ()
                  if isinstance(pair, (list, tuple)) and len(pair) == 2
                  and str(pair[0]).startswith("route.")}

    def frozen_copy(field: str, knob: str) -> bool:
        held = knobs.get(knob) or {}
        return bool((held.get("spec") or {}).get("frozen")) and any(
            bound_field == field and seed_knobs.get(seeded, {}).get("spec") == held.get("spec")
            and seed_knobs.get(seeded, {}).get("value") == held.get("value")
            and (seed_knobs.get(seeded, {}).get("spec") or {}).get("frozen")
            for bound_field, seeded in seed_bound)

    reasons = []
    for node_id, node in sorted(((candidate_state.get("graph") or {}).get("nodes") or {}).items()):
        for pair in node.get("field_bindings") or ():
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2
                    and str(pair[0]).startswith("route.")):
                continue
            field, knob = str(pair[0]), str(pair[1])
            seeded = seed_knobs.get(knob) or {}
            if knob not in seed_knobs and frozen_copy(field, knob):
                continue
            if (not (seeded.get("spec") or {}).get("frozen")
                    or (knobs.get(knob) or {}).get("value") != seeded.get("value")):
                reasons.append(f"E-route: step {node_id!r} binds its {field.removeprefix('route.')} "
                               f"to knob {knob!r}, which the mode did not freeze: a step's route "
                               "settings are the run's, never the search's")
    return reasons


#: Why a run offers no tool step and refuses one when ``new_step_tools`` is off.
NEW_STEP_TOOLS_OFF = "this run turns tools on new steps off (new_step_tools: false)"


def _new_node_tools(seed_state: dict[str, Any], candidate_state: dict[str, Any],
                    sealed: tuple[bool, str] | None = None,
                    check: Callable[[str, dict[str, Any]], tuple[bool, str]] | None = None,
                    allowed: bool = True) -> list[str]:
    """``E-new-node-tools`` for each agent node the search added that would run with tools
    outside a sealed workspace.

    A node an edit adds gets ``tool_policy: none`` by default (`nn.edits`). With a
    ``tools`` list from `workspace.SEALED_TOOLS` in a ``scratch`` workspace it runs them
    sealed (`workspace.sealed`): it sees only the files it reads, no other item, no grader
    file, no network. A copy of a step of the seed that uses tools keeps them in a fresh
    workspace of its own. Any other new node with tools could read anything in the run
    directory or beyond it. A node whose id the seed has is held to the seed's own policy
    instead (`_tool_policy_escalated`). ``sealed`` is the run's answer to whether a
    sealed step can run (`workspace.sealed_steps`, which the request's offer takes too);
    without one, whether the seal runs here. ``check`` judges each new sealed step on the
    client its dispatch takes, its own route included (`workspace.sealed_step`). Without
    ``allowed`` (the run's ``new_step_tools`` off) every new agent node with tools is
    refused, a copy of a seed step's tools included.
    """
    from refactory_dsh.workspace import is_sealed, sealed_steps

    seed_nodes = (seed_state.get("graph") or {}).get("nodes") or {}
    # The tools a step of the seed already runs with: a copy of such a step may keep them
    # when it works in a fresh workspace of its own (the user's decision, recorded in
    # `nn.coverage.DECISIONS`), so it never works where the step it copies works.
    seed_tools = {(node.get("tool_policy"), tuple(node.get("tools") or ()))
                  for node in seed_nodes.values() if node.get("kind") == "agent"
                  and (not no_effect(node.get("tool_policy")) or node.get("tools"))}
    reasons = []
    for node_id, node in sorted(((candidate_state.get("graph") or {}).get("nodes") or {}).items()):
        if node_id in seed_nodes or node.get("kind") != "agent":
            continue
        if no_effect(node.get("tool_policy")) and not node.get("tools"):
            continue
        if not allowed:
            reasons.append(f"E-new-node-tools: new agent node {node_id!r} has tools, and "
                           f"{NEW_STEP_TOOLS_OFF}; give it none")
            continue
        if is_sealed(node):
            sealable, why = sealed if sealed is not None else sealed_steps(None)
            if sealable and check is not None:
                sealable, why = check(node_id, node)
            if not sealable:
                reasons.append(f"E-new-node-tools: new agent node {node_id!r} runs its tools "
                               f"sealed, and this run cannot run a sealed step: {why}")
            continue
        if (node.get("tool_policy"), tuple(node.get("tools") or ())) in seed_tools \
                and _fresh_workspace(node):
            continue
        reasons.append(
            f"E-new-node-tools: new agent node {node_id!r} would run with tools outside a sealed "
            "workspace; list them in 'tools' (a 'scratch' workspace, no tool_policy), or "
            "give it none")
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
    if max_edits is not None and int(max_edits) > 1 and len(steps) > MAX_RAW_EDITS_PER_PROPOSAL:
        return [f"E-too-many-edits: the proposal bundles {len(steps)} edits and no proposal "
                f"may carry more than {MAX_RAW_EDITS_PER_PROPOSAL}, however they count; "
                "split it, or keep the edits the fix needs together"]
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
        internal = _internal_node_fields(step)
        if internal:
            return [_at(f"E-node: {', '.join(map(repr, internal))} is set by the edit layer "
                        "(when an operator gives a code step's outputs new names), never in a "
                        "proposal; leave it out of the node", index, steps)]
    return []


def _internal_node_fields(step: dict[str, Any]) -> list[str]:
    """The node fields only the edit layer sets (`node_schema.INTERNAL_FIELDS`) that a
    node object of ``step`` carries."""
    from refactory_dsh.nn.node_schema import INTERNAL_FIELDS

    nodes: list[Any] = []
    for key in (*NODE_OBJECT_FIELDS, "nodes"):
        value = step.get(key)
        nodes += value if isinstance(value, list) else [value]
    return sorted({name for node in nodes if isinstance(node, dict)
                   for name in node if name in INTERNAL_FIELDS})


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
    #: What that repair dropped or changed of the proposal's text (`repair.lost_content`),
    #: or empty: said in later requests (`repair_loss_lines`).
    dropped: str = ""

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


def refused_lines(refused: Sequence[Mapping[str, Any]],
                  homes: Mapping[str, str] | None = None) -> list[str]:
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
                  describe_edits(steps, text_budget=REFUSED_EDITS_CHARS, homes=homes)]
    return lines


def repair_losses(rejections: Sequence["Rejection"]) -> list[dict[str, Any]]:
    """The ``ProposalContext.repair_losses`` of ``rejections``: each repaired one whose
    repair dropped or changed text."""
    return [{"candidate": r.repaired, "dropped": r.dropped} for r in rejections
            if r.repaired and r.dropped]


def repair_loss_lines(losses: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per repaired proposal whose repair dropped or changed its text."""
    return [f"Not as you wrote it: `{loss.get('candidate')}` is your refused proposal as a "
            f"slim repair fixed it, which dropped or changed {loss.get('dropped')}; propose "
            "it again if you still want it." for loss in losses]


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


#: A prompt line's tail counts as a piece of a reason or a reply (`_settled`) only after a
#: label of at most this many characters ("Sent back because:"), and only when the tail
#: is at least `TAIL_MIN` characters or `TAIL_MIN_WORDS` words: a framing line's last
#: word ("- User: ## Question") is not a piece of a reply that happens to use it.
TAIL_LABEL_MAX = 24
TAIL_MIN = 20
TAIL_MIN_WORDS = 3
#: The shortest piece of an item's data (`_data_lines`) that marks a prompt line as
#: carrying it; a shorter one ("A", "yes") would match framing by chance.
ITEM_DATA_MIN = 8


#: A heading of at most this many words is a bare label (`_bare_label`).
BARE_HEADING_WORDS = 3


def _bare_label(line: str) -> bool:
    """Whether a line is only a label: one word ("Independent") or a short heading
    ("## Question", "## Option Comparison Notes"). A model's free text using one says
    nothing that a framing line holding the same words carries from the item."""
    words = line.split()
    if len(words) == 1:
        return True
    heading = line.lstrip("#")
    return heading != line and len(heading.split()) <= BARE_HEADING_WORDS


def _data_lines(value: Any, *, free: bool = False) -> set[str]:
    """Every line of ``value``'s strings (nested in lists and dicts too), normalised, of at
    least `ITEM_DATA_MIN` characters: an item's input, a step's reply or a code gate's
    reason, which a prompt line holding one carries from that item. With ``free`` (a
    model's reply text), a bare label (`_bare_label`) is left out; a structured field (a
    gate's reason, a string inside a JSON reply) counts at any length from
    `ITEM_DATA_MIN`, one word or a heading included. A string that is JSON counts by its
    own strings too, so a reason extracted from one is matched in the prompt quoting it."""
    from .evidence import _norm

    out: set[str] = set()
    stack: list[tuple[Any, bool]] = [(value, free)]
    while stack:
        current, loose = stack.pop()
        if isinstance(current, Mapping):
            stack += [(v, loose) for v in current.values()]
        elif isinstance(current, (list, tuple)):
            stack += [(v, loose) for v in current]
        elif isinstance(current, str):
            out |= {n for line in current.splitlines()
                    if len(n := _norm(line)) >= ITEM_DATA_MIN
                    and not (loose and _bare_label(n))}
            text = current.strip()
            if text[:1] in ("{", "[", '"'):
                try:
                    stack.append((json.loads(text), False))
                except json.JSONDecodeError:
                    pass
    return out


def _holds(line: str, datum: str) -> bool:
    """Whether ``datum`` is in ``line`` as whole words: a reply's heading "## Option C" is
    not in the op's heading "## Option Comparison Notes"."""
    start = line.find(datum)
    while start != -1:
        end = start + len(datum)
        if (start == 0 or not line[start - 1].isalnum()) and \
                (end == len(line) or not line[end].isalnum()):
            return True
        start = line.find(datum, start + 1)
    return False


def _whole_texts(value: Any) -> set[str]:
    """Each string of ``value`` (nested, and inside a JSON string) whole, normalised: a
    reason or a reply, which a prompt line holding a piece of it carries."""
    from .evidence import _norm

    out: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            stack += list(current.values())
        elif isinstance(current, (list, tuple)):
            stack += list(current)
        elif isinstance(current, str) and current.strip():
            out.add(_norm(current))
            if current.strip()[:1] in ("{", "[", '"'):
                try:
                    stack.append(json.loads(current))
                except json.JSONDecodeError:
                    pass
    return out


def settled_for(knobs: Iterable[Mapping[str, Any]], node_states: Iterable[Mapping[str, Any]],
                records: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """What an item's step does not repeat of its prompt (`ITEM_FORM`), in every request that
    shows items (the proposer's, the analysts', memory's): ``knobs`` and ``node_states``
    are the settings and nodes the request shows, ``records`` its items. It is text that
    comes from no item: the lines of every setting's text ("Each step's
    settings"), of the nodes' own prompt templates (each node in full, in the
    reference), and the framing the step's ops put around its inputs (the model
    client's wrapper, their headings). Framing is a line in every one of the step's
    prompts, over two or more items, that holds nothing from any item's own data
    (its input, a step's reply, or a code gate's reason; as whole words), nor a piece of
    a code gate's reason or of a reply (a reason spread over several prompt lines).

    Never a line just because it recurs: a judge's reason that is not in every one
    of the step's prompts, or a choice text (part of an item's input) said the same
    way in two items, is each item's own evidence and is shown."""
    from .evidence import _norm, settled_lines

    texts = [k.get("value") for k in knobs]
    texts += [n.get(field) for n in node_states
              for field in ("prompt_template", "system_prompt_template")]
    records = list(records)
    every: dict[str, set[str]] = {}
    items: dict[str, set[str]] = {}
    data: set[str] = set()
    # Reasons and replies whole: a prompt line may carry a piece of one (a gate's
    # reason is stored as one line, and reaches a prompt over several). Never an op's
    # output, which is often the framed prompt itself.
    said: set[str] = set()
    for record in records:
        # Of the record's steps, only a code gate's reason is item data: an op's
        # output is often the framed prompt itself, and a visit's node id is no item's.
        reasons = [step.get("reason") for step in record.get("steps") or ()
                   if isinstance(step, Mapping) and step.get("kind") == "gate"]
        data |= _data_lines(record.get("metadata")) | _data_lines(reasons)
        said |= _whole_texts(reasons)
        for exchange in record.get("exchanges") or ():
            node = str(exchange.get("node"))
            lines = {_norm(line) for line in str(exchange.get("prompt") or "").splitlines()
                     if line.strip()}
            every[node] = lines if node not in every else every[node] & lines
            items.setdefault(node, set()).add(str(record.get("id")).split("#")[0])
            data |= _data_lines(exchange.get("response"), free=True)
            said |= _whole_texts(exchange.get("response"))

    def carried(line: str) -> bool:
        if any(_holds(line, datum) for datum in data):
            return True
        # The line inside a reason or a reply, or its tail after a short label
        # ("Feedback: ..."): a piece of one. The bounds keep a framing line's last word
        # ("- User: ## Question") from matching a reply that happens to use it.
        if line in said:
            return True  # a whole reason or reply, however short ("## Reloop")
        if _bare_label(line):
            return False  # a heading or one word is no piece of a reason
        words = line.split(" ")
        for start in range(len(words)):
            label, tail = " ".join(words[:start]), " ".join(words[start:])
            if len(label) > TAIL_LABEL_MAX:
                return False
            if len(tail) < ITEM_DATA_MIN or (start and len(tail) < TAIL_MIN
                                            and len(words) - start < TAIL_MIN_WORDS):
                return False
            if any(tail in text for text in said):
                return True
        return False

    framing = {line for node, lines in every.items() if len(items[node]) > 1
               for line in lines if not carried(line)}
    return settled_lines(texts) | framing


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
    #: The mode author's facts about the mode (`nn.Mode.guidance`), shown under "About this mode".
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
    #: Facts about the edited graph, worked out once by the loop (`graph_facts`): what
    #: ``"sees": ["task"]`` gives, what a fork's new branch can see, and which removals,
    #: replacements and merges an attempt refuses, with why.
    graph_facts: list[str] = field(default_factory=list)
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
    #: The whole request's room in characters (``PROPOSAL_ROOM``; the proposer takes its
    #: system prompt and closing line off it). ``render`` keeps within it by showing
    #: fewer items whole; 0 is no room (every picked item shown, whatever their size).
    request_room: int = PROPOSAL_ROOM
    #: The most characters the items "Trajectories" shows whole take together when not
    #: every item fits (`_fit_shown`); 0 is no limit. ``render`` sets it from
    #: ``request_room`` when the request would pass that, never otherwise.
    shown_items_room: int = 0
    #: What the last ``render`` measured of its sections (``items``: the characters the
    #: items shown whole take), for ``render`` to fit them.
    fit_sizes: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    #: What one ``render`` works out once and reads again while it renders (its copies
    #: share it); emptied when it starts and when it ends.
    rendering: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
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
    #: Each judged candidate's per-item validation scores, by candidate id, under the
    #: default loop's ``val_gate`` and for a parent policy that asks for them
    #: (``ParentSelector.uses_item_scores``); empty otherwise, and always empty in the
    #: context a proposer is handed. Each inner map is
    #: ``Loop.part_item_scores(id, "val")``: item keys are a per-run HMAC of the item id
    #: (`loop.opaque_item_key`), the same keys ``Run.part_item_scores`` gives a
    #: strategy, so a parent policy can line candidates up item by item (a per-item
    #: Pareto frontier) without learning an item id. Validation only, never test, and
    #: never rendered.
    val_item_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    #: How the validation gate selects (``Loop.selection_mode``): ``single``, ``pool``,
    #: or empty without the gate. Under ``pool`` the gate note says what the pool is.
    selection: str = ""
    #: Under ``pool``, whether a population parent picker (``archive``, ``evolutionary``,
    #: ``islands``, ``map_elites``, ``pareto``) chooses the parent from every judged candidate
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
    #: Every evaluated candidate of the run with what its own edits did, its train
    #: fixed/broken counts against its parent and the gate's word (``Loop._ledger``,
    #: ``LoopConfig.run_ledger``), for "Every change so far"; empty says nothing.
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
    #: What slim repairs dropped or changed of recent proposals they turned into
    #: candidates (``{"candidate", "dropped"}``; `Rejection.dropped`), one line each
    #: (`repair_loss_lines`), so the proposer knows that text did not land.
    repair_losses: list[dict[str, Any]] = field(default_factory=list)
    #: Whether a new step's tools can run sealed on this machine (`workspace.sealed_available`);
    #: only then do the compositions show a tool step.
    new_node_tools: bool = False
    #: What a new agent step may choose for its own route, field by field
    #: (`workers.StepRoutes.offered`); empty offers no route, and the request says none.
    step_routes: dict[str, list[str]] = field(default_factory=dict)
    #: Time, in numbers, for "Time per item" (`time_lines`): ``per_item_median_s``/
    #: ``per_item_p90_s``/``per_item_max_s`` (the edited candidate's items' own times,
    #: `loop.item_time_spread`); empty says nothing.
    time_facts: dict[str, Any] = field(default_factory=dict)
    #: Proposals already made earlier in this round (each: ``{"edits", "rationale",
    #: "parent"}``), so the model does not duplicate them. Shown as facts, not
    #: instructions.
    prior_proposals: list[dict[str, Any]] = field(default_factory=list)

    def _settled(self) -> frozenset[str]:
        """The lines an item's step does not repeat (`settled_for`) over this request's
        items, the edited candidate's children included; within one ``render``, worked
        out once (``fit_sizes`` aside, nothing it reads changes while it renders)."""
        held = self.rendering.get("settled")
        if held is None:
            held = settled_for(self.knobs, self.node_states,
                               [*self.item_records,
                                *(r for pair in self.edit_pairs or ()
                                  for side in ("parent_records", "child_records")
                                  for r in pair.get(side) or ())])
            if self.rendering.get("on"):
                self.rendering["settled"] = held
        return held

    def _edit_change(self) -> list[str]:
        """"What the last edit changed" (`_edit_change_lines`), naming a reply
        "Trajectories" shows whole by where it is."""
        shown: dict[str, str] = {}
        if self.item_records:
            whole = trajectories_whole(
                self.item_records, budget=self.evidence_budget, findings=self.reader_findings,
                analysed=(None if self.analysis is None
                          else list(self.analysis.get("findings") or ())),
                shown_room=self.shown_items_room, settled=self._settled())
            for record in whole:
                for exchange in record.get("exchanges") or ():
                    reply = str(exchange.get("response") or "").strip()
                    if reply:
                        shown.setdefault(reply, f"under \"{TRAJECTORIES_HEADING[3:]}\", item "
                                                f"{record.get('id')}, step `{exchange.get('node')}`")
        return _edit_change_lines(self.edit_pairs, homes=knob_homes(self.knobs), shown=shown)

    def _leaderboard_lines(self) -> list[str]:
        """The leaderboard: id, parent, score, structural, gate and cost. A candidate's edits
        are in "What this run has already tried", one line each, so they are not repeated
        here; only a candidate that list leaves out (it was truncated) has its edits here."""
        listed = {c.get("id") for c in self.history}
        missing = [c for c in self.leaderboard if c.get("id") not in listed]
        gate = _gate_column(self.leaderboard)
        return [_table(
            ["id", "parent", "score", "structural", *(["edits"] if missing else []), *gate,
             *(["cost per item"] if self.parsimony else [])],
            [[c["id"], c.get("parent"), fmt_score(c.get("score")), c.get("structural"),
              *([compact_edits(c.get("edits"), self.seed_knob_values)
                 if c.get("id") not in listed else "(below)"] if missing else []),
              *([c.get("gate", "")] if gate else []),
              *([_cost_cell(c.get("cost_per_item"))] if self.parsimony else [])]
             for c in self.leaderboard])]

    def _tried_lines(self) -> list[str]:
        """What this run has tried and how each attempt did: the recent results, the
        traces' numbers, the last edit's children and the reasoning behind recent
        candidates (each with what its edit built), every candidate so far, the run-wide
        feedback, earlier runs and the latest refusals.

        Each long text the optimizer wrote is whole in one place: a knob's current text in
        "Each step's settings" (its exact patch target), the last edit's children's edits
        and predictions under "What the last edit changed", the other recent candidates'
        edits under "Your earlier reasoning". The attempts list, the evidence table and the
        other sections name it and point there."""
        below = [str(p.get("child")) for p in self.edit_pairs or ()]
        elsewhere = {
            **{str(e.get("id")): "Your earlier reasoning"
               for e in reasoning_shown(self.reasoning or [])},
            **{child: "What the last edit changed" for child in below}}
        homes = knob_homes(self.knobs)
        lines = [
            "## Recent results",
            *_gate_note(self.leaderboard, self.history, val_items=self.val_items,
                        selection=self.selection, population=self.population_parents),
            *_cost_note(self.parsimony),
            *self._leaderboard_lines(),
            "",
            "## What this run has already tried",
            (
                f"{len(self.history)} candidate(s), best "
                f"{fmt_score(max((c.get('score') or 0.0) for c in self.history))}"
                if self.history and not _gate_column(self.history)
                else (f"{len(self.history)} candidate(s); "
                      f"{_current_line(self.history, self.selection)}")
                if self.history
                else "(nothing evaluated yet)"
            )
            + (
                " — older candidates omitted, so do not assume the list is complete"
                if self.history_truncated
                else ""
            )
            + (". One line each, oldest first: the edits it added to its parent, what they "
               "built (its first pipeline line, `+` new or `~` changed), its train score "
               "beside its parent's, and the gate's decision." if self.history else ""),
            *_attempt_lines(self.history, self.seed_knob_values, below, elsewhere, homes),
            "",
            "## Evidence from the traces",
            _table(["what", "value"], _evidence_rows(_trace_rows(
                self.traces, omit=UNSCORED_TRACE_KEYS
                if self.parsimony == 0 and self.score_is_item_mean else (),
                current={k.get("name"): k.get("value") for k in self.knobs},
                seed=self.seed_knob_values),
                [] if self.history_truncated else self.history)),
        ]
        if self.edit_pairs:
            lines += ["", *self._edit_change()]
        if self.reasoning:
            predicted = [str(p.get("child")) for p in self.edit_pairs or () if p.get("prediction")]
            lines += ["", *_reasoning_lines(
                self.reasoning, below, predicted, homes,
                reviewed={str(e.get("candidate")) for e in self.added_steps or ()})]
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
                        [r.get("path"), r.get("optimizer"), fmt_score(r.get("seed_score")),
                         fmt_score(r.get("best_score")), r.get("bound"),
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
                *refused_lines(self.refused, homes),
                *repair_loss_lines(self.repair_losses),
            ]
        elif self.repair_losses:
            lines += ["", "## Your repaired proposals", *repair_loss_lines(self.repair_losses)]
        return lines

    def render(self) -> str:
        """The request, as text, for a model optimizer: the score, the pipeline (once, as
        the view), what this run has tried, what the analysts found, the edits it may
        make, the items, then the reference, and the answer's form last. Fills
        ``omissions`` with any text it had to leave out of an item.

        Within ``request_room``, whatever the items' sizes: a request that would pass it
        is made again with the items "Some items in full" shows (``shown_items_room``,
        `_fit_shown`) given what the rest of the request leaves. Items stay whole while
        they fit, in rank order, the rest named; the first, when it alone passes the room,
        has its longest texts cut to fit, or is named. Each cut is marked in place and in
        ``omissions``. Nothing else gives way: the rest of the request is never cut.

        What one render reads again is worked out once (``rendering``)."""
        from . import evidence

        self.rendering.clear()
        self.rendering["on"] = True
        try:
            with evidence.keeping():
                return self._render_fitted()
        finally:
            self.rendering.clear()

    def _render_fitted(self) -> str:
        """`render`'s work: the request, fitted to ``request_room`` by showing fewer items
        whole (``shown_items_room``); the rest of the request is never cut."""
        text = self._render_request()
        if (not self.request_room or self.shown_items_room
                or len(text) <= self.request_room):
            return text
        items = _ordered_items(self.item_records or ())
        if not items or not self.fit_sizes.get("items"):
            return text  # nothing of the items' texts can give way
        # What the items shown whole may take is what the rest of the request leaves; the
        # rest changes a little with what is shown (a pointer to an item shown whole), so
        # the room is made again from the last render, a few times.
        room = self.request_room - (len(text) - self.fit_sizes.get("items", 0))
        for _ in range(5):
            text = replace(self, shown_items_room=max(1, room))._render_request()
            over = len(text) - self.request_room
            if over <= 0:
                break
            room -= over
        return text

    def _render_request(self) -> str:
        """`render`'s request at the context's ``shown_items_room``."""
        self.omissions.clear()
        self.fit_sizes.clear()
        sections = [
            f"# Update request — generation {self.generation}, epoch {self.epoch}",
            "",
            f"Mode: `{self.mode_name}`. "
            + (f"Best score so far: {fmt_score(self.best_score)}. " if not self.selection else
               # Under checkpoint the current candidate's: which one is best on val only
               # picks the winner, and is never said.
               f"Train score of the current candidate, the one being trained: "
               f"{fmt_score(self.best_score)}. " if self.selection == "checkpoint" else
               f"Train score of the candidate the validation gate ranks first: "
               f"{fmt_score(self.best_score)}. ")
            + (f"Mean over the {self.mean_over} candidate(s) scored so far: "
               f"{fmt_score(self.mean_score)}."
             if self.mean_over else f"Mean score so far: {fmt_score(self.mean_score)}."),
            "",
            *objective_lines(self.parsimony, item_mean=self.score_is_item_mean,
                             gated=self.selection),
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
        # 4: what the analysts found, failures and successes.
        if self.analysis is not None:
            from .analysts import section_lines
            sections += ["", *section_lines(self.analysis)]
        # 5: the steps' settings, what the author says, the blocks it may insert.
        sections += [
            "",
            *_settings_lines(self.knobs),
            "## About this mode",
            self.guidance or "(the mode's author offered no guidance)",
            (
                "Operations the author suggests trying first: "
                + ", ".join(f"`{name}`" for name in self.preferred_edits)
                if self.preferred_edits
                else ""
            ),
            "",
            *(["`score_effect: none` on a block means inserting it cannot change the "
               "score. It is available and it may be worth having, but it is not a way "
               "to improve the number.", ""] if self.insertable else []),
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
                omissions=self.omissions, settled=self._settled(),
                shown_room=self.shown_items_room, sizes=self.fit_sizes)]
        if self.prior_proposals:
            sections += ["", *_prior_proposals_lines(self.prior_proposals)]
        # 7: the reference, last and once: the operations, how nodes compose, what each
        # node has before it runs, each node in full with the schemas, the rules.
        sections += [
            "",
            REFERENCE_HEADING,
            "Everything an edit is written from, each said once: the operations, how nodes "
            "compose, what each node has before it runs, each node in full with the node "
            "schema, and the rules the validator checks (the pipeline view above is the "
            "graph's steps and edges).",
            "",
            "### The operations you may use",
            *operation_lines(self.catalogue, self.node_states, self.edges,
                             protected=self.protected,
                             knob_values={k.get("name"): k.get("value") for k in self.knobs},
                             step_routes=self.step_routes, tools=self.new_node_tools,
                             code_nodes=self.code_nodes),
            "",
            *composition_section(self.catalogue, self.node_states, self.edges,
                                 code_nodes=self.code_nodes,
                                 text_knobs=[k.get("name") for k in self.knobs
                                             if k.get("kind") == "str"],
                                 structure_guidance=self.structure_guidance,
                                 tools=self.new_node_tools, protected=self.protected,
                                 knob_values={k.get("name"): k.get("value")
                                              for k in self.knobs},
                                 frozen=[k.get("name") for k in self.knobs if k.get("frozen")]),
            # Only when the view could not be drawn are the nodes and edges listed here.
            *([] if _fit_state(self.node_states, self.edges, self.knobs) is not None else [
                "The pipeline view could not be drawn; the graph as it declares itself:",
                "Nodes:",
                _table(["id", "kind", "call", "knobs", "notes"],
                       [[n["id"], n["kind"], n.get("call", ""), n.get("knobs", []),
                         n.get("notes", "")] for n in self.nodes]),
                "Edges:",
                _table(["source", "target", "condition"],
                       [[e["source"], e["target"], e.get("condition")] for e in self.edges])]),
            *_available_section(self.available, code_nodes=self.code_nodes),
            "",
            *(["### What this graph allows",
               "Worked out on the pipeline you edit, by an attempt of each edit: what a new "
               "step's `sees` can name, and which removals, replacements and merges are "
               "refused, with why; one that could not be worked out says so. An edit not "
               "listed for a step was accepted.", *self.graph_facts, ""] if self.graph_facts else []),
            *_node_reference(self),
            "### The rules the validator checks",
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
                "parameters, under \"Each step's settings\"); the structure below stays as it "
                "is.")
    return ("You may change the pipeline's structure -- its steps and how they connect, "
            "with the operations in the reference at the end -- and each step's settings -- "
            "its prompts and parameters, under \"Each step's settings\". Both are levers of "
            "the same standing: read where the pipeline fails first, then choose by the cause "
            "the evidence shows.")


def _exact_in_table(value: Any) -> bool:
    """Whether a table cell shows ``value`` as it is: a cell puts a text on one line and
    escapes ``|`` (`_cell`), so a text with either is shown whole below the table."""
    return not (isinstance(value, str) and ("\n" in value or "|" in value))


def _fence(text: str) -> str:
    """A code fence longer than any run of backticks in ``text``."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def operation_lines(catalogue: Sequence[Mapping[str, Any]],
                    node_states: Sequence[dict[str, Any]] = (),
                    edges: Sequence[dict[str, Any]] = (), *, protected: Sequence[str] = (),
                    knob_values: Mapping[str, Any] | None = None,
                    step_routes: Mapping[str, Sequence[str]] | None = None,
                    tools: bool = True, code_nodes: bool = False) -> list[str]:
    """"The operations you may use", each said once: the written-out operations in a
    table (fields and description), then the core operators, one line and one example each
    (`operator_lines`, `nn.core_ops.OPERATORS`). Without ``tools`` (the run offers no tool
    step) no operator offers 'tools'."""
    core = set(CORE_OPERATORS)
    written = [c for c in catalogue if c.get("op") not in core]
    lines = [_table(["op", "kind", "macro", "fields", "description"],
                    [[c.get("op"), c.get("kind"), c.get("macro"),
                      _edit_fields(c.get("fields") or ()), c.get("description")]
                     for c in written])] if written else []
    return lines + operator_lines(catalogue, node_states, edges, protected=protected,
                                  knob_values=knob_values, step_routes=step_routes,
                                  tools=tools, code_nodes=code_nodes)


def _prior_proposals_lines(prior: Sequence[dict[str, Any]]) -> list[str]:
    """Facts about proposals already made earlier in this round."""
    lines = [
        "## Proposals already made in this round",
        f"{len(prior)} proposal(s) earlier in this round, each from a separate call. "
        "Do not duplicate them.",
    ]
    for index, entry in enumerate(prior, 1):
        edits = entry.get("edits") or []
        if not edits and entry.get("edit"):
            edits = [entry["edit"]]
        rationale = str(entry.get("rationale") or "")
        parent = str(entry.get("parent") or "")
        edit_summary = compact_edits(edits, {})
        lines.append(f"- Slot {index}: parent {parent}, edit {edit_summary}"
                     + (f" — {rationale}" if rationale else ""))
    return lines


def _settings_lines(knobs: Sequence[dict[str, Any]], text_chars: int | None = None,
                    exact: bool = True) -> list[str]:
    """"Each step's settings": every knob in a table; a text the table cannot show as it
    is (a line break or a ``|``) is shown there by reference and whole below it, fenced,
    so a patch can quote its target text exactly. ``text_chars`` clips each such text
    (head and tail, marked), for an analyst call that must fit a smaller room. Without
    ``exact`` (a reader that never patches: the analysts), a long line an earlier knob's
    text showed is said as where it was (`evidence.dedup_texts`), so the steps' shared
    paragraphs are read once."""
    # Each text clipped once, here (a clipped one says how much was left out), whether
    # the table shows it or it is shown after the table.
    shown = {str(k["name"]): (_clip_middle(k["value"], text_chars)
                              if text_chars is not None and isinstance(k.get("value"), str)
                              else k.get("value")) for k in knobs}
    whole = [k for k in knobs if not _exact_in_table(shown[str(k["name"])])]
    lines = ["## Each step's settings",
             ("The prompts and parameters the steps read, which `set_knob` moves (a new step "
              "brings its own). A patch quotes its target text exactly: a value in the table "
              "is exact as shown"
              + ("; a text with line breaks or `|` is shown whole after the table." if whole
                 else ".")) if exact else
             ("The prompts and parameters the steps read, which `set_knob` moves; a text with "
              "line breaks or `|` is shown after the table, a line an earlier text showed "
              "said as where it is."),
             _table(["name", "value", "kind", "values", "range", "frozen", "reachable",
                     "description"],
                    [[k["name"], shown[str(k["name"])]
                      if _exact_in_table(shown[str(k["name"])])
                      else f"(shown {'whole ' if shown[str(k['name'])] == k['value'] else ''}"
                           f"below, {len(k['value'])} characters)",
                      k["kind"], k.get("values"), k.get("range"),
                      k["frozen"], k.get("reachable", True), k.get("description", "")]
                     for k in knobs])]
    texts = [shown[str(knob["name"])] for knob in whole]
    if not exact:
        from .evidence import dedup_texts

        texts = dedup_texts(texts, [f"`{knob['name']}`" for knob in whole])
    from .evidence import exact_block

    # Byte-exact on every path (`evidence.exact_block`): a fence no line of the text can
    # close, so its own headings read as none and nothing copied from it gains an escape.
    for knob, text in zip(whole, texts):
        lines += [f"`{knob['name']}`:", *exact_block(text)]
    return [*lines, ""]


def _node_reference(ctx: "ProposalContext") -> list[str]:
    """Each node of the edited graph in full and the node schemas (`_node_specs`), under
    their own heading; nothing when the request carries no node states."""
    specs = _node_specs(ctx.node_states, text_chars=ctx.node_spec_text_chars,
                        section_chars=ctx.node_spec_section_chars,
                        knob_values={k.get("name"): k.get("value") for k in ctx.knobs},
                        person_gates=ctx.person_gates, refer=True,
                        catalogue=ctx.catalogue)
    return [*specs, ""] if specs else []


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
    """The run-wide feedback sections (`optim.feedback`): every change so far, whether
    the last round's added steps contributed, the loop's memory note (`optim.memory`),
    and the integrity judge's verdicts on its proposals (`optim.integrity`); each only
    when its field holds something."""
    from . import feedback, integrity, memory

    out: list[str] = []
    above = [str(e.get("id")) for e in reasoning_shown(ctx.reasoning or [])]
    for lines in (feedback.ledger_lines(ctx.ledger, in_full=above, selection=ctx.selection),
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
                "schema in the reference (or copied from a node's JSON under \"Each node in "
                "full\", with a new id).", ""]
    return ["## Blocks you may insert",
            "Use these node objects verbatim in an edit that takes a node, changing nothing "
            "but their `id`, or write a node from the node schema in the reference:",
            "```json", json.dumps(list(insertable), indent=2, default=str), "```", ""]


def _composition_anchor(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                        protected: Sequence[str]) -> tuple[dict[str, Any], str | None] | None:
    """Where the composition examples attach: an agent of this graph with one plain
    out-edge to a step that is not a gate, and that out-edge's target; else an agent with
    no verdict out-edge (one that ends its pass or the run), and ``None`` for its next
    step, so a graph whose only agent step ends its pass still gets examples (each built
    in a form an attempt takes there, `_completed`). ``None`` when the graph has neither."""
    by_id = {str(state.get("id")): state for state in node_states}
    ends: list[dict[str, Any]] = []
    for node_id in sorted(by_id):
        state = by_id[node_id]
        if state.get("kind") != "agent" or node_id in protected or not state.get("writes"):
            continue
        out = [edge for edge in edges if edge.get("source") == node_id]
        if any(edge.get("condition") is not None for edge in out):
            continue
        if len(out) != 1:
            ends += [state] if not out else []
            continue
        target = by_id.get(str(out[0].get("target"))) or {}
        if target.get("kind") in ("agent", "op"):
            return state, str(out[0].get("target"))
    return (ends[0], None) if ends else None


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


def _or_list(names: Sequence[str], last: str = "or") -> str:
    """``names`` in backticks, the last joined by ``last``."""
    quoted = [f"`{name}`" for name in names]
    return f" {last} ".join([", ".join(quoted[:-1]), quoted[-1]]) if len(quoted) > 1 \
        else "".join(quoted)


#: The lead of "Structural operators". It makes no claim about wiring: what a new step
#: sees and where its output goes are each operator's own generated line
#: (`nn.core_ops.OPERATORS`, `core_ops.REACHES`), the single source.
STRUCTURAL_LEAD = (
    "The structural operators, each one line: what kind of step "
    f"({_or_list(STEP_KINDS)}), where (a step id), and "
    "its text (a prompt, or code). The system does the wiring, as each operator's line "
    "below says. Ids, files and facts are generated; each text goes into a knob of its own "
    "(`<id>__task`, `<id>__code`) for a later `set_knob`; the candidate says what was "
    "wired. Every text has a default (the neighbour's, or a neutral line), and each "
    "operator counts as one change. The written-out operations in the table above stay "
    "for what these do not express.")


def composition_section(catalogue: Sequence[dict[str, Any]], node_states: Sequence[dict[str, Any]],
                        edges: Sequence[dict[str, Any]], *, code_nodes: bool = False,
                        text_knobs: Iterable[str] | None = None,
                        structure_guidance: bool = False, tools: bool = False,
                        protected: Sequence[str] = (),
                        knob_values: Mapping[str, Any] | None = None,
                        frozen: Collection[str] = ()) -> list[str]:
    """"How nodes compose": the language's primitives (each node kind, where a `{{fact}}`
    is filled in, tools, gates and loops, running out of output, crashes, forks), facts
    only, then one example of each composition shape with this graph's ids
    (`composition_examples`); each operator's own line and example are under the
    operations (`operator_lines`). With ``structure_guidance``, what makes a new step pay
    off (`feedback.STRUCTURE_GUIDANCE`). Nothing when the run permits no structural op."""
    from refactory_dsh import snippet

    if not structural_ops(catalogue):
        return []
    ops = {op.get("op") for op in catalogue}
    reader = ("a code op that reads the file and `produces` the fact" if code_nodes else
              "an op that reads the file and `produces` the fact")
    filled = _fact_filled_knobs(node_states, text_knobs)
    lines = [
        "### How nodes compose",
        "The pipeline is made of four kinds of step, and any of them may be put anywhere the "
        "validator's rules (listed below) allow:",
        "- **agent**: one model call. Its prompt fills each `{{name}}` from a fact (or knob) "
        "produced upstream; its reply is written to its first `writes` file and, when it "
        "declares `produces`, also becomes that fact. A new agent without `tools` "
        "sees only its prompt and the files it names in `sees` (they become its declared reads, pasted in after the prompt): "
        "hand it anything else as `{{facts}}`. "
        f"A file's text becomes a fact through {reader}, or when the agent that writes the "
        "file declares `produces`.",
        "- **where `{{fact}}` is filled in**: only in an agent's `prompt_template` (its "
        "`<id>__task` knob) and its system prompt (its `<id>__system_prompt` knob)"
        + (f", and in {', '.join(f'`{name}`' for name in filled)}, which an op that fills "
           "facts reads" if filled else "")
        + ". Any other knob is read as written, so a `{{fact}}` there would never arrive "
        "and is refused (V20).",
        tools_line(ops) if tools else NO_TOOLS_LINE,
    ]
    if code_nodes:
        sealed_imports, has_packages = _sealed_code_imports(node_states)
        lines += [
            f"- **code op**: `\"call\": \"{snippet.RUN}\"` with `code`, Python that defines "
            "`main(inp)`. `inp` is `{\"files\": {each declared read whose content is text: "
            "its text}, \"facts\": {each name in its `inputs`: its value}}`; binary reads "
            "(e.g. .xlsx) are silently absent from `inp[\"files\"]`, and without "
            "`\"sealed\": true` a code step cannot read them from disk (the sandbox has no "
            "file access). `main` returns "
            "`{\"facts\": {...}, \"files\": {...}}`, each one declared in its `produces` / "
            "`writes`. An insert takes `produces` (default: one fact named after the step's "
            "id) and `writes` (default: none; a takes_over adds the taken-over files). "
            "`main` must return exactly the declared names. "
            "It costs no model call. Limits: imports only these modules: "
            f"{snippet.ALLOWED_IMPORTS_TEXT}; at most {snippet.MAX_CODE_BYTES} "
            f"bytes, {snippet.WALL_S} s, no files, network, open/exec/eval or dunders (rule "
            "V18). The code goes into the node's own `<id>__code` knob, which a later "
            "`set_knob` patches or rewrites.",
        ]
        if sealed_imports:
            sealed_io = (
                " The files it names in `sees` become its declared reads and are present "
                "as files in the working directory, so it can `open()` them directly "
                "(including binary files such as .xlsx that `inp[\"files\"]` does not carry). "
                "It may also write its declared `writes` "
                "to the working directory; those files are copied back to the run directory "
                "after a successful run, so downstream steps read them.")
            if has_packages:
                importable = "; ".join(sealed_imports)
                lines.append(
                    "  With `\"sealed\": true` on the insert (or loop/route), the code step "
                    "runs in the Task's tool environment instead: it may import anything "
                    f"installed in that environment, including {importable}; there is no "
                    f"import allowlist (the seal is the boundary). Up to "
                    f"{snippet.SEALED_WALL_S} s and {snippet.SEALED_MEMORY_MB} MB. It must "
                    "still define `main(inp)` and parse."
                    f"{sealed_io} "
                    "Without `\"sealed\": true` a code step imports only the standard "
                    "library.")
            else:
                lines.append(
                    "  With `\"sealed\": true` on the insert (or loop/route), the code step "
                    "runs in the Task's tool environment instead: it may import the packages "
                    f"that environment provides ({'; '.join(sealed_imports)}), with no import "
                    f"allowlist (the seal is the boundary), up to {snippet.SEALED_WALL_S} s "
                    f"and {snippet.SEALED_MEMORY_MB} MB. It must still define `main(inp)` "
                    "and parse."
                    f"{sealed_io} "
                    "Without `\"sealed\": true` a code step imports only the standard "
                    "library.")
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
        "without answering is the model failing that question. It is not retried unless the "
        "step opts in (retry.on): the step "
        "has no answer, and the item goes on without it (a gate after it sees no answer and "
        "may reloop; a fork's join sees that branch without one; a code op's `inp['files']` "
        "holds only the files that exist; an item left with nothing to grade scores 0).",
        "- **running out of turns**: an agent that uses its whole turn cap without finishing "
        "is the model failing that question, like running out of output. It is not retried "
        "unless the step opts in (retry.on): "
        "the step has no answer, and the item goes on without it, as above.",
        "- **crashes**: when the worker or product fails (not the model's own output or turn "
        "cap), an agent step in the per-item pass that is safe to rerun (no tools, its own "
        "report files only, or a workspace) is run again in a fresh session, up to 2 more "
        "times; if every session crashes it has no answer, as above. An item that times out "
        f"is a {HARNESS_TIMEOUT}.",
    ]
    if {"parallel", "parallelize"} & ops:
        lines.append(
            "- **fork**: steps run at once between a fork and a join; branches write distinct "
            "files and facts. `parallelize` and `branch` wire what the branches make into the "
            "step after the join; the written-out `parallel` operation leaves that to you.")
    if structure_guidance:
        from .feedback import STRUCTURE_GUIDANCE
        lines.append(STRUCTURE_GUIDANCE)
    examples, refused = composition_examples(catalogue, node_states, edges,
                                             protected=protected, code_nodes=code_nodes,
                                             tools=tools, knob_values=knob_values,
                                             frozen=frozen)
    if examples:
        lines += ["", COMPOSITION_LEAD]
        for shape, bundle in examples:
            lines.append(f"- {shape}: "
                         + json.dumps({"edits": bundle} if len(bundle) > 1 else bundle[0]))
    if refused:
        lines.append(REFUSED_LEAD + "; ".join(f"{shape}: {why}" for shape, why in refused)
                     + ".")
    return lines + [""]


#: What heads the composition examples: vocabulary, one of each shape, none preferred.
COMPOSITION_LEAD = ("One example of each composition shape, with this graph's ids (how to "
                    "write it; none is preferred):")
#: What heads the shapes this graph refuses, each with its refusal.
REFUSED_LEAD = "Shapes not shown here, with the refusal of the forms tried: "
#: The composition shapes, in the order shown (`composition_examples`).
COMPOSITION_SHAPES = ("a new agent step", "a parallel fork and join",
                      "a verdict loop (a judge that can send a step back)",
                      "a route (a judge choosing the next step)", "a code op",
                      "an agent step with tools")


def composition_examples(catalogue: Sequence[dict[str, Any]],
                         node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                         *, protected: Sequence[str] = (), code_nodes: bool = False,
                         tools: bool = False, knob_values: Mapping[str, Any] | None = None,
                         frozen: Collection[str] = ()
                         ) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[tuple[str, str]]]:
    """One short example of each composition shape (`COMPOSITION_SHAPES`) this run offers,
    with this graph's ids and placeholder texts: how each is written, never when to use it.
    A shape is shown only when its operator is offered (a code op with ``code_nodes``, a
    tool step with ``tools`` and an operator that makes one, `tool_step_ops`) and its bundle
    applies to this graph as an attempt takes it and as it runs (`_completed`, then
    `_example_refusal`: no ``frozen`` knob moved, every step it adds read, no step's outputs
    taken over; ``tests/test_dsl_coverage.py`` holds each one to `Optimizer.attempt`). Returns
    the shapes shown and, for each offered shape no form of which this graph takes, the
    refusal of every form tried. A take-over form is shown only when the new step can
    write every file it takes over (`_can_write`), its text saying it must write them.
    Nothing when the graph has no step to anchor them."""
    ops = {op.get("op") for op in catalogue}
    anchor = _composition_anchor(node_states, edges, protected)
    state = _example_state(node_states, edges, knob_values)
    if anchor is None or state is None:
        return [], []
    step = str(anchor[0]["id"])
    code_id = f"{step}_code"
    code = ("def main(inp):\n    result = len(inp['facts'])  # <what it computes>\n"
            f"    return {{'facts': {{{code_id!r}: str(result)}}, 'files': {{}}}}\n")
    agent = {"kind": "agent", "text": "<what the new step does>"}
    # The step just before the anchor, when one plain edge leads to it: branches there feed it.
    into = [e for e in edges if e.get("target") == step]
    feeding = [str(into[0]["source"])] if len(into) == 1 and into[0].get("condition") is None \
        else []
    branches = [{"kind": "agent", "text": "<what one branch does>"},
                {"kind": "agent", "text": "<what another branch does>"}]
    # Each shape: its operator and its forms, the first this graph takes is shown.
    shapes: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("insert", COMPOSITION_SHAPES[0], [
            {"op": "insert", "before": step, **agent}, {"op": "insert", "after": step, **agent}]),
        ("parallelize", COMPOSITION_SHAPES[1], [
            {"op": "parallelize", "after": step, "new": branches},
            *({"op": "parallelize", "after": before, "new": branches} for before in feeding),
            {"op": "parallelize", "steps": [step], "copies": 1}]),
        ("loop", COMPOSITION_SHAPES[2], [
            {"op": "loop", "step": step, "judge": "agent", "text": "<what to check>"}]),
        ("route", COMPOSITION_SHAPES[3], [
            {"op": "route", "after": step, "judge": "agent", "text": "<how to choose>"},
            *({"op": "route", "after": before, "judge": "agent", "text": "<how to choose>",
               "to": {"proceed": step, "revise": {**agent, "to": step}}} for before in feeding)]),
    ]
    if code_nodes:
        sealed_imports, _ = _sealed_code_imports(node_states)
        code_extra: dict[str, Any] = {"sealed": True} if sealed_imports else {}
        shapes.append(("insert", COMPOSITION_SHAPES[4], [
            {"op": "insert", "before": step, "kind": "code", "id": code_id, "text": code,
             "produces": [code_id], **code_extra},
            {"op": "insert", "after": step, "kind": "code", "id": code_id, "text": code,
             "produces": [code_id], **code_extra}]))
    if tools and "insert" in tool_step_ops(ops):
        tool = {"kind": "agent", "tools": ["bash"], "text": "<what to find out with tools>"}
        shapes.append(("insert", COMPOSITION_SHAPES[5], [
            {"op": "insert", "before": step, **tool}, {"op": "insert", "after": step, **tool}]))
    out: list[tuple[str, list[dict[str, Any]]]] = []
    refused: list[tuple[str, str]] = []
    for op, shape, forms in shapes:
        if op not in ops:
            continue
        reasons: list[str] = []
        for form in forms:
            done = _completed(state, [form])
            if done is not None:
                done = _with_outputs_said(done, state)
            why = _example_refusal(state, done if done is not None else [form], frozen)
            if done is not None and why is None:
                out.append((shape, done))
                break
            why = why or "refused by the operator's precondition"
            if why not in reasons:
                reasons.append(why)
        else:
            refused.append((shape, " / ".join(reasons)))
    return out, refused


def tool_step_ops(ops: Iterable[str]) -> list[str]:
    """The operations among ``ops`` that can make a new agent step with `tools`: the
    coverage map's (`nn.coverage.made_by`, which ``tests/test_dsl_coverage.py`` holds to each
    one making a sealed tool step), core operators first in their order, then the
    written-out ones by name."""
    from refactory_dsh.nn import coverage

    makers = set(coverage.made_by("node field", "tools")) & set(ops)
    return [*(op for op in CORE_OPERATORS if op in makers),
            *sorted(makers - set(CORE_OPERATORS))]


def tools_line(ops: Iterable[str]) -> str:
    """What the request says of tools when a new step may have them: an existing step's own
    tool rules (its ``tool_policy``, its ``tools``) are not a limit on a step an edit adds;
    the operations that can add one are generated (`tool_step_ops`), so the list is always
    the exact set this run offers. `NO_TOOLS_LINE` when none of ``ops`` can."""
    makers = tool_step_ops(ops)
    if not makers:
        return NO_TOOLS_LINE
    return ("- **tools**: every existing step keeps its own tool rules (a step that has no "
            "tools or refuses them stays so). A new agent step may still have `tools` "
            f"(made by {_or_list(makers)}: in the edit's own 'tools', or in a step or node "
            "it carries): it runs sealed, in a fresh workspace holding "
            "only the files it reads -- in the per-item part, unless its 'sees' "
            "names its inputs, the item's input files (the pipeline view lists "
            "them) beside what it sees -- with no network and no grader files.")
#: The same when this run offers no tool step (no seal here, or ``new_step_tools`` off).
NO_TOOLS_LINE = ("- **tools**: every existing step keeps its own tool rules, and in this run "
                 "a new step has no tools.")


#: The core operators a request explains, in its order (`nn.core_ops.CORE_OPS`).
CORE_OPERATORS = nn_edits.CORE_OPS


def _example_ids(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                 protected: Sequence[str]) -> dict[str, str]:
    """Ids of this graph for the operators' examples: a step (an agent with one plain
    out-edge to a step) and its next step, and a fork, where the graph has them."""
    ids: dict[str, str] = {}
    anchor = _composition_anchor(node_states, edges, protected)
    if anchor is not None and anchor[1] is not None:
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
    except Exception as exc:  # noqa: BLE001 - logged; the examples then use placeholders
        import logging

        logging.getLogger(__name__).warning("the request's graph does not read back, so the "
                                            "operator examples use placeholders: %s", exc)
        return None
    return {"graph": graph.to_state(),
            "knobs": {k: {"value": v, "spec": {"kind": "str"}}
                      for k, v in (knob_values or {}).items()}}


def _takes_over_parts(takes: Any) -> tuple[str, frozenset[str] | None]:
    """Best-effort parse: on refusal (malformed value) returns ``("", None)``."""
    from refactory_dsh.nn.core_ops import _parse_takes_over

    try:
        return _parse_takes_over(takes)
    except Exception:  # noqa: BLE001 — enrichment, never a refusal
        return "", None


def _can_write(spec: Mapping[str, Any], file: str, files: Sequence[str]) -> bool:
    """Whether a new step of ``spec`` can produce ``file`` among the ``files`` it takes over:
    an agent with tools writes any file, one without only its reply (so one file); the code
    example returns facts, no files, so it takes over none."""
    if spec.get("kind", "agent") == "agent":
        return bool(spec.get("tools")) or len(files) == 1
    return False


def _with_outputs_said(bundle: list[dict[str, Any]], state: Mapping[str, Any]
                       ) -> list[dict[str, Any]]:
    """``bundle`` with each take-over's placeholder text saying which files the new step
    must write: the outputs of the step it takes over, which its readers go on reading."""
    nodes = (state.get("graph") or {}).get("nodes") or {}
    out = []
    for edit in bundle:
        taken = edit.get("takes_over")
        if taken:
            target_id, selected = _takes_over_parts(taken)
            all_files = list((nodes.get(target_id) or {}).get("writes") or ())
            files = ([f for f in all_files if f in selected] if selected is not None
                     else all_files)
        else:
            target_id = None
            files = []
        if files and isinstance(edit.get("text"), str) and edit.get("kind", "agent") == "agent":
            edit = {**edit, "text": f"{edit['text']} It writes {_or_list(files, 'and')}, the outputs "
                                    f"of `{target_id}` it takes over."}
        out.append(edit)
    return out


def _example_refusal(state: Mapping[str, Any], bundle: Sequence[dict[str, Any]],
                     frozen: Collection[str] = ()) -> str | None:
    """Why ``bundle`` is no example for ``state``, or ``None``: an edit refused (a knob of
    ``frozen`` moved included), a step it adds left with no reader
    (`core_ops.unread_by_bundle`), or a step's outputs taken over (the new step would have to
    write what that step writes, which a placeholder text does not say)."""
    from refactory_dsh.nn import core_ops

    nodes = (state.get("graph") or {}).get("nodes") or {}
    for edit in bundle:
        if not edit.get("takes_over"):
            continue
        target_id, selected = _takes_over_parts(edit["takes_over"])
        all_files = list((nodes.get(target_id) or {}).get("writes") or ())
        files = ([f for f in all_files if f in selected] if selected is not None else all_files)
        unable = [f for f in files if not _can_write(edit, f, files)]
        if unable:
            return (f"its new step reaches a reader only by taking over the outputs of "
                    f"`{target_id}` ({_or_list(files, 'and')}), which this new "
                    f"{edit.get('kind', 'agent')} step does not write")
    # Each knob as a spec `set_knob` reads back: its value its default, its kind its
    # value's, frozen as declared.
    kinds = {bool: "bool", int: "int", float: "float", str: "str"}
    knobs = {name: {"value": entry.get("value"),
                    "spec": {"default": entry.get("value"),
                             "kind": kinds.get(type(entry.get("value")), "str"),
                             "frozen": name in frozen}}
             for name, entry in (state.get("knobs") or {}).items()}
    state = {**state, "knobs": knobs}
    states, current = [], dict(state)
    try:
        for edit in bundle:
            current = nn_edits.apply(edit, current)
            states.append(current)
    except Exception as exc:  # noqa: BLE001 - a refused example is said, not shown
        return str(exc)
    return core_ops.unread_by_bundle(state, states, bundle)


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

    placeholders = {**ids, "step": "<step>", "next step": "<next step>",
                    **({"fork": "<fork>"} if "fork" in ids else {})}
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
        if nn_edits.precheck(example, dict(state)) is None \
                and _completed(state, [example]) == [example]:
            return option
    return placeholders


#: How a core operator's refusal names the edit that replaces the step (`core_ops`).
_REPLACE_HINT = "replace it with "


def checked_hint(reason: str, check: Callable[[dict[str, Any]], Any]) -> str:
    """``reason`` with the replacing edit it names checked by ``check`` (an attempt's
    verdict: a candidate, or its refusals): kept when it would be accepted, otherwise
    turned into "replacing it is refused too (<why>)" -- for a rewire giving the step its
    own text (`core_ops.OWN_TEXT_HINT`), "giving it its own text is refused too (<why>)".
    The operators' own precheck knows only the graph; an attempt also refuses on protected
    steps, frozen and guarded knobs, wiring and validation, so the hint is held to that."""
    from refactory_dsh.nn.core_ops import CODE_HINT_NOTE, OWN_TEXT_HINT

    found = [(at, phrase) for phrase in (_REPLACE_HINT, OWN_TEXT_HINT)
             if (at := reason.find(phrase)) >= 0]
    if not found:
        return reason
    start, phrase = min(found)
    try:
        edit, end = json.JSONDecoder().raw_decode(reason, start + len(phrase))
    except ValueError:
        return reason
    verdict = check(edit)
    if not isinstance(verdict, list):
        code = any(isinstance(step, dict) and step.get("kind") == "code"
                   for step in edit.get("into") or ())
        # Accepted without running the new code on recorded inputs (`attempt` skips that
        # for a hint: a process per request), which an attempt of the edit itself does.
        return (f"{reason[:end]} (not dry-run){reason[end:]}" if code else reason)
    why = "; ".join(str(r).removeprefix("E-precondition: ") for r in verdict)
    rest = reason[end:].removeprefix(CODE_HINT_NOTE)
    refused = ("giving it its own text is refused too" if phrase == OWN_TEXT_HINT
               else "replacing it is refused too")
    return f"{reason[:start]}{refused} ({why}){rest}"


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
REFERENCE_HEADING = "## Reference: the operations, the nodes and the rules"
REFERENCE_POINTER = ("The operations you may use, how nodes compose, every node in full (its "
                     "declarations, what it has before it runs), the node schemas and the "
                     "validator's rules are in the reference at the end.")

#: What the pipeline view's words mean.
def _view_lead() -> str:
    from refactory_dsh.nn.pipeline_view import EXHAUSTED, FORMS

    return (
        "The pipeline, one line per step, generated from the graph in the operators' own "
        "words: `sees` lists a step's "
        "inputs as `\"sees\"` names them, `name@maker` (`@run`: no step makes it); a plain "
        "op that only assembles a model step's input is folded in as `name (via op: its "
        "inputs)`; an agent with tools sees the run dir, its own workspace, or a sealed one "
        "(its inputs only); `told` is "
        "each knob whose text reaches the step, its opening words (`*` frozen; `?` read by "
        "an op with several outputs, for all of them), or a code step's code; `->` what it "
        "writes; a per-item block's `item files` are files the task lays into each item's "
        "pass, under the same name for every item, and the steps that read them: a new step "
        "reads one by naming it (or the data step) in 'sees'; "
        + "; ".join(f"`{form}`" for form in FORMS.values())
        + "; a loop that does not go on with its last attempt past its cap says so: `"
        + EXHAUSTED.format(policy="fail_item").lstrip(", ") + "` (only that item fails) or `"
        + EXHAUSTED.format(policy="halt").lstrip(", ") + "` (the whole run stops)"
        + "; `[optional]`; `[protected]`.")


PIPELINE_VIEW_LEAD = _view_lead()


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


def _fork_inputs(fork_id: str, state: dict[str, Any]) -> str:
    """What a new branch of ``fork_id`` can name in ``sees``: each fact or file there before
    the fork as the pipeline view writes it (``name@maker``), one built by a plain op that
    reads a text knob marked with it (it may carry that knob's instructions), and what
    ``"task"`` gives there. Read from the graph; "unavailable: <why>" (and a warning) when
    it does not read back."""
    from refactory_dsh import snippet
    from refactory_dsh.nn import core_ops
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.operators import carried_knobs, made_by
    from refactory_dsh.nn.pipeline_view import RUN_MAKER

    try:
        graph = Graph.from_state(state["graph"])
        there = core_ops._readable(graph, state, set(core_ops._available(graph).get(fork_id, set())))
        task = core_ops._task_inputs(graph, state) & there
    except Exception as exc:  # noqa: BLE001 - said in the row and logged, never raised
        import logging

        logging.getLogger(__name__).warning("no inputs for fork %s: %s", fork_id, exc)
        return f"unavailable: {type(exc).__name__}: {exc}"
    makers = made_by(graph)
    values = {k: (e or {}).get("value") for k, e in (state.get("knobs") or {}).items()}

    def named(kind: str, name: str) -> str:
        made = makers.get((kind, name)) or []
        texts = [k for m in made if graph.nodes[m].kind == "op"
                 and not snippet.is_code_node(graph.nodes[m])
                 for k in carried_knobs(graph.nodes[m]).get((kind, name), ())
                 if isinstance(values.get(k), str)]
        return (f"{name}@{','.join(made) or RUN_MAKER}"
                + (f" (+knob {', '.join(dict.fromkeys(texts))})" if texts else ""))

    order = sorted(there, key=lambda key: (key[0] != "fact", key[1]))
    return (f"a new branch here can see: {', '.join(named(*key) for key in order) or 'nothing'}"
            f"; \"task\" gives {', '.join(n for _k, n in sorted(task)) or 'nothing'}")


def graph_facts(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                knobs: Sequence[Mapping[str, Any]], catalogue: Sequence[Mapping[str, Any]],
                check: Callable[[dict[str, Any]], Any],
                failures: list[dict[str, str]] | None = None) -> list[str]:
    """Facts about this graph an edit is written against, worked out once by the loop
    and carried as data (``ProposalContext.graph_facts``): what ``"sees": ["task"]``
    gives a new step, what a new branch of each fork can see (`_fork_inputs`), and for
    each step the refusal ``check`` (an attempt's verdict on the edited candidate) gives
    to removing it, replacing it by one step that sees the task, and merging it with the
    step after it. Accepted edits are not listed. A fact that cannot be worked out is
    said so in its row (``not worked out (<error>)``), never left out, and appended to
    ``failures`` (``{"fact", "step", "error"}``) for the run's journal. Nothing when the
    run permits no core operator."""
    from refactory_dsh.nn.core_ops import TASK, _readable, _task_inputs
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.operators import TASK_MEANING

    failed = failures if failures is not None else []

    def not_worked_out(fact: str, step: str, exc: Exception) -> str:
        error = f"{type(exc).__name__}: {exc}"
        failed.append({"fact": fact, "step": step, "error": error})
        return f"{fact} not worked out ({error})"

    ops = {op.get("op") for op in catalogue} & set(CORE_OPERATORS)
    if not ops:
        return []
    try:
        state = _fit_state(node_states, edges, knobs)
        if state is None:
            raise ValueError("the graph does not read back")
        graph = Graph.from_state(state["graph"])
    except Exception as exc:  # noqa: BLE001 - said in the section and journaled
        return [f"- The graph's facts are {not_worked_out('the graph', '', exc)}."]
    out: list[str] = []
    try:
        task = sorted(_readable(graph, state, _task_inputs(graph, state)),
                      key=lambda key: (key[0] != "fact", key[1]))
    except Exception as exc:  # noqa: BLE001 - said in the section and journaled
        task = []
        out.append(f"- What `\"sees\": [\"{TASK}\"]` gives is "
                   f"{not_worked_out('the task inputs', '', exc)}.")
    if task:
        out.append(f"- `\"sees\": [\"{TASK}\"]` gives a new step "
                   f"{', '.join(name for _k, name in task)}, where it is there before the "
                   f"step: {TASK_MEANING}.")
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if node.kind == "fork" and "branch" in ops:
            out.append(f"- fork `{node_id}`: {_fork_inputs(node_id, state)}.")
            continue
        if node.kind not in ("agent", "op"):
            continue
        plain_next = [e.target for e in graph.out_edges(node_id) if e.condition is None
                      and graph.nodes[e.target].kind in ("agent", "op")]
        tries = [("remove", {"op": "remove", "step": node_id}),
                 ("replace by one step", {"op": "split", "step": node_id,
                                          "into": [{"kind": "agent", "sees": ["task"]}]}),
                 *([("merge with the next step", {"op": "merge",
                                                   "steps": [node_id, plain_next[0]]})]
                   if len(plain_next) == 1 else [])]
        refused = []
        for label, edit in tries:
            if edit["op"] not in ops:
                continue
            try:
                verdict = check(edit)
            except Exception as exc:  # noqa: BLE001 - said in the row and journaled
                refused.append(not_worked_out(label, node_id, exc))
                continue
            if isinstance(verdict, list):
                reason = "; ".join(str(r).removeprefix("E-precondition: ") for r in verdict)
                refused.append(f"{label} is refused ({reason})")
        if refused:
            out.append(f"- `{node_id}`: " + "; ".join(refused) + ".")
    return out


def operator_lines(catalogue: Sequence[dict[str, Any]], node_states: Sequence[dict[str, Any]],
                   edges: Sequence[dict[str, Any]], *, protected: Sequence[str] = (),
                   knob_values: Mapping[str, Any] | None = None,
                   step_routes: Mapping[str, Sequence[str]] | None = None,
                   tools: bool = True, code_nodes: bool = False) -> list[str]:
    """The structural operators: what the system wires for every one, then each core operator
    the run permits in one line -- its fields and its catalogue description (its
    `nn.core_ops.OPERATORS` line, and a written-out form where it has one) -- with one
    example, its ids taken from this graph where it has them. When the run has routes to
    choose from (``step_routes``), one line says the ``route`` field and its values; without,
    no operator lists it. Nothing when the run permits none."""
    from refactory_dsh.nn import core_ops

    ops = {op.get("op") for op in catalogue}
    by_op = {op.get("op"): op for op in catalogue}
    chosen = [name for name in CORE_OPERATORS if name in ops]
    if not chosen:
        return []
    ids = _example_ids(node_states, edges, protected)
    removable = _removable_step(node_states, edges, protected,
                                _example_state(node_states, edges, knob_values))
    lines = ["", STRUCTURAL_LEAD]
    state = _example_state(node_states, edges, knob_values)
    for name in chosen:
        spec = core_ops.OPERATORS[name]
        # Ids this graph's own steps fill where the operator applies to them; a step this
        # graph can lose for `remove`; placeholders otherwise, never a step it refuses.
        own = {**ids, "step": removable or "<step>"} if name == "remove" else _fitting_ids(
            name, ids, node_states, edges, protected, state)
        entry = by_op.get(name) or {}
        description = str(entry.get("description") or f"{spec['line']}.")
        fields = [f for f in entry.get("fields") or spec["fields"]
                  if step_routes or f != core_ops.ROUTE_FIELD]
        if not tools:
            # A run that offers no tool step offers no 'tools' (`NO_TOOLS_LINE`).
            description, fields = core_ops.without_tools(description, fields)
        lines.append(f"- `{name}` ({_core_fields(fields)}): "
                     f"{description} E.g. " + json.dumps(core_ops.example(name, own)))
    if step_routes:
        lines.append(core_ops.route_field_line(step_routes))
    if tools or code_nodes:
        lines += tool_environment_lines(node_states, tools=tools, code_nodes=code_nodes)
    return lines


def tool_environment_lines(node_states: Sequence[Mapping[str, Any]], *,
                           tools: bool = True, code_nodes: bool = False) -> list[str]:
    """What a sealed step's commands run with in each per-item part whose Task declares
    it (`task.Task.tool_environment`), in the Task's own words: a fact, said only where the
    run offers a tool step or a sealed code step. Nothing for a Task that declares none
    (or cannot be read)."""
    from refactory_dsh.task import TaskError, task_tool_environment

    lines: list[str] = []
    for node in node_states:
        if node.get("kind") != "data" or not node.get("task_ref"):
            continue
        try:
            environment = task_tool_environment(str(node["task_ref"]))
        except (TaskError, ImportError):
            continue
        if environment is not None:
            kinds = []
            if tools:
                kinds.append("a step with `tools`")
            if code_nodes:
                kinds.append("a sealed code step (`\"sealed\": true`)")
            line = (f"- in the per-item part `{node.get('id')}`, "
                    f"{' or '.join(kinds)} runs its commands with {environment.summary}")
            if line not in lines:
                lines.append(line)
    return lines


def _sealed_code_imports(node_states: Sequence[Mapping[str, Any]]
                         ) -> tuple[list[str], bool]:
    """What a sealed code step may import, derived from the Task's tool environment.

    Returns ``(import_lines, has_packages)``: when any environment declares
    ``packages``, each line names the packages by name and version
    (``"openpyxl 3.1.5, pandas 2.3.3"``); otherwise each line is the
    environment's ``summary`` (the proposer knows what is there, but not
    exactly which modules). ``has_packages`` is True when every environment
    declared its packages.
    """
    from refactory_dsh.task import TaskError, task_tool_environment

    seen: list[str] = []
    all_have_packages = True
    for node in node_states:
        if node.get("kind") != "data" or not node.get("task_ref"):
            continue
        try:
            environment = task_tool_environment(str(node["task_ref"]))
        except (TaskError, ImportError):
            continue
        if environment is None:
            continue
        if environment.packages:
            line = ", ".join(f"{name} {version}" for name, version in environment.packages)
        else:
            line = environment.summary
            all_have_packages = False
        if line not in seen:
            seen.append(line)
    return seen, bool(seen) and all_have_packages


#: The core operators whose new step takes 'to' (where its output goes).
_TAKES_TO = ("insert", "branch", "map")


#: The text of the step a completion adds to take over the graded step's outputs.
_TAKE_OVER_TEXT = "<write the final result from the work above>"


def _reader_edit(after: Mapping[str, Any], node_id: str, edit: Mapping[str, Any],
                 before: Mapping[str, Any], first: str | None = None
                 ) -> tuple[str, dict[str, Any]] | None:
    """How a new step ``node_id`` that nothing reads in ``after`` reaches a reader, as one
    edit, the way its refusal's hint says (`core_ops._reader_hint`): ``("append", edit)``
    rewiring the nearest later agent or code step to see it as well as what it sees now,
    or adding a step that takes over the graded step's outputs after it (after its join
    when it is a fork's branch); ``("replace", edit)`` for an ``insert`` given
    ``takes_over`` itself; with ``first``, only a rewire of that step. ``None`` when no
    step could read it."""
    from refactory_dsh.nn import core_ops

    graph = nn_edits._graph(dict(after))
    later = [r for r in core_ops._readers_from(graph, node_id, after)
             if r != node_id and graph.nodes[r].kind == "agent"]
    if first is not None:
        # Only that step (the one the new step was put before), when it is an agent.
        later = [r for r in later if r == first]
        if not later:
            return None
    if later:
        reader = graph.nodes[later[0]]
        sees = sorted({name for _k, name in core_ops._readable(
            graph, after, core_ops._consumed(reader, after))})
        return "append", {"op": "rewire", "step": reader.id, "sees": [*sees, node_id]}
    graded = core_ops._graded_step(graph, node_id, edit, before)
    if graded is None or edit.get("op") == "route":
        return None
    if edit.get("op") == "insert" and "takes_over" not in edit:
        return "replace", {**edit, "takes_over": graded}
    out = [e.target for e in graph.out_edges(node_id)]
    at = out[0] if len(out) == 1 and graph.nodes[out[0]].kind == "join" else node_id
    return "append", {"op": "insert", "after": at, "kind": "agent", "text": _TAKE_OVER_TEXT,
                      "takes_over": graded}


def _completed(state: Mapping[str, Any], bundle: list[dict[str, Any]]
               ) -> list[dict[str, Any]] | None:
    """``bundle`` as an example an attempt takes: a step it adds that nothing reads gets
    'to', the nearest prompt after it that can take it (`core_ops.reader_knobs`: a prompt
    knob, or an agent whose prompt is no knob, which 'to' re-slots), where its
    operator takes one; else the edit its refusal's hint names (`_reader_edit`: rewire a
    later step to see it, or take over the graded step's outputs). None when one stays
    unread (``E-unread``) or an edit of it is refused here. A state built without the
    knobs' values (which say what a prompt takes) returns it as it is."""
    from refactory_dsh.nn import core_ops

    bundle = [dict(edit) for edit in bundle]
    if not state.get("knobs"):
        return bundle
    for _ in range(2 * len(bundle) + 2):
        states, current = [], dict(state)
        try:
            for edit in bundle:
                current = nn_edits.apply(edit, current)
                states.append(current)
        except nn_edits.EditError as exc:
            # An operator's own refusal here (a take-over that reaches nothing) is not
            # shown; another error is this sketch state's, for the example's own check.
            return None if exc.code == "E-precondition" else bundle
        except Exception:  # noqa: BLE001 - not this check's to judge
            return bundle
        unread = core_ops.unread_new_steps(state, current)
        if not unread:
            return bundle
        index = next(i for i, after in enumerate(states)
                     if unread[0] in ((after.get("graph") or {}).get("nodes") or {}))
        edit = bundle[index]
        if edit.get("op") == "insert" and edit.get("before") is not None:
            # A step put before another feeds that one: it is rewired to see it too, the
            # new step's id said, so the rewire names an id the insert declares.
            found = _reader_edit(current, unread[0], edit, state, first=str(edit["before"]))
            if found is not None and found[0] == "append":
                bundle[index] = {**edit, "id": unread[0]}
                bundle.append(found[1])
                continue
        knobs = core_ops.reader_knobs(states[index], unread[0], steps=True)
        if edit.get("op") in _TAKES_TO and "to" not in edit and knobs:
            bundle[index] = {**edit, "to": knobs[0]}
            continue
        new = edit.get("new") if edit.get("op") == "parallelize" else None
        if isinstance(new, list) and knobs and not any(
                isinstance(spec, dict) and "to" in spec for spec in new):
            # Each new step of a parallelize takes its own 'to' (`core_ops.STEP_FIELDS`).
            bundle[index] = {**edit, "new": [{**spec, "to": knobs[0]} if isinstance(spec, dict)
                                             else spec for spec in new]}
            continue
        found = _reader_edit(current, unread[0], edit, state)
        if found is None:
            return None
        how, made = found
        if how == "replace":
            bundle[index] = made
        else:
            if made.get("op") == "rewire" and edit.get("op") == "insert" and "id" not in edit:
                bundle[index] = {**edit, "id": unread[0]}  # the id the rewire names
            bundle.append(made)
    return None


def time_lines(facts: Mapping[str, Any] | None) -> list[str]:
    """"Time per item", in numbers, so time is reasoned about from them: how long the
    edited candidate's train items took each (median, 90th percentile, longest: from each
    item's own record). Nothing when it is not known. A model step has no time limit to
    state (a session ends on inactivity), and an item that timed out is said once, in
    "Item results" (`HARNESS_TIMEOUT`), not here."""
    facts = dict(facts or {})
    median = facts.get("per_item_median_s")
    if median is None:
        return []
    return ["## Time per item",
            f"The candidate you edit took {median} s per item at the median on its train "
            f"evaluation ({facts.get('per_item_p90_s')} s at the 90th percentile, "
            f"{facts.get('per_item_max_s')} s the longest; each item's own time).", ""]


def _cost_note(parsimony: float | None) -> list[str]:
    """With a cost term in the objective, what the cost column of the results says."""
    if not parsimony:
        return []
    return ["The cost per item column is each candidate's measured model cost (USD) over its "
            "train items, so a change's cost can be weighed against the size cost the score "
            "carries; - where it was not measured."]


def fmt_score(value: Any) -> str:
    """A score as every request prints it: 4 significant figures (``0.5714``); anything
    that is not a number as it is (``None`` as "n/a")."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a" if value is None else str(value)
    return f"{float(value):.4g}"


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
#: How ``selection`` ``checkpoint`` decides (``Loop._apply_round``), said in its place.
_CHECKPOINT_RULE = ("Each round's change is applied unless its train run fails (of several, "
                    "the one with the highest train score{train}); the validation score of "
                    "each applied pipeline{none} only picks the winner, the best of them, and "
                    "when the run stops.")


def _gate_rule(gated: bool | str, *, none: str, train: str) -> str:
    """The gate's rule for ``gated`` (a selection, or True for the strict gate)."""
    rule = _CHECKPOINT_RULE if gated == "checkpoint" else _GATE_RULE
    return rule.format(none=none, train=train)


def objective_text(parsimony: float | None, *, item_mean: bool = False,
                   gated: bool | str = False, cost_line: bool = True) -> str:
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
    the current one is rejected (``ValidationGate.judge_round``). ``gated`` is the run's
    selection (``ProposalContext.selection``; True is the strict gate): under
    ``checkpoint`` the rule is that every change is applied unless its train run fails."""
    if parsimony is None:
        return ""
    task = "the task's own score" + (" (the mean of the items' scores)" if item_mean else "")
    unless = "unless the task's own score measures them"
    if not parsimony:
        text = (f"The score is {task} and nothing else. This run adds no cost term to it "
                f"(parsimony is 0): {_UNSCORED} are not part of the score {unless}. A "
                "change is judged only by the task score it reaches.")
        return text + (" " + _gate_rule(gated, none="", train="") if gated else "")
    size = (f"a size cost of {parsimony:g} per node of the pipeline's graph (score = raw "
            f"score - {parsimony:g} x nodes)")
    # Said by the compute policy instead when one follows (``cost_line=False``).
    rest = (" Apart from that size cost, inference cost, latency and the number of model "
            f"calls are not part of the score {unless}." if cost_line else "")
    if gated:
        return (f"The train scores shown are {task} minus {size}. "
                + _gate_rule(gated,
                             none=(" (no size cost)" if gated == "checkpoint"
                                   else ", and that score carries no size cost"),
                             train=", size cost included" + ("" if gated == "checkpoint" else ","))
                + rest)
    return f"The score is {task} minus {size}.{rest}"


def objective_lines(parsimony: float | None, *, item_mean: bool = False,
                    gated: bool | str = False) -> list[str]:
    """"What the score measures" (`objective_with_policy`) as request lines; none when not
    known."""
    text = objective_with_policy(parsimony, item_mean=item_mean, gated=gated)
    return ["## What the score measures", *text.split("\n"), ""] if text else []


def objective_with_policy(parsimony: float | None, *, item_mean: bool = False,
                          gated: bool | str = False) -> str:
    """What the score measures (`objective_text`) and, on its own line, the run's compute
    policy (`compute_policy`): what every optimizer call that recommends or weighs a
    change is told (the proposer, the analysts and their merges, the memory pass), from
    this one place. Empty when the scoring settings are not known."""
    text = objective_text(parsimony, item_mean=item_mean, gated=gated, cost_line=False)
    policy = compute_policy(parsimony, item_mean=item_mean, gated=gated)
    return f"{text}\n{policy}" if text and policy else text


def compute_policy(parsimony: float | None, *, item_mean: bool = False,
                   gated: bool | str = False) -> str:
    """The run's compute policy in one sentence, from its scoring settings: with
    parsimony 0, cost is not a reason for or against a change (extra steps and model
    calls per item are not penalised, so a change that adds them is weighed like any
    other); with a size cost, that cost (in the train scores, under a validation gate,
    whose own score carries none) is the only one counted. Empty when the settings are
    not known."""
    if parsimony is None:
        return ""
    if parsimony:
        where = " in the train scores" if gated else ""
        return (f"Compute policy: apart from the size cost{where}, inference cost, latency "
                "and the number of model calls are no reason for or against a change in "
                "this run, unless the task's own score measures them.")
    # What the score text does not say already (it says what the score is).
    return ("Compute policy: cost is not a reason for or against a change in this run"
            + ("; per-item cost is reported separately." if item_mean else "."))


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
        'for a text parameter, the same with "patch" in place of "value" (its form is in the '
        '`set_knob` row above).',
    ]
    if any(op.get("op") == "set_knob" for op in catalogue):
        # SkillOpt's patch mode edits its skill only by local append/insert/replace/delete
        # patches; a whole new value stays available for what a patch cannot express.
        lines += ["- To change a text parameter such as a prompt, patch it, in the form "
                  "the `set_knob` row above gives.",
                  "  A whole new `value` is for a setting that is not text, a new step's own",
                  "  prompt, or a prompt that has to be replaced outright."]
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
        "  safer default.",
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
    shown = [{"edits": bundle, "rationale": "<why these changes only work together>",
              "prediction": "<expected effect on the score>", "confidence": 0.0}
             for bundle in bundle_examples(catalogue)]
    # Never more proposals in one example than the request asks for: past that, each
    # bundle is an example of its own, one answer each.
    per = proposals if proposals > 0 else len(shown)
    examples = [json.dumps({"proposals": shown[i:i + per]}) for i in range(0, len(shown), per)]
    example = "\n\nor:\n\n".join(examples)
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

Example{"s" if len(examples) > 1 else ""}:

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


#: How a request says, once, an item the harness timed out: the run's own limit on a
#: model call or an item ended it, whatever that limit is (`harness_timeout`).
HARNESS_TIMEOUT = "harness timeout: not caused by the pipeline's design; not a target"

#: The ``error_kind`` (and ``missing_steps`` cause) values the harness sets on its own
#: account: a model call the client ended (``timeout``: inactivity, the item deadline)
#: and a harness fault (``harness_incident``, `model.HarnessIncident`). A wall a step
#: declares (``timeout_s``; ``declared_timeout``) is the pipeline's design, and a session
#: still writing at the backstop (``backstop``) the model's own failure: both ordinary
#: failures the optimizer sees and may target.
HARNESS_TIMEOUT_KINDS = frozenset({"timeout", "harness_incident"})


def harness_timeout(record: Mapping[str, Any]) -> bool:
    """Whether the harness ended the item for time, from structured fields only: its
    ``error_kind`` is ``timeout`` (`items.error_kind`: a model call the client ended, or
    the item deadline), or a step of it has no answer because it timed out. Never from
    an error's text: an op that quotes a reply saying "timed out", a code step's own
    ``TimeoutExpired`` and a ``SnippetTimeout`` are the pipeline's. The item still scores
    what it scored (0); a request lists it once (`HARNESS_TIMEOUT`) and reads none of it
    as the pipeline's failure. A harness incident a rerun did not recover
    is not scored at all (``not_scored``) and is listed on its own line."""
    if record.get("passed"):
        return False
    if str(record.get("error_kind") or "") in HARNESS_TIMEOUT_KINDS:
        return True
    return any(isinstance(info, Mapping) and info.get("cause") in HARNESS_TIMEOUT_KINDS
               for info in (record.get("missing_steps") or {}).values())


def _pipeline_items(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records a request reads as the pipeline's doing: all but harness timeouts."""
    return [r for r in records if not harness_timeout(r)]


def pipeline_item_errors(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """A candidate's item errors (`item_errors`) as every request section says them: a
    harness timeout is none of them ("Item results" lists it once)."""
    return item_errors(_pipeline_items(records))


#: How the optimizer is told why a step gave no answer (`items.missing_steps` causes).
NO_ANSWER_CAUSES = {
    "output_limit": "ran out of output without answering",
    "turn_cap": "used its whole turn cap without finishing",
    "crash": "crashed in every session",
    "timeout": "timed out",
    "backstop": "was still writing at the session time limit (4 h by default)",
    "step_timeout": "ran over its own time limit",
    "failed": "failed (optional branch)",
    "declared_timeout": "ran past its own declared timeout_s",
    "invented_tool_calls": "wrote out tool calls it has no tools for",
}


#: Where "Item results" lists an error's cause, for an item shown in full to point to.
ERROR_CAUSE_REF = 'cause {n} under "Errors, by cause" above'


def _error_causes(records: Sequence[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """Each error text the items failed with (clipped), with the items that failed with it,
    most items first: the rows of "Errors, by cause"."""
    causes: dict[str, list[str]] = {}
    for record in records:
        if record.get("error_text"):
            causes.setdefault(_clip(record["error_text"]), []).append(str(record.get("id", "")))
    return sorted(causes.items(), key=lambda kv: -len(kv[1]))


def error_refs(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    """For each error text "Errors, by cause" shows whole, where it is (`ERROR_CAUSE_REF`):
    an item shown in full says its error there rather than again."""
    return {cause: ERROR_CAUSE_REF.format(n=number)
            for number, (cause, _ids) in enumerate(_error_causes(records)[:ITEM_EVIDENCE_ROWS], 1)
            if cause == str(next(r["error_text"] for r in records
                                 if r.get("error_text") and _clip(r["error_text"]) == cause))}


def _item_evidence(records: Sequence[dict[str, Any]]) -> list[str]:
    """How the candidate being edited did item by item, on its train part.

    A score is a mean over items, and a mean cannot say *why* it is low: an item
    answered wrong and an item whose completion failed (a model call that stopped
    at its output token limit, a timeout) cost the same. The loop already hands the
    proposer these records (``item_records``); this is where it reads them. Train
    records only, as ``OuterLoop.item_records`` enforces. A harness timeout is listed
    once (`HARNESS_TIMEOUT`) and counted nowhere else.
    """
    # An item a harness incident kept from an outcome, run again and still one, is left
    # out of the score (`loop.Loop._settle_incidents`): said once, on its own line.
    incidents = [r for r in records if r.get("not_scored")]
    # One a mode with no item_scores fact could not leave out (``incident_kept``): still
    # scored 0, said once on its own line, never as a timeout.
    kept = [r for r in records if r.get("incident_kept") and not r.get("not_scored")]
    timed_out = [r for r in records if harness_timeout(r) and not r.get("not_scored")
                 and not r.get("incident_kept")]
    scores = [float(r["score"]) for r in records if isinstance(r.get("score"), (int, float))]
    total = len(records) - len(incidents)
    records = _pipeline_items(records)
    errored = [r for r in records if r.get("error_text")]
    passed = [r for r in records if not r.get("error_text") and r.get("passed")]
    unanswered = [r for r in records if not r.get("error_text") and not r.get("passed")
                  and r.get("missing_steps")]
    wrong = [r for r in records if not r.get("error_text") and not r.get("passed")
             and not r.get("missing_steps")]
    mean = f", mean score {fmt_score(sum(scores) / len(scores))}" if scores else ""
    lines = [
        "## Item results of the candidate being edited",
        f"{len(records)} train item(s): {len(passed)} passed, {len(wrong)} answered "
        f"wrong, " + (f"{len(unanswered)} had no answer (a step gave none; cause below), "
                      if unanswered else "")
        + f"{len(errored)} failed with an error{'' if timed_out or kept else mean}. An error is the "
        "item's run failing (its cause is below), not a wrong answer.",
    ]
    if incidents:
        ids = list(dict.fromkeys(str(r.get("id", "")) for r in incidents))
        lines.append(f"{len(ids)} item(s) ({', '.join(ids)}) had a harness incident and are "
                     "not scored: not caused by the pipeline's design; not a target.")
    if kept:
        ids = list(dict.fromkeys(str(r.get("id", "")) for r in kept))
        lines.append(f"{len(ids)} more ({', '.join(ids)}) had a harness incident this mode "
                     "cannot leave out of its score (scored 0): not caused by the pipeline's "
                     "design; not a target"
                     + (f"; mean score over all {total}:{mean.removeprefix(', mean score')}."
                        if mean and not timed_out else "."))
    if timed_out:
        ids = list(dict.fromkeys(str(r.get("id", "")) for r in timed_out))
        lines.append(f"{len(ids)} more ({', '.join(ids)}), each a {HARNESS_TIMEOUT}"
                     + (f"; mean score over all {total}:{mean.removeprefix(', mean score')}."
                        if mean else "."))
    if unanswered:
        rows = [[r.get("id"), step, NO_ANSWER_CAUSES.get(str(info.get("cause")), info.get("cause")),
                 info.get("sessions")]
                for r in unanswered for step, info in sorted((r.get("missing_steps") or {}).items())
                if isinstance(info, dict)]
        lines += [
            "",
            "No answer -- not a wrong answer: a step gave none, so the item was graded "
            "without it. Running out of output or running out of turns is the model's own "
            "failure on the question and is not retried unless the step opts in; a crash is "
            "the worker's or the product's, retried in fresh sessions first:",
            _table(["item", "step", "cause", "sessions"], rows[:ITEM_EVIDENCE_ROWS]),
        ]
        if len(rows) > ITEM_EVIDENCE_ROWS:
            lines.append(f"({len(rows) - ITEM_EVIDENCE_ROWS} more not shown)")
    if errored:
        ranked = _error_causes(records)
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
                   [[r.get("id"), fmt_score(r.get("score")), _clip(r["fail_reason"])]
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


def _trajectory(record: dict[str, Any], text_chars: int = _WHOLE,
                omissions: list[dict[str, Any]] | None = None,
                settled: Collection[str] | None = None,
                errors_said: Mapping[str, str] | None = None) -> str:
    """One train item (`optim.evidence.render_item`, as every call that reads items shows
    one): its verdict, what was expected and what it gave, the grader's reason, its input
    once, then each step's reply, each fenced (heading lines escaped) so nothing in them reads
    as a section of the request; with ``settled`` (`evidence.settled_lines`, the steps'
    settings), each step's ``Sees:`` line and the part of its prompt its settings do not
    hold, as `ITEM_FORM` says.

    The steps are in the order the record holds them, which is the order the pass ran them
    (a fork merges its branches in branch order), never sorted by name. Texts are whole
    unless ``text_chars`` clips them (head and tail, marked; each cut in ``omissions``).
    """
    from . import evidence
    from .analysts import metadata_text

    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return evidence.render_item(
        record, input_keys=TRAJECTORY_INPUT_KEYS, outcome=_outcome(record),
        expected=_expected(record), expected_label=shown_label(record),
        metadata=metadata_text(record), choices=_choice_lines(meta.get("choices")),
        prompts=False,
        prompt_chars=text_chars, reply_chars=text_chars, omissions=omissions, settled=settled,
        errors_said=errors_said)


def _outcome(record: dict[str, Any]) -> str:
    return ("passed" if record.get("passed") else
            "failed with an error" if record.get("error_text") else "answered wrong")


def shown_label(record: dict[str, Any]) -> str:
    """The expected answer's label when one of the item's choices is that label with the
    expected text (or the record has no text for it), so the choice line shown with the
    item says the rest; else empty, and the expected answer is said whole."""
    label, label_text = record.get("label"), record.get("label_text")
    if label in (None, ""):
        return ""
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    choices = meta.get("choices") if isinstance(meta.get("choices"), list) else []
    for index, choice in enumerate(choices):
        own, text = ((choice.get("label", chr(ord("A") + index)), choice.get("text", ""))
                     if isinstance(choice, dict) else (chr(ord("A") + index), choice))
        if str(own) == str(label) and (label_text in (None, "") or str(text) == str(label_text)):
            return str(label)
    return ""


def _expected(record: dict[str, Any], *, label_only: bool = False) -> str:
    """The expected answer: its label and its text; with ``label_only``, only the label
    when a choice shown with the item says the text (`shown_label`)."""
    label, label_text = record.get("label"), record.get("label_text")
    if label_only and shown_label(record):
        return str(label)
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
    then passed, each group by id. A harness timeout is not among them: "Item results"
    lists it once, and no item read as the pipeline's doing is one (`harness_timeout`)."""
    first: dict[str, dict[str, Any]] = {}
    for record in _pipeline_items(records):
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


#: What an item shown here holds -- said once beside the items, and what `_trajectory`
#: renders with the steps' settings as ``settled`` (a test holds the two together).
ITEM_FORM = ("Each item: its verdict, what was expected and what it gave, the grader's "
             "reason, its input once, the steps the harness completed as code, each named "
             "once with how often (`Completed by the harness:`; their order among the model "
             "steps is not recorded), then each model step in the order it ran: at its first "
             "run in the request, how it runs (`Ran as:` a model's reply, the tools its "
             "graph gave it and its real tool calls where the harness records them), what "
             "its prompt "
             "took from the input or an earlier step's reply, by name (`Sees:`), any part of "
             "its prompt that is not in its settings (`" + "Also in its prompt (not in its "
             "settings)" + "`: a reason a judge sent it back with, say), and its reply. A "
             "step's own instructions are its settings, under \"Each step's settings\". A "
             "step run again says in place what its prompt has exactly as the run of it "
             "that fed it had, counted as that run's block and reply show them (`[same as "
             "solve's prompt (step 3), all but its last shown line]`, `[same as solve's reply "
             "(step 3), lines 1-6]`). " + CLAIMS)
#: Whether every train item is shown in full or the rule (`_shown_items`) picks them:
#: every item when all of them together are under this many characters. It only decides
#: that switch; it never cuts an item.
SHOW_ALL_BELOW = 120000
#: A single text of an item shown in full (a step's reply, the input) is cut only past
#: this many characters -- one text too large to send whole; an item is otherwise cut
#: only when it alone passes what the request's room leaves it (`SHOWN_TEXT_FLOOR`).
ITEM_TEXT_MAX = 60000
#: The least a text of an item is cut to (head and tail, marked) when the item must give
#: way to fit the request's room (`_fit_shown`, the forced picks of "What the last edit
#: changed"): an item that does not fit even then is named instead of shown.
SHOWN_TEXT_FLOOR = 200


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
    """The items shown whole when not all fit, by rule, never by size: one failure per
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


def _shown_size(record: dict[str, Any], settled: Collection[str] | None,
                refs: Mapping[str, int] | None,
                how: set[tuple[str, str]] | None = None) -> int:
    """The characters ``record`` takes under "Some items in full", as `_trajectories` shows
    it (its steps' prompt lines with ``settled``, its error as a cause number); with
    ``how`` (added to), without a step's ``Ran as:`` line an item measured before it said
    (`evidence.said_once`): the items kept are then measured as shown together, whatever
    order shows them, since each distinct line is said once either way."""
    text = _trajectory(record, ITEM_TEXT_MAX, None, settled, refs)
    return len(text if how is None else evidence.said_once(text, how)) + 2


def _fit_shown(failing: Sequence[dict[str, Any]], passing: Sequence[dict[str, Any]],
               room: int, size: Callable[[dict[str, Any]], int],
               cut: Callable[[dict[str, Any], int], int | None] | None = None,
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
                          dict[int, int]]:
    """The items `_shown_items` picked that fit ``room`` characters together: failures and
    their contrasting successes in turn (failure 1, success 1, failure 2, ...), each in its
    rank order, stopping at the first that does not fit, each measured as shown (``size``,
    `_shown_size`), each whole. Only the first, when it alone passes ``room``, gives way
    rather than the section showing none: shown with its longest texts cut to the
    longest one length at which it fits (``cut``, `_cut_to_fit`), or, when it does not
    fit even at `SHOWN_TEXT_FLOOR`, named with the rest. ``room`` 0 is no limit
    (`ProposalContext.shown_items_room`). The third value is what was left out, the
    fourth each cut item's length (by ``id``)."""
    if room <= 0:
        return list(failing), list(passing), [], {}
    turns = [r for pair in itertools.zip_longest(failing, passing) for r in pair
             if r is not None]
    kept: list[dict[str, Any]] = []
    cuts: dict[int, int] = {}
    used = 0
    for record in turns:
        chars = size(record)
        if used + chars > room:
            limit = None if kept or cut is None else cut(record, room)
            if limit is None:
                break
            cuts[id(record)] = limit
            chars = room
        kept.append(record)
        used += chars
    keep = {id(r) for r in kept}
    return ([r for r in failing if id(r) in keep], [r for r in passing if id(r) in keep],
            [r for r in turns if id(r) not in keep], cuts)


def _longest_fitting(fits: Callable[[int], bool], low: int, high: int = ITEM_TEXT_MAX) -> int:
    """The longest length in [``low``, ``high``] that ``fits`` (which holds at ``low``,
    and the shorter a cut the more it holds), to within 2%: the ratio between a length
    that fits and one that does not is halved, so about ten tries cover 200 to 60000."""
    if fits(high):
        return high
    while high - low > 1 and high > low * 1.02:
        middle = min(max(int((low * high) ** 0.5), low + 1), high - 1)
        low, high = (middle, high) if fits(middle) else (low, middle)
    return low


def _cut_to_fit(record: dict[str, Any], room: int, settled: Collection[str] | None,
                refs: Mapping[str, str] | None) -> int | None:
    """The longest length every text of ``record`` may keep (head and tail, marked) for it
    to fit ``room`` as "Some items in full" shows it (`_longest_fitting`): only its texts
    longer than that are cut, so its longest go first. ``None`` when it does not fit even
    at `SHOWN_TEXT_FLOOR`."""

    def fits(limit: int) -> bool:
        return len(_trajectory(record, limit, None, settled, refs)) + 2 <= room

    return _longest_fitting(fits, SHOWN_TEXT_FLOOR) if fits(SHOWN_TEXT_FLOOR) else None


TRAJECTORIES_HEADING = "## Trajectories of the candidate being edited"


def _all_whole(items: Sequence[dict[str, Any]], budget: int) -> bool:
    """Whether "Trajectories" shows every item whole (`_trajectories`)."""
    return sum(len(_trajectory(r)) + 2 for r in items) <= min(budget, SHOW_ALL_BELOW)


def _analysed_findings(analysed: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The analysts' ranked findings in the readers' form, for `_shown_items`."""
    return [{"failure_patterns": [f for f in analysed if f.get("source") != "success"],
             "success_patterns": [f for f in analysed if f.get("source") == "success"]}]


def trajectories_whole(records: Sequence[dict[str, Any]], *, budget: int = TRAJECTORY_BUDGET,
                       findings: Sequence[dict[str, Any]] | None = None,
                       analysed: Sequence[dict[str, Any]] | None = None,
                       shown_room: int = 0,
                       settled: Collection[str] | None = None) -> list[dict[str, Any]]:
    """The records "Trajectories of the candidate being edited" shows whole, by the same
    rule `_trajectories` applies: every item when they all fit, else the items
    `_shown_items` picks that fit ``shown_room`` (`_fit_shown`; ``settled`` as
    `_trajectories` has it, so each is measured as shown). Another section names their
    replies by item instead of repeating them."""
    items = _ordered_items(records)
    if not items or _all_whole(items, budget):
        return items
    failing, passing, _cited = _shown_items(
        items, _analysed_findings(analysed) if analysed is not None else findings)
    refs = error_refs(_pipeline_items(records))
    measured: set[tuple[str, str]] = set()
    failing, passing, _left, cuts = _fit_shown(
        failing, passing, shown_room, lambda r: _shown_size(r, settled, refs, measured),
        lambda r, room: _cut_to_fit(r, room, settled, refs))
    # An item cut to fit is not whole: no other section points to it as shown.
    return [r for r in (*failing, *passing) if id(r) not in cuts]


def _trajectories(records: Sequence[dict[str, Any]], *, budget: int = TRAJECTORY_BUDGET,
                  findings: Sequence[dict[str, Any]] | None = None,
                  failures: Sequence[dict[str, Any]] = (),
                  analysed: Sequence[dict[str, Any]] | None = None,
                  omissions: list[dict[str, Any]] | None = None,
                  settled: Collection[str] | None = None,
                  shown_room: int = 0, sizes: dict[str, int] | None = None) -> list[str]:
    """Every train item of the candidate being edited, each as `ITEM_FORM` says; an item's
    error is said where "Item results" lists its cause (`error_refs`). ``sizes`` gets the
    characters the items picked and shown in full take (``items``).

    When all of them together are under ``SHOW_ALL_BELOW`` characters (or the call's
    smaller ``budget``), every item is shown in full here. When not, readers went through
    them in groups of ``budget`` (``AutoresearchOptimizer``): their findings come first,
    then a table of every item, then the items the rule picks (`_shown_items`), each
    whole -- with ``shown_room`` (0: no limit), those of them that fit it together, in rank
    order (`_fit_shown`), the rest named. Train records only.

    ``analysed``: the analysts' ranked findings (`optim.analysts`), shown before this
    section, which take the readers' place: no reader findings are shown, and the
    items they cite are shown in full first (one per finding in turn, failure findings
    first, each kind in rank order).
    """
    items = _ordered_items(records)
    if not items:
        return []
    header = [TRAJECTORIES_HEADING,
              f"The {len(items)} train item(s) counted under \"Item results\"."]
    groups = trajectory_groups(records, budget)
    # Numbered over the records "Errors, by cause" numbers its rows over (`_item_evidence`),
    # so an item's "cause N" is that table's row N.
    refs = error_refs(_pipeline_items(records))
    # How each step runs is said at its first run in the request (`evidence.said_once`).
    how: set[tuple[str, str]] = set()
    if _all_whole(items, budget):
        return header + [f"Every item. {ITEM_FORM}",
                         *(line for r in items
                           for line in ("", evidence.said_once(_trajectory(
                               r, ITEM_TEXT_MAX, omissions, settled, refs), how)))]
    if analysed is not None:
        # The analysts' findings are their own section, before this one; the items they
        # cite are shown in full first (``_cited_items``).
        lines = header + [
            "Not every item is shown in full here. The analysts' ranked findings above "
            "cover them all; here is a table of every item, then the items the rule below "
            "picks, each whole."]
        findings = _analysed_findings(analysed)
    else:
        lines = header + [
            f"Not every item is shown in full here, so readers went through them in "
            f"{len(groups)} groups of whole items. Their findings come first, then a table "
            "of every item, then the items the rule below picks, each whole."]
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
    failing, passing, cited = _shown_items(items, findings)
    measured: set[tuple[str, str]] = set()
    failing, passing, left_out, cuts = _fit_shown(
        failing, passing, shown_room, lambda r: _shown_size(r, settled, refs, measured),
        lambda r, room: _cut_to_fit(r, room, settled, refs))
    chosen = [*failing, *passing]
    # An item shown in full below says its expected choice's text there (its choice
    # line): the table gives only the label for it.
    full = {id(r) for r in chosen}
    # An item's history in the run (`evidence.item_history`: a failing item's, or a flaky
    # pass's); an item shown in full below says it in its heading. Like every empty column,
    # left out when no item has one.
    lines += ["", "### Every item",
              _table_filled(["item", "outcome", "expected", "predicted", "grader", "error",
                             "history"],
                     [[r.get("id"), _outcome(r),
                       _clip(_expected(r, label_only=id(r) in full), _TABLE_CELL_CHARS),
                       _clip(r.get("prediction"), _TABLE_CELL_CHARS),
                       _clip(r.get("fail_reason") if not r.get("passed") else "", _TABLE_CELL_CHARS),
                       "(below)" if id(r) in full and r.get("error_text") else
                       _clip(r.get("error_text"), _TABLE_CELL_CHARS),
                       _history_cell(r, id(r) in full)] for r in items])]
    if not chosen and not left_out:
        return lines
    examples = ["", "### Some items in full",
                f"{len(failing)} failing item(s)"
                + (f" the {'analysts' if analysed is not None else 'readers'} cited, "
                   "one per finding in rank order" if cited else ", one per kind of failure")
                + (f", then {len(passing)} passing item(s) for contrast" if passing else "")
                + f" (of {len(items)}; the table above has every one). {ITEM_FORM}"]
    if left_out:
        examples.append(
            f"The rule picked {len(left_out)} more, left out so this request fits the "
            f"optimizer model's context ({shown_room} characters were left for items in "
            "full"
            + ("; even the first, with every text cut to "
               f"{SHOWN_TEXT_FLOOR} characters, would not fit" if not chosen else "")
            + "): "
            + ", ".join(f"{r.get('id')} ({'passed' if r.get('passed') else 'failed'})"
                        for r in left_out) + ". Each is in the table above.")
    for record in chosen:
        if id(record) in cuts:
            examples += ["", f"Item {record.get('id')} alone is longer than the {shown_room} "
                         "characters left for items in full, so its longest texts are cut to "
                         f"{cuts[id(record)]} characters each, head and tail around a marked "
                         "count; everything else in it is whole."]
        if record.get("passed"):
            examples += ["", "(A success, for contrast.)"]
        examples += ["", evidence.said_once(_trajectory(
            record, cuts.get(id(record), ITEM_TEXT_MAX), omissions, settled, refs), how)]
        if sizes is not None:
            sizes["items"] = sizes.get("items", 0) + len(examples[-1]) + 2
    return lines + examples


def _history_cell(record: Mapping[str, Any], below: bool) -> str:
    """An item's history for the "Every item" table, without the heading's label; "(below)"
    when the item is shown in full below; empty when it has none (a pass that never
    failed)."""
    said = str(record.get("item_history") or "")
    if not said:
        return ""
    return "(below)" if below else said.removeprefix("history: ")


#: The fields of a core operator's older written-out form (`core_ops`: 'target' and a
#: 'gate' or 'into' node objects). An edit may still carry them, but its line does not
#: offer them: each core operator is described by its one-line form only.
WRITTEN_OUT_FIELDS = ("target", "gate")


def _core_fields(fields: Sequence[str]) -> str:
    """A core operator's fields as its line offers them: its one-line form, never a node
    object (``into`` is one or two short steps, as its example shows)."""
    return ", ".join("into: one or two steps, each like {kind, text, sees}" if name == "into"
                     else name for name in fields if name not in WRITTEN_OUT_FIELDS)


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
               knob_values: Mapping[str, Any] | None = None, *, refer: bool = False) -> str:
    """One node in full, as a node object to copy; long texts clipped, and said.

    A field bound to a knob shows the knob's current value (the stored field is not
    what runs), and says to edit the knob; with ``refer``, a pointer to the knob
    (`nn.edits.KNOB_POINTER`), whose text the request shows once elsewhere. The bindings
    themselves, and the bound knobs in ``knobs``, are left out: a copy must not bind the
    original's knob.
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
        if key in bound and refer:
            # Its text is shown once, under "Each step's settings"; a node copied with
            # the pointer gets that text (`nn.edits.KNOB_POINTER`).
            value = nn_edits.KNOB_POINTER.format(knob=bound[key])
        elif key in bound:
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
    and which are the task's own input (``task_inputs``, `nn.operators.task_inputs`;
    ``knob_values`` fills the slots of bound prompts); empty when the graph does not parse."""
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.operators import made_by, task_inputs
    from refactory_dsh.nn.validate import available_inputs

    try:
        graph = Graph.from_state(dict(graph_state))
    except Exception:  # a broken graph is the validator's to report
        return {}
    have = available_inputs(graph)
    makers = made_by(graph)
    try:
        free = task_inputs(graph, knob_values)
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
    written, each as the pipeline view writes it (``name@maker``, ``@run`` when no step
    makes it), nodes with the same row listed together, and how a file's text reaches a
    prompt. Fits ``limit``. Nothing when not known."""
    from refactory_dsh.nn.operators import COLLECT_CALL, COLLECT_TEXT_CHARS, FILE_TEXT_CALL
    from refactory_dsh.nn.pipeline_view import RUN_MAKER

    if not available:
        return []
    how = ("a code op (`\"call\": \"refactory_dsh.snippet:run\"`) that reads the file and "
           "`produces` the fact" if code_nodes else "an op that reads the file and `produces` "
           "the fact")
    lead = (
        "### What each node has before it runs\n"
        "On every path that reaches it, each written as "
        "the pipeline view writes it (`name@maker`; `@run`: no step makes it); nodes with "
        "the same row share it. A step's input is not its output: name the fact or file the "
        "step makes. A prompt's "
        "`{{name}}` may name only a fact listed for its node (or a knob): a file's name is "
        "not a fact, and `{{notes}}` does not read `notes.txt`. An agent without tools sees "
        "its prompt and the files in its `sees` (its declared reads, pasted in after the prompt). To hand "
        "files to a prompt as one `{{name}}`, insert a collect op before the node "
        f"(`\"call\": \"{COLLECT_CALL}\"`, the files in `reads`, the fact in `produces`: "
        "each file under its own header, a missing one said to be missing; one file over "
        f"{COLLECT_TEXT_CHARS} characters is shown as its path and size), or a file_text op "
        f"for one file's whole text (`\"call\": \"{FILE_TEXT_CALL}\"`), or "
        f"{how}, or have the agent that writes the file declare `produces`, which makes its "
        "reply that fact too.")

    def named(entry: Mapping[str, Any], name: str) -> str:
        makers = (entry.get("made_by") or {}).get(name) or ()
        return f"{name}@{','.join(makers) or RUN_MAKER}"

    def cell(entry: Mapping[str, Any], key: str, keep: int | None) -> str:
        names = [named(entry, name) for name in entry.get(key) or ()]
        if not names:
            return "-"
        if keep is None or len(names) <= keep:
            return ", ".join(names)
        return ", ".join(names[:keep]) + f" (+{len(names) - keep} more)"

    def table(rows: Sequence[tuple[str, Mapping[str, Any]]], keep: int | None) -> str:
        # Nodes with the same row share it; a row that holds all of another row's names
        # is said as that row plus the rest, so no list of names is written twice.
        groups: dict[tuple[frozenset, frozenset], list[str]] = {}
        entries: dict[tuple[frozenset, frozenset], Mapping[str, Any]] = {}
        for node_id, entry in rows:
            key = (frozenset(entry.get("facts") or ()), frozenset(entry.get("files") or ()))
            groups.setdefault(key, []).append(node_id)
            entries.setdefault(key, entry)
        out = []
        for key, nodes in groups.items():
            within = [other for other in groups if other != key
                      and other[0] <= key[0] and other[1] <= key[1] and (other[0] or other[1])]
            base = max(within, key=lambda o: (len(o[0]) + len(o[1]), groups[o]), default=None)
            entry = entries[key]
            if base is None:
                facts, files = cell(entry, "facts", keep), cell(entry, "files", keep)
            else:
                said = f"as at `{groups[base][0]}`"
                more = {k: [n for n in entry.get(k) or () if n not in base[i]]
                        for i, k in enumerate(("facts", "files"))}
                facts = said + (", plus " + cell({**entry, "facts": more["facts"]}, "facts",
                                                 keep) if more["facts"] else "")
                files = said + (", plus " + cell({**entry, "files": more["files"]}, "files",
                                                 keep) if more["files"] else "")
            out.append([", ".join(nodes), facts, files])
        return _table(["node(s)", "facts available", "files written before it"], out)

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


def node_object_ops(catalogue: Sequence[Mapping[str, Any]]) -> list[str]:
    """The operations of ``catalogue`` that take a whole node object as their `node` field,
    in its order: generated, so the request names exactly the ones this run offers."""
    return [str(c.get("op")) for c in catalogue if "node" in (c.get("fields") or ())]


def _node_specs(states: Sequence[dict[str, Any]], *, text_chars: int = NODE_SPEC_TEXT_CHARS,
                section_chars: int = NODE_SPEC_SECTION_CHARS,
                knob_values: Mapping[str, Any] | None = None,
                person_gates: bool = True, refer: bool = False,
                catalogue: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    """Each node of the edited graph in full, then the generated node schemas.

    The operations said to take a copied node are those of ``catalogue`` (else of every
    edit) with a `node` field (`node_object_ops`).

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
    takers = node_object_ops(nn_edits.catalogue("broad") if catalogue is None else catalogue)
    ordered = sorted(states, key=lambda s: str(s.get("id")))

    def section(chosen: Sequence[dict[str, Any]], limit: int) -> list[str]:
        lines = ["### Each node in full",
                 "Copy a node object, give it a new id, change what you mean to change "
                 "(prompt, reads, writes, produces, knobs)"
                 + (f", and use it as the `node` field of {_or_list(takers)}" if takers else "")
                 + ". Knob bindings are left out of these copies: a new agent node gets its own "
                 "`<id>__system_prompt` knob holding its `system_prompt_template` and its own "
                 "`<id>__task` knob holding its `prompt_template`, and a "
                 "new node may not bind another node's knob. A field bound to a knob shows "
                 "where its text is (`[the value of knob ...]`); a copy keeping that pointer, "
                 "or an operation's 'text' that is that pointer, gets the knob's current "
                 "text. A clipped text (marked "
                 "`[... N characters omitted ...]`) must not be copied: write your own, or "
                 "edit the knob; an edit carrying the marker is refused."]
        lines += [_node_spec(state, limit, knob_values, refer=refer) for state in chosen]
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


def _current_line(history: Sequence[dict[str, Any]], selection: str = "") -> str:
    """What "What this run has already tried" says of the validation gate's choice: the
    train score of the candidate marked ``current`` (a train score, labelled so; the
    val score that made it current is never shown). Under ``checkpoint`` the current
    candidate is the last one applied, and no gate is named."""
    whose = ("the current candidate" if selection == "checkpoint"
             else "the validation gate's current candidate")
    current = next((c for c in history if c.get("gate") == "current"), None)
    if current is None:
        return ("none judged by the validation gate yet"
                if all(c.get("gate") in (None, "not gated") for c in history)
                else f"{whose} is older than this list")
    score = current.get("score")
    return (f"train score of {whose} "
            + (fmt_score(score) if isinstance(score, (int, float)) else "n/a (its train run failed)"))


def _gate_note(*tables: Sequence[dict[str, Any]], val_items: int = 0,
               selection: str = "", population: bool = False) -> list[str]:
    """What the gate column means, said once, when there is one.

    A rejection says only that the candidate was not shown better on validation; with
    ``val_items`` (the validation items the gate's score is the mean of) the note says
    what one item is worth, so a one-item loss is read as the gate's resolution. Under
    ``selection`` ``pool`` it says what the pool is instead of SkillOpt's single line;
    under ``checkpoint``, what applied means instead of a rejection."""
    if not any(_gate_column(rows) for rows in tables):
        return []
    if selection == "checkpoint":
        # Nothing is rejected on validation: what the column's three words mean, once.
        return ["The gate column says what this run did with each change (the rule is under "
                "\"What the score measures\"): `current` is the last one applied, which you "
                "edit; `kept` was applied and later replaced; `not applied` was not. Scores "
                "here are train scores; held-out scores are never shown."]
    resolution = (f"The gate's score is the mean over {val_items} validation items, so one "
                  f"item is 1/{val_items} = {1 / val_items:.3g} of it, and a rejection can come "
                  "down to a single item. " if val_items > 0 else "")
    if selection == "pool":
        # The same labels as ``single`` (the pool's own membership follows the val
        # ranking, so it is never shown): what changes is that a parent may be any
        # candidate that scored well there (with a population picker, any judged
        # candidate), a rejected one included.
        return ["This run has a validation gate (its rule is under \"What the score "
                "measures\") with a pool of parents; the best so far is marked `current`. "
                + ("A parent may be any candidate judged so far, a `rejected` one included, "
                   "as the parent picker chooses. " if population else
                   "A parent may be any of the candidates that scored well there, a "
                   "`rejected` one included. ")
                + "Rows are not in order of the held-out scores. Scores here are train "
                "scores; held-out scores are never shown. You edit the pipeline shown below. A `rejected` "
                "candidate was not shown better than the current one on validation; that is "
                "all a rejection says. " + resolution + "A change that was rejected must not "
                "be proposed again unchanged. " + REJECTED_BUILD_TEXT]
    return ["This run has a validation gate (its rule is under \"What the score "
            "measures\"). Scores here are train scores; "
            "held-out scores are never shown. You edit the candidate marked `current`. A "
            "`rejected` candidate was not shown better on validation; that is all a "
            "rejection says. " + resolution + "A rejected candidate is never kept, and its "
            "change must not be proposed again unchanged. " + REJECTED_BUILD_TEXT]


#: The most score changes in each direction listed for graded items.
EDIT_CHANGE_GRADED = 3
#: Entries in "Your earlier reasoning", and how much of each text it keeps.
REASONING_ENTRIES = 6
REASONING_TEXT_CHARS = 600
#: The full edits of the latest rejected candidate, whatever its net count, shown under
#: "Your earlier reasoning" with what went wrong in it so it can be built on: every text
#: whole (a new step's instructions, a judge's criteria), the whole within this many
#: characters, head and tail with the cut said past it.
REJECTED_EDITS_CHARS = 20000
#: What the gate note says of building on a rejected change: by its mechanism, not its
#: net count.
REJECTED_BUILD_TEXT = (
    "The latest rejected change's full edits and the items it moved are under \"Your "
    "earlier reasoning\".")


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
    both = sorted(set(parent) & set(child))
    # An item the harness timed out on either side moved for no reason of the edit's.
    timed_out = [i for i in both if harness_timeout(parent[i]) or harness_timeout(child[i])]
    shared = [i for i in both if i not in timed_out]
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
            "deltas": {item: round(delta, 6) for delta, item in moved}, "timed_out": timed_out}


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


#: A text of this many characters that a request shows whole in one place is said there
#: only: anywhere else it is named by its head and where it is whole.
SAID_ONCE_CHARS = 100
#: How much of such a text a pointer to it shows, so the reader can tell which it is.
POINTER_HEAD_CHARS = 60
#: Where a knob's current text is whole in a proposal request: its home, since a patch
#: quotes it exactly from there.
SETTINGS_HOME = "\"Each step's settings\""


def _folded(text: Any) -> str:
    return " ".join(str(text).split())


def knob_homes(knobs: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Each text knob ``SAID_ONCE_CHARS``+ long, by name, its value with whitespace folded:
    what "Each step's settings" shows whole (`refer_to_knobs`)."""
    return {str(k.get("name")): _folded(k["value"]) for k in knobs or ()
            if isinstance(k, Mapping) and isinstance(k.get("value"), str)
            and len(k["value"]) >= SAID_ONCE_CHARS}


def text_pointer(text: str, where: str) -> str:
    """A long text named by its head and its length, and where it is whole."""
    folded = _folded(text)
    return f"\"{folded[:POINTER_HEAD_CHARS]}…\" ({len(text)} chars, whole {where})"


def _knob_home(text: str, homes: Mapping[str, str]) -> str | None:
    """The knob whose current text holds ``text`` (``SAID_ONCE_CHARS``+ long), or ``None``."""
    folded = _folded(text)
    if len(folded) < SAID_ONCE_CHARS:
        return None
    return next((name for name, value in homes.items() if folded in value), None)


def refer_to_knobs(value: Any, homes: Mapping[str, str], away: str | None = None) -> Any:
    """``value`` (an edit, a list of edits) with each free text a current knob holds
    (``homes``, `knob_homes`) named by its head and that knob instead of repeated: the
    settings show it whole. With ``away`` (a section showing these edits whole), every
    other long text is named by its head and that section."""
    if not homes and away is None:
        return value

    def point(text: str) -> str:
        name = _knob_home(text, homes)
        if name is not None:
            return text_pointer(text, f"in `{name}` under {SETTINGS_HOME}")
        if away is not None and len(_folded(text)) >= SAID_ONCE_CHARS:
            return text_pointer(text, f"under \"{away}\"")
        return text

    return _map_text(value, point)


def describe_edits(edits: Any, field_chars: int = EDIT_FIELD_CHARS,
                   text_budget: int | None = None,
                   homes: Mapping[str, str] | None = None) -> str:
    """A candidate's edits as a reader needs them: one line per edit, its op and what it
    names first, whole, then each other field with its wiring whole (every node's id,
    kind, reads, writes, ...; ``EDIT_WIRING_FIELDS``) and only its free text shortened,
    so no edit of a compound proposal, and no step of a split, is cut away by a long
    prompt before it. The free text of all the edits shares ``text_budget`` characters
    (by default ``field_chars`` per field). A text a current knob holds (``homes``,
    `knob_homes`) is named by its head and that knob (`refer_to_knobs`)."""
    steps = list(edits) if isinstance(edits, (list, tuple)) else [edits]
    steps = refer_to_knobs(steps, homes or {})
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


def whole_edits(edits: Any, before: Mapping[str, Any] | None = None,
                homes: Mapping[str, str] | None = None) -> str:
    """`whole_edit_parts`, one edit after another."""
    return "\n".join(whole_edit_parts(edits, before, homes)) or "(none)"


def whole_edit_parts(edits: Any, before: Mapping[str, Any] | None = None,
                     homes: Mapping[str, str] | None = None) -> list[str]:
    """A candidate's own edits as their home section shows them, one text per edit (so a
    bound can cut each and drop none): every text whole, never clipped, with its wiring.
    A text ``set_knob`` whose earlier value is known (``before``: the parent's values,
    then each earlier edit's) is shown as its change -- the lines it added and removed,
    whole -- not as the new value's unchanged start; a text a current knob holds
    (``homes``) is named by that knob, whose text "Each step's settings" shows whole."""
    import difflib

    homes = homes or {}
    values: dict[str, Any] = dict(before or {})
    parts: list[str] = []
    steps = list(edits) if isinstance(edits, (list, tuple)) else [edits]
    for index, edit in enumerate(steps, 1):
        lines: list[str] = []
        parts.append("")
        if not isinstance(edit, dict):
            parts[-1] = f"{index}. {json.dumps(edit, default=str, ensure_ascii=False)}"
            continue
        head = " ".join(f"{key}={json.dumps(edit[key], default=str)}"
                        for key in EDIT_NAMING_FIELDS if key in edit)
        name, new = edit.get("name"), edit.get("value", _UNKNOWN)
        old = values.get(name, _UNKNOWN) if isinstance(name, str) else _UNKNOWN
        if edit.get("op") == "set_knob" and isinstance(new, str) and isinstance(old, str):
            diff = [line[0] + refer_to_knobs(line[1:], homes) if line[:1] in ("+", "-")
                    else line for line in list(difflib.unified_diff(
                        old.splitlines(), new.splitlines(), lineterm="", n=0))[2:]]
            fence = _fence("\n".join(diff))
            lines += [f"{index}. {head} -- {len(old)} -> {len(new)} chars; the lines it "
                      "removed (-) and added (+):", fence + "diff", *(diff or ["(no change)"]),
                      fence]
        else:
            rest = refer_to_knobs({key: value for key, value in edit.items()
                                   if key not in EDIT_NAMING_FIELDS}, homes)
            if isinstance(rest.get("patch"), list):
                # A patch's 'target' quotes the text it replaced, not text it wrote.
                rest["patch"] = [{**op, "target": _cut(op["target"], HISTORY_NODE_PROMPT_CHARS)}
                                 if isinstance(op, dict) and isinstance(op.get("target"), str)
                                 else op for op in rest["patch"]]
            tail = "; ".join(f"{key}: {json.dumps(value, default=str, ensure_ascii=False)}"
                             for key, value in rest.items())
            lines.append(f"{index}. {head}" + (f" -- {tail}" if tail else ""))
        parts[-1] = "\n".join(lines)
        if edit.get("op") == "set_knob" and isinstance(name, str):
            values[name] = new
    return parts


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


def _attempt_lines(history: Sequence[dict[str, Any]],
                   seed_values: Mapping[str, Any] | None, below: Sequence[str] = (),
                   elsewhere: Mapping[str, str] | None = None,
                   homes: Mapping[str, str] | None = None) -> list[str]:
    """Every attempt so far on one line, oldest first: what it added to its parent, its
    first As-built line, its train score beside its parent's, the gate's decision. None is
    dropped (so a failed attempt is not repeated); none repeats its ancestors' edits. One
    in ``below`` ("What the last edit changed" shows what it built) points there. One in
    ``elsewhere`` (id -> the section showing its edits whole) keeps its wiring whole and
    names each long text by its head and that section, and a text a current knob holds
    (``homes``) is named by that knob, so no text is said twice."""
    elsewhere = elsewhere or {}
    scores = {str(c.get("id")): c.get("score") for c in history}

    def score(value: Any) -> str:
        return fmt_score(value)

    lines = []
    for c in history:
        own = c.get("own")
        built = list(c.get("built") or ())
        parent = c.get("parent")
        lines.append(
            f"- `{c.get('id')}` from `{parent or '(start)'}`"
            + (f" (epoch {c['epoch']})" if c.get("epoch") is not None else "") + ": "
            + compact_edits(c.get("edits"), seed_values,
                            own=own if isinstance(own, int) else None, homes=homes,
                            away=elsewhere.get(str(c.get("id"))))
            + ((f" | built: under \"{elsewhere.get(str(c.get('id')), 'What the last edit changed')}\""
                if str(c.get("id")) in below or str(c.get("id")) in elsewhere else
                f" | built {_clip(built[0], AS_BUILT_LINE_CHARS)}"
                + (f" (+{len(built) - 1} more)" if len(built) > 1 else "")) if built else "")
            + f" | train {score(c.get('score'))}"
            + (f" (parent {score(scores.get(str(parent)))})" if parent in scores else "")
            + ("" if c.get("exit_ok", True) else " | its run failed")
            + (f" | gate {c['gate']}" if c.get("gate") else ""))
    return lines


#: How much of an attempt's first As-built line its history line shows.
AS_BUILT_LINE_CHARS = 240


def compact_edits(edits: Any, before: Mapping[str, Any] | None = None,
                  own: int | None = None, homes: Mapping[str, str] | None = None,
                  away: str | None = None) -> str:
    """A candidate's edits (its whole lineage, oldest first) for the history table,
    without whole values: each edit's op and what it names; a text ``set_knob`` as
    old -> new length and a bounded diff against the value it replaced (the previous
    ``set_knob`` of that knob in the lineage, else ``before``, the seed's values); a
    patch's ops; a structural edit's fields with every node's wiring whole and its free
    text (prompts, code) sharing ``HISTORY_EDIT_CHARS`` characters (``fit_edit_text``).
    A diff, value or patch is at most ``HISTORY_EDIT_CHARS`` characters plus a "(N more
    chars)" marker. The "Each step's settings" table already shows the current
    values in full. ``own``: only the last ``own`` edits are shown. A text (or a diff
    line) a current knob holds (``homes``) is named by that knob, and with ``away`` (the
    section showing these edits whole) every other long text by that section
    (`refer_to_knobs`); the wiring stays whole."""
    homes = homes or {}
    values: dict[str, Any] = dict(before or {})
    parts: list[str] = []
    listed = list(edits) if isinstance(edits, (list, tuple)) else [edits]
    # ``own``: only the last ``own`` edits are shown (the candidate's own, its ancestors'
    # are on their own lines); the earlier ones still set the values a diff compares with.
    start = len(listed) - own if own is not None else 0
    for index, edit in enumerate(listed):
        if index < start:
            if isinstance(edit, dict) and edit.get("op") == "set_knob":
                values[edit.get("name")] = edit.get("value", _UNKNOWN)
            continue
        if not isinstance(edit, dict):
            parts.append(_bounded(json.dumps(edit, default=str)))
            continue
        op, name = edit.get("op"), edit.get("name")
        if op == "set_knob" and "value" in edit:
            new, old = edit["value"], values.get(name, _UNKNOWN)
            values[name] = new
            if isinstance(new, str) and isinstance(old, str):
                diff = _text_diff(old, new).split(DIFF_LINE_SEPARATOR)
                diff = [line[0] + refer_to_knobs(line[1:], homes, away)
                        if line[:1] in ("+", "-") else line
                        for line in diff]
                parts.append(f"set_knob {name}: {len(old)} -> {len(new)} chars; diff: "
                             + _bounded(DIFF_LINE_SEPARATOR.join(diff)))
            elif isinstance(new, str):
                # Not pointed ``away``: that section may show this value as its change.
                parts.append(f"set_knob {name} = ({len(new)} chars) "
                             + _bounded(json.dumps(refer_to_knobs(new, homes))))
            else:
                parts.append(f"set_knob {name} = " + _bounded(json.dumps(new, default=str)))
        elif op == "set_knob" and "patch" in edit:
            values[name] = _UNKNOWN  # the patched value is not tracked here
            parts.append(f"set_knob {name} patch: "
                         + _bounded(json.dumps(refer_to_knobs(edit["patch"], homes, away),
                                               default=str)))
        else:
            parts.append(json.dumps(fit_edit_text(refer_to_knobs(edit, homes, away), HISTORY_EDIT_CHARS,
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
    # Short, because it heads every candidate's lines: a 100+ character label said once
    # per candidate is a repeat (`request_sizes.repeated_sentences`).
    return [*said, "As built (the candidate's new + and changed ~ pipeline lines: what ran):",
            "```text", *built, "```",
            *(["Check the description above against As built."] if said else [])]


#: The kinds of changed item, in the order the section picks them in turn.
EDIT_KINDS = (("broken", "broken"), ("fixed", "fixed"), ("down", "score dropped"),
              ("up", "score rose"))
#: An answer as a changed item's one line says it.
MOVED_ANSWER_CHARS = 160
#: The most characters one reply takes in "What the last edit changed" (head and tail).
EDIT_REPLY_CHARS = 4000


def _runs(record: Mapping[str, Any]) -> list[tuple[tuple[str, int], Mapping[str, Any]]]:
    """Each model call of an item keyed by its step and that step's run (a step a loop
    sent back runs more than once), in the order it ran."""
    out, counts = [], {}
    for exchange in record.get("exchanges") or ():
        if isinstance(exchange, Mapping):
            node = str(exchange.get("node") or "agent")
            counts[node] = counts.get(node, 0) + 1
            out.append(((node, counts[node]), exchange))
    return out


def step_moves(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, list]:
    """How an item's steps moved from parent to child, each step's run its own (a loop's
    second run of a step is compared with the parent's second run): ``same`` (the same
    reply), ``changed`` (a different reply), ``new`` (the child ran it, the parent did
    not) and ``gone`` (the reverse), each a list of ``(node, run)`` in the order it ran."""
    before = {key: str(e.get("response") or "").strip() for key, e in _runs(parent)}
    after = _runs(child)
    keys = {key for key, _e in after}
    moves: dict[str, list] = {"same": [], "changed": [], "new": [], "gone": []}
    for key, exchange in after:
        reply = str(exchange.get("response") or "").strip()
        moves["new" if key not in before else "same" if before[key] == reply
              else "changed"].append(key)
    moves["gone"] = [key for key, _e in _runs(parent) if key not in keys]
    return moves


def _verdicts(record: Mapping[str, Any], gate: str) -> str:
    """The verdicts a gate (or loop check) gave on an item, in order, a run of the same
    verdict counted (``reloop x2, proceed``; case aside, as both runtimes route). A
    reloop its cap refused (`items.refused_at_cap`) is said on its own, ``reloop
    (refused: cap of 1 reached)``, and the gate's reruns -- the reloops of that verdict
    it took -- are said once after them; ``not run`` when it gave none."""
    visits = [v for v in record.get("steps") or ()
              if isinstance(v, Mapping) and str(v.get("node")) == gate and v.get("verdict")]
    out: list[list[Any]] = []
    for v in visits:
        verdict = str(v.get("verdict")).strip()
        refused = v.get("cap") if v.get("exhausted") else None
        if out and out[-1][0].lower() == verdict.lower() and out[-1][2] is None \
                and refused is None:
            out[-1][1] += 1
        else:
            out.append([verdict, 1, refused])
    said = [(verdict if count == 1 else f"{verdict} x{count}")
            + (f" (refused: cap of {cap} reached)" if cap is not None else "")
            for verdict, count, cap in out]
    refused_labels = {str(v.get("verdict")).strip().lower() for v in visits if v.get("exhausted")}
    if refused_labels:
        reruns = sum(1 for v in visits if not v.get("exhausted")
                     and str(v.get("verdict")).strip().lower() in refused_labels)
        said.append(f"{reruns} rerun{'' if reruns == 1 else 's'} taken")
    return ", ".join(said) or "not run"


def step_facts(parent: Mapping[str, Any], child: Mapping[str, Any]) -> str:
    """How an item's steps moved, from the records alone and in few words: each step whose
    reply changed, is new or is gone, with its runs on parent and child (``solve 1 -> 2
    run(s), reply changed``), and each gate's verdicts on both (``check: parent proceed;
    child reloop, proceed``). Empty when nothing moved."""
    moves = step_moves(parent, child)
    runs = {"parent": {}, "child": {}}
    for side, record in (("parent", parent), ("child", child)):
        for (node, _run), _e in _runs(record):
            runs[side][node] = runs[side].get(node, 0) + 1
    status: dict[str, str] = {}
    for name, label in (("changed", "reply changed"), ("new", "new"), ("gone", "gone")):
        for node, _run in moves[name]:
            said = label
            if name == "gone" and runs["child"].get(node):
                said = "fewer runs"  # it still runs, less often; "gone" is not at all
            elif name == "new" and runs["parent"].get(node):
                said = "more runs"
            status.setdefault(node, said)
    parts = [f"`{node}` {runs['parent'].get(node, 0)} -> {runs['child'].get(node, 0)} "
             f"run(s), {label}" for node, label in status.items()]
    return "; ".join([*parts, *_gate_facts(parent, child)])


def _gate_facts(parent: Mapping[str, Any], child: Mapping[str, Any]) -> list[str]:
    """Each gate whose verdicts on the item differ between parent and child, both said."""
    gates = list(dict.fromkeys(
        str(v.get("node")) for record in (parent, child) for v in record.get("steps") or ()
        if isinstance(v, Mapping) and v.get("verdict")))
    return [f"`{gate}`: parent {_verdicts(parent, gate)}; child {_verdicts(child, gate)}"
            for gate in gates if _verdicts(parent, gate) != _verdicts(child, gate)]


def _moved_line(parent: Mapping[str, Any], child: Mapping[str, Any], kind: str,
                child_id: Any, why: str = "") -> str:
    """A changed item not shown whole, in one line: both scores, each side's verdict and
    answer, how its steps moved (`step_facts`) and why it is not shown whole (``why``)."""

    def answer(value: Any) -> str:
        text = " ".join(str(value).split())
        if len(text) <= MOVED_ANSWER_CHARS:
            return text
        return f"{text[:MOVED_ANSWER_CHARS]}… (clipped: {len(text):,} chars)"

    def side(record: Mapping[str, Any]) -> str:
        return _outcome(record) + (f", gave {answer(record.get('prediction'))}"
                                   if record.get("prediction") is not None else "")

    steps = step_facts(parent, child)
    return (f"- Item {child.get('id')} -- {kind} (child {child_id}), score "
            f"{fmt_score(parent.get('score'))} -> {fmt_score(child.get('score'))}: parent "
            f"{side(parent)}; child {side(child)}"
            + (f"; {steps}" if steps else "; no reply changed")
            + (f" ({why})" if why else "") + ".")


def _moved_replies(parent: Mapping[str, Any], child: Mapping[str, Any], kind: str,
                   child_id: Any, shown: dict[str, str]) -> list[str]:
    """How the edit moved one item's replies: each step whose reply changed, is new or is
    gone (`step_moves`), the parent's reply and the child's, each at most
    ``EDIT_REPLY_CHARS`` (head and tail, the cut marked), fenced with its headings escaped
    (`evidence.fenced`). A reply shown before -- under
    "Trajectories" or earlier in this section (``shown``: reply -> where) -- is named by
    where it is; each reply shown here is added to ``shown``."""
    from .evidence import fenced

    moves = step_moves(parent, child)
    before, after = dict(_runs(parent)), dict(_runs(child))
    where = f"item {child.get('id')} ({kind}, child {child_id})"

    def reply(side: str, record: Mapping[str, Any]) -> list[str]:
        text = str(record.get("response") or "").strip()
        if not text:
            return [f"{side}: (no reply)"]
        if text in shown:
            return [f"{side}: as {shown[text]}."]
        shown[text] = f"under \"What the last edit changed\", {where}"
        return [f"{side}:", *fenced(_clip_middle(text, EDIT_REPLY_CHARS))]

    lines = [f"#### Replies that moved on {where}"]
    for label, keys in (("changed", moves["changed"]), ("new", moves["new"]),
                        ("gone", moves["gone"])):
        for node, run in keys:
            lines.append(f"Step `{node}` (run {run}, {label}):")
            if label != "new":
                lines += reply("Parent's reply", before[(node, run)])
            if label != "gone":
                lines += reply("Child's reply", after[(node, run)])
    return lines if len(lines) > 1 else []


def _edit_change_lines(pairs: Sequence[dict[str, Any]],
                       homes: Mapping[str, str] | None = None,
                       shown: dict[str, str] | None = None) -> list[str]:
    """'What the last edit changed': each recent child of the edited candidate against
    its parent, on train items only: its edits whole (`whole_edits`: a ``set_knob`` as the
    lines it changed against the pair's ``before`` values), what they built, the counts of
    items fixed and broken, its prediction checked, every changed item (fixed, broken,
    the largest score rises and drops) in one line (`_moved_line`), and, for its first
    broken and its first fixed item, the replies that moved (`_moved_replies`). This is
    the home of each child's edits and prediction; other sections point here. A text a
    current knob holds (``homes``) is named by that knob."""
    if not pairs:
        return []
    shown = {} if shown is None else shown
    labels = dict(EDIT_KINDS)
    lines = ["## What the last edit changed",
             "The latest children of the candidate you edit (every child of its last round, "
             "siblings included), or that candidate itself when it has none yet, each "
             "compared with its parent item by item on the train items both were scored on "
             "(the parent's records from the child's epoch, so the same seed, when it was "
             "scored then; a pair scored with different seeds is marked, and some of its "
             "changes may be run-to-run noise). Each changed item is one line: both scores, "
             "each side's answer and the steps whose reply changed, is new or is gone. For "
             f"the first broken and the first fixed item of each child, the replies that "
             f"moved follow, parent's then child's, each at most {EDIT_REPLY_CHARS:,} "
             "characters (head and tail)."]
    for pair in pairs:
        head = (f"### Child {pair.get('child')} of {pair.get('parent') or '(none)'}"
                + (f" -- gate: {pair['gate']}" if pair.get("gate") else ""))
        lines += ["", head, "Edits:", whole_edits(pair.get("edit"), pair.get("before"), homes)]
        lines += [f"Wired by the system: {note}" for note in pair.get("notes") or ()]
        lines += _as_built_lines(pair)
        if pair.get("same_seed") is False:
            lines.append("Parent and child were scored in different epochs, with different "
                         "seeds: some changes below may be run-to-run noise.")
        child_records = list(pair.get("child_records") or ())
        lines += _item_error_lines(pipeline_item_errors(child_records), len(child_records))
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
                     f"{change['down'] or 'none'}."
                     + (f" Left out: {len(change['timed_out'])} item(s) the harness timed out "
                        "on one side." if change.get("timed_out") else ""))
        if pair.get("effect"):
            from .edit_effect import effect_lines

            lines += effect_lines(pair["effect"])
        if pair.get("prediction"):
            checked = prediction_check(pair["prediction"], pair.get("parent_records"),
                                       pair.get("child_records"), pair.get("steps") or ())
            lines += [f"Its prediction: {_clip(pair['prediction'], REASONING_TEXT_CHARS)}",
                      *([f"Prediction check -- {line}" for line in checked] or [
                          "Prediction check: it names no shared item or step, so only the "
                          "counts above can say whether it held."])]
        parent, child = (_by_item(pair.get("parent_records")),
                         _by_item(pair.get("child_records")))
        changed = [(key, item) for key, _label in EDIT_KINDS for item in change[key]]
        if changed:
            lines += ["", *(_moved_line(parent[item], child[item], labels[key],
                                        pair.get("child")) for key, item in changed)]
        for key in ("broken", "fixed"):
            if change[key]:
                item = change[key][0]
                lines += ["", *_moved_replies(parent[item], child[item], labels[key],
                                              pair.get("child"), shown)]
    return lines


def reasoning_shown(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The reasoning entries 'Your earlier reasoning' shows in full: the latest round's.
    Older ones are in "What this run has already tried", one line each, and in the memory
    note, not repeated there."""
    latest = max((e.get("epoch") for e in entries if isinstance(e.get("epoch"), int)),
                 default=None)
    recent = [e for e in entries if latest is None or e.get("epoch") == latest]
    return list(recent)[-REASONING_ENTRIES:]


#: The gate words of a change that did not become the current candidate (``rejected``, or
#: under ``checkpoint`` ``not applied``), as "Your earlier reasoning" names one.
REFUSED_GATE_WORDS = {"rejected": "rejected", "not applied": "not-applied"}


def _promising(entries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The latest rejected (or, under ``checkpoint``, not-applied) candidate with train
    counts against its parent, whatever its net count, whose edits a request shows whole
    with what went wrong in it, so it can be built on; ``None`` when there is none."""
    rejected = [entry for entry in entries if entry.get("gate") in REFUSED_GATE_WORDS
                and isinstance(entry.get("fixed"), int) and isinstance(entry.get("broken"), int)]
    return rejected[-1] if rejected else None


def _reasoning_lines(entries: Sequence[dict[str, Any]], above: Sequence[str] = (),
                     predicted: Sequence[str] = (),
                     homes: Mapping[str, str] | None = None,
                     reviewed: Collection[str] = ()) -> list[str]:
    """'Your earlier reasoning': what the optimizer said about recent candidates,
    beside what happened to them, with each one's item errors (``item_errors``). A
    candidate in ``above`` ("What the last edit changed" shows its edits, its flow change
    and what they built, and for one in ``predicted`` its prediction) points there instead
    of repeating them; a text a current knob holds (``homes``) is named by that knob."""
    if not entries:
        return []
    lines = ["## Your earlier reasoning and what happened",
             "Recent candidates on the line you edit, and recently refused ones, with what you "
             "predicted when you proposed them and what came of it."]
    for entry in reasoning_shown(entries):
        outcome = ", ".join(part for part in (
            f"gate {entry['gate']}" if entry.get("gate") else "",
            f"train score {fmt_score(entry.get('score'))}",
            (f"{entry['fixed']} fixed / {entry['broken']} broken vs its parent"
             + (" (scored with different seeds)" if entry.get("same_seed") is False else "")
             if entry.get("fixed") is not None else "")) if part)
        lines += ["", f"### {entry.get('id')} (proposed in epoch {entry.get('epoch', '?')}) -- {outcome}",
                  *_item_error_lines(entry.get("errors") or (), int(entry.get("items") or 0)),
                  *[f"Wiring warning this edit added: {w}" for w in entry.get("warnings") or ()],
                  *([("Its edits, flow change, prediction and what they built"
                      if str(entry.get("id")) in predicted else
                      "Its edits, flow change and what they built")
                     + ": under \"What the last edit changed\" above."]
                    if str(entry.get("id")) in above else [
                      "Edits:", whole_edits(entry.get("edit"), entry.get("before"), homes),
                      *[f"Wired by the system: {note}" for note in entry.get("notes") or ()]]),
                  f"Rationale: {_clip(entry.get('rationale'), REASONING_TEXT_CHARS) or '(none)'}",
                  *([f"Addresses: {_clip(entry['addresses'], REASONING_TEXT_CHARS)}"]
                    if entry.get("addresses") else []),
                  *([] if str(entry.get("id")) in above else _as_built_lines(entry)),
                  *([] if str(entry.get("id")) in predicted else [
                      f"Prediction: {_clip(entry.get('prediction'), REASONING_TEXT_CHARS) or '(none)'}"]),
                  f"Diagnosis that round: {_clip(entry.get('diagnosis'), REASONING_TEXT_CHARS) or '(none)'}",
                  *([f"Considered that round: {_clip(entry['considered'], REASONING_TEXT_CHARS)}"]
                    if entry.get("considered") else [])]
    entry = _promising(entries)
    if entry is not None:
        from .edit_effect import effect_lines

        head = (f"### Full edits of {entry.get('id')}, the latest "
                f"{REFUSED_GATE_WORDS[str(entry.get('gate'))]} candidate "
                f"({entry['fixed']} fixed / {entry['broken']} broken)")
        # How it moved the items it changed (`optim.edit_effect`): the mechanism to fix. A
        # change that added no step has no such account; its counts are what there is.
        went = effect_lines(entry.get("effect") or {})
        wrong = ([f"How it moved the items it changed: {went[0]}",
                  *went[1:]] if went else
                 [f"What it did: it broke {entry['broken']} train item(s) and fixed "
                  f"{entry['fixed']} against its parent"
                  + ("; the replies that moved on its first broken and first fixed item are "
                     "under \"What the last edit changed\"." if str(entry.get("id")) in above
                     else ".")])
        # Pointed to only when that section reviews a step of this very candidate.
        review = ("A step it added is reviewed under \"Did the added steps contribute?\"."
                  if str(entry.get("id")) in set(reviewed) else "")
        shown = [str(e.get("id")) for e in reasoning_shown(entries)]
        if str(entry.get("id")) in above or str(entry.get("id")) in shown:
            # Its edits are whole in its own block already (`whole_edits`).
            where = ("under \"What the last edit changed\" above"
                     if str(entry.get("id")) in above else "in its block above")
            lines += ["", head, f"Its edits are whole {where}. It was rejected; proposing it "
                      "again unchanged is refused.", *wrong, *([review] if review else [])]
        else:
            parts = whole_edit_parts(entry.get("edit"), entry.get("before"), homes)
            # Past the bound, the longest edits are cut to one length (each its middle,
            # said in it); no edit is dropped.
            from .analysts import item_cap

            cap = item_cap([len(p) for p in parts], REJECTED_EDITS_CHARS, floor=200)
            # A cut edit is one line (its fence, if any, folded): head and tail around the
            # cut, so no part of it can open a fence or read as a heading.
            shown = [p if cap is None or len(p) <= cap else _one_line_cut(p, cap)
                     for p in parts]
            cut = sum(1 for p, q in zip(parts, shown) if p != q)
            lines += ["", head,
                      "Every edit whole, a new step's instructions and a judge's criteria "
                      "included" + (f" (past {REJECTED_EDITS_CHARS} characters, {cut} of "
                                    f"its {len(parts)} edits have their middle cut, each "
                                    "cut said in it; none is left out)" if cut else "")
                      + ". It was rejected; proposing it again unchanged is refused.",
                      "\n".join(shown) if shown else "(none)",
                      *wrong, *([review] if review else [])]
    return lines


def _one_line_cut(text: str, limit: int) -> str:
    """``text`` on one line (newlines as ``DIFF_LINE_SEPARATOR``) within ``limit``: its
    head and tail around ``[... N characters omitted ...]``."""
    flat = text.replace("\n", DIFF_LINE_SEPARATOR)
    if len(flat) <= limit:
        return flat
    mark = f" [... {len(flat) - limit} characters omitted ...] "
    keep = max(0, limit - len(mark))
    return flat[:(keep + 1) // 2] + mark + (flat[len(flat) - keep // 2:] if keep // 2 else "")


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


def _attribution_key(value: str, knob: str, current: Mapping[str, Any],
                     seed: Mapping[str, Any], first: Any = None) -> str:
    """A knob value as ``knob_attribution`` names it: a long text (`SAID_ONCE_CHARS`+) by
    which one it is -- the current one (whole under "Each step's settings"), the seed's,
    or the one candidate ``first`` was first evaluated with -- with its length; a short
    one as it is. Unique for the knob's values; `_trace_rows` never merges two."""
    if len(value) < SAID_ONCE_CHARS:
        return value
    if isinstance(current.get(knob), str) and value == current[knob]:
        return f"the current text ({len(value)} chars)"
    if isinstance(seed.get(knob), str) and value == seed[knob]:
        return f"the seed's text ({len(value)} chars)"
    if first:
        return f"the text {first} was first scored with ({len(value)} chars)"
    return f"a text of {len(value)} chars, #{hashlib.sha1(value.encode()).hexdigest()[:8]}"


def _trace_rows(traces: Sequence[dict[str, Any]],
                omit: Sequence[str] = (), *, current: Mapping[str, Any] | None = None,
                seed: Mapping[str, Any] | None = None) -> list[tuple[str, Any]]:
    """The traces as the request shows them: no receipts, no paths, short texts, and
    none of the top-level rows ``omit`` names. A text value in ``knob_attribution`` is
    named by which it is (`_attribution_key`: ``current`` and ``seed`` are the knobs'
    current and starting values), never repeated."""

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

    def named(attribution: Any) -> Any:
        if not isinstance(attribution, dict):
            return attribution
        out = {}
        for knob, entry in attribution.items():
            if isinstance(entry, dict) and isinstance(entry.get("tried"), dict):
                tried: dict[str, Any] = {}
                for value, row in entry["tried"].items():
                    key = _attribution_key(str(value), str(knob), current or {}, seed or {},
                                           row.get("first") if isinstance(row, dict) else None)
                    if key in tried:  # never one row for two values
                        key += f" #{hashlib.sha1(str(value).encode()).hexdigest()[:8]}"
                    # Which candidate it came from is in the key when it names a text.
                    tried[key] = ({k: v for k, v in row.items() if k != "first"}
                                  if isinstance(row, dict) else row)
                entry = {**entry, "tried": tried}
            out[knob] = entry
        return out

    return [(key, clean(named(value) if key == "knob_attribution" else value))
            for key, value in _flatten(traces)
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


def _table_filled(headers: list[str], rows: list[list[Any]]) -> str:
    """`_table` without the columns that are empty in every row (nothing to read there)."""
    keep = [i for i in range(len(headers))
            if any(row[i] not in (None, "", []) for row in rows)]
    return _table([headers[i] for i in keep], [[row[i] for i in keep] for row in rows])


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
    #: What a step may choose its own route from (`workers.StepRoutes`); one outside it is
    #: refused with ``E-route``. The loop sets it from its model; None offers nothing.
    step_routes: Any = None
    #: Whether a step that runs its tools sealed can run in this run, and why not
    #: (`workspace.sealed_steps`); the loop sets it from its model, and the request offers
    #: a tool step on the same answer. ``None`` asks whether the seal runs here.
    sealed_tools: tuple[bool, str] | None = None
    #: How a new sealed step is judged on the client its dispatch takes, its own route
    #: included (`workspace.sealed_step`, bound to the loop's model); ``None`` judges by
    #: ``sealed_tools`` alone.
    sealed_check: Any = None
    #: Whether a step an edit adds may have tools at all; the loop sets it from
    #: ``new_step_tools``. Off, every new agent step with tools is refused
    #: (``E-new-node-tools``), sealed or not.
    new_step_tools: bool = True
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
        #: What the last accepted repair dropped or changed of the proposal's text
        #: (`repair.lost_content`), or empty.
        self.last_repair_lost: str = ""
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
                repair_losses=[*ctx.repair_losses, *repair_losses(self.rejections)],
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
                        dropped=self.last_repair_lost if fixed is not None else "",
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
        the last fix's refusal (empty when no fix was tried), for the caller to keep. When
        one is, ``last_repair_lost`` says what it dropped or changed of the proposal's
        text (`repair.lost_content`), for later requests."""
        self.last_repair_refusal = []
        self.last_repair_lost = ""
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
                # What the fix dropped or changed of the proposal as written.
                self.last_repair_lost = repair.lost_content(proposal.steps, fixed.steps)
                return outcome
            current, why = fixed, list(outcome)
            # The fix's own refusal, for the caller to keep with the proposal's.
            self.last_repair_refusal = list(outcome)
        return None

    def review_wiring(self, ctx: ProposalContext, proposal: Proposal, warnings: Sequence[str],
                      parent: Candidate) -> Proposal | None:
        """The proposal with its wiring changed after it was shown the wiring warnings its
        edit adds (`nn.validate.wiring_warnings`), or ``None`` to keep it as it is
        (``Loop._wiring_check``). The default asks nothing and keeps it: an optimizer that
        asks a model overrides it with one slim call (`AutoresearchOptimizer.review_wiring`).
        After a call, ``last_wiring_answer`` holds what the proposer said."""
        self.last_wiring_answer = ""
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
        search added may run with tools only sealed (``E-new-node-tools``). ``graders`` is the mode's ``grader_nodes()``: with the
        protected nodes, the only nodes besides the data nodes that may consume
        grader-derived data (V17).

        ``record`` False records no call (the request's "What this graph allows" asks
        this for every edit it lists). A precondition refusal that names a replacing edit
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
                graders=graders, record=False,
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
        # Each applied edit with the states before and after it, for what a core operator
        # wired (`core_ops.wiring_notes`), said once the whole proposal is applied.
        wired: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        from refactory_dsh.nn import core_ops

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
                # A node object written out whole that runs its tools sealed in a pass and
                # names no reads takes the item's input files, as a core operator's new
                # tool step does; the edit recorded is the one with them.
                defaulted = (core_ops.default_item_inputs(edit, before_step, state)
                             if getattr(self, "new_step_tools", True) else None)
                if defaulted is not None:
                    edit, reports = defaulted, []
                    state = nn_edits.apply(edit, before_step, edit_reports=reports)
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
            wired.append((edit, before_step, state))

        violation = nn_edits.protected_violation(parent.state, state, protected)
        if violation:
            return [f"E-protected: {violation}" + _protected_culprit(applied, parent.state,
                                                                      prefixes, protected)]
        # A grading node is never rewired by an edit either (V17's grader side).
        for grader in graders:
            old = nn_edits._graph(parent.state).nodes.get(grader)
            new = nn_edits._graph(state).nodes.get(grader)
            if old is not None and (new is None or new.to_state() != old.to_state()):
                return [f"E-protected: node {grader!r} grades, and the edit changes its "
                        "declaration"]
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
        wired_state, wiring, wired_fixes = repair.wire_file_slots(parent.state, state)
        if wiring and not nn_edits.protected_violation(parent.state, wired_state, protected):
            state = wired_state
            applied += wiring
            edit_reports += [None] * len(wiring)
            repairs += wired_fixes

        # ── tool-policy escalation check (anchored to the original seed) ──
        escalation = _tool_policy_escalated(seed_state if seed_state is not None else parent.state, state)
        if escalation:
            return escalation
        tooled = _new_node_tools(seed_state if seed_state is not None else parent.state, state,
                                 getattr(self, "sealed_tools", None),
                                 getattr(self, "sealed_check", None),
                                 bool(getattr(self, "new_step_tools", True)))
        if tooled:
            return tooled
        routed = _step_route_refusals(parent.state, state, getattr(self, "step_routes", None))
        if routed:
            return routed
        bound = _route_binding_refusals(seed_state if seed_state is not None else parent.state, state)
        if bound:
            return bound
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
        # A step the edits add has a reader when they are done (finding 16): an output
        # nothing reads changes no result.
        from refactory_dsh.nn.core_ops import unread_by_bundle

        # The last state is the one wired (file_text steps reading a file are readers).
        unread = unread_by_bundle(parent.state, [*prefixes[:-1], state], applied)
        if unread:
            return [f"E-unread: {unread}"]
        # A knob whose text reaches no step any more (an op fills it into a prompt nothing
        # reads) changes no result: setting it is refused, naming what it used to reach.
        from refactory_dsh.nn.core_ops import dead_knob_reason

        for index, step in enumerate(applied):
            name = step.get("name") if step.get("op") == "set_knob" else None
            if isinstance(name, str) and (state.get("knobs") or {}).get(name) \
                    != (parent.state.get("knobs") or {}).get(name):
                dead = dead_knob_reason(name, state, parent.state)
                if dead is not None:
                    return [_at(f"E-unread: {dead}, so setting it changes no result; set "
                                "the knob that step reads now instead", index, applied)]
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
        # What a core operator wired beside the steps it names, said with the candidate: what
        # each edit did, and what is true of the candidate judged on its own state, after
        # every edit (a reader a later edit adds is a reader).
        wired_notes = [note for edit, before_step, after_step in wired
                       for note in nn_edits.core_wiring_notes(edit, before_step, after_step,
                                                              state)]
        if getattr(self, "new_step_tools", True):
            # Whether each sealed tool step the proposal adds or changes in a pass sees the
            # item's input files: said of the candidate, after every edit.
            wired_notes += core_ops.item_input_notes(parent.state, state)
        candidate.repairs, candidate.notes = list(repairs), [*wired_notes, *notes]
        if repairs and record:
            self.record_call(kind="repair", how="deterministic", candidate=candidate.id,
                             parent=parent.id, fixes=repairs)
        return candidate
