"""Reason codes, proposal refusals, code/secret/route refusals, repair losses."""


from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.tool_policy import no_effect

from .types import (
    COUNTING_RULE,
    MAX_RAW_EDITS_PER_PROPOSAL,
    NODE_OBJECT_FIELDS,
    Proposal,
    Rejection,
    candidate_id,
)

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


# --- cross-module imports (resolved at definition time, used at call time) ---
from .request.edits import describe_edits  # noqa: E402
from .request.items import _clip  # noqa: E402
