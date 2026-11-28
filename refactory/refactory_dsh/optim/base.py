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

import json
from dataclasses import replace
from typing import Any, Sequence

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import validate_state
from refactory_dsh.nn.knob import Knob
from refactory_dsh.nn.validate import Invariant

from . import checks, repair
from .refusals import (  # noqa: F401, E402
    NEW_STEP_TOOLS_OFF,
    REASON_LIMIT,
    REASONS_PER_CODE,
    REFUSED_EDITS_CHARS,
    REFUSED_SHOWN,
    UNREADABLE_PROPOSAL,
    _at,
    _code_node_refusals,
    _code_signatures,
    _dead_edit,
    _fresh_workspace,
    _grader_side_knobs,
    _internal_node_fields,
    _lowering_refusal,
    _new_node_tools,
    _patch_targets_missing,
    _protected_culprit,
    _reason_priority,
    _route_binding_refusals,
    _secret_refusals,
    _step_route_refusals,
    _tool_policy_escalated,
    proposal_refusal,
    reason_code,
    refused_entries,
    refused_lines,
    repair_loss_lines,
    repair_losses,
    summarise_reasons,
    unreadable_proposal,
)
from .request.composition import (  # noqa: F401, E402
    _OPTIONAL_EDIT_FIELDS,
    _REPLACE_HINT,
    _STEP_FIELDS,
    _TAKE_OVER_TEXT,
    _TAKES_TO,
    COMPOSITION_LEAD,
    COMPOSITION_SHAPES,
    CORE_OPERATORS,
    NO_TOOLS_LINE,
    PIPELINE_VIEW_LEAD,
    REFERENCE_HEADING,
    REFERENCE_POINTER,
    REFUSED_LEAD,
    STRUCTURAL_LEAD,
    STRUCTURE_KINDS,
    _blocks_section,
    _can_write,
    _composition_anchor,
    _example_ids,
    _example_refusal,
    _example_state,
    _fact_filled_knobs,
    _fit_state,
    _fitting_ids,
    _fork_inputs,
    _item_count,
    _node_op,
    _or_list,
    _reader_edit,
    _removable_step,
    _sealed_code_imports,
    _takes_over_parts,
    _view_lead,
    _with_outputs_said,
    checked_hint,
    composition_examples,
    composition_section,
    example_node,
    graph_facts,
    operation_lines,
    operator_lines,
    pipeline_view_lines,
    structural_example,
    structural_ops,
    tool_environment_lines,
    tool_step_ops,
    tools_line,
)
from .request.context import (  # noqa: F401, E402
    _CODE_STEP_TEXT,
    _DECISION_LEAD,
    _LEVEL_TEXT,
    _NOT_DECISIONS,
    BARE_HEADING_WORDS,
    EDIT_LEVELS,
    ITEM_DATA_MIN,
    NODE_SPEC_SECTION_CHARS,
    NODE_SPEC_TEXT_CHARS,
    NODE_SPEC_TEXT_FLOOR,
    PROPOSAL_ROOM,
    READER_CONCURRENCY,
    READER_FINDINGS_SHARE,
    READER_ITEM_IDS,
    READER_PATTERN_CHARS,
    READER_PATTERNS,
    READER_QUOTE_CHARS,
    READER_QUOTES,
    TAIL_LABEL_MAX,
    TAIL_MIN,
    TAIL_MIN_WORDS,
    TRAJECTORY_BUDGET,
    ProposalContext,
    _bare_label,
    _count,
    _data_lines,
    _evidence_rows,
    _exact_in_table,
    _feedback_sections,
    _fence,
    _holds,
    _level_text,
    _levers_line,
    _node_reference,
    _prior_proposals_lines,
    _settings_lines,
    _strategy_section,
    _whole_texts,
    edit_level,
    levels_instructions,
    settled_for,
    structural_text,
)
from .request.edits import (  # noqa: F401, E402
    _ERROR_KIND,
    _ERROR_STEP,
    _FAILURE_MARKER,
    _UNKNOWN,
    AS_BUILT_LINE_CHARS,
    AS_BUILT_LINES,
    AVAILABLE_SECTION_CHARS,
    DIFF_LINE_SEPARATOR,
    EDIT_CHANGE_GRADED,
    EDIT_FIELD_CHARS,
    EDIT_KINDS,
    EDIT_NAMING_FIELDS,
    EDIT_NODE_FIELDS,
    EDIT_REPLY_CHARS,
    EDIT_TEXT_MIN,
    EDIT_WIRING_FIELDS,
    HISTORY_EDIT_CHARS,
    HISTORY_NODE_PROMPT_CHARS,
    ITEM_ERROR_CHARS,
    MOVED_ANSWER_CHARS,
    POINTER_HEAD_CHARS,
    PREDICTION_NAMED,
    REASONING_ENTRIES,
    REASONING_TEXT_CHARS,
    REFUSED_GATE_WORDS,
    REJECTED_BUILD_TEXT,
    REJECTED_EDITS_CHARS,
    SAID_ONCE_CHARS,
    SETTINGS_HOME,
    WRITTEN_OUT_FIELDS,
    _as_built_lines,
    _attempt_lines,
    _available_section,
    _bounded,
    _by_item,
    _core_fields,
    _current_line,
    _cut,
    _edit_change_lines,
    _edit_fields,
    _error_line,
    _error_origin,
    _folded,
    _gate_column,
    _gate_facts,
    _gate_note,
    _history_cell,
    _item_error_lines,
    _knob_home,
    _map_text,
    _moved_line,
    _moved_replies,
    _named,
    _node_spec,
    _node_specs,
    _one_line_cut,
    _promising,
    _reasoning_lines,
    _runs,
    _score_of,
    _step_replies,
    _text_diff,
    _text_limit,
    _verdicts,
    available_before,
    compact_edits,
    describe_edits,
    edit_change,
    fit_edit_text,
    item_errors,
    knob_homes,
    node_object_ops,
    prediction_check,
    reasoning_shown,
    refer_to_knobs,
    step_facts,
    step_moves,
    text_pointer,
    whole_edit_parts,
    whole_edits,
)
from .request.items import (  # noqa: F401, E402
    _TABLE_CELL_CHARS,
    _WHOLE,
    CONSIDERED_CHARS,
    ERROR_CAUSE_REF,
    HARNESS_TIMEOUT,
    HARNESS_TIMEOUT_KINDS,
    ITEM_EVIDENCE_CHARS,
    ITEM_EVIDENCE_ROWS,
    ITEM_FORM,
    ITEM_TEXT_MAX,
    NO_ANSWER_CAUSES,
    OMITTED,
    READER_SUCCESS_PATTERNS,
    READER_SUCCESS_SHARE,
    SHOW_ALL_BELOW,
    SHOWN_TEXT_FLOOR,
    TRAJECTORIES_HEADING,
    TRAJECTORY_INPUT_KEYS,
    _all_whole,
    _analysed_findings,
    _choice_lines,
    _cited_items,
    _clip,
    _clip_middle,
    _completed,
    _contrast_items,
    _cut_to_fit,
    _error_causes,
    _expected,
    _failure_kind,
    _fit_shown,
    _item_block,
    _item_evidence,
    _longest_fitting,
    _marked,
    _ordered_items,
    _outcome,
    _pipeline_items,
    _reader_patterns,
    _shown_items,
    _shown_size,
    _trajectories,
    _trajectory,
    considered_text,
    error_refs,
    harness_timeout,
    pipeline_item_errors,
    shown_label,
    trajectories_whole,
    trajectory_groups,
)
from .request.objective import (  # noqa: F401, E402
    _CHECKPOINT_RULE,
    _GATE_RULE,
    _UNSCORED,
    ANSWER_INSTRUCTIONS,
    TRACE_OMITTED_KEYS,
    TRACE_TEXT_CHARS,
    UNSCORED_TRACE_KEYS,
    _attribution_key,
    _cell,
    _cost_cell,
    _cost_note,
    _flatten,
    _gate_rule,
    _rules_text,
    _table,
    _table_filled,
    _trace_rows,
    answer_instructions,
    bundle_examples,
    bundle_instructions,
    compute_policy,
    fmt_score,
    objective_lines,
    objective_text,
    objective_with_policy,
    time_lines,
)
from .types import (  # noqa: F401, E402
    COUNTING_RULE,
    MAX_EDITS_PER_PROPOSAL,
    MAX_RAW_EDITS_PER_PROPOSAL,
    NODE_OBJECT_FIELDS,
    REASONING_FIELDS,
    Candidate,
    Proposal,
    Rejection,
    RunStopped,
    _structural_edit,
    candidate_id,
)


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
