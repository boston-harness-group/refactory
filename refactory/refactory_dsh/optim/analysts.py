"""The analyst stage: SkillOpt's per-round analysis, for the model proposer, over a graph.

``LoopConfig.analysts`` turns it on. Before the model proposer writes a round's
proposal, analysts read every train item of the candidate being edited in full, as
SkillOpt's reflect stage reads its rollouts (``refactory_packages/skillopt/analysts.py``
and ``merge.py`` port SkillOpt's ``reflect.py``, ``aggregate.py`` and ``clip.py``):

- **Failure analysts**: the failed items (every item not passed: answered wrong, no
  answer, or an error), shuffled with the round's seed and cut into batches of up to
  ``group_size`` (SkillOpt's ``minibatch_size``, 8), a batch closed early when the next
  item whole would take its call past ``ANALYST_ROOM`` (`batches`, `packer`): no item
  is cut to fit, the evidence is split across more calls. One call per batch. Each analyst
  sees the pipeline in the proposer's words (`pipeline_text`: the pipeline view and each
  step's settings, as SkillOpt's analyst sees the current skill) and each item's full trace
  (`evidence.TRACE_FORM`). It writes at most
  ``FINDINGS_PER_ANALYST`` findings (SkillOpt's analyst edit budget L, 4): what goes
  wrong, where in the graph (nodes, edges, settings, the structure), on which items,
  with quotes and its impact. Analysts report and never prescribe: a finding says what
  happens, not what to do (the proposer alone chooses the change).
- **Success analysts**: the passed items, likewise (seed + 1, as SkillOpt): what made
  them work.
- **Merge**: failure findings merged level by level in batches of ``merge_batch_size``
  (SkillOpt's, 8), then success findings, then a final merge with failure priority
  that returns one de-duplicated list ranked by how many items each covers and its
  impact. SkillOpt's shortcuts hold: one list is not merged, and the final merge runs
  only when both sides found something. An unreadable merge reply keeps every input
  finding (SkillOpt's fallback concatenation).

SkillOpt's patch and slow-update steps are not here: the proposer writes the edits (it
is the step SkillOpt's rank and patch are), so unlike SkillOpt's analysts (ported
unchanged in ``refactory_packages/skillopt``) ours write no suggested change. Its meta
skill is the loop's memory pass
(`optim.memory`): the per-epoch note every analyst and merge call reads here. The
ranked findings reach the proposal request as "What the analysts found, ranked" and
replace the readers' summaries (``AutoresearchOptimizer._analyse``).

Domain-free: the prompts name no dataset or task. Train records only: the analysts see
``ProposalContext.item_records`` (the loop hands over train records only) and the graph.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Callable, Mapping, Sequence

from . import evidence
from .base import (
    TRAJECTORY_INPUT_KEYS, _choice_lines, _clip, _expected, shown_label,
    _ordered_items, _outcome, NO_ANSWER_CAUSES,
)

#: SkillOpt's defaults (``_base_/default.yaml``: ``minibatch_size: 8``,
#: ``merge_batch_size: 8``), the defaults of ``LoopConfig.analyst_group_size`` and
#: ``analyst_merge_batch_size``.
GROUP_SIZE = 8
MERGE_BATCH_SIZE = 8
#: The most findings one analyst writes: SkillOpt's analyst edit budget L
#: (``analyst_edit_budget``, 4).
FINDINGS_PER_ANALYST = 4
#: One analyst call's whole prompt, system prompt included, in characters: the setting
#: tied to the optimizer model's context. Items are never cut to fit it; a batch is
#: closed when the next item would pass it (`packer`), so a larger room makes fewer
#: calls and a smaller one more, with every item whole. The optimizer model (Opus 4.6) reads
#: 200k tokens, of which Claude Code's own system prompt and the reserved reply take
#: about 30-40k. Traces dense in notation run 2.5-3.5 characters a token, so 400000
#: characters is about 115-160k tokens: inside the rest at all but the densest. A call
#: the provider still refuses as too long is retried once at half its length
#: (`context_overflow`, ``AutoresearchOptimizer._ask``).
ANALYST_ROOM = 400000
#: The least an item is cut to when a batch does not fit, however many items it holds.
ITEM_FLOOR = 4000
#: A step input is clipped to no less than this before any output is clipped.
INPUT_FLOOR = 1500
#: No text of an item is clipped below this.
TEXT_FLOOR = 40
#: A node text or setting value in the pipeline an analyst reads, head and tail.
PIPELINE_TEXT_CHARS = 20000
#: What a failed call says when its prompt was longer than the model's context: the
#: Claude Code and DSH runners' wording (`refactory_dsh.model.SETUP_FAILURE_MARKERS`)
#: and the providers' own.
CONTEXT_MARKERS = ("prompt is too long", "input is too long", "context_length_exceeded",
                   "context length", "context window", "maximum context", "too many tokens")


def context_overflow(error: str) -> bool:
    """Whether a failed call's error says its prompt did not fit the model's context."""
    lowered = (error or "").lower()
    return any(marker in lowered for marker in CONTEXT_MARKERS)
#: The most calls of one analyst stage in flight at once, analysts and merges together
#: (the two sides' merges run side by side): 8 runs SkillOpt's 35 train items (about 5
#: or 6 batches of 8) in one wave.
CONCURRENCY = 8
#: The ranked findings the proposal request shows, and their room.
SHOWN_FINDINGS = 16
SECTION_CHARS = 30000
#: The least a finding's text is cut to when the shown findings do
#: not fit ``SECTION_CHARS`` whole (`section_lines`); each cut says how much it cut.
FINDING_FLOOR = 100
#: The evidence quotes a finding keeps (twice SkillOpt's three, for a merged finding).
EVIDENCE_PER_FINDING = 6
ITEM_IDS = 20

KINDS = ("failure", "success")
IMPACTS = ("high", "medium", "low")


def settings(*, group_size: int = GROUP_SIZE, merge_batch_size: int = MERGE_BATCH_SIZE,
             max_batches: int = 0) -> dict[str, int]:
    """The stage's settings as ``ProposalContext.analysts`` carries them (empty is off)."""
    return {"group_size": max(1, int(group_size)),
            # SkillOpt merges at least two lists per call (aggregate.py:102).
            "merge_batch_size": max(2, int(merge_batch_size)),
            "max_batches": max(0, int(max_batches))}


# ── batching ────────────────────────────────────────────────────────────────


def split(records: Sequence[dict[str, Any]], seed: int
          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every train item once (its first repeat), split by ``passed`` as SkillOpt splits
    them (an error is a failure; a harness timeout is neither, `base.harness_timeout`),
    failures shuffled with ``seed`` and passes with ``seed + 1`` (SkillOpt's ``reflect.py``)."""
    items = sorted(_ordered_items(records), key=lambda r: str(r.get("id")))
    failures = [r for r in items if not r.get("passed")]
    successes = [r for r in items if r.get("passed")]
    random.Random(seed).shuffle(failures)
    random.Random(seed + 1).shuffle(successes)
    return failures, successes


def batches(failures: Sequence[dict[str, Any]], successes: Sequence[dict[str, Any]], *,
            group_size: int, max_batches: int = 0,
            fits: Callable[[str, Sequence[dict[str, Any]]], bool] | None = None
            ) -> tuple[list[tuple[str, int, list[dict[str, Any]]]], dict[str, int]]:
    """The analyst batches as ``(kind, index, records)``, failure batches first, and how
    many items of each kind no batch holds. Each kind's items, in order, fill a batch up
    to ``group_size``; with ``fits`` (`packer`), a batch is also closed when adding the
    next item whole would make its request too long (``fits(kind, group)`` false), so
    every item is in exactly one batch, whole. An item too long even alone is a batch of
    its own. ``max_batches`` (0: no limit, SkillOpt's) caps the calls: failure batches
    take the room first, as SkillOpt's merge gives failures priority."""
    out: list[tuple[str, int, list[dict[str, Any]]]] = []
    left = {"failure": 0, "success": 0}
    for kind, items in (("failure", failures), ("success", successes)):
        groups: list[list[dict[str, Any]]] = []
        for record in items:
            if groups and len(groups[-1]) < group_size and (
                    fits is None or fits(kind, [*groups[-1], record])):
                groups[-1].append(record)
            else:
                groups.append([record])
        for index, group in enumerate(groups, 1):
            if max_batches and len(out) >= max_batches:
                left[kind] += len(group)
                continue
            out.append((kind, index, group))
    return out, left


def packer(ctx: Any, pipeline: str, failures: Sequence[dict[str, Any]],
           successes: Sequence[dict[str, Any]], room: int = ANALYST_ROOM
           ) -> Callable[[str, Sequence[dict[str, Any]]], bool]:
    """``fits`` for `batches`: whether a batch's request, measured as `analyst_batch`
    builds it -- the system prompt, the frame with the pipeline, every item whole (each
    instruction once) and the separators between them -- is within ``room``. The frame is measured with the
    largest batch number the kind can reach, so the real request is never longer."""
    from .evidence import dedup

    texts: dict[int, str] = {}
    most = {"failure": len(failures), "success": len(successes)}
    settled = shown_settled(pipeline, ctx.item_records)

    def fits(kind: str, group: Sequence[dict[str, Any]]) -> bool:
        for record in group:
            if id(record) not in texts:
                texts[id(record)] = item_text(record, settled=settled)
        head, tail = _analyst_frame(ctx, kind, most[kind], most[kind], len(group), pipeline)
        # The items as the request joins them: an instruction an earlier item showed is
        # said once (`evidence.dedup`).
        body = "\n\n".join(dedup([texts[id(record)] for record in group]))
        return len(system(kind)) + len(head) + len(tail) + len(body) <= room

    return fits


# ── what an analyst reads ──────────────────────────────────────────────────


def shown_settled(pipeline: str, records: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """The lines an item's step does not repeat in a request that shows ``pipeline`` (the
    pipeline view and each step's settings as shown, clipping included) and ``records``
    (`base.settled_for`): a step's instructions are said once, in the settings. A short
    setting is a cell of the settings table, so each table cell counts as a line too."""
    from .base import settled_for

    cells = [cell.strip().replace("\\|", "|") for line in pipeline.splitlines()
             if line.startswith("|") for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
    return settled_for([{"value": pipeline}, *({"value": cell} for cell in cells)], (),
                       records)


def item_text(record: dict[str, Any], input_chars: int | None = None,
              output_chars: int | None = None, **shown: Any) -> str:
    """One train item's trace (`optim.evidence.render_item`): the input once, each step's
    own instructions (what its prompt took from the input or earlier replies said by
    name) and its reply in the order they ran, the expected answer, the prediction, the
    score and the grader's feedback. ``input_chars``/``output_chars`` clip a step's own
    instructions/replies head and tail; ``shown`` passes ``label``, ``show_input``,
    ``show_expected`` and ``settled`` (`shown_settled`) on (`evidence.render_item`)."""
    from . import evidence

    big = 10 ** 9
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    missing = [f"No answer from step `{step}`: "
               f"{NO_ANSWER_CAUSES.get(str(info.get('cause')), info.get('cause'))}"
               for step, info in sorted((record.get("missing_steps") or {}).items())
               if isinstance(info, dict)]
    return evidence.render_item(
        record, input_keys=TRAJECTORY_INPUT_KEYS, outcome=_outcome(record),
        expected=_expected(record), expected_label=shown_label(record),
        metadata=metadata_text(record), choices=_choice_lines(meta.get("choices")), prompts=True,
        prompt_chars=input_chars or big, reply_chars=output_chars or big, missing=missing,
        **shown)


#: The most characters one declared metadata value shows in an item's header.
METADATA_CHARS = 200


def metadata_text(record: Mapping[str, Any]) -> str:
    """The item's declared metadata (``record["item_meta"]``, the fields its Task names in
    ``Task.item_metadata``, set by the loop; SkillOpt's analysts see each item's task
    type) as ``name: value`` pairs; empty without any."""
    meta = record.get("item_meta") if isinstance(record.get("item_meta"), Mapping) else {}
    parts = []
    for name, value in meta.items():
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value if v not in (None, ""))
        elif isinstance(value, Mapping):
            value = json.dumps(value, default=str, sort_keys=True)
        text = _clip(value, METADATA_CHARS) if value not in (None, "") else ""
        if text:
            parts.append(f"{name}: {text}")
    return "; ".join(parts)


#: What a clipped item says of itself; ``{room}`` names the request it was clipped for.
_CLIPPED = "(This item's texts are clipped head and tail to fit {room}.)"


def fit_item(record: dict[str, Any], budget: int, room_name: str = "the analyst's room",
             **shown: Any) -> str:
    """``item_text`` whole when it fits ``budget``; else step inputs are clipped first
    (down to ``INPUT_FLOOR``), then outputs, each to the largest limit that fits. The
    cut is said, naming ``room_name``: the request it was clipped to fit."""
    whole = item_text(record, **shown)
    if len(whole) <= budget:
        return whole
    clipped = _CLIPPED.format(room=room_name)
    room = budget - len(clipped) - 1

    def largest(make: Any, low: int) -> int | None:
        if len(make(low)) > room:
            return None
        high = max(low, len(whole))
        while low < high:
            middle = (low + high + 1) // 2
            if len(make(middle)) <= room:
                low = middle
            else:
                high = middle - 1
        return low

    inputs = largest(lambda n: item_text(record, n, None, **shown), INPUT_FLOOR)
    if inputs is not None:
        return item_text(record, inputs, None, **shown) + "\n" + clipped
    # Limits stay at TEXT_FLOOR or more (`_clip_middle` keeps head and tail halves).
    outputs = largest(lambda n: item_text(record, INPUT_FLOOR, n, **shown), TEXT_FLOOR)
    if outputs is not None:
        return item_text(record, INPUT_FLOOR, outputs, **shown) + "\n" + clipped
    both = largest(lambda n: item_text(record, n, n, **shown), TEXT_FLOOR) or TEXT_FLOOR
    return item_text(record, both, both, **shown) + "\n" + clipped


def pipeline_text(ctx: Any, text_chars: int = PIPELINE_TEXT_CHARS) -> str:
    """The pipeline as an analyst reads it (SkillOpt's analyst reads the current skill),
    in the proposer's own words from the same renderers: the pipeline view
    (`base.pipeline_view_lines`) and each step's settings with their values
    (`base._settings_lines`, a paragraph the steps' texts share read once: an analyst never
    patches); each long setting text at most ``text_chars``, head and tail (a call retried
    after a context overflow clips it lower)."""
    from .base import _settings_lines, pipeline_view_lines

    view = pipeline_view_lines(ctx.node_states, ctx.edges, ctx.knobs,
                               getattr(ctx, "protected", ()) or (),
                               getattr(ctx, "graders", ()) or ())
    return "\n".join([*view, "", *_settings_lines(ctx.knobs, text_chars, exact=False)]).rstrip()


def _memory_text(ctx: Any) -> str:
    """The loop's memory note (`optim.memory`), as SkillOpt's analysts and merges read
    its optimizer meta skill; empty without one."""
    from .memory import context_block

    return context_block(getattr(ctx, "memory", None))


def _objective_text(ctx: Any) -> str:
    """"What the score measures" and the compute policy (`base.objective_with_policy`),
    from the run's settings the request carries; empty when they are not known."""
    from .base import objective_with_policy

    text = objective_with_policy(getattr(ctx, "parsimony", None),
                                 item_mean=bool(getattr(ctx, "score_is_item_mean", False)),
                                 gated=getattr(ctx, "selection", "") or False)
    return f"## What the score measures\n{text}\n\n" if text else ""


FAILURE_SYSTEM = """\
You are a failure-analysis agent for a multi-step AI pipeline: a graph of steps (nodes)
connected by edges, each step configured by settings (knobs) such as its prompt.

You are given the pipeline and a batch of items it answered wrong or failed on, each
with its full trace (how a trace reads is said with the items). Find the most
important COMMON failure patterns across the batch.

## Analysis process
1. Read ALL items in the batch, every step of each.
2. Identify the most prevalent, systematic failure patterns across them.
3. For each, find where it starts: a step (node), the data passed along an edge, a
   setting, or the pipeline's structure (a missing, redundant or misordered step, a
   check or loop that is absent, work that should run in parallel).
4. Say how the steps took part, as far as the items show it: the step where things
   first went wrong for these items (in a loop or a long multi-turn run, the first step
   that did), any later step whose output replaced good work from an earlier step with
   worse, and any step whose job was to produce something new (an analysis, an
   attempt, a revision) but whose output repeated another step's output -- an earlier
   step, a parallel branch, or its own earlier attempt in a loop -- without adding to
   it. A check, verify or vote step confirming an answer, or parallel attempts agreeing,
   is not repetition: agreement is what such a step reports.
   Leave out what the items cannot show.
5. Report the COMMON patterns, not individual edge cases.
6. Do not propose changes: say what happens, not what to do.

{claims}

You reply with JSON only.""".format(claims=evidence.CLAIMS)

SUCCESS_SYSTEM = """\
You are a success-pattern analyst for a multi-step AI pipeline: a graph of steps (nodes)
connected by edges, each step configured by settings (knobs) such as its prompt.

You are given the pipeline and a batch of items it answered right, each with its full
trace (how a trace reads is said with the items).
Identify what made them work: the behaviour COMMON across the batch and which step or
part of the structure produces it.

## Rules
- Focus on patterns that appear across MULTIPLE items in the batch.
- Say where each pattern comes from: a step (node), an edge, a setting, or the
  pipeline's structure.
- Patterns must generalise beyond specific items.
- Do not propose changes: say what happens, not what to do.

{claims}

You reply with JSON only.""".format(claims=evidence.CLAIMS)

def system(kind: str) -> str:
    """An analyst call's system prompt."""
    return FAILURE_SYSTEM if kind == "failure" else SUCCESS_SYSTEM


#: How a failure finding says the steps took part (optional; `_steps`).
STEP_ROLES = ("first_wrong", "overrode", "repeated")
_STEPS = ('"steps": {"first_wrong": "<the step where things first went wrong for these '
          'items>", "overrode": ["<a later step whose output replaced good work from an '
          'earlier step with worse>", ...], "repeated": ["<a step meant to produce something '
          'new whose output repeated another step\'s output (an earlier step, a parallel '
          'branch, its own earlier attempt) without adding to it; not a check, vote or '
          'agreeing attempt>", ...]}')

_WHERE = ('"where": {"nodes": ["<step id>", ...], "edges": ["<source> -> <target>", ...], '
          '"knobs": ["<setting name>", ...], "structure": "<what about the pipeline\'s '
          'shape, or empty>"}')

FAILURE_ANSWER = f"""Write one JSON object and nothing else:

{{"batch_size": <number of items you read>,
 "findings": [{{"finding": "<what goes wrong, in one or two sentences>",
               {_WHERE},
               {_STEPS},
               "items": ["<item id>", ...],
               "impact": "high|medium|low",
               "evidence": ["<a short verbatim quote from a trace>", ...]}}]}}

Rules:
- At most {{limit}} findings, the most important first; fewer if warranted, none if
  nothing is common.
- Every finding names the items it covers (ids from this batch) and quotes them.
- "steps" is optional: give the parts the items show, with step ids from the pipeline;
  leave out any part that does not apply or cannot be told from them.
- "impact": how much of the score it costs.
"""

SUCCESS_ANSWER = f"""Write one JSON object and nothing else:

{{"batch_size": <number of items you read>,
 "findings": [{{"finding": "<what the passing items do right, in one or two sentences>",
               {_WHERE},
               "items": ["<item id>", ...],
               "impact": "high|medium|low",
               "evidence": ["<a short verbatim quote from a trace>", ...]}}]}}

Rules:
- At most {{limit}} findings, the most important first; fewer if warranted.
- Every finding names the items it covers (ids from this batch) and quotes them.
- "impact": how much of the score it carries.
"""


def item_cap(lengths: Sequence[int], room: int, floor: int = ITEM_FLOOR) -> int | None:
    """The one length the batch's items are cut to so that together they fit ``room``,
    or ``None`` when they fit whole. Room a short item leaves goes to the long ones:
    the cap is the largest that fits (water-filling), so every clipped item is cut to
    the same length, and never below ``floor`` (``ITEM_FLOOR``)."""
    if sum(lengths) <= room:
        return None
    # Never below the floor: when even the floor does not fit, the search ends there.
    low, high = floor, max(floor, max(lengths))
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(n, middle) for n in lengths) <= room:
            low = middle
        else:
            high = middle - 1
    return low


def round_key(ctx: Any, model: str = "") -> str:
    """What one proposal step's analysis is made from: the epoch, the generation, the
    edited candidate, its train records, the stage's settings, the optimizer's model and
    the call's room.
    Not the memory note: a retry in the same step reuses the step's analysis."""
    import hashlib

    # The room sets the batches (`packer`), so it is part of what the analysis is made
    # from: an analysis batched for another room is not reused.
    return hashlib.sha256(json.dumps(
        {"epoch": ctx.epoch, "generation": ctx.generation, "parent": ctx.parent,
         "records": ctx.item_records, "settings": dict(ctx.analysts), "model": model,
         "room": ANALYST_ROOM},
        sort_keys=True, default=str).encode()).hexdigest()


def analyst_prompt(ctx: Any, kind: str, index: int, total: int,
                   group: Sequence[dict[str, Any]], pipeline: str) -> str:
    """One analyst's request (`analyst_batch`), without its clipping counts."""
    return analyst_batch(ctx, kind, index, total, group, pipeline)[0]


def analyst_batch(ctx: Any, kind: str, index: int, total: int,
                  group: Sequence[dict[str, Any]], pipeline: str,
                  room: int = ANALYST_ROOM) -> tuple[str, dict[str, int]]:
    """One analyst's request -- the pipeline, the optimizer's memory note and the batch's items in full -- and what was clipped:
    ``{"items", "chars"}``. The items share what ``room`` (``ANALYST_ROOM``) leaves after
    the system prompt and the rest of the prompt. A batch made by `packer` fits whole;
    only an item longer than the room on its own (alone in its batch), or a call retried
    within half its length after the provider refused it as too long, is cut: the
    longest items to one length (`item_cap`), step inputs first (`fit_item`), the cut
    counted. SkillOpt never clips."""
    head, tail = _analyst_frame(ctx, kind, index, total, len(group), pipeline)
    settled = shown_settled(pipeline, ctx.item_records)
    wholes = [item_text(record, settled=settled) for record in group]
    separators = 2 * max(0, len(group) - 1)
    cap = item_cap([len(text) for text in wholes],
                   room - len(system(kind)) - len(head) - len(tail) - separators)
    # `fit_item` keeps an item no longer than the cap whole.
    blocks = [text if cap is None else fit_item(record, cap, settled=settled)
              for record, text in zip(group, wholes)]
    clipped = [(whole, block) for whole, block in zip(wholes, blocks) if block != whole]
    from .evidence import dedup

    # A clipped batch (the fallback path) keeps its items at the one length `item_cap`
    # chose; a whole batch says each repeated instruction once.
    return head + "\n\n".join(blocks if cap is not None else dedup(blocks)) + tail, {
        "items": len(clipped),
        "chars": sum(max(0, len(whole) - len(block)) for whole, block in clipped)}


def _analyst_frame(ctx: Any, kind: str, index: int, total: int, count: int,
                   pipeline: str) -> tuple[str, str]:
    """An analyst prompt's text before and after its items."""
    label = "failed" if kind == "failure" else "passed"
    answer = (FAILURE_ANSWER if kind == "failure" else SUCCESS_ANSWER).replace(
        "{limit}", str(FINDINGS_PER_ANALYST))
    return ((f"# Analysing {label} items -- batch {index} of {total}\n\n"
             f"Mode: `{ctx.mode_name}`.\n\n"
             f"## The pipeline\n{pipeline}\n\n"
             f"{_objective_text(ctx)}{_memory_text(ctx)}"
             f"## The {label} items ({count} total)\n"
             f"Each item in full: {evidence.TRACE_FORM}.\n\n"),
            f"\n\n## Answer\n{answer}")


# ── what an analyst wrote ──────────────────────────────────────────────────


def _strings(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


def _where(value: Any) -> dict[str, Any]:
    where = value if isinstance(value, dict) else {}
    out = {key: _strings(where.get(key)) for key in ("nodes", "edges", "knobs")}
    structure = where.get("structure")
    out["structure"] = str(structure).strip() if structure not in (None, "") else ""
    if not isinstance(value, dict) and value not in (None, ""):
        out["structure"] = str(value)
    return out


def _step_ids(value: Any) -> list[str]:
    """Step ids from a ``steps`` part: strings only, stripped, each once; anything else
    (a number, an object) is dropped."""
    listed = value if isinstance(value, (list, tuple)) else [value]
    ids = (v.strip().strip("`").strip() for v in listed if isinstance(v, str))
    return list(dict.fromkeys(v for v in ids if v))


def _steps(value: Any) -> dict[str, Any]:
    """A finding's step attribution normalised (`STEP_ROLES`): ``first_wrong`` one step
    id, ``overrode`` and ``repeated`` lists of step ids, each only when given. Optional:
    a finding without it, or with a malformed one, has ``{}`` (or the parts that read)."""
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    first = _step_ids(value.get("first_wrong"))
    if first:
        out["first_wrong"] = first[0]
    for role in ("overrode", "repeated"):
        ids = _step_ids(value.get(role))
        if ids:
            out[role] = ids
    return out


def finding(raw: Any, kind: str, items: set[str]) -> dict[str, Any] | None:
    """One finding normalised: its ``source``, text, ``where``, its step attribution, the
    cited items checked against ``items`` (an id outside them is
    dropped and kept under ``not_in_batch``), ``support`` (the items it covers),
    ``impact``, ``batches`` and evidence. ``None`` for an entry with no text."""
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("finding") or raw.get("pattern") or "").strip()
    if not text:
        return None
    source = raw.get("source") if raw.get("source") in KINDS else kind
    cited = list(dict.fromkeys(_strings(raw.get("items"))))
    impact = str(raw.get("impact") or "").strip().lower()
    try:
        merged_from = max(1, int(raw.get("batches") or 1))
    except (TypeError, ValueError):
        merged_from = 1
    kept = [i for i in cited if i in items]
    return {"source": source, "finding": text, "where": _where(raw.get("where")),
            "steps": _steps(raw.get("steps")),
            "items": kept, "not_in_batch": [i for i in cited if i not in items],
            "support": len(kept), "impact": impact if impact in IMPACTS else "",
            "batches": merged_from, "evidence": _strings(raw.get("evidence"))[:EVIDENCE_PER_FINDING]}


def parse_findings(payload: Any, kind: str, items: Sequence[str],
                   limit: int | None = FINDINGS_PER_ANALYST) -> list[dict[str, Any]] | None:
    """The findings of an analyst's (or a merge's) answer, or ``None`` when the answer
    carries no ``findings`` list; at most ``limit`` (an analyst's L, as SkillOpt truncates
    an analyst's edits to it)."""
    listed = payload.get("findings") if isinstance(payload, dict) else None
    if not isinstance(listed, list):
        return None
    known = {str(i) for i in items}
    out = [f for f in (finding(raw, kind, known) for raw in listed) if f is not None]
    return out[:limit] if limit else out


# ── merging ─────────────────────────────────────────────────────────────────


MERGE_FAILURE_SYSTEM = """\
You coordinate the analysis of a multi-step AI pipeline's failures. You receive several
lists of findings, each from an analyst who read a different batch of failed items.
Merge them into ONE coherent, de-duplicated list, ranked.

Merge guidelines:
1. Deduplicate: findings about the same problem in the same place become one; keep the
   clearest wording, list the union of their items and keep the best evidence.
2. Resolve conflicts: if findings disagree on the cause, choose the one with stronger
   evidence or combine them.
3. Preserve unique insights: keep every non-redundant finding.
4. Prevalent-pattern bias: findings reported by several lists or covering many items are
   systematic; rank them highest. A finding about one item may be dropped if it is
   item-specific.
5. Rank by how many items a finding covers and its impact, most important first.
6. For each merged finding, "batches" is how many source findings support it (add up
   the "batches" of the findings merged into it).

You reply with JSON only."""

MERGE_SUCCESS_SYSTEM = """\
You coordinate the analysis of what a multi-step AI pipeline does right. You receive
several lists of findings, each from an analyst who read a different batch of passed
items. Merge them into ONE de-duplicated list of what works, ranked.

Merge guidelines:
1. Deduplicate: keep the most general version of similar patterns, list the union of
   their items and keep the best evidence.
2. Prevalent-pattern bias: patterns seen across many items and lists rank highest.
3. Rank by how many items a finding covers and its impact, most important first.
4. For each merged finding, "batches" is how many source findings support it (add up
   the "batches" of the findings merged into it).

You reply with JSON only."""

MERGE_FINAL_SYSTEM = """\
You coordinate the FINAL merge of the analysis of a multi-step AI pipeline. You receive
two merged lists:
1. Failure findings: what goes wrong (high priority).
2. Success findings: what works (lower priority).

Merge guidelines:
1. FAILURE FINDINGS TAKE PRIORITY. Keep every failure finding; when one is about the
   same step or behaviour as a well-supported success finding, say so in its text.
2. Deduplicate: when a failure and a success finding describe the same behaviour, keep
   the failure one.
3. Keep success findings not covered by any failure finding.
4. Rank the whole list by how many items a finding covers and its impact, most important
   first; findings with more "batches" represent broader consensus.
5. Keep each finding's "source" ("failure" or "success"), "items" and "batches".
6. Do not propose changes: say what happens, not what to do.

You reply with JSON only."""

MERGE_ANSWER = f"""Write one JSON object and nothing else:

{{"reasoning": "<the key merge and ranking decisions, briefly>",
 "findings": [{{"source": "failure|success",
               "finding": "<...>",
               {_WHERE},
               {_STEPS},
               "items": ["<item id>", ...],
               "batches": <int>,
               "impact": "high|medium|low",
               "evidence": ["<a short verbatim quote>", ...]}}]}}

The list is ranked: the most important finding first. "steps" is optional: keep it
from the findings merged (each part's union; the first "first_wrong" when they differ).
"""


def _for_merge(found: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings as a merge call reads them (without the checker's bookkeeping)."""
    keys = ("source", "finding", "where", "steps", "items", "batches", "impact", "evidence")
    return [{key: f[key] for key in keys if f.get(key) not in (None, "", [], {})} for f in found]


def merge_prompt(ctx: Any, step: str, lists: Sequence[Sequence[dict[str, Any]]],
                 level: int, pipeline: str) -> str:
    """A merge request: the pipeline (SkillOpt's merges read the current skill), the
    optimizer's memory note and the lists to merge."""
    if step == "final":
        body = (f"## Failure findings\n{json.dumps(_for_merge(lists[0]), indent=1)}\n\n"
                f"## Success findings\n{json.dumps(_for_merge(lists[1]), indent=1)}")
        head = "# Final merge of the analysts' findings"
    else:
        body = "\n\n".join(f"## List {number}\n{json.dumps(_for_merge(found), indent=1)}"
                           for number, found in enumerate(lists, 1))
        head = (f"# Merging the {step} analysts' findings -- {len(lists)} lists, "
                f"merge level {level}")
    return (f"{head}\n\nMode: `{ctx.mode_name}`.\n\n"
            f"## The pipeline\n{pipeline}\n\n{_objective_text(ctx)}{_memory_text(ctx)}"
            f"{body}\n\n## Answer\n{MERGE_ANSWER}")


def fit_merge_prompt(ctx: Any, step: str, lists: Sequence[Sequence[dict[str, Any]]],
                     level: int, room: int) -> str:
    """A merge request within ``room`` characters, for a call retried after a context
    overflow: the pipeline's texts are clipped (head and tail) to the largest length
    that fits, down to ``TEXT_FLOOR``; the findings are kept whole."""
    def make(chars: int) -> str:
        return merge_prompt(ctx, step, lists, level, pipeline_text(ctx, chars))

    low, high = TEXT_FLOOR, PIPELINE_TEXT_CHARS
    while low < high:
        middle = (low + high + 1) // 2
        if len(make(middle)) <= room:
            low = middle
        else:
            high = middle - 1
    return make(low)


def merge_system(step: str) -> str:
    return {"failure": MERGE_FAILURE_SYSTEM, "success": MERGE_SUCCESS_SYSTEM,
            "final": MERGE_FINAL_SYSTEM}[step]


# ── the request section ────────────────────────────────────────────────────


def _where_text(where: Mapping[str, Any], cap: int | None = None) -> str:
    parts = [f"step `{n}`" for n in where.get("nodes") or ()]
    parts += [f"edge `{e}`" for e in where.get("edges") or ()]
    parts += [f"setting `{k}`" for k in where.get("knobs") or ()]
    if where.get("structure"):
        parts.append(f"structure: {_cut(where['structure'], cap)}")
    return "; ".join(parts) or "unclear"


def _cut(text: Any, cap: int | None) -> str:
    """``text`` whole, its line breaks kept (a quoted text keeps its layout); past
    ``cap`` characters, cut there with how much was cut said (``… [cut: N chars]``); of
    several lines, fenced byte-exact (`evidence.quoted`)."""
    text = str(text or "").strip()
    if cap is not None and len(text) > cap:
        text = f"{text[:cap]}… [cut: {len(text) - cap} chars]"
    # A text of several lines goes byte-exact in a fence of its own (`evidence.quoted`):
    # a quoted prompt's "## Rules" reads as no heading.
    return evidence.quoted(text)


def _texts(entry: Mapping[str, Any]) -> list[str]:
    """Every text of a finding the section may have to cut: the finding, where it places
    itself in the structure, and each evidence quote."""
    return [str(entry.get("finding") or "").strip(),
            str((entry.get("where") or {}).get("structure") or "").strip(),
            *(str(q).strip() for q in entry.get("evidence") or ())]


def finding_line(rank: int, entry: Mapping[str, Any], cap: int | None = None) -> str:
    """One ranked finding with every text whole -- the finding, where, how the steps took
    part, all its evidence quotes, line breaks kept (SkillOpt never cuts its
    analysts' edits); each text cut to ``cap`` only when the section must fit."""
    ids = list(entry.get("items") or ())
    id_text = ", ".join(ids[:ITEM_IDS]) + (f" and {len(ids) - ITEM_IDS} more"
                                           if len(ids) > ITEM_IDS else "")
    head = [f"{entry.get('source', 'failure')}",
            f"{len(ids)} item(s)" + (f": {id_text}" if ids else "")]
    if entry.get("impact"):
        head.append(f"impact {entry['impact']}")
    if int(entry.get("batches") or 1) > 1:
        head.append(f"from {entry['batches']} findings")
    roles = steps_text(entry.get("steps")) if entry.get("source", "failure") == "failure" else ""
    action = f" -- steps: {roles}" if roles else ""
    quotes = [f'"{_cut(q, cap)}"' for q in entry.get("evidence") or ()]
    return (f"{rank}. [{'; '.join(head)}] {_cut(entry.get('finding'), cap)} "
            f"-- where: {_where_text(entry.get('where') or {}, cap)}{action}"
            + (" -- evidence: " + " | ".join(quotes) if quotes else ""))


#: How `steps_text` and the step map name each role of `STEP_ROLES`.
_ROLE_WORDS = {"first_wrong": "first went wrong at",
               "overrode": "replaced good work from an earlier step with worse",
               "repeated": "repeated other steps' output without adding to it"}


def steps_text(steps: Any) -> str:
    """A finding's step attribution in words; empty without one."""
    roles = _steps(steps)
    parts = [f"{_ROLE_WORDS[role]}: "
             + ", ".join(f"`{step}`" for step in _step_ids(roles[role]))
             for role in STEP_ROLES if roles.get(role)]
    return "; ".join(parts)


def section_lines(analysis: Mapping[str, Any] | None) -> list[str]:
    """"What the analysts found, ranked", for the proposal request; nothing without an
    analysis."""
    if not analysis:
        return []
    done = [b for b in analysis.get("batches") or () if not b.get("error")]
    by_kind = {kind: [b for b in done if b.get("kind") == kind] for kind in KINDS}
    lines = [
        "## What the analysts found, ranked",
        f"Before this request, analysts read the train items of the candidate being edited "
        f"in full (each step's input and output, the answer and the grader's feedback): "
        f"{sum(len(b['items']) for b in by_kind['failure'])} failed item(s) in "
        f"{len(by_kind['failure'])} batch(es) and "
        f"{sum(len(b['items']) for b in by_kind['success'])} passed item(s) in "
        f"{len(by_kind['success'])} batch(es) of up to {analysis.get('group_size')}. Their "
        "findings were merged and ranked, failures first, by how many items each covers "
        "and its impact. Each says what happens, where in the pipeline, on which items, "
        "with quotes from the traces; none says what to change."]
    found = list(analysis.get("findings") or ())
    if not found:
        lines.append("(The analysts found no common pattern.)")
    used = len("\n".join(lines))
    shown = found[:SHOWN_FINDINGS]
    whole = [finding_line(rank, entry) for rank, entry in enumerate(shown, 1)]
    cap = None
    if used + sum(len(line) + 1 for line in whole) > SECTION_CHARS:
        # Too long whole: the findings' texts share the room left
        # after everything else on their lines (each cut is said), longest cut first.
        texts = [len(text) for entry in shown for text in _texts(entry)]
        fixed = sum(len(line) + 1 for line in whole) - sum(texts)
        marks = len(texts) * len("… [cut: 00000 chars]")
        cap = item_cap(texts, SECTION_CHARS - used - fixed - marks, floor=FINDING_FLOOR)
    for rank, entry in enumerate(shown, 1):
        line = whole[rank - 1] if cap is None else finding_line(rank, entry, cap)
        if used + len(line) + 1 > SECTION_CHARS:
            lines.append(f"({len(found) - rank + 1} more finding(s) not shown: the findings "
                         f"are kept within {SECTION_CHARS} characters)")
            break
        lines.append(line)
        used += len(line) + 1
    else:
        if len(found) > SHOWN_FINDINGS:
            lines.append(f"({len(found) - SHOWN_FINDINGS} lower-ranked finding(s) not shown)")
    for batch in analysis.get("batches") or ():
        if batch.get("error"):
            label = "failed" if batch.get("kind") == "failure" else "passed"
            lines.append(f"(Batch {batch.get('index')} of the {label} items -- items "
                         f"{', '.join(batch.get('items') or ())} -- could not be analysed: "
                         f"{_clip(batch['error'], 300)}; those items are in the evidence "
                         "below.)")
    left = analysis.get("not_analysed") or {}
    if any(left.values()):
        lines.append(f"({left.get('failure', 0)} failed and {left.get('success', 0)} passed "
                     f"item(s) were not analysed: at most {analysis.get('max_batches')} "
                     "analyst batch(es) run per round.)")
    return lines


def journal_entry(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """What the loop journals of one round's analysis (``analyst_findings``)."""
    batches_ = list(analysis.get("batches") or ())
    # Retries and clipping count only calls paid in this step: one answered from the
    # cache (a step remade after a resume) was counted when it was paid.
    paid = [b for b in batches_ if not b.get("cached")]
    return {"key": analysis.get("key"), "candidate": analysis.get("candidate"),
            "failed_items": sum(len(b["items"]) for b in batches_ if b.get("kind") == "failure"),
            "passed_items": sum(len(b["items"]) for b in batches_ if b.get("kind") == "success"),
            "batches": [{**{key: b.get(key) for key in ("kind", "index", "items", "findings",
                                                         "clipped", "clipped_chars",
                                                         "context_retry", "error")},
                         **({"cached": True} if b.get("cached") else {})}
                        for b in batches_],
            "context_retries": (sum(1 for b in paid if b.get("context_retry"))
                                + sum(1 for m in analysis.get("merges") or ()
                                      if m.get("context_retry") and not m.get("cached"))),
            "clipped_items": sum(int(b.get("clipped") or 0) for b in paid),
            "clipped_chars": sum(int(b.get("clipped_chars") or 0) for b in paid),
            "merges": list(analysis.get("merges") or ()),
            "not_analysed": dict(analysis.get("not_analysed") or {}),
            "calls": int(analysis.get("calls") or 0),
            "cached": int(analysis.get("cached") or 0),
            **({"error": analysis["error"]} if analysis.get("error") else {}),
            "findings": _for_merge(analysis.get("findings") or ())}
