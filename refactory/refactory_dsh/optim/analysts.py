"""The analyst stage: SkillOpt's per-round analysis, for the model proposer, over a graph.

``LoopConfig.analysts`` turns it on. Before the model proposer writes a round's
proposal, analysts read every train item of the candidate being edited in full, as
SkillOpt's reflect stage reads its rollouts (``refactory_packages/skillopt/analysts.py``
and ``merge.py`` port SkillOpt's ``reflect.py``, ``aggregate.py`` and ``clip.py``):

- **Failure analysts**: the failed items (every item not passed: answered wrong, no
  answer, or an error), shuffled with the round's seed and cut into batches of
  ``group_size`` (SkillOpt's ``minibatch_size``, 8). One call per batch. Each analyst
  sees the pipeline (its nodes, edges, each node in full and every setting's value, as
  SkillOpt's analyst sees the current skill) and each item's full trace: its input,
  every step's input and output in the order they ran, the expected answer, the
  prediction, the score and the grader's feedback. It writes at most
  ``FINDINGS_PER_ANALYST`` findings (SkillOpt's analyst edit budget L, 4): what goes
  wrong, where in the graph (nodes, edges, settings, the structure) and what change
  would fix it.
- **Success analysts**: the passed items, likewise (seed + 1, as SkillOpt): what made
  them work and what must be kept.
- **Merge**: failure findings merged level by level in batches of ``merge_batch_size``
  (SkillOpt's, 8), then success findings, then a final merge with failure priority
  that returns one de-duplicated list ranked by how many items each covers and its
  impact. SkillOpt's shortcuts hold: one list is not merged, and the final merge runs
  only when both sides found something. An unreadable merge reply keeps every input
  finding (SkillOpt's fallback concatenation).

SkillOpt's patch, slow-update and meta-skill steps are not here: the proposer writes the
edits (it is the step SkillOpt's rank and patch are), and its ``lessons`` carry
optimizer-side memory across rounds. The ranked findings reach the proposal request as
"What the analysts found, ranked" and replace the readers' summaries
(``AutoresearchOptimizer._analyse``).

Domain-free: the prompts name no dataset or task. Train records only: the analysts see
``ProposalContext.item_records`` (the loop hands over train records only) and the graph.
"""

from __future__ import annotations

import json
import random
from typing import Any, Mapping, Sequence

from .base import (
    TRAJECTORY_INPUT_KEYS, _choice_lines, _clip, _clip_middle, _expected, _node_spec,
    _ordered_items, _outcome, _table, NO_ANSWER_CAUSES,
)

#: SkillOpt's defaults (``_base_/default.yaml``: ``minibatch_size: 8``,
#: ``merge_batch_size: 8``), the defaults of ``LoopConfig.analyst_group_size`` and
#: ``analyst_merge_batch_size``.
GROUP_SIZE = 8
MERGE_BATCH_SIZE = 8
#: The most findings one analyst writes: SkillOpt's analyst edit budget L
#: (``analyst_edit_budget``, 4).
FINDINGS_PER_ANALYST = 4
#: One analyst call's whole prompt, in characters. The optimizer model (Opus 4.6) reads
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
#: The least a finding's text or suggested change is cut to when the shown findings do
#: not fit ``SECTION_CHARS`` whole (`section_lines`); each cut says how much it cut.
FINDING_FLOOR = 100
#: The evidence quotes a finding keeps (twice SkillOpt's three, for a merged finding).
QUOTES = 3
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
    them (an error is a failure), failures shuffled with ``seed`` and passes with
    ``seed + 1`` (SkillOpt's ``reflect.py``)."""
    items = sorted(_ordered_items(records), key=lambda r: str(r.get("id")))
    failures = [r for r in items if not r.get("passed")]
    successes = [r for r in items if r.get("passed")]
    random.Random(seed).shuffle(failures)
    random.Random(seed + 1).shuffle(successes)
    return failures, successes


def batches(failures: Sequence[dict[str, Any]], successes: Sequence[dict[str, Any]], *,
            group_size: int, max_batches: int = 0
            ) -> tuple[list[tuple[str, int, list[dict[str, Any]]]], dict[str, int]]:
    """The analyst batches as ``(kind, index, records)``, failure batches first, and how
    many items of each kind no batch holds. ``max_batches`` (0: no limit, SkillOpt's)
    caps the calls: failure batches take the room first, as SkillOpt's merge gives
    failures priority."""
    out: list[tuple[str, int, list[dict[str, Any]]]] = []
    left = {"failure": 0, "success": 0}
    for kind, items in (("failure", failures), ("success", successes)):
        for index, start in enumerate(range(0, len(items), group_size), 1):
            group = list(items[start:start + group_size])
            if max_batches and len(out) >= max_batches:
                left[kind] += len(group)
                continue
            out.append((kind, index, group))
    return out, left


# ── what an analyst reads ──────────────────────────────────────────────────


def item_text(record: dict[str, Any], input_chars: int | None = None,
              output_chars: int | None = None) -> str:
    """One train item's full trace: input, every step's input and output in the order
    they ran, the expected answer, the prediction, the score and the grader's feedback.
    ``input_chars``/``output_chars`` clip step inputs/outputs head and tail."""
    big = 10 ** 9
    ins, outs = input_chars or big, output_chars or big
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    exchanges = [e for e in record.get("exchanges") or () if isinstance(e, dict)]
    lines = [f"### Item {record.get('id')} — {_outcome(record)} (score {record.get('score')})"]
    key = next((k for k in TRAJECTORY_INPUT_KEYS if meta.get(k)), None)
    if key is not None:
        lines += [f"Input ({key}):", _clip_middle(meta[key], outs)]
    choices = _choice_lines(meta.get("choices"))
    if choices:
        lines += ["Choices:", *(_clip_middle(line, outs) for line in choices)]
    for number, exchange in enumerate(exchanges, 1):
        node = exchange.get("node") or "agent"
        lines += [f"#### Step `{node}` ({number} of {len(exchanges)})",
                  "Input it was given:", _clip_middle(exchange.get("prompt"), ins),
                  "Output:", _clip_middle(exchange.get("response"), outs)]
    if not exchanges:
        lines.append("(no step recorded a model call for this item)")
    expected = _expected(record)
    if expected:
        lines.append(f"Expected: {_clip_middle(expected, outs)}")
    if record.get("prediction") is not None:
        lines.append(f"Predicted: {_clip_middle(record.get('prediction'), outs)}")
    # An errored item's grader feedback is often its error again: said once, as the error.
    if (record.get("fail_reason") and not record.get("passed")
            and str(record["fail_reason"]).strip() != str(record.get("error_text") or "").strip()):
        lines.append(f"Grader: {_clip_middle(record['fail_reason'], outs)}")
    for step, info in sorted((record.get("missing_steps") or {}).items()):
        if isinstance(info, dict):
            lines.append(f"No answer from step `{step}`: "
                         f"{NO_ANSWER_CAUSES.get(str(info.get('cause')), info.get('cause'))}")
    if record.get("error_text"):
        lines.append(f"Error: {_clip_middle(record['error_text'], outs)}")
    return "\n".join(lines)


_CLIPPED = "(This item's texts are clipped head and tail to fit the analyst's room.)"


def fit_item(record: dict[str, Any], budget: int) -> str:
    """``item_text`` whole when it fits ``budget``; else step inputs are clipped first
    (down to ``INPUT_FLOOR``), then outputs, each to the largest limit that fits."""
    whole = item_text(record)
    if len(whole) <= budget:
        return whole
    room = budget - len(_CLIPPED) - 1

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

    inputs = largest(lambda n: item_text(record, n, None), INPUT_FLOOR)
    if inputs is not None:
        return item_text(record, inputs, None) + "\n" + _CLIPPED
    # Limits stay at TEXT_FLOOR or more (`_clip_middle` keeps head and tail halves).
    outputs = largest(lambda n: item_text(record, INPUT_FLOOR, n), TEXT_FLOOR)
    if outputs is not None:
        return item_text(record, INPUT_FLOOR, outputs) + "\n" + _CLIPPED
    both = largest(lambda n: item_text(record, n, n), TEXT_FLOOR) or TEXT_FLOOR
    return item_text(record, both, both) + "\n" + _CLIPPED


def pipeline_text(ctx: Any, text_chars: int = PIPELINE_TEXT_CHARS) -> str:
    """The pipeline as an analyst reads it (SkillOpt's analyst reads the current skill):
    its nodes and edges, each node in full with the values of the settings bound to it,
    and every setting's value; each long text at most ``text_chars``, head and tail."""
    knob_values = {k.get("name"): k.get("value") for k in ctx.knobs}
    lines = ["Nodes:",
             _table(["id", "kind", "call", "knobs"],
                    [[n.get("id"), n.get("kind"), n.get("call", ""), n.get("knobs", [])]
                     for n in ctx.nodes]),
             "Edges:",
             _table(["source", "target", "condition"],
                    [[e.get("source"), e.get("target"), e.get("condition")] for e in ctx.edges])]
    if ctx.node_states:
        lines += ["", "Each node in full:"]
        lines += [_node_spec(state, text_chars, knob_values)
                  for state in sorted(ctx.node_states, key=lambda s: str(s.get("id")))]
    bound = {str(pair[1]): f"`{state.get('id')}`.{pair[0]}"
             for state in ctx.node_states for pair in state.get("field_bindings") or ()
             if isinstance(pair, (list, tuple)) and len(pair) == 2}
    if ctx.knobs:
        lines += ["", "Settings (knobs) and their current values:"]
        for knob in ctx.knobs:
            if str(knob.get("name")) in bound:
                lines.append(f"- `{knob.get('name')}` ({knob.get('kind')}): the value of "
                             f"{bound[str(knob.get('name'))]}, shown in the node above")
                continue
            value = knob.get("value")
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            lines += [f"- `{knob.get('name')}` ({knob.get('kind')}"
                      + (", frozen" if knob.get("frozen") else "") + "):",
                      _clip_middle(text, text_chars)]
    return "\n".join(lines)


def _changes_text(ctx: Any) -> str:
    """The kinds of change the optimizer may make, so a finding can say which fits."""
    if not ctx.catalogue:
        return "(not listed)"
    return "\n".join(f"- `{op.get('op')}`: {op.get('description', '')}"
                     for op in ctx.catalogue)


def _lessons_text(ctx: Any) -> str:
    text = str((ctx.lessons or {}).get("text") or "").strip() if ctx.ask_lessons else ""
    if not text:
        return ""
    # SkillOpt's analysts and merges read its optimizer meta skill; the proposer's
    # lessons are this loop's optimizer-side memory.
    return ("## The optimizer's lessons so far\n"
            "Optimizer-side memory from earlier rounds of this run. Use it when the "
            "evidence is ambiguous; do not force it where the items contradict it.\n\n"
            f"{text}\n\n")


FAILURE_SYSTEM = """\
You are a failure-analysis agent for a multi-step AI pipeline: a graph of steps (nodes)
connected by edges, each step configured by settings (knobs) such as its prompt.

You are given the pipeline and a batch of items it answered wrong or failed on, each
with its full trace: every step's input and output, the final answer, the expected
answer and the grader's feedback. Find the most important COMMON failure patterns
across the batch.

## Analysis process
1. Read ALL items in the batch, every step of each.
2. Identify the most prevalent, systematic failure patterns across them.
3. For each, find where it starts: a step (node), the data passed along an edge, a
   setting, or the pipeline's structure (a missing, redundant or misordered step, a
   check or loop that is absent, work that should run in parallel).
4. Say what change to the pipeline would fix it: a prompt or setting change, or a
   structural change (add, remove, rewire, split, loop or parallelise steps).
5. Address the COMMON patterns, not individual edge cases. A change must hold across
   items; do not hardcode item-specific values.
6. Do not ask for what the pipeline already does.

You reply with JSON only."""

SUCCESS_SYSTEM = """\
You are a success-pattern analyst for a multi-step AI pipeline: a graph of steps (nodes)
connected by edges, each step configured by settings (knobs) such as its prompt.

You are given the pipeline and a batch of items it answered right, each with its full
trace: every step's input and output, the final answer and the expected answer.
Identify what made them work: the behaviour COMMON across the batch, which step or
part of the structure produces it, and what a change to the pipeline must preserve so
these items keep passing.

## Rules
- Focus on patterns that appear across MULTIPLE items in the batch.
- Say where each pattern comes from: a step (node), an edge, a setting, or the
  pipeline's structure.
- Patterns must generalise beyond specific items.

You reply with JSON only."""

_WHERE = ('"where": {"nodes": ["<step id>", ...], "edges": ["<source> -> <target>", ...], '
          '"knobs": ["<setting name>", ...], "structure": "<what about the pipeline\'s '
          'shape, or empty>"}')

FAILURE_ANSWER = f"""Write one JSON object and nothing else:

{{"batch_size": <number of items you read>,
 "findings": [{{"finding": "<what goes wrong, in one or two sentences>",
               {_WHERE},
               "change": "<the change to the pipeline that would fix it>",
               "items": ["<item id>", ...],
               "impact": "high|medium|low",
               "evidence": ["<a short verbatim quote from a trace>", ...]}}]}}

Rules:
- At most {{limit}} findings, the most important first; fewer if warranted, none if
  nothing is common.
- Every finding names the items it covers (ids from this batch) and quotes them.
- "impact": how much fixing it would raise the score.
"""

SUCCESS_ANSWER = f"""Write one JSON object and nothing else:

{{"batch_size": <number of items you read>,
 "findings": [{{"finding": "<what the passing items do right, in one or two sentences>",
               {_WHERE},
               "preserve": "<what a change to the pipeline must keep so these items pass>",
               "items": ["<item id>", ...],
               "impact": "high|medium|low",
               "evidence": ["<a short verbatim quote from a trace>", ...]}}]}}

Rules:
- At most {{limit}} findings, the most important first; fewer if warranted.
- Every finding names the items it covers (ids from this batch) and quotes them.
- "impact": how much losing it would lower the score.
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
    edited candidate, its train records, the stage's settings and the optimizer's model.
    Not the lessons: a retry in the same step reuses the step's analysis."""
    import hashlib

    return hashlib.sha256(json.dumps(
        {"epoch": ctx.epoch, "generation": ctx.generation, "parent": ctx.parent,
         "records": ctx.item_records, "settings": dict(ctx.analysts), "model": model},
        sort_keys=True, default=str).encode()).hexdigest()


def analyst_prompt(ctx: Any, kind: str, index: int, total: int,
                   group: Sequence[dict[str, Any]], pipeline: str) -> str:
    """One analyst's request (`analyst_batch`), without its clipping counts."""
    return analyst_batch(ctx, kind, index, total, group, pipeline)[0]


def analyst_batch(ctx: Any, kind: str, index: int, total: int,
                  group: Sequence[dict[str, Any]], pipeline: str,
                  room: int = ANALYST_ROOM) -> tuple[str, dict[str, int]]:
    """One analyst's request -- the pipeline, the changes possible, the optimizer's
    lessons and the batch's items in full -- and what was clipped:
    ``{"items", "chars"}``. The items share what ``room`` (``ANALYST_ROOM``) leaves after
    the rest of the prompt; only when together they do not fit are the longest cut, all to one
    length (`item_cap`), step inputs first (`fit_item`). SkillOpt never clips."""
    head, tail = _analyst_frame(ctx, kind, index, total, len(group), pipeline)
    wholes = [item_text(record) for record in group]
    separators = 2 * max(0, len(group) - 1)
    cap = item_cap([len(text) for text in wholes],
                   room - len(head) - len(tail) - separators)
    # `fit_item` keeps an item no longer than the cap whole.
    blocks = [text if cap is None else fit_item(record, cap)
              for record, text in zip(group, wholes)]
    clipped = [(whole, block) for whole, block in zip(wholes, blocks) if block != whole]
    return head + "\n\n".join(blocks) + tail, {
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
             f"## The changes the optimizer can make\n{_changes_text(ctx)}\n\n"
             f"{_lessons_text(ctx)}"
             f"## The {label} items ({count} total)\n"
             "Each item in full: its input, every step's input and output in the order the "
             "steps ran, the expected answer, the prediction, the score and the grader's "
             "feedback.\n\n"),
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


def finding(raw: Any, kind: str, items: set[str]) -> dict[str, Any] | None:
    """One finding normalised: its ``source``, text, ``where``, change (failure) or what
    to keep (success), the cited items checked against ``items`` (an id outside them is
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
            "change": str(raw.get("change") or "").strip(),
            "preserve": str(raw.get("preserve") or "").strip(),
            "items": kept, "not_in_batch": [i for i in cited if i not in items],
            "support": len(kept), "impact": impact if impact in IMPACTS else "",
            "batches": merged_from, "evidence": _strings(raw.get("evidence"))[:QUOTES * 2]}


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
2. Resolve conflicts: if findings disagree on the cause or the change, choose the one
   with stronger evidence or combine them.
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
items. Merge them into ONE de-duplicated list of what works and must be preserved,
ranked.

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
1. Failure findings: what goes wrong and the change that would fix it (high priority).
2. Success findings: what works and must be preserved (lower priority).

Merge guidelines:
1. FAILURE FINDINGS TAKE PRIORITY: the goal is to fix failures. Keep the failure findings
   unless one directly conflicts with a well-supported success finding; then say, in its
   change, what it must preserve.
2. Deduplicate: when a failure and a success finding cover the same point, keep the
   failure one and fold what must be preserved into it.
3. Preserve success insights not covered by any failure finding.
4. Rank the whole list by how many items a finding covers and its impact, most important
   first; findings with more "batches" represent broader consensus.
5. Keep each finding's "source" ("failure" or "success"), "items" and "batches".

You reply with JSON only."""

MERGE_ANSWER = f"""Write one JSON object and nothing else:

{{"reasoning": "<the key merge and ranking decisions, briefly>",
 "findings": [{{"source": "failure|success",
               "finding": "<...>",
               {_WHERE},
               "change": "<failure: the change that would fix it>",
               "preserve": "<success: what must be kept>",
               "items": ["<item id>", ...],
               "batches": <int>,
               "impact": "high|medium|low",
               "evidence": ["<a short verbatim quote>", ...]}}]}}

The list is ranked: the most important finding first.
"""


def _for_merge(found: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings as a merge call reads them (without the checker's bookkeeping)."""
    keys = ("source", "finding", "where", "change", "preserve", "items", "batches",
            "impact", "evidence")
    return [{key: f[key] for key in keys if f.get(key) not in (None, "", [])} for f in found]


def merge_prompt(ctx: Any, step: str, lists: Sequence[Sequence[dict[str, Any]]],
                 level: int, pipeline: str) -> str:
    """A merge request: the pipeline (SkillOpt's merges read the current skill), the
    optimizer's lessons and the lists to merge."""
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
            f"## The pipeline\n{pipeline}\n\n{_lessons_text(ctx)}"
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
    """``text`` whole, its line breaks kept (a proposed prompt keeps its layout); past
    ``cap`` characters, cut there with how much was cut said (``… [cut: N chars]``)."""
    text = str(text or "").strip()
    if cap is None or len(text) <= cap:
        return text
    return f"{text[:cap]}… [cut: {len(text) - cap} chars]"


def _actions(entry: Mapping[str, Any]) -> list[tuple[str, str]]:
    """A finding's advice, each part it carries: what to change and what to keep, the
    one its side leads with first (a success finding's keep, a failure's change)."""
    parts = [("change", str(entry.get("change") or "").strip()),
             ("keep", str(entry.get("preserve") or "").strip())]
    if entry.get("source") == "success":
        parts.reverse()
    return [(label, text) for label, text in parts if text]


def _texts(entry: Mapping[str, Any]) -> list[str]:
    """Every text of a finding the section may have to cut: the finding, its advice,
    where it places itself in the structure, and each evidence quote."""
    return [str(entry.get("finding") or "").strip(),
            *(text for _label, text in _actions(entry)),
            str((entry.get("where") or {}).get("structure") or "").strip(),
            *(str(q).strip() for q in entry.get("evidence") or ())]


def finding_line(rank: int, entry: Mapping[str, Any], cap: int | None = None) -> str:
    """One ranked finding with every text whole -- the finding, what to change and what
    to keep, where, all its evidence quotes, line breaks kept (SkillOpt never cuts its
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
    action = "".join(f" -- {label}: {_cut(text, cap)}" for label, text in _actions(entry))
    quotes = [f'"{_cut(q, cap)}"' for q in entry.get("evidence") or ()]
    return (f"{rank}. [{'; '.join(head)}] {_cut(entry.get('finding'), cap)} "
            f"-- where: {_where_text(entry.get('where') or {}, cap)}{action}"
            + (" -- evidence: " + " | ".join(quotes) if quotes else ""))


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
        "and its impact. Each says where in the pipeline it starts and what change would "
        "fix it (or, for what works, what to keep). Choose your change from them and the "
        "evidence below; the ranking is advice, not a rule."]
    found = list(analysis.get("findings") or ())
    if not found:
        lines.append("(The analysts found no common pattern.)")
    used = len("\n".join(lines))
    shown = found[:SHOWN_FINDINGS]
    whole = [finding_line(rank, entry) for rank, entry in enumerate(shown, 1)]
    cap = None
    if used + sum(len(line) + 1 for line in whole) > SECTION_CHARS:
        # Too long whole: the findings' texts and suggested changes share the room left
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
