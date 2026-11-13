"""The memory pass: SkillOpt's optimizer meta skill, for the model proposer, over a graph.

Once per epoch, from epoch 2, before the epoch's proposal, one call of the optimizer's
model compares two rollouts on the same train items and writes ONE memory note: an
optimizer-only note on how to analyse, merge and propose changes in this run. It is
the loop's cross-round memory, shown to every analyst call, every merge call and the
proposal request (`format_context`, `request_lines`). It never edits the graph.

SkillOpt's mechanism (``skillopt/optimizer/meta_skill.py``, ``slow_update.py``, and
``skillopt/engine/trainer.py:2088-2199``), mirrored here:

- **When**: at the end of every epoch from the second (trainer.py:2097-2103 skips the
  first); the note it writes is read in the next epoch (trainer.py:1190-1194). Our
  loop's epoch ``e`` scores and gates the proposals epoch ``e - 1`` wrote and then
  proposes, so its pass runs after the gate and before the proposal: the note is read
  by the proposal it precedes and by every later one until the next pass.
- **Which two pipelines**: the previous epoch's last-step skill against this epoch's
  (trainer.py:2111-2114, 2176-2180): here the candidate that was current when the
  previous epoch proposed and the one current now (the validation gate's current, or
  without the gate the train leader). When they are one candidate (nothing kept this
  epoch), SkillOpt rolls the same skill out twice; here the two rollouts are that
  candidate's previous-epoch rollout and this epoch's under the gate, and without it its
  rollout when it was scored and a fresh one made for the pass.
- **Which items**: ``slow_update_samples`` (20, default.yaml:99) train items, the train
  ids shuffled with ``seed + epoch * 2000`` (trainer.py:2118-2119, as the slow update's
  sample, :1843-1844). Rolled out with the epoch's own train seed rather than SkillOpt's
  ``seed + epoch * 2000``, so the loop's own rollouts of the epoch are reused; only the
  sampled items a side has no record of are evaluated.
- **What is shown**: both pipelines, the previous note with "reflect on whether it
  improved or harmed" (meta_skill.py:51-60), and the comparison by category
  (slow_update.py:287-349): a summary, then regressions (highest priority),
  persistent failures, improvements, each item with both outcomes, answers, grader
  reasons and both full traces, then stable successes without traces. SkillOpt never
  clips; here the traces share ``ANALYST_ROOM`` (clipped head and tail, each cut said)
  and a call refused as longer than the model's context is retried once at half size.
- **What it writes**: ``{"reasoning", "memory"}`` (``meta_skill_content``), the prompt
  being SkillOpt's ``meta_skill.md`` for a graph. SkillOpt sets no length; its real runs
  wrote 3.4-5.8k characters, so the note is asked for and kept within ``MEMORY_CHARS``.
- **How it is shown**: SkillOpt's ``format_meta_skill_context`` wording (meta_skill.py
  18-30), to analysts, merges and the proposal request.

SkillOpt's slow update (which edits the skill) is not copied. Where SkillOpt leaves no
note after a failed call (trainer.py:384-397), the loop keeps the previous note and
marks it not updated in that epoch, and why.

Domain-free: the prompts name no dataset or task. Train records only.
"""

from __future__ import annotations

import json
import random
from typing import Any, Mapping, Sequence

from .analysts import ANALYST_ROOM, fit_item, item_cap, item_text
from .base import TRAJECTORY_INPUT_KEYS, _by_item, _clip, _clip_middle, _table

#: SkillOpt's ``slow_update_samples`` (``_base_/default.yaml:99``): the train items the
#: two rollouts are compared on.
MEMORY_SAMPLES = 20
#: The most characters of the note kept and shown. SkillOpt sets no length; its real
#: runs' meta skills were 3.4-5.8k characters.
MEMORY_CHARS = 6000
#: The memory call's whole prompt, system prompt included (the analysts' room).
MEMORY_ROOM = ANALYST_ROOM
#: The categories in the order SkillOpt lists them (slow_update.py:308-313), whether
#: each shows the two traces, and its heading.
CATEGORIES = (
    ("regressed", True, "Regressions (right -> wrong) -- HIGHEST PRIORITY"),
    ("persistent_fail", True, "Persistent failures (wrong -> wrong)"),
    ("improved", True, "Improvements (wrong -> right)"),
    ("stable_success", False, "Stable successes (right -> right)"),
)
#: A pipeline text or setting value in the memory request, head and tail.
PIPELINE_TEXT_CHARS = 20000


def sample_ids(train_ids: Sequence[str], seed: int, epoch: int,
               n: int = MEMORY_SAMPLES) -> list[str]:
    """The pass's train items: the train ids (sorted) shuffled with ``seed + epoch *
    2000``, the first ``n`` (trainer.py:1843-1844, :2118-2119)."""
    ids = sorted(str(item) for item in train_ids)
    random.Random(seed + epoch * 2000).shuffle(ids)
    return ids[:max(0, n)]


def category(prev: Mapping[str, Any], curr: Mapping[str, Any]) -> str:
    """SkillOpt's category of one item (slow_update.py:236-243), from ``passed``."""
    before, after = bool(prev.get("passed")), bool(curr.get("passed"))
    if not before and after:
        return "improved"
    if before and not after:
        return "regressed"
    return "persistent_fail" if not before else "stable_success"


def pairs(prev_records: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
          curr_records: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
          ids: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """One entry per sampled item both rollouts hold (its first repeat on each side):
    ``{"id", "category", "prev", "curr"}``, in sample order; and the sampled items
    either side lacks."""
    def by_item(records: Any) -> dict[str, Mapping[str, Any]]:
        if isinstance(records, Mapping):
            return {str(k): v for k, v in records.items()}
        return _by_item(list(records or ()))

    before, after = by_item(prev_records), by_item(curr_records)
    out, missing = [], []
    for item in ids:
        if item in before and item in after:
            out.append({"id": item, "category": category(before[item], after[item]),
                        "prev": dict(before[item]), "curr": dict(after[item])})
        else:
            missing.append(item)
    return out, missing


def counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many items fall in each category."""
    return {name: sum(1 for e in entries if e.get("category") == name)
            for name, _traces, _label in CATEGORIES}


# ── the request ─────────────────────────────────────────────────────────────


MEMORY_SYSTEM = """\
You are an optimizer-coach for a system that optimizes an AI pipeline: a graph of steps
(nodes) connected by edges, each step configured by settings (knobs) such as its prompt.

Your job is not to solve tasks directly and not to write instructions for the pipeline.
Your job is to write a compact OPTIMIZER-SIDE memory that helps future optimizer calls
produce better changes to the pipeline in this run.

## What You Receive

1. The pipeline at the start of this epoch.
2. The pipeline now, and what changed between them.
3. A longitudinal comparison on the SAME sampled items under those two pipelines.
4. The previous optimizer memory, if one existed.

## Your Goal

Write a concise memory that improves future optimizer behavior in stages such as
failure analysis, success analysis, merging findings, and proposing and ranking
changes.

This memory should capture things like:
- Which kinds of changes tend to help in this pipeline: prompt or setting changes, new
  steps, code steps, checks, loops, parallel steps, removals, rewiring.
- Which kinds of changes tend to be too vague, redundant, brittle, or harmful.
- What level of abstraction works best for instructions here.
- What failure-repair patterns should be prioritized.
- What regression risks future optimizer calls should guard against.

## Important Constraints

- Address the FUTURE OPTIMIZER directly, not the pipeline.
- Focus on how to choose and write better changes.
- Use evidence from the comparison, not generic advice.
- Keep it compact and high-signal. Prefer a few durable principles.
- Revise or remove parts of the previous memory if they did not help.
- Do not output instructions for the pipeline's own steps.
- Do not restate the whole pipeline; summarize editing strategy.

You reply with JSON only."""


def answer_text(chars: int = MEMORY_CHARS) -> str:
    return ("Respond ONLY with a valid JSON object:\n"
            "{\n"
            '  "reasoning": "<brief reflection on what changes helped or hurt>",\n'
            '  "memory": "<compact optimizer-side guidance for future analysis, merging '
            f'and proposals, at most {chars} characters>"\n'
            "}")


def parse_note(payload: Any) -> dict[str, str] | None:
    """The answer's note within ``MEMORY_CHARS`` (cut with the cut said) and its
    reasoning, or ``None`` when it carries no note (meta_skill.py:70-75)."""
    if not isinstance(payload, Mapping):
        return None
    note = payload.get("memory", payload.get("meta_skill_content"))
    note = str(note).strip() if note not in (None, "") else ""
    if not note:
        return None
    if len(note) > MEMORY_CHARS:
        cut = len(note) - MEMORY_CHARS
        note = f"{note[:MEMORY_CHARS]}… [cut: {cut} chars]"
    return {"text": note, "reasoning": str(payload.get("reasoning") or "").strip()}


def _answer(record: Mapping[str, Any]) -> str:
    value = record.get("prediction")
    return "N/A" if value in (None, "") else _clip(value, 300)


def _side_line(label: str, record: Mapping[str, Any]) -> str:
    return (f"- {label}: {'PASS' if record.get('passed') else 'FAIL'} "
            f"(score {record.get('score')}) -- answer: {_answer(record)}")


def _reason(record: Mapping[str, Any]) -> str:
    return str(record.get("fail_reason") or record.get("error_text") or "").strip()


def build_prompt(*, epoch: int, mode_name: str, prev: Mapping[str, Any],
                 curr: Mapping[str, Any], changes: Sequence[str], previous_note: str,
                 entries: Sequence[Mapping[str, Any]], room: int = MEMORY_ROOM,
                 system: str = MEMORY_SYSTEM) -> tuple[str, dict[str, int]]:
    """The memory request within ``room`` characters (the system prompt counted), and
    what was clipped: ``{"items", "chars"}``.

    ``prev``/``curr`` describe the two sides: ``{"id", "pipeline", "rollout"}`` (the
    candidate, its pipeline as an analyst reads it, which rollout its records are).
    ``changes`` are the lines saying what changed between them. Every trace is whole
    when they all fit; otherwise the longest are cut to one length (`item_cap`), step
    inputs first (`fit_item`), each cut said."""
    same = prev.get("id") == curr.get("id")
    shown = [e for e in entries if dict((n, t) for n, t, _ in CATEGORIES)[e["category"]]]
    head, tail = _frame(epoch=epoch, mode_name=mode_name, prev=prev, curr=curr, same=same,
                        changes=changes, previous_note=previous_note, entries=entries)
    wholes = {(e["id"], side): item_text(e[side]) for e in shown for side in ("prev", "curr")}
    blank = _comparison(entries, {key: "" for key in wholes}, same)
    fixed = len(system) + len(head) + len(blank) + len(tail)
    cap = item_cap([len(text) for text in wholes.values()], max(0, room - fixed))
    texts = {key: (text if cap is None else fit_item(
        next(e for e in shown if e["id"] == key[0])[key[1]], cap))
        for key, text in wholes.items()}
    clipped = [(wholes[key], texts[key]) for key in wholes if texts[key] != wholes[key]]
    return head + _comparison(entries, texts, same) + tail, {
        "items": len(clipped),
        "chars": sum(max(0, len(whole) - len(text)) for whole, text in clipped)}


def _frame(*, epoch: int, mode_name: str, prev: Mapping[str, Any], curr: Mapping[str, Any],
           same: bool, changes: Sequence[str], previous_note: str,
           entries: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    if same:
        pipelines = (f"## The pipeline (candidate `{curr.get('id')}`)\n"
                     "Nothing this epoch replaced it: the comparison below is two rollouts of "
                     f"this one pipeline ({prev.get('rollout')}, and {curr.get('rollout')}), so "
                     "what moved between them moved without any change to the pipeline.\n\n"
                     f"{curr.get('pipeline')}\n\n")
    else:
        pipelines = (f"## The pipeline at the start of this epoch (candidate `{prev.get('id')}`)\n"
                     f"{prev.get('pipeline')}\n\n"
                     f"## The pipeline now (candidate `{curr.get('id')}`)\n"
                     f"{curr.get('pipeline')}\n\n"
                     "## What changed between them\n"
                     + ("\n".join(changes) if changes else "(not known)") + "\n\n")
    note = previous_note.strip() or "(No previous optimizer memory -- this is the first update.)"
    head = (f"# Writing the optimizer's memory -- epoch {epoch}\n\n"
            f"Mode: `{mode_name}`.\n\n{pipelines}"
            "## Previous optimizer memory\n"
            "The following optimizer memory was available during this epoch. Reflect on "
            "whether it improved or harmed the quality of the changes.\n\n"
            f"{note}\n\n"
            f"## Longitudinal comparison (the same {len(entries)} train items, "
            + ("two rollouts" if same else "two pipelines") + ")\n"
            f"Start of the epoch: {prev.get('rollout')}. Now: {curr.get('rollout')}. Each "
            "trace is the item in full: its input, every step's input and output in the "
            "order the steps ran, the expected answer, the prediction and the grader's "
            "feedback.\n\n")
    return head, f"\n\n## Answer\n{answer_text()}"


def _comparison(entries: Sequence[Mapping[str, Any]], texts: Mapping[tuple[str, str], str],
                same: bool) -> str:
    """SkillOpt's ``format_comparison_text`` (slow_update.py:287-349) over the pairs."""
    tally = counts(entries)
    parts = [f"### Summary\nTotal items: {len(entries)}\n"
             f"- Improved (wrong -> right): {tally['improved']}\n"
             f"- Regressed (right -> wrong): {tally['regressed']}\n"
             f"- Persistent failures (wrong -> wrong): {tally['persistent_fail']}\n"
             f"- Stable successes (right -> right): {tally['stable_success']}"]
    before, after = ("First rollout", "Second rollout") if same else ("Start of epoch", "Now")
    for name, traces, label in CATEGORIES:
        listed = [e for e in entries if e["category"] == name]
        if not listed:
            parts.append(f"### {label}\n(none)")
            continue
        lines = [f"### {label}"]
        for e in listed:
            prev, curr = e["prev"], e["curr"]
            lines += ["", f"#### Task {e['id']}: {_task_text(curr, prev, e['id'])}"
                          f"{_meta_text(curr)}",
                      _side_line(before, prev), _side_line(after, curr)]
            if _reason(curr) and not curr.get("passed"):
                lines.append(f"- Reason now: {_clip(_reason(curr), 600)}")
            if _reason(prev) and not prev.get("passed"):
                lines.append(f"- Reason at the start: {_clip(_reason(prev), 600)}")
            if traces:
                lines += ["", f"**Trace ({before.lower()}):**", texts.get((e["id"], "prev"), ""),
                          "", f"**Trace ({after.lower()}):**", texts.get((e["id"], "curr"), "")]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _task_text(curr: Mapping[str, Any], prev: Mapping[str, Any], item: str) -> str:
    """What the item asks, as SkillOpt heads each comparison entry (slow_update.py:247,
    :330): the question (or task description, or instruction) from its metadata, else
    its id."""
    for record in (curr, prev):
        meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        for key in ("question", "task_description", "instruction", *TRAJECTORY_INPUT_KEYS):
            if meta.get(key):
                return str(meta[key]).strip()
    return item


def _meta_text(record: Mapping[str, Any]) -> str:
    """The item's declared metadata (``item_meta``), for its heading."""
    from .analysts import metadata_text

    text = metadata_text(record)
    return f" -- {text}" if text else ""


def changes_lines(lineage: Mapping[str, Any]) -> list[str]:
    """What separates the start-of-epoch candidate from the current one (``{"common",
    "prev", "curr"}``, `Loop._lineage_edits`): from their common ancestor, each
    candidate on the way to either with its own edits, oldest first; said plainly when
    they are on different branches or share no ancestor."""
    def lines(path: Sequence[Mapping[str, Any]]) -> list[str]:
        out = []
        for step in path:
            edits = json.dumps(step.get("edits") or [], default=str)
            out.append(f"- `{step.get('id')}` (from `{step.get('parent')}`): "
                       f"{_clip_middle(edits, 4000)}"
                       + (f" -- rationale: {_clip(step.get('rationale'), 400)}"
                          if step.get("rationale") else ""))
        return out

    common, before, after = lineage.get("common"), list(lineage.get("prev") or ()), list(
        lineage.get("curr") or ())
    if common is None:
        return ["They share no ancestor in this run. The start-of-epoch candidate's lineage:",
                *lines(before), "The current candidate's lineage:", *lines(after)]
    if not before:
        return [f"The current candidate descends from the start-of-epoch one; the changes "
                "since, oldest first:", *lines(after)]
    return [f"They are on different branches from their common ancestor `{common}`. "
            "From it to the start-of-epoch candidate:", *lines(before),
            "From it to the current candidate:", *(lines(after) or ["(none: it is the "
                                                                   "common ancestor)"])]


def state_view(state: Mapping[str, Any]) -> Any:
    """A candidate's state in the shape `analysts.pipeline_text` reads (``nodes``,
    ``edges``, ``node_states``, ``knobs``), straight from the state: no mode is built."""
    from types import SimpleNamespace

    graph = (state or {}).get("graph") or {}
    nodes = [dict(node) for node in (graph.get("nodes") or {}).values() if isinstance(node, Mapping)]
    knobs = []
    for name, entry in sorted(((state or {}).get("knobs") or {}).items()):
        entry = entry if isinstance(entry, Mapping) else {"value": entry}
        spec = entry.get("spec") if isinstance(entry.get("spec"), Mapping) else {}
        knobs.append({"name": name, "value": entry.get("value"),
                      "kind": spec.get("kind") or type(entry.get("value")).__name__,
                      "frozen": bool(spec.get("frozen"))})
    return SimpleNamespace(
        nodes=[{"id": n.get("id"), "kind": n.get("kind"), "call": n.get("call", ""),
                "knobs": [*(n.get("knobs") or ()),
                          *(pair[1] for pair in n.get("field_bindings") or ()
                            if isinstance(pair, (list, tuple)) and len(pair) == 2)]}
               for n in nodes],
        edges=list(graph.get("edges") or ()), node_states=nodes, knobs=knobs)


# ── what the optimizer is shown ────────────────────────────────────────────


def context_block(memory: Mapping[str, Any] | None) -> str:
    """The note for an analyst or merge call (SkillOpt's ``format_meta_skill_context``,
    meta_skill.py:18-30); empty without one."""
    text = str((memory or {}).get("text") or "").strip()
    if not text:
        return ""
    return ("## Optimizer memory\n"
            "This is optimizer-side memory distilled from earlier epochs of this run. Use it "
            "to improve how you analyse, merge and rank findings. Prefer it when the current "
            "evidence is ambiguous, but do not force it if the current traces clearly "
            f"contradict it.{_stale_text(memory)}\n\n{text}\n\n")


def _stale_text(memory: Mapping[str, Any] | None) -> str:
    stale = (memory or {}).get("not_updated") or {}
    if not stale:
        return ""
    return (f" It was NOT UPDATED in epoch {stale.get('epoch', '?')} ("
            f"{_clip(stale.get('why'), 300)}): this is the note written in epoch "
            f"{(memory or {}).get('epoch', '?')}, unchanged.")


def request_lines(memory: Mapping[str, Any] | None) -> list[str]:
    """"Optimizer memory" for the proposal request: the latest note, when it was written
    and, when a later pass failed (``not_updated``), that it was not updated then and
    why; nothing before any pass wrote one or failed."""
    memory = memory or {}
    text = str(memory.get("text") or "").strip()
    stale = memory.get("not_updated") or {}
    if not text and not stale:
        return []
    lines = ["## Optimizer memory"]
    if text:
        lines.append(f"Optimizer-side memory written in epoch {memory.get('epoch', '?')} from a "
                     "comparison of the same train items under the pipeline at the start of "
                     "that epoch and after it. Use it to improve how you choose and write "
                     "changes. Prefer it when the current evidence is ambiguous, but do not "
                     "force it if the current traces clearly contradict it.")
    if stale:
        lines.append(f"NOT UPDATED in epoch {stale.get('epoch', '?')}: the memory pass failed "
                     f"({_clip(stale.get('why'), 300)})"
                     + ("; this is the earlier note, unchanged." if text
                        else "; there is no note yet."))
    return [*lines, *([text] if text else [])]


# ── the journal and the report ──────────────────────────────────────────────


def report(passes: Sequence[Mapping[str, Any]], memory: Mapping[str, Any] | None
           ) -> dict[str, Any]:
    """What report.json says of the memory: each epoch's pass (without the note's text,
    which is in the journal) and the latest note."""
    keys = ("epoch", "prev", "curr", "same", "sample", "pairs", "missing", "counts",
            "evaluated", "rollouts", "calls", "context_retry", "clipped_items",
            "clipped_chars", "chars", "updated", "error", "skipped")
    return {"passes": [{key: p.get(key) for key in keys if key in p} for p in passes],
            "updated": sum(1 for p in passes if p.get("updated")),
            "failed": sum(1 for p in passes if p.get("error")),
            "evaluations": sum(int(p.get("evaluated") or 0) for p in passes),
            "rollouts": sum(int(p.get("rollouts") or 0) for p in passes),
            "latest": dict(memory or {}) or None}


def report_lines(rep: Mapping[str, Any] | None) -> list[str]:
    """report.md's "Optimizer memory" section."""
    if not rep or not rep.get("passes"):
        return []
    rows = []
    for p in rep["passes"]:
        tally = p.get("counts") or {}
        rows.append([p.get("epoch"), p.get("prev"), p.get("curr"),
                     f"{p.get('pairs', 0)} of {len(p.get('sample') or ())}",
                     "/".join(str(tally.get(name, 0)) for name, _t, _l in CATEGORIES),
                     p.get("evaluated", 0),
                     "written" if p.get("updated") else
                     f"not updated: {_clip(p.get('error') or p.get('skipped'), 120)}",
                     p.get("chars", "")])
    lines = ["## Optimizer memory", "",
             f"The memory pass ran in {len(rep['passes'])} epoch(s): {rep.get('updated', 0)} "
             f"note(s) written, {rep.get('failed', 0)} failed (the earlier note kept, marked "
             f"not updated); {rep.get('evaluations', 0)} extra evaluation(s) "
             f"({rep.get('rollouts', 0)} rollout(s)) for sampled items no rollout of the "
             "epoch held. Each pass is in journal.jsonl (`optimizer_memory`) and its call "
             "in optimizer_trace.jsonl (`memory`).", "",
             _table(["epoch", "start of epoch", "now", "items compared",
                     "regressed/still failing/improved/stable", "extra evaluations",
                     "note", "chars"], rows), ""]
    latest = rep.get("latest") or {}
    if latest.get("text"):
        lines += [f"The latest note (epoch {latest.get('epoch')}):", "",
                  *(f"> {line}" if line else ">" for line in str(latest["text"]).splitlines()),
                  ""]
    return lines
