"""The memory pass: SkillOpt's optimizer meta skill, for the model proposer, over a graph.

Once per epoch, from epoch 2, before the epoch's proposal, the optimizer's model
compares two rollouts on the same train items and writes ONE memory note: an
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
- **What is shown**: both pipelines, what the score measures (`base.objective_text`),
  every change the optimizer tried in the epoch -- evaluated, with its train fixed/broken
  counts and the gate's word, or refused before it ran, with why (`tried_lines`; the facts
  the proposal request already shows, never a validation score) -- so a pass with
  nothing kept still sees what was tried; the previous note with "reflect on whether it
  improved or harmed" (meta_skill.py:51-60), and the comparison by category
  (slow_update.py:287-349): a summary, then regressions (highest priority),
  persistent failures, improvements, each item with both outcomes, answers, grader
  reasons and both full traces, then stable successes without traces. SkillOpt never
  clips, and neither does this pass: when the whole request fits ``MEMORY_ROOM`` (the
  analysts' room, tied to the optimizer model's context) it is one call, SkillOpt's;
  when it does not, the evidence is split (`MemoryRequest`): comparison calls each read
  the pipelines and as many whole item pairs as fit and write observations for the
  optimizer (``COMPARE_SYSTEM``), then one synthesis call reads both pipelines, the tries, the
  previous note and every part's observations and writes the note
  (``SYNTHESIS_SYSTEM``). A larger room makes fewer calls; no trace is cut, except a
  pair longer than the room on its own (counted). A call refused as longer than the
  model's context is retried once at half size.
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

from .analysts import ANALYST_ROOM, TEXT_FLOOR, fit_item, item_cap, item_text
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
#: The most characters "What was tried this epoch" takes: its entries newest first, the
#: older ones past it cut with the cut said. And the most one entry's edits and one
#: refusal's reason take.
TRIED_CHARS = 8000
TRIED_EDITS_CHARS = 300
TRIED_REASON_CHARS = 300
#: "What was tried this epoch" when the round proposed nothing.
NOTHING_TRIED = "(nothing was proposed in that round)"
#: How a clipped trace names this request (`analysts.fit_item`).
ROOM_NAME = "the memory request's room"
#: The most observations one comparison call writes, and the characters they are asked
#: to stay within (asked, not cut: the synthesis reads them whole).
OBSERVATIONS_PER_CALL = 8
OBSERVATION_CHARS = 6000
#: The quotes an observation keeps.
EVIDENCE_PER_OBSERVATION = 4
#: The most characters one comparison call's observations take in the synthesis request
#: (texts and quotes together): past it the longest are cut to one length, never below
#: ``OBSERVATION_FLOOR``, each cut marked and counted (`fit_observations`).
PART_OBSERVATION_CHARS = 2 * OBSERVATION_CHARS
OBSERVATION_FLOOR = 200


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
3. Every change the optimizer tried this epoch: each one evaluated, with what it fixed
   and broke and whether it was kept, and each one refused before it ran, with why.
4. A longitudinal comparison on the SAME sampled items under those two pipelines.
5. The previous optimizer memory, if one existed.

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
- Judge only kinds of change with evidence in front of you: this epoch's tries, or
  what the previous memory records of earlier ones. Any other kind is untested: do
  not advise for or against it, and do not list the untested kinds.
- Keep it compact and high-signal. Prefer a few durable principles.
- Revise or remove parts of the previous memory if they did not help.
- Do not output instructions for the pipeline's own steps.
- Do not restate the whole pipeline; summarize editing strategy.

You reply with JSON only."""

#: The memory system prompt for the synthesis call of a split pass: the same rules, the
#: comparison read through the comparison calls' observations.
SYNTHESIS_SYSTEM = MEMORY_SYSTEM.replace(
    "4. A longitudinal comparison on the SAME sampled items under those two pipelines.",
    "4. A longitudinal comparison on the SAME sampled items under those two pipelines,\n"
    "   read in parts by comparison calls: their observations, each citing its items.",
).replace("- Use evidence from the comparison, not generic advice.",
          "- Use evidence from the comparison (the observations), not generic advice.")

COMPARE_SYSTEM = """\
You read evidence for an optimizer-coach in a system that optimizes an AI pipeline: a
graph of steps (nodes) connected by edges, each step configured by settings (knobs)
such as its prompt.

Your job is not to solve tasks directly and not to write instructions for the pipeline.
You read one part of a longitudinal comparison: the SAME sampled items under two
pipelines (or two rollouts of one pipeline), each item with both full traces. Another
call reads your observations, with those of the other parts, and writes a compact
OPTIMIZER-SIDE memory that helps future optimizer calls produce better changes to the
pipeline in this run.

## What You Receive

1. The pipeline(s) and what changed between the two (or that nothing did).
2. Part of the comparison: items that regressed, persistently failed, improved or
   stayed right, with both outcomes, the grader's reasons and both traces.

## Your Goal

Write observations for the future optimizer: what regressed, what improved, what
persistently fails, and why -- where in the pipeline the traces show it starts (a step,
a setting, the data passed along an edge, the structure), and whether the change
between the pipelines explains it.

## Important Constraints

- Address the FUTURE OPTIMIZER directly, not the pipeline.
- Use evidence from the traces, not generic advice: cite the items and quote briefly.
- Patterns shared by several items first; say when one holds for a single item.
- Report what these traces show; do not judge kinds of change in general (another
  call weighs them against what was tried).
- Do not output instructions for the pipeline's own steps.

You reply with JSON only."""


def compare_answer_text() -> str:
    return ("Respond ONLY with a valid JSON object:\n"
            "{\n"
            '  "observations": [{"category": "regressed|persistent_fail|improved|stable_success",\n'
            '                    "observation": "<what happened and why, for the optimizer>",\n'
            '                    "items": ["<item id>", ...],\n'
            '                    "evidence": ["<a short verbatim quote from a trace>", ...]}]\n'
            "}\n"
            f"At most {OBSERVATIONS_PER_CALL} observations, the most important first, together "
            f"at most {OBSERVATION_CHARS} characters; every one names the items it covers "
            "(ids from this part).")


def parse_observations(payload: Any) -> list[dict[str, Any]] | None:
    """A comparison call's observations (at most ``OBSERVATIONS_PER_CALL``, each with
    its category, text, items and up to ``EVIDENCE_PER_OBSERVATION`` quotes), or
    ``None`` when the answer carries no ``observations`` list. Texts are kept whole."""
    listed = payload.get("observations") if isinstance(payload, Mapping) else None
    if not isinstance(listed, list):
        return None
    out = []
    for raw in listed:
        if not isinstance(raw, Mapping):
            continue
        text = str(raw.get("observation") or "").strip()
        if not text:
            continue
        names = {name for name, _t, _l in CATEGORIES}
        kind = str(raw.get("category") or "").strip()
        items = raw.get("items") if isinstance(raw.get("items"), list) else []
        evidence = raw.get("evidence") if isinstance(raw.get("evidence"), list) else []
        out.append({"category": kind if kind in names else "",
                    "observation": text,
                    "items": [str(i) for i in items if i not in (None, "")],
                    "evidence": [str(q) for q in evidence
                                 if q not in (None, "")][:EVIDENCE_PER_OBSERVATION]})
    return out[:OBSERVATIONS_PER_CALL]


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
                 system: str = MEMORY_SYSTEM, tried: Sequence[str] = (),
                 objective: str = "") -> tuple[str, dict[str, int]]:
    """The memory request within ``room`` characters (the system prompt counted), and
    what was clipped: ``{"items", "chars"}``.

    ``prev``/``curr`` describe the two sides: ``{"id", "pipeline", "rollout"}`` (the
    candidate, its pipeline as an analyst reads it, which rollout its records are).
    ``changes`` are the lines saying what changed between them; ``tried`` the lines of
    "What was tried this epoch" (`tried_lines`) and ``objective`` what the score
    measures (`base.objective_text`), each left out when empty. Every trace is whole
    when they all fit; otherwise the longest are cut to one length (`item_cap`), step
    inputs first (`fit_item`), each cut said."""
    same = prev.get("id") == curr.get("id")
    shown = [e for e in entries if dict((n, t) for n, t, _ in CATEGORIES)[e["category"]]]
    head, tail = _frame(epoch=epoch, mode_name=mode_name, prev=prev, curr=curr, same=same,
                        changes=changes, previous_note=previous_note, entries=entries,
                        tried=tried, objective=objective)
    wholes = {(e["id"], side): item_text(e[side]) for e in shown for side in ("prev", "curr")}
    blank = _comparison(entries, {key: "" for key in wholes}, same)
    fixed = len(system) + len(head) + len(blank) + len(tail)
    cap = item_cap([len(text) for text in wholes.values()], max(0, room - fixed))
    texts = {key: (text if cap is None else fit_item(
        next(e for e in shown if e["id"] == key[0])[key[1]], cap, ROOM_NAME))
        for key, text in wholes.items()}
    clipped = [(wholes[key], texts[key]) for key in wholes if texts[key] != wholes[key]]
    return head + _comparison(entries, texts, same) + tail, {
        "items": len(clipped),
        "chars": sum(max(0, len(whole) - len(text)) for whole, text in clipped)}


def _frame(*, epoch: int, mode_name: str, prev: Mapping[str, Any], curr: Mapping[str, Any],
           same: bool, changes: Sequence[str], previous_note: str,
           entries: Sequence[Mapping[str, Any]], tried: Sequence[str] = (),
           objective: str = "") -> tuple[str, str]:
    head = (_intro(epoch=epoch, mode_name=mode_name, prev=prev, curr=curr, same=same,
                   changes=changes, previous_note=previous_note, tried=tried,
                   objective=objective)
            + f"## Longitudinal comparison (the same {len(entries)} train items, "
            + ("two rollouts" if same else "two pipelines") + ")\n"
            f"Start of the epoch: {prev.get('rollout')}. Now: {curr.get('rollout')}. Each "
            "trace is the item in full: its input, every step's input and output in the "
            "order the steps ran, the expected answer, the prediction and the grader's "
            "feedback.\n\n")
    return head, f"\n\n## Answer\n{answer_text()}"


def _intro(*, epoch: int, mode_name: str, prev: Mapping[str, Any], curr: Mapping[str, Any],
           same: bool, changes: Sequence[str], previous_note: str,
           tried: Sequence[str] = (), objective: str = "") -> str:
    """The memory request up to its comparison: the objective, the pipelines and what
    changed, what was tried and the previous note to reflect on."""
    pipelines = _pipelines_text(prev=prev, curr=curr, same=same, changes=changes, tried=tried)
    note = previous_note.strip() or "(No previous optimizer memory -- this is the first update.)"
    return (f"# Writing the optimizer's memory -- epoch {epoch}\n\n"
            f"Mode: `{mode_name}`.\n\n"
            + (f"## What the score measures\n{objective}\n\n" if objective else "")
            + pipelines
            + ("\n".join(tried) + "\n\n" if tried else "")
            + "## Previous optimizer memory\n"
            "The following optimizer memory was available during this epoch. Reflect on "
            "whether it improved or harmed the quality of the changes.\n\n"
            f"{note}\n\n")


def _pipelines_text(*, prev: Mapping[str, Any], curr: Mapping[str, Any], same: bool,
                    changes: Sequence[str], tried: Sequence[str] = ()) -> str:
    """The pipeline(s) as the memory request shows them: the one pipeline when unchanged,
    else both and what changed between them."""
    if same:
        pipelines = (f"## The pipeline (candidate `{curr.get('id')}`)\n"
                     "Nothing this epoch replaced it"
                     + (" (what was tried and not kept is under \"What was tried this "
                        "epoch\")" if tried and tried[-1] != NOTHING_TRIED else "")
                     + ": the comparison below is two rollouts of "
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
    return pipelines


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


# ── a pass too long for one call: comparison calls, then a synthesis ────────


def _tally_text(entries: Sequence[Mapping[str, Any]]) -> str:
    tally = counts(entries)
    return (f"improved {tally['improved']}, regressed {tally['regressed']}, persistent "
            f"failures {tally['persistent_fail']}, stable successes {tally['stable_success']}")


def _ordered(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The pairs in the order the comparison lists them (`CATEGORIES`: regressions first,
    then persistent failures, improvements, stable successes), sample order within."""
    rank = {name: n for n, (name, _t, _l) in enumerate(CATEGORIES)}
    return sorted(entries, key=lambda e: rank[e["category"]])


def _cut(text: str, cap: int | None) -> str:
    return text if cap is None or len(text) <= cap else f"{text[:cap]}… [cut: {len(text) - cap} chars]"


def fit_observations(found: Sequence[Mapping[str, Any]], chars: int = PART_OBSERVATION_CHARS
                     ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One part's observations within ``chars`` (their texts and quotes together): whole
    when they fit; else the longest texts cut to one length (`item_cap`, never below
    ``OBSERVATION_FLOOR``), each cut marked ``… [cut: N chars]``. Returns them and what
    was cut: ``{"texts", "chars"}``."""
    texts = [len(str(o.get("observation") or "")) for o in found] + [
        len(str(q)) for o in found for q in o.get("evidence") or ()]
    cap = item_cap(texts, chars, floor=OBSERVATION_FLOOR)
    if cap is None:
        return [dict(o) for o in found], {"texts": 0, "chars": 0}
    out = [{**o, "observation": _cut(str(o.get("observation") or ""), cap),
            "evidence": [_cut(str(q), cap) for q in o.get("evidence") or ()]} for o in found]
    over = [n for n in texts if n > cap]
    return out, {"texts": len(over), "chars": sum(n - cap for n in over)}


def _observation_line(entry: Mapping[str, Any]) -> str:
    items = ", ".join(entry.get("items") or ()) or "none named"
    quotes = [f'"{q}"' for q in entry.get("evidence") or ()]
    return (f"- [{entry.get('category') or 'general'}; items: {items}] "
            f"{entry.get('observation')}"
            + (" -- evidence: " + " | ".join(quotes) if quotes else ""))


class MemoryRequest:
    """One pass's memory request, for `AutoresearchOptimizer.write_memory`.

    Called with a room, it is `build_prompt`: the single request, SkillOpt's. When that
    does not fit ``room`` whole, `parts` splits the pairs into comparison calls (each as
    many whole pairs as fit, in the comparison's order), `compare_prompt` makes each
    one's request, and `synthesis_prompt` the one request that writes the note from
    their observations. No trace is cut, except a pair longer than the room on its own,
    counted as clipped."""

    def __init__(self, *, epoch: int, mode_name: str, prev: Mapping[str, Any],
                 curr: Mapping[str, Any], changes: Sequence[str], previous_note: str,
                 entries: Sequence[Mapping[str, Any]], tried: Sequence[str] = (),
                 objective: str = "") -> None:
        self.epoch, self.mode_name = epoch, mode_name
        self.prev, self.curr = dict(prev), dict(curr)
        self.changes, self.previous_note = list(changes), previous_note
        self.entries, self.tried, self.objective = list(entries), list(tried), objective
        self.same = self.prev.get("id") == self.curr.get("id")
        self._texts: dict[tuple[str, str], str] = {}

    def __call__(self, room: int = MEMORY_ROOM) -> tuple[str, dict[str, int]]:
        return build_prompt(epoch=self.epoch, mode_name=self.mode_name, prev=self.prev,
                            curr=self.curr, changes=self.changes,
                            previous_note=self.previous_note, entries=self.entries,
                            room=room, tried=self.tried, objective=self.objective)

    def _whole(self, entry: Mapping[str, Any], side: str) -> str:
        key = (str(entry["id"]), side)
        if key not in self._texts:
            self._texts[key] = item_text(entry[side])
        return self._texts[key]

    def parts(self, room: int = MEMORY_ROOM) -> list[list[Mapping[str, Any]]] | None:
        """``None`` when the single request fits ``room`` whole (one call); else the
        comparison calls' pairs: in the comparison's order, each part filled until the
        next pair whole would take its request past ``room``. Every pair is in exactly
        one part, whole; a pair too long alone is a part of its own."""
        if self._single_length() <= room:
            return None
        out: list[list[Mapping[str, Any]]] = []
        most = len(self.entries)
        for entry in _ordered(self.entries):
            if out and self._length([*out[-1], entry], most, most) <= room:
                out[-1].append(entry)
            else:
                out.append([entry])
        return out

    def _single_length(self) -> int:
        """The single request's length with every trace whole, system prompt included,
        measured as `build_prompt` measures it."""
        head, tail = _frame(epoch=self.epoch, mode_name=self.mode_name, prev=self.prev,
                            curr=self.curr, same=self.same, changes=self.changes,
                            previous_note=self.previous_note, entries=self.entries,
                            tried=self.tried, objective=self.objective)
        texts = {(str(e["id"]), side): self._whole(e, side) for e in self.entries
                 for side in ("prev", "curr") if self._traced(e)}
        return len(MEMORY_SYSTEM) + len(head) + len(_comparison(self.entries, texts, self.same)) \
            + len(tail)

    def _length(self, part: Sequence[Mapping[str, Any]], index: int, total: int) -> int:
        texts = {(str(e["id"]), side): self._whole(e, side) for e in part
                 for side in ("prev", "curr") if self._traced(e)}
        return len(COMPARE_SYSTEM) + len(self._compare_text(part, index, total, texts))

    @staticmethod
    def _traced(entry: Mapping[str, Any]) -> bool:
        return dict((n, t) for n, t, _l in CATEGORIES)[entry["category"]]

    def _compare_text(self, part: Sequence[Mapping[str, Any]], index: int, total: int,
                      texts: Mapping[tuple[str, str], str]) -> str:
        # The pipeline(s) and what changed, as the single call shows them (not what was
        # tried: weighing kinds of change is the synthesis call's).
        changed = _pipelines_text(prev=self.prev, curr=self.curr, same=self.same,
                                  changes=self.changes)
        return (f"# Comparing two rollouts for the optimizer's memory -- epoch {self.epoch}, "
                f"part {index} of {total}\n\n"
                f"Mode: `{self.mode_name}`.\n\n"
                + (f"## What the score measures\n{self.objective}\n\n" if self.objective
                   else "")
                + changed
                + f"## Longitudinal comparison, part {index} of {total} ({len(part)} of the "
                f"same {len(self.entries)} train items, "
                + ("two rollouts" if self.same else "two pipelines") + ")\n"
                f"Start of the epoch: {self.prev.get('rollout')}. Now: "
                f"{self.curr.get('rollout')}. Over all {len(self.entries)} items: "
                f"{_tally_text(self.entries)}. The comparison did not fit one request, so "
                "each part is read by its own call; this part holds the items below, and "
                "its summary counts only them. Each trace is the item in full: its input, "
                "every step's input and output in the order the steps ran, the expected "
                "answer, the prediction and the grader's feedback.\n\n"
                + _comparison(part, texts, self.same)
                + f"\n\n## Answer\n{compare_answer_text()}")

    def compare_prompt(self, part: Sequence[Mapping[str, Any]], index: int, total: int,
                       room: int = MEMORY_ROOM) -> tuple[str, dict[str, int]]:
        """One comparison call's request and what was clipped (``{"items", "chars"}``):
        every trace whole when the part fits ``room`` (a part from `parts` does, unless
        it is one pair too long alone); else its longest traces cut to one length
        (`item_cap`), step inputs first (`fit_item`), each cut said."""
        traced = [e for e in part if self._traced(e)]
        wholes = {(str(e["id"]), side): self._whole(e, side)
                  for e in traced for side in ("prev", "curr")}
        fixed = len(COMPARE_SYSTEM) + len(self._compare_text(
            part, index, total, {key: "" for key in wholes}))
        cap = item_cap([len(text) for text in wholes.values()], max(0, room - fixed))
        by_id = {str(e["id"]): e for e in traced}
        texts = {key: (text if cap is None else fit_item(by_id[key[0]][key[1]], cap, ROOM_NAME))
                 for key, text in wholes.items()}
        clipped = [(wholes[key], texts[key]) for key in wholes if texts[key] != wholes[key]]
        return self._compare_text(part, index, total, texts), {
            "items": len(clipped),
            "chars": sum(max(0, len(whole) - len(text)) for whole, text in clipped)}

    def synthesis_prompt(self, results: Sequence[Mapping[str, Any]],
                         room: int = MEMORY_ROOM) -> tuple[str, dict[str, int]]:
        """The request that writes the note from the comparison calls' ``results``
        (each ``{"part", "items", "observations"}``) and what it cut: the same request
        as the single call's -- the objective, both pipelines and what changed, what was
        tried, the previous note -- with every part's observations in place of the
        traces. Each part's observations are kept within ``PART_OBSERVATION_CHARS``
        (`fit_observations`; a model asked for at most ``OBSERVATION_CHARS`` rarely
        passes it). When the request is still past ``room`` -- pipelines so large that
        they and the observations do not fit, or a call retried within half its length
        after the provider refused it as too long -- the two pipelines' texts are
        clipped head and tail to the largest length that fits. Both cuts are counted:
        ``{"observation_texts", "observation_chars", "pipeline_chars"}``."""
        fitted = [fit_observations(list(result.get("observations") or ())) for result in results]
        cut = {"observation_texts": sum(c["texts"] for _f, c in fitted),
               "observation_chars": sum(c["chars"] for _f, c in fitted), "pipeline_chars": 0}

        def make(chars: int | None) -> str:
            prev, curr = dict(self.prev), dict(self.curr)
            if chars is not None:
                prev["pipeline"] = _clip_middle(prev.get("pipeline"), chars)
                curr["pipeline"] = _clip_middle(curr.get("pipeline"), chars)
            parts = []
            for result, (found, _cut_) in zip(results, fitted):
                parts.append(f"### Part {result.get('part')} of {len(results)}: items "
                             f"{', '.join(str(i) for i in result.get('items') or ())}\n"
                             + ("\n".join(_observation_line(o) for o in found)
                                or "(no observation)"))
            return (_intro(epoch=self.epoch, mode_name=self.mode_name, prev=prev, curr=curr,
                           same=self.same, changes=self.changes,
                           previous_note=self.previous_note, tried=self.tried,
                           objective=self.objective)
                    + f"## Longitudinal comparison (the same {len(self.entries)} train items, "
                    + ("two rollouts" if self.same else "two pipelines") + ")\n"
                    f"Start of the epoch: {self.prev.get('rollout')}. Now: "
                    f"{self.curr.get('rollout')}. Over all {len(self.entries)} items: "
                    f"{_tally_text(self.entries)}. Both full traces of every item did not "
                    f"fit one request, so {len(results)} comparison calls each read part of "
                    "them whole (every item in exactly one part, regressions first, then "
                    "persistent failures, improvements and stable successes) and wrote these "
                    "observations for you. They are your evidence from the comparison.\n\n"
                    + "\n\n".join(parts)
                    + f"\n\n## Answer\n{answer_text()}")

        whole = make(None)
        if len(SYNTHESIS_SYSTEM) + len(whole) <= room:
            return whole, cut
        low, high = TEXT_FLOOR, max(TEXT_FLOOR, len(str(self.prev.get("pipeline") or "")),
                                    len(str(self.curr.get("pipeline") or "")))
        while low < high:
            middle = (low + high + 1) // 2
            if len(SYNTHESIS_SYSTEM) + len(make(middle)) <= room:
                low = middle
            else:
                high = middle - 1
        clipped = make(low)
        pipelines = [str(self.curr.get("pipeline") or "")]
        if not self.same:
            pipelines.append(str(self.prev.get("pipeline") or ""))
        cut["pipeline_chars"] = sum(len(text.strip()) - 2 * (low // 2) for text in pipelines
                                    if len(text.strip()) > low)
        return clipped, cut


def edits_text(edits: Any, chars: int = TRIED_EDITS_CHARS) -> str:
    """A proposal's edits in short: each edit's op and what it touches (the knob it sets,
    the steps it adds, removes, rewires or targets; `feedback.touched`), within
    ``chars``. No value or prompt text: the memory reads what kind of change was tried
    and where, not its wording."""
    from .feedback import touched

    parts = []
    for edit in edits if isinstance(edits, (list, tuple)) else [edits]:
        if not isinstance(edit, Mapping) or not edit.get("op"):
            parts.append("(no readable edit)")
            continue
        names = touched([edit])
        parts.append(f"{edit['op']}" + (f": {', '.join(names)}" if names else ""))
    return _clip("; ".join(parts) or "(none)", chars)


def tried_lines(tried: Sequence[Mapping[str, Any]], refused: Sequence[Mapping[str, Any]], *,
                round_: int | None, current: str | None, gated: bool,
                chars: int = TRIED_CHARS) -> list[str]:
    """"What was tried this epoch": every change the optimizer proposed in ``round_`` (the
    proposal round the start-of-epoch pipeline opened, which this epoch evaluated and
    judged), newest first -- the evaluated ones, then the ones refused before they ran.

    ``tried`` entries (oldest first): ``{"id", "parent", "round", "kind", "structural",
    "edits", "score", "exit_ok", "evaluated", "fixed", "broken", "shared", "same_seed",
    "gate", "integrity"}`` -- train records and the gate's word only, as the proposal
    request's history and change ledger show them; never a validation score.
    ``refused`` entries (oldest first): ``{"round", "attempt", "kind", "structural",
    "edits", "reasons"}``. Within ``chars``: past it, the entries listed last are cut --
    the refusals first, then the oldest evaluated -- and the cut is said."""
    lines = ["## What was tried this epoch",
             f"Every change the optimizer proposed in round {round_ if round_ is not None else '?'}, "
             "while the start-of-epoch pipeline was current (this epoch evaluated and judged "
             "them), newest first: the ones evaluated, then the ones refused before they ran. `fixed/broken` compares a candidate with its parent on the "
             "train items both were scored on with the same seed (parent wrong and child "
             "right / the reverse)"
             + ("; `gate` is what the validation gate decided (kept, rejected, or current: "
                "the pipeline now)" if gated else "")
             + ". A change tried and not kept, or refused, is evidence too."]
    if not tried and not refused:
        return [*lines, NOTHING_TRIED]
    entries = [_tried_line(entry, current, gated) for entry in reversed(list(tried))]
    entries += [_refused_line(entry) for entry in reversed(list(refused))]
    used = sum(len(line) + 1 for line in lines)
    for index, line in enumerate(entries):
        if used + len(line) + 1 > chars and index:
            cut = len(entries) - index
            lines.append(f"[cut: {cut} entr{'y' if cut == 1 else 'ies'} not shown: "
                         "refusals first, then the oldest evaluated]")
            break
        lines.append(line)
        used += len(line) + 1
    return lines


def _change_text(entry: Mapping[str, Any]) -> str:
    """Structural or knob, and the kind (`feedback.change_kind`); ``structural`` ``None``
    is a proposal with no readable edit."""
    if entry.get("structural") is None:
        return "no readable edit"
    kind = entry.get("kind") or "other"
    return (f"structural change: {kind}" if entry.get("structural")
            else f"knob change: {kind}")


def _tried_line(entry: Mapping[str, Any], current: str | None, gated: bool) -> str:
    fixed, broken = entry.get("fixed"), entry.get("broken")
    if not entry.get("evaluated", True):
        outcome = "not evaluated"
    elif entry.get("exit_ok") is False:
        outcome = "its train run failed"
    else:
        outcome = f"train score {entry.get('score')}"
        if isinstance(fixed, int) and isinstance(broken, int):
            outcome += (f"; vs its parent fixed {fixed} / broken {broken} of "
                        f"{entry.get('shared')} train items")
        elif entry.get("same_seed") is False:
            outcome += "; fixed/broken n/a (its parent was not scored with the same seed)"
    gate = entry.get("gate")
    return (f"- `{entry.get('id')}` (round {entry.get('round')}, from `{entry.get('parent')}`; "
            f"{_change_text(entry)}): {entry.get('edits')} -- {outcome}"
            + (f" -- gate: {gate or 'not gated'}" if gated else "")
            + (" -- the pipeline now" if entry.get("id") == current and not gated else "")
            + (f" -- integrity: {entry['integrity']}" if entry.get("integrity") else ""))


def _refused_line(entry: Mapping[str, Any]) -> str:
    reasons = [str(r) for r in entry.get("reasons") or () if r]
    return (f"- refused before it ran (round {entry.get('round')}, attempt "
            f"{entry.get('attempt')}; {_change_text(entry)}): {entry.get('edits')} -- "
            + (" | ".join(_clip(r, TRIED_REASON_CHARS) for r in reasons[:2])
               or "(no reason recorded)"))


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
    if not after:
        return ["The current candidate is an earlier version of the start-of-epoch one: "
                "these changes, oldest first, made the start-of-epoch candidate from it, and "
                "are not in the current one:", *lines(before)]
    return [f"They are on different branches from their common ancestor `{common}`. "
            "From it to the start-of-epoch candidate:", *lines(before),
            "From it to the current candidate:", *lines(after)]


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
            "evaluated", "rollouts", "calls", "parts", "context_retry", "clipped_items",
            "clipped_chars", "synthesis_cuts", "chars", "updated", "error", "skipped")
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
                     f"{p.get('pairs', 0)} of {len(p.get('sample') or ())}"
                     + (f" in {p['parts']} parts" if p.get("parts") else ""),
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
             "in optimizer_trace.jsonl (`memory`; a pass too long for one call also has its "
             "comparison calls, `memory_compare`, and the table says how many).", "",
             _table(["epoch", "start of epoch", "now", "items compared",
                     "regressed/still failing/improved/stable", "extra evaluations",
                     "note", "chars"], rows), ""]
    cut = [p for p in rep["passes"] if p.get("synthesis_cuts")]
    if cut:
        texts = sum(int(p["synthesis_cuts"].get("observation_texts") or 0) for p in cut)
        chars = sum(int(p["synthesis_cuts"].get("observation_chars") or 0) for p in cut)
        piped = sum(int(p["synthesis_cuts"].get("pipeline_chars") or 0) for p in cut)
        lines += [f"In {len(cut)} split pass(es) the synthesis request was cut to fit: "
                  f"{texts} observation text(s) ({chars} characters) and {piped} characters "
                  "of the pipelines' texts.", ""]
    latest = rep.get("latest") or {}
    if latest.get("text"):
        lines += [f"The latest note (epoch {latest.get('epoch')}):", "",
                  *(f"> {line}" if line else ">" for line in str(latest["text"]).splitlines()),
                  ""]
    return lines
