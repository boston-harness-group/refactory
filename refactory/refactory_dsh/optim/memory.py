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
  every change the optimizer tried in the epoch, by what it was (its edits with the text
  it gave a new step or judge, where each new step sits and what it sends back) --
  evaluated, with its train fixed/broken counts and the gate's word, or refused before it
  ran, with why (`tried_lines`; the facts the proposal request already shows, never a
  validation score), and a tally per change, never per kind -- so a pass with
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
  being SkillOpt's ``meta_skill.md`` for a graph, its "which kinds of edit help" asked
  instead per change tried: what it was, how it moved the items and why, and what to
  try next; a kind judged in general only from several different designs that agree. SkillOpt sets no length; its real runs
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

from . import evidence
from .analysts import ANALYST_ROOM, TEXT_FLOOR, fit_item, item_cap, item_text
from .base import TRAJECTORY_INPUT_KEYS, _by_item, _clip, _clip_middle, _table, fmt_score

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
TRIED_CHARS = 16000
TRIED_EDITS_CHARS = 1500
TRIED_REASON_CHARS = 300
#: The most an edit's head (op and names) takes when its edits are cut (`edits_text`).
TRIED_HEAD_CHARS = 120
#: What "What was tried this epoch" keeps free for the line saying a cut.
TRIED_CUT_ROOM = 120
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
2. The pipeline now, and what changed between them -- or, when nothing replaced it, a
   change tried this epoch and rejected, compared with it on the items that change moved.
3. Every change the optimizer tried this epoch: each one evaluated, with what it fixed
   and broke and whether it was kept, and each one refused before it ran, with why.
4. A longitudinal comparison on the SAME sampled items under those two pipelines.
5. The previous optimizer memory, if one existed.

## Your Goal

Write a concise memory that improves future optimizer behavior in stages such as
failure analysis, success analysis, merging findings, and proposing and ranking
changes.

This memory should capture things like:
- For each change tried: what it was -- the step it changed or added, what a new step
  or judge was told to do or check, where it sits and where it sends work back, its
  limits, what text it changed -- how it moved the items (which it fixed, which it
  broke) and why, from the comparison and the traces.
- What to try next given that: what to keep, what in a change that hurt to revise (its
  instructions or criteria, its place, what it sends back, its limit), and what to try
  instead.
- What level of detail works best for a change here: a prompt's wording, a new step's
  job, a judge's criteria, a step's place, its limits.
- What failure-repair patterns should be prioritized.
- What regression risks future optimizer calls should guard against.

## Important Constraints

- Address the FUTURE OPTIMIZER directly, not the pipeline.
- Focus on how to choose and write better changes.
- Use evidence from the comparison, not generic advice.
- Write each lesson about the change that was tried, named by what it did (for example
  "the judge after step X that checks Y sent right answers back, and step X then
  dropped them"), not about every change of its kind (every loop, every new step,
  every prompt edit): two changes of one kind differ in what decides their effect.
- Say something about a kind of change in general only when several different designs
  of that kind were tried (this epoch, or as the previous memory records) and agree.
  Never from one design, however large its effect.
- A result inside the noise the tally gives is weak evidence even about that change:
  say how it moved its items, not that it failed.
- Advise only on changes with evidence in front of you: this epoch's tries, or what
  the previous memory records of earlier ones. Do not advise for or against changes
  nobody tried, and do not list them.
- Keep it compact and high-signal. Prefer a few durable principles.
- Revise or remove parts of the previous memory if they did not help, or if they
  judge a kind of change from one design.
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
- Report what these traces show; leave verdicts on the changes tried to the call that
  weighs each of them against these observations, and judge no kind of change in
  general.
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


#: What a pass calls the change it sets against the unchanged incumbent: under the
#: validation gate it was rejected; without it, it was only not the train leader.
REJECTED_NOUN, NOT_KEPT_NOUN = "rejected change", "unkept change"


def noun(curr: Mapping[str, Any]) -> str:
    """The name of ``curr`` as a change of the unchanged incumbent (``REJECTED_NOUN`` or,
    ungated, ``NOT_KEPT_NOUN``)."""
    return str((curr.get("rejected") or {}).get("noun") or REJECTED_NOUN)


def changes_heading(compared: str | None) -> str:
    """The heading of the section showing what separates the two pipelines: ``compared``
    is the compared change's noun (`noun`), ``None`` for the current pipeline."""
    return f"What the {compared} changed" if compared else "What changed between them"


def _look(prev: Mapping[str, Any], curr: Mapping[str, Any]) -> bool | tuple[str, str]:
    """How the two sides are labelled: ``True`` for two rollouts of one pipeline,
    ``False`` for the start-of-epoch and current pipelines, or the two labels when
    ``curr`` is a rejected change of the unchanged incumbent (``curr["rejected"]``)."""
    if curr.get("rejected"):
        return (f"Incumbent `{prev.get('id')}`",
                f"{noun(curr)[0].upper()}{noun(curr)[1:]} `{curr.get('id')}`")
    return prev.get("id") == curr.get("id")


def _labels(same: bool | tuple[str, str]) -> tuple[str, str]:
    if isinstance(same, tuple):
        return same
    return ("First rollout", "Second rollout") if same else ("Start of epoch", "Now")


def _what(curr: Mapping[str, Any], same: bool) -> str:
    """What the comparison compares, for its heading."""
    if curr.get("rejected"):
        return f"the {noun(curr)} against the incumbent, on the items it changed"
    return "two rollouts" if same else "two pipelines"


def _sides(prev: Mapping[str, Any], curr: Mapping[str, Any]) -> str:
    """Which rollout each side is, and for a rejected change what it did and what the
    comparison leaves out."""
    rejected = curr.get("rejected")
    if not rejected:
        return f"Start of the epoch: {prev.get('rollout')}. Now: {curr.get('rollout')}."
    others, name = list(rejected.get("others") or ()), noun(curr)
    why = ("" if name == REJECTED_NOUN else " (without a validation gate the current is the "
           "train leader; this change scored below it on train)")
    return (f"Nothing this epoch replaced the incumbent; this compares it with the {name} "
            f"that moved the most items{why}. Incumbent: {prev.get('rollout')}. "
            f"{name[0].upper()}{name[1:]}: {curr.get('rollout')}. It fixed {rejected.get('fixed')} and broke "
            f"{rejected.get('broken')} of {rejected.get('shared')} shared train items; "
            + (f"the first {rejected.get('shown')} of the {rejected.get('changed')} it changed "
               "are shown, broken first"
               if rejected.get("shown") != rejected.get("changed")
               else f"all {rejected.get('changed')} it changed are shown, broken first")
            + ", and the items it left as they were are not."
            + (f" The other {name}(s) that moved items ({', '.join(f'`{c}`' for c in others)}) "
               "are under \"What was tried this epoch\", with their edits and how they moved "
               "their items; their traces are not shown." if others else ""))


def _shown(side: str, same: bool | tuple[str, str]) -> dict[str, Any]:
    """How one side's trace is rendered: headed by its rollout, so a "same as" reference
    says which; the input and the expected answer only on the first side (the second is
    the same item)."""
    before, after = _labels(same)
    return ({"label": before, "heading": "#####"} if side == "prev" else
            {"label": after, "heading": "#####", "show_input": False, "show_expected": False})


def _trace(entry: Mapping[str, Any], side: str, same: bool | tuple[str, str],
           settled: frozenset[str] | None = None) -> str:
    return item_text(entry[side], **_shown(side, same), settled=settled)


def _settled(prev: Mapping[str, Any], curr: Mapping[str, Any],
             entries: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """The lines a trace's step does not repeat (`analysts.shown_settled`): what the request
    shows of both pipelines (each step's settings as shown), over both sides' records."""
    from .analysts import shown_settled

    return shown_settled(f"{prev.get('pipeline') or ''}\n{curr.get('pipeline') or ''}",
                         [e[side] for e in entries for side in ("prev", "curr")
                          if isinstance(e.get(side), Mapping)])


def _side_line(label: str, record: Mapping[str, Any]) -> str:
    return (f"- {label}: {'PASS' if record.get('passed') else 'FAIL'} "
            f"(score {fmt_score(record.get('score'))}) -- answer: {_answer(record)}")


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
    same, look = prev.get("id") == curr.get("id"), _look(prev, curr)
    shown = [e for e in entries if dict((n, t) for n, t, _ in CATEGORIES)[e["category"]]]
    head, tail = _frame(epoch=epoch, mode_name=mode_name, prev=prev, curr=curr, same=same,
                        changes=changes, previous_note=previous_note, entries=entries,
                        tried=tried, objective=objective)
    settled = _settled(prev, curr, entries)
    wholes = {(e["id"], side): _trace(e, side, look, settled)
              for e in shown for side in ("prev", "curr")}
    blank = _comparison(entries, {key: "" for key in wholes}, look)
    fixed = len(system) + len(head) + len(blank) + len(tail)
    cap = item_cap([len(text) for text in wholes.values()], max(0, room - fixed))
    texts = {key: (text if cap is None else fit_item(
        next(e for e in shown if e["id"] == key[0])[key[1]], cap, ROOM_NAME,
        **_shown(key[1], look), settled=settled))
        for key, text in wholes.items()}
    clipped = [(wholes[key], texts[key]) for key in wholes if texts[key] != wholes[key]]
    if cap is None:  # clipped traces keep the one length `item_cap` chose
        texts = _deduped(entries, texts, look)
    return head + _comparison(entries, texts, look) + tail, {
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
            + _what(curr, same) + ")\n"
            f"{_sides(prev, curr)} Each trace is the item in full: "
            f"{evidence.TRACE_FORM}.\n\n")
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
    elif curr.get("rejected"):
        pipelines = (f"## The pipeline (candidate `{prev.get('id')}`)\n"
                     "Nothing this epoch replaced it; the comparison below sets it against "
                     f"the {noun(curr)} `{curr.get('id')}` on the items that change "
                     "moved.\n\n"
                     f"{prev.get('pipeline')}\n\n"
                     f"## The {noun(curr)}'s pipeline (candidate `{curr.get('id')}`, not kept)\n"
                     f"{curr.get('pipeline')}\n\n"
                     f"## {changes_heading(noun(curr))}\n"
                     + ("\n".join(changes) if changes else "(not known)") + "\n\n")
    else:
        pipelines = (f"## The pipeline at the start of this epoch (candidate `{prev.get('id')}`)\n"
                     f"{prev.get('pipeline')}\n\n"
                     f"## The pipeline now (candidate `{curr.get('id')}`)\n"
                     f"{curr.get('pipeline')}\n\n"
                     f"## {changes_heading(None)}\n"
                     + ("\n".join(changes) if changes else "(not known)") + "\n\n")
    return pipelines


def _deduped(entries: Sequence[Mapping[str, Any]], texts: Mapping[tuple[str, str], str],
             same: bool | tuple[str, str]) -> dict[tuple[str, str], str]:
    """``texts`` with each instruction text an earlier trace in the comparison's order
    showed said as "same as" (`optim.evidence.dedup`)."""
    from .evidence import dedup

    order = [(e["id"], side) for name, traces, _label in CATEGORIES if traces
             for e in entries if e["category"] == name for side in ("prev", "curr")
             if (e["id"], side) in texts]
    rest = [key for key in texts if key not in order]
    keys = order + rest
    return dict(zip(keys, dedup([texts[key] for key in keys])))


def _comparison(entries: Sequence[Mapping[str, Any]], texts: Mapping[tuple[str, str], str],
                same: bool | tuple[str, str]) -> str:
    """SkillOpt's ``format_comparison_text`` (slow_update.py:287-349) over the pairs."""
    tally = counts(entries)
    parts = [f"### Summary\nTotal items: {len(entries)}\n"
             f"- Improved (wrong -> right): {tally['improved']}\n"
             f"- Regressed (right -> wrong): {tally['regressed']}\n"
             f"- Persistent failures (wrong -> wrong): {tally['persistent_fail']}\n"
             f"- Stable successes (right -> right): {tally['stable_success']}"]
    before, after = _labels(same)
    for name, traces, label in CATEGORIES:
        listed = [e for e in entries if e["category"] == name]
        if not listed:
            parts.append(f"### {label}\n(none)")
            continue
        lines = [f"### {label}"]
        for e in listed:
            prev, curr = e["prev"], e["curr"]
            if traces:
                # Each trace says its verdict, answer and grader's reason, and the first
                # the task's input: the heading names the item only.
                lines += ["", f"#### Task {e['id']}{_meta_text(curr)}",
                          texts.get((e["id"], "prev"), ""), texts.get((e["id"], "curr"), "")]
                continue
            lines += ["", f"#### Task {e['id']}: {_task_text(curr, prev, e['id'])}"
                          f"{_meta_text(curr)}",
                      _side_line(before, prev), _side_line(after, curr)]
            if _reason(curr) and not curr.get("passed"):
                lines.append(f"- Reason now: {_clip(_reason(curr), 600)}")
            if _reason(prev) and not prev.get("passed"):
                lines.append(f"- Reason at the start: {_clip(_reason(prev), 600)}")
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
        self.look = _look(self.prev, self.curr)
        self._texts: dict[tuple[str, str], str] = {}
        self._settled = _settled(self.prev, self.curr, self.entries)

    def __call__(self, room: int = MEMORY_ROOM) -> tuple[str, dict[str, int]]:
        return build_prompt(epoch=self.epoch, mode_name=self.mode_name, prev=self.prev,
                            curr=self.curr, changes=self.changes,
                            previous_note=self.previous_note, entries=self.entries,
                            room=room, tried=self.tried, objective=self.objective)

    def _whole(self, entry: Mapping[str, Any], side: str) -> str:
        key = (str(entry["id"]), side)
        if key not in self._texts:
            self._texts[key] = _trace(entry, side, self.look, self._settled)
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
        return len(MEMORY_SYSTEM) + len(head) + len(_comparison(self.entries, texts, self.look)) \
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
        # tried: weighing each change tried is the synthesis call's).
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
                + _what(self.curr, self.same) + ")\n"
                f"{_sides(self.prev, self.curr)} Over all {len(self.entries)} items: "
                f"{_tally_text(self.entries)}. The comparison did not fit one request, so "
                "each part is read by its own call; this part holds the items below, and "
                "its summary counts only them. Each trace is the item in full: "
                f"{evidence.TRACE_FORM}.\n\n"
                + _comparison(part, _deduped(part, texts, self.look), self.look)
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
        texts = {key: (text if cap is None else fit_item(by_id[key[0]][key[1]], cap, ROOM_NAME,
                                                         **_shown(key[1], self.look),
                                                         settled=self._settled))
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
                    + _what(self.curr, self.same) + ")\n"
                    f"{_sides(self.prev, self.curr)} Over all {len(self.entries)} items: "
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


def edits_text(edits: Any, before: Mapping[str, Any] | None = None,
               chars: int = TRIED_EDITS_CHARS) -> str:
    """A proposal's edits with what identifies them, within ``chars``: a text
    ``set_knob`` as old -> new length and its changed lines against the value before it
    (``before``, the parent's knob values, updated by each earlier ``set_knob`` of the
    same knob in the proposal), else its new value; every other edit whole as JSON -- its
    op, the steps it names, where it sends work back, its limits and the text it gives a
    new step or judge (its instructions, its criteria). One line (JSON and the diff
    separator keep every newline out), so nothing quoted can read as a heading of the
    request. Past ``chars`` no edit is dropped: each keeps its head (op and the names
    it touches) and the longest are cut to one length (`item_cap`), the cut said
    (``[cut: N edits / M chars]``). The memory reads what each change was, not only its
    kind: a loop whose judge checks one thing is a different change from a loop whose
    judge checks another."""
    from .base import DIFF_LINE_SEPARATOR, _text_diff
    from .feedback import touched

    values = dict(before or {})
    listed = list(edits) if isinstance(edits, (list, tuple)) else [edits]
    parts, heads, objects = [], [], []
    for edit in listed:
        if not isinstance(edit, Mapping) or not edit.get("op"):
            parts.append("(no readable edit)")
            heads.append("")
            objects.append(None)
            continue
        name, new = edit.get("name"), edit.get("value")
        old = values.get(str(name))
        names = ", ".join(touched([edit]))
        heads.append(_clip(f"[{edit['op']}" + (f" {names}" if names else "") + "]",
                           TRIED_HEAD_CHARS))
        if edit.get("op") == "set_knob" and isinstance(new, str) and isinstance(old, str):
            diff = _text_diff(old, new)
            parts.append(f"set_knob {name}: {len(old)} -> {len(new)} chars; diff: "
                         + " ".join(diff.replace("\n", DIFF_LINE_SEPARATOR).split()))
            objects.append(None)
        else:
            parts.append(json.dumps(dict(edit), default=str, ensure_ascii=False))
            objects.append(dict(edit))
        if edit.get("op") == "set_knob" and name is not None and "value" in edit:
            values[str(name)] = new
    joined = "; ".join(parts) or "(none)"
    if len(joined) <= chars:
        return joined
    marker_room = 40
    cap = item_cap([len(p) for p in parts], max(0, chars - marker_room - 2 * len(parts)),
                   floor=0) or 0
    shown, cut_edits, cut_chars = [], 0, 0
    for part, head, edit in zip(parts, heads, objects):
        if len(part) <= cap:
            shown.append(part)
            continue
        short = _shrunk_edit(edit, part, head, cap)
        shown.append(short)
        cut_edits += 1
        cut_chars += max(0, len(part) - len(short))
    return "; ".join(shown) + f" [cut: {cut_edits} edits / {cut_chars} chars]"


def _middle(text: str, limit: int) -> str:
    """``text`` within ``limit``: whole, else its head and tail around ``…[N cut]…``."""
    mark = f"…[{len(text)} chars, middle cut]…"
    if len(text) <= max(limit, len(mark)):
        return text  # whole, or too short for a cut to shorten it
    keep = max(0, limit - len(mark))
    return text[:(keep + 1) // 2] + mark + (text[len(text) - keep // 2:] if keep // 2 else "")


def _shrunk_edit(edit: Mapping[str, Any] | None, part: str, head: str, room: int) -> str:
    """One edit's text within ``room``: a JSON edit keeps its op and wiring whole (the
    steps it names, what it sees, reads and writes, where it sends back; `base._map_text`)
    and only its free texts are cut in the middle, all to one length; past that, or for a
    knob's diff, its head (op and names) and the middle-cut text."""
    from .base import _map_text

    if edit is not None:
        def render(limit: int) -> str:
            return json.dumps(_map_text(dict(edit), lambda t: _middle(t, limit)),
                              default=str, ensure_ascii=False)

        low, high = 0, len(part)
        if len(render(0)) <= room:
            while low < high:
                middle = (low + high + 1) // 2
                if len(render(middle)) <= room:
                    low = middle
                else:
                    high = middle - 1
            return render(low)
    rest = max(0, room - len(head) - 1)
    return (f"{head} {_middle(part, rest)}" if head else _middle(part, room))[:max(room, 0)]


#: The heading of the per-change tally (`change_tally_lines`).
TALLY_HEADING = "Each change tried this epoch, against the noise"


def change_tally_lines(tried: Sequence[Mapping[str, Any]], *, val_items: int = 0,
                       spread: float | None = None, gated: bool = False) -> list[str]:
    """Per change tried, grouped by the change itself (`feedback.group_changes`: by a hash
    of its full edits, shown by what they did, its kind only as a label, two changes
    whose labels collide told apart where their texts differ), never by kind: how many tries, and each one's
    fixed/broken train items and the gate's word; then how large a noise one try sits
    in: one validation item's share of the validation part's mean (``1 / val_items``;
    never a validation score) and, when the run re-scored the parent, how far its train
    score moved between re-scores (``spread``). One source for the memory call and the
    proposer request (beside the note). Empty without an evaluated try."""
    from .feedback import group_changes

    changes = group_changes([entry for entry in tried if entry.get("evaluated", True)])
    if not changes:
        return []
    noise = []
    if val_items:
        noise.append(f"one item of the validation part moves its mean by 1/{val_items} = "
                     f"{1 / val_items:.3f}")
    if spread is not None:
        noise.append(f"the parent's train score moved by up to {spread:.3f} between its "
                     "re-scores")
    lines = [f"{TALLY_HEADING}"
             + (" (" + "; ".join(noise) + ")" if noise else "") + ":"]
    for cell, entries in changes:
        tries = []
        for entry in entries:
            fixed, broken = entry.get("fixed"), entry.get("broken")
            counts = (f"fixed {fixed} / broken {broken} of {entry.get('shared')} train items"
                      if isinstance(fixed, int) and isinstance(broken, int) else "fixed/broken n/a")
            tries.append(f"`{entry.get('id')}` {counts}"
                         + (f", gate: {entry.get('gate') or 'not gated'}" if gated else ""))
        lines.append(f"- {cell}: {len(entries)} tr{'y' if len(entries) == 1 else 'ies'}: "
                     + "; ".join(tries))
    return lines


#: What the proposer is told beside the tally; the memory call has it in MEMORY_SYSTEM.
TALLY_NOISE_RULE = ("Each line is one change: its result is evidence about that change -- "
                    "its step, what it was told to do or check, where it sends work back, "
                    "its limits -- not about other changes of its kind. A result inside "
                    "that noise is weak evidence even about the change itself.")


def tried_lines(tried: Sequence[Mapping[str, Any]], refused: Sequence[Mapping[str, Any]], *,
                round_: int | None, current: str | None, gated: bool,
                chars: int = TRIED_CHARS, tally: Sequence[str] = (),
                shown_in: Mapping[str, str] | None = None) -> list[str]:
    """"What was tried this epoch": every change the optimizer proposed in ``round_`` (the
    proposal round the start-of-epoch pipeline opened, which this epoch evaluated and
    judged), newest first -- the evaluated ones, then the ones refused before they ran.

    ``tried`` entries (oldest first): ``{"id", "parent", "round", "kind", "structural",
    "edits", "score", "exit_ok", "evaluated", "fixed", "broken", "shared", "same_seed",
    "gate", "integrity", "effect"}`` -- train records and the gate's word only, as the
    proposal request's history and change ledger show them; never a validation score.
    ``edits`` is what the change was (`edits_text`: each edit with the text it gave a
    new step or judge, a knob's changed lines); ``placed`` (`feedback.placement_lines`)
    where each step it added sits and what it sends back, and a structural change's
    ``effect`` (`optim.edit_effect`) how it moved the items it changed, under its line.
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
    # With the per-change tally (`change_tally_lines`) the counts and the gate's word are
    # said there, once; each entry says its edits whole, where its new steps sit, and its
    # score.
    lines += list(tally)
    # A candidate whose edits the request shows whole in its changes section (``shown_in``:
    # id -> that section's heading) points there, so no text is said twice.
    entries = [_tried_line(entry, current, gated, counted=bool(tally),
                           shown_in=(shown_in or {}).get(str(entry.get("id"))))
               for entry in reversed(list(tried))]
    entries += [_refused_line(entry) for entry in reversed(list(refused))]
    used = sum(len(line) + 1 for line in lines)
    for index, line in enumerate(entries):
        # Room kept for the cut's own line, so the section with it stays within ``chars``.
        if used + len(line) + 1 > chars - TRIED_CUT_ROOM and index:
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


def _tried_line(entry: Mapping[str, Any], current: str | None, gated: bool,
                counted: bool = False, shown_in: str | None = None) -> str:
    fixed, broken = entry.get("fixed"), entry.get("broken")
    if not entry.get("evaluated", True):
        outcome = "not evaluated"
    elif entry.get("exit_ok") is False:
        outcome = "its train run failed"
    else:
        outcome = f"train score {fmt_score(entry.get('score'))}"
        if counted:
            pass  # its fixed/broken counts are in the tally above
        elif isinstance(fixed, int) and isinstance(broken, int):
            outcome += (f"; vs its parent fixed {fixed} / broken {broken} of "
                        f"{entry.get('shared')} train items")
        elif entry.get("same_seed") is False:
            outcome += "; fixed/broken n/a (its parent was not scored with the same seed)"
    gate = entry.get("gate")
    from .edit_effect import effect_lines

    # Where each step it added sits and what it can send back (`feedback.placement_lines`),
    # then how a structural change moved the items it changed (`optim.edit_effect`), so a
    # note can name the change and say the mechanism rather than guess it.
    how = [*(entry.get("placed") or ()), *effect_lines(entry.get("effect") or {})]
    return (f"- `{entry.get('id')}` (round {entry.get('round')}, from `{entry.get('parent')}`; "
            f"{_change_text(entry)}): "
            + (f"its edits are under \"{shown_in}\" above" if shown_in
               else str(entry.get("edits")))
            + f" -- {outcome}"
            + (f" -- gate: {gate or 'not gated'}" if gated and not counted else "")
            + (" -- the pipeline now" if entry.get("id") == current and not gated else "")
            + (f" -- integrity: {entry['integrity']}" if entry.get("integrity") else "")
            + "".join(f"\n  {line}" for line in how))


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


#: How a reader is told to weigh the note: evidence about the changes it names, never a
#: rule against a change nobody tried or a different design of one that failed.
MEMORY_WEIGHT = ("Weigh it as evidence about the changes it names: follow it where the "
                 "current traces agree or say nothing, let the current traces override it, "
                 "and do not let it rule out a change nobody has tried, or a different design "
                 "(other instructions, criteria, place or limit) of a change that failed.")


def context_block(memory: Mapping[str, Any] | None) -> str:
    """The note for an analyst or merge call (SkillOpt's ``format_meta_skill_context``,
    meta_skill.py:18-30, with its "prefer it when the current evidence is ambiguous"
    replaced by `MEMORY_WEIGHT`, so an early note cannot rule out designs nobody has
    tried); empty without one."""
    text = str((memory or {}).get("text") or "").strip()
    if not text:
        return ""
    return ("## Optimizer memory\n"
            "This is optimizer-side memory distilled from earlier epochs of this run. Use it "
            f"to improve how you analyse, merge and rank findings. {MEMORY_WEIGHT}"
            f"{_stale_text(memory)}\n\n{text}\n\n")


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
        rejected = memory.get("rejected") or {}
        compared = (f"the unchanged incumbent `{memory.get('prev')}` and its "
                    f"{rejected.get('noun') or REJECTED_NOUN} `{rejected.get('child')}`, on "
                    "the train items that change moved"
                    if rejected else "the same train items under the pipeline at the start "
                    "of that epoch and after it")
        lines.append(f"Optimizer-side memory written in epoch {memory.get('epoch', '?')} from a "
                     f"comparison of {compared}. Use it to improve how you choose and write "
                     f"changes. {MEMORY_WEIGHT}")
    if stale:
        lines.append(f"NOT UPDATED in epoch {stale.get('epoch', '?')}: the memory pass failed "
                     f"({_clip(stale.get('why'), 300)})"
                     + ("; this is the earlier note, unchanged." if text
                        else "; there is no note yet."))
    tally = list(memory.get("tally") or ())
    if text and tally:
        lines.append("The note was written from this tally of the epoch's tries, so the note "
                     "can be weighed against it:")
    return [*lines, *([*tally, TALLY_NOISE_RULE] if text and tally else []),
            *([text] if text else [])]


# ── the journal and the report ──────────────────────────────────────────────


def report(passes: Sequence[Mapping[str, Any]], memory: Mapping[str, Any] | None
           ) -> dict[str, Any]:
    """What report.json says of the memory: each epoch's pass (without the note's text,
    which is in the journal) and the latest note."""
    keys = ("epoch", "prev", "curr", "same", "rejected", "sample", "pairs", "missing", "counts",
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
        now = (f"{p.get('curr')} ({(p.get('rejected') or {}).get('noun') or REJECTED_NOUN}, "
               "on the items it changed)"
               if p.get("rejected") else p.get("curr"))
        rows.append([p.get("epoch"), p.get("prev"), now,
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
