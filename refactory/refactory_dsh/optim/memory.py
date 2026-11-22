"""The memory pass: the run's record of every change tried, for the model proposer, over a graph.

Once per epoch, from epoch 2, before the epoch's proposal, the loop writes ONE note: a
factual record, the same format for every task and graph (`RECORD_FORMAT`, JSON,
`render_record`). It is the loop's cross-round memory, shown to every analyst call,
every merge call and the proposal request (`context_block`, `request_lines`). It never
edits the graph and gives no advice: it records what happened.

- **The facts are the harness's** (`record_facts`, from the run's journal via
  `Loop._tried_in_round` over every round so far): for each change tried, what it was
  (its step, kind and edits, with the text it gave a new step or judge, where a new step
  sits, how a structural change moved its items), its outcome -- kept, not kept or
  refused -- and why (the gate's word, a failed train run, a refusal's reasons), and the
  train items it fixed and broke, by id; and the train items wrong under every change
  that scored them (``keeps_failing``), with those changes. No model writes them.
- **The observations are the model's**: one optimizer-model call compares two rollouts
  on the same train items and says what the traces show happened on them, each
  observation citing its items and, when it is about one change, that change's id. Its
  prompt (`MEMORY_SYSTEM`) asks for nothing forward-looking -- no next steps, principles,
  rules, recommendations or claims about what can be built -- and the answer is checked:
  an observation that is advice (`advice_in`) or names no compared item is stripped and
  journaled as stripped (`note_from`). Earlier passes' observations are carried, so the
  record holds every pass's.

The comparison is SkillOpt's meta-skill mechanism (``skillopt/optimizer/meta_skill.py``,
``slow_update.py``, ``skillopt/engine/trainer.py:2088-2199``), which this pass keeps:

- **When**: at the end of every epoch from the second (trainer.py:2097-2103 skips the
  first); the note it writes is read in the next epoch (trainer.py:1190-1194). Our
  loop's epoch ``e`` scores and gates the proposals epoch ``e - 1`` wrote and then
  proposes, so its pass runs after the gate and before the proposal: the note is read
  by the proposal it precedes and by every later one until the next pass.
- **Which two pipelines**: the candidate that was current when the previous epoch
  proposed and the one current now (the validation gate's current, or without the gate
  the train leader). When they are one candidate (nothing kept this epoch), the two
  rollouts are that candidate's previous-epoch rollout and this epoch's under the gate,
  and without it its rollout when it was scored and a fresh one made for the pass; or,
  when a change tried this epoch moved items, the incumbent against that change.
- **Which items**: ``slow_update_samples`` (20, default.yaml:99) train items, the train
  ids shuffled with ``seed + epoch * 2000`` (trainer.py:2118-2119, as the slow update's
  sample, :1843-1844). Rolled out with the epoch's own train seed rather than SkillOpt's
  ``seed + epoch * 2000``, so the loop's own rollouts of the epoch are reused; only the
  sampled items a side has no record of are evaluated.
- **What is shown**: both pipelines, what the score measures (`base.objective_text`),
  every change the optimizer tried in the epoch with its id (`tried_lines`: the facts
  the proposal request already shows, never a validation score) and a tally per change,
  and the comparison by category (slow_update.py:287-349): a summary, then regressions,
  persistent failures, improvements, each item with both outcomes, answers, grader
  reasons and both full traces, then stable successes without traces. No earlier note:
  the record is the harness's to carry. SkillOpt never clips, and neither does this
  pass: when the whole request fits ``MEMORY_ROOM`` (the analysts' room, tied to the
  optimizer model's context) it is one call; when it does not, the evidence is split
  (`MemoryRequest`): comparison calls each read the pipelines and as many whole item
  pairs as fit and write observations (``COMPARE_SYSTEM``), then one synthesis call reads
  both pipelines, the tries and every part's observations and writes the note's
  observations (``SYNTHESIS_SYSTEM``). A larger room makes fewer calls; no trace is cut,
  except a pair longer than the room on its own (counted). A call refused as longer than
  the model's context is retried once at half size.
- **What it writes**: ``{"observations": [{"change", "items", "observation",
  "evidence"}]}``, at most ``NOTE_OBSERVATIONS``, together within ``MEMORY_CHARS``; the
  rendered record within ``RECORD_CHARS``, the oldest entries left out first and named.

SkillOpt's own meta skill (the SkillOpt strategy, ``refactory_packages/skillopt``) is
not this module: a custom strategy keeps its own and runs no pass (`Loop._memory_on`).
Where a pass makes no note (a failed call), the loop keeps the previous note and marks
it not updated in that epoch, and why.

Domain-free: the prompts name no dataset or task. Train records only.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Mapping, Sequence

from . import evidence
from .evidence import exact_block
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
    ("regressed", True, "Regressions (right -> wrong)"),
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
#: The most observations one note call (the single call, or a split pass's synthesis)
#: writes; together they are asked to stay within ``MEMORY_CHARS`` and cut past it.
NOTE_OBSERVATIONS = 12
#: The record's format, the first key of every note (`render_record`).
RECORD_FORMAT = "refactory-memory-record/1"
#: The most characters the rendered record takes: past it the oldest changes and the
#: oldest observations not tied to a change are left out, oldest first, and their ids
#: named under ``omitted`` (they stay in the journal's records).
RECORD_CHARS = 20000


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
You add observations to the record of a run that optimizes an AI pipeline: a graph of
steps (nodes) connected by edges, each step configured by settings (knobs) such as its
prompt. Each change the optimizer tried was evaluated on train items and then kept or
not kept, or it was refused before it ran.

The harness writes the record's facts itself, from the run's journal: what each change
was, whether it was kept, not kept or refused and why, and the train items it fixed and
broke. You do not write or restate those facts.

Your job is to say what the traces show happened on the compared items: for an item a
change fixed or broke, and for an item that failed under both pipelines, what the steps
did and where its outcome was decided (a step's reply, a setting, the data passed along
an edge, the output of a step or judge a change added), citing the items and quoting
the traces briefly.

## What You Receive

1. The pipeline(s) compared and what changed between them.
2. Every change the optimizer tried this epoch, each with its id and the harness's facts.
3. A comparison of the SAME train items under the two pipelines (or two rollouts of one
   pipeline): each changed or failing item with both full traces.

## Rules

- Describe what happened, in the past tense, from the traces in front of you.
- Every observation names the items it covers (ids from the comparison) and, when it is
  about one change, that change's id.
- Write nothing about what to do next: no recommendations, suggestions, principles,
  rules, lessons, priorities or risks to guard against, and nothing a change should,
  could or must do, keep or avoid.
- Make no claim about what can or cannot be built, edited, reached or unfrozen.
- Do not judge a kind of change in general, and write no instructions for the
  pipeline's steps.
- An observation that breaks a rule, or names no item of the comparison, is removed
  before the record is saved.

You reply with JSON only."""

#: The memory system prompt for the synthesis call of a split pass: the same rules, the
#: comparison read through the comparison calls' observations.
SYNTHESIS_SYSTEM = MEMORY_SYSTEM.replace(
    "3. A comparison of the SAME train items under the two pipelines (or two rollouts of one\n"
    "   pipeline): each changed or failing item with both full traces.",
    "3. A comparison of the SAME train items under the two pipelines (or two rollouts of one\n"
    "   pipeline), read in parts by comparison calls: their observations, each citing its\n"
    "   items. Combine the observations about the same thing; keep their items and quotes.",
).replace("- Describe what happened, in the past tense, from the traces in front of you.",
          "- Describe what happened, in the past tense, from the observations in front of you.")

COMPARE_SYSTEM = """\
You read evidence for the record of a run that optimizes an AI pipeline: a graph of
steps (nodes) connected by edges, each step configured by settings (knobs) such as its
prompt.

You read one part of a comparison: the SAME train items under two pipelines (or two
rollouts of one pipeline), each changed or failing item with both full traces. Another
call reads your observations, with those of the other parts, and adds them to the
record. The record's facts (what each change was, whether it was kept, the items it
fixed and broke) are written by the harness, not by you.

## What You Receive

1. The pipeline(s) and what changed between the two (or that nothing did).
2. Part of the comparison: items that regressed, persistently failed, improved or
   stayed right, with both outcomes, the grader's reasons and both traces.

## Rules

- Say what the traces show happened on these items: what the steps did and where each
  outcome was decided (a step, a setting, the data passed along an edge, the structure).
- Describe what happened, in the past tense. Cite the items and quote the traces briefly.
- Patterns shared by several items first; say when one holds for a single item.
- Write nothing about what to do next: no recommendations, suggestions, principles,
  rules, lessons, priorities or risks to guard against, and nothing a change should,
  could or must do, keep or avoid.
- Make no claim about what can or cannot be built, edited, reached or unfrozen.
- Do not judge a kind of change in general, and write no instructions for the
  pipeline's steps.

You reply with JSON only."""


def compare_answer_text() -> str:
    return ("Respond ONLY with a valid JSON object:\n"
            "{\n"
            '  "observations": [{"category": "regressed|persistent_fail|improved|stable_success",\n'
            '                    "observation": "<what the traces show happened, past tense>",\n'
            '                    "items": ["<item id>", ...],\n'
            '                    "evidence": ["<a short verbatim quote from a trace>", ...]}]\n'
            "}\n"
            f"At most {OBSERVATIONS_PER_CALL} observations, the most important first, together "
            f"at most {OBSERVATION_CHARS} characters; every one names the items it covers "
            "(ids from this part).")


#: What makes an observation advice, or a claim about what can be built, rather than a
#: record of what happened (`advice_in`): read on its text only, never on its quotes
#: (a trace may say anything). Task-general: no word here names a dataset or a step.
ADVICE_PATTERNS = (
    r"\bshould(?:n't| not)?\b", r"\bought to\b", r"\bmust\b", r"\bneeds? to\b",
    r"\brecommend", r"\bsuggest", r"\badvis", r"\bconsider(?:ing)?\b", r"\bprefer",
    r"\bavoid", r"\bprioriti[sz]", r"\bguard against\b", r"\btry\b", r"\bgoing forward\b",
    r"\bnext (?:time|round|epoch|attempt|proposal|change)\b", r"\bin (?:the )?future\b",
    r"\bfuture (?:optimizer|proposals?|changes?|rounds?|epochs?|edits?|calls?)\b",
    r"\bprinciples?\b", r"\brule of thumb\b", r"\blessons?\b", r"\bbest practice",
    r"\btakeaway", r"\bworth (?:trying|adding|keeping|testing|revisiting)\b",
    r"\b(?:would|could|might) (?:likely |probably )?(?:fix|help|improve|work better)\b",
    r"\bit (?:would|might|could) be (?:better|worth|useful)\b",
    r"\b(?:can(?:not|'t)?|could(?:n't| not)?) (?:not )?be (?:built|edited|changed|added|"
    r"reached|unfrozen|given|modified|moved)\b",
    r"\bonly (?:unfrozen|editable|reachable|changeable|movable)\b", r"\bunfrozen\b",
    r"\b(?:un)?reachable\b", r"\b(?:is|are) frozen\b", r"\b(?:un)?buildable\b",
)
_ADVICE = re.compile("|".join(ADVICE_PATTERNS), re.I)


def advice_in(text: str) -> str | None:
    """The first words of ``text`` that make it advice or a claim about what can be built
    (`ADVICE_PATTERNS`), or ``None`` when it only records what happened."""
    found = _ADVICE.search(str(text or ""))
    return found.group(0) if found else None


def parse_observations(payload: Any) -> list[dict[str, Any]] | None:
    """A comparison call's observations (at most ``OBSERVATIONS_PER_CALL``, each with
    its category, text, items and up to ``EVIDENCE_PER_OBSERVATION`` quotes), or
    ``None`` when the answer carries no ``observations`` list. Texts are kept whole; an
    observation that is advice (`advice_in`) is dropped, as the record's own are."""
    listed = payload.get("observations") if isinstance(payload, Mapping) else None
    if not isinstance(listed, list):
        return None
    out = []
    for raw in listed:
        if not isinstance(raw, Mapping):
            continue
        text = str(raw.get("observation") or "").strip()
        if not text or advice_in(text):
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
            '  "observations": [{"change": "<the id of the change it is about, from \\"What was '
            'tried this epoch\\", or \\"\\" when it is about the comparison as a whole>",\n'
            '                    "items": ["<item id>", ...],\n'
            '                    "observation": "<what the traces show happened on these '
            'items, past tense>",\n'
            '                    "evidence": ["<a short verbatim quote from a trace>", ...]}]\n'
            "}\n"
            f"At most {NOTE_OBSERVATIONS} observations, together at most {chars} characters; "
            "every one names the items it covers (ids from the comparison). An empty list is "
            "an answer.")


# ── the record: the harness's facts, the model's observations ──────────────


def item_ids(parent_records: Sequence[Mapping[str, Any]] | None,
             child_records: Sequence[Mapping[str, Any]] | None) -> dict[str, list[str]]:
    """Which train items a change moved, by id, on the items it and its parent were both
    scored on (`base.edit_change`, an item the harness timed out on left out): ``fixed``,
    ``broken``, ``failing`` (wrong under the change: broken, or wrong under both) and
    ``shared``. The ids behind the fixed/broken counts the ledger and the tally show."""
    from .base import edit_change

    moved = edit_change(list(parent_records or ()), list(child_records or ()))
    parent = _by_item(list(parent_records or ()))
    child = _by_item(list(child_records or ()))
    timed = set(moved["timed_out"])
    shared = sorted(item for item in set(parent) & set(child) if item not in timed)
    return {"fixed": list(moved["fixed"]), "broken": list(moved["broken"]),
            "failing": [item for item in shared if not child[item].get("passed")],
            "shared": shared}


def outcome(entry: Mapping[str, Any], *, gated: bool | str, current: str | None
            ) -> tuple[str, str]:
    """A tried change's outcome for the record, ``kept`` or ``not kept``, and why, from the
    harness's own words: not evaluated, its train run failed, or the gate's word
    (``kept``/``current``/``in pool``/``left pool`` are kept); without a gate, whether it
    is the train leader (``current``) now."""
    from .feedback import KEPT_WORDS

    if not entry.get("evaluated", True):
        return "not kept", "not evaluated"
    if entry.get("exit_ok") is False:
        return "not kept", "its train run failed"
    gate = entry.get("gate")
    if gated:
        return ("kept" if gate in KEPT_WORDS else "not kept"), f"gate: {gate or 'not gated'}"
    if current is not None and str(entry.get("id")) == str(current):
        return "kept", "no validation gate: it is the train leader now"
    return "not kept", "no validation gate: it is not the train leader now"


def record_facts(tried: Sequence[Mapping[str, Any]], refused: Sequence[Mapping[str, Any]], *,
                 gated: bool | str, current: str | None) -> dict[str, Any]:
    """The record's facts, from the harness's own data only (`Loop._tried_in_round` over
    every round so far): ``changes``, one per change tried, oldest round first (in a
    round the evaluated ones, then the refused ones) -- its id, round and parent, its
    kind and the steps it names, what it was (`edits_text`: the text it gave a new step
    or judge, a knob's changed lines), where a step it added sits and how a structural
    change moved its items, its outcome and why (`outcome`; a refusal's reasons), and the
    train items it fixed and broke, by id, of how many it shared with its parent --
    ``None`` where its parent was not scored with the same seed; and ``keeps_failing``:
    each train item wrong under every change that scored it, at least two, with those
    changes' ids (`keeps_failing`)."""
    from .edit_effect import effect_lines

    rows = []
    for order, entry in enumerate(tried):
        result, why = outcome(entry, gated=gated, current=current)
        ids = entry.get("items") if isinstance(entry.get("items"), Mapping) else None
        rows.append((entry.get("round") or 0, 0, order, {
            "id": str(entry.get("id")), "round": entry.get("round"),
            "parent": entry.get("parent"), "kind": _change_text(entry),
            "touches": [str(n) for n in entry.get("touched") or ()],
            "edits": str(entry.get("edits") or ""),
            **({"placed": [str(line) for line in entry["placed"]]} if entry.get("placed") else {}),
            **({"effect": effect_lines(entry["effect"])}
               if isinstance(entry.get("effect"), Mapping) and effect_lines(entry["effect"])
               else {}),
            "outcome": result, "why": why,
            **({"integrity": str(entry["integrity"])} if entry.get("integrity") else {}),
            "fixed": list(ids["fixed"]) if ids else None,
            "broken": list(ids["broken"]) if ids else None,
            "of": len(ids["shared"]) if ids else None}))
    for order, entry in enumerate(refused):
        reasons = [str(r) for r in entry.get("reasons") or () if r]
        rows.append((entry.get("round") or 0, 1, order, {
            "id": None, "round": entry.get("round"), "attempt": entry.get("attempt"),
            "kind": _change_text(entry), "touches": [str(n) for n in entry.get("touched") or ()],
            "edits": str(entry.get("edits") or ""), "outcome": "refused",
            "why": " | ".join(_clip(r, TRIED_REASON_CHARS) for r in reasons[:2])
                   or "(no reason recorded)"}))
    return {"changes": [row for *_key, row in sorted(rows, key=lambda r: r[:3])],
            "keeps_failing": keeps_failing(tried)}


def keeps_failing(tried: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each train item wrong under every evaluated change that scored it with its parent
    (``items``: `item_ids`), when at least two did: ``{"item", "failed_under"}`` (those
    changes' ids, oldest first), the items failed under the most changes first."""
    scored: dict[str, list[str]] = {}
    failed: dict[str, list[str]] = {}
    for entry in tried:
        ids = entry.get("items")
        if not entry.get("evaluated", True) or not isinstance(ids, Mapping):
            continue
        failing = set(ids.get("failing") or ())
        for item in ids.get("shared") or ():
            scored.setdefault(str(item), []).append(str(entry.get("id")))
            if item in failing:
                failed.setdefault(str(item), []).append(str(entry.get("id")))
    out = [{"item": item, "failed_under": under} for item, under in scored.items()
           if len(under) >= 2 and failed.get(item) == under]
    return sorted(out, key=lambda e: (-len(e["failed_under"]), e["item"]))


def note_from(payload: Any, *, epoch: int | None = None,
              facts: Mapping[str, Any] | None = None, earlier: Mapping[str, Any] | None = None,
              items: Sequence[str] | None = None, compared: Mapping[str, Any] | None = None,
              chars: int = MEMORY_CHARS) -> dict[str, Any] | None:
    """The note a memory answer makes: the record (`render_record`) of the harness's
    ``facts`` (`record_facts`), the passes so far and every observation -- the ``earlier``
    record's, of other epochs, then this answer's. ``None`` when the answer carries no
    ``observations`` list (an empty one is an answer: the facts alone).

    Each observation of the answer is kept only when it records what happened: one that
    is advice or a claim about what can be built (`advice_in`) or that names no item of
    the comparison (``items``; any item when ``None``) is stripped, with why, as is one
    past ``NOTE_OBSERVATIONS`` or ``chars`` (its text and quotes counted). Its ``change``
    is kept when it is the id of a change in the facts, else the observation is about the
    comparison as a whole. Returns ``{"text", "reasoning", "record", "stripped"}``."""
    listed = payload.get("observations") if isinstance(payload, Mapping) else None
    if not isinstance(listed, list):
        return None
    facts = facts or {}
    known = {str(c.get("id")) for c in facts.get("changes") or () if c.get("id")}
    allowed = None if items is None else {str(i) for i in items}
    kept: list[dict[str, Any]] = []
    stripped: list[dict[str, str]] = []
    used = 0
    for raw in listed:
        if not isinstance(raw, Mapping):
            continue
        text = str(raw.get("observation") or "").strip()
        if not text:
            continue
        cited = [str(i) for i in (raw.get("items") if isinstance(raw.get("items"), list) else ())
                 if i not in (None, "")]
        cited = [i for i in dict.fromkeys(cited) if allowed is None or i in allowed]
        phrase = advice_in(text)
        if phrase or not cited:
            stripped.append({"text": text, "why": f"advice: {phrase!r}" if phrase
                             else "it names no item of the comparison"})
            continue
        change = str(raw.get("change") or "").strip().strip("`")
        quotes = [str(q) for q in (raw.get("evidence") if isinstance(raw.get("evidence"), list)
                                   else ()) if q not in (None, "")][:EVIDENCE_PER_OBSERVATION]
        size = len(text) + sum(len(q) for q in quotes)
        if len(kept) >= NOTE_OBSERVATIONS or used + size > chars:
            stripped.append({"text": text, "why": "past the note's limit"})
            continue
        used += size
        kept.append({"epoch": epoch, "change": change if change in known else None,
                     "items": cited, "text": text, "evidence": quotes})
    before = dict(earlier or {}) if (earlier or {}).get("format") == RECORD_FORMAT else {}
    record = {"format": RECORD_FORMAT, "epoch": epoch,
              "passes": [*(p for p in before.get("passes") or () if p.get("epoch") != epoch),
                         *([dict(compared)] if compared else [])],
              "changes": [dict(c) for c in facts.get("changes") or ()],
              "keeps_failing": [dict(k) for k in facts.get("keeps_failing") or ()],
              "observations": [*(o for o in before.get("observations") or ()
                                 if o.get("epoch") != epoch), *kept]}
    return {"text": render_record(record), "reasoning": "", "record": record,
            "stripped": stripped}


def parse_note(payload: Any) -> dict[str, Any] | None:
    """A memory answer read without a request's facts (`note_from`): its observations
    only. A `MemoryRequest` reads its answers with its own facts (`MemoryRequest.parse_note`)."""
    return note_from(payload)


#: The record's keys in the order the note says them.
RECORD_KEYS = ("format", "epoch", "passes", "changes", "keeps_failing", "observations",
               "omitted")


def _record_view(record: Mapping[str, Any], dropped: set[tuple[str, int]],
                 failing: int | None = None) -> dict[str, Any]:
    """The record as the note says it: each change with the observations about it,
    the observations about a comparison as a whole after them; the changes and
    observations in ``dropped`` (``("change"|"observation", index)``) left out and named,
    and ``keeps_failing`` cut to its first ``failing`` entries (counted)."""
    def shown(o: Mapping[str, Any], *keys: str) -> dict[str, Any]:
        return {key: o[key] for key in keys if o.get(key) not in (None, [], "")}

    about: dict[str, list[dict[str, Any]]] = {}
    general = []
    for index, o in enumerate(record.get("observations") or ()):
        if ("observation", index) in dropped:
            continue
        if o.get("change"):
            about.setdefault(str(o["change"]), []).append(shown(o, "epoch", "items", "text",
                                                                 "evidence"))
        else:
            general.append(shown(o, "epoch", "items", "text", "evidence"))
    changes, left_out = [], []
    for index, change in enumerate(record.get("changes") or ()):
        if ("change", index) in dropped:
            left_out.append(change.get("id") or f"refused in round {change.get('round')}, "
                                                 f"attempt {change.get('attempt')}")
            continue
        changes.append({**change, **({"observations": about[str(change.get("id"))]}
                                     if change.get("id") and str(change["id"]) in about else {})})
    failing_all = list(record.get("keeps_failing") or ())
    omitted = {**({"changes": left_out} if left_out else {}),
               **({"observations": sum(1 for kind, _i in dropped if kind == "observation")}
                  if any(kind == "observation" for kind, _i in dropped) else {}),
               **({"keeps_failing": len(failing_all) - failing}
                  if failing is not None and failing < len(failing_all) else {})}
    view = {"format": record.get("format"), "epoch": record.get("epoch"),
            "passes": list(record.get("passes") or ()), "changes": changes,
            "keeps_failing": failing_all if failing is None else failing_all[:failing],
            "observations": general}
    return {**view, **({"omitted": omitted} if omitted else {})}


def _layout(view: Mapping[str, Any]) -> str:
    """The record's JSON, one entry of a list per line, so a reader (and a diff) can find
    each change and observation, and ``json.loads`` reads it back whole."""
    parts = []
    for key in [k for k in RECORD_KEYS if k in view]:
        value = view[key]
        if isinstance(value, list) and value:
            inner = ",\n".join("  " + json.dumps(entry, ensure_ascii=False) for entry in value)
            parts.append(f" {json.dumps(key)}: [\n{inner}\n ]")
        else:
            parts.append(f" {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}")
    return "{\n" + ",\n".join(parts) + "\n}"


def render_record(record: Mapping[str, Any], chars: int = RECORD_CHARS) -> str:
    """The note: the record as JSON (`_layout`), within ``chars`` -- whole when it fits;
    else the oldest changes and the oldest observations about a comparison as a whole
    are left out, oldest first (a change by its round, such an observation by the round
    its epoch judged), then the end of ``keeps_failing``, each named or counted under
    ``omitted``. The newest change is never left out, so a record whose newest change
    alone passes ``chars`` is longer."""
    units = sorted(
        [((c.get("round") or 0), 0, ("change", i))
         for i, c in enumerate(record.get("changes") or ())]
        + [((o.get("epoch") or 1) - 1, 1, ("observation", i))
           for i, o in enumerate(record.get("observations") or ()) if not o.get("change")],
        key=lambda u: u[:2])
    changes = sum(1 for u in units if u[2][0] == "change")
    dropped: set[tuple[str, int]] = set()
    text = _layout(_record_view(record, dropped))
    for _age, _kind, unit in units:
        if len(text) <= chars:
            return text
        if unit[0] == "change" and sum(1 for k, _i in dropped if k == "change") == changes - 1:
            continue  # the newest change stays
        dropped.add(unit)
        text = _layout(_record_view(record, dropped))
    failing = len(record.get("keeps_failing") or ())
    while len(text) > chars and failing:
        failing -= 1
        text = _layout(_record_view(record, dropped, failing))
    return text


def _answer(record: Mapping[str, Any]) -> str:
    value = record.get("prediction")
    return "N/A" if value in (None, "") else _clip(value, 300)


#: What a pass calls the change it sets against the unchanged incumbent: under the
#: validation gate it was rejected; without it, it was only not the train leader; under
#: ``checkpoint`` it was not applied (its train run failed, or a sibling scored higher).
REJECTED_NOUN, NOT_KEPT_NOUN = "rejected change", "unkept change"
NOT_APPLIED_NOUN = "not-applied change"


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
    why = ("" if name == REJECTED_NOUN else
           " (it was not applied: its train run failed or a sibling scored higher on train)"
           if name == NOT_APPLIED_NOUN else " (without a validation gate the current is the "
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
                 curr: Mapping[str, Any], changes: Sequence[str],
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
                        changes=changes, entries=entries, tried=tried, objective=objective)
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
           same: bool, changes: Sequence[str], entries: Sequence[Mapping[str, Any]],
           tried: Sequence[str] = (), objective: str = "") -> tuple[str, str]:
    head = (_intro(epoch=epoch, mode_name=mode_name, prev=prev, curr=curr, same=same,
                   changes=changes, tried=tried, objective=objective)
            + f"## Longitudinal comparison (the same {len(entries)} train items, "
            + _what(curr, same) + ")\n"
            f"{_sides(prev, curr)} Each trace is the item in full: "
            f"{evidence.TRACE_FORM}. {evidence.CLAIMS}\n\n")
    return head, f"\n\n## Answer\n{answer_text()}"


#: The memory request's first line (`_intro`), without its epoch.
REQUEST_TITLE = "# Observations for the run's record"


def _intro(*, epoch: int, mode_name: str, prev: Mapping[str, Any], curr: Mapping[str, Any],
           same: bool, changes: Sequence[str], tried: Sequence[str] = (),
           objective: str = "") -> str:
    """The memory request up to its comparison: the objective, the pipelines and what
    changed, and what was tried. No earlier note: the record's facts and earlier
    observations are the harness's to carry (`note_from`), not the model's to rewrite."""
    pipelines = _pipelines_text(prev=prev, curr=curr, same=same, changes=changes, tried=tried)
    return (f"{REQUEST_TITLE} -- epoch {epoch}\n\n"
            f"Mode: `{mode_name}`.\n\n"
            + (f"## What the score measures\n{objective}\n\n" if objective else "")
            + pipelines
            + ("\n".join(tried) + "\n\n" if tried else ""))


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
                     f"{_second_pipeline(prev.get('pipeline'), curr.get('pipeline'))}\n\n"
                     f"## {changes_heading(noun(curr))}\n"
                     + ("\n".join(changes) if changes else "(not known)") + "\n\n")
    else:
        pipelines = (f"## The pipeline at the start of this epoch (candidate `{prev.get('id')}`)\n"
                     f"{prev.get('pipeline')}\n\n"
                     f"## The pipeline now (candidate `{curr.get('id')}`)\n"
                     f"{_second_pipeline(prev.get('pipeline'), curr.get('pipeline'))}\n\n"
                     f"## {changes_heading(None)}\n"
                     + ("\n".join(changes) if changes else "(not known)") + "\n\n")
    return pipelines


#: What the second pipeline's text says by a pointer when the first said it: how a
#: pipeline is drawn (`base.PIPELINE_VIEW_LEAD`) and how its settings read.
SHARED_LINE_NOTE = "(How this pipeline and its settings read is said with the pipeline above.)"


def _shared_frame() -> frozenset[str]:
    from .base import PIPELINE_VIEW_LEAD, _settings_lines

    return frozenset({PIPELINE_VIEW_LEAD, _settings_lines([], exact=False)[1]})


def _second_pipeline(first: Any, second: Any) -> str:
    """The second pipeline's text with the frame lines the first's text already holds
    (`_shared_frame`: the legend of how a pipeline is drawn, the settings' lead) said by
    one pointer, so the request says them once; its steps and settings stay whole."""
    frame = _shared_frame() & set(str(first or "").splitlines())
    if not frame:
        return str(second or "")
    out, pointed = [], False
    for line in str(second or "").splitlines():
        if line in frame:
            if not pointed:
                out.append(SHARED_LINE_NOTE)
                pointed = True
            continue
        out.append(line)
    return "\n".join(out)


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
    counted as clipped.

    ``facts`` (`record_facts`) and the ``earlier`` record are not part of any request:
    `parse_note` reads every answer into the record with them, so the facts are the
    harness's and only the observations are the model's."""

    def __init__(self, *, epoch: int, mode_name: str, prev: Mapping[str, Any],
                 curr: Mapping[str, Any], changes: Sequence[str],
                 entries: Sequence[Mapping[str, Any]], tried: Sequence[str] = (),
                 objective: str = "", facts: Mapping[str, Any] | None = None,
                 earlier: Mapping[str, Any] | None = None) -> None:
        self.epoch, self.mode_name = epoch, mode_name
        self.prev, self.curr = dict(prev), dict(curr)
        self.changes = list(changes)
        self.entries, self.tried, self.objective = list(entries), list(tried), objective
        self.facts = dict(facts or {"changes": [], "keeps_failing": []})
        self.earlier = dict(earlier or {})
        self.same = self.prev.get("id") == self.curr.get("id")
        self.look = _look(self.prev, self.curr)
        self._texts: dict[tuple[str, str], str] = {}
        self._settled = _settled(self.prev, self.curr, self.entries)

    def __call__(self, room: int = MEMORY_ROOM) -> tuple[str, dict[str, int]]:
        return build_prompt(epoch=self.epoch, mode_name=self.mode_name, prev=self.prev,
                            curr=self.curr, changes=self.changes, entries=self.entries,
                            room=room, tried=self.tried, objective=self.objective)

    def compared(self) -> dict[str, Any]:
        """This pass as the record lists it: its epoch, the two candidates, what the
        comparison sets side by side and on how many train items."""
        return {"epoch": self.epoch, "prev": self.prev.get("id"), "curr": self.curr.get("id"),
                "compared": _what(self.curr, self.same), "items": len(self.entries)}

    def parse_note(self, payload: Any) -> dict[str, Any] | None:
        """An answer (the single call's or the synthesis's) read into the record
        (`note_from`) with this pass's facts, its earlier record and its compared items."""
        return note_from(payload, epoch=self.epoch, facts=self.facts, earlier=self.earlier,
                         items=[str(e["id"]) for e in self.entries], compared=self.compared())

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
                            entries=self.entries, tried=self.tried, objective=self.objective)
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
        return (f"# Comparing two rollouts for the run's record -- epoch {self.epoch}, "
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
                f"{evidence.TRACE_FORM}. {evidence.CLAIMS}\n\n"
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
        """The request that writes the note's observations from the comparison calls'
        ``results`` (each ``{"part", "items", "observations"}``) and what it cut: the same
        request as the single call's -- the objective, both pipelines and what changed,
        what was tried -- with every part's observations in place of the
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
                           same=self.same, changes=self.changes, tried=self.tried,
                           objective=self.objective)
                    + f"## Longitudinal comparison (the same {len(self.entries)} train items, "
                    + _what(self.curr, self.same) + ")\n"
                    f"{_sides(self.prev, self.curr)} Over all {len(self.entries)} items: "
                    f"{_tally_text(self.entries)}. Both full traces of every item did not "
                    f"fit one request, so {len(results)} comparison calls each read part of "
                    "them whole (every item in exactly one part, regressions first, then "
                    "persistent failures, improvements and stable successes) and wrote these "
                    "observations. They are your evidence from the comparison.\n\n"
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
    cut = f"{head} {_middle(part, rest)}" if head else _middle(part, room)
    if len(cut) <= room:
        return cut
    # Too little room even for the cut's marker: keep the marker (and the head) whole and
    # drop the text before it, rather than slicing the marker off.
    mark = f"…[{len(part)} chars, middle cut]…"
    return f"{head} {mark}" if head else mark


#: The heading of the per-change tally (`change_tally_lines`).
TALLY_HEADING = "Each change tried this epoch, against the noise"


def change_tally_lines(tried: Sequence[Mapping[str, Any]], *, val_items: int = 0,
                       spread: float | None = None, gated: bool | str = False) -> list[str]:
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


#: What the proposer is told beside the tally (the memory call judges no change, so it
#: has no such rule).
TALLY_NOISE_RULE = ("Each line is one change: its result is evidence about that change -- "
                    "its step, what it was told to do or check, where it sends work back, "
                    "its limits -- not about other changes of its kind. A result inside "
                    "that noise is weak evidence even about the change itself.")


def tried_lines(tried: Sequence[Mapping[str, Any]], refused: Sequence[Mapping[str, Any]], *,
                round_: int | None, current: str | None, gated: bool | str,
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
             + ("; `gate` says whether it was applied (kept, not applied, or current: the "
                "pipeline now)" if gated == "checkpoint" else
                "; `gate` is what the validation gate decided (kept, rejected, or current: "
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


def _tried_line(entry: Mapping[str, Any], current: str | None, gated: bool | str,
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


def changes_lines(lineage: Mapping[str, Any],
                  homes: Mapping[str, tuple[Mapping[str, str], str]] | None = None
                  ) -> list[str]:
    """What separates the start-of-epoch candidate from the current one (``{"common",
    "prev", "curr"}``, `Loop._lineage_edits`): from their common ancestor, each
    candidate on the way to either with its own edits, oldest first; said plainly when
    they are on different branches or share no ancestor. ``homes`` (side -> the knob
    texts of that side's pipeline as the request shows them, `knob_homes`, and its
    heading): a text one of those knobs holds is named by its head and the knob, not
    repeated, so the request says it once."""
    from .base import _knob_home, _map_text, text_pointer

    def lines(path: Sequence[Mapping[str, Any]], side: str = "curr") -> list[str]:
        held, heading = (homes or {}).get(side) or ({}, "")

        def point(text: str) -> str:
            name = _knob_home(text, held) if held else None
            return (text_pointer(text, f"in `{name}` under {heading}'s settings")
                    if name is not None else text)

        out = []
        for step in path:
            edits = json.dumps(_map_text(list(step.get("edits") or []), point), default=str)
            out.append(f"- `{step.get('id')}` (from `{step.get('parent')}`): "
                       f"{_clip_middle(edits, 4000)}"
                       + (f" -- rationale: {_clip(step.get('rationale'), 400)}"
                          if step.get("rationale") else ""))
        return out

    common, before, after = lineage.get("common"), list(lineage.get("prev") or ()), list(
        lineage.get("curr") or ())
    if common is None:
        return ["They share no ancestor in this run. The start-of-epoch candidate's lineage:",
                *lines(before, "prev"), "The current candidate's lineage:", *lines(after)]
    if not before:
        return [f"The current candidate descends from the start-of-epoch one; the changes "
                "since, oldest first:", *lines(after)]
    if not after:
        return ["The current candidate is an earlier version of the start-of-epoch one: "
                "these changes, oldest first, made the start-of-epoch candidate from it, and "
                "are not in the current one:", *lines(before, "prev")]
    return [f"They are on different branches from their common ancestor `{common}`. "
            "From it to the start-of-epoch candidate:", *lines(before, "prev"),
            "From it to the current candidate:", *lines(after)]


def pipeline_homes(state: Mapping[str, Any]) -> dict[str, str]:
    """The knob texts a pipeline's text (`analysts.pipeline_text`) shows whole
    (`base.knob_homes`, at most ``PIPELINE_TEXT_CHARS`` each), for `changes_lines`."""
    from .base import knob_homes

    return {name: value for name, value in knob_homes(state_view(state).knobs).items()
            if len(value) <= PIPELINE_TEXT_CHARS}


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


#: How every reader is told what the note is: the harness's facts and the traces'
#: observations, a record and not advice (the same words for every task and graph).
RECORD_LEAD = ("It is this run's record, as JSON, of every change tried so far. The harness "
               "wrote each change's facts from the run's journal: what it was, whether it was "
               "kept, not kept or refused and why, and the train items it fixed and broke, by "
               "id (`fixed`/`broken`, of the `of` items it shared with its parent); "
               "`keeps_failing` lists each train item wrong under every change that scored it, "
               "with those changes. Each observation says what the traces of a memory pass "
               "showed on the items it names. It records what happened and says nothing about "
               "what to try.")


def context_block(memory: Mapping[str, Any] | None) -> str:
    """The note for an analyst or merge call, under SkillOpt's heading for its meta skill,
    said as the record it is (`RECORD_LEAD`); empty without one."""
    text = str((memory or {}).get("text") or "").strip()
    if not text:
        return ""
    return ("## Optimizer memory\n"
            f"{RECORD_LEAD}{_stale_text(memory)}\n\n" + "\n".join(exact_block(text)) + "\n\n")


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
        lines.append(f"Written in epoch {memory.get('epoch', '?')}; that epoch's observations "
                     f"come from a comparison of {compared}. {RECORD_LEAD}")
    if stale:
        lines.append(f"NOT UPDATED in epoch {stale.get('epoch', '?')}: the memory pass failed "
                     f"({_clip(stale.get('why'), 300)})"
                     + ("; this is the earlier note, unchanged." if text
                        else "; there is no note yet."))
    tally = list(memory.get("tally") or ())
    if text and tally:
        lines.append("The tally of that epoch's tries:")
    return [*lines, *([*tally, TALLY_NOISE_RULE] if text and tally else []),
            *(exact_block(text) if text else [])]


# ── the journal and the report ──────────────────────────────────────────────


def report(passes: Sequence[Mapping[str, Any]], memory: Mapping[str, Any] | None
           ) -> dict[str, Any]:
    """What report.json says of the memory: each epoch's pass (without the note's text,
    which is in the journal) and the latest note."""
    keys = ("epoch", "prev", "curr", "same", "rejected", "sample", "pairs", "missing", "counts",
            "evaluated", "rollouts", "calls", "parts", "context_retry", "clipped_items",
            "clipped_chars", "synthesis_cuts", "chars", "observations", "stripped", "updated",
            "error", "skipped")
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
        lines += [f"The latest record (epoch {latest.get('epoch')}):", "",
                  *(f"> {line}" if line else ">" for line in str(latest["text"]).splitlines()),
                  ""]
    return lines
