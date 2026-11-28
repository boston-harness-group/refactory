"""Items, trajectories, evidence display for the proposal request."""


from __future__ import annotations

import itertools
import json
from typing import Any, Callable, Collection, Mapping, Sequence

from refactory_dsh.nn import edits as nn_edits

from .. import evidence
from ..evidence import CLAIMS

from .context import (
    READER_FINDINGS_SHARE,
    READER_ITEM_IDS,
    READER_PATTERN_CHARS,
    READER_PATTERNS,
    READER_QUOTES,
    READER_QUOTE_CHARS,
    TRAJECTORY_BUDGET,
)

#: The metadata keys an item's input is read from, first found; else the first
#: step's prompt stands in for it.
TRAJECTORY_INPUT_KEYS = ("question", "problem", "task_description", "instruction", "input", "prompt")


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
    from .. import evidence
    from ..analysts import metadata_text

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
    from ..analysts import metadata_text

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
    from ..analysts import metadata_text

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
        from ..analysts import metadata_text

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


# --- cross-module imports (resolved at definition time, used at call time) ---
from .composition import _TAKES_TO, _reader_edit  # noqa: E402
from .edits import _history_cell, item_errors  # noqa: E402
from .objective import _table, _table_filled, fmt_score  # noqa: E402
