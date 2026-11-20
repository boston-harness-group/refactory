"""The proposal request fits its room whatever the items' sizes, and a looped step's
history is shown once.

SpreadsheetBench OL-s42/OL-s44 (2026-11-20): from epoch 2 on every proposal call failed
"Prompt is too long" (requests of 1.0-2.0M characters against a 400k room), and the retry at
half length re-raised because nothing could give way. Two causes:

- a step run 30 times in a loop had, as each run's prompt, the whole conversation so far,
  and each run's prompt was shown in full: one item's text grew with the square of its runs
  (1.2M characters for one item under "What the last edit changed");
- the first item "Some items in full" shows, and the first broken and fixed picks of
  "What the last edit changed", were kept whatever their size.

Now a step run again shows only what is new since the run of it that fed it: what it repeats
exactly is said in place as those lines of what the request shows of that run -- its
prompt's block, its reply as shown -- every other line in place, so each step's block
reads back from the rendered request alone. And the items' texts give way to the room:
cut to fit (their longest texts first, marked and recorded), or named. Nothing here is
specific to a task: any step, any loop.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from refactory_dsh.optim import ProposalContext, analysts, base, evidence
from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer
from test_proposal_trajectories import (_CITING, NODES, _full, _livemath_sized, _ReaderClient,
                                        _record, _TooLongOnce)

TASK = ("Task: total the sales column for each year and write each year's total into the "
        "summary table on the second sheet, leaving every other cell as it is. The table "
        "has one row per year; a year with no sales gets 0.")


def _reply(k: int, size: int = 0) -> str:
    """A reply with short lines among long ones (code), distinct per run, padded to about
    ``size`` characters."""
    lines = [f"Attempt {k}: reading the sheet again and summing the rows per year.",
             "```python", "for row in rows:", "    if not row:", "        continue",
             f"    totals[row.year] += row.amount * {k}  # weight {k}", "```",
             f"I wrote the totals for attempt {k} into the summary table.", "Done."]
    pad, n = [], 0
    while len("\n".join(lines + pad)) < size:
        pad.append(f"note {k}.{n}: checked row block {n} of attempt {k} against the source.")
        n += 1
    return "\n".join(lines[:-1] + pad + lines[-1:])


def _feedback(k: int) -> str:
    return (f"Attempt {k} gave the wrong total in cell B{k + 1}: expected {k * 10}, got "
            f"{k * 11}.\nPlease check the rows again.")


def _looped(item: str, turns: int, *, size: int = 0, offset: int = 0) -> dict[str, Any]:
    """One step, ``solve``, run ``turns`` times by a loop: each run's prompt is the whole
    conversation so far -- the task, then each earlier reply and the feedback on it, each
    turn labelled and indented -- as a chat history renders it."""
    history = [f"- User: {TASK}"]
    exchanges = []
    for k in range(1, turns + 1):
        reply = _reply(k + offset, size)
        exchanges.append({"node": "solve",
                          "prompt": "History:\n" + "\n".join(history)
                          + "\n\nAnswer the latest user request.",
                          "response": reply})
        history.append("- Assistant: " + reply.replace("\n", "\n  "))
        history.append("- User: " + _feedback(k + offset).replace("\n", "\n  "))
    return {"id": item, "passed": False, "score": 0.0, "prediction": "B2=11",
            "label": "B2=10", "fail_reason": "wrong total", "metadata": {"question": TASK},
            "exchanges": exchanges}


# -- 1. A looped step's repeated history is shown once, exactly -------------------------

_HEADING = re.compile(r"^#{4,6} Step `(?P<node>[^`]+)` \((?P<k>\d+) of \d+")
_BLOCK = ("Rest of its prompt:", "Also in its prompt (not in its settings):")
_REPLY = ("Reply:", "Child's reply:")
#: A repeat said in place, parsed back (`evidence._same`, `evidence._shown_span`).
_SAME = re.compile(
    r'^(?P<lead>.*?)\[same as (?P<node>[^\]]+?)\'s (?P<kind>prompt|reply) \(step (?P<k>\d+)\)'
    r'(?:, (?:lines? (?P<a>\d+)(?:-(?P<b>\d+))?'
    r'|all but its (?P<side>first|last) (?:(?P<n>\d+) )?shown lines?'
    r'|its shown lines? (?P<sa>\d+)(?:-(?P<sb>\d+))?))?'
    r'(?:, each line after the first prefixed (?P<p>"(?:[^"\\]|\\.)*"))?\]$')


def _fence_after(lines: list[str], at: int) -> list[str]:
    """The lines of the fenced block opening on ``lines[at]``, its headings unescaped
    (`evidence.escape_headings` undone)."""
    fence, out = lines[at], []
    for line in lines[at + 1:]:
        if line == fence:
            return out
        out.append(re.sub(r"^([ \t]{0,3})\\(#{1,6})(?=[ \t]|$)", r"\1\2", line))
    raise AssertionError("unclosed fence")


def _read_back(text: str) -> dict[int, list[str]]:
    """Each step's prompt block as the rendered item ``text`` alone lets it be read: every
    repeat expanded from the blocks and replies the item shows, recursively."""
    lines = text.splitlines()
    blocks: dict[int, list[str]] = {}
    replies: dict[int, list[str]] = {}
    step = None
    for at, line in enumerate(lines):
        found = _HEADING.match(line)
        if found:
            step = int(found["k"])
        elif step is not None and line in _BLOCK:
            blocks[step] = _fence_after(lines, at + 1)
        elif step is not None and line in _REPLY:
            replies[step] = _fence_after(lines, at + 1)

    def expand(line: str) -> list[str]:
        found = _SAME.match(line)
        if not found:
            return [line]
        k = int(found["k"])
        if found["kind"] == "prompt":
            shown = blocks[k]
            first, end = 0, len(shown)
            if found["side"] == "last":
                end -= int(found["n"] or 1)
            elif found["side"] == "first":
                first = int(found["n"] or 1)
            elif found["sa"]:
                first, end = int(found["sa"]) - 1, int(found["sb"] or found["sa"])
            assert not found["lead"] and not found["p"], line
            return [out for shown_line in shown[first:end] for out in expand(shown_line)]
        shown = replies[k]
        first, end = 0, len(shown)
        if found["a"]:
            first, end = int(found["a"]) - 1, int(found["b"] or found["a"])
        prefix = json.loads(found["p"]) if found["p"] else ""
        part = shown[first:end]
        return [found["lead"] + part[0], *(prefix + line for line in part[1:])]

    return {k: [out for line in block for out in expand(line)] for k, block in blocks.items()}


def _shown_without_repeats(prompt: str, settled: frozenset[str] | None) -> list[str]:
    """What a step's block shows of ``prompt`` with nothing said as a repeat (and nothing
    named as an earlier reply or the input: the fixtures here quote no such line alone):
    every line (`REST`), or, with ``settled`` (`ALSO`), those that are not blank and not
    its settings'."""
    lines = prompt.splitlines()
    if settled is not None:
        return [line for line in lines if line.strip() and evidence._norm(line) not in settled]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _views(record: dict[str, Any], settled: frozenset[str]) -> dict[str, tuple[str, Any]]:
    """The item rendered as each reader shows it: the proposer's (`base._trajectory`, its
    settings shown), an analyst's with and without settings (`analysts.item_text`),
    memory's comparison (`memory._trace`) and the last edit's block (`_moved_block`,
    against a parent whose replies all differ)."""
    from refactory_dsh.optim import memory

    parent = {**record, "passed": not record.get("passed"), "exchanges": [
        {**e, "response": "earlier " + e["response"]} for e in record["exchanges"]]}
    pair = {"child": "c", "parent": "p", "edit": [], "parent_records": [parent],
            "child_records": [record]}
    return {
        "proposer": (base._trajectory(record, settled=settled), settled),
        "analyst": (analysts.item_text(record, settled=settled), settled),
        "analyst, no settings": (analysts.item_text(record), None),
        "memory": (memory._trace({"prev": record}, "prev", False, settled), settled),
        "last edit": ("\n".join(base._edit_change_lines([pair], settled=settled,
                                                        room=10 ** 7)), settled)}


def _assert_reads_back(record: dict[str, Any], settled: frozenset[str] = frozenset()) -> None:
    """In every view, each step's block reads back, from the rendered text alone, as
    exactly what it would show with nothing said as a repeat."""
    prompts = {k: e["prompt"] for k, e in enumerate(record["exchanges"], 1)}
    for view, (text, view_settled) in _views(record, settled).items():
        # The fixtures quote no earlier reply's line on its own, which the older matching
        # would name ("Sees: solve's reply") and leave out: every name is of a repeat.
        for sees in re.findall(r"(?m)^Sees: (.*)$", text):
            assert all(re.search(r"\(step \d+\)$", name) for name in sees.split(", ")), sees
        read = _read_back(text)
        assert read, view
        for k, lines in read.items():
            assert lines == _shown_without_repeats(prompts[k], view_settled), (view, k)


def _record_of(exchanges: list[dict[str, Any]], item: str = "X1") -> dict[str, Any]:
    return {"id": item, "passed": False, "score": 0.0, "prediction": "B2=11",
            "label": "B2=10", "fail_reason": "wrong total", "exchanges": exchanges}


def _blocks_of(record: dict[str, Any]) -> dict[int, tuple[str, list[str]]]:
    """Each step's ``Sees:`` line and block lines, as an analyst's item text shows them."""
    lines = analysts.item_text(record).splitlines()
    out, step, sees = {}, None, ""
    for at, line in enumerate(lines):
        found = _HEADING.match(line)
        if found:
            step, sees = int(found["k"]), ""
        elif line.startswith("Sees: "):
            sees = line
        elif step is not None and line in _BLOCK:
            out[step] = (sees, _fence_after(lines, at + 1))
    return out


@pytest.mark.parametrize("render", ["analyst", "proposer"])
def test_a_looped_step_s_repeated_history_is_shown_once(render):
    """Each run shows only what is new since the run that fed it: the earlier
    conversation and the reply it quotes are said in place, so every reply and every
    feedback turn appears once, and the item grows with its runs, not their square."""

    def text(record: dict[str, Any]) -> str:
        if render == "analyst":
            return analysts.item_text(record)
        return base._trajectory(record, settled=frozenset())

    record = _looped("L1", 30)
    shown = text(record)
    for k in range(1, 31):
        assert shown.count(_reply(k)) == 1, k  # the reply, fenced, once
    for k in range(1, 30):  # the feedback on run k reaches run k + 1's prompt
        assert shown.count(f"Attempt {k} gave the wrong total in cell B{k + 1}") == 1, k
    for k in range(2, 31):
        assert f"Sees: solve's prompt (step {k - 1}), solve's reply (step {k - 1})" in shown
        assert f"[same as solve's prompt (step {k - 1}), all but its last" in shown
        assert (f"- Assistant: [same as solve's reply (step {k - 1}), each line after the "
                'first prefixed "  "]') in shown
    # The closing line is not part of a repeat: it is shown, in place, every run.
    assert shown.count("Answer the latest user request.") == 30
    # Linear: twice the runs is about twice the text (the prompts alone grow 4x).
    half = text(_looped("L1", 15))
    assert len(shown) < 2.3 * len(half), (len(shown), len(half))
    prompts = sum(len(e["prompt"]) for e in record["exchanges"])
    assert len(shown) < prompts / 5, (len(shown), prompts)


def test_a_looped_step_reads_back_from_what_the_request_shows():
    """Every repeat says lines the request shows -- of the earlier run's block as shown,
    of its reply as shown -- so each step's block reads back from the rendered text alone,
    in the proposer's, the analysts', memory's and the last edit's views."""
    _assert_reads_back(_looped("L1", 8))


def test_a_reply_s_lines_are_counted_as_shown():
    """A reply shown stripped of its leading blank lines: a later run quoting it whole,
    then its first six findings, says the lines of the reply as shown (findings 0-5),
    never of the reply as written (the review's repro read findings 2-7)."""
    findings = "\n".join(f"finding {k}: the column total for block {k} is off by one row"
                         for k in range(10))
    exchanges = [{"node": "solve", "prompt": "Check the totals, block by block.",
                  "response": "\n\n" + findings},
                 {"node": "solve", "response": "Fixed.", "prompt": "Your findings were:\n"
                  + findings + "\nFix these first:\n"
                  + "\n".join(findings.splitlines()[:6])}]
    record = _record_of(exchanges)
    sees, block = _blocks_of(record)[2]
    assert block == ["Your findings were:", "[same as solve's reply (step 1)]",
                     "Fix these first:", "[same as solve's reply (step 1), lines 1-6]"]
    _assert_reads_back(record)


def test_a_prompt_s_lines_are_counted_as_its_block_shows_them():
    """Four instruction lines (its settings, not shown in its block), then the history,
    then a closing line: a later run says the lines the earlier block shows, not the
    lines of the prompt as written (the review's repro said "lines 1-6" of a two-line
    block)."""
    rules = [f"Rule {k}: totals are sums of the sales column, rounded to cents ({k})."
             for k in range(4)]
    record = _looped("S1", 5)
    for exchange in record["exchanges"]:
        exchange["prompt"] = "\n".join(rules) + "\n" + exchange["prompt"]
    settled = frozenset(evidence._norm(line) for line in rules)
    text = base._trajectory(record, settled=settled)
    assert "[same as solve's prompt (step 1), all but its last shown line]" in text
    _assert_reads_back(record, settled)


def test_a_turn_repeated_from_an_earlier_turn_is_named_where_it_was_shown():
    """Feedback that repeats word for word what an earlier turn said lies, in the run that
    fed this one, inside that run's own repeat: it is named as the shown lines of the run
    whose block showed it, through the chain, and reads back exactly."""
    record = _looped("F1", 7)
    same = "\n".join(f"  Sheet1!B{k}: your output = {300 + k} (WRONG), expected {400 + k}"
                     for k in range(2, 12))
    for exchange in record["exchanges"]:
        exchange["prompt"] = re.sub(r"(Please check the rows again\.)", r"\1\n" + same.replace(
            "\\", "\\\\"), exchange["prompt"])
    text = base._trajectory(record, settled=frozenset())
    assert re.search(r"\[same as solve's prompt \(step 2\), its shown lines \d+-\d+\]", text)
    assert text.count("Sheet1!B5: your output = 305") == 1
    _assert_reads_back(record)


def test_a_reply_shown_clipped_is_named_as_shown():
    """A reply too long to show whole is shown clipped (head and tail, the cut marked); a
    later run quoting it whole names it, as shown, so the conversation is not shown again
    run after run (one clipped reply once made every later block a full copy)."""
    record = _looped("C2", 12, size=3000)
    text = analysts.item_text(record, output_chars=1000)
    for k in range(2, 12):
        assert f"[same as solve's reply (step {k}), each line after the first prefixed" in text
    half = analysts.item_text(_looped("C2", 6, size=3000), output_chars=1000)
    assert len(text) < 2.3 * len(half), (len(text), len(half))


def test_a_shared_preamble_is_never_said_as_the_earlier_task():
    """Two runs of one step share a five-line preamble and differ in their task. Unless
    the later run was fed the earlier one's reply, nothing of the earlier run is named
    (they may be on parallel paths); when it was, only the preamble's lines are, and the
    later task is shown in place: the January task is never claimed for March."""
    preamble = "\n".join(f"Rule {k}: totals are sums of the sales column of every row, "
                         f"rounded to cents, in the currency of the sheet ({k})."
                         for k in range(5))
    january = f"{preamble}\nSubtask: compute the January totals for the summary table."
    reply = "January totals: 1,204.50 in the summary table, row 2, as the rules say."
    march = f"{preamble}\nSubtask: compute the March totals for the summary table."
    alone = [{"node": "solve", "prompt": january, "response": reply},
             {"node": "solve", "prompt": march, "response": "March totals: 980.00."}]
    sees, block = _blocks_of(_record_of(alone))[2]
    assert sees == "" and "\n".join(block) == march
    fed = [alone[0], {"node": "solve", "prompt": f"{march}\nEarlier: {reply}",
                      "response": "March totals: 980.00."}]
    sees, block = _blocks_of(_record_of(fed))[2]
    assert block[0] == "[same as solve's prompt (step 1), all but its last shown line]"
    assert block[1] == "Subtask: compute the March totals for the summary table."
    _assert_reads_back(_record_of(fed))


def test_a_sliding_window_says_only_the_turns_it_kept():
    """A history of the last three turns: each run drops its oldest turn. The turns it
    kept are said as those shown lines of the earlier block, the dropped one is never
    claimed, and each block reads back exactly. (A kept turn the earlier block names only
    together with the dropped one is shown again: one turn a run, so the item still grows
    with its runs.)"""
    turns, exchanges = [], []
    for k in range(1, 9):
        exchanges.append({"node": "solve", "prompt": "Recent turns:\n" + "\n".join(
            turns[-3:]) + f"\nNow do step {k}.", "response": _reply(k)})
        turns.append("- Assistant: " + _reply(k).replace("\n", "\n> ")
                     + "\n- User: " + _feedback(k).replace("\n", "\n> "))
    record = _record_of(exchanges)
    blocks = _blocks_of(record)
    for k in range(5, 9):
        block = "\n".join(blocks[k][1])
        assert f"Attempt {k - 4} gave the wrong total" not in block  # neither shown nor named
        assert f"[same as solve's prompt (step {k - 1}), its shown lines " in block
    _assert_reads_back(record)


def test_interleaved_runs_of_one_step_name_only_the_run_that_fed_them():
    """Two paths run the same step in turn (A1, B1, A2, B2), each fed only its own
    path's last reply: each later run names its own path's run, never the other's."""
    a1 = {"node": "solve", "prompt": TASK + "\nPath A, first try.", "response": _reply(1)}
    b1 = {"node": "solve", "prompt": TASK + "\nPath B, first try.", "response": _reply(2)}
    a2 = {"node": "solve", "prompt": a1["prompt"] + "\nYou said:\n" + _reply(1)
          + "\nPath A, try again.", "response": _reply(3)}
    b2 = {"node": "solve", "prompt": b1["prompt"] + "\nYou said:\n" + _reply(2)
          + "\nPath B, try again.", "response": _reply(4)}
    record = _record_of([a1, b1, a2, b2])
    blocks = _blocks_of(record)
    assert blocks[3][0] == "Sees: solve's prompt (step 1), solve's reply (step 1)"
    assert blocks[4][0] == "Sees: solve's prompt (step 2), solve's reply (step 2)"
    _assert_reads_back(record)


def test_a_reply_two_runs_wrote_alike_names_the_run_whose_prompt_it_continues():
    """Two earlier runs wrote the same reply, so the reply alone cannot say which fed the
    later run: the one whose prompt the later prompt repeats is named (its own path's);
    when both prompts are the same too, which fed it is not certain, and neither is."""
    same = "The totals are written; the table now has one row per year, as asked."
    a1 = {"node": "solve", "prompt": TASK + "\nPath A, the first sheet.", "response": same}
    b1 = {"node": "solve", "prompt": TASK + "\nPath B, the second sheet.", "response": same}
    a2 = {"node": "solve", "prompt": a1["prompt"] + "\nYou said: " + same + "\nCheck it.",
          "response": "Checked."}
    sees, block = _blocks_of(_record_of([a1, b1, a2]))[3]
    assert sees == "Sees: solve's prompt (step 1)", sees
    tie = [a1, {**a1}, a2]
    sees, block = _blocks_of(_record_of(tie))[3]
    assert sees == "" and "[same as" not in "\n".join(block)


def test_a_first_line_that_changes_every_turn_does_not_defeat_it():
    """A counter on the first line (``Tool calls left: N``) changes every turn: the
    conversation under it is still said once, and the item grows with its runs."""

    def counted(turns: int) -> dict[str, Any]:
        record = _looped("C1", turns)
        for k, exchange in enumerate(record["exchanges"], 1):
            exchange["prompt"] = f"Tool calls left: {31 - k}\n" + exchange["prompt"]
        return record

    _assert_reads_back(counted(8))
    whole, half = base._trajectory(counted(30), settled=frozenset()), base._trajectory(
        counted(15), settled=frozenset())
    assert len(whole) < 2.3 * len(half), (len(whole), len(half))
    assert whole.count("Tool calls left: 1\n") == 1


def test_a_loop_whose_prompt_opens_with_its_last_reply_names_it():
    """A refine loop: each run's prompt is the previous run's reply, then a critique.
    The reply is said as that reply, not shown again; the critique is shown."""
    replies = [_reply(k, 600) for k in range(1, 6)]
    exchanges = [{"node": "refine", "prompt": "Draft the totals.", "response": replies[0]}]
    for k in range(2, 6):
        exchanges.append({"node": "refine",
                          "prompt": f"{replies[k - 2]}\nCritique {k}: the totals of attempt "
                                    f"{k - 1} are off by one row.",
                          "response": replies[k - 1]})
    record = {"id": "R1", "passed": True, "score": 1.0, "exchanges": exchanges}
    _assert_reads_back(record)
    text = analysts.item_text(record)
    for k in range(2, 6):
        assert f"[same as refine's reply (step {k - 1})]" in text
        assert text.count(f"Critique {k}: the totals of attempt {k - 1}") == 1
    assert all(text.count(reply) == 1 for reply in replies)


def test_a_one_line_quote_says_no_prefix():
    """A reply of one line quoted under a label: the line says the label and the reply,
    and no prefix for lines after the first (there are none)."""
    reply = ("The totals for every year are written into the summary table, rows 2 to 9, "
             "each the sum of the sales column over that year's rows, rounded to cents, and "
             "every other cell of both sheets is left as it was.")
    exchanges = [{"node": "solve", "prompt": "Write the totals.", "response": reply},
                 {"node": "solve", "prompt": f"You said:  {reply}\nNow check them.",
                  "response": "Checked."}]
    _sees, block = _blocks_of(_record_of(exchanges))[2]
    assert block[0] == "You said:  [same as solve's reply (step 1)]"


def test_a_step_that_runs_once_reads_as_before():
    """Without an earlier run there is nothing to repeat: a step's prompt is split as it
    always was (what it took from the input or an earlier step's reply, by name)."""
    seen = evidence._Seen()
    seen.add("Problem: find the value of the expression under the rules.", "the input")
    seen.add("plan line one: rewrite the expression first", "plan's reply")
    prompt = ("Problem: find the value of the expression under the rules.\n"
              "plan line one: rewrite the expression first\nYou are the check step.")
    assert seen.split(prompt, "check") == seen.split(prompt) == (
        ["the input", "plan's reply"],
        "[the input]\n[plan's reply]\nYou are the check step.")


def test_another_item_never_says_a_repeat_as_its_own():
    """Across the items of one request, a step's text holding a repeat names that item's
    own runs, so it is never said as "same as item X"; a step that ran more than once
    is named by its run."""
    one, two = _looped("A", 4), _looped("B", 4)
    texts = evidence.dedup([analysts.item_text(one), analysts.item_text(two)])
    for k in (2, 3, 4):  # each holds a repeat of B's own earlier run
        assert f"same as item A, step solve ({k} of 4)" not in texts[1]
    assert texts[1].count("[same as solve's prompt (step 3), all but its last") == 1
    looped = [{"id": f"x{i}", "passed": False, "score": 0.0, "exchanges": [
        {"node": "solve", "prompt": f"Own text of run {k}, long enough to be kept whole.",
         "response": f"reply {i}.{k}"} for k in (1, 2)]} for i in (1, 2)]
    said = evidence.dedup([analysts.item_text(r) for r in looped])[1]
    assert "same as item x1, step solve (2 of 2)." in said


def test_a_quote_keeps_its_label_as_written():
    """The text before a quoted reply on its line is kept as written, its spacing too."""
    first = {"node": "solve", "prompt": "Start the totals.", "response": _reply(1)}
    second = {"node": "solve", "response": _reply(2),
              "prompt": "Model  said:\t" + _reply(1).replace("\n", "\n    ") + "\nGo on."}
    record = _record_of([first, second])
    _sees, block = _blocks_of(record)[2]
    assert block[0] == ("Model  said:\t[same as solve's reply (step 1), each line after the "
                        'first prefixed "    "]')
    _assert_reads_back(record)


def test_the_last_edit_shows_a_looped_step_s_history_once():
    """"What the last edit changed" renders a moved run the same way: the child's
    conversation is said once, each later run naming the run that fed it."""
    parent = {**_looped("a", 12, offset=100), "passed": True, "score": 1.0}
    child = _looped("a", 12)
    pair = {"child": "c", "parent": "p", "edit": [], "parent_records": [parent],
            "child_records": [child]}
    text = "\n".join(base._edit_change_lines([pair], settled=frozenset(), room=10 ** 7))
    for k in range(1, 12):
        assert text.count(f"Attempt {k} gave the wrong total in cell B{k + 1}") == 1, k
    assert "Sees: solve's prompt (step 11), solve's reply (step 11)" in text
    assert ("- Assistant: [same as solve's reply (step 11), each line after the first "
            'prefixed "  "]') in text


# -- 2. The request fits its room whatever the items' sizes -----------------------------


def _huge(item: str, steps: int = 30, size: int = 20000) -> dict[str, Any]:
    """An item longer than the whole request's room on its own: many distinct steps,
    each long (a single text past ``ITEM_TEXT_MAX`` is cut anyway)."""
    record = _livemath_sized(item)
    record["exchanges"] = [{"node": f"s{k:02d}", "prompt": "p",
                            "response": f"s{k:02d} START " + f"{k} " * (size // 3) + " END"}
                           for k in range(steps)]
    return record


def _records_with_huge_first(huge: str = "w20") -> list[dict[str, Any]]:
    """35 LiveMath-sized items, one of them (``huge``) longer than the room on its own:
    the first the rule picks (`_CITING` cites w20 first; readers cite their group's
    items in id order, so w00 comes first)."""
    records = ([_huge(huge)] + [_livemath_sized(f"w{i:02d}") for i in range(25)
                                if f"w{i:02d}" != huge]
               + [_livemath_sized(f"p{i:02d}", passed=True) for i in range(10)])
    assert base._shown_size(records[0], None, None) > base.PROPOSAL_ROOM
    return records


def test_one_item_longer_than_the_room_still_yields_a_request_within_it():
    """The first item the rule picks is past the room on its own: it is shown with its
    longest texts cut to one length (head and tail, each cut marked and recorded), the
    request within its room, everything else in the item whole."""
    records = _records_with_huge_first()
    ctx = ProposalContext(nodes=NODES, item_records=records, reader_findings=_CITING)
    text = ctx.render()
    assert len(text) <= base.PROPOSAL_ROOM
    assert _full(text) == ["w20"]
    section = text.split("### Some items in full", 1)[1]
    note = re.search(r"Item w20 alone is longer than the (\d+) characters left for items in "
                     r"full, so its longest texts are cut to (\d+) characters each", section)
    assert note and int(note.group(2)) < 20000
    block = section[section.index("### Item w20"):]
    assert block.count("chars omitted]") == 30  # each step's reply, marked in place
    assert "Qw20 " + "q" * 1500 in block  # the input, shorter than the cut, whole
    cuts = [o for o in ctx.omissions if o["item"] == "w20"]
    assert len(cuts) == 30 and all(o["chars"] > 0 for o in cuts)


def test_an_item_that_does_not_fit_even_cut_to_the_floor_is_named():
    records = _records_with_huge_first()
    text = ProposalContext(nodes=NODES, item_records=records, reader_findings=_CITING,
                           shown_items_room=1000).render()
    assert _full(text) == []
    note = next(line for line in text.split("### Some items in full", 1)[1].splitlines()
                if "left out so this request fits" in line)
    assert f"even the first, with every text cut to {base.SHOWN_TEXT_FLOOR} characters" in note
    assert note.split(": ", 1)[1].startswith("w20 (failed), ")


def test_the_proposer_s_whole_call_fits_with_one_item_longer_than_the_room():
    client = _ReaderClient()
    optimizer = AutoresearchOptimizer(client)
    optimizer.propose(ProposalContext(nodes=NODES,
                                      item_records=_records_with_huge_first("w00")), 1, [])
    assert len(client.proposals[-1]) + len(optimizer._system) <= base.PROPOSAL_ROOM  # noqa: SLF001
    assert _full(client.proposals[-1])[0] == "w00"  # shown, cut to fit
    assert "Item w00 alone is longer than the" in client.proposals[-1]


def test_a_request_refused_as_too_long_with_a_huge_item_is_asked_again_shorter(tmp_path):
    """The run's failure: a refused request is asked again at half its length, and with
    an item past the room the half-length request is shorter (it used to be the same
    request, so the retry re-raised and the round proposed nothing)."""
    client = _TooLongOnce()
    optimizer = AutoresearchOptimizer(client)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    optimizer.propose(ProposalContext(nodes=NODES,
                                      item_records=_records_with_huge_first("w00")), 1, [])
    first, second = client.proposals[-2:]
    assert len(first) + len(optimizer._system) <= base.PROPOSAL_ROOM  # noqa: SLF001
    assert len(second) <= len(first) // 2
    rows = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    proposals = [r for r in rows if r["kind"] == "proposal"]
    assert proposals[-1].get("context_retry") is True and not proposals[-1].get("error")


def _edit_pair(runs: int = 30, size: int = 20000) -> dict[str, Any]:
    """The last edit broke an item whose step ran ``runs`` times, every run's reply moved:
    even at the section's own floors its block is past its ceiling."""

    def rec(passed: bool, side: str) -> dict[str, Any]:
        return {"id": "a", "passed": passed, "score": float(passed), "prediction": side,
                "exchanges": [{"node": "solve", "prompt": f"turn {k}",
                               "response": f"{side} {k} " + f"{side[0]}{k} " * (size // 4)}
                              for k in range(runs)]}

    return {"child": "c", "parent": "p", "edit": [], "parent_records": [rec(True, "old")],
            "child_records": [rec(False, "new")]}


def test_the_last_edit_s_block_gives_way_when_the_request_would_pass_its_room():
    """A forced pick past the room at its floors no longer keeps the request past it:
    every text of the block is cut to one length to fit what the rest leaves, said in the
    section and recorded; with too little left even at `SHOWN_TEXT_FLOOR` it is listed in
    one line, saying why."""
    pair = _edit_pair()
    unbounded = ProposalContext(nodes=NODES, item_records=[_record("w1")], edit_pairs=[pair],
                                request_room=0).render()
    assert len(unbounded) > 300000  # past its own ceiling at the floors
    room = 150000
    ctx = ProposalContext(nodes=NODES, item_records=[_record("w1")], edit_pairs=[pair],
                          request_room=room)
    text = ctx.render()
    assert len(text) <= room
    assert "#### Item a -- broken (child c)" in text
    said = re.search(r"every text in it cut to at most ([\d,]+) chars \(head and tail, "
                     r"marked\) so that this request fits", text)
    assert said and base.SHOWN_TEXT_FLOOR <= int(said.group(1).replace(",", "")) < 8000
    steps = [o["step"] for o in ctx.omissions]
    assert "what the last edit changed: cut to the request's room" in steps
    assert sum(1 for s in steps if s in ("solve", "solve (parent)")) >= 60

    tiny = ProposalContext(nodes=NODES, item_records=[_record("w1")], edit_pairs=[pair],
                           request_room=len(unbounded) - 300000 - 20000)
    text = tiny.render()
    assert "#### Item a" not in text
    line = next(line for line in text.splitlines() if line.startswith("- Item a -- broken"))
    assert f"even with every text cut to {base.SHOWN_TEXT_FLOOR} chars its block passes" in line
    assert "what the last edit changed: named only, past the request's room" in [
        o["step"] for o in tiny.omissions]


def test_items_in_full_and_the_last_edit_share_what_the_rest_leaves():
    """Both past the room: each gets an even share of what the rest of the request
    leaves (one that needs less is whole and leaves the rest to the other), so neither
    takes it all."""
    assert base._fair_shares([10, 100], 60) == [10, 50]
    assert base._fair_shares([100, 100], 60) == [30, 30]
    assert base._fair_shares([10, 20], 100) == [10, 20]
    assert base._fair_shares([0, 50], 20) == [0, 20]
    assert base._fair_shares([5, 5], -3) == [0, 0]
    records = _records_with_huge_first()
    text = ProposalContext(nodes=NODES, item_records=records, reader_findings=_CITING,
                           edit_pairs=[_edit_pair()]).render()
    assert len(text) <= base.PROPOSAL_ROOM
    assert _full(text) == ["w20"] and "#### Item a -- broken (child c)" in text


def test_a_request_within_its_room_is_not_cut():
    """The rule is unchanged when everything fits: items whole, nothing marked."""
    records = [_livemath_sized(f"w{i:02d}") for i in range(4)]
    ctx = ProposalContext(nodes=NODES, item_records=records)
    text = ctx.render()
    assert "chars omitted" not in text and ctx.omissions == []
    assert ctx.render() == ProposalContext(nodes=NODES, item_records=records,
                                           request_room=0).render()


def test_a_request_that_gives_way_still_fills_its_room():
    """Only what does not fit is cut. The last edit's broken item is the item
    "Trajectories" shows first: shown whole there, the edit's block names it; once it is
    cut there, the block shows it itself and grows. The shares are made again from what
    each render took, so the request is not cut far below its room (one pass on the
    first render's sizes left a SpreadsheetBench request at 278k of 400k)."""
    records = _records_with_huge_first("w00")
    child = records[0]
    parent = {**child, "passed": True, "score": 1.0,
              "exchanges": [{**e, "response": "old " + e["response"]}
                            for e in child["exchanges"]]}
    cites = [{"group": 1, "items": [], "success_patterns": [],
              "failure_patterns": [{"pattern": "A", "step": "s", "items": ["w00"],
                                    "evidence": ["qa"]}]}]
    pair = {"child": "c", "parent": "p", "edit": [], "parent_records": [parent],
            "child_records": [child]}
    whole = ProposalContext(nodes=NODES, item_records=records, reader_findings=cites,
                            edit_pairs=[pair], request_room=0).render()
    assert "Child's reply: as under \"Trajectories of the candidate being edited\"" in whole
    ctx = ProposalContext(nodes=NODES, item_records=records, reader_findings=cites,
                          edit_pairs=[pair])
    text = ctx.render()
    assert 0.9 * base.PROPOSAL_ROOM < len(text) <= base.PROPOSAL_ROOM, len(text)
    assert _full(text) == ["w00"] and "#### Item w00 -- broken (child c)" in text
    assert "Child's reply: as under" not in text  # cut there, so shown here


def test_a_block_named_in_one_line_leaves_its_room_to_the_items():
    """A broken item whose step ran 700 times does not fit even cut to the floor, so it
    is named in one line: it asks for no room then, and the items in full have it (the
    request was 207,873 of 400,000 characters when it kept its share)."""
    ctx = ProposalContext(nodes=NODES, item_records=_records_with_huge_first(),
                          reader_findings=_CITING, edit_pairs=[_edit_pair(runs=700)])
    text = ctx.render()
    assert 0.9 * base.PROPOSAL_ROOM < len(text) <= base.PROPOSAL_ROOM, len(text)
    left = [o["room_left"] for o in ctx.omissions if "named only" in o["step"]]
    assert left and left[0] > 0.25 * base.PROPOSAL_ROOM, left
    named = next(line for line in text.splitlines() if line.startswith("- Item a -- broken"))
    assert "not shown whole: even with every text cut to" in named
    assert _full(text) == ["w20"]
    # Said with the share it was first named at, not the limit it keeps once named.
    share = int(re.search(r"passes the ([\d,]+) chars", named).group(1).replace(",", ""))
    assert share > 0.25 * base.PROPOSAL_ROOM, named


def test_an_item_named_leaves_its_room_to_the_last_edit():
    """The mirror: an item in full too long even cut to the floor is named, and the last
    edit's block, itself past the room, has the room it leaves."""
    records = _records_with_huge_first()
    records[0] = _huge("w20", steps=1500, size=2000)
    text = ProposalContext(nodes=NODES, item_records=records, reader_findings=_CITING,
                           edit_pairs=[_edit_pair(runs=60)]).render()
    assert 0.9 * base.PROPOSAL_ROOM < len(text) <= base.PROPOSAL_ROOM, len(text)
    assert _full(text) == [] and "#### Item a -- broken (child c)" in text


def test_a_request_refused_with_only_an_oversized_last_edit_is_asked_again_shorter(tmp_path):
    """The larger part of the run's failure: the last edit's block alone past the room.
    The request fits it, and the retry at half its length is shorter still."""
    client = _TooLongOnce()
    optimizer = AutoresearchOptimizer(client)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    optimizer.propose(ProposalContext(nodes=NODES, item_records=[_record("w1")],
                                      edit_pairs=[_edit_pair(runs=60)]), 1, [])
    first, second = client.proposals[-2:]
    assert len(first) + len(optimizer._system) <= base.PROPOSAL_ROOM  # noqa: SLF001
    assert len(second) <= len(first) // 2
    assert "#### Item a -- broken (child c)" in second


def test_what_renderings_keep_lasts_one_pass_and_is_bounded():
    """What renderings read again is kept only while a request, an analysts' round or a
    memory pass is made, within `KEPT_CHARS`, and dropped after: nothing is kept outside
    one, so no record's text outlives the pass that showed it."""
    from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer

    assert evidence._KEEP["store"] is None
    analysts.item_text(_looped("K", 6))
    assert evidence._KEEP["store"] is None
    with evidence.keeping():
        store = evidence._KEEP["store"]
        analysts.item_text(_looped("K", 6))
        assert store.used > 0
    assert evidence._KEEP["store"] is None
    small = evidence._Kept(1000)
    for k in range(50):
        small.get(k, lambda: "x" * 100, len)
    assert small.used <= 1000 and len(small.held) < 50
    ProposalContext(nodes=NODES, item_records=[_looped("K", 6)]).render()
    assert evidence._KEEP["store"] is None
    for method in (AutoresearchOptimizer.write_memory, AutoresearchOptimizer._round_analysis):
        assert hasattr(method, "__wrapped__"), method  # made inside `evidence.keeping`
