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

Now a step run again shows only what is new since its previous run (the repeated part said
by name), and the items' texts give way to the room: cut to fit (their longest texts first,
marked and recorded), or named. Nothing here is specific to a task: any step, any loop.
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


# -- 1. A looped step's repeated history is shown once ---------------------------------


@pytest.mark.parametrize("render", ["analyst", "proposer"])
def test_a_looped_step_s_repeated_history_is_shown_once(render):
    """Each run shows only what is new since the step's previous run: the earlier
    conversation and the reply it quotes are said by name, so every reply and every
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
    for k in range(3, 31):
        assert f"Sees: solve's prompt (step {k - 1}), solve's reply (step {k - 1})" in shown
        assert f"- Assistant: [solve's reply (step {k - 1})]" in shown
    # Linear: twice the runs is about twice the text (the prompts alone grow 4x).
    half = text(_looped("L1", 15))
    assert len(shown) < 2.3 * len(half), (len(shown), len(half))
    prompts = sum(len(e["prompt"]) for e in record["exchanges"])
    assert len(shown) < prompts / 5, (len(shown), prompts)


def test_nothing_of_a_looped_step_s_conversation_is_lost():
    """Read in order, the runs say the whole conversation once: each later run's
    prompt is its previous run's prompt (named), that run's reply (named, under its turn
    label) and the new turn, shown."""
    record = _looped("L1", 6)
    seen = evidence._Seen()
    for number, exchange in enumerate(record["exchanges"], 1):
        sources, rest = seen.split(exchange["prompt"], "solve")
        if number >= 3:
            assert sources == [f"solve's prompt (step {number - 1})",
                               f"solve's reply (step {number - 1})"]
            assert rest == (f"[solve's prompt (step {number - 1})]\n"
                            f"- Assistant: [solve's reply (step {number - 1})]\n"
                            f"- User: Attempt {number - 1} gave the wrong total in cell "
                            f"B{number}: expected {(number - 1) * 10}, got "
                            f"{(number - 1) * 11}.\n  Please check the rows again.\n\n"
                            "Answer the latest user request.")
        seen.ran("solve", [(f"solve's prompt (step {number})", exchange["prompt"]),
                           (f"solve's reply (step {number})", exchange["response"])])


def test_a_loop_whose_prompt_opens_with_its_last_reply_names_it():
    """A refine loop: each run's prompt is the previous run's reply, then a critique.
    The reply is named, not shown again; the critique is shown."""
    replies = [_reply(k, 600) for k in range(1, 6)]
    exchanges = [{"node": "refine", "prompt": "Draft the totals.", "response": replies[0]}]
    for k in range(2, 6):
        exchanges.append({"node": "refine",
                          "prompt": f"{replies[k - 2]}\nCritique {k}: the totals of attempt "
                                    f"{k - 1} are off by one row.",
                          "response": replies[k - 1]})
    text = analysts.item_text({"id": "R1", "passed": True, "score": 1.0,
                               "exchanges": exchanges})
    for k in range(2, 6):
        assert f"Sees: refine's reply (step {k - 1})" in text
        assert text.count(f"Critique {k}: the totals of attempt {k - 1}") == 1
    assert all(text.count(reply) == 1 for reply in replies)


def test_a_step_that_runs_once_reads_as_before():
    """Without a previous run there is nothing to repeat: a step's prompt is split as it
    always was (what it took from the input or an earlier step's reply, by name)."""
    seen = evidence._Seen()
    seen.add("Problem: find the value of the expression under the rules.", "the input")
    seen.add("plan line one: rewrite the expression first", "plan's reply")
    prompt = ("Problem: find the value of the expression under the rules.\n"
              "plan line one: rewrite the expression first\nYou are the check step.")
    assert seen.split(prompt, "check") == seen.split(prompt) == (
        ["the input", "plan's reply"],
        "[the input]\n[plan's reply]\nYou are the check step.")


def test_the_last_edit_shows_a_looped_step_s_history_once():
    """"What the last edit changed" renders a moved run the same way: the child's
    conversation is said once, each later run naming its previous run."""
    parent = {**_looped("a", 12, offset=100), "passed": True, "score": 1.0}
    child = _looped("a", 12)
    pair = {"child": "c", "parent": "p", "edit": [], "parent_records": [parent],
            "child_records": [child]}
    text = "\n".join(base._edit_change_lines([pair], settled=frozenset(), room=10 ** 7))
    for k in range(1, 12):
        assert text.count(f"Attempt {k} gave the wrong total in cell B{k + 1}") == 1, k
    assert "Sees: solve's prompt (step 11), solve's reply (step 11)" in text
    assert "- Assistant: [solve's reply (step 11)]" in text


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
