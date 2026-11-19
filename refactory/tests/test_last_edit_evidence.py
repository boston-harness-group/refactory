""""What the last edit changed" obeys the request's evidence discipline.

On live S4 runs (main 4e53948481) the section showed, for each item the last edit moved,
the parent's whole trace and the child's whole trace: 639 characters in round 1, 201,823
in round 2, and the proposal request grew from 141k to 320-428k characters; part of it
was the same trace "Trajectories" shows. Now every changed item is named, a few --
chosen by a stated rule, within a share of the evidence budget -- show the steps the
edit moved whole, and a trace shown elsewhere is named, never repeated. And a model's or
a skill's own Markdown headings, quoted in a request, never read as the request's.
"""

from __future__ import annotations

import re

import pytest

from last_edit_fixture import FLIPPED, context, reply
from refactory_dsh.optim import analysts, base, evidence, memory
from refactory_dsh.tools.request_sizes import own_headings

#: The fixture's request on main 4e53948481 was 296,762 characters (the section 138,301);
#: with this change it is about 187k (the section about 28.5k).
REQUEST_BOUND = 200_000
WINDOW = 400


def _section(text: str, heading: str = "## What the last edit changed") -> str:
    """The section under ``heading``, up to the next ``## `` heading outside a fence."""
    out, inside, fence = [], False, None
    for line in text.splitlines():
        found = re.match(r"^(`{3,})", line)
        if found and (fence is None or found.group(1) == fence):
            fence = None if fence else found.group(1)
        elif fence is None and line.startswith("## "):
            if inside:
                break
            inside = line == heading
            continue
        if inside:
            out.append(line)
    return "\n".join(out)


def _windows(text: str, size: int = WINDOW) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text) - size + 1, size)]


def test_the_s4_round_two_request_stays_bounded_and_says_no_trace_twice():
    text = context().render()
    section = _section(text)
    assert len(text) < REQUEST_BOUND, len(text)
    assert len(section) <= base.EDIT_CHANGE_SHARE * base.TRAJECTORY_BUDGET + 4000, len(section)
    rest = text.replace(section, "")
    repeated = [w for w in _windows(section) if w in rest]
    assert repeated == [], (len(repeated), repeated[0][:200])
    # Inside the section too: no window said twice.
    windows = _windows(section)
    assert len(windows) == len(set(windows))


def test_every_changed_item_is_named_and_the_ones_shown_whole_are_said():
    section = _section(context().render())
    whole = re.findall(r"(?m)^#### Item (\w+) -- (\w+) \(child c2\)$", section)
    listed = re.findall(r"(?m)^- Item (\w+) -- (\w+) \(child c2\)", section)
    assert 1 <= len(whole) <= base.EDIT_ITEMS_WHOLE
    assert sorted(whole + listed) == sorted(FLIPPED.items()), "each changed item once"
    # The rule's first picks: one broken, then one fixed.
    assert [kind for _item, kind in whole[:2]] == ["broken", "fixed"]
    count = re.search(r"(\d+) changed item\(s\); shown whole below: ([\w, ]+)", section)
    assert count and int(count.group(1)) == len(FLIPPED)
    assert count.group(2).split(", ") == [item for item, _kind in whole]
    line = next(line for line in section.splitlines() if line.startswith("- Item "))
    assert ("`check` 1 -> 1 run(s), reply changed; `final` 1 -> 1 run(s), reply changed; "
            "`alt` 0 -> 1 run(s), new; `merge` 0 -> 1 run(s), new (not shown whole: ") in line


def test_a_shown_item_has_its_moved_steps_whole_and_names_what_it_does_not_repeat():
    section = _section(context().render())
    item, kind = re.findall(r"(?m)^#### Item (\w+) -- (\w+)", section)[0]
    block = section.split(f"#### Item {item} -- ")[1].split("\n#### ")[0]
    # The new and changed steps' child replies, whole; the parent's trace is under
    # "Trajectories", so named there; the unchanged step only named.
    for node in ("alt", "merge", "check", "final"):
        assert evidence.escape_headings(reply(node, item, "child")) in block, node
        assert reply(node, item, "parent")[:200] not in block
    assert "Same reply in parent and child: `solve`." in block
    assert (f'Parent\'s reply: as under "Trajectories of the candidate being edited" below, '
            f"item {item}, step `check`.") in block
    assert f'Input: as under "Trajectories of the candidate being edited" below, item {item}.' in block


def test_a_trace_not_shown_elsewhere_is_shown_once_and_a_sibling_names_it():
    """Without "Trajectories" holding the parent, the parent's changed replies are shown;
    a sibling with the same parent names them instead of showing them again."""
    from last_edit_fixture import child_records, parent_records

    pair = {"child": "A", "parent": "p", "edit": [], "parent_records": parent_records(),
            "child_records": child_records()}
    text = "\n".join(base._edit_change_lines([pair, {**pair, "child": "B"}], room=10 ** 6))
    shown = re.findall(r"(?m)^#### Item (\w+) -- \w+ \(child (\w)\)", text)
    assert ("i00", "A") in shown and ("i00", "B") in shown
    parent_check = evidence.escape_headings(reply("check", "i00", "parent"))
    assert text.count(parent_check) == 1
    assert text.count(evidence.escape_headings(reply("alt", "i00", "child"))) == 1
    assert "Child's reply: as under child A's item i00 in this section, step `alt`." in text
    windows = _windows(text)
    assert len(windows) == len(set(windows))


def test_the_first_pick_is_shown_whatever_the_room_and_only_the_parent_s_side_is_cut():
    from last_edit_fixture import child_records, parent_records

    pair = {"child": "c", "parent": "p", "edit": [], "parent_records": parent_records(),
            "child_records": child_records()}
    omissions: list = []
    small = "\n".join(base._edit_change_lines([pair], omissions=omissions, room=100))
    # The first broken and the first fixed item, whatever the room.
    assert "shown whole below: i00, i03; in one line each after them: 4." in small
    assert len(re.findall(r"(?m)^- Item ", small)) == 4
    for item in ("i00", "i03"):
        for node in ("alt", "merge", "check", "final"):
            assert evidence.escape_headings(reply(node, item, "child")) in small, node
    # The parent's changed replies are cut, marked, and recorded: never silently.
    assert small.count("chars omitted]") == 4
    assert sorted(o["step"] for o in omissions) == [
        "check (parent)", "check (parent)", "final (parent)", "final (parent)",
        *["what the last edit changed: over ceiling"] * 2,
        *["what the last edit changed: over room"] * 2]
    assert small.count("chars left of the 100-char room") == 2


def test_a_later_pick_too_large_is_skipped_for_a_smaller_one():
    """Unequal blocks: the second pick does not fit what is left, the third does."""
    def rec(item: str, passed: bool, text: str) -> dict:
        return {"id": item, "passed": passed, "score": float(passed),
                "exchanges": [{"node": "s", "prompt": "p", "response": text}]}

    sizes = {"a": 2000, "b": 20000, "c": 2000}
    pair = {"child": "c1", "parent": "p", "edit": [],
            "parent_records": [rec(i, True, f"OLD {i} " + "o" * n) for i, n in sizes.items()],
            "child_records": [rec(i, False, f"NEW {i} " + "n" * n) for i, n in sizes.items()]}
    text = "\n".join(base._edit_change_lines([pair], room=12000))
    assert "shown whole below: a, c; in one line each after them: 1." in text
    assert "- Item b -- broken (child c1), score 1 -> 0:" in text


def _bare_headings(text: str, planted: set[str]) -> list[str]:
    """Each planted heading line of a step's reply (evidence) that appears in ``text`` as
    a line of its own, unescaped, in or out of a fence."""
    return [line for line in text.splitlines() if line.strip() in planted]


def test_quoted_headings_never_read_as_the_request_s_at_any_level():
    text = context().render()
    planted = {line for n in range(12) for node in ("solve", "check", "final", "alt", "merge")
               for side in ("parent", "child")
               for line in reply(node, f"i{n:02d}", side).splitlines() if line.startswith("#")}
    assert _bare_headings(text, planted) == []
    assert not {h.strip() for h in own_headings(text)} & planted


def test_escaping_keeps_the_words_and_only_disarms_the_markers():
    text = "## Answer\n  #### Deep\n####### seven is no heading\n#tag\nx ## y"
    escaped = evidence.escape_headings(text)
    assert escaped.splitlines() == ["\\## Answer", "  \\#### Deep",
                                    "####### seven is no heading", "#tag", "x ## y"]
    assert re.sub(r"\\(?=#)", "", escaped) == text, "only the markers change"


def test_a_copyable_text_is_fenced_byte_exact_so_nothing_in_it_is_markdown():
    """A heading, a setext underline and a run of backticks inside: the fence is longer
    than any run, so no line closes it, and the text is the same bytes."""
    text = "## Rules\nTitle\n===\nSub\n---\n````\ncode\n````\n~~~"
    block = evidence.exact_block(text)
    assert block[1] == text and block[0] == block[2] == "`" * 5
    assert own_headings("\n".join(block)) == []
    assert evidence.quoted("one line ## x") == "one line ## x"
    assert evidence.quoted(text) == "\n" + "\n".join(block) + "\n"


def test_analysts_and_memory_quote_model_text_with_headings():
    """A finding's proposed prompt and the memory note keep their layout, their headings
    and fences quoted, in the proposer's and the analysts' requests alike."""
    finding = {"source": "failure", "finding": "F\n## Rules\n```\nx", "items": ["i00"],
               "change": "Rewrite:\n### Steps\n- a"}
    lines = analysts.section_lines({"findings": [finding], "batches": []})
    assert own_headings("\n".join(lines)) == ["## What the analysts found, ranked"]
    note = {"text": "## Lessons\n- keep the check\n```", "epoch": 2}
    exact = "\n".join(evidence.exact_block(note["text"]))
    for text in ("\n".join(memory.request_lines(note)), memory.context_block(note)):
        assert own_headings(text) == ["## Optimizer memory"]
        assert exact in text and "\\#" not in text, "byte-exact: the next note gains no escape"


def test_knob_texts_are_byte_exact_and_fenced_for_every_reader():
    value = "## Role\nYou solve.\n#### Rules\n- one\n```\nx\n```"
    knobs = [{"name": "k", "value": value, "kind": "str", "frozen": False}]
    for exact in (True, False):
        text = "\n".join(base._settings_lines(knobs, exact=exact))
        assert "\n".join(evidence.exact_block(value)) in text and "\\#" not in text
        assert own_headings(text) == ["## Each step's settings"]


def test_a_step_reviewer_s_example_and_note_are_quoted():
    from refactory_dsh.optim import feedback

    entry = {"step": "s", "candidate": "c", "kind": "insert", "shared": 3, "fixed": 1,
             "broken": 0, "ran": True,
             "review": {"verdict": "adds", "example": "## Proof\nstep", "note": "### Why\nit"}}
    text = "\n".join(feedback.step_review_lines([entry]))
    assert own_headings(text) == ["## Did the added steps contribute?"]
    assert "\n## Proof\nstep\n" in text and "\n### Why\nit\n" in text, "byte-exact"
    visits = feedback.visit_text({"ran": 2, "fired": {"items": ["a"]}, "silent": {},
                                  "distinct_outputs": 1, "example": "## Out\nTitle\n---"})
    assert own_headings(visits) == [] and "\n## Out\nTitle\n---\n" in visits


@pytest.mark.parametrize("size", [6000, 8000])
def test_with_long_structural_traces_the_first_pick_still_shows_its_moved_replies(size):
    """The reviewer's case: 6-8k replies, the parent under no other section. The first
    pick shows the child's moved replies whole and the parent's, whole or cut and marked."""
    text = context(size, parent_shown=False).render()
    section = _section(text)
    block = section.split("#### Item i00 -- broken (child c2)\n", 1)[1].split(
        "\n- Item ")[0].split("\n#### ")[0]
    for node in ("alt", "merge", "check", "final"):
        assert evidence.escape_headings(reply(node, "i00", "child", size)) in block, node
    for node in ("check", "final"):
        head = evidence.escape_headings(reply(node, "i00", "parent", size))[:300]
        assert head in block, node
    whole = evidence.escape_headings(reply("check", "i00", "parent", size)) in block
    assert whole or block.count("chars omitted]") == 2
    assert len(block) <= 4 * size + 2 * base.EDIT_PARENT_FLOOR + 3000


def test_one_item_of_40k_a_side_is_shown_its_child_side_whole():
    big = 40000
    parent = {"id": "x", "passed": True, "score": 1.0, "prediction": "1",
              "exchanges": [{"node": "s", "prompt": "p", "response": "P " + "p" * big}]}
    child = {"id": "x", "passed": False, "score": 0.0, "prediction": "2",
             "exchanges": [{"node": "s", "prompt": "p", "response": "C " + "c" * big}]}
    omissions: list = []
    text = "\n".join(base._edit_change_lines([{"child": "c", "parent": "p", "edit": [],
                                               "parent_records": [parent],
                                               "child_records": [child]}],
                                             omissions=omissions))
    assert "#### Item x -- broken (child c)" in text and "C " + "c" * big in text
    assert "P " + "p" * big not in text and "chars omitted]" in text
    assert [o["step"] for o in omissions] == ["s (parent)",
                                              "what the last edit changed: over room"]


def test_every_reference_points_back_to_a_text_shown_whole():
    """With "Trajectories" showing only some items (8k replies), each reference names an
    item shown whole there; one to this section names a block printed before it."""
    text = context(8000).render()
    trajectories = _section(text, base.TRAJECTORIES_HEADING)
    assert "Not every item is shown in full here" in trajectories, "setup: a subset"
    section = _section(text)
    for item in re.findall(r'as under "Trajectories of the candidate being edited" below, '
                           r"item (\w+)", section):
        assert re.search(rf"(?m)^### Item {item} — ", trajectories), item
    from last_edit_fixture import child_records, parent_records

    pair = {"child": "A", "parent": "p", "edit": [], "parent_records": parent_records(),
            "child_records": child_records()}
    siblings = "\n".join(base._edit_change_lines([pair, {**pair, "child": "B"}], room=10 ** 6))
    for match in re.finditer(r"as under child (\w)'s item (\w+) in this section", siblings):
        target = f"#### Item {match.group(2)} -- "
        assert siblings.index(target + "") < match.start()
        assert f"(child {match.group(1)})" in siblings[:match.start()]


def test_a_finding_s_proposed_prompt_reaches_the_proposer_byte_exact():
    """Analyst text -> finding -> proposer: a proposed prompt with headings arrives as the
    same bytes, in a fence, so a proposer copying it into a knob adds no escape."""
    proposed = "## Rules\n- check the bound\n#### Steps\n1. solve\nTitle\n---"
    analysis = {"findings": [{"source": "failure", "finding": "F", "items": ["i00"],
                              "change": proposed}], "batches": [], "group_size": 8}
    text = context(analysis=analysis).render()
    found = _section(text, "## What the analysts found, ranked")
    assert "\n" + "\n".join(evidence.exact_block(proposed)) + "\n" in found
    assert "\\#" not in found
    assert own_headings(found) == []


@pytest.mark.parametrize("size", [3000, 6000, 8000])
@pytest.mark.parametrize("parent_shown", [True, False])
def test_no_trace_is_said_twice_at_any_reply_length(size, parent_shown):
    text = context(size, parent_shown).render()
    section = _section(text)
    rest = text.replace(section, "")
    assert [w for w in _windows(section) if w in rest] == []
    assert re.search(r"(?m)^#### Item i00 -- broken", section), "the first pick is shown"


def test_a_cut_to_nothing_keeps_neither_end():
    text = "HEAD" + "x" * 100 + "TAIL"
    assert evidence.clip(text, 0) == "\n[… 108 chars omitted]\n"
    assert evidence.clip(text, 2) == "H\n[… 106 chars omitted]\nL"


def test_the_memory_pass_reads_its_previous_note_byte_exact():
    note = "## Lessons\n- keep the check\nTitle\n---"
    side = {"id": "c1", "pipeline": "(pipeline)", "rollout": "train"}
    request = memory.MemoryRequest(epoch=3, mode_name="m", prev=side, curr=side, changes=[],
                                   previous_note=note, entries=[])
    text = request()[0]
    assert "\n".join(evidence.exact_block(note)) in text and "\\#" not in text
    assert "## Lessons" not in own_headings(text)


def test_a_first_pick_past_the_room_is_said_and_recorded():
    """8k replies: the first broken item's four moved replies alone nearly fill the 30k
    room, and the first fixed one is shown too. Each block past what is left says so,
    with its size, and the run's omissions record it."""
    ctx = context(8000)
    section = _section(ctx.render())
    said = re.findall(r"Item (\w+)'s block is ([\d,]+) chars, past the ([\d,]+) chars left "
                      r"of the 30,000-char room: as the first pick of its kind it is shown "
                      r"all the same", section)
    assert [item for item, _size, _left in said] == ["i00", "i03"], said
    record = [o for o in ctx.omissions if o["step"].endswith("over room")]
    assert [(o["item"], f"{o['block']:,}", f"{o['room_left']:,}") for o in record] == said
    assert all(o["room"] == 30000 and o["chars"] == 0 for o in record)
    assert len(re.findall(r"(?m)^#### Item ", section)) == 2
    small = context(3000)
    assert "chars left of the" not in _section(small.render())
    assert not [o for o in small.omissions if o["step"].endswith("over room")]


@pytest.mark.parametrize("size", [6000, 8000])
@pytest.mark.parametrize("parent_shown", [True, False])
def test_one_broken_and_one_fixed_item_are_shown_whole_at_realistic_sizes(size, parent_shown):
    """The replay's loss: with 4-step traces only one item fit, always a broken one, and
    no fixed item's replies reached the request. The first of each kind is shown."""
    section = _section(context(size, parent_shown).render())
    whole = re.findall(r"(?m)^#### Item (\w+) -- (\w+) \(child c2\)$", section)
    assert [kind for _item, kind in whole] == ["broken", "fixed"], whole
    for item, _kind in whole:
        for node in ("alt", "merge", "check", "final"):
            assert evidence.escape_headings(reply(node, item, "child", size)) in section


def _loop_record(item: str, passed: bool, relooped: int, side: str) -> dict:
    solves = 1 + relooped
    exchanges = [{"node": "solve", "prompt": f"Solve {item}.",
                  "response": f"{side} solve {item} attempt {k}"} for k in range(solves)]
    verdicts = ["reloop"] * relooped + ["proceed"] if side == "child" else []
    return {"id": item, "passed": passed, "score": float(passed), "prediction": "7",
            "exchanges": exchanges,
            "steps": [{"node": "solve_check", "kind": "gate", "verdict": v} for v in verdicts]}


def test_a_one_line_item_says_its_steps_runs_and_its_gate_s_verdicts():
    """The S1-s45 loss: a loop's judge was blamed for broken items it never sent back.
    Each one-line item says how often each moved step ran and what the gate answered."""
    looped = {"b0": 1, "b1": 0, "b2": 1, "b3": 0, "b4": 2}
    pair = {"child": "c", "parent": "p", "edit": [{"op": "loop", "step": "solve"}],
            "parent_records": [_loop_record(i, True, 0, "parent") for i in looped],
            "child_records": [_loop_record(i, False, r, "child") for i, r in looped.items()]}
    text = "\n".join(base._edit_change_lines([pair], room=0))
    lines = {re.match(r"- Item (\w+)", line).group(1): line
             for line in text.splitlines() if line.startswith("- Item ")}
    assert sorted(lines) == ["b1", "b2", "b3", "b4"], "b0, the first pick, is shown whole"
    assert "`solve` 1 -> 3 run(s)" in lines["b4"]
    assert "`solve_check`: parent not run; child reloop x2, proceed" in lines["b4"]
    assert "`solve` 1 -> 2 run(s), reply changed" in lines["b2"]
    assert "`solve_check`: parent not run; child reloop, proceed" in lines["b2"]
    for item in ("b1", "b3"):
        assert "`solve` 1 -> 1 run(s), reply changed" in lines[item]
        assert "`solve_check`: parent not run; child proceed" in lines[item]
        assert "reloop" not in lines[item]
    assert ("Gates whose verdicts moved: `solve_check`: parent not run; child reloop, proceed."
            in text)


def test_at_20k_replies_the_forced_picks_share_the_ceiling_never_below_the_floor():
    """20k replies: the first broken and fixed items alone would be ~170k. They share the
    section's ceiling (3 rooms), each child reply cut head and tail, no shorter than the
    floor, each cut marked and recorded, and the section says so."""
    ctx = context(20000)
    section = _section(ctx.render())
    blocks = re.split(r"(?m)^(?=#### Item )", section.split("\n\n- Item ")[0])[1:]
    assert [b.split(" -- ")[0] for b in blocks] == ["#### Item i00", "#### Item i03"]
    ceiling = base.EDIT_CEILING_ROOMS * 30000
    assert sum(len(b) for b in blocks) <= ceiling + 2000
    said = re.findall(r"its child's 4 moved replies cut to ([\d,]+) chars each \(head and tail, "
                      r"marked\) to keep the blocks shown whole within the section's "
                      rf"{ceiling:,}-char ceiling", section)
    assert len(said) == 2 and all(int(n.replace(",", "")) >= base.EDIT_CHILD_FLOOR for n in said)
    child_cuts = [o for o in ctx.omissions if o["step"] in ("alt", "merge", "check", "final")]
    assert len(child_cuts) == 8 and section.count("chars omitted]") >= 8


@pytest.mark.parametrize("size", [3000, 8000])
def test_graded_kinds_are_shown_whole_only_when_they_fit(size):
    """All four kinds: the first broken and fixed items are forced; a score rise or drop
    only when it fits what is left, else one line saying why."""
    section = _section(context(size, graded=True).render())
    whole = re.findall(r"(?m)^#### Item (\w+) -- ([\w ]+) \(child c2\)$", section)
    assert [kind for _i, kind in whole][:2] == ["broken", "fixed"]
    graded = [line for line in section.splitlines()
              if line.startswith(("- Item i07", "- Item i09"))]
    shown = {item for item, _kind in whole}
    assert len(graded) + len(shown & {"i07", "i09"}) == 2
    assert all("(not shown whole: its block is " in line for line in graded)
    if size == 8000:
        assert not shown & {"i07", "i09"}, "no forced graded pick past the room"


def test_a_cut_of_a_few_hundred_chars_is_not_made():
    def rec(item: str, passed: bool, text: str) -> dict:
        return {"id": item, "passed": passed, "score": float(passed),
                "exchanges": [{"node": "s", "prompt": "p", "response": text}]}

    old = "O" * 3000
    pair = {"child": "c", "parent": "p", "edit": [], "parent_records": [rec("a", True, old)],
            "child_records": [rec("a", False, "N" * 2000)]}
    whole = "\n".join(base._edit_change_lines([pair], room=10 ** 6))
    block = whole[whole.index("#### Item a"):]
    # A room 300 short of the block: the parent's 3,000 chars would lose under 500.
    omissions: list = []
    text = "\n".join(base._edit_change_lines([pair], omissions=omissions,
                                             room=len(block) + 2 - 300))
    assert old in text and "chars omitted" not in text
    assert [o["step"] for o in omissions] == ["what the last edit changed: over room"]
    # 800 short: the cut is made, marked and recorded, and the block then fits.
    omissions = []
    text = "\n".join(base._edit_change_lines([pair], omissions=omissions,
                                             room=len(block) + 2 - 800))
    assert old not in text and "chars omitted]" in text
    assert [o["step"] for o in omissions] == ["s (parent)"]


def test_a_step_that_runs_less_often_reads_fewer_runs_not_gone():
    def rec(passed: bool, solves: int, extra: bool) -> dict:
        exchanges = [{"node": "solve", "prompt": "p", "response": f"r{k} {passed}"}
                     for k in range(solves)]
        if extra:
            exchanges.append({"node": "polish", "prompt": "p", "response": "polished"})
        return {"id": "a", "passed": passed, "score": float(passed), "exchanges": exchanges}

    facts = base.step_facts(rec(True, 3, True), rec(False, 1, False))
    assert facts == "`solve` 3 -> 1 run(s), reply changed; `polish` 1 -> 0 run(s), gone"
    only_fewer = base.step_facts(rec(True, 2, False), {**rec(True, 1, False), "exchanges": [
        {"node": "solve", "prompt": "p", "response": "r0 True"}]})
    assert only_fewer == "`solve` 2 -> 1 run(s), fewer runs"


def test_a_reloop_refused_at_its_cap_is_not_counted_as_run():
    """The walker and the spine mark a reloop the cap refused (`items.refused_at_cap`);
    the request says so instead of counting one more reloop than ran."""
    parent = {"id": "a", "passed": True, "score": 1.0, "prediction": "1",
              "exchanges": [{"node": "solve", "prompt": "p", "response": "first"}],
              "steps": [{"node": "check", "kind": "gate", "verdict": "proceed"}]}
    child = {"id": "a", "passed": False, "score": 0.0, "prediction": "2",
             "exchanges": [{"node": "solve", "prompt": "p", "response": "first"},
                           {"node": "solve", "prompt": "p", "response": "second"}],
             "steps": [{"node": "check", "kind": "gate", "verdict": "reloop"},
                       {"node": "check", "kind": "gate", "verdict": "reloop",
                        "exhausted": True, "cap": 1}]}
    facts = base.step_facts(parent, child)
    assert ("`check`: parent proceed; child reloop, reloop (refused: cap of 1 reached), "
            "1 rerun taken") in facts
    assert "`solve` 1 -> 2 run(s), more runs" in facts


def test_of_two_parent_replies_cut_to_one_length_the_one_barely_over_stays_whole():
    parent = {"id": "a", "passed": True, "score": 1.0,
              "exchanges": [{"node": "s", "prompt": "p", "response": "S" * 3000},
                            {"node": "t", "prompt": "p", "response": "T" * 2300}]}
    child = {"id": "a", "passed": False, "score": 0.0,
             "exchanges": [{"node": "s", "prompt": "p", "response": "s"},
                           {"node": "t", "prompt": "p", "response": "t"}]}
    omissions: list = []
    block = base._moved_block(parent, child, "broken", "c", base._Quoted({}), omissions,
                              None, 2000)
    assert "T" * 2300 in block and "S" * 3000 not in block
    assert [o["step"] for o in omissions] == ["s (parent)"]


def test_forced_blocks_past_the_ceiling_at_the_floor_are_said_and_recorded_honestly():
    """Twelve moved steps of 20k each: even at the 8,000-char floor the first broken item
    passes the ceiling. It is shown, and the section says by how much and why; no line
    claims it was kept within the ceiling."""
    nodes = [f"s{k}" for k in range(12)]

    def rec(passed: bool, side: str) -> dict:
        return {"id": "a", "passed": passed, "score": float(passed),
                "exchanges": [{"node": n, "prompt": "p", "response": f"{side} {n} " + "x" * 20000}
                              for n in nodes]}

    pair = {"child": "c", "parent": "p", "edit": [], "parent_records": [rec(True, "old")],
            "child_records": [rec(False, "new")]}
    omissions: list = []
    text = "\n".join(base._edit_change_lines([pair], omissions=omissions, room=30000))
    found = re.search(r"the blocks shown whole are past the section's 90,000-char limit: "
                      r"([\d,]+) chars, because each of its 12 moved replies keeps at least "
                      r"8,000 chars or is whole\.", text)
    assert found and int(found.group(1).replace(",", "")) > 90000
    assert "within the section's 90,000-char ceiling" not in text
    assert "its child's 12 moved replies cut to 8,000 chars each" in text
    past = [o for o in omissions if o["step"].endswith("over ceiling")]
    assert past == [{"item": "a", "step": "what the last edit changed: over ceiling", "chars": 0,
                     "blocks": int(found.group(1).replace(",", "")), "ceiling": 90000,
                     "child_floor": 8000}]


def test_a_refused_reloop_is_said_on_its_own_and_reruns_once_per_gate():
    """Nested loops: the inner gate refused twice at its cap of 1, a rerun between them.
    Each refusal is said where it happened; the reruns it took are said once."""
    def visit(verdict: str, refused: bool = False) -> dict:
        return {"node": "inner", "kind": "gate", "verdict": verdict,
                **({"exhausted": True, "cap": 1} if refused else {})}

    record = {"steps": [visit("reloop"), visit("reloop", True), visit("Reloop"),
                        visit("reloop", True), visit("proceed")]}
    assert base._verdicts(record, "inner") == (
        "reloop, reloop (refused: cap of 1 reached), Reloop, reloop (refused: cap of 1 "
        "reached), proceed, 2 reruns taken")
