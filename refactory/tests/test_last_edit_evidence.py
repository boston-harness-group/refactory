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
    assert "reply changed: `check`, `final`; new: `alt`, `merge`" in line


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
    assert "shown whole below: i00; in one line each after them: 5." in small
    assert len(re.findall(r"(?m)^- Item ", small)) == 5
    for node in ("alt", "merge", "check", "final"):
        assert evidence.escape_headings(reply(node, "i00", "child")) in small, node
    # The parent's changed replies are cut, marked, and recorded: never silently.
    assert small.count("chars omitted]") == 2
    assert sorted(o["step"] for o in omissions) == ["check (parent)", "final (parent)"]


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
    block = section.split("#### Item i00 -- broken (child c2)\n", 1)[1].split("\n- Item ")[0]
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
    assert [o["step"] for o in omissions] == ["s (parent)"]


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
