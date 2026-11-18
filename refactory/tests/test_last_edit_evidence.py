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

from last_edit_fixture import FLIPPED, context, reply
from refactory_dsh.optim import analysts, base, evidence, memory
from refactory_dsh.tools.request_sizes import own_headings

#: The fixture's request on main 4e53948481 was 296,762 characters (the section 138,301);
#: with this change it is about 187k (the section about 28.5k).
REQUEST_BOUND = 200_000
WINDOW = 400


def _section(text: str, heading: str = "## What the last edit changed") -> str:
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


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


def test_items_are_chosen_to_fit_the_room_never_cut():
    from last_edit_fixture import child_records, parent_records

    pair = {"child": "c", "parent": "p", "edit": [], "parent_records": parent_records(),
            "child_records": child_records()}
    omissions: list = []
    small = "\n".join(base._edit_change_lines([pair], omissions=omissions, room=100))
    assert "shown whole below: none; in one line each after them: 6." in small
    assert len(re.findall(r"(?m)^- Item ", small)) == 6 and omissions == []
    assert "chars omitted" not in small


def _bare_headings(text: str, planted: set[str]) -> list[str]:
    """Each planted heading line that appears in ``text`` as a line of its own, unescaped,
    in or out of a fence."""
    return [line for line in text.splitlines() if line.strip() in planted]


def test_quoted_headings_never_read_as_the_request_s_at_any_level():
    text = context().render()
    planted = {line for n in range(12) for node in ("solve", "check", "final", "alt", "merge")
               for side in ("parent", "child")
               for line in reply(node, f"i{n:02d}", side).splitlines() if line.startswith("#")}
    assert _bare_headings(text, planted) == []
    assert not {h.strip() for h in own_headings(text)} & planted


def test_escaping_keeps_the_words_and_only_disarms_the_markers():
    text = "## Answer\n  #### Deep\n####### seven is no heading\n#tag\nx ## y\n```py\ncode\n```"
    escaped = evidence.escape_headings(text)
    assert escaped.splitlines()[:5] == ["\\## Answer", "  \\#### Deep",
                                        "####### seven is no heading", "#tag", "x ## y"]
    quoted = evidence.inline_quote(text)
    assert "\\```py" in quoted and quoted.count("\\```") == 2
    assert re.sub(r"\\(?=[#`])", "", quoted) == text, "only the markers change"
    assert own_headings(quoted) == []


def test_analysts_and_memory_quote_model_text_with_headings():
    """A finding's proposed prompt and the memory note keep their layout, their headings
    and fences quoted, in the proposer's and the analysts' requests alike."""
    finding = {"source": "failure", "finding": "F\n## Rules\n```\nx", "items": ["i00"],
               "change": "Rewrite:\n### Steps\n- a"}
    lines = analysts.section_lines({"findings": [finding], "batches": []})
    assert own_headings("\n".join(lines)) == ["## What the analysts found, ranked"]
    note = {"text": "## Lessons\n- keep the check\n```", "epoch": 2}
    assert own_headings("\n".join(memory.request_lines(note))) == ["## Optimizer memory"]
    assert own_headings(memory.context_block(note)) == ["## Optimizer memory"]


def test_a_reader_that_never_patches_sees_knob_headings_escaped_the_proposer_exact():
    knobs = [{"name": "k", "value": "## Role\nYou solve.\n#### Rules\n- one", "kind": "str",
              "frozen": False}]
    exact = "\n".join(base._settings_lines(knobs))
    assert "\n## Role\nYou solve.\n#### Rules\n" in exact, "a patch quotes it exactly"
    reader = "\n".join(base._settings_lines(knobs, exact=False))
    assert "\n\\## Role\n" in reader and "\n\\#### Rules\n" in reader


def test_a_step_reviewer_s_example_and_note_are_quoted():
    from refactory_dsh.optim import feedback

    entry = {"step": "s", "candidate": "c", "kind": "insert", "shared": 3, "fixed": 1,
             "broken": 0, "ran": True,
             "review": {"verdict": "adds", "example": "## Proof\nstep", "note": "### Why\nit"}}
    text = "\n".join(feedback.step_review_lines([entry]))
    assert own_headings(text) == ["## Did the added steps contribute?"]
    assert "\\## Proof\nstep" in text and "\\### Why\nit" in text
