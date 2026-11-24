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
    # The edits, counts, one line per item, and two items' moved replies at a fixed cap.
    assert len(section) <= 12000 + 2 * 2 * 5 * base.EDIT_REPLY_CHARS, len(section)
    rest = text.replace(section, "")
    repeated = [w for w in _windows(section) if w in rest]
    assert repeated == [], (len(repeated), repeated[0][:200])
    # Inside the section too: no window said twice.
    windows = _windows(section)
    assert len(windows) == len(set(windows))


def test_every_changed_item_is_named_once_in_one_line_and_no_trace_is_repeated():
    section = _section(context().render())
    listed = re.findall(r"(?m)^- Item (\w+) -- (\w+) \(child c2\)", section)
    assert sorted(listed) == sorted(FLIPPED.items()), "each changed item once"
    blocks = re.findall(r"(?m)^#### Replies that moved on item (\w+) \((\w+), child c2\)$",
                        section)
    assert [kind for _item, kind in blocks] == ["broken", "fixed"], blocks
    line = next(line for line in section.splitlines() if line.startswith("- Item "))
    assert ("`check` 1 -> 1 run(s), reply changed; `final` 1 -> 1 run(s), reply changed; "
            "`alt` 0 -> 1 run(s), new; `merge` 0 -> 1 run(s), new.") in line


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


@pytest.mark.parametrize("size", [3000, 6000, 8000])
@pytest.mark.parametrize("parent_shown", [True, False])
def test_no_trace_is_said_twice_at_any_reply_length(size, parent_shown):
    text = context(size, parent_shown).render()
    section = _section(text)
    rest = text.replace(section, "")
    assert [w for w in _windows(section) if w in rest] == []
    assert re.search(r"(?m)^- Item i00 -- broken", section), "every changed item is named"


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
    text = "\n".join(base._edit_change_lines([pair]))
    lines = {re.match(r"- Item (\w+)", line).group(1): line
             for line in text.splitlines() if line.startswith("- Item ")}
    assert sorted(lines) == ["b0", "b1", "b2", "b3", "b4"]
    assert "`solve` 1 -> 3 run(s)" in lines["b4"]
    assert "`solve_check`: parent not run; child reloop x2, proceed" in lines["b4"]
    assert "`solve` 1 -> 2 run(s), reply changed" in lines["b2"]
    assert "`solve_check`: parent not run; child reloop, proceed" in lines["b2"]
    for item in ("b1", "b3"):
        assert "`solve` 1 -> 1 run(s), reply changed" in lines[item]
        assert "`solve_check`: parent not run; child proceed" in lines[item]
        assert "reloop" not in lines[item]


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
