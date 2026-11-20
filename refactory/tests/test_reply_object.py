"""How a reader, a step review and the integrity judge take the JSON object their reply
answers with (`random_optimizer.reply_object`): a fenced whole object first, else the
first top-level object; a broken span refuses only before any object parsed; never an
object nested inside a broken answer; no proposer-key preference (that is
`answer_json`'s)."""

from __future__ import annotations

import pytest

from refactory_dsh.optim.random_optimizer import answer_json, extract_json, reply_object


def test_a_fenced_answer_beats_a_prose_example_before_it():
    reply = ('Shape: {"verdict": "..."}\n\nMy answer:\n```json\n'
             '{"verdict": "reloop", "why": "the bound is wrong"}\n```')
    assert reply_object(reply) == {"verdict": "reloop", "why": "the bound is wrong"}
    assert reply_object(reply) == extract_json(reply)  # as main took it


@pytest.mark.parametrize("reply", [
    '{"verdict": "pass"}\n\nFor example {"a": 1,} would not parse.',
    '{"verdict": "pass"}\nI used {" to start.',
    '{"verdict": "pass"}\nA cut-off example: {"a": [1, 2',
])
def test_a_broken_fragment_after_a_valid_answer_does_not_undo_it(reply):
    assert reply_object(reply) == {"verdict": "pass"}
    assert reply_object(reply) == extract_json(reply)


def test_the_first_object_wins_whatever_keys_a_later_one_has():
    reply = '{"verdict": "pass"}\nAn aside: {"proposal": {"edits": []}}'
    assert reply_object(reply) == {"verdict": "pass"}
    assert reply_object(reply) == extract_json(reply)


@pytest.mark.parametrize("reply,why", [
    ('{"findings": [{"where": {"step": "solve"}}, ], "verdict": "fail"}',
     "Expecting value at line 1 column 45"),
    ('Here: {"patterns": [{"where": {"step": "a"}}', "is never closed"),
])
def test_a_broken_answer_is_refused_and_no_object_nested_in_it_is_taken(reply, why):
    with pytest.raises(ValueError, match=why) as refused:
        reply_object(reply)
    assert "the answer's JSON object is not valid JSON" in str(refused.value)


def test_braces_in_prose_before_the_answer_are_skipped():
    assert reply_object('Steps {a, b} matter.\n{"verdict": "pass"}') == {"verdict": "pass"}


def test_no_object_and_no_reply_are_said():
    with pytest.raises(ValueError, match="no JSON object"):
        reply_object("I could not decide.")
    with pytest.raises(ValueError, match="returned nothing"):
        reply_object("  ")


def test_a_fenced_object_with_text_after_it_in_the_fence_is_taken_as_main_took_it():
    reply = 'Plan: {"proposals": [\n```json\n{"verdict": "pass"} and {\n```'
    assert reply_object(reply) == {"verdict": "pass"} == extract_json(reply)


def test_a_broken_fenced_answer_is_refused_not_read_for_a_nested_object():
    reply = '```json\n{"findings": [{"where": {"step": "a"}}, ], "verdict": "x"}\n```'
    with pytest.raises(ValueError, match="not valid JSON"):
        reply_object(reply)


# A reply that quotes a code fence inside its strings, as SpreadsheetBench's analysts do
# ("a single ```python``` block"): the fence in a string is not a fence (OL-s42, where
# two analysts and the failure merge were refused for it).
_QUOTED = ('{"findings": [{"finding": "returns a single ```python``` block",\n'
           '  "where": {"nodes": ["solve"], "knobs": ["skill"]},\n'
           '  "evidence": ["```python\\nimport openpyxl\\n```"]},\n'
           ' {"finding": "keeps INPUT_PATH", "where": {"nodes": ["solve"]}}]}')


@pytest.mark.parametrize("reply", [
    "```json\n" + _QUOTED + "\n```",  # fenced, as the analysts answer
    _QUOTED,  # bare, as the merges answer
    # bare, with a lone quoted fence first: the old split read the "where" object
    _QUOTED.replace("a single ```python``` block", "an unclosed ``` fence"),
    "My answer:\n```json\n" + _QUOTED + "\n```\nDone.",
])
def test_a_fence_quoted_inside_a_string_does_not_cut_the_answer(reply):
    for read in (reply_object, extract_json):
        found = read(reply)
        assert len(found["findings"]) == 2
        assert found["findings"][1]["finding"] == "keeps INPUT_PATH"


def test_a_proposer_answer_quoting_a_fence_in_a_fenced_reply_is_read_whole():
    answer = ('{"proposals": [{"edits": [{"op": "set_knob", "knob": "skill", '
              '"value": "Return one ```python``` block:\\n```python\\nx = 1\\n```"}]}]}')
    found, problem = answer_json("```json\n" + answer + "\n```")
    assert problem == ""
    assert found["proposals"][0]["edits"][0]["knob"] == "skill"


@pytest.mark.parametrize("reply", [
    '```json{\n  "findings": [{"finding": "x"}],\n  "verdict": "ok"\n}\n```',
    '```{\n  "findings": [{"finding": "x"}],\n  "verdict": "ok"\n}\n```',
    '```json {"findings": [\n {"finding": "x"}],\n "verdict": "ok"}\n```',
])
def test_an_answer_starting_on_the_opening_fence_line_is_read_whole(reply):
    for read in (reply_object, extract_json):
        assert read(reply) == {"findings": [{"finding": "x"}], "verdict": "ok"}
