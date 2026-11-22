"""How a reader, a step review and the integrity judge take the JSON object their reply
answers with (`random_optimizer.reply_object`): a fenced whole object first, else the
first top-level object; a broken span refuses only before any object parsed; never an
object nested inside a broken answer; no proposer-key preference (that is
`answer_json`'s)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh.agent_json import invented_tool_calls, parse_agent_json
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


def _json_says(text: str) -> str:
    """json's own message and position for *text*, as the refusal reports it.

    Derived, not written out: Python 3.12 says "Expecting value ... column 45" for the
    trailing comma below and 3.14 says "Illegal trailing comma ... column 43"; what the
    test guards is that the refusal carries json's own message and position.
    """
    import json
    import re

    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        return re.escape(f"{error.msg} at line {error.lineno} column {error.colno}")
    raise AssertionError("expected invalid JSON")


@pytest.mark.parametrize("reply,why", [
    ('{"findings": [{"where": {"step": "solve"}}, ], "verdict": "fail"}',
     _json_says('{"findings": [{"where": {"step": "solve"}}, ], "verdict": "fail"}')),
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


# ── parse_agent_json: the shared reader for an agent's JSON answer (C2) ─────────────


#: create benchmark H3's two failing replies, verbatim (2026-11-20, held-out batches ho1
#: and ho2): a tool-less agent invented a tool-call transcript before its answer, bare in
#: one and fenced mid-line in the other. The package's `json.loads` crashed on both.
H3_REPLIES = [Path(__file__).parent / "fixtures" / name
              for name in ("h3_replies_bare_after_invented_transcript.txt",
                           "h3_replies_fenced_after_invented_transcript.txt")]


@pytest.mark.parametrize("path", H3_REPLIES, ids=["bare", "fenced"])
def test_h3s_replies_are_read_past_the_invented_transcript(path):
    reply = path.read_text()
    answer = parse_agent_json(reply, expect=("replies",))
    assert [item["id"] for item in answer["replies"]] == [f"{n:03d}" for n in range(1, 13)]
    # Without the transcript dropped, the first object is the invented tool call.
    assert set(extract_json(reply)) == {"name", "arguments"}, "the replay: extract_json misreads it"
    assert parse_agent_json(reply) == answer, "the transcript never holds the answer"


def test_the_last_object_with_the_expected_keys_is_the_answer():
    """A draft, then the corrected final answer: the final one."""
    reply = ('Draft: {"replies": [{"id": "001", "reply": "draft"}]}\nOn reflection, final:\n'
             '{"replies": [{"id": "001", "reply": "final"}]}\n{"note": 1}')
    assert parse_agent_json(reply, expect=("replies",))["replies"][0]["reply"] == "final"
    assert parse_agent_json(reply) == {"note": 1}, "with no keys named, simply the last object"


def test_the_last_answer_wins_fenced_or_not():
    """An example or a draft before the answer, fenced or bare, never beats it."""
    example_first = ('The format is:\n```json\n{"replies": ["EXAMPLE"]}\n```\n'
                     'My answer: {"replies": ["real"]}')
    assert parse_agent_json(example_first, expect=("replies",)) == {"replies": ["real"]}
    bare_draft_then_fenced = ('Draft: {"replies": ["draft"]}\nFinal:\n```json\n'
                              '{"replies": ["final"]}\n```\n')
    assert parse_agent_json(bare_draft_then_fenced, expect=("replies",)) == {"replies": ["final"]}
    inline_empty_then_fenced = ('I start from {"replies": []} and fill it:\n```json\n'
                                '{"replies": ["filled"]}\n```\n')
    assert parse_agent_json(inline_empty_then_fenced, expect=("replies",)) == {"replies": ["filled"]}
    fenced_only = 'Answer:\n```json\n{"replies": ["in a fence"]}\n```\n'
    assert parse_agent_json(fenced_only, expect=("replies",)) == {"replies": ["in a fence"]}


@pytest.mark.parametrize("tag", ["tool_call", "tool_result", "tool_use", "invoke", "function_call",
                                 "function_calls", "function_result", "function_results"])
def test_every_transcript_spelling_is_left_out(tag):
    reply = f'<{tag} name="bash">{{"replies": ["invented"]}}</{tag}>\n{{"replies": ["real"]}}\n<{tag}>{{"replies": ["after"]}}</{tag}>'
    assert parse_agent_json(reply, expect=("replies",)) == {"replies": ["real"]}


@pytest.mark.parametrize("path", H3_REPLIES, ids=["bare", "fenced"])
def test_h3s_invented_transcripts_are_found(path):
    """What `node_retry.deliver` refuses from a step with no tools (D14)."""
    assert "tool_call" in invented_tool_calls(path.read_text())


@pytest.mark.parametrize("reply, found", [
    ('<function_calls><invoke name="py">isprime(7)</invoke></function_calls>\n'
     "<function_results>True</function_results>", ["function_calls", "function_results"]),
    ('<invoke name="bash">ls</invoke>', ["invoke"]),
    ('<function_call>{"name": "f", "arguments": {}}</function_call>', ["function_call"]),
    # The tags outside a fence count even when the call's body is fenced.
    ('<tool_call>\n```json\n{"name": "sympy"}\n```\n</tool_call>', ["tool_call"]),
])
def test_a_call_written_as_made_is_found(reply, found):
    assert invented_tool_calls(reply) == found


@pytest.mark.parametrize("reply", [
    "Use `<invoke>` or `<tool_call>{\"name\": \"x\"}</tool_call>` in your answer.",
    "The <tool_call> tag opens a call.\nMuch later in the text, </tool_call> closes it.",
    # JSON as a whole: a verdict quoting a person's feedback, markup and all.
    json.dumps({"verdict": "document",
                "reason": "they pasted <invoke name='bash'>ls</invoke> into design.md"}),
    'Example:\n\n    <tool_call>{"name": "x"}</tool_call>\n    <tool_result>1</tool_result>\n\nDone.',
    '> <tool_call>{"name": "x"}</tool_call>\n> <tool_result>1</tool_result>\nTheir log.',
    'Shape:\n```xml\n<tool_call>{"name": "x"}</tool_call>\n```\nDone.',
    '~~~~\n<invoke name="x">y</invoke>\n~~~\n~~~~\nafter',
    '```\n<invoke name="x">y</invoke>',
    "<tool_call>prose, not a call</tool_call>",
    "<tool_result>a result with no call before it</tool_result>",
], ids=["inline-code", "prose-tags", "json-reply", "indented-code", "quote", "fenced", "tildes",
        "unclosed-fence", "no-call-body", "result-alone"])
def test_a_reply_that_only_shows_or_names_markup_is_left_alone(reply):
    assert invented_tool_calls(reply) == []


def test_an_answer_that_mentions_a_transcript_tag_reads_as_written():
    answer = {"replies": [{"id": "001", "reply": "Your log shows <tool_call>retry</tool_call> twice."}]}
    reply = f"<tool_call>{{\"name\": \"bash\"}}</tool_call>\n{json.dumps(answer)}"
    assert parse_agent_json(reply, expect=("replies",)) == answer


def test_an_object_nested_in_another_is_never_the_answer():
    reply = '{"outer": {"replies": [1]}}'
    with pytest.raises(ValueError, match=r"has the keys \['replies'\]; the objects it has carry: \{outer\}"):
        parse_agent_json(reply, expect=("replies",))


@pytest.mark.parametrize("reply, said", [
    ("", "reply is empty"),
    ("no object here", "no JSON object in the agent's reply"),
    ('<tool_call>{"name": "bash"}</tool_call>', "no JSON object in the agent's reply"),
])
def test_a_reply_with_no_answer_is_refused(reply, said):
    with pytest.raises(ValueError, match=said):
        parse_agent_json(reply)


def test_create_tells_generated_packages_to_parse_agent_json_with_it():
    from refactory_dsh import create_ops

    assert "refactory_dsh.agent_json.parse_agent_json(text, expect=(...))" in create_ops.API


def test_the_judges_and_agent_replies_share_one_scanner():
    """One top-level object scanner, not a seventh (C0 reuse audit, D7)."""
    from refactory_dsh import agent_json, judges

    assert judges._objects is agent_json.objects
