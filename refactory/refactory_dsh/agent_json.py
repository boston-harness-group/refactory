"""Read the JSON object an agent answered with, whatever the model wrapped around it.

One reader for agent replies meant to be a JSON object: the library's own steps, and
the packages create mode writes, which import it rather than each growing a parser of
their own (packages cannot import each other, so it lives here). Its scanner,
:func:`objects`, is the judges' (`judges._objects`, which now imports it from here);
the other extractors that predate it are listed for folding in, not yet repointed.

What a model wraps around its answer, in the replies that broke a hand-rolled
``json.loads``: prose before or after it, a fenced block, and a transcript of tool calls
it never made (``<tool_call>…</tool_call>``, ``<tool_result>…</tool_result>``). A
tool-less agent told to read a file writes that transcript itself, and the first JSON
object in the reply is then the invented call, not the answer (create benchmark H3,
2026-11-20). So nothing inside a transcript is read as the answer, and a caller names
the keys its answer has (``expect``): the answer is the last top-level object outside a
transcript that has them all.

The same transcript in the reply of a step with no tools at all is the reply itself
being wrong, not noise around it: the step "ran" a check it could not run and reports
its invented result (``isprime(1999997)=True``; 1999997 = 19 x 105263; create benchmark H2).
:func:`invented_tool_calls` finds one outside fenced code blocks, for `node_retry.deliver`
to refuse such a reply.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

__all__ = ["invented_tool_calls", "objects", "parse_agent_json"]

#: A tool-call transcript written into a reply as text: a call or its result, in the
#: tag spellings models use. Never the agent's answer -- an agent that really called a
#: tool did so through its runtime, not in its reply.
TRANSCRIPT = re.compile(
    r"<(tool_call|tool_result|tool_use|invoke|function_calls?|function_results?)\b[^>]*>.*?</\1\s*>",
    re.DOTALL)

#: A code fence line (CommonMark): up to three spaces, then three or more backticks or
#: tildes. A fence closes on a line of the same character, at least as long, and nothing
#: else; an unclosed one runs to the end of the text.
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")


def _outside_fences(text: str) -> str:
    """``text`` without its fenced code blocks, fence lines included."""
    kept: list[str] = []
    fence = ""
    for line in text.splitlines(keepends=True):
        match = _FENCE.match(line)
        if not fence:
            # A backtick fence's info string has no backticks, or the line is inline code.
            if match and not (match.group(1)[0] == "`" and "`" in match.group(2)):
                fence = match.group(1)
            else:
                kept.append(line)
        elif (match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence)
              and not match.group(2).strip()):
            fence = ""
    return "".join(kept)


def invented_tool_calls(text: str) -> list[str]:
    """The tag of each tool-call transcript (`TRANSCRIPT`) ``text`` writes outside its
    fenced code blocks, in order; empty when it writes none.

    A transcript inside a fenced block is the reply showing markup, not making a call, so
    it is left alone; a transcript whose tags sit outside a fence counts even when its
    body is fenced (``<tool_call>`` then a fenced snippet then ``</tool_call>``).
    """
    return [match.group(1) for match in TRANSCRIPT.finditer(_outside_fences(text))]

def _spans(text: str) -> list[tuple[int, int, Any]]:
    """Every top-level JSON object in ``text`` with where it starts and ends."""
    decoder = json.JSONDecoder()
    found: list[tuple[int, int, Any]] = []
    index = text.find("{")
    while index >= 0:
        try:
            value, end = decoder.raw_decode(text, index)
        except ValueError:
            index = text.find("{", index + 1)
            continue
        found.append((index, end, value))
        index = text.find("{", end)
    return found


def objects(text: str) -> list[Any]:
    """Every top-level JSON object in ``text``, bare, fenced or among prose, in order.

    A brace that does not start a JSON object (prose such as ``{x}``) is skipped, and an
    object's own nested objects are part of it, not further objects.
    """
    return [value for _start, _end, value in _spans(text)]


def _within(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(low <= start and end <= high for low, high in ranges)


def parse_agent_json(text: str, *, expect: Iterable[str] = ()) -> dict[str, Any]:
    """The JSON object an agent's reply answers with: the **last** top-level object that
    has every key in ``expect`` (any object, with no ``expect``), fenced or not -- so a
    corrected final answer beats a draft, and an example given before the answer does not
    win.

    An object inside a tool-call transcript the reply wrote (:data:`TRANSCRIPT`) is never
    the answer. Objects are left out by where they are, and nothing is cut out of the text,
    so an answer whose own text mentions ``<tool_call>`` reads as written.

    Raises ``ValueError`` saying what was there when nothing qualifies, rather than
    returning a wrong object: an answer missing its keys is the agent's failure to
    report, not data.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the agent's reply is empty")
    keys = tuple(expect)
    spans = _spans(text)
    transcripts = [(match.start(), match.end()) for match in TRANSCRIPT.finditer(text)]
    replied = [(start, end, value) for start, end, value in spans
               if isinstance(value, dict) and not _within(start, end, transcripts)]
    candidates = [value for _start, _end, value in replied if all(key in value for key in keys)]
    if candidates:
        return candidates[-1]
    if replied:
        shown = "; ".join("{" + ", ".join(sorted(value)[:8]) + "}" for _s, _e, value in replied[:4])
        raise ValueError(f"no JSON object in the agent's reply has the keys {list(keys)}; "
                         f"the objects it has carry: {shown}")
    raise ValueError(f"no JSON object in the agent's reply: {' '.join(text[:400].split())!r}")
