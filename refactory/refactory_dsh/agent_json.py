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
2026-11-20). So the transcript is dropped before anything is read, and a caller names
the keys its answer has (``expect``): the answer is the first top-level object that has
them all.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

__all__ = ["objects", "parse_agent_json"]

#: A tool-call transcript written into a reply as text: a call or its result, in the
#: tag spellings models use. Never the agent's answer -- an agent that really called a
#: tool did so through its runtime, not in its reply.
TRANSCRIPT = re.compile(
    r"<(tool_call|tool_result|tool_use|function_calls?|function_results?)\b[^>]*>.*?</\1\s*>",
    re.DOTALL)


def objects(text: str) -> list[Any]:
    """Every top-level JSON object in ``text``, bare, fenced or among prose, in order.

    A brace that does not start a JSON object (prose such as ``{x}``) is skipped, and an
    object's own nested objects are part of it, not further objects.
    """
    decoder = json.JSONDecoder()
    found: list[Any] = []
    index = text.find("{")
    while index >= 0:
        try:
            value, end = decoder.raw_decode(text, index)
        except ValueError:
            index = text.find("{", index + 1)
            continue
        found.append(value)
        index = text.find("{", end)
    return found


def parse_agent_json(text: str, *, expect: Iterable[str] = ()) -> dict[str, Any]:
    """The JSON object an agent's reply answers with.

    Tool-call transcripts written into the reply (:data:`TRANSCRIPT`) are dropped, then
    the first top-level object (:func:`objects`) that has every key in ``expect`` is the
    answer; with no ``expect``, simply the first object.

    Raises ``ValueError`` saying what was there when nothing qualifies, rather than
    returning a wrong object: an answer missing its keys is the agent's failure to
    report, not data.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the agent's reply is empty")
    keys = tuple(expect)
    found = objects(TRANSCRIPT.sub("\n", text))
    for candidate in found:
        if all(key in candidate for key in keys):
            return candidate
    if found:
        shown = "; ".join("{" + ", ".join(sorted(item)[:8]) + "}" for item in found[:4])
        raise ValueError(f"no JSON object in the agent's reply has the keys {list(keys)}; "
                         f"the objects it has carry: {shown}")
    raise ValueError(f"no JSON object in the agent's reply: {' '.join(text[:400].split())!r}")
