"""The ops of the swe mode (`swe_mode.Swe`): SpreadsheetBench 2's agent setup around a sealed step.

::

    prepare -> has_case -> solve -> finalize
                  |                    ^
                  +------- skip -------+

`prepare` lays the session's workspace files in the pass -- the input as ``input.xlsx``
and the skill as ``SKILL.md`` -- and returns the prompt's slots (`swe_port.prompt_facts`).
The solve step works on a copy of them and leaves ``output.xlsx`` when it writes one.
`finalize` takes that file as the item's output (`tasks.PRED`), graded by V2's grade; no
``output.xlsx`` fails the item (``output-not-found``). Nothing the agent wrote is run. The
session's facts the adapter left (`swe_port.SESSION_PATH`: Claude Code's turn count, and
whether the session stopped at its turn cap) go into `tasks.FINAL` as ``n_turns`` and
``turn_limit``, and the Task's verdict says when the cap was reached; its tool calls (each
tool, its input and output cut to 2,000 characters, and whether it was an error) go into
`TOOL_CALLS` with the turn count.
"""
from __future__ import annotations

import shutil
import time
from typing import Any

from refactory_packages.spreadsheetbench import ops
from refactory_packages.spreadsheetbench import skillopt_port as port
from refactory_packages.spreadsheetbench.tasks import FINAL, PRED, TURN

from . import swe_port

#: The solve step's reply: its final message.
REPLY = "reply.txt"
#: The session's tool calls, kept in the item's folder for comparing runs step by step with
#: upstream's trajectories (`session_facts`). Neither the solver nor either arm's analysts
#: see it: no record field, view or item context reads it.
TOOL_CALLS = "tool_calls.json"
#: Why an item whose session left no output fails.
NO_OUTPUT = f"output-not-found: the agent left no {swe_port.OUTPUT_PATH}"


def prepare(ctx: Any) -> dict[str, Any]:
    """Record the item's start; write the workspace files and, for the analysts, the
    session's system prompt (upstream's system template), its prompt and the input's
    preview; return the prompt's slots."""
    started = time.time()
    item = ops._item(ctx)  # noqa: SLF001 - the solver's reader of the item
    source = ops._input(ctx)  # noqa: SLF001
    skill = str(ctx.knob("skill", "") or "")
    facts = swe_port.prompt_facts(str(item.get("instruction", "")), skill)
    if source is None:
        ctx.write_json(TURN, {"start": started, "status": "no_case"})
        return {**facts, "has_case": False}
    try:
        preview = port.preview_workbook(str(source))
    except Exception:  # noqa: BLE001 - as SkillOpt's
        preview = "(preview failed)"
    ctx._note_write(swe_port.INPUT_PATH)  # noqa: SLF001 - a declared write, copied in place
    shutil.copyfile(source, ctx._writable(swe_port.INPUT_PATH))  # noqa: SLF001
    ctx.write_text(swe_port.SKILL_PATH, skill if skill.strip() else "")
    ctx.write_text(ops.PREVIEW_FILE, preview)
    ctx.write_text(ops.SYSTEM_FILE, swe_port.SYSTEM_TEMPLATE)
    ctx.write_text(ops.USER_FILE, swe_port.render(str(item.get("instruction", "")), skill))
    ctx.write_json(TURN, {"start": started, "status": ""})
    return {**facts, "has_case": True}


def finalize(ctx: Any) -> dict[str, Any]:
    """The item's outcome before grading: the session's ``output.xlsx`` as the item's
    output, or ``output-not-found`` without one. Writes `tasks.FINAL` and the
    conversation (the final message alone). A solve step that ended without an answer
    ends the item with no answer (`ops.NoReply`), as in the agent mode; one stopped at its
    turn cap answers with the adapter's note, and its ``output.xlsx`` is graded."""
    turn = ctx.read_json(TURN, {}) or {}
    fail_reason, n_turns, capped = "", 0, {}
    if turn.get("status") == "no_case":
        fail_reason = "no-test-cases"
    else:
        if not ctx.exists(REPLY):
            raise ops.NoReply(f"the solve step left no {REPLY}: its model call ended without an "
                              "answer, so the item ends here")
        reply = ctx.read_text(REPLY)
        session = session_facts(ctx)
        n_turns = session.get("num_turns")
        if session.get("turn_limit_reached"):
            capped = {"turn_limit": session.get("max_turns")}
        if session:
            ctx.write_json(TOOL_CALLS, {"n_turns": n_turns,
                                        "turn_limit_reached": session["turn_limit_reached"],
                                        "calls": session["tool_calls"]})
        ctx.write_text(ops.RAW, reply)
        ctx.write_json(ops.CONVERSATION, [{"role": "assistant", "content": reply}])
        if ctx.exists(swe_port.OUTPUT_PATH):
            ctx._note_write(PRED)  # noqa: SLF001 - a declared write, copied in place
            shutil.copyfile(ctx.path(swe_port.OUTPUT_PATH), ctx._writable(PRED))  # noqa: SLF001
        else:
            fail_reason = NO_OUTPUT
    elapsed = time.time() - float(turn.get("start") or time.time())
    limit = ops.item_limit_s()
    ctx.write_json(FINAL, {"fail_reason": fail_reason, "n_turns": n_turns, **capped,
                           "elapsed_s": round(elapsed, 3), "backstop": elapsed > limit,
                           "limit_s": limit})
    return {"finished": True}


def session_facts(ctx: Any) -> dict[str, Any]:
    """What the adapter recorded of the session (`swe_port.SESSION_PATH`), or ``{}`` when it
    left nothing (a stand-in model). Its turn count is Claude Code's own (``num_turns``),
    ``None`` when it reported none."""
    facts = ctx.read_json(swe_port.SESSION_PATH, {}) if ctx.exists(swe_port.SESSION_PATH) else {}
    if not isinstance(facts, dict) or not facts:
        return {}
    turns = facts.get("num_turns")
    cap = facts.get("max_turns")
    calls = [call for call in facts.get("tool_calls") or () if isinstance(call, dict)]
    return {"num_turns": turns if isinstance(turns, int) and not isinstance(turns, bool) else None,
            "max_turns": cap if isinstance(cap, int) and not isinstance(cap, bool) else None,
            "turn_limit_reached": facts.get("turn_limit_reached") is True,
            "tool_calls": calls}
