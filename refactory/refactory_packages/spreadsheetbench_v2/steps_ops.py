"""The ops of the steps mode (`steps_mode.Steps`), on swe's code paths.

`prepare` is swe's (`swe_ops.lay_workspace`): the input as ``input.xlsx``, the skill as
``SKILL.md``, the preview and the prompts' slots; the steps' prompts are their own knobs,
so it writes no prompt files. `finalize` takes the ``output.xlsx`` in the pass after the
last step as the item's output, as swe's does, and records every step: its turns (Claude
Code's own count, from the session facts the adapter left beside its workspace,
`steps_mode.session_path`), whether it reached its turn cap, and its tool calls.
"""
from __future__ import annotations

import shutil
import time
from typing import Any

from refactory_packages.spreadsheetbench import ops
from refactory_packages.spreadsheetbench.tasks import FINAL, PRED, TURN

from refactory_dsh import workspace

from . import swe_ops, swe_port
from .steps_mode import STEPS, reply_path, session_path

#: The steps' tool calls, in step order (`swe_ops.TOOL_CALLS`: read by no record, view or
#: item context).
TOOL_CALLS = swe_ops.TOOL_CALLS
NO_OUTPUT = swe_ops.NO_OUTPUT


def prepare(ctx: Any) -> dict[str, Any]:
    """Record the item's start; write the steps' workspace files and the input's preview;
    return the prompts' slots."""
    return swe_ops.lay_workspace(ctx, prompts=False)


def finalize(ctx: Any) -> dict[str, Any]:
    """The item's outcome before grading: the ``output.xlsx`` the steps left as the item's
    output, or ``output-not-found`` without one. Writes `tasks.FINAL` with each step's
    turns (``step_turns``, ``None`` where unknown) and their total (``n_turns``, the sum of
    the known counts), the steps stopped at their turn cap (``turn_limits``), the steps the
    adapter left no facts of (``session_facts_missing``), each step's hand-offs that were
    absent when it ran (``missing_inputs``, `workspace.absent_reads`: it ran without them)
    and the steps that left no reply (``steps_without_reply``: missing after their
    retries, or a reply with no text); the conversation is every step's final message in
    order, and `TOOL_CALLS` every step's calls. An item where no step answered ends with no
    answer (`ops.NoReply`), as swe's does."""
    turn = ctx.read_json(TURN, {}) or {}
    fail_reason, extra = "", {}
    n_turns: int | None = 0
    if turn.get("status") == "no_case":
        fail_reason, n_turns = "no-test-cases", 0
    else:
        # A reply file with no text is no reply (the step said nothing).
        replies = {step: text for step in STEPS if ctx.exists(reply_path(step))
                   for text in (ctx.read_text(reply_path(step)),) if text.strip()}
        if not replies:
            raise ops.NoReply("no step left a reply: every step's model call ended without an "
                              "answer, so the item ends here")
        sessions = {step: swe_ops.session_facts(ctx, session_path(step)) for step in STEPS}
        step_turns = {step: (sessions[step] or {}).get("num_turns") for step in STEPS}
        known = [turns for turns in step_turns.values() if turns is not None]
        n_turns = sum(known) if known else None
        extra["step_turns"] = step_turns
        capped = {step: facts["max_turns"] for step, facts in sessions.items()
                  if facts and facts["turn_limit_reached"]}
        if capped:
            extra["turn_limits"] = capped
        missing = [step for step in STEPS if step in replies and not sessions[step]]
        if missing:
            extra["session_facts_missing"] = missing
        absent = workspace.absent_reads(ctx.workdir)
        if absent:
            extra["missing_inputs"] = absent
        silent = [step for step in STEPS if step not in replies]
        if silent:
            extra["steps_without_reply"] = silent
        recorded = [{"step": step, "n_turns": step_turns[step],
                     "turn_limit_reached": sessions[step]["turn_limit_reached"],
                     "calls": sessions[step]["tool_calls"]}
                    for step in STEPS if sessions[step]]
        if recorded:
            ctx.write_json(TOOL_CALLS, {"n_turns": n_turns, "steps": recorded})
        last = replies[[step for step in STEPS if step in replies][-1]]
        ctx.write_text(ops.RAW, last)
        ctx.write_json(ops.CONVERSATION, [{"role": "assistant", "step": step, "content": text}
                                          for step, text in replies.items()])
        if ctx.exists(swe_port.OUTPUT_PATH):
            ctx._note_write(PRED)  # noqa: SLF001 - a declared write, copied in place
            shutil.copyfile(ctx.path(swe_port.OUTPUT_PATH), ctx._writable(PRED))  # noqa: SLF001
        else:
            fail_reason = NO_OUTPUT
    elapsed = time.time() - float(turn.get("start") or time.time())
    limit = ops.item_limit_s()
    ctx.write_json(FINAL, {"fail_reason": fail_reason, "n_turns": n_turns, **extra,
                           "elapsed_s": round(elapsed, 3), "backstop": elapsed > limit,
                           "limit_s": limit})
    return {"finished": True}
