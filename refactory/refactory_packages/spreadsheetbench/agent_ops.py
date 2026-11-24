"""The ops of SkillOpt's SpreadsheetBench agent harness (``claude_code_exec``), as graph nodes.

``codegen_agent.run_single``'s exec branch (skillopt/envs/spreadsheetbench/codegen_agent.py:
466-504) and ``rollout.process_one_codegen`` (rollout.py:610-863), split at the agent so
the agent is a sealed Claude Code step (`agent_mode.Agent`)::

    prepare -> has_case -> solve -> finalize
                  |                    ^
                  +------- skip -------+

`prepare` writes SkillOpt's workspace files into the pass (`agent_port`), and the solve
step's workspace gets copies of them (its declared reads). The step leaves ``solution.py``
there when it writes one; `finalize` takes that, or the reply's first fenced block, runs it
on the input with SkillOpt's runner in the code sandbox (`ops._sandbox_run`), and records
the item's outcome for the Task's grade, as the solver's `ops.finalize` does.

As in the solver there is no item budget (SkillOpt's 600 s ``exec_timeout``): the call ends
on inactivity, and the one limit is the harness backstop (`ops.item_limit_s`); an item past
it is the model's own failure (``backstop``).
"""
from __future__ import annotations

import shutil
import time
from typing import Any

from . import agent_port
from . import ops
from . import skillopt_port as port
from .tasks import FINAL, PRED, TURN

#: The solve step's reply: its final message (`agent_port.conversation`).
REPLY = "reply.txt"


def prepare(ctx: Any) -> dict[str, Any]:
    """Record the item's start; write SkillOpt's workspace files (codegen_agent.py:319-358):
    the input as ``input.xlsx``, SKILL.md, task.md and run_solution.py; and what
    ``process_one_codegen`` saves for its analysts before the call (rollout.py:700-721:
    the chat path's system and user prompts and the preview, as SkillOpt saves them in
    this harness too)."""
    started = time.time()
    item = ops._item(ctx)  # noqa: SLF001 - the solver's reader of the item
    source = ops._input(ctx)  # noqa: SLF001
    if source is None:
        # rollout.py:682-685: no init+golden pair, no model call.
        ctx.write_json(TURN, {"start": started, "status": "no_case"})
        return {"has_case": False}
    skill = ctx.knob("skill", "")
    instruction = item.get("instruction", "")
    itype = item.get("instruction_type", "")
    position = item.get("answer_position_eval", "")
    try:
        preview = port.preview_workbook(str(source))
    except Exception:  # noqa: BLE001 - rollout.py:702-705
        preview = "(preview failed)"
    write_workspace(ctx, source, skill,
                    agent_port.build_codex_task(instruction, str(source), itype, position))
    ctx.write_text(ops.PREVIEW_FILE, preview)
    ctx.write_text(ops.SYSTEM_FILE, port.build_system(skill))
    ctx.write_text(ops.USER_FILE, port.build_user(instruction, str(source), itype, position))
    ctx.write_json(TURN, {"start": started, "status": ""})
    return {"has_case": True}


def write_workspace(ctx: Any, source: Any, skill: Any, task: str) -> None:
    """SkillOpt's workspace files in the pass (codegen_agent.py:319-358): the input as
    ``input.xlsx``, SKILL.md for ``skill``, ``task`` as task.md, and run_solution.py."""
    ctx._note_write(agent_port.INPUT_PATH)  # noqa: SLF001 - a declared write, copied in place
    shutil.copyfile(source, ctx._writable(agent_port.INPUT_PATH))  # noqa: SLF001
    ctx.write_text(agent_port.SKILL_PATH, agent_port.build_codex_skill(skill))
    ctx.write_text(agent_port.TASK_PATH, task)
    ctx.write_text(agent_port.DRIVER_PATH, agent_port.RUN_SOLUTION)


def finalize(ctx: Any) -> dict[str, Any]:
    """The item's outcome before grading (codegen_agent.py:491-504, rollout.py:783-863):
    the code is ``solution.py`` when the step left one, else the reply's first fenced block;
    an empty one fails (``empty-code-block``); otherwise it runs once on the input, and an
    error or a missing output fails. Writes `tasks.FINAL` and SkillOpt's
    ``conversation.json`` (the final message alone). A solve step that ended without an
    answer ends the item with no answer (`ops.NoReply`), as in the solver."""
    return finalize_item(ctx, ops._data_roots())  # noqa: SLF001


def finalize_item(ctx: Any, data_roots: list[str]) -> dict[str, Any]:
    """`finalize`, with model code blind to ``data_roots`` (the Task's dataset)."""
    item = ops._item(ctx)  # noqa: SLF001
    turn = ctx.read_json(TURN, {}) or {}
    fail_reason, n_turns, exec_error = "", 0, None
    if turn.get("status") == "no_case":
        fail_reason = "no-test-cases"
    else:
        if not ctx.exists(REPLY):
            raise ops.NoReply(f"the solve step left no {REPLY}: its model call ended without an "
                              "answer, so the item ends here")
        reply = ctx.read_text(REPLY)
        solution = (ctx.read_text(agent_port.SOLUTION_PATH)
                    if ctx.exists(agent_port.SOLUTION_PATH) else None)
        code = agent_port.item_code(solution, reply)
        n_turns = 1  # codegen_agent.py:500
        ctx.write_text(ops.CODE, code)
        ctx.write_text(ops.RAW, reply)
        ctx.write_json(ops.CONVERSATION, agent_port.conversation(reply))
        if not code.strip():
            fail_reason = "empty-code-block"  # rollout.py:783-786
        else:
            ok, err = ops._sandbox_run(ctx, item, code, data_roots)  # noqa: SLF001
            if not ok:
                fail_reason = f"exec-error: {port.tail_line(err)}"
                exec_error = err[:500]  # rollout.py:802-812, shown to the analysts
            elif not ctx.exists(PRED):
                fail_reason = "output-not-found"
    elapsed = time.time() - float(turn.get("start") or time.time())
    limit = ops.item_limit_s()
    ctx.write_json(FINAL, {"fail_reason": fail_reason, "n_turns": n_turns,
                           **({"exec_error": exec_error} if exec_error is not None else {}),
                           "elapsed_s": round(elapsed, 3), "backstop": elapsed > limit,
                           "limit_s": limit})
    return {"finished": True}
