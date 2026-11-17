"""The ops of SkillOpt's SpreadsheetBench multi-turn loop, as graph nodes.

``codegen_agent.run_multi`` (skillopt/envs/spreadsheetbench/codegen_agent.py:681-772,
the chat path) and ``rollout.process_one_codegen`` (rollout.py:610-863), split at the
model call so the call is an agent node (`mode.Solver`)::

    prepare -> has_case -> frame -> solve -> run_turn -> gold_check -> next_turn -> turn_gate
                  |          ^                                                        |
                  |          +----------------------- reloop -------------------------+
                  +-- skip --> finalize <------------------ proceed -------------------+

The item budget (SkillOpt's ``exec_timeout``, the ``item_budget_s`` knob) is kept the
way the port spec's interim says: `prepare` records the item's start, `turn_gate` stops
the loop when 10 s or less of it are left (codegen_agent.py:704-711), and `finalize`
(then `tasks.SpreadsheetBenchTask.verify`, which counts grading too) scores an item past
the budget 0 as SkillOpt's batch runner does (rollout.py:983-1011). A model call that
starts inside the budget is not cut short at its end: the solve node's own timeout is the
whole budget, so a late call costs time but never changes a score.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from refactory_dsh.nn import Verdict

from refactory_packages.skillopt.claude_backend import claude_backend_messages

from . import skillopt_port as port
from .tasks import FINAL, INPUT_DIR, PRED, TURN

HISTORY = "history.json"
REPLY = "reply.txt"
#: What a turn left (SkillOpt saves the last ones, rollout.py:771-778).
CODE = "code.py"
RAW = "raw.txt"
CONVERSATION = "conversation.json"
#: What SkillOpt saves for its analysts before the loop (rollout.py:700-721).
SYSTEM_FILE = "target_system_prompt.txt"
USER_FILE = "target_user_prompt.txt"
PREVIEW_FILE = "spreadsheet_preview.txt"
#: codegen_agent.py:707-710: no new turn with this little of the item budget left.
LAST_TURN_MARGIN_S = 10
#: executor.py's default per-run limit (codegen_agent.py:729-734, rollout.py:799-800).
RUN_TIMEOUT_S = 120


def _item(ctx: Any) -> dict[str, Any]:
    return ctx.read_json("current_item.json", {}) or {}


def _input(ctx: Any, item: dict[str, Any]) -> Path | None:
    name = str(item.get("input_name") or "")
    path = Path(ctx.workdir) / INPUT_DIR / name
    return path if name and path.is_file() else None


def prepare(ctx: Any) -> dict[str, Any]:
    """Record the item's start; build SkillOpt's system and first user message
    (rollout.py:700-721; codegen_agent.py:684-697) and the history that starts with it."""
    started = time.time()
    item = _item(ctx)
    source = _input(ctx, item)
    if source is None:
        # rollout.py:681-685: no init+golden pair, no model call.
        ctx.write_json(TURN, {"start": started, "turn": 0, "status": "no_case", "done": True})
        return {"skill_section": "", "has_case": False}
    skill = ctx.knob("skill", "")
    try:
        preview = port.preview_workbook(str(source))
    except Exception:  # noqa: BLE001 - rollout.py:702-705
        preview = "(preview failed)"
    user = port.build_user(item.get("instruction", ""), str(source),
                           item.get("instruction_type", ""), item.get("answer_position_eval", ""))
    ctx.write_text(PREVIEW_FILE, preview)
    ctx.write_text(SYSTEM_FILE, port.build_system(skill))
    ctx.write_text(USER_FILE, user)
    ctx.write_json(HISTORY, [{"role": "user", "content": user}])
    ctx.write_json(TURN, {"start": started, "turn": 0, "status": "", "done": False, "code": ""})
    return {"skill_section": port.skill_section(skill), "has_case": True}


def has_case(ctx: Any) -> Any:
    """Gate: an item with a test case goes to the loop; one without goes to finalize."""
    if ctx.facts.get("has_case"):
        return "proceed"
    return Verdict("skip", "no-test-cases: the item has no init and golden pair")


def frame(ctx: Any) -> dict[str, Any]:
    """The prompt the target's ``claude -p`` gets for the history so far
    (`claude_backend_messages`: the ``History:`` block, then ``Answer the latest user
    request.``); the system message travels separately, as the solve node's system
    prompt."""
    _system, prompt = claude_backend_messages(ctx.read_json(HISTORY, []) or [])
    return {"solver_prompt": prompt}


def _sandbox_run(ctx: Any, item: dict[str, Any], code: str) -> tuple[bool, str]:
    """Run ``code`` as SkillOpt's executor does, in a fresh confined directory holding a
    copy of the input and of the prediction so far; the prediction comes back after."""
    from refactory_dsh.items import PASS_MARKER

    from . import interpreter, sandbox

    python = interpreter.interpreter()
    sandbox.require(python)
    workdir = Path(ctx.workdir)
    # The run the pass belongs to (``<run>/passes/<node>/<item>``), else the directory.
    run_dir = workdir.parents[2] if (workdir / PASS_MARKER).exists() else workdir
    hidden = sandbox.hidden_paths([str(run_dir), *_data_roots()])
    pred_name = f"{item.get('case_no') or 1}_pred.xlsx"
    with tempfile.TemporaryDirectory(prefix="sb-item-") as work:
        root = Path(work)
        (root / "in").mkdir()
        (root / "out").mkdir()
        (root / "cwd").mkdir()
        source = _input(ctx, item)
        assert source is not None
        input_path = root / "in" / source.name
        shutil.copyfile(source, input_path)
        output_path = root / "out" / pred_name
        if (workdir / PRED).exists():
            shutil.copyfile(workdir / PRED, output_path)
        ok, err = port.run_generated_code(
            code, str(input_path), str(output_path), RUN_TIMEOUT_S,
            python=python, wrap=sandbox.wrapper(hidden), cwd=str(root / "cwd"))
        if output_path.exists():
            ctx._note_write(PRED)  # noqa: SLF001 - a declared write, copied in place
            shutil.copyfile(output_path, ctx._writable(PRED))  # noqa: SLF001
    return ok, err


def _data_roots() -> list[str]:
    """The dataset directory, hidden from model code."""
    import os

    from .tasks import DATA_ENV

    root = os.environ.get(DATA_ENV, "").strip()
    return [str(Path(root).expanduser())] if root else []


def run_turn(ctx: Any) -> dict[str, Any]:
    """One turn after the model's reply (codegen_agent.py:713-761): the cooldown, the
    code, its run, and the execution feedback."""
    time.sleep(float(ctx.knob("turn_pause_s", 3)))  # codegen_agent.py:713
    item = _item(ctx)
    history = ctx.read_json(HISTORY, []) or []
    turn = ctx.read_json(TURN, {}) or {}
    raw = ctx.read_text(REPLY) if ctx.exists(REPLY) else ""
    code = port.extract_code(raw)
    history.append({"role": "assistant", "content": raw})
    turn["turn"] = int(turn.get("turn", 0)) + 1
    turn["code"] = code
    if not code.strip():
        history.append({"role": "user", "content": port.NO_CODE})
        turn["status"] = "no_code"
    else:
        ok, err = _sandbox_run(ctx, item, code)
        if ok:
            turn["status"] = "ran"
        else:
            history.append({"role": "user", "content": port.exec_error_feedback(err)})
            turn["status"] = "exec_error"
    ctx.write_text(CODE, code)
    ctx.write_text(RAW, raw)
    ctx.write_json(HISTORY, history)
    ctx.write_json(TURN, turn)
    return {"turn_status": turn["status"]}


def next_turn(ctx: Any) -> dict[str, Any]:
    """After a clean run: the item is done, unless a train-feedback node says the output
    is wrong -- then its feedback is the next user message (codegen_agent.py:735-753)."""
    turn = ctx.read_json(TURN, {}) or {}
    history = ctx.read_json(HISTORY, []) or []
    if turn.get("status") == "ran":
        feedback = ctx.facts.get("train_feedback")
        if isinstance(feedback, dict) and feedback.get("passed") is False:
            history.append({"role": "user", "content": str(feedback.get("feedback", ""))})
            turn["status"] = "wrong"
        else:
            turn["done"] = True
    ctx.write_json(HISTORY, history)
    ctx.write_json(TURN, turn)
    return {"turn_status": turn.get("status", "")}


def turn_gate(ctx: Any) -> Any:
    """Gate: another turn unless the item is done, the turns are spent, or 10 s or less
    of the item budget are left (codegen_agent.py:703-711)."""
    turn = ctx.read_json(TURN, {}) or {}
    if turn.get("done"):
        return "proceed"
    if int(turn.get("turn", 0)) >= int(ctx.knob("max_turns", 30)):
        return Verdict("proceed", "the turns are spent")
    remaining = float(turn.get("start", 0)) + float(ctx.knob("item_budget_s", 0)) - time.time()
    if remaining <= LAST_TURN_MARGIN_S:
        turn["stop"] = "budget"
        ctx.write_json(TURN, turn)
        return Verdict("proceed", f"{max(remaining, 0):.0f} s of the item budget left")
    return Verdict("reloop", f"turn {turn.get('turn')} ended {turn.get('status')}")


def finalize(ctx: Any) -> dict[str, Any]:
    """The item's outcome before grading (rollout.py:779-812, :983-1011): an empty last
    code block fails; a missing prediction gets one more run of the last code; an item
    past its budget is a timeout. Writes `tasks.FINAL` and SkillOpt's
    ``conversation.json`` (the history after the first user message)."""
    item = _item(ctx)
    turn = ctx.read_json(TURN, {}) or {}
    history = ctx.read_json(HISTORY, []) or []
    budget = float(ctx.knob("item_budget_s", 0))
    code = str(turn.get("code") or "")
    fail_reason = ""
    calls = int(turn.get("turn", 0))
    # run_multi's n_turns is its loop index + 1: a loop the budget stopped before a
    # call counts that turn too (codegen_agent.py:768).
    n_turns = calls + 1 if turn.get("stop") == "budget" and calls < int(ctx.knob("max_turns", 30)) \
        else calls
    if turn.get("status") == "no_case":
        fail_reason = "no-test-cases"
        n_turns = 0
    elif not code.strip():
        fail_reason = "empty-code-block"
    elif not ctx.exists(PRED):
        ok, err = _sandbox_run(ctx, item, code)
        if not ok:
            fail_reason = f"exec-error: {port.tail_line(err)}"
        elif not ctx.exists(PRED):
            fail_reason = "output-not-found"
    elapsed = time.time() - float(turn.get("start") or time.time())
    if budget and elapsed > budget:
        fail_reason = f"task-timeout-{int(budget)}s"
    ctx.write_json(CONVERSATION, history[1:])
    ctx.write_json(FINAL, {"fail_reason": fail_reason, "n_turns": n_turns,
                           "elapsed_s": round(elapsed, 3), "item_budget_s": budget})
    return {"finished": True}
