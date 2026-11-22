"""The ops of SkillOpt's SpreadsheetBench multi-turn loop, as graph nodes.

``codegen_agent.run_multi`` (skillopt/envs/spreadsheetbench/codegen_agent.py:681-772,
the chat path) and ``rollout.process_one_codegen`` (rollout.py:610-863), split at the
model call so the call is an agent node (`mode.Solver`)::

    prepare -> has_case -> frame -> solve -> run_turn -> gold_check -> next_turn -> turn_gate
                  |          ^                                                        |
                  |          +----------------------- reloop -------------------------+
                  +-- skip --> finalize <------------------ proceed -------------------+

There is no item budget (SkillOpt's 600 s ``exec_timeout``): calls end on inactivity,
and the one limit is the harness backstop (`item_limit_s`, ``REFACTORY_MODEL_BACKSTOP_S``).
SkillOpt's check is kept against it: `prepare` records the item's start, `turn_gate`
starts no turn with 10 s or less of it left (codegen_agent.py:704-711), and `finalize`
records an item past it, which the Task's ``verify`` reports as
`refactory_dsh.model.HarnessBackstop`: the model's own failure (``error_kind``
``backstop``), scored 0 like a wrong answer.
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
from .tasks import FINAL, PRED, TURN

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
#: codegen_agent.py:707-710: no new turn with this little time left (here, before the backstop).
LAST_TURN_MARGIN_S = 10
#: executor.py's default per-run limit (codegen_agent.py:729-734, rollout.py:799-800).
RUN_TIMEOUT_S = 120


def item_limit_s() -> float:
    """The most an item's turn loop may take: the harness backstop."""
    from refactory_dsh import model

    return float(model.DEFAULT_BACKSTOP_S)


def _item(ctx: Any) -> dict[str, Any]:
    return ctx.read_json("current_item.json", {}) or {}


def _input(ctx: Any) -> Path | None:
    """The item's input workbook, its item file (`tasks.INPUT`), through the declaration
    that names it (`RunContext.item_file`); ``None`` for an item with no case."""
    return ctx.item_file("input")


def prepare(ctx: Any) -> dict[str, Any]:
    """Record the item's start; build SkillOpt's system and first user message
    (rollout.py:700-721; codegen_agent.py:684-697) and the history that starts with it."""
    started = time.time()
    item = _item(ctx)
    source = _input(ctx)
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
    from . import interpreter, sandbox

    python = interpreter.interpreter()
    sandbox.require(python)
    workdir = Path(ctx.workdir)
    hidden = sandbox.hidden_paths([*_run_roots(workdir), *_data_roots()])
    pred_name = f"{item.get('case_no') or 1}_pred.xlsx"
    with tempfile.TemporaryDirectory(prefix="sb-item-") as work:
        root = Path(work)
        (root / "in").mkdir()
        (root / "out").mkdir()
        (root / "cwd").mkdir()
        source = _input(ctx)
        assert source is not None
        # Under the input's own name, as SkillOpt's code found it (``INPUT_PATH``).
        input_path = root / "in" / (str(item.get("input_name") or "") or source.name)
        shutil.copyfile(source, input_path)
        output_path = root / "out" / pred_name
        if (workdir / PRED).exists():
            shutil.copyfile(workdir / PRED, output_path)
        ok, err = port.run_generated_code(
            code, str(input_path), str(output_path), RUN_TIMEOUT_S,
            python=python, wrap=sandbox.wrapper(hidden, visible=[root / "cwd", root / "in"]),
            cwd=str(root / "cwd"))
        ctx._note_write(PRED)  # noqa: SLF001 - a declared write, copied in place
        if output_path.exists():
            shutil.copyfile(output_path, ctx._writable(PRED))  # noqa: SLF001
        else:
            # The code removed it: SkillOpt's file would be gone too.
            ctx._writable(PRED).unlink(missing_ok=True)  # noqa: SLF001
    return ok, err


def _run_roots(workdir: Path) -> list[str]:
    """The run the pass belongs to and, above it, the outer loop's root (every
    candidate's records): what model code must not see. A concurrent pass works through
    an alias (``.refactory/live/<node>/<item>``), so the pass is resolved first."""
    from refactory_dsh.history import JOURNAL, RUN_FILE
    from refactory_dsh.items import PASS_MARKER

    real = workdir.resolve()
    # ``<run>/passes/<node>/<item>``, else the directory itself.
    run = real.parents[2] if (real / PASS_MARKER).exists() else real
    roots = [str(run)]
    for parent in run.parents:
        if (parent / RUN_FILE).exists() or (parent / JOURNAL).exists():
            roots.append(str(parent))
    return roots


def _data_roots() -> list[str]:
    """The dataset directory, hidden from model code."""
    import os

    from .tasks import DATA_ENV

    root = os.environ.get(DATA_ENV, "").strip()
    return [str(Path(root).expanduser())] if root else []


class NoReply(RuntimeError):
    """The solve step left no reply: its model call ended without an answer (every
    session crashed, or it ran out of output), so the step is missing
    (`refactory_dsh.node_retry.leaves_no_answer`) and its output was set aside. The item
    ends here: a failed call is never read as an empty reply. Its message names `REPLY`,
    so both runtimes record the item as having no answer, with the step's failure
    (`refactory_dsh.items.no_answer`)."""


def run_turn(ctx: Any) -> dict[str, Any]:
    """One turn after the model's reply (codegen_agent.py:713-761): the cooldown, the
    code, its run, and the execution feedback. A turn whose model call failed has no
    reply at all (`NoReply`), unlike an empty reply, which gets SkillOpt's no-code
    message."""
    if not ctx.exists(REPLY):
        raise NoReply(f"the solve step left no {REPLY}: its model call ended without an "
                      "answer, so the item ends here")
    time.sleep(float(ctx.knob("turn_pause_s", 3)))  # codegen_agent.py:713
    item = _item(ctx)
    history = ctx.read_json(HISTORY, []) or []
    turn = ctx.read_json(TURN, {}) or {}
    raw = ctx.read_text(REPLY)
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
    are left before the harness backstop (codegen_agent.py:703-711)."""
    turn = ctx.read_json(TURN, {}) or {}
    if turn.get("done"):
        return "proceed"
    if int(turn.get("turn", 0)) >= int(ctx.knob("max_turns", 30)):
        return Verdict("proceed", "the turns are spent")
    remaining = float(turn.get("start", 0)) + item_limit_s() - time.time()
    if remaining <= LAST_TURN_MARGIN_S:
        turn["stop"] = "budget"
        ctx.write_json(TURN, turn)
        return Verdict("proceed", f"{max(remaining, 0):.0f} s left before the harness backstop")
    return Verdict("reloop", f"turn {turn.get('turn')} ended {turn.get('status')}")


def finalize(ctx: Any) -> dict[str, Any]:
    """The item's outcome before grading (rollout.py:779-812): an empty last code block
    fails; a missing prediction gets one more run of the last code. Writes `tasks.FINAL`
    and SkillOpt's ``conversation.json`` (the history after the first user message), with
    ``backstop`` set when the item's turns ran past the harness backstop."""

    item = _item(ctx)
    turn = ctx.read_json(TURN, {}) or {}
    history = ctx.read_json(HISTORY, []) or []
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
    ctx.write_json(CONVERSATION, history[1:])
    ctx.write_json(FINAL, {"fail_reason": fail_reason, "n_turns": n_turns,
                           "elapsed_s": round(elapsed, 3),
                           "backstop": elapsed > item_limit_s(), "limit_s": item_limit_s()})
    return {"finished": True}
