"""SpreadsheetBench item records as SkillOpt's rollout results, for its analysts.

The SkillOpt strategy (``refactory_packages/skillopt/env.py``) renders each train
record through this module, as SkillOpt's generic formatter renders what
``skillopt/envs/spreadsheetbench/rollout.py`` (microsoft/SkillOpt, MIT) wrote for an
item: the result dict of ``process_one_codegen`` (rollout.py:657-676, :845-863), its
``conversation.json`` -- the messages after the first user message, then the
``[POST-EXECUTION VERIFICATION]`` system entry (rollout.py:838-858) -- and the saved
``target_system_prompt.txt``, ``target_user_prompt.txt`` and ``spreadsheet_preview.txt``
(rollout.py:700-721).

The conversation comes from the record's model calls: the last call's framed prompt
holds every message before its reply (`unframe_claude_backend_messages`). A user
message after the last call -- feedback on a turn that ended the loop by using up its
turns or its budget -- is not in any call, so it is not shown.

The agent mode's records (`agent_mode.Agent`, SkillOpt's ``claude_code_exec`` harness) are
rendered as that harness's results: one turn, the conversation its final message alone
(codegen_agent.py:501) then the verification block, and the chat path's system and user
prompts and preview, which ``process_one_codegen`` saves for the analysts in this harness
too (rollout.py:700-718) -- never the agent's tool calls or its ``solution.py``, which
SkillOpt's CLI path shows them nowhere (`agent_port.conversation`). Its user prompt is
rebuilt from the item's own input in ``SB_DATA`` (its ``spreadsheet_path`` and
``input_name``, both in the record), as SkillOpt built it from the same file.

``skillopt_prompts/analyst_error.md`` and ``analyst_success.md`` are SkillOpt's
SpreadsheetBench analyst prompts, copied verbatim from
``skillopt/envs/spreadsheetbench/prompts/`` (MIT License, Copyright (c) 2026 Microsoft
Corporation).
"""
from __future__ import annotations

from typing import Any

from refactory_dsh.items import NO_ANSWER

from refactory_packages.skillopt.claude_backend import unframe_claude_backend_messages

from . import skillopt_port as port

_PREVIEW_HEAD = "# Input spreadsheet preview\n"
_PREVIEW_TAIL = "\n\n# Task\n"


def preview_of(user: str) -> str:
    """The preview inside a first user message (`skillopt_port.build_user`)."""
    start = user.find(_PREVIEW_HEAD)
    end = user.rfind(_PREVIEW_TAIL)
    return user[start + len(_PREVIEW_HEAD):end] if start != -1 and end > start else ""


def _agent_user(meta: dict[str, Any]) -> str:
    """The chat path's first user message for the item, as ``process_one_codegen`` builds
    it from the item's input (rollout.py:697-718), read from ``SB_DATA``."""
    import os

    from .tasks import DATA_ENV

    root = os.environ.get(DATA_ENV, "").strip()
    sp = str(meta.get("spreadsheet_path") or "")
    path = sp if os.path.isabs(sp) else os.path.join(os.path.expanduser(root), sp)
    return port.build_user(str(meta.get("instruction", "")),
                           os.path.join(path, str(meta.get("input_name") or "")),
                           str(meta.get("instruction_type", "")),
                           str(meta.get("answer_position_eval", "")))


def agent_rollout_result(record: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    """One agent-mode item record as SkillOpt's ``claude_code_exec`` result."""
    from . import agent_port

    result = _base(record)
    exchanges = [e for e in (record.get("exchanges") or []) if isinstance(e, dict)]
    if record.get("error_text") is not None or not exchanges \
            or str(record.get("fail_reason") or "").startswith(NO_ANSWER):
        return result
    conversation = agent_port.conversation(str(exchanges[-1].get("response") or ""))
    verification = str(record.get("prediction") or "")
    if verification:
        conversation.append({"role": "system",
                             "content": f"[POST-EXECUTION VERIFICATION]\n\n{verification}"})
    user = _agent_user(record.get("metadata") or {})
    result.update({
        "n_turns": 1,
        "target_system_prompt": port.build_system(knobs.get("skill", "")),
        "target_user_prompt": user,
        "spreadsheet_preview": preview_of(user),
        "conversation": conversation,
    })
    return result


def _base(record: dict[str, Any]) -> dict[str, Any]:
    """rollout.py:657-676: the result's fields before the model call."""
    meta = record.get("metadata") or {}
    passed = bool(record.get("passed"))
    return {
        "id": str(record.get("id", "")),
        "instruction_type": meta.get("instruction_type", ""),
        "task_type": meta.get("task_type", ""),
        "task_description": meta.get("instruction", ""),
        "hard": 1 if passed else 0,
        "soft": 1.0 if passed else 0.0,
        "fail_reason": "" if passed else str(record.get("fail_reason")
                                              or record.get("error_text") or ""),
        "n_turns": 0,
        "conversation": None,
    }


def rollout_result(record: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    """One item record (``refactory_dsh/items.py:make_item_record``) as SkillOpt's result:
    the agent mode's (its frozen prompt knob is among the knobs) as its harness's."""
    from .agent_mode import PROMPT_KNOB

    if PROMPT_KNOB in knobs:
        return agent_rollout_result(record, knobs)
    result = _base(record)
    exchanges = [e for e in (record.get("exchanges") or []) if isinstance(e, dict)]
    if record.get("error_text") is not None or not exchanges \
            or str(record.get("fail_reason") or "").startswith(NO_ANSWER):
        # An item that failed or timed out, or had no case, saved no conversation; nor
        # did one whose model call failed (no answer): its last reply is not its end.
        return result
    history = unframe_claude_backend_messages(str(exchanges[-1].get("prompt") or "")) or []
    user = history[0]["content"] if history else ""
    conversation: list[dict[str, Any]] = [
        *history[1:], {"role": "assistant", "content": str(exchanges[-1].get("response") or "")}]
    verification = str(record.get("prediction") or "")
    if verification:
        conversation.append({"role": "system",
                             "content": f"[POST-EXECUTION VERIFICATION]\n\n{verification}"})
    result.update({
        "n_turns": len(exchanges),
        "target_system_prompt": port.build_system(knobs.get("skill", "")),
        "target_user_prompt": user,
        "spreadsheet_preview": preview_of(user),
        "conversation": conversation,
    })
    return result
