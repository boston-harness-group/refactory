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

``skillopt_prompts/analyst_error.md`` and ``analyst_success.md`` are SkillOpt's
SpreadsheetBench analyst prompts, copied verbatim from
``skillopt/envs/spreadsheetbench/prompts/`` (MIT License, Copyright (c) 2026 Microsoft
Corporation).
"""
from __future__ import annotations

from typing import Any

from refactory_packages.skillopt.claude_backend import unframe_claude_backend_messages

from . import skillopt_port as port

_PREVIEW_HEAD = "# Input spreadsheet preview\n"
_PREVIEW_TAIL = "\n\n# Task\n"


def preview_of(user: str) -> str:
    """The preview inside a first user message (`skillopt_port.build_user`)."""
    start = user.find(_PREVIEW_HEAD)
    end = user.rfind(_PREVIEW_TAIL)
    return user[start + len(_PREVIEW_HEAD):end] if start != -1 and end > start else ""


def rollout_result(record: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    """One item record (``refactory_dsh/items.py:make_item_record``) as SkillOpt's result."""
    meta = record.get("metadata") or {}
    passed = bool(record.get("passed"))
    result: dict[str, Any] = {
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
    exchanges = [e for e in (record.get("exchanges") or []) if isinstance(e, dict)]
    if record.get("error_text") is not None or not exchanges:
        # An item that failed or timed out, or had no case, saved no conversation.
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
