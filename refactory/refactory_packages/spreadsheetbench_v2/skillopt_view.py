"""SpreadsheetBench 2 agent records as SkillOpt rollout results, for its analysts.

As `spreadsheetbench.skillopt_view.agent_rollout_result` renders SB's agent records: one
turn, the conversation the agent's final message then the verification or execution-error
block (the record's ``prediction``), never its tool calls or ``solution.py``. The target's
prompts are what this agent was given: Claude Code's own system prompt (nothing appended,
so none shown) and ``task.md`` (V2's instructions and SkillOpt's harness section), with the
input's preview -- the same preview the default loop shows as the item's context
(`tasks.SpreadsheetBenchV2Task.item_context`). SkillOpt has no V2 environment, so its
generic analyst prompts are used (no ``skillopt_prompts/`` here).
"""
from __future__ import annotations

import os
from typing import Any

from refactory_dsh.items import NO_ANSWER

from refactory_packages.spreadsheetbench import agent_port
from refactory_packages.spreadsheetbench import skillopt_view as sb_view


def _preview(meta: dict[str, Any]) -> str:
    from . import tasks

    roots = tasks.data_roots()
    rel = str(meta.get("input_rel") or "")
    if not roots or not rel:
        return ""
    return tasks._preview(os.path.join(roots[0], rel))  # noqa: SLF001


def agent_rollout_result(record: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    """One V2 agent item record as SkillOpt's ``claude_code_exec`` result."""
    from .agent_ops import build_task

    result = sb_view._base(record)  # noqa: SLF001
    meta = record.get("metadata") or {}
    result["task_type"] = meta.get("category", "")
    exchanges = [e for e in (record.get("exchanges") or []) if isinstance(e, dict)]
    if record.get("error_text") is not None or not exchanges \
            or str(record.get("fail_reason") or "").startswith(NO_ANSWER):
        return result
    conversation = agent_port.conversation(str(exchanges[-1].get("response") or ""))
    verification = str(record.get("prediction") or "")
    if verification:
        conversation.append({"role": "system",
                             "content": f"[POST-EXECUTION VERIFICATION]\n\n{verification}"})
    result.update({
        "n_turns": 1,
        "target_system_prompt": "",
        "target_user_prompt": build_task(str(meta.get("instruction", ""))),
        "spreadsheet_preview": _preview(meta),
        "conversation": conversation,
    })
    return result


def swe_rollout_result(record: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    """One swe item record as SkillOpt's ``claude_code_exec`` result, as the agent mode's:
    the final message then the verification block (the record's ``prediction``); the
    target's prompts are V2's system template (appended to Claude Code's) and the prompt
    the session was sent, with the run's skill."""
    from . import swe_port

    result = sb_view._base(record)  # noqa: SLF001
    meta = record.get("metadata") or {}
    result["task_type"] = meta.get("category", "")
    exchanges = [e for e in (record.get("exchanges") or []) if isinstance(e, dict)]
    if record.get("error_text") is not None or not exchanges \
            or str(record.get("fail_reason") or "").startswith(NO_ANSWER):
        return result
    conversation = agent_port.conversation(str(exchanges[-1].get("response") or ""))
    verification = str(record.get("prediction") or record.get("fail_reason") or "")
    if verification:
        conversation.append({"role": "system",
                             "content": f"[POST-EXECUTION VERIFICATION]\n\n{verification}"})
    result.update({
        "n_turns": 1,
        "target_system_prompt": swe_port.SYSTEM_TEMPLATE,
        "target_user_prompt": swe_port.render(str(meta.get("instruction", "")),
                                              str(knobs.get("skill", "") or "")),
        "spreadsheet_preview": _preview(meta),
        "conversation": conversation,
    })
    return result


rollout_result = agent_rollout_result
#: The package's modes.
MODE_VIEWS = {"agent": agent_rollout_result, "swe": swe_rollout_result}
