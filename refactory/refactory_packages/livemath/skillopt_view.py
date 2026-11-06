"""LiveMath's item records as SkillOpt's LiveMathematicianBench rollout results.

The SkillOpt strategy (``refactory_packages/skillopt/env.py``) renders each
train record through this module, as SkillOpt's generic analyst formatter
renders what ``skillopt/envs/livemathematicianbench/rollout.py`` (MIT) wrote:
the result dict of ``process_one`` (rollout.py:127-144, :250-270), its
``conversation.json`` (one ``{"type": "message"}`` entry per turn, then the
``{"role": "system"}`` evaluation detail, rollout.py:246, :272-281), and the
``target_system_prompt.txt`` / ``target_user_prompt.txt`` it saved
(rollout.py:254-257).

Every LiveMath value comes from this package: the system prompt from
``mode.SYSTEM_PROMPT`` and ``ops.skill_section``, the grading from
``grading.evaluate`` -- the same code the solver graph and ``LiveMathTask.verify``
run.

``skillopt_prompts/analyst_error.md`` and ``skillopt_prompts/analyst_success.md``
are SkillOpt's LiveMathematicianBench analyst prompts, copied verbatim from
``skillopt/envs/livemathematicianbench/prompts/`` (microsoft/SkillOpt, MIT
License, Copyright (c) 2026 Microsoft Corporation). The SkillOpt strategy reads them
in place of its generic analyst prompts, as ``load_prompt(name, env)`` does.
"""
from __future__ import annotations

import re
from typing import Any

from .grading import evaluate
from .mode import SYSTEM_PROMPT, Solver
from .ops import skill_section, unwrap_claude_backend_prompt


def build_system(skill: Any) -> str:
    """The solver's system prompt for a skill (rollout.py:16-21 ``_build_system``)."""
    return SYSTEM_PROMPT.replace("{{skill_section}}", skill_section(skill))


def task_type(metadata: dict) -> str:
    """rollout.py:132: the item's first theorem type, else ``math_mcq``."""
    theorem_type = metadata.get("theorem_type")
    return theorem_type[0] if theorem_type else "math_mcq"


#: The exceptions a model call raises when it runs out of time
#: (``refactory_dsh.model.CompletionTimeout`` and the subprocess timeouts it wraps).
TIMEOUT_ERRORS = ("CompletionTimeout", "TimeoutExpired", "InactivityTimeout")

# A failed pass's error text: "<node>: <ExceptionType>: <message>"
# (refactory_dsh/nn/executor.py ``_walk_branch``).
_NODE_ERROR = re.compile(r"^[A-Za-z0-9_.-]+: ([A-Za-z_][A-Za-z0-9_]*): (.*)$", re.DOTALL)


def task_timeout_s(solve_timeout_s: int | None = None) -> int:
    """SkillOpt's task timeout for an exec timeout (rollout.py:308-313).

    The adapter passes ``task_timeout=exec_timeout`` (adapter.py:125), so it is
    ``max(exec_timeout, exec_timeout + 60)``.
    """
    exec_timeout = int(Solver.solve_timeout_s if solve_timeout_s is None else solve_timeout_s)
    return max(exec_timeout, exec_timeout + 60)


def error_fail_reason(record: dict[str, Any]) -> str:
    """A failed pass's fail reason, in SkillOpt's words.

    A pass that ran out of time is ``_timeout_result``'s
    ``task-timeout-{task_timeout}s`` (rollout.py:359-376); any other failure is
    ``process_one``'s ``error: {e}`` (rollout.py:286-287), ``e`` being the
    exception's message.
    """
    text = str(record.get("error_text") or "")
    match = _NODE_ERROR.match(text)
    kind, message = (match.group(1), match.group(2)) if match else ("", text)
    if record.get("error_kind") == "timeout" or kind in TIMEOUT_ERRORS:
        return f"task-timeout-{task_timeout_s()}s"
    return f"error: {message or 'no response'}"


def _target_system_prompt(knobs: dict[str, Any], node: str) -> str:
    """``node``'s system prompt: a ``skill_graph`` step's own whole-prompt knob
    (``<step>__system_prompt``) when the record's knobs have one, else the solver's
    system prompt built from the ``skill`` knob."""
    from .skill_graph import STEPS, system_prompt_knob

    own = system_prompt_knob(node) if node in STEPS else ""
    if own and own in knobs:
        return str(knobs[own])
    return build_system(knobs.get("skill", ""))


def rollout_result(record: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    """One F6 record (``refactory_dsh/items.py:make_item_record``) as SkillOpt's result.

    ``knobs`` are the solver knobs the record was rolled out with; ``skill``
    gives the target system prompt. A ``skill_graph`` record takes its first
    step's system-prompt knob (``<step>__system_prompt``) instead, and when its
    exchanges come from more than one node each message is labelled with its
    node (``[compare] ...``), so the analysts can tell which step wrote what. A one-node record (the solver's) is unchanged.

    A record that ended in an error has no conversation and ``n_turns`` 0
    (rollout.py:286-289, :359-376): the failed or timed-out ``process_one``
    wrote no ``conversation.json`` in time, so the analysts skip it. Its fail
    reason is :func:`error_fail_reason`.
    """
    meta = record.get("metadata") or {}
    result: dict[str, Any] = {
        "id": str(record.get("id", "")),
        "question": meta.get("question", ""),
        "task_type": task_type(meta),
        "hard": 1 if record.get("passed") else 0,
        "soft": 0.0,
        "predicted_answer": "",
        "fail_reason": "",
        "n_turns": 0,
        "conversation": None,
    }
    exchanges = [e for e in (record.get("exchanges") or []) if isinstance(e, dict)]
    if record.get("error_text") is not None or not exchanges:
        result["fail_reason"] = error_fail_reason(record)
        return result

    # rollout.py:221-252: one message per turn; the response graded is the last.
    labelled = len({str(exchange.get("node") or "") for exchange in exchanges}) > 1
    conversation: list[dict[str, Any]] = [
        {"type": "message", "turn": turn, "content": str(exchange.get("response") or ""),
         **({"role": str(exchange.get("node") or "agent")} if labelled else {})}
        for turn, exchange in enumerate(exchanges, 1)
    ]
    response = conversation[-1]["content"]
    result["n_turns"] = len(conversation)

    # rollout.py:254-257: the first turn's system and user prompts.
    result["target_system_prompt"] = _target_system_prompt(knobs, str(exchanges[0].get("node") or ""))
    # SkillOpt saves the user message (rollout.py:254-257), not the Claude
    # backend's framing of it that the solve node sends.
    result["target_user_prompt"] = unwrap_claude_backend_prompt(str(exchanges[0].get("prompt") or ""))

    eval_result = evaluate(response, meta.get("correct_choice") or {}, meta.get("choices") or [])
    result["soft"] = eval_result["f1"]  # rollout.py:261
    result["predicted_answer"] = eval_result["predicted_answer"]  # rollout.py:262
    if not eval_result["em"]:
        # rollout.py:266-270
        result["fail_reason"] = (
            f"MCQ=0: predicted '{eval_result['predicted_label'] or eval_result['predicted_answer']}' "
            f"but expected '{eval_result['correct_label']}'"
        )
    # rollout.py:272-281
    eval_detail = (
        f"[EVALUATION RESULT]\n"
        f"Question: {meta.get('question', '')}\n"
        f"Predicted label: {eval_result['predicted_label']!r}\n"
        f"Predicted text: {eval_result['predicted_text']!r}\n"
        f"Correct label: {eval_result['correct_label']!r}\n"
        f"Correct text: {eval_result['correct_text']!r}\n"
        f"Exact Match: {eval_result['em']}"
    )
    conversation.append({"role": "system", "content": eval_detail})
    result["conversation"] = conversation
    return result
