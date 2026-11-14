"""LiveMathematicianBench operations.

Prompt formatting ported from SkillOpt (MIT) —
skillopt/envs/livemathematicianbench/rollout.py:16-53, and the framing arm A's
Claude backend puts around the user message — skillopt/model/claude_backend.py:83-138.
"""
from __future__ import annotations

from typing import Any


# Ported from skillopt/envs/livemathematicianbench/rollout.py:24-28 (MIT)
def _format_choices(choices: list[dict]) -> str:
    return "\n".join(
        f"{choice['label']}. {choice['text']}"
        for choice in choices
    )


# Ported from skillopt/envs/livemathematicianbench/rollout.py:16-21 (MIT)
def skill_section(skill: Any) -> str:
    """The ``## Skill`` block ``_build_system`` puts in the system prompt."""
    text = str(skill).strip()
    return f"## Skill\n{text}\n\n" if text else ""


# Ported from skillopt/envs/livemathematicianbench/rollout.py:31-53 (MIT)
def build_user_prompt(item: dict) -> str:
    """The solver's user prompt for one item (``_build_user``).

    SkillOpt's ``use_theorem`` / ``use_sketch`` options (rollout.py:41-44, both off in
    configs/livemathematicianbench/default.yaml:21-22) are not ported: an item's
    theorem states the correct option, so the prompt never carries it or its sketch.
    """
    question = item.get("question", "")
    choices = item.get("choices", [])
    return "\n\n".join([
        f"## Question\n{question}",
        f"## Choices\n{_format_choices(choices)}",
    ])


#: What arm A's Claude backend puts before and after the one user message.
_USER_HEAD = "History:\n- User:"
_ANSWER_TAIL = "\n\nAnswer the latest user request."


# Ported from skillopt/model/claude_backend.py:83-138 ``_build_prompt_from_messages`` (MIT)
def claude_backend_prompt(user: str) -> str:
    """The prompt arm A's ``claude -p`` receives for a ``[system, user]`` call.

    SkillOpt's Claude backend (``chat_target``, claude_backend.py:332-334) does not
    send the user message as it is: ``_build_prompt_from_messages`` puts it in a
    ``History:`` block as ``- User: <text>`` (stripped, every continuation line
    indented two spaces; ``- User:`` alone when empty) and appends ``Answer the
    latest user request.`` (no tools, no structured output). The system message
    goes to ``--append-system-prompt-file``, so it is not part of this prompt.
    """
    stripped = user.strip()
    if not stripped:
        return _USER_HEAD + _ANSWER_TAIL
    return f"{_USER_HEAD} {stripped.replace(chr(10), chr(10) + '  ')}{_ANSWER_TAIL}"


def unwrap_claude_backend_prompt(prompt: str) -> str:
    """The user message inside a :func:`claude_backend_prompt` prompt.

    Exact for a message without leading or trailing whitespace (the framing strips
    it). A prompt without the framing -- a record from before the solve node sent
    it -- is returned unchanged.
    """
    if not (prompt.startswith(_USER_HEAD) and prompt.endswith(_ANSWER_TAIL)):
        return prompt
    body = prompt[len(_USER_HEAD):len(prompt) - len(_ANSWER_TAIL)]
    if not body:
        return ""
    if not body.startswith(" "):
        return prompt
    return body[1:].replace("\n  ", "\n")


def format_prompt(ctx: Any) -> dict[str, Any]:
    """Build the user prompt, the solve node's prompt and the skill section.

    ``user_prompt`` is SkillOpt's ``_build_user`` message, unchanged. ``solver_prompt``
    is what the solve node sends: that message framed as arm A's Claude backend
    frames it (:func:`claude_backend_prompt`).
    """
    item = ctx.read_json("current_item.json", {}) or {}
    user_prompt = build_user_prompt(item)
    return {
        "user_prompt": user_prompt,
        "solver_prompt": claude_backend_prompt(user_prompt),
        "skill_section": skill_section(ctx.knob("skill", "")),
    }


def finish(ctx: Any) -> dict[str, Any]:
    return {"finished": True}
