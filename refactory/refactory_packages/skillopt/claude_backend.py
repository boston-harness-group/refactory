"""Arm A's Claude backend framing for optimizer calls.

SkillOpt's Claude backend (microsoft/SkillOpt, MIT) sends every optimizer call
(``chat_optimizer``, skillopt/model/claude_backend.py:327-329) through
``_build_prompt_from_messages`` (claude_backend.py:83-138): the user message goes
into a ``History:`` block as ``- User: <text>`` (stripped, every continuation line
indented two spaces; ``- User:`` alone when empty), followed by ``Answer the latest
user request.``. The system message goes to ``--append-system-prompt-file``, so it
is not part of the prompt. The strategy frames every optimizer call this way: arm A
runs on that backend.
"""
from __future__ import annotations

from typing import Any

#: What arm A's Claude backend puts before and after the one user message.
_USER_HEAD = "History:\n- User:"
_ANSWER_TAIL = "\n\nAnswer the latest user request."


# Ported from skillopt/model/claude_backend.py:83-138 ``_build_prompt_from_messages`` (MIT)
def claude_backend_prompt(user: str) -> str:
    """The prompt arm A's ``claude -p`` receives for a ``[system, user]`` call."""
    stripped = user.strip()
    if not stripped:
        return _USER_HEAD + _ANSWER_TAIL
    return f"{_USER_HEAD} {stripped.replace(chr(10), chr(10) + '  ')}{_ANSWER_TAIL}"


# Ported from skillopt/model/claude_backend.py:89-94 ``_history_line`` (MIT)
def _history_line(label: str, body: str) -> str:
    stripped = body.strip()
    if not stripped:
        return f"- {label}:"
    indented = stripped.replace("\n", "\n  ")
    return f"- {label}: {indented}"


# Ported from skillopt/model/claude_backend.py:83-142 ``_build_prompt_from_messages`` (MIT),
# text messages with no tools and no structured output.
def claude_backend_messages(messages: list[dict]) -> tuple[str, str]:
    """``(system, prompt)``: what arm A's ``claude -p`` receives for a message list.

    The system texts, each stripped, joined by a blank line and stripped again, go to
    ``--append-system-prompt-file``. Every other message becomes one history line --
    ``- Assistant: <text>`` for the assistant, ``- <Role>: <text>`` otherwise (the role
    capitalised: ``- User:``) -- stripped, every continuation line indented two spaces,
    the label alone when the text is empty. The prompt is ``History:`` and those lines
    (joined by newlines, then stripped), a blank line, and ``Answer the latest user
    request.``; with no history line, the tail alone.
    """
    system_parts: list[str] = []
    history_parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        text = str(message.get("content", ""))
        if role == "system":
            if text.strip():
                system_parts.append(text.strip())
            continue
        label = "Assistant" if role == "assistant" else role.capitalize()
        history_parts.append(_history_line(label, text))
    prompt_parts: list[str] = []
    history_text = "\n".join(part for part in history_parts if part).strip()
    if history_text:
        prompt_parts.append("History:\n" + history_text)
    prompt_parts.append("Answer the latest user request.")
    return "\n\n".join(part for part in system_parts if part).strip(), "\n\n".join(prompt_parts)


class ClaudeBackendFraming:
    """An optimizer client that sends each prompt framed by :func:`claude_backend_prompt`.

    Everything else -- the system prompt, the call's keywords, every other attribute
    -- passes through to the wrapped client unchanged. The trace still records the
    bare message the strategy built (`trace.ask`).
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def complete(self, prompt: str, **options: Any) -> str:
        """Send ``prompt`` framed; the call's keywords (``system`` and any other) pass
        through exactly as the caller gave them."""
        return self.client.complete(claude_backend_prompt(prompt), **options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


def unframe_claude_backend_messages(prompt: str) -> list[dict] | None:
    """The history inside a :func:`claude_backend_messages` prompt, as ``role``/``content``
    messages (``User`` and ``Assistant`` lines; the label lowercased), or ``None`` for a
    prompt without that framing. Exact for messages with no leading or trailing
    whitespace (the framing strips it)."""
    head, tail = "History:\n", "\n\nAnswer the latest user request."
    if not (prompt.startswith(head) and prompt.endswith(tail)):
        return None
    messages: list[dict] = []
    for line in prompt[len(head):-len(tail)].split("\n"):
        label = next((name for name in ("User", "Assistant")
                      if line == f"- {name}:" or line.startswith(f"- {name}: ")), None)
        if label is not None:
            messages.append({"role": label.lower(), "content": line[len(label) + 4:]})
        elif messages and (line.startswith("  ") or line == ""):
            messages[-1]["content"] += "\n" + line[2:]
        else:
            return None
    return messages
