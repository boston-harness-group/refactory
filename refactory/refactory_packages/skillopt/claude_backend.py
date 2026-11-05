"""Arm A's Claude backend framing for optimizer calls (``claude_backend_wrapper``).

SkillOpt's Claude backend (microsoft/SkillOpt, MIT) sends every optimizer call
(``chat_optimizer``, skillopt/model/claude_backend.py:327-329) through
``_build_prompt_from_messages`` (claude_backend.py:83-138): the user message goes
into a ``History:`` block as ``- User: <text>`` (stripped, every continuation line
indented two spaces; ``- User:`` alone when empty), followed by ``Answer the latest
user request.``. The system message goes to ``--append-system-prompt-file``, so it
is not part of the prompt. SkillOpt's other backends send the messages as they are,
so the strategy frames its optimizer calls only when asked to
(``--strategy-option claude_backend_wrapper=true``).
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


class ClaudeBackendFraming:
    """An optimizer client that sends each prompt framed by :func:`claude_backend_prompt`.

    Everything else -- the system prompt, the call's keywords, every other attribute
    -- passes through to the wrapped client unchanged. The trace still records the
    bare message the strategy built (`trace.ask`).
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        workdir: Any = None,
        cwd: Any = None,
        env: Any = None,
        confine: bool = False,
    ) -> str:
        return self.client.complete(
            claude_backend_prompt(prompt), system=system, workdir=workdir, cwd=cwd,
            env=env, confine=confine,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)
