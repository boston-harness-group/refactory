"""Operations for non-baseline solver blocks.

``read_draft`` reads the draft agent's output so the verifier / self-check can
see it as a template variable.  ``merge_attempts`` reads both fork branches'
outputs so the judge can compare them.
"""
from __future__ import annotations

from typing import Any


def read_draft(ctx: Any) -> dict[str, Any]:
    """Read ``draft_answer.txt`` and expose it as ``draft_response``."""
    text = ctx.read_text("draft_answer.txt")
    return {"draft_response": text}


def merge_attempts(ctx: Any) -> dict[str, Any]:
    """Read both fork branches' outputs and expose them for the judge."""
    return {
        "attempt_a_text": ctx.read_text("attempt_a.txt"),
        "attempt_b_text": ctx.read_text("attempt_b.txt"),
    }
