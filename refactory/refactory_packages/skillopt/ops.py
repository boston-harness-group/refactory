"""SkillOpt replay operations.

A minimal op so the replay mode has a resolvable call in its graph.
"""
from __future__ import annotations

from typing import Any


def replay_noop(ctx: Any) -> dict[str, Any]:
    """No-op placeholder for the replay mode's graph validation."""
    return {"replay_ready": True}
