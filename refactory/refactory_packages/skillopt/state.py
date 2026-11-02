"""Serialize / deserialize SkillOpt strategy state for resume.

The strategy keeps no random generator between steps: every shuffle and
sample is reseeded from the configured seed and the epoch, as in SkillOpt
(reflect.py:481, trainer.py:1181, :1844). Its state is therefore plain JSON.
"""

from __future__ import annotations

from typing import Any


def serialize_state(fields: dict[str, Any]) -> dict[str, Any]:
    """Convert strategy state to a JSON-safe dict."""
    return dict(fields)


def deserialize_state(data: dict[str, Any]) -> dict[str, Any]:
    """Restore strategy state from a JSON dict."""
    return dict(data)
