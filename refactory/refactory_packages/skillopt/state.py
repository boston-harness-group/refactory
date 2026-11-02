"""Serialize / deserialize SkillOpt strategy state for resume.

RNG state (``random.Random.getstate()``) returns ``(version, tuple_of_625_ints,
gauss_next)``; the inner tuple is converted to a list for JSON and back to a
tuple on restore.
"""

from __future__ import annotations

import random
from typing import Any


def serialize_rng(rng: random.Random) -> dict[str, Any]:
    version, internalstate, gauss_next = rng.getstate()
    return {
        "version": version,
        "internalstate": list(internalstate),
        "gauss_next": gauss_next,
    }


def deserialize_rng(data: dict[str, Any]) -> random.Random:
    rng = random.Random()
    rng.setstate((
        data["version"],
        tuple(data["internalstate"]),
        data["gauss_next"],
    ))
    return rng


def serialize_state(fields: dict[str, Any]) -> dict[str, Any]:
    """Convert strategy state to a JSON-safe dict."""
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if key == "rng_state" and isinstance(value, random.Random):
            out[key] = serialize_rng(value)
        else:
            out[key] = value
    return out


def deserialize_state(data: dict[str, Any]) -> dict[str, Any]:
    """Restore strategy state from a JSON dict."""
    out: dict[str, Any] = dict(data)
    if "rng_state" in out and isinstance(out["rng_state"], dict):
        out["rng_state"] = deserialize_rng(out["rng_state"])
    return out
