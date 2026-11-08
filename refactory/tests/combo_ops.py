"""Node callables for ``test_search_combo``'s toy mode."""
from __future__ import annotations

from typing import Any


def prep(ctx: Any) -> dict[str, Any]:
    """The current item's id, as the ``question`` fact the solve step's prompt reads."""
    item = ctx.read_json("current_item.json", {}) or {}
    return {"question": str(item.get("id"))}


def finish(ctx: Any) -> dict[str, Any]:
    return {"done": True}


def explode(ctx: Any) -> None:
    """A step that always fails, for a candidate whose run breaks."""
    raise RuntimeError("this step always fails")
