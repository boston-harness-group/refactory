"""Ops for a data pass whose gate reloops (`test_data_parallel`)."""
from __future__ import annotations

from typing import Any

#: How many attempts an item takes before its gate proceeds.
ATTEMPTS = 5


def attempt(ctx: Any) -> dict[str, Any]:
    count = int(ctx.read_json("attempts.json", 0) or 0) + 1
    ctx.write_json("attempts.json", count)
    return {"attempted": count}


def again(ctx: Any) -> str:
    return "proceed" if int(ctx.read_json("attempts.json", 0) or 0) >= ATTEMPTS else "reloop"
