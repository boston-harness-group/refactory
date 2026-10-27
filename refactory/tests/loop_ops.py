"""Operations the executor tests resolve by name.

They live here rather than in the library because they exist to exercise the
executor, not to be an operation any real mode would call.
"""

from __future__ import annotations

from typing import Any


def write_number(ctx: Any) -> dict[str, Any]:
    value = int(ctx.knob("step", 1))
    ctx.write_json("number.json", {"value": value})
    return {"value": value}


def double(ctx: Any) -> dict[str, Any]:
    data = ctx.read_json("number.json", {"value": 0})
    doubled = int(data["value"]) * 2
    ctx.write_json("doubled.json", {"value": doubled})
    return {"doubled": doubled}


def undeclared_write(ctx: Any) -> None:
    ctx.write_json("surprise.json", {})


def always_retry(ctx: Any) -> str:
    return "retry"


def three_passes(ctx: Any) -> str:
    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    return "retry" if visits < 3 else "proceed"


def explained_passes(ctx: Any) -> str:
    """`three_passes`, saying why each retry happens."""
    from refactory_dsh.nn import Verdict

    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    return Verdict("retry", f"pass {visits} of 3") if visits < 3 else "proceed"


def bogus_verdict(ctx: Any) -> str:
    return "nonsense"


def boom(ctx: Any) -> None:
    raise ValueError("this op fails on purpose")


def write_best(ctx: Any) -> dict[str, Any]:
    return {"best": 42.0}


def noop(ctx: Any) -> None:
    """Do nothing — a placeholder for graphs that need a start node."""
    pass


def record_item(ctx: Any) -> dict[str, Any]:
    """Append the current item's id, so a test can see one pass per item."""
    item = ctx.read_json("current_item.json", {}) or {}
    seen = ctx.read_json("seen.json", {"ids": []}) or {"ids": []}
    seen.setdefault("ids", []).append(str(item.get("id")))
    ctx.write_json("seen.json", seen)
    return {"seen": seen["ids"]}


def explained_passes_in_a_fact(ctx: Any) -> str:
    """`explained_passes`, also leaving the reason as a fact a template can render."""
    from refactory_dsh.nn import Verdict

    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    ctx.set_fact("why", f"pass {visits} of 3")
    return Verdict("retry", f"pass {visits} of 3") if visits < 3 else "proceed"


def pause(ctx: Any) -> None:
    """Take a moment, so a sibling branch finishes first."""
    import time

    time.sleep(0.3)


def retry_each_item_once(ctx: Any) -> str:
    """Send the current item's pass back once, saying why; then go on."""
    from refactory_dsh.nn import Verdict

    item = str((ctx.read_json("current_item.json", {}) or {}).get("id"))
    tried = ctx.read_json("tried.json", []) or []
    if item in tried:
        return "proceed"
    ctx.write_json("tried.json", [*tried, item])
    return Verdict("retry", f"item {item} needs another look")


#: A reason past the runner's 4000-character limit: a whole check log, line by line.
LONG_REASON = "\n".join(f"check log line {number}: tests/test_widget.py::test_{number} failed"
                        for number in range(150))


def long_reason_once(ctx: Any) -> str:
    """Send the work back once with a reason longer than the runner keeps, left as a fact too."""
    from refactory_dsh.nn import Verdict

    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    ctx.set_fact("why", LONG_REASON)
    return Verdict("retry", LONG_REASON) if visits < 2 else "proceed"


def always_proceed(ctx: Any) -> str:
    return "proceed"


def fail_on_second_visit(ctx: Any) -> None:
    """Pass the first time, fail every time after: a run stopped right after a reloop."""
    visits = (ctx.read_json("visits.json", 0) or 0) + 1
    ctx.write_json("visits.json", visits)
    if visits > 1:
        raise RuntimeError("this op fails on its second visit")


def reloop_twice(ctx: Any) -> str:
    """`three_passes` on an explicit `reloop` edge, for a cycle with another gate in it."""
    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    return "reloop" if visits < 3 else "proceed"
