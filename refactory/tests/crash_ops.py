"""Op and gate helpers for test_crashed_step_retry: graded from the pass's files.

Each tolerates a missing answer, which is what a crashed agent step whose retries are
spent leaves behind (`nn.graph.Graph.missing_on_crash`).
"""
from __future__ import annotations

from typing import Any


def _answer(ctx: Any, name: str = "answer.txt") -> str:
    return ctx.read_text(name).strip() if ctx.exists(name) else ""


def check(ctx: Any) -> Any:
    """The checker gate: proceed on a right answer, else send the solver back with why."""
    from refactory_dsh.nn import Verdict

    answer = _answer(ctx)
    if answer == "right":
        return "proceed"
    return Verdict("reloop", reason=f"{answer} is not right" if answer else "there is no answer")


def score(ctx: Any) -> dict[str, Any]:
    """Grade the answer as it stands: 1.0 right, 0.25 wrong, 0.0 none."""
    answer = _answer(ctx)
    ctx.write_json("item_result.json", {"score": 1.0 if answer == "right" else 0.25 if answer else 0.0})
    return {}


def pick(ctx: Any) -> dict[str, Any]:
    """Keep the first branch answer that is right (a vote over the fork's branches)."""
    answers = [_answer(ctx, name) for name in ("a.txt", "b.txt")]
    ctx.write_json("item_result.json", {"score": 1.0 if "right" in answers else 0.0,
                                        "answered": sum(1 for answer in answers if answer)})
    return {}


def pick_with_metadata(ctx: Any) -> dict[str, Any]:
    """`pick`, then, on item ``0``, a genuine bug: it reads ``metadata.txt``, which nothing
    writes."""
    pick(ctx)
    if str((ctx.read_json("current_item.json", {}) or {}).get("id")) == "0":
        ctx.read_text("metadata.txt")
    return {}


def extract(ctx: Any) -> dict[str, Any]:
    """Grade ``answer.txt``, which this step declares as a required input (the Countdown
    seed's ``extract_answer``): it never runs without one."""
    answer = ctx.read_text("answer.txt").strip()
    ctx.write_json("item_result.json", {"score": 1.0 if answer == "right" else 0.25})
    return {}


def note_unless_first(ctx: Any) -> dict[str, Any]:
    """Write ``notes.txt`` for the next step -- except, by a bug, on item ``0``."""
    if str((ctx.read_json("current_item.json", {}) or {}).get("id")) != "0":
        ctx.write_text("notes.txt", "noted\n")
    return {}
