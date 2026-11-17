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


def right_unless_first(ctx: Any) -> dict[str, Any]:
    """A solving op that answers ``right`` -- except on item ``0``, where it raises (the
    ``fallback`` operator's genuine failure)."""
    if str((ctx.read_json("current_item.json", {}) or {}).get("id")) == "0":
        raise ValueError("item 0 breaks the solver")
    ctx.write_text("answer.txt", "right\n")
    return {}


def _first_item(ctx: Any) -> bool:
    return str((ctx.read_json("current_item.json", {}) or {}).get("id")) == "0"


def _tries(ctx: Any) -> int:
    """How many times this op has run in the item's pass, this run included (a file of
    its own in the pass directory, which both runtimes run it in)."""
    from pathlib import Path

    counter = Path(ctx.workdir) / f".{ctx.node_id}-tries"
    tries = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(tries))
    return tries


def right_from_second_try(ctx: Any) -> dict[str, Any]:
    """Answers ``right``; on item ``0`` its first run raises (the runner's one op retry
    answers it)."""
    if _first_item(ctx) and _tries(ctx) < 2:
        raise ValueError("item 0 breaks the solver once")
    ctx.write_text("answer.txt", "right\n")
    return {}


def right_from_third_try(ctx: Any) -> dict[str, Any]:
    """Answers ``right``; on item ``0`` its first visit (two runs, with the op retry)
    raises, so its fallback answers, and a gate's reloop back to it finds it working."""
    if _first_item(ctx) and _tries(ctx) < 3:
        raise ValueError("item 0 breaks the solver on its first visit")
    ctx.write_text("answer.txt", "right\n")
    return {}


def partial_unless_first(ctx: Any) -> dict[str, Any]:
    """Answers ``right``; on item ``0`` it writes a partial answer, then raises."""
    if _first_item(ctx):
        ctx.write_text("answer.txt", "partial\n")
        raise ValueError("item 0 breaks the solver half way")
    ctx.write_text("answer.txt", "right\n")
    return {}


def snippet_timeout_on_first(ctx: Any) -> dict[str, Any]:
    """Answers ``right``; on item ``0`` it runs over a code step's wall."""
    from refactory_dsh.snippet import SnippetTimeout

    if _first_item(ctx):
        raise SnippetTimeout("the code ran over its 10 s limit")
    ctx.write_text("answer.txt", "right\n")
    return {}


def sandbox_down_on_first(ctx: Any) -> dict[str, Any]:
    """Answers ``right``; on item ``0`` the sandbox does not start."""
    from refactory_dsh.snippet import SandboxUnavailable

    if _first_item(ctx):
        raise SandboxUnavailable("the sandbox did not start within 60 s")
    ctx.write_text("answer.txt", "right\n")
    return {}


def own_timeout(ctx: Any) -> dict[str, Any]:
    """An op whose own subprocess runs over its limit on item ``0``."""
    import subprocess

    if _first_item(ctx):
        raise subprocess.TimeoutExpired(["slow-tool"], 1)
    ctx.write_text("a.txt", "right\n")
    return {}


def right_b(ctx: Any) -> dict[str, Any]:
    ctx.write_text("b.txt", "right\n")
    return {}


def draft_right(ctx: Any) -> dict[str, Any]:
    ctx.write_text("answer.txt", "right\n")
    return {}


def rewrite_unless_first(ctx: Any) -> dict[str, Any]:
    """Reads the draft and rewrites it; on item ``0`` it rewrites it badly, then raises."""
    draft = ctx.read_text("answer.txt")
    if _first_item(ctx):
        ctx.write_text("answer.txt", "garbled\n")
        raise ValueError("item 0 breaks the solver after rewriting")
    ctx.write_text("answer.txt", draft)
    return {}


def keep_answer(ctx: Any) -> dict[str, Any]:
    """A fallback that keeps the draft it is given (and notes what it saw in kept.txt)."""
    from pathlib import Path

    draft = ctx.read_text("answer.txt") if ctx.exists("answer.txt") else "(not written)\n"
    (Path(ctx.workdir) / "kept.txt").write_text(draft)
    ctx.write_text("answer.txt", draft)
    return {}
