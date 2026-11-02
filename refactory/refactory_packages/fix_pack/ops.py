"""The operations `make_pass` calls: its own prepare, check and report, on `repo_change`'s steps."""

from __future__ import annotations

import json
import re
from typing import Any

from refactory_dsh import repo_change
from refactory_dsh.nn import Verdict

FEEDBACK = "feedback.txt"


def prepare(ctx: Any) -> dict[str, Any]:
    """Read the check to make pass, and ask `repo_change.locate` where the run works."""
    task = dict(getattr(ctx, "task", {}) or {})
    check = str(task.get("check") or ctx.knob("check", "") or "").strip()
    if not check:
        raise ValueError("make_pass needs the check to make pass, e.g. {'check': 'python -m pytest -q'}")
    brief = str(task.get("brief") or ctx.knob("brief", "") or "").strip() or f"make `{check}` pass"
    # Named as design names a run without one: its first words, as a new project's name.
    named = str(task.get("name") or ctx.knob("name", "") or "").strip()
    name = named or "_".join(re.findall(r"[a-z0-9]+", brief.lower())[:3]) or "make_pass"
    located = repo_change.locate(ctx, name, task)
    feedback = "(no check has run yet)"
    ctx.write_text(FEEDBACK, feedback + "\n")
    ctx.write_json("brief.json", {"name": name, "brief": brief, "check": check})
    return {"name": name, "brief": brief, "check": check, "feedback": feedback, **located}


def check(ctx: Any) -> str:
    """Run the check where the run works; another attempt, with what it said, until it passes."""
    refused = repo_change.refused(ctx)
    if refused:
        return _again(ctx, f"materialise could not apply emit.txt, so nothing was checked: {refused}")
    result = repo_change.run_check(ctx, str(ctx.facts["check"]), int(ctx.knob("check_timeout_s", 300)))
    if result.ok:
        return "proceed"
    return _again(ctx, "\n".join((result.problem(), result.stdout or "", result.stderr or "")).strip())


def _again(ctx: Any, feedback: str) -> Verdict:
    used = int(ctx.facts.get("attempts_used", 1))
    if used >= int(ctx.knob("attempts", 3)):
        raise ValueError(f"the check still fails after {used} attempt(s): {feedback}")
    ctx.write_text(FEEDBACK, feedback + "\n")
    ctx.set_fact("feedback", feedback)
    ctx.set_fact("attempts_used", used + 1)
    return Verdict("reloop", feedback)


def report(ctx: Any) -> dict[str, Any]:
    """Say which branch holds the fix, in result.json (`repo_change.report`)."""
    payload = repo_change.report(ctx, "result.json", {
        "check": ctx.facts["check"],
        "files": ctx.facts.get("files", []),
        "attempts": int(ctx.facts.get("attempts_used", 1)),
    })
    return {"result": json.dumps(payload, default=str)}
