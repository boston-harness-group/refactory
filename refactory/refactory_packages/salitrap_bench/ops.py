"""Operations for the Salitrap benchmark mode.

Minimal study, answer-file gate, standard merge.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from refactory_dsh.nn import Verdict

STUDY_OUTPUT = "study-output.md"
FEEDBACK = "feedback.txt"
ANSWER_FILE = "/workspace/answer.txt"


def study(ctx: Any) -> dict[str, Any]:
    """Discover workspace contents and read the task instruction."""
    workdir = ctx.workdir
    parts = []
    parts.append("=== Workspace Contents ===")
    try:
        result = subprocess.run(
            ["find", ".", "-type", "f"],
            capture_output=True, text=True, timeout=30, cwd=workdir,
        )
        lines = result.stdout.strip().splitlines()[:200]
        parts.append("\n".join(lines) if lines else "(empty workspace)")
    except (subprocess.TimeoutExpired, OSError):
        parts.append("(could not list files)")

    parts.append("\n=== Task Instruction ===")
    try:
        with open("/tmp/task-instruction.md") as f:
            parts.append(f.read())
    except FileNotFoundError:
        parts.append("No task instruction file found at /tmp/task-instruction.md")

    output = "\n".join(parts)
    ctx.write_text(STUDY_OUTPUT, output)
    return {"study_output": output[:500]}


def gate_verify(ctx: Any) -> str:
    """Check whether the solver wrote an answer file and committed."""
    workdir = ctx.workdir

    if not os.path.isfile(ANSWER_FILE):
        return _again(ctx, "solver did not write /workspace/answer.txt")

    try:
        with open(ANSWER_FILE) as f:
            content = f.read().strip()
    except OSError:
        content = ""

    if not content:
        return _again(ctx, "/workspace/answer.txt is empty")

    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD~1", "--stat"],
            capture_output=True, text=True, timeout=30, cwd=workdir,
        )
        changes = diff.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        changes = ""

    if not changes:
        return _again(ctx, "solver did not commit any changes")

    return "proceed"


def _again(ctx: Any, feedback: str) -> Verdict:
    used = int(ctx.facts.get("attempts_used", 1))
    if used >= int(ctx.knob("attempts", 3)):
        raise ValueError(f"gate still failing after {used} attempt(s): {feedback}")
    ctx.write_text(FEEDBACK, feedback + "\n")
    ctx.set_fact("feedback", feedback)
    ctx.set_fact("attempts_used", used + 1)
    return Verdict("reloop", feedback)


def auto_merge(ctx: Any) -> dict[str, Any]:
    """Update the main branch ref to include the solver's answer."""
    workdir = ctx.workdir
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=workdir,
        )
        current = result.stdout.strip()
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=workdir,
        )
        head_short = result.stdout.strip()
        subprocess.run(
            ["git", "update-ref", "refs/heads/main", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=workdir,
        )
        msg = f"Updated main to {head_short} (was on {current})"
    except (subprocess.TimeoutExpired, OSError) as exc:
        msg = f"auto_merge failed: {exc}"
    return {"merge_result": msg}
