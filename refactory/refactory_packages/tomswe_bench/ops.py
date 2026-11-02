"""Operations for the TomSWE benchmark mode.

Same study/gate/merge pattern as swebench — the differentiation is in
the builder prompt, not the ops.
"""

from __future__ import annotations

import subprocess
from typing import Any

from refactory_dsh.nn import Verdict

STUDY_OUTPUT = "study-output.md"
FEEDBACK = "feedback.txt"


def study(ctx: Any) -> dict[str, Any]:
    """Discover repository structure and read the task instruction."""
    workdir = ctx.workdir
    parts = []
    parts.append("=== Repository Structure ===")
    try:
        result = subprocess.run(
            ["find", ".", "-type", "f", "-name", "*.py"],
            capture_output=True, text=True, timeout=30, cwd=workdir,
        )
        lines = result.stdout.strip().splitlines()[:200]
        parts.append("\n".join(lines) if lines else "(no .py files found)")
    except (subprocess.TimeoutExpired, OSError):
        parts.append("(could not list files)")

    parts.append("\n=== Test Files ===")
    try:
        result = subprocess.run(
            ["find", ".", "-type", "f", "(",
             "-name", "test_*.py", "-o", "-name", "*_test.py", ")"],
            capture_output=True, text=True, timeout=30, cwd=workdir,
        )
        lines = result.stdout.strip().splitlines()[:50]
        parts.append("\n".join(lines) if lines else "(no test files found)")
    except (subprocess.TimeoutExpired, OSError):
        parts.append("(could not list test files)")

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
    """Check whether the builder committed changes and reports tests passing."""
    workdir = ctx.workdir
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD~1", "--stat"],
            capture_output=True, text=True, timeout=30, cwd=workdir,
        )
        changes = diff.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        changes = ""

    if not changes:
        return _again(ctx, "builder did not commit any changes")

    builder_output = ""
    try:
        builder_output = ctx.read_text("builder-latest.md")
    except (FileNotFoundError, OSError):
        pass

    lower = builder_output.lower()
    if any(w in lower for w in ("tests pass", "test pass", "tests succeed", "passed")):
        return "proceed"
    if any(w in lower for w in ("tests fail", "test fail", "failed", "error")):
        return _again(ctx, "builder needs to retry — tests did not pass")

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
    """Update the main branch ref to include the builder's fix."""
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
