"""Operations for the TerminalBench benchmark mode.

Broad tool detection in study, wider success/failure matching in gate.
"""

from __future__ import annotations

import subprocess
from typing import Any

from refactory_dsh.nn import Verdict

STUDY_OUTPUT = "study-output.md"
FEEDBACK = "feedback.txt"

_LANGUAGES = ("python3", "gcc", "g++", "rustc", "go", "node", "java", "R")
_PKG_MANAGERS = ("pip", "apt", "npm", "cargo", "gem", "luarocks")
_TOOLS = (
    "make", "cmake", "gdb", "strace", "valgrind",
    "sqlite3", "ffmpeg", "openssl", "nmap",
)

_SUCCESS_WORDS = ("pass", "succeed", "ok", "complete", "done", "verified", "correct", "works")
_FAILURE_WORDS = ("fail", "error", "broken", "cannot", "unable", "wrong")


def _check_tools(tools: tuple[str, ...]) -> list[str]:
    available = []
    for tool in tools:
        try:
            result = subprocess.run(
                ["which", tool],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                available.append(f"  {tool}: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, OSError):
            pass
    return available


def study(ctx: Any) -> dict[str, Any]:
    """Detect installed tools and languages, read the task instruction."""
    workdir = ctx.workdir
    parts = []

    parts.append("=== Installed Languages ===")
    langs = _check_tools(_LANGUAGES)
    parts.append("\n".join(langs) if langs else "(none detected)")

    parts.append("\n=== Package Managers ===")
    pkgs = _check_tools(_PKG_MANAGERS)
    parts.append("\n".join(pkgs) if pkgs else "(none detected)")

    parts.append("\n=== Developer Tools ===")
    tools = _check_tools(_TOOLS)
    parts.append("\n".join(tools) if tools else "(none detected)")

    parts.append("\n=== Repository Structure ===")
    try:
        result = subprocess.run(
            ["find", ".", "-type", "f", "-name", "*.py"],
            capture_output=True, text=True, timeout=30, cwd=workdir,
        )
        lines = result.stdout.strip().splitlines()[:200]
        parts.append("\n".join(lines) if lines else "(no .py files found)")
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
    """Check whether the builder committed changes and task succeeded."""
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
    if any(w in lower for w in _SUCCESS_WORDS):
        return "proceed"
    if any(w in lower for w in _FAILURE_WORDS):
        return _again(ctx, "builder needs to retry — task did not succeed")

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
    """Update the main branch ref to include the solution."""
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
