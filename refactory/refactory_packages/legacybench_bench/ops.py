"""Operations for the LegacyBench benchmark mode.

Multi-language study, build-based gate verification (make / make test).
"""

from __future__ import annotations

import subprocess
from typing import Any

from refactory_dsh.nn import Verdict

STUDY_OUTPUT = "study-output.md"
FEEDBACK = "feedback.txt"

_SOURCE_EXTENSIONS = (
    "*.c", "*.h", "*.f", "*.f90", "*.cob", "*.cbl",
    "*.java", "*.s", "*.asm", "*.py",
)


def study(ctx: Any) -> dict[str, Any]:
    """Discover multi-language repository structure and check build state."""
    workdir = ctx.workdir
    parts = []

    parts.append("=== Source Files ===")
    find_args = ["find", ".", "-type", "f", "("]
    for i, ext in enumerate(_SOURCE_EXTENSIONS):
        if i > 0:
            find_args.append("-o")
        find_args.extend(["-name", ext])
    find_args.append(")")
    try:
        result = subprocess.run(
            find_args,
            capture_output=True, text=True, timeout=30, cwd=workdir,
        )
        lines = result.stdout.strip().splitlines()[:200]
        parts.append("\n".join(lines) if lines else "(no source files found)")
    except (subprocess.TimeoutExpired, OSError):
        parts.append("(could not list files)")

    parts.append("\n=== Build Configuration ===")
    for name in ("Makefile", "makefile", "GNUmakefile", "build.sh", "compile.sh"):
        path = f"{workdir}/{name}"
        try:
            with open(path) as f:
                content = f.read(2000)
            parts.append(f"\n--- {name} ---\n{content}")
        except FileNotFoundError:
            pass

    parts.append("\n=== Build Check ===")
    try:
        result = subprocess.run(
            ["make"],
            capture_output=True, text=True, timeout=120, cwd=workdir,
        )
        if result.returncode == 0:
            parts.append("make succeeded")
        else:
            parts.append(f"make failed (rc={result.returncode}):\n{result.stderr[:500]}")
    except (subprocess.TimeoutExpired, OSError) as exc:
        parts.append(f"make not available or timed out: {exc}")

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
    """Run make and make test; reloop on compilation or test failure."""
    workdir = ctx.workdir

    try:
        build = subprocess.run(
            ["make"],
            capture_output=True, text=True, timeout=600, cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return _again(ctx, "make timed out after 600s")
    except OSError as exc:
        return _again(ctx, f"make not available: {exc}")

    if build.returncode != 0:
        stderr = build.stderr.strip()[:500]
        return _again(ctx, f"compilation failed (rc={build.returncode}):\n{stderr}")

    try:
        test = subprocess.run(
            ["make", "test"],
            capture_output=True, text=True, timeout=600, cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return _again(ctx, "make test timed out after 600s")
    except OSError as exc:
        return _again(ctx, f"make test not available: {exc}")

    if test.returncode != 0:
        stderr = test.stderr.strip()[:500]
        stdout_tail = test.stdout.strip()[-500:]
        return _again(
            ctx,
            f"tests failed (rc={test.returncode}):\n{stderr}\n{stdout_tail}",
        )

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
