"""Operations for the DevOpsGym benchmark mode.

Build-system-aware study and gate; standard merge.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from refactory_dsh.nn import Verdict

STUDY_OUTPUT = "study-output.md"
FEEDBACK = "feedback.txt"

BUILD_FILES = (
    "pom.xml", "build.gradle", "build.gradle.kts",
    "go.mod", "Makefile", "Dockerfile",
    "Cargo.toml", "package.json", "CMakeLists.txt",
    ".gitlab-ci.yml",
)


def study(ctx: Any) -> dict[str, Any]:
    """Discover build system configuration and read the task instruction."""
    workdir = ctx.workdir
    parts = []
    parts.append("=== Build System Detection ===")

    found = []
    for name in BUILD_FILES:
        path = os.path.join(workdir, name)
        if os.path.isfile(path):
            found.append(name)
            try:
                with open(path) as f:
                    content = f.read(1000)
                parts.append(f"\n--- {name} ---\n{content}")
            except OSError:
                parts.append(f"\n--- {name} --- (could not read)")

    gh_dir = os.path.join(workdir, ".github", "workflows")
    if os.path.isdir(gh_dir):
        try:
            for fname in os.listdir(gh_dir):
                if fname.endswith((".yml", ".yaml")):
                    found.append(f".github/workflows/{fname}")
                    fpath = os.path.join(gh_dir, fname)
                    with open(fpath) as f:
                        content = f.read(1000)
                    parts.append(f"\n--- .github/workflows/{fname} ---\n{content}")
        except OSError:
            pass

    if not found:
        parts.append("(no recognized build files found)")
    else:
        parts.insert(1, f"Found: {', '.join(found)}\n")

    parts.append("\n=== Repository Structure ===")
    try:
        result = subprocess.run(
            ["find", ".", "-maxdepth", "3", "-type", "f"],
            capture_output=True, text=True, timeout=30, cwd=workdir,
        )
        lines = result.stdout.strip().splitlines()[:150]
        parts.append("\n".join(lines) if lines else "(empty)")
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
    """Try the project's build system; reloop on failure."""
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
        return _again(ctx, "solver did not commit any changes")

    build_commands = []
    if os.path.isfile(os.path.join(workdir, "pom.xml")):
        build_commands.append(("Maven", ["mvn", "verify"]))
    if os.path.isfile(os.path.join(workdir, "build.gradle")) or \
       os.path.isfile(os.path.join(workdir, "build.gradle.kts")):
        build_commands.append(("Gradle", ["gradle", "build"]))
    if os.path.isfile(os.path.join(workdir, "go.mod")):
        build_commands.append(("Go", ["go", "build", "./..."]))
    if os.path.isfile(os.path.join(workdir, "Makefile")):
        build_commands.append(("Make", ["make"]))

    if not build_commands:
        return "proceed"

    for name, cmd in build_commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=600, cwd=workdir,
            )
            if result.returncode != 0:
                stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
                return _again(ctx, f"{name} build failed (exit {result.returncode}):\n{stderr_tail}")
        except subprocess.TimeoutExpired:
            return _again(ctx, f"{name} build timed out after 600s")
        except OSError as exc:
            return _again(ctx, f"{name} build could not start: {exc}")

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
    """Update the main branch ref to include the solver's fix."""
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
