"""Operations for the ProgramBench benchmark mode.

No study op — the builder starts directly. Gate checks TODOs and
compile.sh output.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from refactory_dsh.nn import Verdict

FEEDBACK = "feedback.txt"
TEST_RESULTS = "test-results.txt"


def gate_verify(ctx: Any) -> str:
    """Check TODOs from reviewer and verify compile.sh produces a working binary."""
    workdir = ctx.workdir

    todos_path = Path(workdir) / "todos.md"
    has_todos = False
    if todos_path.exists():
        content = todos_path.read_text()
        if "## TODO" in content:
            todo_lines = [
                ln for ln in content.splitlines()
                if ln.strip().startswith("- ") or ln.strip().startswith("* ")
            ]
            if todo_lines:
                has_todos = True

    compile_sh = Path(workdir) / "compile.sh"
    compile_ok = False
    results = []

    if compile_sh.exists():
        try:
            result = subprocess.run(
                ["bash", "compile.sh"],
                capture_output=True, text=True, timeout=120, cwd=workdir,
            )
            if result.returncode == 0:
                compile_ok = True
                results.append("compile.sh: SUCCESS")
            else:
                results.append(f"compile.sh: FAILED (rc={result.returncode})")
                results.append(result.stderr[:500] if result.stderr else "(no stderr)")
        except subprocess.TimeoutExpired:
            results.append("compile.sh: TIMEOUT (120s)")
        except OSError as exc:
            results.append(f"compile.sh: ERROR ({exc})")
    else:
        results.append("compile.sh: NOT FOUND")

    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD~1", "--stat"],
            capture_output=True, text=True, timeout=30, cwd=workdir,
        )
        has_commits = bool(diff.stdout.strip())
    except (subprocess.TimeoutExpired, OSError):
        has_commits = False

    results_text = "\n".join(results)
    ctx.write_text(TEST_RESULTS, results_text + "\n")

    if not has_commits:
        return _again(ctx, "builder did not commit any changes")
    if not compile_ok:
        return _again(ctx, f"compilation failed:\n{results_text}")
    if has_todos:
        return _again(ctx, "reviewer identified remaining TODOs")

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
    """Update the main branch ref to include the reconstruction."""
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
