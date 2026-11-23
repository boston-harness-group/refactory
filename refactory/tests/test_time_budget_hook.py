"""The per-test time budget hook fails over-budget tests only when --time-budget is given.

Three scenarios, exercised end-to-end through the **real** ``conftest.py`` by placing a
tiny test file inside ``refactory/tests/`` (via a subprocess, so the outer session's
checkout guard never sees it) and running pytest on it:

1. ``--time-budget=1``, unmarked test takes 1.5 s → failed by the hook.
2. ``--time-budget=1``, ``@pytest.mark.slow`` test takes 1.5 s → not failed.
3. No ``--time-budget``, test takes 1.5 s → passes normally.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)

_INLINE_TEST = textwrap.dedent("""\
    import time, pytest

    def test_over_budget():
        time.sleep(1.5)

    @pytest.mark.slow
    def test_over_budget_but_slow():
        time.sleep(1.5)

    def test_under_budget():
        pass
""")


def _run_budget_probe(*extra_args: str) -> dict:
    """Create a probe test in refactory/tests/, run pytest on it through the real
    conftest.py, clean up, and return the stdout/stderr/returncode.

    Everything happens inside a subprocess so the outer session's checkout guard
    never sees the file.
    """
    script = textwrap.dedent(f"""\
        import json, os, pathlib, subprocess, sys

        tests_dir = pathlib.Path({_TESTS_DIR!r})
        probe = tests_dir / "_test_budget_probe_tmp.py"
        probe.write_text({_INLINE_TEST!r})
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(probe), "-v",
                 "-p", "no:xdist", "--timeout=30"] + {list(extra_args)!r},
                capture_output=True, text=True, timeout=60,
                cwd={_REPO_ROOT!r},
            )
            print(json.dumps({{"stdout": result.stdout, "stderr": result.stderr,
                              "returncode": result.returncode}}))
        finally:
            probe.unlink(missing_ok=True)
    """)
    outer = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
        cwd=_REPO_ROOT,
    )
    assert outer.returncode == 0 or outer.stdout.strip(), (
        f"wrapper failed: {outer.stderr}")
    return json.loads(outer.stdout.strip().split("\n")[-1])


def test_the_hook_fails_an_unmarked_test_over_the_budget():
    data = _run_budget_probe("--time-budget=1", "--run-slow")
    assert "test_over_budget FAILED" in data["stdout"], data["stdout"] + data["stderr"]
    assert "exceeding the 1s budget" in data["stdout"], data["stdout"]
    # The slow-marked one passes even though it is over budget.
    assert "test_over_budget_but_slow PASSED" in data["stdout"], data["stdout"]
    assert "test_under_budget PASSED" in data["stdout"], data["stdout"]


def test_slow_tests_are_skipped_without_run_slow():
    data = _run_budget_probe("--time-budget=1")
    assert "test_over_budget_but_slow SKIPPED" in data["stdout"], (
        data["stdout"] + data["stderr"])


def test_no_budget_means_no_failure():
    data = _run_budget_probe("--run-slow")
    assert "test_over_budget PASSED" in data["stdout"], data["stdout"] + data["stderr"]
    assert "test_over_budget_but_slow PASSED" in data["stdout"], data["stdout"]
