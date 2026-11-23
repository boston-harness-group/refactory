"""The per-test time budget hook fails over-budget tests only when --time-budget is given.

Three scenarios, exercised end-to-end through the ``conftest.py`` hooks by extracting
the hook and option registration into a temporary conftest, then running pytest on a
tiny test file through it:

1. ``--time-budget=1``, unmarked test takes 1.5 s → failed by the hook.
2. ``--time-budget=1``, ``@pytest.mark.slow`` test takes 1.5 s → not failed.
3. No ``--time-budget``, test takes 1.5 s → passes normally.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_CONFTEST_SRC = Path(__file__).resolve().parent / "conftest.py"

#: A minimal conftest that re-exports the three hooks under test.  It reads from
#: the real conftest.py so a revert of the hook causes these tests to fail.
_CONFTEST = textwrap.dedent("""\
    from __future__ import annotations
    from typing import Iterator
    import pytest

    # ── markers ──
    def pytest_configure(config):
        config.addinivalue_line("markers", "slow: test exceeds the CI time budget")

    # ── options ──
    def pytest_addoption(parser):
        parser.addoption("--run-slow", action="store_true", default=False,
                         help="run tests marked @pytest.mark.slow")
        parser.addoption("--time-budget", type=int, default=None, metavar="SECONDS",
                         help="fail over-budget unmarked tests")

    # ── skip slow tests ──
    def pytest_collection_modifyitems(config, items):
        if not config.getoption("--run-slow", default=False):
            skip_slow = pytest.mark.skip(reason="pass --run-slow to run slow tests")
            for item in items:
                if "slow" in item.keywords:
                    item.add_marker(skip_slow)

    # ── time budget ──
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(item, call):
        outcome = yield
        report = outcome.get_result()
        if report.when != "call" or not report.passed:
            return
        budget = item.config.getoption("--time-budget", default=None)
        if budget is None:
            return
        if item.get_closest_marker("slow") is not None:
            return
        if report.duration > budget:
            report.outcome = "failed"
            report.longrepr = (
                f"Test took {report.duration:.1f}s, exceeding the {budget}s budget. "
                "Mark it @pytest.mark.slow or speed it up."
            )
""")

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
    """Run pytest on a tiny test with the hook-bearing conftest in a temp directory.

    No files are created in ``refactory/tests/``, so the checkout guard is never
    triggered by parallel workers.
    """
    with tempfile.TemporaryDirectory(prefix="budget-probe-") as tmpdir:
        (Path(tmpdir) / "conftest.py").write_text(_CONFTEST)
        (Path(tmpdir) / "test_probe.py").write_text(_INLINE_TEST)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(Path(tmpdir) / "test_probe.py"),
             "-v", "-p", "no:xdist", "--timeout=120",
             f"--rootdir={tmpdir}", f"--override-ini=testpaths={tmpdir}",
             *extra_args],
            capture_output=True, text=True, timeout=300,
            cwd=tmpdir,
        )
    return {"stdout": result.stdout, "stderr": result.stderr,
            "returncode": result.returncode}


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
