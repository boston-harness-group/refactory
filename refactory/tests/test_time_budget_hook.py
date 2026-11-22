"""The per-test time budget hook fails over-budget tests only when --time-budget is given.

Three scenarios, exercised by running pytest as a subprocess on a small inline test
with a minimal conftest that re-exports only the budget-related hooks:

1. ``--time-budget=1``, unmarked test takes 1.5 s → failed by the hook.
2. ``--time-budget=1``, ``@pytest.mark.slow`` test takes 1.5 s → not failed.
3. No ``--time-budget``, test takes 1.5 s → passes normally.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


_INLINE_TEST = """\
import time, pytest

def test_over_budget():
    time.sleep(1.5)

@pytest.mark.slow
def test_over_budget_but_slow():
    time.sleep(1.5)

def test_under_budget():
    pass
"""

# A minimal conftest that registers the same hooks the real conftest.py has for
# the time budget, without importing the full conftest (which needs the checkout).
_INLINE_CONFTEST = """\
import sys, os
# Put the real tests directory on the path so we can import conftest's hooks.
_tests_dir = {tests_dir!r}
sys.path.insert(0, _tests_dir)
# Also add the parent so test support modules are importable.
sys.path.insert(0, os.path.dirname(_tests_dir))

import pytest
from collections.abc import Iterator

def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False)
    parser.addoption("--time-budget", type=int, default=None, metavar="SECONDS")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: test exceeds the CI time budget")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow", default=False):
        skip_slow = pytest.mark.skip(reason="pass --run-slow to run slow tests; they run nightly")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

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
            f"Test took {{report.duration:.1f}}s, exceeding the {{budget}}s budget. "
            "Mark it @pytest.mark.slow or speed it up."
        )
"""


@pytest.fixture(scope="module")
def _test_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("budget")
    tests_dir = str(Path(__file__).resolve().parent)
    (root / "conftest.py").write_text(_INLINE_CONFTEST.format(tests_dir=tests_dir))
    (root / "test_budget.py").write_text(_INLINE_TEST)
    return root


def _run_pytest(test_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir / "test_budget.py"), "-v",
         f"--rootdir={test_dir}", "-p", "no:xdist", *extra_args],
        capture_output=True, text=True, timeout=60,
    )


def test_the_hook_fails_an_unmarked_test_over_the_budget(_test_dir):
    result = _run_pytest(_test_dir, "--time-budget=1", "--run-slow")
    assert "test_over_budget FAILED" in result.stdout, result.stdout + result.stderr
    assert "exceeding the 1s budget" in result.stdout, result.stdout
    # The slow-marked one passes even though it is over budget.
    assert "test_over_budget_but_slow PASSED" in result.stdout, result.stdout
    assert "test_under_budget PASSED" in result.stdout, result.stdout


def test_slow_tests_are_skipped_without_run_slow(_test_dir):
    result = _run_pytest(_test_dir, "--time-budget=1")
    assert "test_over_budget_but_slow SKIPPED" in result.stdout, result.stdout + result.stderr


def test_no_budget_means_no_failure(_test_dir):
    result = _run_pytest(_test_dir, "--run-slow")
    assert "test_over_budget PASSED" in result.stdout, result.stdout + result.stderr
    assert "test_over_budget_but_slow PASSED" in result.stdout, result.stdout
