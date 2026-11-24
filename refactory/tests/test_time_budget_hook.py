"""The per-test time budget and the ``slow`` mark, through the real ``conftest.py``.

Each test runs a separate pytest session over a tiny test file in ``tmp_path``, with this
directory's real ``conftest.py`` loaded as a plugin (``-p conftest``), so the hooks under
test are the ones PR CI runs, not a copy. Nothing is written into the checkout.

The four sessions are the four ways the suite is run:

- PR CI: ``--time-budget`` given, ``--run-slow`` not.
- Locally and in design-run checks: neither option.
- Nightly ``slow-tests``: ``--run-slow``, no budget.
- ``--time-budget`` with ``--run-slow``: the budget never fails a ``slow``-marked test.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from refactory_dsh.design import CI_TIME_BUDGET_S

_TESTS_DIR = Path(__file__).resolve().parent
_REPO = _TESTS_DIR.parent.parent
_WORKFLOWS = _REPO / ".github" / "workflows"

#: The inner tests' call phases: one far under the budget, two far over it.
_BUDGET = "0.3"
_OVER_S = 0.8

_INNER = f"""\
import time

import pytest


def test_fast():
    pass


def test_over_budget():
    time.sleep({_OVER_S})


@pytest.mark.slow
def test_over_budget_marked_slow():
    time.sleep({_OVER_S})
"""


def _session(tmp_path: Path, *args: str) -> dict[str, tuple[str, str]]:
    """Run the inner file through the real conftest; return name -> (outcome, message)."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_inner.py").write_text(_INNER)
    report = tmp_path / "report.xml"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_TESTS_DIR), str(_REPO / "refactory"), env.get("PYTHONPATH", "")])
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop("PYTEST_ADDOPTS", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_inner.py", "-p", "conftest",
         "-p", "no:cacheprovider", "-p", "no:xdist", "-c", "pytest.ini",
         f"--rootdir={tmp_path}", f"--basetemp={tmp_path / 'inner-tmp'}",
         f"--junitxml={report}", "-q", "--timeout=120", *args],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert report.exists(), proc.stdout + proc.stderr
    results: dict[str, tuple[str, str]] = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                outcome = tag
                message = (node.get("message") or "") + (node.text or "")
                break
        results[case.get("name", "")] = (outcome, message)
    assert set(results) == {"test_fast", "test_over_budget", "test_over_budget_marked_slow"}, (
        proc.stdout + proc.stderr)
    return results


def test_pr_ci_fails_an_unmarked_test_over_the_budget_and_skips_slow_tests(tmp_path):
    results = _session(tmp_path, f"--time-budget={_BUDGET}")
    assert results["test_fast"] == ("passed", "")
    outcome, message = results["test_over_budget"]
    assert outcome == "failure", results
    assert f"exceeding the {_BUDGET}s budget" in message, message
    assert re.search(r"Test took \d+\.\ds", message), message
    assert "@pytest.mark.slow" in message, message
    outcome, message = results["test_over_budget_marked_slow"]
    assert outcome == "skipped", results
    assert "--run-slow" in message, message


def test_the_budget_is_off_by_default(tmp_path):
    results = _session(tmp_path)
    assert results["test_fast"] == ("passed", "")
    assert results["test_over_budget"] == ("passed", ""), results
    # Slow tests still skip without --run-slow, budget or not.
    assert results["test_over_budget_marked_slow"][0] == "skipped", results


def test_nightly_runs_slow_tests_without_a_budget(tmp_path):
    results = _session(tmp_path, "--run-slow")
    assert results == {name: ("passed", "") for name in results}, results


def test_the_budget_never_fails_a_test_marked_slow(tmp_path):
    results = _session(tmp_path, f"--time-budget={_BUDGET}", "--run-slow")
    assert results["test_over_budget_marked_slow"] == ("passed", ""), results
    assert results["test_over_budget"][0] == "failure", results
    assert results["test_fast"] == ("passed", "")


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_a_budget_that_is_not_a_positive_number_is_a_usage_error(tmp_path, value):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_inner.py").write_text("def test_x():\n    pass\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(_TESTS_DIR), str(_REPO / "refactory")]))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_inner.py", "-p", "conftest",
         "-p", "no:cacheprovider", "-p", "no:xdist", "-c", "pytest.ini",
         f"--rootdir={tmp_path}", "-q", f"--time-budget={value}"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert "--time-budget" in proc.stderr, proc.stderr


def _pytest_commands(workflow: Path) -> list[str]:
    """Every ``pytest`` invocation in a workflow file, folded scalars joined into one line."""
    commands: list[str] = []
    lines = workflow.read_text().splitlines()
    for index, line in enumerate(lines):
        if "python -m pytest" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        parts = [line.strip()]
        for follow in lines[index + 1:]:
            if not follow.strip() or len(follow) - len(follow.lstrip()) != indent:
                break
            parts.append(follow.strip())
        commands.append(" ".join(parts))
    return commands


def test_only_the_pr_ci_python_suite_passes_the_budget():
    ci = _pytest_commands(_WORKFLOWS / "ci.yml")
    suite = [command for command in ci if "--shard" in command]
    assert len(suite) == 1, ci
    assert f"--time-budget={CI_TIME_BUDGET_S}" in suite[0].split(), suite[0]
    assert "--run-slow" not in suite[0], suite[0]
    others = [command for command in ci if command not in suite]
    assert others and not any("--time-budget" in command for command in others), others
    nightly = _pytest_commands(_WORKFLOWS / "nightly.yml")
    assert not any("--time-budget" in command for command in nightly), nightly
    slow = [command for command in nightly if "-m slow" in command]
    assert len(slow) == 1 and "--run-slow" in slow[0].split(), nightly
