"""Pytest configuration for the refactory-dsh test harness."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Benchmark controllers are repository-only test infrastructure, deliberately
# excluded from the installed runtime package. Make them importable under the
# documented pytest command without requiring a caller-supplied PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ops declared by a mode are resolved by module name at run time, so the test ops
# have to be importable by name rather than only by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The credential probe and the hook that skips the marked tests live together in
# `model_credential`, and a module imported by a test is not a plugin: its hooks are
# never called unless it is registered. Registering it here is what makes the skip
# fire, and it is also why the probe can stay deferred -- pytest calls the hook only
# at collection, and only this conftest pulls the module in.
pytest_plugins = ("model_credential",)

# PROMPT_MUTATE and expandable-KNOB_MUTATE shell out to `claude` by default.
# For the fast, deterministic harness we point that at /bin/true (empty output),
# which makes `default_prompt_rewriter` fall back to its deterministic prompt
# variants instead of calling a live LLM. The real DSH inner loop test runs the
# actual CEO route, so this only affects the surrogate/unit surface.
os.environ.setdefault("FACTORY_CLAUDE_BIN", "/bin/true")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run slow tests that spawn a real DSH CEO run",
    )
    parser.addoption(
        "--toolchain-gate",
        action="store_true",
        default=False,
        help=(
            "fail if a test marked `toolchain` did not run against its real "
            "external tool, and report what was verified"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: slow tests that spawn a real DSH CEO run"
    )
    config.addinivalue_line(
        "markers",
        "toolchain(name): needs a real external tool (lean, mathlib, sympy); "
        "checked by --toolchain-gate",
    )
    config.addinivalue_line(
        "markers",
        "needs_model_credential: needs a model credential the authoring sandbox "
        "accepts; the tests/model_credential.py hook skips it with a stated reason "
        "when this machine has none",
    )


#: nodeid -> tool, for every test that declares it needs a real external tool.
#:
#: A skip is invisible: the suite goes green and says nothing about the path the
#: skipped test covered, which is how two real Lean/mathlib defects hid. The
#: marker is the one place the set is written down, and the gate below reads it,
#: so it is not a list that drifts from the tests it is about (issue #44).
_TOOLCHAIN: dict[str, str] = {}
#: nodeid -> ("passed" | "failed" | "skipped", reason)
_TOOLCHAIN_RESULT: dict[str, tuple[str, str]] = {}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        marker = item.get_closest_marker("toolchain")
        if marker is not None:
            _TOOLCHAIN[item.nodeid] = str(marker.args[0]) if marker.args else "unknown"
    if config.getoption("--run-integration", default=False):
        return
    skip = pytest.mark.skip(reason="pass --run-integration to run DSH inner-loop tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


def pytest_deselected(items: list[pytest.Item]) -> None:
    """A deselected toolchain test did not run; `-k` must not hide it from the gate."""
    for item in items:
        marker = item.get_closest_marker("toolchain")
        if marker is not None:
            _TOOLCHAIN[item.nodeid] = str(marker.args[0]) if marker.args else "unknown"
            _TOOLCHAIN_RESULT[item.nodeid] = ("deselected", "")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record what happened to each marked test, including the reason it skipped."""
    if report.nodeid not in _TOOLCHAIN:
        return
    if report.skipped:
        reason = ""
        if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
            reason = str(report.longrepr[2])
        _TOOLCHAIN_RESULT[report.nodeid] = ("skipped", reason)
    elif report.failed:
        _TOOLCHAIN_RESULT[report.nodeid] = ("failed", "")
    elif report.when == "call" and report.passed:
        _TOOLCHAIN_RESULT[report.nodeid] = ("passed", "")


def _unverified() -> list[tuple[str, str, str, str]]:
    """Every marked test that did not pass, with its tool, outcome, and reason."""
    problems: list[tuple[str, str, str, str]] = []
    for nodeid, tool in sorted(_TOOLCHAIN.items()):
        outcome, reason = _TOOLCHAIN_RESULT.get(nodeid, ("not run", ""))
        if outcome not in ("passed", "failed"):
            problems.append((nodeid, tool, outcome, reason))
    return problems


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """The gate: a green suite that skipped a tool-dependent test is not a pass."""
    if not session.config.getoption("--toolchain-gate", default=False):
        return
    if _unverified():
        session.exitstatus = 1


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config
) -> None:
    """Say which tool-dependent tests ran and which did not, whenever any are marked."""
    if not _TOOLCHAIN:
        return
    gated = config.getoption("--toolchain-gate", default=False)
    verified: dict[str, int] = {}
    for nodeid, tool in _TOOLCHAIN.items():
        if _TOOLCHAIN_RESULT.get(nodeid, ("", ""))[0] == "passed":
            verified[tool] = verified.get(tool, 0) + 1
    problems = _unverified()
    if not gated and not problems:
        return
    terminalreporter.write_sep("=", "toolchain gate")
    if verified:
        summary = ", ".join(f"{tool}: {count}" for tool, count in sorted(verified.items()))
        terminalreporter.write_line(f"verified against real tools: {summary}")
    if problems:
        terminalreporter.write_line(
            "UNVERIFIED: these tests need a real external tool and did not run:"
        )
        for nodeid, tool, outcome, reason in problems:
            terminalreporter.write_line(
                f"  [{tool}] {nodeid}: {outcome} -- {reason or 'no reason given'}"
            )
        terminalreporter.write_line(
            "a green suite here says nothing about the checker path; install the "
            "tools or accept that this machine did not verify it"
        )
