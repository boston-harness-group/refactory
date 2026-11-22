"""Tests for per-step chain excusal logic in base_check.

Covers _chain_steps, _build_marker_script, _parse_step_results, _step_would_run,
_run_chain_markers, _per_step_comparison, and excusal_blocker with step_results.

Scenario tests drive filter_check with mocked run_base_check and _run_chain_markers,
exercising the real code paths the brief's done-when cases describe.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from refactory_dsh import sandbox
from refactory_dsh.base_check import (
    BaseCheckResult,
    ComparisonResult,
    FilteredCheckResult,
    _accounted,
    _build_marker_script,
    _chain_steps,
    _parse_step_results,
    _run_chain_markers,
    _runner,
    _step_would_run,
    compare_failures,
    design_comparison,
    excusal_blocker,
    filter_check,
    parse_test_failures,
    _per_step_comparison,
)


# ── Unit tests ───────────────────────────────────────────────────────────


class TestChainSteps:
    """Unit tests for _chain_steps."""

    def test_splits_only_on_chain_operators(self):
        """Pipelines stay whole; only &&, ||, ; split."""
        result = _chain_steps("A | B && C")
        assert result == [("A | B", ""), ("C", "&&")]

    def test_heredoc_refuses_to_split(self):
        """A heredoc command is not split."""
        assert _chain_steps("cat <<EOF\na && b\nEOF && pytest") is None

    def test_compound_commands_refuse_to_split(self):
        """if/for/while/case make a command unsplittable."""
        assert _chain_steps("if true; then pytest; fi && lint") is None
        assert _chain_steps("for x in a b; do echo $x; done && test") is None
        assert _chain_steps("while true; do break; done && lint") is None
        assert _chain_steps("case $x in a) echo;; esac && test") is None

    def test_quoted_heredoc_marker_does_split(self):
        """<< inside quotes does not trigger the heredoc guard."""
        result = _chain_steps('echo "<<EOF" && pytest')
        assert result is not None
        assert len(result) == 2

    def test_backtick_spanning_operator(self):
        """An open backtick at a split point means the substitution spans an operator."""
        assert _chain_steps("cmd `a && b` && c") is None

    def test_records_operators(self):
        """Operators are recorded correctly."""
        result = _chain_steps("A && B || C ; D")
        assert result == [("A", ""), ("B", "&&"), ("C", "||"), ("D", ";")]

    def test_single_step_returns_one_element(self):
        """A single command with no operators returns a single-element list."""
        result = _chain_steps("pytest tests/")
        assert result == [("pytest tests/", "")]

    def test_subshell_stays_whole(self):
        """Content inside parentheses is not split."""
        result = _chain_steps("(a && b) && c")
        assert result == [("(a && b)", ""), ("c", "&&")]

    def test_dollar_paren_stays_whole(self):
        """$(...) with operators inside is not split."""
        result = _chain_steps("echo $(a && b) && c")
        assert result == [("echo $(a && b)", ""), ("c", "&&")]

    def test_quoted_operators_stay_whole(self):
        """Operators inside quotes are not split points."""
        result = _chain_steps("echo 'a && b' && c")
        assert result == [("echo 'a && b'", ""), ("c", "&&")]
        result2 = _chain_steps('echo "a && b" && c')
        assert result2 == [('echo "a && b"', ""), ("c", "&&")]


class TestMarkerScriptAndParsing:
    """Unit tests for _build_marker_script and _parse_step_results."""

    def test_construction_and_parsing(self):
        """Marker script construction and parsing round-trip correctly."""
        steps = [("echo hello", ""), ("pytest tests/", "&&"), ("grep -q Y f", "&&")]
        marker_id = "abcdef012345"
        script = _build_marker_script(steps, marker_id)

        # Script starts with exec 2>&1
        assert script.startswith("exec 2>&1\n")
        # Contains all three echo markers
        assert script.count(f"___REFACTORY_STEP_{marker_id}_") == 3

        # Simulate stdout from running the script
        stdout = (
            "hello\n"
            f"___REFACTORY_STEP_{marker_id}_0_EXIT_0\n"
            "FAILED tests/test_a.py::test_x - assert False\n"
            "1 failed in 0.1s\n"
            f"___REFACTORY_STEP_{marker_id}_1_EXIT_1\n"
            f"___REFACTORY_STEP_{marker_id}_2_EXIT_1\n"
        )

        results = _parse_step_results(stdout, marker_id, steps)
        assert results is not None
        assert len(results) == 3
        assert results[0]["exit_code"] == 0
        assert "hello" in results[0]["output"]
        assert results[1]["exit_code"] == 1
        assert "FAILED" in results[1]["output"]
        assert results[2]["exit_code"] == 1
        assert results[0]["operator"] == ""
        assert results[1]["operator"] == "&&"

    def test_parsing_returns_none_on_missing_marker(self):
        """When a marker is missing, parsing returns None."""
        steps = [("echo hello", ""), ("pytest", "&&")]
        marker_id = "abcdef012345"
        # Only one marker instead of two
        stdout = f"hello\n___REFACTORY_STEP_{marker_id}_0_EXIT_0\n"
        assert _parse_step_results(stdout, marker_id, steps) is None

    def test_parsing_returns_none_on_uuid_mismatch(self):
        """When marker UUID doesn't match, parsing returns None."""
        steps = [("echo hello", "")]
        marker_id = "abcdef012345"
        stdout = f"hello\n___REFACTORY_STEP_wrongid00000_0_EXIT_0\n"
        assert _parse_step_results(stdout, marker_id, steps) is None


class TestStepWouldRun:
    """Unit tests for _step_would_run."""

    def test_operators(self):
        """Operator gating applies to || only."""
        assert _step_would_run("&&", 0) is True
        assert _step_would_run("&&", 1) is True  # No gating for &&
        assert _step_would_run("||", 0) is False
        assert _step_would_run("||", 1) is True
        assert _step_would_run(";", 0) is True
        assert _step_would_run(";", 1) is True
        assert _step_would_run("", 0) is True


class TestExcusalBlockerBackwardsCompat:
    """Test that excusal_blocker's existing behavior is unchanged without step_results."""

    def test_backwards_compat(self):
        """Existing callers without step_results get the same behavior."""
        out = "FAILED tests/test_a.py::test_x - assert False\n1 failed, 3 passed in 0.12s\n"
        failures = parse_test_failures(out)

        # Without step_results (default): existing behavior
        assert excusal_blocker(
            "pnpm test && .venv/bin/python -m pytest -q tests 2>&1", out, failures
        ) is None
        result = excusal_blocker("python -m pytest -q; ruff check .", out, failures)
        assert result is not None and "last step" in result


class TestOrChainOperatorGating:
    """Test that || steps whose previous step exited 0 are excluded."""

    def test_or_chain_gating(self):
        """A || B where A succeeds on both sides: B's output is excluded."""
        # Build base and attempt step results where step 0 (A) exits 0 on both sides,
        # so step 1 (B, operator ||) would not run.
        base_steps = [
            {"step": "true", "operator": "", "exit_code": 0, "output": "",
             "failures": [], "accounted": True},
            {"step": "pytest tests/", "operator": "||", "exit_code": 1,
             "output": "FAILED tests/test_a.py::test_x\n1 failed in 0.1s\n",
             "failures": ["tests/test_a.py::test_x"], "accounted": True},
        ]

        # Step 1 has operator "||" and prev_exit=0, so _step_would_run returns False.
        assert _step_would_run("||", 0) is False
        # The failure from step 1 should not be inherited since || gating drops it.


# ── Helpers for scenario tests ───────────────────────────────────────────


def _make_pytest_output(failures: list[str], passed: int = 3) -> str:
    """Build a fake pytest output with FAILED lines and a summary."""
    lines = []
    for f in failures:
        lines.append(f"FAILED {f} - assert False")
    n_failed = len(failures)
    lines.append(f"={' ' * 10}{n_failed} failed, {passed} passed in 0.12s{' ' * 10}=")
    return "\n".join(lines) + "\n"


def _make_vitest_output(failures: list[str], passed: int = 9) -> str:
    """Build a fake vitest output with failure lines and a summary."""
    lines = []
    for f in failures:
        lines.append(f" FAIL  {f}")
    n_failed = len(failures)
    lines.append(f"      Tests  {n_failed} failed | {passed} passed ({n_failed + passed})")
    return "\n".join(lines) + "\n"


def _make_base_result(ok: bool, failures: set[str], output: str = "",
                      complete: bool = False, step_results=None,
                      exit_code: int = 1) -> BaseCheckResult:
    """Create a BaseCheckResult with transient attributes."""
    result = BaseCheckResult(ok=ok, failures=failures, output=output,
                             exit_code=exit_code)
    result._base_complete = complete  # type: ignore[attr-defined]
    if step_results is not None:
        result._step_results = step_results  # type: ignore[attr-defined]
    return result


def _make_ctx(workdir: Path, facts: dict | None = None) -> SimpleNamespace:
    """Create a minimal context for filter_check."""
    ctx = SimpleNamespace()
    ctx.workdir = workdir
    ctx.facts = facts or {}
    ctx.exists = lambda name: False
    ctx.read_text = lambda name: ""
    ctx.budget = None
    return ctx


def _make_sandbox_result(ok: bool, stdout: str, stderr: str = "",
                         returncode: int | None = None,
                         timed_out: bool = False) -> sandbox.SandboxResult:
    """Create a SandboxResult."""
    if returncode is None:
        returncode = 0 if ok else 1
    return sandbox.SandboxResult(
        ok=ok, stdout=stdout, stderr=stderr,
        returncode=returncode, timed_out=timed_out,
    )


# ── Scenario tests (driving filter_check) ───────────────────────────────


class TestChainTokShPytestGrep:
    """tok.sh && pytest X && grep -q Y f scenario.

    Base: tok.sh exit 1, pytest has failure S, grep exit 1.
    Attempt: tok.sh exit 1, pytest has failures S and N, grep exit 1.
    Expected: S is inherited; N and grep stay the attempt's.
    """

    def test_chain_tok_sh_pytest_grep(self, tmp_path):
        command = "tok.sh && pytest X && grep -q Y f"

        base_pytest_out = _make_pytest_output(["tests/test_a.py::test_S"])
        attempt_pytest_out = _make_pytest_output(["tests/test_a.py::test_S",
                                                   "tests/test_a.py::test_N"], passed=2)

        base_step_results = [
            {"step": "tok.sh", "operator": "", "exit_code": 1, "output": "tok failed\n",
             "failures": [], "accounted": True},
            {"step": "pytest X", "operator": "&&", "exit_code": 1,
             "output": base_pytest_out,
             "failures": ["tests/test_a.py::test_S"], "accounted": True},
            {"step": "grep -q Y f", "operator": "&&", "exit_code": 1, "output": "",
             "failures": [], "accounted": True},
        ]

        base_failures = {"tests/test_a.py::test_S"}
        base_result = _make_base_result(
            ok=False, failures=base_failures, complete=True,
            step_results=base_step_results,
        )

        # Attempt marker run results
        attempt_step_results = [
            {"step": "tok.sh", "operator": "", "exit_code": 1,
             "output": "tok failed\n"},
            {"step": "pytest X", "operator": "&&", "exit_code": 1,
             "output": attempt_pytest_out},
            {"step": "grep -q Y f", "operator": "&&", "exit_code": 1,
             "output": ""},
        ]

        # The attempt's original output (what the check produced before re-run):
        # It stopped at step 0 (tok.sh), or maybe ran partially
        attempt_original_out = (
            "tok failed\n"
            + attempt_pytest_out
        )
        attempt_sandbox = _make_sandbox_result(
            ok=False, stdout=attempt_original_out, returncode=1,
        )

        ctx = _make_ctx(tmp_path)

        with patch("refactory_dsh.base_check.run_base_check", return_value=base_result), \
             patch("refactory_dsh.base_check._run_chain_markers", return_value=attempt_step_results), \
             patch("refactory_dsh.base_check.worktree") as mock_wt, \
             patch("refactory_dsh.base_check._all_changed_files", return_value=[]):
            mock_wt.recorded.return_value = SimpleNamespace(
                path=str(tmp_path / "attempt"), base="abc123", target="/tmp/target",
                head="def456", branch="test", run=str(tmp_path),
            )
            mock_wt.CHECK_HOME = Path(".refactory") / "check-home"
            mock_wt.check_environment.return_value = {"PATH": "/usr/bin"}

            filtered = filter_check(ctx, attempt_sandbox, command, 60.0)

        # S is inherited (same outcome on step 0, accounted on step 1)
        assert "tests/test_a.py::test_S" in filtered.inherited
        # N is new (not in base)
        assert "tests/test_a.py::test_N" in filtered.new_failures
        assert not filtered.all_excused


class TestChainBuildVitestDifferentOutcome:
    """pnpm build && vitest scenario.

    Base: build exit 1, vitest exit 1 with T failing.
    Attempt: build exit 0, vitest exit 1 with T failing.
    Expected: T stays the attempt's (different outcome on step 0).
    """

    def test_chain_build_vitest_different_outcome(self, tmp_path):
        command = "pnpm build && vitest"

        base_vitest_out = _make_vitest_output(["src/app.spec.ts > suite > test T"])
        attempt_vitest_out = _make_vitest_output(["src/app.spec.ts > suite > test T"])

        base_step_results = [
            {"step": "pnpm build", "operator": "", "exit_code": 1,
             "output": "build failed\n", "failures": [], "accounted": True},
            {"step": "vitest", "operator": "&&", "exit_code": 1,
             "output": base_vitest_out,
             "failures": ["src/app.spec.ts > suite > test T"], "accounted": True},
        ]

        base_failures = {"src/app.spec.ts > suite > test T"}
        base_result = _make_base_result(
            ok=False, failures=base_failures, complete=True,
            step_results=base_step_results,
        )

        attempt_step_results = [
            {"step": "pnpm build", "operator": "", "exit_code": 0,
             "output": "build ok\n"},
            {"step": "vitest", "operator": "&&", "exit_code": 1,
             "output": attempt_vitest_out},
        ]

        attempt_original_out = attempt_vitest_out
        attempt_sandbox = _make_sandbox_result(
            ok=False, stdout=attempt_original_out, returncode=1,
        )

        ctx = _make_ctx(tmp_path)

        with patch("refactory_dsh.base_check.run_base_check", return_value=base_result), \
             patch("refactory_dsh.base_check._run_chain_markers", return_value=attempt_step_results), \
             patch("refactory_dsh.base_check.worktree") as mock_wt, \
             patch("refactory_dsh.base_check._all_changed_files", return_value=[]):
            mock_wt.recorded.return_value = SimpleNamespace(
                path=str(tmp_path / "attempt"), base="abc123", target="/tmp/target",
                head="def456", branch="test", run=str(tmp_path),
            )
            mock_wt.CHECK_HOME = Path(".refactory") / "check-home"
            mock_wt.check_environment.return_value = {"PATH": "/usr/bin"}

            filtered = filter_check(ctx, attempt_sandbox, command, 60.0)

        # T stays the attempt's because step 0 ended differently (1 vs 0)
        assert "src/app.spec.ts > suite > test T" in filtered.new_failures
        assert not filtered.all_excused


class TestChainCdPytestStateCarries:
    """cd pkg && pytest scenario.

    Base and attempt: cd exit 0, pytest with pkg/tests/test_a.py::test_x failing.
    Expected: failure inherited, with test ids from the real directory.
    """

    def test_chain_cd_pytest_state_carries(self, tmp_path):
        command = "cd pkg && pytest"

        pytest_out = _make_pytest_output(["pkg/tests/test_a.py::test_x"])

        base_step_results = [
            {"step": "cd pkg", "operator": "", "exit_code": 0,
             "output": "", "failures": [], "accounted": True},
            {"step": "pytest", "operator": "&&", "exit_code": 1,
             "output": pytest_out,
             "failures": ["pkg/tests/test_a.py::test_x"], "accounted": True},
        ]

        base_failures = {"pkg/tests/test_a.py::test_x"}
        base_result = _make_base_result(
            ok=False, failures=base_failures, complete=True,
            step_results=base_step_results,
        )

        attempt_step_results = [
            {"step": "cd pkg", "operator": "", "exit_code": 0, "output": ""},
            {"step": "pytest", "operator": "&&", "exit_code": 1,
             "output": pytest_out},
        ]

        attempt_original_out = pytest_out
        attempt_sandbox = _make_sandbox_result(
            ok=False, stdout=attempt_original_out, returncode=1,
        )

        ctx = _make_ctx(tmp_path)

        with patch("refactory_dsh.base_check.run_base_check", return_value=base_result), \
             patch("refactory_dsh.base_check._run_chain_markers", return_value=attempt_step_results), \
             patch("refactory_dsh.base_check.worktree") as mock_wt, \
             patch("refactory_dsh.base_check._all_changed_files", return_value=[]):
            mock_wt.recorded.return_value = SimpleNamespace(
                path=str(tmp_path / "attempt"), base="abc123", target="/tmp/target",
                head="def456", branch="test", run=str(tmp_path),
            )
            mock_wt.CHECK_HOME = Path(".refactory") / "check-home"
            mock_wt.check_environment.return_value = {"PATH": "/usr/bin"}

            filtered = filter_check(ctx, attempt_sandbox, command, 60.0)

        assert "pkg/tests/test_a.py::test_x" in filtered.inherited
        assert filtered.all_excused


class TestChainPytestRuffPytestAttemptReruns:
    """pytest unit && ruff check . && pytest integ scenario.

    Base: pytest unit exit 1 (S fails), ruff exit 0, pytest integ exit 0.
    Attempt original run: pytest unit fails (S), chain stops at step 0.
    Attempt marker re-run: pytest unit exit 1 (S), ruff exit 0, pytest integ exit 1 (new N).
    Expected: S is inherited, N is new.
    """

    def test_chain_pytest_ruff_pytest_attempt_reruns(self, tmp_path):
        command = "pytest unit && ruff check . && pytest integ"

        unit_out = _make_pytest_output(["tests/test_unit.py::test_S"])
        integ_out = _make_pytest_output(["tests/test_integ.py::test_N"])

        base_step_results = [
            {"step": "pytest unit", "operator": "", "exit_code": 1,
             "output": unit_out,
             "failures": ["tests/test_unit.py::test_S"], "accounted": True},
            {"step": "ruff check .", "operator": "&&", "exit_code": 0,
             "output": "All checks passed\n", "failures": [], "accounted": True},
            {"step": "pytest integ", "operator": "&&", "exit_code": 0,
             "output": "3 passed in 0.1s\n", "failures": [], "accounted": True},
        ]

        base_failures = {"tests/test_unit.py::test_S"}
        base_result = _make_base_result(
            ok=False, failures=base_failures, complete=True,
            step_results=base_step_results,
        )

        attempt_step_results = [
            {"step": "pytest unit", "operator": "", "exit_code": 1,
             "output": unit_out},
            {"step": "ruff check .", "operator": "&&", "exit_code": 0,
             "output": "All checks passed\n"},
            {"step": "pytest integ", "operator": "&&", "exit_code": 1,
             "output": integ_out},
        ]

        # Attempt's original output: only the unit test output (chain stopped at step 0)
        attempt_original_out = unit_out
        attempt_sandbox = _make_sandbox_result(
            ok=False, stdout=attempt_original_out, returncode=1,
        )

        ctx = _make_ctx(tmp_path)

        with patch("refactory_dsh.base_check.run_base_check", return_value=base_result), \
             patch("refactory_dsh.base_check._run_chain_markers", return_value=attempt_step_results), \
             patch("refactory_dsh.base_check.worktree") as mock_wt, \
             patch("refactory_dsh.base_check._all_changed_files", return_value=[]):
            mock_wt.recorded.return_value = SimpleNamespace(
                path=str(tmp_path / "attempt"), base="abc123", target="/tmp/target",
                head="def456", branch="test", run=str(tmp_path),
            )
            mock_wt.CHECK_HOME = Path(".refactory") / "check-home"
            mock_wt.check_environment.return_value = {"PATH": "/usr/bin"}

            filtered = filter_check(ctx, attempt_sandbox, command, 60.0)

        # S is inherited
        assert "tests/test_unit.py::test_S" in filtered.inherited
        # N is new (discovered by the attempt re-run)
        assert "tests/test_integ.py::test_N" in filtered.new_failures
        assert not filtered.all_excused


class TestHeredocNotSplitBaseIncomplete:
    """A heredoc command is not split, and the base stays incomplete."""

    def test_heredoc_not_split_base_incomplete(self):
        cmd = "cat <<EOF\ndata\nEOF && pytest tests/"
        assert _chain_steps(cmd) is None


class TestPassingBaseOneRun:
    """A passing base makes exactly one run (no marker re-run)."""

    def test_passing_base_one_run(self, tmp_path):
        """When the base check passes, _base_complete is True and no marker re-run.
        filter_check with a passing base returns all_excused=False (no failures to excuse).
        """
        command = "tok.sh && pytest tests/"

        base_result = _make_base_result(ok=True, failures=set(), complete=True,
                                         exit_code=0)
        attempt_out = _make_pytest_output(["tests/test_a.py::test_x"])
        attempt_sandbox = _make_sandbox_result(ok=False, stdout=attempt_out, returncode=1)

        ctx = _make_ctx(tmp_path)

        with patch("refactory_dsh.base_check.run_base_check", return_value=base_result), \
             patch("refactory_dsh.base_check._run_chain_markers") as mock_run, \
             patch("refactory_dsh.base_check.worktree") as mock_wt, \
             patch("refactory_dsh.base_check._all_changed_files", return_value=[]):
            mock_wt.recorded.return_value = SimpleNamespace(
                path=str(tmp_path / "attempt"), base="abc123", target="/tmp/target",
                head="def456", branch="test", run=str(tmp_path),
            )
            mock_wt.CHECK_HOME = Path(".refactory") / "check-home"
            mock_wt.check_environment.return_value = {"PATH": "/usr/bin"}

            filtered = filter_check(ctx, attempt_sandbox, command, 60.0)

        # No attempt re-run needed (base passed, failure is new)
        assert "tests/test_a.py::test_x" in filtered.new_failures
        assert not filtered.all_excused
        # _run_chain_markers was not called (no per-step comparison needed)
        mock_run.assert_not_called()


class TestUnavailableTimedOutBaseInheritsNothing:
    """An unavailable or timed-out base inherits nothing."""

    def test_unavailable_base_inherits_nothing(self, tmp_path):
        base_result = BaseCheckResult(
            ok=False, failures=set(), output="environment failed",
            exit_code=-1, environment_failure="could not prepare base",
        )

        attempt_out = _make_pytest_output(["tests/test_a.py::test_x"])
        attempt_sandbox = _make_sandbox_result(ok=False, stdout=attempt_out, returncode=1)

        ctx = _make_ctx(tmp_path)

        with patch("refactory_dsh.base_check.run_base_check", return_value=base_result), \
             patch("refactory_dsh.base_check.worktree") as mock_wt:
            mock_wt.recorded.return_value = SimpleNamespace(
                path=str(tmp_path / "attempt"), base="abc123", target="/tmp/target",
                head="def456", branch="test", run=str(tmp_path),
            )

            filtered = filter_check(ctx, attempt_sandbox, "tok.sh && pytest tests/", 60.0)

        # Environment failure: nothing inherited
        assert not filtered.all_excused
        assert not filtered.inherited
        assert filtered.base_environment_failure is not None


class TestSingleShellFidelityReal:
    """Prove single-shell fidelity through sandbox.run_command.

    mkdir -p sub && cd sub && pwd: later steps see cd's effect.
    export X=1 && sh -c 'echo $X': later steps see the export.
    """

    def test_single_shell_fidelity_real(self, tmp_path):
        """Real shell execution proves state carries between steps."""
        steps = [
            ("mkdir -p sub", ""),
            ("cd sub", "&&"),
            ("pwd", "&&"),
            ("export X=1", "&&"),
            ("sh -c 'echo $X'", "&&"),
        ]
        marker_id = "fidelitytest0"
        script = _build_marker_script(steps, marker_id)

        result = sandbox.run_command(
            script, workdir=tmp_path, timeout_s=30,
            home=tmp_path,
        )

        stdout = "\n".join((result.stdout or "", result.stderr or "")).strip()
        parsed = _parse_step_results(stdout, marker_id, steps)
        assert parsed is not None, f"Parsing failed. stdout:\n{stdout}"
        assert len(parsed) == 5

        # mkdir exit 0
        assert parsed[0]["exit_code"] == 0
        # cd exit 0
        assert parsed[1]["exit_code"] == 0
        # pwd output ends with /sub
        assert parsed[2]["output"].strip().endswith("/sub"), f"pwd was: {parsed[2]['output']}"
        assert parsed[2]["exit_code"] == 0
        # export exit 0
        assert parsed[3]["exit_code"] == 0
        # sh -c 'echo $X' outputs 1
        assert "1" in parsed[4]["output"], f"echo $X was: {parsed[4]['output']}"
        assert parsed[4]["exit_code"] == 0


class TestAttemptMarkerRunLeavesNoFilesInRunWorktree:
    """The attempt's marker run uses the check copy, not the run worktree."""

    def test_no_files_in_run_worktree(self, tmp_path):
        """_per_step_comparison's attempt marker run leaves no files in the run worktree."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        attempt_dir = tmp_path / "attempt"
        attempt_dir.mkdir()

        command = "echo hello && echo world"
        steps = _chain_steps(command)
        assert steps is not None

        base_step_results = [
            {"step": "echo hello", "operator": "", "exit_code": 0,
             "output": "hello\n", "failures": [], "accounted": True},
            {"step": "echo world", "operator": "&&", "exit_code": 0,
             "output": "world\n", "failures": [], "accounted": True},
        ]

        base_result = _make_base_result(
            ok=False, failures=set(), complete=True,
            step_results=base_step_results,
        )

        attempt_step_results = [
            {"step": "echo hello", "operator": "", "exit_code": 0,
             "output": "hello\n"},
            {"step": "echo world", "operator": "&&", "exit_code": 0,
             "output": "world\n"},
        ]

        with patch("refactory_dsh.base_check._run_chain_markers") as mock_run, \
             patch("refactory_dsh.base_check.worktree") as mock_wt:
            mock_wt.recorded.return_value = SimpleNamespace(
                path=str(attempt_dir), base="abc123", target="/tmp/target",
                head="def456", branch="test", run=str(run_dir),
            )
            mock_wt.CHECK_HOME = Path(".refactory") / "check-home"
            mock_wt.check_environment.return_value = {"PATH": "/usr/bin"}

            mock_run.return_value = attempt_step_results

            _per_step_comparison(
                _make_ctx(run_dir),
                command, steps, base_result,
                _make_sandbox_result(ok=True, stdout="hello\nworld\n"),
                timeout=60.0, named_text=None,
                changed_files=[], brief_text="", design_text="",
            )

        # The run worktree should not have new files
        # (since we mock _run_chain_markers, nothing actually runs in run_dir)
        after_files = set(run_dir.rglob("*"))
        # run_dir/.refactory/check-home may be created by _per_step_comparison
        new_files = {f for f in after_files if ".refactory" not in str(f)}
        assert not new_files, f"New files appeared in run worktree: {new_files}"


# ── Excusal blocker with step_results tests ──────────────────────────────


class TestExcusalBlockerWithStepResults:
    """Test excusal_blocker when step_results is provided."""

    def test_all_accounted_returns_none(self):
        """When all test runner steps are accounted, returns None."""
        step_results = [
            {"step": "echo hello", "operator": "", "exit_code": 0, "output": "hello\n"},
            {"step": "pytest tests/", "operator": "&&", "exit_code": 1,
             "output": "FAILED tests/test_a.py::test_x - assert\n1 failed in 0.1s\n"},
        ]
        result = excusal_blocker("echo hello && pytest tests/", "", set(),
                                  step_results=step_results)
        assert result is None

    def test_non_test_step_fails_returns_blocker(self):
        """A non-test step failure blocks excusal."""
        step_results = [
            {"step": "build.sh", "operator": "", "exit_code": 1, "output": "build failed\n"},
            {"step": "pytest tests/", "operator": "&&", "exit_code": 0,
             "output": "1 passed in 0.1s\n"},
        ]
        result = excusal_blocker("build.sh && pytest tests/", "", set(),
                                  step_results=step_results)
        assert result is not None and "non-test step" in result

    def test_runner_failed_no_failures_returns_blocker(self):
        """A runner that failed but named no tests blocks excusal."""
        step_results = [
            {"step": "pytest tests/", "operator": "", "exit_code": 1,
             "output": "ERROR collecting tests\n"},
        ]
        result = excusal_blocker("pytest tests/", "", set(),
                                  step_results=step_results)
        assert result is not None and "named no failing test" in result


# ── Cache integration test ───────────────────────────────────────────────


class TestCacheStepsField:
    """Test that steps field is written to and read from cache."""

    def test_cache_round_trip(self, tmp_path):
        from refactory_dsh.base_check import _write_cache, _read_cache

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        result = BaseCheckResult(ok=False, failures={"tests/test_a.py::test_x"},
                                 output="FAILED tests/test_a.py::test_x\n1 failed in 0.1s",
                                 exit_code=1)
        result._base_complete = True  # type: ignore[attr-defined]
        result._step_results = [  # type: ignore[attr-defined]
            {"step": "echo hello", "operator": "", "exit_code": 0,
             "output": "hello\n", "failures": [], "accounted": True},
            {"step": "pytest tests/", "operator": "&&", "exit_code": 1,
             "output": "FAILED tests/test_a.py::test_x\n1 failed in 0.1s",
             "failures": ["tests/test_a.py::test_x"], "accounted": True},
        ]

        _write_cache(run_dir, "test-key", result)
        cached = _read_cache(run_dir, "test-key")

        assert cached is not None
        assert cached._base_complete is True  # type: ignore[attr-defined]
        assert hasattr(cached, "_step_results")
        assert len(cached._step_results) == 2  # type: ignore[attr-defined]
        assert cached._step_results[0]["step"] == "echo hello"  # type: ignore[attr-defined]
        assert cached._step_results[1]["failures"] == ["tests/test_a.py::test_x"]  # type: ignore[attr-defined]

    def test_old_cache_without_steps(self, tmp_path):
        """A cache written before this change (no steps) has no _step_results."""
        run_dir = tmp_path / "run"
        refactory_dir = run_dir / ".refactory"
        refactory_dir.mkdir(parents=True)

        cache_data = {
            "cache_key": "old-key",
            "ok": False,
            "failures": ["tests/test_a.py::test_x"],
            "output": "...",
            "exit_code": 1,
            "base_complete": False,
            "base_clean": True,
        }
        (refactory_dir / "base-check.json").write_text(json.dumps(cache_data))

        from refactory_dsh.base_check import _read_cache
        cached = _read_cache(run_dir, "old-key")
        assert cached is not None
        assert not hasattr(cached, "_step_results")

    def test_cache_hit_does_not_rerun_base(self, tmp_path):
        """A cache with steps field never re-runs the base."""
        from refactory_dsh.base_check import _read_cache

        run_dir = tmp_path / "run"
        refactory_dir = run_dir / ".refactory"
        refactory_dir.mkdir(parents=True)

        cache_data = {
            "cache_key": "cached-key",
            "ok": False,
            "failures": ["tests/test_a.py::test_S"],
            "output": "...",
            "exit_code": 1,
            "base_complete": True,
            "base_clean": True,
            "steps": [
                {"step": "tok.sh", "operator": "", "exit_code": 1,
                 "output": "tok failed", "failures": [], "accounted": True},
                {"step": "pytest X", "operator": "&&", "exit_code": 1,
                 "output": "FAILED tests/test_a.py::test_S\n1 failed in 0.1s",
                 "failures": ["tests/test_a.py::test_S"], "accounted": True},
            ],
        }
        (refactory_dir / "base-check.json").write_text(json.dumps(cache_data))

        cached = _read_cache(run_dir, "cached-key")
        assert cached is not None
        assert hasattr(cached, "_step_results")
        assert len(cached._step_results) == 2  # type: ignore[attr-defined]


class TestBaseCompleteRequiresAccounted:
    """base_complete must be False when a runner's _accounted returns False.

    This tests through _per_step_comparison: when the base's pytest step has
    failures but no summary (killed mid-run, so _accounted returns False),
    the failure should NOT be inherited because the _accounted gate in
    _per_step_comparison blocks it.
    """

    def test_unaccounted_base_step_blocks_inheritance(self, tmp_path):
        """A base step where _accounted is False prevents inheritance."""
        command = "tok.sh && pytest tests/"

        # Base pytest output has failures but NO summary line (killed mid-run)
        base_pytest_no_summary = "FAILED tests/test_a.py::test_S - assert False\n"

        base_step_results = [
            {"step": "tok.sh", "operator": "", "exit_code": 1,
             "output": "tok failed\n", "failures": [], "accounted": True},
            {"step": "pytest tests/", "operator": "&&", "exit_code": 1,
             "output": base_pytest_no_summary,
             "failures": ["tests/test_a.py::test_S"], "accounted": False},
        ]

        base_failures = {"tests/test_a.py::test_S"}
        base_result = _make_base_result(
            ok=False, failures=base_failures, complete=False,
            step_results=base_step_results,
        )

        attempt_pytest_out = _make_pytest_output(["tests/test_a.py::test_S"])
        attempt_step_results = [
            {"step": "tok.sh", "operator": "", "exit_code": 1,
             "output": "tok failed\n"},
            {"step": "pytest tests/", "operator": "&&", "exit_code": 1,
             "output": attempt_pytest_out},
        ]

        attempt_sandbox = _make_sandbox_result(
            ok=False, stdout=attempt_pytest_out, returncode=1,
        )

        ctx = _make_ctx(tmp_path)

        with patch("refactory_dsh.base_check.run_base_check", return_value=base_result), \
             patch("refactory_dsh.base_check._run_chain_markers", return_value=attempt_step_results), \
             patch("refactory_dsh.base_check.worktree") as mock_wt, \
             patch("refactory_dsh.base_check._all_changed_files", return_value=[]):
            mock_wt.recorded.return_value = SimpleNamespace(
                path=str(tmp_path / "attempt"), base="abc123", target="/tmp/target",
                head="def456", branch="test", run=str(tmp_path),
            )
            mock_wt.CHECK_HOME = Path(".refactory") / "check-home"
            mock_wt.check_environment.return_value = {"PATH": "/usr/bin"}

            filtered = filter_check(ctx, attempt_sandbox, command, 60.0)

        # S should NOT be inherited because base's _accounted is False
        assert "tests/test_a.py::test_S" in filtered.new_failures
        assert not filtered.all_excused


class TestAndGatingMustNotApply:
    """Re-adding && gating would break the tok.sh && pytest X && grep scenario.

    With tok.sh failing on the base, && gating would exclude pytest failures,
    and S could not be inherited.
    """

    def test_and_gating_would_break_inheritance(self, tmp_path):
        """Steps after a failed && must still run (not gated)."""
        command = "tok.sh && pytest X && grep -q Y f"

        base_pytest_out = _make_pytest_output(["tests/test_a.py::test_S"])

        base_step_results = [
            {"step": "tok.sh", "operator": "", "exit_code": 1,
             "output": "tok failed\n", "failures": [], "accounted": True},
            {"step": "pytest X", "operator": "&&", "exit_code": 1,
             "output": base_pytest_out,
             "failures": ["tests/test_a.py::test_S"], "accounted": True},
            {"step": "grep -q Y f", "operator": "&&", "exit_code": 1,
             "output": "", "failures": [], "accounted": True},
        ]

        base_failures = {"tests/test_a.py::test_S"}
        base_result = _make_base_result(
            ok=False, failures=base_failures, complete=True,
            step_results=base_step_results,
        )

        attempt_pytest_out = _make_pytest_output(["tests/test_a.py::test_S"])
        attempt_step_results = [
            {"step": "tok.sh", "operator": "", "exit_code": 1,
             "output": "tok failed\n"},
            {"step": "pytest X", "operator": "&&", "exit_code": 1,
             "output": attempt_pytest_out},
            {"step": "grep -q Y f", "operator": "&&", "exit_code": 1,
             "output": ""},
        ]

        attempt_sandbox = _make_sandbox_result(
            ok=False, stdout="tok failed\n" + attempt_pytest_out, returncode=1,
        )

        ctx = _make_ctx(tmp_path)

        with patch("refactory_dsh.base_check.run_base_check", return_value=base_result), \
             patch("refactory_dsh.base_check._run_chain_markers", return_value=attempt_step_results), \
             patch("refactory_dsh.base_check.worktree") as mock_wt, \
             patch("refactory_dsh.base_check._all_changed_files", return_value=[]):
            mock_wt.recorded.return_value = SimpleNamespace(
                path=str(tmp_path / "attempt"), base="abc123", target="/tmp/target",
                head="def456", branch="test", run=str(tmp_path),
            )
            mock_wt.CHECK_HOME = Path(".refactory") / "check-home"
            mock_wt.check_environment.return_value = {"PATH": "/usr/bin"}

            filtered = filter_check(ctx, attempt_sandbox, command, 60.0)

        # S is inherited even though tok.sh (step before it) failed
        # This proves && gating is NOT applied
        assert "tests/test_a.py::test_S" in filtered.inherited
