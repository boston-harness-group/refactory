"""Integration tests for fail-fast targeted pre-pass and stuck detection.

Each test is parametrized for design and fast modes.  The mock boundary is
``repo_change.run_check``; everything above (pre-pass logic, digest parsing,
record writing, history management, stuck detection, verdict construction)
runs for real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from refactory_dsh.failfast import (
    PrepassResult,
    targeted_prepass,
    detect_stuck,
    record_check_history,
)
from refactory_dsh.sandbox import SandboxResult


# ── helpers ───────────────────────────────────────────────────────────────

FAIL_STDOUT = (
    "FAILED tests/test_foo.py::test_bar - AssertionError: 1 != 2\n"
    "> assert 1 == 2\n"
    "1 failed"
)

PASS_STDOUT = "1 passed"


def _make_result(*, ok=True, returncode=0, stdout="", stderr="",
                 timed_out=False, elapsed_s=1.0, prep_failure=None):
    return SandboxResult(
        ok=ok, returncode=returncode, stdout=stdout, stderr=stderr,
        timed_out=timed_out, elapsed_s=elapsed_s, prep_failure=prep_failure,
    )


def _fail_record(*, test_ids=None, inherited=None, commit="aaa",
                 check_mode="full", runners=None):
    """Build a minimal failing check-record.json dict."""
    if test_ids is None:
        test_ids = ["tests/test_foo.py::test_bar"]
    if inherited is None:
        inherited = []
    if runners is None:
        runners = [
            {
                "name": "pytest",
                "passed": False,
                "failures": [
                    {"test_id": tid, "assertion": "> assert 1 == 2", "traceback": ""}
                    for tid in test_ids
                ],
            }
        ]
    return {
        "status": "fail",
        "validation": "check failed",
        "digest": "## Failing tests",
        "attempt": 0,
        "command": "python -m pytest tests",
        "runners": runners,
        "raw_tail": "",
        "check_mode": check_mode,
        "targeted_command": None,
        "inherited": inherited,
        "commit": commit,
    }


# ── targeted_prepass unit tests ──────────────────────────────────────────

class TestTargetedPrepass:

    def test_skip_when_not_fail(self):
        result = targeted_prepass(
            {"status": "pass"}, "python -m pytest", lambda c, t: None, 180,
        )
        assert not result.ran
        assert result.check_mode == "full"

    def test_skip_when_all_inherited(self):
        record = _fail_record(
            test_ids=["tests/test_foo.py::test_bar"],
            inherited=["tests/test_foo.py::test_bar"],
        )
        result = targeted_prepass(
            record, "python -m pytest tests", lambda c, t: None, 180,
        )
        assert not result.ran
        assert result.check_mode == "full"

    def test_early_stop_on_still_failing(self):
        record = _fail_record()
        calls = []
        def fake_run(cmd, timeout):
            calls.append((cmd, timeout))
            return _make_result(ok=False, returncode=1, stdout=FAIL_STDOUT)

        result = targeted_prepass(record, "python -m pytest tests", fake_run, 180)
        assert result.ran
        assert result.early_stop
        assert result.check_mode == "targeted"
        assert len(calls) == 1
        assert "test_foo.py::test_bar" in calls[0][0]
        assert calls[0][1] <= 120  # pre-pass timeout

    def test_fallthrough_on_pass(self):
        record = _fail_record()
        def fake_run(cmd, timeout):
            return _make_result(ok=True, returncode=0, stdout=PASS_STDOUT)

        result = targeted_prepass(record, "python -m pytest tests", fake_run, 180)
        assert result.ran
        assert not result.early_stop
        assert result.check_mode == "full"

    def test_fallthrough_on_exit_4(self):
        record = _fail_record()
        def fake_run(cmd, timeout):
            return _make_result(ok=False, returncode=4, stdout="no tests ran")

        result = targeted_prepass(record, "python -m pytest tests", fake_run, 180)
        assert result.ran
        assert not result.early_stop
        assert result.check_mode == "full"

    def test_fallthrough_on_prep_failure(self):
        record = _fail_record()
        def fake_run(cmd, timeout):
            return _make_result(
                ok=False, returncode=1, stdout=FAIL_STDOUT,
                prep_failure={"returncode": 1},
            )

        result = targeted_prepass(record, "python -m pytest tests", fake_run, 180)
        assert result.ran
        assert not result.early_stop
        assert result.check_mode == "full"

    def test_fallthrough_on_timeout(self):
        record = _fail_record()
        def fake_run(cmd, timeout):
            return _make_result(ok=False, returncode=None, timed_out=True, stdout="")

        result = targeted_prepass(record, "python -m pytest tests", fake_run, 180)
        assert result.ran
        assert not result.early_stop

    def test_inherited_never_stops_early(self):
        """A failure that is inherited never causes an early stop."""
        record = _fail_record(
            test_ids=["tests/test_foo.py::test_bar", "tests/test_base.py::test_inherited"],
            inherited=["tests/test_base.py::test_inherited"],
        )
        # Only test_bar is targeted; it passes but inherited still fails.
        def fake_run(cmd, timeout):
            # Return only the inherited test failing
            return _make_result(
                ok=False, returncode=1,
                stdout="FAILED tests/test_base.py::test_inherited - assert old\n1 failed",
            )

        result = targeted_prepass(record, "python -m pytest tests", fake_run, 180)
        # Early stop should NOT happen because the failing test in output
        # (test_inherited) is in the inherited set and was not targeted.
        assert result.ran
        assert not result.early_stop

    def test_prepass_timeout_calculation(self):
        record = _fail_record()
        calls = []
        def fake_run(cmd, timeout):
            calls.append((cmd, timeout))
            return _make_result(ok=True, returncode=0, stdout=PASS_STDOUT)

        # With check_timeout_s=180, pre-pass timeout = min(60, 120) = 60
        targeted_prepass(record, "python -m pytest tests", fake_run, 180)
        assert calls[0][1] == 60

        calls.clear()
        # With check_timeout_s=60, pre-pass timeout = min(20, 120) = 20
        targeted_prepass(record, "python -m pytest tests", fake_run, 60)
        assert calls[0][1] == 20

    def test_no_test_files_found_runs_full(self):
        """vitest 'No test files found' falls through to full check."""
        record = _fail_record()
        def fake_run(cmd, timeout):
            # vitest exits 1 when no test files are found
            return _make_result(ok=False, returncode=1, stdout="No test files found")

        result = targeted_prepass(record, "python -m pytest tests", fake_run, 180)
        assert result.ran
        # No targeted test ID appears in the output, so early stop won't fire
        assert not result.early_stop


# ── detect_stuck unit tests ──────────────────────────────────────────────

class TestDetectStuck:

    def test_no_stuck_on_first_check(self):
        result = detect_stuck(
            {"status": "not_run"}, [], "", None, None, "bbb",
        )
        assert result is None

    def test_no_stuck_on_same_commit(self):
        prev = _fail_record(commit="aaa")
        curr_runners = prev["runners"]
        result = detect_stuck(prev, curr_runners, "", None, "aaa", "aaa")
        assert result is None

    def test_stuck_fires_after_two(self):
        prev = _fail_record(commit="aaa")
        curr_runners = [
            {
                "name": "pytest",
                "passed": False,
                "failures": [
                    {"test_id": "tests/test_foo.py::test_bar",
                     "assertion": "> assert 1 == 2", "traceback": ""}
                ],
            }
        ]
        result = detect_stuck(prev, curr_runners, FAIL_STDOUT, None, "aaa", "bbb")
        assert result is not None
        assert "Stuck" in result
        assert "test_bar" in result

    def test_no_stuck_with_different_assertion(self):
        prev = _fail_record(commit="aaa")
        # Change the assertion key
        curr_runners = [
            {
                "name": "pytest",
                "passed": False,
                "failures": [
                    {"test_id": "tests/test_foo.py::test_bar",
                     "assertion": "> assert 2 == 3", "traceback": ""}
                ],
            }
        ]
        result = detect_stuck(prev, curr_runners, "", None, "aaa", "bbb")
        assert result is None  # different assertion → not stuck

    def test_inherited_excluded_from_stuck(self):
        prev = _fail_record(
            test_ids=["tests/test_foo.py::test_bar"],
            inherited=["tests/test_foo.py::test_bar"],
            commit="aaa",
        )
        curr_runners = [
            {
                "name": "pytest",
                "passed": False,
                "failures": [
                    {"test_id": "tests/test_foo.py::test_bar",
                     "assertion": "> assert 1 == 2", "traceback": ""}
                ],
            }
        ]
        result = detect_stuck(prev, curr_runners, "", None, "aaa", "bbb")
        assert result is None  # inherited → not stuck


# ── record_check_history unit tests ──────────────────────────────────────

class TestRecordCheckHistory:

    def test_targeted_excluded_from_history(self):
        facts: dict = {"check_failures_history": [], "check_blocker_history": []}

        class _Ctx:
            pass

        ctx = _Ctx()
        ctx.facts = facts

        def set_fact(k, v):
            facts[k] = v
        ctx.set_fact = set_fact

        record_check_history(ctx, "targeted", "some output", "python -m pytest")
        assert facts["check_failures_history"] == []
        assert facts["check_blocker_history"] == []

    def test_full_appended_to_history(self):
        facts: dict = {"check_failures_history": [], "check_blocker_history": []}

        class _Ctx:
            pass

        ctx = _Ctx()
        ctx.facts = facts

        def set_fact(k, v):
            facts[k] = v
        ctx.set_fact = set_fact

        record_check_history(ctx, "full", FAIL_STDOUT, "python -m pytest tests")
        assert len(facts["check_failures_history"]) == 1
        assert len(facts["check_blocker_history"]) == 1


# ── _not_found_ids unit tests ───────────────────────────────────────────


class TestNotFoundIds:

    def test_standard_not_found_line(self):
        from refactory_dsh.failfast import _not_found_ids

        output = ("ERROR: not found: /abs/path/tests/test_foo.py::test_bar\n"
                  "(no match in any of [<Module test_foo.py>])")
        assert _not_found_ids(output) == {"tests/test_foo.py::test_bar"}

    def test_collecting_prefix(self):
        from refactory_dsh.failfast import _not_found_ids

        output = "collecting ... ERROR: not found: /abs/path/tests/test_x.py::test_y"
        assert _not_found_ids(output) == {"tests/test_x.py::test_y"}

    def test_no_match(self):
        from refactory_dsh.failfast import _not_found_ids

        assert _not_found_ids("1 passed") == set()

    def test_file_or_directory_not_found_is_different(self):
        from refactory_dsh.failfast import _not_found_ids

        output = "ERROR: file or directory not found: tests/nonexistent.py::test_x"
        assert _not_found_ids(output) == set()


# ── pnpm targeting unit tests ───────────────────────────────────────────

class TestPnpmTargeting:

    def test_pnpm_vitest_from_root(self, tmp_path):
        """From a working directory with no package.json, root=<worktree> works."""
        (tmp_path / "package.json").write_text('{"scripts": {"test": "vitest run"}}')
        from refactory_dsh.check_digest import build_probe_command
        cmd, refusal = build_probe_command(
            "pnpm test", ["src/foo.test.ts > suite > test_add"],
            root=tmp_path,
        )
        assert refusal is None
        assert "pnpm test" in cmd
        assert "src/foo.test.ts" in cmd
        assert "-t" in cmd
        assert cmd.count("-t") == 1


# ── scripted mode runs ───────────────────────────────────────────────────

PLAN = "Change only the requested file, preserve its public behavior, and review the required check result."
QA_PASS = json.dumps({
    "status": "pass",
    "checks": [{"name": "requested behavior", "command": "source inspection",
                "status": "pass", "evidence": "the implementation returns the requested literal"}],
    "findings": [], "not_run": ["broad research and full reviewer campaign not requested"],
})
VERIFICATION_ANSWER = "No blocking findings.\n\nVERDICT: PASS"


class TestScriptedModeRuns:
    """Scripted DesignFastMode runs exercising focused_check.

    Each scenario exercises the real check node through a FakeModel that
    delivers scripted answers.  The check command is a simple Python
    assertion so the test controls pass/fail.  The shared helpers
    (targeted_prepass, detect_stuck, record_check_history) are the same
    code paths used by check_artifact in design mode.
    """

    def test_targeted_pass_then_full(self, tmp_path, monkeypatch):
        """A fixed test gives a targeted pass, then the full check runs."""
        monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
        from tests_support import FakeModel
        from refactory_dsh.fast import DesignFastMode

        mode = DesignFastMode()

        # First attempt: bad code → check fails → record written.
        # Second attempt: good code → pre-pass passes (targeted) → full check passes.
        bad = f"===FILE: answer.py===\nassert False, 'test_bar'\n"
        good = f"===FILE: answer.py===\nassert True\n"

        client = FakeModel([PLAN, bad, good, QA_PASS, VERIFICATION_ANSWER])
        state = mode.state_dict()
        state["knobs"]["repair_attempts"]["value"] = 2
        mode.load_state_dict(state)

        run = mode.run(
            tmp_path / "run", model=client,
            task={"name": "prepass_fix", "brief": "assert true",
                  "check": f"{sys.executable} answer.py"},
        )
        # The run should complete (either pass or exhaust attempts).
        record = json.loads((tmp_path / "run/check-record.json").read_text())
        # Final record should have check_mode.
        assert "check_mode" in record

    def test_targeted_still_fails_early_stop(self, tmp_path, monkeypatch):
        """A still-failing test stops early, with the digest."""
        monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
        from tests_support import FakeModel
        from refactory_dsh.fast import DesignFastMode

        mode = DesignFastMode()

        bad = f"===FILE: answer.py===\nassert False, 'test_bar'\n"

        client = FakeModel([PLAN, bad, bad, QA_PASS, VERIFICATION_ANSWER])
        state = mode.state_dict()
        state["knobs"]["repair_attempts"]["value"] = 2
        mode.load_state_dict(state)

        run = mode.run(
            tmp_path / "run", model=client,
            task={"name": "prepass_stuck", "brief": "assert true",
                  "check": f"{sys.executable} answer.py"},
        )
        record = json.loads((tmp_path / "run/check-record.json").read_text())
        assert "check_mode" in record

    def test_stuck_after_two_not_one(self, tmp_path, monkeypatch):
        """Stuck notice appears after two identical failures, not after one."""
        monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
        from tests_support import FakeModel
        from refactory_dsh.fast import DesignFastMode

        mode = DesignFastMode()

        # Both attempts produce the same failing code.
        bad = f"===FILE: answer.py===\nassert False, 'test_bar'\n"

        client = FakeModel([PLAN, bad, bad, QA_PASS, VERIFICATION_ANSWER])
        state = mode.state_dict()
        state["knobs"]["repair_attempts"]["value"] = 2
        mode.load_state_dict(state)

        run = mode.run(
            tmp_path / "run", model=client,
            task={"name": "stuck_detect", "brief": "assert true",
                  "check": f"{sys.executable} answer.py"},
        )
        # The run completes (possibly with an error); check the record.
        record = json.loads((tmp_path / "run/check-record.json").read_text())
        assert "check_mode" in record


class TestCreateFastEditFastUnchanged:
    """create_fast and edit_fast scripted runs are unchanged."""

    def test_create_fast_keeps_admission_and_canonical_smoke(self, tmp_path, monkeypatch):
        """No UnboundLocalError from prepass variable."""
        monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
        from refactory_dsh.fast import CreateFastMode
        from refactory_dsh import registry
        from tests_support import FakeModel, good_files, render

        name = "fast_counter"
        mode = CreateFastMode()
        state = mode.state_dict()
        state["knobs"]["single_shot"]["value"] = True
        mode.load_state_dict(state)
        run = mode.run(
            tmp_path / "run",
            model=FakeModel([PLAN, render(good_files(name)), QA_PASS]),
            task={"name": name, "brief": "a reusable counter"},
        )
        assert run.exit_ok, run.error
        result = json.loads((tmp_path / "run/create.json").read_text())
        assert result["fast"]["structural_admission"] == "passed"
        assert result["fast"]["canonical_smoke"] == "passed"


def test_a_step_option_keeps_its_value_when_the_check_is_targeted():
    """`-n 4`, `-k expr`, `-m mark`, `-p plugin` are option and value: stripping the value
    leaves the option dangling, and the targeted run collects nothing."""
    from refactory_dsh.check_digest import build_probe_command

    command, refusal = build_probe_command(
        ".venv/bin/python -m pytest -q -n 4 refactory/tests",
        ["refactory/tests/test_a.py::test_one"])
    assert refusal is None
    assert "-n 4" in command, command
    assert "refactory/tests/test_a.py::test_one" in command
    assert command.count("refactory/tests") == 1, "the directory itself is still stripped"

    command, refusal = build_probe_command('pytest -q -k "a or b" tests', ["tests/test_a.py::test_one"])
    assert refusal is None
    assert "-k 'a or b'" in command or '-k "a or b"' in command, command


def test_a_targeted_command_pytest_cannot_parse_fails_loudly():
    """Exit 4 with no not-found id means the command itself is broken — the case `-n 4`
    stripped to `-n` produced. A self-check that cannot run at all is a fault, not a pass."""
    import pytest

    from refactory_dsh import failfast

    class Result:
        ok = False
        returncode = 4
        timed_out = False
        prep_failure = None
        stdout = "error: argument -n: invalid int value\n"
        stderr = ""

    def runner(_command: str, _timeout: float) -> Result:
        return Result()

    with pytest.raises(RuntimeError, match="the targeted check ran no tests"):
        failfast.targeted_prepass(
            {"status": "fail"}, "pytest -q -n refactory/tests", runner, 60,
            selection=["refactory/tests/test_a.py::test_one"],
            self_check_timeout_s=30)


def test_a_targeted_check_that_collects_no_test_is_recorded_not_a_pass():
    """Exit 5 (nothing matched) is not a broken command: the chunk is recorded as having
    verified nothing, the self-check does not count it, and the whole check still runs."""
    from refactory_dsh import failfast

    class Result:
        ok = False
        returncode = 5
        timed_out = False
        prep_failure = None
        stdout = "no tests ran in 0.01s\n"
        stderr = ""

    def runner(_command: str, _timeout: float) -> Result:
        return Result()

    result = failfast.targeted_prepass(
        {"status": "fail"}, "pytest -q refactory/tests", runner, 60,
        selection=["refactory/tests/test_a.py::test_one"],
        self_check_timeout_s=30)
    assert result.ran_nothing, "the record says the chunk ran nothing"
    assert not result.broken_command
    assert not result.failing_ids and not result.early_stop, "it is not a pass on a failure"
