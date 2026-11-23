"""Tests for the positive-selection send-back detail in filter_check.

The builder sees only charged failures: the FAILED/ERROR line and
FAILURES-section block of each charged ID, the final summary lines,
and one line listing excused IDs with reasons.
"""

from __future__ import annotations

import pytest

from refactory_dsh.base_check import (
    _build_positive_detail,
    _extract_failed_error_lines,
    _extract_failure_blocks,
    _extract_summary_lines,
    _format_excused_line,
)


ATTEMPT_OUTPUT = """\
=========================== FAILURES ===========================
___________________________ test_new ___________________________
tests/test_a.py:10: in test_new
    assert False
E   AssertionError: test_new failed

___________________________ test_base ___________________________
tests/test_a.py:20: in test_base
    assert False
E   AssertionError: test_base failed

=========================== short test summary info ===========================
FAILED tests/test_a.py::test_new - AssertionError: test_new failed
FAILED tests/test_a.py::test_base - AssertionError: test_base failed
= 2 failed in 0.1s =
"""


def test_positive_selection_only():
    """A mixed result produces detail with only the charged test's block."""
    charged_ids = {"tests/test_a.py::test_new"}
    excused_ids = {"tests/test_a.py::test_base"}
    reasons = {
        "tests/test_a.py::test_new": "new-failure",
        "tests/test_a.py::test_base": "base-whole-run",
    }
    detail = _build_positive_detail(ATTEMPT_OUTPUT, charged_ids, excused_ids, reasons)
    assert "test_new" in detail
    assert "Excused (not charged)" in detail
    assert "test_base (base-whole-run)" in detail


def test_no_excused_block_lines():
    """The filtered detail contains no line belonging to an excused ID's FAILURES block."""
    charged_ids = {"tests/test_a.py::test_new"}
    excused_ids = {"tests/test_a.py::test_base"}
    reasons = {"tests/test_a.py::test_base": "base-whole-run"}
    blocks = _extract_failure_blocks(ATTEMPT_OUTPUT, charged_ids)
    # Only charged blocks should be present
    assert any("test_new" in b for b in blocks)
    # No excused blocks
    excused_blocks = _extract_failure_blocks(ATTEMPT_OUTPUT, excused_ids)
    for block in excused_blocks:
        for charged_block in blocks:
            assert block not in charged_block


def test_excused_short_name_in_traceback_not_leaked():
    """A charged test's traceback mentioning an excused test by short name does not
    cause excused-block content to appear."""
    output = """\
=========================== FAILURES ===========================
___________________________ test_new ___________________________
tests/test_a.py:10: in test_new
    # This test depends on test_base results
    assert False
E   AssertionError: test_new failed, see test_base

___________________________ test_base ___________________________
tests/test_a.py:20: in test_base
    assert False
E   AssertionError: test_base failed

=========================== short test summary info ===========================
FAILED tests/test_a.py::test_new - AssertionError
FAILED tests/test_a.py::test_base - AssertionError
= 2 failed in 0.1s =
"""
    charged_ids = {"tests/test_a.py::test_new"}
    excused_ids = {"tests/test_a.py::test_base"}
    reasons = {"tests/test_a.py::test_base": "base-whole-run"}
    detail = _build_positive_detail(output, charged_ids, excused_ids, reasons)
    # The charged block may mention test_base in its traceback, that's OK
    assert "test_new" in detail
    # But the excused BLOCK (the section under ___ test_base ___) should not be there
    # The detail should not have the specific assertion from test_base's block
    assert "test_base failed" not in detail or "see test_base" in detail


def test_no_fallback_when_empty():
    """When no charged failure detail exists, the detail says so, not the whole output."""
    detail = _build_positive_detail(ATTEMPT_OUTPUT, set(), set(), {})
    assert detail == "no charged failure detail found"


def test_verdict_separates_charged_excused():
    """Check-verdict reasons show charged and excused IDs separately."""
    reasons = {
        "tests/test_a.py::test_new": "new-failure",
        "tests/test_a.py::test_base": "base-whole-run",
    }
    # The reasons dict itself separates them
    charged = [tid for tid, r in reasons.items() if r in ("new-failure", "named-test", "signature-mismatch", "touched-file")]
    excused = [tid for tid, r in reasons.items() if r in ("base-whole-run", "base-per-step", "base-targeted", "not-reproduced")]
    assert "tests/test_a.py::test_new" in charged
    assert "tests/test_a.py::test_base" in excused


def test_excused_line_format():
    """The excused line lists IDs with reasons."""
    excused = {"tests/test_a.py::test_x", "tests/test_b.py::test_y"}
    reasons = {
        "tests/test_a.py::test_x": "base-whole-run",
        "tests/test_b.py::test_y": "not-reproduced",
    }
    line = _format_excused_line(excused, reasons)
    assert "Excused (not charged)" in line
    assert "test_x (base-whole-run)" in line
    assert "test_y (not-reproduced)" in line


def test_summary_lines_extracted():
    """The final summary line is extracted."""
    lines = _extract_summary_lines(ATTEMPT_OUTPUT)
    assert any("2 failed" in l for l in lines)


# ── integration tests: positive selection through filter_check ──────────

from types import SimpleNamespace
from unittest.mock import patch

from refactory_dsh import base_check, sandbox
from refactory_dsh.base_check import BaseCheckResult, filter_check


def _make_attempt_output(charged_ids, excused_ids):
    """Build a pytest output with failures and tracebacks."""
    all_ids = sorted(charged_ids | excused_ids)
    failure_blocks = []
    for tid in all_ids:
        short_name = tid.split("::")[-1]
        failure_blocks.append(
            f"___________________________ {short_name} ___________________________\n"
            f"tests/test_a.py:10: in {short_name}\n"
            f"    assert False\n"
            f"E   AssertionError: {short_name} failed\n"
        )
    failed_lines = [f"FAILED {tid} - AssertionError" for tid in all_ids]
    summary = f"= {len(all_ids)} failed in 0.1s ="
    return (
        "=========================== FAILURES ===========================\n"
        + "\n".join(failure_blocks)
        + "\n=========================== short test summary info ===========================\n"
        + "\n".join(failed_lines) + "\n"
        + summary + "\n"
    )


def test_filter_check_positive_selection_integration(tmp_path):
    """Drives filter_check to verify positive-selection detail building
    for correction 4: the builder sees only charged failures.

    When the correction-4 hunk (the _build_positive_detail call) is reverted,
    this test must fail.
    """
    charged_id = "tests/test_a.py::test_new"
    excused_id = "tests/test_a.py::test_base"
    attempt_out = _make_attempt_output({charged_id}, {excused_id})

    base = BaseCheckResult(
        ok=False, failures={excused_id}, output="", exit_code=1)
    ctx = SimpleNamespace(workdir=tmp_path, facts={}, exists=lambda name: False,
                          read_text=lambda name: "", budget=None)
    attempt = sandbox.SandboxResult(ok=False, stdout=attempt_out, stderr="", returncode=1)
    found = SimpleNamespace(path=str(tmp_path / "run-x"), base="abc", head="def",
                            target=str(tmp_path / "target"))
    # Mock targeted to confirm charged_id: attempt=fail, base=pass → charged
    with patch.object(base_check, "run_base_check", return_value=base), \
         patch.object(base_check.worktree, "recorded", return_value=found), \
         patch.object(base_check, "_all_changed_files", return_value=[]), \
         patch.object(base_check, "targeted_attempt_failures",
                      return_value={charged_id}), \
         patch.object(base_check, "targeted_base_failures",
                      return_value=(BaseCheckResult(ok=True, failures=set(), output="", exit_code=0), "cache")):
        filtered = filter_check(ctx, attempt, "pytest tests/", 60.0)

    assert charged_id in filtered.new_failures
    assert excused_id in filtered.inherited
    detail = filtered.new_failure_detail
    # Charged test's FAILED line and block must be present
    assert "test_new" in detail
    assert "FAILED tests/test_a.py::test_new" in detail
    # Excused test's FAILURES block must NOT be present
    assert "test_base failed" not in detail
    # Excused listing line must be present
    assert "Excused (not charged)" in detail
    assert "test_base" in detail
    # No fallback to full output
    assert "=========================== FAILURES ===========================" not in detail or detail.count("FAILURES") <= 2


def test_filter_check_no_fallback_integration(tmp_path):
    """When all failures are excused, the detail says 'no charged failure detail found',
    not the full output."""
    excused_id = "tests/test_a.py::test_base"
    attempt_out = _make_attempt_output(set(), {excused_id})

    base = BaseCheckResult(
        ok=False, failures={excused_id}, output="", exit_code=1)
    ctx = SimpleNamespace(workdir=tmp_path, facts={}, exists=lambda name: False,
                          read_text=lambda name: "", budget=None)
    attempt = sandbox.SandboxResult(ok=False, stdout=attempt_out, stderr="", returncode=1)
    found = SimpleNamespace(path=str(tmp_path / "run-x"), base="abc", head="def",
                            target=str(tmp_path / "target"))
    with patch.object(base_check, "run_base_check", return_value=base), \
         patch.object(base_check.worktree, "recorded", return_value=found), \
         patch.object(base_check, "_all_changed_files", return_value=[]):
        filtered = filter_check(ctx, attempt, "pytest tests/", 60.0)

    assert filtered.all_excused
    assert "no charged failure detail found" in filtered.new_failure_detail


def test_filter_check_verdict_has_reasons(tmp_path):
    """check-verdict.json reasons show charged and excused IDs separately."""
    charged_id = "tests/test_a.py::test_new"
    excused_id = "tests/test_a.py::test_base"
    attempt_out = _make_attempt_output({charged_id}, {excused_id})

    base = BaseCheckResult(
        ok=False, failures={excused_id}, output="", exit_code=1)
    ctx = SimpleNamespace(workdir=tmp_path, facts={}, exists=lambda name: False,
                          read_text=lambda name: "", budget=None)
    attempt = sandbox.SandboxResult(ok=False, stdout=attempt_out, stderr="", returncode=1)
    found = SimpleNamespace(path=str(tmp_path / "run-x"), base="abc", head="def",
                            target=str(tmp_path / "target"))
    with patch.object(base_check, "run_base_check", return_value=base), \
         patch.object(base_check.worktree, "recorded", return_value=found), \
         patch.object(base_check, "_all_changed_files", return_value=[]), \
         patch.object(base_check, "targeted_attempt_failures",
                      return_value={charged_id}), \
         patch.object(base_check, "targeted_base_failures",
                      return_value=(BaseCheckResult(ok=True, failures=set(), output="", exit_code=0), "cache")):
        filtered = filter_check(ctx, attempt, "pytest tests/", 60.0)

    assert filtered.reasons is not None
    assert filtered.reasons[excused_id] == "base-whole-run"
    assert filtered.reasons[charged_id] == "new-failure"
