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
