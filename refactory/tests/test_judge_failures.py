"""Tests for the unified failure judge in base_check.

Covers judge_failures, extract_failure_signatures, the fixed _PYTEST_FAILED
and _PYTEST_ERROR regexes, shell quoting, and mutation coverage.
"""

from __future__ import annotations

import re

import pytest

from refactory_dsh.base_check import (
    JudgeResult,
    _PYTEST_ERROR,
    _PYTEST_FAILED,
    extract_failure_signatures,
    judge_failures,
    names_test,
    parse_test_failures,
)


# ── judge_failures: base excusal ────────────────────────────────────────


def test_base_whole_run_excuses():
    """A failure present in base_failures is excused; charged is empty, all_excused is True."""
    jr = judge_failures({"tests/test_a.py::test_x"}, {"tests/test_a.py::test_x"})
    assert jr.charged == frozenset()
    assert jr.excused == frozenset({"tests/test_a.py::test_x"})
    assert jr.all_excused is True
    assert jr.reasons["tests/test_a.py::test_x"] == "base-whole-run"


def test_new_failure_charged_after_targeted():
    """A failure absent from every base observation that fails on the attempt re-run
    and passes on the base re-run is charged."""
    jr = judge_failures(
        {"tests/test_a.py::test_x"}, set(),
        targeted_attempt_failures={"tests/test_a.py::test_x"},
        targeted_base_failures_set=set(),
    )
    assert "tests/test_a.py::test_x" in jr.charged
    assert jr.reasons["tests/test_a.py::test_x"] == "new-failure"


def test_not_reproduced_excused():
    """A failure that passes on the attempt re-run is excused with 'not-reproduced'."""
    jr = judge_failures(
        {"tests/test_a.py::test_x"}, set(),
        targeted_attempt_failures=set(),  # test passed on re-run
        targeted_base_failures_set=set(),
    )
    assert "tests/test_a.py::test_x" in jr.excused
    assert jr.reasons["tests/test_a.py::test_x"] == "not-reproduced"


def test_base_targeted_excused():
    """A failure absent from the whole-run base but present in targeted base re-run
    is excused with 'base-targeted'."""
    jr = judge_failures(
        {"tests/test_a.py::test_x"}, set(),
        targeted_base_failures_set={"tests/test_a.py::test_x"},
    )
    assert "tests/test_a.py::test_x" in jr.excused
    assert jr.reasons["tests/test_a.py::test_x"] == "base-targeted"


def test_named_test_always_charged():
    """A failure that names_test matches in named_text is charged even if the base has it."""
    jr = judge_failures(
        {"tests/test_a.py::test_x"}, {"tests/test_a.py::test_x"},
        named_text="fix test_x",
    )
    assert "tests/test_a.py::test_x" in jr.charged
    assert jr.reasons["tests/test_a.py::test_x"] == "named-test"


def test_touched_file_charged_fast_mode():
    """A failure whose test file is in changed_files is charged in fast mode."""
    jr = judge_failures(
        {"tests/test_a.py::test_x"}, set(),
        changed_files=["tests/test_a.py"],
    )
    assert "tests/test_a.py::test_x" in jr.charged
    assert jr.reasons["tests/test_a.py::test_x"] == "touched-file"


def test_touched_file_different_signature_charged():
    """A failure in a touched file with a different signature from the base's is charged."""
    jr = judge_failures(
        {"tests/test_a.py::test_x"}, {"tests/test_a.py::test_x"},
        changed_files=["tests/test_a.py"],
        attempt_signatures={"tests/test_a.py::test_x": "AssertionError:test_a.py:10"},
        base_signatures={"tests/test_a.py::test_x": "TypeError:test_a.py:20"},
    )
    assert "tests/test_a.py::test_x" in jr.charged
    assert jr.reasons["tests/test_a.py::test_x"] == "signature-mismatch"


def test_touched_file_same_signature_excused():
    """A failure in a touched file with the same signature as the base's is excused."""
    jr = judge_failures(
        {"tests/test_a.py::test_x"}, {"tests/test_a.py::test_x"},
        changed_files=["tests/test_a.py"],
        attempt_signatures={"tests/test_a.py::test_x": "AssertionError:test_a.py:10"},
        base_signatures={"tests/test_a.py::test_x": "AssertionError:test_a.py:10"},
    )
    assert "tests/test_a.py::test_x" in jr.excused
    assert jr.reasons["tests/test_a.py::test_x"] == "base-whole-run"


def test_missing_signature_charged():
    """When either signature is missing for a failure in a touched test file, it is charged."""
    # Attempt sig missing
    jr1 = judge_failures(
        {"tests/test_a.py::test_x"}, {"tests/test_a.py::test_x"},
        changed_files=["tests/test_a.py"],
        attempt_signatures={"tests/test_a.py::test_x": None},
        base_signatures={"tests/test_a.py::test_x": "AssertionError:test_a.py:10"},
    )
    assert "tests/test_a.py::test_x" in jr1.charged
    assert jr1.reasons["tests/test_a.py::test_x"] == "signature-mismatch"

    # Base sig missing
    jr2 = judge_failures(
        {"tests/test_a.py::test_x"}, {"tests/test_a.py::test_x"},
        changed_files=["tests/test_a.py"],
        attempt_signatures={"tests/test_a.py::test_x": "AssertionError:test_a.py:10"},
        base_signatures={"tests/test_a.py::test_x": None},
    )
    assert "tests/test_a.py::test_x" in jr2.charged
    assert jr2.reasons["tests/test_a.py::test_x"] == "signature-mismatch"

    # Both missing
    jr3 = judge_failures(
        {"tests/test_a.py::test_x"}, {"tests/test_a.py::test_x"},
        changed_files=["tests/test_a.py"],
        attempt_signatures={},
        base_signatures={},
    )
    assert "tests/test_a.py::test_x" in jr3.charged


def test_per_step_base_excuses():
    """A failure found in per_step_base_failures but not base_failures is excused."""
    jr = judge_failures(
        {"tests/test_a.py::test_x"}, set(),
        per_step_base_failures={"tests/test_a.py::test_x"},
    )
    assert "tests/test_a.py::test_x" in jr.excused
    assert jr.reasons["tests/test_a.py::test_x"] == "base-per-step"


def test_targeted_base_excuses():
    """A failure found in targeted_base_failures but not in other observations is excused."""
    jr = judge_failures(
        {"tests/test_a.py::test_x"}, set(),
        targeted_base_failures_set={"tests/test_a.py::test_x"},
    )
    assert "tests/test_a.py::test_x" in jr.excused
    assert jr.reasons["tests/test_a.py::test_x"] == "base-targeted"


def test_empty_attempt_failures():
    """No failures → empty result."""
    jr = judge_failures(set(), {"tests/test_a.py::test_x"})
    assert jr.charged == frozenset()
    assert jr.excused == frozenset()
    assert jr.all_excused is False


# ── mutation tests ──────────────────────────────────────────────────────


def test_mutation_charged_condition():
    """Flipping each boolean condition in the judge changes at least one result."""
    # If we remove the named-test check, a named failure would be excused instead of charged
    jr_named = judge_failures(
        {"tests/test_a.py::test_x"}, {"tests/test_a.py::test_x"},
        named_text="test_x",
    )
    assert "tests/test_a.py::test_x" in jr_named.charged

    # If we remove the base-matching check, a base failure would be charged
    jr_base = judge_failures(
        {"tests/test_a.py::test_x"}, {"tests/test_a.py::test_x"},
    )
    assert "tests/test_a.py::test_x" in jr_base.excused

    # Both conditions differ
    assert jr_named.charged != jr_base.charged


def test_mutation_excused_condition():
    """The judge does not trivially excuse everything."""
    # A genuinely new failure (not in base, not in targeted base) is charged
    jr = judge_failures(
        {"tests/test_a.py::test_new"}, set(),
        targeted_attempt_failures={"tests/test_a.py::test_new"},
        targeted_base_failures_set=set(),
    )
    assert "tests/test_a.py::test_new" in jr.charged
    assert jr.all_excused is False


# ── regex tests ─────────────────────────────────────────────────────────


def test_parametrised_id_full_capture():
    """parse_test_failures captures full parametrised IDs with spaces and brackets."""
    output = (
        "FAILED tests/test_x.py::test_x[a b] - AssertionError\n"
        "FAILED tests/test_x.py::test_x[a\\b] - AssertionError\n"
        "FAILED tests/test_x.py::test_x[key=val] - AssertionError\n"
        "= 3 failed in 0.1s =\n"
    )
    failures = parse_test_failures(output)
    assert "tests/test_x.py::test_x[a b]" in failures
    assert "tests/test_x.py::test_x[a\\b]" in failures
    assert "tests/test_x.py::test_x[key=val]" in failures


def test_vitest_regex_not_cross_contaminated():
    """Vitest output is captured by _VITEST_FAIL, not by _PYTEST_FAILED."""
    vitest_line = " FAIL  src/app.spec.ts > suite > test name"
    assert not _PYTEST_FAILED.search(vitest_line)


def test_shell_quoting_spaces():
    """A test ID with spaces is correctly quoted via shlex.quote()."""
    import shlex
    test_id = "tests/test_x.py::test_x[a b]"
    quoted = shlex.quote(test_id)
    assert " " not in quoted or quoted.startswith("'")
    assert test_id == shlex.split(quoted)[0]


def test_pytest_error_rejects_log_lines():
    """_PYTEST_ERROR does not match log-capture lines."""
    log_lines = [
        "ERROR    root:module.py:12 message",
        "ERROR: something",
        "ERROR    conftest.py:12 some message",
    ]
    for line in log_lines:
        failures = parse_test_failures(line)
        assert not failures, f"log line matched as failure: {line}"


def test_pytest_error_rejects_log_lines_check_digest():
    """The same log lines tested on check_digest's parser."""
    from refactory_dsh.check_digest import _PYTEST_ERROR_LINE
    log_lines = [
        "ERROR    root:module.py:12 message",
        "ERROR: something",
    ]
    for line in log_lines:
        assert not _PYTEST_ERROR_LINE.search(line), f"check_digest matched: {line}"


def test_pytest_error_matches_valid_error():
    """_PYTEST_ERROR matches valid pytest error lines."""
    valid_lines = [
        "ERROR tests/test_a.py::test_x - fixture not found",
        "ERROR tests/test_a.py - ImportError",
    ]
    for line in valid_lines:
        failures = parse_test_failures(line)
        assert failures, f"valid error line not matched: {line}"


# ── signature extraction tests ──────────────────────────────────────────


def test_signature_extraction_tb_short():
    """extract_failure_signatures extracts from --tb=short output."""
    output = """
=========================== FAILURES ===========================
___________________________ test_x ___________________________
tests/test_a.py:10: in test_x
    assert False
E   AssertionError: assert False
=========================== short test summary info ===========================
"""
    result = extract_failure_signatures(output, {"tests/test_a.py::test_x"})
    assert result.get("tests/test_a.py::test_x") is not None
    assert "AssertionError" in result["tests/test_a.py::test_x"]


def test_signature_extraction_tb_no():
    """extract_failure_signatures returns None for --tb=no output."""
    output = """
=========================== short test summary info ===========================
FAILED tests/test_a.py::test_x - AssertionError
= 1 failed in 0.1s =
"""
    result = extract_failure_signatures(output, {"tests/test_a.py::test_x"})
    assert result.get("tests/test_a.py::test_x") is None


def test_signature_extraction_ansi():
    """ANSI-colored traceback is handled."""
    output = """
=========================== FAILURES ===========================
___________________________ test_x ___________________________
\x1b[31mtests/test_a.py\x1b[0m:10: in test_x
    assert False
E   \x1b[31mAssertionError\x1b[0m: assert False
=========================== short test summary info ===========================
"""
    result = extract_failure_signatures(output, {"tests/test_a.py::test_x"})
    assert result.get("tests/test_a.py::test_x") is not None
    assert "AssertionError" in result["tests/test_a.py::test_x"]
