"""Tests for the empty-emission classifier and its integration with design and fast mode.

The classifier lives in ``failfast.classify_empty_emission``; the adapters are
``design_ops._refused_verdict`` (design mode) and ``fast_ops.gate_emit`` (fast mode).
"""

from __future__ import annotations

import pytest

from refactory_dsh.failfast import classify_empty_emission, EmissionDecision
from refactory_dsh import design_ops, repo_change
from refactory_dsh import nn
from refactory_dsh.nn import Verdict
from refactory_dsh.run import BuildLoopStop


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests for classify_empty_emission
# ═══════════════════════════════════════════════════════════════════════════════


def test_classify_reuse():
    """Case 1: passing check, no corrections, matching commit, no review findings → reuse."""
    result = classify_empty_emission(
        check_record={"status": "pass", "commit": "abc123", "runners": []},
        corrections_pending=[],
        consecutive_refusals=0,
        current_commit="abc123",
        review_findings=[],
    )
    assert result.case == "reuse"
    assert result.failing_tests == []
    assert result.blockers == []
    assert result.fingerprint is not None


def test_classify_specific_failure():
    """Case 2: failing check with specific test IDs and corrections → specific_failure."""
    result = classify_empty_emission(
        check_record={
            "status": "fail",
            "runners": [{
                "failures": [
                    {"test_id": "tests/test_a.py::test_x", "assertion": "> assert 1 == 2"},
                    {"test_id": "tests/test_b.py::test_y", "assertion": "> assert True"},
                ],
            }],
        },
        corrections_pending=["fix imports"],
        consecutive_refusals=0,
        current_commit="abc",
        send_back_reason="check failed: 2 tests",
    )
    assert result.case == "specific_failure"
    assert len(result.failing_tests) == 2
    assert len(result.blockers) == 1  # correction IDs
    assert "tests/test_a.py::test_x" in result.detail
    assert "tests/test_b.py::test_y" in result.detail
    assert "check failed: 2 tests" in result.detail


def test_classify_stuck():
    """Case 3: second consecutive empty emission with same fingerprint → stuck."""
    check = {
        "status": "fail",
        "commit": "abc",
        "runners": [{
            "failures": [
                {"test_id": "tests/test_a.py::test_x", "assertion": "> assert 1 == 2"},
            ],
        }],
    }
    # First call: always gives specific_failure (first empty emission).
    result1 = classify_empty_emission(
        check_record=check,
        corrections_pending=[],
        consecutive_refusals=0,
        current_commit="abc",
        prev_fingerprint=None,
    )
    assert result1.case == "specific_failure"

    # Second call with same fingerprint → stuck.
    result2 = classify_empty_emission(
        check_record=check,
        corrections_pending=[],
        consecutive_refusals=1,
        current_commit="abc",
        prev_fingerprint=result1.fingerprint,
    )
    assert result2.case == "stuck"
    assert "tests/test_a.py::test_x" in result2.detail


def test_reuse_blocked_by_pending_corrections():
    """An unmet correction must not be waved through by an empty diff."""
    result = classify_empty_emission(
        check_record={"status": "pass", "commit": "abc", "runners": []},
        corrections_pending=["fix X"],
        consecutive_refusals=0,
        current_commit="abc",
        review_findings=[],
    )
    assert result.case != "reuse"


def test_reuse_blocked_by_failing_check():
    """A failing check must not be waved through by an empty diff."""
    result = classify_empty_emission(
        check_record={"status": "fail", "runners": [], "commit": "abc"},
        corrections_pending=[],
        consecutive_refusals=0,
        current_commit="abc",
        review_findings=[],
    )
    assert result.case != "reuse"


def test_reuse_blocked_by_commit_mismatch():
    """Stale evidence (different commit) must not be reused."""
    result = classify_empty_emission(
        check_record={"status": "pass", "commit": "abc", "runners": []},
        corrections_pending=[],
        consecutive_refusals=0,
        current_commit="def",
        review_findings=[],
    )
    assert result.case != "reuse"


def test_reuse_blocked_by_review_findings():
    """A send-back with review findings blocks reuse, gives specific_failure naming them."""
    result = classify_empty_emission(
        check_record={"status": "pass", "commit": "abc", "runners": []},
        corrections_pending=[],
        consecutive_refusals=0,
        current_commit="abc",
        review_findings=["qa_gate: missing error handling in foo.py"],
    )
    assert result.case == "specific_failure"
    assert "qa_gate: missing error handling in foo.py" in result.detail


def test_first_empty_emission_gives_specific_failure():
    """The first empty emission always gives Case 2 with specific failures, never stuck."""
    result = classify_empty_emission(
        check_record={
            "status": "fail",
            "runners": [{"failures": [{"test_id": "t::x", "assertion": "> y"}]}],
        },
        corrections_pending=[],
        consecutive_refusals=0,
        current_commit="abc",
        prev_fingerprint=None,
    )
    assert result.case == "specific_failure"


def test_second_empty_emission_same_fingerprint_gives_stuck():
    """Second consecutive empty emission with unchanged fingerprint → stuck."""
    check = {
        "status": "fail",
        "runners": [{"failures": [{"test_id": "t::x", "assertion": "> y"}]}],
    }
    result1 = classify_empty_emission(
        check_record=check, corrections_pending=[], consecutive_refusals=0,
        current_commit="a", prev_fingerprint=None,
    )
    result2 = classify_empty_emission(
        check_record=check, corrections_pending=[], consecutive_refusals=1,
        current_commit="a", prev_fingerprint=result1.fingerprint,
    )
    assert result2.case == "stuck"


def test_second_empty_emission_different_fingerprint_gives_specific_failure():
    """Second consecutive empty emission with changed fingerprint → specific_failure again."""
    check_a = {
        "status": "fail",
        "runners": [{"failures": [{"test_id": "t::a", "assertion": "> y"}]}],
    }
    check_b = {
        "status": "fail",
        "runners": [{"failures": [{"test_id": "t::b", "assertion": "> z"}]}],
    }
    result1 = classify_empty_emission(
        check_record=check_a, corrections_pending=[], consecutive_refusals=0,
        current_commit="a", prev_fingerprint=None,
    )
    result2 = classify_empty_emission(
        check_record=check_b, corrections_pending=[], consecutive_refusals=1,
        current_commit="a", prev_fingerprint=result1.fingerprint,
    )
    assert result2.case == "specific_failure"


def test_first_attempt_empty_emission_names_brief_asks():
    """A first-attempt empty emission's message names the brief's asks explicitly."""
    result = classify_empty_emission(
        check_record={},
        corrections_pending=[],
        consecutive_refusals=0,
        current_commit="",
        review_findings=[],
        send_back_reason="",
        brief_asks=["add retry logic to client.py", "add tests for retry"],
    )
    assert result.case == "specific_failure"
    assert "your workspace has no change and nothing has been built yet; the brief asks for" in result.detail
    assert "add retry logic to client.py" in result.detail


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests for _refused_verdict (design mode)
# ═══════════════════════════════════════════════════════════════════════════════


def test_design_case1_reuses_evidence_and_proceeds(tmp_path, monkeypatch):
    """Case 1: passing check + matching commit + no corrections + no review findings → proceed.

    Includes the validation fact (always present after a check in a real run)
    to prove that validation alone does not block reuse — it is the send-back
    reason, not a review finding.
    """
    monkeypatch.setattr(repo_change, "refused", lambda ctx: "edit text not found")
    monkeypatch.setattr(design_ops, "unapplied_corrections", lambda workdir, facts: [])

    ctx = nn.RunContext(tmp_path, facts={
        "check": "python tool.py",
        "commit": "abc123",
        "validation": "(the check passed)",
    }, knobs={"attempts": 3})
    ctx._declared = ("check.txt", "check-record.json", "check-prep.json", "check-verdict.json")
    ctx._declared_facts = ("validation", "attempts_used", "empty_emission_fingerprint")
    ctx.enforce_writes = False

    # Write a passing check-record.json.
    ctx.write_json("check-record.json", {"status": "pass", "commit": "abc123", "runners": []})

    result = design_ops._refused_verdict(ctx)
    assert isinstance(result, Verdict)
    assert str(result) == "proceed"

    # Check-record must NOT be overwritten — the pass record is preserved.
    cr = ctx.read_json("check-record.json", {})
    assert cr["status"] == "pass"

    # CHECK_PREP and CHECK_VERDICT must be written so check_beside_reviews
    # sees a decided attempt and does not crash.
    assert ctx.exists("check-prep.json")
    assert ctx.exists("check-verdict.json")

    # No attempt spent.
    assert not ctx.facts.get("attempts_used")


def test_design_case2_returns_specific_failures(tmp_path, monkeypatch):
    """Case 2: failing check with specific test → reloop with specific failure detail."""
    monkeypatch.setattr(repo_change, "refused", lambda ctx: "edit text not found")
    monkeypatch.setattr(design_ops, "unapplied_corrections", lambda workdir, facts: [])

    ctx = nn.RunContext(tmp_path, facts={
        "check": "python tool.py",
        "commit": "abc",
    }, knobs={"attempts": 3})
    ctx._declared = ("check.txt", "check-record.json")
    ctx._declared_facts = ("validation", "attempts_used", "empty_emission_fingerprint")
    ctx.enforce_writes = False

    # Write a failing check-record.
    ctx.write_json("check-record.json", {
        "status": "fail",
        "runners": [{"failures": [
            {"test_id": "tests/test_foo.py::test_bar", "assertion": "> assert 1 == 2"},
        ]}],
    })

    result = design_ops._refused_verdict(ctx)
    assert isinstance(result, Verdict)
    assert str(result) == "reloop"
    assert "materialise could not apply" in result.reason
    assert "tests/test_foo.py::test_bar" in result.reason
    assert not ctx.facts.get("attempts_used"), "refused emission must not spend an attempt"


def test_design_case2_review_finding_blocks_reuse(tmp_path, monkeypatch):
    """A passing check with a review finding blocks reuse; gives specific_failure naming it."""
    monkeypatch.setattr(repo_change, "refused", lambda ctx: "edit text not found")
    monkeypatch.setattr(design_ops, "unapplied_corrections", lambda workdir, facts: [])

    ctx = nn.RunContext(tmp_path, facts={
        "check": "python tool.py",
        "commit": "abc123",
        "qa_gate": "missing error handling in foo.py",
    }, knobs={"attempts": 3})
    ctx._declared = ("check.txt", "check-record.json")
    ctx._declared_facts = ("validation", "attempts_used", "empty_emission_fingerprint")
    ctx.enforce_writes = False

    ctx.write_json("check-record.json", {"status": "pass", "commit": "abc123", "runners": []})

    result = design_ops._refused_verdict(ctx)
    assert isinstance(result, Verdict)
    assert str(result) == "reloop"  # NOT "proceed"
    assert "missing error handling in foo.py" in result.reason


def test_design_case3_stuck_raises_buildloopstop(tmp_path, monkeypatch):
    """Case 3: two consecutive empty emissions with same failures → BuildLoopStop.

    Simulates the production flow: the first empty emission fires Case 2 (which
    overwrites check-record.json), then the second fires Case 3 using the
    fingerprint recorded by the first call and the runners preserved in the
    overwritten check-record.  The refusal string must indicate a genuine
    no-change emission (not a malformed edit) for fingerprint tracking.
    """
    monkeypatch.setattr(repo_change, "refused",
                        lambda ctx: "the emission changed nothing: every file and edit in it is already in the last attempt")
    monkeypatch.setattr(design_ops, "unapplied_corrections", lambda workdir, facts: [])

    check_record = {
        "status": "fail",
        "runners": [{"failures": [
            {"test_id": "tests/test_stuck.py::test_loop", "assertion": "> assert False"},
        ]}],
    }

    ctx = nn.RunContext(tmp_path, facts={
        "check": "python tool.py",
        "commit": "abc",
        "consecutive_refusals": 0,
    }, knobs={"attempts": 3})
    ctx._declared = ("check.txt", "check-record.json")
    ctx._declared_facts = ("validation", "attempts_used", "empty_emission_fingerprint")
    ctx.enforce_writes = False

    # Write the original failing check-record (as the real check would).
    ctx.write_json("check-record.json", check_record)

    # First empty emission → Case 2 (specific_failure).
    result1 = design_ops._refused_verdict(ctx)
    assert str(result1) == "reloop", "first empty emission must be Case 2"
    assert ctx.facts.get("empty_emission_fingerprint") is not None

    # The overwritten check-record must preserve runners so the next call
    # can compute the same fingerprint.
    cr = ctx.read_json("check-record.json", {})
    assert "runners" in cr, "Case 2 must preserve runners in check-record"

    # Simulate the consecutive_refusals counter that _count_refusal manages.
    ctx.set_fact("consecutive_refusals", 1)

    # Second empty emission with same check-record → Case 3 (stuck).
    with pytest.raises(BuildLoopStop) as exc_info:
        design_ops._refused_verdict(ctx)
    assert "tests/test_stuck.py::test_loop" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests for gate_emit (fast mode)
# ═══════════════════════════════════════════════════════════════════════════════


def test_fast_case1_proceeds(tmp_path, monkeypatch):
    """Case 1 in fast mode: passing check + matching commit → proceed."""
    monkeypatch.setattr(repo_change, "refused", lambda ctx: "edit text not found")

    ctx = nn.RunContext(tmp_path, facts={
        "check": "python tool.py",
        "commit": "abc123",
    }, knobs={})
    ctx._declared = ("fast-report.json", "repair.txt")
    ctx._declared_facts = ("repair_feedback", "repairs_used", "empty_emission_fingerprint")
    ctx.enforce_writes = False

    ctx.write_json("check-record.json", {"status": "pass", "commit": "abc123", "runners": []})

    from refactory_dsh import fast_ops
    result = fast_ops.gate_emit(ctx)
    assert str(result) == "proceed"


def test_fast_case2_reloops_with_specifics(tmp_path, monkeypatch):
    """Case 2 in fast mode: failing check → reloop with specific test IDs."""
    monkeypatch.setattr(repo_change, "refused", lambda ctx: "edit text not found")

    ctx = nn.RunContext(tmp_path, facts={
        "check": "python tool.py",
        "commit": "abc",
    }, knobs={})
    ctx._declared = ("fast-report.json", "repair.txt")
    ctx._declared_facts = ("repair_feedback", "repairs_used", "empty_emission_fingerprint")
    ctx.enforce_writes = False

    ctx.write_json("check-record.json", {
        "status": "fail",
        "runners": [{"failures": [
            {"test_id": "tests/test_q.py::test_w", "assertion": "> assert 0"},
        ]}],
    })

    from refactory_dsh import fast_ops
    result = fast_ops.gate_emit(ctx)
    assert isinstance(result, Verdict)
    assert str(result) == "reloop"
    assert "tests/test_q.py::test_w" in result.reason


def test_fast_case3_stuck_stops(tmp_path, monkeypatch):
    """Case 3 in fast mode: two consecutive empty emissions → BuildLoopStop.

    Simulates the production flow: the first empty emission fires Case 2
    (recording the fingerprint), then the second fires Case 3.  The refusal
    string must indicate a genuine no-change emission for fingerprint tracking.
    """
    monkeypatch.setattr(repo_change, "refused",
                        lambda ctx: "the emission changed nothing: every file and edit in it is already in the last attempt")

    check_record = {
        "status": "fail",
        "runners": [{"failures": [
            {"test_id": "tests/test_fast_stuck.py::test_z", "assertion": "> assert False"},
        ]}],
    }

    ctx = nn.RunContext(tmp_path, facts={
        "check": "python tool.py",
        "commit": "abc",
        "consecutive_refusals": 0,
    }, knobs={})
    ctx._declared = ("fast-report.json", "repair.txt")
    ctx._declared_facts = ("repair_feedback", "repairs_used", "empty_emission_fingerprint")
    ctx.enforce_writes = False

    ctx.write_json("check-record.json", check_record)

    from refactory_dsh import fast_ops

    # First empty emission → Case 2 (specific_failure).
    result1 = fast_ops.gate_emit(ctx)
    assert str(result1) == "reloop", "first empty emission must be Case 2"
    assert ctx.facts.get("empty_emission_fingerprint") is not None

    # Simulate the consecutive_refusals counter that _count_refusal manages.
    ctx.set_fact("consecutive_refusals", 1)

    # Second empty emission with same check-record → Case 3 (stuck).
    with pytest.raises(BuildLoopStop) as exc_info:
        fast_ops.gate_emit(ctx)
    assert "tests/test_fast_stuck.py::test_z" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════════
# Fingerprint lifecycle: cleared on successful materialise
# ═══════════════════════════════════════════════════════════════════════════════


def test_materialise_clears_stale_fingerprint(tmp_path, monkeypatch):
    """A successful materialise clears the fingerprint so the next cycle's first
    empty emission gets Case 2 (specific_failure), not Case 3 from a stale match.

    Correction 2: 'The first empty emission always gets Case 2 with the
    specific failures.'
    """
    from refactory_dsh import repo_change as rc, worktree

    # Pretend there is no worktree (directory-mode materialise).
    monkeypatch.setattr(worktree, "recorded", lambda workdir: None)
    monkeypatch.setattr(rc, "_materialise_directory", lambda ctx, source, files: {
        "artifact": str(tmp_path / "art"), "files": [], "commit": "none",
    })

    ctx = nn.RunContext(tmp_path, facts={
        "name": "test_art",
        "empty_emission_fingerprint": {"failures": [["t::x", "> y"]], "corrections": []},
        "consecutive_refusals": 2,
    })
    ctx._declared = ("emit.txt", "changes.diff", "changed-functions.json")
    ctx._declared_facts = ("artifact", "files", "commit", "change",
                           "emission_refused", "changed_functions",
                           "consecutive_refusals", "empty_emission_fingerprint")
    ctx.enforce_writes = False

    result = rc.materialise(ctx)

    # The fingerprint must be explicitly set to None in the return dict
    # (not merely absent), so the executor writes it back as a fact and
    # overwrites the stale value.  consecutive_refusals must be reset.
    assert "empty_emission_fingerprint" in result, \
        "materialise must return empty_emission_fingerprint to overwrite the stale fact"
    assert result["empty_emission_fingerprint"] is None
    assert result.get("consecutive_refusals") == 0
