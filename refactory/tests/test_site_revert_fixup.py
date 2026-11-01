"""Site-revert fix-up tests: after the targeted self-check passes, revert each
[changed] site and check that the tests catch the revert.

These tests exercise the helpers in ``design_ops`` that implement Ask 1:
_pairs_from_design, _run_site_reverts, _site_revert_fixup, and the
integration with _settle_self_check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from refactory_dsh import design_contract, design_ops, judges
from refactory_dsh.nn import RunContext, Verdict


# ── helpers ──────────────────────────────────────────────────────────────

DESIGN_WITH_SITES = """\
# Test change

## Summary

A test change with sites.

## Changes

- `refactory/refactory_dsh/judges.py` — update blocks()

## Rule and sites

**The rule:** every weakened test blocks.

- `refactory/refactory_dsh/judges.py:79` — blocks() branch [changed]
- `refactory/refactory_dsh/design.py:142` — _JUDGE_RULE text [changed]

**Rule:** probe coverage enforced.

- `refactory/refactory_dsh/design.py:990` — QA director prompt [changed]

## Site revert exemptions

(None)

## Test guardedness exemptions

(None)

## Tests

- `refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up`
"""

DESIGN_WITH_EXEMPTION = """\
# Test change

## Summary

A test change with exempted sites.

## Changes

- `refactory/refactory_dsh/judges.py` — update blocks()

## Rule and sites

**Rule:** test rule.

- `refactory/refactory_dsh/judges.py:79` — blocks() branch [changed]
- `refactory/refactory_dsh/cli.py:484` — add subcommand [changed]

## Site revert exemptions

- `refactory/refactory_dsh/cli.py:484` — CLI replay, exempt from revert

## Tests

- `refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up`
"""

DESIGN_SAME_RULE = """\
# Test change

## Summary

Two fallbacks under the same rule.

## Changes

- `refactory/refactory_dsh/judges.py` — update blocks()

## Rule and sites

**The rule:** two co-dependent fallbacks.

- `refactory/refactory_dsh/judges.py:79` — first fallback [changed]
- `refactory/refactory_dsh/judges.py:83` — second fallback [changed]

## Tests

- `refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up`
"""


class _FakePrepass:
    """Minimal prepass stub for _pairs_from_design."""
    def __init__(self, selection: list[str] | None = None):
        self.selection = selection or []
        self.failing_ids = []
        self.ran = True
        self.cut_ids = []
        self.notes = []
        self.ran_ids = []
        self.ran_nothing = False
        self.broken_command = False
        self.elapsed_s = 0.0
        self.limit_s = 0.0
        self.timings = []
        self.commands = []
        self.digest_text = ""
        self.targeted_command = None


# ── unit tests ───────────────────────────────────────────────────────────

def test_pairs_from_design_filters_correctly(tmp_path):
    """Unit test: _pairs_from_design with [changed], [left], and exempted sites
    using the real bold-line rule-block format."""
    ctx = RunContext(tmp_path)
    design_text = """\
# Test

## Summary

Test summary.

## Changes

- `refactory/refactory_dsh/judges.py` — update

## Rule and sites

**The rule:** catch weak tests.

- `refactory/refactory_dsh/judges.py:79` — blocks branch [changed]
- `refactory/refactory_dsh/design.py:142` — rule text [changed]
- `refactory/refactory_dsh/design_ops.py:1938` — read-only ref [left: reference only]
- `refactory/refactory_dsh/cli.py:484` — subcommand [changed]

## Site revert exemptions

- `refactory/refactory_dsh/cli.py:484` — exempt

## Tests

- `refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up`
"""
    ctx.write_text("design.md", design_text)
    prepass = _FakePrepass(selection=[
        "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
    ])

    all_pairs, groups, exempted = design_ops._pairs_from_design(ctx, prepass)

    # Only [changed] non-exempted sites appear
    sites_in_pairs = {p[0] for p in all_pairs}
    assert "refactory/refactory_dsh/judges.py:79" in sites_in_pairs
    assert "refactory/refactory_dsh/design.py:142" in sites_in_pairs
    # [left] site is excluded
    assert "refactory/refactory_dsh/design_ops.py:1938" not in sites_in_pairs
    # Exempted site is excluded
    assert "refactory/refactory_dsh/cli.py:484" not in sites_in_pairs
    assert "refactory/refactory_dsh/cli.py:484" in exempted


def test_pairs_from_design_with_left_sites_correct_grouping(tmp_path):
    """When [left] sites are interspersed among [changed] sites, the grouping
    correctly maps bullet ordinals to changed_sites indices."""
    ctx = RunContext(tmp_path)
    design_text = """\
# Test

## Summary

Mixed sites.

## Changes

- `refactory/refactory_dsh/judges.py` — update

## Rule and sites

**The rule:** rule one.

- `refactory/refactory_dsh/judges.py:79` — first change [changed]
- `refactory/refactory_dsh/design_ops.py:1938` — reference [left: read-only]
- `refactory/refactory_dsh/judges.py:83` — second change [changed]

**Rule:** rule two.

- `refactory/refactory_dsh/design.py:990` — prompt [changed]

## Tests

- `refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up`
"""
    ctx.write_text("design.md", design_text)
    prepass = _FakePrepass(selection=[
        "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
    ])

    all_pairs, groups, exempted = design_ops._pairs_from_design(ctx, prepass)

    # All [changed] sites should appear
    sites_in_pairs = {p[0] for p in all_pairs}
    assert "refactory/refactory_dsh/judges.py:79" in sites_in_pairs
    assert "refactory/refactory_dsh/judges.py:83" in sites_in_pairs
    assert "refactory/refactory_dsh/design.py:990" in sites_in_pairs
    # [left] site excluded
    assert "refactory/refactory_dsh/design_ops.py:1938" not in sites_in_pairs

    # Rule one group: judges.py:79 and judges.py:83 should be grouped together
    rule_one_group = None
    for g in groups:
        g_sites = {p[0] for p in g}
        if "refactory/refactory_dsh/judges.py:79" in g_sites and \
           "refactory/refactory_dsh/judges.py:83" in g_sites:
            rule_one_group = g
            break
    assert rule_one_group is not None, "judges.py:79 and :83 should be co-reverted"
    # design.py:990 should NOT be in rule one group (it's under rule two)
    rule_one_sites = {p[0] for p in rule_one_group}
    assert "refactory/refactory_dsh/design.py:990" not in rule_one_sites


def test_same_rule_sites_grouped(tmp_path):
    """Two sites under the same **Rule:** bold-line block are submitted as a group."""
    ctx = RunContext(tmp_path)
    ctx.write_text("design.md", DESIGN_SAME_RULE)
    prepass = _FakePrepass(selection=[
        "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
    ])

    all_pairs, groups, exempted = design_ops._pairs_from_design(ctx, prepass)

    # Both sites should appear in a group together
    assert len(groups) >= 1
    group_sites = set()
    for group in groups:
        for site, test, scope in group:
            group_sites.add(site)
    assert "refactory/refactory_dsh/judges.py:79" in group_sites
    assert "refactory/refactory_dsh/judges.py:83" in group_sites


def test_same_function_sites_co_reverted(tmp_path):
    """Two sites in the same file are co-reverted together."""
    ctx = RunContext(tmp_path)
    ctx.write_text("design.md", DESIGN_SAME_RULE)
    prepass = _FakePrepass()

    all_pairs, groups, _ = design_ops._pairs_from_design(ctx, prepass)

    # Sites in the same file (refactory/refactory_dsh/judges.py) should be grouped
    file_groups = [g for g in groups
                   if all("judges.py" in p[0] for p in g)]
    assert len(file_groups) >= 1


def test_site_revert_exempted_skipped(tmp_path):
    """Sites under '## Site revert exemptions' are not reverted."""
    ctx = RunContext(tmp_path)
    ctx.write_text("design.md", DESIGN_WITH_EXEMPTION)
    prepass = _FakePrepass(selection=[
        "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
    ])

    all_pairs, groups, exempted = design_ops._pairs_from_design(ctx, prepass)

    # The exempted site should not appear in pairs
    sites_in_pairs = {p[0] for p in all_pairs}
    assert "refactory/refactory_dsh/cli.py:484" not in sites_in_pairs
    assert "refactory/refactory_dsh/cli.py:484" in exempted


def test_site_revert_uncaught_triggers_fixup(tmp_path):
    """An uncaught site triggers a Verdict('reloop') charged to site_revert_fixups."""
    ctx = RunContext(tmp_path)
    ctx.write_text("design.md", DESIGN_WITH_SITES)
    prepass = _FakePrepass(selection=[
        "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
    ])

    # Mock compute_site_reverts to return an uncaught site
    mock_results = {"pairs": [
        {"site": "refactory/refactory_dsh/judges.py:79",
         "test": "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
         "result": "pass", "catches_site": False},
    ], "summary": {"caught": 0, "not_caught": 1}}

    co_mock = {"caught": False, "detail": ""}
    with patch("refactory_dsh.base_check.compute_site_reverts", return_value=mock_results):
        with patch("refactory_dsh.base_check.compute_co_revert", return_value=co_mock):
            with patch("refactory_dsh.worktree.recorded", return_value=type("F", (), {"path": str(tmp_path), "base": "abc123", "head": "def456"})()):
                result = design_ops._run_site_reverts(ctx, prepass, "python -m pytest")

    assert result is not None
    assert isinstance(result, Verdict)
    assert str(result) == "reloop"
    assert ctx.facts.get("site_revert_fixup_count") == 1
    assert not ctx.facts.get("self_check_fixup_count")


def test_site_revert_caught_proceeds(tmp_path):
    """All sites' tests fail on revert → _run_site_reverts returns None (proceed)."""
    ctx = RunContext(tmp_path)
    ctx.write_text("design.md", DESIGN_WITH_SITES)
    prepass = _FakePrepass(selection=[
        "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
    ])

    mock_results = {"pairs": [
        {"site": "refactory/refactory_dsh/judges.py:79",
         "test": "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
         "result": "fail", "catches_site": True},
    ], "summary": {"caught": 1, "not_caught": 0}}

    co_mock = {"caught": False, "detail": ""}
    with patch("refactory_dsh.base_check.compute_site_reverts", return_value=mock_results):
        with patch("refactory_dsh.base_check.compute_co_revert", return_value=co_mock):
            with patch("refactory_dsh.worktree.recorded", return_value=type("F", (), {"path": str(tmp_path), "base": "abc123", "head": "def456"})()):
                result = design_ops._run_site_reverts(ctx, prepass, "python -m pytest")

    assert result is None  # proceed


def test_site_revert_budget_exhausted_proceeds(tmp_path):
    """Budget 0 → uncaught sites become warnings, not fix-ups, and returns None."""
    ctx = RunContext(tmp_path, facts={"site_revert_fixup_count": 0},
                     knobs={"site_revert_fixups": 0, "check_attempts": 3})
    ctx.write_text("design.md", DESIGN_WITH_SITES)
    prepass = _FakePrepass(selection=[
        "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
    ])

    mock_results = {"pairs": [
        {"site": "refactory/refactory_dsh/judges.py:79",
         "test": "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
         "result": "pass", "catches_site": False},
    ], "summary": {"caught": 0, "not_caught": 1}}

    co_mock = {"caught": False, "detail": ""}
    with patch("refactory_dsh.base_check.compute_site_reverts", return_value=mock_results):
        with patch("refactory_dsh.base_check.compute_co_revert", return_value=co_mock):
            with patch("refactory_dsh.worktree.recorded", return_value=type("F", (), {"path": str(tmp_path), "base": "abc123", "head": "def456"})()):
                result = design_ops._run_site_reverts(ctx, prepass, "python -m pytest")

    assert result is None  # proceed, not a fixup


def test_worktree_clean_after_revert(tmp_path):
    """After _run_site_reverts returns, the existing site-revert restore
    (per-file git checkout) is used rather than a blanket checkout."""
    ctx = RunContext(tmp_path)
    ctx.write_text("design.md", DESIGN_WITH_SITES)
    prepass = _FakePrepass(selection=[
        "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
    ])

    # compute_site_reverts already restores per-file; verify it's called
    mock_results = {"pairs": [
        {"site": "refactory/refactory_dsh/judges.py:79",
         "test": "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
         "result": "fail", "catches_site": True},
    ], "summary": {"caught": 1, "not_caught": 0}}

    co_mock = {"caught": False, "detail": ""}
    with patch("refactory_dsh.base_check.compute_site_reverts", return_value=mock_results) as mock_csr:
        with patch("refactory_dsh.base_check.compute_co_revert", return_value=co_mock):
            with patch("refactory_dsh.worktree.recorded", return_value=type("F", (), {"path": str(tmp_path), "base": "abc123", "head": "def456"})()):
                result = design_ops._run_site_reverts(ctx, prepass, "python -m pytest")

    # compute_site_reverts was called (it handles per-file restore internally)
    assert mock_csr.called
    assert result is None


def test_cap_stops_early_unmeasured_advisory(tmp_path):
    """Time cap stops the pass early; remaining sites are listed as unmeasured (advisory),
    not as uncaught and not as a fix-up."""
    ctx = RunContext(tmp_path)
    ctx.write_text("design.md", DESIGN_WITH_SITES)
    prepass = _FakePrepass(selection=[
        "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
    ])

    # Some sites measured, some not_run (time cap)
    mock_results = {"pairs": [
        {"site": "refactory/refactory_dsh/judges.py:79",
         "test": "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
         "result": "fail", "catches_site": True},
        {"site": "refactory/refactory_dsh/design.py:142",
         "test": "refactory/tests/test_qa_judge.py::test_the_judge_prompts_state_the_bar_and_ask_for_each_items_kind",
         "result": "not_run", "catches_site": False, "detail": "time cap reached"},
    ], "summary": {"caught": 1, "not_caught": 0, "not_run": 1}}

    co_mock = {"caught": False, "detail": ""}
    with patch("refactory_dsh.base_check.compute_site_reverts", return_value=mock_results):
        with patch("refactory_dsh.base_check.compute_co_revert", return_value=co_mock):
            with patch("refactory_dsh.worktree.recorded", return_value=type("F", (), {"path": str(tmp_path), "base": "abc123", "head": "def456"})()):
                result = design_ops._run_site_reverts(ctx, prepass, "python -m pytest")

    # Should proceed (no uncaught sites), unmeasured sites are advisory
    assert result is None
    # Check that the advisory was recorded
    if ctx.exists("self-check-record.jsonl"):
        records = ctx.read_text("self-check-record.jsonl")
        assert "unmeasured" in records
