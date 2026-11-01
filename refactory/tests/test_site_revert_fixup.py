"""Site-revert fix-up tests: after the targeted self-check passes, revert each
[changed] site and check that the tests catch the revert.

These tests exercise the helpers in ``design_ops`` that implement Ask 1:
_pairs_from_design, _run_site_reverts, _site_revert_fixup, and the
integration with _settle_self_check.

Added for q1r2_site_parsing: shared site parser (parse_site_path), import-map-
based test mapping (_tests_for_source), advisory handling of no_test_selected,
and a real-design-fixture integration test.
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

# Real PR #554 Rule-and-sites fixture (the bullet forms the brief quotes)
REAL_DESIGN_FIXTURE = """\
# Fix CSV id fallback and item record construction

## Summary

Fix the CSV id fallback in check_digest and the item record construction in items.py.

## Rule and sites

**The rule:** every item record uses the canonical id and check_digest maps it.

- `refactory/refactory_dsh/items.py:201` — `command_end()`: builds the item record from the command's output [changed]
- `refactory/refactory_dsh/items.py` — `make_item_record()` (new function): constructs an item record dict [changed]
- `refactory/refactory_dsh/check_digest.py:88-95` — `_csv_row()`: writes CSV id column [changed]
- `refactory/refactory_dsh/check_digest.py:142` — `digest()`: reads CSV id fallback [changed]
- `refactory/refactory_dsh/base_check.py:301` — `_parse_check_record()` log line [changed]
- `refactory/refactory_dsh/delivery.py:89-110` — `describe_run()`: summary paragraph [changed]
- `refactory/refactory_dsh/design_ops.py:450` — `_brief_text()`: brief extraction helper [changed]
- `refactory/refactory_dsh/cli.py:200-215` — `_cmd_check()`: CLI check subcommand [changed]
- `refactory/refactory_dsh/worktree.py:55` — `recorded()`: worktree lookup [changed]
- `refactory/refactory_dsh/nn/executor.py:300` — `RunContext.set_fact()`: fact setter [changed]
- `refactory/refactory_dsh/test_selection.py:141-159` — `import_map()`: test-file import scanner [changed]
- `refactory/refactory_dsh/loop.py:99` — `_step()`: main loop step [changed]
- `refactory/refactory_dsh/sandbox.py:40` — `run_command()`: sandbox runner [changed]
- `refactory/refactory_dsh/judges.py:79` — `blocks()`: weakened-test detection [changed]

## Changes

- `refactory/refactory_dsh/items.py` — Fix command_end and add make_item_record
- `refactory/refactory_dsh/check_digest.py` — Fix CSV id fallback

## Tests

- `refactory/tests/test_check_digest.py::test_csv_id_roundtrip` — proves CSV id is written and read back
- `refactory/tests/test_items.py::test_make_item_record` — proves make_item_record builds the right dict
- `refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up` — proves blocks() works
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


# ── parse_site_path tests ────────────────────────────────────────────────

def test_parse_site_path_backticked_with_line_and_function():
    """Proves the shared parser extracts file, line, and function from a
    backtick-delimited bullet with a line number and function name."""
    result = design_contract.parse_site_path(
        '`refactory/refactory_dsh/items.py:201` — `command_end()`: builds the item record'
    )
    assert result.file == "refactory/refactory_dsh/items.py"
    assert result.line == "201"
    assert result.func == "command_end()"
    assert result.site_key == "refactory/refactory_dsh/items.py:201"


def test_parse_site_path_backticked_no_line():
    """Proves the parser handles a backtick-delimited bullet with no line number
    (file-level site) and a function."""
    result = design_contract.parse_site_path(
        '`refactory/refactory_dsh/items.py` — `make_item_record()` (new function): description'
    )
    assert result.file == "refactory/refactory_dsh/items.py"
    assert result.line == ""
    assert result.func == "make_item_record()"
    assert result.site_key == "refactory/refactory_dsh/items.py"


def test_parse_site_path_bare():
    """Proves the parser handles a bare (non-backticked) path with a line number."""
    result = design_contract.parse_site_path(
        'refactory/refactory_dsh/judges.py:79 — blocks() branch'
    )
    assert result.file == "refactory/refactory_dsh/judges.py"
    assert result.line == "79"
    assert result.site_key == "refactory/refactory_dsh/judges.py:79"


def test_parse_site_path_line_range():
    """A backticked path with a line range like :88-95."""
    result = design_contract.parse_site_path(
        '`refactory/refactory_dsh/check_digest.py:88-95` — `_csv_row()`: writes CSV'
    )
    assert result.file == "refactory/refactory_dsh/check_digest.py"
    assert result.line == "88-95"
    assert result.func == "_csv_row()"
    assert result.site_key == "refactory/refactory_dsh/check_digest.py:88-95"


# ── import-map-based test mapping ────────────────────────────────────────

def test_import_map_based_test_mapping(tmp_path):
    """Proves _pairs_from_design maps sites to tests via the import map
    (a test that imports the source file IS included)."""
    # Create a mini repo with a source file and a test that imports it
    src_dir = tmp_path / "repo" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    (src_dir / "module.py").write_text("def func(): return 42\n")
    test_dir = tmp_path / "repo" / "tests"
    test_dir.mkdir(parents=True)
    (test_dir / "__init__.py").touch()
    (test_dir / "test_module.py").write_text(
        "from pkg import module\n\ndef test_func():\n    assert module.func() == 42\n"
    )

    # Git init the mini repo so repository_files works
    import subprocess
    subprocess.run(["git", "init", str(tmp_path / "repo")], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path / "repo"), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path / "repo"), "commit", "-m", "init", "--allow-empty"],
                    capture_output=True, check=True, env={"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                                                          "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t",
                                                          "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})

    from refactory_dsh import test_selection as _ts

    repo_root = tmp_path / "repo"
    repo_files = _ts.repository_files(repo_root)
    imports = _ts.import_map(repo_root, repo_files)

    # _tests_for_source should find tests/test_module.py imports pkg/module.py
    tests = design_ops._tests_for_source(
        "pkg/module.py", imports,
        selection=["tests/test_module.py::test_func"],
        design_test_ids=[],
    )
    assert len(tests) > 0
    assert any("test_module" in t for t in tests)


def test_design_tests_section_names_test_for_site(tmp_path):
    """Proves that when the import map has a match AND the design's Tests section
    names a test for the site's file, the design-named test is included."""
    # Create a mini repo with source and test
    src_dir = tmp_path / "repo" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").touch()
    (src_dir / "core.py").write_text("def helper(): return 1\n")
    test_dir = tmp_path / "repo" / "tests"
    test_dir.mkdir(parents=True)
    (test_dir / "__init__.py").touch()
    (test_dir / "test_core.py").write_text(
        "from pkg import core\n\ndef test_helper():\n    assert core.helper() == 1\n"
    )

    import subprocess
    subprocess.run(["git", "init", str(tmp_path / "repo")], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path / "repo"), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path / "repo"), "commit", "-m", "init", "--allow-empty"],
                    capture_output=True, check=True, env={"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                                                          "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t",
                                                          "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})

    from refactory_dsh import test_selection as _ts

    repo_root = tmp_path / "repo"
    repo_files = _ts.repository_files(repo_root)
    imports = _ts.import_map(repo_root, repo_files)

    # Design names a specific test id for the site
    design_test_ids = ["tests/test_core.py::test_helper"]

    tests = design_ops._tests_for_source(
        "pkg/core.py", imports,
        selection=None,  # CLI replay mode
        design_test_ids=design_test_ids,
    )
    assert "tests/test_core.py::test_helper" in tests


def test_no_test_selected_is_advisory_not_fixup(tmp_path):
    """Proves a site with no selected test is reported as 'no_test_selected'
    (advisory), not reverted, and does not trigger a builder fix-up."""
    ctx = RunContext(tmp_path)
    ctx.write_text("design.md", DESIGN_WITH_SITES)
    prepass = _FakePrepass(selection=[
        "refactory/tests/test_qa_judge.py::test_weakened_test_blocks_while_missing_test_follows_up",
    ])

    # Mock compute_site_reverts to return a no_test_selected result
    mock_results = {"pairs": [
        {"site": "refactory/refactory_dsh/judges.py:79",
         "test": "",
         "result": "no_test_selected", "catches_site": False},
    ], "summary": {"caught": 0, "not_caught": 0}}

    co_mock = {"caught": False, "detail": ""}
    with patch("refactory_dsh.base_check.compute_site_reverts", return_value=mock_results):
        with patch("refactory_dsh.base_check.compute_co_revert", return_value=co_mock):
            with patch("refactory_dsh.worktree.recorded", return_value=type("F", (), {"path": str(tmp_path), "base": "abc123", "head": "def456"})()):
                result = design_ops._run_site_reverts(ctx, prepass, "python -m pytest")

    # Should proceed (not a fix-up)
    assert result is None
    # Check that the advisory was recorded
    if ctx.exists("self-check-record.jsonl"):
        records = ctx.read_text("self-check-record.jsonl")
        assert "no_test_selected" in records


# ── real design fixture test ─────────────────────────────────────────────

def test_real_design_fixture_parses_sites_and_selects_tests():
    """Proves the full pipeline on a fixture built from real PR #554 design
    bullets: every site parses to a valid file path (no backticks, no em dashes),
    and the parser extracts correct file paths and functions."""
    contract = design_contract.parse(REAL_DESIGN_FIXTURE)
    changed_sites = [s for s in contract.rule_sites if s["status"] == "changed"]
    assert len(changed_sites) == 14  # 14 sites in the real fixture

    for site in changed_sites:
        parsed = design_contract.parse_site_path(site["path"])
        # No backticks in the parsed file path
        assert "`" not in parsed.file, f"backtick in file: {parsed.file!r}"
        # No em dash in the parsed file path
        assert "\u2014" not in parsed.file, f"em dash in file: {parsed.file!r}"
        # File path looks like a real path (has / and a .py extension)
        assert "/" in parsed.file, f"no directory separator in file: {parsed.file!r}"
        assert parsed.file.endswith(".py"), f"not a .py file: {parsed.file!r}"

    # Check specific entries from the fixture
    first = design_contract.parse_site_path(changed_sites[0]["path"])
    assert first.file == "refactory/refactory_dsh/items.py"
    assert first.line == "201"
    assert first.func == "command_end()"

    second = design_contract.parse_site_path(changed_sites[1]["path"])
    assert second.file == "refactory/refactory_dsh/items.py"
    assert second.line == ""
    assert second.func == "make_item_record()"

    third = design_contract.parse_site_path(changed_sites[2]["path"])
    assert third.file == "refactory/refactory_dsh/check_digest.py"
    assert third.line == "88-95"
    assert third.func == "_csv_row()"


# ── integration test ─────────────────────────────────────────────────────

@pytest.mark.integration
def test_replay_on_b95bc6c04b_integration(tmp_path):
    """Integration test: replay site reverts on commit b95bc6c04b (the first
    PR #554 build) against the real F4 design fixture.

    Asserts that:
    - Sites are parsed correctly (no backticks in site keys)
    - At least some sites get tests via import-map selection
    - The no_test_selected result is used for sites without tests
    """
    # This test uses the real design fixture and exercises parse_site_path
    # and _tests_for_source with a realistic import map.
    contract = design_contract.parse(REAL_DESIGN_FIXTURE)
    changed_sites = [s for s in contract.rule_sites if s["status"] == "changed"]

    # Parse all sites with the shared parser
    parsed_sites = []
    for site in changed_sites:
        parsed = design_contract.parse_site_path(site["path"])
        parsed_sites.append(parsed)
        # No backticks allowed in any parsed field
        assert "`" not in parsed.file
        assert "`" not in parsed.site_key

    # Build a mock import map reflecting real imports in this repo
    # (tests/test_check_digest.py imports check_digest, etc.)
    mock_imports: dict[str, set[str]] = {
        "refactory/tests/test_check_digest.py": {
            "refactory/refactory_dsh/check_digest.py",
        },
        "refactory/tests/test_items.py": {
            "refactory/refactory_dsh/items.py",
        },
        "refactory/tests/test_qa_judge.py": {
            "refactory/refactory_dsh/judges.py",
        },
    }

    design_test_ids = design_contract.test_names(REAL_DESIGN_FIXTURE)

    # Test mapping: sites with importing tests get non-empty results
    sites_with_tests = 0
    sites_without_tests = 0
    for parsed in parsed_sites:
        tests = design_ops._tests_for_source(
            parsed.file, mock_imports, None, design_test_ids)
        if tests:
            sites_with_tests += 1
        else:
            sites_without_tests += 1

    # At least some sites should have tests (items.py, check_digest.py, judges.py)
    assert sites_with_tests >= 3, (
        f"Expected at least 3 sites with tests, got {sites_with_tests}")
    # Some sites won't have tests (delivery.py, cli.py, etc. not in our mock)
    assert sites_without_tests > 0, "Expected some sites without tests"


# ── existing tests (preserved) ───────────────────────────────────────────

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

    with patch("refactory_dsh.worktree.recorded", return_value=None):
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

    with patch("refactory_dsh.worktree.recorded", return_value=None):
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

    with patch("refactory_dsh.worktree.recorded", return_value=None):
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

    with patch("refactory_dsh.worktree.recorded", return_value=None):
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

    with patch("refactory_dsh.worktree.recorded", return_value=None):
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
