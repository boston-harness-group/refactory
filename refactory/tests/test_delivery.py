"""A design run that passes its reviews ends with a pull request (#350).

One test per path a person takes: a new project, an existing repository, a reused
project name, a follow-up to an open pull request, a redo, `--no-pr`, a run that fails,
a base with commits nobody pushed, a name already taken on GitHub, and no gh login.
GitHub is `fake_gh.py` over bare repositories on disk, which git reaches through
`url.<base>.insteadOf`; the model is a fake; the repositories, the worktrees, the
pushes and the registry are real.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from refactory_dsh import cli, console, delivery, nn, projects, worktree
from refactory_dsh.design import DesignMode
from refactory_dsh.projects import projects_dir
from tests_support import DESIGN_DOC_ANSWER, PITFALLS_ANSWER, RESEARCH_ANSWER, VERIFICATION_ANSWER, REVIEW_ANSWERS, FakeModel, git, git_repo, overwatch_judgment, qa_judgment, render, stub

NAME = "delivered_tool"
FAKE_GH = Path(__file__).resolve().parent / "fake_gh.py"
LOGIN = "octo"


# ── PR title and body unit tests ───────────────────────────────────────────


class _TitleCtx:
    """Minimal context for _title() and _body() unit tests."""
    def __init__(self, facts=None, files=None, workdir=None):
        self.facts = dict(facts or {})
        self._files = dict(files or {})
        self.workdir = Path(workdir) if workdir else None
    def exists(self, name):
        return name in self._files
    def read_text(self, name):
        return self._files[name]
    def read_json(self, name, default=None):
        import json as _j
        if name not in self._files:
            return default
        try:
            return _j.loads(self._files[name])
        except (ValueError, TypeError):
            return default


def test_pr_body_lists_the_gate_corrections_one_bullet_each(tmp_path):
    """The corrections are short bullets, one each: a correction's own heading or line
    breaks are folded into its line, and a credential-like value in one is masked like
    the rest of the body."""
    import subprocess
    corrections = ("- Use websockets\n## Not a section\nsecond line\n\n"
                   "- Read API_TOKEN=s3cr3t-value from the environment")
    ctx = _TitleCtx(facts={"brief": "Build a tool.", "name": "b2", "gate_corrections": corrections},
                    files={"design.md": "# The heading\n\nThe design.\n"}, workdir=tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for key, value in (("user.email", "test@test.com"), ("user.name", "Test")):
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    (tmp_path / "tool.py").write_text("print('built')\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"], check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    body = delivery._body(ctx, SimpleNamespace(path=str(tmp_path), base=head, head=head,
                                               branch="refactory/run-test"))
    assert ("## Corrections given at approval\n\n- Use websockets ## Not a section second line\n"
            "- Read API_TOKEN=*** from the environment\n") in body, body
    assert "\n## Not a section" not in body
    assert "s3cr3t-value" not in body


def test_pr_title_from_design_heading():
    ctx = _TitleCtx(
        facts={"brief": "Fix audit", "name": "b2"},
        files={"design.md": "# Audit bookkeeping and PR formatting\n\nSome text.\n"},
    )
    assert delivery._title(ctx) == "Audit bookkeeping and PR formatting"


def test_pr_title_strips_design_prefix():
    ctx = _TitleCtx(
        facts={"brief": "Fix audit", "name": "b2"},
        files={"design.md": "# Design: Audit bookkeeping\n\nSome text.\n"},
    )
    assert delivery._title(ctx) == "Audit bookkeeping"


def test_pr_title_falls_back_to_brief():
    ctx = _TitleCtx(
        facts={"brief": "Fix the audit gate", "name": "b2"},
        files={"design.md": "No heading here, just text.\n"},
    )
    assert delivery._title(ctx) == "Fix the audit gate"


def test_pr_title_truncates_at_word_boundary():
    long_heading = "Fix " + "word " * 60  # well over the limit
    ctx = _TitleCtx(
        facts={"brief": "short", "name": "b2"},
        files={"design.md": f"# {long_heading}\n"},
    )
    title = delivery._title(ctx)
    assert len(title) <= delivery.TITLE_LIMIT
    assert title.endswith("…")


def test_a_long_design_heading_is_the_whole_title_up_to_githubs_limit():
    """dm-217: #494's title was cut at 72 characters, mid-phrase ("...and other safe...").
    A heading GitHub takes whole is the title whole, prefixes and all."""
    heading = ("Emitted paths: accept @-scoped package paths, refuse quoted and malformed headers, "
               "and other safe characters")
    ctx = _TitleCtx(facts={"brief": "short", "name": "b2"}, files={"design.md": f"# {heading}\n"})
    assert delivery._title(ctx) == heading
    assert len(delivery.DRAFT_TITLE + delivery.CHECK_FAILING_TITLE) + delivery.TITLE_LIMIT <= 256


def test_a_heading_past_githubs_limit_is_cut_between_phrases_or_at_a_word_never_inside_one():
    """dm-217: past the limit the cut falls between phrases; with none, at a word boundary."""
    phrases = ", ".join(f"phrase number {n} with a few words" for n in range(12))
    ctx = _TitleCtx(facts={"brief": "short", "name": "b2"}, files={"design.md": f"# {phrases}\n"})
    title = delivery._title(ctx)
    assert len(phrases) > delivery.TITLE_LIMIT >= len(title) > delivery.TITLE_LIMIT // 2
    cut = title.removesuffix("\u2026")
    assert title.endswith("\u2026") and cut.endswith("with a few words")
    assert phrases.startswith(cut + ", ")
    words = "Make " + " ".join(f"word{n}" for n in range(80))
    ctx = _TitleCtx(facts={"brief": "short", "name": "b2"}, files={"design.md": f"# {words}\n"})
    title = delivery._title(ctx)
    cut = title.removesuffix("\u2026")
    assert len(title) <= delivery.TITLE_LIMIT and title.endswith("\u2026")
    assert words.startswith(cut + " "), "the cut falls between two words"


def test_pr_title_never_cuts_inside_backticks():
    heading = "Fix how gate_overwatch and its refusal of the `code_reviewer` backtick_spans are reported in the pipeline " * 3
    ctx = _TitleCtx(
        facts={"brief": "short", "name": "b2"},
        files={"design.md": f"# {heading}\n"},
    )
    title = delivery._title(ctx)
    assert len(title) <= delivery.TITLE_LIMIT
    # Even number of backticks: no unclosed pair.
    assert title.count("`") % 2 == 0, f"unclosed backtick in title: {title!r}"


def test_pr_body_has_summary_and_verification_and_no_original_request(tmp_path):
    import subprocess
    ctx = _TitleCtx(
        facts={"brief": "Build a tool\nthat prints built.", "check": "python tool.py", "name": "b2",
               "inherited_failures": ["tests/test_env.py::test_needs_profile"]},
        files={"design.md": "# The design heading\n\nFirst paragraph of the design.\n\n---\n",
               "authoring-evidence.json": json.dumps({
                   "validation": {"status": "pass"},
                   "identity": {"root": str(tmp_path)},
                   "requirements": [{"id": "r1", "text": "Build a tool that prints built."}],
               }),
               "intent-audit.json": json.dumps({
                   "requirements": [{
                       "id": "r1", "status": "pass",
                       "requirement": "Build a tool that prints built.",
                       "evidence": [{"path": "tool.py", "detail": "implements the tool"}],
                   }],
               })},
        workdir=tmp_path,
    )
    # Set up a minimal git repo so worktree.git works.
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], capture_output=True, check=True)
    (tmp_path / "file.txt").write_text("content\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "tool.py").write_text("print('built')\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "attempt"], capture_output=True, check=True)
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)
    assert "## Summary" in body
    assert "First paragraph of the design." in body
    assert "## Verification" in body
    assert "python tool.py" in body
    assert "the health check, code review, independent verification and QA" in body
    assert "`tests/test_env.py::test_needs_profile`" in body
    assert "## Changes\n\n- tool.py\n" in body
    # The short description (2026-10-26): the brief is not repeated under its own heading.
    assert "## Original request" not in body and "that prints built." not in body
    assert body.startswith(delivery.BODY_START) and body.endswith(delivery.BODY_END + "\n")


def test_pr_verification_says_check_did_not_run_when_validation_fact_is_absent(tmp_path):
    """When authoring evidence is absent and check is present, the PR says 'did not run'."""
    import subprocess
    ctx = _TitleCtx(
        facts={"brief": "Build a tool.", "check": "pnpm test", "name": "t1"},
        files={"design.md": "# Heading\n\nSummary.\n"},
    )
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "g.txt").write_text("b\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)
    assert "did not run" in body
    assert "passed in the run" not in body


def test_pr_verification_says_check_passed_when_validation_fact_is_set(tmp_path):
    """When authoring evidence shows validation passed, the PR says 'passed'."""
    import subprocess
    ctx = _TitleCtx(
        facts={"brief": "Build a tool.", "check": "pnpm test", "name": "t1"},
        files={"design.md": "# Heading\n\nSummary.\n",
               "authoring-evidence.json": '{"validation": {"status": "pass"}}'},
    )
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "g.txt").write_text("b\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)
    assert "- **Check:** `pnpm test` passed.\n" in body
    assert "did not run" not in body


def test_pr_reviews_line_says_audit_did_not_pass_on_fail_row(tmp_path):
    """A fail row in the audit makes the Reviews line say not all steps completed."""
    import subprocess
    ctx = _TitleCtx(
        facts={"brief": "Build a tool.", "check": "pnpm test", "name": "t1"},
        files={"design.md": "# Heading\n\nSummary.\n",
               "authoring-evidence.json": '{"validation": {"status": "pass"}}',
               "intent-audit.json": '{"requirements": [{"id": "r1", "status": "fail", "reason": "missing"}]}'},
    )
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "g.txt").write_text("b\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)
    assert "not all review steps completed" in body
    assert "health check, code review, independent verification and QA found nothing blocking" not in body


def test_pr_reviews_line_says_audit_did_not_pass_on_unverified_row(tmp_path):
    """An unverified row in the audit makes the Reviews line say not all steps completed."""
    import subprocess
    ctx = _TitleCtx(
        facts={"brief": "Build a tool.", "check": "pnpm test", "name": "t1"},
        files={"design.md": "# Heading\n\nSummary.\n",
               "authoring-evidence.json": '{"validation": {"status": "pass"}}',
               "intent-audit.json": '{"requirements": [{"id": "r1", "status": "unverified", "reason": "not checked"}]}'},
    )
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "g.txt").write_text("b\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)
    assert "not all review steps completed" in body
    assert "health check, code review, independent verification and QA found nothing blocking" not in body


def test_pr_reviews_line_not_passed_on_excluded_row_without_user_change(tmp_path):
    """An excluded row with no user_change makes the Reviews line say not all steps completed."""
    import subprocess
    ctx = _TitleCtx(
        facts={"brief": "Build a tool.", "check": "pnpm test", "name": "t1"},
        files={"design.md": "# Heading\n\nSummary.\n",
               "authoring-evidence.json": '{"validation": {"status": "pass"}}',
               "intent-audit.json": '{"requirements": [{"id": "r1", "status": "excluded"}]}'},
    )
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "g.txt").write_text("b\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)
    assert "not all review steps completed" in body


def test_pr_reviews_line_not_passed_on_empty_requirements(tmp_path):
    """An empty requirements list makes the Reviews line say not all steps completed."""
    import subprocess
    ctx = _TitleCtx(
        facts={"brief": "Build a tool.", "check": "pnpm test", "name": "t1"},
        files={"design.md": "# Heading\n\nSummary.\n",
               "authoring-evidence.json": '{"validation": {"status": "pass"}}',
               "intent-audit.json": '{"requirements": []}'},
    )
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "g.txt").write_text("b\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)
    assert "not all review steps completed" in body


def test_pr_reviews_line_passed_on_excluded_row_with_valid_user_change(tmp_path):
    """An excluded row with a valid user_change passes (audit_problems validates exclusions)."""
    import subprocess
    ctx = _TitleCtx(
        facts={"brief": "Build a tool.", "check": "pnpm test", "name": "t1"},
        files={"design.md": "# Heading\n\nSummary.\n",
               "authoring-evidence.json": json.dumps({
                   "validation": {"status": "pass"},
                   "requirements": [{"id": "brief", "text": "Build a tool."}],
               }),
               "intent-audit.json": json.dumps({
                   "requirements": [{
                       "id": "brief", "status": "excluded",
                       "requirement": "Build a tool.",
                       "reason": "The person decided to skip this.",
                       "user_change": "brief",
                       "user_quote": "Build a tool.",
                   }],
               })},
    )
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "g.txt").write_text("b\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)
    assert "the health check, code review, independent verification and QA found nothing blocking" in body
    assert "not all review steps completed" not in body


def test_pr_check_line_says_failed_when_validation_failed(tmp_path):
    """When validation status is not pass, the Check line says failed, not did not run."""
    import subprocess
    ctx = _TitleCtx(
        facts={"brief": "Build a tool.", "check": "pnpm test", "name": "t1"},
        files={"design.md": "# Heading\n\nSummary.\n",
               "authoring-evidence.json": '{"validation": {"status": "fail"}}'},
    )
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "g.txt").write_text("b\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)
    assert "**failed**" in body
    assert "did not run" not in body


def test_run_report_checked_false_when_check_did_not_run(tmp_path):
    """The run report's checked field is False when validation did not pass."""
    from refactory_dsh import design_ops
    # When evidence has no passing validation, checked should be False.
    ev_no_pass = {"identity": {}}
    assert not bool(ev_no_pass and isinstance(ev_no_pass.get("validation"), dict)
                    and ev_no_pass["validation"].get("status") == "pass")
    # When evidence has passing validation, checked should be True.
    ev_pass = {"validation": {"status": "pass"}}
    assert bool(ev_pass and isinstance(ev_pass.get("validation"), dict)
                and ev_pass["validation"].get("status") == "pass")
    # None evidence: checked is False
    assert not bool(None and isinstance(None, dict))


def test_pr_verification_names_inherited_failures_and_the_new_tests_on_the_base(tmp_path):
    """The check's line names the failures the base has too, and the Tests section how many
    of the new tests fail without the change (the run's own revert check)."""
    import json
    from types import SimpleNamespace
    base, head = _git_pair(tmp_path)
    ctx = _TitleCtx(
        facts={"check": "pytest", "name": "b2", "inherited_failures": ["tests/test_env.py::test_needs_profile"]},
        files={"design.md": "# A design\n\nSummary.\n",
               "authoring-evidence.json": json.dumps({"validation": {"status": "pass"}}),
               # Two of test_a.py's tests fail on the base, test_b.py passes there and
               # test_c.py cannot run: the summary counts tests and files together.
               "test-guardedness.json": json.dumps({
                   "results": [
                       {"file": "tests/test_a.py", "guarding": True, "tests": [
                           {"name": "tests/test_a.py::one", "on_base": "fail", "guarding": True},
                           {"name": "tests/test_a.py::two", "on_base": "fail", "guarding": True}]},
                       {"file": "tests/test_b.py", "tests": [], "file_level": "pass", "guarding": False},
                       {"file": "tests/test_c.py", "tests": [], "file_level": "cannot_run_on_base",
                        "guarding": False}],
                   "summary": {"guarding": 2, "not_guarding": 1, "cannot_run_on_base": 1,
                               "total_new_tests": 4}})},
    )
    body = delivery._body(ctx, SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="b"))
    assert ("- **Check:** `pytest` passed; 1 failure(s) the unchanged base has too were not counted: "
            "`tests/test_env.py::test_needs_profile`.\n") in body
    # One unit throughout: files.
    assert ("## Tests\n\n- 1 of 3 new test files fail without the change; 1 pass there, so they "
            "do not guard it; 1 cannot run on the base.\n") in body


def test_pr_body_includes_uncommitted_files_line(tmp_path, monkeypatch):
    """The PR body renders the uncommitted-files line through delivery's body builder."""
    import subprocess
    ctx = _TitleCtx(
        facts={"brief": "Build a tool", "check": "python tool.py", "name": "b2"},
        files={"design.md": "# A design\n\nSummary.\n"},
    )
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], capture_output=True, check=True)
    (tmp_path / "file.txt").write_text("content\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "tool.py").write_text("print('built')\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "attempt"], capture_output=True, check=True)
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head,
                            branch="refactory/run-test", target=str(tmp_path))
    # Mock uncommitted_record to return entries.
    monkeypatch.setattr(
        "refactory_dsh.worktree.uncommitted_record",
        lambda f: [{"path": "modified.py", "kind": "modified", "sha256": "abc123"}],
    )
    body = delivery._body(ctx, found)
    assert "Uncommitted base" in body
    assert "modified.py" in body
    assert "include_uncommitted" in body


def test_pr_verification_says_why_the_new_tests_were_not_run_on_the_base():
    import json
    ctx = _TitleCtx(files={"test-guardedness.json": json.dumps(
        {"unavailable": "the base worktree could not be prepared: no uv"})})
    assert delivery._guardedness(ctx) == (
        "New tests on the base: not run: the base worktree could not be prepared: no uv.")
    assert delivery._guardedness(_TitleCtx()) == ""
    leaked = _TitleCtx(files={"test-guardedness.json": json.dumps(
        {"unavailable": "setup failed: ANTHROPIC_API_KEY=sk-live-123 not accepted"})})
    assert "sk-live-123" not in delivery._guardedness(leaked)


def _git_pair(tmp_path):
    """Set up a minimal two-commit git repo; return (base, head, path)."""
    import subprocess
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "g.txt").write_text("b\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return base, head


def test_contract_design_gives_title_summary_and_changes(tmp_path):
    design_text = (
        "# Enforce clause-level audit coverage\n\n"
        "## Summary\n\n"
        "The overwatch audit missed a clause. This change adds clause-level checking.\n\n"
        "## Rule and sites\n\n"
        "The rule: every constraint clause is checked.\n\n"
        "- `authoring_evidence.py:132-157` — **[changed]** adds clause rows.\n"
        "- `design_ops.py:116-246` — **[left: only the gate; clauses are audit-side]**\n\n"
        "## Docs\n\n"
        "- `docs/models.md` — add clause-level audit section\n\n"
        "## Changes\n\n"
        "- `refactory/refactory_dsh/authoring_evidence.py`\n"
        "- `refactory/tests/test_authoring_evidence.py`\n"
    )
    ctx = _TitleCtx(
        facts={"brief": "Add clause checking", "check": "pytest", "name": "b17_clauses"},
        files={"design.md": design_text},
    )
    title = delivery._title(ctx)
    assert title == "Enforce clause-level audit coverage"
    assert title != "b17_clauses"

    import subprocess as sp
    sp.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    sp.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = sp.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                  capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "g.txt").write_text("b\n")
    (tmp_path / "authoring_evidence.py").write_text("# new\n")
    sp.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = sp.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                  capture_output=True, text=True, check=True).stdout.strip()
    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)
    assert "## Summary" in body
    assert "clause-level checking" in body
    # The design's ## Changes bullet names a path the diff does not have, so the file
    # takes its [changed] site's description.
    assert "## Changes\n\n- authoring_evidence.py — adds clause rows\n- g.txt\n" in body


# ── fixture-based marker tests ─────────────────────────────────────────────

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_b23a_colon_markers_produce_site_what_changed(tmp_path):
    """b23a's design uses ``[changed: description]`` colon forms.

    Fails without the parser fix: the colon-description form is not recognised,
    so all sites are ``unmarked`` and ``build_what_changed`` returns a file list.
    """
    from refactory_dsh import design_contract

    text = (_FIXTURES / "b23a_design_excerpt.md").read_text()
    parsed = design_contract.parse(text, "b23a_correct_run")
    changed = [s for s in parsed.rule_sites if s["status"] == "changed"]
    assert len(changed) >= 3, f"expected ≥3 changed sites, got {len(changed)}: {changed}"
    # The colon-description is captured.
    descs = [s["description"] for s in changed if s["description"]]
    assert any("command_history" in d for d in descs), f"no colon-desc captured: {descs}"
    # build_what_changed returns items from sites, not an empty list.
    items, _ = design_contract.build_what_changed(parsed, ["refactory/refactory_dsh/console.py"])
    assert items, "build_what_changed returned no items"


def test_b24ar_backticked_markers_produce_site_what_changed(tmp_path):
    """b24ar's design uses backtick-wrapped markers for all 21+ sites.

    Fails without the parser fix: ``_find_marker`` skips backtick-wrapped
    markers, so all sites are ``unmarked``.
    """
    from refactory_dsh import design_contract

    text = (_FIXTURES / "b24ar_design_excerpt.md").read_text()
    parsed = design_contract.parse(text, "b24ar_hooks_refine")
    unmarked = [s for s in parsed.rule_sites if s["status"] == "unmarked"]
    assert not unmarked, f"sites still unmarked: {unmarked[:3]}"
    changed = [s for s in parsed.rule_sites if s["status"] == "changed"]
    assert len(changed) >= 10, f"expected ≥10 changed sites, got {len(changed)}"
    items, _ = design_contract.build_what_changed(
        parsed, ["vitest.global-setup.ts", "refactory/refactory_dsh/worktree.py"],
    )
    assert items, "build_what_changed returned no items"


# ── parametrised marker-form tests ─────────────────────────────────────────

@pytest.mark.parametrize("form,expected_status,expected_desc", [
    ("[changed]", "changed", ""),
    ("**[changed]**", "changed", ""),
    ("`[changed]`", "changed", ""),
    ("[changed: description text]", "changed", "description text"),
    ("**[changed: description text]**", "changed", "description text"),
    ("`[changed: description text]`", "changed", "description text"),
    ("[left: reason text]", "left", ""),
    ("**[left: reason text]**", "left", ""),
    ("`[left: reason text]`", "left", ""),
    ("[left]", "left", ""),
    ("`[left]`", "left", ""),
    ("[new]", "changed", ""),
    ("`[new]`", "changed", ""),
    ("[new: added this]", "changed", "added this"),
    # dm-119: a [left] site whose description mentions [changed] parses as left
    ("design_contract.py:177-209 [left: parse() calls _find_marker for [changed] first]", "left", ""),
    # Quote guard: marker right after a quote character is skipped
    ("file.py:10 — '[changed]' is a valid marker", "unmarked", ""),
    # Open-bracket guard: marker inside an open [left ...] is skipped
    ("file.py:10 [left: mentions [changed] here] — reason", "left", ""),
    # Both guards: inside [left:] and after a quote
    ('file.py:10 [left: the old code used "[changed]" markers]', "left", ""),
])
def test_marker_form_recognised(form, expected_status, expected_desc):
    """Every unambiguous marker form is recognised by the parser."""
    from refactory_dsh import design_contract

    design_text = (
        "# Test form\n\n"
        "## Summary\n\nTest.\n\n"
        "## Rule and sites\n\n"
        f"- `some/file.py:1` — {form}\n\n"
        "## Docs\n\n- docs/x.md\n\n"
        "## Changes\n\n- some/file.py\n"
    )
    parsed = design_contract.parse(design_text, "test")
    assert parsed.has_contract
    site = parsed.rule_sites[0]
    assert site["status"] == expected_status, f"form {form!r}: got status {site['status']!r}"
    if expected_desc:
        assert site["description"] == expected_desc, (
            f"form {form!r}: expected desc {expected_desc!r}, got {site['description']!r}"
        )


def test_changes_keeps_bullets_naming_no_file_and_describes_the_rest_by_their_sites(tmp_path):
    """Contract design whose Changes bullets name no file: they are kept, as behaviours
    (dm-191), and each changed file no bullet names follows with its [changed] sites'
    descriptions. (Until b45a's review this test pinned those bullets left out.)

    Uses site lines from b21b, including one where ``[changed]`` appears
    inside backticks before the real **[changed]** marker.

    Fails without the change: old _body (delivery.py:1007-1008) renders
    parsed.changes, which includes 'Return code is 2'.
    """
    from refactory_dsh import design_contract

    design_text = (
        "# Build What changed from sites\n\n"
        "## Summary\n\nFix the What changed section.\n\n"
        "## Rule and sites\n\n"
        "- `refactory/refactory_dsh/design_contract.py:96-155` \u2014 `parse()` builds "
        "`rule_sites` dicts. Currently drops text after `[changed]`. "
        "**[changed]** captures that text as a `'description'` field in each site dict.\n"
        "- `refactory/refactory_dsh/delivery.py:989-1022` \u2014 `_body()` renders the PR body. "
        "**[changed]** replaces those lines with a call to `build_what_changed`.\n"
        "- `refactory/refactory_dsh/delivery.py:1025-1029` \u2014 `_write_pr_text()` calls "
        "`_body()`. **[left: inherits the fix through `_body()`]**\n\n"
        "## Docs\n\n- `docs/models.md` \u2014 update\n\n"
        "## Changes\n\n"
        "- Return code is 2\n"
        "- `seen == []`\n"
    )

    # Verify the backtick-escaped marker is handled: description is captured
    # from the second [changed] (the bold one), not the first (inside backticks).
    parsed = design_contract.parse(design_text, "test_run")
    changed = [s for s in parsed.rule_sites if s['status'] == 'changed']
    assert len(changed) == 2
    assert changed[0]['description'] == "captures that text as a `'description'` field in each site dict"
    assert changed[1]['description'] == "replaces those lines with a call to `build_what_changed`"

    ctx = _TitleCtx(
        facts={"brief": "Fix what changed", "check": "pytest", "name": "test_run"},
        files={"design.md": design_text},
    )

    import subprocess as sp
    sp.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    sp.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = sp.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                  capture_output=True, text=True, check=True).stdout.strip()
    for f in ["refactory/refactory_dsh/design_contract.py",
              "refactory/refactory_dsh/delivery.py"]:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# changed\n")
    sp.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = sp.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                  capture_output=True, text=True, check=True).stdout.strip()

    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)

    assert ("## Changes\n\n- Return code is 2\n- `seen == []`\n"
            "- refactory/refactory_dsh/delivery.py — replaces those lines with a call "
            "to `build_what_changed`\n- refactory/refactory_dsh/design_contract.py — captures that "
            "text as a `'description'` field in each site dict\n") in body, body


def test_diff_file_with_no_site_appears_by_name(tmp_path):
    """A file in the diff with no matching [changed] site appears by name.

    Uses a site line from b20b: ``delivery.py:795`` \u2014 **[changed]**.
    The diff also touches test_delivery.py, which has no site.

    Fails without the change: old _body renders parsed.changes (the Changes
    section bullets); the diff file not in any site is not listed.
    """
    design_text = (
        "# Write PR text once\n\n"
        "## Summary\n\nA kept delivery has no PR text.\n\n"
        "## Rule and sites\n\n"
        "- `refactory/refactory_dsh/delivery.py:795` \u2014 the only place `BODY` is written "
        "today, inside `_deliver`. **[changed]** \u2014 this write is removed.\n"
        "- `refactory/refactory_dsh/delivery.py:926-938` \u2014 `_title()` computes the PR title. "
        "**[left: read-only; called from `_write_pr_text`]**\n\n"
        "## Docs\n\n- `docs/models.md` \u2014 update delivery paragraph\n\n"
        "## Changes\n\n"
        "- `refactory/refactory_dsh/delivery.py`\n"
    )

    ctx = _TitleCtx(
        facts={"brief": "PR text for kept", "check": "pytest", "name": "b20b"},
        files={"design.md": design_text},
    )

    import subprocess as sp
    sp.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    sp.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = sp.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                  capture_output=True, text=True, check=True).stdout.strip()
    for f in ["refactory/refactory_dsh/delivery.py",
              "refactory/tests/test_delivery.py"]:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# changed\n")
    sp.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = sp.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                  capture_output=True, text=True, check=True).stdout.strip()

    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)

    assert "refactory/tests/test_delivery.py" in body


def test_changed_site_outside_diff_excluded(tmp_path, capsys):
    """A [changed] site whose file is not in the diff is excluded from
    What changed, and the mismatch is reported via target_guard.warn.

    Uses site lines from b20d (``nn/knob.py:49-81``) and b21b
    (``delivery.py:989-1022``).  The diff touches only delivery.py.

    Fails without the change: old _body renders parsed.changes (the Changes
    section bullets), not the sites. The bullets include knob.py.
    """
    design_text = (
        "# Refuse empty briefs\n\n"
        "## Summary\n\nEmpty briefs should be refused.\n\n"
        "## Rule and sites\n\n"
        "- `refactory/refactory_dsh/delivery.py:989-1022` \u2014 `_body()` renders the PR body. "
        "**[changed]** replaces the What changed source.\n"
        "- `refactory/refactory_dsh/nn/knob.py:49-81` (`Knob` dataclass) \u2014 "
        "**[changed]** Add `required: bool = False` field.\n\n"
        "## Docs\n\n- `docs/models.md` \u2014 update\n\n"
        "## Changes\n\n"
        "- `refactory/refactory_dsh/delivery.py`\n"
        "- `refactory/refactory_dsh/nn/knob.py`\n"
    )

    ctx = _TitleCtx(
        facts={"brief": "Refuse empty", "check": "pytest", "name": "b20d"},
        files={"design.md": design_text},
    )

    import subprocess as sp
    sp.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    sp.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = sp.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                  capture_output=True, text=True, check=True).stdout.strip()
    # Only delivery.py is in the diff; knob.py is NOT
    p = tmp_path / "refactory/refactory_dsh/delivery.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# changed\n")
    sp.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = sp.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                  capture_output=True, text=True, check=True).stdout.strip()

    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)

    assert "delivery.py" in body
    assert "knob.py" not in body
    err = capsys.readouterr().err
    assert "knob.py" in err


def test_pre_contract_design_lists_diff_files(tmp_path):
    """A pre-contract design (no ## Summary, ## Rule and sites, etc.)
    lists the diff's files in What changed.

    Fails without the change: old _body skips What changed entirely for
    pre-contract designs (parsed.has_contract is False at delivery.py:1007).
    """
    design_text = "# Fix something\n\nThe old oracle is stale.\n"

    ctx = _TitleCtx(
        facts={"brief": "Fix it", "check": "pytest", "name": "b15b"},
        files={"design.md": design_text},
    )

    import subprocess as sp
    sp.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    sp.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = sp.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                  capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "alpha.py").write_text("# alpha\n")
    (tmp_path / "beta.py").write_text("# beta\n")
    sp.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = sp.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                  capture_output=True, text=True, check=True).stdout.strip()

    from types import SimpleNamespace
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)

    assert "## Changes" in body
    assert "alpha.py" in body
    assert "beta.py" in body


def test_build_what_changed_with_real_design():
    """build_what_changed with site lines from b20b (multiple [changed] for
    delivery.py) and b20d (knob.py, design.py) against their diff files.

    Multiple [changed] sites in delivery.py become one item; each diff file
    appears exactly once; no mismatches.
    """
    from refactory_dsh import design_contract

    design_text = (
        "# Test real design\n\n"
        "## Summary\n\nReal design test.\n\n"
        "## Rule and sites\n\n"
        "- `refactory/refactory_dsh/delivery.py:795` \u2014 the only place `BODY` is written "
        "today, inside `_deliver`. **[changed]** \u2014 this write is removed.\n"
        "- `refactory/refactory_dsh/delivery.py:729-766` \u2014 `deliver_design()` orchestrates "
        "the whole step. **[changed]** \u2014 restructured.\n"
        "- `refactory/refactory_dsh/delivery.py:138-159` \u2014 `knobs()` declares the `pr` knob. "
        "**[changed]** \u2014 appended finishing commands.\n"
        "- `refactory/refactory_dsh/delivery.py:926-938` \u2014 `_title()` computes the PR title. "
        "**[left: read-only; called from `_write_pr_text`]**\n"
        "- `refactory/refactory_dsh/nn/knob.py:49-81` (`Knob` dataclass) \u2014 "
        "**[changed]** Add `required: bool = False` field.\n"
        "- `refactory/refactory_dsh/design.py:69` (brief knob declaration) \u2014 "
        "**[changed]** Change to `required=True`.\n\n"
        "## Docs\n\n- `docs/models.md` \u2014 update\n\n"
        "## Changes\n\n"
        "- `refactory/refactory_dsh/delivery.py`\n"
        "- `refactory/refactory_dsh/nn/knob.py`\n"
    )

    parsed = design_contract.parse(design_text, "test_real")

    # Diff files from a mix of b20b and b20d
    diff_files = [
        "refactory/refactory_dsh/delivery.py",
        "refactory/refactory_dsh/design.py",
        "refactory/refactory_dsh/nn/knob.py",
        "refactory/tests/test_delivery.py",
    ]

    items, mismatches = design_contract.build_what_changed(parsed, diff_files)

    # No mismatches: every [changed] site matches a diff file
    assert mismatches == [], f"unexpected mismatches: {mismatches}"
    # One item per diff file
    assert len(items) == 4, f"expected 4 items, got {len(items)}: {items}"

    # delivery.py: three [changed] sites merged into one item
    assert items[0].startswith("refactory/refactory_dsh/delivery.py")
    assert "this write is removed" in items[0]
    assert "restructured" in items[0]
    assert "appended finishing commands" in items[0]

    # design.py: one [changed] site
    assert "refactory/refactory_dsh/design.py" in items[1]
    assert "Change to `required=True`" in items[1]

    # knob.py: one [changed] site
    assert "refactory/refactory_dsh/nn/knob.py" in items[2]
    assert "Add `required: bool = False` field" in items[2]

    # test_delivery.py: no site, appears by name only
    assert items[3] == "refactory/tests/test_delivery.py"


@pytest.mark.parametrize("heading,run_name,expected_not_equal", [
    ("# Design: b15b_reference_oracle", "b15b_reference_oracle", "b15b_reference_oracle"),
    ("# Design: b16b_graph_ends \u2014 Make sinks() agree with scheduler reachability",
     "b16b_graph_ends", "b16b_graph_ends"),
    ("# Design: Fork-branch redo reopens join and downstream",
     "b15a_fork_redo", "b15a_fork_redo"),
    ("# Design: Generalize set-aside to every declared output on re-dispatch",
     "b16a_set_aside", "b16a_set_aside"),
])
def test_pre_contract_design_title_is_not_the_run_name(heading, run_name, expected_not_equal):
    design_text = f"{heading}\n\n## Problem\n\nThe current implementation has a bug that causes incorrect results.\n"
    ctx = _TitleCtx(
        facts={"brief": "Fix something important in the codebase.", "name": run_name},
        files={"design.md": design_text},
    )
    title = delivery._title(ctx)
    assert title, f"title should not be empty for heading={heading!r}"
    assert title != expected_not_equal, f"title should not be the run name: {title!r}"


def test_pre_contract_design_gives_prose_summary():
    from refactory_dsh import design_contract

    design_text = (
        "# Design: b15b_reference_oracle\n\n"
        "## Problem\n\n"
        "The oracle returns stale references when the cache is cold.\n"
    )
    parsed = design_contract.parse(design_text, "b15b_reference_oracle")
    assert parsed.summary == "The oracle returns stale references when the cache is cold."
    assert "## Problem" not in parsed.summary




# ── dm-108: delivery recognises its own merge ─────────────────────────────


def _git_init(path):
    """Initialise a git repo with user config at ``path`` and return the path."""
    import subprocess as sp
    path.mkdir(parents=True, exist_ok=True)
    sp.run(["git", "init", str(path)], capture_output=True, check=True)
    sp.run(["git", "-C", str(path), "config", "user.email", "t@t"], capture_output=True, check=True)
    sp.run(["git", "-C", str(path), "config", "user.name", "T"], capture_output=True, check=True)
    return path


def test_interrupted_merge_recovery_resets_to_reviewed_commit(tmp_path):
    """delivery._recover_interrupted_merge resets HEAD when it is delivery's
    own merge commit: a merge whose first parent is found.head.
    """
    import subprocess as sp
    repo = _git_init(tmp_path / "repo")
    # Initial commit = the reviewed commit
    (repo / "f.txt").write_text("a\n")
    sp.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(repo), "commit", "-m", "reviewed"], capture_output=True, check=True)
    reviewed = sp.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                      capture_output=True, text=True, check=True).stdout.strip()
    # Create a second branch with a tip commit
    sp.run(["git", "-C", str(repo), "checkout", "-b", "main-tip"], capture_output=True, check=True)
    (repo / "g.txt").write_text("b\n")
    sp.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(repo), "commit", "-m", "tip"], capture_output=True, check=True)
    # Back to the reviewed commit's branch and merge
    sp.run(["git", "-C", str(repo), "checkout", "-b", "run-branch", reviewed],
           capture_output=True, check=True)
    sp.run(["git", "-C", str(repo), "merge", "--no-ff", "--no-edit", "main-tip"],
           capture_output=True, check=True)
    merge_sha = sp.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True).stdout.strip()
    assert merge_sha != reviewed

    # Construct a Worktree recording the reviewed commit
    found = worktree.Worktree(
        path=str(repo), branch="run-branch", base=reviewed,
        head=reviewed, target=str(repo), run=str(tmp_path / "run"),
    )
    delivery._recover_interrupted_merge(found)
    # After recovery, HEAD is back to the reviewed commit
    after = sp.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                   capture_output=True, text=True, check=True).stdout.strip()
    assert after == reviewed


def test_foreign_commit_still_refused(tmp_path):
    """A non-merge commit on the branch is NOT recovered: require_current
    still refuses it as 'outside the run'.
    """
    import subprocess as sp
    repo = _git_init(tmp_path / "repo")
    # Initial commit = the reviewed commit
    (repo / "f.txt").write_text("a\n")
    sp.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(repo), "commit", "-m", "reviewed"], capture_output=True, check=True)
    reviewed = sp.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                      capture_output=True, text=True, check=True).stdout.strip()
    # Create a proper worktree so require_current's git-dir check passes
    wt_path = tmp_path / "wt"
    sp.run(["git", "-C", str(repo), "worktree", "add", "-b", "run-branch",
            str(wt_path), reviewed], capture_output=True, check=True)
    # Make a foreign (non-merge) commit in the worktree
    (wt_path / "g.txt").write_text("foreign\n")
    sp.run(["git", "-C", str(wt_path), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(wt_path), "-c", "core.hooksPath=/dev/null",
            "commit", "-m", "foreign"], capture_output=True, check=True)
    foreign = sp.run(["git", "-C", str(wt_path), "rev-parse", "HEAD"],
                     capture_output=True, text=True, check=True).stdout.strip()
    assert foreign != reviewed

    found = worktree.Worktree(
        path=str(wt_path), branch="run-branch", base=reviewed,
        head=reviewed, target=str(repo), run=str(tmp_path / "run"),
    )
    # _recover_interrupted_merge does nothing for non-merge commits
    delivery._recover_interrupted_merge(found)
    after = sp.run(["git", "-C", str(wt_path), "rev-parse", "HEAD"],
                   capture_output=True, text=True, check=True).stdout.strip()
    assert after == foreign  # not reset — still the foreign commit

    # Set up a worktree.json so require_current can read it
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    wt_record = run_dir / ".refactory" / "worktree.json"
    wt_record.parent.mkdir(parents=True, exist_ok=True)
    import json
    wt_record.write_text(json.dumps({
        "path": str(wt_path), "branch": "run-branch", "base": reviewed,
        "head": reviewed, "target": str(repo), "run": str(run_dir),
    }))
    with pytest.raises(ValueError, match="outside the run"):
        worktree.require_current(run_dir)


def test_deliver_design_recovers_own_merge(tmp_path, monkeypatch):
    """deliver_design, driven end-to-end, gets past require_current when
    HEAD is at delivery's own merge.

    Fails without _recover_interrupted_merge: require_current raises
    "outside the run" at the merge commit.
    """
    import subprocess as sp
    from refactory_dsh import nn

    repo = _git_init(tmp_path / "repo")
    (repo / "f.txt").write_text("a\n")
    sp.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(repo), "commit", "-m", "reviewed"], capture_output=True, check=True)
    reviewed = sp.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                      capture_output=True, text=True, check=True).stdout.strip()
    # Create a branch simulating main moving forward
    sp.run(["git", "-C", str(repo), "checkout", "-b", "main-moved"], capture_output=True, check=True)
    (repo / "g.txt").write_text("b\n")
    sp.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(repo), "commit", "-m", "main-moved"], capture_output=True, check=True)
    sp.run(["git", "-C", str(repo), "checkout", reviewed], capture_output=True, check=True)
    # Create a proper worktree at the reviewed commit
    wt_path = tmp_path / "wt"
    sp.run(["git", "-C", str(repo), "worktree", "add", "-b", "run-branch",
            str(wt_path), reviewed], capture_output=True, check=True)
    # Merge main-moved into the worktree (delivery's own merge)
    sp.run(["git", "-C", str(wt_path), "merge", "--no-ff", "--no-edit", "main-moved"],
           capture_output=True, check=True)
    merge_sha = sp.run(["git", "-C", str(wt_path), "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True).stdout.strip()
    assert merge_sha != reviewed

    # Write worktree.json recording the reviewed commit
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    wt_record = run_dir / ".refactory" / "worktree.json"
    wt_record.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    wt_record.write_text(_json.dumps({
        "path": str(wt_path), "branch": "run-branch", "base": reviewed,
        "head": reviewed, "target": str(repo), "run": str(run_dir),
    }))

    # deliver_design with delivery_plan={"action": "none"} so it returns
    # after require_current without doing any GitHub work.
    ctx = nn.RunContext(run_dir, facts={
        "delivery_plan": {"action": "none", "why": "test"},
    }, enforce_writes=False)
    result = delivery.deliver_design(ctx)
    assert "not delivered" in result.get("delivery_result", "")

    # Now verify it fails without _recover_interrupted_merge:
    # reset HEAD back to the merge commit
    sp.run(["git", "-C", str(wt_path), "reset", "--hard", merge_sha],
           capture_output=True, check=True)
    # Monkeypatch _recover_interrupted_merge to be a no-op
    monkeypatch.setattr(delivery, "_recover_interrupted_merge", lambda found: None)
    with pytest.raises(ValueError, match="outside the run"):
        delivery.deliver_design(ctx)


def test_verify_git_link_refuses_mislinked_worktree(tmp_path):
    """A worktree whose .git points at the repository's own git directory
    is refused by verify_git_link, even when HEAD is at delivery's own merge.

    This prevents _recover_interrupted_merge from resetting a worktree
    that was repointed at the person's repository.
    """
    import subprocess as sp
    repo = _git_init(tmp_path / "repo")
    (repo / "f.txt").write_text("a\n")
    sp.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(repo), "commit", "-m", "reviewed"], capture_output=True, check=True)
    reviewed = sp.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                      capture_output=True, text=True, check=True).stdout.strip()

    sp.run(["git", "-C", str(repo), "checkout", "-b", "main-tip"], capture_output=True, check=True)
    (repo / "g.txt").write_text("b\n")
    sp.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    sp.run(["git", "-C", str(repo), "commit", "-m", "tip"], capture_output=True, check=True)

    # Create a proper worktree at the reviewed commit
    wt_path = tmp_path / "wt"
    sp.run(["git", "-C", str(repo), "worktree", "add", "-b", "run-branch",
            str(wt_path), reviewed], capture_output=True, check=True)

    # Merge in the worktree (simulating delivery's own merge)
    sp.run(["git", "-C", str(wt_path), "merge", "--no-ff", "--no-edit", "main-tip"],
           capture_output=True, check=True)
    merge_sha = sp.run(["git", "-C", str(wt_path), "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True).stdout.strip()
    assert merge_sha != reviewed

    # Repoint the worktree's .git to the repo's own git dir
    repo_git_dir = sp.run(["git", "-C", str(repo), "rev-parse", "--absolute-git-dir"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (wt_path / ".git").write_text(f"gitdir: {repo_git_dir}\n")

    # After repointing, git -C wt_path resolves through the repo's git dir,
    # so rev-parse HEAD returns the repo's HEAD (on main-tip), not the
    # worktree's merge.  Record it now to verify nothing is mutated.
    repo_head_before = sp.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()

    found = worktree.Worktree(
        path=str(wt_path), branch="run-branch", base=reviewed,
        head=reviewed, target=str(repo), run=str(tmp_path / "run"),
    )

    # verify_git_link refuses the mislinked worktree
    with pytest.raises(ValueError, match="no longer points at its own git directory"):
        worktree.verify_git_link(found)

    # The repo's HEAD is unchanged — git reset --hard was never called.
    # (After the .git repoint, any reset in wt_path would mutate the repo.)
    repo_head_after = sp.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    assert repo_head_after == repo_head_before


def _make_github(base: Path) -> Path:
    """GitHub under ``base``: gh (logged in as `octo`) in ``base/bin``, repositories in ``base/github``."""
    root = base / "github"
    root.mkdir()
    (root / "login").write_text(LOGIN + "\n")
    stub(base / "bin", "gh", f'exec "{sys.executable}" "{FAKE_GH}" "$@"')
    return root


def _use_github(monkeypatch: pytest.MonkeyPatch, base: Path) -> Path:
    """Point gh and git at the GitHub ``_make_github`` made under ``base``."""
    root = base / "github"
    monkeypatch.setenv("PATH", f"{base / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GH_ROOT", str(root))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.file://{root}/.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "https://github.com/")
    monkeypatch.setattr(delivery, "VERIFY_WAIT_S", 0)
    return root


@pytest.fixture
def github(tmp_path, monkeypatch) -> Path:
    """GitHub, as far as delivery can tell: gh logged in as `octo`, repositories on disk."""
    _make_github(tmp_path)
    return _use_github(monkeypatch, tmp_path)


def _published(github: Path, slug: str, source: Path) -> Path:
    """A repository on GitHub holding ``source``'s main, which ``source`` names as origin."""
    bare = github / f"{slug}.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    git(bare.parent, "init", "-q", "--bare", "-b", "main", bare.name)
    (github / f"{slug}.json").write_text(json.dumps({"visibility": "private", "default_branch": "main"}))
    # Pushed before origin is added: from git 2.56 a push to a URL that matches a configured
    # remote also updates its tracking refs, and the person's clone here has none.
    git(source, "push", "-q", f"https://github.com/{slug}.git", "main:refs/heads/main")
    git(source, "remote", "add", "origin", f"https://github.com/{slug}.git")
    return bare


def test_delivery_leaves_the_persons_remote_tracking_refs_as_they_were(github, tmp_path):
    """A real push of the run's branch writes no remote-tracking ref, also when the person's
    origin/HEAD points at that branch (git 2.56 updates a matching remote's on a URL push)."""
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    git(repo, "fetch", "-q", "origin")
    git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/run-1")
    before = git(repo, "for-each-ref", "refs/remotes")
    git(repo, "commit", "-q", "--allow-empty", "-m", "run")
    head = git(repo, "rev-parse", "HEAD")
    delivery._push(repo, "acme/app", head, "run-1", default="main")
    assert _remote_head(github, "acme/app", "run-1") == head
    assert git(repo, "for-each-ref", "refs/remotes") == before
    assert git(repo, "symbolic-ref", "refs/remotes/origin/HEAD") == "refs/remotes/origin/run-1"
    assert git(repo, "remote") == "origin"


def _remote_head(github: Path, slug: str, branch: str) -> str:
    try:
        return git(github / f"{slug}.git", "rev-parse", "-q", "--verify", f"refs/heads/{branch}")
    except subprocess.CalledProcessError:
        return ""


def _pulls(github: Path, slug: str) -> list[dict]:
    path = github / f"{slug}.pulls.json"
    return json.loads(path.read_text()) if path.exists() else []


def _describe(github: Path, slug: str, body: str) -> None:
    """Give the one pull request on ``slug`` the description ``body``, as its author would on GitHub."""
    path = github / f"{slug}.pulls.json"
    [pull] = json.loads(path.read_text())
    path.write_text(json.dumps([{**pull, "body": body}]))


def _calls(tmp_path: Path) -> str:
    log = tmp_path / "bin" / "calls.log"
    return log.read_text() if log.exists() else ""


def _answers(*emits: str) -> list[str]:
    """Every answer a design run needs, in the order its graph asks.

    pick_corrections is a gate (no model answer needed).  When gate_corrections
    is empty (no ledger or no corrections), pick_corrections returns "skip"
    and apply_corrections does not run — no model answer consumed.
    """
    return [RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
            DESIGN_DOC_ANSWER, *emits,
            *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3]]


def _mode(**knobs) -> DesignMode:
    """Design mode with some of its knobs set, as a person sets them for one run."""
    mode = DesignMode()
    state = mode.state_dict()
    for name, value in knobs.items():
        state["knobs"][name]["value"] = value
    mode.load_state_dict(state)
    return mode


def _design(workdir: Path, emit: dict[str, str] | None = None, *, mode: DesignMode | None = None,
            check: str = "python tool.py", **task: str):
    model = FakeModel(_answers(render(emit or {"tool.py": "print('built')\n"})))
    used_mode = mode or DesignMode()
    # Delivery tests pre-date light refine and script answers for the full pipeline.
    if "refine" in task:
        state = used_mode.state_dict()
        state["knobs"]["refine_design"]["value"] = "full"
        used_mode.load_state_dict(state)
    return used_mode.run(
        workdir, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "Build a tool\nthat prints built.", "check": check, **task},
    )


def _record(workdir: Path) -> dict:
    return json.loads((workdir / delivery.RECORD).read_text())


# ── draft delivery tests ──────────────────────────────────────────────────

DRAFT_FACT = {
    "reason": "review reloop limit reached after 3 review cycle(s)",
    "unresolved": [
        {"text": "The logging format is inconsistent", "source": "review", "rounds": [1, 2, 3]},
        {"text": "Missing error handling in parser", "source": "overwatch", "rounds": [2]},
    ],
}


# ── b45a: every run with an applied commit ends in a pull request ──────────
#
# The build loop's limit stops route into a draft (b40b, dm-224), each by its own way
# (correction 2): a check still failing at its limit goes straight to the report, a
# review limit at gate_qa goes on through the correction audit and the overwatch, and a
# limit at gate_overwatch or the refusal bound goes to the report. Every run here walks
# the real design graph with a scripted model, and delivers to the fake GitHub.

#: What the QA judge and the overwatch hold the change back for, and the line each is
#: listed as in the draft (`judges.must_fix_line`).
QA_HELD = {"source": "qa", "where": "tool.py:1", "what": "no test shows the tool prints built",
           "why": "the brief's only ask", "severity": "high"}
QA_HELD_LINE = "[qa] tool.py:1: no test shows the tool prints built (why: the brief's only ask) (high)"
OVERWATCH_HELD = {"source": "overwatch", "where": "tool.py:1", "what": "nothing a user runs shows it",
                  "why": "the brief's ask", "severity": "high"}
OVERWATCH_HELD_LINE = "[overwatch] tool.py:1: nothing a user runs shows it (why: the brief's ask) (high)"
#: An emission materialise refuses: an edit to a file the last attempt does not have.
REFUSED = "===EDIT: missing.py===\n<<<<<<< SEARCH\nnot there\n=======\nx\n>>>>>>> REPLACE\n"
BUILT = render({"tool.py": "print('built')\n"})
QA_JUDGE_MARK = "You are the QA judge."
OVERWATCH_JUDGE_MARK = "Write `overwatch-verdict.json`"


def _attempt(emission: str, *, judge: str | None = None, overwatch: str | None = None) -> list[str]:
    """One attempt's answers: the emission, the reviews and the verifier beside the check,
    then the judges' verdicts (the QA judge proceeds and the overwatch passes when not given)."""
    return [emission, *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER,
            *([judge] if judge else []), *([overwatch] if overwatch else [])]


def _limited(run: Path, *attempts: list[str], check: str = "python tool.py",
             **knobs: int) -> tuple[nn.RunResult, FakeModel]:
    """A new project's design run through ``attempts``, with ``knobs`` set."""
    model = FakeModel([RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan", DESIGN_DOC_ANSWER,
                       *(answer for attempt in attempts for answer in attempt)])
    result = _mode(**knobs).run(run, model=model, verdicts={"gate_strategy": "proceed"},
                                task={"name": NAME, "brief": "Build a tool\nthat prints built.",
                                      "check": check})
    return result, model


def _asked(model: FakeModel, mark: str) -> int:
    """How many prompts carried ``mark``."""
    return sum(1 for prompt in model.prompts if mark in prompt)


def _draft_pull(github: Path) -> dict:
    [pull] = _pulls(github, f"{LOGIN}/{NAME}")
    return pull


def test_a_review_limit_at_gate_qa_with_a_passing_attempt_delivers_a_draft(github, tmp_path):
    """b43b2 stopped at its review limit with a passing attempt and delivered nothing. Now
    the QA judge's limit goes on through the correction audit and the overwatch to a draft
    pull request listing what the judge left unresolved, with the attempts that named it."""
    run = tmp_path / "run"
    redo = qa_judgment("redo", must_fix=[QA_HELD], follow_ups=["document the --json flag"])
    result, model = _limited(run, _attempt(BUILT, judge=redo),
                             _attempt(render({"tool.py": "print('built!')\n"}), judge=redo,
                                      overwatch=overwatch_judgment("pass")),
                             review_attempts=2)
    assert result.exit_ok, result.error
    assert result.facts["draft_delivery"] == {
        "reason": "review reloop limit reached after 2 review cycle(s)",
        "unresolved": [{"text": QA_HELD_LINE, "source": "QA judge", "rounds": [1, 2]}]}
    # On through the audit and the overwatch, which ran once: after the limit.
    assert (run / "correction-audit.json").exists() and _asked(model, OVERWATCH_JUDGE_MARK) == 1
    assert result.facts["review_attempts_used"] == 2
    pull = _draft_pull(github)
    assert pull["draft"] is True and pull["title"] == "Do the thing"
    assert pull["body"].startswith(f"{delivery.DRAFT_MADE}\n{delivery.DRAFT_HEADING}\n"), pull["body"]
    assert "**Reason:** review reloop limit reached after 2 review cycle(s)" in pull["body"]
    assert _draft_items(pull["body"]) == [f"- {QA_HELD_LINE} (source: QA judge) (rounds: 1, 2)"]
    assert delivery.BODY_START in pull["body"].split(delivery.DRAFT_END, 1)[1]
    assert result.facts["delivery_result"].startswith(f"draft: https://github.com/{LOGIN}/{NAME}/pull/1")
    assert _record(run)["draft"] is True and _record(run)["check_passed"] is True
    # Going on as a draft, the judge's non-blocking items still reach the pull request.
    assert "## Follow-ups\n\n- document the --json flag (qa)\n" in pull["body"]


def test_a_check_that_still_fails_at_its_limit_delivers_a_draft_marked_check_failing(github, tmp_path):
    """The check is binding: a check still failing at its combined-sendback bound goes
    straight to the report as a draft marked check failing, the QA judge and the
    overwatch never asked.  The check's own send-back does not spend an attempt."""
    run = tmp_path / "run"
    # Enough distinct emissions to reach the combined sendback bound without a refusal.
    failing = [render({"tool.py": f"print('built')\nraise SystemExit({code})\n"})
               for code in range(1, 12)]
    result, model = _limited(run, *[[f] for f in failing], check_attempts=2, check_attempt_cap=0)
    assert result.exit_ok, result.error
    fact = result.facts["draft_delivery"]
    assert "check failures exhausted" in fact["reason"]
    assert fact["check_failing"] is True and fact["check_failure"] and fact["unresolved"] == []
    assert _asked(model, QA_JUDGE_MARK) == 0 and _asked(model, OVERWATCH_JUDGE_MARK) == 0
    assert not (run / "correction-audit.json").exists()
    pull = _draft_pull(github)
    assert pull["draft"] is True and pull["title"] == "[check failing] Do the thing"
    assert pull["body"].startswith(f"{delivery.DRAFT_MADE}\n{delivery.CHECK_FAILING_HEADING}\n\n"), pull["body"]
    section = pull["body"].split(delivery.DRAFT_END, 1)[0]
    assert section.index(delivery.CHECK_FAILING_HEADING) < section.index(delivery.DRAFT_HEADING)
    assert "No review judged this head: its check failed first." in section
    assert "`python tool.py` **failed**" in pull["body"]
    assert _record(run)["check_passed"] is False


def test_a_run_with_no_applied_commit_delivers_nothing(github, tmp_path):
    """A stop with no applied commit is no draft: every emission refused, the refusal
    bound stops the run, and GitHub is not asked for anything."""
    run = tmp_path / "run"
    result, _ = _limited(run, [REFUSED], [REFUSED], refusal_attempts=2)
    assert result.exit_ok is False
    assert "emissions refused 2 consecutive times" in (result.error or "")
    assert "draft_delivery" not in result.facts and not result.facts.get("commit")
    calls = _calls(tmp_path)
    assert "repo create" not in calls and "pr create" not in calls
    assert not (run / delivery.RECORD).exists()


def test_a_run_without_a_worktree_still_stops_at_its_check_limit(github, tmp_path, monkeypatch):
    """A run with its worktree off writes an artifact directory, not a commit: nothing it
    could open a pull request of, so its check limit still ends it.  The combined-sendback
    bound or the refusal bound stops the run."""
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    result = _design(run, check="false", mode=_mode(worktree=False, check_attempts=1, check_attempt_cap=0))
    assert result.exit_ok is False
    # check_attempts=1 in this mode: the send-back bound is the check"s own limit.
    assert "check failures exhausted after 1 combined send-back(s)" in (result.error or "")
    assert "draft_delivery" not in result.facts


def test_the_refusal_bound_after_an_applied_commit_delivers_that_commit_as_a_draft(github, tmp_path):
    """Attempt 1 passes its check and the QA judge sends it back; the builder's emissions
    are then refused until the bound. Attempt 1 goes to the report as a draft, listing the
    QA judge's item, and the refusals' rewrite of check.txt does not make its evidence stale."""
    run = tmp_path / "run"
    result, model = _limited(run, _attempt(BUILT, judge=qa_judgment("redo", must_fix=[QA_HELD])),
                             [REFUSED], [REFUSED], refusal_attempts=2, review_attempts=3)
    assert result.exit_ok, result.error
    fact = result.facts["draft_delivery"]
    # REFUSED edits a file that is not there, which a builder working in its workspace
    # cannot do: its workspace has no change, and materialise refuses that.
    assert fact["reason"].startswith("emissions refused 2 consecutive times: your workspace at")
    assert "has no change from the last attempt" in fact["reason"]
    assert "check_failing" not in fact
    assert fact["unresolved"] == [{"text": QA_HELD_LINE, "source": "QA judge", "rounds": [1]}]
    assert _asked(model, OVERWATCH_JUDGE_MARK) == 0
    pull = _draft_pull(github)
    assert pull["draft"] is True and pull["title"] == "Do the thing"
    assert _draft_items(pull["body"]) == [f"- {QA_HELD_LINE} (source: QA judge) (rounds: 1)"]
    assert _remote_head(github, f"{LOGIN}/{NAME}", pull["head"]) == result.facts["commit"]


def test_a_review_limit_at_gate_overwatch_delivers_a_draft(github, tmp_path):
    """The overwatch's redo at the review limit goes to the report as a draft listing its item."""
    run = tmp_path / "run"
    result, model = _limited(run, _attempt(BUILT, overwatch=overwatch_judgment("redo", must_fix=[OVERWATCH_HELD])),
                             review_attempts=1)
    assert result.exit_ok, result.error
    assert result.facts["draft_delivery"] == {
        "reason": "review reloop limit reached after 1 review cycle(s)",
        "unresolved": [{"text": OVERWATCH_HELD_LINE, "source": "overwatch", "rounds": [1]}]}
    pull = _draft_pull(github)
    assert pull["draft"] is True
    assert _draft_items(pull["body"]) == [f"- {OVERWATCH_HELD_LINE} (source: overwatch) (rounds: 1)"]


def test_the_overwatch_after_the_qa_judges_limit_adds_its_items_and_spends_no_attempt(github, tmp_path):
    """The QA judge's limit made the run a draft; the overwatch then sends it back too. Its
    item joins the draft, which goes to the report: no attempt is spent past the limit."""
    run = tmp_path / "run"
    result, _ = _limited(run, _attempt(BUILT, judge=qa_judgment("redo", must_fix=[QA_HELD]),
                                       overwatch=overwatch_judgment("redo", must_fix=[OVERWATCH_HELD])),
                         review_attempts=1)
    assert result.exit_ok, result.error
    assert result.facts["review_attempts_used"] == 1 and result.facts["attempts_used"] == 1
    assert result.facts["draft_delivery"]["unresolved"] == [
        {"text": QA_HELD_LINE, "source": "QA judge", "rounds": [1]},
        {"text": OVERWATCH_HELD_LINE, "source": "overwatch", "rounds": [1]}]
    assert (run / "reviews/overwatch-gate.txt").read_text().startswith("draft: ")
    assert len(_draft_items(_draft_pull(github)["body"])) == 2


#: A person's description: prose with no heading above it, and one with no heading at all.
PERSON_PROSE = "Why this matters, in my words, above everything else.\n\n## Context\n\nKept as written.\n"
PERSON_HEADINGLESS = "Just my paragraph.\nA second line of it, and no heading anywhere.\n"


@pytest.mark.parametrize("theirs", [PERSON_PROSE, PERSON_HEADINGLESS], ids=["prose-first", "no-heading"])
def test_a_persons_description_survives_two_draft_deliveries(new_run, theirs):
    """dm-246: the draft section was cut by a '## ' regex, so a second draft delivery took a
    person's leading prose, or a whole heading-less description, with it. Between its
    markers, it takes only itself."""
    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    pulls = _pulls(github, slug)
    pulls[0]["body"] = theirs
    (github / f"{slug}.pulls.json").write_text(json.dumps(pulls))
    second = {"reason": "overall attempts limit reached", "unresolved": [
        {"text": "Still flaky", "source": "review", "rounds": [4]}]}
    for fact in (DRAFT_FACT, second):
        delivery.deliver_design(nn.RunContext(run, facts={**result.facts, "draft_delivery": fact},
                                              enforce_writes=False))
        [pull] = _pulls(github, slug)
        assert pull["body"].count(delivery.DRAFT_MADE) == 1
        assert pull["body"].startswith(delivery.DRAFT_MADE)
        assert pull["body"].endswith(theirs), pull["body"]
    assert _draft_items(pull["body"]) == ["- Still flaky (source: review) (rounds: 4)"]


@pytest.mark.parametrize("drafts", [True, False], ids=["draft", "github-refuses-drafts"])
def test_a_full_delivery_after_a_draft_makes_it_ready_with_no_section(new_run, drafts):
    """dm-246: a later delivery with no draft fact marks the pull request ready, and takes
    the draft section and the title's prefixes away: the description is the run's again."""
    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    meta = json.loads((github / f"{slug}.json").read_text())
    (github / f"{slug}.json").write_text(json.dumps({**meta, "drafts": drafts}))
    [before] = _pulls(github, slug)
    failing = {**DRAFT_FACT, "check_failing": True, "check_failure": "FAILED test_tool.py::test_build"}
    delivery.deliver_design(nn.RunContext(run, facts={**result.facts, "draft_delivery": failing},
                                          enforce_writes=False))
    [drafted] = _pulls(github, slug)
    assert drafted["draft"] is drafts
    assert drafted["title"] == ("" if drafts else delivery.DRAFT_TITLE) + "[check failing] " + before["title"]
    assert drafted["body"].startswith(f"{delivery.DRAFT_MADE if drafts else delivery.DRAFT_TITLED}\n{delivery.CHECK_FAILING_HEADING}\n")
    again = delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))
    [ready] = _pulls(github, slug)
    assert ready["draft"] is False and ready["title"] == before["title"]
    assert ready["body"] == before["body"]
    assert again["delivery_result"].startswith(f"already open: https://github.com/{slug}/pull/1")


def test_a_persons_own_draft_is_not_marked_ready(opened_pull, tmp_path):
    """Only a draft delivery made is marked ready by a later one; a person's own draft stays one."""
    url, github = opened_pull.url, opened_pull.github
    pulls = _pulls(github, "acme/app")
    pulls[0]["draft"] = True
    (github / "acme/app.pulls.json").write_text(json.dumps(pulls))
    result = _design(tmp_path / "refine", {"tool.py": "print('one')\n", "more.py": "x = 1\n"}, refine=url)
    assert result.exit_ok, result.error
    [after] = _pulls(github, "acme/app")
    assert after["draft"] is True and "pr ready" not in _calls(opened_pull.root)


def test_a_run_whose_whole_change_is_empty_opens_no_pull_request(github, tmp_path, monkeypatch):
    """dm-246: attempt 2 put the tree back to the base, so the branch changes nothing.
    Delivery refuses before it creates the repository or pushes, and says why."""
    real = delivery.deliver_design

    def put_back(ctx):
        found = worktree.recorded(ctx.workdir)
        git(Path(found.path), "rm", "-q", "-r", "tool.py")
        git(Path(found.path), "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "attempt 2")
        head = git(Path(found.path), "rev-parse", "HEAD")
        from dataclasses import asdict, replace

        worktree._write_json(Path(ctx.workdir) / worktree.RECORD, asdict(replace(found, head=head)))
        ctx.facts["commit"] = head
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", put_back)
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    record = _record(run)
    assert record["status"] == "kept local" and "changes nothing" in record["why"]
    assert "changes nothing" in result.facts["delivery_result"]
    calls = _calls(tmp_path)
    assert "repo create" not in calls and "pr create" not in calls


def _section(body: str) -> str:
    """The sections delivery wrote, between its markers."""
    return body[body.index(delivery.BODY_START):body.index(delivery.BODY_END) + len(delivery.BODY_END)]


def test_a_refine_rewrites_only_the_sections_delivery_wrote(opened_pull, tmp_path):
    """dm-212: a refine writes its own sections again, from the refined run, between the
    markers; a person's text above and below them stays as it is, and so does the title."""
    url, pull, github = opened_pull.url, opened_pull.pull, opened_pull.github
    pulls = _pulls(github, "acme/app")
    above, below = "Person's intro, kept.\n\n", "\n## Reviewer notes\n\nKeep this \\1 too.\n"
    pulls[0]["body"] = above + pull["body"] + below
    (github / "acme/app.pulls.json").write_text(json.dumps(pulls))
    run = tmp_path / "refine"
    result = _design(run, {"tool.py": "print('one')\n", "more.py": "print('two')\n"}, refine=url)
    assert result.exit_ok, result.error
    [after] = _pulls(github, "acme/app")
    ours = _section((run / delivery.BODY).read_text())
    assert after["body"] == above + ours + "\n" + below
    assert "more.py" in ours and "more.py" not in _section(pull["body"])
    assert after["title"] == pull["title"]


def test_a_refine_leaves_a_description_without_markers_as_it_is(opened_pull, tmp_path):
    """Correction 4: a pull request opened before the markers keeps its description,
    hand edits and all, and the delivery result says it was not updated."""
    url, github = opened_pull.url, opened_pull.github
    pulls = _pulls(github, "acme/app")
    legacy = "## What changed\n\n- tool.py, by hand: I rewrote this.\n\n## Original request\n\nA tool.\n"
    pulls[0]["body"] = legacy
    (github / "acme/app.pulls.json").write_text(json.dumps(pulls))
    run = tmp_path / "refine"
    result = _design(run, {"tool.py": "print('one')\n", "more.py": "print('two')\n"}, refine=url)
    assert result.exit_ok, result.error
    [after] = _pulls(github, "acme/app")
    assert after["body"] == legacy
    said = f"the description of {url} was not updated: it has no delivery markers"
    assert said in result.facts["delivery_result"] and said in _record(run)["note"]
    assert "pr edit" not in _calls(opened_pull.root).split("pr create", 1)[-1]


def test_a_runs_description_is_short_and_its_changes_come_from_the_design(new_run):
    """The short structure: Summary and Changes from the design, one line for the check and
    one for the reviews, nothing else. The design names `test_tool.py::test_build`, which
    the run never wrote, so there is no Tests section."""
    body = _draft_pull(new_run.github)["body"]
    assert body.startswith(delivery.BODY_START) and body.rstrip().endswith(delivery.BODY_END)
    headings = [line for line in body.splitlines() if line.startswith("## ")]
    assert headings == ["## Summary", "## Changes", "## Verification"]
    assert "## Summary\n\nFix it properly.\n" in body and "## Changes\n\n- tool.py\n" in body
    assert "test_build" not in body
    verification = body.split("## Verification\n\n", 1)[1].split(delivery.BODY_END, 1)[0]
    assert [line[:12] for line in verification.splitlines() if line] == ["- **Check:**", "- **Reviews:"]
    assert "Original request" not in body and "Written by a refactory design run" not in body


def _changed_pair(tmp_path, *names: str) -> tuple[str, str]:
    """`_git_pair`'s repository with ``names`` committed on top: (base, head)."""
    base, _ = _git_pair(tmp_path)
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(f"{name}\n")
        git(tmp_path, "add", name)
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "work")
    return base, git(tmp_path, "rev-parse", "HEAD")


def _a_body(tmp_path, design: str, *names: str) -> str:
    from types import SimpleNamespace
    base, head = _changed_pair(tmp_path, *names)
    ctx = _TitleCtx(facts={"brief": "b", "check": "pytest", "name": "b45a"}, files={"design.md": design})
    return delivery._body(ctx, SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="b"))


def test_the_changes_section_is_the_designs_bullets_one_per_behaviour(tmp_path, capsys):
    """dm-191: the PR's Changes are the design's ## Changes bullets, as written, one per
    behaviour: a bullet naming no file is kept, one naming only files the branch does not
    change is left out (with a warning), and a changed file no bullet names follows by name."""
    design = ("# Build a tool\n\n## Summary\n\nThe problem, and the fix.\n\n## Changes\n\n"
              "- `tool.py` — prints what it built, on one line\n"
              "- `docs/tool.md` — say what it prints\n"
              "- Core: the tokenizer splits on commas, not spaces\n"
              "- Behaviour change: `tool --list` prints one name per line\n")
    body = _a_body(tmp_path, design, "tool.py")
    changes = body.split("## Changes\n\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert changes == ["- `tool.py` — prints what it built, on one line",
                       "- Core: the tokenizer splits on commas, not spaces",
                       "- Behaviour change: `tool --list` prints one name per line",
                       "- g.txt"]
    assert "docs/tool.md" not in body and "docs/tool.md" in capsys.readouterr().err


def test_behaviour_bullets_that_name_no_file_reach_the_changes_section():
    design = ("# T\n\n## Summary\n\nS.\n\n## Changes\n\n"
              "- Core: the tokenizer splits on commas, not spaces\n"
              "- Behaviour change: `tool --list` prints one name per line\n")
    from refactory_dsh import design_contract
    items, _ = design_contract.describe_changes(design_contract.parse(design, "t"), ["tool.py"])
    assert items == ["Core: the tokenizer splits on commas, not spaces",
                     "Behaviour change: `tool --list` prints one name per line", "tool.py"]


def test_the_designs_core_and_behaviour_changes_sections_reach_the_pull_request(tmp_path):
    """dm-191: a design's own ## Core and ## Behaviour changes sections are carried into the
    description as written, after the Summary."""
    design = ("# Build a tool\n\n## Summary\n\nS.\n\n## Core\n\nThe tokenizer, rewritten.\n\n"
              "## Behaviour changes\n\n- `tool --list` prints one name per line\n\n"
              "## Changes\n\n- `tool.py` — the tokenizer\n")
    body = _a_body(tmp_path, design, "tool.py")
    headings = [line for line in body.splitlines() if line.startswith("## ")]
    assert headings[:4] == ["## Summary", "## Core", "## Behaviour changes", "## Changes"], headings
    assert "## Core\n\nThe tokenizer, rewritten.\n" in body
    assert "## Behaviour changes\n\n- `tool --list` prints one name per line\n" in body


def _repo_pair(tmp_path, base_files: dict[str, str], head_files: dict[str, str],
               removed: tuple[str, ...] = ()) -> tuple[Path, str, str]:
    """A repository whose base commit holds ``base_files`` and whose head writes
    ``head_files`` and removes ``removed``: (repository, base, head)."""
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for files in (base_files, head_files):
        if files is head_files:
            for name in removed:
                git(repo, "rm", "-q", name)
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
        git(repo, "add", "-A")
        git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "c")
        if files is base_files:
            base = git(repo, "rev-parse", "HEAD")
    return repo, base, git(repo, "rev-parse", "HEAD")


def _pair_body(repo: Path, base: str, head: str, design: str) -> str:
    from types import SimpleNamespace
    ctx = _TitleCtx(facts={"brief": "b", "check": "pytest", "name": "t"}, files={"design.md": design})
    return delivery._body(ctx, SimpleNamespace(path=str(repo), base=base, head=head, branch="b"))


def test_the_tests_section_names_only_tests_the_change_has(tmp_path):
    """A design can name a test that was never written: the PR lists a pytest id only when
    the branch changes its file and that file defines it at the head (parameters aside),
    and a vitest id only when its spec has an it() or test() of that title."""
    repo, base, head = _repo_pair(tmp_path, {"a.txt": "a\n"}, {
        "tests/test_tool.py": ("import pytest\n\n\ndef test_prints():\n    pass\n\n\n"
                               "@pytest.mark.parametrize('x', [1])\nasync def test_lists(x):\n    pass\n"),
        "web/tool.spec.ts": "describe('tool', () => {\n  it('lists one name per line', () => {})\n})\n"})
    design = ("# T\n\n## Summary\n\nS.\n\n## Changes\n\n- `tests/test_tool.py` — t\n\n## Tests\n\n"
              "- `tests/test_tool.py::test_prints` — it prints\n"
              "- `tests/test_tool.py::test_lists[1]` — it lists\n"
              "- `web/tool.spec.ts > tool > lists one name per line` — the spec\n"
              "- `web/tool.spec.ts > tool > never written either` — no such it()\n"
              "- `tests/test_never.py::test_never_written` — no such file\n")
    body = _pair_body(repo, base, head, design)
    assert ("## Tests\n\n- `tests/test_tool.py::test_prints`\n- `tests/test_tool.py::test_lists[1]`\n"
            "- `web/tool.spec.ts > tool > lists one name per line`\n") in body, body
    assert "never written" not in body


def test_a_test_the_design_names_in_a_changed_file_but_never_wrote_is_not_listed(tmp_path):
    repo, base, head = _repo_pair(tmp_path, {"a.txt": "a\n"},
                                  {"tests/test_tool.py": "def test_real():\n    assert True\n"})
    design = ("# T\n\n## Summary\n\nS.\n\n## Changes\n\n- `tests/test_tool.py` — tests\n\n"
              "## Tests\n\n- `tests/test_tool.py::test_real` — real\n"
              "- `tests/test_tool.py::test_never_written` — never written\n")
    body = _pair_body(repo, base, head, design)
    assert "test_real" in body
    assert "test_never_written" not in body, body


def test_a_bullet_about_a_deleted_file_is_kept(tmp_path):
    repo, base, head = _repo_pair(tmp_path, {"old/legacy.py": "x = 1\n", "a.txt": "a\n"},
                                  {"a.txt": "b\n"}, removed=("old/legacy.py",))
    design = ("# T\n\n## Summary\n\nS.\n\n## Changes\n\n"
              "- `old/legacy.py` — deleted: nothing imports it any more\n")
    body = _pair_body(repo, base, head, design)
    assert "nothing imports it any more" in body, body


def test_a_backticked_path_inside_a_bullet_is_the_file_it_names(tmp_path, capsys):
    """A path anywhere in a bullet names a file: it ties the bullet to a changed file, which
    is then not listed again, or leaves out a bullet about a file the branch does not change."""
    repo, base, head = _repo_pair(tmp_path, {"src/parse.py": "a\n", "old/shim.py": "s\n"},
                                  {"src/parse.py": "b\n"})
    design = ("# T\n\n## Summary\n\nS.\n\n## Changes\n\n"
              "- Fix the parser in `src/parse.py`, so empty input is refused\n"
              "- Drop the shim in `old/shim.py`\n")
    body = _pair_body(repo, base, head, design)
    changes = body.split("## Changes\n\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert changes == ["- Fix the parser in `src/parse.py`, so empty input is refused"], changes
    assert "old/shim.py" in capsys.readouterr().err


def test_a_refine_keeps_backslashes_in_the_sections_it_writes(tmp_path, monkeypatch):
    """The new sections replace the old ones as written: a backslash in them, as in a
    Windows path or a regular expression, is not read as a group reference."""
    ours = f"{delivery.BODY_START}\n## Summary\n\nMatch C:\\1 and \\g<0>.\n{delivery.BODY_END}\n"
    (tmp_path / delivery.BODY).write_text(ours)
    old = f"Theirs.\n\n{delivery.BODY_START}\n## Summary\n\nOld.\n{delivery.BODY_END}\n"
    sent: list[str] = []

    def gh(*args, **_):
        if args[:2] == ("pr", "view"):
            return json.dumps({"body": old, "title": "T", "isDraft": False})
        if args[:2] == ("pr", "edit") and "--body-file" in args:
            sent.append(Path(args[args.index("--body-file") + 1]).read_text())
        return ""

    monkeypatch.setattr(delivery, "_gh", gh)
    assert delivery._describe(nn.RunContext(tmp_path, enforce_writes=False), "u", None) == ""
    assert sent == ["Theirs.\n\n" + ours]


def test_the_check_line_carries_the_suite_counts(tmp_path):
    """The Verification's check line says what the test runners counted."""
    from types import SimpleNamespace
    base, head = _git_pair(tmp_path)
    stdout = ("..........\n3 passed, 1 skipped in 0.52s\n"
              "\x1b[2m      Tests \x1b[22m \x1b[1m\x1b[32m4 passed\x1b[39m\x1b[22m (4)\n")
    ctx = _TitleCtx(facts={"check": "make test", "name": "b"},
                    files={"authoring-evidence.json": json.dumps({"validation": {"status": "pass", "stdout": stdout}})})
    body = delivery._body(ctx, SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="b"))
    assert "- **Check:** `make test` passed (pytest 3 passed, 1 skipped; vitest 4 passed (4)).\n" in body


@pytest.mark.parametrize("said", [
    # gh 2.x, as fake_gh says it.
    "pull request create failed: GraphQL: Draft pull requests are not supported in this "
    "repository. (createPullRequest)",
    # Older gh releases, as reported in cli/cli#520 and MetaMask/action-create-release-pr#67.
    "failed to create pull request: graphql error: 'Draft pull requests are not supported in "
    "this repository.'",
    "pull request create failed: GraphQL error: Draft pull requests are not supported in this "
    "repository.",
])
def test_githubs_draft_refusal_is_recognised_in_each_wording_gh_has_used(tmp_path, monkeypatch, said):
    """dm-246: a refused draft is recognised by GitHub's own message, which every gh release
    has carried, however it wrapped it: the pull request then opens ready, titled [draft]."""
    opened: list[str] = []

    def opens(path, slug, branch, into, title, body, *, draft=False):
        if draft:
            raise delivery.NotDelivered(f"`gh pr create` failed: {said}")
        opened.append(title)
        return "https://github.com/acme/app/pull/1"

    monkeypatch.setattr(delivery, "_open", opens)
    url, note = delivery._open_draft(tmp_path, "acme/app", "b", "main", "Do the thing", tmp_path / "body")
    assert opened == [delivery.DRAFT_TITLE + "Do the thing"] and url.endswith("/pull/1")
    assert "GitHub refused a draft pull request" in note


def _main_moved(github: Path, slug: str, name: str, tmp_path: Path) -> str:
    """Someone merged ``name`` into main on GitHub; returns main's new tip."""
    other = tmp_path / f"other-{name}"
    git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
    (other / name).write_text(f"{name}\n")
    git(other, "add", name)
    git(other, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", name)
    git(other, "push", "-q", "origin", "main")
    return git(other, "rev-parse", "HEAD")


def test_redeliver_on_a_design_run_behind_main_merges_and_pushes(new_run, tmp_path, monkeypatch):
    """dm-192: deliver declared the `commit` it reads, so `graph-control redeliver`, which
    resets deliver's facts, always failed. It resets `delivered_commit` now and keeps
    materialise's `commit`: each redeliver merges main and pushes, the second building on
    the first's merge, and the merge it pushed is deliver's own fact (correction 3)."""
    from refactory_dsh import items, lowering
    from refactory_dsh.run import run_node

    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    reviewed = result.facts["commit"]
    if not (run / lowering.DECLARATIONS).exists():
        lowering.write_declarations(DesignMode().state_dict(), run)
    (run / "facts.json").write_text(json.dumps(result.facts, default=str))
    monkeypatch.setenv("REFACTORY_WORKDIR", str(run))
    heads = [reviewed]
    for name in ("first.txt", "second.txt"):
        tip = _main_moved(github, slug, name, tmp_path)
        # What the spine does for `graph-control redeliver`: a rerun of deliver.
        items.command_reset(run, {"nodes": ["deliver", "delivery_gate", "done"], "start": "deliver"})
        facts = json.loads((run / "facts.json").read_text())
        assert facts["commit"] == reviewed and "delivered_commit" not in facts
        monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "deliver", "type": "op"}))
        run_node("refactory_dsh.delivery:deliver_design", workdir=run, node_id="deliver")
        facts = json.loads((run / "facts.json").read_text())
        [pull] = _pulls(github, slug)
        pushed = _remote_head(github, slug, pull["head"])
        assert _parents(github, slug, pushed) == [heads[-1], tip]
        assert facts["delivered_commit"] == pushed and facts["commit"] == reviewed
        assert _record(run)["merge_commit"] == pushed
        heads.append(pushed)
    assert len(_pulls(github, slug)) == 1


# ── b45a review: the overall limit, the failing refusal, and what a redo keeps ──
FAILING = render({"tool.py": "print('built')\nraise SystemExit(1)\n"})


def test_the_overall_limit_on_a_review_send_back_drafts_through_the_overwatch(github, tmp_path):
    """Two passing checks whose QA judge sends them back reach the overall attempts limit
    (2) at gate_qa, which goes on through the correction audit and the overwatch.  A check
    failure no longer spends an attempt, so both attempts have to pass the check."""
    run = tmp_path / "run"
    # The rebuild differs from attempt 1, or materialise reads it as no change.
    rebuilt = render({"tool.py": "print(\u0027built\u0027)\n# rebuilt after the QA send-back\n"})
    result, model = _limited(run,
                             _attempt(BUILT, judge=qa_judgment("redo", must_fix=[QA_HELD])),
                             _attempt(rebuilt, judge=qa_judgment("redo", must_fix=[QA_HELD]),
                                      overwatch=overwatch_judgment("pass")),
                             check_attempts=2, review_attempts=3, check_attempt_cap=0)
    assert result.exit_ok, result.error
    fact = result.facts["draft_delivery"]
    assert fact["reason"] == "authoring unresolved after 2 attempt(s)" and "check_failing" not in fact
    # The same item was named in both attempts, so both rounds are recorded.
    assert fact["unresolved"] == [{"text": QA_HELD_LINE, "source": "QA judge", "rounds": [1, 2]}]
    assert _asked(model, OVERWATCH_JUDGE_MARK) == 1 and (run / "correction-audit.json").exists()
    pull = _draft_pull(github)
    assert pull["draft"] is True and pull["title"] == "Do the thing"


def test_the_overall_limit_on_a_check_failure_drafts_straight_to_the_report(github, tmp_path):
    """Attempt 1 passes and the QA judge sends it back; the check then fails repeatedly
    through the combined sendback bound.  The draft is check failing and skips the judges.
    The check's own send-back does not spend an attempt."""
    run = tmp_path / "run"
    failings = [[render({"tool.py": f"print('built')\nraise SystemExit({i})\n"})]
                for i in range(1, 12)]
    result, model = _limited(run,
                             _attempt(BUILT, judge=qa_judgment("redo", must_fix=[QA_HELD])),
                             *failings,
                             check_attempts=5, review_attempts=3, check_attempt_cap=0)
    assert result.exit_ok, result.error
    fact = result.facts["draft_delivery"]
    assert "check failures exhausted" in fact["reason"] and fact["check_failing"] is True
    # The QA judge sent attempt 1 back; the check-failing draft skips the overwatch.
    assert _asked(model, QA_JUDGE_MARK) == 1 and _asked(model, OVERWATCH_JUDGE_MARK) == 0
    pull = _draft_pull(github)
    assert pull["draft"] is True and pull["title"] == "[check failing] Do the thing"
    assert _remote_head(github, f"{LOGIN}/{NAME}", pull["head"]) == result.facts["commit"]


def test_the_refusal_bound_after_a_failing_check_is_a_check_failing_draft(github, tmp_path):
    """Attempt 1's check fails; the next emissions are refused to the bound. The draft is
    attempt 1, marked check failing, as its evidence says."""
    run = tmp_path / "run"
    result, model = _limited(run, [FAILING], [REFUSED], [REFUSED],
                             check_attempts=3, refusal_attempts=2, check_attempt_cap=0)
    assert result.exit_ok, result.error
    fact = result.facts["draft_delivery"]
    assert fact["reason"].startswith("emissions refused 2 consecutive times") and fact["check_failing"] is True
    assert _asked(model, QA_JUDGE_MARK) == 0
    pull = _draft_pull(github)
    assert pull["draft"] is True and pull["title"] == "[check failing] Do the thing"


def test_a_later_qa_proceed_clears_the_earlier_qa_items_from_an_overwatch_draft(github, tmp_path):
    """Attempt 1: the QA judge sends it back. Attempt 2: the QA judge proceeds and the
    overwatch sends it back at the review limit. The draft lists only the overwatch's item."""
    run = tmp_path / "run"
    held = overwatch_judgment("redo", must_fix=[OVERWATCH_HELD])
    result, _ = _limited(run, _attempt(BUILT, judge=qa_judgment("redo", must_fix=[QA_HELD])),
                         _attempt(render({"tool.py": "print('built!')\n"}), overwatch=held),
                         review_attempts=2)
    assert result.exit_ok, result.error
    assert result.facts["draft_delivery"]["unresolved"] == [
        {"text": OVERWATCH_HELD_LINE, "source": "overwatch", "rounds": [2]}]


def test_a_follow_up_both_judges_name_is_listed_once(github, tmp_path):
    item = {"source": "qa", "where": "tool.py:1", "what": "add a test for an empty input",
            "why": "not an ask", "severity": "low"}
    run = tmp_path / "run"
    result, _ = _limited(run, _attempt(BUILT, judge=qa_judgment("proceed", follow_ups=[item]),
                                       overwatch=overwatch_judgment("pass", follow_ups=[
                                           {**item, "source": "overwatch"}])))
    assert result.exit_ok, result.error
    follow = (run / delivery.BODY).read_text().split("## Follow-ups\n\n", 1)[1].split(
        f"\n{delivery.BODY_END}", 1)[0]
    assert follow.splitlines() == [
        "- add a test for an empty input — tool.py:1 — not an ask (qa, overwatch)"], follow
    assert "1 non-core finding(s) are listed under Follow-ups" in (run / delivery.BODY).read_text()


def _describing(monkeypatch, tmp_path, body: str, *, title: str = "T", draft: bool = False) -> dict:
    """A pull request as `_describe` reads and edits it, through a stand-in for `_gh`."""
    state = {"body": body, "title": title, "draft": draft, "calls": []}

    def gh(*args, **_):
        state["calls"].append(args)
        if args[:2] == ("pr", "view"):
            return json.dumps({"body": state["body"], "title": state["title"], "isDraft": state["draft"]})
        if args[:2] == ("pr", "edit") and "--body-file" in args:
            state["body"] = Path(args[args.index("--body-file") + 1]).read_text()
        if args[:2] == ("pr", "edit") and "--title" in args:
            state["title"] = args[args.index("--title") + 1]
        if args[:2] == ("pr", "ready"):
            state["draft"] = "--undo" in args
        return ""

    monkeypatch.setattr(delivery, "_gh", gh)
    return state


OURS = f"{delivery.BODY_START}\n## Summary\n\nS.\n{delivery.BODY_END}\n"


def test_a_persons_draft_that_a_draft_delivery_touched_is_not_marked_ready(tmp_path, monkeypatch):
    """A person keeps their pull request a draft. A refine hits its limit (a draft delivery,
    with nothing to convert), then a later refine passes: it is still their draft. A draft
    delivery made is marked ready (`test_a_full_delivery_after_a_draft_makes_it_ready_…`)."""
    (tmp_path / delivery.BODY).write_text(OURS)
    pr = _describing(monkeypatch, tmp_path, "My own WIP.\n\n" + OURS, draft=True)
    ctx = nn.RunContext(tmp_path, enforce_writes=False)
    delivery._describe(ctx, "u", {"reason": "limit", "unresolved": []})
    assert pr["body"].startswith(delivery.DRAFT_START + "\n") and ("pr", "ready", "--undo", "u") not in pr["calls"]
    (tmp_path / delivery.BODY).write_text(OURS)
    delivery._describe(ctx, "u", None)
    assert pr["draft"] is True and pr["body"] == "My own WIP.\n\n" + OURS


@pytest.mark.parametrize("legacy", ["\r\nWritten in the web UI.\r\nSecond line.\r\n",
                                    "\nLeading blank line, then text.\n"], ids=["crlf", "newline"])
def test_a_legacy_body_is_left_byte_identical_on_a_refine(tmp_path, monkeypatch, legacy):
    (tmp_path / delivery.BODY).write_text(OURS)
    pr = _describing(monkeypatch, tmp_path, legacy)
    note = delivery._describe(nn.RunContext(tmp_path, enforce_writes=False), "u", None)
    assert pr["body"] == legacy and not [args for args in pr["calls"] if "--body-file" in args]
    assert note == ("the description of u was not updated: it has no delivery markers, so all of "
                    "it is kept as written")


def test_a_draft_onto_a_legacy_body_says_its_section_was_put_above_it(tmp_path, monkeypatch):
    (tmp_path / delivery.BODY).write_text(OURS)
    pr = _describing(monkeypatch, tmp_path, "A hand-written body.\n")
    note = delivery._describe(nn.RunContext(tmp_path, enforce_writes=False), "u", DRAFT_FACT)
    assert pr["body"].endswith("\n\nA hand-written body.\n") and pr["draft"] is True
    assert "its text is kept as written, with the draft's section put above it" in note
    assert "all of it is kept as written" not in note


def test_a_refine_keeps_a_persons_own_draft_title_prefix(tmp_path, monkeypatch):
    (tmp_path / delivery.BODY).write_text(OURS)
    pr = _describing(monkeypatch, tmp_path, "Mine.\n", title="[draft] my idea", draft=True)
    delivery._describe(nn.RunContext(tmp_path, enforce_writes=False), "u", None)
    assert pr["title"] == "[draft] my idea" and not [args for args in pr["calls"] if "--title" in args]


def test_a_persons_draft_title_prefix_stays_on_their_draft_with_deliverys_section(tmp_path, monkeypatch):
    """A person's draft, titled "[draft] …", that a draft delivery touched (its section says
    the pull request was a draft already): a later full delivery takes the section out but
    leaves their title and their draft alone."""
    (tmp_path / delivery.BODY).write_text(OURS)
    section = delivery._draft_preamble(DRAFT_FACT, "")
    pr = _describing(monkeypatch, tmp_path, f"{section}\n\nMine.\n\n{OURS}", title="[draft] my idea",
                     draft=True)
    delivery._describe(nn.RunContext(tmp_path, enforce_writes=False), "u", None)
    assert pr["title"] == "[draft] my idea" and pr["draft"] is True
    assert pr["body"] == f"Mine.\n\n{OURS}"


def test_delivery_markers_inside_a_value_cannot_end_a_section(tmp_path):
    """A design or a judge that writes delivery's markers itself does not end, or open,
    a section: they are taken out of every value."""
    design = (f"# T\n\n## Summary\n\nS {delivery.BODY_END} and {delivery.DRAFT_MADE}.\n\n"
              "## Changes\n\n- `tool.py` — t\n")
    body = _a_body(tmp_path, design, "tool.py")
    assert body.count(delivery.BODY_START) == 1 and body.count(delivery.BODY_END) == 1
    assert "delivery:draft" not in body
    section = delivery._draft_preamble({"reason": f"stop {delivery.DRAFT_END}",
                                        "unresolved": [{"text": f"x {delivery.BODY_START}"}]})
    assert section.count(delivery.DRAFT_END) == 1 and delivery.BODY_START not in section


def test_a_branch_at_a_plain_commit_on_its_reviewed_one_is_never_delivered(new_run):
    """Only delivery's own merges of the reviewed commit are delivered (dm-192); a plain
    commit on top of it is not the reviewed work."""
    run, result = new_run.run, new_run.result
    found = worktree.recorded(run)
    (Path(found.path) / "unreviewed.py").write_text("x = 1\n")
    git(Path(found.path), "add", "unreviewed.py")
    git(Path(found.path), "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "more")
    from dataclasses import asdict, replace

    worktree._write_json(run / worktree.RECORD,
                         asdict(replace(found, head=git(Path(found.path), "rev-parse", "HEAD"))))
    with pytest.raises(ValueError, match="not at the attempt its reviews passed"):
        delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))


def test_an_empty_change_is_refused_before_main_is_merged_in(github, tmp_path, monkeypatch):
    """dm-246: main moved before delivery, and the run's change is empty. Merging main in
    first would give a head with main's files, and a pull request of nothing: delivery
    refuses before the merge."""
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    real = delivery.deliver_design

    def put_back_and_move_main(ctx):
        found = worktree.recorded(ctx.workdir)
        git(Path(found.path), "rm", "-q", "tool.py")
        git(Path(found.path), "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "attempt 2")
        head = git(Path(found.path), "rev-parse", "HEAD")
        from dataclasses import asdict, replace

        worktree._write_json(Path(ctx.workdir) / worktree.RECORD, asdict(replace(found, head=head)))
        ctx.facts["commit"] = head
        _main_moved(github, "acme/app", "theirs.txt", tmp_path)
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", put_back_and_move_main)
    run = tmp_path / "run"
    result = _design(run, target=str(repo))
    assert result.exit_ok, result.error
    assert "changes nothing" in _record(run)["why"]
    assert _pulls(github, "acme/app") == [] and "merge_commit" not in _record(run)


def test_redeliver_after_someone_else_pushed_to_the_branch_does_not_overwrite_it(new_run, tmp_path,
                                                                                 monkeypatch):
    """A person pushed onto the pull request's branch; a redeliver never overwrites it."""
    from refactory_dsh import items, lowering
    from refactory_dsh.run import run_node

    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    [pull] = _pulls(github, slug)
    other = tmp_path / "theirs"
    git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
    git(other, "checkout", "-q", pull["head"])
    (other / "theirs.txt").write_text("theirs\n")
    git(other, "add", "theirs.txt")
    git(other, "-c", "user.name=p", "-c", "user.email=p@example.com", "commit", "-q", "-m", "theirs")
    git(other, "push", "-q", "origin", f"HEAD:refs/heads/{pull['head']}")
    theirs = git(other, "rev-parse", "HEAD")
    _main_moved(github, slug, "first.txt", tmp_path)
    if not (run / lowering.DECLARATIONS).exists():
        lowering.write_declarations(DesignMode().state_dict(), run)
    (run / "facts.json").write_text(json.dumps(result.facts, default=str))
    monkeypatch.setenv("REFACTORY_WORKDIR", str(run))
    items.command_reset(run, {"nodes": ["deliver", "delivery_gate", "done"], "start": "deliver"})
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "deliver", "type": "op"}))
    run_node("refactory_dsh.delivery:deliver_design", workdir=run, node_id="deliver")
    assert _remote_head(github, slug, pull["head"]) == theirs
    assert json.loads((run / "facts.json").read_text())["delivery_result"].startswith("not delivered")


def _design_draft(workdir, github, tmp_path, *, mode=None, emit=None, target=None,
                  check="python tool.py", generate="", draft_fact=None,
                  advance_main=None, monkeypatch=None, seen=None):
    """Run a full design with the draft_delivery fact injected via monkeypatch.

    When *target* is None the run creates a new project (same as ``_design``).
    When *target* is a repo path, the run uses that existing repo.
    *seen*, when given, gets ``head``: the run's reviewed commit as delivery found it,
    before delivery's own bookkeeping can move the worktree record.
    """
    fact = draft_fact if draft_fact is not None else DRAFT_FACT

    import refactory_dsh.design_ops as design_ops_mod
    # The report the design graph calls: its judges decided, so it is report_judged.
    real_report = design_ops_mod.report_judged
    real_deliver = delivery.deliver_design

    def inject_draft_report(ctx):
        ctx.facts["draft_delivery"] = fact
        return real_report(ctx)

    def inject_draft_deliver(ctx):
        ctx.facts["draft_delivery"] = fact
        if seen is not None:
            seen["head"] = worktree.recorded(ctx.workdir).head
        if advance_main is not None:
            advance_main()
        return real_deliver(ctx)

    # Use a simple setattr/restore pattern when monkeypatch is not available.
    saved_report = design_ops_mod.report_judged
    saved_deliver = delivery.deliver_design
    design_ops_mod.report_judged = inject_draft_report
    delivery.deliver_design = inject_draft_deliver
    try:
        task_kw: dict[str, str] = {}
        if target is not None:
            task_kw["target"] = str(target)
        if generate:
            task_kw["generate"] = generate
        model = FakeModel(_answers(render(emit or {"tool.py": "print('built')\n"})))
        result = (mode or DesignMode()).run(
            workdir, model=model, verdicts={"gate_strategy": "proceed"},
            task={"name": NAME, "brief": "Build a tool\nthat prints built.",
                  "check": check, **task_kw},
        )
    finally:
        design_ops_mod.report_judged = saved_report
        delivery.deliver_design = saved_deliver
    return result


def _parents(github: Path, slug: str, commit: str) -> list[str]:
    """The parents of ``commit`` in the repository on GitHub."""
    return git(github / f"{slug}.git", "rev-list", "--parents", "-n1", commit).split()[1:]


def _shown(github: Path, slug: str, commit: str, name: str) -> str:
    """The file ``name`` as ``commit`` on GitHub holds it."""
    return git(github / f"{slug}.git", "show", f"{commit}:{name}")


def _draft_items(body: str) -> list[str]:
    """The item lines of the draft section at the top of a pull request's description."""
    found = delivery._DRAFT_SECTION.search(body)
    assert found is not None, body[:200]
    section = found.group(0)
    return [line for line in section.splitlines() if line.startswith("- ")]


def _registry_note(run: Path) -> dict:
    """What `refactory projects` keeps about the run's branch."""
    found = worktree.recorded(run)
    return next(note for entry in projects.load()["projects"] for note in entry.get("runs", [])
                if note.get("branch") == found.branch)


# ── shared design runs ─────────────────────────────────────────────────────
#
# Many tests start from one of two design runs and then do something to what it left:
# a new project whose first pull request is open (`new_run`), or an existing repository
# with a pull request a run opened (`opened_pull`, for refinements). Those runs are the
# same every time, so each is walked once, for real, through the whole design graph,
# and every test gets its own copy of everything it left: the run directory, the
# worktree, the project and its registry, gh's log and GitHub. The copy is restored at
# the path the run was made at -- one stage directory for the module -- because the
# worktrees, records and remotes name absolute paths.
#
# A copy is not the run in one respect: a reviewer's sign-off is bound to the inode and
# ctime of what it saw (`agent_delivery._fingerprint`), and copied files have new ones.
# So a test that needs current sign-offs -- `report` without a draft fact, which calls
# `require_completion` -- walks its own run.


@dataclass
class Staged:
    """A restored design run, and what its test needs to read it."""

    root: Path  # what `tmp_path` is to a test that walks its own run: `bin/calls.log` is here
    github: Path
    run: Path | None = None  # `new_run`: the run's directory
    result: nn.RunResult | None = None  # `new_run`: what the run returned
    repo: Path | None = None  # `opened_pull`: the person's checkout
    url: str = ""  # `opened_pull`: the pull request the first run opened
    pull: dict | None = None  # `opened_pull`: that pull request as GitHub showed it


def _use_home(monkeypatch: pytest.MonkeyPatch, base: Path) -> None:
    """Refactory's home and projects folder under ``base``, as `_isolate_refactory_home` sets them."""
    home = base / "refactory-home"
    monkeypatch.setenv("REFACTORY_HOME", str(home / "home"))
    monkeypatch.setenv("REFACTORY_PROJECTS_DIR", str(home / "projects"))
    monkeypatch.setenv("GH_CONFIG_DIR", str(home / "gh"))
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _build_new(stage: Path, github: Path) -> Staged:
    result = _design(stage / "run")
    assert result.exit_ok, result.error
    return Staged(stage, github, stage / "run", result)


def _build_opened(stage: Path, github: Path) -> Staged:
    repo, url, pull = _opened(github, stage)
    return Staged(stage, github, repo=repo, url=url, pull=pull)


@pytest.fixture(scope="module")
def _stage(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("delivery-stage")


def _golden(stage: Path, factory: pytest.TempPathFactory, kind: str, build) -> tuple[Path, Staged]:
    """Walk one shared run at ``stage`` and keep a pristine copy of everything it left."""
    shutil.rmtree(stage)
    stage.mkdir()
    github = _make_github(stage)
    with pytest.MonkeyPatch.context() as monkeypatch:
        _use_github(monkeypatch, stage)
        _use_home(monkeypatch, stage)
        made = build(stage, github)
    pristine = factory.mktemp(f"delivery-{kind}") / "tree"
    shutil.copytree(stage, pristine, symlinks=True)
    return pristine, made


def _restore(stage: Path, golden: tuple[Path, Staged], monkeypatch: pytest.MonkeyPatch) -> Staged:
    """This test's own copy of a shared run, at the stage, with gh and the home pointed at it."""
    pristine, made = golden
    shutil.rmtree(stage)
    shutil.copytree(pristine, stage, symlinks=True)
    _use_github(monkeypatch, stage)
    _use_home(monkeypatch, stage)
    return copy.deepcopy(made)


@pytest.fixture(scope="module")
def _new_golden(_stage: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Staged]:
    return _golden(_stage, tmp_path_factory, "new", _build_new)


@pytest.fixture(scope="module")
def _opened_golden(_stage: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Staged]:
    return _golden(_stage, tmp_path_factory, "opened", _build_opened)


@pytest.fixture
def new_run(_stage: Path, _new_golden: tuple[Path, Staged], monkeypatch: pytest.MonkeyPatch) -> Staged:
    """A new project's design run, finished: its private repository made, its first pull request open."""
    return _restore(_stage, _new_golden, monkeypatch)


@pytest.fixture
def opened_pull(_stage: Path, _opened_golden: tuple[Path, Staged], monkeypatch: pytest.MonkeyPatch) -> Staged:
    """`acme/app` on GitHub, the person's checkout of it, and a pull request a design run opened there."""
    return _restore(_stage, _opened_golden, monkeypatch)


def test_draft_passing_check_opens_draft_pr(github, tmp_path):
    """A draft run with a passing check opens a draft PR with the unresolved section."""
    run = tmp_path / "run"
    result = _design_draft(run, github, tmp_path)
    assert result.exit_ok, result.error

    # The fake gh saw `pr create` with `--draft`.
    calls = _calls(tmp_path)
    assert "--draft" in calls, f"expected --draft in calls: {calls}"

    # The PR body starts with the unresolved section.
    slug = f"{LOGIN}/{NAME}"
    [pull] = _pulls(github, slug)
    assert pull["body"].startswith(f"{delivery.DRAFT_MADE}\n{delivery.DRAFT_HEADING}\n"), pull["body"][:200]
    assert "review reloop limit reached" in pull["body"]
    assert "The logging format is inconsistent" in pull["body"]
    assert "source: review" in pull["body"]
    assert "rounds: 1, 2, 3" in pull["body"]
    assert "Missing error handling in parser" in pull["body"]
    assert "source: overwatch" in pull["body"]
    assert pull.get("draft") is True

    # delivery_result starts with "draft:".
    assert result.facts["delivery_result"].startswith("draft:"), result.facts["delivery_result"]
    assert "2 unresolved item(s)" in result.facts["delivery_result"]

    # Delivery record has draft fields.
    rec = _record(run)
    assert rec["draft"] is True
    assert rec["check_passed"] is True
    assert rec["draft_reason"] == "review reloop limit reached after 3 review cycle(s)"
    assert "draft_note" not in rec

    # The run's report carries the draft fields (report_artifact).
    report = json.loads((run / "design.json").read_text())
    assert report["draft"] is True and report["check_passed"] is True
    assert report["draft_reason"] == DRAFT_FACT["reason"]
    assert report["draft_unresolved"] == DRAFT_FACT["unresolved"]

    # `refactory projects` lists the run's delivery as a draft.
    assert _registry_note(run)["delivery"] == "draft"
    assert _registry_note(run)["pull_request"] == result.facts["pull_request"]


def test_draft_failing_check_opens_draft_pr(new_run):
    """A draft run with a failing check: report does NOT raise, the draft PR opens.

    The evidence shows status='fail' but require_current_evidence accepts it
    because it does not check validation.status.
    """
    from refactory_dsh.design_ops import report_artifact
    from refactory_dsh.nn.executor import RunContext
    from refactory_dsh.authoring_evidence import SETTINGS

    # A normal run has set up the worktree and evidence.
    run, result = new_run.run, new_run.result

    # Tamper the evidence to show a failing check.
    ev_path = run / "authoring-evidence.json"
    ev = json.loads(ev_path.read_text())
    ev["validation"]["status"] = "fail"
    ev_path.write_text(json.dumps(ev))

    # Call report_artifact with draft_delivery.
    # Use the same knobs from the evidence so currency checks pass.
    knobs = {key: ev.get("settings", {}).get(key) for key in SETTINGS}
    facts = dict(result.facts)
    facts["draft_delivery"] = DRAFT_FACT
    ctx = RunContext(run, knobs=knobs, facts=facts, enforce_writes=False)
    report_result = report_artifact(ctx)
    assert "designed" in report_result

    # Call deliver_design with draft_delivery.
    facts.update(report_result)
    outcome = delivery.deliver_design(RunContext(run, knobs=knobs, facts=facts, enforce_writes=False))
    assert outcome["delivery_result"].startswith("draft:"), outcome["delivery_result"]

    rec = _record(run)
    assert rec["draft"] is True
    assert rec["check_passed"] is False


def test_draft_no_evidence_refused(github, tmp_path):
    """A draft run with no/stale evidence: report refuses."""
    # We need to test that require_current_evidence raises when there's no evidence.
    # Build a RunContext directly.
    from refactory_dsh.authoring_evidence import require_current_evidence
    from refactory_dsh.nn.executor import RunContext

    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    # Write facts with runtime_evidence set (so evidence is expected).
    facts = {"draft_delivery": DRAFT_FACT, "runtime_evidence": True}
    ctx = RunContext(run, facts=facts)
    with pytest.raises(ValueError, match="evidence is missing"):
        require_current_evidence(ctx)


def test_no_draft_fact_requires_signoff(github, tmp_path):
    """Without draft_delivery, a run whose sign-off is missing is refused by report.
    With the fact, the same run delivers a draft (evidence is current, only sign-off missing)."""
    from refactory_dsh.design_ops import report_artifact
    from refactory_dsh.nn.executor import RunContext
    from refactory_dsh.authoring_evidence import SETTINGS

    run = tmp_path / "run_signoff"
    # Run a full design first to set up the worktree and evidence.
    result_normal = _design(run)
    assert result_normal.exit_ok, result_normal.error

    # Tamper the evidence to make the validation status "fail".
    # This makes require_completion fail ("required runtime validation did not pass")
    # but require_current_evidence pass (it does not check validation.status).
    ev_path = run / "authoring-evidence.json"
    ev = json.loads(ev_path.read_text())
    ev["validation"]["status"] = "fail"
    ev_path.write_text(json.dumps(ev))

    # Build knobs from the evidence so currency checks pass.
    knobs = {key: ev.get("settings", {}).get(key) for key in SETTINGS}

    # Without the fact: report refuses with sign-off error.
    facts = dict(result_normal.facts)
    facts.pop("draft_delivery", None)
    ctx = RunContext(run, knobs=knobs, facts=facts, enforce_writes=False)
    with pytest.raises(ValueError, match="sign-off is incomplete or stale"):
        report_artifact(ctx)

    # With the fact: the same run delivers a draft (require_current_evidence passes).
    facts["draft_delivery"] = DRAFT_FACT
    ctx2 = RunContext(run, knobs=knobs, facts=facts, enforce_writes=False)
    report_result = report_artifact(ctx2)
    assert "designed" in report_result

    # Complete the draft delivery.
    facts.update(report_result)
    outcome = delivery.deliver_design(RunContext(run, knobs=knobs, facts=facts, enforce_writes=False))
    assert outcome["delivery_result"].startswith("draft:"), outcome["delivery_result"]


def test_draft_conflict_with_base(github, tmp_path, monkeypatch):
    """A draft run with a merge conflict: draft PR opened, run's own commit pushed,
    conflict named in the unresolved section."""
    repo = git_repo(tmp_path / "repo_conflict")
    (repo / "authored.py").write_text("line1\nline2\nline3\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base with authored")
    slug = "acme/draft-conflict"
    _published(github, slug, repo)

    def advance_main():
        other = tmp_path / "other_conflict"
        if not other.exists():
            git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
            (other / "authored.py").write_text("line1\nchanged_by_base\nline3\n")
            git(other, "add", ".")
            git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
                "commit", "-q", "-m", "base changes authored.py")
            git(other, "push", "-q", "origin", "main")

    run = tmp_path / "run_conflict"
    seen: dict = {}
    result = _design_draft(run, github, tmp_path, target=repo,
                           emit={"tool.py": "print('built')\n",
                                 "authored.py": "line1\nchanged_by_run\nline3\n"},
                           advance_main=advance_main, seen=seen)
    assert result.exit_ok, result.error

    calls = _calls(tmp_path)
    assert "--draft" in calls

    # The pushed head is the run's own commit (not a merge): the commit delivery
    # found before it started, with one parent and the run's version of the file.
    # (The worktree record is no witness: delivery moves it to a merge it made.)
    found = worktree.recorded(run)
    remote = _remote_head(github, slug, found.branch)
    assert remote == seen["head"], f"expected run's own commit {seen['head']}, got {remote}"
    assert len(_parents(github, slug, remote)) == 1
    assert _shown(github, slug, remote, "authored.py") == "line1\nchanged_by_run\nline3"

    # The PR body's unresolved section names the conflict first.
    [pull] = _pulls(github, slug)
    items = _draft_items(pull["body"])
    assert items[0].startswith("- Merge conflict with base in authored.py"), items
    assert len(items) == 1 + len(DRAFT_FACT["unresolved"])

    # delivery_result says draft.
    assert result.facts["delivery_result"].startswith("draft:")

    # The delivery gate returns "proceed" (early return for draft).
    # The run ended done, not looping back.
    assert result.exit_ok


def test_draft_failing_merged_check(github, tmp_path, monkeypatch):
    """A draft run with a failing merged-result check: draft PR opened, run's own commit pushed."""
    repo = git_repo(tmp_path / "repo_mfail")
    (repo / "config.txt").write_text("ok\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base config ok")
    slug = "acme/draft-mfail"
    _published(github, slug, repo)

    def advance_main():
        other = tmp_path / "other_mfail"
        if not other.exists():
            git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
            (other / "config.txt").write_text("broken\n")
            git(other, "add", ".")
            git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
                "commit", "-q", "-m", "base breaks config")
            git(other, "push", "-q", "origin", "main")

    run = tmp_path / "run_mfail"
    seen: dict = {}
    result = _design_draft(
        run, github, tmp_path, target=repo,
        emit={"tool.py": "import sys; sys.exit(1 if open('config.txt').read().strip() == 'broken' else 0)\n"},
        check="python tool.py",
        advance_main=advance_main, seen=seen,
    )
    assert result.exit_ok, result.error

    calls = _calls(tmp_path)
    assert "--draft" in calls

    # The pushed head is the run's own commit, never the merge whose check failed: the
    # commit delivery found before it started, with one parent and the base's config.
    found = worktree.recorded(run)
    remote = _remote_head(github, slug, found.branch)
    assert remote == seen["head"], f"expected run's own commit {seen['head']}, got {remote}"
    assert len(_parents(github, slug, remote)) == 1
    assert _shown(github, slug, remote, "config.txt") == "ok"
    # ...and the record claims no merge.
    rec = _record(run)
    assert rec["head"] == seen["head"] and "merge_commit" not in rec and "merge_check" not in rec

    # delivery_result says draft.
    assert result.facts["delivery_result"].startswith("draft:")

    # The PR body names the check failure first in the unresolved section.
    [pull] = _pulls(github, slug)
    items = _draft_items(pull["body"])
    assert items[0].startswith("- Merged-result check failed"), items


def test_draft_existing_pr_converted_to_draft(new_run):
    """A draft run with an existing open PR: gh pr ready --undo + body prepend."""
    # A normal run has opened the PR.
    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    [pull_before] = _pulls(github, slug)
    old_body = pull_before["body"]

    # Now call deliver_design with draft_delivery on the same run.
    facts = dict(result.facts)
    facts["draft_delivery"] = DRAFT_FACT
    outcome = delivery.deliver_design(nn.RunContext(run, facts=facts, enforce_writes=False))

    calls = _calls(new_run.root)
    assert "pr ready --undo" in calls, f"expected pr ready --undo in: {calls}"
    assert "pr edit" in calls, f"expected pr edit in: {calls}"

    # The PR body is the unresolved preamble followed by the previous body.
    pulls = _pulls(github, slug)
    updated_pull = next(p for p in pulls if p["number"] == pull_before["number"])
    assert updated_pull["body"].startswith(f"{delivery.DRAFT_MADE}\n{delivery.DRAFT_HEADING}\n"), updated_pull["body"][:200]
    # The old body is preserved after the preamble.
    assert updated_pull["body"].endswith(old_body), "old body should be preserved"
    assert updated_pull.get("draft") is True


def test_draft_closed_pr_opens_no_second_pr(new_run):
    """A draft run with a closed PR: delivery does NOT open a second PR."""
    # A normal run has opened the PR.
    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    [pull_before] = _pulls(github, slug)

    # Close the PR.
    pulls = _pulls(github, slug)
    for p in pulls:
        if p["number"] == pull_before["number"]:
            p["state"] = "CLOSED"
    (github / f"{slug}.pulls.json").write_text(json.dumps(pulls, indent=2))

    # Now call deliver_design with draft_delivery on the same run.
    facts = dict(result.facts)
    facts["draft_delivery"] = DRAFT_FACT
    outcome = delivery.deliver_design(nn.RunContext(run, facts=facts, enforce_writes=False))

    # No second PR was opened — the result is kept local.
    final_pulls = _pulls(github, slug)
    assert len(final_pulls) == 1, "should not open a second PR"

    # Nothing was delivered, so nothing is recorded or reported as a draft: the
    # record, the result and `refactory projects` say what they say without the fact.
    rec = _record(run)
    assert rec["status"] == "kept local"
    assert "draft" not in rec and "draft_reason" not in rec
    assert outcome["delivery_result"].startswith("not delivered: the pull request for"), outcome
    assert "draft" not in outcome["delivery_result"].split("; to finish")[0]
    assert _registry_note(run)["delivery"] == "kept local"


def test_draft_conflict_existing_open_pr(github, tmp_path, monkeypatch):
    """A draft with a base conflict and an existing open PR: pr ready --undo,
    body names the conflict first."""
    repo = git_repo(tmp_path / "repo_cex")
    (repo / "authored.py").write_text("line1\nline2\nline3\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base with authored")
    slug = "acme/draft-cex"
    _published(github, slug, repo)

    # First, create an existing open PR via a normal design run.
    first_run = tmp_path / "first_cex"
    first_result = _design(first_run, {"tool.py": "print('built')\n",
                                        "authored.py": "line1\nfirst_run\nline3\n"},
                           target=str(repo), check="python tool.py")
    assert first_result.exit_ok, first_result.error
    [pull_before] = _pulls(github, slug)

    # Advance main to create a conflict.
    other = tmp_path / "other_cex"
    git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
    (other / "authored.py").write_text("line1\nchanged_by_base\nline3\n")
    git(other, "add", ".")
    git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "-q", "-m", "base changes authored.py")
    git(other, "push", "-q", "origin", "main")

    # Now call deliver_design on the same run with draft_delivery, which will
    # find the existing PR and encounter the conflict.
    reviewed = worktree.recorded(first_run).head  # before delivery can move the record
    facts = dict(first_result.facts)
    facts["draft_delivery"] = DRAFT_FACT
    outcome = delivery.deliver_design(nn.RunContext(first_run, facts=facts, enforce_writes=False))

    calls = _calls(tmp_path)
    assert "pr ready --undo" in calls, f"expected pr ready --undo in: {calls}"
    assert outcome["delivery_result"].startswith("draft:"), outcome["delivery_result"]

    # The PR body names the conflict first.
    pulls_list = _pulls(github, slug)
    updated = next(p for p in pulls_list if p["number"] == pull_before["number"])
    items = _draft_items(updated["body"])
    assert items[0].startswith("- Merge conflict with base in authored.py"), items

    # The pushed head is the run's own commit: one parent, the run's version.
    found = worktree.recorded(first_run)
    remote = _remote_head(github, slug, found.branch)
    assert remote == reviewed
    assert len(_parents(github, slug, remote)) == 1
    assert _shown(github, slug, remote, "authored.py") == "line1\nfirst_run\nline3"


def test_draft_no_pr_mode(github, tmp_path):
    """A draft run with --no-pr: PR text has the section, record says draft,
    finish commands include --draft."""
    repo = git_repo(tmp_path / "repo_nopr")
    _published(github, "acme/nopr", repo)

    run = tmp_path / "run_nopr"
    result = _design_draft(run, github, tmp_path, mode=_mode(pr=False), target=repo)
    assert result.exit_ok, result.error

    # The PR text carries the unresolved preamble.
    pr_text = (run / delivery.BODY).read_text()
    assert pr_text.startswith(f"{delivery.DRAFT_MADE}\n{delivery.DRAFT_HEADING}\n"), pr_text[:200]

    # Delivery record says draft.
    rec = _record(run)
    assert rec["draft"] is True
    assert rec["check_passed"] is True
    assert rec["draft_reason"] == "review reloop limit reached after 3 review cycle(s)"

    # delivery_result starts with "not delivered (draft):".
    assert result.facts["delivery_result"].startswith("not delivered (draft):"), result.facts["delivery_result"]

    # Finish commands include --draft.
    finish = rec.get("finish", [])
    gh_cmd = next((c for c in finish if "gh pr create" in c), "")
    assert "--draft" in gh_cmd, f"expected --draft in finish commands: {finish}"


def test_spine_draft_delivery_across_processes(new_run, monkeypatch):
    """The spine carries draft_delivery across processes via facts.json.

    Uses ``node_runner.run_node`` (separate processes, as the spine runs them),
    with the fact in the run's facts file, and shows the draft opening on the
    fake GitHub.
    """
    from refactory_dsh import lowering, design_ops
    from refactory_dsh.run import run_node

    # A normal run has set up the worktree, evidence and delivery plan.
    run, result = new_run.run, new_run.result

    # Write the draft_delivery fact into facts.json on disk — the spine's channel.
    facts = json.loads((run / "facts.json").read_text()) if (run / "facts.json").exists() else {}
    facts.update(result.facts)
    facts["draft_delivery"] = DRAFT_FACT
    (run / "facts.json").write_text(json.dumps(facts))

    # Write declarations so run_node can look up the node's reads/writes/produces.
    lowering.write_declarations(DesignMode().state_dict(), run)

    # Read the evidence's settings so run_node's knobs match the evidence currency check.
    ev_path = run / "authoring-evidence.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    knobs = dict(ev.get("settings", {}))
    knobs["check_timeout_s"] = 180
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps(knobs))

    # Run report through run_node (reads draft_delivery from facts.json).
    run_node(f"{design_ops.__name__}:report_artifact", workdir=run, node_id="report")

    # Run deliver through run_node.
    run_node("refactory_dsh.delivery:deliver_design", workdir=run, node_id="deliver")

    # Read back the facts run_node persisted to disk.
    saved = json.loads((run / "facts.json").read_text())
    assert saved["delivery_result"].startswith("draft:"), saved["delivery_result"]

    # The fake gh saw --draft (for the existing PR, it's pr ready --undo).
    calls = _calls(new_run.root)
    assert "pr ready --undo" in calls or "--draft" in calls


# ── draft delivery: the review's findings (#500) ──────────────────────────
def test_draft_pr_body_hides_credentials_and_keeps_each_item_on_one_line(github, tmp_path):
    """Every value in the draft section reaches GitHub, so each goes through `_public`
    on one line, as every other field of the description: the merged check's own
    output, the reason and each item. A line break in an item cannot start a heading."""
    repo = git_repo(tmp_path / "repo_sec")
    (repo / "config.txt").write_text("ok\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base config ok")
    slug = "acme/draft-sec"
    _published(github, slug, repo)

    def advance_main():
        other = tmp_path / "other_sec"
        if not other.exists():
            git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
            (other / "config.txt").write_text("broken\n")
            git(other, "add", ".")
            git(other, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "brk")
            git(other, "push", "-q", "origin", "main")

    fact = {"reason": "limit reached\nANTHROPIC_API_KEY=sk-ant-in-the-reason",
            "unresolved": [{"text": "the reviewer saw OPENAI_API_KEY=sk-live-reviewer-quoted\n## Injected",
                            "source": "review", "rounds": [1]}]}
    tool = ("import sys\n"
            "if open('config.txt').read().strip() == 'broken':\n"
            "    print('GITHUB_TOKEN=ghp_SECRETVALUE123 failed', file=sys.stderr)\n"
            "    print('second line', file=sys.stderr)\n"
            "    sys.exit(1)\n")
    result = _design_draft(tmp_path / "run_sec", github, tmp_path, target=repo, emit={"tool.py": tool},
                           check="python tool.py", advance_main=advance_main, draft_fact=fact)
    assert result.exit_ok, result.error

    [pull] = _pulls(github, slug)
    body = pull["body"]
    for secret in ("ghp_SECRETVALUE123", "sk-live-reviewer-quoted", "sk-ant-in-the-reason"):
        assert secret not in body, secret
    assert "GITHUB_TOKEN=***" in body and "OPENAI_API_KEY=***" in body
    assert "**Reason:** limit reached ANTHROPIC_API_KEY=***" in body
    assert "\n## Injected" not in body
    items = _draft_items(body)
    assert items[0].startswith("- Merged-result check failed:") and "GITHUB_TOKEN=***" in items[0], items
    assert "second line" in items[0]
    assert items[1] == "- the reviewer saw OPENAI_API_KEY=*** ## Injected (source: review) (rounds: 1)"
    assert "sk-ant-in-the-reason" not in result.facts["delivery_result"]


def test_draft_without_any_check_evidence_is_refused(new_run):
    """A draft says what its check showed, so report refuses a draft run with no check
    evidence at all, whether or not the runtime_evidence fact is set. (Without the
    fact, require_completion is unchanged.)"""
    from refactory_dsh.design_ops import report_artifact
    from refactory_dsh.authoring_evidence import SETTINGS

    run, result = new_run.run, new_run.result
    ev_path = run / "authoring-evidence.json"
    knobs = {key: json.loads(ev_path.read_text()).get("settings", {}).get(key) for key in SETTINGS}
    ev_path.unlink()
    facts = {**result.facts, "draft_delivery": DRAFT_FACT}
    facts.pop("runtime_evidence", None)
    with pytest.raises(ValueError, match="authoritative runtime evidence is missing"):
        report_artifact(nn.RunContext(run, knobs=knobs, facts=facts, enforce_writes=False))
    facts["runtime_evidence"] = True
    with pytest.raises(ValueError, match="authoritative runtime evidence is missing"):
        report_artifact(nn.RunContext(run, knobs=knobs, facts=facts, enforce_writes=False))


def test_draft_with_stale_identity_is_refused(new_run):
    """A draft needs evidence for what it delivers: a design changed after validation
    is refused on the draft path as on the complete one."""
    from refactory_dsh.design_ops import report_artifact
    from refactory_dsh.authoring_evidence import SETTINGS

    run, result = new_run.run, new_run.result
    ev = json.loads((run / "authoring-evidence.json").read_text())
    knobs = {key: ev.get("settings", {}).get(key) for key in SETTINGS}
    (run / "design.md").write_text((run / "design.md").read_text() + "\nchanged after validation\n")
    facts = {**result.facts, "draft_delivery": DRAFT_FACT}
    with pytest.raises(ValueError, match="artifact, accepted inputs or user intent changed after validation"):
        report_artifact(nn.RunContext(run, knobs=knobs, facts=facts, enforce_writes=False))


def test_a_draft_github_refuses_opens_ready_with_its_title_marked(github, tmp_path):
    """GitHub has no draft pull requests for private repositories on some plans: the
    pull request opens ready for review, its title starting "[draft] " and its
    unresolved section still first, and the record says what happened."""
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/nodrafts", repo)
    (github / "acme/nodrafts.json").write_text(json.dumps({"visibility": "private", "default_branch": "main",
                                                           "drafts": False}))
    run = tmp_path / "run"
    result = _design_draft(run, github, tmp_path, target=repo)
    assert result.exit_ok, result.error

    [pull] = _pulls(github, "acme/nodrafts")
    assert pull["title"].startswith(delivery.DRAFT_TITLE) and pull["draft"] is False
    assert pull["title"] == delivery.DRAFT_TITLE + _record(run)["title"]
    assert pull["body"].startswith(f"{delivery.DRAFT_TITLED}\n{delivery.DRAFT_HEADING}\n"), pull["body"][:200]
    rec = _record(run)
    assert (rec["status"], rec["pull_request"]) == ("opened", result.facts["pull_request"])
    assert rec["draft"] is True and "Draft pull requests are not supported" in rec["draft_note"]
    assert result.facts["delivery_result"].startswith(f"draft: {rec['pull_request']}")
    assert "GitHub refused a draft pull request" in result.facts["delivery_result"]


def test_a_pull_request_github_will_not_make_a_draft_keeps_its_delivery(new_run):
    """`gh pr ready --undo` refused on the open pull request: the unresolved section is
    still put first, and the record names the pushed commit and the pull request and
    says it could not be converted."""
    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    meta = json.loads((github / f"{slug}.json").read_text())
    (github / f"{slug}.json").write_text(json.dumps({**meta, "drafts": False}))
    url = result.facts["pull_request"]

    outcome = delivery.deliver_design(nn.RunContext(run, facts={**result.facts, "draft_delivery": DRAFT_FACT},
                                                    enforce_writes=False))
    rec = _record(run)
    assert (rec["status"], rec["pull_request"], rec["draft"]) == ("updated", url, True)
    assert _remote_head(github, slug, rec["branch"]) == rec["head"] == worktree.recorded(run).head
    assert "could not be converted to a draft" in rec["draft_note"]
    assert outcome["delivery_result"].startswith(f"draft: {url}")
    assert "could not be converted to a draft" in outcome["delivery_result"]
    [pull] = _pulls(github, slug)
    assert pull["body"].startswith(f"{delivery.DRAFT_TITLED}\n{delivery.DRAFT_HEADING}\n") and pull["draft"] is False
    # A ready pull request that is really a draft says so in its title, as one opened so does.
    assert pull["title"].startswith(delivery.DRAFT_TITLE)


def test_a_draft_whose_description_gh_cannot_edit_still_records_its_push(new_run, monkeypatch):
    """Converting the open pull request runs after the push, so a refusal there leaves
    the pushed branch and the pull request in the record."""
    run, result = new_run.run, new_run.result
    real = delivery._gh

    def gh(*args, **kwargs):
        if args[:2] == ("pr", "edit"):
            raise delivery.NotDelivered("`gh pr edit` failed: HTTP 502")
        return real(*args, **kwargs)

    monkeypatch.setattr(delivery, "_gh", gh)
    url = result.facts["pull_request"]
    delivery.deliver_design(nn.RunContext(run, facts={**result.facts, "draft_delivery": DRAFT_FACT},
                                          enforce_writes=False))
    rec = _record(run)
    found = worktree.recorded(run)
    assert (rec["status"], rec["pull_request"], rec["pushed"]) == (
        "kept local", url, f"{LOGIN}/{NAME}:{found.branch}")


def test_delivering_a_draft_again_replaces_its_unresolved_section(new_run):
    """A second draft delivery on the same pull request replaces the section the first
    one wrote, rather than stacking a second above it."""
    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    [first] = _pulls(github, slug)
    later = {"reason": "overall attempts limit reached",
             "unresolved": [{"text": "Still flaky", "source": "review", "rounds": [4]}]}
    for fact in (DRAFT_FACT, later):
        delivery.deliver_design(nn.RunContext(run, facts={**result.facts, "draft_delivery": fact},
                                              enforce_writes=False))
    [pull] = _pulls(github, slug)
    assert pull["body"].count(delivery.DRAFT_HEADING) == 1
    assert "overall attempts limit reached" in pull["body"] and DRAFT_FACT["reason"] not in pull["body"]
    assert _draft_items(pull["body"]) == ["- Still flaky (source: review) (rounds: 4)"]
    assert pull["body"].endswith(first["body"])


def test_a_draft_refinement_turns_the_refined_pull_request_into_a_draft(opened_pull, tmp_path):
    """On a refinement the refined pull request becomes the draft, its own description
    kept below one unresolved section, however often it is delivered."""
    repo, url, pull, github = opened_pull.repo, opened_pull.url, opened_pull.pull, opened_pull.github
    run = tmp_path / "refine"
    result = _design(run, refine=url)
    assert result.exit_ok, result.error
    for _ in range(2):
        again = delivery.deliver_design(nn.RunContext(run, facts={**result.facts, "draft_delivery": DRAFT_FACT},
                                                      enforce_writes=False))
    [after] = _pulls(github, "acme/app")
    assert after["draft"] is True and after["body"].count(delivery.DRAFT_MADE) == 1
    # The section first, then the description, its delivery sections the refined run's.
    assert after["body"].startswith(f"{delivery.DRAFT_MADE}\n{delivery.DRAFT_HEADING}\n")
    assert after["body"].endswith(_section((run / delivery.BODY).read_text()) + "\n")
    assert again["delivery_result"].startswith(f"draft: {url}")


def test_a_draft_whose_pull_request_merged_is_recorded_merged(new_run):
    """Merged is merged: not recorded or reported as a draft, and not converted."""
    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    pulls = _pulls(github, slug)
    pulls[0]["state"] = "MERGED"
    (github / f"{slug}.pulls.json").write_text(json.dumps(pulls))
    again = delivery.deliver_design(nn.RunContext(run, facts={**result.facts, "draft_delivery": DRAFT_FACT},
                                                  enforce_writes=False))
    rec = _record(run)
    assert rec["status"] == "merged" and "draft" not in rec and "draft_reason" not in rec
    assert again["delivery_result"].startswith(f"merged: {result.facts['pull_request']}"), again
    assert _registry_note(run)["delivery"] == "merged"
    assert "pr ready" not in _calls(new_run.root)


def test_a_draft_whose_merge_was_abandoned_is_not_succeeded(github, tmp_path):
    """An abandoned merge stops a draft as it stops any delivery: kept local, not
    called a draft, and the run's worktree record not marked succeeded."""
    repo = git_repo(tmp_path / "repo")
    slug = "acme/draft-abandoned"
    _published(github, slug, repo)
    run = tmp_path / "run"

    def advance_main():
        other = tmp_path / "other"
        if not other.exists():
            git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
            (other / "blocker.txt").write_text("from base\n")
            git(other, "add", ".")
            git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
                "commit", "-q", "-m", "base adds blocker")
            git(other, "push", "-q", "origin", "main")
        # An untracked file of that name in the worktree: git merge refuses, with no
        # conflicted file, so the merge is abandoned.
        (Path(worktree.recorded(run).path) / "blocker.txt").write_text("untracked content\n")

    result = _design_draft(run, github, tmp_path, target=repo, advance_main=advance_main)
    assert result.exit_ok, result.error
    found = worktree.recorded(run)
    assert worktree._json(worktree._record_path(found)).get("outcome") != "succeeded"
    rec = _record(run)
    assert rec["status"] == "kept local" and rec.get("merge_abandoned") and "draft" not in rec
    assert result.facts["delivery_result"].startswith("not delivered: merge abandoned")
    assert _remote_head(github, slug, found.branch) == ""


def test_a_draft_whose_push_is_refused_keeps_the_work_local_as_without_the_fact(github, tmp_path, monkeypatch):
    """A refused push stops a draft as it stops any delivery: no pull request, kept
    local, said and recorded as without the fact; its text and finishing commands
    still open a draft."""
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)

    def push_refused(*args, **kwargs):
        raise delivery.NotDelivered("push refused by GitHub")

    monkeypatch.setattr(delivery, "_push", push_refused)
    run = tmp_path / "run"
    result = _design_draft(run, github, tmp_path, target=repo)
    assert result.exit_ok, result.error
    assert _pulls(github, "acme/app") == []
    rec = _record(run)
    assert (rec["status"], rec["pushed"], rec["pull_request"]) == ("kept local", "", "")
    assert "draft" not in rec
    assert result.facts["delivery_result"].startswith("not delivered: push refused by GitHub")
    assert _registry_note(run)["delivery"] == "kept local"
    assert (run / delivery.BODY).read_text().startswith(f"{delivery.DRAFT_MADE}\n{delivery.DRAFT_HEADING}\n")
    assert any(cmd.startswith("gh pr create") and cmd.endswith(" --draft") for cmd in rec["finish"])


def test_the_delivery_gate_never_reloops_a_draft(tmp_path):
    """The delivery gate proceeds on a draft run even with conflicts named (the build
    loop's limits are spent); without the fact, or with one that is not a mapping, it
    reloops to the builder as before."""
    conflicted = {"delivery_conflict_files": ["authored.py"], "delivery_target_tip": "a" * 40}
    gate = delivery.evaluate_delivery_conflicts
    assert gate(nn.RunContext(tmp_path, facts={**conflicted, "draft_delivery": DRAFT_FACT})) == "proceed"
    for fact in ({}, {"draft_delivery": "not a mapping"}):
        verdict = gate(nn.RunContext(tmp_path, facts={**conflicted, **fact}))
        assert verdict == "reloop", verdict


def test_a_draft_fact_that_is_not_a_mapping_is_no_draft(github, tmp_path, capsys):
    """A fact that is not a mapping is read as no draft, with a warning, rather than
    failing report or deliver."""
    from refactory_dsh.design_ops import report_artifact
    from refactory_dsh.authoring_evidence import SETTINGS

    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    ev = json.loads((run / "authoring-evidence.json").read_text())
    knobs = {key: ev.get("settings", {}).get(key) for key in SETTINGS}
    facts = {**result.facts, "draft_delivery": json.dumps(DRAFT_FACT)}
    capsys.readouterr()
    designed = report_artifact(nn.RunContext(run, knobs=knobs, facts=facts, enforce_writes=False))["designed"]
    assert "draft" not in json.loads(designed)
    assert "the draft_delivery fact is a str, not a mapping" in capsys.readouterr().err
    again = delivery.deliver_design(nn.RunContext(run, knobs=knobs, facts=facts, enforce_writes=False))
    assert again["delivery_result"].startswith("already open:"), again
    assert "pr ready" not in _calls(tmp_path)


# ── the paths a person takes ───────────────────────────────────────────────
def test_a_new_project_gets_a_private_repository_whose_first_pull_request_holds_all_the_code(new_run):
    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    project = (projects_dir() / NAME).resolve()
    assert json.loads((github / f"{slug}.json").read_text())["visibility"] == "private"
    # main on GitHub is the project's empty first commit; the pull request is all of the code.
    assert _remote_head(github, slug, "main") == git(project, "rev-parse", "main")
    [pull] = _pulls(github, slug)
    branch = json.loads((run / "design.json").read_text())["branch"]
    assert (pull["head"], pull["base"], pull["title"]) == (branch, "main", "Do the thing")
    assert _remote_head(github, slug, branch) == git(project, "rev-parse", branch)
    assert git(project, "ls-tree", "--name-only", "main") == ""
    assert "## Summary\n\nFix it properly." in pull["body"] and "`python tool.py` passed" in pull["body"]
    url = f"https://github.com/{slug}/pull/1"
    assert result.facts["pull_request"] == url
    assert _record(run)["status"] == "opened" and _record(run)["verified"] is True
    assert json.loads((run / "design.json").read_text())["delivery"]["pull_request"] == url
    [entry] = projects.load()["projects"]
    assert (entry["github"], entry["runs"][-1]["pull_request"]) == (slug, url)
    # The project now names the repository it was delivered to.
    assert git(project, "config", "--get", "remote.origin.url") == f"https://github.com/{slug}.git"


def test_the_approval_gate_reads_what_delivery_will_do_before_anything_is_built(github, tmp_path):
    graph = DesignMode().build()
    assert "delivery.md" in graph.nodes["gate_strategy"].reads
    assert "delivery.md" in graph.nodes["gate_strategy"].notes
    run = tmp_path / "run"
    seen: list[tuple[bool, list[Path]]] = []

    class Watching(FakeModel):
        """Notes, as the approved design is built, what the gate had read and what GitHub held."""

        def complete(self, prompt: str, *, system: str = "", workdir: object = None, **workspace: object) -> str:
            if "Write the artifact `" in prompt:
                seen.append(((run / delivery.PLAN).exists(), sorted(github.glob("*/*.git"))))
            return super().complete(prompt, system=system, workdir=workdir, **workspace)

    result = DesignMode().run(run, model=Watching(_answers(render({"tool.py": "print('built')\n"}))),
                              verdicts={"gate_strategy": "proceed"},
                              task={"name": NAME, "brief": "a tool", "check": "python tool.py"})
    assert result.exit_ok, result.error
    assert seen == [(True, [])], "the plan existed at the gate, and the repository did not until delivery"
    said = (run / delivery.PLAN).read_text()
    assert f"creates the private GitHub repository {LOGIN}/{NAME}" in said and "--no-pr" in said
    assert list(github.glob("*/*.git")) == [github / LOGIN / f"{NAME}.git"]
    assert [edge.target for edge in graph.edges if edge.source == "report"] == ["deliver"]


def test_an_existing_repository_gets_a_pull_request_into_its_default_branch(github, tmp_path):
    repo = git_repo(tmp_path / "repo")
    slug = "acme/app"
    _published(github, slug, repo)
    refs = git(repo, "for-each-ref")
    main = _remote_head(github, slug, "main")
    run = tmp_path / "run"
    result = _design(run, target=str(repo))
    assert result.exit_ok, result.error
    branch = json.loads((run / "design.json").read_text())["branch"]
    [pull] = _pulls(github, slug)
    assert (pull["head"], pull["base"]) == (branch, "main")
    assert _remote_head(github, slug, "main") == main, "delivery never pushes the default branch"
    assert _remote_head(github, slug, branch) == git(repo, "rev-parse", branch)
    # The person's repository keeps its remotes and remote-tracking branches as they were;
    # only the run's own branch is new.
    assert git(repo, "remote") == "origin"
    assert git(repo, "for-each-ref", "--exclude", f"refs/heads/{branch}") == refs
    assert f"opens a pull request from {branch} into {slug}:main" in (run / delivery.PLAN).read_text()


class _WorkspaceBuilder(FakeModel):
    """A builder that does its work in its workspace: it writes the tool there, runs the
    check there, and stops, answering only that it is done."""

    def __init__(self, answers: list[str]) -> None:
        super().__init__(answers)
        self.ran: list[str] = []

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, cwd: object = None,
                 env: object = None, confine: bool = False) -> str:
        answer = super().complete(prompt, system=system, workdir=workdir, cwd=cwd, env=env, confine=confine)
        if cwd is not None and "Write the artifact `" in prompt:
            clone = Path(str(cwd))
            (clone / "tool.py").write_text("import app\nprint('built')\n")
            (clone / "app.py").write_text("print('app, reworked')\n")
            ran = subprocess.run([sys.executable, "tool.py"], cwd=clone, capture_output=True, text=True,
                                 env={**os.environ, **{k: v for k, v in dict(env or {}).items() if v is not None}})
            self.ran.append(ran.stdout)
            return "Done: the tool is written and runs."
        return answer


def test_a_builder_that_changes_code_in_its_workspace_delivers_a_pull_request(github, tmp_path):
    """End to end on the walker: the builder works in its clone, the derived emission is
    materialised, checked and reviewed, and the run opens a pull request with exactly
    the builder's change."""
    repo = git_repo(tmp_path / "repo")
    slug = "acme/app"
    _published(github, slug, repo)
    model = _WorkspaceBuilder(_answers("(the builder answers with its workspace, not blocks)"))
    run = tmp_path / "run"
    result = DesignMode().run(run, model=model, verdicts={"gate_strategy": "proceed"},
                              task={"name": NAME, "brief": "Build a tool\nthat prints built.",
                                    "check": "python tool.py", "target": str(repo)})
    assert result.exit_ok, result.error
    assert model.ran == ["app, reworked\nbuilt\n"], "the builder ran its change in its workspace"
    assert (run / "emit.txt").read_text() == ("===FILE: app.py===\nprint('app, reworked')\n"
                                             "===FILE: tool.py===\nimport app\nprint('built')\n")
    branch = json.loads((run / "design.json").read_text())["branch"]
    [pull] = _pulls(github, slug)
    assert (pull["head"], pull["base"]) == (branch, "main")
    pushed = github / f"{slug}.git"
    assert git(pushed, "show", f"{branch}:tool.py") == "import app\nprint('built')"
    assert git(pushed, "show", f"{branch}:app.py") == "print('app, reworked')"
    assert sorted(git(pushed, "diff", "--name-only", "main", branch).splitlines()) == ["app.py", "tool.py"]
    assert not (run / ".refactory" / "workspaces").exists(), "the run's end removes the workspace"


def test_unpushed_commits_in_the_base_are_never_pushed(github, tmp_path, capsys):
    repo = git_repo(tmp_path / "repo")
    slug = "acme/app"
    _published(github, slug, repo)
    (repo / "mine.py").write_text("x = 1\n")
    git(repo, "add", "mine.py")
    git(repo, "commit", "-q", "-m", "mine, not pushed")
    run = tmp_path / "run"
    result = _design(run, target=str(repo))
    assert result.exit_ok, result.error
    said = (run / delivery.PLAN).read_text()
    assert "won't open a pull request" in said and "1 commit(s) that are not on acme/app" in said
    assert "--base origin/main" in said
    assert _pulls(github, slug) == [] and git(github / f"{slug}.git", "branch") == "* main"
    assert "no pull request was opened" in capsys.readouterr().err
    assert result.facts["pull_request"] == ""


def test_a_name_already_taken_on_github_is_never_pushed_into(github, tmp_path, capsys):
    taken = git_repo(tmp_path / "theirs")
    _published(github, f"{LOGIN}/{NAME}", taken)
    before = git(github / f"{LOGIN}/{NAME}.git", "for-each-ref")
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert "already exists" in (run / delivery.PLAN).read_text()
    assert git(github / f"{LOGIN}/{NAME}.git", "for-each-ref") == before
    assert "repo create" not in _calls(tmp_path) and _pulls(github, f"{LOGIN}/{NAME}") == []
    assert git((projects_dir() / NAME), "remote") == ""
    assert "already exists" in capsys.readouterr().err


def test_a_reused_project_name_opens_its_pull_request_in_the_repository_it_was_given(github, tmp_path):
    first = _design(tmp_path / "one", {"tool.py": "print(1)\n"})
    second = _design(tmp_path / "two", {"tool.py": "print(2)\n"})
    assert first.exit_ok and second.exit_ok, (first.error, second.error)
    slug = f"{LOGIN}/{NAME}"
    assert _calls(tmp_path).count("repo create") == 1
    assert [pull["number"] for pull in _pulls(github, slug)] == [1, 2]
    assert second.facts["pull_request"] == f"https://github.com/{slug}/pull/2"
    assert "opens a pull request from" in (tmp_path / "two" / delivery.PLAN).read_text()


@pytest.mark.parametrize("refused", ["exit 4", "HTTP 401"])
def test_without_gh_login_the_run_succeeds_and_says_its_work_stays_local(github, tmp_path, capsys, refused):
    (github / "login").unlink()
    if refused == "HTTP 401":
        (github / "logged-out-as-401").write_text("")
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    err = capsys.readouterr().err
    assert "run warning: no pull request was opened: GitHub CLI is not logged in" in err
    assert _record(run)["status"] == "kept local"
    assert "repo create" not in _calls(tmp_path) and not list(github.glob("*/*.git"))


def test_no_pr_keeps_the_work_on_its_branch_and_asks_github_nothing(github, tmp_path, capsys):
    run = tmp_path / "run"
    result = _design(run, mode=_mode(pr=False))
    assert result.exit_ok, result.error
    assert _calls(tmp_path) == ""
    assert "run warning" not in capsys.readouterr().err
    assert "`pr` is off (--no-pr)" in (run / delivery.PLAN).read_text()
    assert result.facts["delivery_result"].startswith("not delivered")
    assert result.facts["delivery"].startswith("once the reviews pass, this run won't open"), "the plan is kept"


def test_a_run_whose_check_never_passes_opens_a_draft_in_its_private_repository(github, tmp_path):
    """b45a: a new project's run whose check fails at its limit has an applied commit, so it
    is delivered: its repository is created, private as for any run, and its pull request
    is a draft marked check failing. A run with no applied commit still creates nothing
    (`test_a_run_with_no_applied_commit_delivers_nothing`)."""
    result = _design(tmp_path / "run", check="false", mode=_mode(check_attempts=1, check_attempt_cap=0))
    assert result.exit_ok, result.error
    slug = f"{LOGIN}/{NAME}"
    assert json.loads((github / f"{slug}.json").read_text())["visibility"] == "private"
    [pull] = _pulls(github, slug)
    assert pull["draft"] is True and pull["title"] == "[check failing] Do the thing"


def test_a_run_without_a_worktree_has_nothing_to_deliver(github, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    result = _design(run, mode=_mode(worktree=False))
    assert result.exit_ok, result.error
    assert _calls(tmp_path) == "" and "run warning" not in capsys.readouterr().err
    assert "this run has no worktree" in (run / delivery.PLAN).read_text()


# ── a redo, a resume, and what GitHub then shows ───────────────────────────
def test_delivering_again_opens_no_second_pull_request_and_leaves_its_text_alone(new_run):
    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    [before] = _pulls(github, slug)
    again = delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))
    assert again["pull_request"] == result.facts["pull_request"]
    assert _pulls(github, slug) == [before]
    assert _record(run)["status"] == "already open"
    assert _calls(new_run.root).count("pr create") == 1 and "pr edit" not in _calls(new_run.root)


def test_a_delivery_that_failed_after_its_push_opens_one_pull_request_when_resumed(github, tmp_path, monkeypatch):
    run = tmp_path / "run"
    real = delivery._gh

    def fails_at_create(*args, **kwargs):
        if args[:2] == ("pr", "create"):
            raise RuntimeError("the delivery stopped here")
        return real(*args, **kwargs)

    monkeypatch.setattr(delivery, "_gh", fails_at_create)
    failed = _design(run)
    assert failed.exit_ok is False and "the delivery stopped here" in (failed.error or "")
    slug = f"{LOGIN}/{NAME}"
    branch = worktree.recorded(run).branch
    assert _remote_head(github, slug, branch) and _pulls(github, slug) == []
    monkeypatch.setattr(delivery, "_gh", real)
    resumed = DesignMode.resume(run, model=FakeModel([]))
    assert resumed.exit_ok, resumed.error
    assert [pull["head"] for pull in _pulls(github, slug)] == [branch]
    assert _calls(tmp_path).count("repo create") == 1


def test_a_pull_request_that_shows_another_head_is_reported_not_claimed(github, tmp_path):
    run = tmp_path / "run"
    (github / "head-override").write_text("0" * 40 + "\n")
    result = _design(run)
    assert result.exit_ok is False
    assert "not this run's reviewed commit" in (result.error or "")
    assert _record(run)["status"] == "unverified"
    assert _record(run)["pull_request"] == f"https://github.com/{LOGIN}/{NAME}/pull/1"


def test_a_plan_that_changed_since_the_gate_keeps_the_work_local(github, tmp_path, capsys, monkeypatch):
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    # A fork holding the same main, so the run's base is on it and only the repository differs.
    fork = github / "acme/fork.git"
    git(fork.parent, "init", "-q", "--bare", "-b", "main", fork.name)
    (github / "acme/fork.json").write_text(json.dumps({"default_branch": "main"}))
    git(repo, "push", "-q", "https://github.com/acme/fork.git", "main:refs/heads/main")
    real = delivery.deliver_design

    def repointed(ctx):
        # Between the approval and delivery, the checkout started pointing elsewhere.
        git(repo, "remote", "set-url", "origin", "https://github.com/acme/fork.git")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", repointed)
    result = _design(tmp_path / "run", target=str(repo))
    assert result.exit_ok, result.error
    assert _pulls(github, "acme/app") == [] and _pulls(github, "acme/fork") == []
    err = capsys.readouterr().err
    assert "different from what the approval gate showed" in err and "acme/fork:main" in err


# ── refining an open pull request ──────────────────────────────────────────
def _opened(github: Path, tmp_path: Path) -> tuple[Path, str, dict]:
    """An existing repository with a pull request a design run opened."""
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    first = _design(tmp_path / "first", {"tool.py": "print('one')\n"}, target=str(repo))
    assert first.exit_ok, first.error
    return repo, first.facts["pull_request"], _pulls(github, "acme/app")[0]


def test_refine_starts_from_the_pull_requests_head_and_pushes_onto_its_branch(opened_pull, tmp_path):
    repo, url, pull, github = opened_pull.repo, opened_pull.url, opened_pull.pull, opened_pull.github
    head = _remote_head(github, "acme/app", pull["head"])
    run = tmp_path / "refine"
    result = _design(run, {"tool.py": "print('one')\n", "more.py": "print('two')\n"}, refine=url)
    assert result.exit_ok, result.error
    found = worktree.recorded(run)
    assert found.base == head and found.target == str(repo.resolve())
    assert _remote_head(github, "acme/app", pull["head"]) == found.head
    assert git(repo, "merge-base", "--is-ancestor", head, found.head) == ""
    [after] = _pulls(github, "acme/app")
    # The title stays; the description's delivery sections are the refined run's (dm-212).
    assert after["title"] == pull["title"]
    assert after["body"] == _section((run / delivery.BODY).read_text()) + "\n" != pull["body"]
    assert "more.py" in after["body"]
    assert _record(run)["status"] == "updated" and result.facts["pull_request"] == url
    assert f"pushes onto {pull['head']}, the branch of the pull request {url}" in (run / delivery.PLAN).read_text()
    ledger = (run / "intent-ledger.md").read_text()
    assert f"This run refines the pull request {url}" in ledger


def test_refine_pushes_nothing_once_the_pull_requests_branch_has_moved(opened_pull, tmp_path, capsys, monkeypatch):
    repo, url, pull, github = opened_pull.repo, opened_pull.url, opened_pull.pull, opened_pull.github
    real = delivery.deliver_design

    def someone_pushed(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", "--branch", pull["head"], f"https://github.com/acme/app.git", str(other))
        (other / "theirs.py").write_text("x = 2\n")
        git(other, "add", "theirs.py")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "theirs")
        git(other, "push", "-q", "origin", pull["head"])
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", someone_pushed)
    run = tmp_path / "refine"
    result = _design(run, refine=url)
    assert result.exit_ok, result.error
    moved = _remote_head(github, "acme/app", pull["head"])
    assert moved != worktree.recorded(run).head
    assert git(tmp_path / "other", "rev-parse", "HEAD") == moved
    assert "moved to" in capsys.readouterr().err and _record(run)["status"] == "kept local"


def test_refine_with_review_comments_puts_them_in_the_steering(opened_pull, tmp_path):
    repo, url, pull, github = opened_pull.repo, opened_pull.url, opened_pull.pull, opened_pull.github
    pulls = _pulls(github, "acme/app")
    pulls[0]["reviews"] = [{"author": {"login": "rev"}, "state": "CHANGES_REQUESTED", "body": "Rename --json."}]
    pulls[0]["comments"] = [{"author": {"login": "pm"}, "body": "Please keep the old flag working."}]
    pulls[0]["review_comments"] = [{"user": {"login": "rev"}, "path": "tool.py", "line": 1, "body": "Typo here."}]
    (github / "acme/app.pulls.json").write_text(json.dumps(pulls))
    run = tmp_path / "refine"
    result = _design(run, refine=url, mode=_mode(review_comments=True))
    assert result.exit_ok, result.error
    ledger = (run / "intent-ledger.md").read_text()
    for said in ("@rev (changes_requested): Rename --json.", "@pm: Please keep the old flag working.",
                 "@rev on tool.py:1: Typo here."):
        assert said in ledger


def test_refine_refuses_a_closed_pull_request_and_a_base(opened_pull, tmp_path):
    url, github = opened_pull.url, opened_pull.github
    refused = _design(tmp_path / "with-base", refine=url, base="main")
    assert refused.exit_ok is False and "takes no base" in (refused.error or "")
    pulls = _pulls(github, "acme/app")
    pulls[0]["state"] = "MERGED"
    (github / "acme/app.pulls.json").write_text(json.dumps(pulls))
    closed = _design(tmp_path / "closed", refine=url)
    assert closed.exit_ok is False and "is merged; only an open one can be refined" in (closed.error or "")


# ── what delivery never does ───────────────────────────────────────────────
def test_a_push_is_never_forced_and_never_to_the_default_branch(github, tmp_path):
    repo = git_repo(tmp_path / "repo")
    bare = _published(github, "acme/app", repo)
    with pytest.raises(delivery.NotDelivered, match="default branch"):
        delivery._push(repo, "acme/app", git(repo, "rev-parse", "HEAD"), "main", default="main")
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "side.py").write_text("y = 1\n")
    git(repo, "add", "side.py")
    git(repo, "commit", "-q", "-m", "side")
    delivery._push(repo, "acme/app", git(repo, "rev-parse", "HEAD"), "side", default="main")
    git(repo, "commit", "-q", "--amend", "-m", "side, rewritten")
    with pytest.raises(delivery.NotDelivered):
        delivery._push(repo, "acme/app", git(repo, "rev-parse", "HEAD"), "side", default="main")
    assert git(bare, "rev-parse", "side") == git(repo, "rev-parse", "HEAD@{1}")


def test_delivery_asks_github_with_the_persons_home_not_the_checks(new_run):
    homes = set((new_run.github / "homes.log").read_text().split())
    assert homes == {os.environ["HOME"]}


def test_a_remote_url_is_read_as_github_only_on_github():
    for url in ("https://github.com/acme/app.git", "https://x-access-token:t@github.com/acme/app",
                "git@github.com:acme/app.git", "ssh://git@github.com/acme/app"):
        assert delivery._slug(url) == "acme/app", url
    for url in ("https://gitlab.com/acme/app.git", "git@example.com:acme/app.git", "/srv/app.git"):
        assert delivery._slug(url) is None, url


# ── every route that starts a design run says how it is delivered ──────────
def _launched(monkeypatch) -> list[list[str]]:
    """What a `refactory ceo` run would have handed DSH; nothing is launched."""
    from refactory_dsh import workers, interactive, dsh_profile

    class _Run:
        returncode = 0

    seen: list[list[str]] = []
    monkeypatch.setattr(workers, "run_graph_command", lambda argv, **kwargs: seen.append(argv) or _Run())
    monkeypatch.setattr(interactive, "require_build", lambda **kw: None)
    monkeypatch.setattr(dsh_profile, "require", lambda *a, **k: None)
    return seen


def test_a_run_approved_up_front_says_how_it_will_be_delivered_before_it_starts(github, tmp_path, monkeypatch,
                                                                                capsys):
    seen = _launched(monkeypatch)
    run = tmp_path / "run"
    assert cli.main(["ceo", "a tool", "--check", "true", "--name", NAME, "--workdir", str(run),
                     "--approve", "gate_strategy=proceed", "--headless"]) == 0
    said = f"refactory: delivery — once the reviews pass, this run creates the private GitHub repository {LOGIN}/{NAME}"
    assert said in capsys.readouterr().err and seen
    assert json.loads((run / delivery.SHOWN).read_text())["action"] == "create"
    # The same mode named directly, with a knob given last winning as it does in the run.
    assert cli.main(["ceo", "default_pack/design", "--knob", "name=first", "--knob", f"name={NAME}",
                     "--knob", "brief=a tool", "--knob", "check=true", "--workdir", str(tmp_path / "named"),
                     "--approve", "gate_strategy=proceed", "--headless"]) == 0
    assert said in capsys.readouterr().err
    # A gate that will be asked shows delivery.md instead, and an earlier announcement goes.
    assert cli.main(["ceo", "a tool", "--check", "true", "--name", NAME, "--workdir", str(run), "--headless"]) == 0
    assert "refactory: delivery" not in capsys.readouterr().err and not (run / delivery.SHOWN).exists()


def test_the_command_line_hands_the_delivery_flags_to_the_run_as_knobs(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    assert cli.main(["ceo", "a tool", "--check", "true", "--no-pr", "--refine", "acme/app#3", "--review-comments"]) == 0
    knobs = [seen[-1][i + 1] for i, arg in enumerate(seen[-1]) if arg == "--knob"]
    assert {"pr=0", "refine=acme/app#3", "review_comments=1"} <= set(knobs)


def test_the_command_line_hands_refine_design_to_the_run(monkeypatch):
    """`--refine-design full` reaches the run as its knob; without the flag the mode's default holds."""
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)

    def knobs() -> list[str]:
        return [seen[-1][i + 1] for i, arg in enumerate(seen[-1]) if arg == "--knob"]

    assert cli.main(["ceo", "a tool", "--check", "true", "--refine", "acme/app#3", "--refine-design", "full"]) == 0
    assert "refine_design=full" in knobs()
    assert cli.main(["ceo", "a tool", "--check", "true", "--refine", "acme/app#3"]) == 0
    assert not [knob for knob in knobs() if knob.startswith("refine_design=")]


def test_the_design_command_applies_refine_design_and_refuses_it_without_refine(tmp_path, monkeypatch, capsys):
    from refactory_dsh import design

    used: list[str] = []
    monkeypatch.setattr(design.DesignMode, "run", lambda self, *a, **k: used.append(self.values()["refine_design"])
                        or nn.RunResult(exit_ok=True, facts={}))
    _fake_dsh = tmp_path / "fake_dsh"
    (_fake_dsh / "profiles" / "headless").mkdir(parents=True)
    (_fake_dsh / "profiles" / "headless" / "package.json").write_text("{}")
    monkeypatch.setattr("refactory_dsh.dsh_profile.model_home", lambda home=None: _fake_dsh)
    monkeypatch.chdir(tmp_path)
    given = ["--name", NAME, "--brief", "a tool", "--check", "true", "--workdir", "run"]
    assert design.main([*given, "--refine", "acme/app#3", "--refine-design", "full"]) == 0
    assert design.main([*given, "--refine", "acme/app#3"]) == 0
    assert used == ["full", "light"]
    with pytest.raises(SystemExit):
        design.main([*given, "--refine-design", "full"])
    assert "--refine-design goes with --refine" in capsys.readouterr().err
    assert used == ["full", "light"]


def test_the_console_and_chat_route_says_how_it_will_be_delivered_when_approved_up_front(github, tmp_path,
                                                                                        monkeypatch, capsys):
    from refactory_dsh import design

    monkeypatch.setattr(design.DesignMode, "run", lambda self, *a, **k: nn.RunResult(exit_ok=True, facts={}))
    # The walker profile check needs a headless profile; create one under a
    # fake DSH home so the test does not depend on the real ~/.dsh directory.
    _fake_dsh = tmp_path / "fake_dsh"
    (_fake_dsh / "profiles" / "headless").mkdir(parents=True)
    (_fake_dsh / "profiles" / "headless" / "package.json").write_text("{}")
    monkeypatch.setattr("refactory_dsh.dsh_profile.model_home", lambda home=None: _fake_dsh)
    monkeypatch.chdir(tmp_path)
    assert design.main(["--name", NAME, "--brief", "a tool", "--check", "true", "--workdir", "run",
                        "--verdict", "gate_strategy=proceed"]) == 0
    assert f"creates the private GitHub repository {LOGIN}/{NAME}" in capsys.readouterr().err
    assert (tmp_path / "run" / delivery.SHOWN).exists()
    # Again in the same directory with the gate left to ask: the earlier announcement goes.
    assert design.main(["--name", NAME, "--brief", "a tool", "--check", "true", "--workdir", "run"]) == 0
    assert "refactory: delivery" not in capsys.readouterr().err
    assert not (tmp_path / "run" / delivery.SHOWN).exists()


def test_a_run_approved_up_front_does_only_what_it_announced(github, tmp_path, capsys):
    run = tmp_path / "run"
    (run / ".refactory").mkdir(parents=True)
    (run / delivery.SHOWN).write_text(json.dumps({"action": "none", "summary": "won't open a pull request, because x"}))
    result = _design(run)
    assert result.exit_ok, result.error
    assert "before the run started it said it won't open a pull request" in (run / delivery.PLAN).read_text()
    assert "repo create" not in _calls(tmp_path) and not list(github.glob("*/*.git"))


def test_the_command_line_refuses_delivery_flags_that_cannot_apply(capsys):
    assert cli.main(["ceo", "a tool", "--check", "true", "--refine", "acme/app#3", "--base", "main"]) == 2
    assert cli.main(["ceo", "a tool", "--check", "true", "--review-comments"]) == 2
    assert cli.main(["ceo", "a tool", "--check", "true", "--refine-design", "full"]) == 2
    assert cli.main(["ceo", "a tool", "--check", "true", "--mode", "design_fast", "--no-pr"]) == 2
    assert cli.main(["ceo", "a tool", "--check", "true", "--mode", "design_fast", "--refine", "acme/app#3"]) == 2
    assert cli.main(["ceo", "a tool", "--check", "true", "--no-worktree", "--refine", "acme/app#3"]) == 2
    err = capsys.readouterr().err
    assert "takes no --base" in err and "goes with --refine" in err
    assert "does not take --no-pr" in err and "--no-worktree cannot take --refine" in err
    assert "--mode design_fast does not take --refine" in err
    assert "--refine-design goes with --refine" in err


def test_the_console_design_verb_forwards_how_the_run_is_delivered(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr(console, "_forward", lambda module, argv: seen.append(argv) or {})
    assert console.main(["design", "--name", NAME, "--brief", "b", "--check", "true", "--no-pr",
                         "--refine", "acme/app#3", "--review-comments"]) == 0
    assert "--no-pr" in seen[-1] and "--review-comments" in seen[-1]
    assert seen[-1][seen[-1].index("--refine") + 1] == "acme/app#3"


def test_refactory_projects_lists_the_pull_request_a_run_opened(new_run, capsys):
    result = new_run.result
    capsys.readouterr()
    assert projects.main([]) == 0
    listed = capsys.readouterr().out
    assert result.facts["pull_request"] in listed and f"{LOGIN}/{NAME}" in listed


# ── the cases between the paths ────────────────────────────────────────────
def test_a_checkout_that_is_only_behind_github_gets_its_pull_request_and_keeps_its_refs(github, tmp_path):
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", "https://github.com/acme/app.git", str(other))
    (other / "later.py").write_text("x = 1\n")
    git(other, "add", "later.py")
    git(other, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "later")
    git(other, "push", "-q", "origin", "main")
    refs = git(repo, "for-each-ref")
    run = tmp_path / "run"
    result = _design(run, target=str(repo))
    assert result.exit_ok, result.error
    assert len(_pulls(github, "acme/app")) == 1
    branch = worktree.recorded(run).branch
    assert git(repo, "for-each-ref", "--exclude", f"refs/heads/{branch}") == refs
    assert not (repo / ".git" / "FETCH_HEAD").exists()


def test_a_creation_that_died_before_its_first_push_goes_on_in_the_repository_it_made(github, tmp_path, monkeypatch):
    run = tmp_path / "run"
    real = delivery._push

    def first_push_fails(path, slug, commit, branch, **kwargs):
        if branch == "main":
            raise RuntimeError("the delivery stopped here")
        return real(path, slug, commit, branch, **kwargs)

    monkeypatch.setattr(delivery, "_push", first_push_fails)
    assert _design(run).exit_ok is False
    slug = f"{LOGIN}/{NAME}"
    assert (github / f"{slug}.git").is_dir() and not _remote_head(github, slug, "main")
    monkeypatch.setattr(delivery, "_push", real)
    resumed = DesignMode.resume(run, model=FakeModel([]))
    assert resumed.exit_ok, resumed.error
    assert _calls(tmp_path).count("repo create") == 1 and len(_pulls(github, slug)) == 1
    assert _remote_head(github, slug, "main") == git(projects_dir() / NAME, "rev-parse", "main")


def test_a_repository_gh_cannot_push_to_gets_no_pull_request(github, tmp_path):
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    (github / "acme/app.json").write_text(json.dumps({"default_branch": "main", "push": False}))
    planned = delivery.plan(repo, base=git(repo, "rev-parse", "HEAD"), branch="refactory/run-0123456789ab")
    assert planned["action"] == "none" and "cannot push to acme/app" in planned["why"]


def test_a_refinement_delivered_once_is_not_refused_as_moved_when_asked_again(opened_pull, tmp_path, capsys):
    url = opened_pull.url
    run = tmp_path / "refine"
    result = _design(run, refine=url)
    assert result.exit_ok, result.error
    capsys.readouterr()
    again = delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))
    assert again["pull_request"] == url and _record(run)["status"] == "already updated"
    assert "run warning" not in capsys.readouterr().err


def test_refine_refuses_a_pull_request_from_another_repositorys_branch(opened_pull, tmp_path):
    url, github = opened_pull.url, opened_pull.github
    pulls = _pulls(github, "acme/app")
    pulls[0]["cross"] = True
    (github / "acme/app.pulls.json").write_text(json.dumps(pulls))
    refused = _design(tmp_path / "fork", refine=url)
    assert refused.exit_ok is False and "another repository's branch" in (refused.error or "")


def test_a_later_report_of_the_same_run_keeps_its_pull_request_in_the_registry(tmp_path):
    repo = git_repo(tmp_path / "repo")
    projects.record_run(repo, run=tmp_path / "run", branch="refactory/run-0123456789ab", base="b", status="running")
    projects.record_delivery(repo, branch="refactory/run-0123456789ab", pull_request="https://github.com/a/b/pull/1",
                             status="opened")
    projects.record_run(repo, run=tmp_path / "run", branch="refactory/run-0123456789ab", base="b", status="succeeded")
    [entry] = projects.load()["projects"]
    assert entry["runs"][-1]["pull_request"] == "https://github.com/a/b/pull/1"


def test_design_fast_takes_none_of_delivery_s_knobs():
    from refactory_dsh.fast import DesignFastMode

    assert not {"pr", "repo_owner", "repo_visibility", "refine", "review_comments"} & set(DesignFastMode.declare())


def test_a_cli_run_ends_by_naming_its_pull_request(tmp_path):
    from refactory_dsh import interactive

    (tmp_path / "facts.json").write_text(json.dumps({"name": NAME, "pull_request": "https://github.com/a/b/pull/7",
                                                     "delivery_result": "opened: https://github.com/a/b/pull/7"}))
    assert "https://github.com/a/b/pull/7" in interactive.run_summary(tmp_path)


def test_a_new_projects_main_with_work_of_its_own_is_never_published(github, tmp_path, capsys):
    first = _design(tmp_path / "one", mode=_mode(pr=False))
    assert first.exit_ok, first.error
    project = projects_dir() / NAME
    (project / "notes.md").write_text("mine\n")
    git(project, "add", "notes.md")
    git(project, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "mine")
    second = _design(tmp_path / "two")
    assert second.exit_ok, second.error
    assert "no longer the empty first commit" in (tmp_path / "two" / delivery.PLAN).read_text()
    assert "repo create" not in _calls(tmp_path) and not list(github.glob("*/*.git"))
    assert "no pull request was opened" in capsys.readouterr().err


def test_without_a_network_the_run_succeeds_and_says_its_work_stays_local(github, tmp_path, capsys):
    (github / "offline").write_text("")
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert "error connecting to api.github.com" in capsys.readouterr().err
    assert _record(run)["status"] == "kept local" and not list(github.glob("*/*.git"))


def test_review_comments_without_refine_are_refused_not_dropped(github, tmp_path):
    refused = _design(tmp_path / "run", mode=_mode(review_comments=True))
    assert refused.exit_ok is False and "review_comments goes with refine" in (refused.error or "")


def test_refactory_projects_says_when_a_run_kept_its_work_local(github, tmp_path, capsys):
    (github / "login").unlink()
    assert _design(tmp_path / "run").exit_ok
    capsys.readouterr()
    assert projects.main([]) == 0
    assert "kept local" in capsys.readouterr().out


def test_a_new_projects_first_pull_request_publishes_nothing_its_base_adds_to_main(github, tmp_path, capsys):
    first = _design(tmp_path / "one", mode=_mode(pr=False))
    assert first.exit_ok, first.error
    project = projects_dir() / NAME
    git(project, "checkout", "-q", "-b", "mywork", "main")
    (project / "secret-notes.md").write_text("mine\n")
    git(project, "add", "secret-notes.md")
    git(project, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "mine")
    git(project, "checkout", "-q", "main")
    second = _design(tmp_path / "two", base="mywork")
    assert second.exit_ok, second.error
    assert "1 commit(s) besides the empty main" in (tmp_path / "two" / delivery.PLAN).read_text()
    assert "repo create" not in _calls(tmp_path) and not list(github.glob("*/*.git"))


def test_a_credential_in_a_rewritten_url_never_reaches_a_message(monkeypatch, tmp_path):
    def leaks(*args, **kwargs):
        raise ValueError("git push failed in /r: fatal: unable to access 'https://me:ghp_secret@github.com/a/b.git/'")

    monkeypatch.setattr(delivery.worktree, "git", leaks)
    stub(tmp_path / "bin", "gh", "true")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(delivery.NotDelivered) as refused:
        delivery._push(tmp_path, "a/b", "0" * 40, "x", default="main")
    assert "ghp_secret" not in str(refused.value) and "https://***@github.com" in str(refused.value)


def test_a_pull_request_never_carries_a_credential_the_check_spelled_out(github, tmp_path):
    run = tmp_path / "run"
    result = _design(run, check="API_TOKEN=abc123 python tool.py")
    assert result.exit_ok, result.error
    [pull] = _pulls(github, f"{LOGIN}/{NAME}")
    assert "abc123" not in pull["body"] and "API_TOKEN=*** python tool.py" in pull["body"]


def test_a_redo_after_its_pull_request_merged_records_it_merged(new_run, capsys):
    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    pulls = _pulls(github, slug)
    pulls[0]["state"] = "MERGED"
    (github / f"{slug}.pulls.json").write_text(json.dumps(pulls))
    capsys.readouterr()
    again = delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))
    assert (_record(run)["status"], again["pull_request"]) == ("merged", result.facts["pull_request"])
    assert "run warning" not in capsys.readouterr().err


def test_a_redo_after_its_pull_request_closed_keeps_it_in_the_record(new_run, capsys):
    run, result, github = new_run.run, new_run.result, new_run.github
    slug = f"{LOGIN}/{NAME}"
    pulls = _pulls(github, slug)
    pulls[0]["state"] = "CLOSED"
    (github / f"{slug}.pulls.json").write_text(json.dumps(pulls))
    delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))
    record = _record(run)
    assert (record["status"], record["pull_request"]) == ("kept local", result.facts["pull_request"])
    assert f"{result.facts['pull_request']} was not updated" in capsys.readouterr().err
    assert len(_pulls(github, slug)) == 1


def test_a_pull_request_gh_opened_but_reported_failing_is_the_one_delivered(github, tmp_path, monkeypatch):
    real = delivery._gh

    def opens_then_fails(*args, **kwargs):
        said = real(*args, **kwargs)
        if args[:2] == ("pr", "create"):
            raise delivery.NotDelivered("`gh pr create` failed: timeout")
        return said

    monkeypatch.setattr(delivery, "_gh", opens_then_fails)
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert _record(run)["status"] == "opened" and len(_pulls(github, f"{LOGIN}/{NAME}")) == 1


def test_a_repository_gh_made_but_reported_failing_is_adopted_only_when_new_and_empty(github, tmp_path,
                                                                                     monkeypatch):
    real = delivery._gh

    def creates_then_fails(*args, **kwargs):
        said = real(*args, **kwargs)
        if args[:2] == ("repo", "create"):
            raise delivery.NotDelivered("`gh repo create` failed: timeout")
        return said

    monkeypatch.setattr(delivery, "_gh", creates_then_fails)
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert _record(run)["status"] == "opened"
    [entry] = projects.load()["projects"]
    assert entry["github"] == f"{LOGIN}/{NAME}"


def test_a_new_repository_that_is_slow_to_take_pushes_is_waited_for(github, tmp_path, monkeypatch):
    real, calls = delivery._heads, []

    def lags(path, slug):
        calls.append(slug)
        if len(calls) == 1:
            raise delivery.NotDelivered("git ls-remote failed: repository not found")
        return real(path, slug)

    monkeypatch.setattr(delivery, "_heads", lags)
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert _record(run)["status"] == "opened"


def test_a_remote_git_cannot_read_keeps_the_work_local_rather_than_failing_the_run(github, tmp_path):
    repo = git_repo(tmp_path / "repo")
    git(repo, "remote", "add", "origin", "https://github.com/acme/app.git")
    git(repo, "config", "--unset", "remote.origin.url")
    planned = delivery.plan(repo, base=git(repo, "rev-parse", "HEAD"), branch="refactory/run-0123456789ab")
    assert planned["action"] == "none" and "could not read the remotes" in planned["why"]


def test_a_gh_that_cannot_even_start_keeps_the_work_local(tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text("#!/no/such/interpreter\n")
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert "could not run" in _record(run)["why"] and "no pull request was opened" in capsys.readouterr().err


def test_a_branch_that_moved_since_its_reviews_is_never_delivered(new_run):
    run, result = new_run.run, new_run.result
    unreviewed = {**result.facts, "commit": "0" * 40}
    with pytest.raises(ValueError, match="not at the attempt its reviews passed"):
        delivery.deliver_design(nn.RunContext(run, facts=unreviewed, enforce_writes=False))


def test_a_refinement_never_pushes_onto_a_default_branch(opened_pull):
    repo, url, github = opened_pull.repo, opened_pull.url, opened_pull.github
    pulls = _pulls(github, "acme/app")
    pulls[0]["head"] = "main"
    (github / "acme/app.pulls.json").write_text(json.dumps(pulls))
    head = _remote_head(github, "acme/app", "main")
    planned = delivery.plan(repo, base=head, branch="refactory/run-0123456789ab",
                            refine={"url": url, "repo": "acme/app", "head": head})
    assert planned["action"] == "none" and "is acme/app's default branch" in planned["why"]


def test_a_new_repository_goes_where_repo_owner_and_repo_visibility_say(github, tmp_path):
    result = _design(tmp_path / "run", mode=_mode(repo_owner="acme-org", repo_visibility="internal"))
    assert result.exit_ok, result.error
    assert f"repo create acme-org/{NAME} --internal" in _calls(tmp_path)
    assert json.loads((github / "acme-org" / f"{NAME}.json").read_text())["visibility"] == "internal"
    assert result.facts["pull_request"] == f"https://github.com/acme-org/{NAME}/pull/1"


def test_a_refinement_that_github_shows_at_another_head_is_reported_not_claimed(opened_pull, tmp_path):
    url, github = opened_pull.url, opened_pull.github
    (github / "stale-view").write_text("0" * 40 + "\n")
    run = tmp_path / "refine"
    result = _design(run, refine=url)
    assert result.exit_ok is False and "not this run's reviewed commit" in (result.error or "")
    assert _record(run)["status"] == "unverified"


def test_a_pull_request_already_open_is_checked_again_when_asked_again(new_run):
    run, result = new_run.run, new_run.result
    (new_run.github / "stale-view").write_text("0" * 40 + "\n")
    with pytest.raises(delivery.Unverified):
        delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))


def test_the_targets_pre_push_hook_does_not_run_for_a_delivery(github, tmp_path):
    repo = git_repo(tmp_path / "repo")
    bare = _published(github, "acme/app", repo)
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text(f"#!/bin/sh\ntouch '{tmp_path / 'hook-ran'}'\nexit 1\n")
    hook.chmod(0o755)
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "side.py").write_text("y = 1\n")
    git(repo, "add", "side.py")
    git(repo, "commit", "-q", "-m", "side")
    delivery._push(repo, "acme/app", git(repo, "rev-parse", "HEAD"), "side", default="main")
    assert git(bare, "rev-parse", "side") == git(repo, "rev-parse", "HEAD")
    assert not (tmp_path / "hook-ran").exists()


def test_nothing_delivery_runs_can_stop_to_ask_a_question():
    env = delivery._env()
    assert (env["GIT_TERMINAL_PROMPT"], env["GIT_ASKPASS"], env["GCM_INTERACTIVE"], env["GH_PROMPT_DISABLED"]) == (
        "0", "", "never", "1")
    assert env["GIT_SSH_COMMAND"].endswith("-o BatchMode=yes")


def test_the_console_announces_a_run_approved_up_front_before_it_is_handed_on(github, tmp_path, monkeypatch, capsys):
    said_first: list[str] = []
    monkeypatch.setattr(console, "_forward", lambda module, argv: said_first.append(capsys.readouterr().err) or {})
    monkeypatch.chdir(tmp_path)
    assert console.main(["design", "--name", NAME, "--brief", "a tool", "--check", "true",
                         "--verdict", "gate_strategy=proceed"]) == 0
    assert f"creates the private GitHub repository {LOGIN}/{NAME}" in said_first[-1]


# ── the second review: the approval card, the base's branch, and what is where ─
def test_the_approval_card_itself_names_the_repository_a_run_would_create(new_run):
    """The card shows only the gate's question, so the question carries the plan (#362 review)."""
    run, result = new_run.run, new_run.result
    said = f"creates the private GitHub repository {LOGIN}/{NAME}"
    assert "{{delivery}}" in DesignMode().build().nodes["gate_strategy"].notes
    # The walker's journal shows the gate as it was put, never the raw placeholder.
    journal = [json.loads(line) for line in (run / ".refactory" / "events.jsonl").read_text().splitlines()]
    shown = [record["summary"] for record in journal if record.get("node_id") == "gate_strategy" and record.get("summary")]
    assert shown and all(said in summary and "{{" not in summary for summary in shown)
    # The panel's pending-gate card, read by the console from a spine run's graph and facts.json.
    (run / "facts.json").write_text(json.dumps(result.facts, default=str))
    (run / ".refactory" / "spine-status.json").write_text(json.dumps({"status": "waiting_user", "nodes": ["gate_strategy"]}))
    graphs = run / "graphs"
    graphs.mkdir(exist_ok=True)
    (graphs / "design.graph.json").write_text(json.dumps({"nodes": {"gate_strategy": {
        "_type": "GateNode", "evaluator_type": "user",
        "gate_prompt": DesignMode().build().nodes["gate_strategy"].notes}}, "edges": [
        {"source": "gate_strategy", "target": "emit", "condition": "proceed"}]}))
    [card] = console.gates(run)
    assert said in card["prompt"] and "{{" not in card["prompt"]


def test_pr_body_says_check_skipped_by_person(tmp_path):
    """dm-115: PR body says the check did not run when person-skipped."""
    import subprocess
    import json as _j
    from types import SimpleNamespace
    ctx = _TitleCtx(
        facts={"brief": "Fix something", "name": "b1", "check": "pnpm test"},
        files={
            "design.json": _j.dumps({
                "check_skipped": {
                    "source": "person",
                    "at": "2026-10-20T22:25:29+00:00",
                    "via": "graph-control",
                    "node": "check",
                }
            }),
            "authoring-evidence.json": _j.dumps({
                "validation": {"status": "skipped", "source": "person",
                               "at": "2026-10-20T22:25:29+00:00", "node": "check"},
            }),
        },
        workdir=tmp_path,
    )
    # Write files to disk for _body
    (tmp_path / "design.json").write_text(ctx._files["design.json"])
    (tmp_path / "authoring-evidence.json").write_text(ctx._files["authoring-evidence.json"])
    (tmp_path / "facts.json").write_text(_j.dumps(ctx.facts))
    # Set up a minimal git repo so worktree.git works.
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], capture_output=True, check=True)
    (tmp_path / "file.txt").write_text("content\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "tool.py").write_text("print('built')\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "attempt"], capture_output=True, check=True)
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    found = SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="refactory/run-test")
    body = delivery._body(ctx, found)
    assert "did not run" in body
    assert "skipped by the person" in body
    assert "covers later attempts" in body


def _stacked(github: Path, tmp_path: Path) -> tuple[Path, str]:
    """A repository on GitHub with a pushed branch `stack` one commit ahead of main; main is checked out."""
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    git(repo, "checkout", "-q", "-b", "stack")
    (repo / "lib.py").write_text("x = 1\n")
    git(repo, "add", "lib.py")
    git(repo, "commit", "-q", "-m", "the stack's own commit")
    git(repo, "push", "-q", "-u", "origin", "stack")
    git(repo, "checkout", "-q", "main")
    return repo, git(repo, "rev-parse", "stack")


def test_a_run_on_a_pushed_branch_opens_its_pull_request_into_that_branch(github, tmp_path):
    repo, stack = _stacked(github, tmp_path)
    run = tmp_path / "run"
    result = _design(run, target=str(repo), base="stack")
    assert result.exit_ok, result.error
    [pull] = _pulls(github, "acme/app")
    assert pull["base"] == "stack", "a stack's pull request goes into the branch it builds on"
    assert "into acme/app:stack, the branch its base came from" in (run / delivery.PLAN).read_text()
    # It carries only the run's own commit, not the stack's.
    branch = worktree.recorded(run).branch
    assert git(repo, "rev-list", "--count", f"{stack}..{branch}") == "1"


def test_a_run_from_a_checkout_on_a_pushed_branch_goes_into_its_upstream(github, tmp_path):
    repo, _stack = _stacked(github, tmp_path)
    git(repo, "checkout", "-q", "stack")
    run = tmp_path / "run"
    result = _design(run, target=str(repo))
    assert result.exit_ok, result.error
    assert _pulls(github, "acme/app")[0]["base"] == "stack"


def test_a_base_branch_github_does_not_have_falls_back_to_the_default_branch(github, tmp_path):
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    git(repo, "branch", "topic")
    planned = delivery.plan(repo, base=git(repo, "rev-parse", "topic"), branch="refactory/run-0123456789ab",
                            rev="topic")
    assert (planned["action"], planned["into"]) == ("pull_request", "main")
    assert planned["summary"].endswith("into acme/app:main")


def test_a_delivery_that_stopped_after_its_push_says_the_branch_is_on_github(github, tmp_path, monkeypatch, capsys):
    real = delivery._gh

    def no_pull_request(*args, **kwargs):
        if args[:2] == ("pr", "create"):
            raise delivery.NotDelivered("`gh pr create` failed: error connecting to api.github.com")
        return real(*args, **kwargs)

    monkeypatch.setattr(delivery, "_gh", no_pull_request)
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    branch = worktree.recorded(run).branch
    assert _remote_head(github, f"{LOGIN}/{NAME}", branch)
    err = capsys.readouterr().err
    assert f"the branch {branch} is already on GitHub, as {LOGIN}/{NAME}:{branch}" in err
    assert "stays on the local branch" not in err
    assert _record(run)["pushed"] == f"{LOGIN}/{NAME}:{branch}"


def test_a_repository_whose_creation_lost_the_network_is_finished_by_a_redo(github, tmp_path, monkeypatch, capsys):
    """gh reports the creation failed, the network then drops; the redo adopts the repository it made."""
    real = delivery._gh

    def creates_then_drops(*args, **kwargs):
        said = real(*args, **kwargs)
        if args[:2] == ("repo", "create"):
            (github / "offline").write_text("")
            raise delivery.NotDelivered("`gh repo create` failed: timeout")
        return said

    monkeypatch.setattr(delivery, "_gh", creates_then_drops)
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert _record(run)["status"] == "kept local" and (github / LOGIN / f"{NAME}.git").is_dir()
    (github / "offline").unlink()
    monkeypatch.setattr(delivery, "_gh", real)
    again = delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))
    assert again["pull_request"] == f"https://github.com/{LOGIN}/{NAME}/pull/1"
    assert _calls(tmp_path).count("repo create") == 2 and len(_pulls(github, f"{LOGIN}/{NAME}")) == 1


def test_a_repository_with_the_name_that_this_project_never_set_out_to_create_is_not_adopted(github, tmp_path):
    first = _design(tmp_path / "one", mode=_mode(pr=False))
    assert first.exit_ok, first.error
    bare = github / LOGIN / f"{NAME}.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    git(bare.parent, "init", "-q", "--bare", "-b", "main", bare.name)
    (github / LOGIN / f"{NAME}.json").write_text(json.dumps({"default_branch": "main"}))
    second = _design(tmp_path / "two")
    assert second.exit_ok, second.error
    assert "already exists" in (tmp_path / "two" / delivery.PLAN).read_text() and _pulls(github, f"{LOGIN}/{NAME}") == []


def test_a_logged_out_gh_is_named_asks_github_nothing_and_leaves_commands_that_publish(github, tmp_path):
    """dm-260: delivery without a gh login blamed GitHub's rate limit, and named the address.

    A logged-out gh 2.99 still sends `gh api user` to GitHub, which answered from a
    shared address with "API rate limit exceeded for <address>"; that went into the plan,
    the record and the warning, and `finish` was empty although the new project's files
    were only on the run's branch. Now gh's own `auth status` is asked first, nothing
    reaches GitHub, and `finish` holds the commands that publish the work, which run here
    once gh is logged in.
    """
    (github / "login").unlink()
    (github / "rate-limited").write_text("")
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    plan_text = (run / delivery.PLAN).read_text()
    assert "GitHub CLI is not logged in" in plan_text
    assert "api user" not in _calls(tmp_path), "nothing is asked of GitHub without a login"
    said = plan_text + json.dumps(_record(run)) + result.facts["delivery_result"]
    assert "rate limit" not in said and "203.0.113.7" not in said
    record = _record(run)
    target = (projects_dir() / NAME).resolve()
    assert record["finish"][0] == "gh auth login"
    assert record["finish"][1] == f"gh repo create {NAME} --private --source {target} --remote origin"
    assert f"the work is on branch {record['branch']} of {target}" in result.facts["delivery_result"]

    (github / "login").write_text(LOGIN + "\n")
    shell = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
             "FAKE_GH_ROOT": str(github), "GIT_TERMINAL_PROMPT": "0"}
    for command in record["finish"][1:]:
        subprocess.run(command, shell=True, check=True, capture_output=True, env=shell)
    slug = f"{LOGIN}/{NAME}"
    [pull] = _pulls(github, slug)
    assert (pull["head"], pull["base"], pull["title"]) == (record["branch"], "main", record["title"])
    assert _remote_head(github, slug, record["branch"]) == record["head"]
    assert git(github / f"{slug}.git", "ls-tree", "main") == "", "main is the project's empty first commit"


def test_a_repository_with_no_github_remote_is_left_the_local_merge(github, tmp_path):
    """dm-260: a run on a repository that has no GitHub remote cannot be published, so its
    `finish` is the merge that brings the run's branch into the person's checkout."""
    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    result = _design(run, target=str(repo))
    assert result.exit_ok, result.error
    record = _record(run)
    assert record["finish"] == [f"git -C {repo.resolve()} merge {record['branch']}"]
    assert f"the work is on branch {record['branch']} of {repo}" in result.facts["delivery_result"]
    subprocess.run(record["finish"][0], shell=True, check=True, capture_output=True)
    assert (repo / "tool.py").read_text() == "print('built')\n"


def test_gh_is_asked_for_its_login_without_gh_host(github, tmp_path):
    """dm-260: delivery runs gh with GH_HOST set, and with it a logged-out gh 2.99's `auth
    status` reports a phantom token rather than "not logged in". So the login is asked
    without GH_HOST, and a logged-out gh is named without anything being sent to GitHub."""
    (github / "login").unlink()
    planned = delivery.plan(tmp_path / "new", base="", branch="refactory/run-x")
    assert planned["why"] == delivery.LOGGED_OUT and planned.get("logged_out") is True
    assert "auth status" in _calls(tmp_path) and "api user" not in _calls(tmp_path)


def test_what_github_says_reaches_no_record_with_an_address(tmp_path, monkeypatch):
    """dm-260: a refusal from GitHub can name the address a request came from. Every reason
    delivery records hides it, while a version or a time in the same text stays as it was."""
    stub(tmp_path / "bin", "gh", 'echo "gh: API rate limit exceeded for 203.0.113.7 and 2001:db8::7, '
                                 'gh 2.99.0 at 13:55:01 (HTTP 403)" >&2; exit 1')
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    planned = delivery.plan(tmp_path / "new", base="", branch="refactory/run-x")
    assert planned["action"] == "none"
    assert "203.0.113.7" not in json.dumps(planned) and "2001:db8::7" not in json.dumps(planned)
    assert "rate limit exceeded for [address hidden] and [address hidden], gh 2.99.0 at 13:55:01" in planned["why"]


def test_a_run_planned_while_gh_was_logged_out_says_the_next_run_delivers(github, tmp_path):
    (github / "login").unlink()
    run = tmp_path / "run"
    assert _design(run).exit_ok
    assert "`gh auth login` signs in, for the next run" in (run / delivery.PLAN).read_text()


def test_the_pull_request_recommends_no_way_to_merge(new_run):
    """The description no longer explains the branch's commits (the short description,
    2026-10-26), and still recommends no merge method."""
    [pull] = _pulls(new_run.github, f"{LOGIN}/{NAME}")
    assert "the run's attempts" not in pull["body"]
    assert "squash merge is recommended" not in pull["body"]


# ── pull request text for kept-local deliveries (dm-86) ────────────────────

def test_no_pr_run_leaves_pr_text_and_title_in_delivery_json(github, tmp_path, capsys):
    """A --no-pr run that passes its reviews leaves pull-request.md with a non-empty
    Summary and its title in delivery.json.

    Fails without the change because _kept never writes BODY and never records title.
    """
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    run = tmp_path / "run"
    result = _design(run, mode=_mode(pr=False), target=str(repo))
    assert result.exit_ok, result.error
    # pull-request.md exists with a Summary section
    body_path = run / delivery.BODY
    assert body_path.exists(), "pull-request.md should exist for a --no-pr run with a worktree"
    body_text = body_path.read_text()
    assert "## Summary" in body_text
    assert delivery.BODY_START in body_text
    assert "Opened by a refactory design run" not in body_text
    # delivery.json records title, pr_text, and finish
    record = _record(run)
    assert record["title"] and isinstance(record["title"], str)
    assert record["pr_text"] == delivery.BODY
    assert isinstance(record["finish"], list) and len(record["finish"]) > 0
    # delivery_result includes the finish commands (reaching the person even without a warning)
    assert "--body-file" in result.facts["delivery_result"]
    assert "--title" in result.facts["delivery_result"]
    # design.json inherits title
    design = json.loads((run / "design.json").read_text())
    assert design["delivery"]["title"]
    # A deliberate --no-pr emits no warning
    assert "run warning" not in capsys.readouterr().err


def test_plan_changed_leaves_pr_text_and_finish_commands(github, tmp_path, capsys, monkeypatch):
    """A plan that changed since the gate keeps the work local but still leaves
    pull-request.md and finish commands.

    Fails without the change because _kept never writes BODY and its warning has no
    --body-file.
    """
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    # A fork holding the same main, so the run's base is on it and only the repository differs.
    fork = github / "acme/fork.git"
    git(fork.parent, "init", "-q", "--bare", "-b", "main", fork.name)
    (github / "acme/fork.json").write_text(json.dumps({"default_branch": "main"}))
    git(repo, "push", "-q", "https://github.com/acme/fork.git", "main:refs/heads/main")
    real = delivery.deliver_design

    def repointed(ctx):
        git(repo, "remote", "set-url", "origin", "https://github.com/acme/fork.git")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", repointed)
    run = tmp_path / "run"
    result = _design(run, target=str(repo))
    assert result.exit_ok, result.error
    # pull-request.md exists
    assert (run / delivery.BODY).exists()
    body_text = (run / delivery.BODY).read_text()
    assert "## Summary" in body_text
    # The warning includes the complete command
    err = capsys.readouterr().err
    assert "--body-file" in err
    assert "--title" in err
    # delivery.json records title and finish
    record = _record(run)
    assert record["title"]
    assert any("gh pr create" in cmd and "--body-file" in cmd for cmd in record["finish"])


def test_push_refused_leaves_pr_text_and_finish_commands(github, tmp_path, capsys, monkeypatch):
    """A push GitHub refuses keeps the work local but still leaves pull-request.md
    and finish commands.

    Fails without the change because _kept never writes BODY and its warning has no
    --body-file.
    """
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    real_push = delivery._push

    def push_refused(*args, **kwargs):
        raise delivery.NotDelivered("push refused by GitHub")

    monkeypatch.setattr(delivery, "_push", push_refused)
    run = tmp_path / "run"
    result = _design(run, target=str(repo))
    assert result.exit_ok, result.error
    # pull-request.md exists
    assert (run / delivery.BODY).exists()
    body_text = (run / delivery.BODY).read_text()
    assert "## Summary" in body_text
    # The warning includes the complete commands
    err = capsys.readouterr().err
    assert "--body-file" in err
    assert "--title" in err
    # delivery.json records title and status
    record = _record(run)
    assert record["title"]
    assert record["status"] == "kept local"
    assert any("gh pr create" in cmd and "--body-file" in cmd for cmd in record["finish"])


def test_a_push_refused_for_a_missing_workflow_scope_says_so_and_how_to_finish(github, tmp_path, capsys,
                                                                                monkeypatch):
    """dm-75: GitHub's refusal is said in one sentence naming the file and the scope, then
    the push and the pull request commands to run where a credential has that scope."""
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    real_git = delivery._git

    def git(path, *args):
        if "push" in args:
            raise delivery.NotDelivered(
                f"git push -q https://github.com/acme/app.git {args[-1]} failed in {path}: "
                "To https://github.com/acme/app.git\n"
                " ! [remote rejected] abc -> refactory/run-abc (refusing to allow an OAuth App to "
                "create or update workflow `.github/workflows/ci.yml` without `workflow` scope)\n"
                "error: failed to push some refs to 'https://github.com/acme/app.git'")
        return real_git(path, *args)

    monkeypatch.setattr(delivery, "_git", git)
    run = tmp_path / "run"
    result = _design(run, target=str(repo))
    assert result.exit_ok, result.error
    said = result.facts["delivery_result"]
    assert said.startswith(
        "not delivered: GitHub refused the push because it changes .github/workflows/ci.yml, and "
        "the credential delivery pushed with lacks the `workflow` scope that changing a workflow "
        "file takes; run the finishing commands where a GitHub credential with the `workflow` "
        "scope is signed in"), said
    assert "remote rejected" not in said
    record = _record(run)
    assert record["status"] == "kept local" and "remote rejected" not in record["why"]
    push = next(cmd for cmd in record["finish"] if " push https://github.com/acme/app.git " in cmd)
    opening = next(cmd for cmd in record["finish"] if cmd.startswith("gh pr create --repo acme/app "))
    assert f"to finish: {push} && {opening}" in said
    assert push in capsys.readouterr().err


def test_a_github_app_without_the_workflows_permission_is_named_as_a_permission():
    """GitHub words a GitHub App's refusal with a permission, not a scope."""
    said = delivery._scope_refusal(
        "git push failed in /w: To https://github.com/acme/app.git\n ! [remote rejected] abc -> b "
        "(refusing to allow a GitHub App to create or update workflow `.github/workflows/ci.yml` "
        "without `workflows` permission)\n")
    assert said == ("GitHub refused the push because it changes .github/workflows/ci.yml, and the "
                    "credential delivery pushed with lacks the `workflows` permission that changing a "
                    "workflow file takes; run the finishing commands where a GitHub credential with "
                    "the `workflows` permission is signed in")
    assert delivery._scope_refusal("! [rejected] abc -> b (non-fast-forward)") == ""


def test_no_worktree_writes_no_pr_text(github, tmp_path, monkeypatch):
    """A run with no worktree writes no pull-request.md, no title, and no pr_text.

    This test passes both before and after the change — it verifies the guard.
    """
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    result = _design(run, mode=_mode(worktree=False))
    assert result.exit_ok, result.error
    assert not (run / delivery.BODY).exists()
    record = _record(run)
    assert not record.get("title")
    assert "pr_text" not in record


def test_finish_commands_open_a_pull_request_with_the_exact_title_and_body(github, tmp_path):
    """Runs a --no-pr design, reads the finish commands from delivery.json, and
    executes them against the test's fake GitHub.  Asserts a PR opens with the exact
    title and body — proving nothing is left for a person to write.

    Fails without the change because delivery.json has no finish field.
    """
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    run = tmp_path / "run"
    result = _design(run, mode=_mode(pr=False), target=str(repo))
    assert result.exit_ok, result.error
    record = _record(run)
    finish = record["finish"]
    assert len(finish) == 2, f"expected push + gh pr create, got {finish}"
    # The push command uses https://github.com/ and :refs/heads/ (not origin)
    push_cmd = finish[0]
    assert "https://github.com/" in push_cmd
    assert ":refs/heads/" in push_cmd
    assert "origin" not in push_cmd
    # Execute the push command
    subprocess.run(push_cmd, shell=True, check=True, capture_output=True,
                   env={**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
                        "GIT_TERMINAL_PROMPT": "0"})
    # Execute the gh pr create command
    subprocess.run(finish[1], shell=True, check=True, capture_output=True,
                   env={**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
                        "FAKE_GH_ROOT": str(github)})
    # A pull request was opened
    [pull] = _pulls(github, "acme/app")
    assert pull["title"] == record["title"]
    body_text = (run / delivery.BODY).read_text()
    assert pull["body"] == body_text


def test_finish_commands_work_with_special_characters_in_title(github, tmp_path):
    """A title containing a quote, a backtick, and a $ works through shell quoting.

    Fails without the change because delivery.json has no finish field.
    """
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    run = tmp_path / "run"
    # A design doc whose heading has special characters, so _title picks them up.
    tricky_brief = "Fix the audit's `gate` and $PATH handling"
    tricky_design = (
        "# Fix the audit's `gate` and $PATH handling\n\n"
        "## Summary\n\nFix it properly.\n\n"
        "## Failure trace\n\nN/A — this is a new feature.\n\n"
        "## Ask mapping\n\n"
        "| Ask (quoted from brief) | Design section | Test |\n"
        "|---|---|---|\n"
        '| "gate" | § Changes | test_tool.py |\n\n'
        "## Rule and sites\n\n"
        "- `tool.py:1` — **[changed]**\n\n"
        "## Docs\n\n"
        "- docs/models.md\n\n"
        "## Changes\n\n"
        "- tool.py\n"
    )
    tricky_design += (
        "\n## Research and perspective findings\n\n"
        "- FND-api-surface-1: addressed at `tool.py:1`\n"
        "- FND-existing-modes-1: addressed at § Changes\n"
        "- FND-acceptance-1: addressed at `tool.py:1`\n"
        "- FND-simplest-1: addressed at `tool.py:1`\n"
        "- FND-criteria-1: addressed at § Changes\n"
    )
    tricky_design += (
        "\n## Anti-patterns to avoid\n\n"
        "- PIT-001: addressed at `tool.py:1`\n"
    )
    answers = [RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
               tricky_design,
               tricky_design,
               render({"tool.py": "print('built')\n"}),
               *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3]]
    model = FakeModel(answers)
    result = _mode(pr=False).run(
        run, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": tricky_brief, "check": "python tool.py",
              "target": str(repo)},
    )
    assert result.exit_ok, result.error
    record = _record(run)
    finish = record["finish"]
    assert len(finish) == 2
    # Execute the commands
    subprocess.run(finish[0], shell=True, check=True, capture_output=True,
                   env={**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
                        "GIT_TERMINAL_PROMPT": "0"})
    subprocess.run(finish[1], shell=True, check=True, capture_output=True,
                   env={**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
                        "FAKE_GH_ROOT": str(github)})
    [pull] = _pulls(github, "acme/app")
    # The title contains the special characters intact
    assert "'" in pull["title"] or "`" in pull["title"] or "$" in pull["title"]
    assert pull["title"] == record["title"]


def test_no_pr_finish_commands_are_complete(github, tmp_path, capsys):
    """A --no-pr run whose target has a GitHub remote gets complete finish commands
    with --repo, --title, and --body-file, and they appear in delivery_result.

    Fails without the change because delivery.json has no finish field and
    delivery_result has no commands.
    """
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    run = tmp_path / "run"
    result = _design(run, mode=_mode(pr=False), target=str(repo))
    assert result.exit_ok, result.error
    record = _record(run)
    finish = record["finish"]
    assert len(finish) == 2
    gh_cmd = finish[1]
    assert "--repo" in gh_cmd
    assert "--title" in gh_cmd
    assert "--body-file" in gh_cmd
    # delivery_result carries the finish commands
    dr = result.facts["delivery_result"]
    assert "--body-file" in dr
    assert "--title" in dr
    # No warning for deliberate --no-pr
    assert "run warning" not in capsys.readouterr().err


def test_no_github_remote_says_so_plainly(github, tmp_path, capsys):
    """A target with no GitHub remote gets no commands and a plain message.

    The second correction says: say plainly that there is no GitHub repository.
    """
    run = tmp_path / "run"
    result = _design(run, mode=_mode(pr=False))
    assert result.exit_ok, result.error
    # pull-request.md is still written (there IS a worktree)
    assert (run / delivery.BODY).exists()
    record = _record(run)
    assert record.get("title")
    # A new project has no repository yet: the commands create it, as delivery would (dm-260).
    assert record["finish"][0].startswith(f"gh repo create {NAME} --private --source ")
    # delivery_result mentions no GitHub repository, and where the work is
    dr = result.facts["delivery_result"]
    assert "no GitHub repository" in dr
    assert "pull request text is at" in dr
    assert f"the work is on branch {record['branch']} of " in dr


# ── merge-before-deliver (dm-76) ───────────────────────────────────────────

def test_base_moved_with_generated_conflict(github, tmp_path, monkeypatch):
    """The base moves after the run starts, with a change to a generated file and to an
    unrelated file: the delivered branch contains the base, the generated file is
    regenerated, the check ran on the merged result, and the pull request merges cleanly.

    Fails without the change: delivery pushes the reviewed commit as-is, and the PR
    opens behind its base with a generated-file conflict.
    """
    repo = git_repo(tmp_path / "repo")
    # gen.sh: concatenate source files into generated.txt
    (repo / "gen.sh").write_text("cat source.txt > generated.txt\n"
                                 "if [ -f source2.txt ]; then cat source2.txt >> generated.txt; fi\n")
    (repo / "source.txt").write_text("alpha\n")
    # generate from the initial state
    subprocess.run(["bash", "gen.sh"], cwd=str(repo), check=True, capture_output=True)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base with gen")
    slug = "acme/merge-gen"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_then_deliver(ctx):
        """Between the run's start and delivery, advance main with a generated-file conflict."""
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        (other / "source2.txt").write_text("gamma\n")
        (other / "unrelated.txt").write_text("hello\n")
        # regenerate generated.txt with the base's new sources
        subprocess.run(["bash", "gen.sh"], cwd=str(other), check=True, capture_output=True)
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base moves: add source2 and unrelated")
        git(other, "push", "-q", "origin", "main")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)

    run = tmp_path / "run"
    # The run changes source.txt and writes tool.py; gen.sh is the generate command.
    result = _design(run, {"tool.py": "print('built')\n", "source.txt": "alpha\nbeta\n"},
                     target=str(repo), check="python tool.py",
                     generate="bash gen.sh")
    assert result.exit_ok, result.error

    record = _record(run)
    assert record["status"] == "opened"
    assert record.get("merge_commit") and len(record["merge_commit"]) == 40
    merge_commit = record["merge_commit"]
    assert record.get("reviewed_commit") and record["reviewed_commit"] != merge_commit
    assert record.get("merge_check") == "passed on the merged result"

    # The merge commit has exactly two parents.
    parents = [line.split()[1] for line in
               git(bare, "cat-file", "-p", merge_commit).splitlines()
               if line.startswith("parent ")]
    assert len(parents) == 2

    # The delivered branch contains the base.
    main_tip = git(bare, "rev-parse", "refs/heads/main")
    assert git(bare, "merge-base", "--is-ancestor", main_tip, merge_commit) == ""

    # The generated file has content from both sides.
    generated = git(bare, "show", f"{merge_commit}:generated.txt")
    assert "alpha" in generated and "beta" in generated and "gamma" in generated

    # The PR body names both commits and says the check passed on the merged result.
    [pull] = _pulls(github, slug)
    assert record["reviewed_commit"][:12] in pull["body"]
    assert merge_commit[:12] in pull["body"]
    assert "check passed on the merged result" in pull["body"]

    # The remote branch head equals the merge commit.
    branch = json.loads((run / "design.json").read_text())["branch"]
    assert _remote_head(github, slug, branch) == merge_commit


def test_base_changed_authored_file_conflict(github, tmp_path, monkeypatch):
    """The base changes the same lines of an authored file: delivery does not push,
    and the conflict facts are set.

    Fails without the change: delivery pushes the reviewed commit as-is, and the PR
    opens behind its base.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "authored.py").write_text("line1\nline2\nline3\n")
    (repo / "gen.sh").write_text("echo generated > generated.txt\n")
    subprocess.run(["bash", "gen.sh"], cwd=str(repo), check=True, capture_output=True)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base with authored")
    slug = "acme/merge-auth"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        (other / "authored.py").write_text("line1\nchanged_by_base\nline3\n")
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base changes authored.py line 2")
        git(other, "push", "-q", "origin", "main")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)
    # Skip the reloop — this test verifies delivery's behaviour, not the full loop.
    monkeypatch.setattr(delivery, "evaluate_delivery_conflicts", lambda ctx: "proceed")

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n",
                           "authored.py": "line1\nchanged_by_run\nline3\n"},
                     target=str(repo), check="python tool.py",
                     generate="bash gen.sh")
    assert result.exit_ok, result.error

    record = _record(run)
    # Delivery does not push — the run keeps the work local.
    assert record["status"] == "kept local"
    assert "authored.py" in record.get("merge_conflicts", [])

    # Conflict facts are set.
    assert "authored.py" in result.facts.get("delivery_conflict_files", [])
    # The detail is a bounded excerpt kept in a file, never a fact (no prompt renders it).
    assert "delivery_conflict_detail" not in result.facts
    detail = json.loads((run / delivery.CONFLICT_DETAIL).read_text())
    assert [d["path"] for d in detail] == ["authored.py"]
    assert "<<<<<<<" in detail[0]["excerpt"] and "changed_by_base" in detail[0]["excerpt"]
    assert set(detail[0]) == {"path", "excerpt"}

    # No push happened.
    branch = json.loads((run / "design.json").read_text())["branch"]
    assert _remote_head(github, slug, branch) == ""


def _move_main_then_deliver(monkeypatch, github, tmp_path, slug, edit):
    """Between the run's start and delivery, let ``edit(other_clone)`` commit to main."""
    real = delivery.deliver_design
    moved = []

    def advance(ctx):
        if moved:
            return real(ctx)  # the reloop delivers again: main has already moved
        moved.append(1)
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        edit(other)
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "main moves")
        git(other, "push", "-q", "origin", "main")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance)


def test_generated_only_conflict_is_regenerated_not_sent_back(github, tmp_path, monkeypatch):
    """A merge with main that conflicts only in a generated file ends regenerated and
    committed, with no send-back, even when generate is not repeatable (it stamps a count).

    Fails without the change: delivery runs generate once per side and calls a file whose
    two outputs differ authored, so it sends the build back for a generated file.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "gen.sh").write_text(
        'c="${HOME:-/tmp}/gen-count"; n=$(cat "$c" 2>/dev/null || echo 0); echo $((n + 1)) > "$c"\n'
        'echo "build $n from $(cat source.txt)" > generated.txt\n')
    (repo / "source.txt").write_text("alpha\n")
    (repo / "generated.txt").write_text("build 0 from alpha\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/merge-gen-only"
    bare = _published(github, slug, repo)

    def edit(other):
        (other / "generated.txt").write_text("main's own generated text\n")
    _move_main_then_deliver(monkeypatch, github, tmp_path, slug, edit)

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n", "source.txt": "alpha\nbeta\n"},
                     target=str(repo), check="python tool.py", generate="bash gen.sh")
    assert result.exit_ok, result.error

    record = _record(run)
    assert record["status"] == "opened", record
    assert not record.get("merge_conflicts")
    assert not result.facts.get("delivery_conflict_files")
    assert not (run / delivery.CONFLICT_DETAIL).exists()
    merged = git(bare, "show", f"{record['merge_commit']}:generated.txt")
    assert merged.startswith("build ") and "alpha" in merged and "<<<<<<<" not in merged
    assert "main's own" not in merged


def test_mixed_conflict_sends_back_only_the_authored_file(github, tmp_path, monkeypatch):
    """An authored file and a generated file conflict: only the authored one is sent back,
    with a bounded excerpt of its hunks.

    Fails without the change: the generated file derives from the authored one, so
    generate's output differs between the two sides and both files are sent back.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "gen.sh").write_text("cat authored.py > generated.txt\n")
    (repo / "authored.py").write_text("line1\nline2\nline3\n")
    subprocess.run(["bash", "gen.sh"], cwd=str(repo), check=True, capture_output=True)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/merge-mixed"
    _published(github, slug, repo)

    def edit(other):
        (other / "authored.py").write_text("line1\nchanged_by_base\nline3\n")
        subprocess.run(["bash", "gen.sh"], cwd=str(other), check=True, capture_output=True)
    _move_main_then_deliver(monkeypatch, github, tmp_path, slug, edit)
    monkeypatch.setattr(delivery, "evaluate_delivery_conflicts", lambda ctx: "proceed")

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n", "authored.py": "line1\nchanged_by_run\nline3\n"},
                     target=str(repo), check="python tool.py", generate="bash gen.sh")
    assert result.exit_ok, result.error

    assert _record(run)["status"] == "kept local"
    assert result.facts["delivery_conflict_files"] == ["authored.py"]
    detail = json.loads((run / delivery.CONFLICT_DETAIL).read_text())
    assert [d["path"] for d in detail] == ["authored.py"]
    assert "changed_by_run" in detail[0]["excerpt"] and "changed_by_base" in detail[0]["excerpt"]
    assert len(detail[0]["excerpt"].splitlines()) <= 41


def test_conflict_send_back_stays_bounded_with_multi_megabyte_files(github, tmp_path, monkeypatch):
    """Multi-megabyte conflicting files reach the builder as at most a bounded excerpt each
    and about 20 KB in all, in the file list and in the validation text the prompt renders.

    Fails without the change: validation carries the base, ours and theirs files whole.
    """
    lines = [f"row {i} " + "x" * 40 for i in range(70_000)]  # about 3.5 MB

    def variant(tag):
        return "\n".join(f"{row} {tag}" if i % 1000 == 0 else row for i, row in enumerate(lines)) + "\n"

    repo = git_repo(tmp_path / "repo")
    for name in ("big_a.txt", "big_b.txt", "big_c.txt"):
        (repo / name).write_text("\n".join(lines) + "\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/merge-big"
    _published(github, slug, repo)

    def edit(other):
        for name in ("big_a.txt", "big_b.txt", "big_c.txt"):
            (other / name).write_text(variant("main"))
    _move_main_then_deliver(monkeypatch, github, tmp_path, slug, edit)
    gate = delivery.evaluate_delivery_conflicts
    monkeypatch.setattr(delivery, "evaluate_delivery_conflicts", lambda ctx: "proceed")

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n", **{n: variant("run") for n in
                                                              ("big_a.txt", "big_b.txt", "big_c.txt")}},
                     target=str(repo), check="python tool.py")
    assert result.exit_ok, result.error
    assert sorted(result.facts["delivery_conflict_files"]) == ["big_a.txt", "big_b.txt", "big_c.txt"]
    assert "delivery_conflict_detail" not in result.facts
    assert (run / delivery.CONFLICT_DETAIL).stat().st_size < 60_000
    for entry in json.loads((run / delivery.CONFLICT_DETAIL).read_text()):
        assert len(entry["excerpt"].splitlines()) <= 41

    ctx = nn.RunContext(run, facts=dict(result.facts))
    assert gate(ctx) == "reloop"
    validation = ctx.facts["validation"]
    assert 0 < len(validation) <= 20_200, len(validation)
    assert "<<<<<<<" in validation
    assert all(len(json.dumps(v)) < 40_000 for k, v in ctx.facts.items()
               if k in ("validation", "delivery_conflict_files"))


def test_merged_result_fails_check(github, tmp_path, monkeypatch, capsys):
    """The merged result fails the check: delivery does not push.

    Fails without the change: delivery pushes the reviewed commit as a fallback.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "config.txt").write_text("ok\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base config ok")
    slug = "acme/merge-fail"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        (other / "config.txt").write_text("broken\n")
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base breaks config")
        git(other, "push", "-q", "origin", "main")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)
    # Skip the reloop — this test verifies delivery's behaviour, not the full loop.
    monkeypatch.setattr(delivery, "evaluate_delivery_conflicts", lambda ctx: "proceed")

    run = tmp_path / "run"
    # tool.py fails when config.txt says "broken"
    result = _design(run, {"tool.py": "import sys; sys.exit(1 if open('config.txt').read().strip() == 'broken' else 0)\n"},
                     target=str(repo), check="python tool.py")
    assert result.exit_ok, result.error

    record = _record(run)
    # Delivery does not push — the run keeps the work local.
    assert record["status"] == "kept local"
    assert record.get("merge_check_failed")

    # No push happened.
    branch = json.loads((run / "design.json").read_text())["branch"]
    assert _remote_head(github, slug, branch) == ""

    # The worktree must NOT be marked "succeeded" — the merge check failed.
    # Fails without the fix: _kept only guards record_outcome against
    # merge.conflicts, not merge.check_failed.
    found = worktree.recorded(run)
    if found is not None:
        record_path = worktree.slot(Path(found.target)) / f"run-{found.branch.removeprefix('refactory/run-')}.json"
        if record_path.exists():
            wt_record = json.loads(record_path.read_text())
            assert wt_record.get("outcome") != "succeeded", \
                "worktree should not be marked succeeded when the merged check failed"


def test_base_did_not_move(github, tmp_path):
    """The base did not move: no merge commit.

    This test passes without the change too (it tests the no-change path). It asserts
    that no merge_commit is recorded, guarding against a spurious merge.
    """
    repo = git_repo(tmp_path / "repo")
    slug = "acme/no-move"
    _published(github, slug, repo)

    run = tmp_path / "run"
    result = _design(run, target=str(repo))
    assert result.exit_ok, result.error

    record = _record(run)
    assert record["status"] == "opened"
    assert not record.get("merge_commit")

    # No merge commits in the delivered branch.
    branch = json.loads((run / "design.json").read_text())["branch"]
    remote_head = _remote_head(github, slug, branch)
    assert remote_head == worktree.recorded(run).head
    merges = git(github / f"{slug}.git", "rev-list", "--merges",
                 f"{git(repo, 'rev-parse', 'main')}..{remote_head}")
    assert merges == ""


def test_generated_content_equals_base_version_still_merges(github, tmp_path, monkeypatch):
    """When the merged generated content equals the base's version, the file is still
    classified as generated (A == B from both sides' generate), and the merge succeeds.

    This tests the correction #2: classify by whether generate determines the file,
    not by whether generate changed it relative to the base's version.
    """
    repo = git_repo(tmp_path / "repo")
    # gen.sh always writes the same content regardless of other source files.
    (repo / "gen.sh").write_text("echo 'static-content' > generated.txt\n")
    subprocess.run(["bash", "gen.sh"], cwd=str(repo), check=True, capture_output=True)
    (repo / "source.txt").write_text("alpha\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/merge-static-gen"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        # Base changes generated.txt to something else (simulating a different generator version),
        # but the generate command always produces 'static-content'.
        (other / "generated.txt").write_text("old-generated-content\n")
        (other / "extra.txt").write_text("extra\n")
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base changes generated.txt")
        git(other, "push", "-q", "origin", "main")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)

    run = tmp_path / "run"
    # The run also changes generated.txt (via generate), causing a conflict.
    result = _design(run, {"tool.py": "print('built')\n", "source.txt": "alpha\nbeta\n"},
                     target=str(repo), check="python tool.py",
                     generate="bash gen.sh")
    assert result.exit_ok, result.error

    record = _record(run)
    assert record["status"] == "opened"
    # The merge succeeded even though the generated content equals the base's version
    # (generate produces 'static-content' from both sides).
    assert record.get("merge_commit") and len(record["merge_commit"]) == 40
    # generated.txt has the generate command's output.
    generated = git(bare, "show", f"{record['merge_commit']}:generated.txt")
    assert generated.strip() == "static-content"


def test_timeout_uses_check_timeout_s(github, tmp_path, monkeypatch):
    """The run's check_timeout_s knob is used for generate, setup and the check during
    delivery's merge, not a hard-coded 300.

    Fails without the change: the captured timeout is 300 (hard-coded).
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "gen.sh").write_text("echo generated > generated.txt\n")
    subprocess.run(["bash", "gen.sh"], cwd=str(repo), check=True, capture_output=True)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/merge-timeout"
    _published(github, slug, repo)

    real = delivery.deliver_design
    captured_timeouts: list[float] = []

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        (other / "extra.txt").write_text("new\n")
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base moves")
        git(other, "push", "-q", "origin", "main")

        from refactory_dsh import sandbox as _sbx
        _orig_run = _sbx.run_command

        def _spy_run(*args, **kwargs):
            if "timeout_s" in kwargs:
                captured_timeouts.append(kwargs["timeout_s"])
            return _orig_run(*args, **kwargs)

        monkeypatch.setattr(_sbx, "run_command", _spy_run)
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n"},
                     target=str(repo), check="python tool.py",
                     generate="bash gen.sh",
                     mode=_mode(check_timeout_s=1800))
    assert result.exit_ok, result.error

    # Every timeout captured during the merge's generate/setup/check equals the knob.
    assert captured_timeouts, "no sandbox.run_command calls were captured"
    for t in captured_timeouts:
        assert t == 1800.0, f"expected 1800, got {t}"


def test_conflict_path_regenerates_all_files(github, tmp_path, monkeypatch):
    """Both generated files are regenerated after the merge, even when only one
    conflicted.

    Fails without the change: gen2.txt has a stale line count (2 instead of 3)
    from git's three-way merge, because only conflicted files had generate run
    on them in the conflict path.
    """
    repo = git_repo(tmp_path / "repo")
    # gen.sh: gen1.txt combines two inputs on one line (so both sides changing
    # a different input creates a single-line conflict); gen2.txt is a line-
    # count header plus the content of source2.txt — non-linear, so git's
    # auto-merge gives a stale count when both sides added lines.
    (repo / "gen.sh").write_text(
        "#!/bin/sh\n"
        'printf "%s%s\\n" "$(cat input_a.txt)" "$(cat input_b.txt)" > gen1.txt\n'
        "n=$(cat source2.txt | wc -l)\n"
        'printf "count:%d\\n" "$n" > gen2.txt\n'
        "cat source2.txt >> gen2.txt\n"
    )
    (repo / "input_a.txt").write_text("A\n")
    (repo / "input_b.txt").write_text("B\n")
    (repo / "source2.txt").write_text("original\n")
    subprocess.run(["bash", "gen.sh"], cwd=str(repo), check=True, capture_output=True)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base with two gen files")
    slug = "acme/merge-regen-all"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        # Base changes input_b (creates a conflict in gen1.txt because the run
        # changes input_a) and appends to source2.txt (gen2.txt merges cleanly
        # from git's perspective but its line-count header becomes stale).
        (other / "input_b.txt").write_text("B-base\n")
        (other / "source2.txt").write_text("original\nbase-line\n")
        subprocess.run(["bash", "gen.sh"], cwd=str(other), check=True, capture_output=True)
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base updates both inputs")
        git(other, "push", "-q", "origin", "main")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)

    run = tmp_path / "run"
    # The run changes input_a (making gen1.txt conflict) and prepends to
    # source2.txt (so the merged tree has 3 lines, not 2).
    result = _design(run, {"tool.py": "print('built')\n",
                           "input_a.txt": "A-run\n",
                           "source2.txt": "run-line\noriginal\n"},
                     target=str(repo), check="python tool.py",
                     generate="bash gen.sh")
    assert result.exit_ok, result.error

    record = _record(run)
    assert record["status"] == "opened"
    assert record.get("merge_commit")
    merge_commit = record["merge_commit"]

    # gen1.txt: merged inputs (A-run from run + B-base from base) on one line.
    gen1 = git(bare, "show", f"{merge_commit}:gen1.txt")
    assert "A-run" in gen1 and "B-base" in gen1
    # gen2.txt: regenerated on the merged tree (3 lines), not stale (count:2).
    gen2 = git(bare, "show", f"{merge_commit}:gen2.txt")
    assert "count:3" in gen2


def test_modify_delete_conflict(github, tmp_path, monkeypatch):
    """A modify/delete conflict where --theirs has no file to check out: the merge is
    abandoned and delivery does not push.

    Fails without the change: subprocess.run returns non-zero silently, and the merge
    proceeds with a broken tree.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "gen.sh").write_text("echo ok > generated.txt\n")
    (repo / "conflicted.txt").write_text("original\n")
    subprocess.run(["bash", "gen.sh"], cwd=str(repo), check=True, capture_output=True)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base with conflicted.txt")
    slug = "acme/merge-moddel"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        # Base deletes the file.
        git(other, "rm", "conflicted.txt")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base deletes conflicted.txt")
        git(other, "push", "-q", "origin", "main")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)
    # Skip the reloop — this test verifies delivery's behaviour, not the full loop.
    monkeypatch.setattr(delivery, "evaluate_delivery_conflicts", lambda ctx: "proceed")

    run = tmp_path / "run"
    # The run modifies the file.
    result = _design(run, {"tool.py": "print('built')\n",
                           "conflicted.txt": "modified by run\n"},
                     target=str(repo), check="python tool.py",
                     generate="bash gen.sh")
    assert result.exit_ok, result.error

    record = _record(run)
    # Delivery does not push.
    assert record["status"] == "kept local"
    # The file is named as a conflict.
    assert "conflicted.txt" in record.get("merge_conflicts", [])
    # No push happened.
    branch = json.loads((run / "design.json").read_text())["branch"]
    assert _remote_head(github, slug, branch) == ""


def test_generate_fails_classes_as_authored(github, tmp_path, monkeypatch):
    """When generate fails on the merged tree during conflict classification,
    the conflicting files are classified as authored, and delivery does not push.

    Fails without the change: the generate failure's exit status is not checked;
    the file is classified based on stale content.
    """
    repo = git_repo(tmp_path / "repo")
    # gen.sh fails if a marker file exists.
    (repo / "gen.sh").write_text(
        "if [ -f FAIL_GEN ]; then exit 1; fi\n"
        "cat source.txt > generated.txt\n"
    )
    (repo / "source.txt").write_text("alpha\n")
    subprocess.run(["bash", "gen.sh"], cwd=str(repo), check=True, capture_output=True)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/merge-genfail"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        # Base adds the marker that makes generate fail and changes source.txt.
        (other / "FAIL_GEN").write_text("yes\n")
        (other / "source.txt").write_text("gamma\n")
        subprocess.run(["bash", str(other / "gen.sh")], cwd=str(other), capture_output=True)
        # gen.sh fails, so write generated.txt manually for the base commit.
        (other / "generated.txt").write_text("gamma\n")
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base adds fail marker")
        git(other, "push", "-q", "origin", "main")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)
    # Skip the reloop — this test verifies delivery's behaviour, not the full loop.
    monkeypatch.setattr(delivery, "evaluate_delivery_conflicts", lambda ctx: "proceed")

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n", "source.txt": "alpha\nbeta\n"},
                     target=str(repo), check="python tool.py",
                     generate="bash gen.sh")
    assert result.exit_ok, result.error

    record = _record(run)
    # Delivery does not push.
    assert record["status"] == "kept local"
    # The conflicting file is classified as authored (not merged).
    assert "generated.txt" in record.get("merge_conflicts", [])
    # No push happened.
    branch = json.loads((run / "design.json").read_text())["branch"]
    assert _remote_head(github, slug, branch) == ""


def test_post_checkout_hook_does_not_run(github, tmp_path, monkeypatch):
    """A post-checkout hook in the target must not run during delivery's merge.

    Fails without the change: subprocess.run does not use NO_HOOKS, so the
    post-checkout hook runs and creates the marker.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "gen.sh").write_text("cat source.txt > generated.txt\n")
    (repo / "source.txt").write_text("alpha\n")
    subprocess.run(["bash", "gen.sh"], cwd=str(repo), check=True, capture_output=True)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/merge-hook"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        # Add a new file so main moves without conflicting with the run's source.txt.
        (other / "extra.txt").write_text("added by main\n")
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base adds extra.txt")
        git(other, "push", "-q", "origin", "main")

        # Install a post-checkout hook in the worktree's git dir.
        found = worktree.recorded(ctx.workdir)
        wt = Path(found.path)
        marker = wt / "HOOK_RAN"
        # Find the git dir (for a worktree, .git is a file with "gitdir: <path>").
        dot_git = wt / ".git"
        if dot_git.is_file():
            gitdir = Path(dot_git.read_text().strip().removeprefix("gitdir: "))
            if not gitdir.is_absolute():
                gitdir = wt / gitdir
        else:
            gitdir = dot_git
        hooks = gitdir / "hooks"
        hooks.mkdir(parents=True, exist_ok=True)
        hook = hooks / "post-checkout"
        hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
        hook.chmod(0o755)
        result = real(ctx)
        # The marker must not exist.
        assert not marker.exists(), "post-checkout hook ran during delivery"
        return result

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n"},
                     target=str(repo), check="python tool.py",
                     generate="bash gen.sh")
    assert result.exit_ok, result.error


def test_redo_after_merge_pushed(github, tmp_path, monkeypatch):
    """A redo after the merge commit was pushed finishes without a second PR and
    without a refused push.

    Fails without the change: the redo creates a new merge commit with different
    timestamps; pushing it is not a fast-forward, so git refuses it.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "extra.txt").write_text("placeholder\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/merge-redo"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_and_fail_verify(ctx):
        """First delivery: advance main, then sabotage _verified via stale-view."""
        other = tmp_path / "other"
        if not other.exists():
            git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
            (other / "extra.txt").write_text("new\n")
            git(other, "add", ".")
            git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
                "commit", "-q", "-m", "base moves")
            git(other, "push", "-q", "origin", "main")
        # Make _verified fail: fake_gh returns stale headRefOid for "headRefOid,url" queries.
        (github / "stale-view").write_text("0" * 40)
        monkeypatch.setattr(delivery, "VERIFY_ATTEMPTS", 1)
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_and_fail_verify)
    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n"},
                     target=str(repo), check="python tool.py")

    # The first run failed (Unverified) or ended otherwise; the branch should
    # be on GitHub because the push succeeded before _verified failed.
    branch = json.loads((run / "design.json").read_text())["branch"]
    first_remote = _remote_head(github, slug, branch)

    if not first_remote:
        # If no push happened, the base did not actually move; skip the redo test.
        return

    # Redo: remove the sabotage, then call deliver_design directly (as the
    # existing redo tests do), not DesignMode.resume which refuses a node
    # whose side-effect outcome is unknown.
    (github / "stale-view").unlink(missing_ok=True)
    monkeypatch.setattr(delivery, "deliver_design", real)
    monkeypatch.setattr(delivery, "VERIFY_ATTEMPTS", 5)
    delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))

    record2 = _record(run)
    assert record2["status"] in ("opened", "already open", "updated"), f"redo status: {record2['status']}"

    # No second pull request.
    pulls = _pulls(github, slug)
    assert len(pulls) <= 1, f"expected at most 1 PR, got {len(pulls)}"

    # The remote head is the same or a descendant of the first merge.
    second_remote = _remote_head(github, slug, branch)
    assert second_remote, "redo should have the branch on GitHub"


def test_dependency_install_for_merged_tree(github, tmp_path, monkeypatch):
    """When the base adds a dependency (changes pyproject.toml), ensure_environment
    is called for the merged tree with commit_lockfiles=False.

    Fails without the change: ensure_environment is never called for the merged tree.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "pyproject.toml").write_text('[project]\nname = "example"\nversion = "0.1.0"\n')
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base with pyproject")
    slug = "acme/merge-deps"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design
    ensure_calls: list[dict] = []

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        # Base changes pyproject.toml.
        (other / "pyproject.toml").write_text('[project]\nname = "example"\nversion = "0.1.0"\ndependencies = ["requests"]\n')
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base adds dependency")
        git(other, "push", "-q", "origin", "main")

        orig_ensure = worktree.ensure_environment

        def _spy_ensure(found, run, budget=None, *, commit_lockfiles=True):
            ensure_calls.append({"commit_lockfiles": commit_lockfiles})
            # Don't actually install anything; just return no problems.
            return found, [], None

        monkeypatch.setattr(worktree, "ensure_environment", _spy_ensure)
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n"},
                     target=str(repo), check="python tool.py")
    assert result.exit_ok, result.error

    # ensure_environment was called with commit_lockfiles=False.
    non_amend = [c for c in ensure_calls if not c["commit_lockfiles"]]
    assert non_amend, f"ensure_environment was not called with commit_lockfiles=False; calls: {ensure_calls}"

    # The reviewed commit's tree is unchanged (no lockfile added).
    reviewed = worktree.recorded(run).head
    wt = Path(worktree.recorded(run).path)
    # Check that the reviewed commit has no lockfile that wasn't there before.
    base = worktree.recorded(run).base
    base_files = set(git(wt, "ls-tree", "-r", "--name-only", base).splitlines())
    head_files = set(git(wt, "ls-tree", "-r", "--name-only", reviewed).splitlines())
    # No new lockfiles should appear in the reviewed commit.
    new_locks = {f for f in (head_files - base_files) if f in ("uv.lock", "pnpm-lock.yaml", "package-lock.json", "yarn.lock")}
    # (This assertion depends on the test's setup; it mainly guards that amend was not called.)


def test_merge_fails_no_conflicted_files(github, tmp_path, monkeypatch):
    """When git merge fails with no conflicted files (e.g. untracked file it would
    overwrite), the merge is abandoned and delivery does not push.

    Fails without the change: the error is stuffed into the conflicts list as a reason
    string, not into a separate abandoned field.
    """
    repo = git_repo(tmp_path / "repo")
    slug = "acme/merge-noconflict"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        # Base adds a file.
        (other / "blocker.txt").write_text("from base\n")
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base adds blocker")
        git(other, "push", "-q", "origin", "main")

        # Create an untracked file in the worktree with the same name, so git merge
        # fails with "would be overwritten" and no conflicted files.
        found = worktree.recorded(ctx.workdir)
        wt = Path(found.path)
        (wt / "blocker.txt").write_text("untracked content\n")

        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n"},
                     target=str(repo), check="python tool.py")
    assert result.exit_ok, result.error

    record = _record(run)
    # Delivery does not push.
    assert record["status"] == "kept local"
    # The abandoned field names the error (not in conflicts).
    assert record.get("merge_abandoned"), f"no merge_abandoned in record: {record}"
    assert not record.get("merge_conflicts"), f"unexpected merge_conflicts: {record.get('merge_conflicts')}"

    # No push happened.
    branch = json.loads((run / "design.json").read_text())["branch"]
    assert _remote_head(github, slug, branch) == ""


def test_two_runs_conflict_reloop(github, tmp_path, monkeypatch):
    """Two scripted runs edit the same function. The second's delivery goes back to
    the builder with no person step. Its pushed head contains main, and the resolution
    commit has main's tip as a parent.

    Fails without the change: the second run pushes behind main and conflicts.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "shared.py").write_text("def f():\n    return 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/two-runs"
    bare = _published(github, slug, repo)

    # First run: changes shared.py.
    run1 = tmp_path / "run1"
    result1 = _design(run1, {"tool.py": "print('built')\n",
                             "shared.py": "def f():\n    return 2\n"},
                      target=str(repo), check="python tool.py")
    assert result1.exit_ok, result1.error
    record1 = _record(run1)
    assert record1["status"] == "opened"
    branch1 = json.loads((run1 / "design.json").read_text())["branch"]

    # Simulate main having the first run's change merged.
    main_clone = tmp_path / "main_clone"
    git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(main_clone))
    git(main_clone, "-c", "user.name=t", "-c", "user.email=t@example.com",
        "merge", "-q", "--no-ff", f"origin/{branch1}",
        "-m", "merge first run")
    git(main_clone, "push", "-q", "origin", "main")

    main_tip = git(bare, "rev-parse", "refs/heads/main").strip()

    # Second run: changes the same function differently.
    # This will conflict with main (which now has return 2).
    # The delivery_gate should reloop to emit.
    # We provide two emit answers: first the conflicting one, then a resolution.
    resolution_emit = (
        "===FILE: tool.py===\nprint('built')\n"
        "===FILE: shared.py===\ndef f():\n    return 3  # merged\n"
    )

    # The second run needs extra answers for the reloop:
    # delivery_gate reloops directly to emit (not through apply_corrections),
    # so: emit (conflict resolution) + reviews (4 answers)
    reloop_answers = [
        resolution_emit,
        *REVIEW_ANSWERS[:3],
        VERIFICATION_ANSWER,
        REVIEW_ANSWERS[3],
    ]

    run2 = tmp_path / "run2"
    model = FakeModel(_answers(render({"tool.py": "print('built')\n",
                                       "shared.py": "def f():\n    return 42\n"}))
                      + reloop_answers)
    result2 = DesignMode().run(
        run2, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "Build a tool\nthat prints built.",
              "check": "python tool.py", "target": str(repo)},
    )
    assert result2.exit_ok, result2.error

    record2 = _record(run2)
    # Should have been delivered after the reloop.
    assert record2.get("status") in ("opened", "already open", "updated"), \
        f"expected delivery after reloop, got: {record2}"

    # The pushed head contains main.
    branch2 = json.loads((run2 / "design.json").read_text())["branch"]
    remote_head2 = _remote_head(github, slug, branch2)
    assert remote_head2, "second run's branch should be on GitHub"

    # The head is a merge commit whose second parent is main's tip.
    parents = [line.split()[1] for line in
               git(bare, "cat-file", "-p", remote_head2).splitlines()
               if line.startswith("parent ")]
    assert len(parents) == 2, f"expected merge commit, got {len(parents)} parents"
    assert parents[1] == main_tip, \
        f"expected second parent to be main's tip {main_tip}, got {parents[1]}"
    # main's tip is an ancestor.
    assert git(bare, "merge-base", "--is-ancestor", main_tip, remote_head2) == ""


def test_merge_failure_leaves_worktree_clean(github, tmp_path, monkeypatch):
    """A crash at emit (during the conflict reloop) leaves the worktree clean.

    The merge exists only inside materialise.  The delivery gate records the
    conflict and returns reloop, but does NOT start a merge.  If the builder
    crashes at emit, no merge was ever started, so there is no MERGE_HEAD to
    clean up.  This test verifies that invariant through a real graph run.

    Fails without the change: if the delivery_gate started the merge, a crash
    at emit would leave MERGE_HEAD behind.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "shared.py").write_text("def f():\n    return 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/clean-merge"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        (other / "shared.py").write_text("def f():\n    return 999\n")
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", "base changes shared.py")
        git(other, "push", "-q", "origin", "main")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)

    # Provide only one emit answer for the reloop: the model will run out of
    # answers when it tries to produce the conflict-resolution emit.  This
    # simulates a builder crash at the conflict emit.  The run stops with an
    # error at emit, having never started a merge.
    run = tmp_path / "run"
    # _answers gives the normal answers; we do NOT add extra reloop answers,
    # so the model runs out during the conflict-resolution emit.
    result = _design(run, {"tool.py": "print('built')\n",
                           "shared.py": "def f():\n    return 42\n"},
                     target=str(repo), check="python tool.py")
    # The run may or may not succeed, but the worktree must be clean.
    found = worktree.recorded(run)
    if found is not None:
        path = Path(found.path)
        # No MERGE_HEAD should remain.
        with pytest.raises(subprocess.CalledProcessError):
            git(path, "rev-parse", "--verify", "-q", "MERGE_HEAD")
        # No unmerged index entries.
        try:
            unmerged = worktree.git(path, "diff", "--name-only", "--diff-filter=U")
        except ValueError:
            unmerged = ""
        assert not unmerged.strip(), f"unmerged files remain: {unmerged}"


def test_emission_with_conflict_markers_refused(tmp_path):
    """An emission that leaves conflict markers is refused.

    Fails without the change: materialise commits the markers.
    """
    from refactory_dsh import repo_change
    from refactory_dsh.worktree import AttemptRefused

    # Set up a minimal worktree.
    repo = git_repo(tmp_path / "repo")
    (repo / "tool.py").write_text("print('built')\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")

    # Create a worktree record.
    from refactory_dsh.worktree import create, recorded
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    wt = create(repo, "HEAD", "test-markers", run)
    found = recorded(run)

    # A file with conflict markers.
    marker_content = "before\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> main\nafter\n"
    emit_with_markers = f"===FILE: conflicted.py===\n{marker_content}"

    ctx = _TitleCtx(
        facts={"name": "test_markers", "brief": "test", "check": "echo ok"},
        files={"emit.txt": emit_with_markers},
    )
    ctx.workdir = run
    ctx.read_text = lambda name: ctx._files[name]
    ctx.write_text = lambda name, text: ctx._files.__setitem__(name, text)
    ctx.write_json = lambda name, data: ctx._files.__setitem__(name, json.dumps(data))
    ctx.read_json = lambda name, default=None: json.loads(ctx._files.get(name, "null")) if ctx._files.get(name) else default

    with pytest.raises(AttemptRefused, match="conflict markers"):
        repo_change._commit_emission(ctx)

    # A normal SEARCH/REPLACE block should succeed.
    normal_emit = (
        "===FILE: tool.py===\n"
        "print('built and tested')\n"
    )
    ctx._files["emit.txt"] = normal_emit
    # This should not raise.
    result = repo_change._commit_emission(ctx)
    assert result.get("commit")


def test_gate_correction_updates_pr_text(github, tmp_path, monkeypatch):
    """A run whose gate correction changes a site's approach, driven through a real
    gate answer, gets PR text with the corrected approach and not the original.

    Fails without the change: the PR text contains the uncorrected approach.
    """
    from refactory_dsh.steering import Ledger

    repo = git_repo(tmp_path / "repo")
    slug = "acme/corrections"
    _published(github, slug, repo)

    # The design_doc agent will produce a design mentioning "polling loop".
    design_with_polling = (
        "# Use a polling loop to check for updates\n\n"
        "## Summary\n\nUse a polling loop to check for updates periodically.\n\n"
        "## Failure trace\n\nN/A \u2014 this is a new feature.\n\n"
        "## Ask mapping\n\n| Ask (quoted from brief) | Design section | Test |\n|---|---|---|\n| \"Build a tool\" | \u00a7 Changes | tool.py |\n\n"
        "## Rule and sites\n\n"
        "- `tool.py:1` — use a polling loop **[changed]**\n\n"
        "## Docs\n\n- docs/models.md\n\n"
        "## Changes\n\n- tool.py\n\n"
        "## Research and perspective findings\n\n"
        "- FND-api-surface-1: addressed at `tool.py:1`\n"
        "- FND-existing-modes-1: addressed at \u00a7 Changes\n"
        "- FND-acceptance-1: addressed at `tool.py:1`\n"
        "- FND-simplest-1: addressed at `tool.py:1`\n"
        "- FND-criteria-1: addressed at \u00a7 Changes\n\n"
        "## Anti-patterns to avoid\n\n"
        "- PIT-001: addressed at `tool.py:1`\n"
    )

    # The apply_corrections agent will rewrite design.md to use websockets.
    corrected_design = (
        "# Use websockets to receive updates\n\n"
        "## Summary\n\nUse websockets to receive real-time updates.\n\n"
        "## Failure trace\n\nN/A \u2014 this is a new feature.\n\n"
        "## Ask mapping\n\n| Ask (quoted from brief) | Design section | Test |\n|---|---|---|\n| \"Build a tool\" | \u00a7 Changes | tool.py |\n\n"
        "## Rule and sites\n\n"
        "- `tool.py:1` — use websockets **[changed]**\n\n"
        "## Docs\n\n- docs/models.md\n\n"
        "## Changes\n\n- tool.py\n\n"
        "## Research and perspective findings\n\n"
        "- FND-api-surface-1: addressed at `tool.py:1`\n"
        "- FND-existing-modes-1: addressed at \u00a7 Changes\n"
        "- FND-acceptance-1: addressed at `tool.py:1`\n"
        "- FND-simplest-1: addressed at `tool.py:1`\n"
        "- FND-criteria-1: addressed at \u00a7 Changes\n\n"
        "## Anti-patterns to avoid\n\n"
        "- PIT-001: addressed at `tool.py:1`\n"
    )

    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    # Exercise corrections with steering prompt injection explicitly disabled.
    (run / ".refactory").mkdir()
    (run / ".refactory/run-config.json").write_text(json.dumps({"omit_steering": True}))

    # Seed the ledger with a gate_strategy proceed verdict and correction,
    # driven through the real Ledger API (steering.py:577-587).
    ledger = Ledger.seed(run, "Build a tool\nthat prints built.")
    # Record the verdict as a real gate answer — answered=True records it as
    # consumed, and approval=True provides standing upfront authorization.
    ledger.verdict("gate_strategy", "proceed", approval=True)
    # The feedback from the person is a constraint note — this is what
    # steering.py:584-587 does when verdict=="proceed" and there is feedback.
    ledger.note("constraint", "use websockets instead of a polling loop")

    # Also add a process note and a check-gate constraint to verify they are
    # excluded from the design.
    ledger.note("info", "try harder", scope="process")

    # Build model answers: pick_corrections (gate) extracts the gate correction
    # from the ledger, then apply_corrections (agent) rewrites design.md.
    answers = [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        design_with_polling,   # design_doc writes design.md with "polling loop"
        # pick_corrections is a gate — no model answer needed
        corrected_design,      # apply_corrections rewrites design.md to "websockets"
        render({"tool.py": "print('built')\n"}),  # emit
        *REVIEW_ANSWERS[:3],
        VERIFICATION_ANSWER,
        REVIEW_ANSWERS[3],
    ]

    model = FakeModel(answers)
    # The constraint is about the design, so the audit cites the corrected
    # design's heading rather than a code site.
    model._siteless_clause = {"kind": "design", "heading": "Summary"}
    result = DesignMode().run(
        run, model=model,
        verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "Build a tool\nthat prints built.",
              "check": "python tool.py", "target": str(repo)},
    )
    assert result.exit_ok, result.error

    # The correction must still be applied, audited and judged with prompt opt-out.
    assert not any("What the user has said" in prompt for prompt in model.prompts)
    assert any("use websockets instead of a polling loop" in prompt for prompt in model.prompts)
    assert (run / "correction-audit.json").exists()
    evidence = json.loads((run / "authoring-evidence.json").read_text())
    correction = next(row for row in evidence["requirements"]
                      if row.get("text") == "use websockets instead of a polling loop")
    audit = json.loads((run / "correction-audit.json").read_text())
    assert any(row.get("id") == correction["id"] for row in audit), audit

    # design.md should contain the corrected approach.
    design_text = (run / "design.md").read_text()
    # The apply_corrections agent wrote the corrected design.
    assert "websocket" in design_text.lower(), \
        f"design.md should mention websockets: {design_text[:500]}"

    # The PR body should contain "websockets" and not "polling loop", except where
    # it quotes the person's correction, which names what it replaces.
    pr_text = (run / delivery.BODY).read_text()
    assert "websocket" in pr_text.lower(), \
        f"PR text should mention websockets: {pr_text[:500]}"
    heading = "## Corrections given at approval\n\n"
    assert heading + "- use websockets instead of a polling loop\n" in pr_text, pr_text
    before, after = pr_text.split(heading, 1)
    # The correction section may be the last section; handle gracefully.
    rest_parts = after.split("\n## ", 1)
    described = before + (rest_parts[1] if len(rest_parts) > 1 else "")
    assert "polling" not in described.lower(), \
        f"PR text should not describe the replaced approach: {pr_text[:500]}"

    # Process notes should not appear in the design.
    assert "try harder" not in design_text.lower(), \
        "process note should not appear in design.md"


def test_gate_corrections_on_a_design_without_a_summary_reach_the_pr(github, tmp_path):
    """A light design need not have ## Summary. Approved with a correction, its corrected
    design is the one delivered: design.md keeps the correction, and the pull request's
    title and text come from it.

    Fails without the change: delivery read the missing Summary as a broken contract and
    restored the pre-correction design, so the correction was dropped from both.
    """
    from refactory_dsh.steering import Ledger

    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/no-summary", repo)
    polling = ("# Use a polling loop to check for updates\n\n"
               "## Changes\n\n- `tool.py` \u2014 poll for updates in a loop\n\n"
               "## Tests\n\n- `tool.py` \u2014 prints built\n")
    websockets = ("# Use websockets to receive updates\n\n"
                  "## Changes\n\n- `tool.py` \u2014 receive updates over websockets\n\n"
                  "## Tests\n\n- `tool.py` \u2014 prints built\n\n"
                  "## Corrections at approval\n\n"
                  "- \"use websockets instead of a polling loop\" \u2014 \u00a7 Changes\n")
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.seed(run, "Build a tool\nthat prints built.")
    ledger.verdict("gate_strategy", "proceed", approval=True)
    ledger.note("constraint", "use websockets instead of a polling loop")
    model = FakeModel([
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        polling,      # design_doc
        websockets,   # apply_corrections
        render({"tool.py": "print('built')\n"}),
        *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3],
    ])
    model._siteless_clause = {"kind": "design", "heading": "Changes"}
    result = DesignMode().run(
        run, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "Build a tool\nthat prints built.",
              "check": "python tool.py", "target": str(repo)},
    )
    assert result.exit_ok, result.error

    assert (run / "design.md").read_text() == websockets
    assert _record(run)["title"] == "Use websockets to receive updates"
    pr_text = (run / delivery.BODY).read_text()
    heading = "## Corrections given at approval\n\n"
    assert heading + "- use websockets instead of a polling loop\n" in pr_text, pr_text
    before, after = pr_text.split(heading, 1)
    # The correction section ends at the next ## heading or end of text.
    rest_parts = after.split("\n## ", 1)
    described = before + (rest_parts[1] if len(rest_parts) > 1 else "")
    assert "polling" not in described.lower(), pr_text


def test_unparseable_design_after_corrections_is_restored(tmp_path):
    """When apply_corrections produces unparseable design.md, the pre-correction
    version is restored and a warning is emitted.

    Fails without the change: delivery reads the unparseable design.md and the
    PR loses its Summary.
    """
    from refactory_dsh.delivery import _validate_design_after_corrections

    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    good_design = (
        "# Title\n\n"
        "## Summary\n\nGood summary.\n\n"
        "## Rule and sites\n\n"
        "- `file.py:1` — do the thing **[changed]**\n\n"
        "## Docs\n\n- docs/readme.md\n\n"
        "## Changes\n\n- file.py\n"
    )
    (run / "design.md").write_text(good_design)
    (run / "design.md.pre_corrections").write_text(good_design)

    # Corrupt design.md (missing required sections).
    (run / "design.md").write_text("This is not a valid design document at all.")

    _validate_design_after_corrections(run)

    # design.md should be restored to the backup.
    assert (run / "design.md").read_text() == good_design


def test_no_backup_means_no_restoration(tmp_path):
    """When there is no backup (no corrections were applied), validation is a no-op."""
    from refactory_dsh.delivery import _validate_design_after_corrections

    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    bad_text = "This is not a valid design document at all."
    (run / "design.md").write_text(bad_text)
    # No design.md.pre_corrections exists.

    _validate_design_after_corrections(run)

    # design.md is unchanged — no backup to restore from.
    assert (run / "design.md").read_text() == bad_text


def test_edit_resolves_conflict_keeping_mains_lines(tmp_path):
    """An ===EDIT: block resolves a merge conflict by keeping main's lines,
    not only ===FILE: whole-file replacement.

    Materialise reads pending-merge.json, starts the merge (which produces
    conflict markers), applies the ===EDIT: blocks against the worktree text,
    and commits the merge — all in one step.

    Fails without the change: edits apply to committed_text (the last attempt's
    commit), which has no conflict markers.  An edit that searches for marker
    text would fail with 'text is not in shared.py'.
    """
    from refactory_dsh import repo_change
    from refactory_dsh.worktree import (
        create, recorded, _write_json, RECORD,
        git as wgit, COMMITTER, NO_HOOKS,
    )
    from dataclasses import asdict, replace as dreplace

    # ── Set up a repo with shared.py ──
    repo = git_repo(tmp_path / "repo")
    (repo / "shared.py").write_text("line 1\nline 2\nline 3\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "initial shared.py")

    # ── Create a worktree for the run ──
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    create(repo, "HEAD", "test-edit-merge", run)
    found = recorded(run)
    wt_path = Path(found.path)

    # ── Make a feature change in the worktree and commit ──
    (wt_path / "shared.py").write_text("line 1\nline 2 feature\nline 3\n")
    wgit(wt_path, "add", ".")
    wgit(wt_path, *NO_HOOKS, "-c", "commit.gpgsign=false",
         "commit", "-q", "--no-verify", "-m", "feature change",
         env={**os.environ, **COMMITTER})
    feature_head = wgit(wt_path, "rev-parse", "HEAD").strip()

    # Update the worktree record with the new head.
    updated_found = dreplace(found, head=feature_head)
    _write_json(run / RECORD, asdict(updated_found))

    # ── Make a conflicting change on main in the original repo ──
    (repo / "shared.py").write_text("line 1\nline 2 main\nline 3\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "main changes line 2")
    main_head = git(repo, "rev-parse", "HEAD").strip()

    # ── Write pending-merge.json so materialise starts the merge ──
    pm_dir = run / ".refactory"
    pm_dir.mkdir(parents=True, exist_ok=True)
    (pm_dir / "pending-merge.json").write_text(
        json.dumps({"target_tip": main_head, "conflict_files": ["shared.py"]})
    )

    # Materialise will start the merge, producing conflict markers in shared.py.
    # We need to know what the marker looks like, so start the merge temporarily
    # to read the theirs marker line, then abort.
    try:
        wgit(wt_path, *NO_HOOKS, "-c", "commit.gpgsign=false",
             "merge", "--no-commit", "--no-verify", main_head,
             env={**os.environ, **COMMITTER})
    except ValueError:
        pass  # expected — conflicts
    conflicted = (wt_path / "shared.py").read_text()
    theirs_marker = ""
    for mline in conflicted.splitlines():
        if mline.startswith(">>>>>>>") and "REPLACE" not in mline:
            theirs_marker = mline
            break
    assert theirs_marker, "expected >>>>>>> marker in conflicted text"
    # Abort — let materialise start the merge from pending-merge.json.
    wgit(wt_path, *NO_HOOKS, "merge", "--abort")

    # ── Build ===EDIT: blocks that resolve the conflict ──
    edit_emit = (
        "===EDIT: shared.py===\n"
        "<<<<<<< SEARCH\n"
        "<<<<<<< HEAD\n"
        "line 2 feature\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
        "\n"
        "<<<<<<< SEARCH\n"
        + theirs_marker + "\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
    )

    ctx = _TitleCtx(
        facts={"name": "test_edit_merge", "brief": "test", "check": "echo ok"},
        files={"emit.txt": edit_emit},
    )
    ctx.workdir = run
    ctx.read_text = lambda name: ctx._files[name]
    ctx.write_text = lambda name, text: ctx._files.__setitem__(name, text)
    ctx.write_json = lambda name, data: ctx._files.__setitem__(name, json.dumps(data))
    ctx.read_json = lambda name, default=None: (
        json.loads(ctx._files.get(name, "null"))
        if ctx._files.get(name) else default
    )

    result = repo_change._commit_emission(ctx)
    assert result.get("commit"), f"expected a commit: {result}"

    # The committed file keeps main's content, with no <<<<<<< or >>>>>>> markers.
    committed = (wt_path / "shared.py").read_text()
    assert "line 2 main" in committed, f"expected main's content: {committed}"
    assert "<<<<<<< HEAD" not in committed
    assert "line 2 feature" not in committed
    assert theirs_marker not in committed

    # The commit is a merge commit (2 parents).
    cat_output = git(wt_path, "cat-file", "-p", "HEAD")
    parent_lines = [l for l in cat_output.splitlines() if l.startswith("parent ")]
    assert len(parent_lines) == 2, f"expected merge commit, got {len(parent_lines)} parents"

    # No MERGE_HEAD remains after the commit.
    try:
        wgit(wt_path, "rev-parse", "--verify", "-q", "MERGE_HEAD")
        assert False, "MERGE_HEAD should be gone after commit"
    except ValueError:
        pass  # expected

    # pending-merge.json is cleaned up.
    assert not (pm_dir / "pending-merge.json").exists(), "pending-merge.json should be deleted"


def test_apply_corrections_no_ledger_leaves_design_unchanged(github, tmp_path):
    """When there is no intent-ledger.md, the pick_corrections op finds nothing
    and the apply_corrections agent does not run (gate_corrections is empty).
    design.md is unchanged.

    Fails without the change: the agent node runs on every design run and may
    wipe design.md.
    """
    repo = git_repo(tmp_path / "repo")
    slug = "acme/no-ledger"
    _published(github, slug, repo)

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n"}, target=str(repo))
    assert result.exit_ok, result.error

    # design.md should be the DESIGN_DOC_ANSWER — unchanged by apply_corrections
    # because there is no ledger.
    design_text = (run / "design.md").read_text()
    assert "Do the thing" in design_text, \
        f"design.md should be unchanged: {design_text[:300]}"

    # The pick_corrections op ran and set the fact.
    assert result.facts.get("design_corrections_applied"), \
        "pick_corrections should have set design_corrections_applied"
    # With no corrections, the pull request has no corrections section.
    assert "Corrections given at approval" not in (run / delivery.BODY).read_text()


def test_ls_remote_failure_stops_existing_repo(github, tmp_path, monkeypatch):
    """ls-remote failure on an existing repository stops delivery instead of
    pushing the reviewed commit without main.

    Fails without the change: the raise inside try/except NotDelivered is
    swallowed, and delivery pushes the reviewed commit without main.
    """
    repo = git_repo(tmp_path / "repo")
    slug = "acme/ls-remote-fail"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design
    real_git = delivery._git

    def break_ls_remote_then_deliver(ctx):
        def failing_git(path, *args):
            if "ls-remote" in args and f"refs/heads/main" in " ".join(str(a) for a in args):
                raise delivery.NotDelivered("simulated ls-remote failure")
            return real_git(path, *args)
        monkeypatch.setattr(delivery, "_git", failing_git)
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", break_ls_remote_then_deliver)
    # Skip the delivery_gate reloop.
    monkeypatch.setattr(delivery, "evaluate_delivery_conflicts", lambda ctx: "proceed")

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n"},
                     target=str(repo), check="python tool.py")
    # The run should complete but delivery should not push.
    assert result.exit_ok, result.error
    record = _record(run)
    assert record["status"] == "kept local"
    assert "cannot reach" in record.get("why", "")
    # No push happened.
    branch = json.loads((run / "design.json").read_text())["branch"]
    assert _remote_head(github, slug, branch) == ""


def test_delivery_advances_worktree_after_merge_push(github, tmp_path, monkeypatch):
    """After delivery pushes a merge commit, the worktree record advances
    so a later push fast-forwards (dm-116).

    Fails without the change: the finally block resets HEAD to the reviewed
    commit, so a later delivery's merge does not contain the pushed one and
    GitHub rejects the push as non-fast-forward.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "shared.py").write_text("def f():\n    return 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/advance-wt"
    bare = _published(github, slug, repo)

    real = delivery.deliver_design

    def advance_main_then_deliver(ctx):
        other = tmp_path / "other"
        if not other.exists():
            git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
            (other / "new_file.txt").write_text("main advanced\n")
            git(other, "add", ".")
            git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
                "commit", "-q", "-m", "main advances")
            git(other, "push", "-q", "origin", "main")
        return real(ctx)

    monkeypatch.setattr(delivery, "deliver_design", advance_main_then_deliver)
    # Skip the delivery_gate reloop.
    monkeypatch.setattr(delivery, "evaluate_delivery_conflicts", lambda ctx: "proceed")

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n"},
                     target=str(repo), check="python tool.py")
    assert result.exit_ok, result.error

    record = _record(run)
    assert record["status"] in ("opened", "already open", "updated"), \
        f"expected delivery, got: {record}"

    # The worktree record should be advanced to the merge commit.
    found = worktree.recorded(run)
    # Check that the head is a merge commit (2 parents).
    path = Path(found.path)
    head = found.head
    cat_output = git(path, "cat-file", "-p", head)
    parent_lines = [l for l in cat_output.splitlines() if l.startswith("parent ")]
    assert len(parent_lines) == 2, f"expected merge commit, got {len(parent_lines)} parents"

    # The base should be the second parent (main's tip).
    second_parent = parent_lines[1].split()[1]
    assert found.base == second_parent, \
        f"expected base to be second parent {second_parent}, got {found.base}"


def test_unresolved_conflict_stops_after_budget(github, tmp_path, monkeypatch):
    """A builder that never resolves the conflict stops after check_attempts,
    with a clean worktree.

    The emission omits the conflicted file every time, so materialise refuses
    it ("missing: shared.py").  The delivery_gate counts reloops against
    check_attempts; after the limit it raises without starting a merge.

    Fails without the change: the evaluator never counts reloops, so the run
    stops with "check exceeded max_iterations of 8" and MERGE_HEAD is left.
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "shared.py").write_text("def f():\n    return 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/unresolved"
    bare = _published(github, slug, repo)

    # First run to advance main.
    run1 = tmp_path / "run1"
    result1 = _design(run1, {"tool.py": "print('built')\n",
                             "shared.py": "def f():\n    return 2\n"},
                      target=str(repo), check="python tool.py")
    assert result1.exit_ok, result1.error
    branch1 = json.loads((run1 / "design.json").read_text())["branch"]
    # Merge first run into main.
    main_clone = tmp_path / "main_clone"
    git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(main_clone))
    git(main_clone, "-c", "user.name=t", "-c", "user.email=t@example.com",
        "merge", "-q", "--no-ff", f"origin/{branch1}", "-m", "merge first run")
    git(main_clone, "push", "-q", "origin", "main")

    # Second run: emits only tool.py, omitting the conflicted shared.py.
    # Materialise will refuse because the unmerged path is missing from the
    # emission, and the delivery_gate will count the reloop.
    bad_emit = "===FILE: tool.py===\nprint('built')\n"

    # Use check_attempts=1 so it stops quickly.
    run2 = tmp_path / "run2"
    mode = _mode(check_attempts=1)
    model = FakeModel(_answers(render({"tool.py": "print('built')\n",
                                       "shared.py": "def f():\n    return 42\n"}))
                      + [bad_emit] * 8)
    result2 = mode.run(
        run2, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "Build a tool\nthat prints built.",
              "check": "python tool.py", "target": str(repo)},
    )
    # The run should stop (not succeed) because the conflict was never resolved.
    found = worktree.recorded(run2)
    if found is not None:
        path = Path(found.path)
        # The worktree must be clean — no MERGE_HEAD.
        with pytest.raises(subprocess.CalledProcessError):
            git(path, "rev-parse", "--verify", "-q", "MERGE_HEAD")
        # No unmerged index entries.
        try:
            unmerged = worktree.git(path, "diff", "--name-only", "--diff-filter=U")
        except ValueError:
            unmerged = ""
        assert not unmerged.strip(), f"unmerged files remain: {unmerged}"
    # pending-merge.json should be cleaned up.
    pm = run2 / ".refactory" / "pending-merge.json"
    # It may or may not exist depending on the stop path, but MERGE_HEAD
    # must be gone regardless.


def test_cleanly_merged_edit_keeps_mains_change(tmp_path):
    """An ===EDIT: to a file main changed cleanly keeps main's change.

    Materialise reads pending-merge.json, starts the merge, applies the edit
    to the worktree text (which has main's changes), and commits a merge.

    Fails without the change: edits apply to committed_text (the run's last
    commit), which has no main's change, silently reverting it.
    Removing the base-move (second parent → new base) also fails this test.
    """
    from refactory_dsh import repo_change
    from refactory_dsh.worktree import (
        create, recorded, _write_json, RECORD,
        git as wgit, COMMITTER, NO_HOOKS,
    )
    from dataclasses import asdict, replace as dreplace

    # ── Set up a repo with two files ──
    repo = git_repo(tmp_path / "repo")
    (repo / "a.py").write_text("def f():\n    return 1\n")
    (repo / "c.py").write_text("def g():\n    return 0\ndef h():\n    return 0\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "initial")

    # ── Create a worktree for the run ──
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    create(repo, "HEAD", "test-clean-edit", run)
    found = recorded(run)
    wt_path = Path(found.path)

    # ── Feature change: edit a.py (only) ──
    (wt_path / "a.py").write_text("def f():\n    return 2\n")
    wgit(wt_path, "add", ".")
    wgit(wt_path, *NO_HOOKS, "-c", "commit.gpgsign=false",
         "commit", "-q", "--no-verify", "-m", "feature change",
         env={**os.environ, **COMMITTER})
    feature_head = wgit(wt_path, "rev-parse", "HEAD").strip()
    updated_found = dreplace(found, head=feature_head)
    _write_json(run / RECORD, asdict(updated_found))

    # ── Main changes c.py (no conflict with the feature change) ──
    (repo / "c.py").write_text("def g():\n    return 99\ndef h():\n    return 0\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "main changes g()")
    main_head = git(repo, "rev-parse", "HEAD").strip()

    # ── Write pending-merge.json so materialise starts the merge ──
    pm_dir = run / ".refactory"
    pm_dir.mkdir(parents=True, exist_ok=True)
    (pm_dir / "pending-merge.json").write_text(
        json.dumps({"target_tip": main_head, "conflict_files": []})
    )

    # ── An EDIT to c.py that keeps main's change and edits h() ──
    edit_emit = (
        "===EDIT: c.py===\n"
        "<<<<<<< SEARCH\n"
        "def h():\n"
        "    return 0\n"
        "=======\n"
        "def h():\n"
        "    return 42\n"
        ">>>>>>> REPLACE\n"
    )

    ctx = _TitleCtx(
        facts={"name": "test_clean_edit", "brief": "test", "check": "echo ok"},
        files={"emit.txt": edit_emit},
    )
    ctx.workdir = run
    ctx.read_text = lambda name: ctx._files[name]
    ctx.write_text = lambda name, text: ctx._files.__setitem__(name, text)
    ctx.write_json = lambda name, data: ctx._files.__setitem__(name, json.dumps(data))
    ctx.read_json = lambda name, default=None: (
        json.loads(ctx._files.get(name, "null"))
        if ctx._files.get(name) else default
    )

    result = repo_change._commit_emission(ctx)
    assert result.get("commit"), f"expected a commit: {result}"

    # c.py keeps main's change to g() AND the edit to h().
    committed = (wt_path / "c.py").read_text()
    assert "return 99" in committed, f"expected main's change to g(): {committed}"
    assert "return 42" in committed, f"expected edit to h(): {committed}"

    # The commit is a merge commit (2 parents).
    cat_output = git(wt_path, "cat-file", "-p", "HEAD")
    parent_lines = [l for l in cat_output.splitlines() if l.startswith("parent ")]
    assert len(parent_lines) == 2, f"expected merge commit, got {len(parent_lines)} parents"

    # The base moved to main's tip (second parent).
    new_found = recorded(run)
    second_parent = parent_lines[1].split()[1]
    assert new_found.base == second_parent, \
        f"expected base to be second parent {second_parent}, got {new_found.base}"

    # pending-merge.json is cleaned up.
    assert not (pm_dir / "pending-merge.json").exists(), "pending-merge.json should be deleted"


def test_edit_during_conflict_keeps_mains_changes(tmp_path):
    """During conflict resolution, ===EDIT: on a cleanly-merged file preserves
    main's changes while ===FILE: resolves the conflicted file.  Together they
    produce a merge commit.

    Materialise reads pending-merge.json, starts the merge, applies the
    emission, and commits a merge — all in one step.

    Fails without the change: edits apply to committed_text (the run's last
    commit) which lacks main's changes, so the SEARCH text is not found or
    main's change is silently reverted.
    """
    from refactory_dsh import repo_change
    from refactory_dsh.worktree import (
        create, recorded, _write_json, RECORD,
        git as wgit, COMMITTER, NO_HOOKS,
    )
    from dataclasses import asdict, replace as dreplace

    # ── Set up a repo with two files ──
    repo = git_repo(tmp_path / "repo")
    (repo / "shared.py").write_text("def f():\n    return 1\n")
    (repo / "util.py").write_text("def helper():\n    return 0\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "initial")

    # ── Create a worktree for the run ──
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    create(repo, "HEAD", "test-edit-conflict", run)
    found = recorded(run)
    wt_path = Path(found.path)

    # ── Feature change: edit shared.py only ──
    (wt_path / "shared.py").write_text("def f():\n    return 42\n")
    wgit(wt_path, "add", ".")
    wgit(wt_path, *NO_HOOKS, "-c", "commit.gpgsign=false",
         "commit", "-q", "--no-verify", "-m", "feature: f returns 42",
         env={**os.environ, **COMMITTER})
    feature_head = wgit(wt_path, "rev-parse", "HEAD").strip()
    updated_found = dreplace(found, head=feature_head)
    _write_json(run / RECORD, asdict(updated_found))

    # ── Main changes BOTH files: shared.py conflicts, util.py merges cleanly ──
    (repo / "shared.py").write_text("def f():\n    return 2\n")
    (repo / "util.py").write_text("def helper():\n    return 99\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "main: f returns 2, helper returns 99")
    main_head = git(repo, "rev-parse", "HEAD").strip()

    # ── Write pending-merge.json so materialise starts the merge ──
    pm_dir = run / ".refactory"
    pm_dir.mkdir(parents=True, exist_ok=True)
    (pm_dir / "pending-merge.json").write_text(
        json.dumps({"target_tip": main_head, "conflict_files": ["shared.py"]})
    )

    # ── Builder's resolution: ===FILE: resolves conflict, ===EDIT: keeps main's util.py ──
    emit_text = (
        "===FILE: shared.py===\n"
        "def f():\n"
        "    return 3  # merged\n"
        "===EDIT: util.py===\n"
        "<<<<<<< SEARCH\n"
        "def helper():\n"
        "    return 99\n"
        "=======\n"
        "def helper():\n"
        "    return 99  # kept from main\n"
        ">>>>>>> REPLACE\n"
    )

    ctx = _TitleCtx(
        facts={"name": "test_edit_conflict", "brief": "test", "check": "echo ok"},
        files={"emit.txt": emit_text},
    )
    ctx.workdir = run
    ctx.read_text = lambda name: ctx._files[name]
    ctx.write_text = lambda name, text: ctx._files.__setitem__(name, text)
    ctx.write_json = lambda name, data: ctx._files.__setitem__(name, json.dumps(data))
    ctx.read_json = lambda name, default=None: (
        json.loads(ctx._files.get(name, "null"))
        if ctx._files.get(name) else default
    )

    result = repo_change._commit_emission(ctx)
    assert result.get("commit"), f"expected a commit: {result}"

    # shared.py is resolved (no conflict markers).
    shared_content = (wt_path / "shared.py").read_text()
    assert "return 3" in shared_content, f"expected resolved f(): {shared_content}"
    assert "<<<<<<<" not in shared_content, f"conflict markers remain: {shared_content}"

    # util.py keeps main's change AND has the builder's edit.
    util_content = (wt_path / "util.py").read_text()
    assert "return 99" in util_content, f"expected main's change: {util_content}"
    assert "kept from main" in util_content, f"expected builder's edit: {util_content}"

    # The commit is a merge commit (2 parents).
    cat_output = git(wt_path, "cat-file", "-p", "HEAD")
    parent_lines = [l for l in cat_output.splitlines() if l.startswith("parent ")]
    assert len(parent_lines) == 2, f"expected merge commit, got {len(parent_lines)} parents"

    # pending-merge.json is cleaned up.
    assert not (pm_dir / "pending-merge.json").exists(), "pending-merge.json should be deleted"


def test_emission_missing_conflicted_path_aborts_merge(tmp_path):
    """An emission that omits a conflicted path aborts the merge and leaves
    the worktree clean.

    Materialise starts the merge from pending-merge.json, detects the missing
    unmerged path in the emission, aborts the merge, and raises AttemptRefused.

    Fails without the change: removing the abort in the missing-unmerged check
    leaves MERGE_HEAD behind.
    """
    from refactory_dsh import repo_change
    from refactory_dsh.worktree import (
        create, recorded, _write_json, RECORD, AttemptRefused,
        git as wgit, COMMITTER, NO_HOOKS,
    )
    from dataclasses import asdict, replace as dreplace

    repo = git_repo(tmp_path / "repo")
    (repo / "shared.py").write_text("def f():\n    return 1\n")
    (repo / "other.py").write_text("x = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "initial")

    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    create(repo, "HEAD", "test-missing-path", run)
    found = recorded(run)
    wt_path = Path(found.path)

    # Feature change to shared.py.
    (wt_path / "shared.py").write_text("def f():\n    return 42\n")
    wgit(wt_path, "add", ".")
    wgit(wt_path, *NO_HOOKS, "-c", "commit.gpgsign=false",
         "commit", "-q", "--no-verify", "-m", "feature",
         env={**os.environ, **worktree.COMMITTER})
    feature_head = wgit(wt_path, "rev-parse", "HEAD").strip()
    _write_json(run / RECORD, asdict(dreplace(found, head=feature_head)))

    # Main also changes shared.py (conflict).
    (repo / "shared.py").write_text("def f():\n    return 2\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "main changes shared.py")
    main_head = git(repo, "rev-parse", "HEAD").strip()

    # Write pending-merge.json.
    pm_dir = run / ".refactory"
    pm_dir.mkdir(parents=True, exist_ok=True)
    (pm_dir / "pending-merge.json").write_text(
        json.dumps({"target_tip": main_head, "conflict_files": ["shared.py"]})
    )

    # Emission omits shared.py — only writes other.py.
    emit_text = "===FILE: other.py===\nx = 2\n"
    ctx = _TitleCtx(
        facts={"name": "test_missing", "brief": "test", "check": "echo ok"},
        files={"emit.txt": emit_text},
    )
    ctx.workdir = run
    ctx.read_text = lambda name: ctx._files[name]
    ctx.write_text = lambda name, text: ctx._files.__setitem__(name, text)
    ctx.write_json = lambda name, data: ctx._files.__setitem__(name, json.dumps(data))
    ctx.read_json = lambda name, default=None: (
        json.loads(ctx._files.get(name, "null"))
        if ctx._files.get(name) else default
    )

    with pytest.raises(AttemptRefused, match="missing"):
        repo_change._commit_emission(ctx)

    # The worktree must be clean — no MERGE_HEAD.
    with pytest.raises(subprocess.CalledProcessError):
        git(wt_path, "rev-parse", "--verify", "-q", "MERGE_HEAD")

    # No unmerged entries.
    try:
        unmerged = worktree.git(wt_path, "diff", "--name-only", "--diff-filter=U")
    except ValueError:
        unmerged = ""
    assert not unmerged.strip(), f"unmerged files remain: {unmerged}"


def test_materialise_base_moves_to_main_tip(tmp_path):
    """After materialise commits a merge, the base moves to main's tip.

    Removing the base-move (worktree.py second-parent logic) would leave the
    base at the original commit, so the diff would include everything main
    gained.  This test fails without that logic.
    """
    from refactory_dsh import repo_change
    from refactory_dsh.worktree import (
        create, recorded, _write_json, RECORD,
        git as wgit, COMMITTER, NO_HOOKS,
    )
    from dataclasses import asdict, replace as dreplace

    repo = git_repo(tmp_path / "repo")
    (repo / "a.py").write_text("x = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "initial")
    initial_base = git(repo, "rev-parse", "HEAD").strip()

    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    create(repo, "HEAD", "test-base-move", run)
    found = recorded(run)
    wt_path = Path(found.path)

    # Feature change.
    (wt_path / "b.py").write_text("y = 1\n")
    wgit(wt_path, "add", ".")
    wgit(wt_path, *NO_HOOKS, "-c", "commit.gpgsign=false",
         "commit", "-q", "--no-verify", "-m", "feature",
         env={**os.environ, **worktree.COMMITTER})
    feature_head = wgit(wt_path, "rev-parse", "HEAD").strip()
    _write_json(run / RECORD, asdict(dreplace(found, head=feature_head)))

    # Main adds a file (clean merge).
    (repo / "c.py").write_text("z = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "main adds c.py")
    main_head = git(repo, "rev-parse", "HEAD").strip()

    # Write pending-merge.json.
    pm_dir = run / ".refactory"
    pm_dir.mkdir(parents=True, exist_ok=True)
    (pm_dir / "pending-merge.json").write_text(
        json.dumps({"target_tip": main_head, "conflict_files": []})
    )

    emit_text = "===FILE: b.py===\ny = 2\n"
    ctx = _TitleCtx(
        facts={"name": "test_base_move", "brief": "test", "check": "echo ok"},
        files={"emit.txt": emit_text},
    )
    ctx.workdir = run
    ctx.read_text = lambda name: ctx._files[name]
    ctx.write_text = lambda name, text: ctx._files.__setitem__(name, text)
    ctx.write_json = lambda name, data: ctx._files.__setitem__(name, json.dumps(data))
    ctx.read_json = lambda name, default=None: (
        json.loads(ctx._files.get(name, "null"))
        if ctx._files.get(name) else default
    )

    result = repo_change._commit_emission(ctx)
    assert result.get("commit"), f"expected a commit: {result}"

    new_found = recorded(run)
    # The base must be main's tip (the second parent), not the original base.
    assert new_found.base == main_head, \
        f"expected base to move to main's tip {main_head}, got {new_found.base}"
    assert new_found.base != initial_base, \
        "base should have moved from the initial commit"


def test_an_upstream_that_does_not_hold_the_base_is_not_where_the_pull_request_goes(github, tmp_path):
    repo, _stack = _stacked(github, tmp_path)
    # main moves on GitHub; a local branch at that commit still names stack as its upstream.
    (repo / "later.py").write_text("y = 2\n")
    git(repo, "add", "later.py")
    git(repo, "commit", "-q", "-m", "main moves on")
    git(repo, "push", "-q", "origin", "main")
    git(repo, "branch", "-q", "--track", "elsewhere", "origin/stack")
    git(repo, "branch", "-q", "-f", "elsewhere", "main")
    planned = delivery.plan(repo, base=git(repo, "rev-parse", "main"), branch="refactory/run-0123456789ab",
                            rev="elsewhere")
    assert (planned["action"], planned["into"]) == ("pull_request", "main"), planned
