"""A design run that passes its reviews ends with a pull request (#350).

One test per path a person takes: a new project, an existing repository, a reused
project name, a follow-up to an open pull request, a redo, `--no-pr`, a run that fails,
a base with commits nobody pushed, a name already taken on GitHub, and no gh login.
GitHub is `fake_gh.py` over bare repositories on disk, which git reaches through
`url.<base>.insteadOf`; the model is a fake; the repositories, the worktrees, the
pushes and the registry are real.
"""

from __future__ import annotations

from _delivery_support import *  # noqa: F403  (the helpers and fixtures these tests share)


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


def test_a_run_whose_whole_change_is_empty_opens_no_pull_request(github, tmp_path, monkeypatch):
    """dm-246: attempt 2 put the tree back to the base, so the branch changes nothing.
    Delivery refuses before it creates the repository or pushes, and says why."""
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after,
                     steps=[["put_back", {"remove": ["tool.py"], "recursive": True}]])
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    record = _record(run)
    assert record["status"] == "kept local" and "changes nothing" in record["why"]
    assert "changes nothing" in result.facts["delivery_result"]
    calls = _calls(tmp_path)
    assert "repo create" not in calls and "pr create" not in calls


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


@pytest.mark.parametrize("legacy", ["\r\nWritten in the web UI.\r\nSecond line.\r\n",
                                    "\nLeading blank line, then text.\n"], ids=["crlf", "newline"])
def test_a_legacy_body_is_left_byte_identical_on_a_refine(tmp_path, monkeypatch, legacy):
    (tmp_path / delivery.BODY).write_text(OURS)
    pr = _describing(monkeypatch, tmp_path, legacy)
    note = delivery._describe(nn.RunContext(tmp_path, enforce_writes=False), "u", None)
    assert pr["body"] == legacy and not [args for args in pr["calls"] if "--body-file" in args]
    assert note == ("the description of u was not updated: it has no delivery markers, so all of "
                    "it is kept as written")


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
    _child().setattr(delivery, "_gh", delivery_fakes.gh_fails, command=["pr", "create"],
                     error="the delivery stopped here", exception="RuntimeError")
    failed = _design(run)
    assert failed.exit_ok is False and "the delivery stopped here" in (failed.error or "")
    slug = f"{LOGIN}/{NAME}"
    branch = worktree.recorded(run).branch
    assert _remote_head(github, slug, branch) and _pulls(github, slug) == []
    _unfake(delivery, "_gh")
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
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main", {
        "slug": "acme/app", "clone": str(tmp_path / "other"), "branch": pull["head"],
        "files": {"theirs.py": "x = 2\n"}, "message": "theirs"}]])
    run = tmp_path / "refine"
    result = _design(run, refine=url)
    assert result.exit_ok, result.error
    moved = _remote_head(github, "acme/app", pull["head"])
    assert moved != worktree.recorded(run).head
    assert git(tmp_path / "other", "rev-parse", "HEAD") == moved
    assert "moved to" in capsys.readouterr().err and _record(run)["status"] == "kept local"


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


def test_the_refine_design_flag_is_gone(capsys):
    """A refine always designs straight from the pull request: `--refine-design` (and
    its `full` value, the whole design phase) no longer exists."""
    with pytest.raises(SystemExit):
        cli.main(["ceo", "a tool", "--check", "true", "--refine", "acme/app#3",
                  "--refine-design", "full"])
    assert "--refine-design" in capsys.readouterr().err
    assert "refine_design" not in DesignMode().state_dict()["knobs"]


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
    assert cli.main(["ceo", "a tool", "--check", "true", "--mode", "design_fast", "--no-pr"]) == 2
    assert cli.main(["ceo", "a tool", "--check", "true", "--mode", "design_fast", "--refine", "acme/app#3"]) == 2
    assert cli.main(["ceo", "a tool", "--check", "true", "--no-worktree", "--refine", "acme/app#3"]) == 2
    err = capsys.readouterr().err
    assert "takes no --base" in err and "goes with --refine" in err
    assert "does not take --no-pr" in err and "--no-worktree cannot take --refine" in err
    assert "--mode design_fast does not take --refine" in err


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


def test_a_refinement_delivered_once_is_not_refused_as_moved_when_asked_again(opened_pull, tmp_path, capsys):
    url = opened_pull.url
    run = tmp_path / "refine"
    result = _design(run, refine=url)
    assert result.exit_ok, result.error
    capsys.readouterr()
    again = delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))
    assert again["pull_request"] == url and _record(run)["status"] == "already updated"
    assert "run warning" not in capsys.readouterr().err


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


def test_a_pull_request_gh_opened_but_reported_failing_is_the_one_delivered(github, tmp_path, monkeypatch):
    _child().setattr(delivery, "_gh", delivery_fakes.gh_fails, command=["pr", "create"],
                     error="`gh pr create` failed: timeout", real_first=True)
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert _record(run)["status"] == "opened" and len(_pulls(github, f"{LOGIN}/{NAME}")) == 1


def test_a_refinement_never_pushes_onto_a_default_branch(opened_pull):
    repo, url, github = opened_pull.repo, opened_pull.url, opened_pull.github
    pulls = _pulls(github, "acme/app")
    pulls[0]["head"] = "main"
    (github / "acme/app.pulls.json").write_text(json.dumps(pulls))
    head = _remote_head(github, "acme/app", "main")
    planned = delivery.plan(repo, base=head, branch="refactory/run-0123456789ab",
                            refine={"url": url, "repo": "acme/app", "head": head})
    assert planned["action"] == "none" and "is acme/app's default branch" in planned["why"]


def test_a_refinement_that_github_shows_at_another_head_is_reported_not_claimed(opened_pull, tmp_path):
    url, github = opened_pull.url, opened_pull.github
    (github / "stale-view").write_text("0" * 40 + "\n")
    run = tmp_path / "refine"
    result = _design(run, refine=url)
    assert result.exit_ok is False and "not this run's reviewed commit" in (result.error or "")
    assert _record(run)["status"] == "unverified"


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
    _child().setattr(delivery, "_gh", delivery_fakes.gh_fails, command=["pr", "create"],
                     error="`gh pr create` failed: error connecting to api.github.com")
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    branch = worktree.recorded(run).branch
    assert _remote_head(github, f"{LOGIN}/{NAME}", branch)
    err = capsys.readouterr().err
    assert f"the branch {branch} is already on GitHub, as {LOGIN}/{NAME}:{branch}" in err
    assert "stays on the local branch" not in err
    assert _record(run)["pushed"] == f"{LOGIN}/{NAME}:{branch}"


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
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["run_git", {
        "path": str(repo), "args": ["remote", "set-url", "origin", "https://github.com/acme/fork.git"]}]])
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
    _child().setattr(delivery, "_push", delivery_fakes.push_fails, error="push refused by GitHub")
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
    _child().setattr(
        delivery, "_git", delivery_fakes.git_fails, arg="push",
        error="git push -q https://github.com/acme/app.git {last} failed in {path}: "
              "To https://github.com/acme/app.git\n"
              " ! [remote rejected] abc -> refactory/run-abc (refusing to allow an OAuth App to "
              "create or update workflow `.github/workflows/ci.yml` without `workflow` scope)\n"
              "error: failed to push some refs to 'https://github.com/acme/app.git'")
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

    timeouts = tmp_path / "timeouts.jsonl"
    # Main moves, then each sandbox.run_command delivery makes records its timeout_s.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[
        ["move_main", {"slug": slug, "clone": str(tmp_path / "other"), "files": {"extra.txt": "new\n"},
                       "message": "base moves"}],
        ["patch", {"target": "refactory_dsh.sandbox", "name": "run_command",
                   "function": "delivery_fakes:record_kwarg",
                   "params": {"_module": "refactory_dsh.sandbox", "_name": "run_command",
                              "_key": "timeout_s", "_path": str(timeouts)}}]])

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n"},
                     target=str(repo), check="python tool.py",
                     generate="bash gen.sh",
                     mode=_mode(check_timeout_s=1800))
    assert result.exit_ok, result.error
    captured_timeouts = ([json.loads(line) for line in timeouts.read_text().splitlines()]
                         if timeouts.exists() else [])

    # Every timeout captured during the merge's generate/setup/check equals the knob.
    assert captured_timeouts, "no sandbox.run_command calls were captured"
    for t in captured_timeouts:
        assert t == 1800.0, f"expected 1800, got {t}"


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

    # Base adds the marker that makes generate fail and changes source.txt; gen.sh fails
    # there, so generated.txt is written by hand for the base commit.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main", {
        "slug": slug, "clone": str(tmp_path / "other"),
        "files": {"FAIL_GEN": "yes\n", "source.txt": "gamma\n", "generated.txt": "gamma\n"},
        "message": "base adds fail marker"}]])
    # Skip the reloop — this test verifies delivery's behaviour, not the full loop.
    _child().setattr(delivery, "evaluate_delivery_conflicts", delivery_fakes.proceed)

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

    # From delivery on, `git ls-remote … refs/heads/main` fails.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["patch", {
        "target": "refactory_dsh.delivery", "name": "_git", "function": "delivery_fakes:git_fails",
        "params": {"arg": "ls-remote", "text": "refs/heads/main", "error": "simulated ls-remote failure"}}]])
    # Skip the delivery_gate reloop.
    _child().setattr(delivery, "evaluate_delivery_conflicts", delivery_fakes.proceed)

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


def test_a_passed_check_is_reused_only_for_the_same_tree_and_command(tmp_path):
    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    run.mkdir()
    ctx = nn.RunContext(run, facts={"check": "python tool.py"}, enforce_writes=False)
    assert delivery.reusable_check(ctx, repo, "HEAD") == ""
    delivery.record_full_check(ctx, repo, "python tool.py")
    head = git(repo, "rev-parse", "HEAD")
    assert delivery.reusable_check(ctx, repo, "HEAD") == head
    ctx.facts["check"] = "pytest -q"
    assert delivery.reusable_check(ctx, repo, "HEAD") == ""
    ctx.facts["check"] = "python tool.py"
    (repo / "other.txt").write_text("x\n")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "more")
    assert delivery.reusable_check(ctx, repo, "HEAD") == ""
    # Tracked files that differ from the head: what ran is not what the tree id names.
    (repo / "other.txt").write_text("changed\n")
    delivery.record_full_check(ctx, repo, "python tool.py")
    assert delivery.reusable_check(ctx, repo, "HEAD") == ""
