"""Delivery: drafts, limits, refusal bounds and the overwatch."""

from __future__ import annotations

from _delivery_support import *  # noqa: F403  (the helpers and fixtures these tests share)


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


def test_a_review_limit_at_gate_qa_with_a_passing_attempt_delivers_a_draft(github, tmp_path):
    """b43b2 stopped at its review limit with a passing attempt and delivered nothing. Now
    the QA judge's limit goes on through the correction audit and the overwatch to a draft
    pull request listing what the judge left unresolved, with the attempts that named it."""
    run = tmp_path / "run"
    redo = qa_judgment("redo", must_fix=[QA_HELD], follow_ups=["document the --json flag"])
    result, model = _limited(run, _attempt(BUILT, judge=redo),
                             _attempt(render({"tool.py": "print('built!')\n"}), judge=redo,
                                      overwatch=overwatch_judgment("pass")),
                             review_attempts=2, fix_rounds=10)
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
                             check_attempts=2, review_attempts=3, check_attempt_cap=0, fix_rounds=10)
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
    # The old body sections are preserved; Open items may appear inside the delivery markers.
    old_sections = old_body.replace(delivery.BODY_END + "\n", "").rstrip()
    assert old_sections in updated_pull["body"], "old body sections should be preserved"
    assert updated_pull["body"].endswith(delivery.BODY_END + "\n")
    assert "<!-- refactory:open-items -->" in updated_pull["body"]
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

    advance_main = [["move_main", {"slug": slug, "clone": str(tmp_path / "other_sec"),
                                   "files": {"config.txt": "broken\n"}, "message": "brk"}]]

    fact = {"reason": "limit reached\nANTHROPIC_API_KEY=sk-ant-in-the-reason",
            "unresolved": [{"text": "the reviewer saw OPENAI_API_KEY=sk-live-reviewer-quoted\n## Injected",
                            "source": "review", "rounds": [1]}]}
    tool = ("import sys\n"
            "if open('config.txt').read().strip() == 'broken':\n"
            "    print('GITHUB_TOKEN=ghp_SECRETVALUE123 failed', file=sys.stderr)\n"
            "    print('second line', file=sys.stderr)\n"
            "    sys.exit(1)\n")
    result = _design_draft(tmp_path / "run_sec", github, tmp_path, target=repo, emit={"tool.py": tool},
                           check="python tool.py", before=advance_main, draft_fact=fact)
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
    first_sections = first["body"].replace(delivery.BODY_END + "\n", "").rstrip()
    assert first_sections in pull["body"], "first body sections should be preserved"
    assert pull["body"].endswith(delivery.BODY_END + "\n")


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


def test_review_comments_without_refine_are_refused_not_dropped(github, tmp_path):
    refused = _design(tmp_path / "run", mode=_mode(review_comments=True))
    assert refused.exit_ok is False and "review_comments goes with refine" in (refused.error or "")


def test_a_branch_that_moved_since_its_reviews_is_never_delivered(new_run):
    run, result = new_run.run, new_run.result
    unreviewed = {**result.facts, "commit": "0" * 40}
    with pytest.raises(ValueError, match="not at the attempt its reviews passed"):
        delivery.deliver_design(nn.RunContext(run, facts=unreviewed, enforce_writes=False))


def test_a_pull_request_already_open_is_checked_again_when_asked_again(new_run):
    run, result = new_run.run, new_run.result
    (new_run.github / "stale-view").write_text("0" * 40 + "\n")
    with pytest.raises(delivery.Unverified):
        delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))
