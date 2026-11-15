"""Delivery: runs whose work stays local (no pull request, no gh login)."""

from __future__ import annotations

from _delivery_support import *  # noqa: F403  (the helpers and fixtures these tests share)


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


def test_a_draft_whose_push_is_refused_keeps_the_work_local_as_without_the_fact(github, tmp_path, monkeypatch):
    """A refused push stops a draft as it stops any delivery: no pull request, kept
    local, said and recorded as without the fact; its text and finishing commands
    still open a draft."""
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)

    _child().setattr(delivery, "_push", delivery_fakes.push_fails, error="push refused by GitHub")
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


def test_a_plan_that_changed_since_the_gate_keeps_the_work_local(github, tmp_path, capsys, monkeypatch):
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    # A fork holding the same main, so the run's base is on it and only the repository differs.
    fork = github / "acme/fork.git"
    git(fork.parent, "init", "-q", "--bare", "-b", "main", fork.name)
    (github / "acme/fork.json").write_text(json.dumps({"default_branch": "main"}))
    git(repo, "push", "-q", "https://github.com/acme/fork.git", "main:refs/heads/main")
    # Between the approval and delivery, the checkout started pointing elsewhere.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["run_git", {
        "path": str(repo), "args": ["remote", "set-url", "origin", "https://github.com/acme/fork.git"]}]])
    result = _design(tmp_path / "run", target=str(repo))
    assert result.exit_ok, result.error
    assert _pulls(github, "acme/app") == [] and _pulls(github, "acme/fork") == []
    err = capsys.readouterr().err
    assert "different from what the approval gate showed" in err and "acme/fork:main" in err


def test_without_a_network_the_run_succeeds_and_says_its_work_stays_local(github, tmp_path, capsys):
    (github / "offline").write_text("")
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert "error connecting to api.github.com" in capsys.readouterr().err
    assert _record(run)["status"] == "kept local" and not list(github.glob("*/*.git"))


def test_refactory_projects_says_when_a_run_kept_its_work_local(github, tmp_path, capsys):
    (github / "login").unlink()
    assert _design(tmp_path / "run").exit_ok
    capsys.readouterr()
    assert projects.main([]) == 0
    assert "kept local" in capsys.readouterr().out


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


def test_gh_is_asked_for_its_login_without_gh_host(github, tmp_path):
    """dm-260: delivery runs gh with GH_HOST set, and with it a logged-out gh 2.99's `auth
    status` reports a phantom token rather than "not logged in". So the login is asked
    without GH_HOST, and a logged-out gh is named without anything being sent to GitHub."""
    (github / "login").unlink()
    planned = delivery.plan(tmp_path / "new", base="", branch="refactory/run-x")
    assert planned["why"] == delivery.LOGGED_OUT and planned.get("logged_out") is True
    assert "auth status" in _calls(tmp_path) and "api user" not in _calls(tmp_path)


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
