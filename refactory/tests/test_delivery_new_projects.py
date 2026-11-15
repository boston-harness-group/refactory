"""Delivery: new projects and the repositories they create, adopt or publish to."""

from __future__ import annotations

from _delivery_support import *  # noqa: F403  (the helpers and fixtures these tests share)


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


def test_a_reused_project_name_opens_its_pull_request_in_the_repository_it_was_given(github, tmp_path):
    first = _design(tmp_path / "one", {"tool.py": "print(1)\n"})
    second = _design(tmp_path / "two", {"tool.py": "print(2)\n"})
    assert first.exit_ok and second.exit_ok, (first.error, second.error)
    slug = f"{LOGIN}/{NAME}"
    assert _calls(tmp_path).count("repo create") == 1
    assert [pull["number"] for pull in _pulls(github, slug)] == [1, 2]
    assert second.facts["pull_request"] == f"https://github.com/{slug}/pull/2"
    assert "opens a pull request from" in (tmp_path / "two" / delivery.PLAN).read_text()


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


def test_a_creation_that_died_before_its_first_push_goes_on_in_the_repository_it_made(github, tmp_path, monkeypatch):
    run = tmp_path / "run"
    _child().setattr(delivery, "_push", delivery_fakes.push_fails, error="the delivery stopped here",
                     exception="RuntimeError", only_branch="main")
    assert _design(run).exit_ok is False
    slug = f"{LOGIN}/{NAME}"
    assert (github / f"{slug}.git").is_dir() and not _remote_head(github, slug, "main")
    _unfake(delivery, "_push")
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


def test_refine_refuses_a_pull_request_from_another_repositorys_branch(opened_pull, tmp_path):
    url, github = opened_pull.url, opened_pull.github
    pulls = _pulls(github, "acme/app")
    pulls[0]["cross"] = True
    (github / "acme/app.pulls.json").write_text(json.dumps(pulls))
    refused = _design(tmp_path / "fork", refine=url)
    assert refused.exit_ok is False and "another repository's branch" in (refused.error or "")


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


def test_a_repository_gh_made_but_reported_failing_is_adopted_only_when_new_and_empty(github, tmp_path,
                                                                                     monkeypatch):
    _child().setattr(delivery, "_gh", delivery_fakes.gh_fails, command=["repo", "create"],
                     error="`gh repo create` failed: timeout", real_first=True)
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert _record(run)["status"] == "opened"
    [entry] = projects.load()["projects"]
    assert entry["github"] == f"{LOGIN}/{NAME}"


def test_a_new_repository_that_is_slow_to_take_pushes_is_waited_for(github, tmp_path, monkeypatch):
    _child().setattr(delivery, "_heads", delivery_fakes.heads_lag, counter=str(tmp_path / "heads-calls"),
                     error="git ls-remote failed: repository not found")
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert _record(run)["status"] == "opened"


def test_a_new_repository_goes_where_repo_owner_and_repo_visibility_say(github, tmp_path):
    result = _design(tmp_path / "run", mode=_mode(repo_owner="acme-org", repo_visibility="internal"))
    assert result.exit_ok, result.error
    assert f"repo create acme-org/{NAME} --internal" in _calls(tmp_path)
    assert json.loads((github / "acme-org" / f"{NAME}.json").read_text())["visibility"] == "internal"
    assert result.facts["pull_request"] == f"https://github.com/acme-org/{NAME}/pull/1"


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
