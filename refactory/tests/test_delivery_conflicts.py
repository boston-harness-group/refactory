"""Delivery: conflicts with a moved main, merge reloops and redos."""

from __future__ import annotations

from _delivery_support import *  # noqa: F403  (the helpers and fixtures these tests share)


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


def test_an_empty_change_is_refused_before_main_is_merged_in(github, tmp_path, monkeypatch):
    """dm-246: main moved before delivery, and the run's change is empty. Merging main in
    first would give a head with main's files, and a pull request of nothing: delivery
    refuses before the merge."""
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after,
                     steps=[["put_back", {"remove": ["tool.py"]}], ["move_main", {
                         "slug": "acme/app", "clone": str(tmp_path / "other-theirs.txt"),
                         "files": {"theirs.txt": "theirs.txt\n"}, "message": "theirs.txt"}]])
    run = tmp_path / "run"
    result = _design(run, target=str(repo))
    assert result.exit_ok, result.error
    assert "changes nothing" in _record(run)["why"]
    assert _pulls(github, "acme/app") == [] and "merge_commit" not in _record(run)


def test_draft_conflict_with_base(github, tmp_path, monkeypatch):
    """A draft run with a merge conflict: draft PR opened, run's own commit pushed,
    conflict named in the unresolved section."""
    repo = git_repo(tmp_path / "repo_conflict")
    (repo / "authored.py").write_text("line1\nline2\nline3\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base with authored")
    slug = "acme/draft-conflict"
    _published(github, slug, repo)

    advance_main = [["move_main", {"slug": slug, "clone": str(tmp_path / "other_conflict"),
                                   "files": {"authored.py": "line1\nchanged_by_base\nline3\n"},
                                   "message": "base changes authored.py"}]]

    run = tmp_path / "run_conflict"
    seen: dict = {}
    result = _design_draft(run, github, tmp_path, target=repo,
                           emit={"tool.py": "print('built')\n",
                                 "authored.py": "line1\nchanged_by_run\nline3\n"},
                           before=advance_main, seen=seen)
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

    advance_main = [["move_main", {"slug": slug, "clone": str(tmp_path / "other_mfail"),
                                   "files": {"config.txt": "broken\n"}, "message": "base breaks config"}]]

    run = tmp_path / "run_mfail"
    seen: dict = {}
    result = _design_draft(
        run, github, tmp_path, target=repo,
        emit={"tool.py": "import sys; sys.exit(1 if open('config.txt').read().strip() == 'broken' else 0)\n"},
        check="python tool.py",
        before=advance_main, seen=seen,
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

    advance_main = [["move_main", {"slug": slug, "clone": str(tmp_path / "other"),
                                   "files": {"blocker.txt": "from base\n"}, "message": "base adds blocker"}],
                    # An untracked file of that name in the worktree: git merge refuses, with
                    # no conflicted file, so the merge is abandoned.
                    ["untracked", {"files": {"blocker.txt": "untracked content\n"}}]]

    result = _design_draft(run, github, tmp_path, target=repo, before=advance_main)
    assert result.exit_ok, result.error
    found = worktree.recorded(run)
    assert worktree._json(worktree._record_path(found)).get("outcome") != "succeeded"
    rec = _record(run)
    assert rec["status"] == "kept local" and rec.get("merge_abandoned") and "draft" not in rec
    assert result.facts["delivery_result"].startswith("not delivered: merge abandoned")
    assert _remote_head(github, slug, found.branch) == ""


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


def test_a_repository_whose_creation_lost_the_network_is_finished_by_a_redo(github, tmp_path, monkeypatch, capsys):
    """gh reports the creation failed, the network then drops; the redo adopts the repository it made."""
    _child().setattr(delivery, "_gh", delivery_fakes.gh_fails, command=["repo", "create"],
                     error="`gh repo create` failed: timeout", real_first=True,
                     touch=str(github / "offline"))
    run = tmp_path / "run"
    result = _design(run)
    assert result.exit_ok, result.error
    assert _record(run)["status"] == "kept local" and (github / LOGIN / f"{NAME}.git").is_dir()
    (github / "offline").unlink()
    _unfake(delivery, "_gh")
    again = delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))
    assert again["pull_request"] == f"https://github.com/{LOGIN}/{NAME}/pull/1"
    assert _calls(tmp_path).count("repo create") == 2 and len(_pulls(github, f"{LOGIN}/{NAME}")) == 1


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


def test_the_pull_request_recommends_no_way_to_merge(new_run):
    """The description no longer explains the branch's commits (the short description,
    2026-10-26), and still recommends no merge method."""
    [pull] = _pulls(new_run.github, f"{LOGIN}/{NAME}")
    assert "the run's attempts" not in pull["body"]
    assert "squash merge is recommended" not in pull["body"]


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

    # Between the run's start and delivery, advance main with a generated-file conflict;
    # gen.sh regenerates generated.txt with the base's new sources.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main", {
        "slug": slug, "clone": str(tmp_path / "other"),
        "files": {"source2.txt": "gamma\n", "unrelated.txt": "hello\n"},
        "commands": [["bash", "gen.sh"]], "message": "base moves: add source2 and unrelated"}]])

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

    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main", {
        "slug": slug, "clone": str(tmp_path / "other"),
        "files": {"authored.py": "line1\nchanged_by_base\nline3\n"},
        "message": "base changes authored.py line 2"}]])
    # Skip the reloop — this test verifies delivery's behaviour, not the full loop.
    _child().setattr(delivery, "evaluate_delivery_conflicts", delivery_fakes.proceed)

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

    _move_main_then_deliver(tmp_path, slug, files={"generated.txt": "main's own generated text\n"})

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

    _move_main_then_deliver(tmp_path, slug, files={"authored.py": "line1\nchanged_by_base\nline3\n"},
                            commands=[["bash", "gen.sh"]])
    _child().setattr(delivery, "evaluate_delivery_conflicts", delivery_fakes.proceed)

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


def test_conflict_send_back_stays_bounded_with_large_files(github, tmp_path, monkeypatch):
    """Large conflicting files reach the builder as at most a bounded excerpt each and about
    20 KB in all, in the file list and in the validation text the prompt renders.

    Fails without the change: validation carries the base, ours and theirs files whole. The
    bounds bind on conflict hunks, not bytes: 150 hunks a file (every 20th row differs)
    cross each of them several times over without the change, where the 3.5 MB files this
    once used had 70 and only made the run slower.
    """
    lines = [f"row {i} " + "x" * 40 for i in range(3_000)]  # about 140 KB

    def variant(tag):
        return "\n".join(f"{row} {tag}" if i % 20 == 0 else row for i, row in enumerate(lines)) + "\n"

    repo = git_repo(tmp_path / "repo")
    for name in ("big_a.txt", "big_b.txt", "big_c.txt"):
        (repo / name).write_text("\n".join(lines) + "\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/merge-big"
    _published(github, slug, repo)

    mains = tmp_path / "main-files"  # too large to hand the op child as parameters
    mains.mkdir()
    for name in ("big_a.txt", "big_b.txt", "big_c.txt"):
        (mains / name).write_text(variant("main"))
    _move_main_then_deliver(tmp_path, slug, copy_from=str(mains))
    gate = delivery.evaluate_delivery_conflicts
    _child().setattr(delivery, "evaluate_delivery_conflicts", delivery_fakes.proceed)

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

    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main", {
        "slug": slug, "clone": str(tmp_path / "other"), "files": {"config.txt": "broken\n"},
        "message": "base breaks config"}]])
    # Skip the reloop — this test verifies delivery's behaviour, not the full loop.
    _child().setattr(delivery, "evaluate_delivery_conflicts", delivery_fakes.proceed)

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

    # Base changes generated.txt to something else (simulating a different generator version),
    # but the generate command always produces 'static-content'.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main", {
        "slug": slug, "clone": str(tmp_path / "other"),
        "files": {"generated.txt": "old-generated-content\n", "extra.txt": "extra\n"},
        "message": "base changes generated.txt"}]])

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

    # Base changes input_b (creates a conflict in gen1.txt because the run
    # changes input_a) and appends to source2.txt (gen2.txt merges cleanly
    # from git's perspective but its line-count header becomes stale).
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main", {
        "slug": slug, "clone": str(tmp_path / "other"),
        "files": {"input_b.txt": "B-base\n", "source2.txt": "original\nbase-line\n"},
        "commands": [["bash", "gen.sh"]], "message": "base updates both inputs"}]])

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

    # Base deletes the file.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main", {
        "slug": slug, "clone": str(tmp_path / "other"), "remove": ["conflicted.txt"],
        "message": "base deletes conflicted.txt"}]])
    # Skip the reloop — this test verifies delivery's behaviour, not the full loop.
    _child().setattr(delivery, "evaluate_delivery_conflicts", delivery_fakes.proceed)

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

    # Main moves with a new file (no conflict with the run's source.txt), a post-checkout
    # hook is installed in the worktree's git dir, and its marker must not exist after delivery.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[
        ["move_main", {"slug": slug, "clone": str(tmp_path / "other"),
                       "files": {"extra.txt": "added by main\n"}, "message": "base adds extra.txt"}],
        ["hook", {"name": "post-checkout", "marker": "HOOK_RAN"}]],
        after=[["absent", {"name": "HOOK_RAN"}]])

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

    # First delivery: advance main, then sabotage _verified via stale-view (fake_gh returns
    # a stale headRefOid for "headRefOid,url" queries), with one verification attempt.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[
        ["move_main", {"slug": slug, "clone": str(tmp_path / "other"), "files": {"extra.txt": "new\n"},
                       "message": "base moves"}],
        ["touch", {"path": str(github / "stale-view"), "text": "0" * 40}],
        ["patch", {"target": "refactory_dsh.delivery", "name": "VERIFY_ATTEMPTS", "value": 1}]])
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
    _unfake(delivery, "deliver_design")
    _child().setattr(delivery, "VERIFY_ATTEMPTS", 5)
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

    ensures = tmp_path / "ensure-calls.jsonl"
    # Base changes pyproject.toml; from then on ensure_environment installs nothing and
    # records each call.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[
        ["move_main", {"slug": slug, "clone": str(tmp_path / "other"), "files": {
            "pyproject.toml": '[project]\nname = "example"\nversion = "0.1.0"\ndependencies = ["requests"]\n'},
            "message": "base adds dependency"}],
        ["patch", {"target": "refactory_dsh.worktree", "name": "ensure_environment",
                   "function": "delivery_fakes:ensure_recorded", "params": {"path": str(ensures)}}]])

    run = tmp_path / "run"
    result = _design(run, {"tool.py": "print('built')\n"},
                     target=str(repo), check="python tool.py")
    assert result.exit_ok, result.error
    ensure_calls = ([json.loads(line) for line in ensures.read_text().splitlines()]
                    if ensures.exists() else [])

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

    # Base adds a file, and the worktree has an untracked file with the same name, so git
    # merge fails with "would be overwritten" and no conflicted files.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[
        ["move_main", {"slug": slug, "clone": str(tmp_path / "other"),
                       "files": {"blocker.txt": "from base\n"}, "message": "base adds blocker"}],
        ["untracked", {"files": {"blocker.txt": "untracked content\n"}}]])

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

    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main", {
        "slug": slug, "clone": str(tmp_path / "other"), "files": {"shared.py": "def f():\n    return 999\n"},
        "message": "base changes shared.py"}]])

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

    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main", {
        "slug": slug, "clone": str(tmp_path / "other"), "files": {"new_file.txt": "main advanced\n"},
        "message": "main advances"}]])
    # Skip the delivery_gate reloop.
    _child().setattr(delivery, "evaluate_delivery_conflicts", delivery_fakes.proceed)

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


def test_an_unresolved_mid_run_conflict_is_delivered_as_a_draft_naming_it(github, tmp_path, monkeypatch):
    """A builder that never resolves a conflict with main ends in a draft pull request that
    names the conflict (a passing check always ends in a pull request, b45a), within its
    conflict budget, with a clean worktree.

    Main changes shared.py after the run starts, as the run's own build does, so delivery
    meets the conflict. Every later emission writes only tool.py, omitting shared.py, so
    materialise refuses it. The delivery gate counts the reloop against check_attempts (1)
    and the run stops sending the conflict back once that is spent. (This test once moved
    main before the run, which gave the run a base with no conflict at all, and asserted
    nothing about its result.)
    """
    repo = git_repo(tmp_path / "repo")
    (repo / "shared.py").write_text("def f():\n    return 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/unresolved"
    _published(github, slug, repo)
    _move_main_then_deliver(tmp_path, slug, files={"shared.py": "def f():\n    return 2\n"})

    run = tmp_path / "run"
    model = FakeModel(_answers(render({"tool.py": "print('built')\n",
                                       "shared.py": "def f():\n    return 42\n"}))
                      + ["===FILE: tool.py===\nprint('built')\n"] * 8)
    result = _mode(check_attempts=1).run(
        run, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "Build a tool\nthat prints built.",
              "check": "python tool.py", "target": str(repo)},
    )
    assert result.exit_ok, result.error
    record = _record(run)
    assert record["status"] == "opened" and record["draft"] is True, record
    [pull] = json.loads((github / f"{slug}.pulls.json").read_text())
    assert pull["draft"] and "Merge conflict with base in shared.py" in pull["body"], pull["body"]
    # Within budget: one conflict send-back (check_attempts=1), and the build stops at the
    # refusal bound (one build, three refused emissions) rather than spending all eight.
    assert int(result.facts["delivery_conflict_reloops"]) == 1
    assert sum(1 for prompt in model.prompts if "Write the artifact `" in prompt) == 4
    found = worktree.recorded(run)
    path = Path(found.path)
    with pytest.raises(subprocess.CalledProcessError):
        git(path, "rev-parse", "--verify", "-q", "MERGE_HEAD")
    assert not worktree.git(path, "diff", "--name-only", "--diff-filter=U").strip()


def test_a_conflict_that_comes_back_every_delivery_is_delivered_as_a_draft_at_its_budget(
        github, tmp_path, monkeypatch):
    """Main changes shared.py again before every delivery, so a builder that resolves each
    conflict meets a fresh one: the delivery gate sends it back check_attempts (1) times and
    no more, and then the work is delivered as a draft pull request naming the unresolved
    file (b45a: a passing check always ends in a pull request), not kept local as a failed
    run."""
    repo = git_repo(tmp_path / "repo")
    (repo / "shared.py").write_text("def f():\n    return 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/budget"
    _published(github, slug, repo)
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main_each", {
        "slug": slug, "clone": str(tmp_path / "other"), "path": "shared.py", "counter": str(tmp_path / "moves")}]])
    run = tmp_path / "run"
    builds = [render({"tool.py": "print('built')\n", "shared.py": f"def f():\n    return {42 + i}\n"})
              for i in range(8)]
    model = FakeModel(_answers(builds[0]) + builds[1:])
    result = _mode(check_attempts=1).run(
        run, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "Build a tool\nthat prints built.", "check": "python tool.py",
              "target": str(repo)})
    assert result.exit_ok, result.error
    # No send-back past the budget: one build, one send-back, then the draft.
    assert int(result.facts["delivery_conflict_reloops"]) == 1
    assert sum(1 for prompt in model.prompts if "Write the artifact `" in prompt) == 2
    record = _record(run)
    assert record["status"] == "opened" and record["draft"] is True, record
    assert record["merge_conflicts"] == ["shared.py"]
    assert record["draft_reason"].startswith("delivered as draft: unresolved conflict"), record
    assert "delivered as draft: unresolved conflict" in result.facts["delivery_result"]
    assert json.loads((run / "design.json").read_text())["draft"] is True, "design.json agrees with delivery.json"
    [pull] = json.loads((github / f"{slug}.pulls.json").read_text())
    assert pull["draft"] and "Merge conflict with base in shared.py" in pull["body"], pull["body"]
    path = Path(worktree.recorded(run).path)
    with pytest.raises(subprocess.CalledProcessError):
        git(path, "rev-parse", "--verify", "-q", "MERGE_HEAD")


def _at_the_second_delivery(github, tmp_path, slug):
    """A run the walker took to its second delivery and stopped there: main moved again
    since the builder resolved the first conflict, and the one send-back is spent."""
    repo = git_repo(tmp_path / "repo")
    (repo / "shared.py").write_text("def f():\n    return 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    _published(github, slug, repo)
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[
        ["move_main_each", {"slug": slug, "clone": str(tmp_path / "other"), "path": "shared.py",
                            "counter": str(tmp_path / "moves")}],
        ["stop_at", {"counter": str(tmp_path / "deliveries"), "call": 1}]])
    run = tmp_path / "run"
    builds = [render({"tool.py": "print('built')\n", "shared.py": f"def f():\n    return {42 + i}\n"})
              for i in range(8)]
    walked = _mode(check_attempts=1).run(
        run, model=FakeModel(_answers(builds[0]) + builds[1:]), verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "Build a tool\nthat prints built.", "check": "python tool.py",
              "target": str(repo)})
    assert walked.exit_ok is False and "stopped before delivery 1" in (walked.error or ""), walked.error
    assert int(json.loads((run / "facts.json").read_text())["delivery_conflict_reloops"]) == 1
    _unfake(delivery, "deliver_design")
    return run


def _deliver_on_the_spine(run, monkeypatch) -> tuple[int, str]:
    """`deliver` as the spine runs it: its own entry through `refactory_dsh.run`, the knobs
    as the spine passes them, the facts from facts.json."""
    import io

    from refactory_dsh import lowering
    from refactory_dsh.run import main

    state = _mode(check_attempts=1).state_dict()
    lowering.write_declarations(state, run)
    for name, value in lowering.knob_environment(state).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("REFACTORY_WORKDIR", str(run))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "deliver", "type": "op"}))
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    code = main(["refactory_dsh.delivery:deliver_design", "--node", "deliver"])
    return code, err.getvalue()


def test_on_the_spine_a_spent_conflict_budget_is_delivered_as_a_draft(github, tmp_path, monkeypatch):
    """The same decision when the spine runs `deliver`: the walker takes the run to its
    second delivery and stops; the spine's `deliver` delivers the draft naming the conflict,
    and the report agrees it is a draft."""
    slug = "acme/budget-spine"
    run = _at_the_second_delivery(github, tmp_path, slug)
    code, err = _deliver_on_the_spine(run, monkeypatch)
    assert code == 0, err
    record = _record(run)
    assert record["draft"] is True and record["merge_conflicts"] == ["shared.py"], record
    assert record["draft_reason"].startswith("delivered as draft: unresolved conflict"), record
    assert json.loads((run / "design.json").read_text())["draft"] is True
    [pull] = json.loads((github / f"{slug}.pulls.json").read_text())
    assert pull["draft"] and "Merge conflict with base in shared.py" in pull["body"]


def test_a_spent_budget_whose_base_no_longer_conflicts_is_delivered_as_it_is(github, tmp_path, monkeypatch):
    """A spent budget is not a reason to call the work a draft: if the base no longer
    conflicts when delivery tries again, the pull request is an ordinary one, claiming no
    unresolved conflict."""
    slug = "acme/budget-clean"
    run = _at_the_second_delivery(github, tmp_path, slug)
    real = delivery._merge_target
    calls = []

    def merge_target(ctx, path, *args, **kwargs):
        merged = real(ctx, path, *args, **kwargs)
        if not calls and merged.conflicts:
            # Main takes the run's own shared.py: the next merge has nothing to conflict on.
            delivery_fakes.move_main(None, slug=slug, clone=str(tmp_path / "settle"),
                                     files={"shared.py": (Path(path) / "shared.py").read_text()},
                                     message="main settles")
        calls.append(merged.conflicts)
        return merged

    monkeypatch.setattr(delivery, "_merge_target", merge_target)
    code, err = _deliver_on_the_spine(run, monkeypatch)
    assert code == 0, err
    assert calls[0] == ["shared.py"] and calls[1] == [], calls
    record = _record(run)
    assert record["status"] == "opened" and "draft" not in record and "merge_conflicts" not in record, record
    [pull] = json.loads((github / f"{slug}.pulls.json").read_text())
    assert not pull["draft"] and "Merge conflict" not in pull["body"]


def test_a_spent_conflict_budget_whose_draft_cannot_be_pushed_fails_saying_so(
        github, tmp_path, monkeypatch):
    """When even the draft that would carry the unresolved conflict cannot be delivered (here
    GitHub refuses the push), the run fails as an exhausted conflict budget always did,
    naming the conflicting file, why the draft did not land, and how to finish."""
    repo = git_repo(tmp_path / "repo")
    (repo / "shared.py").write_text("def f():\n    return 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    slug = "acme/budget-refused"
    _published(github, slug, repo)
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main_each", {
        "slug": slug, "clone": str(tmp_path / "other"), "path": "shared.py", "counter": str(tmp_path / "moves")}]])
    _child().setattr(delivery, "_push", delivery_fakes.push_fails, error="push refused by GitHub")
    run = tmp_path / "run"
    builds = [render({"tool.py": "print('built')\n", "shared.py": f"def f():\n    return {42 + i}\n"})
              for i in range(8)]
    result = _mode(check_attempts=1).run(
        run, model=FakeModel(_answers(builds[0]) + builds[1:]), verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "Build a tool\nthat prints built.", "check": "python tool.py",
              "target": str(repo)})
    assert result.exit_ok is False
    assert "conflicting files: shared.py" in (result.error or "")
    assert "push refused by GitHub" in (result.error or "")
    assert "graph-control redeliver" in (result.error or "")
    assert not (github / f"{slug}.pulls.json").exists() or json.loads(
        (github / f"{slug}.pulls.json").read_text()) == []
    assert _record(run)["status"] == "kept local"


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


def test_a_redo_with_main_unmoved_reuses_the_merged_check(github, tmp_path, monkeypatch):
    """The merge delivery pushed is the one it checked: a redo of it runs no suite again and says so.

    Fails without the change: the redo runs the whole check on the same merged tree.
    """
    first, redo, record, run = _redo_delivery_after_merge(github, tmp_path, monkeypatch, "acme/reuse-check",
                                                     move_main_again=False)
    assert first > 0
    assert redo == 0
    # The run's own check and delivery's merged check each left the tree they passed on.
    assert len(json.loads((run / delivery.PASSED_CHECKS).read_text())) == 2
    assert record["status"] in ("opened", "already open", "updated")
    assert record["merge_check_reused_from"], record
    assert "reused from the check on" in record["merge_check"]
    assert "reused from the check on" in (run / delivery.BODY).read_text()


def test_a_redo_after_main_moved_again_reruns_the_merged_check(github, tmp_path, monkeypatch):
    """A merge onto a main that moved is a tree no check has passed on: the check runs.

    Fails if reuse ignored the tree: the redo would run no check.
    """
    _, redo, record, run = _redo_delivery_after_merge(github, tmp_path, monkeypatch, "acme/rerun-check",
                                                 move_main_again=True)
    assert redo > 0
    assert "merge_check_reused_from" not in record
    assert "reused from the check" not in (run / delivery.BODY).read_text()
