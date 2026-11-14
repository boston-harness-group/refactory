"""A design run's own worktree: created, committed to, checked and pruned (#351).

Ported from remote-factory's `factory/worktree.py`, and so is much of what is tested
here: where a test stands in for one of its test classes, its docstring says which.
Real repositories throughout; only the installers (`uv`, `pnpm`) are stand-ins, on the
PATH.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from refactory_dsh import authoring_evidence, sandbox, worktree
from refactory_dsh.paths import venv_bin
from tests_support import git, git_repo, on_stdin, stub


@pytest.fixture
def repo(tmp_path) -> Path:
    return git_repo(tmp_path / "repo")


def _run(tmp_path: Path, name: str = "run") -> Path:
    run = tmp_path / name
    run.mkdir(parents=True, exist_ok=True)
    return run


def _spine(run: Path, status: str) -> None:
    """Leave the status the spine writes for a run it drives."""
    (run / ".refactory").mkdir(parents=True, exist_ok=True)
    (run / ".refactory" / "spine-status.json").write_text(json.dumps({"status": status}))


# ── a worktree of its own, from a pinned base (#330 I1) ────────────────────
def test_a_run_gets_its_own_worktree_on_its_own_branch_outside_the_target(repo, tmp_path):
    """The person's checkout keeps its branch and gains nothing; the run works elsewhere."""
    found = worktree.create(repo, "HEAD", "abcdef123456", _run(tmp_path))
    path = Path(found.path)
    assert path.is_relative_to(worktree.root()) and not path.is_relative_to(repo)
    assert found.branch == "refactory/run-abcdef123456"
    assert git(path, "branch", "--show-current") == found.branch
    assert (path / "app.py").read_text() == "print('app')\n"
    assert git(repo, "branch", "--show-current") == "main"
    assert git(repo, "status", "--porcelain") == "", "nothing was added to the target"
    assert worktree.recorded(tmp_path / "run") == found


def test_the_base_is_a_commit_pinned_at_prepare_even_if_head_is_amended(repo, tmp_path):
    """remote-factory's ``TestSHAResolution``."""
    start = git(repo, "rev-parse", "HEAD")
    found = worktree.create(repo, "HEAD", "aaaaaaaa1111", _run(tmp_path))
    git(repo, "commit", "-q", "--amend", "-m", "amended")
    assert found.base == start != git(repo, "rev-parse", "HEAD")
    assert git(Path(found.path), "rev-parse", "HEAD") == start


def test_a_base_other_than_head_is_where_the_run_starts(repo, tmp_path):
    """remote-factory's ``test_worktree_uses_custom_base_branch``."""
    first = git(repo, "rev-parse", "HEAD")
    (repo / "later.py").write_text("x = 1\n")
    git(repo, "add", "later.py")
    git(repo, "commit", "-q", "-m", "later")
    found = worktree.create(repo, first, "abab0000cdcd", _run(tmp_path))
    assert found.base == first and not (Path(found.path) / "later.py").exists()
    with pytest.raises(ValueError, match="the base 'no-such-branch' is not a commit"):
        worktree.create(repo, "no-such-branch", "abab1111cdcd", _run(tmp_path, "other"))


def test_a_redo_of_prepare_keeps_the_worktree_and_the_base_it_was_pinned_to(repo, tmp_path):
    run = _run(tmp_path)
    first = worktree.create(repo, "HEAD", "bbbbbbbb2222", run)
    (repo / "later.py").write_text("x = 1\n")
    git(repo, "add", "later.py")
    git(repo, "commit", "-q", "-m", "later")
    assert worktree.create(repo, "HEAD", "bbbbbbbb2222", run) == first


def test_a_run_that_died_before_recording_its_worktree_finds_it_again(repo, tmp_path):
    """The target's record is written first, so nothing is orphaned (``TestPruneStale``)."""
    run = _run(tmp_path)
    first = worktree.create(repo, "HEAD", "bcbcbcbc2323", run)
    (run / worktree.RECORD).unlink()
    assert worktree.create(repo, "HEAD", "bcbcbcbc2323", run) == first


def test_a_resumable_run_whose_worktree_was_removed_gets_it_back_with_its_attempts(repo, tmp_path):
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "cdcdcdcd3434", run)
    attempt = worktree.commit_attempt(run, {Path("tool.py"): "v1\n"}, "attempt 1")
    git(repo, "worktree", "remove", found.path)
    again = worktree.create(repo, "HEAD", "cdcdcdcd3434", run)
    assert (again.path, again.base, again.head) == (found.path, found.base, attempt.head)
    assert (Path(again.path) / "tool.py").read_text() == "v1\n"


def test_a_worktree_deleted_by_hand_is_added_again_from_its_branch(repo, tmp_path):
    import shutil

    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "cececece3434", run)
    attempt = worktree.commit_attempt(run, {Path("tool.py"): "v1\n"}, "attempt 1")
    shutil.rmtree(found.path)
    again = worktree.create(repo, "HEAD", "cececece3434", run)
    assert again.head == attempt.head and (Path(again.path) / "tool.py").read_text() == "v1\n"


def test_a_branch_whose_checkout_failed_is_found_again_by_the_redo(repo, tmp_path):
    """git keeps the branch when only the checkout fails, as an offline git-lfs does."""
    (repo / ".gitattributes").write_text("app.py filter=broken\n")
    git(repo, "add", ".gitattributes")
    git(repo, "commit", "-q", "-m", "a filter")
    git(repo, "config", "filter.broken.smudge", "false")
    git(repo, "config", "filter.broken.required", "true")
    run = _run(tmp_path)
    with pytest.raises(ValueError):
        worktree.create(repo, "HEAD", "cfcfcfcf3434", run)
    git(repo, "config", "--unset", "filter.broken.required")
    git(repo, "config", "--unset", "filter.broken.smudge")
    found = worktree.create(repo, "HEAD", "cfcfcfcf3434", run)
    assert (Path(found.path) / "app.py").exists()


def test_a_symlink_in_refactorys_home_does_not_hide_a_runs_own_worktree(repo, tmp_path, monkeypatch):
    real = tmp_path / "real-home"
    real.mkdir()
    (tmp_path / "linked-home").symlink_to(real)
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "linked-home"))
    run = _run(tmp_path)
    first = worktree.create(repo, "HEAD", "c1c1c1c13434", run)
    (run / worktree.RECORD).unlink()
    assert worktree.create(repo, "HEAD", "c1c1c1c13434", run) == first
    _spine(run, "failed")
    worktree.prune(repo, idle_hours=0)
    assert not Path(first.path).exists()


def test_a_branch_the_run_did_not_make_is_refused_and_left_alone(repo, tmp_path):
    git(repo, "branch", "refactory/run-dededede4545")
    with pytest.raises(ValueError, match="already exists and is not this run's"):
        worktree.create(repo, "HEAD", "dededede4545", _run(tmp_path))
    assert not (worktree.slot(repo) / "run-dededede4545.json").exists()
    assert git(repo, "branch", "--list", "refactory/run-dededede4545")


#: The real `worktree.git`, which a stand-in passes everything else to.
_GIT = worktree.git

#: git's words when an add read another add's half-written entry (CI, 2026-10-23).
HALF_WRITTEN = "fatal: failed to read .git/worktrees/run-cccccccc0001/commondir: Success"


def _refuse_adds(monkeypatch, said: str = HALF_WRITTEN, *, every: bool = False) -> list[tuple]:
    """Fail the first `git worktree add` (or ``every`` one) in git's words ``said``.

    git makes a new branch before it scans `.git/worktrees/`, so a refused ``-b`` add
    leaves its branch behind, as it does in git. Returns the adds that were refused.
    """
    real = _GIT
    refused: list[tuple] = []
    lock = threading.Lock()

    def git(path, *args, env=None):
        if "worktree" in args and "add" in args:
            with lock:
                refuse = every or not refused
                if refuse:
                    refused.append(args)
            if refuse:
                if "-b" in args:
                    real(path, "branch", args[args.index("-b") + 1], args[-1])
                raise ValueError(f"git {' '.join(args)} failed in {path}: {said}")
        return real(path, *args, env=env)

    monkeypatch.setattr(worktree, "git", git)
    monkeypatch.setattr(worktree, "ADD_JITTER_S", 0.01)
    return refused


def test_an_add_that_read_a_half_written_entry_is_tried_again(repo, tmp_path, monkeypatch):
    """dm-197: concurrent adds on one repository collide in git; the one that fails is
    retried, so every run still gets its own worktree."""
    refused = _refuse_adds(monkeypatch)
    made: list[worktree.Worktree] = []
    threads = [threading.Thread(target=lambda index=index: made.append(
        worktree.create(repo, "HEAD", f"cccccccc{index:04d}", _run(tmp_path, f"run{index}"))))
        for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(refused) == 1, "the injected failure did not happen"
    assert len({found.path for found in made}) == 4
    assert all(Path(found.path, "app.py").exists() for found in made)
    assert all(git(Path(found.path), "branch", "--show-current") == found.branch for found in made)


def test_only_a_half_written_entry_is_retried_and_only_so_often(repo, tmp_path, monkeypatch):
    """Any other failure, or one that left something behind, is refused at once."""
    refused = _refuse_adds(monkeypatch, "fatal: refused for another reason", every=True)
    with pytest.raises(ValueError, match="refused for another reason"):
        worktree.create(repo, "HEAD", "dddddddd0000", _run(tmp_path))
    assert len(refused) == 1
    # Nor is one that left something where the worktree goes.
    tried: list[tuple] = []

    def leaves_a_path(path, *args, env=None):
        tried.append(args)
        (tmp_path / "elsewhere").mkdir()
        raise ValueError(f"git worktree add failed in {path}: {HALF_WRITTEN}")

    monkeypatch.setattr(worktree, "git", leaves_a_path)
    with pytest.raises(ValueError, match="failed to read"):
        worktree._add(repo, tmp_path / "elsewhere", "some-branch")
    assert len(tried) == 1
    # The half-written failure itself is retried a bounded number of times.
    refused = _refuse_adds(monkeypatch, every=True)
    with pytest.raises(ValueError, match="failed to read"):
        worktree.create(repo, "HEAD", "dddddddd0001", _run(tmp_path, "run1"))
    assert len(refused) == worktree.ADD_ATTEMPTS


def test_two_runs_on_one_target_get_separate_worktrees_even_at_once(repo, tmp_path):
    """remote-factory's ``TestFilelockConcurrency``: concurrent adds each get a worktree;
    an add that collides with another in git is retried (dm-197)."""
    made: list[worktree.Worktree] = []
    threads = [threading.Thread(target=lambda index=index: made.append(
        worktree.create(repo, "HEAD", f"cccccccc{index:04d}", _run(tmp_path, f"run{index}"))))
        for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len({found.path for found in made}) == 4
    assert all(Path(found.path, "app.py").exists() for found in made)


def test_a_target_that_is_not_a_repository_root_or_has_no_commits_is_refused(repo, tmp_path):
    (repo / "sub").mkdir()
    with pytest.raises(ValueError, match="inside the repository"):
        worktree.repository_root(repo / "sub")
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="not a git repository"):
        worktree.repository_root(plain)
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-q")
    with pytest.raises(ValueError, match="has no commits"):
        worktree.repository_root(empty)


def test_none_of_the_targets_hooks_run_for_the_run(repo, tmp_path):
    """`--no-verify` stops pre-commit only; a post-checkout or post-commit hook is the person's."""
    log = tmp_path / "hooks.log"
    for hook in ("post-checkout", "post-commit", "pre-commit"):
        script = repo / ".git" / "hooks" / hook
        script.write_text(f"#!/bin/sh\necho {hook} >> {log}\n")
        script.chmod(0o755)
    run = _run(tmp_path)
    worktree.create(repo, "HEAD", "efefefef5656", run)
    worktree.commit_attempt(run, {Path("app.py"): "changed\n"}, "attempt 1")
    worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt 2")
    assert not log.exists()


# ── attempts ───────────────────────────────────────────────────────────────
def test_each_attempt_is_one_commit_that_builds_on_the_last(repo, tmp_path):
    """A retry changes only what it needs to: an earlier attempt's files stay until one deletes them."""
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "dddddddd3333", run)
    first = worktree.commit_attempt(run, {Path("tool.py"): "v1\n", Path("app.py"): "changed\n"}, "attempt 1")
    second = worktree.commit_attempt(run, {Path("other.py"): "v2\n"}, "attempt 2", (Path("tool.py"),))
    path = Path(found.path)
    assert git(path, "rev-list", "--count", f"{found.base}..HEAD") == "2"
    assert second.head != first.head != found.base
    assert sorted(git(path, "ls-tree", "--name-only", "HEAD").splitlines()) == ["app.py", "other.py"]
    assert (path / "app.py").read_text() == "changed\n", "an earlier attempt's edit is kept"
    assert not (path / "tool.py").exists(), "and a file it wrote goes only when an attempt deletes it"
    assert "other.py" in worktree.diff(second) and "app.py" in worktree.diff(second)


def test_a_file_written_over_a_directory_or_inside_a_file_is_refused_not_crashed_on(repo, tmp_path):
    """An earlier attempt's files now stay, so a later one can collide with them."""
    run = _run(tmp_path)
    worktree.create(repo, "HEAD", "fefefefe3333", run)
    worktree.commit_attempt(run, {Path("pkg/a.py"): "x\n"}, "attempt 1")
    with pytest.raises(worktree.AttemptRefused, match="is a directory of the last attempt"):
        worktree.commit_attempt(run, {Path("pkg"): "x\n"}, "attempt 2")
    with pytest.raises(worktree.AttemptRefused, match="inside app.py, which is a file"):
        worktree.commit_attempt(run, {Path("app.py/x"): "x\n"}, "attempt 2")
    with pytest.raises(worktree.AttemptRefused, match="nothing to delete"):
        worktree.commit_attempt(run, {}, "attempt 2", (Path("pkg"),))


def test_a_file_that_is_not_text_can_still_be_deleted(repo, tmp_path):
    """It cannot be edited, which is no reason it cannot be removed."""
    (repo / "logo.bin").write_bytes(b"\xff\xfe\x00")
    git(repo, "add", "logo.bin")
    git(repo, "commit", "-q", "-m", "a binary file")
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "bdbdbdbd3333", run)
    worktree.commit_attempt(run, {}, "attempt", (Path("logo.bin"),))
    assert "logo.bin" not in git(Path(found.path), "ls-tree", "--name-only", "HEAD").split()


def test_an_attempt_reads_and_deletes_only_what_the_last_attempt_committed(repo, tmp_path):
    """Edits apply to the commit, whatever a check left on disk; a deletion needs a file to delete."""
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "dededede3333", run)
    (Path(found.path) / "app.py").write_text("rewritten by a check\n")
    assert worktree.committed_text(found, Path("app.py")) == "print('app')\n"
    assert worktree.committed_text(found, Path("missing.py")) is None
    with pytest.raises(worktree.AttemptRefused, match="nothing to delete"):
        worktree.commit_attempt(run, {}, "attempt", (Path("missing.py"),))


def test_a_tracked_file_the_run_changed_on_disk_is_put_back_before_the_next_attempt(repo, tmp_path, capsys):
    """A check that rewrites a lockfile must not wedge the run; the change is reported."""
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "eeeeeeee4444", run)
    (Path(found.path) / "app.py").write_text("rewritten by a check\n")
    advanced = worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt")
    assert (Path(found.path) / "app.py").read_text() == "print('app')\n"
    assert git(Path(found.path), "show", f"{advanced.head}:app.py") == "print('app')"
    assert "were put back before this one: app.py" in capsys.readouterr().err


def test_a_commit_made_in_the_worktree_outside_the_run_is_refused_and_kept(repo, tmp_path):
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "e0e0e0e04444", run)
    (Path(found.path) / "app.py").write_text("someone else's commit\n")
    git(Path(found.path), "commit", "-q", "-am", "outside")
    head = git(Path(found.path), "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="not at its last attempt"):
        worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt")
    assert git(Path(found.path), "rev-parse", "HEAD") == head, "nothing is reverted"


def test_a_check_finds_the_worktree_put_back_to_its_last_attempt(repo, tmp_path):
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "e2e2e2e24444", run)
    (Path(found.path) / "app.py").write_text("edited after materialise\n")
    assert worktree.restore(run) == (found, ["app.py"])
    assert (Path(found.path) / "app.py").read_text() == "print('app')\n"


def test_a_worktree_whose_git_link_was_replaced_is_refused(repo, tmp_path):
    """Its `.git` file is what keeps its commits off the person's own branch."""
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "e3e3e3e34444", run)
    (Path(found.path) / ".git").write_text(f"gitdir: {repo / '.git'}\n")
    with pytest.raises(ValueError, match="no longer points at its own git directory"):
        worktree.require_current(run)


def test_a_pruned_worktree_comes_back_from_its_branch_when_the_run_goes_on(repo, tmp_path):
    """A failed node can be retried after its idle worktree was pruned."""
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "e1e1e1e14444", run)
    attempt = worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt")
    git(repo, "worktree", "remove", found.path)
    assert worktree.require_current(run).head == attempt.head
    assert (Path(found.path) / "tool.py").read_text() == "x\n"


def test_a_run_whose_worktree_and_branch_are_both_gone_is_refused(repo, tmp_path):
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "e4e4e4e44444", run)
    git(repo, "worktree", "remove", found.path)
    git(repo, "branch", "-D", found.branch)
    with pytest.raises(ValueError, match="and its branch refactory/run-e4e4e4e44444 are both gone"):
        worktree.require_current(run)


def test_an_attempt_that_died_before_its_commit_is_redone_cleanly(repo, tmp_path, capsys):
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "ffffffff5555", run)
    (run / worktree.PENDING).write_text(json.dumps({"files": ["app.py", "half.py"], "parent": found.head}))
    (Path(found.path) / "app.py").write_text("half-written\n")
    (Path(found.path) / "half.py").write_text("half-written\n")
    worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt")
    assert (Path(found.path) / "app.py").read_text() == "print('app')\n"
    assert not (Path(found.path) / "half.py").exists()
    assert not (run / worktree.PENDING).exists()
    assert "put back" not in capsys.readouterr().err, "the attempt's own files are not someone else's edits"


def test_an_attempt_that_died_after_its_commit_is_replaced_by_the_redo(repo, tmp_path):
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "f0f0f0f05555", run)
    (run / worktree.PENDING).write_text(json.dumps({"files": ["half.py"], "parent": found.head}))
    (Path(found.path) / "half.py").write_text("committed, never recorded\n")
    git(Path(found.path), "add", "half.py")
    git(Path(found.path), "commit", "-q", "-m", "attempt 1")
    advanced = worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt 1")
    assert git(Path(found.path), "rev-list", "--count", f"{found.base}..{advanced.head}") == "1"
    assert git(Path(found.path), "ls-tree", "--name-only", "HEAD").splitlines() == ["app.py", "tool.py"]


def test_an_attempt_never_commits_a_run_artifact_its_emission_names_or_the_worktree_holds(
        repo, tmp_path, capsys):
    """b40a committed `.refactory/verifier-findings.md` onto its branch (dm-242)."""
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "a2a2a2a2b4b4", run)
    path = Path(found.path)
    (path / ".refactory").mkdir()
    (path / ".refactory" / "x.md").write_text("a reviewer's stray notes\n")
    git(path, "add", "-f", ".refactory/x.md")
    advanced = worktree.commit_attempt(
        run, {Path("tool.py"): "x\n", Path(".refactory/verifier-findings.md"): "findings\n",
              Path(".Refactory/notes.md"): "notes\n"}, "attempt 1")
    assert git(path, "ls-tree", "-r", "--name-only", advanced.head).splitlines() == ["app.py", "tool.py"]
    assert not (path / ".refactory" / "verifier-findings.md").exists(), "never written either"
    assert ("were left out of it: .Refactory/notes.md, .refactory/verifier-findings.md"
            in capsys.readouterr().err)


def test_a_run_artifact_staged_in_the_worktree_is_never_amended_into_the_attempt(repo, tmp_path, monkeypatch):
    """An amendment commits what is staged: a lockfile amend must not carry a staged run artifact."""
    stub(tmp_path / "bin", "npm", "echo '{\"lockfileVersion\": 3}' > package-lock.json")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "a3a3a3a3b4b4", run)
    attempt = worktree.commit_attempt(run, {Path("package.json"): "{}\n"}, "attempt 1")
    path = Path(found.path)
    for name in (".refactory/verifier-findings.md", ".Refactory/notes.md"):
        (path / name).parent.mkdir(exist_ok=True)
        (path / name).write_text("findings\n")
        git(path, "add", "-f", name)
    amended, problems, _pf = worktree.ensure_environment(attempt, run)
    assert problems == [] and amended.head != attempt.head
    assert sorted(git(path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()) == [
        "app.py", "package-lock.json", "package.json"]
    assert (path / ".refactory" / "verifier-findings.md").read_text() == "findings\n", "left on disk, unstaged"


def test_an_attempt_never_deletes_a_run_artifact_the_project_tracks(repo, tmp_path, capsys):
    """Only the top-level `.refactory/` is the run's; a delete there is left out like a write."""
    (repo / ".refactory").mkdir()
    (repo / ".refactory" / "kept.md").write_text("the project's own\n")
    git(repo, "add", "-f", ".refactory/kept.md")
    git(repo, "commit", "-q", "-m", "tracked run-artifact path")
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "a5a5a5a5b4b4", run)
    path = Path(found.path)
    advanced = worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt 1", (Path(".refactory/kept.md"),))
    assert ".refactory/kept.md" in git(path, "ls-tree", "-r", "--name-only", advanced.head).splitlines()
    assert (path / ".refactory" / "kept.md").read_text() == "the project's own\n"
    assert "were left out of it: .refactory/kept.md" in capsys.readouterr().err


def test_a_merge_attempt_never_commits_a_staged_run_artifact(repo, tmp_path):
    """While a merge is in progress the index is kept as it is, so a staged run artifact
    is unstaged on its own before the merge commit."""
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "a6a6a6a6b4b4", run)
    path = Path(found.path)
    (repo / "later.py").write_text("x = 1\n")
    git(repo, "add", "later.py")
    git(repo, "commit", "-q", "-m", "main moved on")
    git(path, "-c", "core.hooksPath=/dev/null", "merge", "-q", "--no-ff", "--no-commit", "main")
    (path / ".refactory").mkdir()
    (path / ".refactory" / "x.md").write_text("stray\n")
    git(path, "add", "-f", ".refactory/x.md")
    advanced = worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt 1")
    assert sorted(git(path, "ls-tree", "-r", "--name-only", advanced.head).splitlines()) == [
        "app.py", "later.py", "tool.py"]


def test_an_emission_of_only_run_artifacts_is_refused_as_such(repo, tmp_path):
    run = _run(tmp_path)
    worktree.create(repo, "HEAD", "a7a7a7a7b4b4", run)
    with pytest.raises(worktree.AttemptRefused, match="names only run artifacts under .refactory/"):
        worktree.commit_attempt(run, {Path(".refactory/verifier-findings.md"): "findings\n"}, "attempt 1")


@pytest.mark.parametrize("name", [".git/config", ".GIT", "sub/.Git/HEAD"])
def test_an_emitted_path_inside_gits_own_state_is_refused_in_any_case(repo, tmp_path, name):
    """A `.GIT` on a case-insensitive disk would replace the worktree's link to its repository."""
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "abababab6666", run)
    with pytest.raises(ValueError, match="git's own state"):
        worktree.commit_attempt(run, {Path(name): "gitdir: elsewhere\n"}, "attempt")
    assert worktree.is_run_worktree(Path(found.path))


def test_an_attempt_never_writes_through_a_symlink(repo, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("theirs\n")
    (repo / "link.txt").symlink_to(outside)
    git(repo, "add", "link.txt")
    git(repo, "commit", "-q", "-m", "a link")
    run = _run(tmp_path)
    worktree.create(repo, "HEAD", "acacacac6767", run)
    with pytest.raises(ValueError, match="goes through the symlink link.txt"):
        worktree.commit_attempt(run, {Path("link.txt"): "mine\n"}, "attempt")
    assert outside.read_text() == "theirs\n"


def test_a_refused_attempt_never_makes_the_next_one_delete_through_a_symlink(repo, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "notes.txt").write_text("theirs\n")
    (repo / "docs").symlink_to(outside)
    git(repo, "add", "docs")
    git(repo, "commit", "-q", "-m", "a linked directory")
    run = _run(tmp_path)
    worktree.create(repo, "HEAD", "adadadad6767", run)
    with pytest.raises(ValueError, match="goes through the symlink docs"):
        worktree.commit_attempt(run, {Path("docs/notes.txt"): "mine\n"}, "attempt 1")
    worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt 1")
    assert (outside / "notes.txt").read_text() == "theirs\n"


def test_a_file_left_in_the_worktree_is_gone_before_the_next_attempt_but_installs_stay(
        repo, tmp_path, monkeypatch, capsys):
    """A reviewer's stray `conftest.py` must not be in a check without being on the branch.

    Removed without a warning, as a check's `__pycache__` is: only a changed tracked
    file says that a step changed the source.
    """
    bin_dir = tmp_path / "bin"
    stub(bin_dir, "pnpm", "mkdir -p node_modules/pkg && echo x > node_modules/pkg/index.js")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    for name in ("package.json", "pnpm-lock.yaml"):
        (repo / name).write_text("{}\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "js")
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "afafafaf6767", run)
    worktree.ensure_environment(found, run)
    (Path(found.path) / "conftest.py").write_text("stray\n")
    worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt")
    assert not (Path(found.path) / "conftest.py").exists()
    assert (Path(found.path) / "node_modules" / "pkg").is_dir()
    assert "put back" not in capsys.readouterr().err


def test_only_what_an_install_left_untracked_is_kept_between_attempts(repo, tmp_path, monkeypatch):
    """Found live: pnpm leaves ignored `node_modules` inside tracked package directories."""
    (repo / ".gitignore").write_text("build/\n")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "index.js").write_text("x\n")
    for name in ("package.json", "pnpm-lock.yaml"):
        (repo / name).write_text("{}\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "js")
    stub(tmp_path / "bin", "pnpm", "mkdir -p pkg/build node_modules && echo x > pkg/build/out.js "
                                   "&& echo x > node_modules/dep.js")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "abcdabcd6767", run)
    worktree.ensure_environment(found, run)
    record = json.loads((worktree.slot(repo) / "run-abcdabcd6767.json").read_text())
    assert record["installed"] == ["node_modules/"]
    (Path(found.path) / "pkg" / "stray.js").write_text("left by a reviewer\n")
    worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt")
    assert not (Path(found.path) / "pkg" / "stray.js").exists()
    assert (Path(found.path) / "node_modules" / "dep.js").exists() and (Path(found.path) / "pkg" / "build").is_dir()


# ── whether a run still needs its worktree ─────────────────────────────────
def test_a_run_whose_state_cannot_be_read_counts_as_active(tmp_path):
    """Fails closed, where remote-factory's session guard fails open."""
    run = _run(tmp_path)
    (run / ".refactory").mkdir()
    (run / ".refactory" / "spine-status.json").write_text("{not json")
    assert worktree.state(run) == "active"


@pytest.mark.parametrize("status", ["halted", "cancelled", "paused", "waiting_user"])
def test_a_run_that_can_still_be_resumed_counts_as_active(tmp_path, status):
    run = _run(tmp_path)
    _spine(run, status)
    assert worktree.state(run) == "active"


def test_a_held_invocation_lock_or_a_pending_delivery_keeps_a_run_active(tmp_path):
    import fcntl

    run = _run(tmp_path)
    _spine(run, "failed")
    lock = run / ".refactory" / "invocation.lock"
    with open(lock, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        assert worktree.state(run) == "active"
    assert worktree.state(run) == "ended"
    pending = run / ".refactory" / "agent-delivery" / "pending"
    pending.mkdir(parents=True)
    (pending / "x.json").write_text("{}")
    assert worktree.state(run) == "active"


def test_a_walked_run_ends_with_an_error_or_with_its_walker(tmp_path):
    run = _run(tmp_path)
    (run / ".refactory").mkdir()
    (run / ".refactory" / "progress.json").write_text(json.dumps({"node": "check"}))
    assert worktree.state(run, os.getpid()) == "active"
    walker = subprocess.Popen(["true"])
    walker.wait()
    assert worktree.state(run, walker.pid) == "ended"
    (run / ".refactory" / "progress.json").write_text(json.dumps({"node": "check", "error": "boom"}))
    assert worktree.state(run, os.getpid()) == "ended"


def test_a_successful_run_keeps_its_worktree_after_its_directory_is_reused(repo, tmp_path):
    """A second run in one session reuses the run directory; the first one's success stands."""
    run = _run(tmp_path)
    first = worktree.create(repo, "HEAD", "1a1a1a1a0000", run)
    worktree.record_outcome(first, "succeeded")
    second = worktree.create(repo, "HEAD", "2b2b2b2b0000", run)
    _spine(run, "failed")
    worktree.prune(repo, idle_hours=0)
    assert Path(first.path).exists() and not Path(second.path).exists()


def test_a_run_whose_directory_moved_on_to_a_later_run_has_ended(repo, tmp_path):
    run = _run(tmp_path)
    first = worktree.create(repo, "HEAD", "3c3c3c3c0000", run)
    second = worktree.create(repo, "HEAD", "4d4d4d4d0000", run)
    _spine(run, "running")
    worktree.prune(repo, idle_hours=0)
    assert not Path(first.path).exists() and Path(second.path).exists()


def test_prune_removes_an_idle_ended_runs_worktree_and_keeps_active_and_succeeded_ones(repo, tmp_path):
    ended, active, succeeded = (_run(tmp_path, name) for name in ("ended", "active", "succeeded"))
    made = {run.name: worktree.create(repo, "HEAD", f"1234567{index}abcd", run)
            for index, run in enumerate((ended, active, succeeded))}
    _spine(ended, "failed")
    _spine(active, "ready")
    (succeeded / "design.json").write_text("{}")
    notes = worktree.prune(repo, idle_hours=0)
    assert not Path(made["ended"].path).exists()
    assert Path(made["active"].path).exists() and Path(made["succeeded"].path).exists()
    assert git(repo, "branch", "--list", made["ended"].branch), "the branch outlives the worktree"
    assert any("removed the worktree of a run" in note for note in notes)


def test_a_recently_ended_run_keeps_its_worktree_until_it_has_been_idle_long_enough(tmp_path):
    """remote-factory's ``test_fresh_worktree_not_touched``, and its idle counterpart."""
    old = time.time() - (worktree.IDLE_HOURS + 1) * 3600
    stamp = f"@{int(old)} +0000"
    repo = git_repo(tmp_path / "repo")
    git(repo, "commit", "-q", "--amend", "--no-edit", "--date", stamp,
        env={**os.environ, "GIT_COMMITTER_DATE": stamp})
    fresh, stale = _run(tmp_path, "fresh"), _run(tmp_path, "stale")
    kept = worktree.create(repo, "HEAD", "5e5e5e5e0000", fresh)
    gone = worktree.create(repo, "HEAD", "6f6f6f6f0000", stale)
    _spine(fresh, "failed")
    _spine(stale, "failed")
    os.utime(stale / ".refactory" / "spine-status.json", (old, old))
    worktree.prune(repo)
    assert Path(kept.path).exists() and not Path(gone.path).exists()


def test_prune_keeps_a_worktree_where_a_file_appeared_after_the_runs_last_step(repo, tmp_path):
    """Untracked files from the run's own steps go; one someone left there later keeps it."""
    old = time.time() - 3600
    left, later = _run(tmp_path, "left"), _run(tmp_path, "later")
    removed = worktree.create(repo, "HEAD", "9a9a9a9a0000", left)
    kept = worktree.create(repo, "HEAD", "9b9b9b9b0000", later)
    for found, run in ((removed, left), (kept, later)):
        (Path(found.path) / "__pycache__").mkdir()
        (Path(found.path) / "__pycache__" / "app.pyc").write_text("from the check\n")
        os.utime(Path(found.path) / "__pycache__" / "app.pyc", (old, old))
        _spine(run, "failed")
    (Path(kept.path) / "notes.md").write_text("written after the run ended\n")
    future = time.time() + 2 * worktree.SLACK_S
    os.utime(Path(kept.path) / "notes.md", (future, future))
    notes = worktree.prune(repo, idle_hours=0)
    assert not Path(removed.path).exists()
    assert (Path(kept.path) / "notes.md").exists()
    assert any(note.startswith(f"kept {kept.path}: notes.md changed there after") for note in notes)


def test_prune_takes_a_succeeded_runs_worktree_only_when_asked_and_never_its_branch(repo, tmp_path):
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "9d9d9d9d0000", run)
    worktree.record_outcome(found, "succeeded")
    worktree.prune(repo, idle_hours=0)
    assert Path(found.path).exists()
    worktree.prune(repo, idle_hours=0, succeeded=True)
    assert not Path(found.path).exists() and git(repo, "branch", "--list", found.branch)


def test_a_report_left_by_another_run_in_the_directory_does_not_make_this_one_succeed(tmp_path):
    """A walker run again in one directory finds the earlier run's `design.json` there."""
    run = _run(tmp_path)
    (run / "design.json").write_text(json.dumps({"branch": "refactory/run-aaaa0000aaaa"}))
    _spine(run, "failed")
    assert worktree.state(run, branch="refactory/run-aaaa0000aaaa") == "succeeded"
    assert worktree.state(run, branch="refactory/run-bbbb0000bbbb") == "ended"


def test_prune_never_forces_a_worktree_with_uncommitted_work(repo, tmp_path):
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "9999aaaa0000", run)
    _spine(run, "failed")
    (Path(found.path) / "app.py").write_text("unsaved work\n")
    notes = worktree.prune(repo, idle_hours=0)
    assert (Path(found.path) / "app.py").read_text() == "unsaved work\n"
    assert any(note.startswith(f"kept {found.path}") and "--force" in note for note in notes)


def test_prune_removes_a_worktree_holding_only_what_the_run_left_untracked(repo, tmp_path, monkeypatch):
    """Found live: a check leaves `__pycache__` in a new project, which has no `.gitignore`."""
    bin_dir = tmp_path / "bin"
    stub(bin_dir, "pnpm", "mkdir -p node_modules/pkg && echo x > node_modules/pkg/index.js")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    for name in ("package.json", "pnpm-lock.yaml"):
        (repo / name).write_text("{}\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "js")
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "7a7a7a7a0000", run)
    worktree.ensure_environment(found, run)
    for left in ("__pycache__/app.cpython-312.pyc", ".pytest_cache/v/cache/nodeids", "dist/out.js"):
        (Path(found.path) / left).parent.mkdir(parents=True, exist_ok=True)
        (Path(found.path) / left).write_text("left by the check\n")
    _spine(run, "failed")
    worktree.prune(repo, idle_hours=0)
    assert not Path(found.path).exists()


def test_prune_removes_a_worktree_whose_installs_wrote_a_lockfile(repo, tmp_path, monkeypatch):
    """`uv sync` without a `uv.lock` writes one, untracked; it is the run's, not the person's."""
    stub(tmp_path / "bin", "uv", "mkdir -p .venv/bin && echo lock > uv.lock")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    _python_project(repo)
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "7c7c7c7c0000", run)
    worktree.ensure_environment(found, run)
    _spine(run, "failed")
    worktree.prune(repo, idle_hours=0)
    assert not Path(found.path).exists()


def test_a_worktree_with_submodules_is_removed_once_nothing_in_it_is_uncommitted(tmp_path, monkeypatch):
    """git refuses any worktree holding submodules; one that is clean throughout can go."""
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'protocol.file.allow'='always'")
    sub = git_repo(tmp_path / "sub")
    repo = git_repo(tmp_path / "repo")
    git(repo, "submodule", "-q", "add", str(sub), "sub")
    git(repo, "commit", "-q", "-m", "a submodule")
    # As a person's clone is: the submodule is declared, but nothing initialised it here.
    git(repo, "config", "--remove-section", "submodule.sub")
    clean, dirty = _run(tmp_path, "clean"), _run(tmp_path, "dirty")
    removed = worktree.create(repo, "HEAD", "7d7d7d7d0000", clean)
    kept = worktree.create(repo, "HEAD", "7e7e7e7e0000", dirty)
    for found, run in ((removed, clean), (kept, dirty)):
        assert worktree.ensure_environment(found, run)[1] == []
        assert worktree.shared_changed(found) is False, "initialising submodules is refactory's own doing"
        _spine(run, "failed")
    (Path(kept.path) / "sub" / "app.py").write_text("work inside the submodule\n")
    worktree.prune(repo, idle_hours=0)
    assert not Path(removed.path).exists()
    assert (Path(kept.path) / "sub" / "app.py").read_text() == "work inside the submodule\n"


def test_an_ended_runs_branch_goes_after_its_idle_days_but_a_named_branch_never(repo, tmp_path, monkeypatch):
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "8888bbbb1111", run)
    _spine(run, "failed")
    named = worktree.slot(repo) / "run-person.json"
    named.write_text(json.dumps({"branch": "feature/mine", "path": str(tmp_path / "nowhere"),
                                 "run": str(tmp_path / "gone")}))
    git(repo, "branch", "feature/mine")
    monkeypatch.setattr(worktree, "_idle_hours", lambda *args: worktree.FAILED_BRANCH_DAYS * 24 + 1)
    worktree.prune(repo)
    assert not git(repo, "branch", "--list", found.branch)
    assert git(repo, "branch", "--list", "feature/mine"), "a branch a person named is never touched"
    assert named.exists()


def test_prune_leaves_the_targets_own_worktrees_alone(repo, tmp_path):
    """git's own `worktree prune` would drop a person's worktree on a disk that is not mounted."""
    theirs = tmp_path / "theirs"
    git(repo, "worktree", "add", "-q", "-b", "theirs", str(theirs))
    (theirs / "app.py").write_text("their staged work\n")
    git(theirs, "add", "app.py")
    hidden = tmp_path / "unmounted"
    theirs.rename(hidden)
    run = _run(tmp_path)
    worktree.create(repo, "HEAD", "9c9c9c9c0000", run)
    _spine(run, "failed")
    worktree.prune(repo, idle_hours=0)
    hidden.rename(theirs)
    assert git(theirs, "diff", "--cached", "--name-only") == "app.py"


# ── the check environment (remote-factory's test_worktree_venv) ───────────
def test_a_check_in_a_worktree_sees_none_of_refactorys_variables(repo, tmp_path, monkeypatch):
    """`REFACTORY_KNOBS` once failed a target's own test suite (#330 E1)."""
    monkeypatch.setenv("PYTHONPATH", "/launcher/refactory")
    monkeypatch.setenv("REFACTORY_KNOBS", '{"x": 1}')
    monkeypatch.setenv("DSH_WORKFLOW_NODE", "{}")
    found = worktree.create(repo, "HEAD", "7777cccc2222", _run(tmp_path))
    env = worktree.check_environment(Path(found.path), tmp_path / "home")
    assert set(env) <= {"PATH", "HOME", "VIRTUAL_ENV", *worktree.CHECK_ENV, *worktree.HOOK_SUPPRESS}
    assert env["HOME"] == str(tmp_path / "home")
    assert env["PATH"].split(os.pathsep)[0] == str(venv_bin()), "none of its own, so refactory's"


def test_a_managed_environment_comes_first_and_an_unmarked_one_is_not_adopted(repo, tmp_path):
    found = worktree.create(repo, "HEAD", "6666dddd3333", _run(tmp_path))
    bin_dir = Path(found.path) / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    unmarked = worktree.check_environment(Path(found.path), tmp_path)
    assert str(bin_dir) not in unmarked["PATH"] and "VIRTUAL_ENV" not in unmarked
    (Path(found.path) / ".venv" / worktree.MARKER).write_text("")
    marked = worktree.check_environment(Path(found.path), tmp_path)
    assert marked["PATH"].split(os.pathsep)[0] == str(bin_dir)
    assert marked["VIRTUAL_ENV"] == str(Path(found.path) / ".venv")
    assert str(venv_bin()) not in marked["PATH"].split(os.pathsep)


def test_a_check_uses_the_pnpm_store_its_install_used(repo, tmp_path):
    """pnpm finds its store through HOME: under the check's own it saw another store and tried to reinstall."""
    found = worktree.create(repo, "HEAD", "5e5e5e5e3333", _run(tmp_path))
    assert "pnpm_config_store_dir" not in worktree.check_environment(Path(found.path), tmp_path / "home")
    (Path(found.path) / "node_modules").mkdir()
    (Path(found.path) / "node_modules" / ".modules.yaml").write_text(
        json.dumps({"storeDir": "/home/someone/.local/share/pnpm/store/v11"}))
    env = worktree.check_environment(Path(found.path), tmp_path / "home")
    assert env["pnpm_config_store_dir"] == "/home/someone/.local/share/pnpm/store/v11"
    assert env["HOME"] == str(tmp_path / "home"), "only the store is shared; caches stay out of the worktree"


def _python_project(repo: Path) -> None:
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    git(repo, "add", "pyproject.toml")
    git(repo, "commit", "-q", "-m", "manifest")


def test_installs_run_once_per_manifest_content_and_again_when_it_changes(repo, tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    stub(bin_dir, "uv", "mkdir -p .venv/bin")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    _python_project(repo)
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "5555eeee4444", run)
    assert worktree.ensure_environment(found, run) == (found, [], None)
    assert worktree.ensure_environment(found, run) == (found, [], None)
    assert (bin_dir / "calls.log").read_text().count("uv sync") == 1
    assert (Path(found.path) / ".venv" / worktree.MARKER).exists()
    (Path(found.path) / "pyproject.toml").write_text("[project]\nname = 'y'\n")
    worktree.ensure_environment(found, run)
    assert (bin_dir / "calls.log").read_text().count("uv sync") == 2


def test_what_was_installed_is_known_per_worktree_not_per_run_directory(repo, tmp_path, monkeypatch):
    """A walker that runs again in the same directory, or a worktree added back, installs again."""
    stub(tmp_path / "bin", "uv", "mkdir -p .venv/bin")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    _python_project(repo)
    run = _run(tmp_path)
    first = worktree.create(repo, "HEAD", "5b5b5b5b4444", run)
    worktree.ensure_environment(first, run)
    second = worktree.create(repo, "HEAD", "5c5c5c5c4444", run)
    worktree.ensure_environment(second, run)
    git(repo, "worktree", "remove", "--force", second.path)
    worktree.ensure_environment(worktree.require_current(run), run)
    assert (tmp_path / "bin" / "calls.log").read_text().count("uv sync") == 3
    assert (Path(second.path) / ".venv" / worktree.MARKER).exists()


def test_installs_run_without_a_memory_cap(repo, tmp_path, monkeypatch):
    """A package manager under a 2 GB address space is killed before it installs anything."""
    stub(tmp_path / "bin", "uv", f'echo "cap=$(ulimit -v)" >> "{tmp_path}/bin/calls.log"; mkdir -p .venv/bin')
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    _python_project(repo)
    run = _run(tmp_path)
    worktree.ensure_environment(worktree.create(repo, "HEAD", "5d5d5d5d4444", run), run)
    assert "cap=unlimited" in (tmp_path / "bin" / "calls.log").read_text()


def test_an_install_that_failed_is_tried_again_and_a_missing_installer_is_named(repo, tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    stub(bin_dir, "uv", f'test -e "{bin_dir}/second" || {{ touch "{bin_dir}/second"; exit 1; }}; mkdir -p .venv/bin')
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    _python_project(repo)
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "5a5a5a5a4444", run)
    assert worktree.ensure_environment(found, run)[1][0].startswith("`uv sync --all-groups` failed")
    assert worktree.ensure_environment(found, run)[1] == []
    assert (bin_dir / "calls.log").read_text().count("uv sync") == 2
    only_git = tmp_path / "only-git"
    only_git.mkdir()
    (only_git / "git").symlink_to(subprocess.run(["sh", "-c", "command -v git"], capture_output=True,
                                                 text=True, check=True).stdout.strip())
    monkeypatch.setenv("PATH", str(only_git))
    (Path(found.path) / "pyproject.toml").write_text("[project]\nname = 'changed'\n")
    assert worktree.ensure_environment(found, run)[1] == [
        "UNAVAILABLE: uv is not installed, so `uv sync --all-groups` could not run"]


def test_a_lockfile_an_install_writes_is_amended_into_the_attempt(repo, tmp_path, monkeypatch):
    """A new JavaScript project has no lockfile; the one npm writes is part of the change."""
    stub(tmp_path / "bin", "npm", "echo '{\"lockfileVersion\": 3}' > package-lock.json")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "4444ffff5555", run)
    attempt = worktree.commit_attempt(run, {Path("package.json"): "{}\n"}, "attempt 1")
    amended, problems, _pf = worktree.ensure_environment(attempt, run)
    assert problems == [] and amended.head != attempt.head
    assert worktree.recorded(run) == amended
    assert git(Path(found.path), "rev-list", "--count", f"{found.base}..HEAD") == "1", "one commit per attempt"
    assert sorted(git(Path(found.path), "ls-tree", "--name-only", "HEAD").splitlines()) == [
        "app.py", "package-lock.json", "package.json"]
    assert "npm install" in (tmp_path / "bin" / "calls.log").read_text()


def test_a_lockfile_written_before_any_attempt_never_rewrites_the_base(repo, tmp_path, monkeypatch):
    stub(tmp_path / "bin", "npm", "echo '{\"lockfileVersion\": 3}' > package-lock.json")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    (repo / "package.json").write_text("{}\n")
    git(repo, "add", "package.json")
    git(repo, "commit", "-q", "-m", "js")
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "4646ffff5555", run)
    amended, _problems, _pf = worktree.ensure_environment(found, run)
    assert git(Path(found.path), "rev-parse", f"{amended.head}^") == found.base == git(repo, "rev-parse", "main")


def test_an_install_is_frozen_only_while_the_attempt_leaves_its_manifest_alone(repo, tmp_path, monkeypatch):
    """A change that adds a dependency must be installable (`uv sync --frozen` would ignore it)."""
    stub(tmp_path / "bin", "uv", "mkdir -p .venv/bin")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (repo / "uv.lock").write_text("version = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "locked")
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "4545ffff5555", run)
    worktree.ensure_environment(worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt 1"), run)
    worktree.ensure_environment(
        worktree.commit_attempt(run, {Path("pyproject.toml"): "[project]\nname = 'x'\ndependencies = ['six']\n"},
                                "attempt 2"), run)
    calls = (tmp_path / "bin" / "calls.log").read_text().splitlines()
    assert calls == ["uv sync --all-groups --frozen", "uv sync --all-groups"]


def test_installs_run_unattended_without_hook_managers_secrets_or_refactorys_variables(repo, tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    stub(bin_dir, "pnpm", f'echo "HUSKY=$HUSKY LEFTHOOK=$LEFTHOOK CI=$CI DSH=${{DSH_HOME-unset}} '
                          f'VENV=${{VENV_BIN-unset}} KEY=${{ANTHROPIC_API_KEY-unset}} '
                          f'NPM=${{NPM_TOKEN-unset}}" >> "{bin_dir}/calls.log"\n'
                          f'read answer && echo "read a prompt answer" >> "{bin_dir}/calls.log"')
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DSH_HOME", "/launcher/dsh")
    monkeypatch.setenv("VENV_BIN", "/launcher/.venv/bin")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-model-key")
    monkeypatch.setenv("NPM_TOKEN", "a-registry-token")
    for name in ("package.json", "pnpm-lock.yaml"):
        (repo / name).write_text("{}\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "js")
    run = _run(tmp_path)
    with on_stdin("y\n"):
        worktree.ensure_environment(worktree.create(repo, "HEAD", "3333aaaa6666", run), run)
    calls = (bin_dir / "calls.log").read_text()
    assert "HUSKY=0 LEFTHOOK=0 CI=true DSH=unset VENV=unset KEY=unset NPM=a-registry-token" in calls
    assert "read a prompt answer" not in calls, "an install's stdin is closed, so a prompt cannot wait on it"


def test_a_change_to_the_hooks_shared_with_the_target_is_reported_once(repo, tmp_path):
    """A worktree shares its target's hooks, so whatever "sets them up" changes the person's repo."""
    found = worktree.create(repo, "HEAD", "3a3a3a3a6666", _run(tmp_path))
    assert worktree.shared_changed(found) is False
    hooks = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")) / "hooks"
    (hooks / "pre-commit").write_text("#!/bin/sh\n")
    assert worktree.shared_changed(found) is True
    assert worktree.shared_changed(found) is False


def test_check_environment_suppresses_hook_managers(repo, tmp_path):
    """HOOK_SUPPRESS is merged into check_environment's output (#347)."""
    found = worktree.create(repo, "HEAD", "hook0000aaaa", _run(tmp_path))
    home = tmp_path / "home"
    home.mkdir()
    env = worktree.check_environment(Path(found.path), home)
    assert env["LEFTHOOK"] == "0", "LEFTHOOK must be suppressed"
    assert env["HUSKY"] == "0", "HUSKY must be suppressed"


def test_the_wrapper_suppresses_hook_managers(repo, tmp_path):
    """HOOK_SUPPRESS flows into the wrapper script via check_environment (#347)."""
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "hook1111bbbb", run)
    script = worktree.write_wrapper(found, run)
    text = script.read_text()
    assert "LEFTHOOK=" in text, "LEFTHOOK must appear in the wrapper"
    assert "HUSKY=" in text, "HUSKY must appear in the wrapper"


def test_the_wrapper_runs_a_command_in_the_worktree_with_the_check_environment(repo, tmp_path, monkeypatch):
    """Reviewers run commands through it, so a target that is refactory is not tested from the launcher."""
    monkeypatch.setenv("PYTHONPATH", "/launcher/refactory")
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "2222bbbb7777", run)
    script = worktree.write_wrapper(found, run)
    shown = subprocess.run([str(script), "sh", "-c", 'pwd; echo "$HOME"; echo "${PYTHONPATH-unset}"'],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    assert shown == [str(Path(found.path).resolve()), str(run / ".refactory" / "check-home"), "unset"]


def test_a_check_outside_a_worktree_keeps_todays_environment(tmp_path, monkeypatch):
    """Create, edit and design_fast checks are unchanged by #351: HOME is their directory, as before."""
    monkeypatch.setenv("PYTHONPATH", "/kept")
    result = sandbox.run_command('echo "$HOME"; echo "${PYTHONPATH-unset}"', workdir=tmp_path / "artifact")
    assert result.stdout.splitlines() == [str(tmp_path / "artifact"), "/kept"]
    # Without worktree_root the sentinel is absent and no false positive occurs.
    assert result.cwd_outside is None
    assert "__REFACTORY_CWD_ESCAPED" not in result.stderr
    assert "__refactory" not in result.stderr


def test_global_memory_lives_in_refactorys_home():
    """`REFACTORY_HOME` moves all of refactory's per-user state, not only the worktrees."""
    from refactory_dsh.memory import MemoryStore
    from refactory_dsh.paths import refactory_home

    store = MemoryStore([], run_root=Path("run"))
    assert store._roots["global"] == refactory_home() / "memory"


# ── identity: what a sign-off is about ────────────────────────────────────
def test_a_worktrees_identity_is_its_committed_attempt(repo, tmp_path):
    """Build output and a check's rewrites leave a sign-off current; a new attempt does not.

    The review comment on #330: identity must not count `lib/`, `dist/` or caches.
    """
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "1111cccc8888", run)
    path = Path(found.path)
    before = authoring_evidence.source_files(path)
    (path / "lib").mkdir()
    (path / "lib" / "built.js").write_text("built\n")
    (path / "app.py").write_text("rewritten on disk\n")
    assert authoring_evidence.source_files(path) == before
    worktree.commit_attempt(run, {Path("tool.py"): "x\n"}, "attempt")
    assert authoring_evidence.source_files(path) != before


def test_signal_killed_install_propagates_signal_name(repo, tmp_path, monkeypatch):
    """A signal-killed install (after retry) carries the signal name in its problem text."""
    _python_project(repo)
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "sig0000test", run)

    def fake_run_command(cmd, **kwargs):
        return sandbox.SandboxResult(
            ok=False, returncode=-6, elapsed_s=1.0,
            stderr="killed by SIGABRT", stdout="",
        )

    monkeypatch.setattr(sandbox, "run_command", fake_run_command)
    _, problems, prep_fail = worktree.ensure_environment(found, run)
    assert any("killed by SIGABRT" in p for p in problems)
    assert prep_fail is not None
    assert prep_fail["returncode"] == -6
    assert prep_fail["step"] == "uv sync --all-groups"


def test_identity_outside_a_worktree_still_counts_every_file(tmp_path):
    """A package's or an artifact directory's identity is what it was before #351."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "lib").mkdir()
    (tmp_path / "pkg" / "lib" / "built.js").write_text("built\n")
    assert list(authoring_evidence.source_files(tmp_path / "pkg")) == ["lib/built.js"]


# ── per-worktree hook isolation (#dm-96) ───────────────────────────────────


@pytest.mark.toolchain("lefthook")
def test_worktree_gets_isolated_hooks_path_and_hook_manager_leaves_target_unchanged(repo, tmp_path):
    """A worktree refactory creates has its own hooks path; lefthook writes there, not to shared hooks."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    lefthook = repo_root / "node_modules" / ".bin" / "lefthook"
    if not lefthook.is_file():
        pytest.skip("lefthook not found at node_modules/.bin/lefthook")

    # Commit lefthook.yml so the worktree sees it.
    (repo / "lefthook.yml").write_text("pre-commit:\n  commands:\n    echo:\n      run: echo ok\n")
    git(repo, "add", "lefthook.yml")
    git(repo, "commit", "-q", "-m", "add lefthook config")

    from refactory_dsh.worktree import _shared_snapshot
    before = _shared_snapshot(repo)

    found = worktree.create(repo, "HEAD", "hook_iso_001", _run(tmp_path))
    wt_path = Path(found.path)

    # The worktree has its own hooks path inside its git dir.
    wt_git_dir = Path(git(wt_path, "rev-parse", "--absolute-git-dir"))
    hooks_isolated = wt_git_dir / "hooks.isolated"
    assert hooks_isolated.is_dir()
    assert git(wt_path, "config", "--worktree", "core.hooksPath") == str(hooks_isolated)

    # git rev-parse --git-path hooks respects core.hooksPath.
    hooks_resolved = Path(git(wt_path, "rev-parse", "--git-path", "hooks"))
    if not hooks_resolved.is_absolute():
        hooks_resolved = wt_path / hooks_resolved
    assert hooks_resolved.resolve() == hooks_isolated.resolve()

    # Real lefthook writes into the isolated directory, not the shared hooks.
    result = subprocess.run(
        [str(lefthook), "install"], cwd=str(wt_path),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"lefthook install failed: {result.stderr}"
    assert (hooks_isolated / "pre-commit").is_file(), "lefthook should write to hooks.isolated"

    # Target's shared state is unchanged (after exclusions).
    after = _shared_snapshot(repo)
    assert before["hook_files"] == after["hook_files"], "target hooks should be unchanged"
    assert before["config_lines"] == after["config_lines"], "target config lines should be unchanged"

    # shared_changed returns False because only excluded lines differ.
    assert worktree.shared_changed(found) is False

    # But a real change to shared hooks is still detected.
    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    (common / "hooks").mkdir(exist_ok=True)
    (common / "hooks" / "pre-push").write_text("#!/bin/sh\n")
    assert worktree.shared_changed(found) is True


def test_standin_hook_manager_writes_to_isolated_hooks(repo, tmp_path):
    """A tool that uses git rev-parse --git-path hooks writes to the isolated dir, not shared hooks."""
    found = worktree.create(repo, "HEAD", "hook_iso_002", _run(tmp_path))
    wt_path = Path(found.path)

    # Stand-in hook manager: resolves the hooks dir the way lefthook does.
    hooks_dir = Path(git(wt_path, "rev-parse", "--git-path", "hooks"))
    if not hooks_dir.is_absolute():
        hooks_dir = wt_path / hooks_dir
    hook_path = hooks_dir / "pre-commit"
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text("#!/bin/sh\necho standin\n")

    # The hook was written to the isolated dir, not the shared hooks.
    wt_git_dir = Path(git(wt_path, "rev-parse", "--absolute-git-dir"))
    assert hook_path.resolve().is_relative_to((wt_git_dir / "hooks.isolated").resolve())

    from refactory_dsh.worktree import _shared_snapshot
    snapshot = _shared_snapshot(repo)
    assert "pre-commit" not in snapshot["hook_files"], "shared hooks must not have pre-commit"


def test_hook_isolation_skipped_when_target_has_core_bare(repo, tmp_path, capsys):
    """A target with core.bare=true is not changed; the run says why."""
    git(repo, "config", "core.bare", "true")
    found = worktree.create(repo, "HEAD", "hook_bare_001", _run(tmp_path))

    # extensions.worktreeConfig should NOT be set.
    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    result = subprocess.run(
        ["git", "config", "--file", str(common / "config"), "--get", "extensions.worktreeConfig"],
        capture_output=True, text=True)
    assert result.returncode != 0, "extensions.worktreeConfig should not be set"

    captured = capsys.readouterr()
    assert "core.bare" in captured.err


def test_hook_isolation_skipped_when_target_has_core_worktree(repo, tmp_path, capsys):
    """A target with core.worktree is not changed; the run says why."""
    git(repo, "config", "core.worktree", str(repo))
    found = worktree.create(repo, "HEAD", "hook_cwt_001", _run(tmp_path))

    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    result = subprocess.run(
        ["git", "config", "--file", str(common / "config"), "--get", "extensions.worktreeConfig"],
        capture_output=True, text=True)
    assert result.returncode != 0, "extensions.worktreeConfig should not be set"

    captured = capsys.readouterr()
    assert "core.worktree" in captured.err


def test_no_hooks_overrides_worktree_hooks_path(repo, tmp_path):
    """NO_HOOKS (-c core.hooksPath=/dev/null) overrides the worktree's core.hooksPath.

    Uses a post-commit hook (not stopped by --no-verify) so NO_HOOKS is what blocks it.
    """
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "hook_noh_001", run)
    wt_path = Path(found.path)

    wt_git_dir = Path(git(wt_path, "rev-parse", "--absolute-git-dir"))
    hooks_dir = wt_git_dir / "hooks.isolated"

    marker = tmp_path / "hook-ran"
    hook = hooks_dir / "post-commit"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)

    # commit_attempt uses NO_HOOKS: hook should NOT run.
    worktree.commit_attempt(run, {Path("nohooks.txt"): "test\n"}, "no-hooks attempt")
    assert not marker.exists(), "NO_HOOKS should prevent post-commit from running"

    # Positive control: a plain git commit in the worktree runs the hook.
    (wt_path / "control.txt").write_text("control\n")
    git(wt_path, "add", "control.txt")
    git(wt_path, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "control")
    assert marker.exists(), "positive control: post-commit hook should have run"


def test_config_lock_does_not_block_isolate_hooks_when_extension_already_set(repo, tmp_path):
    """With the extension already on, _isolate_hooks succeeds despite a held config.lock."""
    from refactory_dsh.worktree import _isolate_hooks

    git(repo, "config", "extensions.worktreeConfig", "true")

    wt_path = tmp_path / "lock-test-wt"
    git(repo, "worktree", "add", str(wt_path), "HEAD")

    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    lock = common / "config.lock"
    lock.write_text("held by test\n")
    try:
        result = _isolate_hooks(repo, wt_path)
        assert result is True, "_isolate_hooks should have succeeded"
        hooks_path = git(wt_path, "config", "--worktree", "core.hooksPath")
        assert "hooks.isolated" in hooks_path
    finally:
        lock.unlink(missing_ok=True)


# ── session-level repository guard ────────────────────────────────────────


def _guard_conftest(tests_dir: Path) -> str:
    """A conftest.py that loads the real repo_guard module."""
    return (
        f"import sys\nsys.path.insert(0, {str(tests_dir)!r})\n"
        "pytest_plugins = ('repo_guard',)\n"
    )


def test_session_guard_catches_worktree_in_tmp(tmp_path):
    """The guard fails a session whose test adds a worktree under tmp_path (scenario a)."""
    tests_dir = Path(__file__).resolve().parent
    test_repo = tmp_path / "guarded-repo-a"
    git_repo(test_repo)
    (test_repo / "conftest.py").write_text(_guard_conftest(tests_dir))

    (test_repo / "test_bad.py").write_text(
        "import subprocess\nfrom pathlib import Path\n\n"
        "def test_adds_worktree(tmp_path):\n"
        f"    repo = Path({str(test_repo)!r})\n"
        "    wt = tmp_path / 'bad-wt'\n"
        "    subprocess.run(['git', '-C', str(repo), 'worktree', 'add',\n"
        "                    str(wt), 'HEAD'], check=True, capture_output=True)\n"
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-xvs", str(test_repo / "test_bad.py")],
        capture_output=True, text=True, timeout=60, cwd=str(test_repo))
    combined = result.stdout + result.stderr
    assert result.returncode != 0, f"guard should have failed:\n{combined}"
    assert "test_adds_worktree PASSED" in combined
    assert "FAIL: worktree added by session:" in combined
    assert "bad-wt" in combined


@pytest.mark.parametrize("variant", ["hook", "config"])
def test_session_guard_catches_hook_or_config_change(tmp_path, variant):
    """The guard fails when a test rewrites a hook (hook) or a config line (config) (scenario b)."""
    tests_dir = Path(__file__).resolve().parent
    test_repo = tmp_path / "guarded-repo-b"
    git_repo(test_repo)
    common = Path(git(test_repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    (common / "hooks").mkdir(exist_ok=True)
    (test_repo / "conftest.py").write_text(_guard_conftest(tests_dir))

    if variant == "hook":
        (test_repo / "test_bad.py").write_text(
            "from pathlib import Path\n\n"
            "def test_changes_hook():\n"
            f"    hook = Path({str(common / 'hooks' / 'pre-commit')!r})\n"
            "    hook.write_text('#!/bin/sh\\necho bad\\n')\n"
        )
    else:
        (test_repo / "test_bad.py").write_text(
            "import subprocess\n\n"
            "def test_changes_config():\n"
            f"    subprocess.run(['git', '-C', {str(test_repo)!r},\n"
            "                     'config', 'user.name', 'BadActor'], check=True)\n"
        )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-xvs", str(test_repo / "test_bad.py")],
        capture_output=True, text=True, timeout=60, cwd=str(test_repo))
    combined = result.stdout + result.stderr
    assert result.returncode != 0, f"guard should have failed:\n{combined}"
    if variant == "hook":
        assert "test_changes_hook PASSED" in combined
        assert "FAIL: hook added: pre-commit" in combined
    else:
        assert "test_changes_config PASSED" in combined
        assert "FAIL: config line added: user.name=BadActor" in combined


def test_session_guard_catches_subprocess_worktree_in_refactory_home(tmp_path):
    """The guard fails when a subprocess creates a worktree under REFACTORY_HOME set by the guard (scenario c)."""
    tests_dir = Path(__file__).resolve().parent
    test_repo = tmp_path / "guarded-repo-c"
    git_repo(test_repo)
    (test_repo / "conftest.py").write_text(_guard_conftest(tests_dir))

    script = tmp_path / "create_worktree.py"
    script.write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "from refactory_dsh import worktree\n"
        f"repo = Path({str(test_repo)!r})\n"
        "run_dir = Path(sys.argv[1])\n"
        "run_dir.mkdir(parents=True, exist_ok=True)\n"
        "worktree.create(repo, 'HEAD', 'sub_bad_01', run_dir)\n"
    )

    (test_repo / "test_bad.py").write_text(
        "import subprocess, sys\nfrom pathlib import Path\n\n"
        "def test_subprocess_creates_worktree(tmp_path):\n"
        f"    subprocess.run([sys.executable, {str(script)!r},\n"
        "                     str(tmp_path / 'run')], check=True)\n"
    )

    # Isolate TMPDIR and REFACTORY_HOME so only the guard's own override puts
    # the worktree into session space.
    sub_tmpdir = tmp_path / "sub-tmp"
    sub_tmpdir.mkdir()
    outer_home = tmp_path / "outer-home"
    outer_home.mkdir()

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-xvs", str(test_repo / "test_bad.py")],
        capture_output=True, text=True, timeout=60, cwd=str(test_repo),
        env={**os.environ, "TMPDIR": str(sub_tmpdir), "REFACTORY_HOME": str(outer_home)})
    combined = result.stdout + result.stderr
    assert result.returncode != 0, f"guard should have failed:\n{combined}"
    assert "test_subprocess_creates_worktree PASSED" in combined
    assert "FAIL: worktree added by session:" in combined
    assert "sub_bad_01" in combined


def test_session_guard_reports_external_worktree_without_failing(tmp_path):
    """The guard reports but does not fail when a worktree is added outside the session's temp space (scenario d)."""
    tests_dir = Path(__file__).resolve().parent
    test_repo = tmp_path / "guarded-repo-d"
    git_repo(test_repo)
    (test_repo / "conftest.py").write_text(_guard_conftest(tests_dir))

    # This path is OUTSIDE the subprocess's temp space.
    external_wt = tmp_path / "external-wt"

    sub_tmpdir = tmp_path / "sub-tmp"
    sub_tmpdir.mkdir()

    (test_repo / "test_external.py").write_text(
        "import subprocess\nfrom pathlib import Path\n\n"
        "def test_adds_external_worktree():\n"
        f"    repo = Path({str(test_repo)!r})\n"
        f"    wt = Path({str(external_wt)!r})\n"
        "    subprocess.run(['git', '-C', str(repo), 'worktree', 'add',\n"
        "                    str(wt), 'HEAD'], check=True, capture_output=True)\n"
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-xvs", str(test_repo / "test_external.py")],
        capture_output=True, text=True, timeout=60, cwd=str(test_repo),
        env={**os.environ, "TMPDIR": str(sub_tmpdir)})
    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"guard should have passed:\n{combined}"
    assert str(external_wt) in combined, f"should report the external worktree path:\n{combined}"


def test_shared_state_digest_matches_old_formula(repo, tmp_path):
    """_shared_state returns the same digest the old code returned for the same state."""
    import hashlib as _hashlib
    from refactory_dsh.worktree import _shared_state

    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    hooks_dir = common / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    (hooks_dir / "pre-commit").write_text("#!/bin/sh\necho hook\n")
    (hooks_dir / "post-commit").write_text("#!/bin/sh\necho post\n")

    # Old formula: config text, then name + "\0" + raw bytes per hook, sorted.
    config_text = git(repo, "config", "--file", str(common / "config"), "--list")
    settings = config_text.splitlines()
    config_lines = [
        line for line in settings
        if not line.startswith("submodule.")
        and not line.lower().startswith("extensions.worktreeconfig=")
    ]
    expected = _hashlib.sha256()
    expected.update("\n".join(config_lines).encode())
    for item in sorted(hooks_dir.glob("*")):
        if item.is_file():
            expected.update(item.name.encode() + b"\0" + item.read_bytes())

    assert _shared_state(repo) == expected.hexdigest()


def test_session_guard_resolves_symlinked_tmp_root(tmp_path):
    """The guard catches a worktree under a tmp root reached through a symlink."""
    tests_dir = Path(__file__).resolve().parent
    test_repo = tmp_path / "guarded-repo-sym"
    git_repo(test_repo)
    (test_repo / "conftest.py").write_text(_guard_conftest(tests_dir))

    real_tmp = tmp_path / "real-tmp"
    real_tmp.mkdir()
    linked_tmp = tmp_path / "linked-tmp"
    linked_tmp.symlink_to(real_tmp)

    # The bad test creates its worktree under tempfile.mkdtemp() which uses the
    # symlinked TMPDIR, while git reports the resolved path.
    (test_repo / "test_bad.py").write_text(
        "import subprocess, tempfile\nfrom pathlib import Path\n\n"
        "def test_adds_worktree(tmp_path):\n"
        f"    repo = Path({str(test_repo)!r})\n"
        "    wt = Path(tempfile.mkdtemp(prefix='bad-wt-'))\n"
        "    subprocess.run(['git', '-C', str(repo), 'worktree', 'add',\n"
        "                    str(wt), 'HEAD'], check=True, capture_output=True)\n"
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-xvs", str(test_repo / "test_bad.py")],
        capture_output=True, text=True, timeout=60, cwd=str(test_repo),
        env={**os.environ, "TMPDIR": str(linked_tmp)})
    combined = result.stdout + result.stderr
    assert result.returncode != 0, f"guard should have failed:\n{combined}"
    assert "test_adds_worktree PASSED" in combined
    assert "FAIL: worktree added by session:" in combined
    assert "bad-wt" in combined


def test_session_guard_reports_inactive_on_non_git_dir(tmp_path):
    """The pytest guard reports inactive when run on a directory that is not a git repo."""
    tests_dir = Path(__file__).resolve().parent
    test_dir = tmp_path / "not-a-repo"
    test_dir.mkdir()
    (test_dir / "conftest.py").write_text(_guard_conftest(tests_dir))
    (test_dir / "test_ok.py").write_text(
        "def test_passes():\n"
        "    assert True\n"
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-xvs", str(test_dir / "test_ok.py"),
         "--rootdir", str(test_dir)],
        capture_output=True, text=True, timeout=60, cwd=str(test_dir))
    combined = result.stdout + result.stderr
    assert "repository guard inactive:" in combined, (
        f"should report guard inactive:\n{combined}")


@pytest.mark.toolchain("node")
def test_vitest_guard_catches_mismatch(tmp_path):
    """The vitest globalSetup fails a vitest run that changes the repository's hooks.

    Points globalSetup at the real vitest.global-setup.ts with the test repository
    as root, so this test exercises the real guard file.  Removing the teardown throw
    from vitest.global-setup.ts makes this test fail.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    vitest_bin = repo_root / "node_modules" / ".bin" / "vitest"
    real_setup = repo_root / "vitest.global-setup.ts"

    if not vitest_bin.is_file():
        pytest.skip("vitest not found at node_modules/.bin/vitest")

    test_repo = tmp_path / "vitest-guarded"
    git_repo(test_repo)
    common = Path(git(test_repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    (common / "hooks").mkdir(exist_ok=True)

    # Symlink node_modules so vitest can resolve its dependencies.
    (tmp_path / "node_modules").symlink_to(repo_root / "node_modules")

    vitest_cfg = tmp_path / "vitest.config.mjs"
    vitest_cfg.write_text(
        "import { defineConfig } from 'vitest/config'\n"
        "export default defineConfig({\n"
        "  test: {\n"
        "    include: ['**/*.spec.mjs'],\n"
        f"    root: {json.dumps(str(test_repo))},\n"
        f"    globalSetup: [{json.dumps(str(real_setup))}],\n"
        "  },\n"
        "})\n"
    )

    bad_spec = test_repo / "bad.spec.mjs"
    bad_spec.write_text(
        "import { writeFileSync } from 'node:fs'\n"
        "import { test, expect } from 'vitest'\n"
        f"test('changes hooks', () => {{\n"
        f"  writeFileSync({json.dumps(str(common / 'hooks' / 'pre-commit'))}, "
        "'#!/bin/sh\\necho bad\\n')\n"
        "  expect(true).toBe(true)\n"
        "})\n"
    )

    result = subprocess.run(
        [str(vitest_bin), "run", "--config", str(vitest_cfg)],
        capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        f"vitest should have exited non-zero:\n{combined}")
    assert "repo-guard" in combined.lower(), (
        f"output should mention repo-guard:\n{combined}")


@pytest.mark.toolchain("node")
def test_vitest_guard_reports_inactive_when_python_missing(tmp_path):
    """The vitest guard prints inactive when the venv's python does not exist."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    vitest_bin = repo_root / "node_modules" / ".bin" / "vitest"
    real_setup = repo_root / "vitest.global-setup.ts"

    if not vitest_bin.is_file():
        pytest.skip("vitest not found at node_modules/.bin/vitest")

    test_repo = tmp_path / "vitest-no-python"
    git_repo(test_repo)

    # Symlink node_modules so vitest can resolve its dependencies.
    (tmp_path / "node_modules").symlink_to(repo_root / "node_modules")

    # Derive a globalSetup from the real vitest.global-setup.ts by replacing
    # the python path with a nonexistent one.
    real_text = real_setup.read_text()
    patched_text = real_text.replace(
        "const python = resolve(dir, '.venv', 'bin', 'python')",
        "const python = '/nonexistent/venv/bin/python'",
    )
    assert patched_text != real_text, "patch target not found in vitest.global-setup.ts"
    derived_setup = tmp_path / "derived-global-setup.ts"
    derived_setup.write_text(patched_text)

    vitest_cfg = tmp_path / "vitest.config.mjs"
    vitest_cfg.write_text(
        "import { defineConfig } from 'vitest/config'\n"
        "export default defineConfig({\n"
        "  test: {\n"
        "    include: ['**/*.spec.mjs'],\n"
        f"    root: {json.dumps(str(test_repo))},\n"
        f"    globalSetup: [{json.dumps(str(derived_setup))}],\n"
        "  },\n"
        "})\n"
    )

    (test_repo / "ok.spec.mjs").write_text(
        "import { test, expect } from 'vitest'\n"
        "test('passes', () => { expect(true).toBe(true) })\n"
    )

    result = subprocess.run(
        [str(vitest_bin), "run", "--config", str(vitest_cfg)],
        capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"vitest should pass when guard is inactive:\n{combined}")
    assert "repository guard inactive:" in combined, (
        f"should report guard inactive:\n{combined}")


@pytest.mark.toolchain("node")
def test_vitest_guard_reports_inactive_on_non_git_dir(tmp_path):
    """The vitest guard reports inactive when the repo is not a git checkout."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    vitest_bin = repo_root / "node_modules" / ".bin" / "vitest"
    real_setup = repo_root / "vitest.global-setup.ts"

    if not vitest_bin.is_file():
        pytest.skip("vitest not found at node_modules/.bin/vitest")

    test_dir = tmp_path / "not-a-repo"
    test_dir.mkdir()

    # Symlink node_modules so vitest can resolve its dependencies.
    (tmp_path / "node_modules").symlink_to(repo_root / "node_modules")

    vitest_cfg = tmp_path / "vitest.config.mjs"
    vitest_cfg.write_text(
        "import { defineConfig } from 'vitest/config'\n"
        "export default defineConfig({\n"
        "  test: {\n"
        "    include: ['**/*.spec.mjs'],\n"
        f"    root: {json.dumps(str(test_dir))},\n"
        f"    globalSetup: [{json.dumps(str(real_setup))}],\n"
        "  },\n"
        "})\n"
    )

    (test_dir / "ok.spec.mjs").write_text(
        "import { test, expect } from 'vitest'\n"
        "test('passes', () => { expect(true).toBe(true) })\n"
    )

    result = subprocess.run(
        [str(vitest_bin), "run", "--config", str(vitest_cfg)],
        capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"vitest should pass when guard is inactive:\n{combined}")
    assert "repository guard inactive:" in combined, (
        f"should report guard inactive:\n{combined}")


# ── literal pathspec protection ────────────────────────────────────────────
def test_delete_bracket_path_does_not_glob_expand(repo, tmp_path):
    """Deleting app/[id]/page.ts must not also delete app/i/page.ts.

    Without GIT_LITERAL_PATHSPECS, git rm treats [id] as a character-class
    pathspec matching 'i', so the sibling is removed.
    """
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "bracket01234", run)
    path = Path(found.path)

    # Set up both files through commit_attempt so the worktree record stays current
    worktree.commit_attempt(run, {
        Path("app/[id]/page.ts"): "bracket\n",
        Path("app/i/page.ts"): "sibling\n",
    }, "setup files")

    # Now delete only the bracket path
    worktree.commit_attempt(run, {}, "delete bracket", (Path("app/[id]/page.ts"),))

    tree = git(path, "ls-tree", "--name-only", "-r", "HEAD").splitlines()
    assert "app/i/page.ts" in tree, "sibling must be kept"
    assert "app/[id]/page.ts" not in tree, "target must be deleted"


def test_colon_prefixed_path_does_not_expand_pathspec(repo, tmp_path):
    """An emitted path starting with ':' must be committed literally.

    Without GIT_LITERAL_PATHSPECS, git add treats ':(glob)**' as a pathspec
    magic that matches everything including ignored files.
    """
    run = _run(tmp_path)
    found = worktree.create(repo, "HEAD", "colonpath1234", run)
    path = Path(found.path)

    # Commit .gitignore through commit_attempt so the worktree record stays current
    worktree.commit_attempt(run, {Path(".gitignore"): ".env\n"}, "setup gitignore")

    # Write .env to disk — it is ignored by .gitignore so git clean won't touch it
    (path / ".env").write_text("SECRET=x\n")

    # Now add the colon-prefixed file
    worktree.commit_attempt(run, {Path(":(glob)**"): "content\n"}, "colon path")

    tree = git(path, "ls-tree", "--name-only", "-r", "HEAD").splitlines()
    assert ":(glob)**" in tree, "the literal file must be committed"
    assert ".env" not in tree, "the ignored file must NOT be committed"


# ── tracking config from ordinary branch creation, and naming what changed ────

def _with_origin(repo: Path) -> None:
    """Give ``repo`` an ``origin`` remote with a remote-tracking ``origin/main``."""
    git(repo, "remote", "add", "origin", "https://example.invalid/repo.git")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")


def test_a_tracking_branch_made_in_the_person_s_checkout_is_not_a_shared_change(repo, tmp_path):
    """`git worktree add -b <b> origin/main` writes branch.<b>.remote/merge: not the run's doing."""
    _with_origin(repo)
    found = worktree.create(repo, "HEAD", "trk0000aaaa", _run(tmp_path))
    git(repo, "worktree", "add", "-q", "-b", "fix/early-check-probe",
        str(tmp_path / "session"), "origin/main")
    assert "branch.fix/early-check-probe.remote=origin" in git(repo, "config", "--list")
    git(repo, "branch", "--track", "local-tracker", "main")  # names the remote "."
    assert worktree.shared_change(found) is None


@pytest.mark.parametrize("change, named", [
    (("config", "branch.main.pushRemote", "origin"), "config branch.main.pushremote added"),
    (("config", "branch.main.rebase", "true"), "config branch.main.rebase added"),
    (("config", "core.hooksPath", "/elsewhere"), "config core.hookspath added"),
    (("config", "alias.ci", "commit"), "config alias.ci added"),
    (("config", "remote.origin.url", "https://example.invalid/other.git"),
     "config remote.origin.url changed"),
])
def test_every_other_shared_config_change_is_still_caught_and_named(repo, tmp_path, change, named):
    _with_origin(repo)
    git(repo, "branch", "--set-upstream-to", "origin/main", "main")  # main's own tracking is left out
    found = worktree.create(repo, "HEAD", "trk1111bbbb", _run(tmp_path))
    git(repo, *change)
    assert worktree.shared_change(found) == named
    assert worktree.shared_change(found) is None, "each change is reported once"


def test_tracking_a_remote_added_meanwhile_is_caught(repo, tmp_path):
    _with_origin(repo)
    found = worktree.create(repo, "HEAD", "trk2222cccc", _run(tmp_path))
    git(repo, "remote", "add", "upstream", "https://example.invalid/fork.git")
    git(repo, "config", "branch.other.remote", "upstream")
    git(repo, "config", "branch.other.merge", "refs/heads/main")
    change = worktree.shared_change(found)
    assert change is not None and "config remote.upstream.url added" in change
    assert "config branch.other.remote added" in change, "the remote is judged as it was before"


def test_tracking_an_undefined_remote_is_caught(repo, tmp_path):
    found = worktree.create(repo, "HEAD", "trk3333dddd", _run(tmp_path))
    git(repo, "config", "branch.other.remote", "nowhere")
    git(repo, "config", "branch.other.merge", "refs/heads/main")
    assert worktree.shared_change(found) == (
        "config branch.other.merge added; config branch.other.remote added")


def test_tracking_config_for_the_run_s_own_branch_is_caught(repo, tmp_path):
    _with_origin(repo)
    found = worktree.create(repo, "HEAD", "trk4444eeee", _run(tmp_path))
    git(repo, "config", f"branch.{found.branch}.remote", "origin")
    assert worktree.shared_change(found) == f"config branch.{found.branch}.remote added"


def test_a_shared_hook_change_is_named_by_file(repo, tmp_path):
    found = worktree.create(repo, "HEAD", "trk5555ffff", _run(tmp_path))
    hooks = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")) / "hooks"
    hooks.mkdir(exist_ok=True)
    (hooks / "pre-commit").write_text("#!/bin/sh\n")
    assert worktree.shared_change(found) == "hook pre-commit added"
    (hooks / "pre-commit").write_text("#!/bin/sh\nexit 1\n")
    assert worktree.shared_change(found) == "hook pre-commit changed"


def test_a_record_from_before_the_detail_still_works(repo, tmp_path):
    """A live run's record holds only the old all-lines digest: no spurious report on upgrade."""
    _with_origin(repo)
    git(repo, "config", "branch.main.remote", "origin")
    found = worktree.create(repo, "HEAD", "trk6666aaaa", _run(tmp_path))
    record_path = worktree._record_path(found)
    record = json.loads(record_path.read_text())
    legacy = worktree._shared_state(repo)
    record = {key: value for key, value in record.items()
              if key not in ("shared_config", "shared_hooks")}
    record_path.write_text(json.dumps({**record, "shared": legacy}))
    assert worktree.shared_change(found) is None
    git(repo, "config", "core.hooksPath", "/elsewhere")
    assert worktree.shared_change(found) == "config core.hookspath added"


def test_an_old_record_s_change_is_reported_without_detail(repo, tmp_path):
    found = worktree.create(repo, "HEAD", "trk7777bbbb", _run(tmp_path))
    record_path = worktree._record_path(found)
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps({key: value for key, value in record.items()
                                       if key not in ("shared_config", "shared_hooks")}))
    git(repo, "config", "core.hooksPath", "/elsewhere")
    assert worktree.shared_change(found) == ""
    assert worktree.shared_changed(found) is False


def _with_fork(repo: Path) -> None:
    git(repo, "remote", "add", "fork", "https://example.invalid/fork.git")
    git(repo, "update-ref", "refs/remotes/fork/main", "HEAD")


def test_retargeting_the_person_s_tracked_branch_is_caught(repo, tmp_path):
    """Review of #644: an existing branch's tracking moved to another remote is a change."""
    _with_origin(repo)
    _with_fork(repo)
    git(repo, "branch", "--set-upstream-to", "origin/main", "main")
    found = worktree.create(repo, "HEAD", "trk8888cccc", _run(tmp_path))
    git(repo, "config", "branch.main.remote", "fork")
    git(repo, "config", "branch.main.merge", "refs/heads/evil")
    assert worktree.shared_change(found) == (
        "config branch.main.merge changed; config branch.main.remote changed")


def test_giving_an_existing_branch_an_upstream_is_caught(repo, tmp_path):
    """Review of #644: `--set-upstream-to` on a branch that was there is not branch creation."""
    _with_origin(repo)
    _with_fork(repo)
    git(repo, "branch", "feature")
    found = worktree.create(repo, "HEAD", "trk9999dddd", _run(tmp_path))
    git(repo, "branch", "--set-upstream-to", "fork/main", "feature")
    assert worktree.shared_change(found) == (
        "config branch.feature.merge added; config branch.feature.remote added")


def test_a_branch_made_in_one_step_and_given_an_upstream_in_a_later_one_is_caught(repo, tmp_path):
    _with_origin(repo)
    found = worktree.create(repo, "HEAD", "trkaaaaeeee", _run(tmp_path))
    git(repo, "branch", "later")
    assert worktree.shared_change(found) is None
    git(repo, "branch", "--set-upstream-to", "origin/main", "later")
    assert worktree.shared_change(found) == (
        "config branch.later.merge added; config branch.later.remote added")


def test_a_new_branch_s_other_settings_are_caught(repo, tmp_path):
    """Only a new branch's remote and merge are left out; its pushRemote is not."""
    _with_origin(repo)
    _with_fork(repo)
    found = worktree.create(repo, "HEAD", "trkbbbbffff", _run(tmp_path))
    git(repo, "worktree", "add", "-q", "-b", "newb", str(tmp_path / "session"), "origin/main")
    git(repo, "config", "branch.newb.pushRemote", "fork")
    assert worktree.shared_change(found) == "config branch.newb.pushremote added"


def test_retargeting_config_left_by_a_deleted_branch_is_caught(repo, tmp_path):
    """A branch with config but no ref is not new: its tracking lines are guarded."""
    _with_origin(repo)
    _with_fork(repo)
    git(repo, "config", "branch.ghost.remote", "origin")
    git(repo, "config", "branch.ghost.merge", "refs/heads/main")
    found = worktree.create(repo, "HEAD", "trkcccc0000", _run(tmp_path))
    git(repo, "config", "branch.ghost.remote", "fork")
    assert worktree.shared_change(found) == "config branch.ghost.remote changed"


def test_deleting_a_branch_and_making_it_again_with_an_upstream_is_caught(repo, tmp_path):
    """Re-review of #644: a branch once seen is never new again."""
    _with_origin(repo)
    _with_fork(repo)
    git(repo, "branch", "feature")
    found = worktree.create(repo, "HEAD", "trkdddd1111", _run(tmp_path))
    git(repo, "branch", "-D", "feature")
    assert worktree.shared_change(found) is None
    git(repo, "branch", "--track", "feature", "fork/main")
    assert worktree.shared_change(found) == (
        "config branch.feature.merge added; config branch.feature.remote added")


def test_a_remote_is_matched_by_name_not_prefix(repo, tmp_path):
    """Re-review of #644: an existing remote `a.b` does not make remote `a` defined."""
    git(repo, "remote", "add", "a.b", "https://example.invalid/ab.git")
    found = worktree.create(repo, "HEAD", "trkeeee2222", _run(tmp_path))
    git(repo, "branch", "newb")
    git(repo, "config", "branch.newb.remote", "a")
    git(repo, "config", "branch.newb.merge", "refs/heads/main")
    assert worktree.shared_change(found) == (
        "config branch.newb.merge added; config branch.newb.remote added")


def test_a_branch_named_like_a_tag_is_known_by_its_name(repo, tmp_path):
    """Re-review of #644: with a tag `x`, `refname:short` would list branch `x` as `heads/x`."""
    _with_origin(repo)
    git(repo, "tag", "x")
    git(repo, "branch", "x")
    found = worktree.create(repo, "HEAD", "trkffff3333", _run(tmp_path))
    git(repo, "branch", "--set-upstream-to", "origin/main", "x")
    assert worktree.shared_change(found) == "config branch.x.merge added; config branch.x.remote added"


# ── the checkout's config.worktree and included files (review of #644) ────────

def test_the_checkout_s_worktree_config_is_shared_state(repo, tmp_path):
    """`_isolate_hooks` turns on worktreeConfig, so the checkout's config.worktree takes effect."""
    found = worktree.create(repo, "HEAD", "wtc0000aaaa", _run(tmp_path))
    assert git(repo, "config", "extensions.worktreeConfig").strip() == "true"
    assert worktree.shared_change(found) is None, "the run's own isolated hooks path is not a change"
    git(repo, "config", "--worktree", "core.hooksPath", "/evil")
    assert git(repo, "config", "core.hooksPath").strip() == "/evil"
    assert worktree.shared_change(found) == "config worktree:core.hookspath added"
    assert worktree.shared_change(found) is None, "each change is reported once"
    git(repo, "config", "--worktree", "core.hooksPath", "/eviler")
    assert worktree.shared_change(found) == "config worktree:core.hookspath changed"


def test_a_worktree_config_written_before_the_extension_is_on_is_seen(repo, tmp_path):
    """A file git does not read yet is recorded too: turning the extension on would load it."""
    found = worktree.create(repo, "HEAD", "wtc1111bbbb", _run(tmp_path))
    git_dir = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-dir").strip())
    (git_dir / "config.worktree").write_text("[core]\n\thooksPath = /evil\n")
    assert worktree.shared_change(found) == "config worktree:core.hookspath added"


def test_a_person_s_linked_worktree_checkout_has_its_own_worktree_config(repo, tmp_path):
    """When the target is itself a linked worktree, its own config.worktree is the one read."""
    git(repo, "worktree", "add", "-q", "--no-track", "-b", "person", str(tmp_path / "checkout"), "HEAD")
    checkout = tmp_path / "checkout"
    found = worktree.create(checkout, "HEAD", "wtc2222cccc", _run(tmp_path))
    git(checkout, "config", "--worktree", "core.hooksPath", "/evil")
    assert worktree.shared_change(found) == "config worktree:core.hookspath added"


def test_a_change_inside_an_included_file_is_caught(repo, tmp_path):
    included = tmp_path / "extra.gitconfig"
    included.write_text("[alias]\n\tci = commit\n")
    git(repo, "config", "include.path", str(included))
    found = worktree.create(repo, "HEAD", "inc0000aaaa", _run(tmp_path))
    assert worktree.shared_change(found) is None
    included.write_text("[alias]\n\tci = commit\n[core]\n\thooksPath = /evil\n")
    assert worktree.shared_change(found) == f"config include:{included} changed"
    assert worktree.shared_change(found) is None, "each change is reported once"


def test_includes_are_followed_relative_conditional_nested_and_from_worktree_config(repo, tmp_path):
    """A relative path, an includeIf whatever its condition, a nested include, a missing file."""
    git_dir = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-dir").strip())
    (git_dir / "rel.inc").write_text("[include]\n\tpath = nested.inc\n")
    (git_dir / "nested.inc").write_text("[alias]\n\tst = status\n")
    (tmp_path / "cond.inc").write_text("[alias]\n\tco = checkout\n")
    git(repo, "config", "include.path", "rel.inc")
    git(repo, "config", "includeIf.gitdir:/nowhere/.path", str(tmp_path / "cond.inc"))
    found = worktree.create(repo, "HEAD", "inc1111bbbb", _run(tmp_path))
    git(repo, "config", "--worktree", "include.path", str(tmp_path / "wt.inc"))
    assert worktree.shared_change(found) == (
        f"config include:{tmp_path / 'wt.inc'} added; config worktree:include.path added")
    (tmp_path / "wt.inc").write_text("[core]\n\thooksPath = /evil\n")
    assert worktree.shared_change(found) == f"config include:{tmp_path / 'wt.inc'} changed"
    (git_dir / "nested.inc").write_text("[core]\n\thooksPath = /evil\n")
    assert worktree.shared_change(found) == f"config include:{git_dir / 'nested.inc'} changed"
    (tmp_path / "cond.inc").write_text("[core]\n\thooksPath = /evil\n")
    assert worktree.shared_change(found) == f"config include:{tmp_path / 'cond.inc'} changed"


def test_an_include_cycle_ends_after_reading_each_file_once(repo, tmp_path, monkeypatch):
    """An inactive includeIf may cycle (git never opens it); each file is listed once."""
    (tmp_path / "a.inc").write_text(f"[include]\n\tpath = {tmp_path / 'b.inc'}\n")
    (tmp_path / "b.inc").write_text(f"[include]\n\tpath = {tmp_path / 'a.inc'}\n")
    git(repo, "config", "includeIf.gitdir:/elsewhere/.path", str(tmp_path / "a.inc"))
    calls: list[tuple[str, ...]] = []
    real = worktree.git

    def counted(where, *args, **kwargs):
        calls.append(args)
        return real(where, *args, **kwargs)

    monkeypatch.setattr(worktree, "git", counted)
    assert set(worktree._shared_snapshot(repo)["included"]) == {
        str(tmp_path / "a.inc"), str(tmp_path / "b.inc")}
    assert len(calls) == 4, "rev-parse, the shared config, a.inc and b.inc"


def test_ordinary_work_with_includes_and_worktree_config_is_not_a_change(repo, tmp_path):
    """Branches, commits and a tracking branch in the checkout trip nothing new."""
    _with_origin(repo)
    (tmp_path / "extra.gitconfig").write_text("[alias]\n\tci = commit\n")
    git(repo, "config", "include.path", str(tmp_path / "extra.gitconfig"))
    git(repo, "config", "extensions.worktreeConfig", "true")
    git(repo, "config", "--worktree", "user.name", "Person")
    found = worktree.create(repo, "HEAD", "inc2222cccc", _run(tmp_path))
    git(repo, "worktree", "add", "-q", "-b", "fix/thing", str(tmp_path / "session"), "origin/main")
    git(repo, "commit", "-q", "--allow-empty", "-m", "work")
    worktree.commit_attempt(Path(found.run), {Path("f.txt"): "x\n"}, "attempt")
    assert worktree.shared_change(found) is None


def test_an_old_record_still_sees_a_worktree_config_change(repo, tmp_path):
    """A record from before the detail compares only the digest, which covers config.worktree."""
    found = worktree.create(repo, "HEAD", "wtc3333dddd", _run(tmp_path))
    record_path = worktree._record_path(found)
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps({key: value for key, value in record.items()
                                       if key not in ("shared_config", "shared_hooks")}))
    git(repo, "config", "--worktree", "core.hooksPath", "/evil")
    assert worktree.shared_change(found) == ""


def test_a_record_from_644_takes_worktree_config_and_includes_as_its_baseline(repo, tmp_path):
    """A live run's record from before they were read is not failed on upgrade, only after."""
    (tmp_path / "extra.gitconfig").write_text("[alias]\n\tci = commit\n")
    git(repo, "config", "include.path", str(tmp_path / "extra.gitconfig"))
    found = worktree.create(repo, "HEAD", "upg0000aaaa", _run(tmp_path))
    git(repo, "config", "--worktree", "core.sparseCheckout", "true")  # as `sparse-checkout init`
    record_path = worktree._record_path(found)
    snapshot = worktree._shared_snapshot(repo)
    legacy = worktree._shared_digest({**snapshot, "worktree_lines": [], "included": {}})
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps({
        **{key: value for key, value in record.items() if key != "shared_reads"},
        "shared": legacy,
        "shared_config": {key: value for key, value in record["shared_config"].items()
                          if not key.startswith(("worktree:", "include:"))}}))
    assert worktree.shared_change(found) is None
    (tmp_path / "extra.gitconfig").write_text("[core]\n\thooksPath = /evil\n")
    assert worktree.shared_change(found) == f"config include:{tmp_path / 'extra.gitconfig'} changed"


def test_a_digest_only_record_takes_worktree_config_as_its_baseline(repo, tmp_path):
    found = worktree.create(repo, "HEAD", "upg1111bbbb", _run(tmp_path))
    git(repo, "config", "--worktree", "core.sparseCheckout", "true")
    record_path = worktree._record_path(found)
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps({key: value for key, value in record.items()
                                       if key not in ("shared_config", "shared_hooks", "shared_reads")}))
    assert worktree.shared_change(found) is None
    git(repo, "config", "--worktree", "core.hooksPath", "/evil")
    assert worktree.shared_change(found) == "config worktree:core.hookspath added"


def test_an_include_whose_condition_holds_an_equals_sign_is_followed(repo, tmp_path):
    (tmp_path / "eq.inc").write_text("[alias]\n\tst = status\n")
    git(repo, "config", "includeIf.onbranch:a=b.path", str(tmp_path / "eq.inc"))
    found = worktree.create(repo, "HEAD", "inc3333dddd", _run(tmp_path))
    (tmp_path / "eq.inc").write_text("[core]\n\thooksPath = /evil\n")
    assert worktree.shared_change(found) == f"config include:{tmp_path / 'eq.inc'} changed"


def test_an_include_git_does_not_read_here_may_be_unparsable(repo, tmp_path):
    """An includeIf for another checkout is never opened by git, so it need not be config."""
    (tmp_path / "other.inc").write_text("not [ config\n")
    git(repo, "config", "includeIf.gitdir:/elsewhere/.path", str(tmp_path / "other.inc"))
    found = worktree.create(repo, "HEAD", "inc4444eeee", _run(tmp_path))
    assert worktree.shared_change(found) is None
    (tmp_path / "other.inc").write_text("still not [ config\n")
    assert worktree.shared_change(found) == f"config include:{tmp_path / 'other.inc'} changed"


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="root reads any file")
def test_an_include_git_does_not_read_here_may_be_unreadable(repo, tmp_path):
    (tmp_path / "locked.inc").write_text("[alias]\n\tst = status\n")
    (tmp_path / "locked.inc").chmod(0)
    git(repo, "config", "includeIf.gitdir:/elsewhere/.path", str(tmp_path / "locked.inc"))
    try:
        assert worktree._shared_snapshot(repo)["included"] == {str(tmp_path / "locked.inc"): "unreadable"}
    finally:
        (tmp_path / "locked.inc").chmod(0o600)


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="root reads any file")
def test_an_include_git_does_not_read_here_may_sit_in_a_locked_folder(repo, tmp_path):
    (tmp_path / "locked").mkdir()
    (tmp_path / "locked" / "cfg").write_text("[alias]\n\tst = status\n")
    git(repo, "config", "includeIf.gitdir:/elsewhere/.path", str(tmp_path / "locked" / "cfg"))
    (tmp_path / "locked").chmod(0)
    try:
        assert worktree._shared_snapshot(repo)["included"] == {str(tmp_path / "locked" / "cfg"): "missing"}
    finally:
        (tmp_path / "locked").chmod(0o700)


def test_the_snapshot_spawns_git_twice_for_a_plain_target(repo, monkeypatch):
    """One rev-parse and one listing of the shared config: it runs after every agent step."""
    calls: list[tuple[str, ...]] = []
    real = worktree.git

    def counted(where, *args, **kwargs):
        calls.append(args)
        return real(where, *args, **kwargs)

    monkeypatch.setattr(worktree, "git", counted)
    worktree._shared_snapshot(repo)
    assert len(calls) == 2, calls


def test_the_config_lines_are_what_git_config_list_prints(repo):
    """Read with -z, they still match `--list` line for line, so live digests do not move."""
    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    with (common / "config").open("a") as config:
        config.write('[alias]\n\tbare\n\tempty =\n\tmulti = "one\\ntwo"\n')
    listed = git(repo, "config", "--file", str(common / "config"), "--list").splitlines()
    assert "alias.bare" in listed and "alias.empty=" in listed and "two" in listed
    assert worktree._shared_snapshot(repo)["config_lines"] == [
        line for line in listed if not line.lower().startswith("extensions.worktreeconfig=")]


def test_an_include_path_is_normalised_so_a_dot_dot_loop_ends(repo, tmp_path):
    """Review of #646: `sub/../loop` including itself made a new name at every level."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "loop").write_text("[include]\n\tpath = sub/../loop\n")
    git(repo, "config", "includeIf.gitdir:/elsewhere/.path", str(tmp_path / "sub" / ".." / "loop"))
    assert worktree._shared_snapshot(repo)["included"] == {
        str(tmp_path / "loop"): hashlib.sha256((tmp_path / "loop").read_bytes()).hexdigest()}


def test_includes_are_followed_no_deeper_than_git_follows_them(repo, tmp_path):
    for level in range(1, 13):
        (tmp_path / f"f{level}").write_text(f"[include]\n\tpath = f{level + 1}\n")
    git(repo, "config", "includeIf.gitdir:/elsewhere/.path", str(tmp_path / "f1"))
    assert sorted(worktree._shared_snapshot(repo)["included"]) == sorted(
        str(tmp_path / f"f{level}") for level in range(1, 11))


# ── another run's setup enabling worktreeConfig (live, scale-first build) ─────

def test_another_run_enabling_worktree_config_does_not_trip_this_run(repo, tmp_path):
    """Run B's setup turns on extensions.worktreeConfig; run A's next step is not failed."""
    run_a = worktree.create(repo, "HEAD", "two0000aaaa", _run(tmp_path, "a"))
    git(repo, "config", "--unset", "extensions.worktreeConfig")  # as before any run's setup
    assert worktree.shared_change(run_a) is None
    worktree.create(repo, "HEAD", "two1111bbbb", _run(tmp_path, "b"))  # its setup enables it
    assert git(repo, "config", "extensions.worktreeConfig").strip() == "true"
    assert worktree.shared_change(run_a) is None


@pytest.mark.parametrize("before", [None, "false"])
def test_enabling_worktree_config_from_absent_or_false_is_not_a_change(repo, tmp_path, before):
    found = worktree.create(repo, "HEAD", "ext0000aaaa", _run(tmp_path))
    git(repo, "config", "--unset", "extensions.worktreeConfig")
    if before is not None:
        git(repo, "config", "extensions.worktreeConfig", before)
    worktree.shared_change(found)  # the record moves on to the starting state
    git(repo, "config", "extensions.worktreeConfig", "true")
    assert worktree.shared_change(found) is None


@pytest.mark.parametrize("change, named", [
    (("config", "extensions.worktreeConfig", "false"), "config extensions.worktreeconfig changed"),
    (("config", "--unset", "extensions.worktreeConfig"), "config extensions.worktreeconfig removed"),
    (("config", "extensions.noop", "true"), "config extensions.noop added"),
])
def test_turning_worktree_config_off_or_any_other_extension_is_caught(repo, tmp_path, change, named):
    found = worktree.create(repo, "HEAD", "ext1111bbbb", _run(tmp_path))
    assert worktree.shared_change(found) is None, "the run's own isolation turned it on"
    git(repo, *change)
    assert worktree.shared_change(found) == named


def test_a_record_from_before_holds_worktree_config_off_as_its_baseline(repo, tmp_path):
    """A run live at upgrade whose target has the extension set false is not failed for it."""
    found = worktree.create(repo, "HEAD", "ext2222cccc", _run(tmp_path))
    git(repo, "config", "extensions.worktreeConfig", "false")
    record_path = worktree._record_path(found)
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps({key: value for key, value in record.items() if key != "shared_reads"}))
    assert worktree.shared_change(found) is None
    git(repo, "config", "extensions.worktreeConfig", "true")
    assert worktree.shared_change(found) is None
    git(repo, "config", "extensions.worktreeConfig", "false")
    assert worktree.shared_change(found) == "config extensions.worktreeconfig changed"
