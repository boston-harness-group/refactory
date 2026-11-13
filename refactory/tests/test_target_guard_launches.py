"""The target guard reads a git target in three git commands, and reads it as it did in seven.

Every agent step snapshots the target before and after it runs (`target_guard.watch`,
`target_guard.strayed`), so the number of git commands in one snapshot is paid on every
step; on a Mac under an endpoint-security agent each costs 30 ms or more. `state` used
to ask git seven times. These tests hold it to three -- plus one the first time a
commit is seen -- and compare it, case by case, with the seven-command reading it
replaced, which is kept here as the reference.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from refactory_dsh import target_guard


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout


def _porcelain_v1(listing: str) -> list[str]:
    fields = listing.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if len(entry) < 4:
            continue
        paths.append(entry[3:])
        if "R" in entry[:2] or "C" in entry[:2]:
            index += 1
    return paths


def _reference_state(root: Path, *, exclude: tuple[Path, ...] = ()) -> dict:
    """`target_guard.state` as it was before it was cut to three git commands."""
    git = target_guard._git
    root = Path(root).resolve()
    skipped = [root / ".factory", *(Path(path).resolve() for path in exclude)]
    found: dict = {"root": str(root), "git": False, "head": None, "branch": None, "tree": None}
    if root in skipped:
        return {**found, "changes": {}}
    inside = git(root, "rev-parse", "--is-inside-work-tree")
    if inside is not None and inside.strip() == "true" and git(root, "check-ignore", "-q", str(root)) is None:
        top = Path((git(root, "rev-parse", "--show-toplevel") or str(root)).strip()).resolve()
        specs = [f":(exclude,literal){path.relative_to(root)}" for path in skipped if path.is_relative_to(root)]
        listing = git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ".", *specs)
        assert listing is not None
        paths = [os.path.relpath(top / path, root) for path in _porcelain_v1(listing)]
        prefix = "" if root == top else root.relative_to(top).as_posix()
        found.update(git=True, head=(git(root, "rev-parse", "-q", "--verify", "HEAD") or "").strip() or None,
                     branch=(git(root, "symbolic-ref", "-q", "--short", "HEAD") or "").strip() or None,
                     tree=(git(root, "rev-parse", "-q", "--verify", f"HEAD:{prefix}") or "").strip() or None)
    else:
        paths = []
        for directory, names, files in os.walk(root):
            here = Path(directory)
            names[:] = [name for name in names if name not in target_guard.UNWATCHED and here / name not in skipped]
            paths.extend(str((here / name).relative_to(root)) for name in files if name not in target_guard.UNWATCHED)
    return {**found, "changes": {path: target_guard._digest(root / path) for path in sorted(paths)}}


def _repo(root: Path) -> Path:
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    (root / "app.py").write_text("print('app')\n")
    (root / "lib").mkdir()
    (root / "lib" / "util.py").write_text("x = 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _commit(root: Path, message: str = "more") -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--allow-empty", "-m", message)


# Each case changes a fresh checkout, then names the directory the guard watches in it.
def _clean(root: Path) -> Path:
    return root


def _edits(root: Path) -> Path:
    (root / "app.py").write_text("print('changed')\n")
    (root / "new file.txt").write_text("spaces\n")
    (root / "lib" / "util.py").unlink()
    return root


def _staged_rename(root: Path) -> Path:
    _git(root, "mv", "app.py", "main.py")
    return root


def _worktree_rename(root: Path) -> Path:
    # A rename only in the work tree (` R`), after `git add -N`.
    os.rename(root / "app.py", root / "moved.py")
    _git(root, "add", "-N", "moved.py")
    return root


def _detached(root: Path) -> Path:
    _commit(root)
    _git(root, "checkout", "-q", "--detach", "HEAD~1")
    (root / "scratch.txt").write_text("x\n")
    return root


def _unborn(root: Path) -> Path:
    _git(root, "checkout", "-q", "--orphan", "fresh")
    _git(root, "rm", "-q", "-r", "--cached", ".")
    return root


def _other_branch(root: Path) -> Path:
    _git(root, "checkout", "-q", "-b", "feature")
    _commit(root, "feature work")
    return root


def _subdirectory(root: Path) -> Path:
    (root / "lib" / "util.py").write_text("x = 2\n")
    (root / "lib" / "extra.py").write_text("y = 1\n")
    (root / "app.py").write_text("outside the target\n")
    return root / "lib"


def _subdirectory_uncommitted(root: Path) -> Path:
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text("a\n")
    return root / "pkg"


def _ignored_subdirectory(root: Path) -> Path:
    (root / ".gitignore").write_text("build/\n")
    (root / "build" / "out").mkdir(parents=True)
    (root / "build" / "out" / "a.o").write_text("object\n")
    return root / "build"


def _top_ignored_by_its_own_rules(root: Path) -> Path:
    (root / ".gitignore").write_text("*\n")
    (root / "extra.txt").write_text("x\n")
    return root


def _not_a_checkout(root: Path) -> Path:
    plain = root.parent / "plain"
    (plain / "src").mkdir(parents=True)
    (plain / "src" / "a.py").write_text("a\n")
    (plain / "node_modules").mkdir()
    (plain / "node_modules" / "x.js").write_text("x\n")
    return plain


def _git_directory(root: Path) -> Path:
    return root / ".git"


def _factory_and_run_excluded(root: Path) -> Path:
    (root / ".factory").mkdir()
    (root / ".factory" / "state.json").write_text("{}\n")
    (root / "run").mkdir()
    (root / "run" / "out.txt").write_text("run output\n")
    (root / "app.py").write_text("print('edited')\n")
    return root


CASES = [_clean, _edits, _staged_rename, _worktree_rename, _detached, _unborn, _other_branch,
         _subdirectory, _subdirectory_uncommitted, _ignored_subdirectory,
         _top_ignored_by_its_own_rules, _not_a_checkout, _git_directory, _factory_and_run_excluded]


@pytest.mark.parametrize("case", CASES, ids=[case.__name__.strip("_") for case in CASES])
def test_state_reads_a_target_as_the_seven_command_reading_did(tmp_path, case):
    watched = case(_repo(tmp_path / "repo"))
    exclude = (tmp_path / "repo" / "run",)

    assert target_guard.state(watched, exclude=exclude) == _reference_state(watched, exclude=exclude)


def test_a_commit_is_seen_after_the_tree_of_an_earlier_one_was_remembered(tmp_path):
    """The tree under the target is remembered per commit; a new commit is asked about anew."""
    root = _repo(tmp_path / "repo")
    before = target_guard.state(root / "lib")
    (root / "lib" / "util.py").write_text("x = 3\n")
    _commit(root)
    after = target_guard.state(root / "lib")

    assert after == _reference_state(root / "lib")
    assert after["tree"] != before["tree"] and after["head"] != before["head"]
    assert target_guard._compare(before, after, commits=False)


def _launches(monkeypatch) -> list[list[str]]:
    seen: list[list[str]] = []
    original = subprocess.run

    def counted(argv, *args, **kwargs):
        seen.append(list(argv))
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(target_guard.subprocess, "run", counted)
    return seen


def test_a_snapshot_of_a_git_target_is_three_git_commands(tmp_path, monkeypatch):
    """Three on every snapshot, and one more only the first time a commit is seen."""
    root = _repo(tmp_path / "repo")
    _edits(root)
    target_guard._tree.cache_clear()
    launches = _launches(monkeypatch)

    target_guard.state(root)
    assert len(launches) == 4
    launches.clear()
    target_guard.state(root)
    assert len(launches) == 3


def test_watching_an_agent_step_launches_a_bounded_number_of_git_commands(tmp_path, monkeypatch):
    """`watch` then `strayed` around one step of a worktree run: the target, the worktree
    and the launcher checkout are each read before and after, three commands a read."""
    target = _repo(tmp_path / "target")
    worktree = _repo(tmp_path / "worktree")
    launcher = _repo(tmp_path / "launcher")
    workdir = tmp_path / "run"
    target_guard.record(workdir, target, output_only=("emit",), also=(worktree,), warn_also=(launcher,))
    target_guard._tree.cache_clear()
    target_guard.state(target), target_guard.state(worktree), target_guard.state(launcher)
    launches = _launches(monkeypatch)

    before = target_guard.watch(workdir, "emit")
    assert target_guard.strayed(workdir, before) == ([], [])
    assert len(launches) == 2 * 3 * 3
