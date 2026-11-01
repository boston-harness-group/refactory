"""Test guardedness extension for modified test files.

Tests for Ask 2: enforce_guardedness now reports changed test functions in
modified test files, but only when the change relaxed an assertion.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from refactory_dsh import design_ops, worktree
from refactory_dsh.nn import RunContext


# ── helpers ──────────────────────────────────────────────────────────────

def _git_repo(path: Path) -> Path:
    """Create a minimal git repo for testing."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test.com"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"],
                   capture_output=True, check=True)
    return path


def _commit(repo: Path, files: dict[str, str], msg: str = "commit") -> str:
    """Write files and commit, returning the commit hash."""
    for name, content in files.items():
        file_path = repo / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", msg, "--allow-empty"],
                   capture_output=True, check=True)
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True)
    return result.stdout.strip()


# ── tests ────────────────────────────────────────────────────────────────

def test_changed_test_fns_identifies_relaxed_assertions(tmp_path):
    """Unit test of _changed_test_fns: two versions of a file where test_foo has a
    relaxed assertion and test_bar has only a rename; returns ['test_foo'] only."""
    repo = _git_repo(tmp_path / "repo")

    base_test = '''\
def test_foo():
    result = compute()
    assert result == 42
    assert result > 0

def test_bar():
    old_name = get_value()
    assert old_name == "hello"
'''
    base_commit = _commit(repo, {"tests/test_example.py": base_test}, "base")

    head_test = '''\
def test_foo():
    result = compute()
    assert result  # relaxed: was assert result == 42

def test_bar():
    new_name = get_value()
    assert new_name == "hello"
'''
    _commit(repo, {"tests/test_example.py": head_test}, "head")

    relaxed = design_ops._changed_test_fns(repo, base_commit, "tests/test_example.py")
    assert "test_foo" in relaxed
    assert "test_bar" not in relaxed


def test_modified_test_function_with_relaxed_assertion_reported(tmp_path):
    """A changed test function whose change relaxed an assertion is reported
    by enforce_guardedness."""
    repo = _git_repo(tmp_path / "repo")

    base_test = '''\
def test_check():
    assert compute() == 42
'''
    base_commit = _commit(repo, {"tests/test_mod.py": base_test}, "base")

    head_test = '''\
def test_check():
    assert compute()  # relaxed
'''
    _commit(repo, {"tests/test_mod.py": head_test}, "head")

    # Set up context with guardedness record
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ctx = RunContext(run_dir)

    guardedness = {
        "summary": {"not_guarding": 1},
        "results": [
            {"file": "tests/test_mod.py", "guarding": False, "file_level": "not_guarding"},
        ],
    }
    ctx.write_json("test-guardedness.json", guardedness)

    # Record the worktree
    wt_dir = run_dir / ".refactory"
    wt_dir.mkdir(parents=True, exist_ok=True)
    head_commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    (wt_dir / "worktree.json").write_text(json.dumps({
        "path": str(repo), "branch": "main", "base": base_commit,
        "head": head_commit, "target": str(repo), "run": str(run_dir),
    }))

    result = design_ops.enforce_guardedness(ctx)
    assert "tests/test_mod.py" in result


def test_modified_test_function_with_rename_only_excused(tmp_path):
    """A test function edited only for renames/refactors is not flagged."""
    repo = _git_repo(tmp_path / "repo")

    base_test = '''\
def test_check():
    old_var = compute()
    assert old_var == 42
'''
    base_commit = _commit(repo, {"tests/test_mod.py": base_test}, "base")

    head_test = '''\
def test_check():
    new_var = compute()
    assert new_var == 42
'''
    _commit(repo, {"tests/test_mod.py": head_test}, "head")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ctx = RunContext(run_dir)

    guardedness = {
        "summary": {"not_guarding": 1},
        "results": [
            {"file": "tests/test_mod.py", "guarding": False, "file_level": "not_guarding"},
        ],
    }
    ctx.write_json("test-guardedness.json", guardedness)

    wt_dir = run_dir / ".refactory"
    wt_dir.mkdir(parents=True, exist_ok=True)
    head_commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    (wt_dir / "worktree.json").write_text(json.dumps({
        "path": str(repo), "branch": "main", "base": base_commit,
        "head": head_commit, "target": str(repo), "run": str(run_dir),
    }))

    result = design_ops.enforce_guardedness(ctx)
    # Rename-only change should not be reported as a weakened test
    assert "tests/test_mod.py" not in result


def test_new_function_in_modified_file_not_flagged(tmp_path):
    """A new test function added to a modified file is not flagged as weakened."""
    repo = _git_repo(tmp_path / "repo")

    base_test = '''\
def test_existing():
    assert compute() == 42
'''
    base_commit = _commit(repo, {"tests/test_mod.py": base_test}, "base")

    head_test = '''\
def test_existing():
    assert compute() == 42

def test_new_one():
    assert other() == 99
'''
    _commit(repo, {"tests/test_mod.py": head_test}, "head")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ctx = RunContext(run_dir)

    guardedness = {
        "summary": {"not_guarding": 1},
        "results": [
            {"file": "tests/test_mod.py", "guarding": False, "file_level": "not_guarding"},
        ],
    }
    ctx.write_json("test-guardedness.json", guardedness)

    wt_dir = run_dir / ".refactory"
    wt_dir.mkdir(parents=True, exist_ok=True)
    head_commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    (wt_dir / "worktree.json").write_text(json.dumps({
        "path": str(repo), "branch": "main", "base": base_commit,
        "head": head_commit, "target": str(repo), "run": str(run_dir),
    }))

    result = design_ops.enforce_guardedness(ctx)
    # A new function (not in base) should not be flagged
    assert "tests/test_mod.py" not in result
