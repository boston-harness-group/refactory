"""refactory reads a repository with the git macOS's /usr/bin/git trampoline would start.

The trampoline is a second process launch on every git command, which on a Mac under an
endpoint-security agent doubles what each one costs. Reads skip it; anything that can
run a hook (a commit, a merge, a clone) still goes through `git` on PATH, because the
trampoline gives the git it starts a compiler environment a hook may rely on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from refactory_dsh import git_exe


@pytest.fixture(autouse=True)
def _fresh_resolution():
    git_exe._resolve.cache_clear()
    yield
    git_exe._resolve.cache_clear()


def _developer(tmp_path: Path, inside: str = "usr/bin") -> Path:
    directory = tmp_path / "Developer"
    (directory / inside).mkdir(parents=True)
    binary = directory / inside / "git"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    return directory


def _as_a_mac_with_the_trampoline(monkeypatch):
    monkeypatch.setattr(git_exe.sys, "platform", "darwin")
    monkeypatch.setattr(git_exe.shutil, "which", lambda name, path=None: git_exe.TRAMPOLINE)


def test_on_a_mac_the_trampoline_s_git_in_the_chosen_developer_directory_is_run(tmp_path, monkeypatch):
    _as_a_mac_with_the_trampoline(monkeypatch)
    developer = _developer(tmp_path)
    monkeypatch.setenv("DEVELOPER_DIR", str(developer))

    assert git_exe.git() == str(developer / "usr/bin/git")


def test_a_developer_directory_that_names_xcode_app_itself_is_looked_inside(tmp_path, monkeypatch):
    _as_a_mac_with_the_trampoline(monkeypatch)
    xcode = _developer(tmp_path, "Contents/Developer/usr/bin")
    monkeypatch.setenv("DEVELOPER_DIR", str(xcode))

    assert git_exe.git() == str(xcode / "Contents/Developer/usr/bin/git")


def test_without_a_developer_directory_setting_the_xcode_select_choice_is_used(tmp_path, monkeypatch):
    _as_a_mac_with_the_trampoline(monkeypatch)
    developer = _developer(tmp_path)
    link = tmp_path / "xcode_select_link"
    link.symlink_to(developer)
    monkeypatch.delenv("DEVELOPER_DIR", raising=False)
    monkeypatch.setattr(git_exe, "SELECTED", str(link))

    assert git_exe.git() == str(developer / "usr/bin/git")


def test_with_no_choice_at_all_xcode_comes_before_the_command_line_tools(tmp_path, monkeypatch):
    _as_a_mac_with_the_trampoline(monkeypatch)
    xcode, tools = _developer(tmp_path / "xcode"), _developer(tmp_path / "tools")
    monkeypatch.delenv("DEVELOPER_DIR", raising=False)
    monkeypatch.setattr(git_exe, "SELECTED", str(tmp_path / "no-link"))
    monkeypatch.setattr(git_exe, "DEFAULT_DEVELOPERS", (str(xcode), str(tools)))
    assert git_exe.git() == str(xcode / "usr/bin/git")

    git_exe._resolve.cache_clear()
    monkeypatch.setattr(git_exe, "DEFAULT_DEVELOPERS", (str(tmp_path / "no-xcode"), str(tools)))
    assert git_exe.git() == str(tools / "usr/bin/git")


def test_a_developer_directory_with_no_git_keeps_plain_git(tmp_path, monkeypatch):
    _as_a_mac_with_the_trampoline(monkeypatch)
    monkeypatch.setenv("DEVELOPER_DIR", str(tmp_path / "empty"))

    assert git_exe.git() == "git"


def test_off_a_mac_git_is_git(monkeypatch, tmp_path):
    monkeypatch.setattr(git_exe.sys, "platform", "linux")
    monkeypatch.setenv("DEVELOPER_DIR", str(_developer(tmp_path)))

    assert git_exe.git() == "git"


def test_a_git_ahead_of_the_trampoline_on_path_is_the_one_run(tmp_path, monkeypatch):
    """Homebrew's git, or a test's stub, is found on PATH and run as `git`, as before."""
    monkeypatch.setattr(git_exe.sys, "platform", "darwin")
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "git").write_text("#!/bin/sh\n")
    (stub / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub}:/usr/bin:/bin")
    monkeypatch.setenv("DEVELOPER_DIR", str(_developer(tmp_path)))

    assert git_exe.git() == "git"


@pytest.mark.parametrize("args", [
    ("status", "--porcelain=v2"),
    ("--no-optional-locks", "-C", "/repo", "rev-parse", "HEAD"),
    ("-C", "/repo", "-c", "core.quotePath=false", "ls-tree", "-r", "HEAD"),
    ("config", "--file", "/repo/.git/config", "--list"),
])
def test_a_read_skips_the_trampoline(args, tmp_path, monkeypatch):
    _as_a_mac_with_the_trampoline(monkeypatch)
    developer = _developer(tmp_path)
    monkeypatch.setenv("DEVELOPER_DIR", str(developer))

    assert git_exe.program(*args) == str(developer / "usr/bin/git")


@pytest.mark.parametrize("args", [
    ("commit", "-q", "-m", "x"),
    ("-C", "/repo", "-c", "core.hooksPath=/dev/null", "merge", "--no-ff", "topic"),
    ("-C", "/repo", "checkout", "-q", "main"),
    ("clone", "--shared", "/a", "/b"),
    ("-C", "/repo", "worktree", "add", "/w"),
    ("-C", "/repo", "status-like-alias"),
    (),
])
def test_anything_that_can_run_a_hook_still_goes_through_git_on_path(args, tmp_path, monkeypatch):
    _as_a_mac_with_the_trampoline(monkeypatch)
    monkeypatch.setenv("DEVELOPER_DIR", str(_developer(tmp_path)))

    assert git_exe.program(*args) == "git"


@pytest.mark.skipif(sys.platform != "darwin", reason="the trampoline is macOS's")
def test_on_this_mac_the_git_run_is_the_one_the_trampoline_runs(tmp_path):
    """Same version and same exec path, so its helpers (remote-https, ...) are found."""
    def ask(program: str, *args: str) -> str:
        return subprocess.run([program, *args], check=True, capture_output=True, text=True).stdout

    found = git_exe.git()
    assert ask(found, "--version") == ask("git", "--version")
    assert ask(found, "--exec-path") == ask("git", "--exec-path")
