"""Session-level repository guard: fail the test suite when any test changes the repository under test.

Loaded as a pytest plugin via ``pytest_plugins`` in conftest.py.  Also provides a
``__main__`` entry point that the vitest globalSetup calls to snapshot and compare.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest


# ── snapshot and comparison logic ─────────────────────────────────────────


def _take_snapshot(repo: Path) -> dict[str, Any]:
    """Take a snapshot of the repository's shared state and worktree list."""
    from refactory_dsh.worktree import _shared_snapshot

    shared = _shared_snapshot(repo)
    result = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        capture_output=True, text=True, timeout=30,
    )
    worktree_list = result.stdout if result.returncode == 0 else ""
    return {
        "config_lines": shared["config_lines"],
        "hook_files": shared["hook_files"],
        "worktrees": _parse_worktrees(worktree_list),
        "files": _working_files(repo),
    }


def _working_files(repo: Path) -> dict[str, str]:
    """Each path `git status` lists (changed or untracked, ignored ones not), with a digest
    of its bytes: a test that writes into the checkout shows here, even over a file the
    person had already changed."""
    result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        capture_output=True, timeout=60,
    )
    top = subprocess.run(["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, timeout=30)
    if result.returncode != 0 or top.returncode != 0:
        return {}
    # Porcelain paths are relative to the top of the checkout, wherever pytest started.
    root = Path(top.stdout.strip())
    files: dict[str, str] = {}
    entries = iter(result.stdout.split(b"\0"))
    for entry in entries:
        if len(entry) < 4:
            continue
        name = entry[3:].decode(errors="surrogateescape")
        if entry[:1] in (b"R", b"C"):
            next(entries, None)  # the rename's source
        path = root / name
        try:
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"
        except OSError:
            files[name] = "unreadable"
    return files


def _parse_worktrees(porcelain: str) -> set[str]:
    """Extract worktree paths from porcelain output."""
    paths: set[str] = set()
    for line in porcelain.splitlines():
        if line.startswith("worktree "):
            paths.add(line.removeprefix("worktree "))
    return paths


def _compare_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
    temp_roots: list[str],
    refactory_home: str | None,
) -> tuple[list[str], list[str]]:
    """Compare two snapshots.  Returns ``(failures, warnings)``.

    Failures are changes attributed to this session.  Warnings are external changes
    that are reported without failing the session.
    """
    failures: list[str] = []
    warnings: list[str] = []

    # Config changes always fail.
    before_lines = set(before["config_lines"])
    after_lines = set(after["config_lines"])
    for line in sorted(after_lines - before_lines):
        failures.append(f"config line added: {line}")
    for line in sorted(before_lines - after_lines):
        failures.append(f"config line removed: {line}")

    # Hook changes always fail.
    before_hooks: dict[str, str] = before["hook_files"]
    after_hooks: dict[str, str] = after["hook_files"]
    for name in sorted(set(before_hooks) | set(after_hooks)):
        if name not in before_hooks:
            failures.append(f"hook added: {name}")
        elif name not in after_hooks:
            failures.append(f"hook removed: {name}")
        elif before_hooks[name] != after_hooks[name]:
            failures.append(f"hook changed: {name}")

    # Working-tree files: a test writes only under its own temporary directories. In CI
    # nobody else edits the checkout, so a change fails; locally it is more often the
    # person saving a file during a long run, so it is reported, not failed. (A file a
    # test leaves in refactory/tests fails everywhere: conftest checks that per test.)
    before_files: dict[str, str] = before.get("files") or {}
    after_files: dict[str, str] = after.get("files") or {}
    changed_files = failures if os.environ.get("CI", "").lower() not in ("", "0", "false") else warnings
    for name in sorted(set(before_files) | set(after_files)):
        if before_files.get(name) != after_files.get(name):
            changed_files.append(f"file changed in the checkout under test (a test wrote there, "
                                 f"or it was edited during the run): {name}")

    # Worktree changes: attribute to session or report as external.
    before_wts = set(before.get("worktrees") or set())
    after_wts = set(after.get("worktrees") or set())
    for wt in sorted(after_wts - before_wts):
        if _is_session_worktree(wt, temp_roots, refactory_home):
            failures.append(f"worktree added by session: {wt}")
        else:
            warnings.append(f"worktree added externally (not failing): {wt}")
    for wt in sorted(before_wts - after_wts):
        if _is_session_worktree(wt, temp_roots, refactory_home):
            failures.append(f"worktree removed by session: {wt}")
        else:
            warnings.append(f"worktree removed externally (not failing): {wt}")

    return failures, warnings


def _is_session_worktree(
    wt_path: str, temp_roots: list[str], refactory_home: str | None,
) -> bool:
    """Whether a worktree path is inside one of the session's own spaces."""
    p = Path(wt_path).resolve()
    for root in temp_roots:
        try:
            if root and p.is_relative_to(Path(root).resolve()):
                return True
        except (TypeError, ValueError):
            pass
    if refactory_home:
        try:
            if p.is_relative_to(Path(refactory_home).resolve()):
                return True
        except (TypeError, ValueError):
            pass
    return False


# ── pytest hooks ──────────────────────────────────────────────────────────

_GUARD_KEY = pytest.StashKey[dict]()
_GUARD_CHANGES_KEY = pytest.StashKey[tuple]()
_GUARD_INACTIVE_KEY = pytest.StashKey[str]()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Record the repository's state before any test runs."""
    repo = session.config.rootpath

    try:
        snapshot = _take_snapshot(repo)
    except (ValueError, OSError) as exc:
        session.config.stash[_GUARD_INACTIVE_KEY] = str(exc)
        return

    # Session-wide REFACTORY_HOME so subprocesses don't write into the real one.
    guard_home = Path(tempfile.mkdtemp(prefix="repo-guard-home-"))
    os.environ["REFACTORY_HOME"] = str(guard_home / "home")
    os.environ["REFACTORY_PROJECTS_DIR"] = str(guard_home / "projects")

    try:
        basetemp = str(session.config._tmp_path_factory.getbasetemp())
    except Exception:
        basetemp = ""

    session.config.stash[_GUARD_KEY] = {
        "snapshot": snapshot,
        "repo": str(repo),
        "temp_roots": [basetemp, tempfile.gettempdir()],
        "refactory_home": str(guard_home / "home"),
        "guard_home": str(guard_home),
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Compare the repository's state and fail the session on changes."""
    stored = session.config.stash.get(_GUARD_KEY, None)
    if stored is None:
        return

    repo = Path(stored["repo"])
    try:
        after = _take_snapshot(repo)
    except (ValueError, OSError) as exc:
        session.config.stash[_GUARD_INACTIVE_KEY] = str(exc)
        return

    failures, warnings = _compare_snapshots(
        stored["snapshot"],
        after,
        stored["temp_roots"],
        stored.get("refactory_home"),
    )

    if failures or warnings:
        session.config.stash[_GUARD_CHANGES_KEY] = (failures, warnings)
    if failures:
        session.exitstatus = 1

    # Clean up the guard's temp directory.
    guard_home = stored.get("guard_home")
    if guard_home:
        shutil.rmtree(guard_home, ignore_errors=True)


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Report what the repository guard found."""
    inactive = config.stash.get(_GUARD_INACTIVE_KEY, None)
    if inactive is not None:
        terminalreporter.write_line(f"repository guard inactive: {inactive}")

    changes = config.stash.get(_GUARD_CHANGES_KEY, None)
    if changes is None:
        return

    failures, warnings = changes
    if failures:
        terminalreporter.write_sep("=", "repository guard FAILED")
        for msg in failures:
            terminalreporter.write_line(f"  FAIL: {msg}")
    if warnings:
        if not failures:
            terminalreporter.write_sep("=", "repository guard (external changes noted)")
        for msg in warnings:
            terminalreporter.write_line(f"  NOTE: {msg}")


# ── __main__ entry point for vitest ───────────────────────────────────────


def _snapshot_json(repo: Path) -> str:
    """Return the snapshot as a JSON string."""
    snapshot = _take_snapshot(repo)
    # Convert sets to sorted lists for JSON serialisation.
    snapshot["worktrees"] = sorted(snapshot["worktrees"])
    return json.dumps(snapshot)


def _compare_main(
    repo: Path,
    before_file: Path,
    temp_root: str,
    refactory_home: str,
) -> tuple[list[str], list[str]]:
    """Compare and return ``(failures, warnings)``."""
    before = json.loads(before_file.read_text())
    before["worktrees"] = set(before.get("worktrees", []))
    after = _take_snapshot(repo)
    return _compare_snapshots(before, after, [temp_root], refactory_home)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Repository guard for the test suite")
    sub = parser.add_subparsers(dest="command")

    snap_parser = sub.add_parser("snapshot")
    snap_parser.add_argument("repo", type=Path)

    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("repo", type=Path)
    cmp_parser.add_argument("before_file", type=Path)
    cmp_parser.add_argument("--temp-root", required=True)
    cmp_parser.add_argument("--refactory-home", required=True)

    args = parser.parse_args()

    if args.command == "snapshot":
        print(_snapshot_json(args.repo))
    elif args.command == "compare":
        failures, warnings = _compare_main(
            args.repo, args.before_file, args.temp_root, args.refactory_home,
        )
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        for msg in warnings:
            print(f"NOTE: {msg}", file=sys.stderr)
        sys.exit(1 if failures else 0)
    else:
        parser.print_help()
        sys.exit(2)
