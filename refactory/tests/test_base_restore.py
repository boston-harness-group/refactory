"""Tests for base worktree integrity (dm-162).

Each test verifies one of the three layers of defence:
  Layer 1 — restore-in-finally in compute_test_guardedness
  Layer 2 — dirty-base detection and repair in _ensure_base_worktree
  Layer 3 — base_clean cache guard

# guardedness-exempt: these tests pin new behaviour that does not exist on
# the base commit.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import patch

import pytest

from refactory_dsh import base_check, sandbox, worktree
from refactory_dsh import target_guard
from refactory_dsh.worktree import _json as _read_json, NO_HOOKS


# ── helpers ────────────────────────────────────────────────────────────────


#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def _git_init(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "T"],
                   capture_output=True, check=True)
    return path


def _commit(repo: Path, name: str, text: str) -> str:
    p = repo / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    subprocess.run(["git", "-C", str(repo), "add", name],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", f"add {name}"],
                   capture_output=True, check=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


class _Ctx:
    """Minimal RunContext stand-in for unit-level tests."""

    def __init__(self, workdir: Path, *, facts: dict | None = None,
                 knobs: dict | None = None):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.facts: dict[str, Any] = dict(facts or {})
        self.knobs_dict: dict[str, Any] = dict(knobs or {})
        self.budget = None
        self.enforce_writes = False
        self._produced: set[str] = set()
        self._declared_facts: tuple[str, ...] = ()

    def knob(self, name: str, default: Any = None) -> Any:
        return self.knobs_dict.get(name, default)

    def set_fact(self, name: str, value: Any) -> None:
        self.facts[name] = value
        self._produced.add(name)

    def exists(self, name: str) -> bool:
        return (self.workdir / name).exists()

    def read_text(self, name: str) -> str:
        return (self.workdir / name).read_text()

    def write_text(self, name: str, text: str) -> Path:
        target = self.workdir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    def write_json(self, name: str, payload: Any) -> Path:
        return self.write_text(name, json.dumps(payload, indent=2, default=str))

    def read_json(self, name: str, default: Any = None) -> Any:
        p = self.workdir / name
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return default


class _Env(NamedTuple):
    repo: Path
    base_commit: str
    run_dir: Path
    attempt_path: Path
    found: worktree.Worktree
    ctx: _Ctx


@pytest.fixture
def base_worktree_env(tmp_path) -> _Env:
    """A real worktree environment for testing base worktree integrity."""
    repo = _git_init(tmp_path / "repo")

    # Create .gitignore first
    _commit(repo, ".gitignore", "node_modules/\n.venv/\n__pycache__/\n")

    # Create tracked test files and helpers
    _commit(repo, "tests/test_math.py",
            "def test_add():\n    assert 1 + 1 == 2\n")
    _commit(repo, "tests/helpers/graph-shapes.ts",
            "export const square = (n: number) => n * n;\n")
    _commit(repo, "tests/test_example.py",
            "def test_example():\n    assert True\n")
    base_commit = _commit(repo, "app.py", "print('app')\n")

    # Create the run directory and worktree
    run_dir = tmp_path / "run"
    found = worktree.create(repo, base_commit, "test0base01", run_dir)
    wt = Path(found.path)

    # Make attempt changes
    (wt / "tests" / "test_math.py").write_text(
        "def test_add():\n    assert 1 + 1 == 2\n"
        "def test_sub():\n    assert 2 - 1 == 1\n"
    )
    (wt / "tests" / "helpers" / "graph-shapes.ts").write_text(
        "export const square = (n: number) => n * n;\n"
        "export const cube = (n: number) => n * n * n;\n"
    )
    (wt / "tests" / "test_new_feature.py").write_text(
        "def test_new():\n    assert True\n"
    )
    subprocess.run(["git", "-C", str(wt), "add", "."],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(wt), "commit", "-q", "-m", "attempt"],
                   capture_output=True, check=True)

    # Update found with the new head
    new_head = subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    found = worktree.Worktree(
        path=found.path, branch=found.branch,
        base=base_commit, head=new_head,
        target=found.target, run=found.run,
    )
    worktree.write_wrapper(found, run_dir)

    ctx = _Ctx(run_dir, facts={"check": "python -m pytest -q"})

    return _Env(
        repo=repo, base_commit=base_commit, run_dir=run_dir,
        attempt_path=wt, found=found, ctx=ctx,
    )


# ── Test 1: changed test file restored after guardedness ────────────────


def test_changed_test_file_restored_after_guardedness(base_worktree_env, monkeypatch):
    """After guardedness, a changed test file is still present in the base and
    byte-identical to the base commit's version. This includes a changed helper."""
    env = base_worktree_env

    # Patch sandbox.run_command to return a successful result
    ok_result = sandbox.SandboxResult(ok=True, returncode=0,
                                      stdout="1 passed\n", stderr="")
    monkeypatch.setattr(sandbox, "run_command", lambda *a, **kw: ok_result)

    base_check.compute_test_guardedness(env.ctx, "python -m pytest -q", 60)

    # Get the base worktree path
    found = worktree.recorded(env.ctx.workdir)
    target = Path(found.target)
    run_id = Path(found.path).name.removeprefix("run-")
    base_path = base_check._base_worktree_path(target, run_id)

    # Changed files must exist and be byte-identical to the base commit's version
    for tf in ["tests/test_math.py", "tests/helpers/graph-shapes.ts"]:
        assert (base_path / tf).exists(), f"{tf} was deleted from the base"
        expected = worktree.git(
            env.repo, "show", f"{env.base_commit}:{tf}"
        ).encode()
        actual = (base_path / tf).read_bytes()
        assert actual == expected, (
            f"{tf} in the base is not byte-identical to the base commit's version"
        )


# ── Test 2: new test file removed after guardedness ─────────────────────


def test_new_test_file_removed_after_guardedness(base_worktree_env, monkeypatch):
    """A new test file copied for guardedness is removed from the base afterwards."""
    env = base_worktree_env

    ok_result = sandbox.SandboxResult(ok=True, returncode=0,
                                      stdout="1 passed\n", stderr="")
    monkeypatch.setattr(sandbox, "run_command", lambda *a, **kw: ok_result)

    base_check.compute_test_guardedness(env.ctx, "python -m pytest -q", 60)

    found = worktree.recorded(env.ctx.workdir)
    target = Path(found.target)
    run_id = Path(found.path).name.removeprefix("run-")
    base_path = base_check._base_worktree_path(target, run_id)

    assert not (base_path / "tests" / "test_new_feature.py").exists(), (
        "new test file was not removed from the base after guardedness"
    )


# ── Test 3: crash during guardedness restores base ──────────────────────


def test_crash_during_guardedness_restores_base_timeout(base_worktree_env, monkeypatch):
    """Guardedness with a sandbox timeout still restores the base."""
    env = base_worktree_env

    timeout_result = sandbox.SandboxResult(
        ok=False, returncode=-1,
        stdout="", stderr="timed out",
        timed_out=True,
    )
    monkeypatch.setattr(sandbox, "run_command", lambda *a, **kw: timeout_result)

    base_check.compute_test_guardedness(env.ctx, "python -m pytest -q", 60)

    found = worktree.recorded(env.ctx.workdir)
    target = Path(found.target)
    run_id = Path(found.path).name.removeprefix("run-")
    base_path = base_check._base_worktree_path(target, run_id)

    for tf in ["tests/test_math.py", "tests/helpers/graph-shapes.ts"]:
        assert (base_path / tf).exists(), f"{tf} missing after timeout"
        expected = worktree.git(env.repo, "show", f"{env.base_commit}:{tf}").encode()
        assert (base_path / tf).read_bytes() == expected
    assert not (base_path / "tests" / "test_new_feature.py").exists()


def test_crash_during_guardedness_restores_base_budget_exhausted(base_worktree_env, monkeypatch):
    """Guardedness with BudgetExhausted still restores the base."""
    env = base_worktree_env
    from refactory_dsh.budgets import BudgetExhausted

    def raise_budget(*a, **kw):
        raise BudgetExhausted("out of budget")

    monkeypatch.setattr(sandbox, "run_command", raise_budget)

    with pytest.raises(BudgetExhausted):
        base_check.compute_test_guardedness(env.ctx, "python -m pytest -q", 60)

    found = worktree.recorded(env.ctx.workdir)
    target = Path(found.target)
    run_id = Path(found.path).name.removeprefix("run-")
    base_path = base_check._base_worktree_path(target, run_id)

    for tf in ["tests/test_math.py", "tests/helpers/graph-shapes.ts"]:
        assert (base_path / tf).exists(), f"{tf} missing after BudgetExhausted"
        expected = worktree.git(env.repo, "show", f"{env.base_commit}:{tf}").encode()
        assert (base_path / tf).read_bytes() == expected
    assert not (base_path / "tests" / "test_new_feature.py").exists()


def test_crash_during_guardedness_restores_base_keyboard_interrupt(base_worktree_env, monkeypatch):
    """Guardedness with KeyboardInterrupt still restores the base."""
    env = base_worktree_env

    def raise_keyboard(*a, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(sandbox, "run_command", raise_keyboard)

    with pytest.raises(KeyboardInterrupt):
        base_check.compute_test_guardedness(env.ctx, "python -m pytest -q", 60)

    found = worktree.recorded(env.ctx.workdir)
    target = Path(found.target)
    run_id = Path(found.path).name.removeprefix("run-")
    base_path = base_check._base_worktree_path(target, run_id)

    for tf in ["tests/test_math.py", "tests/helpers/graph-shapes.ts"]:
        assert (base_path / tf).exists(), f"{tf} missing after KeyboardInterrupt"
        expected = worktree.git(env.repo, "show", f"{env.base_commit}:{tf}").encode()
        assert (base_path / tf).read_bytes() == expected
    assert not (base_path / "tests" / "test_new_feature.py").exists()


def test_sigterm_followed_by_layer2_restores_base(base_worktree_env):
    """A SIGTERM'd process followed by the next use of the base is covered by Layer 2."""
    env = base_worktree_env

    # Create the base worktree first
    base_path, base_found, _ = base_check._ensure_base_worktree(env.ctx)

    # Simulate SIGTERM damage: overwrite a tracked file, add stray files
    (base_path / "tests" / "test_math.py").write_text("CORRUPTED BY SIGTERM\n")
    (base_path / "stray_after_sigterm.txt").write_text("leftover\n")

    # The next _ensure_base_worktree call (Layer 2) should repair it
    base_path2, _, repaired = base_check._ensure_base_worktree(env.ctx)

    expected = worktree.git(env.repo, "show", f"{env.base_commit}:tests/test_math.py").encode()
    assert (base_path2 / "tests" / "test_math.py").read_bytes() == expected
    assert not (base_path2 / "stray_after_sigterm.txt").exists()
    assert len(repaired) > 0, "repair should have reported tracked edits"


# ── Test 4: dirty base worktree repaired ────────────────────────────────


def test_dirty_base_worktree_repaired(base_worktree_env, capsys):
    """A dirty base worktree is reset before callers use it, and installs survive."""
    env = base_worktree_env

    # Create the base worktree
    base_path, base_found, _ = base_check._ensure_base_worktree(env.ctx)

    # Dirty the base: overwrite tracked file
    (base_path / "tests" / "test_math.py").write_text("CORRUPTED\n")
    # Untracked non-ignored file
    (base_path / "stray.txt").write_text("stray\n")
    # Untracked directory
    (base_path / "stray_dir").mkdir(exist_ok=True)
    (base_path / "stray_dir" / "file.txt").write_text("x\n")
    # Stage a new file
    (base_path / "staged.txt").write_text("staged\n")
    worktree.git(base_path, "add", "staged.txt")
    # File with a space in its name
    (base_path / "a file.txt").write_text("space\n")
    # Gitignored file (should survive)
    (base_path / "node_modules").mkdir(exist_ok=True)
    (base_path / "node_modules" / "pkg.js").write_text("cached\n")

    # Call _ensure_base_worktree again to trigger Layer 2
    base_path2, _, repaired = base_check._ensure_base_worktree(env.ctx)

    # Tracked file restored
    expected = worktree.git(
        env.repo, "show", f"{env.base_commit}:tests/test_math.py"
    ).encode()
    assert (base_path2 / "tests" / "test_math.py").read_bytes() == expected

    # Untracked files removed
    assert not (base_path2 / "stray.txt").exists()
    assert not (base_path2 / "stray_dir").exists()
    assert not (base_path2 / "staged.txt").exists()
    assert not (base_path2 / "a file.txt").exists()

    # Gitignored files survive
    assert (base_path2 / "node_modules" / "pkg.js").exists()

    # Warning emitted
    captured = capsys.readouterr()
    assert "run warning:" in captured.err, (
        f"expected 'run warning:' in stderr, got: {captured.err!r}"
    )

    # base_reset accumulated on ctx
    base_reset = getattr(env.ctx, '_base_reset', [])
    assert len(base_reset) > 0, "base_reset should have entries"


@pytest.mark.parametrize("tracked", [False, True], ids=["writes-lockfile", "rewrites-lockfile"])
def test_preparing_the_base_never_commits_the_lockfile_its_install_writes(tmp_path, monkeypatch, tracked):
    """The base is a fixed commit: a lockfile its install writes or rewrites stays in the
    base worktree, uncommitted, and the run's worktree record still names the run's own
    worktree (dm-188). The next preparation does not reinstall."""
    repo = _git_init(tmp_path / "repo")
    _commit(repo, "package.json", '{"name": "x"}\n')
    if tracked:
        _commit(repo, "package-lock.json", '{"lockfileVersion": 1}\n')
    base_commit = _commit(repo, "tests/test_math.py", "def test_add():\n    assert 1 + 1 == 2\n")
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "npm").write_text("#!/bin/sh\necho \"$@\" >> \"$(dirname \"$0\")/calls.log\"\n"
                              "mkdir -p node_modules/pkg && echo installed > node_modules/pkg/index.js\n"
                              "echo '{\"lockfileVersion\": 3}' > package-lock.json\n")
    (stub / "npm").chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub}{os.pathsep}{os.environ['PATH']}")
    run_dir = tmp_path / "run"
    found = worktree.create(repo, base_commit, "test0base88", run_dir)
    ctx = _Ctx(run_dir)

    base_path, _, _ = base_check._ensure_base_worktree(ctx)

    assert worktree.git(base_path, "rev-parse", "HEAD").strip() == base_commit
    assert (base_path / "package-lock.json").read_text() == '{"lockfileVersion": 3}\n'
    assert worktree.recorded(run_dir) == found
    base_check._ensure_base_worktree(ctx)
    assert worktree.git(base_path, "rev-parse", "HEAD").strip() == base_commit
    calls = (stub / "calls.log").read_text().splitlines()
    assert len(calls) == 1, f"the base was rebuilt and reinstalled: {calls}"


def test_unignored_install_survives_reset_without_reinstall(tmp_path, monkeypatch):
    """An install git does not ignore is kept by the install record, and the re-check
    does not count it as dirt, so a reset neither rebuilds nor reinstalls the base (C1)."""
    repo = _git_init(tmp_path / "repo")
    _commit(repo, "package.json", '{"name": "x"}\n')
    base_commit = _commit(repo, "tests/test_math.py", "def test_add():\n    assert 1 + 1 == 2\n")
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "npm").write_text("#!/bin/sh\necho \"$@\" >> \"$(dirname \"$0\")/calls.log\"\n"
                              "mkdir -p node_modules/pkg && echo installed > node_modules/pkg/index.js\n")
    (stub / "npm").chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub}{os.pathsep}{os.environ['PATH']}")
    run_dir = tmp_path / "run"
    worktree.create(repo, base_commit, "test0base02", run_dir)
    ctx = _Ctx(run_dir)

    base_path, _, _ = base_check._ensure_base_worktree(ctx)
    assert (base_path / "node_modules" / "pkg" / "index.js").exists()
    (base_path / "tests" / "test_math.py").write_text("CORRUPTED\n")
    (base_path / "stray.txt").write_text("stray\n")

    _, _, repaired = base_check._ensure_base_worktree(ctx)

    assert repaired == ["tests/test_math.py"]
    assert not (base_path / "stray.txt").exists()
    assert (base_path / "node_modules" / "pkg" / "index.js").read_text() == "installed\n"
    calls = (stub / "calls.log").read_text().splitlines()
    assert len(calls) == 1, f"the base was rebuilt and reinstalled: {calls}"


def test_dirty_base_through_run_base_check(base_worktree_env, monkeypatch, capsys):
    """Layer 2 fires before run_base_check uses the base."""
    env = base_worktree_env

    # Create and dirty the base
    base_path, _, _ = base_check._ensure_base_worktree(env.ctx)
    (base_path / "tests" / "test_math.py").write_text("CORRUPTED\n")

    # Patch run_command for the base check
    ok_result = sandbox.SandboxResult(ok=True, returncode=0,
                                      stdout="all passed\n", stderr="")
    monkeypatch.setattr(sandbox, "run_command", lambda *a, **kw: ok_result)

    result = base_check.run_base_check(env.ctx, "python -m pytest -q", 60)

    # The base should have been repaired before the check ran
    expected = worktree.git(
        env.repo, "show", f"{env.base_commit}:tests/test_math.py"
    ).encode()
    assert (base_path / "tests" / "test_math.py").read_bytes() == expected

    captured = capsys.readouterr()
    assert "run warning:" in captured.err

    # Cache skipped when base was dirty (base_clean_flag=False).
    assert not (env.run_dir / base_check.CACHE_FILE).exists(), \
        "base-check cache should not be written when base was dirty"


def test_dirty_base_through_compare_prep_failure(base_worktree_env, monkeypatch, capsys):
    """Layer 2 fires before compare_prep_failure uses the base."""
    env = base_worktree_env

    # Create and dirty the base
    base_path, _, _ = base_check._ensure_base_worktree(env.ctx)
    (base_path / "tests" / "test_math.py").write_text("CORRUPTED\n")

    # Set up facts for compare_prep_failure
    env.ctx.facts["generate"] = "echo gen"
    env.ctx.facts["setup"] = ""

    # Patch run_command for the prep commands
    ok_result = sandbox.SandboxResult(ok=True, returncode=0,
                                      stdout="ok\n", stderr="")
    monkeypatch.setattr(sandbox, "run_command", lambda *a, **kw: ok_result)

    found = worktree.recorded(env.ctx.workdir)
    prep_fail = {"step": "echo gen", "returncode": 1, "timed_out": False}
    base_check.compare_prep_failure(
        env.ctx, prep_fail, ["gen failed"], found,
        worktree.check_environment(Path(found.path), env.run_dir / "home"),
        env.run_dir / "home", 60.0,
    )

    # The base should have been repaired
    expected = worktree.git(
        env.repo, "show", f"{env.base_commit}:tests/test_math.py"
    ).encode()
    assert (base_path / "tests" / "test_math.py").read_bytes() == expected

    captured = capsys.readouterr()
    assert "run warning:" in captured.err

    # Prep cache skipped when base was dirty (prep_base_clean=False).
    assert not (env.run_dir / base_check.PREP_CACHE_FILE).exists(), \
        "base-prep cache should not be written when base was dirty"


# ── Test 4b: reset failure rebuilds, not environment failure ────────────


def test_reset_failure_rebuilds_not_environment_failure(base_worktree_env, monkeypatch):
    """A reset that fails never becomes a cached environment failure."""
    env = base_worktree_env

    # Create and dirty the base
    base_path, _, _ = base_check._ensure_base_worktree(env.ctx)
    (base_path / "tests" / "test_math.py").write_text("CORRUPTED\n")

    # Make git reset --hard fail
    orig_git = worktree.git
    def failing_git(repo, *args, **kwargs):
        if "reset" in args and "--hard" in args:
            raise ValueError("simulated reset failure")
        return orig_git(repo, *args, **kwargs)
    monkeypatch.setattr(worktree, "git", failing_git)

    # _ensure_base_worktree should rebuild, not become environment_failure
    base_path2, base_found2, repaired = base_check._ensure_base_worktree(env.ctx)

    # The worktree should exist and be clean (rebuilt from scratch)
    head = worktree.git(base_path2, "rev-parse", "HEAD").strip()
    assert head == env.base_commit

    # No cache entries should exist
    assert not (env.run_dir / base_check.CACHE_FILE).exists()
    assert not (env.run_dir / base_check.PREP_CACHE_FILE).exists()


def test_fast_ops_base_reset_in_check_record(base_worktree_env, monkeypatch):
    """fast_ops writes base_reset to check-record.json when ctx._base_reset is set."""
    from refactory_dsh import fast_ops, repo_change
    from refactory_dsh import authoring_evidence as evidence
    from refactory_dsh import unbuilt_check

    env = base_worktree_env

    # Set up ctx with necessary facts and declarations for focused_check.
    env.ctx.facts["artifact"] = str(env.attempt_path)
    env.ctx.facts["check"] = "python -m pytest -q"
    env.ctx.facts["name"] = "test"
    env.ctx.facts["brief"] = "test"
    env.ctx.facts["files"] = []
    env.ctx.knobs_dict["check_timeout_s"] = 60
    env.ctx.knobs_dict["repair_attempts"] = 1
    env.ctx._declared = (
        "focused-check.json", "focused-source.json",
        "authoring-evidence.json", "check-record.json",
        "check.txt", "test-guardedness.json",
    )
    env.ctx._declared_facts = (
        "focused_check", "output_path", "review_files",
        "runtime_evidence", "inherited_failures", "base_unavailable",
    )
    env.ctx.enforce_writes = False

    # Pre-set _base_reset to simulate dirty base repair from an earlier call.
    env.ctx._base_reset = ["tests/test_math.py"]

    # Provide required files.
    (env.run_dir / "focused-check.json").write_text("{}")
    (env.run_dir / "focused-source.json").write_text("{}")
    (env.run_dir / "authoring-evidence.json").write_text("{}")

    # Mock run_check to return a failing result.
    fail_result = sandbox.SandboxResult(
        ok=False, returncode=1,
        stdout="FAILED tests/test_math.py::test_add\n1 failed in 0.1s\n",
        stderr="",
    )
    monkeypatch.setattr(repo_change, "run_check", lambda *a, **kw: fail_result)

    # Mock evidence functions.
    monkeypatch.setattr(evidence, "record", lambda *a, **kw: None)
    monkeypatch.setattr(evidence, "check_failure_kind", lambda *a, **kw: "code_failure")

    # Mock filter_check to return new failures (not all excused).
    filtered = base_check.FilteredCheckResult(
        all_excused=False,
        inherited=set(),
        new_failures={"tests/test_math.py::test_add"},
        new_failure_detail="FAILED tests/test_math.py::test_add",
        inherited_detail="",
        base_environment_failure=None,
        guardedness=None,
        base_failures=set(),
        base_complete=True,
    )
    monkeypatch.setattr(base_check, "filter_check", lambda *a, **kw: filtered)
    monkeypatch.setattr(base_check, "compute_test_guardedness", lambda *a, **kw: None)

    # Mock unbuilt_check.
    monkeypatch.setattr(unbuilt_check, "scan_unbuilt_packages", lambda *a, **kw: [])

    fast_ops.focused_check(env.ctx)

    # Check that check-record.json has base_reset.
    cr = json.loads((env.run_dir / "check-record.json").read_text())
    assert cr.get("base_reset") == ["tests/test_math.py"], (
        f"check-record.json should have base_reset, got: {cr}"
    )


def test_design_ops_base_reset_in_check_record(base_worktree_env, monkeypatch):
    """design_ops writes base_reset to check-record.json when ctx._base_reset is set."""
    from refactory_dsh import design_ops, repo_change
    from refactory_dsh import authoring_evidence as evidence
    from refactory_dsh import unbuilt_check
    from refactory_dsh import failfast
    from refactory_dsh import check_digest as cd_mod
    from refactory_dsh import design_contract as dc_mod
    from refactory_dsh import create_ops

    env = base_worktree_env

    env.ctx.facts["artifact"] = str(env.attempt_path)
    env.ctx.facts["check"] = "python -m pytest -q"
    env.ctx.facts["name"] = "test"
    env.ctx.facts["brief"] = "test"
    env.ctx.facts["files"] = []
    env.ctx.facts["setup"] = ""
    env.ctx.facts["generate"] = ""
    env.ctx.facts["commit"] = "abc123"
    env.ctx.facts["attempts_used"] = 1
    env.ctx.knobs_dict["check_timeout_s"] = 60
    env.ctx.knobs_dict["check_attempts"] = 3
    env.ctx._base_reset = ["tests/test_math.py"]
    env.ctx.enforce_writes = False

    (env.run_dir / "check-record.json").write_text("{}")

    monkeypatch.setattr(repo_change, "refused", lambda *a, **kw: None)

    class _Prepass:
        early_stop = False
        ran = False
        targeted_command = None
        result = None
        runners = []
        digest_text = ""
        raw_tail = ""

    monkeypatch.setattr(failfast, "targeted_prepass", lambda *a, **kw: _Prepass())

    fail_result = sandbox.SandboxResult(
        ok=False, returncode=1,
        stdout="FAILED tests/test_math.py::test_add\n1 failed in 0.1s\n",
        stderr="",
    )
    monkeypatch.setattr(repo_change, "run_check", lambda *a, **kw: fail_result)
    monkeypatch.setattr(design_ops, "apply_cwd_escaped", lambda *a, **kw: None)
    monkeypatch.setattr(evidence, "record", lambda *a, **kw: None)
    monkeypatch.setattr(evidence, "check_failure_kind", lambda *a, **kw: "code_failure")
    monkeypatch.setattr(base_check, "compute_test_guardedness", lambda *a, **kw: None)

    filtered = base_check.FilteredCheckResult(
        all_excused=False, inherited=set(),
        new_failures={"tests/test_math.py::test_add"},
        new_failure_detail="FAILED tests/test_math.py::test_add",
        inherited_detail="",
        base_environment_failure=None,
        guardedness=None,
        base_failures=set(), base_complete=True,
    )
    monkeypatch.setattr(base_check, "filter_check", lambda *a, **kw: filtered)
    monkeypatch.setattr(unbuilt_check, "scan_unbuilt_packages", lambda *a, **kw: [])

    class _Digest:
        raw_tail = ""
        def to_dict(self):
            return {"runners": []}

    monkeypatch.setattr(cd_mod, "digest_check_output", lambda *a, **kw: _Digest())
    monkeypatch.setattr(cd_mod, "format_digest", lambda *a, **kw: "")

    class _Parsed:
        rule_sites = []

    monkeypatch.setattr(dc_mod, "parse", lambda *a, **kw: _Parsed())
    monkeypatch.setattr(failfast, "detect_stuck", lambda *a, **kw: None)
    monkeypatch.setattr(failfast, "record_check_history", lambda *a, **kw: None)
    monkeypatch.setattr(create_ops, "repair", lambda *a, **kw: "reloop")

    design_ops.check_artifact(env.ctx)

    cr = json.loads((env.run_dir / "check-record.json").read_text())
    assert cr.get("base_reset") == ["tests/test_math.py"], (
        f"check-record.json should have base_reset, got: {cr}"
    )


def test_compare_prep_failure_ignores_dirty_cache(base_worktree_env, monkeypatch):
    """compare_prep_failure re-runs the base when a cached entry has base_clean=False.

    If the _read_prep_cache guard is removed, the dirty cache entry (base_ok=False,
    same step) causes _resolve_prep_comparison to return None (inherited).  With
    the guard, the cache is ignored, the base is re-prepared (succeeds), and the
    result is a change-caused dict (not None).
    """
    env = base_worktree_env

    env.ctx.facts["generate"] = "echo gen"
    env.ctx.facts["setup"] = ""

    found = worktree.recorded(env.ctx.workdir)
    key = base_check.cache_key(found.base, "", "echo gen", "")

    # Pre-seed with base_ok=False (same step) — if used, result would be None.
    base_check._write_prep_cache(env.ctx.workdir, key, {
        "base_ok": False,
        "base_step": "echo gen",
        "base_error": "gen failed",
        "base_returncode": 1,
    }, base_clean=True)
    # Tamper: set base_clean to False so it looks like a dirty-base entry.
    cache_path = env.ctx.workdir / base_check.PREP_CACHE_FILE
    data = json.loads(cache_path.read_text())
    data["base_clean"] = False
    cache_path.write_text(json.dumps(data))

    ok_result = sandbox.SandboxResult(ok=True, returncode=0,
                                       stdout="ok\n", stderr="")
    monkeypatch.setattr(sandbox, "run_command", lambda *a, **kw: ok_result)

    prep_fail = {"step": "echo gen", "returncode": 1, "timed_out": False}
    result = base_check.compare_prep_failure(
        env.ctx, prep_fail, ["gen failed"], found,
        worktree.check_environment(Path(found.path), env.run_dir / "home"),
        env.run_dir / "home", 60.0,
    )

    # If the dirty cache was used: base_ok=False, same step → None (inherited).
    # If the dirty cache was ignored: actual base prep succeeds → change-caused dict.
    assert result is not None, (
        "compare_prep_failure should have re-run the base (dirty cache ignored) "
        "and returned a change-caused dict"
    )


# ── Test 5: dirty base cache miss ──────────────────────────────────────


def test_dirty_base_check_cache_miss(tmp_path):
    """A cached environment failure computed on a dirty base is not replayed."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    key = "test_key_123"

    # Write a valid cache entry
    result = base_check.BaseCheckResult(
        ok=False, failures=set(), output="base failed",
        exit_code=1, environment_failure="docs:generate failed",
    )
    base_check._write_cache(run_dir, key, result, base_clean=True)

    # Verify it can be read
    cached = base_check._read_cache(run_dir, key)
    assert cached is not None

    # Tamper: set base_clean to false
    cache_path = run_dir / base_check.CACHE_FILE
    data = json.loads(cache_path.read_text())
    data["base_clean"] = False
    cache_path.write_text(json.dumps(data))

    cached = base_check._read_cache(run_dir, key)
    assert cached is None, "dirty-base cache entry should be a miss"

    # Tamper: remove base_clean entirely (old format)
    data.pop("base_clean")
    cache_path.write_text(json.dumps(data))

    cached = base_check._read_cache(run_dir, key)
    assert cached is None, "old-format cache entry (no base_clean) should be a miss"


def test_dirty_base_prep_cache_miss(tmp_path):
    """A cached base-prep result computed on a dirty base is not replayed."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    key = "test_prep_key_123"

    base_check._write_prep_cache(run_dir, key, {
        "base_ok": False, "base_step": None,
        "base_error": "tsc failed", "base_returncode": 1,
    }, base_clean=True)

    # Verify it can be read
    cached = base_check._read_prep_cache(run_dir, key)
    assert cached is not None

    # Tamper: set base_clean to false
    cache_path = run_dir / base_check.PREP_CACHE_FILE
    data = json.loads(cache_path.read_text())
    data["base_clean"] = False
    cache_path.write_text(json.dumps(data))

    cached = base_check._read_prep_cache(run_dir, key)
    assert cached is None, "dirty-base prep cache entry should be a miss"

    # Tamper: remove base_clean entirely (old format)
    data.pop("base_clean")
    cache_path.write_text(json.dumps(data))

    cached = base_check._read_prep_cache(run_dir, key)
    assert cached is None, "old-format prep cache entry should be a miss"


def test_dirty_base_cache_not_written(tmp_path, capsys):
    """Cache write is skipped when base_clean=False."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    key = "test_no_write_key"

    result = base_check.BaseCheckResult(
        ok=False, failures=set(), output="base failed",
        exit_code=1, environment_failure="env fail",
    )
    base_check._write_cache(run_dir, key, result, base_clean=False)
    assert not (run_dir / base_check.CACHE_FILE).exists(), \
        "cache should not be written when base_clean=False"

    captured = capsys.readouterr()
    assert "run warning:" in captured.err

    base_check._write_prep_cache(run_dir, key, {
        "base_ok": False, "base_step": None,
        "base_error": "fail", "base_returncode": 1,
    }, base_clean=False)
    assert not (run_dir / base_check.PREP_CACHE_FILE).exists(), \
        "prep cache should not be written when base_clean=False"


# ── Test 6: end-to-end design and fast survive guarded test ─────────────


def _make_e2e_repo(tmp_path: Path) -> Path:
    """Create a repo with a TS helper and a Python test linked from a doc,
    plus generate/setup steps that read the copied files."""
    repo = _git_init(tmp_path / "repo")

    # .gitignore
    _commit(repo, ".gitignore", "node_modules/\n.venv/\n__pycache__/\n")

    # A TS helper that other tests import
    _commit(repo, "tests/helpers/graph-shapes.ts",
            "export const square = (n: number) => n * n;\n")

    # A dependent test that imports the helper
    _commit(repo, "tests/test_imports.py",
            "import os, sys\n"
            "def test_helper_exists():\n"
            "    assert os.path.exists('tests/helpers/graph-shapes.ts')\n")

    # A Python test linked from a doc
    _commit(repo, "tests/test_linked.py",
            "def test_linked():\n    assert True\n")
    _commit(repo, "docs/api.md",
            "See [tests/test_linked.py](../tests/test_linked.py) for examples.\n")

    # install.sh — no-op
    (repo / "install.sh").write_text("#!/bin/sh\nexit 0\n")
    (repo / "install.sh").chmod(0o755)
    _commit(repo, "install.sh", "#!/bin/sh\nexit 0\n")

    # A generate step that checks the helper exists (stands in for tsc -b).
    # This makes a missing helper cause an environment failure on the base.
    _commit(repo, "generate.sh",
            "#!/bin/sh\n"
            "if [ ! -f tests/helpers/graph-shapes.ts ]; then\n"
            "  echo 'ERROR: tests/helpers/graph-shapes.ts missing' >&2\n"
            "  exit 1\n"
            "fi\n"
            "if [ ! -f tests/test_linked.py ]; then\n"
            "  echo 'ERROR: tests/test_linked.py missing (doc link broken)' >&2\n"
            "  exit 1\n"
            "fi\n"
            "exit 0\n")
    (repo / "generate.sh").chmod(0o755)

    # app.py
    _commit(repo, "app.py", "print('app')\n")

    return repo


def test_e2e_design_survives_guarded_test(tmp_path, monkeypatch):
    """DesignMode: an attempt that changes a TS helper and a linked Python test
    gets a base comparison and goes back to the builder (not an environment failure)."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from refactory_dsh.design import DesignMode
    from tests_support import (FakeModel, render, DESIGN_DOC_ANSWER,
                               PITFALLS_ANSWER,
                               RESEARCH_ANSWER, VERIFICATION_ANSWER,
                               REVIEW_ANSWERS)

    repo = _make_e2e_repo(tmp_path)

    # First attempt: modifies the TS helper and adds a failing test
    first_emit = render({
        "tests/helpers/graph-shapes.ts": (
            "export const square = (n: number) => n * n;\n"
            "export const cube = (n: number) => n * n * n;\n"
        ),
        "tests/test_linked.py": (
            "def test_linked():\n    assert True\n"
            "def test_new():\n    raise SystemExit(1)\n"
        ),
        "app.py": "raise SystemExit(1)\n",
    })
    # Second attempt: passes
    second_emit = render({
        "tests/helpers/graph-shapes.ts": (
            "export const square = (n: number) => n * n;\n"
            "export const cube = (n: number) => n * n * n;\n"
        ),
        "tests/test_linked.py": (
            "def test_linked():\n    assert True\n"
            "def test_new():\n    assert 1 + 1 == 2\n"
        ),
        "app.py": "print('fixed')\n",
    })

    answers = [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER,
        # The first attempt's failing test goes back from the self-check, before any
        # review runs (b41a); the reviews run once, on the second emission.
        first_emit, second_emit,
        *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3],
    ]
    # Allow enough answers for a two-attempt run
    answers = answers + [""] * 20

    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["check_attempts"]["value"] = 3
    mode.load_state_dict(state)

    result = mode.run(
        tmp_path / "run",
        model=FakeModel(answers),
        verdicts={"gate_strategy": "proceed"},
        task={
            "name": "e2e_base_restore",
            "brief": "a tool that uses helpers",
            "check": "python -m pytest -q",
            "target": str(repo),
            "generate": "sh generate.sh",
        },
    )

    # The run must succeed (possibly after rework), NOT stop as environment failure
    assert result.exit_ok is True, (
        f"expected exit_ok, got error: {result.error}"
    )


def test_e2e_fast_survives_guarded_test(tmp_path, monkeypatch):
    """DesignFastMode: same scenario — goes back to builder, not environment failure."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from refactory_dsh.fast import DesignFastMode
    from tests_support import (FakeModel, render, VERIFICATION_ANSWER)

    repo = _make_e2e_repo(tmp_path)

    FAST_QA = json.dumps({
        "status": "pass",
        "checks": [{"name": "runs", "command": "python -m pytest -q",
                     "status": "pass", "evidence": "exit 0"}],
        "findings": [], "not_run": [],
    })

    first_emit = render({
        "tests/helpers/graph-shapes.ts": (
            "export const square = (n: number) => n * n;\n"
            "export const cube = (n: number) => n * n * n;\n"
        ),
        "tests/test_linked.py": (
            "def test_linked():\n    assert True\n"
            "def test_new():\n    raise SystemExit(1)\n"
        ),
        "app.py": "raise SystemExit(1)\n",
    })
    second_emit = render({
        "tests/helpers/graph-shapes.ts": (
            "export const square = (n: number) => n * n;\n"
            "export const cube = (n: number) => n * n * n;\n"
        ),
        "tests/test_linked.py": (
            "def test_linked():\n    assert True\n"
            "def test_new():\n    assert 1 + 1 == 2\n"
        ),
        "app.py": "print('fixed')\n",
    })

    # Since #475 the verifier is a branch of the review fork, which runs only once
    # the check passes: focused QA, then the verifier.
    answers = [
        "Change only the requested file.",   # strategy
        first_emit,                          # first emit (the check fails)
        second_emit,                         # second emit (the check passes)
        FAST_QA,                             # focused QA
        VERIFICATION_ANSWER,                 # verifier
    ]
    answers = answers + [""] * 20

    mode = DesignFastMode()
    state = mode.state_dict()
    state["knobs"]["repair_attempts"]["value"] = 3
    mode.load_state_dict(state)

    result = mode.run(
        tmp_path / "run",
        model=FakeModel(answers),
        task={
            "name": "e2e_fast_restore",
            "brief": "a tool that uses helpers",
            "check": "python -m pytest -q",
            "target": str(repo),
            "generate": "sh generate.sh",
        },
    )

    assert result.exit_ok is True, (
        f"expected exit_ok, got error: {result.error}"
    )


# ── Test 7: end-to-end via DSH runner ───────────────────────────────────


def test_e2e_dsh_runner_survives_guarded_test(tmp_path, monkeypatch):
    """One of the end-to-end cases runs through the DSH runner (run.main)
    as ``python -m refactory_dsh.run …:check_artifact --node check``
    in its own process, so both runtimes are covered."""
    from refactory_dsh.design import DesignMode
    from refactory_dsh import lowering, design_ops, repo_change
    from refactory_dsh.run import run_node, NODE_ENV, WORKDIR_ENV
    from tests_support import render

    repo = _make_e2e_repo(tmp_path)

    # Set up the run directory as if earlier graph nodes have already run.
    run_dir = tmp_path / "run"
    worktree.create(repo, "HEAD", "test7dsh01234", run_dir)
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["check_attempts"]["value"] = 3
    lowering.write_declarations(state, run_dir)

    (run_dir / "facts.json").write_text(json.dumps({
        "name": "e2e_dsh_runner",
        "brief": "a tool that uses helpers",
        "check": "python -m pytest -q",
        "generate": "sh generate.sh",
    }))

    knobs_json = json.dumps({
        "check_attempts": 3, "check_timeout_s": 60,
    })
    monkeypatch.setenv("REFACTORY_KNOBS", knobs_json)

    # Write the first attempt: modifies the TS helper and the linked test,
    # adds a deliberately failing test so the check fails.
    emit_text = render({
        "tests/helpers/graph-shapes.ts": (
            "export const square = (n: number) => n * n;\n"
            "export const cube = (n: number) => n * n * n;\n"
        ),
        "tests/test_linked.py": (
            "def test_linked():\n    assert True\n"
            "def test_new():\n    raise SystemExit(1)\n"
        ),
        "app.py": "raise SystemExit(1)\n",
    })
    (run_dir / "emit.txt").write_text(emit_text)

    # Materialise the emit in-process (apply changes to the worktree).
    run_node(
        f"{repo_change.OPS}:materialise",
        workdir=run_dir, node_id="materialise",
    )

    # Run check_artifact through the DSH runner in its own process.
    # Build the env from the current process (which has conftest's
    # REFACTORY_HOME, REFACTORY_PROJECTS_DIR, etc.).
    env = {**os.environ}
    env[WORKDIR_ENV] = str(run_dir)
    env[NODE_ENV] = json.dumps({"id": "check", "type": "GateNode"})
    env["REFACTORY_KNOBS"] = knobs_json
    env["REFACTORY_PROJECT"] = str(tmp_path)

    proc = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.run",
         f"{design_ops.__name__}:check_artifact", "--node", "check"],
        capture_output=True, text=True, timeout=120,
        env=env,
    )

    # Exit 0 means the gate returned a verdict normally (not an
    # environment-failure ValueError).  On unfixed code the base
    # worktree's generate.sh fails because the helper is missing,
    # evidence.stop_unrepairable raises, and main() returns 1.
    assert proc.returncode == 0, (
        f"DSH runner failed (exit {proc.returncode}):\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
    )

    # facts.json must not contain an environment failure in base_unavailable.
    facts = json.loads((run_dir / "facts.json").read_text())
    base_unavailable = facts.get("base_unavailable", "")
    assert "the base check could not run" not in base_unavailable, (
        f"base check became environment failure: {base_unavailable}"
    )

    # The verdict (last non-empty stdout line) should be "reloop"
    # (a failing check that goes back to the builder), not "stop".
    stdout_lines = [
        l.strip() for l in proc.stdout.strip().splitlines() if l.strip()
    ]
    assert stdout_lines, "DSH runner produced no stdout"
    assert stdout_lines[-1] == "reloop", (
        f"expected reloop verdict, got {stdout_lines[-1]!r}:\n"
        f"stderr: {proc.stderr[-2000:]}"
    )


# ── Per-step recovery tests ─────────────────────────────────────────────
# guardedness-exempt: these tests pin new per-step recovery behaviour that
# does not exist on the base commit.


def _make_per_step_ctx(tmp_path: Path) -> tuple[_Ctx, Path]:
    """Set up a minimal repo + run_dir + ctx for per-step recovery tests."""
    repo = _git_init(tmp_path / "repo")
    _commit(repo, ".gitignore", "node_modules/\n.venv/\n__pycache__/\n")
    base_commit = _commit(repo, "app.py", "print('app')\n")
    run_dir = tmp_path / "run"
    found = worktree.create(repo, base_commit, "perstep001", run_dir)
    ctx = _Ctx(run_dir, facts={"check": "true", "setup": "", "generate": ""})
    return ctx, Path(found.path)


# 1. test_per_step_recovery_finds_failures_hidden_by_early_step
def test_per_step_recovery_finds_failures_hidden_by_early_step(tmp_path, monkeypatch):
    """Command ``A && pytest X && grep -q Y f``.

    The whole-command output has no test output (the canary-repair shape from
    Gate 1). The base's whole run fails on A so later steps never run and
    ``failures`` is empty.  Per-step recovery runs each step individually:
    pytest finds one shared failure.  Then ``filter_check`` / ``design_comparison``
    with an attempt that fails A, the same pytest test, a new pytest test, and the
    grep: the shared pytest failure is inherited, the new pytest failure and the
    grep remain the attempt's.
    """
    ctx, attempt_path = _make_per_step_ctx(tmp_path)
    command = "bash canary-repair-check.sh tok && pytest tests/ && grep -q marker output.txt"

    call_count = 0

    def fake_run_command(cmd, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Whole-command run: first step fails, no test output at all
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout="canary-repair-check.sh: line 3: command failed\n",
                stderr="",
            )
        # Per-step recovery calls:
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        if "canary-repair-check" in cmd_str:
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout="canary-repair-check.sh: line 3: command failed\n",
                stderr="",
            )
        if "pytest" in cmd_str:
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout=(
                    "FAILED tests/test_math.py::test_shared\n"
                    "======= 1 failed in 0.5s =======\n"
                ),
                stderr="",
            )
        if "grep" in cmd_str:
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout="", stderr="",
            )
        return sandbox.SandboxResult(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox, "run_command", fake_run_command)
    # Bypass worktree preparation
    monkeypatch.setattr(
        base_check, "_ensure_base_worktree",
        lambda ctx: (attempt_path, worktree.recorded(ctx.workdir), []),
    )
    monkeypatch.setattr("refactory_dsh.repo_change.prepare_commands",
                        lambda *a, **kw: (None, [], None))

    result = base_check.run_base_check(ctx, command, 60.0)

    # Per-step recovery should have found the pytest failure
    assert "tests/test_math.py::test_shared" in result.failures
    assert result._base_complete is True

    # Now test through filter_check / design_comparison with an attempt that
    # has: same test, a new test, and failing grep output
    attempt_result = sandbox.SandboxResult(
        ok=False, returncode=1,
        stdout=(
            "FAILED tests/test_math.py::test_shared\n"
            "FAILED tests/test_math.py::test_new_one\n"
            "======= 2 failed in 0.5s =======\n"
            "grep: output.txt: No such file or directory\n"
        ),
        stderr="",
    )
    named_text = "implement the math helpers"

    # Mock run_base_check to return our already-computed result
    monkeypatch.setattr(base_check, "run_base_check", lambda *a, **kw: result)

    filtered = base_check.filter_check(
        ctx, attempt_result, command, 60.0,
        named_text=named_text,
    )

    # The shared failure is inherited
    assert "tests/test_math.py::test_shared" in filtered.inherited
    # The new failure stays
    assert "tests/test_math.py::test_new_one" in filtered.new_failures
    # base_complete is True
    assert filtered.base_complete is True


# 2. test_per_step_skipped_when_whole_command_complete
def test_per_step_skipped_when_whole_command_complete(tmp_path, monkeypatch):
    """A single-step ``pytest tests/`` whose whole-command output has a matching
    summary: ``_base_complete`` is True and ``sandbox.run_command`` is called
    exactly once (no per-step recovery)."""
    ctx, attempt_path = _make_per_step_ctx(tmp_path)
    command = "pytest tests/"

    calls = []

    def fake_run_command(cmd, **kw):
        calls.append(cmd)
        return sandbox.SandboxResult(
            ok=False, returncode=1,
            stdout=(
                "FAILED tests/test_math.py::test_add\n"
                "======= 1 failed in 0.3s =======\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(sandbox, "run_command", fake_run_command)
    monkeypatch.setattr(
        base_check, "_ensure_base_worktree",
        lambda ctx: (attempt_path, worktree.recorded(ctx.workdir), []),
    )
    monkeypatch.setattr("refactory_dsh.repo_change.prepare_commands",
                        lambda *a, **kw: (None, [], None))

    result = base_check.run_base_check(ctx, command, 60.0)

    assert result._base_complete is True
    assert len(calls) == 1, f"expected 1 run_command call, got {len(calls)}"


# 3. test_passing_chain_skips_per_step_recovery
def test_passing_chain_skips_per_step_recovery(tmp_path, monkeypatch):
    """Command ``bash setup.sh && pytest tests/ && grep -q X f``.

    Whole-command run passes (``check_result.ok``). ``_base_complete`` is True
    and ``sandbox.run_command`` called exactly once (no per-step recovery).
    """
    ctx, attempt_path = _make_per_step_ctx(tmp_path)
    command = "bash setup.sh && pytest tests/ && grep -q X f"

    calls = []

    def fake_run_command(cmd, **kw):
        calls.append(cmd)
        return sandbox.SandboxResult(
            ok=True, returncode=0,
            stdout="all passed\n", stderr="",
        )

    monkeypatch.setattr(sandbox, "run_command", fake_run_command)
    monkeypatch.setattr(
        base_check, "_ensure_base_worktree",
        lambda ctx: (attempt_path, worktree.recorded(ctx.workdir), []),
    )
    monkeypatch.setattr("refactory_dsh.repo_change.prepare_commands",
                        lambda *a, **kw: (None, [], None))

    result = base_check.run_base_check(ctx, command, 60.0)

    assert result._base_complete is True
    assert len(calls) == 1, f"expected 1 run_command call, got {len(calls)}"


# 4. test_unavailable_base_inherits_nothing
def test_unavailable_base_inherits_nothing(tmp_path, monkeypatch):
    """``_ensure_base_worktree`` raises ``ValueError``.  ``result.environment_failure``
    is set, ``result.failures`` is empty, no ``sandbox.run_command`` calls."""
    ctx, attempt_path = _make_per_step_ctx(tmp_path)
    command = "pytest tests/"

    calls = []
    orig_run = sandbox.run_command

    def counting_run(cmd, **kw):
        calls.append(cmd)
        return orig_run(cmd, **kw)

    monkeypatch.setattr(sandbox, "run_command", counting_run)
    monkeypatch.setattr(
        base_check, "_ensure_base_worktree",
        lambda ctx: (_ for _ in ()).throw(ValueError("cannot prepare base")),
    )

    result = base_check.run_base_check(ctx, command, 60.0)

    assert result.environment_failure is not None
    assert len(result.failures) == 0
    assert len(calls) == 0


# 5. test_pnpm_test_vitest_detected_from_output
def test_pnpm_test_vitest_detected_from_output(tmp_path, monkeypatch):
    """Command ``pnpm test && pytest tests/``.

    The whole-command run fails early (pnpm exits before vitest produces its
    summary), so the whole-command output has no vitest FAIL lines.
    ``_runner('pnpm test')`` returns ``None``; per-step ``pnpm test`` output
    contains vitest summary.  ``_runner_from_output`` detects ``'vitest'``.
    The vitest failure appears only via per-step recovery.
    Mutation-checked: removing ``_runner_from_output`` fallback drops the
    vitest failure from ``result.failures``.
    """
    ctx, attempt_path = _make_per_step_ctx(tmp_path)
    command = "pnpm test && pytest tests/"

    call_count = 0

    def fake_run_command(cmd, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Whole-command run: pnpm test fails early with a build error,
            # no vitest FAIL lines in the output
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout="ERR_PNPM_RECURSIVE_RUN_FIRST_FAIL in /app\n",
                stderr="",
            )
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        if "pnpm" in cmd_str:
            # Per-step: pnpm test actually runs vitest
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout=(
                    " FAIL  src/utils.test.ts > Utils > should validate\n"
                    " Tests  1 failed\n"
                    " Test Files  1 failed\n"
                ),
                stderr="",
            )
        if "pytest" in cmd_str:
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout=(
                    "FAILED tests/test_math.py::test_add\n"
                    "======= 1 failed in 0.3s =======\n"
                ),
                stderr="",
            )
        return sandbox.SandboxResult(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox, "run_command", fake_run_command)
    monkeypatch.setattr(
        base_check, "_ensure_base_worktree",
        lambda ctx: (attempt_path, worktree.recorded(ctx.workdir), []),
    )
    monkeypatch.setattr("refactory_dsh.repo_change.prepare_commands",
                        lambda *a, **kw: (None, [], None))

    result = base_check.run_base_check(ctx, command, 60.0)

    # Vitest failure detected via _runner_from_output during per-step recovery
    assert "src/utils.test.ts > Utils > should validate" in result.failures
    # Pytest failure also detected
    assert "tests/test_math.py::test_add" in result.failures
    assert result._base_complete is True

    # Now test inheritance through filter_check: attempt has same vitest
    # failure plus a new one
    attempt_result = sandbox.SandboxResult(
        ok=False, returncode=1,
        stdout=(
            " FAIL  src/utils.test.ts > Utils > should validate\n"
            " FAIL  src/utils.test.ts > Utils > should format\n"
            " Tests  2 failed\n"
            " Test Files  1 failed\n"
            "FAILED tests/test_math.py::test_add\n"
            "======= 1 failed in 0.3s =======\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(base_check, "run_base_check", lambda *a, **kw: result)
    filtered = base_check.filter_check(
        ctx, attempt_result, command, 60.0,
        named_text="implement the utils module",
    )

    # Shared vitest failure is inherited
    assert "src/utils.test.ts > Utils > should validate" in filtered.inherited
    # New vitest failure stays
    assert "src/utils.test.ts > Utils > should format" in filtered.new_failures


# 6. test_base_complete_requires_accounted_summary
def test_base_complete_requires_accounted_summary(tmp_path, monkeypatch):
    """Command ``bash setup.sh && pytest tests/``.  Per-step pytest returns 2
    failures but summary says ``1 failed``.  ``_accounted`` returns False.
    ``result._base_complete`` is False but both failures are in ``result.failures``.
    """
    ctx, attempt_path = _make_per_step_ctx(tmp_path)
    command = "bash setup.sh && pytest tests/"

    call_count = 0

    def fake_run_command(cmd, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Whole-command run: setup fails
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout="setup.sh failed\n", stderr="",
            )
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        if "setup" in cmd_str:
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout="setup.sh failed\n", stderr="",
            )
        if "pytest" in cmd_str:
            # 2 failures but summary claims 1
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout=(
                    "FAILED tests/test_a.py::test_one\n"
                    "FAILED tests/test_a.py::test_two\n"
                    "======= 1 failed in 0.5s =======\n"
                ),
                stderr="",
            )
        return sandbox.SandboxResult(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox, "run_command", fake_run_command)
    monkeypatch.setattr(
        base_check, "_ensure_base_worktree",
        lambda ctx: (attempt_path, worktree.recorded(ctx.workdir), []),
    )
    monkeypatch.setattr("refactory_dsh.repo_change.prepare_commands",
                        lambda *a, **kw: (None, [], None))

    result = base_check.run_base_check(ctx, command, 60.0)

    assert result._base_complete is False
    assert "tests/test_a.py::test_one" in result.failures
    assert "tests/test_a.py::test_two" in result.failures


# 7. test_timed_out_base_no_per_step_recovery
def test_timed_out_base_no_per_step_recovery(tmp_path, monkeypatch):
    """Whole-command run returns ``timed_out=True``.  ``sandbox.run_command``
    called exactly once and ``result._base_complete`` is False."""
    ctx, attempt_path = _make_per_step_ctx(tmp_path)
    command = "bash setup.sh && pytest tests/"

    calls = []

    def fake_run_command(cmd, **kw):
        calls.append(cmd)
        return sandbox.SandboxResult(
            ok=False, returncode=124,
            stdout="", stderr="timed out",
            timed_out=True,
        )

    monkeypatch.setattr(sandbox, "run_command", fake_run_command)
    monkeypatch.setattr(
        base_check, "_ensure_base_worktree",
        lambda ctx: (attempt_path, worktree.recorded(ctx.workdir), []),
    )
    monkeypatch.setattr("refactory_dsh.repo_change.prepare_commands",
                        lambda *a, **kw: (None, [], None))

    result = base_check.run_base_check(ctx, command, 60.0)

    assert result._base_complete is False
    assert len(calls) == 1


# 8. test_per_step_cache_roundtrip
def test_per_step_cache_roundtrip(tmp_path, monkeypatch):
    """A BaseCheckResult with per-step-enriched failures and ``_base_complete=True``
    survives cache write / read.  Then mock ``_ensure_base_worktree`` to raise
    and call ``run_base_check``: the cached result is returned without calling
    ``_ensure_base_worktree``.  No new cache fields (Gate 4)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    result = base_check.BaseCheckResult(
        ok=False,
        failures={"tests/test_a.py::test_one", "tests/test_a.py::test_two"},
        output="FAILED tests/test_a.py::test_one\nFAILED tests/test_a.py::test_two\n"
               "======= 2 failed in 0.5s =======\n",
        exit_code=1,
    )
    result._base_complete = True  # type: ignore[attr-defined]

    key = "test_cache_key_42"
    base_check._write_cache(run_dir, key, result)

    cached = base_check._read_cache(run_dir, key)
    assert cached is not None
    assert cached.failures == result.failures
    assert cached._base_complete is True

    # Verify no new cache fields (Gate 4)
    cache_data = json.loads((run_dir / base_check.CACHE_FILE).read_text())
    expected_keys = {"cache_key", "ok", "failures", "output", "exit_code",
                     "environment_failure", "base_complete", "base_clean"}
    assert set(cache_data.keys()) == expected_keys

    # Now set up a ctx where _ensure_base_worktree would raise
    repo = _git_init(tmp_path / "repo")
    base_commit = _commit(repo, "app.py", "print('hello')\n")
    found = worktree.create(repo, base_commit, "cache_rt01", run_dir)
    ctx = _Ctx(run_dir, facts={"setup": "", "generate": ""})

    # Write cache with the correct key for this ctx
    setup = ""
    generate = ""
    command = "pytest tests/"
    real_key = base_check.cache_key(found.base, setup, generate, command)
    base_check._write_cache(run_dir, real_key, result)

    ensure_called = []

    def raising_ensure(ctx):
        ensure_called.append(True)
        raise ValueError("should not be called — cached")

    monkeypatch.setattr(base_check, "_ensure_base_worktree", raising_ensure)

    cached_result = base_check.run_base_check(ctx, command, 60.0)

    assert len(ensure_called) == 0, "_ensure_base_worktree should not be called"
    assert cached_result.failures == result.failures
    assert cached_result._base_complete is True


# 9. test_runner_from_output_with_ansi
def test_runner_from_output_with_ansi():
    """``_runner_from_output`` with ANSI-escaped vitest, pytest, and jest output.

    Jest input uses ``Tests:       1 failed, 2 passed`` (with colon).
    """
    # pytest with ANSI
    pytest_output = "\x1b[31m======= 1 failed in 0.5s =======\x1b[0m\n"
    assert base_check._runner_from_output(pytest_output) == "pytest"

    # vitest with ANSI
    vitest_output = "\x1b[31m Tests  1 failed\x1b[0m\n\x1b[31m Test Files  1 failed\x1b[0m\n"
    assert base_check._runner_from_output(vitest_output) == "vitest"

    # jest with ANSI (note: colon after "Tests")
    jest_output = "\x1b[1m\x1b[31mTests:\x1b[0m       \x1b[31m1 failed\x1b[0m, 2 passed\n"
    assert base_check._runner_from_output(jest_output) == "jest"

    # No runner
    plain_output = "some shell output\nexit 1\n"
    assert base_check._runner_from_output(plain_output) is None


# 10. test_piped_step_kept_whole
def test_piped_step_kept_whole(tmp_path, monkeypatch):
    """Command ``pytest tests/ | tee log.txt && grep -q X f``.

    Per-step recovery splits into two steps: ``pytest tests/ | tee log.txt``
    (kept whole) and ``grep -q X f``.  Pytest failures from the piped step
    are detected.
    """
    ctx, attempt_path = _make_per_step_ctx(tmp_path)
    command = "pytest tests/ | tee log.txt && grep -q X f"

    call_count = 0
    step_cmds = []

    def fake_run_command(cmd, **kw):
        nonlocal call_count
        call_count += 1
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        if call_count == 1:
            # Whole-command run: fails, last step is grep (not recognised)
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout=(
                    "FAILED tests/test_a.py::test_one\n"
                    "======= 1 failed in 0.3s =======\n"
                ),
                stderr="",
            )
        # Per-step recovery
        step_cmds.append(cmd_str)
        if "pytest" in cmd_str:
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout=(
                    "FAILED tests/test_a.py::test_one\n"
                    "======= 1 failed in 0.3s =======\n"
                ),
                stderr="",
            )
        if "grep" in cmd_str:
            return sandbox.SandboxResult(
                ok=False, returncode=1,
                stdout="", stderr="",
            )
        return sandbox.SandboxResult(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox, "run_command", fake_run_command)
    monkeypatch.setattr(
        base_check, "_ensure_base_worktree",
        lambda ctx: (attempt_path, worktree.recorded(ctx.workdir), []),
    )
    monkeypatch.setattr("refactory_dsh.repo_change.prepare_commands",
                        lambda *a, **kw: (None, [], None))

    result = base_check.run_base_check(ctx, command, 60.0)

    # The pipeline was kept whole
    assert any("pytest tests/ | tee log.txt" in c for c in step_cmds), \
        f"piped step should be kept whole, got: {step_cmds}"
    # No fragmented "tee log.txt" step
    assert not any(c.strip() == "tee log.txt" for c in step_cmds), \
        f"pipeline was fragmented: {step_cmds}"
    # Pytest failures detected from piped step
    assert "tests/test_a.py::test_one" in result.failures
    assert result._base_complete is True
