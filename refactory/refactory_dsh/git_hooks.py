"""Core hooks-isolation helpers for git worktrees.

Shared between pin.py (core) and worktree.py (product).  Both need
:data:`NO_HOOKS` (a ``git -c core.hooksPath=/dev/null`` pair) and
:func:`_isolate_hooks` (per-worktree hooks isolation) but the boundary
forbids core modules from importing product.  This module lives on the
core side of the boundary; worktree re-exports what it needs.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from refactory_dsh import git_exe


#: Given to every git command that changes a worktree so hooks do not run
#: during the operation.  ``--no-verify`` stops only pre-commit and
#: commit-msg; a post-checkout or post-commit hook would still run.
NO_HOOKS = ("-c", "core.hooksPath=/dev/null")


# ── helpers ──────────────────────────────────────────────────────────────

def _git(repo: Path, *args: str) -> str:
    """Run a git command in *repo* and return its stdout.

    Mirrors :func:`worktree.git` in contract so callers move cleanly.
    """
    try:
        result = subprocess.run(
            [git_exe.program(*args), "-C", str(repo), *args],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            f"git {' '.join(args)} could not run in {repo}: {error}"
        ) from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"git {' '.join(args)} failed in {repo}: {detail}")
    return result.stdout


def _config_value(
    config_file: Path, key: str, *, boolean: bool = False,
) -> str | None:
    """Read a single git-config key from a config file.

    Returns the stripped value when the key is set (exit 0), ``None``
    when the key is unset (exit 1), and raises ``ValueError`` on any
    other failure — matching :func:`_git`'s error contract.
    """
    cmd = [git_exe.git(), "config", "--file", str(config_file)]
    if boolean:
        cmd.append("--type=bool")
    cmd += ["--get", key]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            f"git config --get {key} could not run: {error}"
        ) from error
    if result.returncode == 0:
        return result.stdout.strip()
    if result.returncode == 1:
        return None
    raise ValueError(f"git config --get {key} failed: {result.stderr.strip()}")


# ── per-worktree hooks isolation ─────────────────────────────────────────

def _isolate_hooks(target: Path, worktree_path: Path) -> bool:
    """Give the worktree its own hooks path so a hook manager writes there, not to the shared hooks.

    Enables ``extensions.worktreeConfig`` on the target (the only change to the shared
    config) and sets a per-worktree ``core.hooksPath`` pointing to an isolated directory
    inside the worktree's own git dir.  Returns True when isolation was enabled, False
    when skipped because the target's common config sets ``core.bare=true`` or
    ``core.worktree``, which conflict with ``extensions.worktreeConfig``.
    """
    from refactory_dsh import target_guard

    common = Path(
        _git(target, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    )
    config_file = common / "config"

    # ── check for blocking config in the common config file ─────────────
    bare = _config_value(config_file, "core.bare", boolean=True)
    if bare == "true":
        target_guard.warn(
            "hook isolation skipped: target config sets core.bare=true, "
            "which conflicts with extensions.worktreeConfig"
        )
        return False

    worktree_val = _config_value(config_file, "core.worktree")
    if worktree_val is not None:
        target_guard.warn(
            "hook isolation skipped: target config sets core.worktree, "
            "which conflicts with extensions.worktreeConfig"
        )
        return False

    # ── enable extensions.worktreeConfig (read-before-write, lock retry) ──
    ext_val = _config_value(config_file, "extensions.worktreeConfig")
    need_write = ext_val is None or ext_val.lower() != "true"

    if need_write:
        for attempt in range(4):
            try:
                _git(target, "config", "extensions.worktreeConfig", "true")
                break
            except ValueError as exc:
                if (
                    "config.lock" not in str(exc)
                    and "could not lock config file" not in str(exc)
                ):
                    raise
                # Another process may have set it while we waited.
                retry_val = _config_value(
                    config_file, "extensions.worktreeConfig"
                )
                if retry_val is not None and retry_val.lower() == "true":
                    break
                if attempt < 3:
                    time.sleep(0.1)
                else:
                    raise

    # ── per-worktree hooks path ─────────────────────────────────────────
    wt_git_dir = Path(
        _git(worktree_path, "rev-parse", "--absolute-git-dir").strip()
    )
    hooks_dir = wt_git_dir / "hooks.isolated"
    hooks_dir.mkdir(exist_ok=True)
    _git(worktree_path, "config", "--worktree", "core.hooksPath", str(hooks_dir))
    return True
