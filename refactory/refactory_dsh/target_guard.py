"""Watch the target a run works on: say when it moves, refuse writes nobody declared.

The target is the repository a design run changes: the one it names in `target`, the
new project it created, or -- for a run without a worktree -- the project it runs from
(`project_for_nodes()`). A checkout other sessions work in can move under a run (#330
I1), and a worker with full write access can edit files directly instead of answering
through its declared output (#319). A run with its own worktree (`worktree.py`, #351)
never works in the target, so this module is its backstop rather than its isolation:
the baseline is recorded at `prepare`, and every agent node of a worktree run fails
when it changes the target while it ran (dm-159). An output-only agent also fails
when the run's own worktree changed. Every agent node of a recorded run also fails
when it changes refactory's own checkout -- the driver, or the run's pin -- where that
is not the target (:func:`checkouts`, dm-164).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from refactory_dsh.invocations import TARGET_RECORD as RECORD  # was: RECORD = Path(…)
from refactory_dsh.paths import refactory_home

#: Names a walk of a target that is not a git checkout skips, as directories or files:
#: installs, caches and editor state that running anything rewrites, which say nothing
#: about what a worker chose to change. A git checkout is read through its own ignore
#: rules instead.
UNWATCHED = frozenset({".git", ".factory", ".venv", "venv", "node_modules", "__pycache__",
                       ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".idea",
                       ".DS_Store", ".coverage"})

#: The most uncommitted paths a target may have and still be watched. Every one is
#: hashed before and after any agent node, so a run launched from a home
#: directory or an unignored data tree is reported as unwatched rather than walked.
LIMIT = 20_000

#: Files larger than this are compared by size and modification time, not content.
HASHED_BYTES = 16 * 1024 * 1024

#: How many changed paths one message names before it counts the rest.
SHOWN = 8

#: What starts a warning meant for the person, not the log. The workflow spine passes
#: lines carrying it to the CEO under `ran.warnings`, and leaves a routine
#: `refactory: warning:` (an unverifiable credential, say) in the log, where it was.
WARNING = "refactory: run warning: "


def _git(root: Path, *args: str) -> str | None:
    """What a git command prints in ``root``, or None when git cannot answer there.

    ``--no-optional-locks`` keeps `status` from refreshing the index, so watching the
    target never writes to it.
    """
    try:
        result = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                                capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.decode("utf-8", "surrogateescape") if result.returncode == 0 else None


def _digest(path: Path) -> str:
    """What a path holds now: a content hash, or ``-`` when it is gone.

    Content rather than metadata, so a file put back exactly as it was reads as
    unchanged.
    """
    try:
        if path.is_symlink():
            return "symlink:" + os.readlink(path)
        if path.is_dir():
            # git lists an untracked nested checkout as one directory entry.
            return "directory"
        stat = path.stat()
        if stat.st_size > HASHED_BYTES:
            return f"large:{stat.st_size}:{stat.st_mtime_ns}"
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return "-"
    except OSError as error:
        return f"unreadable:{error.errno}"


def _porcelain(listing: str) -> list[str]:
    """The paths `git status --porcelain=v1 -z` names; a rename's old path is skipped.

    A rename or copy is followed by its old path as a field of its own, whether it is
    staged (`R `) or only in the work tree (` R`, after `git add -N`), so both status
    columns are read; reading only the first takes the old path for an entry.
    """
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


def state(root: Path, *, exclude: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Where the target stands: its commit, its branch, and what is uncommitted.

    ``changes`` maps each uncommitted path, relative to the target, to what it holds,
    so a further edit to an already-dirty file is still a change. In a git checkout
    that is what `git status` lists; anywhere else it is every file, there being no
    commit to compare with. The target's ``.factory/`` and the ``exclude``d
    directories (the run's own) are never part of it, because a run writes there by
    design. ``changes`` is None when there are more than :data:`LIMIT` paths to watch.
    """
    root = Path(root).resolve()
    skipped = [root / ".factory", *(Path(path).resolve() for path in exclude)]
    found: dict[str, Any] = {"root": str(root), "git": False, "head": None, "branch": None, "tree": None}
    if root in skipped:
        # The run's own directory is the target: everything in it is the run's.
        return {**found, "changes": {}}
    inside = _git(root, "rev-parse", "--is-inside-work-tree")
    # A directory an enclosing checkout ignores is not that checkout's to report on:
    # its status would list nothing there, and the guard would be off without a word.
    if inside is not None and inside.strip() == "true" and _git(root, "check-ignore", "-q", str(root)) is None:
        top = Path((_git(root, "rev-parse", "--show-toplevel") or str(root)).strip()).resolve()
        specs = [f":(exclude,literal){path.relative_to(root)}" for path in skipped if path.is_relative_to(root)]
        listing = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ".", *specs)
        if listing is None:
            raise ValueError(f"git cannot report the state of the target {root}")
        paths = [os.path.relpath(top / path, root) for path in _porcelain(listing)]
        prefix = "" if root == top else root.relative_to(top).as_posix()
        found.update(git=True, head=(_git(root, "rev-parse", "-q", "--verify", "HEAD") or "").strip() or None,
                     branch=(_git(root, "symbolic-ref", "-q", "--short", "HEAD") or "").strip() or None,
                     # The committed content under the target alone: a commit elsewhere in
                     # the checkout, or a new branch at the same commit, leaves it as it was.
                     tree=(_git(root, "rev-parse", "-q", "--verify", f"HEAD:{prefix}") or "").strip() or None)
    else:
        paths = []
        for directory, names, files in os.walk(root):
            here = Path(directory)
            names[:] = [name for name in names if name not in UNWATCHED and here / name not in skipped]
            paths.extend(str((here / name).relative_to(root)) for name in files if name not in UNWATCHED)
            if len(paths) > LIMIT:
                break
    if len(paths) > LIMIT:
        return {**found, "changes": None}
    return {**found, "changes": {path: _digest(root / path) for path in sorted(paths)}}


def _read(workdir: Path) -> dict[str, Any] | None:
    """The baseline this run recorded, or None when it recorded none."""
    try:
        loaded = json.loads((Path(workdir) / RECORD).read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _commit(sha: str | None) -> str:
    """A commit as a reader recognises it: abbreviated, or said to be absent."""
    return sha[:12] if sha else "no commit"


def _named(paths: list[str]) -> str:
    """A list of paths short enough to read in one message."""
    shown = ", ".join(paths[:SHOWN])
    return shown if len(paths) <= SHOWN else f"{shown} and {len(paths) - SHOWN} more"


def _differ(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """The paths whose content, presence or absence differs between two states."""
    return sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))


def _compare(before: dict[str, Any], after: dict[str, Any], *, commits: bool = True) -> list[str]:
    """What differs between two states of one target, as phrases one message can list.

    With ``commits`` a moved HEAD or branch is a difference of its own. Without, only
    what is in the target is: its files, and its committed content (so a worker that
    commits its edits is still caught), but not a commit elsewhere in the checkout or a
    new branch at the same commit. A target that went over :data:`LIMIT` is itself a
    difference: a flood of new files is exactly what must not pass as no change.
    """
    found = []
    if commits and after["head"] != before.get("head"):
        found.append(f"HEAD moved from {_commit(before.get('head'))} to {_commit(after['head'])}")
    if commits and after["branch"] != before.get("branch"):
        found.append(f"the branch changed from {before.get('branch') or 'a detached HEAD'} "
                     f"to {after['branch'] or 'a detached HEAD'}")
    if not commits and after.get("tree") != before.get("tree"):
        found.append(f"the committed files under it changed (HEAD {_commit(before.get('head'))} "
                     f"is now {_commit(after['head'])})")
    if before.get("changes") is not None and after["changes"] is None:
        found.append(f"more than {LIMIT} paths now differ, too many to compare one by one")
    elif before.get("changes") is not None:
        changed = _differ(before["changes"], after["changes"])
        if changed:
            kind = "uncommitted changes differ" if after["git"] else "files differ"
            found.append(f"{kind} in {_named(changed)}")
    return found


def record(workdir: Path, root: Path, *, output_only: tuple[str, ...] = (),
           also: tuple[Path, ...] = (),
           warn_also: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Record the target's baseline for this run, and which nodes may not touch it.

    ``output_only`` names agent nodes whose only output is their declared file: the
    agent-delivery bracket refuses one of them that changed the target while it ran.
    ``also`` names directories whose changes **fail** an output-only step -- the
    run's own worktree, where a direct edit bypasses materialise.
    ``warn_also`` names directories whose changes produce a run **warning** -- the
    launcher checkout, where another session's edit is not this run's result but
    should be reported.
    """
    workdir = Path(workdir).resolve()
    baseline = {**state(root, exclude=(workdir,)), "output_only": list(output_only),
                "also": [str(Path(path).resolve()) for path in also],
                "warn_also": [str(Path(path).resolve()) for path in warn_also],
                "recorded": datetime.now(timezone.utc).isoformat()}
    path = workdir / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(baseline, indent=2) + "\n")
    pending.replace(path)
    return baseline


def moved(workdir: Path) -> str | None:
    """How the target differs from this run's baseline, or None when it does not."""
    baseline = _read(workdir)
    if baseline is None:
        return None
    try:
        now = state(Path(baseline["root"]), exclude=(Path(workdir),))
    except (OSError, ValueError) as error:
        # This path only warns, so a target it cannot read is reported, not fatal.
        return f"the target {baseline['root']} could not be compared with this run's baseline: {error}"
    differences = _compare(baseline, now)
    if not differences:
        return None
    return (f"the target {baseline['root']} changed since this run recorded it at prepare: "
            f"{'; '.join(differences)}. Another session or person may be working in this "
            "checkout, so the artifact, its check and its reviews may no longer see the "
            "baseline the run started from.")


def warn(message: str) -> None:
    """Tell the person driving the run, through the step that noticed."""
    print(f"{WARNING}{message}", file=sys.stderr)


def warn_if_moved(workdir: Path) -> str | None:
    """Warn when the target moved, so the step that noticed reports it.

    A warning, not a refusal: the change may be the person's own and intended, and
    only they can say whether the run should go on.
    """
    message = moved(workdir)
    if message is not None:
        warn(message)
    return message


def _also_state(path: str, workdir: Path) -> dict[str, Any]:
    """A further watched directory's state, without the environments installs create there."""
    root = Path(path)
    return state(root, exclude=(Path(workdir), root / ".venv", root / "node_modules"))


def watch(workdir: Path, node: str) -> dict[str, Any] | None:
    """The target's state before an agent node runs; None when it is not watched.

    For output-only agents in worktree runs, returns the full snapshot (target +
    also + warn_also).  For non-output-only agents in worktree runs, returns a
    target-only snapshot so ``strayed()`` can refuse target changes while leaving
    their worktree changes allowed.  For non-worktree runs, only output-only
    agents are watched (unchanged from before dm-159).
    """
    baseline = _read(workdir)
    if baseline is None:
        return None
    if node not in baseline.get("output_only", ()):
        if not baseline.get("also"):
            return None
        # Worktree run: target-only guard (no also/warn_also tracking).
        found = state(Path(baseline["root"]), exclude=(Path(workdir),))
        found["watched_at"] = datetime.now(timezone.utc).isoformat()
        return found
    found = state(Path(baseline["root"]), exclude=(Path(workdir),))
    found["also"] = {path: _also_state(path, workdir) for path in baseline.get("also", [])}
    found["warn_also"] = {path: _also_state(path, workdir) for path in baseline.get("warn_also", [])}
    found["watched_at"] = datetime.now(timezone.utc).isoformat()
    return found


def _stray_matches(stray: dict[str, Any] | None, pre: str, now: str, watched_at: str) -> bool:
    """Whether a path's change since watch() is another node's refusal undoing its stray.

    True when the stray was recorded at or after this node's watch, this node's
    pre-node bytes were that stray (``digest``), and the path now holds what the
    stray's refusal restored (``pre_digest``).
    """
    return (stray is not None and stray.get("at", "") >= watched_at
            and pre == stray.get("digest") and now == stray.get("pre_digest"))


def _discount_strays(before: dict[str, Any], now: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *now* with paths discounted by a concurrent refusal's stray record.

    A path is treated as unchanged when this node's pre-node digest is a stray that
    another node's refusal set aside at or after ``before['watched_at']``, and the
    path now holds what that refusal restored (dm-159, correction 4.1).
    """
    watched_at = before.get("watched_at", "")
    if not watched_at or now.get("changes") is None or before.get("changes") is None:
        return now
    strays = _read_stray_record(Path(before["root"]))
    if not strays:
        return now
    adjusted = dict(now)
    adjusted["changes"] = dict(now["changes"])
    for path in before["changes"].keys() | now["changes"].keys():
        pre = before["changes"].get(path, "-")
        if _stray_matches(strays.get(path), pre, now["changes"].get(path, "-"), watched_at):
            if path in before["changes"]:
                adjusted["changes"][path] = pre
            else:
                adjusted["changes"].pop(path, None)
    return adjusted


def strayed(workdir: Path, before: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    """What changed since :func:`watch`, split into failures and warnings.

    A watched worktree is held to its commits as well as its files.

    Returns ``(fail, warn)``.

    ``fail`` lists changes in ``also`` directories (the run's worktree): writing
    there bypasses materialise.  When ``also`` is set (a worktree exists), target-root
    changes also go to fail (dm-159).  When the snapshot has no ``also`` key (a
    non-output-only agent's target-only snapshot), target changes go to fail.  When
    there are no ``also`` entries and no worktree -- a run without a worktree --
    every target-root change goes to fail, preserving today's behaviour where the
    target is the only protection.

    ``warn`` lists only ``warn_also`` changes for worktree runs (launcher checkout):
    those are not where the run works, so a change there cannot be its result, but
    the person is told their files moved.
    """
    if before is None:
        return ([], [])
    now = state(Path(before["root"]), exclude=(Path(workdir),))
    # Discount paths that another node's refusal already set aside.
    now = _discount_strays(before, now)
    target_changes = _compare(before, now, commits=False)
    also_changes: list[str] = []
    for path, earlier in (before.get("also") or {}).items():
        also_changes.extend(f"in {path}, {difference}"
                            for difference in _compare(earlier, _also_state(path, workdir)))
    warn_also_changes: list[str] = []
    for path, earlier in (before.get("warn_also") or {}).items():
        warn_also_changes.extend(f"in {path}, {difference}"
                                 for difference in _compare(earlier, _also_state(path, workdir)))
    if also_changes or before.get("also"):
        return (also_changes + target_changes, warn_also_changes)
    return (target_changes, warn_also_changes)


def refusal(node: str, root: str, changed: list[str], writes: list[str], *,
            set_aside: str | None = None,
            worktree_path: str | None = None,
            restored: dict | None = None,
            is_output_only: bool = False) -> str:
    """Why an agent node is refused, and what the person should do about it.

    The target can change while a node runs for reasons that are not the node's (another
    session in the same checkout), and nothing here can tell which, so the message says
    what changed and when, not who changed it. ``set_aside`` is where the refused
    attempt's outputs were moved, so its redo starts without them.
    """
    outputs = ", ".join(writes) or "none"
    moved = (f" This attempt's output was moved to {set_aside}, so the redo writes its own."
             if set_aside else "")
    if restored is not None:
        restore_parts: list[str] = []
        if restored.get("head_moved"):
            restore_parts.append("HEAD moved during the node; commits are not rewritten.")
        if restored["restored"]:
            restore_parts.append(
                f"The target was restored to its state before the agent ran: "
                f"{len(restored['restored'])} path(s) restored. "
                f"The agent's copies are in {restored['set_aside']}.")
        if restored["errors"]:
            restore_parts.append(f"Could not restore: {'; '.join(restored['errors'])}.")
        restore_text = " ".join(restore_parts) if restore_parts else ""
    else:
        restore_text = "Nothing in the target was reverted."
    where_text = ""
    if is_output_only:
        where_text = (" For an output-only node, its changes reach the worktree only"
                      " through the node's outputs.")
    elif worktree_path is not None:
        where_text = f" Write to the worktree at {worktree_path} instead."
    action_text = ""
    if restored is None:
        action_text = (f" If the agent made the change, revert it and redo {node}; if "
                       f"another session or person did, redo {node} once they are done.")
    return (f"the target {root} changed while agent {node!r} ran, outside its declared "
            f"outputs ({outputs}): {'; '.join(changed)}. Its whole answer belongs in its "
            "declared output, which a later step turns into the artifact, so a direct change "
            f"skips that step, the check and the reviews (#319). "
            f"{restore_text}{moved}{where_text}{action_text}")


# ── lossless target restoration (dm-159) ─────────────────────────────────────

MAX_BASELINE_SIZE = 16 * 1024 * 1024  # 16 MB


def save_baseline(root: Path, changes: dict[str, str], baseline_dir: Path) -> list[str]:
    """Save pre-node bytes of uncommitted/untracked target paths for lossless restore.

    Returns a list of paths that could not be saved (directories, oversized, errors).
    """
    import shutil

    not_saved: list[str] = []
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in sorted(changes):
        src = root / rel_path
        try:
            if src.is_dir() and not src.is_symlink():
                not_saved.append(f"{rel_path} (directory)")
                continue
            if src.is_symlink():
                dest = baseline_dir / (rel_path + ".symlink.json")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps({"target": os.readlink(src)}))
                continue
            if not src.exists():
                continue
            stat = src.stat()
            if stat.st_size > MAX_BASELINE_SIZE:
                not_saved.append(f"{rel_path} (>{MAX_BASELINE_SIZE // (1024*1024)} MB)")
                continue
            dest = baseline_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        except OSError as error:
            not_saved.append(f"{rel_path} ({error})")
    return not_saved


def _target_slot(root: Path) -> str:
    """A stable hash-based directory name for a target root."""
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]


def _stray_record_path(root: Path) -> Path:
    return refactory_home() / "target-guard" / _target_slot(root)


def _read_stray_record(root: Path) -> dict[str, Any]:
    path = _stray_record_path(root) / "strays.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _restore_plan(root: Path, path: str, before: dict[str, Any],
                  baseline_dir: Path) -> tuple[str, Any] | None:
    """How ``path`` gets its pre-node content back, or None when nothing can give it back.

    ``("symlink", target)``, ``("copy", saved file)``, ``("git", (ls-tree line, content))``,
    or ``("absent", None)`` when the path did not exist before the node, so moving it
    aside is the restore. Decided before anything is moved, so a file whose earlier
    bytes were never kept is left where it is rather than moved away for good.
    """
    symlink_marker = baseline_dir / (path + ".symlink.json")
    if symlink_marker.exists():
        return "symlink", json.loads(symlink_marker.read_text()).get("target", "")
    if (baseline_dir / path).exists():
        return "copy", baseline_dir / path
    pre_digest = before.get("changes", {}).get(path, "-")
    if before.get("git") and path not in before.get("changes", {}):
        # Not in git status before the node: a clean tracked file, or no file at all.
        prefix = ""
        top = Path((_git(root, "rev-parse", "--show-toplevel") or str(root)).strip()).resolve()
        if root.resolve() != top:
            prefix = str(root.resolve().relative_to(top)) + "/"
        ls = _git(root, "ls-tree", "HEAD", f"{prefix}{path}")
        if not ls:
            return "absent", None
        content = _git(root, "show", f"HEAD:{prefix}{path}")
        return ("git", (ls, content)) if content is not None else None
    if pre_digest == "-":
        return "absent", None
    return None


def restore_target(root: Path, before: dict[str, Any],
                   baseline_dir: Path, workdir: Path, *,
                   now: dict[str, Any] | None = None) -> dict[str, Any]:
    """Restore the target to its pre-node state on refusal.

    Moves current target files to ``target-set-aside/``, writes pre-node bytes back.
    Uses a per-target stray record under ``fcntl`` lock to prevent parallel agents
    from undoing each other's restores (dm-159). ``now`` is the state to compare with
    ``before``, read the way ``before`` was; by default, :func:`state`. A file whose
    pre-node content cannot be put back is left in place and named in ``errors``.

    Returns ``{'restored': [paths], 'set_aside': str, 'errors': [msgs], 'head_moved': bool}``.
    """
    import fcntl
    import shutil

    set_aside_dir = baseline_dir.parent / "target-set-aside"
    result: dict[str, Any] = {"restored": [], "set_aside": str(set_aside_dir),
                               "errors": [], "head_moved": False}

    if before.get("changes") is None:
        result["errors"].append("too many paths to compare")
        return result

    after = now if now is not None else state(root, exclude=(Path(workdir),))
    if after.get("changes") is None:
        result["errors"].append("too many paths to compare")
        return result

    changed = _differ(before.get("changes", {}), after.get("changes", {}))
    result["head_moved"] = after.get("head") != before.get("head")

    if not changed:
        return result

    # Acquire stray record lock
    stray_dir = _stray_record_path(root)
    stray_dir.mkdir(parents=True, exist_ok=True)
    lock_path = stray_dir / "lock"
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        strays = _read_stray_record(root)
        watched_at = before.get("watched_at", "")

        # Filter out paths already set aside by another node's refusal
        effective = []
        # A path whose pre-node bytes are another node's stray, set aside since this
        # node's watch, is restored to what that refusal restored, never to the stray.
        undone: dict[str, str] = {}
        for path in changed:
            pre_digest = before.get("changes", {}).get(path, "-")
            stray_entry = strays.get(path)
            if (stray_entry is not None and stray_entry.get("at", "") >= watched_at
                    and pre_digest == stray_entry.get("digest")):
                now_digest = after.get("changes", {}).get(path, "-")
                if now_digest == stray_entry.get("pre_digest"):
                    continue
                undone[path] = stray_entry.get("pre_digest", "-")
            effective.append(path)

        for path in effective:
            try:
                full = root / path
                if full.is_dir() and not full.is_symlink():
                    result["errors"].append(f"{path} (directory, not restorable)")
                    continue
                if path in undone:
                    # Another node's refusal restored this path; what it held before is
                    # that refusal's to give back, and only an absent one is given back here.
                    plan = ("absent", None) if undone[path] == "-" else None
                    if plan is None:
                        result["errors"].append(f"{path} (another node's refusal restored it; left as it is)")
                        continue
                else:
                    plan = _restore_plan(root, path, before, baseline_dir)
                    if plan is None:
                        result["errors"].append(f"{path} (its content before the node was not kept; "
                                                "left as it is)")
                        continue
                # (a) Set aside the current file, now that it can be given back.
                if full.exists() or full.is_symlink():
                    dest = set_aside_dir / path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    full.rename(dest)
                strays[path] = {
                    "digest": after.get("changes", {}).get(path, "-"),
                    "pre_digest": before.get("changes", {}).get(path, "-"),
                    "at": datetime.now(timezone.utc).isoformat(),
                }
                # (b) Restore pre-node content
                kind, value = plan
                if kind == "symlink":
                    full.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(value, full)
                elif kind == "copy":
                    full.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(value, full)
                elif kind == "git":
                    ls, content = value
                    full.parent.mkdir(parents=True, exist_ok=True)
                    if ls.startswith("120000"):
                        os.symlink(content, full)
                    else:
                        full.write_bytes(content.encode("utf-8", "surrogateescape"))
                    # Restore mode from git ls-tree
                    if ls.startswith("100755"):
                        full.chmod(0o755)
                    elif ls.startswith("100644"):
                        full.chmod(0o644)
                result["restored"].append(path)
            except OSError as error:
                result["errors"].append(f"{path} ({error})")

        # Write updated stray record
        stray_file = stray_dir / "strays.json"
        stray_file.write_text(json.dumps(strays, indent=2) + "\n")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    return result


# ── refactory's own checkouts (dm-164) ───────────────────────────────────────
#
# An agent that runs a refactory tool with the contract's Python imports refactory from
# the checkout that Python belongs to -- the driver the run launched from, or the pin it
# moved to when the driver moved -- and the tool writes there from any directory: an emit
# builder ran a generator that way and wrote into the driver (b32l, 2026-10-21). When that
# checkout is not the run's target, nothing above watches it, so every agent node of a
# recorded run gets a snapshot of its tracked files, and a change there is refused; what
# happens to the files is :data:`CHECKOUT_POLICY`'s.

#: Where a run's launch is recorded: `refactory ceo` writes the ``driver`` (`pin.record_driver`)
#: and a pinned resume adds its ``pin_checkout`` (`pin.resolve_pin`).
RUN_CONFIG = Path(".refactory/run-config.json")

#: Checkouts never watched as refactory's own, separated like PATH, honoured only inside a
#: pytest test (``PYTEST_CURRENT_TEST``, which the processes a test starts inherit and a
#: person's shell never has). The test suite names its own checkout here
#: (`tests/conftest.py`): its tests dispatch thousands of scripted agents, and a person
#: editing that checkout while they run is not a stray write. Anywhere else the variable
#: would switch the guard off without a word, so it is ignored, with a warning.
UNWATCHED_ENV = "REFACTORY_GUARD_UNWATCHED"

#: Whether this process has said it ignores :data:`UNWATCHED_ENV`; once is enough.
_IGNORED_UNWATCHED: list[bool] = []


def _unwatched() -> set[Path]:
    """The checkouts :data:`UNWATCHED_ENV` names, when a test is running; otherwise none."""
    named = [path for path in os.environ.get(UNWATCHED_ENV, "").split(os.pathsep) if path]
    if not named:
        return set()
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        if not _IGNORED_UNWATCHED:
            _IGNORED_UNWATCHED.append(True)
            print(f"refactory: warning: {UNWATCHED_ENV} is set, but it is only for refactory's own test "
                  "suite, so it is ignored: refactory's own checkout is watched as usual", file=sys.stderr)
        return set()
    return {Path(path).expanduser().resolve() for path in named}


def _overlaps(one: Path, other: Path) -> bool:
    return one == other or one.is_relative_to(other) or other.is_relative_to(one)


def _recorded_driver(workdir: Path) -> Path | None:
    """The checkout the run's record says its workers import refactory from: the pin, else the driver."""
    try:
        driver = json.loads((Path(workdir) / RUN_CONFIG).read_text()).get("driver")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(driver, dict):
        return None
    found = driver.get("pin_checkout") or driver.get("checkout")
    return Path(found) if isinstance(found, str) and found else None


def checkouts(workdir: Path, source: Path) -> list[Path]:
    """The refactory checkouts an agent node of this run must leave alone, besides its target.

    ``source`` is the directory the contract's Python imports refactory from (the
    contract's ``refactory_source``): the driver on the walker, and the pin on a pinned
    run's spine worker, which re-executes under the pin's Python. The run's record
    (:data:`RUN_CONFIG`) adds the pin when the run is pinned and the driver otherwise,
    for a dispatch whose own Python is not the run's. Only git checkouts count (not an
    install a checkout ignores, such as one in its .venv), and none that is, holds or
    lies in the run's target or its worktree: those are the target
    guard's own, so a run on refactory itself is watched exactly as before. Nor are the
    checkouts :data:`UNWATCHED_ENV` names. A run that recorded no target (no `record`)
    is not watched here either.
    """
    baseline = _read(workdir)
    if baseline is None:
        return []
    candidates: list[Path] = []
    top = _source_checkout(str(source))
    if top is not None:
        candidates.append(Path(top))
    recorded = _recorded_driver(workdir)
    if recorded is not None:
        candidates.append(recorded)
    own = [Path(baseline["root"]).resolve(), *(Path(path).resolve() for path in baseline.get("also") or ())]
    unwatched = _unwatched()
    found: list[Path] = []
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if (path in found or path in unwatched or not (path / ".git").exists()
                or any(_overlaps(path, mine) for mine in own)):
            continue
        found.append(path)
    return found


@functools.lru_cache(maxsize=None)
def _source_checkout(source: str) -> str | None:
    """The checkout a package directory belongs to, asked once per process: it never moves under one.

    A package a checkout ignores (an install in its .venv) is not that checkout's code.
    """
    top = _git(Path(source), "rev-parse", "--show-toplevel")
    if not top or not top.strip() or _git(Path(source), "check-ignore", "-q", source) is not None:
        return None
    return top.strip()


#: How many space-separated fields come before the path in each `git status --porcelain=v2`
#: entry of a tracked file: changed, renamed or copied (whose old path follows as a field of
#: its own), unmerged.
_V2_FIELDS = {"1": 8, "2": 9, "u": 10}


def _porcelain_v2(listing: str) -> tuple[str | None, list[str]]:
    """The HEAD commit and the paths `git status --porcelain=v2 --branch -z` names.

    As :func:`_porcelain` reads v1, a rename's old path is skipped.
    """
    fields = listing.split("\0")
    head: str | None = None
    paths: list[str] = []
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if entry.startswith("# branch.oid "):
            oid = entry.removeprefix("# branch.oid ").strip()
            head = None if oid == "(initial)" else oid
            continue
        count = _V2_FIELDS.get(entry[:1])
        if count is None:
            continue
        parts = entry.split(" ", count)
        if len(parts) > count:
            paths.append(parts[count])
        if entry.startswith("2 "):
            index += 1
    return head, paths


def _checkout_state(root: Path) -> dict[str, Any]:
    """A checkout's tracked files that differ from HEAD, and HEAD itself, in one git call.

    Untracked files are not watched: an editor's swap or lock file, a log, a person's new
    notes or a copied tree are where false refusals came from, and a generator run with
    the contract's Python rewrites tracked files, which is what b32l did. HEAD stands for
    the committed content, so a commit there is a change. The root is the checkout's top
    (:func:`checkouts` takes only those). Every agent node takes two of these.
    """
    root = Path(root).resolve()
    listing = _git(root, "status", "--porcelain=v2", "--branch", "--no-ahead-behind", "-z",
                   "--untracked-files=no")
    if listing is None:
        raise ValueError(f"git cannot report the state of {root}")
    head, paths = _porcelain_v2(listing)
    found: dict[str, Any] = {"root": str(root), "git": True, "head": head, "branch": None, "tree": head}
    if len(paths) > LIMIT:
        return {**found, "changes": None}
    return {**found, "changes": {path: _digest(root / path) for path in sorted(paths)}}


def watch_checkouts(workdir: Path, paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Each checkout's state before an agent node runs, keyed by its path.

    A checkout git cannot report on is left unwatched with a warning, not a refusal: the
    node has done nothing yet, and the target's own guard still holds.
    """
    found: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            snapshot = _checkout_state(path)
        except (OSError, ValueError) as error:
            warn(f"refactory's own checkout {path} cannot be watched during this node: {error}")
            continue
        found[str(path)] = {**snapshot, "watched_at": datetime.now(timezone.utc).isoformat()}
    return found


def strayed_checkouts(workdir: Path, before: dict[str, dict[str, Any]] | None) -> dict[str, list[str]]:
    """What changed in each watched checkout since :func:`watch_checkouts`; unchanged ones are left out.

    Read like the target of a worktree run: its tracked files and its committed content,
    and a change another node's refusal already undid is not this node's
    (:func:`_discount_strays`). A checkout git cannot report on afterwards is unknown, not
    changed: it is named in a warning, and the node is not refused for it.
    """
    changed: dict[str, list[str]] = {}
    for path, earlier in (before or {}).items():
        try:
            now = _discount_strays(earlier, _checkout_state(Path(path)))
        except (OSError, ValueError) as error:
            warn(f"refactory's own checkout {path} could not be compared with its state before "
                 f"this node, so whether the node changed it is unknown: {error}")
            continue
        differences = _compare(earlier, now, commits=False)
        if differences:
            changed[path] = differences
    return changed


def checkout_baseline(directory: Path, path: str | Path) -> Path:
    """Where one dispatch keeps a checkout's pre-node bytes; its set-aside copies go beside it."""
    return Path(directory) / "checkouts" / _target_slot(Path(path)) / "target-baseline"


# ── what a refused checkout change does to the files: one decision, one place ──

#: Restore policy for a refused change to refactory's own checkout; the user chose
#: restore-if-clean on 2026-10-27 (dm-164). Each is tested:
#:
#: - ``"restore"`` (A): put the changed files back as they were, keeping the
#:   changed copies aside;
#: - ``"report"`` (B): leave the files as they are, and name them with their diff;
#: - ``"restore-if-clean"`` (C, the default): restore when the checkout had no uncommitted tracked
#:   change when the node began, and report otherwise.
CHECKOUT_POLICY = "restore-if-clean"
POLICIES = ("restore", "report", "restore-if-clean")

#: How much of a reported change's diff one warning carries.
DIFF_SHOWN = 4000


def restores(before: dict[str, Any]) -> bool:
    """Whether :data:`CHECKOUT_POLICY` puts a checkout back, given its state when the node began."""
    if CHECKOUT_POLICY not in POLICIES:
        raise ValueError(f"CHECKOUT_POLICY is {CHECKOUT_POLICY!r}; it must be one of {', '.join(POLICIES)}")
    if CHECKOUT_POLICY == "restore-if-clean":
        return before.get("changes") == {}
    return CHECKOUT_POLICY == "restore"


def _node_diff(root: Path, path: str, before: dict[str, Any], baseline_dir: Path) -> str:
    """What the node changed in one file: from its bytes at `begin` (the saved baseline, or
    HEAD for a file that was clean), not from HEAD, so a person's earlier edit is not shown."""
    import difflib

    def text(data: bytes | None) -> list[str]:
        return [] if data is None else data.decode("utf-8", "replace").splitlines(keepends=True)

    plan = _restore_plan(root, path, before, baseline_dir)
    if plan is None:
        return f"{path}: its content at the node's start was not kept, so no diff\n"
    kind, value = plan
    earlier: bytes | None
    if kind == "copy":
        earlier = Path(value).read_bytes()
    elif kind == "git":
        earlier = value[1].encode("utf-8", "surrogateescape")
    elif kind == "symlink":
        earlier = f"symlink to {value}\n".encode()
    else:
        earlier = None
    full = root / path
    try:
        now = (f"symlink to {os.readlink(full)}\n".encode() if full.is_symlink()
               else full.read_bytes() if full.exists() else None)
    except OSError as error:
        return f"{path}: unreadable now ({error})\n"
    return "".join(difflib.unified_diff(text(earlier), text(now), f"a/{path}", f"b/{path}"))


def restore_log(root: Path) -> Path:
    """The checkout's own log of what refusals did to it, under refactory's home."""
    return _stray_record_path(Path(root)) / "restores.log"


def _settled(path: str, node: str, result: dict[str, Any]) -> str:
    """One line on what a refusal did to a checkout's files, for the person and the log."""
    text = f"agent {node!r} changed refactory's own checkout {path}"
    if result.get("restored"):
        text += (f"; put back as before the node: {_named(result['restored'])}; the changed copies "
                 f"are in {result['set_aside']}")
    if result.get("left"):
        text += f"; left as they are: {_named(result['left'])}"
    if result.get("errors"):
        text += f"; could not restore: {'; '.join(result['errors'])}"
    if result.get("head_moved"):
        text += (f"; HEAD moved from {_commit(result.get('before_head'))} to {_commit(result.get('head'))}, "
                 f"and commits are not undone: run `git -C {path} reset --keep {result.get('before_head')}`"
                 " to undo them")
    return text


def settle_checkout(path: str, before: dict[str, Any], directory: Path, workdir: Path, *,
                    node: str) -> dict[str, Any]:
    """Do what :data:`CHECKOUT_POLICY` says to a checkout a refused node changed, and say so.

    Every outcome is a run warning and a line in the checkout's :func:`restore_log`, naming
    the files and where their changed bytes were put.
    """
    root = Path(path)
    try:
        now = _checkout_state(root)
    except (OSError, ValueError) as error:
        result: dict[str, Any] = {"mode": "unknown", "restored": [], "left": [], "set_aside": None,
                                  "errors": [f"git cannot report its state now: {error}"], "head_moved": False}
        now = {"head": before.get("head")}
    else:
        if restores(before):
            result = {**restore_target(root, before, checkout_baseline(directory, path), workdir, now=now),
                      "mode": "restored", "left": []}
        else:
            left = _differ(before.get("changes") or {}, now.get("changes") or {})
            diff = "".join(_node_diff(root, name, before, checkout_baseline(directory, path)) for name in left)
            result = {"mode": "reported", "restored": [], "left": left, "set_aside": None, "errors": [],
                      "head_moved": now.get("head") != before.get("head"),
                      "diff": (diff or "")[:DIFF_SHOWN]}
    result.update(before_head=before.get("head"), head=now.get("head"))
    note = _settled(path, node, result)
    warn(note + (f"\n{result['diff']}" if result.get("diff") else ""))
    try:
        log = restore_log(root)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} run {Path(workdir)}: {note}\n")
    except OSError as error:
        warn(f"could not write {restore_log(root)}: {error}")
    return result


def checkout_refusal(node: str, target: str | None, changed: dict[str, list[str]],
                     settled: dict[str, dict[str, Any]], *, set_aside: str | None = None) -> str:
    """Why an agent node that changed one of refactory's own checkouts is refused."""
    parts: list[str] = []
    for path, differences in changed.items():
        text = f"refactory's own checkout {path} changed while agent {node!r} ran: {'; '.join(differences)}."
        result = settled.get(path) or {}
        if result.get("restored"):
            text += (f" Its changed files were put back as they were before the node: "
                     f"{_named(result['restored'])}. The changed copies are in {result['set_aside']}.")
        if result.get("left"):
            text += f" They were left as they are: {_named(result['left'])}."
        if result.get("errors"):
            text += f" Could not restore: {'; '.join(result['errors'])}."
        if result.get("head_moved"):
            text += (f" HEAD moved during the node, and commits are not undone; to undo them, run "
                     f"`git -C {path} reset --keep {result.get('before_head')}`.")
        parts.append(text)
    where = f"this run's target, {target}" if target else "this run's target"
    moved = (f" This attempt's output was moved to {set_aside}, so the redo writes its own."
             if set_aside else "")
    return (f"{' '.join(parts)} It is not {where}: it is the checkout the contract's Python imports "
            "refactory from (the run's driver, or its pin), so a refactory tool run with that Python "
            "writes there from any directory. A node's work goes only to its declared outputs and "
            "the run's own steps (materialise, generate, the check), never into refactory's checkout; "
            f"redo {node} without writing there.{moved}")
