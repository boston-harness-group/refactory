"""Watch the target a run works on: say when it moves, refuse writes nobody declared.

The target is the repository a design run changes: the one it names in `target`, the
new project it created, or -- for a run without a worktree -- the project it runs from
(`project_for_nodes()`). A checkout other sessions work in can move under a run (#330
I1), and a worker with full write access can edit files directly instead of answering
through its declared output (#319). A run with its own worktree (`worktree.py`, #351)
never works in the target, so this module is its backstop rather than its isolation:
the baseline is recorded at `prepare`, and every agent node of a worktree run fails
when it changes the target while it ran (dm-159). An output-only agent also fails
when the run's own worktree changed.
"""

from __future__ import annotations

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


def restore_target(root: Path, before: dict[str, Any],
                   baseline_dir: Path, workdir: Path) -> dict[str, Any]:
    """Restore the target to its pre-node state on refusal.

    Moves current target files to ``target-set-aside/``, writes pre-node bytes back.
    Uses a per-target stray record under ``fcntl`` lock to prevent parallel agents
    from undoing each other's restores (dm-159).

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

    after = state(root, exclude=(Path(workdir),))
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
                # (a) Set aside the current file
                if full.exists() or full.is_symlink():
                    if full.is_dir() and not full.is_symlink():
                        result["errors"].append(f"{path} (directory, not restorable)")
                        continue
                    dest = set_aside_dir / path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    full.rename(dest)

                # Record the stray
                pre_digest = before.get("changes", {}).get(path, "-")
                strays[path] = {
                    "digest": after.get("changes", {}).get(path, "-"),
                    "pre_digest": pre_digest,
                    "at": datetime.now(timezone.utc).isoformat(),
                }

                if path in undone:
                    if undone[path] == "-":
                        result["restored"].append(path)
                    else:
                        result["errors"].append(f"{path} (another node's refusal restored it)")
                    continue

                # (b) Restore pre-node content
                symlink_marker = baseline_dir / (path + ".symlink.json")
                baseline_copy = baseline_dir / path

                if symlink_marker.exists():
                    target_link = json.loads(symlink_marker.read_text()).get("target", "")
                    full.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(target_link, full)
                    result["restored"].append(path)
                elif baseline_copy.exists():
                    full.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(baseline_copy, full)
                    result["restored"].append(path)
                elif before.get("git") and path not in before.get("changes", {}):
                    # Path was not in git status before the node: either a clean
                    # tracked file, or a path that did not exist at all.
                    prefix = ""
                    top_str = (_git(root, "rev-parse", "--show-toplevel") or str(root)).strip()
                    top = Path(top_str).resolve()
                    if root.resolve() != top:
                        prefix = str(root.resolve().relative_to(top)) + "/"
                    ls = _git(root, "ls-tree", "HEAD", f"{prefix}{path}")
                    if not ls:
                        # Not tracked in git: the file did not exist before the
                        # node.  The set-aside in (a) already removed it.
                        result["restored"].append(path)
                    else:
                        # Clean tracked file: restore from git.
                        content = _git(root, "show", f"HEAD:{prefix}{path}")
                        if content is not None:
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
                        else:
                            result["errors"].append(f"{path} (git show failed)")
                elif pre_digest == "-":
                    # Path did not exist before the node; set-aside is the restore.
                    result["restored"].append(path)
                else:
                    result["errors"].append(f"{path} (no baseline copy)")
            except OSError as error:
                result["errors"].append(f"{path} ({error})")

        # Write updated stray record
        stray_file = stray_dir / "strays.json"
        stray_file.write_text(json.dumps(strays, indent=2) + "\n")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    return result
