"""Each design run works in its own git worktree, on its own branch, from a pinned commit.

Concurrent sessions in one checkout stashed, checked out and amended over each other's
work (#330 I1), and a run's check and reviews read whatever the checkout held at that
moment (I2). A run that works in a worktree of its own cannot touch the person's
checkout, and sees the same base from its first step to its last: every attempt is a
commit on the run's branch, and what the reviewers judge is that branch's diff.

Ported from remote-factory's `factory/worktree.py`, which isolates every CEO run this
way. Three things differ on purpose. Worktrees live under refactory's own home rather
than inside the target, because the target may be someone else's repository and a
checkout nested in it confuses its tests and package managers. A run whose state cannot
be read counts as active, where remote-factory assumes the opposite, so a worktree is
never taken from a run that may still need it. And removal never discards what someone
did after the run: a worktree with an edited tracked file, or with an untracked file
newer than the run's last step, is kept and reported, not forced.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from refactory_dsh.paths import refactory_home, venv_bin
from refactory_dsh.run_ownership import held

from refactory_dsh.invocations import WORKTREE_RECORD as RECORD  # canonical in invocations
from refactory_dsh.invocations import PENDING  # canonical in invocations

#: The prefix of every branch this module creates. Only branches matching
#: :data:`STANDARD_BRANCH` are ever deleted: a branch a person named is never touched.
BRANCH_PREFIX = "refactory/run-"
STANDARD_BRANCH = re.compile(r"^refactory/run-[0-9a-f]{8,}$")

#: How long a run that ended without success keeps its worktree, for inspection.
IDLE_HOURS = 24

#: How long such a run keeps its branch once its worktree is gone.
FAILED_BRANCH_DAYS = 14

#: Who commits a run's attempts. Each commit only records an attempt; who authors what
#: is finally delivered is decided when it is delivered (#350).
COMMITTER = {
    "GIT_AUTHOR_NAME": "Refactory", "GIT_AUTHOR_EMAIL": "refactory@localhost",
    "GIT_COMMITTER_NAME": "Refactory", "GIT_COMMITTER_EMAIL": "refactory@localhost",
}

#: The files whose content decides what a worktree's environment holds.
MANIFESTS = ("pyproject.toml", "uv.lock", "package.json", "pnpm-lock.yaml",
             "package-lock.json", "yarn.lock", ".gitmodules")

#: Left in a virtual environment refactory created, so a check only ever adopts an
#: environment it knows the provenance of (remote-factory's ``WORKTREE_VENV_MARKER``).
MARKER = ".refactory-managed"

from refactory_dsh.invocations import WRAPPER  # canonical in invocations

#: The environment names a check in a worktree is given, besides PATH and HOME. Nothing
#: of refactory's own -- ``PYTHONPATH``, ``REFACTORY_KNOBS``, ``DSH_WORKFLOW_NODE`` --
#: reaches a target's tests (#330 E1).
CHECK_ENV = ("LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")

from refactory_dsh.invocations import CHECK_HOME  # canonical in invocations
from refactory_dsh.invocations import BASE_CHECK_HOME  # canonical in invocations
from refactory_dsh.invocations import INSTALL_TIMEOUT_S  # canonical in invocations

#: The largest file an install may write, in MB: native packages exceed a check's limit.
INSTALL_FILE_MB = 16384

#: Credential-like variables an install keeps: a package registry's own sign-in.
REGISTRY_ENV = ("NPM_", "NODE_AUTH_", "UV_", "PIP_", "YARN_", "PNPM_")

#: The lockfiles an install writes. One it creates or changes is part of the attempt.
LOCKFILES = ("uv.lock", "pnpm-lock.yaml", "package-lock.json", "yarn.lock")

#: How far an untracked file's time may run past the run's last step and still be the
#: run's own, for filesystems that keep times coarsely.
SLACK_S = 60

#: Spine statuses after which a run cannot be resumed (`recovery`): it failed, or it
#: finished, which a design run does only after it reported.
ENDED = ("failed", "done")

#: Given to every git command that changes a run's worktree. The target's hooks are the
#: person's, and `--no-verify` stops only pre-commit and commit-msg: a post-checkout or
#: post-commit hook would still run on the run's behalf.
#: Canonical definition in ``git_hooks`` (core); re-imported here so
#: ``worktree.NO_HOOKS`` keeps working for product callers.
from refactory_dsh.git_hooks import NO_HOOKS  # noqa: F401 — re-export

#: Suppress git hook managers (lefthook, husky) so installs and commands in the
#: worktree do not rewrite the shared repository's hooks (#347). Canonical
#: definition in invocations.py; re-imported here for internal use.
from refactory_dsh.invocations import HOOK_SUPPRESS  # canonical in invocations

#: Where the spine records the invocation that owns a run directory (`invocations`).
INVOCATION = Path(".refactory") / "invocation.json"

#: The per-worktree record of files copied from the person's uncommitted working tree.
UNCOMMITTED_RECORD = "uncommitted-files.json"

#: Where a run's own artifacts go. One a step writes into the worktree -- a reviewer's
#: findings, say -- is never part of an attempt: b40a once committed
#: ``.refactory/verifier-findings.md`` onto its branch (dm-242).
RUN_ARTIFACTS = ".refactory"


def run_artifact(relative: Path | str) -> bool:
    """Whether ``relative`` is under the worktree's top-level ``.refactory/``, in any case.

    Only the top-level one: a run's state lives there, and a nested ``.refactory/`` (say
    ``docs/.refactory/``) is the project's own.
    """
    parts = Path(relative).parts
    return bool(parts) and parts[0].lower() == RUN_ARTIFACTS


def _unstage_run_artifacts(path: Path) -> None:
    """Unstage whatever is staged under the top-level ``.refactory/``, in any case (dm-242).

    Named one by one rather than by a pathspec, which `git` here reads literally, so
    ``.Refactory/`` on a case-insensitive disk is unstaged too.
    """
    staged = [name for name in git(path, "diff", "--cached", "--name-only", "--no-renames", "-z").split("\0")
              if name and run_artifact(name)]
    if staged:
        git(path, "reset", "-q", "--", *staged)


class AttemptRefused(ValueError):
    """An attempt's files cannot be applied as they were emitted.

    The emitting step's to fix, not the run's to end on: a design run sends the reason
    back to ``emit`` and tries again (#330 I4).
    """


@dataclass(frozen=True)
class Worktree:
    """One run's worktree: where it is, its branch, the commit it started from and its last."""

    path: str
    branch: str
    base: str
    head: str
    target: str
    run: str


def root() -> Path:
    """Where every run's worktree lives: refactory's own home, never inside a target.

    Resolved, because git names worktrees by their real path, and a symlink in the home
    directory would otherwise make a run's own worktree look like someone else's.
    """
    return (refactory_home() / "worktrees").resolve()


def slot(target: Path) -> Path:
    """The directory one target's worktrees share, named so two targets never collide."""
    target = Path(target).resolve()
    return root() / f"{target.name}-{hashlib.sha256(str(target).encode()).hexdigest()[:8]}"


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    """What a git command prints in ``repo``; a failure is refused in git's own words."""
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                                timeout=300, env={**(env or os.environ), "GIT_LITERAL_PATHSPECS": "1"})
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"git {' '.join(args)} could not run in {repo}: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"git {' '.join(args)} failed in {repo}: {detail}")
    return result.stdout


def _json(path: Path) -> dict[str, Any]:
    """A JSON object from disk, or empty when there is none to read."""
    try:
        loaded = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace ``path`` whole, so a reader never sees half a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pending.write_text(json.dumps(payload, indent=2) + "\n")
    pending.replace(path)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` while two runs could race on the same repository.

    Two runs that start on one target at once both prune it, so they take turns: one
    never removes a worktree the other is still deciding about. Creating takes no lock:
    an add that reads another's half-written entry is retried (`_add`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":  # pragma: no cover - posix only
        yield
        return
    import fcntl

    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def repository_root(path: Path) -> Path:
    """The root of the git repository at ``path``, refusing anything that is not one.

    A worktree is of a whole repository, and the artifact's paths are relative to its
    root, so a subdirectory is refused rather than silently widened to the repository.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"the target {path} is not a directory")
    try:
        top = Path(git(path, "rev-parse", "--show-toplevel").strip()).resolve()
    except ValueError as error:
        raise ValueError(
            f"the target {path} is not a git repository. Pass the root of a repository to "
            "change, or leave `target` empty to start a new project."
        ) from error
    if top != path:
        raise ValueError(f"the target {path} is inside the repository {top}; pass its root, {top}")
    try:
        git(path, "rev-parse", "-q", "--verify", "HEAD^{commit}")
    except ValueError as error:
        raise ValueError(
            f"the repository {path} has no commits, so a run has nothing to branch from. "
            "Commit something first: refactory adds a first commit only to projects it creates."
        ) from error
    return path


def recorded(run: Path) -> Worktree | None:
    """The worktree ``run`` works in, or None when it works without one."""
    found = _json(Path(run) / RECORD)
    try:
        return Worktree(**{field: str(found[field]) for field in Worktree.__dataclass_fields__})
    except KeyError:
        return None


def is_run_worktree(path: Path) -> bool:
    """Whether ``path`` is one of the worktrees this module created."""
    path = Path(path)
    return path.resolve().is_relative_to(root().resolve()) and (path / ".git").is_file()


# Canonical definitions live in ``git_hooks`` (core).  Re-imported here
# so ``worktree._config_value`` and ``worktree._isolate_hooks`` keep
# working for product callers (base_check, delivery, tests).
from refactory_dsh.git_hooks import _config_value  # noqa: F401 — re-export
from refactory_dsh.git_hooks import _isolate_hooks  # noqa: F401 — re-export


def create(target: Path, rev: str, run_id: str, run: Path) -> Worktree:
    """Give ``run`` its own worktree of ``target``, on a new branch from ``rev``'s commit.

    The base is resolved to a commit once, here, so a branch that moves later -- even
    a HEAD that was amended -- never changes what the run started from
    (remote-factory's ``TestSHAResolution``). The target's record of the run is written
    before the worktree is added, so a redo of the run's first step finds what it made
    whatever point it died at: the worktree, or only its branch when the worktree was
    removed while the run could still be resumed. Either way it keeps the base it was
    pinned to then.
    """
    target, run = Path(target).resolve(), Path(run).resolve()
    branch = BRANCH_PREFIX + run_id
    path = slot(target) / f"run-{run_id}"
    record_path = slot(target) / f"run-{run_id}.json"
    earlier = _json(record_path)
    if earlier.get("run") == str(run) and _branch_exists(target, branch):
        base = str(earlier["base"])
        if not path.exists():
            _readd(target, path, branch)
        elif not _registered(target, path, branch):
            raise ValueError(f"{path} already exists and is not this run's worktree; it is kept as it is")
    elif path.exists() or _branch_exists(target, branch):
        raise ValueError(f"{path} or its branch {branch} already exists and is not this run's; "
                         "both are kept as they are")
    else:
        base = resolve_base(target, rev)
        _write_json(record_path, {"path": str(path), "branch": branch, "base": base, "target": str(target),
                                  "run": str(run), "pid": os.getpid(), "shared": _shared_state(target),
                                  "created": datetime.now(timezone.utc).isoformat()})
        try:
            _add(target, path, branch, base)
        except ValueError:
            # git keeps the branch it made when only the checkout failed (a smudge filter,
            # say); the record stays with it, so a redo finds the branch and adds the worktree.
            if not _branch_exists(target, branch):
                record_path.unlink(missing_ok=True)
            raise
    _isolate_hooks(target, path)
    found = Worktree(path=str(path), branch=branch, base=base, head=git(path, "rev-parse", "HEAD").strip(),
                     target=str(target), run=str(run))
    _write_json(run / RECORD, asdict(found))
    return found


#: How many times `git worktree add` is tried when it fails on another add's half-written
#: entry, and the most it waits, in seconds, before the next try.
ADD_ATTEMPTS = 5
ADD_JITTER_S = 0.2

#: git's words when an add read another add's entry under `.git/worktrees/` half-written.
_HALF_WRITTEN = re.compile(r"failed to read \S*\.git/worktrees/")


def _add(target: Path, path: Path, branch: str, base: str | None = None) -> None:
    """Add the worktree at ``path`` of ``branch``, made from ``base`` when one is given.

    Two adds on one repository at once can collide: one scans `.git/worktrees/` while the
    other has written only part of its entry there, and fails on it ("failed to read
    .git/worktrees/<name>/commondir"). Nothing of the failed add is at ``path`` then, so it
    is tried again, a bounded number of times, after a short random wait (dm-197). git
    makes a new branch before it scans, so a retry checks out a branch the failed add made
    rather than making it again. Any other failure, or one that left something at
    ``path``, is refused in git's words.
    """
    for attempt in range(ADD_ATTEMPTS):
        made = base is None or (attempt > 0 and _branch_exists(target, branch))
        try:
            git(target, *NO_HOOKS, "worktree", "add", "-q",
                *((str(path), branch) if made else ("-b", branch, str(path), str(base))))
            return
        except ValueError as error:
            if (attempt == ADD_ATTEMPTS - 1 or not _HALF_WRITTEN.search(str(error))
                    or path.exists()):
                raise
        time.sleep(random.uniform(0, ADD_JITTER_S))


def resolve_base(target: Path, rev: str) -> str:
    """The commit ``rev`` names in ``target``, refused when it names none."""
    try:
        return git(target, "rev-parse", "-q", "--verify", f"{rev}^{{commit}}").strip()
    except ValueError as error:
        raise ValueError(f"the base {rev!r} is not a commit in {target}") from error


def _readd(target: Path, path: Path, branch: str) -> None:
    """Add ``path`` back as the worktree of ``branch``, whose commits hold every attempt.

    git still lists a worktree deleted by hand, and refuses to add it again until that
    stale entry, which nothing else needs, is removed.
    """
    if _registered(target, path, branch):
        git(target, "worktree", "remove", str(path))
    _add(target, path, branch)
    _isolate_hooks(target, path)


def copy_uncommitted(target: Path, worktree_path: Path, run: Path) -> list[dict[str, str]]:
    """Copy every uncommitted change from ``target`` into ``worktree_path`` and record it.

    Copies modified tracked files, staged files, untracked unignored files, and deletes
    tracked files the person deleted — including the old side of staged renames.  The
    worktree is then committed as the run's base ("working-tree base for run-<id>"), so
    the attempts' diffs show only the run's own change, and the PR's Verification lists
    which files came from the working tree.

    Uses ``git status --porcelain`` to capture renames (``git mv a b`` must delete ``a``
    in the worktree) rather than separate ``--name-only --diff-filter`` calls that miss
    the old side of a staged rename.

    The person's checkout is never modified.
    """
    import hashlib as _hl

    target = Path(target).resolve()
    worktree_path = Path(worktree_path).resolve()
    run = Path(run).resolve()
    wt_name = worktree_path.name
    run_id = wt_name.removeprefix("run-") if wt_name.startswith("run-") else wt_name

    # Parse ``git status --porcelain=v1 -z --untracked-files=all`` for a unified
    # view of all changes including renames, staged deletes, modified, added and
    # untracked files.  The ``-z`` flag gives NUL-separated output so paths with
    # spaces or special characters are not quoted, and ``--untracked-files=all``
    # enumerates individual files inside new directories instead of collapsing
    # them to the directory name.
    try:
        status_output = git(target, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    except ValueError:
        status_output = ""
    copy_paths: set[str] = set()
    delete_paths: set[str] = set()
    untracked_set: set[str] = set()

    # NUL-separated entries: each is "XY path\0" except renames which are
    # "XY new\0old\0" (two path entries — with -z the field order is reversed:
    # the entry's path is the destination, the next NUL field is the source).
    entries_raw = status_output.split("\0") if status_output else []
    i = 0
    while i < len(entries_raw):
        entry = entries_raw[i]
        if len(entry) < 4:
            i += 1
            continue
        xy = entry[:2]
        path = entry[3:]
        # Renames: with -z, the entry path is the NEW name (destination) and
        # the next NUL-separated field is the OLD name (source).
        if xy[0] == "R" or xy[1] == "R":
            new_name = path
            i += 1
            old_name = entries_raw[i] if i < len(entries_raw) else ""
            copy_paths.add(new_name)
            if old_name:
                delete_paths.add(old_name)
            i += 1
            continue
        if not path:
            i += 1
            continue
        # Deleted: either index or worktree column is 'D'
        if "D" in xy:
            delete_paths.add(path)
            i += 1
            continue
        # Untracked
        if xy == "??":
            copy_paths.add(path)
            untracked_set.add(path)
            i += 1
            continue
        # Modified, Added, Copied, etc.
        copy_paths.add(path)
        i += 1

    entries: list[dict[str, str]] = []

    # Delete files that the person deleted or renamed away.
    for rel in sorted(delete_paths):
        dest = worktree_path / rel
        if dest.exists():
            dest.unlink()
        entries.append({"path": rel, "kind": "deleted", "sha256": ""})

    # Copy files that exist in the working tree.
    # Symlink safety: recreate symlinks as symlinks (os.readlink + os.symlink)
    # instead of copying the target's contents.  This prevents two hazards:
    #   1. A retargeted tracked absolute symlink being written through,
    #      overwriting the old link's target outside the worktree.
    #   2. An untracked symlink to a credentials file having its contents
    #      committed into the run's base and delivered with the PR.
    # Before writing any destination, unlink it if it is already a symlink so
    # we never follow an existing link to a location outside the worktree.
    # Regular-file destinations whose resolved path escapes the worktree (e.g.
    # because a parent component is a symlink) are refused.
    wt_resolved = worktree_path.resolve()
    for rel in sorted(copy_paths):
        source = target / rel
        dest = worktree_path / rel
        if not source.exists() and not source.is_symlink():
            # The person deleted it after staging, or it's gone; treat as deleted.
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            entries.append({"path": rel, "kind": "deleted", "sha256": ""})
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Unlink an existing destination symlink before writing anything,
        # so we never write through a link to outside the worktree.
        if dest.is_symlink():
            dest.unlink()
        if source.is_symlink():
            # Recreate the symlink faithfully rather than copying the target
            # file's contents.
            link_target = os.readlink(source)
            os.symlink(link_target, dest)
            sha = _hl.sha256(link_target.encode()).hexdigest()
            kind = "untracked" if rel in untracked_set else "modified"
            entries.append({"path": rel, "kind": kind, "sha256": sha})
        else:
            # Refuse a regular-file destination that resolves outside the
            # worktree (a parent directory might be a symlink).
            resolved_dest = dest.resolve()
            if not (str(resolved_dest).startswith(str(wt_resolved) + os.sep)
                    or resolved_dest == wt_resolved):
                entries.append({"path": rel, "kind": "refused", "sha256": ""})
                continue
            try:
                content = source.read_bytes()
            except OSError:
                continue
            sha = _hl.sha256(content).hexdigest()
            dest.write_bytes(content)
            kind = "untracked" if rel in untracked_set else "modified"
            entries.append({"path": rel, "kind": kind, "sha256": sha})

    if not entries:
        return entries

    # Stage and commit the working-tree state as the run's base commit.
    git(worktree_path, *NO_HOOKS, "add", "--all")
    env = {**os.environ, **COMMITTER}
    git(worktree_path, *NO_HOOKS, "commit", "-q", "--allow-empty",
        "-m", f"working-tree base for run-{run_id}", env=env)

    # Update the run's worktree record so `found.head` and `found.base` reflect
    # the new commit (the base is now the working-tree commit, and so is head).
    new_head = git(worktree_path, "rev-parse", "HEAD").strip()
    run_record = _json(run / RECORD)
    if run_record:
        run_record["head"] = new_head
        run_record["base"] = new_head
        _write_json(run / RECORD, run_record)
    # Also update the per-slot record.
    slot_record_path = slot(target) / f"run-{run_id}.json"
    slot_data = _json(slot_record_path)
    if slot_data:
        slot_data["base"] = new_head
        _write_json(slot_record_path, slot_data)

    # Record the uncommitted files alongside the worktree record.
    record_path = slot(target) / f"run-{run_id}.{UNCOMMITTED_RECORD}"
    _write_json(record_path, {"files": entries})

    return entries


def uncommitted_record(found: "Worktree") -> list[dict[str, str]]:
    """The uncommitted files that were copied into ``found``'s worktree, or empty."""
    if not hasattr(found, "target"):
        return []
    run_id = Path(found.path).name.removeprefix("run-")
    path = slot(Path(found.target)) / f"run-{run_id}.{UNCOMMITTED_RECORD}"
    data = _json(path)
    return data.get("files", [])


def _branch_exists(target: Path, branch: str) -> bool:
    try:
        git(target, "rev-parse", "-q", "--verify", f"refs/heads/{branch}")
    except ValueError:
        return False
    return True


def _registered(target: Path, path: Path, branch: str) -> bool:
    """Whether git lists ``path`` as a worktree of ``target`` on ``branch``."""
    for entry in git(target, "worktree", "list", "--porcelain").split("\n\n"):
        lines = entry.splitlines()
        listed = next((line.removeprefix("worktree ") for line in lines if line.startswith("worktree ")), "")
        if listed and Path(listed) == Path(path) and f"branch refs/heads/{branch}" in lines:
            return True
    return False


def _record_path(found: Worktree) -> Path:
    """The target's record of the run that made ``found``."""
    return slot(Path(found.target)) / f"run-{found.branch.removeprefix(BRANCH_PREFIX)}.json"


def record_outcome(found: Worktree, outcome: str) -> None:
    """Note how the run ended in the target's record of it, which outlives the run directory.

    A run directory is reused by a later invocation (`invocations.prepare` archives this
    one's files), so a run that succeeded must not have to be recognised from it again
    (:func:`run_state`).
    """
    record_path = _record_path(found)
    _write_json(record_path, {**_json(record_path), "outcome": outcome,
                              "ended": datetime.now(timezone.utc).isoformat()})


def shared_changed(found: Worktree) -> bool:
    """Whether the git configuration or hooks the worktree shares with its target changed.

    A worktree shares them, so an install, a setup command, a check or a reviewer that
    "sets up" hooks there changes the person's own repository. Each change is reported
    once: the record is moved on to what is there now.
    """
    record_path = _record_path(found)
    record = _json(record_path)
    now = _shared_state(Path(found.target))
    if record.get("shared") in (None, now):
        return False
    _write_json(record_path, {**record, "shared": now})
    return True


import types as _types
_worktree_state_bridge = _types.SimpleNamespace(
    recorded=recorded, shared_changed=shared_changed,
)


# ── attempts: each one a commit on the run's branch ─────────────────────────
def _tracked_edits(path: Path) -> list[str]:
    """Tracked paths whose content differs from HEAD in the worktree at ``path``."""
    listing = git(path, "--no-optional-locks", "status", "--porcelain=v1", "-z", "--untracked-files=no")
    return sorted(entry[3:] for entry in listing.split("\0") if len(entry) > 3)


def verify_git_link(found: Worktree) -> None:
    """Refuse a worktree whose ``.git`` does not point at its target's worktree directory.

    Extracted from :func:`_require_current` so delivery can verify the link
    before running ``git reset --hard`` in ``_recover_interrupted_merge``.
    A worktree whose ``.git`` was repointed at the person's repository must be
    refused before any mutating git command runs in it.
    """
    path = Path(found.path)
    if not path.is_dir():
        return  # the worktree does not exist; _require_current will re-add it
    common = Path(git(Path(found.target), "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    if Path(git(path, "rev-parse", "--absolute-git-dir").strip()).parent != common / "worktrees":
        raise ValueError(f"the run's worktree {path} no longer points at its own git directory; "
                         "nothing was changed")


def _require_current(found: Worktree) -> None:
    """Refuse a worktree whose HEAD moved outside the run's own steps; add back one pruned.

    A run whose node failed can still be retried, and an idle one may have been pruned
    meanwhile (:func:`prune`), so a missing worktree is added back from its branch.
    """
    path = Path(found.path)
    if not path.is_dir():
        if not _branch_exists(Path(found.target), found.branch):
            raise ValueError(f"the run's worktree {path} and its branch {found.branch} are both gone, "
                             "so its attempts are too; start the run again")
        _readd(Path(found.target), path, found.branch)
    verify_git_link(found)
    head = git(path, "rev-parse", "HEAD").strip()
    if head != found.head:
        raise ValueError(
            f"the run's worktree {path} is at {head[:12]}, not at its last attempt "
            f"{found.head[:12]}: something committed or checked out there outside the run. "
            f"Nothing was reverted. If that work is not needed, `git -C {path} reset --hard "
            f"{found.head[:12]}` puts the run back where it was."
        )


def _restore(found: Worktree, written: frozenset[str] = frozenset()) -> list[str]:
    """Put the run's worktree back to its last attempt; name the tracked files that had changed.

    The worktree is the run's own, so whatever changed it since -- the check, its setup,
    a reviewer -- did so for the run, and is undone rather than refused: a check that
    rewrites a lockfile must not wedge the run, and a file a reviewer left must not be
    in the next check without being on the branch. Tracked files are reset, and are
    worth reporting: a step changed the source. Untracked files are removed without a
    word -- a check's `__pycache__` in a project with no `.gitignore` would otherwise be
    reported on every reloop -- and what the run installed, and what git ignores, stay.
    ``written`` are the files an attempt that died had written, which are undone
    without being reported.
    """
    path = Path(found.path)
    edited = _tracked_edits(path)
    if edited:
        git(path, *NO_HOOKS, "reset", "-q", "--hard", "HEAD")
    keep = [argument for name in _json(_record_path(found)).get("installed", []) for argument in ("-e", f"/{name}")]
    stray = [line.removeprefix("Would remove ") for line in git(path, "clean", "-fdn", *keep).splitlines() if line]
    if stray:
        git(path, "clean", "-fdq", *keep)
    return [name for name in edited if name not in written]


def _inside(root: Path, relative: Path) -> Path:
    """``root / relative``, refused when a part of it is a symlink, which could lead outside."""
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise AttemptRefused(f"{relative} goes through the symlink {current.relative_to(root)}; an attempt "
                                 "writes only real files in its worktree, so change the file the link points "
                                 "to by its own path, or leave it")
    return current


def committed_text(found: Worktree, relative: Path, rev: str = "") -> str | None:
    """The file at ``relative`` as commit ``rev`` (the run's last attempt) has it; None when it has none.

    Read from the commit, not the disk, so an edit applies to what the last attempt left
    whatever a check has done to the worktree since, and byte for byte, line endings kept.
    """
    rev = rev or found.head
    if not _committed_file(found, relative, rev):
        return None
    try:
        result = subprocess.run(["git", "-C", found.path, "cat-file", "blob", f"{rev}:{relative.as_posix()}"],
                                capture_output=True, timeout=300,
                                env={**os.environ, "GIT_LITERAL_PATHSPECS": "1"})
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"git cat-file could not run in {found.path}: {error}") from error
    if result.returncode != 0:
        raise ValueError(f"git cat-file failed in {found.path}: {result.stderr.decode(errors='replace').strip()}")
    try:
        return result.stdout.decode()
    except UnicodeDecodeError:
        raise AttemptRefused(f"{relative} is not text, so it cannot be edited; leave it as it is, or remove it "
                             "with ===DELETE:") from None


def _committed_file(found: Worktree, relative: Path, rev: str = "") -> bool:
    """Whether commit ``rev`` (the run's last attempt) has a file at ``relative``: not a directory."""
    entry = git(Path(found.path), "ls-tree", rev or found.head, "--", relative.as_posix()).split()
    return len(entry) >= 2 and entry[1] == "blob"


def commit_attempt(run: Path, files: dict[Path, str], message: str, deletes: tuple[Path, ...] = (),
                   *, amending: bool = False) -> Worktree:
    """Make one attempt's files the run's worktree, as one commit on its branch.

    Each attempt builds on the last: ``files`` are written and ``deletes`` removed on top
    of the run's last attempt, so a retry changes only what it needs to. Only the named
    paths are staged, never ``-A``, and none of the target's hooks run: the run's check is
    its gate. An attempt that died after its commit but before recording it is dropped,
    and this one replaces it.

    With ``amending``, the files are staged the same way and folded into the last attempt
    through `amend`, the one path every amendment takes: a self-check fix-up finishes
    the attempt it was sent back from (b41a).

    A path under ``.refactory/`` is a run artifact, not a file of the change: it is left
    out, with a warning, and never written or staged, and an emission of nothing else is
    refused, saying so (dm-242).
    """
    found = recorded(run)
    if found is None:
        raise ValueError("this run has no worktree")
    path = Path(found.path)
    artifacts = sorted(str(name) for name in (*files, *deletes) if run_artifact(name))
    if artifacts:
        from refactory_dsh import target_guard

        files = {name: content for name, content in files.items() if not run_artifact(name)}
        deletes = tuple(name for name in deletes if not run_artifact(name))
        target_guard.warn(f"run artifacts under {RUN_ARTIFACTS}/ are never part of an attempt, so these "
                          f"were left out of it: {target_guard._named(artifacts)}")
    for relative in (*files, *deletes):
        if any(part.lower() == ".git" for part in relative.parts):
            raise AttemptRefused(f"{relative} is inside git's own state, not a file of the artifact; leave it out")
        _inside(path, relative)
    for relative in deletes:
        if not _committed_file(found, relative):
            raise AttemptRefused(f"{relative} is not a file of the last attempt, so there is nothing to delete; "
                                 "leave its ===DELETE: out")
    pending = Path(run) / PENDING
    died = _json(pending)
    if died.get("parent") == found.head and path.is_dir():
        head = git(path, "rev-parse", "HEAD").strip()
        if head != found.head and _died_after_commit(path, found.head, bool(died.get("amending"))):
            git(path, *NO_HOOKS, "reset", "-q", "--hard", found.head)
    _require_current(found)
    # When a merge is in progress (MERGE_HEAD present), skip _restore: the staged
    # merge state (cleanly-merged files, unmerged entries, MERGE_HEAD) is intentional
    # and must not be reset.
    try:
        git(path, "rev-parse", "--verify", "-q", "MERGE_HEAD")
        merge_in_progress = True
    except ValueError:
        merge_in_progress = False
    if artifacts and not files and not deletes and not merge_in_progress:
        raise AttemptRefused(f"the emission names only run artifacts under {RUN_ARTIFACTS}/, which are never "
                             "part of an attempt: emit the change's own files")
    if not merge_in_progress:
        restored = _restore(found, frozenset(died.get("files", [])))
        if restored:
            from refactory_dsh import target_guard

            target_guard.warn(f"tracked files in the run's worktree {path} had changed since its last attempt and "
                              f"were put back before this one: {target_guard._named(restored)}")
    for relative in files:
        if (path / relative).is_dir() and not (path / relative).is_symlink():
            raise AttemptRefused(f"{relative} is a directory of the last attempt, not a file; write the files "
                                 "inside it, or choose another path")
        for parent in relative.parents:
            if parent != Path(".") and (path / parent).exists() and not (path / parent).is_dir():
                raise AttemptRefused(f"{relative} would be inside {parent}, which is a file of the last attempt; "
                                     "choose another path")
    _write_json(pending, {"files": sorted(str(name) for name in (*files, *deletes)), "parent": found.head,
                          "amending": amending})
    for relative in deletes:
        git(path, "rm", "-q", "-f", "--", str(relative))
    for relative, content in files.items():
        destination = _inside(path, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    if files:
        git(path, "add", "-f", "--", *(str(name) for name in files))
    # An emission that changes nothing -- every file and edit already in the last
    # attempt -- is refused before committing, so the run loops back to emit
    # rather than through the check and reviews on identical code.
    # During a merge-in-progress, a commit is ALWAYS needed to create the merge
    # commit (making main an ancestor of HEAD), even when content matches HEAD.
    if not merge_in_progress:
        try:
            git(path, "diff", "--cached", "--quiet")
        except ValueError:
            pass  # a non-zero exit from git diff --cached --quiet -- the normal path when something is staged
        else:
            pending.unlink(missing_ok=True)
            raise AttemptRefused(
                "the emission changed nothing: every file and edit in it is already in the last attempt"
            )
        # A fix-up that takes the attempt back to what it built on would leave it changing
        # nothing: refused, as an emission that changes nothing is, not amended away.
        if amending and _staged_as_parent(path):
            git(path, *NO_HOOKS, "reset", "-q", "--hard", "HEAD")
            pending.unlink(missing_ok=True)
            raise AttemptRefused(
                "the emission undoes the whole attempt: with it, the attempt would change nothing "
                "against the commit it builds on. Fix what failed without reverting the change; "
                "if part of the change is wrong, change that part"
            )
    if merge_in_progress:
        # `_restore` did not run, so nothing else has cleared the index of a run artifact.
        _unstage_run_artifacts(path)
    try:
        if amending:
            amend(found, run, [str(name) for name in files], message)
        else:
            git(path, *NO_HOOKS, "-c", "commit.gpgsign=false", "commit", "-q", "--no-verify",
                "-m", message, env={**os.environ, **COMMITTER})
    except ValueError as exc:
        # When committing with unmerged paths, git refuses.  This can happen if
        # not every conflicted file was resolved by the emission.
        if "unmerged" in str(exc).lower() or "not possible" in str(exc).lower():
            raise AttemptRefused(
                f"the commit failed because unmerged paths remain; every conflicted file "
                f"must appear in the emission and be fully resolved: {exc}"
            ) from exc
        raise
    new_head = git(path, "rev-parse", "HEAD").strip()
    # When the commit was a merge (MERGE_HEAD was present before commit), update the
    # base to main's tip (the second parent) so base_check and the diff cover only
    # this run's own change, not everything main gained.
    try:
        parents_output = git(path, "rev-parse", f"{new_head}^1", f"{new_head}^2")
        parent_lines = [p.strip() for p in parents_output.strip().splitlines() if p.strip()]
        if len(parent_lines) >= 2:
            # This is a merge commit.  Update base to the second parent (main's tip).
            advanced = replace(found, head=new_head, base=parent_lines[1])
        else:
            advanced = replace(found, head=new_head)
    except ValueError:
        advanced = replace(found, head=new_head)
    # Clean up pending-merge record after a successful merge commit.
    pending_merge = Path(run) / ".refactory" / "pending-merge.json"
    pending_merge.unlink(missing_ok=True)
    _write_json(Path(run) / RECORD, asdict(advanced))
    pending.unlink(missing_ok=True)
    return advanced


def _staged_as_parent(path: Path) -> bool:
    """Whether what is staged is the tree of HEAD's parent: amended in, it empties HEAD."""
    try:
        git(path, "diff", "--cached", "--quiet", "HEAD^", "--")
    except ValueError:
        return False  # it differs, or HEAD has no parent
    return True


def _died_after_commit(path: Path, head: str, amended: bool) -> bool:
    """Whether HEAD is the commit an attempt that died made on ``head``, so it can be dropped.

    A new attempt is a child of ``head``; an amendment replaces ``head``, so it is a
    sibling: another commit on ``head``'s parent.
    """
    try:
        if amended:
            return git(path, "rev-parse", "HEAD^").strip() == git(path, "rev-parse", f"{head}^").strip()
        return git(path, "rev-parse", "HEAD^").strip() == head
    except ValueError:
        return False


def require_current(run: Path) -> Worktree:
    """The run's worktree, refusing one that is gone or moved since its last attempt."""
    found = recorded(run)
    if found is None:
        raise ValueError("this run has no worktree")
    # Recover from a crash between a merge commit and the pending-merge record
    # write: if pending-merge.json exists and MERGE_HEAD is gone, check whether
    # HEAD is already a merge commit whose second parent matches the recorded tip.
    pending_merge = Path(run) / ".refactory" / "pending-merge.json"
    if pending_merge.exists():
        pending = _json(pending_merge)
        recorded_tip = str(pending.get("target_tip") or "")
        path = Path(found.path)
        if path.is_dir():
            has_merge_head = False
            try:
                git(path, "rev-parse", "--verify", "-q", "MERGE_HEAD")
                has_merge_head = True
            except ValueError:
                pass
            if not has_merge_head and recorded_tip:
                # Check if HEAD is already a merge commit with the recorded tip
                # as its second parent — adopt it rather than refusing.
                try:
                    head = git(path, "rev-parse", "HEAD").strip()
                    if head != found.head:
                        parents_output = git(path, "rev-parse", f"{head}^1", f"{head}^2")
                        parent_lines = [p.strip() for p in parents_output.strip().splitlines() if p.strip()]
                        if len(parent_lines) >= 2 and parent_lines[1] == recorded_tip:
                            # Adopt the existing merge commit.
                            advanced = replace(found, head=head, base=parent_lines[1])
                            _write_json(Path(run) / RECORD, asdict(advanced))
                            pending_merge.unlink(missing_ok=True)
                            found = advanced
                except ValueError:
                    pass
    _require_current(found)
    return found


def restore(run: Path) -> tuple[Worktree, list[str]]:
    """The run's worktree at its last attempt, and the tracked files that had to be put back."""
    found = require_current(run)
    return found, _restore(found)


def tracked_edits(found: Worktree) -> list[str]:
    """Tracked files in the worktree that differ from its last attempt, so are not on its branch."""
    return _tracked_edits(Path(found.path))


def diff(found: Worktree) -> str:
    """What the run changed: its branch against the commit it started from."""
    return git(Path(found.path), "diff", found.base, "HEAD")


def committed(path: Path) -> dict[str, str]:
    """The files of the attempt the worktree at ``path`` holds: its HEAD's tree, as git hashes it."""
    listing = git(path, "ls-tree", "-r", "-z", "HEAD")
    files = {}
    for entry in filter(None, listing.split("\0")):
        meta, name = entry.split("\t", 1)
        mode, _kind, blob = meta.split()
        files[name] = f"symlink:{blob}" if mode == "120000" else blob
    return files


# ── whether a run still needs its worktree ─────────────────────────────────
def _alive(pid: Any) -> bool:
    """Whether process ``pid`` still exists; True when that cannot be told."""
    if not isinstance(pid, int) or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _marks() -> tuple[Path, Path, Path]:
    """Where a run directory says how its run stands: spine status, walker progress, deliveries."""
    from refactory_dsh.agent_delivery import ROOT as DELIVERIES
    from refactory_dsh.console import SPINE_STATUS
    from refactory_dsh.nn.executor import PROGRESS

    return SPINE_STATUS, Path(PROGRESS), DELIVERIES / "pending"


def state(run: Path, claimant: Any = None, *, branch: str = "") -> str:
    """Where ``run`` stands: ``succeeded``, ``ended`` (failed), or ``active``.

    Fails closed. A run whose state cannot be read counts as active, so its worktree is
    kept; remote-factory's session guard fails open instead. A run the spine drives is
    over only once its status says it failed or finished: a halted, cancelled or paused
    run can be resumed (`recovery`), and so can one whose GUI restarted. That is
    stricter than `repo_change._still_open`, which releases an artifact name once the
    owner exits: a name released early costs a refusal, a worktree removed early costs
    adding it back from its branch. A walked run is over once its progress records an
    error, or once ``claimant``, the walking process, has exited. ``branch`` names the
    run when its directory may hold a later run's report: `design.json` then counts only
    if it reports that branch.
    """
    run = Path(run)
    spine_status, progress_path, pending = _marks()
    try:
        if not run.is_dir():
            return "ended"
        report = _json(run / "design.json")
        if report and (not branch or report.get("branch") == branch):
            return "succeeded"
        if held(run) or any((run / pending).glob("*.json")):
            return "active"
        spine = run / spine_status
        # A walked run writes the status file too, but whether it is over is its progress's
        # and its walking process's to say (`run_state.walked`).
        from refactory_dsh.run_state import walked

        if spine.exists() and not walked(run):
            return "ended" if _json(spine).get("status") in ENDED else "active"
        progress = _json(run / progress_path)
        # A walked run names its walking process in its status file when no claimant did.
        from refactory_dsh.run_state import owner_pid

        if progress.get("error") or not _alive(claimant if claimant is not None else owner_pid(run)):
            return "ended"
        return "active"
    except OSError:
        return "active"


def _superseded(found: dict[str, Any]) -> bool:
    """Whether the record's run directory has since been taken by a later invocation.

    A run directory is reused when a session opens a second run (`invocations.prepare`
    archives the first one's files), and from then on it describes the later run, which
    could start only once this one was over.
    """
    run = Path(str(found.get("run") or ""))
    current = recorded(run)
    if current is not None:
        return current.path != str(found.get("path"))
    invocation = str(_json(run / INVOCATION).get("id") or "")
    return bool(invocation) and not invocation.startswith(str(found.get("branch") or "").removeprefix(BRANCH_PREFIX))


def run_state(found: dict[str, Any]) -> str:
    """Where the run that made a worktree stands, from the target's record of it.

    ``succeeded`` once the run said so (:func:`record_outcome`); ``ended`` when its run
    directory has moved on to a later invocation; otherwise what its run directory says
    (:func:`state`).
    """
    if found.get("outcome") == "succeeded":
        return "succeeded"
    if _superseded(found):
        return "ended"
    return state(Path(str(found.get("run") or "")), found.get("pid"), branch=str(found.get("branch") or ""))


def _last_active(target: Path, found: dict[str, Any]) -> float | None:
    """When the run last did anything: its last commit, or the last status it recorded.

    A run whose directory moved on to a later one stopped by the time that one began,
    so the directory's newer records date its end instead. None when nothing says.
    """
    times = []
    try:
        times.append(float(git(target, "log", "-1", "--format=%ct", str(found.get("branch"))).strip()))
    except ValueError:
        pass
    run = Path(str(found.get("run") or ""))
    spine_status, progress_path, _pending = _marks()
    for mark in (RECORD, INVOCATION) if _superseded(found) else (spine_status, progress_path):
        try:
            times.append((run / mark).stat().st_mtime)
        except OSError:
            pass
    return max(times) if times else None


def _idle_hours(target: Path, found: dict[str, Any]) -> float:
    """How long the run has been idle; 0 when that cannot be told, so nothing is pruned on a guess.

    remote-factory's ``_compute_idle_seconds`` reads the worktree's mtime instead, which
    an edit to a file in it does not change; what someone did there after the run is
    judged file by file when the worktree is removed (:func:`_remove`).
    """
    last = _last_active(target, found)
    return (time.time() - last) / 3600 if last is not None else 0.0


def _untracked(path: Path) -> list[str]:
    """Every file git neither tracks nor ignores in the worktree at ``path``."""
    listing = git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    return sorted(entry[3:] for entry in listing.split("\0") if entry.startswith("?? "))


def _remove(target: Path, found: dict[str, Any], path: Path) -> str:
    """Remove a run's worktree, never discarding what someone did there after the run; say what happened.

    Untracked files from before the run's last step are the run's -- what it installed,
    what its check left, such as `__pycache__` in a project with no `.gitignore` -- and
    are cleared, as before every attempt (:func:`_restore`); one changed since keeps the
    worktree, as does an edited tracked file, which `git worktree remove` refuses. git
    refuses any worktree that holds submodules, so one whose status shows nothing
    uncommitted, submodules included, is removed with ``--force``.
    """
    from refactory_dsh.target_guard import _named

    last = _last_active(target, found)
    later = [name for name in _untracked(path)
             if last is None or (path / name).lstat().st_mtime > last + SLACK_S]
    if later:
        return (f"kept {path}: {_named(later)} changed there after the run's last step. Remove it yourself "
                f"with `git -C {target} worktree remove --force {path}` once nothing in it is needed")
    git(path, "clean", "-fdq")
    try:
        git(target, "worktree", "remove", str(path))
    except ValueError as error:
        clean = (path / ".gitmodules").is_file() and not git(
            path, "status", "--porcelain", "--ignore-submodules=none").strip()
        if not clean:
            return (f"kept {path}: {error}. It holds work no run committed; remove it yourself with "
                    f"`git -C {target} worktree remove --force {path}` once nothing in it is needed")
        git(target, "worktree", "remove", "--force", str(path))
    return f"removed the worktree of a run: {path}"


def prune(target: Path, *, idle_hours: float | None = None, succeeded: bool = False) -> list[str]:
    """Remove what runs on ``target`` no longer need, and say what was removed or kept.

    Only runs that ended without success are pruned. A successful run keeps its
    worktree and branch until its work is delivered (#350), unless ``succeeded`` asks
    for its worktree -- never its branch -- to go too; an active or resumable one is
    never touched. An ended run loses its worktree after :data:`IDLE_HOURS` idle, never
    discarding what someone did there after it (:func:`_remove`), and its branch after
    :data:`FAILED_BRANCH_DAYS`, with `git branch -D`: a failed run's commits were never
    merged. Only branches matching :data:`STANDARD_BRANCH` are ever deleted, and nothing
    of the target's own worktrees is touched: git's own `worktree prune` would drop the
    records of a person's worktree on a disk that is not mounted. A pruned run that goes
    on gets its worktree back from its branch (:func:`_require_current`).
    """
    target = Path(target).resolve()
    limit = float(os.environ.get("REFACTORY_WORKTREE_IDLE_HOURS", IDLE_HOURS)) if idle_hours is None else idle_hours
    notes: list[str] = []
    here = slot(target)
    if not here.is_dir():
        return notes
    with _locked(here / ".lock"):
        for record_path in sorted(here.glob("run-*.json")):
            found = _json(record_path)
            branch, path = str(found.get("branch") or ""), Path(str(found.get("path") or ""))
            if not STANDARD_BRANCH.match(branch):
                continue
            outcome = run_state(found)
            if outcome == "active" or (outcome == "succeeded" and not succeeded):
                continue
            idle = _idle_hours(target, found)
            if path.is_dir() and (outcome == "succeeded" or idle >= limit):
                notes.append(_remove(target, found, path))
            if outcome == "ended" and not path.is_dir() and idle >= FAILED_BRANCH_DAYS * 24:
                try:
                    if _registered(target, path, branch):
                        git(target, "worktree", "remove", str(path))
                    if _branch_exists(target, branch):
                        git(target, "branch", "-D", branch)
                        notes.append(f"deleted the branch of an ended run: {branch}")
                except ValueError as error:
                    notes.append(f"kept the branch {branch}: {error}")
                    continue
                record_path.unlink(missing_ok=True)
    return notes


# ── the environment a check runs in ─────────────────────────────────────────
def check_environment(path: Path, home: Path) -> dict[str, str]:
    """What a check in the worktree at ``path`` runs with.

    The worktree's own tools come first: its virtual environment and its package
    binaries. Refactory's interpreter is used only when the target has no Python
    environment of its own, and never with ``PYTHONPATH``, so a target that is
    refactory itself is tested from its worktree and not from the launcher's checkout.
    ``home`` keeps tool caches a check writes out of the worktree and its diff (#330 E6).
    pnpm finds its store through HOME, so it is pointed at the store the install used;
    otherwise it takes the installed packages for another store's and tries to reinstall.
    """
    path = Path(path)
    managed = (path / ".venv" / MARKER).exists()
    own = [*([path / ".venv" / "bin"] if managed else []), path / "node_modules" / ".bin"]
    entries = [str(directory) for directory in own if directory.is_dir()]
    if not managed:
        entries.append(str(venv_bin()))
    entries += [entry for entry in os.environ.get("PATH", "").split(os.pathsep)
                if entry and Path(entry) != venv_bin() and entry not in entries]
    env = {"PATH": os.pathsep.join(entries), "HOME": str(home)}
    env.update({name: os.environ[name] for name in CHECK_ENV if name in os.environ})
    if managed:
        env["VIRTUAL_ENV"] = str(path / ".venv")
    store = _json(path / "node_modules" / ".modules.yaml").get("storeDir")
    if isinstance(store, str) and store:
        env["pnpm_config_store_dir"] = store
    # Suppress git hook managers so commands run through the worktree's
    # environment do not rewrite the shared repository's hooks (#347).
    env.update(HOOK_SUPPRESS)
    return env


def _shared_snapshot(target: Path) -> dict:
    """A snapshot of the git configuration and hooks the worktree shares with the target.

    Returns ``{"config_lines": [...], "hook_files": {"name": "sha256hex", ...},
    "hooks_dir": "<path>"}``.  The ``hooks_dir`` key carries the path so
    ``_shared_digest`` can re-read raw bytes for backward-compatible hashing.
    The exclusion rules match what ``shared_changed`` ignores: ``submodule.`` lines
    (initialising submodules is refactory's own doing) and the
    ``extensions.worktreeconfig=`` line ``_isolate_hooks`` sets.
    """
    common = Path(git(target, "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    settings = git(target, "config", "--file", str(common / "config"), "--list").splitlines()
    config_lines = [
        line for line in settings
        if not line.startswith("submodule.")
        and not line.lower().startswith("extensions.worktreeconfig=")
    ]
    hook_files: dict[str, str] = {}
    hooks_dir = common / "hooks"
    if hooks_dir.is_dir():
        for item in sorted(hooks_dir.glob("*")):
            if item.is_file():
                hook_files[item.name] = hashlib.sha256(item.read_bytes()).hexdigest()
    return {"config_lines": config_lines, "hook_files": hook_files, "hooks_dir": str(hooks_dir)}


def _shared_digest(snapshot: dict) -> str:
    """A SHA-256 hex digest of a shared-state snapshot.

    Feeds raw file bytes (not hex-digest strings) to match the digest the code
    produced before the snapshot/digest split.  The snapshot carries per-file hashes
    for the guard's comparison; the digest re-reads raw bytes so existing worktree
    records stay compatible.
    """
    digest = hashlib.sha256()
    digest.update("\n".join(snapshot["config_lines"]).encode())
    hooks_dir = Path(snapshot["hooks_dir"])
    for name in sorted(snapshot["hook_files"]):
        digest.update(name.encode() + b"\0" + (hooks_dir / name).read_bytes())
    return digest.hexdigest()


def _shared_state(target: Path) -> str:
    """A digest of the git configuration and hooks the worktree shares with the target."""
    return _shared_digest(_shared_snapshot(target))


def _loose(path: Path) -> set[str]:
    """What git neither tracks nor ignores in the worktree at ``path``, as git names it.

    Only these need keeping between attempts: `git clean` leaves ignored files alone,
    and a top-level directory that is tracked must not be kept whole because an
    install left ignored files deep inside it.
    """
    listing = git(path, "status", "--porcelain=v1", "-z", "--untracked-files=normal")
    return {entry[3:] for entry in listing.split("\0") if entry.startswith("?? ")}


def _install_env() -> dict[str, str]:
    """The environment a dependency install runs with: the person's, less refactory's and secrets.

    An install runs the project's build backend and install scripts, which a model may
    have written, so credentials are dropped as a check drops them (`sandbox.is_secret`)
    -- except a package registry's own. Unlike a check it keeps the rest of the person's
    environment and their real HOME, whose `~/.npmrc`, `~/.netrc`, uv configuration and
    keyrings are how most private indexes sign in.
    """
    from refactory_dsh.sandbox import is_secret

    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("REFACTORY_", "DSH_"))
           and name not in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "VENV_BIN", "NODE_BIN")
           and (not is_secret(name) or name.upper().startswith(REGISTRY_ENV))}
    # Suppress git hook managers so installs in the worktree do not rewrite the
    # shared repository's hooks (#347).  Nobody answers an install's prompts
    # either, so it is told it runs unattended.
    env.update(**HOOK_SUPPRESS, CI="true")
    return env


def _node_manager(path: Path) -> str:
    """The package manager a JavaScript project uses: its lockfile's, its `packageManager`'s, or npm."""
    for lockfile, manager in (("pnpm-lock.yaml", "pnpm"), ("yarn.lock", "yarn"), ("package-lock.json", "npm")):
        if (path / lockfile).is_file():
            return manager
    declared = str(_json(path / "package.json").get("packageManager") or "").split("@")[0]
    return declared if declared in ("pnpm", "yarn", "npm") else "npm"


def _installs(found: Worktree) -> list[list[str]]:
    """The installs the worktree's manifests call for, in the order they run.

    Frozen only while the attempt leaves a manifest as the base has it. One that adds a
    dependency must be installable, and the lockfile the install then writes is part
    of the change (:func:`ensure_environment`).
    """
    path = Path(found.path)
    changed = set(git(path, "diff", "--name-only", found.base, "HEAD").split())
    commands: list[list[str]] = []
    if (path / "pyproject.toml").is_file():
        frozen = (path / "uv.lock").is_file() and "pyproject.toml" not in changed
        # Every dependency group, so the project's own test tools are there for its check.
        commands.append(["uv", "sync", "--all-groups", *(["--frozen"] if frozen else [])])
    if (path / "package.json").is_file():
        manager = _node_manager(path)
        frozen = "package.json" not in changed and any(
            (path / name).is_file() for name in ("pnpm-lock.yaml", "yarn.lock", "package-lock.json"))
        commands.append({
            "pnpm": ["pnpm", "install", "--frozen-lockfile" if frozen else "--no-frozen-lockfile"],
            "npm": ["npm", "ci"] if frozen else ["npm", "install"],
            "yarn": ["yarn", "install", *(["--frozen-lockfile"] if frozen else [])],
        }[manager])
    if (path / ".gitmodules").is_file():
        commands.append(["git", *NO_HOOKS, "submodule", "update", "--init", "--recursive"])
    return commands


def ensure_environment(found: Worktree, run: Path, budget: Any = None, *,
                       commit_lockfiles: bool = True,
                       record: Path | None = None) -> tuple[Worktree, list[str], dict[str, Any] | None]:
    """Install what the worktree's manifests declare, once per their content; say what went wrong.

    Lazily: before the first check, and again only when a manifest changed, the last
    install failed, or what it made is gone (a worktree added back from its branch).
    The record lives with the worktree, in the target's record of the run, not in the
    run directory, which a later run may reuse. Installs run through the sandbox, under
    the run's budget. A lockfile an install creates or changes is amended into
    the attempt, so the change and its reviewers include it; anything else an install
    leaves untracked is recorded, so the clean-up before each attempt keeps it. A
    Python environment refactory created is marked, so :func:`check_environment` adopts
    only that one. Returns the worktree, whose head an amended lockfile moves, the
    problems as text a failed check reports, and the prep-failure dict (step, returncode,
    timed_out) of the first failing install when one failed; an installer that is missing
    is reported as unavailable, which no code repair can fix.

    When ``commit_lockfiles`` is False the installs still run but lockfiles are not
    amended into the attempt.  Delivery uses this for the merged tree, where amending
    would corrupt the worktree record, and base preparation
    (`base_check._ensure_base_worktree`) for the base, whose commit must not move (dm-188).

    ``record`` keeps the record somewhere else than the target's record of the run: an
    agent workspace (`clones.prepare`) keeps its own, so its installs are skipped while
    its manifests are unchanged without it touching the run's.
    """
    from refactory_dsh import sandbox

    path = Path(found.path)
    manifests = _manifests(path)
    key = hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()
    record_path = record if record is not None else _record_path(found)
    done = _json(record_path).get("environment") or {}
    if (done.get("key") == key and not done.get("problems")
            and all((path / name).exists() for name in done.get("made", []))):
        return found, [], None
    problems: list[str] = []
    prep_fail: dict[str, Any] | None = None
    before = _loose(path)
    for command in _installs(found):
        shown = " ".join(command)
        if shutil.which(command[0]) is None:
            problems.append(f"UNAVAILABLE: {command[0]} is not installed, so `{shown}` could not run")
            if prep_fail is None:
                prep_fail = {"step": shown, "returncode": 127, "timed_out": False}
            continue
        result = sandbox.run_command(
            shlex.join(command), workdir=path, timeout_s=INSTALL_TIMEOUT_S, file_mb=INSTALL_FILE_MB,
            env=_install_env(), home=Path.home(), exact_env=True, budget=budget)
        if not result.ok:
            detail = "\n".join((result.problem(), result.stderr or "", result.stdout or "")).strip()
            problems.append(f"`{shown}` failed: {detail[-2000:]}")
            # Classify signal crashes: normalise shell-wrapped signals to negative
            # return codes so check_failure_kind returns 'uncertain' (infrastructure).
            crashed, sig_label = sandbox.is_signal_crash(result)
            if crashed and sig_label:
                problems[-1] = f"`{shown}` {sig_label}: {detail[-2000:]}"
            rc_for_fail = result.returncode
            if crashed and result.returncode is not None and result.returncode > 0:
                rc_for_fail = -(result.returncode - 128)
            if prep_fail is None:
                prep_fail = {"step": shown, "returncode": rc_for_fail, "timed_out": result.timed_out}
        elif command[0] == "uv" and (path / ".venv").is_dir():
            (path / ".venv" / MARKER).write_text("created by refactory for this run's checks\n")
    left = _loose(path) - before
    lockfiles = sorted(name for name in {*_tracked_edits(path), *left} if name in LOCKFILES)
    if lockfiles and commit_lockfiles:
        found = amend(found, run, lockfiles, "lockfile written by an install")
    record = _json(record_path)
    _write_json(record_path, {
        **record,
        "installed": sorted((left - set(lockfiles)) | set(record.get("installed", []))),
        "environment": {"key": key, "manifests": manifests, "problems": problems,
                        "made": [name for name in (".venv", "node_modules") if (path / name).is_dir()],
                        "at": datetime.now(timezone.utc).isoformat()},
    })
    return found, problems, prep_fail


def _manifests(path: Path) -> dict[str, str]:
    return {name: hashlib.sha256((Path(path) / name).read_bytes()).hexdigest()
            for name in MANIFESTS if (Path(path) / name).is_file()}


def manifest_key(path: Path) -> str:
    """A digest of the dependency manifests and lockfiles at ``path``: what installs depend on.

    The key `ensure_environment` records its installs under, so a caller can tell
    whether they would run again.
    """
    return hashlib.sha256(json.dumps(_manifests(path), sort_keys=True).encode()).hexdigest()


def amend(found: Worktree, run: Path, paths: list[str], message: str) -> Worktree:
    """Stage ``paths`` and amend them into the attempt, or commit with ``message`` when there is none.

    When the head is still the base -- before there is one, a commit of its own, so
    the base the run was pinned to is never rewritten -- the amendment is a fresh
    commit rather than an amend.

    One code path for every amendment -- a lockfile an install writes, a file a
    generate command rewrites, a self-check fix-up -- so the git add, the
    amend-or-commit, the new head and the record write are in one place.

    An amendment that leaves the attempt's tree its parent's is kept, as an attempt that
    changes nothing: a generate command can regenerate the only files an attempt changed
    back to what they were. git refuses such an amend without ``--allow-empty``, and the
    run would end there; kept, the attempt goes to its check and its reviews, which judge
    a change that does nothing. A fix-up's own emission that does this is refused before
    it gets here (`commit_attempt`).

    A run artifact under ``.refactory/`` is never amended in, whether it is named here or
    something else staged it: it is unstaged before the commit (dm-242).
    """
    path = Path(found.path)
    if paths:
        git(path, "add", "-f", "--", *paths)
    _unstage_run_artifacts(path)
    how = ("-m", message)
    if found.head != found.base:
        how = ("--amend", "--no-edit", *(("--allow-empty",) if _staged_as_parent(path) else ()))
    git(path, *NO_HOOKS, "-c", "commit.gpgsign=false", "commit", "-q", "--no-verify", *how,
        env={**os.environ, **COMMITTER})
    found = replace(found, head=git(path, "rev-parse", "HEAD").strip())
    _write_json(Path(run) / RECORD, asdict(found))
    return found


def _worktree_environment_overlay(workdir: Path) -> dict[str, str] | None:
    """Environment overlay for model._environment().

    Called via importlib.metadata entry_points when the product is installed.
    Wraps recorded() + check_environment() to provide the worktree's PATH
    and VIRTUAL_ENV.
    """
    found = recorded(Path(workdir))
    if found is None:
        return None
    home = Path(workdir).resolve() / CHECK_HOME
    home.mkdir(parents=True, exist_ok=True)
    target_env = check_environment(Path(found.path), home)
    result: dict[str, str] = {"PATH": target_env["PATH"]}
    if "VIRTUAL_ENV" in target_env:
        result["VIRTUAL_ENV"] = target_env["VIRTUAL_ENV"]
    return result


def write_wrapper(found: Worktree, run: Path) -> Path:
    """Leave the script every role runs commands through: the worktree, with its check environment.

    Reviewers and other roles run commands through this wrapper to get ``cwd``
    switched to the worktree and a pristine ``env -i`` environment matching the
    check. The wrapper strips all inherited variables for reproducible,
    check-equivalent execution, and provides dual access: agent tooling for
    run-directory files, plus this wrapper for isolated check commands.
    """
    run = Path(run).resolve()
    script = run / WRAPPER
    home = run / CHECK_HOME
    home.mkdir(parents=True, exist_ok=True)
    env = check_environment(Path(found.path), home)
    assignments = " ".join(f"{name}={shlex.quote(value)}" for name, value in sorted(env.items()))
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/bin/sh\n"
        "# Run a command in this design run's worktree, with the environment its check uses.\n"
        f"cd {shlex.quote(found.path)} || exit 1\n"
        f'exec env -i {assignments} "$@"\n'
    )
    script.chmod(0o755)
    return script
