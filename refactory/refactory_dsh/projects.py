"""The projects design runs work on, and where to find them again.

A design run works on a **target**: an existing git repository, or a new project
refactory creates for it. New projects live in one discoverable folder
(:func:`projects_dir`, `~/refactory-projects` by default) rather
than under a checkout's git-ignored `.factory/`, and every target a run worked on is
listed in a registry beside refactory's other per-user state, with the run's branch
and how it ended. `refactory projects` reads it; `refactory projects prune` removes
what ended runs no longer need (:func:`refactory_dsh.worktree.prune`).

Ported from remote-factory's `~/factory-projects` and `factory/registry.py`. The
registry is written under a file lock as well as atomically, because two runs can end
at the same moment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from refactory_dsh import worktree
from refactory_dsh.paths import refactory_home

#: The registry's file name, under :func:`~refactory_dsh.paths.refactory_home`.
REGISTRY = "projects.json"

#: The registry's format version.
SCHEMA = 1


def projects_dir() -> Path:
    """Where design mode creates a new project: one git repository per name.

    Discoverable on purpose -- a person looks here for what refactory built -- rather
    than under a checkout's git-ignored `.factory/`. ``REFACTORY_PROJECTS_DIR``
    overrides it, as remote-factory's ``FACTORY_PROJECTS_DIR`` does its own.
    """
    return Path(os.environ.get("REFACTORY_PROJECTS_DIR") or Path.home() / "refactory-projects").expanduser()


def registry_path() -> Path:
    """Where the registry of projects and their runs lives."""
    return refactory_home() / REGISTRY


def load() -> dict[str, Any]:
    """The registry, or an empty one when there is none to read yet."""
    try:
        loaded = json.loads(registry_path().read_text())
    except (OSError, ValueError):
        loaded = {}
    if not isinstance(loaded, dict) or not isinstance(loaded.get("projects"), list):
        return {"schema": SCHEMA, "projects": []}
    return loaded


def _entry(registry: dict[str, Any], path: Path) -> dict[str, Any] | None:
    return next((project for project in registry["projects"] if project.get("path") == str(path)), None)


def register(path: Path, *, kind: str) -> dict[str, Any]:
    """List ``path`` as a project; ``kind`` says whether refactory created it (``new``)."""
    path = Path(path).resolve()
    with worktree._locked(registry_path().with_suffix(".lock")):
        registry = load()
        entry = _entry(registry, path)
        if entry is None:
            entry = {"name": path.name, "path": str(path), "kind": kind,
                     "created": datetime.now(timezone.utc).isoformat(), "runs": []}
            registry["projects"].append(entry)
            worktree._write_json(registry_path(), registry)
        return entry


def record_run(path: Path, *, run: Path, branch: str, base: str, status: str) -> None:
    """Note a run on ``path`` and how it stands, replacing that run's earlier note.

    A run is its branch: a run directory can hold several runs one after another. A
    registry that cannot be written is reported and the run goes on: the registry is how
    a person finds a project again, not part of the run's own record (remote-factory's
    ``TestEventEmissionFailure``).
    """
    path, run = Path(path).resolve(), Path(run).resolve()
    try:
        with worktree._locked(registry_path().with_suffix(".lock")):
            registry = load()
            entry = _entry(registry, path)
            if entry is None:
                entry = {"name": path.name, "path": str(path), "kind": "existing",
                         "created": datetime.now(timezone.utc).isoformat(), "runs": []}
                registry["projects"].append(entry)
            earlier = next((note for note in entry.get("runs", []) if note.get("branch") == branch), {})
            entry["runs"] = [note for note in entry.get("runs", []) if note.get("branch") != branch]
            entry["runs"].append({"run": str(run), "branch": branch, "base": base, "status": status,
                                  "at": datetime.now(timezone.utc).isoformat(),
                                  **{key: earlier[key] for key in ("pull_request", "delivery") if earlier.get(key)}})
            worktree._write_json(registry_path(), registry)
    except OSError as error:
        from refactory_dsh.target_guard import warn

        warn(f"the project registry {registry_path()} could not be written ({error}); the run goes on, "
             "but `refactory projects` will not list it")


def _update(path: Path, change: Any) -> None:
    """Apply ``change`` to ``path``'s entry, reporting a registry that cannot be written."""
    path = Path(path).resolve()
    try:
        with worktree._locked(registry_path().with_suffix(".lock")):
            registry = load()
            entry = _entry(registry, path)
            if entry is not None:
                change(entry)
                worktree._write_json(registry_path(), registry)
    except OSError as error:
        from refactory_dsh.target_guard import warn

        warn(f"the project registry {registry_path()} could not be written ({error}); "
             "`refactory projects` will not show it")


def record_delivery(path: Path, *, branch: str, pull_request: str, status: str) -> None:
    """Note how a run's branch was delivered (#350): its pull request, if any, and how that went.

    A run reported as succeeded can still deliver nothing, or a pull request that shows
    another head, and the listing says so rather than leaving "succeeded" to stand alone.
    """
    def change(entry: dict[str, Any]) -> None:
        for note in entry.get("runs", []):
            if note.get("branch") == branch:
                note["delivery"] = status
                if pull_request:
                    note["pull_request"] = pull_request
    _update(path, change)


def record_repository(path: Path, repository: str) -> None:
    """Note the GitHub repository refactory created for the new project at ``path`` (#350).

    It is how a later delivery tells the repository it made from one that merely has
    the project's name, which it never pushes into.
    """
    _update(path, lambda entry: entry.__setitem__("github", repository))


def record_creating(path: Path, repository: str) -> None:
    """Note, before asking GitHub, the repository a delivery is about to create for ``path`` (#350).

    A creation gh reports as failed may still have made it; this is how the delivery,
    or its redo, tells that repository from one that merely has the project's name.
    """
    _update(path, lambda entry: entry.__setitem__("github_creating", repository))


def new_project(name: str) -> Path:
    """The project called ``name`` in the projects folder, created when it does not exist.

    A new project is a git repository whose first commit is empty, so a run's first
    pull request holds all of its code (#350). A name refactory already created is the
    same project again: the next run branches from its default branch. A directory
    there that refactory did not create is refused rather than adopted. The project is
    made under a temporary name and moved into place once it is registered, so a run
    that dies part way leaves nothing a later run is refused over, and two runs that
    create it at once share one project.
    """
    path = projects_dir() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{name}.", suffix=".new", dir=path.parent))
        try:
            worktree.git(staging, "init", "-q", "-b", "main")
            # gc.auto=0 and maintenance.auto=false stop git from starting background
            # maintenance after this commit: the staging directory is renamed or deleted
            # immediately, and a background process writing inside it races with that.
            worktree.git(staging, *worktree.NO_HOOKS,
                         "-c", "gc.auto=0", "-c", "maintenance.auto=false",
                         "-c", "commit.gpgsign=false",
                         "commit", "-q", "--no-verify",
                         "--allow-empty", "-m", f"Start {name}",
                         env={**os.environ, **worktree.COMMITTER})
            register(path, kind="new")
            try:
                staging.rename(path)
            except OSError:
                if not (path / ".git").exists():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return path.resolve()
    entry = _entry(load(), path.resolve())
    if entry is None or entry.get("kind") != "new":
        raise ValueError(
            f"{path} already exists and is not a project refactory created. Choose another "
            f"name, or pass target={path} if it is a git repository you want this run to change."
        )
    return worktree.repository_root(path)


def status(project: dict[str, Any], note: dict[str, Any]) -> str:
    """How a run stands now: its note, unless it ended without saying so.

    A run notes only that it is running and that it succeeded; one that failed, or was
    stopped and never resumed, is read from the target's record of it. A run that can
    still go on says how it stands: running, or the spine's word for why it waits.
    """
    if note.get("status") != "running":
        return str(note.get("status"))
    record = slot_record(Path(str(project.get("path"))), str(note.get("branch")))
    if not record:
        return "ended"
    now = worktree.run_state(record)
    if now != "active":
        return now
    from refactory_dsh.console import SPINE_STATUS

    waiting = str(worktree._json(Path(str(record.get("run"))) / SPINE_STATUS).get("status") or "")
    return waiting if waiting in ("paused", "halted", "cancelled", "waiting_user", "refused") else "running"


def slot_record(path: Path, branch: str) -> dict[str, Any]:
    """The target's record of the run on ``branch``; empty once pruning removed it."""
    return worktree._json(worktree.slot(path) / f"run-{branch.removeprefix(worktree.BRANCH_PREFIX)}.json")


def _describe(project: dict[str, Any]) -> str:
    runs = project.get("runs", [])
    tail = f"; last run {status(project, runs[-1])} on {runs[-1].get('branch')}" if runs else ""
    if runs and (runs[-1].get("pull_request") or runs[-1].get("delivery")):
        tail += f", {runs[-1].get('delivery') or 'delivered'} {runs[-1].get('pull_request') or ''}".rstrip()
    github = f", {project['github']}" if project.get("github") else ""
    return f"{project.get('name')}  {project.get('path')}  ({project.get('kind')}{github}, {len(runs)} run(s){tail})"


def main(argv: list[str] | None = None) -> int:
    """`refactory projects [--json]` lists; `refactory projects prune [name] [--succeeded]` cleans up."""
    parser = argparse.ArgumentParser(prog="refactory projects",
                                     description="the projects design runs work on, and their runs")
    parser.add_argument("action", nargs="?", choices=("list", "prune"), default="list")
    parser.add_argument("name", nargs="?", default="", help="prune only this project")
    parser.add_argument("--json", action="store_true", help="print the registry as JSON")
    parser.add_argument("--succeeded", action="store_true",
                        help="prune: also remove succeeded runs' worktrees; their branches keep the work")
    args = parser.parse_args(argv)
    registry = load()
    if args.action == "list":
        if args.json:
            for project in registry["projects"]:
                for note in project.get("runs", []):
                    note["now"] = status(project, note)
            print(json.dumps(registry, indent=2))
        elif not registry["projects"]:
            print(f"no projects yet; new ones are created in {projects_dir()}")
        else:
            for project in registry["projects"]:
                print(_describe(project))
        return 0
    chosen = [project for project in registry["projects"] if not args.name or project.get("name") == args.name]
    if args.name and not chosen:
        print(f"refactory: no project called {args.name!r}", file=sys.stderr)
        return 2
    for project in chosen:
        path = Path(str(project.get("path")))
        notes = (worktree.prune(path, succeeded=args.succeeded) if path.is_dir()
                 else worktree.prune_gone(path) or [f"{path} is gone; nothing of it to prune"])
        for note in notes or [f"{project.get('name')}: nothing to prune"]:
            print(note)
    return 0


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
