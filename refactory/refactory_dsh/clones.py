"""Clone workspaces: a prepared clone of a run's worktree for an agent node.

The product half of `workspace.py`, reached through the ``refactory.workspace_clone``
entry point because it needs the run's worktree. It generalises the builder probe's
warm clone (#503), which is now a client of `clone_shared` and of the environment it
prepares:

- **One clone per node per run.** It persists across commands, fix-ups and attempts.
  After materialise commits the emission derived from it, the run's new head holds the
  clone's content, so the next dispatch only verifies that, by comparing trees, and moves
  the clone's base to the new head (``verified``). A head whose tree differs -- a check
  amended a lockfile, an emission was refused and another applied -- resets the clone to
  it (``reset``). A head that did not move keeps the clone as the agent left it
  (``kept``), so a refused emission's work is still there to fix.
- **Installs and setup run again only when the manifests change**: any dependency
  manifest or lockfile anywhere in the clone (`dependency_fingerprint`), or after a reset.
  Generate runs on every dispatch.
- **A cheap first clone.** It shares the target's objects (``git clone --shared``) and
  keeps its own refs, no remote and no hooks, as the probe's clone does.

A run without a worktree has no attempt to clone: its workspace is a folder seeded
with the source project's files, if it has one, whose diff from that seed is the
whole artifact, as materialise writes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from refactory_dsh import worktree
from refactory_dsh.git_hooks import NO_HOOKS
from refactory_dsh.invocations import HOOK_SUPPRESS
from refactory_dsh.workspace import (EXCLUDED_DIRS, OWN, WorkspaceError, _excluded, cache_environment,
                                     derive_emission, file_state, strip_run_paths)

#: The branch name a clone's environment record is keyed by. Nothing creates the branch.
BRANCH = f"{worktree.BRANCH_PREFIX}workspace"

#: The files whose content decides what an install makes, wherever in the repository they
#: are: a nested workspace member's manifest changes an install as the root's does.
MANIFEST_NAMES = frozenset({
    "pyproject.toml", "setup.py", "setup.cfg", "uv.lock", "poetry.lock", "Pipfile", "Pipfile.lock",
    "package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", "package-lock.json", "npm-shrinkwrap.json",
    "yarn.lock", ".npmrc", ".gitmodules", "Cargo.toml", "Cargo.lock", "go.mod", "go.sum", "Gemfile",
    "Gemfile.lock",
})

#: What a tree comparison leaves out, as ignore patterns: refactory's own directory and
#: the environments. Patterns rather than exclude pathspecs, because `git add` refuses a
#: pathspec that names an ignored path, and the clone ignores its ``.refactory/``.
_TREE_IGNORES = (f"/{OWN}", *EXCLUDED_DIRS)


def clone_shared(source: Path, dest: Path, rev: str, *, timeout: float = 120) -> str:
    """Clone ``source`` into ``dest`` at ``rev``, sharing its objects; return the commit.

    ``--shared`` borrows the source's object store instead of copying it, so a clone of
    a large repository takes seconds, and every commit of the run -- its worktree's
    attempts live in that store -- can be checked out without a fetch. The clone keeps its
    own refs and index, has no remote to push to, and runs no hooks; git's own state is
    ignored at ``/.refactory/``, where the agent's HOME is.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", *NO_HOOKS, "clone", "--shared", "--no-checkout", "--quiet", str(source), str(dest)],
                       check=True, capture_output=True, timeout=timeout)
    except subprocess.CalledProcessError as error:
        raise WorkspaceError(f"could not clone {source} for a workspace: "
                             f"{error.stderr.decode(errors='replace').strip()}") from error
    worktree.git(dest, "remote", "remove", "origin")
    worktree.git(dest, "config", "core.hooksPath", "/dev/null")
    # Only the repository's own ignore rules decide what the agent's change is: a person's
    # global ignore file would silently drop a new file from it.
    worktree.git(dest, "config", "core.excludesFile", "/dev/null")
    worktree.git(dest, "config", "advice.detachedHead", "false")
    worktree.git(dest, *NO_HOOKS, "checkout", "-q", "--detach", rev)
    exclude = dest / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a") as stream:
        stream.write(f"\n/{OWN}/\n")
    return worktree.git(dest, "rev-parse", "HEAD").strip()


def tree_of(repo: Path) -> str:
    """The tree ``repo``'s working files make, untracked ones included, as git would commit them.

    Through a copy of the index, so the clone's own index and HEAD are untouched, and its
    stat cache spares re-hashing unchanged files. Refactory's own directory and the
    environments are left out, as `workspace.collect` leaves them out.
    """
    repo = Path(repo)
    with tempfile.TemporaryDirectory() as scratch:
        index = Path(scratch) / "index"
        real = Path(worktree.git(repo, "rev-parse", "--git-path", "index").strip())
        real = real if real.is_absolute() else repo / real
        if real.is_file():
            shutil.copyfile(real, index)
        env = {**os.environ, "GIT_INDEX_FILE": str(index)}
        if not real.is_file():
            worktree.git(repo, "read-tree", "HEAD", env=env)
        ignores = Path(scratch) / "ignore"
        ignores.write_text("".join(f"{pattern}\n" for pattern in _TREE_IGNORES))
        worktree.git(repo, "-c", f"core.excludesFile={ignores}", "add", "-A", "--", ".", env=env)
        return worktree.git(repo, "write-tree", env=env).strip()


def _json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def dependency_fingerprint(repo: Path) -> str:
    """A digest of every dependency manifest and lockfile in ``repo``, at any depth.

    Found by name (`MANIFEST_NAMES`, and ``requirements*.txt``) among the files git tracks
    or would add, outside the environments, so it holds for any project's layout.
    """
    listing = worktree.git(Path(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    found: dict[str, str] = {}
    for name in sorted(set(listing.split("\0"))):
        relative = Path(name)
        if not name or _excluded(relative):
            continue
        if relative.name in MANIFEST_NAMES or (relative.name.startswith("requirements")
                                               and relative.suffix == ".txt"):
            target = Path(repo) / relative
            if target.is_file():
                found[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(found, sort_keys=True).encode()).hexdigest()


def _reset(repo: Path, rev: str) -> None:
    """Put ``repo`` at ``rev`` exactly: a merge in progress or committed in it is dropped too."""
    worktree.git(repo, *NO_HOOKS, "reset", "-q", "--hard", rev)
    _clean(repo)


def _clean(repo: Path) -> None:
    """Put ``repo`` back to its HEAD: tracked files restored, untracked ones removed.

    Ignored files stay (build caches), and so do the environments and refactory's own
    directory, which an install or the agent's HOME made.
    """
    worktree.git(repo, *NO_HOOKS, "reset", "-q", "--hard", "HEAD")
    worktree.git(repo, "clean", "-fdq", *(f"--exclude={name}" for name in (*EXCLUDED_DIRS, f"/{OWN}")))


def prepare(workdir: Path, root: Path, of: str, *, facts: dict[str, Any], command_timeout_s: int,
            budget: Any = None) -> dict[str, Any]:
    """Bring the node's clone at ``root`` to this dispatch; what `workspace.prepare` needs of it.

    The agent also gets ``refactory-targeted-tests`` on its PATH (`targeted_tests.install`).
    See `_prepare` for what is returned.
    """
    from refactory_dsh import targeted_tests

    prepared = _prepare(workdir, root, of, facts=facts, command_timeout_s=command_timeout_s,
                        budget=budget)
    targeted_tests.install(workdir, root, prepared, facts)
    return prepared


def _prepare(workdir: Path, root: Path, of: str, *, facts: dict[str, Any], command_timeout_s: int,
             budget: Any = None) -> dict[str, Any]:
    """The clone's preparation (`prepare` adds the agent's tools to it).

    Returns its ``path``, the ``base`` its diff is taken from, the agent's ``env``, how it
    was ``synced``, the ``conflicts`` a pending merge left, whether it may emit
    ``deletes``, what was ``prepared``, and the ``problems`` an install, generate or
    setup reported. Raises `workspace.WorkspaceError` when the clone itself cannot be
    made, naming why.
    """
    workdir, root = Path(workdir), Path(root)
    found = worktree.recorded(workdir)
    if found is None:
        return _prepare_detached(workdir, root, facts)
    run_worktree = Path(found.path)
    if not run_worktree.is_dir():
        raise WorkspaceError(f"the run's worktree {run_worktree} does not exist; a clone of it cannot be made")
    head = worktree.git(run_worktree, "rev-parse", "HEAD").strip()
    if head != found.head:
        raise WorkspaceError(f"the run's worktree is at {head[:12]}, not at its recorded attempt "
                             f"{found.head[:12]}; materialise again before working from it")
    want = found.head if of == "attempt" else found.base
    repo = root / "repo"
    state_path = root / "state.json"
    state = _json(state_path)
    if (not (repo / ".git").exists() or state.get("target") != found.target
            or state.get("branch") != found.branch or state.get("of") != of):
        shutil.rmtree(repo, ignore_errors=True)
        (root / "environment.json").unlink(missing_ok=True)
        clone_shared(Path(found.target), repo, want)
        state = {"target": found.target, "branch": found.branch, "of": of, "base": want}
        synced = "cloned"
    elif of == "base":
        # A clone of the base starts every dispatch from it.
        _reset(repo, want)
        synced = "reset"
    elif state.get("base") == want and state.get("merge") and not _pending_tip(workdir):
        # The merge the clone holds was given up -- materialise refused it and dropped the
        # pending merge -- so what it brought in is main's, not the agent's change: the
        # clone goes back to the attempt rather than emit main's files as its own.
        _reset(repo, want)
        synced = "reset"
    elif state.get("base") == want:
        # Nothing was materialised since: the clone holds the agent's last work.
        synced = "kept"
    elif _holds(repo, want, state.get("prepared") or {}):
        # Materialise committed exactly the clone's change: only the base moves.
        worktree.git(repo, *NO_HOOKS, "reset", "-q", want)
        synced = "verified"
    else:
        _reset(repo, want)
        synced = "reset"
    if synced == "reset":
        # What setup made in the working tree went with the reset: it runs again.
        state.update(ready=False, prepared={})
    state["base"] = want
    base, conflicts = want, ()
    if of == "attempt":
        base, conflicts = _pending_merge(workdir, repo, state, synced)
    home = repo / OWN / "home"
    home.mkdir(parents=True, exist_ok=True)
    # The run's base, as the check's worktree has it, so an install is frozen exactly when
    # the check's would be (`worktree._installs`).
    clone_found = worktree.Worktree(path=str(repo), branch=BRANCH, base=found.base, head=want,
                                    target=found.target, run=str(workdir))
    deps = dependency_fingerprint(repo)
    before = file_state(repo)
    generate_owned: dict[str, str] = dict(state.get("generate_owned") or {})
    if deps != state.get("deps") or not state.get("ready"):
        from refactory_dsh.repo_change import prepare_check_environment

        if deps != state.get("deps"):
            # The install record knows only the root's manifests; a nested one changed too.
            (root / "environment.json").unlink(missing_ok=True)
        state["ready"] = False
        _write_json(state_path, state)
        _, env, problems, _ = prepare_check_environment(
            clone_found, facts, amend=False, commit_lockfiles=False, run=workdir, home=home,
            timeout=command_timeout_s, budget=budget, record=root / "environment.json",
            snapshot=lambda: file_state(repo), generate_owned=generate_owned)
        prepared = "installs, generate and setup"
    else:
        from refactory_dsh.repo_change import _PrepCtx, prepare_commands

        env = worktree.check_environment(repo, home)
        _, problems, _ = prepare_commands(_PrepCtx({"generate": facts.get("generate") or ""}, workdir=workdir),
                                          clone_found, env, home, command_timeout_s, budget, amend=False,
                                          snapshot=lambda: file_state(repo), generate_owned=generate_owned)
        prepared = "generate"
    # What the preparation changed in the working tree -- a file setup builds, a tracked
    # file generate rewrites -- is the run's steps' work, not the agent's: the check runs
    # them again on the attempt. It is left out of the emission while the agent leaves it
    # as the preparation made it (`workspace.derive_emission`'s ``skip``).
    after = file_state(repo)
    made = {name: digest for name, digest in after.items() if before.get(name) != digest}
    state["prepared"] = {**(state.get("prepared") or {}), **made}
    # What the generate command itself wrote is the run's own output, kept by digest across
    # dispatches: a later one where generate is a no-op must not forget it. Setup's output
    # is not in the set: a lockfile setup refreshes is carried when the agent changes it.
    state["generate_owned"] = {name: digest for name, digest in generate_owned.items()}
    # A step that failed does not stop the node: the agent can work, and may be the one
    # to fix it. It is told, the check runs the same steps and judges them, and the next
    # dispatch prepares in full again.
    state.update(deps=deps, ready=not problems)
    _write_json(state_path, state)
    return {"path": str(repo), "base": base, "env": agent_environment(env, found.path, home),
            "synced": synced, "conflicts": list(conflicts), "prepared": prepared,
            "problems": list(problems), "skip": dict(state["prepared"]),
            "generate_owned": dict(state["generate_owned"])}


def _holds(repo: Path, rev: str, prepared: dict[str, str]) -> bool:
    """Whether the clone's working tree is ``rev``'s, apart from what its preparation made.

    True after materialise committed the emission derived from it: the change it holds
    is then the commit's, and nothing more.
    """
    emission, refused, _ = derive_emission(repo, rev, skip=prepared)
    return not emission and not refused


def agent_environment(env: dict[str, str], run_worktree: str | None, home: Path) -> dict[str, str | None]:
    """What an agent working in a prepared clone runs with, over the run's environment.

    The clone's own tools first (`worktree.check_environment`), the run worktree's taken
    off PATH, its own HOME, no hook managers, and a git identity for a commit it makes
    there. A clone without its own virtual environment does not inherit the worktree's.
    Every cache a tool writes is inside that HOME, which is inside what the sandbox lets
    the agent write: pnpm's store (the person's, which the check's environment names, is
    read-only there), uv's cache and the XDG cache directory.
    """
    result: dict[str, str | None] = dict(strip_run_paths(env, run_worktree))
    result.update(HOME=str(home), **HOOK_SUPPRESS, **worktree.COMMITTER, **cache_environment(home))
    result.setdefault("VIRTUAL_ENV", None)
    return result


def _pending_tip(workdir: Path) -> str:
    return str(_json(Path(workdir) / ".refactory" / "pending-merge.json").get("target_tip") or "")


def _pending_merge(workdir: Path, repo: Path, state: dict[str, Any], synced: str) -> tuple[str, tuple[str, ...]]:
    """When delivery sent the run back to merge main, the same merge, started in the clone.

    Materialise starts that merge in the run's worktree and needs every conflicted file
    in the emission (`repo_change._commit_emission`), so the agent resolves it here. The
    diff is taken from the merge as git left it, conflict markers and all, and the
    conflicted files are always part of it: the emission then holds the resolutions and
    the agent's own changes, and git merges the rest again in the worktree.

    A conflicted file the run did not author (a generated one) is resolved to main's side,
    as materialise resolves it, so only the authored conflicts are the agent's. A merge
    the agent finished -- still in progress, or committed in the clone -- is kept across a
    refused round rather than started again over its resolution.
    """
    pending = _json(Path(workdir) / ".refactory" / "pending-merge.json")
    tip = str(pending.get("target_tip") or "")
    if not tip:
        state.pop("merge", None)
        return str(state["base"]), ()
    merge = state.get("merge") or {}
    in_progress = True
    try:
        worktree.git(repo, "rev-parse", "--verify", "-q", "MERGE_HEAD")
    except ValueError:
        in_progress = False
    holds_merge = in_progress or _contains(repo, tip)
    if not (merge.get("tip") == tip and holds_merge and synced == "kept"):
        if in_progress:
            worktree.git(repo, *NO_HOOKS, "merge", "--abort")
        try:
            worktree.git(repo, *NO_HOOKS, "-c", "commit.gpgsign=false", "merge", "--no-commit", "--no-ff",
                         "--no-verify", tip, env={**os.environ, **worktree.COMMITTER})
        except ValueError:
            pass  # conflicts: the agent resolves them
        authored = set(pending.get("conflict_files") or ())
        for name in worktree.git(repo, "diff", "--name-only", "--diff-filter=U").split():
            if name not in authored:
                try:
                    worktree.git(repo, *NO_HOOKS, "checkout", "--theirs", "--", name)
                    worktree.git(repo, *NO_HOOKS, "add", "--", name)
                except ValueError:
                    pass  # as materialise leaves it
        unmerged = worktree.git(repo, "diff", "--name-only", "--diff-filter=U").split()
        merge = {"tip": tip, "tree": tree_of(repo), "conflicts": sorted(unmerged)}
        state["merge"] = merge
    return str(merge["tree"]), tuple(merge.get("conflicts") or ())


def _contains(repo: Path, rev: str) -> bool:
    """Whether the clone's HEAD has ``rev`` in its history: a merge the agent committed."""
    try:
        worktree.git(repo, "merge-base", "--is-ancestor", rev, "HEAD")
    except ValueError:
        return False
    return True


def _prepare_detached(workdir: Path, root: Path, facts: dict[str, Any]) -> dict[str, Any]:
    """A workspace for a run without a worktree: a folder seeded with its source project.

    Such a run's artifact is the source project's files, if it has one, with the run's
    over them, written afresh each attempt (`repo_change._materialise_directory`). The
    folder persists across dispatches, so its diff from the seed is that whole artifact.
    A source file it removes cannot be carried, so deletions are refused.
    """
    repo = root / "repo"
    state_path = root / "state.json"
    state = _json(state_path)
    if (repo / ".git").exists() and state.get("seeded"):
        synced = "kept"
    else:
        shutil.rmtree(repo, ignore_errors=True)
        repo.mkdir(parents=True)
        source = str(facts.get("source_project") or "")
        if source:
            manifest = _json(Path(workdir) / "source-manifest.json").get("files") or {}
            for relative in sorted(manifest):
                destination = repo / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(source) / relative, destination)
        subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True, timeout=60)
        worktree.git(repo, "config", "core.hooksPath", "/dev/null")
        worktree.git(repo, "config", "core.excludesFile", "/dev/null")
        (repo / ".git" / "info").mkdir(parents=True, exist_ok=True)
        with (repo / ".git" / "info" / "exclude").open("a") as stream:
            stream.write(f"\n/{OWN}/\n")
        worktree.git(repo, "add", "-A")
        worktree.git(repo, *NO_HOOKS, "-c", "commit.gpgsign=false", "commit", "-q", "--no-verify",
                     "--allow-empty", "-m", "the run's seed", env={**os.environ, **worktree.COMMITTER})
        state = {"seeded": True, "base": worktree.git(repo, "rev-parse", "HEAD").strip()}
        _write_json(state_path, state)
        synced = "seeded"
    home = repo / OWN / "home"
    home.mkdir(parents=True, exist_ok=True)
    env: dict[str, str | None] = {"HOME": str(home), **HOOK_SUPPRESS, **worktree.COMMITTER,
                                  **cache_environment(home)}
    return {"path": str(repo), "base": state["base"], "env": env, "synced": synced, "deletes": False,
            "prepared": "nothing: a run without a worktree installs nothing before its check"}
