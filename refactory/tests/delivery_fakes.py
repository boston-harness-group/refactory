"""Fakes for delivery tests that reach the op child (`child_patches`).

Each is a module-level function the op's child imports by name; what a test varies is
passed as JSON keyword parameters, and state kept across dispatches lives in files.
"""
from __future__ import annotations

import functools
import importlib
import json
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from child_patches import real
from refactory_dsh import delivery, worktree
from refactory_dsh.nn.resolve import resolve
from tests_support import git

AUTHOR = ("-c", "user.name=t", "-c", "user.email=t@example.com")


def move_main(ctx: Any, *, slug: str, clone: str, files: dict[str, str] | None = None,
              remove: list[str] | None = None, commands: list[list[str]] | None = None,
              message: str = "main moves", branch: str = "main", copy_from: str = "") -> None:
    """Commit to ``branch`` (main) on GitHub from a clone at ``clone``, once: the clone is
    the mark. ``copy_from`` is a directory whose files are copied in first (for files too
    large to pass as parameters)."""
    other = Path(clone)
    if other.exists():
        return
    git(other.parent, "clone", "-q", *(["--branch", branch] if branch != "main" else []),
        f"https://github.com/{slug}.git", str(other))
    if copy_from:
        for source in Path(copy_from).iterdir():
            shutil.copyfile(source, other / source.name)
    for name, text in (files or {}).items():
        (other / name).parent.mkdir(parents=True, exist_ok=True)
        (other / name).write_text(text)
    for name in remove or []:
        git(other, "rm", "-q", name)
    for command in commands or []:
        subprocess.run(command, cwd=str(other), check=True, capture_output=True)
    git(other, "add", ".")
    git(other, *AUTHOR, "commit", "-q", "-m", message)
    git(other, "push", "-q", "origin", branch)


def put_back(ctx: Any, *, remove: list[str], recursive: bool = False) -> None:
    """A second attempt that removes ``remove`` from the run's worktree and commits it."""
    found = worktree.recorded(ctx.workdir)
    git(Path(found.path), "rm", "-q", *(["-r"] if recursive else []), *remove)
    git(Path(found.path), *AUTHOR, "commit", "-q", "-m", "attempt 2")
    head = git(Path(found.path), "rev-parse", "HEAD")
    worktree._write_json(Path(ctx.workdir) / worktree.RECORD, asdict(replace(found, head=head)))
    ctx.facts["commit"] = head


def untracked(ctx: Any, *, files: dict[str, str]) -> None:
    """Files left untracked in the run's worktree."""
    for name, text in files.items():
        (Path(worktree.recorded(ctx.workdir).path) / name).write_text(text)


def draft(ctx: Any, *, fact: dict[str, Any]) -> None:
    """The ``draft_delivery`` fact the review limit leaves."""
    ctx.facts["draft_delivery"] = fact


def seen_head(ctx: Any, *, path: str) -> None:
    """The run's reviewed commit as delivery found it, into ``path``."""
    Path(path).write_text(worktree.recorded(ctx.workdir).head)


def run_git(ctx: Any, *, path: str, args: list[str]) -> None:
    """``git args`` in ``path`` (a checkout repointed between the gate and delivery, say)."""
    git(Path(path), *args)


def touch(ctx: Any, *, path: str, text: str = "") -> None:
    """Write ``text`` to ``path``."""
    Path(path).write_text(text)


def seen_record(ctx: Any, *, path: str) -> None:
    """The run's worktree record as delivery found it, as JSON into ``path``."""
    Path(path).write_text(json.dumps(asdict(worktree.recorded(ctx.workdir))))


def hook(ctx: Any, *, name: str, marker: str) -> None:
    """A git hook ``name`` in the run's worktree that touches ``marker`` there."""
    wt = Path(worktree.recorded(ctx.workdir).path)
    dot_git = wt / ".git"
    if dot_git.is_file():
        gitdir = Path(dot_git.read_text().strip().removeprefix("gitdir: "))
        if not gitdir.is_absolute():
            gitdir = wt / gitdir
    else:
        gitdir = dot_git
    hooks = gitdir / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / name).write_text(f"#!/bin/sh\ntouch {wt / marker}\n")
    (hooks / name).chmod(0o755)


def absent(ctx: Any, *, name: str) -> None:
    """Fail delivery if ``name`` exists in the run's worktree."""
    marker = Path(worktree.recorded(ctx.workdir).path) / name
    assert not marker.exists(), f"{name} exists in the worktree during delivery"


#: What :func:`patch` replaced in this child, by ``(module, attribute)``.
_BEFORE: dict[tuple[str, str], Any] = {}


def _original(module: str, name: str) -> Any:
    """What ``module.name`` was before :func:`patch` or a test's fake replaced it."""
    found = _BEFORE.get((module, name))
    return found if found is not None else real(importlib.import_module(module), name)


def patch(ctx: Any, *, target: str, name: str, function: str = "", params: dict[str, Any] | None = None,
          value: Any = None) -> None:
    """From here on in this op's child, ``target.name`` is ``function`` (a ``module:name``
    bound to ``params``) or the constant ``value``: a fake that starts mid-delivery."""
    module = importlib.import_module(target)
    _BEFORE.setdefault((target, name), getattr(module, name))
    if function:
        found = resolve(function)
        setattr(module, name, functools.partial(found, **params) if params else found)
    else:
        setattr(module, name, value)


STEPS = {"move_main": move_main, "put_back": put_back, "untracked": untracked, "draft": draft,
         "seen_head": seen_head, "run_git": run_git, "touch": touch, "seen_record": seen_record,
         "hook": hook, "absent": absent, "patch": patch}


def deliver_after(ctx: Any, *, steps: list[list[Any]], after: list[list[Any]] | None = None) -> Any:
    """`delivery.deliver_design`, after ``steps`` and before ``after``: ``[name, params]``
    pairs of :data:`STEPS`."""
    for name, params in steps:
        STEPS[name](ctx, **params)
    said = real(delivery, "deliver_design")(ctx)
    for name, params in after or []:
        STEPS[name](ctx, **params)
    return said


def report_with_draft(ctx: Any, *, fact: dict[str, Any]) -> Any:
    """`design_ops.report_judged`, with the ``draft_delivery`` fact the review limit leaves."""
    from refactory_dsh import design_ops

    ctx.facts["draft_delivery"] = fact
    return real(design_ops, "report_judged")(ctx)


def proceed(ctx: Any) -> str:
    """A conflict evaluation that always lets delivery go on."""
    return "proceed"


def counted(path: str) -> int:
    """How many times :func:`counted` was called for ``path`` before, across children."""
    file = Path(path)
    seen = int(file.read_text()) if file.exists() else 0
    file.write_text(str(seen + 1))
    return seen


def record_call(path: str, entry: Any) -> None:
    """Append ``entry`` to the JSON lines at ``path``."""
    with Path(path).open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


_EXCEPTIONS = {"NotDelivered": delivery.NotDelivered, "RuntimeError": RuntimeError}


def as_real(*args: Any, _module: str, _name: str, **kwargs: Any) -> Any:
    """``_module._name`` as it was before any fake: puts a faked function back for later ops."""
    return real(importlib.import_module(_module), _name)(*args, **kwargs)


def gh_fails(*args: str, command: list[str], error: str, exception: str = "NotDelivered",
             real_first: bool = False, touch: str = "", **kwargs: Any) -> str:
    """`delivery._gh` that raises ``error`` for ``gh command…`` (after running it for real,
    when ``real_first``, and writing ``touch``, when given); every other call is the real one."""
    said = _original("refactory_dsh.delivery", "_gh")(*args, **kwargs) if real_first else ""
    if list(args[:len(command)]) == command:
        if touch:
            Path(touch).write_text("")
        raise _EXCEPTIONS[exception](error)
    return said if real_first else _original("refactory_dsh.delivery", "_gh")(*args, **kwargs)


def push_fails(path: Path, slug: str, commit: str, branch: str, *, error: str,
               exception: str = "NotDelivered", only_branch: str = "", **kwargs: Any) -> None:
    """`delivery._push` that raises ``error`` (only for ``only_branch``, when given)."""
    if not only_branch or branch == only_branch:
        raise _EXCEPTIONS[exception](error)
    return _original("refactory_dsh.delivery", "_push")(path, slug, commit, branch, **kwargs)


def git_fails(path: Path, *args: str, arg: str, error: str, text: str = "") -> str:
    """`delivery._git` that raises ``error`` for a call with ``arg`` among its arguments and
    ``text`` in them; ``error`` may name ``{path}`` and ``{last}`` (the last argument)."""
    if arg in args and text in " ".join(str(a) for a in args):
        raise delivery.NotDelivered(error.format(path=path, last=args[-1]))
    return _original("refactory_dsh.delivery", "_git")(path, *args)


def heads_lag(path: Path, slug: str, *, counter: str, error: str) -> dict[str, str]:
    """`delivery._heads` that raises ``error`` the first time, across children."""
    if counted(counter) == 0:
        raise delivery.NotDelivered(error)
    return _original("refactory_dsh.delivery", "_heads")(path, slug)


def record_kwarg(*args: Any, _module: str, _name: str, _key: str, _path: str, **kwargs: Any) -> Any:
    """``_module._name``, recording each call's keyword ``_key`` (when given) into ``_path``."""
    if _key in kwargs:
        record_call(_path, kwargs[_key])
    return _original(_module, _name)(*args, **kwargs)


def ensure_recorded(found: Any, run: Any, budget: Any = None, *, commit_lockfiles: bool = True,
                    path: str) -> Any:
    """`worktree.ensure_environment` that installs nothing and records ``commit_lockfiles``."""
    record_call(path, {"commit_lockfiles": commit_lockfiles})
    return found, [], None
