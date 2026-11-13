"""The git refactory runs: the one `git` names on PATH, without macOS's trampoline in front.

On a Mac, ``/usr/bin/git`` is not git: it is a trampoline that asks which developer
directory is selected and then starts that directory's git, so every git command is two
process launches. Under an endpoint-security agent, which inspects each launch, that
doubles the cost of the dozens of git commands every agent step runs (27 ms against
12 ms a command on one such Mac). :func:`git` names the binary the trampoline would have
started, found the way the trampoline finds it -- ``DEVELOPER_DIR``, else the
``xcode-select`` choice, else Xcode, else the Command Line Tools -- and anything it
cannot be sure of keeps plain ``git``.

The trampoline also hands the git it starts a compiler environment (``SDKROOT``,
``CPATH``, ``LIBRARY_PATH``, ``MANPATH``), which reaches whatever git itself runs: a
hook, an alias. So only commands that read the repository and run no hook skip it
(:data:`READS`, through :func:`program`); a commit, a merge, a checkout or a clone
still goes through ``git`` on PATH exactly as before.
"""

from __future__ import annotations

import functools
import os
import shutil
import sys
from pathlib import Path

#: The trampoline: what `git` names on PATH on a Mac with no other git installed.
TRAMPOLINE = "/usr/bin/git"

#: Where `xcode-select --switch` records the chosen developer directory.
SELECTED = "/var/db/xcode_select_link"

#: The developer directories used when none was chosen, in the order xcrun tries them:
#: Xcode, then the Command Line Tools.
DEFAULT_DEVELOPERS = ("/Applications/Xcode.app/Contents/Developer", "/Library/Developer/CommandLineTools")


def _developer_dir() -> Path | None:
    """The developer directory the trampoline would use, or None when it cannot be told."""
    chosen = os.environ.get("DEVELOPER_DIR")
    if chosen:
        return Path(chosen)
    try:
        return Path(os.readlink(SELECTED))
    except OSError:
        return next((Path(path) for path in DEFAULT_DEVELOPERS if Path(path).is_dir()), None)


@functools.lru_cache(maxsize=None)
def _resolve(path_env: str, developer_env: str | None) -> str:
    if sys.platform != "darwin" or shutil.which("git", path=path_env) != TRAMPOLINE:
        return "git"
    developer = _developer_dir()
    if developer is None:
        return "git"
    # `DEVELOPER_DIR` may name Xcode.app itself; the trampoline looks inside it too.
    for candidate in (developer / "usr/bin/git", developer / "Contents/Developer/usr/bin/git"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return "git"


#: Subcommands that only read the repository. Nothing they can start (an fsmonitor, a
#: textconv or external diff driver) is a compiler, which is all that environment is for.
READS = frozenset({"rev-parse", "status", "check-ignore", "symbolic-ref", "ls-tree", "ls-files",
                   "cat-file", "config", "diff", "show", "log", "rev-list", "merge-base",
                   "for-each-ref", "show-ref", "diff-tree", "diff-index", "describe"})

#: Options before the subcommand that take the next argument as their value.
_VALUED = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path",
                     "--config-env"})


def _subcommand(args: tuple[str, ...] | list[str]) -> str | None:
    """The git subcommand ``args`` (what follows `git`) runs, past its leading options."""
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in _VALUED:
            index += 2
        elif arg.startswith("-"):
            index += 1
        else:
            return arg
    return None


def program(*args: str) -> str:
    """The program to start for `git *args`: :func:`git` for a read, plain ``git`` otherwise.

    A `config` that writes (``--add``, ``--unset``, a value after the key, ...) runs no
    hook either, so it is the trampoline's environment, not the write, that rules a
    command out.
    """
    return git() if _subcommand(args) in READS else "git"


def git() -> str:
    """The program to run for `git`: the trampoline's target on a Mac, else ``git`` itself.

    Asked again whenever PATH or ``DEVELOPER_DIR`` changes, so a git put ahead on PATH
    (Homebrew's, a test's stub) is always the one run.
    """
    return _resolve(os.environ.get("PATH", os.defpath), os.environ.get("DEVELOPER_DIR"))
