"""The check's probe: whether the check can run at all, before research starts.

A check that cannot run -- an option no installed pytest plugin provides, a program that
is not installed -- used to be found only by the build's self-check, after research,
design and the build (o55-dm149c: 82 minutes). Design's prepare probes it instead
(`probe_check`): each pytest step collects (``--collect-only -q``, its options kept) and
each other step's program is looked up. The probe never runs the check, and never stops
the run: what it finds is a warning on the approval card and in the run's log.

It runs where the check will run, with the check's environment: in the run's worktree,
still at its base, after the install the check itself makes (`worktree.ensure_environment`,
the same call and record, so the check does not install again), or, without a worktree,
in an empty directory with the environment the artifact's check gets.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from refactory_dsh import sandbox, target_guard, worktree
from refactory_dsh.base_check import _steps, strip_ansi

#: How long the probe may run: it collects or looks a program up, it runs no test.
PROBE_TIMEOUT_S = 120

#: Where the probe runs when the run has no worktree: an empty directory of the run's.
PROBE_DIR = Path(".refactory") / "check-probe"

#: What a check step's exit status means when the step is pytest, or the shell.
EXIT_MEANINGS = {
    2: "pytest was interrupted, usually by an error while collecting",
    3: "pytest hit an internal error",
    4: "pytest's usage error: it could not parse the command line",
    5: "pytest collected no tests",
    126: "the shell found a program it could not execute",
    127: "the shell could not find a program",
}

#: Builtins whose step a probe keeps as it is: the steps after them depend on what they set.
_KEPT_BUILTINS = ("cd", "pushd", "export", "set", "unset", "ulimit", "umask", "source", ".")
#: Where an install, not a change, puts a check's programs: a path under one is looked up.
_INSTALLED = (".venv/", "./.venv/", "node_modules/", "./node_modules/")
#: Commands that run the program after them: ``uv run pytest`` runs pytest.
_RUNNERS = ("uv", "poetry", "pipenv", "pdm", "hatch", "rye")
_ASSIGNMENT = re.compile(r"^[A-Za-z_]\w*=")
_PYTHON = re.compile(r"^python[\d.]*$")
#: What a missing program's probe step says, so its failure is told from the check's own.
PROBE_MISSING = "refactory: check probe: not found:"
#: pytest's own errors for a path or test id that is not on disk.
_PYTEST_NOT_FOUND = re.compile(r"^ERROR: (?:file or directory not found|not found): ", re.MULTILINE)


def exit_meaning(returncode: int | None) -> str:
    """What ``returncode`` means for a check step, as far as pytest and the shell say."""
    return EXIT_MEANINGS.get(returncode, "") if returncode is not None else ""


def strip_comments(command: str) -> str:
    """``command`` without its shell comments: a ``#`` that starts a word, to the line's end."""
    out, quote, i = "", "", 0
    while i < len(command):
        char = command[i]
        if quote:
            out += char
            if char == "\\" and quote == '"' and i + 1 < len(command):
                out += command[i + 1]
                i += 1
            elif char == quote:
                quote = ""
        elif char == "\\" and i + 1 < len(command):
            out += command[i:i + 2]
            i += 1
        elif char in "'\"":
            quote = char
            out += char
        elif char == "#" and (not out or out[-1] in " \t\n;&|()"):
            while i < len(command) and command[i] != "\n":
                i += 1
            continue
        else:
            out += char
        i += 1
    return out


def _command_words(words: list[str]) -> list[str]:
    """``words`` from the command word on: no leading assignments, ``env`` or ``uv run``."""
    while words and _ASSIGNMENT.match(words[0]):
        words = words[1:]
    if words and words[0] == "env":
        words = words[1:]
        while words and (words[0].startswith("-") or _ASSIGNMENT.match(words[0])):
            words = words[1:]
    if len(words) > 1 and Path(words[0]).name in _RUNNERS and words[1] == "run":
        words = words[2:]
        while words and words[0].startswith("-"):
            words = words[1:]
    return words


def is_pytest(step: str) -> bool:
    """Whether the step's command word runs pytest: ``pytest``, or ``python -m pytest``."""
    try:
        words = _command_words(shlex.split(step))
    except ValueError:
        return False
    if not words:
        return False
    name = Path(words[0]).name
    if name in ("pytest", "py.test"):
        return True
    return bool(_PYTHON.match(name)) and "-m" in words and words.index("-m") + 1 < len(words) \
        and words[words.index("-m") + 1] == "pytest"


def _probe_step(step: str) -> str | None:
    """A cheap stand-in for one check step: pytest collects, any other program is looked up."""
    if is_pytest(step):
        return f"{step} --collect-only -q"
    try:
        words = _command_words(shlex.split(step))
    except ValueError:
        return None
    if not words:
        return None
    program = words[0]
    if program in _KEPT_BUILTINS:
        return step
    if any(char in program for char in "$`(){}<>|;&*?[]"):
        return None
    if "/" in program and not program.startswith(("/", *_INSTALLED)):
        return None  # a script in the repository: the change may be what adds it
    quoted = shlex.quote(program)
    return (f"{{ command -v {quoted} >/dev/null 2>&1 || "
            f"{{ echo {shlex.quote(f'{PROBE_MISSING} {program}')} >&2; exit 127; }}; }}")


def _has_alternative(command: str) -> bool:
    """Whether ``command`` chains steps with ``||`` (outside quotes, comments and subshells)."""
    steps = _steps(command)
    at = 0
    for step in steps[:-1]:
        end = command.index(step, at) + len(step)
        if command[end:].lstrip(" \t").startswith("||"):
            return True
        at = end
    return False


def probe_command(command: str) -> str | None:
    """A cheap probe of ``command``: whether it could run at all, without running any test.

    Each step pytest runs collects instead (``--collect-only -q``, every option kept);
    each other step's program is looked up on ``PATH``; ``cd``, ``source`` and the like
    stay, as the steps after them depend on them. The probes are joined with ``&&``, so
    one that fails is not hidden behind a later one. Comments are dropped first. None
    when no step can be probed, and for a command with ``a || b``: there one runnable
    branch is enough, and a looked-up fallback would pass whatever the first did.
    """
    command = strip_comments(command)
    if _has_alternative(command):
        return None
    steps = _steps(command)
    probes = [probe for probe in map(_probe_step, steps) if probe]
    if all(probe in steps for probe in probes):
        return None  # only builtins kept as they are: nothing is probed
    return " && ".join(probes)


def probe_fault(returncode: int | None, output: str) -> bool:
    """Whether a probe's failure is worth a warning: the check may not run at all.

    A command line pytest cannot parse (4), and a program that is missing or cannot run
    (127, 126, or one the probe looked up and did not find). A test path or id that is
    not on disk yet is not -- the change may add it -- and neither is a failing import,
    which the change may fix.
    """
    if PROBE_MISSING in output or returncode in (126, 127):
        return True
    if returncode != 4:
        return False
    lines = [line for line in strip_ansi(output).splitlines() if line.startswith("ERROR:")]
    return not lines or not all(_PYTEST_NOT_FOUND.match(line) for line in lines)


#: Where the warning says the check may not run: the run's worktree is at its base; a run
#: without one checks the artifact's own directory, which has no base.
AT_BASE = "at base"
IN_ARTIFACT = "in the artifact's directory"


def probe_warning(probe: str, returncode: int | None, output: str, *, where: str = AT_BASE) -> str:
    """What the approval card and the run's log say about a probe that found a fault.

    A program the probe looked up and did not find is named; otherwise the exit status
    and what it means. Then the tool's own words: the tail of its output.
    """
    lines = strip_ansi(output).splitlines()
    missing = [line.split(PROBE_MISSING, 1)[1].strip() for line in lines if PROBE_MISSING in line]
    if missing:
        head = f"`{missing[0]}` is not installed or not on the check's PATH"
    else:
        meaning = exit_meaning(returncode)
        head = f"its probe exited {returncode}" + (f" ({meaning})" if meaning else "")
    tail = "\n".join(line for line in lines[-12:] if line.strip())
    return f"Check may not run {where}: {head}. Probe: `{probe}`. Its output ends:\n{tail}"


def probe_check(ctx: Any, command: str) -> str:
    """Probe the check where it will run; a warning when it may not run at all, else "".

    Advisory: the change may add what the base lacks (a plugin, a tool, a test), so a
    probe never stops the run. Its warning goes to the run's log here and to the
    approval card (`design_ops.design_review_status`), where a person decides.

    In the run's worktree, still at its base, after the install its check makes -- the
    same `worktree.ensure_environment` call and record, so the check does not install
    again -- and with the check's environment. Without a worktree, in an empty directory
    with the environment the artifact's check gets. An install that fails, or a probe
    that times out or crashes, says nothing about the command: it is warned about in the
    log, and the check reports it as it always has.
    """
    probe = probe_command(command)
    if probe is None:
        return ""
    found = worktree.recorded(ctx.workdir)
    budget = getattr(ctx, "budget", None)
    if found is not None:
        _, problems, _ = worktree.ensure_environment(found, ctx.workdir, budget)
        if problems:
            target_guard.warn("the check could not be probed before research: its environment "
                              "could not be installed: " + "\n".join(problems)[:500])
            return ""
        where = Path(found.path)
        home = Path(ctx.workdir).resolve() / worktree.CHECK_HOME
        home.mkdir(parents=True, exist_ok=True)
        env = worktree.check_environment(where, home)
    else:
        where = Path(ctx.workdir) / PROBE_DIR
        where.mkdir(parents=True, exist_ok=True)
        home, env = None, sandbox.refactory_env()
    result = sandbox.run_command(probe, workdir=where, timeout_s=PROBE_TIMEOUT_S, budget=budget,
                                 env=env, home=home, exact_env=True)
    output = "\n".join((result.stdout or "", result.stderr or "")).strip()
    if result.ok or result.timed_out or sandbox.is_signal_crash(result)[0]:
        return ""
    if not probe_fault(result.returncode, output):
        return ""
    warning = probe_warning(probe, result.returncode, output,
                            where=AT_BASE if found is not None else IN_ARTIFACT)
    target_guard.warn(warning)
    return warning
