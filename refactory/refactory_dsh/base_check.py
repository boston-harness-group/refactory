"""Base-comparison gate: decide which check failures the attempt caused.

Before a failed check spends a code attempt, compare with the unchanged base in the
same environment. A failure the base has too is inherited, not the attempt's, only if
all three conditions hold:

(a) the same test identifier fails on both base and attempt;
(b) its test file is not one the attempt changes;
(c) its test id or leaf name appears in neither the brief text nor design.md.

Design mode uses one rule instead, in its self-check and its whole check alike
(`design_comparison`): a failure the base has too is inherited unless the brief or an
approved correction names the test or its file (`names_test`).

When the check cannot identify individual tests, no failure is excused (conservative
fallback), and an excusal stands only if the identified failures are the whole of what
failed and nothing was left unrun behind them (`excusal_blocker`). When the base
environment cannot be prepared, the run stops as unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from refactory_dsh import sandbox, worktree
from refactory_dsh.worktree import _json as _read_json, NO_HOOKS, _isolate_hooks
from refactory_dsh import target_guard

# ── ANSI stripping ───────────────────────────────────────────────────────

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_ESCAPE.sub('', text)


# ── parse test failures ──────────────────────────────────────────────────

_PYTEST_FAILED = re.compile(r"^FAILED\s+(\S+::.+)$", re.MULTILINE)
_VITEST_FAIL = re.compile(r"^\s*(?:FAIL|×)\s+(.+)$", re.MULTILINE)
_TEST_FILE_PATTERNS = re.compile(
    r"(?:^|/)(?:test_[\w]+\.py|[\w]+_test\.py|[\w]+\.test\.[tj]sx?|[\w]+\.spec\.[tj]sx?|test_[\w]+\.[tj]s)$"
)
_PYTEST_ERROR = re.compile(
    r"^(?:ERROR|ERRORS?)\s+(\S+\.py\b.*)$",
    re.MULTILINE,
)


def _strip_pytest_reason(raw: str) -> str:
    """Strip trailing `` - reason`` from a pytest FAILED/ERROR line.

    Respects balanced brackets so that parametrised IDs like ``test[a - b]``
    are not truncated.
    """
    raw = raw.strip()
    idx = len(raw)
    while True:
        pos = raw.rfind(" - ", 0, idx)
        if pos < 0:
            return raw
        before = raw[:pos]
        if before.count("[") <= before.count("]"):
            return before.rstrip()
        idx = pos

CACHE_FILE = Path(".refactory") / "base-check.json"
BASE_WORKTREE_PREFIX = "base-"


def parse_test_failures(output: str) -> set[str]:
    """Extract individual test identifiers from check output.

    Returns empty set when no recognisable test names are found (conservative fallback).
    ANSI codes are stripped before matching so coloured lines are not missed.
    """
    output = strip_ansi(output)
    found: set[str] = set()
    for m in _PYTEST_FAILED.finditer(output):
        found.add(_strip_pytest_reason(m.group(1)))
    # An error (a module that does not import, a fixture that fails) is a failure too.
    for m in _PYTEST_ERROR.finditer(output):
        tid = _strip_pytest_reason(m.group(1))
        # Reject log-capture lines.  A pytest ERROR line's ID is a bare file path
        # (``tests/test_x.py``) or a node ID (``tests/test_x.py::test_x``).  Log
        # lines look like ``root:module.py:12 msg`` or ``conftest.py:12 msg``
        # (colon-digit after ``.py``).
        if re.search(r"\.py:\d", tid):
            continue  # ``file.py:12`` is a log line, not a pytest node ID
        if "::" not in tid and "/" not in tid:
            if re.match(r"^\w+:", tid):
                continue  # logger:module pattern
        found.add(tid)
    for m in _VITEST_FAIL.finditer(output):
        candidate = m.group(1).strip()
        if " > " in candidate:
            found.add(candidate)
    return found


#: A flag that stops a runner at its first failure, so nothing after it runs: pytest's
#: ``-x`` (alone or in a cluster such as ``-xq``; ``-r`` takes report letters instead),
#: ``--exitfirst`` and ``--maxfail``, and vitest's and jest's ``--bail``.
_EXIT_FIRST = re.compile(r"^(?:-(?!r)[a-zA-Z]*x[a-zA-Z]*|--exitfirst|--maxfail(?:=.*)?|--bail(?:=.*)?)$")
_PYTEST_SUMMARY = re.compile(r"^=*\s*(\d+ (?:failed|passed|errors?|skipped)\b.*\bin [\d.]+s\b.*)$", re.MULTILINE)


def _steps(command: str) -> list[str]:
    """The shell steps ``command`` chains with ``&&``, ``||``, ``;``, ``|``, ``&`` or newlines.

    Quotes, backslashes and parentheses are respected; a step inside a subshell stays
    part of the step that opens it.
    """
    steps, current, depth, quote, i = [], "", 0, "", 0
    while i < len(command):
        char = command[i]
        if quote:
            current += char
            if char == "\\" and quote == '"' and i + 1 < len(command):
                current += command[i + 1]
                i += 1
            elif char == quote:
                quote = ""
        elif char == "\\" and i + 1 < len(command):
            current += command[i:i + 2]
            i += 1
        elif char in "'\"":
            quote = char
            current += char
        elif char == "(":
            depth += 1
            current += char
        elif char == ")":
            depth -= 1
            current += char
        elif char == "&" and (current.endswith((">", "<")) or command[i + 1:i + 2] == ">"):
            current += char  # a redirection (2>&1, &>log), not a step
        elif depth == 0 and (char in ";|&\n"):
            steps.append(current.strip())
            current = ""
            if command[i:i + 2] in ("&&", "||"):
                i += 1
        else:
            current += char
        i += 1
    steps.append(current.strip())
    return [step for step in steps if step]


# ── chain-step splitting for per-step excusal ───────────────────────────

#: Shell keywords that make a command unsplittable.
_UNSPLITTABLE_KEYWORDS = frozenset({
    "if", "then", "else", "elif", "fi",
    "for", "while", "until", "do", "done",
    "case", "esac",
})


def _chain_steps(command: str, *, reason: list[str] | None = None) -> list[tuple[str, str]] | None:
    """Split *command* on top-level ``&&``, ``||`` and ``;`` only.

    Returns a list of ``(step_text, operator)`` tuples where the first step's
    operator is ``""``.  Keeps ``|`` (pipe) and ``&`` (background) inside the
    step text, and drops comments (an unquoted ``#`` starting a word, to the end
    of its line).  Returns ``None`` when the command is unsplittable: a heredoc,
    a compound keyword (``if``/``case``/``for``/``while``), or a backtick
    substitution that spans an operator.

    When *reason* is a list, appends a human-readable string explaining why
    the command could not be split.
    """

    def _refuse(msg: str) -> None:
        if reason is not None:
            reason.append(msg)

    steps: list[tuple[str, str]] = []
    current = ""
    depth = 0        # parenthesis depth
    quote = ""       # current quoting character
    backtick = False # inside backtick substitution
    operator = ""    # operator before this step
    i = 0

    while i < len(command):
        char = command[i]

        if quote:
            current += char
            if char == "\\" and quote == '"' and i + 1 < len(command):
                current += command[i + 1]
                i += 2
                continue
            elif char == quote:
                quote = ""
            i += 1
            continue

        if char == "\\" and i + 1 < len(command):
            current += command[i:i + 2]
            i += 2
            continue

        if char in "'\"":
            quote = char
            current += char
            i += 1
            continue

        # An unquoted ``#`` that starts a word starts a comment, which runs to the end
        # of the line: nothing in it is a step, an operator or a runner name.
        if char == "#" and (not current or (current[-1] in " \t\n&|("
                                            and current[-2:-1] != "\\")):
            end = command.find("\n", i)
            i = len(command) if end == -1 else end
            continue

        if char == "`":
            backtick = not backtick
            current += char
            i += 1
            continue

        if char == "(":
            depth += 1
            current += char
            i += 1
            continue

        if char == ")":
            depth -= 1
            current += char
            i += 1
            continue

        # Redirection with & (e.g. 2>&1, &>log)
        if char == "&" and (current.endswith((">", "<")) or command[i + 1:i + 2] == ">"):
            current += char
            i += 1
            continue

        # At top level (depth 0): check for split operators
        if depth == 0:
            # Check for heredoc: unquoted << not preceded by < and not <<<
            if char == "<" and command[i + 1:i + 2] == "<" and command[i + 2:i + 3] != "<":
                # Check it's not preceded by another <
                if not current.rstrip().endswith("<"):
                    _refuse("heredoc (unquoted <<) detected")
                    return None

            # Check for && or ||
            if char == "&" and command[i + 1:i + 2] == "&":
                # Backtick substitution spans an operator: unsplittable
                if backtick:
                    _refuse("backtick substitution spans && operator")
                    return None
                step_text = current.strip()
                if step_text:
                    # Check first word for shell keywords
                    try:
                        words = shlex.split(step_text)
                    except ValueError:
                        words = step_text.split()
                    if words and words[0] in _UNSPLITTABLE_KEYWORDS:
                        _refuse(f"shell keyword '{words[0]}' in step")
                        return None
                    steps.append((step_text, operator))
                operator = "&&"
                current = ""
                i += 2
                continue

            if char == "|" and command[i + 1:i + 2] == "|":
                if backtick:
                    _refuse("backtick substitution spans || operator")
                    return None
                step_text = current.strip()
                if step_text:
                    try:
                        words = shlex.split(step_text)
                    except ValueError:
                        words = step_text.split()
                    if words and words[0] in _UNSPLITTABLE_KEYWORDS:
                        _refuse(f"shell keyword '{words[0]}' in step")
                        return None
                    steps.append((step_text, operator))
                operator = "||"
                current = ""
                i += 2
                continue

            if char == ";":
                if backtick:
                    _refuse("backtick substitution spans ; operator")
                    return None
                step_text = current.strip()
                if step_text:
                    try:
                        words = shlex.split(step_text)
                    except ValueError:
                        words = step_text.split()
                    if words and words[0] in _UNSPLITTABLE_KEYWORDS:
                        _refuse(f"shell keyword '{words[0]}' in step")
                        return None
                    steps.append((step_text, operator))
                operator = ";"
                current = ""
                i += 1
                continue

        # Pipe (not ||) stays in current step
        current += char
        i += 1

    # Handle the last step
    step_text = current.strip()
    if step_text:
        try:
            words = shlex.split(step_text)
        except ValueError:
            words = step_text.split()
        if words and words[0] in _UNSPLITTABLE_KEYWORDS:
            _refuse(f"shell keyword '{words[0]}' in step")
            return None
        steps.append((step_text, operator))

    # If backtick is still open, the substitution spans operators
    if backtick:
        _refuse("unclosed backtick substitution spans operators")
        return None

    for step_text, _operator in steps:
        unsafe = _unsafe_step(step_text)
        if unsafe:
            _refuse(unsafe)
            return None

    return steps if steps else None


def _unsafe_step(step: str) -> str | None:
    """Why ``step`` cannot run in the shell the chain's steps share; None when it can.

    The marker script runs every step past a failure itself, so a step that ends,
    replaces (``exec``) or traps that shell, runs a file's commands in it (``.``,
    ``source``: the file may turn on errexit), or turns on errexit in a way
    `_without_errexit` cannot take back out, would stop the later steps' markers.
    """
    try:
        words = shlex.split(step)
    except ValueError:
        words = step.split()
    if not words:
        return None
    if words[0] in ("exit", "trap", "exec"):
        return f"a `{words[0]}` step ends, replaces or traps the shell the steps share"
    if words[0] in (".", "source"):
        return f"a `{words[0]}` step runs a file's commands in the shell the steps share"
    if words[0] == "set":
        if _without_errexit(step) is None:
            return "a `set` step whose options cannot be read"
        return None
    if re.search(r"\berrexit\b|\bSHELLOPTS\b", step) or (
            words[0] in ("eval", "builtin", "command") and "set" in words):
        return "a step that may turn on errexit some other way than `set`"
    return None


def _without_errexit(step: str) -> str | None:
    """A ``set`` step without errexit (``-e``, ``-o errexit``), its other options kept.

    ``set -euo pipefail`` becomes ``set -u -o pipefail``; ``set -e`` alone becomes ``:``.
    A step that is not ``set`` is returned as it is; None when its options cannot be read
    (a variable, a missing ``-o`` name).
    """
    try:
        words = shlex.split(step)
    except ValueError:
        return None
    if not words or words[0] != "set":
        return step
    if len(words) == 1:
        return step
    if any(char in step for char in "$`"):
        return None
    kept = ["set"]
    i = 1
    while i < len(words):
        word = words[i]
        if word in ("-", "--") or not word.startswith(("-", "+")):
            kept += words[i:]  # positional parameters
            break
        if not re.fullmatch(r"[-+][A-Za-z]+", word):
            return None
        sign, letters = word[0], word[1:]
        name = None
        if "o" in letters:
            if not letters.endswith("o") or letters.count("o") > 1 or i + 1 >= len(words):
                return None
            name = words[i + 1]
            letters = letters[:-1]
            i += 1
        if sign == "-":
            letters = letters.replace("e", "")
        if letters:
            kept.append(sign + letters)
        if name is not None and not (sign == "-" and name == "errexit"):
            kept += [sign + "o", name]
        i += 1
    return shlex.join(kept) if len(kept) > 1 else ":"


def _build_marker_script(steps: list[tuple[str, str]], marker_id: str) -> str:
    """A shell script that runs the chain's steps in one shell, each followed by a marker.

    It begins with ``exec 2>&1``, so each step's output includes what it printed to
    stderr (runners print summaries and errors there). A step after ``&&`` runs even
    when the chain has failed by then -- running what ``&&`` skips is the point -- but
    it leaves the chain's status as the shell would have it. A step after ``||`` runs
    only when the shell would run it, on the chain's status so far, not only on the
    previous step's exit: with ``a || b || c`` and ``a`` passing, neither ``b`` nor
    ``c`` runs, and its marker says so. A ``set`` step runs without errexit
    (`_without_errexit`), its other options kept.
    """
    status, code = "__refactory_status", "__refactory_exit"
    lines = ["exec 2>&1", f"{status}=0"]
    for idx, (step_text, operator) in enumerate(steps):
        # errexit would end the shell at the first failing step, before the later markers.
        step_text = _without_errexit(step_text) or step_text
        mark = f"___REFACTORY_STEP_{marker_id}_{idx}_"
        record = f"{code}=$?; echo '{mark}EXIT_'${code}"
        if operator == "||":
            lines += [f'if [ "${status}" -ne 0 ]; then', step_text, record,
                      f"{status}=${code}", f"else echo '{mark}SKIPPED'; fi"]
        elif operator == "&&":
            lines += [step_text, record, f'if [ "${status}" -eq 0 ]; then {status}=${code}; fi']
        else:
            lines += [step_text, record, f"{status}=${code}"]
    return "\n".join(lines)


def _parse_step_results(
    stdout: str, marker_id: str, steps: list[tuple[str, str]],
) -> list[dict] | None:
    """Parse marker lines from stdout.

    Returns a list of dicts with ``step``, ``operator``, ``ran``, ``output`` and
    ``exit_code`` (None for a step that did not run) per step, or ``None`` when
    markers are missing, out of order or incomplete.
    """
    pattern = (r'___REFACTORY_STEP_' + re.escape(marker_id)
               + r'_(\d+)_(?:EXIT_(\d+)|SKIPPED)')
    markers = list(re.finditer(pattern, stdout))
    if len(markers) != len(steps):
        return None

    results = []
    prev_end = 0
    for i, m in enumerate(markers):
        if int(m.group(1)) != i:
            return None  # out of order
        ran = m.group(2) is not None
        # The step's output is everything between the previous marker and this one.
        output = stdout[prev_end:m.start()].rstrip()
        prev_end = m.end()
        if prev_end < len(stdout) and stdout[prev_end] == '\n':
            prev_end += 1
        step_text, operator = steps[i]
        results.append({
            "step": step_text,
            "operator": operator,
            "ran": ran,
            "output": output if ran else "",
            "exit_code": int(m.group(2)) if ran else None,
        })
    return results


def _roots(*places: tuple[Path | str, str]) -> list[tuple[str, str]]:
    """Run-specific paths with the placeholder each becomes in a step's `_signature`."""
    roots = []
    for path, label in places:
        roots += [(str(path), label), (str(Path(path).resolve()), label)]
    return roots


def _normalized(output: str, roots: list[tuple[str, str]]) -> str:
    """A step's output with each side's own paths -- the exact worktree or check copy,
    HOME and run directory it ran with -- replaced. Any other path, under /tmp or not,
    stays as printed, so a new error that differs only there still differs."""
    text = strip_ansi(output)
    for path, label in sorted(roots, key=lambda root: -len(root[0])):
        if path and path != "/":
            text = text.replace(path, label)
    return text


def _signature(output: str, roots: list[tuple[str, str]]) -> str:
    """`_normalized` output, hashed, so the same output on base and attempt matches."""
    return hashlib.sha256(_normalized(output, roots).encode()).hexdigest()


def _step_record(sr: dict, roots: list[tuple[str, str]] = ()) -> dict:  # type: ignore[assignment]
    """A parsed step with what the comparison reads: its failures, runner, accounting and signature."""
    failures = parse_test_failures(sr["output"]) if sr["ran"] else set()
    runner = (_runner(sr["step"]) or _runner_from_output(sr["output"])) if sr["ran"] else None
    # A runner that passed names nothing and counts nothing failed: nothing is unaccounted.
    passed = sr["exit_code"] == 0 and not failures
    return {**sr, "failures": sorted(failures), "runner": runner,
            "accounted": runner is not None and (passed or _accounted(runner, sr["output"], failures)),
            "signature": _signature(sr["output"], list(roots)),
            "quiet": not _normalized(sr["output"], list(roots)).strip()}


def _run_chain_markers(
    steps: list[tuple[str, str]], run: Callable[[str], Any],
    roots: list[tuple[str, str]] = (),  # type: ignore[assignment]
) -> list[dict] | None:
    """Run all chain steps in one shell with markers, returning per-step records.

    ``run`` runs a script where the side being compared runs its check, and returns
    its `sandbox.SandboxResult`; ``roots`` are that side's own paths (`_signature`).
    Returns ``None``, with a warning, when the per-step
    results cannot be read: a timeout, a signal crash, or markers missing (the kept
    output was cut, or a step ended the shell).
    """
    marker_id = uuid.uuid4().hex[:12]
    result = run(_build_marker_script(steps, marker_id))
    if result.timed_out:
        target_guard.warn("the chain's per-step run timed out; its steps cannot be compared")
        return None
    crashed, signal_name = sandbox.is_signal_crash(result)
    if crashed:
        target_guard.warn(f"the chain's per-step run was {signal_name}; its steps cannot be compared")
        return None
    stdout = "\n".join((result.stdout or "", result.stderr or "")).strip()
    parsed = _parse_step_results(stdout, marker_id, steps)
    if parsed is None:
        found = len(re.findall(r'___REFACTORY_STEP_' + re.escape(marker_id) + r'_', stdout))
        target_guard.warn(f"the chain's per-step results could not be read: {found} of "
                          f"{len(steps)} step markers in its kept output")
        return None
    return [_step_record(sr, roots) for sr in parsed]


def _runner(step: str) -> str | None:
    """``pytest`` or ``vitest`` when ``step`` names that runner (jest reports like vitest)."""
    import shlex

    try:
        words = shlex.split(step)
    except ValueError:
        return None
    names = {Path(word).name for word in words}
    if "pytest" in names or "py.test" in names:
        return "pytest"
    if "vitest" in names or "jest" in names:
        return "vitest"
    return None


def _runner_from_output(output: str) -> str | None:
    """Detect the runner from its output when the step text does not name it.

    Falls back to ``None`` when neither pytest's nor vitest's summary is found.
    """
    if _PYTEST_SUMMARY.search(output):
        return "pytest"
    if re.search(r"^\s*Tests\s+\d+\s+(?:failed|passed)", output, re.MULTILINE):
        return "vitest"
    return None


def _accounted(runner: str, output: str, failures: set[str]) -> bool:
    """Whether the runner's own summary counts exactly the failures that were identified."""
    if runner == "pytest":
        summaries = _PYTEST_SUMMARY.findall(output)
        if not summaries:
            return False
        last = summaries[-1]
        counted = sum(int(n) for n in re.findall(r"(\d+) (?:failed|errors?)\b", last))
        return counted == len(failures)
    tests = re.findall(r"^\s*Tests\s+(\d+) failed", output, re.MULTILINE)
    files = re.findall(r"^\s*Test Files\s+(\d+) failed", output, re.MULTILINE)
    if not tests:
        return False
    named_files = {failure.split(" > ")[0].strip() for failure in failures}
    return int(tests[-1]) == len(failures) and (not files or int(files[-1]) <= len(named_files))


def chain_incomplete(command: str, run: Callable[[str], Any],
                     roots: list[tuple[str, str]] = ()) -> str | None:  # type: ignore[assignment]
    """Why a failed ``command`` may not have run all its steps; None when it did.

    A chain such as ``pytest a && pytest b`` stops at its first failing step, so a
    failure the base also has can hide that the steps after it never ran. The chain's
    steps run again one by one (`_run_chain_markers`, through ``run``), and the shell's
    own rule says which the real run skipped: a step after ``&&`` once the chain has
    failed. A step after ``||`` the shell skips after a success is not missing. A
    command whose steps cannot be split or read is never called complete.
    """
    if len(_steps(command)) <= 1:
        return None
    chain = _chain_steps(command)
    if chain is None:
        return "the check's steps cannot be told apart, so whether each one ran is unknown"
    if len(chain) <= 1:
        return None
    steps = _run_chain_markers(chain, run, roots)
    if steps is None:
        return ("the check's steps could not be run one by one, so whether each one ran "
                "is unknown")
    status = 0
    for sr in steps:
        if sr["operator"] == "&&" and status != 0:
            return f"step `{sr['step']}` never ran: an earlier step of the check failed"
        if sr["ran"]:
            status = sr["exit_code"]
    return None


def excusal_blocker(command: str, output: str, failures: set[str]) -> str | None:
    """Why the check's failures, all inherited, still cannot excuse it; None when they can.

    Matching tests is not enough: the check must have failed only for them, and nothing
    may have gone unrun behind them. So the failing runner has to be the command's last
    step, named so it can be recognised; it must not stop at its first failure; and its
    own summary must count exactly the failures identified. A later step, a module that
    does not import, or a step skipped after a failing one is otherwise unaccounted for.
    """
    steps = _steps(command)
    runner = _runner(steps[-1]) if steps else None
    if runner is None:
        return "the check's last step is not a test runner refactory recognises (pytest, vitest, jest)"
    kind = {"pytest": lambda failure: " > " not in failure, "vitest": lambda failure: " > " in failure}[runner]
    if not all(kind(failure) for failure in failures):
        return f"a failure is not from the check's last step ({runner}), so what follows it never ran"
    import shlex

    if any(_EXIT_FIRST.match(word) for word in shlex.split(steps[-1])):
        return f"{runner} stops at its first failure here, so the tests after it never ran"
    if not _accounted(runner, output, failures):
        return f"{runner}'s summary counts failures that are not identified by name"
    return None


def _parse_collection_errors(output: str) -> tuple[set[str], bool]:
    """Extract test files that had collection or import errors.

    Returns ``(files, has_unlocatable)`` where *has_unlocatable* is True when
    a collection error was detected but the importing file could not be
    determined (e.g. vitest's "does not provide an export named").
    """
    files: set[str] = set()
    has_unlocatable = False
    for m in _PYTEST_ERROR.finditer(output):
        ident = m.group(1).strip()
        # Extract just the file path (before any ::)
        path = ident.split("::")[0] if "::" in ident else ident
        if path.endswith(".py"):
            files.add(path)
    # Also catch "ImportError while importing" and "ModuleNotFoundError" patterns
    for line in output.splitlines():
        if "ImportError" in line or "ModuleNotFoundError" in line or "CollectionError" in line:
            # Try to extract the file from the surrounding context
            m = re.search(r"([\w/.-]+\.py)", line)
            if m:
                files.add(m.group(1))
    # Vitest collection errors — use non-greedy .*? to capture the full file path
    for m in re.finditer(r'Failed to collect\s+([\w/.-]+?\.(?:spec\.ts|ts))', output):
        files.add(m.group(1))
    for m in re.finditer(r'Error:.*?Cannot find module.*?([\w/.-]+?\.(?:spec\.ts|ts))', output):
        files.add(m.group(1))
    # "does not provide an export named" — record the importing file from surrounding output
    for m in re.finditer(r"does not provide an export named\s+'(\w+)'", output):
        found_file = False
        for line in output.splitlines():
            if m.group(0) in line:
                fm = re.search(r'([\w/.-]+?\.(?:spec\.ts|ts))', line)
                if fm:
                    files.add(fm.group(1))
                    found_file = True
                    break
        if not found_file:
            has_unlocatable = True
    # "Failed to load url" pattern
    for line in output.splitlines():
        if "Failed to load url" in line:
            m = re.search(r'([\w/.-]+?\.(?:spec\.ts|ts))', line)
            if m:
                files.add(m.group(1))
            else:
                has_unlocatable = True
    return files, has_unlocatable


def cache_key(base: str, setup: str, generate: str, check: str) -> str:
    payload = json.dumps(
        {"base": base, "setup": setup, "generate": generate, "check": check},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ── data types ───────────────────────────────────────────────────────────


@dataclass
class BaseCheckResult:
    ok: bool
    failures: set[str]
    output: str
    exit_code: int
    environment_failure: str | None = None
    #: Each failure's signature (`extract_failure_signatures`), when a targeted run kept it.
    signatures: dict[str, str | None] | None = None


@dataclass
class ComparisonResult:
    inherited: set[str]
    new_failures: set[str]
    all_excused: bool
    base_ok: bool
    reasons: dict[str, str] | None = None


@dataclass(frozen=True)
class JudgeResult:
    """The unified verdict on which failures are the change's."""
    charged: frozenset[str]       # test IDs the change is responsible for
    excused: frozenset[str]       # test IDs inherited from the base
    reasons: dict[str, str]       # test_id → classification reason
    all_excused: bool             # True iff charged is empty and excused is non-empty


@dataclass
class FilteredCheckResult:
    all_excused: bool
    inherited: set[str]
    new_failures: set[str]
    new_failure_detail: str
    inherited_detail: str
    base_environment_failure: str | None
    guardedness: dict | None
    base_failures: set[str] | None = None
    base_complete: bool = False
    base_crashed: bool = False
    reasons: dict[str, str] | None = None


# ── failure signatures ───────────────────────────────────────────────────

_FAILURES_SECTION = re.compile(r"^={3,}\s*FAILURES\s*={3,}$", re.MULTILINE)
_HEADER_LINE = re.compile(r"^_{3,}\s*(.+?)\s*_{3,}$", re.MULTILINE)


def extract_failure_signatures(output: str, test_ids: set[str]) -> dict[str, str | None]:
    """Extract failure signatures for the given test IDs from check output.

    A signature is 'exception_type:file_path:line_number' extracted from the
    traceback following a FAILED line. Returns None for a test ID whose
    traceback cannot be found (e.g. --tb=no).
    """
    clean = strip_ansi(output)
    result: dict[str, str | None] = {tid: None for tid in test_ids}

    # Find the FAILURES section
    section_match = _FAILURES_SECTION.search(clean)
    if section_match is None:
        return result

    section_start = section_match.end()
    # Find the end of the section (next === line)
    section_end_match = re.search(r"^={3,}", clean[section_start:], re.MULTILINE)
    section_text = clean[section_start:section_start + section_end_match.start()] if section_end_match else clean[section_start:]

    headers = list(_HEADER_LINE.finditer(section_text))
    for i, hdr in enumerate(headers):
        header_name = hdr.group(1).strip()
        block_start = hdr.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(section_text)
        block = section_text[block_start:block_end]

        # Match header name to test IDs — the header may be just the test name part.
        # Use exact segment matching, not endswith, to avoid matching
        # 'base_test_foo' to 'test_foo' (review finding m1-sig-4).
        matched_tid = None
        for tid in test_ids:
            if header_name == tid or tid.endswith("::" + header_name):
                matched_tid = tid
                break
            # The last :: segment must match exactly
            last_part = tid.split("::")[-1]
            if header_name == last_part:
                matched_tid = tid
                break
        if matched_tid is None:
            # Try matching on any :: segment exactly
            for tid in test_ids:
                parts = tid.split("::")
                if any(header_name == p for p in parts[1:]):
                    matched_tid = tid
                    break
        if matched_tid is None:
            continue

        # Extract the signature: last exception line + file:line
        lines = block.strip().splitlines()
        exc_type = None
        file_line = None
        for line in reversed(lines):
            line_stripped = line.strip()
            # Exception line: may start with pytest's "E   " marker, then
            # an uppercase word followed by Error/Exception/Failure/Warning.
            exc_match = re.match(r'^(?:E\s+)?([A-Z]\w*(?:Error|Exception|Failure|Warning))', line_stripped)
            if exc_type is None and exc_match:
                exc_type = exc_match.group(1)
            # File:line pattern
            if file_line is None:
                fl_match = re.search(r'(\S+\.py):(\d+)', line_stripped)
                if fl_match:
                    file_line = f"{fl_match.group(1)}:{fl_match.group(2)}"
        if exc_type and file_line:
            result[matched_tid] = f"{exc_type}:{file_line}"

    return result


# ── unified failure judge ────────────────────────────────────────────────


def judge_failures(
    attempt_failures: set[str],
    base_failures: set[str],
    *,
    named_text: str | None = None,
    changed_files: list[str] | None = None,
    brief_text: str = "",
    design_text: str = "",
    per_step_base_failures: set[str] | None = None,
    targeted_base_failures_set: set[str] | None = None,
    targeted_attempt_failures: set[str] | None = None,
    attempt_signatures: dict[str, str | None] | None = None,
    base_signatures: dict[str, str | None] | None = None,
) -> JudgeResult:
    """Classify attempt failures as charged or excused.

    A pure function (no I/O): callers collect the observations and pass them in.
    Every check path uses this single classifier.
    """
    if not attempt_failures:
        return JudgeResult(
            charged=frozenset(), excused=frozenset(),
            reasons={}, all_excused=False,
        )

    charged: set[str] = set()
    excused: set[str] = set()
    reasons: dict[str, str] = {}

    search_text = (brief_text + "\n" + design_text).lower() if named_text is None else ""
    changed_set = set(changed_files) if changed_files else set()

    all_base = set(base_failures)
    if per_step_base_failures:
        all_base |= per_step_base_failures
    if targeted_base_failures_set:
        all_base |= targeted_base_failures_set

    for failure in attempt_failures:
        # Step 1: Named test — always charged
        if named_text is not None:
            if names_test(named_text, failure):
                charged.add(failure)
                reasons[failure] = "named-test"
                continue
        else:
            # Fast mode: check if any part appears in brief_text + design_text
            if any(part.lower() in search_text for part in _test_name_parts(failure)):
                charged.add(failure)
                reasons[failure] = "named-test"
                continue

        # Step 2: A failure in a test file the change touched. A base match (whole run,
        # per step or targeted) excuses it only when both sides' failure signatures are
        # available and equal; otherwise the change may have broken it differently, so
        # it is charged. Fast modes also charge one the base does not fail at all.
        test_file = _test_file(failure)
        if test_file and changed_set and any(
                path == test_file or path.endswith("/" + test_file) for path in changed_set):
            if failure in all_base:
                a_sig = (attempt_signatures or {}).get(failure)
                b_sig = (base_signatures or {}).get(failure)
                if a_sig is None or b_sig is None or a_sig != b_sig:
                    charged.add(failure)
                    reasons[failure] = "signature-mismatch"
                    continue
            elif named_text is None:
                charged.add(failure)
                reasons[failure] = "touched-file"
                continue

        # Step 4: Base matching
        if failure in base_failures:
            excused.add(failure)
            reasons[failure] = "base-whole-run"
            continue
        if per_step_base_failures and failure in per_step_base_failures:
            excused.add(failure)
            reasons[failure] = "base-per-step"
            continue
        if targeted_base_failures_set and failure in targeted_base_failures_set:
            excused.add(failure)
            reasons[failure] = "base-targeted"
            continue

        # Step 5: Targeted confirmation
        if targeted_attempt_failures is not None:
            if failure not in targeted_attempt_failures:
                excused.add(failure)
                reasons[failure] = "not-reproduced"
                continue
            if targeted_base_failures_set and failure in targeted_base_failures_set:
                excused.add(failure)
                reasons[failure] = "base-targeted"
                continue
            charged.add(failure)
            reasons[failure] = "new-failure"
            continue

        # No targeted info available — charged
        charged.add(failure)
        reasons[failure] = "new-failure"

    return JudgeResult(
        charged=frozenset(charged),
        excused=frozenset(excused),
        reasons=reasons,
        all_excused=not charged and bool(excused),
    )


# ── three-condition excusal ──────────────────────────────────────────────


def _test_name_parts(identifier: str) -> list[str]:
    """The parts of a test identifier to search for in the brief/design.

    For "tests/test_a.py::TestClass::test_method", yields:
    ["test_method", "TestClass::test_method", "tests/test_a.py::TestClass::test_method"]
    """
    parts = []
    if "::" in identifier:
        segments = identifier.split("::")
        for i in range(len(segments) - 1, -1, -1):
            parts.append("::".join(segments[i:]))
    elif " > " in identifier:
        segments = identifier.split(" > ")
        for i in range(len(segments) - 1, -1, -1):
            parts.append(" > ".join(segments[i:]))
    else:
        parts.append(identifier)
    return parts


def _test_file(identifier: str) -> str | None:
    """Extract the test file path from a test identifier, or a file-level error's file."""
    if identifier.endswith((".py", ".ts", ".tsx", ".js", ".jsx")):
        return identifier
    if "::" in identifier:
        return identifier.split("::")[0]
    if " > " in identifier:
        return identifier.split(" > ")[0].strip()
    return None


def compare_failures(
    base_failures: set[str],
    attempt_failures: set[str],
    changed_files: list[str],
    brief_text: str,
    design_text: str,
    *,
    targeted_base_failures_set: set[str] | None = None,
    targeted_attempt_failures: set[str] | None = None,
    per_step_base_failures: set[str] | None = None,
    attempt_signatures: dict[str, str | None] | None = None,
    base_signatures: dict[str, str | None] | None = None,
) -> ComparisonResult:
    """Classify attempt failures as inherited or new using the three-condition rule.

    Thin wrapper around ``judge_failures`` preserving the public signature.
    """
    jr = judge_failures(
        attempt_failures, base_failures,
        named_text=None,
        changed_files=changed_files,
        brief_text=brief_text,
        design_text=design_text,
        targeted_base_failures_set=targeted_base_failures_set,
        targeted_attempt_failures=targeted_attempt_failures,
        per_step_base_failures=per_step_base_failures,
        attempt_signatures=attempt_signatures,
        base_signatures=base_signatures,
    )
    return ComparisonResult(
        inherited=set(jr.excused),
        new_failures=set(jr.charged),
        all_excused=jr.all_excused,
        base_ok=not bool(base_failures),
    )


# ── design mode's rule: what the base fails, less what the asks name ─────

#: A test file's extensions, which a name that is followed by one is part of.
_FILE_SUFFIX = r"\.(?:py|[cm]?[tj]sx?)\b"


def names_test(text: str, test_id: str) -> bool:
    """Whether ``text`` names ``test_id``: by its id, a qualified or leaf name, or its file.

    A name counts only whole: ``test_a`` is not named by ``test_all_paths``, nor by
    ``tests.test_a`` or ``test_a.py`` (a module and a file, which the file rule reads).
    A file counts as a path or a trailing part of one (``tests/test_a.py`` is named by
    ``refactory/tests/test_a.py`` and by ``test_a.py``), but not when it is followed by
    ``::``: that names one test in the file, which the id rule reads.
    """
    if not text or not test_id:
        return False
    parts = _test_name_parts(test_id)
    # A parametrized test is named by its name without the parameters too.
    parts += [re.sub(r"\[[^\]]*\]$", "", part) for part in parts if part.endswith("]")]
    for part in parts:
        if re.search(rf"(?<![\w/.]){re.escape(part)}(?!\w)(?!{_FILE_SUFFIX})", text, re.IGNORECASE):
            return True
    path = _test_file(test_id)
    if not path:
        return False
    pieces = path.split("/")
    for start in range(len(pieces)):
        tail = "/".join(pieces[start:])
        if re.search(rf"(?<![\w.\-]){re.escape(tail)}(?![\w/\-])(?!\.\w)(?!::)", text,
                     re.IGNORECASE):
            return True
    return False


def design_comparison(base_failures: set[str], attempt_failures: set[str],
                      named_text: str, *,
                      targeted_base_failures_set: set[str] | None = None,
                      targeted_attempt_failures: set[str] | None = None,
                      per_step_base_failures: set[str] | None = None,
                      changed_files: list[str] | None = None,
                      attempt_signatures: dict[str, str | None] | None = None,
                      base_signatures: dict[str, str | None] | None = None,
                      ) -> ComparisonResult:
    """Design mode's one inheritance rule — thin wrapper around ``judge_failures``.

    A failure is inherited when the unchanged base fails the same test, whatever file the
    attempt touched and whatever design.md says, unless ``named_text`` (the brief and
    the approved corrections) names it (`names_test`): a test the run exists to fix is
    sent back while it fails.
    """
    jr = judge_failures(
        attempt_failures, base_failures,
        named_text=named_text,
        changed_files=changed_files,
        targeted_base_failures_set=targeted_base_failures_set,
        targeted_attempt_failures=targeted_attempt_failures,
        per_step_base_failures=per_step_base_failures,
        attempt_signatures=attempt_signatures,
        base_signatures=base_signatures,
    )
    return ComparisonResult(
        inherited=set(jr.excused),
        new_failures=set(jr.charged),
        all_excused=jr.all_excused,
        base_ok=not bool(base_failures),
    )


# ── base worktree lifecycle ──────────────────────────────────────────────


def _base_worktree_path(target: Path, run_id: str) -> Path:
    return worktree.slot(target) / f"{BASE_WORKTREE_PREFIX}{run_id}"


def _check_base_clean(base_path: Path, base_commit: str, base_found: worktree.Worktree) -> list[str]:
    """Check that the base worktree is clean; repair it if dirty.

    Returns the list of tracked paths that were repaired (empty if already clean).
    Warns via target_guard.warn listing repaired tracked paths only.
    On failure, rebuilds the base worktree (remove and re-add) or raises as a crash.
    """
    # What the base installed and git does not ignore stays, as worktree._restore keeps
    # it: the clean below keeps it, so neither status check may count it as dirt.
    kept = {name.rstrip("/") for name in
            _read_json(worktree._record_path(base_found)).get("installed", [])}

    def _dirt(listing: str) -> list[str]:
        return [entry for entry in listing.split("\0") if len(entry) > 3
                and not (entry[:2] == "??" and entry[3:].rstrip("/") in kept)]

    status_out = worktree.git(
        base_path, "--no-optional-locks", "status", "--porcelain=v1", "-z",
        "--untracked-files=normal",
    )
    if not _dirt(status_out):
        return []

    # Parse NUL-delimited entries to find tracked edits for the warning.
    tracked_edits: list[str] = []
    for entry in status_out.split("\0"):
        if not entry or len(entry) < 4:
            continue
        code = entry[:2]
        path = entry[3:]
        if code != "??":
            tracked_edits.append(path)

    # Reset tracked files to the base commit.
    try:
        worktree.git(base_path, *NO_HOOKS, "reset", "-q", "--hard", base_commit)
    except ValueError:
        raise ValueError(f"base worktree reset failed (crashed): git reset --hard {base_commit[:12]}")

    # Remove untracked files/dirs, preserving installs from the install record.
    keep = [argument for name in sorted(kept) for argument in ("-e", f"/{name}")]
    try:
        worktree.git(base_path, "clean", "-fdq", *keep)
    except ValueError:
        raise ValueError(f"base worktree clean failed (crashed): git clean -fdq")

    # Re-check: status must be empty now.
    recheck = worktree.git(
        base_path, "--no-optional-locks", "status", "--porcelain=v1", "-z",
        "--untracked-files=normal",
    )
    if _dirt(recheck):
        raise ValueError(
            f"base worktree still dirty after reset: {recheck[:200]}"
        )

    # Warn only about restored tracked files (not __pycache__ or other untracked),
    # following worktree._restore's pattern.
    if tracked_edits:
        target_guard.warn(
            f"base worktree repaired: restored {len(tracked_edits)} tracked "
            f"file(s) to {base_commit[:12]}: {', '.join(sorted(tracked_edits)[:10])}"
        )

    return tracked_edits


def _rebuild_base_worktree(target: Path, run_id: str, base_path: Path) -> None:
    """Remove and re-add the base worktree from scratch."""
    try:
        worktree.git(target, "worktree", "remove", "--force", str(base_path))
    except ValueError:
        shutil.rmtree(base_path, ignore_errors=True)


def _ensure_base_worktree(ctx: Any) -> tuple[Path, worktree.Worktree, list[str]]:
    """Create and prepare the base worktree on demand, caching it per run.

    Returns the base worktree path, a Worktree descriptor for it, and a list of
    tracked paths that were repaired (empty when the base was already clean).
    Raises ValueError with an environment-failure message on failure.
    """
    found = worktree.recorded(ctx.workdir)
    if found is None:
        raise ValueError("no worktree recorded for this run")

    target = Path(found.target)
    run_id = Path(found.path).name.removeprefix("run-")
    base_path = _base_worktree_path(target, run_id)

    # Create the worktree if it does not exist or points to the wrong commit
    if base_path.exists():
        try:
            head = worktree.git(base_path, "rev-parse", "HEAD").strip()
            if head != found.base:
                worktree.git(target, "worktree", "remove", "--force", str(base_path))
                worktree.git(
                    target, *NO_HOOKS, "worktree", "add", "-q", "--detach",
                    str(base_path), found.base,
                )
        except ValueError as exc:
            raise ValueError(
                f"could not verify base worktree for {found.base}: {exc}"
            ) from exc
    else:
        try:
            worktree.git(
                target, *NO_HOOKS, "worktree", "add", "-q", "--detach",
                str(base_path), found.base,
            )
        except ValueError as exc:
            raise ValueError(
                f"could not create base worktree for {found.base}: {exc}"
            ) from exc

    # Build a Worktree descriptor for the base; use a synthetic branch name so
    # ensure_environment's record path does not collide with the run's own.
    base_found = worktree.Worktree(
        path=str(base_path),
        branch=f"{worktree.BRANCH_PREFIX}base-{run_id}",
        base=found.base,
        head=found.base,
        target=found.target,
        run=found.run,
    )

    # Layer 2: check and repair dirty base before any caller uses it.
    # On failure, rebuild from scratch rather than becoming a cached env failure.
    try:
        repaired = _check_base_clean(base_path, found.base, base_found)
    except ValueError:
        # Reset failed — rebuild the base worktree from scratch.
        _rebuild_base_worktree(target, run_id, base_path)
        try:
            worktree.git(
                target, *NO_HOOKS, "worktree", "add", "-q", "--detach",
                str(base_path), found.base,
            )
        except ValueError as exc:
            raise ValueError(
                f"could not rebuild base worktree for {found.base}: {exc}"
            ) from exc
        # Rebuild the descriptor for the fresh worktree.
        base_found = worktree.Worktree(
            path=str(base_path),
            branch=f"{worktree.BRANCH_PREFIX}base-{run_id}",
            base=found.base,
            head=found.base,
            target=found.target,
            run=found.run,
        )
        repaired = []  # Fresh worktree is clean.

    # Accumulate base_reset on ctx for check-record.json.
    if repaired:
        prev = getattr(ctx, '_base_reset', [])
        ctx._base_reset = prev + repaired

    _isolate_hooks(target, base_path)

    # Prepare the environment (installs). A lockfile the install writes stays in the base
    # worktree, uncommitted: the base is a fixed commit, and an amend would also point
    # the run's worktree record at the base (dm-188).
    budget = getattr(ctx, "budget", None)
    _, problems, _ = worktree.ensure_environment(base_found, ctx.workdir, budget, commit_lockfiles=False)
    if problems:
        unavailable = [p for p in problems if "UNAVAILABLE:" in p]
        if unavailable:
            raise ValueError("\n".join(unavailable))
        raise ValueError(
            "environment preparation failed on the base:\n" + "\n".join(problems)
        )

    return base_path, base_found, repaired


def remove_base_worktree(target: Path, run_id: str) -> None:
    """Remove the base worktree when the run ends, and the reference worktree delivery's
    merged-result check prepared (`targeted_base_failures` with ``reference``)."""
    for path in (_base_worktree_path(target, run_id), _reference_worktree_path(target, run_id)):
        if path.exists():
            try:
                worktree.git(target, "worktree", "remove", "--force", str(path))
            except ValueError:
                # Best effort — the worktree may already be gone
                shutil.rmtree(path, ignore_errors=True)


def _reference_worktree_path(target: Path, run_id: str) -> Path:
    return worktree.slot(target) / f"ref-{run_id}"


# ── identify changed files ───────────────────────────────────────────────


def _changed_test_files(path: Path, base: str) -> list[str]:
    try:
        changed = worktree.git(
            path, "diff", "--name-only", "--diff-filter=d", base, "HEAD"
        )
    except ValueError:
        return []
    return [f for f in changed.splitlines() if _TEST_FILE_PATTERNS.search(f)]


def _all_changed_files(path: Path, base: str) -> list[str]:
    try:
        changed = worktree.git(path, "diff", "--name-only", base, "HEAD")
    except ValueError:
        return []
    return [f.strip() for f in changed.splitlines() if f.strip()]


def _changed_files_in_test_dirs(path: Path, base: str, test_files: list[str]) -> list[str]:
    """All new or changed files in the same directories as the test files.

    Covers helpers, fixtures and other files the tests import (correction 3).
    The project root is excluded: a test at the root imports production code,
    and copying that code into the base worktree defeats the guardedness check.
    Only proper test subdirectories are considered.
    """
    if not test_files:
        return []
    test_dirs = {str(Path(tf).parent) for tf in test_files}
    # A test file at the project root (parent == ".") means every changed file
    # in the root would be copied — including production code the test is meant
    # to exercise.  Exclude the root so only actual test subdirectories supply
    # helpers and fixtures.
    test_dirs.discard(".")
    if not test_dirs:
        return []
    all_changed = _all_changed_files(path, base)
    return [
        f for f in all_changed
        if f not in test_files and any(
            f.startswith(d + "/") or str(Path(f).parent) == d
            for d in test_dirs
        )
    ]


# ── run base check ───────────────────────────────────────────────────────


def _read_cache(run: Path, key: str) -> BaseCheckResult | None:
    data = _read_json(run / CACHE_FILE)
    if data.get("cache_key") == key:
        if data.get("base_clean") is not True:
            return None  # dirty-base or old entry — re-run
        # Guard: reject d4b-shaped entries — base failed without naming any
        # failing test and without an environment_failure explanation.
        if not data["ok"] and not data.get("failures") and data.get("environment_failure") is None:
            return None  # d4b-shaped entry: base failed without failures or environment_failure
        result = BaseCheckResult(
            ok=data["ok"],
            failures=set(data.get("failures", [])),
            output=data.get("output", ""),
            exit_code=data.get("exit_code", 1),
            environment_failure=data.get("environment_failure"),
        )
        # Attach base_complete as a transient attribute for filter_check (R5).
        result._base_complete = data.get("base_complete", False)  # type: ignore[attr-defined]
        # Attach per-step results from cache if present.
        cached_steps = data.get("steps")
        if cached_steps is not None:
            result._step_results = cached_steps  # type: ignore[attr-defined]
        if data.get("steps_unreadable"):
            result._steps_unreadable = True  # type: ignore[attr-defined]
        return result
    return None


def _write_cache(run: Path, key: str, result: BaseCheckResult,
                 *, base_clean: bool = True) -> None:
    # Never cache a crash result — R2: later checks get a fresh attempt.
    # Deterministic (non-crash) environment failures ARE cached so filter_check
    # and the next attempt do not re-prepare the base (v5 correction 2).
    if getattr(result, '_crashed', False):
        return  # pragma: no cover — guarded by run_base_check
    if not base_clean:
        target_guard.warn("skipping base-check cache write (base was dirty)")
        return
    cache_dir = run / CACHE_FILE.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Compute completeness for R5/R6.
    base_complete = getattr(result, '_base_complete', False)
    data = {
        "cache_key": key,
        "ok": result.ok,
        "failures": sorted(result.failures),
        "output": result.output[-10000:],
        "exit_code": result.exit_code,
        "environment_failure": result.environment_failure,
        "base_complete": base_complete,
        "base_clean": True,
    }
    # Per-step results from chain marker runs.
    step_results = getattr(result, '_step_results', None)
    if step_results is not None:
        data["steps"] = step_results
    if getattr(result, '_steps_unreadable', False):
        data["steps_unreadable"] = True
    (run / CACHE_FILE).write_text(json.dumps(data, indent=2) + "\n")


def run_base_check(ctx: Any, command: str, timeout: float) -> BaseCheckResult:
    """Run the check on the unchanged base in a separate worktree, caching the result.

    When any step crashes (a signal that survives ``run_command``'s retry),
    the base's HOME, worktree and cache are discarded and rebuilt from scratch
    (R2).  On double failure the result is returned with ``environment_failure``
    set but never cached, so later checks get a fresh attempt.  Deterministic
    (non-crash) failures are cached as before.
    """
    found = worktree.recorded(ctx.workdir)
    if found is None:
        return BaseCheckResult(ok=True, failures=set(), output="", exit_code=0)

    setup = str(ctx.facts.get("setup") or "").strip()
    generate = str(ctx.facts.get("generate") or "").strip()
    key = cache_key(found.base, setup, generate, command)

    cached = _read_cache(ctx.workdir, key)
    if cached is not None:
        return cached

    base_clean_flag = True  # Track whether the base was clean for cache writes.

    def _try_base() -> tuple[BaseCheckResult, bool]:
        """Attempt to prepare the base worktree and run the check.

        Returns ``(result, is_crash)``.  ``is_crash`` is True when a signal
        killed a preparation step and the base environment should be rebuilt.
        """
        nonlocal base_clean_flag
        try:
            base_path, base_found, repaired = _ensure_base_worktree(ctx)
            if repaired:
                base_clean_flag = False
        except ValueError as exc:
            err_str = str(exc)
            # Classify: use structured result from ensure_environment.
            # prep_fail returned by ensure_environment carries returncode.
            is_crash = any(tok in err_str for tok in ("killed by SIG", "killed by signal"))
            r = BaseCheckResult(
                ok=False, failures=set(), output=err_str, exit_code=-1,
                environment_failure=err_str,
            )
            r._crashed = is_crash  # type: ignore[attr-defined]
            return r, is_crash

        home = Path(ctx.workdir).resolve() / worktree.BASE_CHECK_HOME
        home.mkdir(parents=True, exist_ok=True)
        budget_inner = getattr(ctx, "budget", None)
        env = worktree.check_environment(base_path, home)

        from refactory_dsh.repo_change import prepare_commands as _prep

        _, problems, prep_fail = _prep(
            ctx, base_found, env, home, timeout, budget_inner, amend=False)
        if problems:
            # Check if any problem is a signal crash.
            is_crash = False
            if prep_fail is not None and isinstance(prep_fail.get("returncode"), int):
                prc = prep_fail["returncode"]
                is_crash = prc < 0 or prc in sandbox._AMBIGUOUS_SHELL_EXITS
            detail = "on the base: " + "\n".join(problems)
            r = BaseCheckResult(
                ok=False, failures=set(), output=detail, exit_code=-1,
                environment_failure=detail,
            )
            r._crashed = is_crash  # type: ignore[attr-defined]
            return r, is_crash

        check_result = sandbox.run_command(
            command, workdir=base_path, timeout_s=timeout,
            budget=budget_inner, env=env, home=home, exact_env=True,
        )
        crashed_chk, sig_chk = sandbox.is_signal_crash(check_result)
        if crashed_chk:
            r = BaseCheckResult(
                ok=False, failures=set(), output=f"base check command {sig_chk}",
                exit_code=-1, environment_failure=f"base check command {sig_chk}",
            )
            r._crashed = True  # type: ignore[attr-defined]
            return r, True

        output = "\n".join(
            (check_result.stdout or "", check_result.stderr or "")
        ).strip()
        failures = parse_test_failures(output)

        result = BaseCheckResult(
            ok=check_result.ok, failures=failures,
            output=output, exit_code=check_result.returncode or 0,
        )
        # Compute completeness for R5.
        blocker = excusal_blocker(command, output, failures) if failures else None
        result._base_complete = (blocker is None) if failures else check_result.ok  # type: ignore[attr-defined]

        # A failed chain of two or more steps: run its steps again in one shell, each
        # recorded, so the comparison can tell which step every failure came from and
        # how each earlier step ended -- whatever the whole-command blocker says, because
        # a base that stopped at its first step looks complete to it.
        if not check_result.ok:
            unsplit_reason: list[str] = []
            chain = _chain_steps(command, reason=unsplit_reason)
            if chain is None and unsplit_reason:
                target_guard.warn(f"base chain not split: {unsplit_reason[0]}")
            if chain is not None and len(chain) > 1:
                step_results = _run_chain_markers(chain, lambda script: sandbox.run_command(
                    script, workdir=base_path, timeout_s=timeout,
                    budget=budget_inner, env=env, home=home, exact_env=True,
                ), _roots((base_path, "<root>"), (home, "<home>"), (ctx.workdir, "<run>")))
                if step_results is None:
                    # Recorded, so the comparison charges rather than reading it as a cache
                    # written before per-step results existed.
                    result._steps_unreadable = True  # type: ignore[attr-defined]
                else:
                    for sr in step_results:
                        result.failures |= set(sr["failures"])
                    # Complete when every test runner that ran counts exactly the
                    # failures it named.
                    result._base_complete = all(  # type: ignore[attr-defined]
                        sr["accounted"] for sr in step_results
                        if sr["ran"] and sr["runner"] is not None)
                    result._step_results = [  # type: ignore[attr-defined]
                        {**sr, "output": sr["output"][-5000:]} for sr in step_results]

        # Guard: an unfinished base (a timeout, whatever it printed before it was
        # stopped, or a non-zero exit with no failures and no per-step results) is
        # never cached -- not as clean, not as failure-free, not as the failures it
        # named before it stopped.  Applied AFTER the per-step re-run.
        if check_result.timed_out:
            result.environment_failure = (
                f"the base check was stopped at its time limit "
                f"(exit {check_result.returncode})")
            return result, False
        if not check_result.ok and not result.failures:
            if _runner(command) or _runner_from_output(output):
                # The check command names or produced output from a test runner
                # (pytest/vitest/etc.) but named no failures: environment refusal
                # (canary-repair case — credentials guard, etc.).
                result.environment_failure = (
                    f"the base check exited {check_result.returncode} "
                    "without naming a failing test")
            # Not cached at all — neither as clean nor as environment failure
            # (correction 2).  The _read_cache guard rejects existing entries.
            return result, False  # not a crash, but not cached either

        return result, False

    result, is_crash = _try_base()
    if is_crash:
        # Discard and rebuild from scratch — R2.
        home = Path(ctx.workdir).resolve() / worktree.BASE_CHECK_HOME
        shutil.rmtree(home, ignore_errors=True)
        if found is not None:
            run_id = Path(found.path).name.removeprefix("run-")
            remove_base_worktree(Path(found.target), run_id)
        (ctx.workdir / CACHE_FILE).unlink(missing_ok=True)
        result, is_crash = _try_base()
        if is_crash or result.environment_failure:
            # Double crash or deterministic failure after rebuild.
            # Discard HOME and the base worktree so the NEXT check gets a
            # fresh start too (R2: never reuse a base environment whose
            # preparation crashed — the b27a rerun failure).
            home = Path(ctx.workdir).resolve() / worktree.BASE_CHECK_HOME
            shutil.rmtree(home, ignore_errors=True)
            if found is not None:
                run_id = Path(found.path).name.removeprefix("run-")
                remove_base_worktree(Path(found.target), run_id)
            (ctx.workdir / CACHE_FILE).unlink(missing_ok=True)
            result._crashed = is_crash  # type: ignore[attr-defined]
            return result

    # An unfinished base (environment_failure set by the non-crash guard) is
    # never cached — the caller falls back to targeted base runs.
    if not result.environment_failure:
        _write_cache(ctx.workdir, key, result, base_clean=base_clean_flag)
    return result


#: What the self-check learned about single tests on the base (b41a), by `cache_key`, and
#: whether it prepared the base for that key already.
TARGETED_CACHE_FILE = Path(".refactory") / "base-targeted.json"


def targeted_base_failures(ctx: Any, command: str, test_ids: list[str], *,
                           prepare_timeout: float, timeout: float,
                           reference: str | None = None) -> tuple[BaseCheckResult, str]:
    """Which of ``test_ids`` fail on the unchanged base: the self-check's comparison (b41a).

    It is the base the check compares with. When the check's own base result is cached
    for this base and command (`run_base_check`), that answers. Otherwise only these
    tests run, in the base worktree `_ensure_base_worktree` prepares, after the run's
    generate and setup there, with the base's own HOME, as `run_base_check` runs the
    whole check. What a run learns about a test is kept, so a fix-up does not prepare
    the base again for the same tests.

    When ``reference`` is provided (e.g. for delivery's merged-result check), the
    targeted tests run at that commit in a separate worktree with a cache key that
    includes ``reference`` and does not collide with the in-run base cache.

    A base that cannot be prepared, runs out of ``timeout`` or reports no result is
    unavailable: ``environment_failure`` says why, and nothing may be inherited from it.
    Returns the result and where it came from.
    """
    found = worktree.recorded(ctx.workdir)

    def unavailable(why: str) -> tuple[BaseCheckResult, str]:
        return BaseCheckResult(ok=False, failures=set(), output=why, exit_code=-1,
                               environment_failure=why), "unavailable"

    if found is None:
        return unavailable("the run has no worktree, so it has no base")

    # When reference is provided, use it as the base commit for cache and worktree.
    base_commit = reference if reference else found.base

    setup = str(ctx.facts.get("setup") or "").strip()
    generate = str(ctx.facts.get("generate") or "").strip()
    key = cache_key(base_commit, setup, generate, command)

    # Only check the in-run base cache when not using a different reference.
    if not reference:
        cached = _read_cache(ctx.workdir, key)
        if cached is not None and not cached.environment_failure:
            cached.signatures = extract_failure_signatures(
                cached.output, cached.failures & set(test_ids))
            return cached, "the check's base result"

    # Use a separate targeted cache file when reference is provided.
    if reference:
        ref_short = reference[:12]
        targeted_cache_name = Path(".refactory") / f"base-targeted-{ref_short}.json"
    else:
        targeted_cache_name = TARGETED_CACHE_FILE
    known_path = ctx.workdir / targeted_cache_name
    known = _read_json(known_path)
    same_key = known.get("cache_key") == key
    learned: dict[str, bool] = dict(known.get("tests", {})) if same_key else {}
    signatures: dict[str, str | None] = dict(known.get("signatures", {})) if same_key else {}
    missing = [test_id for test_id in test_ids if test_id not in learned]
    source = "earlier runs of these tests on the base"
    if missing:
        from refactory_dsh.check_digest import build_probe_command
        from refactory_dsh.failfast import self_check_failures
        from refactory_dsh.repo_change import prepare_commands

        home = Path(ctx.workdir).resolve() / worktree.BASE_CHECK_HOME
        home.mkdir(parents=True, exist_ok=True)
        budget = getattr(ctx, "budget", None)
        base_path = _base_worktree_path(Path(found.target), Path(found.path).name.removeprefix("run-"))

        # When reference differs from found.base, use a separate worktree so the
        # in-run base worktree is never contaminated (correction 6).
        if reference and reference != found.base:
            base_path = _reference_worktree_path(
                Path(found.target), Path(found.path).name.removeprefix("run-"))
            home = Path(ctx.workdir).resolve() / ".refactory" / "ref-check-home"
            home.mkdir(parents=True, exist_ok=True)
            if not (same_key and known.get("prepared") and _at(base_path, reference)):
                why = _prepare_reference(ctx, found, base_path, reference, home,
                                         prepare_timeout)
                if why:
                    return unavailable(why)
                _write_targeted(known_path, key, learned, prepared=True, signatures=signatures)
        elif not (same_key and known.get("prepared") and _at(base_path, base_commit)):
            try:
                base_path, base_found, _ = _ensure_base_worktree(ctx)
            except ValueError as exc:
                return unavailable(f"the base could not be prepared: {exc}")
            _, problems, _ = prepare_commands(ctx, base_found, worktree.check_environment(base_path, home),
                                              home, prepare_timeout, budget, amend=False)
            if problems:
                return unavailable("the base's generate or setup failed: " + "\n".join(problems)[:1000])
            _write_targeted(known_path, key, learned, prepared=True, signatures=signatures)
        from refactory_dsh.test_selection import runnable

        env = worktree.check_environment(base_path, home)
        targets = list(dict.fromkeys(runnable(test_id) for test_id in missing))
        files = [name for name in targets if "::" not in name and " > " not in name]
        ids = [name for name in targets if name not in files]
        targeted, refusal = build_probe_command(command, ids, root=base_path, test_files=files)
        if refusal:
            return unavailable(f"the tests cannot be targeted on the base: {refusal}")
        result = sandbox.run_command(targeted, workdir=base_path, timeout_s=timeout, budget=budget,
                                     env=env, home=home, exact_env=True)
        if result.timed_out:
            return unavailable(f"the base run of {len(missing)} test(s) was stopped at its "
                               f"{timeout:.0f} s limit")
        crashed, signal = sandbox.is_signal_crash(result)
        if crashed:
            return unavailable(f"the base run {signal}")
        output = "\n".join((result.stdout or "", result.stderr or "")).strip()
        failed = self_check_failures(output)
        if not result.ok and not failed:
            return unavailable(f"the base run failed (exit {result.returncode}) without "
                               "naming a failing test")
        learned.update({test_id: test_id in failed for test_id in missing})
        signatures.update(extract_failure_signatures(
            output, {test_id for test_id in missing if test_id in failed}))
        _write_targeted(known_path, key, learned, prepared=True, signatures=signatures)
        source = f"a run of {len(missing)} test(s) on the base"
    failures = {test_id for test_id in test_ids if learned.get(test_id)}
    return BaseCheckResult(ok=not failures, failures=failures, output="", exit_code=0,
                           signatures={test_id: signatures.get(test_id)
                                       for test_id in failures}), source


def _prepare_reference(ctx: Any, found: worktree.Worktree, path: Path, reference: str,
                       home: Path, timeout: float) -> str:
    """Put a worktree at ``reference`` at ``path`` and prepare it as the base worktree is
    prepared (`_ensure_base_worktree`): its installs, then the run's generate and setup,
    run there with ``home``. Returns why it could not be prepared; empty when it was.

    The targeted cache records that it was prepared, per ``reference``, so a later call
    for the same reference runs its tests without preparing again.
    """
    from refactory_dsh.repo_change import prepare_commands

    target = Path(found.target)
    run_id = Path(found.path).name.removeprefix("run-")

    def add() -> None:
        if path.is_dir():
            worktree.git(target, *worktree.NO_HOOKS, "worktree", "remove", "--force", str(path))
        worktree.git(target, *worktree.NO_HOOKS, "worktree", "add", "-q", "--detach",
                     str(path), reference)

    if not _at(path, reference):
        try:
            add()
        except ValueError:
            try:
                worktree.git(target, *worktree.NO_HOOKS, "fetch", "origin", reference)
                add()
            except ValueError as exc:
                return f"could not create a worktree at reference {reference[:12]}: {exc}"
    _isolate_hooks(target, path)
    ref_found = worktree.Worktree(
        path=str(path), branch=f"{worktree.BRANCH_PREFIX}ref-{run_id}",
        base=reference, head=reference, target=found.target, run=found.run)
    budget = getattr(ctx, "budget", None)
    _, problems, _ = worktree.ensure_environment(ref_found, ctx.workdir, budget,
                                                 commit_lockfiles=False)
    if problems:
        return "the reference's install failed: " + "\n".join(problems)[:1000]
    _, problems, _ = prepare_commands(ctx, ref_found, worktree.check_environment(path, home),
                                      home, timeout, budget, amend=False)
    if problems:
        return "the reference's generate or setup failed: " + "\n".join(problems)[:1000]
    return ""


def _at(path: Path, commit: str) -> bool:
    """Whether the worktree at ``path`` exists and is checked out at ``commit``."""
    try:
        return path.is_dir() and worktree.git(path, "rev-parse", "HEAD").strip() == commit
    except ValueError:
        return False


def _write_targeted(path: Path, key: str, learned: dict[str, bool], *, prepared: bool,
                    signatures: dict[str, str | None] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cache_key": key, "prepared": prepared, "tests": learned,
                                "signatures": signatures or {}},
                               indent=2) + "\n")


def targeted_attempt_failures(ctx: Any, command: str, test_ids: list[str], *,
                              timeout: float,
                              place: tuple[Path, dict[str, str], Path] | None = None,
                              signatures: dict[str, str | None] | None = None,
                              ) -> set[str] | None:
    """Run ``test_ids`` targeted on the attempt and return which ones fail.

    Returns ``None`` when the attempt cannot be re-run (the targeted confirmation
    then falls through to the "no targeted info" branch of the judge, which
    charges the failure).

    It runs in the run's check copy (`repo_change._check_copy`), at the attempt's
    commit with the copy's own environment and HOME, so reviewers working in the run's
    worktree are not disturbed. ``place`` -- a directory, its environment and HOME --
    runs it somewhere already prepared instead: delivery's merged result.

    A test that passes when run alone may fail only in its suite's order, after
    another test in its file left state behind. So each id that passes alone runs
    again with its whole file; it counts as passing only when it passes there too.
    ``signatures``, when given, receives each failing id's signature from its run alone.
    """
    from refactory_dsh.check_digest import build_probe_command
    from refactory_dsh.failfast import self_check_failures
    from refactory_dsh.test_selection import runnable

    found = worktree.recorded(ctx.workdir)
    if found is None:
        return None
    budget = getattr(ctx, "budget", None)
    if place is None:
        from refactory_dsh import repo_change

        copy, env, home, problems, _ = repo_change._check_copy(ctx, found, timeout)
        if problems:
            return None
        where = Path(copy.path)
    else:
        where, env, home = place

    def run(ids: list[str], files: list[str]) -> set[str] | None:
        targeted_cmd, refusal = build_probe_command(command, ids, root=where, test_files=files)
        if refusal:
            return None
        result = sandbox.run_command(targeted_cmd, workdir=where, timeout_s=timeout,
                                     budget=budget, env=env, home=home, exact_env=True)
        if result.timed_out or sandbox.is_signal_crash(result)[0]:
            return None
        output = "\n".join((result.stdout or "", result.stderr or "")).strip()
        if not result.ok and (result.returncode or 0) != 1:
            # Exit code 1 means test failures; 2+ means collection errors or an
            # interrupt -- inconclusive.
            return None
        failing = self_check_failures(output)
        if signatures is not None:
            for test_id, sig in extract_failure_signatures(
                    output, failing & set(test_ids)).items():
                signatures.setdefault(test_id, sig)
        return failing

    targets = list(dict.fromkeys(runnable(test_id) for test_id in test_ids))
    files = [name for name in targets if "::" not in name and " > " not in name]
    failed = run([name for name in targets if name not in files], files)
    if failed is None:
        return None
    for test_id in test_ids:
        # An id run as its whole file (`runnable`) fails when anything in that file does.
        name = runnable(test_id)
        if name != test_id and any(f == name or f.startswith(name + "::") for f in failed):
            failed.add(test_id)
    passed_alone = [test_id for test_id in test_ids
                    if test_id not in failed and runnable(test_id) not in failed]
    modules = sorted({module for test_id in passed_alone
                      if (module := _test_file(test_id)) and module not in files})
    if modules:
        in_order = run([], modules)
        for test_id in passed_alone:
            module = _test_file(test_id)
            if module not in modules:
                continue
            # Inconclusive, or failing in its file's order: not a pass.
            if in_order is None or test_id in in_order or module in in_order:
                failed.add(test_id)
    return failed


# ── prep triage: compare a preparation failure against the base ──────────

PREP_CACHE_FILE = Path(".refactory") / "base-prep.json"


def _read_prep_cache(run: Path, key: str) -> dict | None:
    """Read the cached base-prep comparison result, if it matches the key."""
    data = _read_json(run / PREP_CACHE_FILE)
    if data.get("cache_key") == key:
        if data.get("base_clean") is not True:
            return None  # dirty-base or old entry — re-run
        return data
    return None


def _write_prep_cache(run: Path, key: str, result: dict,
                      *, base_clean: bool = True) -> None:
    """Write the base-prep comparison result to cache."""
    if not base_clean:
        target_guard.warn("skipping base-prep cache write (base was dirty)")
        return
    cache_dir = run / PREP_CACHE_FILE.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = {"cache_key": key, "base_clean": True, **result}
    (run / PREP_CACHE_FILE).write_text(json.dumps(data, indent=2) + "\n")


def _step_label(step_cmd: str, generate: str, setup: str) -> str:
    """Classify a prep step command as generate, setup or install."""
    if step_cmd == generate:
        return "generate"
    if step_cmd == setup:
        return "setup"
    return "install"


_STEP_ORDER = {"install": 0, "generate": 1, "setup": 2}


def _fails_differently(returncode: Any, error_text: str, base_returncode: Any,
                       base_error: str, found_wt: worktree.Worktree) -> bool:
    """Whether the attempt's preparation failed otherwise than the base's at the same step.

    A different exit code, when the base's is known, or a different error once what
    differs between the two places only by where they run reads alike: each worktree's
    path as ``<root>``, each one's HOME as ``<home>``, the run directory as ``<run>`` and
    the temporary directory as ``<tmp>``; and every run of digits as ``#``, since the
    exit code is compared on its own and a PID, a timestamp, a duration or a log file
    named by the time is not a different failure.
    """
    import tempfile

    from refactory_dsh.invocations import BASE_CHECK_HOME, CHECK_HOME

    if base_returncode is not None and returncode != base_returncode:
        return True
    run_id = Path(found_wt.path).name.removeprefix("run-")
    run_dir = Path(found_wt.run)
    places = {
        str(found_wt.path): "<root>",
        str(_base_worktree_path(Path(found_wt.target), run_id)): "<root>",
        str(run_dir / CHECK_HOME): "<home>",
        str(run_dir / BASE_CHECK_HOME): "<home>",
        str(run_dir): "<run>",
        tempfile.gettempdir(): "<tmp>",
    }
    places.update({str(Path(place).resolve()): name for place, name in list(places.items())})

    def normal(text: str) -> str:
        for place in sorted(places, key=len, reverse=True):
            text = text.replace(place, places[place])
        return re.sub(r"\d+", "#", text).strip()

    return normal(error_text) != normal(base_error)


def _resolve_prep_comparison(
    base_ok: bool,
    base_step: str | None,
    base_error: str | None,
    prep_fail: dict[str, Any],
    problems: list[str],
    found_wt: worktree.Worktree,
    step_label_val: str,
    generate: str,
    setup: str,
    base_returncode: int | None = None,
) -> dict[str, Any] | None:
    """Given a base-prep result, decide whether the failure is change-caused.

    Returns a change-caused dict or None (inherited / infrastructure). When both fail
    the same step, the attempt's failure is the change's if it names a file the change
    touched that the base's does not, or if the base's names a file the change touched
    and the attempt fails differently (`_fails_differently`): the base failed on what
    the change replaced, so its failure says nothing about the attempt's.
    """
    error_text = "\n".join(problems)
    step = prep_fail["step"]

    if base_ok:
        return {
            "step": step, "step_label": step_label_val,
            "returncode": prep_fail["returncode"],
            "error_text": error_text, "command": step,
        }

    # Base also had problems.
    if base_step == step:
        # Same step — check the "names files the attempt changed" rule.
        changed = _all_changed_files(Path(found_wt.path), found_wt.base)
        if changed:
            safe_base_error = base_error or ""
            for f in changed:
                basename = Path(f).name
                if (f in error_text and f not in safe_base_error) or \
                   (basename in error_text and basename not in safe_base_error):
                    return {
                        "step": step, "step_label": step_label_val,
                        "returncode": prep_fail["returncode"],
                        "error_text": error_text, "command": step,
                    }
            base_names_changed = any(f in safe_base_error or Path(f).name in safe_base_error
                                     for f in changed)
            if base_names_changed and _fails_differently(
                    prep_fail.get("returncode"), error_text, base_returncode,
                    safe_base_error, found_wt):
                return {
                    "step": step, "step_label": step_label_val,
                    "returncode": prep_fail["returncode"],
                    "error_text": error_text, "command": step,
                }
        return None  # Same failure, no changed file uniquely named → inherited.

    # Different steps.
    # No base step means _ensure_base_worktree failed: the base failed its install.
    base_label = _step_label(base_step, generate, setup) if base_step else "install"
    attempt_order = _STEP_ORDER.get(step_label_val, 0)
    base_order = _STEP_ORDER.get(base_label, 0)
    if base_order > attempt_order:
        # Base fails a later step → base passed the attempt's failing step.
        return {
            "step": step, "step_label": step_label_val,
            "returncode": prep_fail["returncode"],
            "error_text": error_text, "command": step,
        }
    return None  # Base fails earlier → inherited.


def compare_prep_failure(
    ctx: Any,
    prep_fail: dict[str, Any] | None,
    problems: list[str],
    found: worktree.Worktree,
    env: dict[str, str],
    home: Path,
    timeout: float,
) -> dict[str, Any] | None:
    """Compare a preparation failure against the unchanged base.

    Returns a dict when the failure is change-caused and should go to
    the builder for repair.  Returns None when the failure is
    infrastructure or inherited (today's handling).
    """
    from refactory_dsh import authoring_evidence

    # ── 1. Infrastructure fault guard ────────────────────────────────────
    if prep_fail is None:
        return None
    rc = prep_fail.get("returncode")
    if isinstance(rc, int) and rc < 0:
        return None
    if prep_fail.get("timed_out"):
        return None
    if rc in (126, 127):
        return None
    if rc in sandbox._AMBIGUOUS_SHELL_EXITS:
        return None
    if rc in authoring_evidence._RESOURCE_EXIT_CODES:
        return None

    # ── 2. Determine step label ──────────────────────────────────────────
    generate = str(ctx.facts.get("generate") or "").strip()
    setup = str(ctx.facts.get("setup") or "").strip()
    step_label_val = _step_label(prep_fail["step"], generate, setup)

    found_wt = worktree.recorded(ctx.workdir)
    if found_wt is None:
        return None

    # ── 3. Check the per-run prep cache ──────────────────────────────────
    key = cache_key(found_wt.base, setup, generate, "")
    cached = _read_prep_cache(ctx.workdir, key)
    if cached is not None:
        return _resolve_prep_comparison(
            cached["base_ok"], cached.get("base_step"),
            cached.get("base_error"),
            prep_fail, problems, found_wt, step_label_val, generate, setup,
            cached.get("base_returncode"),
        )

    # ── 4. Prepare the base worktree and run its prep commands ───────────
    prep_base_clean = True  # Track whether the base was clean for cache writes.

    def _try_base_prep() -> tuple[dict, bool]:
        nonlocal prep_base_clean
        try:
            base_path, base_found, repaired = _ensure_base_worktree(ctx)
            if repaired:
                prep_base_clean = False
        except ValueError as exc:
            err = str(exc)
            crashed = any(tok in err for tok in ("killed by SIG", "killed by signal",
                                                 "exceeded its time limit"))
            return {"base_ok": False, "base_step": None,
                    "base_error": err, "base_returncode": None}, crashed

        base_home = Path(ctx.workdir).resolve() / worktree.BASE_CHECK_HOME
        base_home.mkdir(parents=True, exist_ok=True)
        budget = getattr(ctx, "budget", None)
        base_env = worktree.check_environment(base_path, base_home)

        from refactory_dsh.repo_change import prepare_commands as _prep_cmds
        _, base_problems, base_pf = _prep_cmds(
            ctx, base_found, base_env, base_home, timeout, budget, amend=False,
        )

        is_crash = False
        if base_pf is not None:
            brc = base_pf.get("returncode")
            is_crash = (
                (isinstance(brc, int) and brc < 0)
                or base_pf.get("timed_out", False)
                or brc in sandbox._AMBIGUOUS_SHELL_EXITS
            )

        return {
            "base_ok": not bool(base_problems),
            "base_step": base_pf["step"] if base_pf else None,
            "base_error": "\n".join(base_problems) if base_problems else None,
            "base_returncode": base_pf.get("returncode") if base_pf else None,
        }, is_crash

    result_dict, is_crash = _try_base_prep()

    if is_crash:
        base_home = Path(ctx.workdir).resolve() / worktree.BASE_CHECK_HOME
        shutil.rmtree(base_home, ignore_errors=True)
        run_id = Path(found_wt.path).name.removeprefix("run-")
        remove_base_worktree(Path(found_wt.target), run_id)
        (ctx.workdir / PREP_CACHE_FILE).unlink(missing_ok=True)

        result_dict, is_crash = _try_base_prep()
        if is_crash:
            base_home = Path(ctx.workdir).resolve() / worktree.BASE_CHECK_HOME
            shutil.rmtree(base_home, ignore_errors=True)
            run_id = Path(found_wt.path).name.removeprefix("run-")
            remove_base_worktree(Path(found_wt.target), run_id)
            return None

    # ── 5. Cache (never cache a crash or timeout) ────────────────────────
    _write_prep_cache(ctx.workdir, key, result_dict, base_clean=prep_base_clean)

    return _resolve_prep_comparison(
        result_dict["base_ok"], result_dict.get("base_step"),
        result_dict.get("base_error"),
        prep_fail, problems, found_wt, step_label_val, generate, setup,
        result_dict.get("base_returncode"),
    )


# ── test guardedness ─────────────────────────────────────────────────────


def _infer_test_runner(check_command: str) -> str | None:
    if "pytest" in check_command:
        return "pytest"
    if "vitest" in check_command or "jest" in check_command:
        return "vitest"
    return None


def compute_test_guardedness(
    ctx: Any, command: str, timeout: float,
) -> dict | None:
    """Run the attempt's new/changed test files against the base source.

    Returns a guardedness record dict; when the tests cannot be run on the base, the
    record says why under ``unavailable``. None only when the run has no worktree. Computed per attempt (never cached). Creates the base worktree on
    demand if it does not already exist.
    """
    found = worktree.recorded(ctx.workdir)
    if found is None:
        return None

    attempt_path = Path(found.path)
    new_test_files = _changed_test_files(attempt_path, found.base)
    if not new_test_files:
        return {
            "new_test_files": [],
            "results": [],
            "summary": {
                "guarding": 0, "not_guarding": 0, "total_new_tests": 0,
            },
        }

    runner = _infer_test_runner(command)
    if runner is None:
        return {"new_test_files": sorted(new_test_files),
                "unavailable": "the check command runs neither pytest nor vitest, so the new tests "
                               "could not be run on the base"}

    # Ensure the base worktree exists (on demand, correction 1)
    try:
        base_path, _, _ = _ensure_base_worktree(ctx)
    except ValueError as exc:
        return {"new_test_files": sorted(new_test_files),
                "unavailable": f"the base worktree could not be prepared: {exc}"}

    # Copy the attempt's new/changed test files AND their helpers/fixtures
    # into the base worktree (correction 3)
    helper_files = _changed_files_in_test_dirs(attempt_path, found.base, new_test_files)
    all_to_copy = new_test_files + helper_files

    # Layer 1: determine which destinations exist in the base commit using git ls-tree.
    # This distinguishes pre-existing tracked files (restore from base) from new files
    # (unlink after) and pre-existing ignored files (save and write back bytes).
    pre_existing: set[str] = set()
    new_copied: list[Path] = []
    saved_ignored: dict[str, bytes] = {}

    try:
        ls_output = worktree.git(
            base_path, "--literal-pathspecs", "ls-tree", "--name-only",
            found.base, "--", *all_to_copy,
        )
        base_tree_paths = set(ls_output.strip().split("\n")) if ls_output.strip() else set()
    except ValueError:
        base_tree_paths = set()

    for tf in all_to_copy:
        dst = base_path / tf
        if tf in base_tree_paths:
            pre_existing.add(tf)
        elif dst.exists():
            # File exists but is not in the base commit (ignored file).
            # Save its bytes so we can write them back after the restore.
            try:
                saved_ignored[tf] = dst.read_bytes()
            except OSError:
                pass

    copied: list[Path] = []
    try:
        for tf in all_to_copy:
            src = attempt_path / tf
            dst = base_path / tf
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied.append(dst)
                if tf not in pre_existing and tf not in saved_ignored:
                    new_copied.append(dst)

        # Run the test files against the base source
        home = Path(ctx.workdir).resolve() / worktree.BASE_CHECK_HOME
        env = worktree.check_environment(base_path, home)
        budget = getattr(ctx, "budget", None)

        if runner == "pytest":
            test_cmd = f"python -m pytest -q {' '.join(new_test_files)}"
        else:
            test_cmd = f"npx vitest run {' '.join(new_test_files)}"

        test_result = sandbox.run_command(
            test_cmd, workdir=base_path, timeout_s=timeout,
            budget=budget, env=env, home=home, exact_env=True,
        )
        test_output = "\n".join(
            (test_result.stdout or "", test_result.stderr or "")
        ).strip()
        base_failures = parse_test_failures(test_output)
        collection_errors, _ = _parse_collection_errors(test_output)
    finally:
        # Restore pre-existing files from the base commit.
        if pre_existing:
            try:
                worktree.git(
                    base_path, "checkout", found.base, "--", *sorted(pre_existing)
                )
            except ValueError as exc:
                target_guard.warn(
                    f"could not restore base worktree files: {exc}"
                )
        # Write back pre-existing ignored files that were overwritten.
        for tf, content in saved_ignored.items():
            dst = base_path / tf
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(content)
        # Remove files that were new (not in the base commit).
        for dst in new_copied:
            dst.unlink(missing_ok=True)

    # Build the guardedness record (correction 4: distinguish import errors)
    guarding = 0
    not_guarding = 0
    cannot_run = 0
    results = []
    for tf in new_test_files:
        tests_in_file = sorted(
            f for f in base_failures
            if f.startswith(tf + "::") or f.startswith(tf + " > ")
        )
        has_collection_error = tf in collection_errors or any(
            e.startswith(tf) or e.endswith("/" + tf)
            for e in collection_errors
        )
        if has_collection_error and not tests_in_file:
            # The test cannot be collected or imported on the base (correction 4):
            # reported as "cannot run on the base", not evidence of guarding.
            cannot_run += 1
            results.append({
                "file": tf, "tests": [], "file_level": "cannot_run_on_base",
                "guarding": False,
            })
        elif tests_in_file:
            # Assertion-level failures: real evidence of guarding
            guarding += len(tests_in_file)
            results.append({
                "file": tf,
                "tests": [
                    {"name": t, "on_base": "fail", "guarding": True}
                    for t in tests_in_file
                ],
            })
        elif not test_result.ok and not has_collection_error:
            # The test file failed but we can't parse which tests
            # Fall back to file-level assessment
            guarding += 1
            results.append({
                "file": tf, "tests": [], "file_level": "fail", "guarding": True,
            })
        else:
            not_guarding += 1
            results.append({
                "file": tf, "tests": [], "file_level": "pass", "guarding": False,
            })

    return {
        "new_test_files": sorted(new_test_files),
        "results": results,
        "summary": {
            "guarding": guarding,
            "not_guarding": not_guarding,
            "cannot_run_on_base": cannot_run,
            "total_new_tests": guarding + not_guarding + cannot_run,
        },
    }


# ── site-revert check ────────────────────────────────────────────────────


def vitest_name_pattern(name: str) -> str:
    """A cited vitest name as `vitest -t` matches it (dm-193).

    A citation names a test as vitest reports it, ``describe > test``, but ``-t`` is a
    regular expression matched against the names joined by a space: ``describe test``. So
    the separator becomes a space, and the name is escaped, so a ``(`` in it is a
    character and not a group vitest refuses at startup.
    """
    return re.sub(r"[.*+?^${}()|[\]\\/]", r"\\\g<0>", name.replace(" > ", " "))


#: Vitest's own summary of what it ran, ``Tests  1 passed | 2 skipped (3)``; pytest's is
#: `_PYTEST_SUMMARY`.
_VITEST_SUMMARY = re.compile(r"^\s*Tests\s+(.+)$", re.MULTILINE)
#: A vitest file that failed to load, so none of its tests ran: ``FAIL  a.test.ts [ a.test.ts ]``.
_VITEST_UNLOADED = re.compile(r"^\s*FAIL\s+(\S+)\s+\[\s*\1\s*\]", re.MULTILINE)
_RAN = re.compile(r"\b\d+ (?:passed|failed|errors?|xfailed|xpassed)\b")


def _nothing_ran(output: str, *, vitest: bool) -> bool:
    """Whether the run passed or failed no test: every selected test skipped, or none collected.

    Vitest exits 0 when ``-t`` selects nothing, which would read as a test that still
    passes with the site reverted; it prints no ``Tests`` line at all when it never got
    to collect (dm-193). A pytest summary of only skipped tests is the same.
    """
    text = strip_ansi(output)
    if vitest:
        summary = _VITEST_SUMMARY.findall(text)
        return not summary or not _RAN.search(summary[-1])
    summary = _PYTEST_SUMMARY.findall(text)
    return bool(summary) and not _RAN.search(summary[-1])


def _is_test_file(path: str) -> bool:
    """Whether *path* is a test file (not a code site for revert purposes)."""
    parts = Path(path).parts
    name = Path(path).name
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py"):
        return True
    if name.endswith(".spec.ts"):
        return True
    if "tests" in parts or "test" in parts:
        return True
    return False


def _extract_defined_names(added_lines: list[str]) -> set[str]:
    """Extract def/class/assignment names from ``+`` lines of a hunk."""
    names: set[str] = set()
    for line in added_lines:
        m = re.match(r'\s*(?:def|class)\s+(\w+)', line)
        if m:
            names.add(m.group(1))
        m = re.match(r'\s*(\w+)\s*=', line)
        if m:
            names.add(m.group(1))
    return names


def _is_import_only_failure(output: str, defined_names: set[str]) -> bool:
    """Whether the failure is only NameError/AttributeError/ImportError on defined names.

    Classifies from pytest's ``E   <ErrorType>: <message>`` lines, not the
    truncated short-summary line ("…has no attrib..."), which leaves only the
    module name quoted.  A failure is import-only when every NameError,
    AttributeError or ImportError line names a definition the reverted hunks
    removed.
    """
    if not defined_names:
        return False
    found_errors = False
    all_names_defined = True
    # Extract from pytest's E-lines: ``E   NameError: name 'foo' is not defined``
    # and from full-output error lines.
    error_types = ("NameError", "AttributeError", "ImportError")
    for line in output.splitlines():
        stripped = line.strip()
        # Match pytest E-lines: "E   ErrorType: message"
        # Also match plain "ErrorType: message" lines for non-pytest output.
        for error_type in error_types:
            if error_type not in stripped:
                continue
            # A pytest E-line, or a plain "ErrorType: message" line. pytest's
            # "ImportError while importing test module '...'" header is neither.
            m = (re.match(rf'E\s+({error_type}):?\s+(.*)', stripped)
                 or re.match(rf'({error_type}):\s+(.*)', stripped))
            if not m:
                continue
            found_errors = True
            snippet = m.group(2)
            quoted = re.findall(r"['\"](\w+)['\"]", snippet)
            if quoted:
                if not any(n in defined_names for n in quoted):
                    all_names_defined = False
            else:
                all_names_defined = False
    return found_errors and all_names_defined


#: Total and per-pair time for the site-revert pass, below the gate's DSH timeout.
SITE_REVERT_TOTAL_S = 240
SITE_REVERT_PAIR_S = 60

#: A pair the pass could not measure: the time cap ran out, the test timed out or
#: was killed, the patch did not apply, or git failed. Never cached, so a later
#: gate measures it; the audit reports it as advisory, never as a failure.
NOT_RUN = "not_run"

#: Languages whose comments can be told from code exactly: Python by its own tokenizer,
#: and JavaScript/TypeScript without JSX by a scan that follows strings, template literals
#: and regular expressions. In any other file no line is taken as comment-only.
_PYTHON_SOURCE = (".py", ".pyi")
_SCRIPT_SOURCE = (".ts", ".mts", ".cts", ".mjs", ".cjs")
_REGEX_AFTER = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_KEYWORD = re.compile(r"(?:^|[^\w$])(?:return|typeof|case|in|of|delete|void|throw|new|yield|await)\s*$")


def _python_code_lines(text: str) -> set[int] | None:
    import io
    import tokenize

    quiet = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
             tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
    code: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in quiet:
                code.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, SyntaxError):
        return None
    return code


def _script_code_lines(text: str) -> set[int] | None:
    """Lines of JS/TS *text* holding anything outside a comment; None when it does not scan."""
    code: set[int] = set()
    line, i, n = 1, 0, len(text)
    state, braces, prev, line_start = "code", [], "", 0
    while i < n:
        c, nxt = text[i], text[i + 1] if i + 1 < n else ""
        if c == "\n":
            line, line_start = line + 1, i + 1
            state = "code" if state in ("line", "sq", "dq") else state
            i += 1
            continue
        if state == "line":
            i += 1
        elif state == "block":
            if c == "*" and nxt == "/":
                state = "code"
                i += 1
            i += 1
        elif state in ("sq", "dq", "tpl"):
            code.add(line)
            if c == "\\":
                if nxt == "\n":
                    line, line_start = line + 1, i + 2
                i += 2
                continue
            if (state, c) in (("sq", "'"), ("dq", '"'), ("tpl", "`")):
                state, prev = "code", c
            elif state == "tpl" and c == "$" and nxt == "{":
                braces.append(0)
                state = "code"
                i += 1
            i += 1
        elif c in " \t\r":
            i += 1
        elif c == "/" and nxt in "/*":
            state = "line" if nxt == "/" else "block"
            i += 2
        else:
            code.add(line)
            if c == "/" and (prev in _REGEX_AFTER or not prev
                             or _REGEX_KEYWORD.search(text[line_start:i])):
                i, in_class = i + 1, False
                while i < n and text[i] != "\n" and (text[i] != "/" or in_class):
                    if text[i] == "\\":
                        i += 1
                    elif text[i] in "[]":
                        in_class = text[i] == "["
                    i += 1
                prev = "/"
                i += 1
                continue
            if c in "'\"`":
                state = {"'": "sq", '"': "dq", "`": "tpl"}[c]
            elif c == "{" and braces:
                braces[-1] += 1
            elif c == "}" and braces:
                if braces[-1] == 0:
                    braces.pop()
                    state = "tpl"
                else:
                    braces[-1] -= 1
            elif c == "\\":
                i += 1
            prev = c
            i += 1
    if state in ("block", "sq", "dq", "tpl") or braces:
        return None
    return code


def _code_lines(path: str, text: str) -> set[int] | None:
    """The 1-based lines of *text* that hold anything but comments, in *path*'s language,
    or None when the language is not one read exactly here."""
    suffix = Path(path).suffix
    if suffix in _PYTHON_SOURCE:
        return _python_code_lines(text)
    if suffix in _SCRIPT_SOURCE:
        return _script_code_lines(text)
    return None


def is_comment_only(path: str, added_lines: list[str], *, text: str | None = None,
                    numbers: list[int] | None = None) -> bool:
    """Whether every added line is blank or wholly a comment in *path*'s language.

    Given the post-image *text* and the added lines' 1-based *numbers*, each line is
    judged where it stands, so a ``#`` line inside a Python string or a ``//`` line in a
    template literal is code; without them the added lines are read on their own. At
    least one line must be a comment. A language not read exactly is never comment-only.
    """
    if text is None:
        text = "\n".join(added_lines) + "\n"
        numbers = list(range(1, len(added_lines) + 1))
    lines = text.splitlines()
    if not numbers or any(n < 1 or n > len(lines) for n in numbers):
        return False
    if [lines[n - 1] for n in numbers] != [line.rstrip("\r\n") for line in added_lines]:
        return False
    code = _code_lines(path, text)
    if code is None or any(n in code for n in numbers):
        return False
    return any(lines[n - 1].strip() for n in numbers)


def _site_range(site: str) -> tuple[str, int, int] | None:
    """``(path, first line, last line)`` of a ``path:line[-line]`` site, or None."""
    parts = site.split(":")
    if len(parts) < 2:
        return None
    spec = parts[1]
    try:
        if "-" in spec:
            start, end = spec.split("-", 1)
            return parts[0], int(start), int(end)
        return parts[0], int(spec), int(spec)
    except ValueError:
        return parts[0], 0, 0


def _overlapping_hunks(attempt_path: Path, base: str, file_path: str,
                       line_start: int, line_end: int) -> tuple[str, list[str]]:
    """The ``-U0`` diff of *file_path* from *base* to HEAD, and the hunk lines whose
    added range overlaps the site. Raises when git fails."""
    import subprocess as _sp

    diff_output = _sp.run(
        ["git", "-C", str(attempt_path), "diff", "-U0", base, "HEAD", "--", file_path],
        capture_output=True, text=True, timeout=15, check=True,
        env={**os.environ, "GIT_LITERAL_PATHSPECS": "1"},
    ).stdout
    overlapping: list[str] = []
    current: list[str] = []
    overlaps = False
    for line in diff_output.splitlines():
        if line.startswith("@@"):
            if current and overlaps:
                overlapping.extend(current)
            current = [line]
            m = re.search(r'\+(\d+)(?:,(\d+))?', line)
            if m:
                hunk_start = int(m.group(1))
                hunk_count = int(m.group(2)) if m.group(2) else 1
                if hunk_count == 0:
                    # Pure deletion hunk: site line is the position or next
                    overlaps = line_start <= hunk_start + 1 and line_end >= hunk_start
                else:
                    overlaps = line_start <= hunk_start + hunk_count - 1 and line_end >= hunk_start
            else:
                overlaps = False
        elif current:
            current.append(line)
    if current and overlaps:
        overlapping.extend(current)
    return diff_output, overlapping


def _added(hunk_lines: list[str]) -> list[str]:
    return [line[1:] for line in hunk_lines if line.startswith("+") and not line.startswith("+++")]


def _added_numbers(hunk_lines: list[str]) -> list[int]:
    """The post-image line numbers of the lines ``-U0`` hunks add."""
    numbers: list[int] = []
    for line in hunk_lines:
        m = re.match(r'@@ -\S+ \+(\d+)(?:,(\d+))? @@', line)
        if m:
            start = int(m.group(1))
            numbers.extend(range(start, start + (int(m.group(2)) if m.group(2) else 1)))
    return numbers


def _hunks_comment_only(attempt_path: Path, file_path: str, hunk_lines: list[str]) -> bool:
    """Whether the lines *hunk_lines* add to HEAD's *file_path* are all comments, in place."""
    import subprocess as _sp

    try:
        text = _sp.run(["git", "-C", str(attempt_path), "show", f"HEAD:{file_path}"],
                       capture_output=True, text=True, timeout=10, check=True).stdout
    except Exception:
        return False
    return is_comment_only(file_path, _added(hunk_lines), text=text,
                           numbers=_added_numbers(hunk_lines))


def site_in_diff(ctx: Any, site: str) -> bool:
    """Whether a ``path:line`` site falls inside a hunk of the run's diff."""
    found = worktree.recorded(ctx.workdir)
    parsed = _site_range(site)
    if found is None or parsed is None:
        return False
    try:
        _, hunks = _overlapping_hunks(Path(found.path), found.base, *parsed)
    except Exception:
        return False
    return bool(hunks)


def site_is_comment_only(ctx: Any, site: str) -> bool:
    """Whether every line a site's hunks add is a comment; False when it is not in the diff."""
    found = worktree.recorded(ctx.workdir)
    parsed = _site_range(site)
    if found is None or parsed is None:
        return False
    file_path, line_start, line_end = parsed
    try:
        _, hunks = _overlapping_hunks(Path(found.path), found.base, file_path, line_start, line_end)
    except Exception:
        return False
    return bool(hunks) and _hunks_comment_only(Path(found.path), file_path, hunks)


def site_outside_change(ctx: Any, site: str) -> str:
    """Why a ``path:line`` site that no hunk of the run's diff adds to is outside the change.

    ``"deleted"`` when the change deletes lines the site names, counted in the base: the
    code it cites is gone. ``"unchanged"`` when the file is in HEAD with the site's lines and
    the change leaves them as the base has them: what a correction that drops a planned
    change leaves, and also what a correction the change never applied leaves. ``""`` when
    it names no code (a line past the end of its file, or a file HEAD does not have) or the
    run has no worktree. The design-mode correction audit reports ``deleted`` as removed
    and ``unchanged`` as measured not met, for the overwatch to decide
    (`authoring_evidence.correction_audit_problems` with ``removed_sites``).
    """
    import subprocess as _sp

    found = worktree.recorded(ctx.workdir)
    parsed = _site_range(site)
    if found is None or parsed is None:
        return ""
    file_path, line_start, line_end = parsed
    if line_start < 1 or line_end < line_start:
        return ""
    root = Path(found.path)
    try:
        diff = _sp.run(
            ["git", "-C", str(root), "diff", "-U0", found.base, "HEAD", "--", file_path],
            capture_output=True, text=True, timeout=15, check=True,
            env={**os.environ, "GIT_LITERAL_PATHSPECS": "1"},
        ).stdout
    except Exception:
        return ""
    for header in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", diff, re.M):
        start = int(header[1])
        count = int(header[2]) if header[2] is not None else 1
        if count and line_start <= start + count - 1 and line_end >= start:
            return "deleted"
    try:
        text = _sp.run(["git", "-C", str(root), "show", f"HEAD:{file_path}"],
                       capture_output=True, text=True, timeout=10, check=True).stdout
    except Exception:
        return ""
    return "unchanged" if line_end <= len(text.splitlines()) else ""


def compute_site_reverts(
    ctx: Any, pairs: list[tuple[str, str, str]],
) -> dict[str, Any]:
    """Revert each site and run its named test to check reachability.

    *pairs* is a list of ``(site_path_line, test_file_and_name, scope)`` tuples.
    Operates on the run's own worktree (the attempt path), applies a reverse
    patch of only the overlapping hunks, runs the named test, and restores
    the file in a ``finally`` block. A site whose added lines are all comments,
    or that sits in a test file, is checked against the diff and not reverted.
    Measured results are cached in a dedicated file; ``not_run`` never is.
    """
    found = worktree.recorded(ctx.workdir)
    if found is None:
        return {"no_base": True}

    attempt_path = Path(found.path)
    base = found.base
    head = found.head

    # Load or initialise cache
    cache_file = ctx.workdir / ".refactory" / "site-revert-cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    except (ValueError, OSError):
        cache = {}

    results: list[dict[str, Any]] = []
    summary = {"caught": 0, "not_caught": 0, "collection_error": 0, NOT_RUN: 0}

    import subprocess as _sp
    import time as _time

    start_time = _time.monotonic()

    def record(entry: dict[str, Any], key: str) -> None:
        results.append(entry)
        if entry["result"] != NOT_RUN:
            cache[key] = entry
        if entry.get("catches_site"):
            summary["caught"] += 1
        elif entry["result"] == "collection_error":
            summary["collection_error"] += 1
        elif entry["result"] == NOT_RUN:
            summary[NOT_RUN] += 1
        else:
            summary["not_caught"] += 1

    def not_run(site: str, test_name: str, detail: str) -> dict[str, Any]:
        return {"site": site, "test": test_name, "result": NOT_RUN,
                "catches_site": False, "detail": detail}

    for site, test_name, scope in pairs:
        cache_key_str = f"{head}|{site}|{test_name}"
        if not isinstance(test_name, str):
            # A citation that names no test: a wrong test id, measured without a run.
            record({"site": site, "test": test_name,
                    "result": "no_tests", "catches_site": False}, cache_key_str)
            continue
        if not test_name.strip():
            # No test was selected for this site (advisory, not a fix-up).
            record({"site": site, "test": test_name,
                    "result": "no_test_selected", "catches_site": False}, cache_key_str)
            continue
        cached = cache.get(cache_key_str)
        # "unavailable" is what earlier versions cached for an unmeasured pair.
        if cached and cached.get("result") not in (NOT_RUN, "unavailable"):
            record(cached, cache_key_str)
            continue

        parsed = _site_range(site)
        if parsed is None:
            record({"site": site, "test": test_name,
                    "result": "not_in_diff", "catches_site": False}, cache_key_str)
            continue
        file_path, line_start, line_end = parsed

        # The file must exist inside the worktree: a path into another checkout is not
        # in this diff (git diff would refuse it, which is not a measurement gap).
        target = (attempt_path / file_path).resolve()
        if not target.is_relative_to(attempt_path.resolve()) or not target.exists():
            record({"site": site, "test": test_name,
                    "result": "not_in_diff", "catches_site": False}, cache_key_str)
            continue

        try:
            diff_output, overlapping_hunks = _overlapping_hunks(
                attempt_path, base, file_path, line_start, line_end)
        except Exception as exc:
            record(not_run(site, test_name, f"git diff failed: {exc}"), cache_key_str)
            continue

        if not overlapping_hunks:
            record({"site": site, "test": test_name,
                    "result": "not_in_diff", "catches_site": False}, cache_key_str)
            continue

        # A site inside a test file is not a code site: it is in the diff, which is
        # all a revert of a test could show.
        if _is_test_file(file_path):
            record({"site": site, "test": test_name,
                    "result": "test_file", "catches_site": False}, cache_key_str)
            continue

        all_added = _added(overlapping_hunks)
        # A comment changes no behaviour, so no test can catch its revert.
        if _hunks_comment_only(attempt_path, file_path, overlapping_hunks):
            record({"site": site, "test": test_name,
                    "result": "comment_only", "catches_site": False}, cache_key_str)
            continue

        # Check if file exists on base (new file)
        try:
            _sp.run(
                ["git", "-C", str(attempt_path), "cat-file", "-e", f"{base}:{file_path}"],
                check=True, capture_output=True, timeout=10,
            )
            is_new_file = False
        except _sp.CalledProcessError:
            is_new_file = True
        except Exception:
            is_new_file = False

        if is_new_file:
            record({"site": site, "test": test_name,
                    "result": "new_file", "catches_site": False}, cache_key_str)
            continue

        # Time cap: it bounds test runs only, so every check above is made for every pair.
        if _time.monotonic() - start_time > SITE_REVERT_TOTAL_S:
            record(not_run(site, test_name, "time cap reached"), cache_key_str)
            continue

        # Build a reverse patch containing only overlapping hunks
        header_lines = []
        for line in diff_output.splitlines():
            if line.startswith("@@"):
                break
            header_lines.append(line)

        reverse_patch = "\n".join(header_lines + overlapping_hunks) + "\n"
        defined_names = _extract_defined_names(all_added)

        # Apply reverse patch and run the test
        try:
            _sp.run(
                ["git", "-C", str(attempt_path), "apply", "--reverse", "--unidiff-zero"],
                input=reverse_patch, capture_output=True, text=True, timeout=10, check=True,
            )
        except Exception as exc:
            record(not_run(site, test_name, f"patch apply failed: {exc}"), cache_key_str)
            continue

        detail = ""
        try:
            # Pick runner and build command. Each argument is quoted: a vitest name
            # such as `suite > test` would otherwise redirect into a stray file in the
            # worktree and select the wrong tests (dm-193).
            test_path = test_name.split("::")[0] if "::" in test_name else test_name
            fn_name = test_name.split("::")[-1] if "::" in test_name else ""

            if test_path.endswith(".py"):
                if fn_name:
                    argv = ["python", "-B", "-m", "pytest", "-x", "-q", test_name]
                else:
                    argv = ["python", "-B", "-m", "pytest", "-x", "-q", test_path]
            elif test_path.endswith(".ts") or test_path.endswith(".spec.ts"):
                if fn_name:
                    argv = ["npx", "vitest", "run", test_path, "-t", vitest_name_pattern(fn_name)]
                else:
                    argv = ["npx", "vitest", "run", test_path]
            else:
                argv = ["python", "-B", "-m", "pytest", "-x", "-q", test_name]
            cmd = shlex.join(argv)

            home = Path(ctx.workdir).resolve() / worktree.CHECK_HOME
            env = worktree.check_environment(attempt_path, home)

            test_result = sandbox.run_command(
                cmd, workdir=attempt_path,
                timeout_s=max(1.0, min(SITE_REVERT_PAIR_S,
                                       SITE_REVERT_TOTAL_S - (_time.monotonic() - start_time))),
                env=env, home=home, exact_env=True,
            )

            test_output = "\n".join(
                (test_result.stdout or "", test_result.stderr or "")
            ).strip()
            rc = test_result.returncode

            # Classify result
            if getattr(test_result, "timed_out", False):
                result, detail = NOT_RUN, "the test timed out"
            elif rc is not None and rc < 0:
                result, detail = NOT_RUN, f"the test was killed by signal {-rc}"
            else:
                coll_files, has_unlocatable = _parse_collection_errors(test_output)
                # A spec the revert stopped from loading is a collection error, not "no tests".
                coll_files |= set(_VITEST_UNLOADED.findall(strip_ansi(test_output)))
                test_path_for_coll = test_name.split("::")[0] if "::" in test_name else test_name
                # pytest names the module by its absolute path, the citation by a
                # path relative to the worktree.
                has_collection = rc != 0 and (any(e == test_path_for_coll
                                      or e.endswith("/" + test_path_for_coll)
                                      or test_path_for_coll.startswith(e) for e in coll_files)
                                  or has_unlocatable)
                # A module that no longer imports after the revert is a collection
                # error (pytest then also reports the test as not found), and an
                # import-only one when it names a definition the hunks removed.
                if has_collection:
                    result = ("import_only" if _is_import_only_failure(test_output, defined_names)
                              else "collection_error")
                elif (rc in (4, 5) or "no tests ran" in test_output.lower()
                      or _nothing_ran(test_output, vitest=argv[:2] == ["npx", "vitest"])):
                    # pytest exit 4 = no tests collected, 5 = no tests ran: the cited
                    # test does not exist, a wrong citation rather than a measurement gap.
                    # So is a run that passed or failed nothing (`_nothing_ran`).
                    result = "no_tests"
                elif rc != 0:
                    # Test failed — check if it's import-only
                    if _is_import_only_failure(test_output, defined_names):
                        result = "import_only"
                    else:
                        result = "fail"  # genuine assertion failure = site is caught
                else:
                    result = "pass"  # test still passes = site NOT caught

        except Exception as exc:
            result, detail = NOT_RUN, f"the test could not be run: {exc}"
        finally:
            # Restore the worktree
            try:
                _sp.run(
                    ["git", "-C", str(attempt_path), "checkout", "HEAD", "--", file_path],
                    capture_output=True, timeout=10,
                    env={**os.environ, "GIT_LITERAL_PATHSPECS": "1"},
                )
            except Exception:
                pass

        entry = {"site": site, "test": test_name,
                 "result": result, "catches_site": result == "fail"}
        if detail:
            entry["detail"] = detail
        record(entry, cache_key_str)

    # Write cache
    try:
        cache_file.write_text(json.dumps(cache, indent=2) + "\n")
    except OSError:
        pass

    return {"pairs": results, "summary": summary}


def compute_co_revert(
    ctx: Any, group: list[tuple[str, str, str]],
) -> dict[str, Any]:
    """Revert all sites in *group* simultaneously and run the tests.

    Unlike :func:`compute_site_reverts` which processes each pair independently,
    this reverts *every* site's hunks at once before running any test.  This
    catches co-dependent fallbacks that survive a solo revert.

    Returns ``{"caught": bool, "detail": str}``.
    """
    found = worktree.recorded(ctx.workdir)
    if found is None:
        return {"caught": False, "detail": "no worktree"}

    import subprocess as _sp
    import time as _time

    attempt_path = Path(found.path)
    base = found.base

    # Collect the unique sites and their reverse patches
    reverted_files: list[str] = []  # for restore
    patches_applied = False

    try:
        for site, _test_name, _scope in group:
            parsed = _site_range(site)
            if parsed is None:
                continue
            file_path, line_start, line_end = parsed
            target = (attempt_path / file_path).resolve()
            if not target.is_relative_to(attempt_path.resolve()) or not target.exists():
                continue
            if _is_test_file(file_path):
                continue
            try:
                diff_output, overlapping_hunks = _overlapping_hunks(
                    attempt_path, base, file_path, line_start, line_end)
            except Exception:
                continue
            if not overlapping_hunks:
                continue
            # Build reverse patch
            header_lines = []
            for line in diff_output.splitlines():
                if line.startswith("@@"):
                    break
                header_lines.append(line)
            reverse_patch = "\n".join(header_lines + overlapping_hunks) + "\n"
            try:
                _sp.run(
                    ["git", "-C", str(attempt_path), "apply", "--reverse", "--unidiff-zero"],
                    input=reverse_patch, capture_output=True, text=True, timeout=10, check=True,
                )
                reverted_files.append(file_path)
                patches_applied = True
            except Exception:
                continue

        if not patches_applied:
            return {"caught": False, "detail": "no patches applied"}

        # Run all unique tests in the group
        tests: set[str] = set()
        for _site, test_name, _scope in group:
            if isinstance(test_name, str) and test_name.strip():
                tests.add(test_name)

        caught = False
        detail_parts: list[str] = []
        home = Path(ctx.workdir).resolve() / worktree.CHECK_HOME
        env = worktree.check_environment(attempt_path, home)

        for test_name in sorted(tests):
            test_path = test_name.split("::")[0] if "::" in test_name else test_name
            fn_name = test_name.split("::")[-1] if "::" in test_name else ""
            if test_path.endswith(".py"):
                if fn_name:
                    argv = ["python", "-B", "-m", "pytest", "-x", "-q", test_name]
                else:
                    argv = ["python", "-B", "-m", "pytest", "-x", "-q", test_path]
            else:
                argv = ["python", "-B", "-m", "pytest", "-x", "-q", test_name]
            cmd = shlex.join(argv)
            try:
                test_result = sandbox.run_command(
                    cmd, workdir=attempt_path,
                    timeout_s=SITE_REVERT_PAIR_S, env=env, home=home, exact_env=True,
                )
                rc = test_result.returncode
                if rc is not None and rc != 0 and rc > 0:
                    caught = True
                    detail_parts.append(f"{test_name}: fail (rc={rc})")
                else:
                    detail_parts.append(f"{test_name}: pass")
            except Exception as exc:
                detail_parts.append(f"{test_name}: error ({exc})")

        return {"caught": caught, "detail": "; ".join(detail_parts)}
    finally:
        # Restore all reverted files
        for file_path in reverted_files:
            try:
                _sp.run(
                    ["git", "-C", str(attempt_path), "checkout", "HEAD", "--", file_path],
                    capture_output=True, timeout=10,
                    env={**os.environ, "GIT_LITERAL_PATHSPECS": "1"},
                )
            except Exception:
                pass


# ── per-step comparison for chain commands ───────────────────────────────


def _attempt_marker_run(ctx: Any, timeout: float
                        ) -> tuple[Callable[[str], Any], list[tuple[str, str]]]:
    """How the attempt's per-step script runs, and that place's own paths (`_signature`).

    It runs where and as the attempt's check beside the reviews runs: the run's check
    copy, with the copy's environment and HOME and the same worktree-root and restore
    handling (`repo_change.run_check` with ``beside_reviews``), so nothing the steps
    write reaches the run's worktree.
    """
    from refactory_dsh import repo_change
    from refactory_dsh.invocations import CHECK_COPY_HOME

    found = worktree.recorded(ctx.workdir)
    if found is None:
        raise ValueError("the run has no worktree")
    run_id = Path(found.path).name.removeprefix("run-")
    copy = repo_change._check_copy_path(Path(found.target), run_id)
    roots = _roots((copy, "<root>"), (Path(found.path), "<root>"),
                   (Path(ctx.workdir).resolve() / CHECK_COPY_HOME, "<home>"), (ctx.workdir, "<run>"))
    return (lambda script: repo_change.run_check(ctx, script, timeout, beside_reviews=True)), roots


def _same_end(attempt_step: dict, base_step: dict) -> bool:
    """Whether a step ended the same way on both sides: it ran on both, with one exit code, or on neither.

    A step that failed without naming a failing test -- a lint, a build -- ends the same
    way only when its output is the same too, its run-specific paths aside (`_signature`),
    and not empty: a new lint or build error behind the base's own is the attempt's.
    """
    if attempt_step["ran"] != base_step["ran"] or attempt_step["exit_code"] != base_step["exit_code"]:
        return False
    if attempt_step["ran"] and attempt_step["exit_code"] != 0 and not attempt_step["failures"]:
        # A failure that prints nothing (``build >/dev/null``, ``grep -q``) cannot be told
        # from a new one, so it never ends the same way.
        if attempt_step.get("quiet", True):
            return False
        return attempt_step.get("signature") == base_step.get("signature") is not None
    return True


def _per_step_comparison(
    ctx: Any, steps: list[tuple[str, str]],
    base_result: "BaseCheckResult", attempt_failures: set[str],
    timeout: float, named_text: str | None,
    changed_files: list[str], brief_text: str, design_text: str,
    *, attempt_signatures: dict[str, str | None] | None = None,
    base_signatures: dict[str, str | None] | None = None,
) -> tuple[ComparisonResult | None, str]:
    """Classify a failed chain's failures step by step: ``(comparison, why)``.

    The attempt's steps run again in one shell (`_run_chain_markers`), and are compared
    with the base's, recorded when the base ran (or read from its cache). A failure in
    step k is excused only when it is inherited by the mode's rule, it failed in step k
    on both sides, every step before k ended the same way on both (`_same_end`), and
    step k's runner counts exactly its named failures on both sides (`_accounted`). A
    step that failed on the attempt without naming a failing test is charged unless it
    ended the same way on the base without naming one either.

    When per-step results cannot be read on either side, the comparison is ``None``
    and ``why`` says why: nothing is then excused, because which step a failure came
    from, and whether the steps before it ran alike, is unknown.
    """
    base_steps = getattr(base_result, '_step_results', None)
    if base_steps is None or len(base_steps) != len(steps):
        return None, "the base's per-step results could not be read"
    base_steps = [{"ran": True, **sr} for sr in base_steps]

    try:
        attempt_steps = _run_chain_markers(steps, *_attempt_marker_run(ctx, timeout))
    except ValueError as exc:
        return None, f"the attempt's steps could not be run again: {exc}"
    if attempt_steps is None:
        return None, "the attempt's per-step results could not be read"

    all_attempt_failures = set(attempt_failures)
    for sr in attempt_steps:
        all_attempt_failures |= set(sr["failures"])
    # Each side's steps carry the tracebacks of failures its whole run never reached.
    attempt_signatures = _with_step_signatures(attempt_signatures, attempt_steps)
    base_signatures = _with_step_signatures(base_signatures, base_steps)
    judge_kwargs: dict[str, Any] = dict(
        named_text=named_text, changed_files=changed_files,
        brief_text=brief_text if named_text is None else "",
        design_text=design_text if named_text is None else "",
        attempt_signatures=attempt_signatures, base_signatures=base_signatures)
    jr = judge_failures(all_attempt_failures, base_result.failures, **judge_kwargs)

    # Targeted confirmation (correction 1) for what the base never failed: a failure only
    # the attempt's per-step re-run found, or one the whole-run comparison excused by its
    # own targeted runs, which this comparison has not repeated. Each runs targeted once
    # on the attempt and once on the base, as `filter_check` confirms the whole run's,
    # and is judged again with what those runs said.
    to_confirm = {failure for failure in jr.charged if jr.reasons.get(failure) == "new-failure"}
    if to_confirm:
        cmd_for_targeted = " && ".join(s for s, _ in steps)
        ids_to_confirm = sorted(to_confirm)
        t_attempt_set = targeted_attempt_failures(
            ctx, cmd_for_targeted, ids_to_confirm, timeout=timeout)
        t_base_result, _ = targeted_base_failures(
            ctx, cmd_for_targeted, ids_to_confirm,
            prepare_timeout=timeout, timeout=timeout,
        )
        t_base_set = (None if t_base_result.environment_failure
                      else set(t_base_result.failures))
        if t_attempt_set is not None or t_base_set is not None:
            jr = judge_failures(all_attempt_failures, base_result.failures,
                                targeted_base_failures_set=t_base_set,
                                targeted_attempt_failures=t_attempt_set, **judge_kwargs)
    reasons = dict(jr.reasons)

    # A failure the base's whole run also had stays excused -- UNLESS the per-step
    # analysis shows the base's step ended differently (e.g. the build failed on base
    # so the test never actually ran there). A flaky test that was in the base's
    # failures but does not appear in the per-step re-run stays excused (the d16 fix).
    # One excused by a targeted run (the base fails it alone, or the attempt does not)
    # was judged on its own, so no step comparison applies to it.
    excused: set[str] = set()
    for failure in jr.excused:
        steps_with = [k for k, sr in enumerate(attempt_steps) if failure in sr["failures"]]
        if (jr.reasons.get(failure) != "base-whole-run" or not steps_with
                or all(_excusable_in_step(k, failure, attempt_steps, base_steps)
                       for k in steps_with)):
            excused.add(failure)
        else:
            reasons[failure] = "step-ended-differently"
    comparison = ComparisonResult(inherited=set(), new_failures=set(jr.charged),
                                  all_excused=False, base_ok=not bool(base_result.failures))

    # A step that failed without naming a failing test -- a build, a grep, a runner that
    # collected nothing or crashed first -- is the attempt's unless the base's same step
    # ended the same way without naming one either.
    unnamed = [
        sr["step"] for sr, base_sr in zip(attempt_steps, base_steps)
        if sr["ran"] and sr["exit_code"] != 0 and not sr["failures"]
        and not (_same_end(sr, base_sr) and not base_sr.get("failures"))
    ]
    new_failures = (all_attempt_failures | comparison.new_failures) - excused
    detail = ""
    shown = [sr for sr in attempt_steps
             if sr["ran"] and sr["exit_code"] != 0
             and (set(sr["failures"]) & new_failures or sr["step"] in unnamed)]
    if shown:
        # Positive selection, as the whole run's detail: a step that named failures
        # shows only its charged ones' lines and blocks and its summary, never an
        # excused failure's; a step that named none shows its output.
        detail = "\n".join(
            f"step `{sr['step']}` exited {sr['exit_code']} when the chain's steps were run "
            f"one by one:\n"
            + (_charged_lines(sr["output"], set(sr["failures"]) & new_failures)
               if sr["failures"] else sr["output"][-4000:])
            for sr in shown)
    for failure in new_failures:
        reasons.setdefault(failure, "new-failure")
    return ComparisonResult(
        inherited=excused,
        new_failures=new_failures,
        all_excused=bool(excused) and not new_failures and not unnamed,
        base_ok=comparison.base_ok,
        reasons=reasons,
    ), detail


def _with_step_signatures(signatures: dict[str, str | None] | None,
                          steps: list[dict]) -> dict[str, str | None]:
    """``signatures`` with each step's failures' signatures added where it has none."""
    merged = dict(signatures or {})
    for sr in steps:
        failures = set(sr.get("failures") or ())
        if failures and sr.get("output"):
            for test_id, sig in extract_failure_signatures(sr["output"], failures).items():
                if merged.get(test_id) is None:
                    merged[test_id] = sig
    return merged


def _excusable_in_step(k: int, failure: str, attempt_steps: list[dict],
                       base_steps: list[dict]) -> bool:
    """Whether ``failure``, failing in the attempt's step ``k``, is excused by the base's step ``k``."""
    attempt_sr, base_sr = attempt_steps[k], base_steps[k]
    if not base_sr["ran"] or failure not in base_sr.get("failures", []):
        return False
    if not all(_same_end(attempt_steps[j], base_steps[j]) for j in range(k)):
        return False
    if attempt_sr["runner"] is None or not attempt_sr["accounted"]:
        return False
    if not base_sr.get("runner") or not base_sr.get("accounted"):
        return False
    try:
        words = shlex.split(attempt_sr["step"])
    except ValueError:
        words = attempt_sr["step"].split()
    return not any(_EXIT_FIRST.match(word) for word in words)


# ── integration function ────────────────────────────────────────────────


def _vitest_blocks(output: str, test_ids: set[str]) -> list[str]:
    """Each charged vitest failure's error block: its ``FAIL`` header and the lines after
    it, up to the next header, separator or summary line."""
    vitest_ids = {tid for tid in test_ids if " > " in tid}
    if not vitest_ids:
        return []
    blocks: list[str] = []
    current: list[str] | None = None
    for line in strip_ansi(output).splitlines():
        stripped = line.strip()
        header = stripped.startswith(("FAIL ", "× "))
        if header or stripped.startswith("⎯") or _VITEST_FINAL.match(stripped):
            if current is not None and len(current) > 1:
                blocks.append("\n".join(current[:60]))
            current = None
            if header:
                name = stripped.split(None, 1)[1].strip() if " " in stripped else ""
                if any(name == tid or name.startswith(tid + " ") for tid in vitest_ids):
                    current = [stripped]
            continue
        if current is not None:
            current.append(line.rstrip())
    if current is not None and len(current) > 1:
        blocks.append("\n".join(current[:60]))
    return blocks


def _extract_failure_blocks(output: str, test_ids: set[str]) -> list[str]:
    """Extract the FAILURES-section block for each test ID (positive selection), and
    each charged vitest failure's error block."""
    clean = strip_ansi(output)
    blocks: list[str] = _vitest_blocks(clean, test_ids)
    section_match = _FAILURES_SECTION.search(clean)
    if section_match is None:
        return blocks
    section_start = section_match.end()
    section_end_match = re.search(r"^={3,}", clean[section_start:], re.MULTILINE)
    section_text = clean[section_start:section_start + section_end_match.start()] if section_end_match else clean[section_start:]

    headers = list(_HEADER_LINE.finditer(section_text))
    for i, hdr in enumerate(headers):
        header_name = hdr.group(1).strip()
        block_start = hdr.start()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(section_text)
        # Check if this header belongs to a charged test ID using strict matching.
        # Use exact segment matching, not endswith, to avoid leaking an excused
        # test's block into charged output (review finding m1-blocks-3).
        for tid in test_ids:
            if (header_name == tid
                    or tid.endswith("::" + header_name)
                    or header_name == tid.split("::")[-1]):
                blocks.append(section_text[block_start:block_end].strip())
                break
        else:
            # Fallback: try matching on any :: segment exactly
            for tid in test_ids:
                parts = tid.split("::")
                if any(header_name == p for p in parts[1:]):
                    blocks.append(section_text[block_start:block_end].strip())
                    break
    return blocks


def _extract_failed_error_lines(output: str, test_ids: set[str]) -> list[str]:
    """Extract the FAILED/ERROR short-summary lines for each charged test ID."""
    clean = strip_ansi(output)
    lines: list[str] = []
    for line in clean.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("FAILED ", "ERROR ", "FAIL ", "× ")):
            continue
        # Match: the line must contain one of the charged test IDs as a whole
        # token (not as a substring of a longer ID).
        for tid in test_ids:
            # The FAILED/ERROR line format is: "FAILED tid" or "FAILED tid - reason"
            # Match if the line contains the full tid followed by end, space, or " -"
            pos = stripped.find(tid)
            if pos >= 0:
                after = pos + len(tid)
                if after >= len(stripped) or stripped[after] in (" ", "\t"):
                    lines.append(stripped)
                    break
    return lines


#: A pytest final summary line (``=== 1 failed, 3 passed in 0.12s ===``, or bare under ``-q``).
_PYTEST_FINAL = re.compile(
    r"^=*\s*\d+ (?:failed|passed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b.*$")
#: vitest's and jest's summary block: ``Test Files  1 failed | 2 passed (3)``, ``Tests  ...``.
_VITEST_FINAL = re.compile(r"^(?:Test Files|Tests|Test Suites)\s+\d+ .*$")


def _extract_summary_lines(output: str) -> list[str]:
    """The runner's final summary lines: pytest's last summary line, and vitest's (or
    jest's) ``Test Files`` / ``Tests`` lines. Nothing else -- when no summary line is
    found the result is empty, never the output it was looking in."""
    summary: list[str] = []
    pytest_last = ""
    for line in strip_ansi(output).splitlines():
        stripped = line.strip()
        if _PYTEST_FINAL.match(stripped) and " in " in stripped:
            pytest_last = stripped
        elif _VITEST_FINAL.match(stripped):
            summary.append(stripped)
    return ([pytest_last] if pytest_last else []) + summary


def _charged_lines(output: str, charged_ids: set[str]) -> str:
    """What ``output`` says of ``charged_ids`` alone: their FAILED/ERROR lines, their
    FAILURES-section blocks, and the runner's summary lines."""
    parts = (_extract_failed_error_lines(output, charged_ids)
             + _extract_failure_blocks(output, charged_ids)
             + _extract_summary_lines(output))
    return "\n".join(parts) if parts else "no charged failure detail found"


def _format_excused_line(excused: set[str], reasons: dict[str, str]) -> str:
    """Format one line listing excused IDs with their reasons."""
    if not excused:
        return ""
    parts = []
    for tid in sorted(excused):
        reason = reasons.get(tid, "base")
        parts.append(f"{tid} ({reason})")
    return "Excused (not charged): " + ", ".join(parts)


def _build_positive_detail(attempt_output: str, charged_ids: set[str],
                           excused_ids: set[str], judge_reasons: dict[str, str]) -> str:
    """Build the send-back detail using positive selection only."""
    if not charged_ids:
        return "no charged failure detail found"

    blocks = _extract_failure_blocks(attempt_output, charged_ids)
    failed_error_lines = _extract_failed_error_lines(attempt_output, charged_ids)
    summary_lines = _extract_summary_lines(attempt_output)
    excused_line = _format_excused_line(excused_ids, judge_reasons)

    parts = failed_error_lines + blocks + summary_lines
    if excused_line:
        parts.append(excused_line)
    return "\n".join(parts) if parts else "no charged failure detail found"


def filter_check(
    ctx: Any, result: Any, command: str, timeout: float,
    guardedness: dict | None = None, *, named_text: str | None = None,
) -> FilteredCheckResult:
    """The integration point: run the base check, compare, reuse guardedness.

    ``guardedness`` is passed in (correction 2: computed once per attempt,
    reused here rather than recomputed). ``named_text`` selects design mode's rule
    (`design_comparison`, the same one its self-check uses); without it, the
    three-condition rule (`compare_failures`) the fast modes use.
    """
    found = worktree.recorded(ctx.workdir)
    if found is None:
        return FilteredCheckResult(
            all_excused=False, inherited=set(), new_failures=set(),
            new_failure_detail="", inherited_detail="",
            base_environment_failure=None, guardedness=guardedness,
        )

    # Parse the attempt's failures early — needed for targeted fallback.
    attempt_output = "\n".join(
        (result.stdout or "", result.stderr or "")
    ).strip()
    attempt_failures = parse_test_failures(attempt_output)

    # Run or load the base check (cached per run)
    base_result = run_base_check(ctx, command, timeout)

    # Base unavailable: fall back to targeted base runs of the attempt's
    # failing IDs (correction 2).  Only if those targeted runs are also
    # unavailable is the check reported as "base unavailable".
    if base_result.environment_failure:
        is_crashed = getattr(base_result, '_crashed', False)
        if attempt_failures:
            # Fall back to targeted base runs.
            targeted_result, _src = targeted_base_failures(
                ctx, command, sorted(attempt_failures),
                prepare_timeout=timeout, timeout=timeout,
            )
            if not targeted_result.environment_failure:
                # Targeted runs succeeded — use them as the base observation.
                base_result = targeted_result
                # Clear the environment_failure so we continue below.
            else:
                # Targeted runs also unavailable — report as "base unavailable"
                # with every failure listed (never excused, never charged as if clean).
                return FilteredCheckResult(
                    all_excused=False, inherited=set(), new_failures=set(),
                    new_failure_detail=attempt_output,
                    inherited_detail=f"base unavailable: {base_result.environment_failure}",
                    base_environment_failure=base_result.environment_failure,
                    guardedness=guardedness,
                    base_failures=None,
                    base_complete=False,
                    base_crashed=is_crashed,
                )
        else:
            return FilteredCheckResult(
                all_excused=False, inherited=set(), new_failures=set(),
                new_failure_detail="", inherited_detail="",
                base_environment_failure=base_result.environment_failure,
                guardedness=guardedness,
                base_failures=None,
                base_complete=False,
                base_crashed=is_crashed,
            )

    # Get changed files and texts for the three-condition rule
    attempt_path = Path(found.path)
    changed_files = _all_changed_files(attempt_path, found.base)
    brief_text = str(ctx.facts.get("brief") or "")
    design_text = ""
    if ctx.exists("design.md"):
        design_text = ctx.read_text("design.md")

    # Extract signatures for touched-file comparison
    attempt_sigs = extract_failure_signatures(attempt_output, attempt_failures) if attempt_failures else None
    base_sigs: dict[str, str | None] = dict(base_result.signatures or {})
    if base_result.failures and base_result.output:
        base_sigs.update(extract_failure_signatures(base_result.output, base_result.failures))

    # First-pass comparison: judge with whole-run base observations only.
    # changed_files is passed in both fast and design mode so the signature
    # check for touched test files (correction 3) fires in every mode.
    judge_kwargs: dict[str, Any] = dict(
        named_text=named_text,
        changed_files=changed_files,
        brief_text=brief_text if named_text is None else "",
        design_text=design_text if named_text is None else "",
        attempt_signatures=attempt_sigs,
        base_signatures=base_sigs,
    )
    first_jr = judge_failures(attempt_failures, base_result.failures, **judge_kwargs)

    # Targeted confirmation (correction 1): run new failures targeted on
    # both the attempt and the base, then re-judge with that information.
    # Even when the base passed, a flaky test that passes on the attempt
    # re-run is excused with "not-reproduced" (brief: "A failure that does
    # not reproduce in a re-run is never charged by default").
    new_from_first = set(first_jr.charged)
    targeted_base_set: set[str] | None = None
    targeted_attempt_set: set[str] | None = None
    if new_from_first:
        ids_to_confirm = sorted(new_from_first)
        try:
            targeted_attempt_set = targeted_attempt_failures(
                ctx, command, ids_to_confirm, timeout=timeout)
        except Exception:
            targeted_attempt_set = None
        try:
            t_base_result, _ = targeted_base_failures(
                ctx, command, ids_to_confirm, prepare_timeout=timeout, timeout=timeout)
            if not t_base_result.environment_failure:
                targeted_base_set = t_base_result.failures
                for test_id, sig in (t_base_result.signatures or {}).items():
                    base_sigs.setdefault(test_id, sig)
        except Exception:
            pass  # targeted base unavailable — fall through without targeted info
        # Re-judge with targeted info (if any was collected).
        if targeted_base_set is not None or targeted_attempt_set is not None:
            jr = judge_failures(
                attempt_failures, base_result.failures,
                targeted_base_failures_set=targeted_base_set,
                targeted_attempt_failures=targeted_attempt_set,
                **judge_kwargs,
            )
        else:
            jr = first_jr
    else:
        jr = first_jr

    comparison = ComparisonResult(
        inherited=set(jr.excused),
        new_failures=set(jr.charged),
        all_excused=jr.all_excused,
        base_ok=not bool(base_result.failures),
    )
    judge_reasons = jr.reasons

    blocker = None
    step_detail = ""
    chain = None
    # A chain whose base failed is compared step by step unless a failure is already
    # the attempt's: the attempt's check may have stopped at a step that fails on the
    # base too (naming no test, or only inherited ones) before reaching what it broke.
    if not base_result.ok and not comparison.new_failures:
        unsplit_reason: list[str] = []
        chain = _chain_steps(command, reason=unsplit_reason)
        if chain is None and unsplit_reason:
            target_guard.warn(f"attempt chain not split: {unsplit_reason[0]}")
    if (chain is not None and len(chain) > 1 and getattr(base_result, '_step_results', None) is None
            and not getattr(base_result, '_steps_unreadable', False)):
        # A base cache written before per-step results existed: the whole-command path.
        chain = None
    if chain is not None and len(chain) > 1:
        # The per-step comparison runs for multi-step chains to detect "earlier
        # step ending differently" scenarios.  Single-step commands (the d16/c3
        # cases) never enter this block (len(chain) <= 1), so the 11–26 min
        # re-run cost identified in the brief is already eliminated.
        revised, why = _per_step_comparison(
            ctx, chain, base_result, attempt_failures, timeout,
            named_text, changed_files, brief_text, design_text,
            attempt_signatures=attempt_sigs, base_signatures=base_sigs,
        )
        if revised is None:
            target_guard.warn(f"no failure of the chained check is excused: {why}")
            blocker = why
        else:
            comparison, step_detail = revised, why
            judge_reasons = {**judge_reasons, **(revised.reasons or {})}
    elif comparison.all_excused:
        blocker = excusal_blocker(command, attempt_output, attempt_failures)

    if blocker:
        comparison = ComparisonResult(inherited=set(), new_failures=set(), all_excused=False,
                                      base_ok=comparison.base_ok)

    # Build detail strings
    inherited_detail = f"no failure was excused: {blocker}" if blocker else ""
    if comparison.inherited:
        inherited_detail = (
            "inherited failures from the base (not the attempt's): "
            + ", ".join(sorted(comparison.inherited))
        )

    # Positive selection: the builder sees only charged failures (correction 4).
    new_failure_detail = _build_positive_detail(
        attempt_output, comparison.new_failures, comparison.inherited, judge_reasons,
    )
    if step_detail:
        new_failure_detail = f"{new_failure_detail}\n\n{step_detail}"

    # Populate base_failures and base_complete for R5 (broken guarantees).
    base_complete = getattr(base_result, '_base_complete', False)

    return FilteredCheckResult(
        all_excused=comparison.all_excused,
        inherited=comparison.inherited,
        new_failures=comparison.new_failures,
        new_failure_detail=new_failure_detail,
        inherited_detail=inherited_detail,
        base_environment_failure=None,
        guardedness=guardedness,
        base_failures=base_result.failures,
        base_complete=base_complete,
        reasons=judge_reasons,
    )
