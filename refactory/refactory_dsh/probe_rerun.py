"""Rerun the design verifier's probes with deterministic code, and check its PROBED claims.

The verifier saves every probe as its own file under ``reviews/probes/``
(``<requirement-id>-<n>.py`` or ``.sh``) and names that file on each PROBED line of
``reviews/verification.md`` (`verification._HUNK_REVERT_METHOD`). A replay of the verifier
on a known-buggy commit showed why the file matters: where the bugs were, it wrote
"PROBED via <the builder's test>" and so inherited the builder's wrong assumptions. Here
"probed" is checked mechanically:

- each probe file runs against the attempt's commit in a scratch checkout -- never the
  builder's workspace or the committed tree -- with `PROBE_TIMEOUT_S` per probe and
  `PROBES_TOTAL_S` for all of them; the probes past the total are recorded as not run
  (advisory);
- ``reviews/probe-results.md`` (and ``.json``) records each probe's exit code, pass/fail
  and the last lines of its output;
- a PROBED line naming no probe file, a missing one, or one that only invokes a test the
  repository had before the change is UNVERIFIED (`design_ops._check_unverified_coverage`
  gives it the coverage rule's one QA-director re-run, then lists it as advisory);
- each probe runs with the check's interpreter and the checkout's sources importable
  (`probe_environment`); one that stops before reaching the change because of that
  environment -- the project's own package not importable, a missing interpreter or Node
  package -- is "could not run (environment)", advisory (`environment_failure`);
- any other probe that fails is a defect finding (kind defect, severity high, breaks its
  requirement) that `design_ops.gate_qa` sends back;
- the same probes run on the base at the same time (`rerun_probes`): a requirement none of
  whose passing probes fails there -- they pass with the change reverted, declare only
  preserved behaviour, or could not run on the base -- is an advisory finding for the
  person (`base_revert_findings`), never a re-run or a reloop.

Design mode runs this as the ``rerun_probes`` op before ``synthesize_qa``, which both
runtimes run through the same callable; ``refactory replay-verifier`` runs `run_probes`
after the replayed verifier.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

#: Where the verifier writes its probes, in its workspace and (copied) in the run directory.
PROBES = "reviews/probes"
RESULTS_MD = "reviews/probe-results.md"
RESULTS_JSON = "reviews/probe-results.json"
#: Each probe's deadline, and the deadline for all of them together.
PROBE_TIMEOUT_S = 120
PROBES_TOTAL_S = 600
#: How long preparing the scratch checkout's environment may take per command.
PREPARE_TIMEOUT_S = 600
#: How many trailing output lines the table keeps.
TAIL_LINES = 5

PROBE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.(?:py|sh)$")
_PROBE_REF = re.compile(r"reviews/probes/([A-Za-z0-9_.-]+\.(?:py|sh))\b")
_REQUIREMENT = re.compile(r"^#{2,6}\s*Requirement:\s*(.+?)\s*$", re.I)
#: A PROBED claim: the uppercase word, not "READ, not probed".
_PROBED = re.compile(r"\bPROBED\b")
#: A test file and optional test id a probe hands to pytest, or a test module it imports.
_TEST_ID = re.compile(r"([\w./-]*test[\w./-]*\.py)(?:::(\w+))?")
_FROM_IMPORT = re.compile(r"^[ \t]*from[ \t]+([\w.]+)[ \t]+import[ \t]+(?:\(([\w\s,]+)\)|([\w \t,]+))", re.M)
#: A check of the probe's own: an assertion or an expected raise. Not an exit status:
#: `sys.exit(subprocess.run([... "pytest", "<existing test>"]).returncode)` is the obvious
#: way to write "just run the repository's test", which is the one thing a probe may not be.
_OWN_CHECK = {".py": re.compile(r"\bassert\b|pytest\.raises|\braise\b"),
              ".sh": re.compile(r"\[\s|\btest\s|python3?\s+(?:-c|-\s*<<)|<<")}


def requirement_of(name: str) -> str:
    """The requirement id a probe file is named for: ``R1-2.py`` → ``R1``."""
    stem = Path(name).stem
    return re.sub(r"-\d+$", "", stem)


def header_of(text: str) -> str:
    """The probe's one-line header comment (requirement, failure point, expected outcome)."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#!"):
            continue
        return stripped.lstrip("#").strip() if stripped.startswith("#") else ""
    return ""


def probed_claims(report: str) -> list[dict[str, str]]:
    """Every PROBED line of the verifier's report: its requirement and the probe file it names."""
    claims: list[dict[str, str]] = []
    requirement = ""
    for line in report.splitlines():
        heading = _REQUIREMENT.match(line.strip())
        if heading:
            requirement = heading[1]
            continue
        if not _PROBED.search(line):
            continue
        named = _PROBE_REF.search(line)
        claims.append({"requirement": requirement or (requirement_of(named[1]) if named else "?"),
                       "file": named[1] if named else "", "line": line.strip()[:300]})
    return claims


def _base_files(repo: Path, base: str) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", base],
                             capture_output=True, text=True, timeout=60, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return out.splitlines()


def _base_text(repo: Path, base: str, name: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "show", f"{base}:{name}"],
                              capture_output=True, text=True, timeout=60, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def _existing(files: list[str], relative: str) -> str | None:
    """The base's file that ``relative`` names, from the repository root or any subdirectory."""
    relative = relative.lstrip("./")
    for name in files:
        if name == relative or name.endswith("/" + relative):
            return name
    return None


def existing_tests_invoked(text: str, repo: Path, base: str,
                           files: list[str] | None = None) -> list[str]:
    """The tests the repository had at ``base`` that the probe runs or imports."""
    files = _base_files(repo, base) if files is None else files
    found: list[str] = []
    for match in _TEST_ID.finditer(text):
        name = _existing(files, match[1])
        if name is None:
            continue
        if match[2] is None or re.search(rf"^\s*(?:async\s+)?def\s+{match[2]}\b",
                                         _base_text(repo, base, name), re.M):
            found.append(name + (f"::{match[2]}" if match[2] else ""))
    for match in _FROM_IMPORT.finditer(text):
        names = match[2] if match[2] is not None else match[3]
        tests = [part.strip() for part in names.split(",") if part.strip().startswith("test")]
        name = _existing(files, match[1].replace(".", "/") + ".py") if tests else None
        if name is None:
            continue
        source = _base_text(repo, base, name)
        found += [f"{name}::{test}" for test in tests
                  if re.search(rf"^\s*(?:async\s+)?def\s+{test}\b", source, re.M)]
    return sorted(set(found))


#: A run that checked nothing: nothing collected, or everything skipped or deselected.
_NOTHING_RAN = re.compile(r"no tests ran|collected 0 items|no tests collected", re.I)
_ALL_SKIPPED = re.compile(r"\b\d+ (?:skipped|deselected)\b", re.I)
#: Any outcome that means a test actually executed and reported.
_SOMETHING_RAN = re.compile(r"\b\d+ (?:passed|failed|error|errors|xpassed|xfailed)\b", re.I)


def has_own_check(name: str, text: str) -> bool:
    """Whether a probe file contains a check of its own at all.

    A probe is model-authored, and `print("ok")` or `def test_nothing(): pass` exits 0.
    Without a floor, "every probe passed" says nothing about the requirements, so a
    probe with no assertion is reported unverified rather than passing.
    """
    pattern = _OWN_CHECK.get(Path(name).suffix)
    if pattern is None:
        return True
    body = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
    return bool(pattern.search(body))


def ran_nothing(output: str) -> bool:
    """Whether a probe's own output says it checked nothing.

    Either it collected nothing, or it reported only skips and deselections with no
    test actually executing. A run with one skip beside one pass has still checked
    something, so only the absence of any executed outcome counts.
    """
    if _SOMETHING_RAN.search(output):
        return False
    return bool(_NOTHING_RAN.search(output) or _ALL_SKIPPED.search(output))


def only_existing_tests(name: str, text: str, repo: Path, base: str,
                        files: list[str] | None = None) -> list[str]:
    """The pre-existing tests a probe merely invokes, or ``[]`` when it has a check of its own."""
    invoked = existing_tests_invoked(text, repo, base, files)
    if not invoked:
        return []
    body = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
    own = _OWN_CHECK[Path(name).suffix].search(body)
    return [] if own else invoked


#: Where a project keeps its virtual environment's interpreter, from the project's root.
_VENV_PYTHONS = (".venv/bin/python", "venv/bin/python", ".venv/Scripts/python.exe")
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
#: Directories a search for the checkout's Python projects does not enter.
_SKIP_DIRS = {"node_modules", "__pycache__", "dist", "build", "vendor", "site-packages"}


def _python_steps(check: str) -> list[list[str]]:
    """The check command's steps that run Python (pytest or an interpreter), as words."""
    import shlex

    from refactory_dsh.base_check import _steps

    found = []
    for step in _steps(check or ""):
        try:
            words = shlex.split(step)
        except ValueError:
            continue
        if any("python" in word or "pytest" in word for word in words):
            found.append(words)
    return found


def _split_assignments(words: list[str]) -> tuple[dict[str, str], list[str]]:
    """A step's leading ``NAME=value`` words (after an ``env``), and the command after them."""
    assigned: dict[str, str] = {}
    rest = list(words)
    if rest[:1] == ["env"]:
        rest = rest[1:]
    while rest and _ASSIGNMENT.match(rest[0]):
        name, value = _ASSIGNMENT.match(rest[0]).groups()  # type: ignore[union-attr]
        assigned[name] = value
        rest = rest[1:]
    return assigned, rest


def check_interpreter(check: str, roots: list[Path]) -> str | None:
    """The interpreter the check runs Python with: the one its command names by path (such
    as ``.venv/bin/python -m pytest``), else the project's virtual environment, looked up in
    each of ``roots`` in turn (the scratch checkout first). ``None`` when there is none."""
    named = []
    for words in _python_steps(check):
        _, command = _split_assignments(words)
        if command and "python" in Path(command[0]).name and "/" in command[0]:
            named.append(command[0])
    for root in roots:
        for relative in (*named, *_VENV_PYTHONS):
            candidate = Path(relative) if Path(relative).is_absolute() else Path(root) / relative
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _project_roots(checkout: Path, depth: int = 3) -> list[Path]:
    """The checkout's Python projects' import roots: each directory with a ``pyproject.toml``,
    ``setup.py`` or ``setup.cfg`` (its ``src/`` in a src layout), nearest the root first."""
    found: list[Path] = []
    level = [checkout]
    for _ in range(depth):
        following: list[Path] = []
        for directory in level:
            if any((directory / name).is_file() for name in ("pyproject.toml", "setup.py", "setup.cfg")):
                found.append(directory / "src" if (directory / "src").is_dir() else directory)
            try:
                children = sorted(directory.iterdir())
            except OSError:
                continue
            following += [child for child in children if child.is_dir() and not child.is_symlink()
                          and not child.name.startswith(".") and child.name not in _SKIP_DIRS
                          and not (child / "pyvenv.cfg").exists()]
        level = following
    return found


def _test_roots(check: str, checkout: Path) -> list[Path]:
    """The directories pytest puts on the import path for the check's test paths: each named
    path's directory, up to the first that is not a package (pytest's ``prepend`` mode)."""
    found: list[Path] = []
    for words in _python_steps(check):
        _, command = _split_assignments(words)
        for word in command[1:]:
            if word.startswith("-") or "::" in word:
                continue
            path = checkout / word
            if not (path.is_dir() or (path.is_file() and path.suffix == ".py")):
                continue
            directory = path if path.is_dir() else path.parent
            while (directory / "__init__.py").is_file() and directory != checkout:
                directory = directory.parent
            found.append(directory)
    return found


def probe_environment(checkout: Path, env: dict[str, str], check: str = "",
                      roots: list[Path] | None = None) -> tuple[dict[str, str], str]:
    """The environment and interpreter a probe runs with: the check's.

    The interpreter is the check's (`check_interpreter`), its directory first on PATH so a
    ``.sh`` probe's ``python`` is the same one. The scratch checkout's sources come first on
    the import path, ahead of any installed copy: the ``PYTHONPATH`` the check's command
    sets (relative to the checkout), each Python project's import root, and the check's
    test directories. An inherited ``PYTHONPATH`` is dropped, as the check's environment
    never has one (`worktree.check_environment`): it would import another checkout's code.
    """
    checkout = Path(checkout)
    env = dict(env)
    python = check_interpreter(check, [checkout, *(roots or [])])
    if python is not None:
        bin_dir = str(Path(python).parent)
        env["PATH"] = os.pathsep.join([bin_dir, *(entry for entry in env.get("PATH", "").split(os.pathsep)
                                                  if entry and entry != bin_dir)])
        if (Path(python).parent.parent / "pyvenv.cfg").is_file():
            env["VIRTUAL_ENV"] = str(Path(python).parent.parent)
    else:
        python = (shutil.which("python", path=env.get("PATH")) or shutil.which("python3", path=env.get("PATH"))
                  or sys.executable)
    paths: list[Path] = []
    for words in _python_steps(check):
        assigned, _ = _split_assignments(words)
        for name, value in assigned.items():
            if name == "PYTHONPATH":
                paths += [entry if Path(entry).is_absolute() else checkout / entry
                          for entry in value.split(os.pathsep) if entry and "$" not in entry]
            elif "$" not in value:
                env[name] = value
    paths += [*_project_roots(checkout), *_test_roots(check, checkout), checkout]
    ordered = list(dict.fromkeys(str(Path(path)) for path in paths))
    env["PYTHONPATH"] = os.pathsep.join(ordered)
    return env, python


def _command(probe: Path, text: str, python: str) -> list[str]:
    if probe.suffix == ".sh":
        return ["bash", str(probe)]
    if re.search(r"^\s*(?:async\s+)?def\s+test_", text, re.M):
        return [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(probe)]
    return [python, str(probe)]


_NO_MODULE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")
_UNKNOWN_LOCATION = re.compile(r"ImportError: cannot import name '\w+' from '([\w.]+)' \(unknown location\)")
_NOT_FOUND = re.compile(r"(?:^|\s)([\w.+-]+): (?:command )?not found\s*$", re.M)
_NODE_MISSING = re.compile(r"Cannot find (?:module|package) '([^']+)'")
#: pytest's own exit codes for a run that never got to test anything: internal error (3),
#: usage error (4), nothing collected (5). Not 2: pytest exits 2 on an error during
#: collection, which is what a probe importing a function the change never added, or a
#: changed module that no longer loads, produces -- the probe's verdict on the change.
_PYTEST_NOT_RUN = {3, 4, 5}
#: A test function pytest could not call, because a parameter is not a fixture.
_MISSING_FIXTURE = re.compile(r"^E\s+fixture '[^']+' not found", re.M)


def _not_a_test_run(command: list[str], code: int | None, output: str) -> bool:
    """Whether a failing pytest-run probe failed only because pytest could not run it."""
    if "pytest" not in command or code is None:
        return False
    if code in _PYTEST_NOT_RUN:
        return True
    return code == 1 and bool(_MISSING_FIXTURE.search(output)) and not re.search(
        r"\b\d+ passed\b|^FAILED ", output, re.M)


#: Test runners a probe is run with: missing from the interpreter, the probe never started.
_RUNNERS = {"pytest", "_pytest", "pluggy"}


def _own_package(name: str, roots: list[Path]) -> bool:
    return any((root / name / "__init__.py").is_file() or (root / f"{name}.py").is_file()
               or (root / name).is_dir() and any((root / name).glob("*.py")) for root in roots)


def environment_failure(code: int | None, output: str, checkout: Path, env: dict[str, str]) -> str:
    """Why a failing probe could not run in this environment, or ``""`` when it ran and failed.

    A probe could not run when it stops before reaching the change: the project's own
    package (one the checkout has) or the test runner is not importable, its interpreter or
    a command it calls is missing, or a Node package it imports is not installed. Anything
    else -- an assertion, an error inside the code, a module the change was to add -- is
    the probe's verdict on the change.
    """
    if code is None or code == 0:
        return ""
    roots = [Path(entry) for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry] or [checkout]
    for match in _NO_MODULE.finditer(output):
        name = match[1]
        if name in _RUNNERS or ("." not in name and _own_package(name, roots)):
            return f"the interpreter cannot import {name!r} (No module named {name!r})"
    for match in _UNKNOWN_LOCATION.finditer(output):
        if "." not in match[1] and _own_package(match[1], roots):
            return f"the interpreter cannot import {match[1]!r} (unknown location)"
    if code == 127:
        # Only when the output actually names what was not found. A bare 127 is not
        # evidence of a missing command: a probe whose last line is `exit 127`, or
        # whose own shell exits 127 for another reason, would otherwise mark itself
        # "environment" and stop being a defect -- a blocking check that any probe
        # can opt out of.
        missing = _NOT_FOUND.search(output)
        if missing:
            return f"{missing[1]} is not installed (exit 127)"
    for match in _NODE_MISSING.finditer(output):
        name = match[1]
        if not name.startswith((".", "/")) and not (checkout / "node_modules" / name).exists():
            return f"the Node package {name!r} is not installed in the checkout"
    return ""


def _run_one(command: list[str], cwd: Path, env: dict[str, str], timeout: float) -> tuple[int | None, str]:
    """Run one probe; ``(exit code or None on timeout, output)``.

    A probe is a file the verifier agent wrote, so it runs the way every other command
    we did not write runs (`sandbox.run_command`, as `base_check` runs reverted code):
    CPU and file-size limits, the whole process group stopped at the deadline, no
    stdin, and the probe's own environment and HOME. That is resource and lifetime
    bounding, not filesystem confinement -- a probe can still write wherever the run
    can, as the agent that wrote it can (no agent's file tools are confined). A probe
    that leaves its checkout says so in its output.
    """
    from refactory_dsh import sandbox

    result = sandbox.run_command(
        shlex.join(command), workdir=cwd, timeout_s=max(1, math.ceil(timeout)), env=env,
        exact_env=True, home=Path(env["HOME"]) if env.get("HOME") else None, worktree_root=cwd,
    )
    output = "\n".join(part for part in (result.stdout or "", result.stderr or "") if part)
    if result.cwd_outside:
        output += f"\nthe probe left its checkout for {result.cwd_outside}"
    return (None if result.timed_out else result.returncode), output


def _tail(output: str) -> str:
    return "\n".join(output.rstrip().splitlines()[-TAIL_LINES:])


def reset_checkout(repo: Path) -> None:
    """Put a probe checkout back to its commit, ignored files included.

    `clones._clean` keeps ignored files on purpose (build caches), but between probes
    that lets one probe change the next one's result -- a file under a gitignored
    directory such as ``reviews/`` survives, and probes run in an order the verifier
    chooses by naming them. So ignored files go too; only the installed environments
    and refactory's own directory (`workspace.EXCLUDED_DIRS`, `workspace.OWN`) stay,
    because reinstalling them per probe would cost far more than a probe. A probe that
    writes into one of those (a ``pip install`` into ``.venv``) still affects the next.
    """
    from refactory_dsh import worktree
    from refactory_dsh.git_hooks import NO_HOOKS
    from refactory_dsh.workspace import EXCLUDED_DIRS, OWN

    worktree.git(repo, *NO_HOOKS, "reset", "-q", "--hard", "HEAD")
    worktree.git(repo, "clean", "-fdqx", *(f"--exclude={name}" for name in (*EXCLUDED_DIRS, f"/{OWN}")))


def run_probes(probes: Path, checkout: Path, report: str, *, base: str, env: dict[str, str] | None = None,
               reset: Callable[[Path], None] | None = None, per_probe_s: float = PROBE_TIMEOUT_S,
               total_s: float = PROBES_TOTAL_S, check: str = "",
               roots: list[Path] | None = None) -> dict[str, Any]:
    """Run each probe file in ``probes`` in the scratch ``checkout``; check ``report``'s claims.

    ``checkout`` is a clone at the attempt's commit, made for this; ``reset`` puts it back
    to that commit before each probe. ``base`` is the commit the change started from, which
    decides what a pre-existing test is. Each probe runs with the check's interpreter and
    the checkout's sources importable (`probe_environment`; ``check`` is the run's check
    command, ``roots`` other directories to find its interpreter in). A probe that could
    not run in that environment (`environment_failure`) is advisory, not a defect. Returns
    the results `write_results` records.
    """
    env, python = probe_environment(Path(checkout), dict(os.environ if env is None else env),
                                    check, roots)
    files = _base_files(checkout, base)
    names = sorted(path.name for path in probes.iterdir()
                   if path.is_file() and PROBE_NAME.match(path.name)) if probes.is_dir() else []
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for name in names:
        text = (probes / name).read_text(errors="replace")
        entry: dict[str, Any] = {"file": name, "requirement": requirement_of(name),
                                 "header": header_of(text), "exit_code": None, "tail": "",
                                 "existing_tests": only_existing_tests(name, text, checkout, base, files)}
        left = total_s - (time.monotonic() - started)
        if left <= 0:
            entry["status"] = "unrun"
            entry["tail"] = f"not run: the {int(total_s)} s total for all probes was spent"
            results.append(entry)
            continue
        if reset is not None:
            reset(checkout)
        # The probe runs from inside the checkout, at the path the verifier gave it.
        target = checkout / PROBES / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(probes / name, target)
        code, output = _run_one(_command(target, text, python), checkout, env, min(per_probe_s, left))
        entry["exit_code"] = code
        entry["tail"] = _tail(output)
        entry["status"] = ("timed_out" if code is None else "pass" if code == 0 else "fail")
        why = environment_failure(code, output, Path(checkout), env)
        if why:
            entry["status"], entry["environment"] = "environment", why
        elif entry["status"] == "fail" and _not_a_test_run(_command(target, text, python), code, output):
            # pytest ran the file but found nothing it could run as a test (a script
            # with a helper or a method called test_*): the probe checked nothing, which
            # is unverified, not a defect in the change.
            entry["status"], entry["no_check"] = "no_check", "pytest could not run it as a test"
        elif entry["status"] == "pass":
            # A pass only counts when the probe actually checked something.
            if not has_own_check(name, text):
                entry["status"], entry["no_check"] = "no_check", "the probe makes no check of its own"
            elif ran_nothing(output):
                entry["status"], entry["no_check"] = "no_check", "the probe's run checked nothing"
        results.append(entry)
    if reset is not None:
        reset(checkout)
    by_file = {entry["file"]: entry for entry in results}
    unverified: list[dict[str, str]] = []
    for claim in probed_claims(report):
        entry = by_file.get(claim["file"]) if claim["file"] else None
        if not claim["file"]:
            why = "its PROBED line names no probe file under reviews/probes/"
        elif entry is None:
            why = f"its probe file reviews/probes/{claim['file']} does not exist"
        elif entry["existing_tests"]:
            why = (f"its probe file {claim['file']} only invokes tests the repository already had: "
                   + ", ".join(entry["existing_tests"]))
        elif entry.get("status") == "no_check":
            why = f"its probe file {claim['file']} passed without checking anything: " + str(
                entry.get("no_check") or "")
        else:
            continue
        unverified.append({"id": claim["requirement"], "requirement": f"{claim['line']} ({why})",
                           "why": why})
    return {"probes": results, "unverified": unverified, "per_probe_s": per_probe_s,
            "total_s": total_s}


def defects(results: dict[str, Any]) -> list[dict[str, str]]:
    """Each failing probe as a must-fix item the QA gate sends back (`judges.blocks`)."""
    items = []
    for entry in results.get("probes", []):
        if entry.get("status") not in ("fail", "timed_out") or entry.get("existing_tests"):
            continue
        how = (f"timed out after {int(results.get('per_probe_s', PROBE_TIMEOUT_S))} s"
               if entry["status"] == "timed_out" else f"exits {entry['exit_code']}")
        last = entry.get("tail", "").splitlines()[-1:] or [""]
        items.append({"source": "probe rerun", "where": f"{PROBES}/{entry['file']}",
                      "what": f"the verifier's probe for {entry['requirement']} {how} on this "
                              f"attempt: {last[0]}".rstrip(": "),
                      "why": entry.get("header") or "a probe of the requirement fails",
                      "kind": "defect", "severity": "high", "breaks": "ask"})
    return items


def render(results: dict[str, Any]) -> str:
    """``reviews/probe-results.md``: the table, then the unverified claims and the failures."""
    lines = ["# Probe results", "",
             f"Each probe file the verifier wrote, rerun on this attempt's commit in a scratch "
             f"checkout ({int(results.get('per_probe_s', PROBE_TIMEOUT_S))} s per probe, "
             f"{int(results.get('total_s', PROBES_TOTAL_S))} s for all).", "",
             "| file | requirement | exit code | result | last lines of output | discriminating |",
             "| --- | --- | --- | --- | --- | --- |"]
    for entry in results.get("probes", []):
        result = {"pass": "pass", "fail": "FAIL", "timed_out": "FAIL (timed out)",
                  "unrun": "not run (advisory)", "no_check": "no check of its own: UNVERIFIED",
                  "environment": "could not run (environment)"}[entry["status"]]
        if entry.get("existing_tests"):
            result += " — only invokes existing tests: UNVERIFIED"
        tail = entry.get("tail", "").replace("|", "\\|").replace("\n", "<br>")
        code = "" if entry.get("exit_code") is None else str(entry["exit_code"])
        disc = entry.get("discriminating")
        disc_col = ("✓" if disc is True else "✗" if disc is False
                    else "non-regression" if disc == "non_regression"
                    else "base unavailable" if disc == "base_unavailable" else "—")
        lines.append(f"| {entry['file']} | {entry['requirement']} | {code} | {result} | {tail} | {disc_col} |")
    if not results.get("probes"):
        lines.append("| (no probe files) | | | | | |")
    base_wall = results.get("base_rerun_wall_s")
    if base_wall is not None:
        lines += ["", f"Base rerun wall-clock: {base_wall} s"]
    base_revert = base_revert_findings(results)
    if base_revert:
        lines += ["", "## Not verified against the base (advisory)", "",
                  "Each probe also ran with the change reverted. A requirement none of whose "
                  "probes failed there is not shown to work by them; it is listed for the "
                  "person, not sent back or re-run.", "",
                  *(f"- {finding}" for finding in base_revert)]
    if results.get("unverified"):
        lines += ["", "## UNVERIFIED claims", ""]
        lines += [f"- UNVERIFIED {item['id']}: {item['requirement']}"
                  for item in results["unverified"]]
    found = defects(results)
    if found:
        lines += ["", "## Failing probes (defects, blocking)", ""]
        lines += [f"- {item['where']}: {item['what']} ({item['why']})" for item in found]
    blocked = [entry for entry in results.get("probes", []) if entry["status"] == "environment"]
    if blocked:
        lines += ["", "## Could not run (environment, advisory)", "",
                  "These probes stopped before reaching the change, so they say nothing about it; "
                  "they are listed for the person, not sent back.", "",
                  *(f"- {entry['file']}: {entry.get('environment', '')}" for entry in blocked)]
    unrun = [entry["file"] for entry in results.get("probes", []) if entry["status"] == "unrun"]
    if unrun:
        lines += ["", "## Not run (advisory)", "", *(f"- {name}" for name in unrun)]
    return "\n".join(lines) + "\n"


def write_results(out: Path, results: dict[str, Any]) -> str:
    """Write ``reviews/probe-results.md`` and ``.json`` under ``out``; return the markdown."""
    text = render(results)
    (out / RESULTS_MD).parent.mkdir(parents=True, exist_ok=True)
    (out / RESULTS_MD).write_text(text)
    (out / RESULTS_JSON).write_text(json.dumps(results, indent=2, sort_keys=True))
    return text


def read_results(ctx: Any) -> dict[str, Any]:
    """This round's probe results, or ``{}`` when none were recorded."""
    if not ctx.exists(RESULTS_JSON):
        return {}
    try:
        found = json.loads(ctx.read_text(RESULTS_JSON))
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


#: A probe header that declares it checks preserved behaviour: it passes on the base too
#: by design, so it never shows on its own that the change works (`_base_revert_discriminate`).
_NON_REGRESSION = re.compile(r"non-regression|preserved behaviou?r", re.I)
#: Where the base side's own checkout and HOME live under the run's workspaces.
BASE_ROOT = "probe_rerun_base"
#: The fallback base checkout, recorded once it is prepared, so later passes reuse it.
BASE_READY = "ready.json"


def _base_unavailable(base_entry: dict[str, Any] | None, note: str) -> str:
    """Why a probe's base-side result says nothing about discrimination, or ``""``."""
    if base_entry is None:
        return note or "the base run has no result for it"
    if base_entry.get("status") == "environment":
        return f"could not run on the base (environment): {base_entry.get('environment', '')}".rstrip(": ")
    if base_entry.get("status") == "unrun":
        return "not run on the base: the total time for all probes was spent"
    return ""


def _base_revert_discriminate(results: dict[str, Any], base_results: dict[str, Any],
                              base_wall_s: float) -> None:
    """Annotate the head-side results with base-revert discrimination, per requirement.

    Each probe that passed on the head is compared with its run on the base: one that
    fails or times out there is ``discriminating=True``; one that passes there too is
    ``False``, or ``"non_regression"`` when its header declares ``non-regression`` or
    ``preserved behaviour``; one the base could not run (an environment failure, not
    run, or no base at all) is ``"base_unavailable"`` -- it says nothing either way.

    A requirement is verified when at least one of its passing probes discriminates; a
    helper probe beside it that does not is shown in the table only. A requirement with
    passing probes and none that discriminates is listed in ``results["base_revert"]``
    as an advisory finding (`base_revert_findings`): never ``unverified``, which the
    coverage rule re-runs, and never a defect.
    """
    base_by_file = {entry["file"]: entry for entry in base_results.get("probes", [])}
    note = str(base_results.get("note") or "")
    by_requirement: dict[str, list[dict[str, Any]]] = {}
    for entry in results.get("probes", []):
        if entry.get("status") != "pass":
            continue
        base_entry = base_by_file.get(entry["file"])
        why = _base_unavailable(base_entry, note)
        if why:
            entry["discriminating"], entry["base_unavailable"] = "base_unavailable", why
        else:
            assert base_entry is not None
            entry["base_exit_code"] = base_entry.get("exit_code")
            if entry["base_exit_code"] != 0:
                # Failed, crashed or timed out (None) on the base: it tells the two apart.
                entry["discriminating"] = True
            elif _NON_REGRESSION.search(entry.get("header") or ""):
                entry["discriminating"] = "non_regression"
            else:
                entry["discriminating"] = False
        by_requirement.setdefault(entry["requirement"], []).append(entry)
    findings: list[dict[str, Any]] = []
    for requirement, entries in by_requirement.items():
        values = [entry["discriminating"] for entry in entries]
        if any(value is True for value in values):
            continue
        files = [entry["file"] for entry in entries]
        named = ", ".join(files)
        if "base_unavailable" in values:
            reasons = "; ".join(dict.fromkeys(entry["base_unavailable"] for entry in entries
                                              if entry.get("base_unavailable")))
            kind, text = "base_unavailable", (
                f"base unavailable: {requirement} could not be checked against the base "
                f"({named}: {reasons})")
        elif "non_regression" in values:
            kind, text = "non_regression_only", (
                f"verified by non-regression probes only: {requirement} ({named} declare "
                "preserved behaviour and pass on the base too; no probe shows the change)")
        else:
            kind, text = "not_discriminating", (
                f"unverified: probes pass on the base too: {requirement} ({named} pass with "
                "the change reverted, so they do not show the change works)")
        findings.append({"id": requirement, "kind": kind, "probes": files, "finding": text})
    results["base_revert"] = findings
    results["base_run"] = base_results
    results["base_rerun_wall_s"] = base_wall_s


def base_revert_findings(results: dict[str, Any]) -> list[str]:
    """The advisory findings for requirements no probe verified against the base."""
    return [str(item["finding"]) for item in results.get("base_revert", [])
            if isinstance(item, dict) and item.get("finding")]


def _overlay(changes: dict[str, Any]) -> dict[str, str]:
    """This process's environment with ``changes`` applied (``None`` unsets a name)."""
    env = dict(os.environ)
    for key, value in changes.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def _shared_base(found: Any) -> Path | None:
    """The base check's worktree (``base-<run id>``) when it is checked out at the run's
    base, else ``None``: a worktree at any other commit is not reused."""
    from refactory_dsh import base_check

    run_id = Path(found.path).name.removeprefix("run-")
    path = base_check._base_worktree_path(Path(found.target), run_id)
    return path if base_check._at(path, found.base) else None


def _base_clone(workdir: Path, found: Any, facts: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    """The probes' own base checkout when the base check left none: prepared once per run.

    `clones._prepare` puts a base clone back to the base on every call, so it is called
    only when no ready checkout is recorded (`BASE_READY`) for this run's target and base,
    or the recorded one is no longer at the base. A preparation that reported problems is
    not recorded, so the next pass prepares again. Its environment carries the clone's own
    HOME (``.refactory/home`` inside it), never this process's.
    """
    from refactory_dsh import base_check, clones
    from refactory_dsh.workspace import OWN, ROOT

    root = Path(workdir) / ROOT / BASE_ROOT
    ready = root / BASE_READY
    try:
        kept = json.loads(ready.read_text())
    except (OSError, ValueError):
        kept = {}
    if (isinstance(kept, dict) and kept.get("base") == found.base and kept.get("target") == found.target
            and base_check._at(Path(str(kept.get("path"))), found.base)):
        path, changes = Path(kept["path"]), dict(kept.get("env") or {})
    else:
        ready.unlink(missing_ok=True)
        prepared = clones._prepare(workdir, root, "base", facts=facts, command_timeout_s=PREPARE_TIMEOUT_S)
        path, changes = Path(prepared["path"]), dict(prepared.get("env") or {})
        if not prepared.get("problems"):
            ready.parent.mkdir(parents=True, exist_ok=True)
            ready.write_text(json.dumps({"base": found.base, "target": found.target,
                                         "path": str(path), "env": changes}))
    home = path / OWN / "home"
    home.mkdir(parents=True, exist_ok=True)
    return path, {**_overlay(changes), "HOME": str(home)}


def _run_on_shared_base(probes: Path, base_wt: Path, env: dict[str, str], *, base: str,
                        check: str, roots: list[Path]) -> dict[str, Any]:
    """Run the probes in the base check's own worktree without disturbing it.

    No reset and no clean: the worktree holds what the base check built and installed
    (``lib/``, generated files, a refreshed lockfile), which `base_check.targeted_base_failures`
    and later base checks reuse. The probe copies are removed afterwards (a file the base
    already had at a probe's path is put back), and Python writes no bytecode there.
    """
    names = sorted(path.name for path in probes.iterdir()
                   if path.is_file() and PROBE_NAME.match(path.name)) if probes.is_dir() else []
    # The directories the copies need that the base does not have (``reviews/``, then
    # ``reviews/probes/``), outermost first: removed afterwards with what is in them.
    made = [directory for directory in reversed((base_wt / PROBES).parents)
            if base_wt in directory.parents and not directory.exists()]
    if not (base_wt / PROBES).exists():
        made.append(base_wt / PROBES)
    saved = {name: (base_wt / PROBES / name).read_bytes() for name in names
             if (base_wt / PROBES / name).is_file()}
    try:
        return run_probes(probes, base_wt, "", base=base, env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
                          reset=None, check=check, roots=roots, total_s=PROBES_TOTAL_S)
    finally:
        for name in names:
            copy = base_wt / PROBES / name
            if name in saved:
                copy.write_bytes(saved[name])
            else:
                copy.unlink(missing_ok=True)
        if made:
            shutil.rmtree(made[0], ignore_errors=True)


def rerun_probes(ctx: Any) -> dict[str, Any]:
    """Op before ``synthesize_qa``: rerun the verifier's probe files on the attempt and the base.

    The probes come from the verifier's workspace clone (``reviews/probes/`` there; a
    directory is not a declared write, because an agent node fails on a declared output it
    leaves unchanged) and are copied to the run directory, replacing the last round's. They
    run in this op's own clone at the attempt (`clones._prepare`), with that clone's
    prepared environment, reset before each probe.

    The same probes run on the base at the same time, within the same total
    (``PROBES_TOTAL_S``): in the base check's worktree when it is at the base, left as it
    is (`_run_on_shared_base`); otherwise in a base clone prepared once per run, beside the
    attempt's preparation (`_base_clone`). The base side runs with its own HOME. A probe
    that passes on both does not discriminate; a requirement none of whose probes does is
    an advisory finding (`_base_revert_discriminate`), never a re-run or a reloop.
    """
    import concurrent.futures

    from refactory_dsh import clones, worktree
    from refactory_dsh.workspace import OWN, ROOT, _slug

    workdir = Path(ctx.workdir)
    probes = workdir / PROBES
    shutil.rmtree(probes, ignore_errors=True)
    source = workdir / ROOT / _slug("verifier") / "repo" / PROBES
    if source.is_dir():
        probes.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and not path.is_symlink() and PROBE_NAME.match(path.name):
                shutil.copyfile(path, probes / path.name)
    report = ctx.read_text("reviews/verification.md") if ctx.exists("reviews/verification.md") else ""
    found = worktree.recorded(workdir)
    if found is None:
        results: dict[str, Any] = {"probes": [], "unverified": [], "per_probe_s": PROBE_TIMEOUT_S,
                                   "total_s": PROBES_TOTAL_S,
                                   "note": "the run has no worktree, so there is no attempt to rerun probes on"}
        text = write_results(workdir, results)
        return {"probe_results": text}
    facts = dict(ctx.facts)
    check = str(ctx.facts.get("check") or "")
    roots = [Path(found.path)]
    shared = _shared_base(found)
    base_wall: list[float] = []

    def timed(run: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
        began = time.monotonic()
        try:
            return run(*args, **kwargs)
        finally:
            base_wall.append(time.monotonic() - began)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        head_prepare = pool.submit(clones._prepare, workdir, workdir / ROOT / "probe_rerun", "attempt",
                                   facts=facts, command_timeout_s=PREPARE_TIMEOUT_S)
        base_prepare = (None if shared is not None
                        else pool.submit(_base_clone, workdir, found, facts))
        base_note = ""
        base_side = None
        if shared is not None:
            # Its own HOME, beside the base side's clone rather than inside the shared
            # worktree, which the base check would otherwise find untracked files in.
            home = workdir / ROOT / BASE_ROOT / OWN / "home"
            home.mkdir(parents=True, exist_ok=True)
            base_env = {**os.environ, **worktree.check_environment(shared, home)}
            base_side = pool.submit(timed, _run_on_shared_base, probes, shared, base_env,
                                    base=found.base, check=check, roots=roots)
        prepared = head_prepare.result()
        head = pool.submit(run_probes, probes, Path(prepared["path"]), report, base=found.base,
                           env=_overlay(prepared.get("env") or {}), reset=reset_checkout,
                           check=check, roots=roots, total_s=PROBES_TOTAL_S)
        if shared is None:
            assert base_prepare is not None
            try:
                base_path, base_env = base_prepare.result()
            except (ValueError, OSError) as exc:
                base_note = f"the base could not be prepared: {exc}"
            else:
                base_side = pool.submit(timed, run_probes, probes, base_path, "", base=found.base,
                                        env=base_env, reset=clones._clean, check=check,
                                        roots=roots, total_s=PROBES_TOTAL_S)
        results = head.result()
        base_results = base_side.result() if base_side is not None else {"probes": [], "note": base_note}
    base_results["checkout"] = "the base check's worktree" if shared is not None else "the probes' base clone"
    _base_revert_discriminate(results, base_results, round(sum(base_wall), 2))
    text = write_results(workdir, results)
    return {"probe_results": text}
