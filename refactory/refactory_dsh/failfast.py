"""Fail-fast targeted pre-pass, stuck detection, and convergence history helpers.

Shared by ``check_artifact`` (design mode) and ``focused_check`` (fast mode)
so that the logic is not duplicated across two entry points. Design mode's
``prepare_check`` runs the same pre-pass as its self-check (b41a): given a
selection, it runs those tests in chunks within a time limit.
"""

from __future__ import annotations

import math
import re
import os
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: How many chunks the self-check cuts its selection into. Each chunk is its own
#: targeted command, so a limit that runs out stops between chunks and every
#: finished chunk reports; more chunks cut closer to the limit, but each pays the
#: runner's start-up again.
SELF_CHECK_CHUNKS = 6

#: The most test files a chunk holds, so a chunk the limit stops loses little.
SELF_CHECK_CHUNK_FILES = 4

#: pytest's summary line for a test module that does not import, ``ERROR tests/test_x.py``:
#: `base_check.parse_test_failures` reads only error lines that give a reason.
_COLLECTION_ERROR = re.compile(r"^ERROR\s+([\w/.\[\]-]+\.py)\s*(?:-\s.*)?$", re.MULTILINE)
#: pytest stops before running anything once a module does not import.
_INTERRUPTED = re.compile(r"Interrupted: \d+ errors? during collection")
#: pytest's "not found" line for an id that does not exist on disk.
_NOT_FOUND = re.compile(r"ERROR:\s+not found:\s+\S+?(tests/[\w/.\-]+(?:::[\w\[\]\-]+)+)", re.MULTILINE)


@dataclass
class PrepassResult:
    """Outcome of a targeted pre-pass run."""

    ran: bool  # whether a targeted command was executed
    early_stop: bool  # whether the pre-pass stopped early
    targeted_command: str | None  # the targeted command string, or None
    check_mode: str  # "targeted" if early_stop, else "full"
    result: Any | None  # SandboxResult from the targeted run, or None
    digest_text: str  # formatted digest, or ""
    runners: list[dict] = field(default_factory=list)  # parsed runner dicts
    raw_tail: str = ""  # raw_tail from the targeted digest
    # ── the self-check's account (b41a); empty for the dm-128 pre-pass ──
    selection: list[str] = field(default_factory=list)  # what it set out to run, in order
    ran_ids: list[str] = field(default_factory=list)  # in chunks that finished
    cut_ids: list[str] = field(default_factory=list)  # not started, or killed by the limit
    failing_ids: list[str] = field(default_factory=list)  # failed, and not inherited
    commands: list[str] = field(default_factory=list)  # one per chunk that ran
    #: Chunks whose targeted command collected no test at all (pytest exit 4 or 5): the
    #: self-check proves nothing about them, so the record says so rather than a pass.
    ran_nothing: list[str] = field(default_factory=list)
    #: Chunks whose command pytest could not parse at all (exit 4 with no not-found id):
    #: a broken command is a harness fault, and the self-check stops instead of pretending.
    broken_command: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # why a chunk did not count
    limit_s: float = 0.0
    elapsed_s: float = 0.0
    outputs: list[str] = field(default_factory=list)  # each failing chunk's output
    left_out: list[str] = field(default_factory=list)  # failures not shown: inherited
    timings: list[dict] = field(default_factory=list)  # each chunk run: its tests, seconds


def _skip() -> PrepassResult:
    return PrepassResult(
        ran=False, early_stop=False, targeted_command=None,
        check_mode="full", result=None, digest_text="", runners=[], raw_tail="",
    )


def self_check_limit(self_check_timeout_s: float, check_timeout_s: float) -> float:
    """The self-check's time limit: its own knob, and never more than a third of the check's."""
    return float(min(self_check_timeout_s, check_timeout_s // 3))


def _runner_of(name: str) -> str:
    from refactory_dsh.check_digest import _SCRIPT_TEST_FILE

    if " > " in name or _SCRIPT_TEST_FILE.search(name.split("::")[0]):
        return "vitest"
    return "pytest" if "::" in name or name.endswith(".py") else "other"


def _chunks(selection: list[str], alone: list[str] = ()) -> list[list[str]]:
    """The selection in order, cut into about `SELF_CHECK_CHUNKS` chunks of one runner each,
    none of more than `SELF_CHECK_CHUNK_FILES` tests.

    The targeted command keeps the check's ``&&`` between steps, so a chunk that held
    two runners' tests would not run the second runner's once the first failed. A file
    of ``alone`` (`test_selection.run_alone`) shares its chunk with no other file.
    """
    from refactory_dsh.test_selection import _file_of

    size = min(SELF_CHECK_CHUNK_FILES, max(1, math.ceil(len(selection) / SELF_CHECK_CHUNKS)))
    single = set(alone)

    def own(name: str) -> str | None:
        """``name``'s file when that file runs alone, else None: a chunk holds one value."""
        return _file_of(name) if _file_of(name) in single else None

    chunks: list[list[str]] = []
    for name in selection:
        if (chunks and len(chunks[-1]) < size and _runner_of(chunks[-1][-1]) == _runner_of(name)
                and own(chunks[-1][-1]) == own(name)):
            chunks[-1].append(name)
        else:
            chunks.append([name])
    return chunks


def _not_found_ids(output: str) -> set[str]:
    """The test ids pytest reports as ``ERROR: not found:`` — ids that do not exist."""
    return set(_NOT_FOUND.findall(output))


def _is_dropped(name: str, dropped: set[str]) -> bool:
    """Whether *name* matches any id in *dropped* by path suffix or bracket prefix.

    pytest prints absolute paths and strips ``[...]`` from parametrized ids,
    so ``tests/test_x.py::test_y`` must match ``refactory/tests/test_x.py::test_y``
    and ``tests/test_x.py::test_p`` must cover ``refactory/tests/test_x.py::test_p[2.5]``.
    """
    for d in dropped:
        if name == d or name.endswith("/" + d):
            return True
        # A dropped id without brackets covers that test's parametrized ids.
        if "[" not in d:
            base = name.split("[")[0]
            if base == d or base.endswith("/" + d):
                return True
    return False


def _is_single_pytest_command(command: str) -> bool:
    """Whether *command* is a single pytest invocation (no ``&&``)."""
    return "pytest" in command and "&&" not in command


def _parse_junitxml(path: str) -> list[dict[str, Any]] | None:
    """Parse a pytest --junitxml file into per-test durations.

    Returns a list of ``{"test_id": str, "seconds": float}`` or ``None``
    when the file is missing or malformed.
    """
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError):
        return None
    durations: list[dict[str, Any]] = []
    for case in tree.getroot().iter("testcase"):
        name = case.get("name", "")
        seconds = float(case.get("time", 0))
        fpath = _junit_file(case.get("classname", ""))
        test_id = f"{fpath}::{name}" if name else fpath
        durations.append({"test_id": test_id, "seconds": seconds})
    return durations if durations else None


def _junit_file(classname: str) -> str:
    """The test file a junit ``classname`` (``pkg.tests.test_x``, or ``pkg.tests.test_x.Class``
    for a test in a class) stands for: the dotted path without its trailing class names, so
    a capitalised package directory stays (``src/MyPkg/tests/test_x.py``, not ``src.py``)."""
    parts = classname.split(".")
    while len(parts) > 1 and parts[-1][:1].isupper():
        parts.pop()
    return "/".join(parts) + ".py"


def _output(result: Any) -> str:
    return "\n".join(filter(None, [
        getattr(result, "stdout", None) or "",
        getattr(result, "stderr", None) or "",
    ])).strip()


def self_check_failures(output: str) -> set[str]:
    """The failures a run names: `base_check.parse_test_failures`'s, and each module that
    did not import (its file stands for its failure)."""
    from refactory_dsh.base_check import parse_test_failures, strip_ansi

    return parse_test_failures(output) | set(_COLLECTION_ERROR.findall(strip_ansi(output)))


def _chunk_report(result: Any, inherited: set[str]) -> tuple[set[str], str]:
    """The chunk's failures that are not inherited, or why the chunk says nothing.

    The same rule as the pre-pass's early stop: a preparation failure, a failure that
    is not the code's (a timeout, a signal), and pytest's "nothing collected" exits are
    not a verdict on the tests, and neither is a failure that names no test.
    """
    from refactory_dsh import authoring_evidence as evidence

    if getattr(result, "ok", False):
        return set(), ""
    if getattr(result, "prep_failure", None) is not None:
        return set(), "its environment could not be prepared"
    if evidence.check_failure_kind(result) != "code_failure":
        return set(), "it did not fail on the code (a timeout or a signal)"
    rc = getattr(result, "returncode", None)
    if rc in (4, 5):
        return set(), f"pytest collected no test (exit {rc})"
    found = self_check_failures(_output(result))
    if not found:
        return set(), f"it failed (exit {rc}) without naming a failing test"
    return found - inherited, ""


def _self_check(
    selection: list[str],
    prev_record: dict[str, Any],
    check_command: str,
    run_check_fn: Callable[[str, float], Any],
    limit_s: float,
    root: str | None,
    alone: list[str] = (),
) -> PrepassResult:
    """Run ``selection`` in chunks until the limit is spent; report what failed.

    The cut is never made by killing the runner, because a killed runner reports
    no failure: no new chunk starts once the limit is spent, and a finished chunk's
    failures count even when a later chunk is killed. A chunk that reports nothing
    (it names no failure, or its environment failed) ran nothing that counts, so its
    tests are recorded as not run. A module that does not import is that file's
    failure; pytest then stops before the chunk's other tests, so they run again
    without it. Tests ``build_probe_command`` cannot target are left out and named.
    """
    from refactory_dsh.check_digest import build_probe_command

    inherited = set(prev_record.get("inherited", []))
    root_path = Path(root) if root else None
    started = time.monotonic()
    ran: list[str] = []
    cut: list[str] = []
    failing: list[str] = []
    commands: list[str] = []
    ran_nothing: list[str] = []
    broken_command: list[str] = []
    notes: list[str] = []
    failed_outputs: list[str] = []
    timings: list[dict[str, Any]] = []
    last = None
    queue = _chunks(selection, alone)
    number = 0
    while queue:
        chunk = queue.pop(0)
        number += 1
        of = number + len(queue)
        remaining = limit_s - (time.monotonic() - started)
        if remaining < 1:
            for later in (chunk, *queue):
                cut.extend(later)
            notes.append(f"the {limit_s:.0f} s limit was spent before chunk {number} "
                          f"of {of}; it and the rest did not run")
            break
        ids = [name for name in chunk if "::" in name or " > " in name]
        files = [name for name in chunk if name not in ids]
        command, refusal = build_probe_command(check_command, ids, root=root_path, test_files=files)
        if refusal:
            cut.extend(chunk)
            notes.append(f"chunk {number} could not be targeted: {refusal}")
            continue
        # Optionally collect per-test durations via --junitxml (single pytest
        # invocation only, written outside the worktree).
        junit_path: str | None = None
        actual_command = command
        if _is_single_pytest_command(command):
            fd, junit_path = tempfile.mkstemp(suffix=".xml", prefix=f"refactory-chunk-{number}-")
            os.close(fd)
            actual_command = f"{command} --junitxml={junit_path}"
        began = time.monotonic()
        result = run_check_fn(actual_command, remaining)
        last = result
        chunk_durations: list[dict[str, Any]] | None = None
        if junit_path is not None:
            try:
                chunk_durations = _parse_junitxml(junit_path)
            finally:
                try:
                    os.unlink(junit_path)
                except OSError:
                    pass
        if getattr(result, "timed_out", False):
            elapsed = round(time.monotonic() - began, 1)
            entry_t: dict[str, Any] = {"tests": chunk, "s": elapsed,
                                       "timed_out": True, "counted": True}
            if chunk_durations:
                entry_t["durations"] = chunk_durations
            timings.append(entry_t)
            for later in (chunk, *queue):
                cut.extend(later)
            notes.append(f"chunk {number} of {of} was stopped at the "
                         f"{limit_s:.0f} s limit, so it and the rest did not report")
            break
        commands.append(command)
        found, why_not = _chunk_report(result, inherited)
        entry_f: dict[str, Any] = {"tests": chunk, "s": round(time.monotonic() - began, 1),
                                   "counted": not why_not}
        if chunk_durations:
            entry_f["durations"] = chunk_durations
        timings.append(entry_f)
        if why_not:
            dropped: list[str] = []
            if "exit 4" in why_not:
                dropped = _not_found_ids(_output(result))
                if dropped:
                    rest_of_chunk = [name for name in chunk
                                     if not _is_dropped(name, dropped)]
                    remaining_for_rerun = limit_s - (time.monotonic() - started)
                    if remaining_for_rerun < 1:
                        timings.append({"tests": rest_of_chunk or chunk,
                                        "s": round(time.monotonic() - began, 1),
                                        "timed_out": True, "counted": True})
                        cut.extend(chunk)
                        notes.append(f"chunk {number}: dropped not-found ids "
                                     f"{sorted(dropped)}, but no time left for a rerun")
                        notes.append(f"chunk {number} did not count: {why_not}")
                        if "collected no test" in why_not:
                            ran_nothing.append(command)
                        continue
                    if rest_of_chunk:
                        ids2 = [name for name in rest_of_chunk if "::" in name or " > " in name]
                        files2 = [name for name in rest_of_chunk if name not in ids2]
                        cmd2, ref2 = build_probe_command(check_command, ids2,
                                                         root=root_path, test_files=files2)
                        if not ref2:
                            result2 = run_check_fn(cmd2, remaining_for_rerun)
                            found2, why2 = _chunk_report(result2, inherited)
                            timings.append({"tests": rest_of_chunk,
                                            "s": round(time.monotonic() - began, 1),
                                            "counted": not why2})
                            commands.append(cmd2)
                            if not why2:
                                for test_id in sorted(found2):
                                    if test_id not in failing:
                                        failing.append(test_id)
                                if found2:
                                    failed_outputs.append(_output(result2))
                                # Check for _INTERRUPTED in the rerun (module did not import).
                                if _INTERRUPTED.search(_output(result2)):
                                    named2 = self_check_failures(_output(result2))
                                    broken2 = [name for name in rest_of_chunk
                                               if name in named2]
                                    rest2 = [name for name in rest_of_chunk
                                             if name not in broken2]
                                    ran.extend(broken2)
                                    if broken2 and rest2:
                                        queue.insert(0, rest2)
                                        notes.append(
                                            f"chunk {number}: dropped not-found ids "
                                            f"{sorted(dropped)}, reran; "
                                            f"{', '.join(broken2)} did not import")
                                        continue
                                ran.extend(rest_of_chunk)
                                notes.append(f"chunk {number}: dropped not-found ids "
                                             f"{sorted(dropped)}, reran the rest")
                                continue
                            # Rerun also failed: note and fall through to cut.
                            notes.append(f"chunk {number}: dropped not-found ids "
                                         f"{sorted(dropped)}, reran but the rest also failed")
                            cut.extend(chunk)
                            notes.append(f"chunk {number} did not count: {why_not}")
                            if why2 and "collected no test" in why2:
                                ran_nothing.append(cmd2)
                            continue
                    # Nothing left after dropping not-found ids, or couldn't target.
                    notes.append(f"chunk {number}: dropped not-found ids "
                                 f"{sorted(dropped)}, nothing left after dropping")
                    ran_nothing.append(command)
            cut.extend(chunk)
            notes.append(f"chunk {number} did not count: {why_not}")
            if "collected no test" in why_not:
                # Exit 4 with no not-found id is a command pytest cannot parse: a harness
                # fault, not a selection that matched nothing (exit 5, or ids a rename
                # removed, are recorded and go on to the whole check).
                if "exit 4" in why_not and not dropped:
                    broken_command.append(command)
                elif not dropped:
                    ran_nothing.append(command)
            continue
        for test_id in sorted(found):
            if test_id not in failing:
                failing.append(test_id)
        if found:
            failed_outputs.append(_output(result))
        if _INTERRUPTED.search(_output(result)):
            named = self_check_failures(_output(result))
            broken = [name for name in chunk if name in named]
            rest = [name for name in chunk if name not in broken]
            ran.extend(broken)
            if broken and rest:
                queue.insert(0, rest)
                notes.append(f"chunk {number}: {', '.join(broken)} did not import, so pytest "
                             "stopped before the chunk's other tests; they run again without it")
            elif rest:
                cut.extend(rest)
                notes.append(f"chunk {number}: a module did not import, so pytest stopped "
                             "before the chunk's tests")
            continue
        ran.extend(chunk)
    digest_text, runners, raw_tail = failure_digest(failed_outputs, inherited)
    return PrepassResult(
        ran=bool(commands), early_stop=bool(failing),
        targeted_command="\n".join(commands) or None,
        check_mode="targeted" if failing else "full", result=last,
        digest_text=digest_text, runners=runners, raw_tail=raw_tail,
        selection=list(selection), ran_ids=ran, cut_ids=cut, failing_ids=failing,
        commands=commands, notes=notes, limit_s=limit_s, ran_nothing=ran_nothing,
        broken_command=broken_command,
        elapsed_s=round(time.monotonic() - started, 1),
        outputs=failed_outputs, left_out=sorted(inherited), timings=timings,
    )


def without_failures(prepass: PrepassResult, test_ids: list[str]) -> PrepassResult:
    """The self-check's result with ``test_ids`` gone from its failures and its digest."""
    from dataclasses import replace

    leave_out = {*prepass.left_out, *test_ids}
    digest_text, runners, raw_tail = failure_digest(prepass.outputs, leave_out)
    failing = [test_id for test_id in prepass.failing_ids if test_id not in leave_out]
    return replace(prepass, failing_ids=failing, early_stop=bool(failing),
                   check_mode="targeted" if failing else "full",
                   digest_text=digest_text if failing else "", runners=runners if failing else [],
                   raw_tail=raw_tail if failing else "", left_out=sorted(leave_out))


def failure_digest(outputs: list[str], leave_out: set[str]) -> tuple[str, list[dict], str]:
    """The digest of the failing chunks' outputs, without the failures in ``leave_out``.

    Each output is digested on its own, as one runner's report, and the runners merged:
    the builder is shown the failures it is sent back for, and no inherited one.
    """
    from refactory_dsh.check_digest import CheckDigest, RunnerDigest, digest_check_output, format_digest

    if not outputs:
        return "", [], ""
    merged: dict[str, RunnerDigest] = {}
    tail = ""
    for output in outputs:
        digest = digest_check_output(output)
        tail = digest.raw_tail or tail
        for runner in digest.runners:
            kept = [entry for entry in runner.failures if entry.test_id not in leave_out]
            if not kept:
                continue
            into = merged.setdefault(runner.name, RunnerDigest(name=runner.name, passed=False))
            into.failures.extend(entry for entry in kept
                                 if entry.test_id not in {f.test_id for f in into.failures})
    combined = CheckDigest(runners=list(merged.values()), raw_tail=tail)
    return format_digest(combined), combined.to_dict()["runners"], tail[-500:]


def targeted_prepass(
    prev_record: dict[str, Any],
    check_command: str,
    run_check_fn: Callable[[str, float], Any],
    check_timeout_s: float,
    *,
    root: str | None = None,
    selection: list[str] | None = None,
    self_check_timeout_s: float | None = None,
    alone: list[str] = (),
) -> PrepassResult:
    """Run previously-failing tests first and stop early while one still fails.

    Parameters
    ----------
    prev_record:
        The previous ``check-record.json`` contents (read before running).
    check_command:
        The full check command string.
    run_check_fn:
        ``(command, timeout) -> SandboxResult`` callback.
    check_timeout_s:
        The full check's timeout.  The full check still gets this value
        unmodified.
    root:
        Working directory root for ``build_probe_command``'s package.json
        lookup (correction item 4).
    selection:
        The self-check's tests, in the order they run (`test_selection`): whole test
        files and test ids, the last check's failures among them. Given, the pre-pass
        runs on every attempt, round 1 included, in chunks within
        ``self_check_limit(self_check_timeout_s, check_timeout_s)``, and reports every
        failure it found that is not inherited. Not given (``check_artifact`` and
        ``focused_check``), it is the dm-128 pre-pass, unchanged.
    self_check_timeout_s:
        The self-check's own time limit (its knob), used with ``selection``.
    alone:
        The selection's files that run one per chunk (`test_selection.run_alone`).
    """
    if selection is not None:
        if not selection:
            return _skip()
        limit = self_check_limit(300 if self_check_timeout_s is None else self_check_timeout_s,
                                 check_timeout_s)
        result = _self_check(selection, prev_record, check_command, run_check_fn, limit, root,
                             alone)
        if result.broken_command:
            # pytest could not parse the targeted command, so no test could run. The command
            # is the harness's, and a self-check that cannot run is a fault, not a verdict.
            raise RuntimeError(
                "the targeted check ran no tests: "
                + result.broken_command[0]
                + " (pytest could not parse it, exit 4). The check command's options were not "
                  "carried into the targeted one, so this self-check could not run at all.")
        return result

    # 1. Skip when previous record is not a failure.
    if prev_record.get("status") != "fail":
        return _skip()

    # 2. Extract failing test IDs from runners.
    failing_ids: list[str] = []
    for runner in prev_record.get("runners", []):
        for f in runner.get("failures", []):
            tid = f.get("test_id")
            if tid:
                failing_ids.append(tid)

    if not failing_ids:
        return _skip()

    # 3. Read inherited set from the previous record's "inherited" field.
    inherited_set = set(prev_record.get("inherited", []))

    # 4. Filter to non-inherited.
    non_inherited = [tid for tid in failing_ids if tid not in inherited_set]
    if not non_inherited:
        return _skip()

    # 5. Build targeted command.
    from pathlib import Path
    from refactory_dsh.check_digest import build_probe_command
    root_path = Path(root) if root else None
    targeted_cmd, refusal = build_probe_command(check_command, non_inherited, root=root_path)
    if refusal:
        return _skip()

    # 6. Pre-pass timeout.
    prepass_timeout = min(check_timeout_s // 3, 120)

    # 7. Run the targeted command.
    result = run_check_fn(targeted_cmd, prepass_timeout)

    # 8. Apply the early-stop rule.
    from refactory_dsh import authoring_evidence as evidence
    from refactory_dsh import base_check

    early_stop = True

    # Condition a: no prep failure.
    if getattr(result, "prep_failure", None) is not None:
        early_stop = False

    # Condition b: must be a code_failure.
    if early_stop and evidence.check_failure_kind(result) != "code_failure":
        early_stop = False

    # Condition c: exit code not 4 or 5.
    rc = getattr(result, "returncode", None)
    if early_stop and rc is not None and rc in (4, 5):
        early_stop = False

    # Condition d: parse failures and check that a targeted test still fails.
    if early_stop:
        raw_output = "\n".join(filter(None, [
            getattr(result, "stdout", None) or "",
            getattr(result, "stderr", None) or "",
        ])).strip()
        found_failures = base_check.parse_test_failures(raw_output)
        non_inherited_found = found_failures - inherited_set
        if not any(tid in non_inherited_found for tid in non_inherited):
            early_stop = False

    # Build the digest for the targeted output.
    from refactory_dsh.check_digest import digest_check_output, format_digest
    raw_combined = "\n".join(filter(None, [
        getattr(result, "stdout", None) or "",
        getattr(result, "stderr", None) or "",
    ])).strip()
    digest = digest_check_output(raw_combined)
    digest_text = format_digest(digest)
    runners = digest.to_dict().get("runners", [])
    raw_tail = digest.raw_tail[-500:]

    if early_stop:
        return PrepassResult(
            ran=True, early_stop=True, check_mode="targeted",
            targeted_command=targeted_cmd, result=result,
            digest_text=digest_text, runners=runners, raw_tail=raw_tail,
        )
    else:
        return PrepassResult(
            ran=True, early_stop=False, check_mode="full",
            targeted_command=targeted_cmd, result=result,
            digest_text=digest_text, runners=runners, raw_tail=raw_tail,
        )


def _extract_assertion_key(assertion: str) -> str:
    """Extract the ``>`` source line from an assertion string.

    "Same assertion" means the same ``>`` source line, even when the
    values differ (correction item 14).
    """
    for line in assertion.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            return stripped
    return assertion.strip()


def detect_stuck(
    prev_record: dict[str, Any],
    curr_runners: list[dict[str, Any]],
    curr_raw_output: str,
    worktree_path: str | None,
    prev_commit: str | None,
    curr_commit: str | None,
) -> str | None:
    """Detect when the same non-inherited test fails with the same assertion.

    Returns a formatted stuck notice string, or ``None``.
    """
    # Guard 1: previous record must be a failure.
    if prev_record.get("status") != "fail":
        return None

    # Guard 2: skip when same commit (retry/resume of the same attempt).
    if prev_commit is not None and curr_commit is not None and prev_commit == curr_commit:
        return None

    # Build prev_failures: test_id → assertion key.
    prev_failures: dict[str, str] = {}
    for runner in prev_record.get("runners", []):
        for f in runner.get("failures", []):
            tid = f.get("test_id", "")
            assertion = f.get("assertion", "")
            if tid:
                prev_failures[tid] = _extract_assertion_key(assertion)

    # Build curr_failures: test_id → assertion key.
    curr_failures: dict[str, str] = {}
    for runner in curr_runners:
        for f in runner.get("failures", []):
            tid = f.get("test_id", "")
            assertion = f.get("assertion", "")
            if tid:
                curr_failures[tid] = _extract_assertion_key(assertion)

    # Inherited set from previous record.
    inherited_set = set(prev_record.get("inherited", []))

    # Compare only tests both checks ran.
    common_ids = set(prev_failures) & set(curr_failures)
    stuck_tests = [
        (tid, curr_failures[tid])
        for tid in sorted(common_ids)
        if tid not in inherited_set and prev_failures[tid] == curr_failures[tid]
    ]

    if not stuck_tests:
        return None

    # Build diff output.
    diff_output = ""
    if worktree_path is not None:
        # Extract test file from the first stuck test.
        diff_files: set[str] = set()
        for tid, _ in stuck_tests:
            if "::" in tid:
                diff_files.add(tid.split("::")[0])
            elif " > " in tid:
                diff_files.add(tid.split(" > ")[0].strip())

        # The path:NN: lookup is documented as a known limitation — the whole
        # HEAD^..HEAD fallback is the documented behaviour.
        try:
            file_args = ["--"] + sorted(diff_files) if diff_files else []
            cp = subprocess.run(
                ["git", "diff", "HEAD^..HEAD"] + file_args,
                cwd=worktree_path, capture_output=True, text=True, timeout=10,
                env={**os.environ, "GIT_LITERAL_PATHSPECS": "1"},
            )
            diff_output = cp.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            diff_output = ""

        # If diff is empty for the specific files, show the whole diff bounded.
        if not diff_output:
            try:
                cp = subprocess.run(
                    ["git", "diff", "HEAD^..HEAD"],
                    cwd=worktree_path, capture_output=True, text=True, timeout=10,
                )
                lines = cp.stdout.strip().splitlines()
                diff_output = "\n".join(lines[:200])
            except (subprocess.TimeoutExpired, OSError):
                diff_output = "(diff unavailable)"

    # Format the notice.
    parts = [
        "## Stuck: same failure on consecutive checks\n",
        "The following test(s) failed with the same assertion as the previous attempt.",
        "Try a different approach.\n",
    ]
    for tid, assertion_key in stuck_tests:
        parts.append(f"- **{tid}**: {assertion_key}")
    parts.append("")

    if diff_output:
        parts.append("### Your last change (HEAD^..HEAD) for the relevant files:")
        parts.append("```diff")
        parts.append(diff_output)
        parts.append("```")

    return "\n".join(parts)


def record_check_history(
    ctx: Any,
    check_mode: str,
    result_output: str,
    check_command: str,
) -> None:
    """Append to convergence-detection history, guarding on check_mode.

    Targeted runs are excluded so a subset never looks like convergence.
    """
    if check_mode == "targeted":
        return

    from refactory_dsh import base_check

    current_failures = sorted(base_check.parse_test_failures(result_output))
    history = list(ctx.facts.get("check_failures_history", []))
    history.append(current_failures)
    ctx.set_fact("check_failures_history", history)

    blocker = base_check.excusal_blocker(
        check_command, result_output, set(current_failures)
    ) if current_failures else "empty failure set"
    blocker_history = list(ctx.facts.get("check_blocker_history", []))
    blocker_history.append(blocker)
    ctx.set_fact("check_blocker_history", blocker_history)
