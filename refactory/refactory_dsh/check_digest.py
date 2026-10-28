"""Parse check output into a structured per-runner failure digest.

Extracts each failing test's id, assertion and short traceback from pytest,
vitest and node's test runner output.  Unknown runners fall back to the raw
tail.  ANSI codes, passing-test lines and source-map noise are stripped.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from refactory_dsh.base_check import parse_test_failures, strip_ansi

# ── data ──────────────────────────────────────────────────────────────────────

RAW_TAIL_LIMIT = 100


@dataclass
class FailureEntry:
    test_id: str
    assertion: str
    traceback: str


@dataclass
class RunnerDigest:
    name: str
    passed: bool
    failures: list[FailureEntry] = field(default_factory=list)


@dataclass
class CheckDigest:
    runners: list[RunnerDigest] = field(default_factory=list)
    raw_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "runners": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "failures": [
                        {"test_id": f.test_id, "assertion": f.assertion, "traceback": f.traceback}
                        for f in r.failures
                    ],
                }
                for r in self.runners
            ],
            "raw_tail": self.raw_tail,
        }


# ── ANSI / noise filters ─────────────────────────────────────────────────────

_SOURCEMAP_WARNING = re.compile(
    r"^.*Sourcemap\s.*points\sto\smissing\ssource\sfiles.*$", re.MULTILINE
)
_PASSING_PYTEST = re.compile(r"^.*\s+PASSED\s*$", re.MULTILINE)
_PASSING_VITEST = re.compile(r"^\s*[✓✔]\s+.*$", re.MULTILINE)


def _clean(output: str) -> str:
    """Strip ANSI codes, source-map warnings and passing-test lines."""
    text = strip_ansi(output)
    text = _SOURCEMAP_WARNING.sub("", text)
    text = _PASSING_PYTEST.sub("", text)
    text = _PASSING_VITEST.sub("", text)
    # collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ── pytest parser ─────────────────────────────────────────────────────────────

_PYTEST_SECTION = re.compile(
    r"^={3,}\s*(FAILURES|ERRORS|short test summary info)\s*={3,}$", re.MULTILINE
)
_PYTEST_HEADER = re.compile(r"^_{3,}\s*(.+?)\s*_{3,}$", re.MULTILINE)
_PYTEST_FAILED_LINE = re.compile(r"^FAILED\s+([\w/.:\[\]-]+(?:::[\w\[\]-]+)+)", re.MULTILINE)
_PYTEST_ERROR_LINE = re.compile(
    r"^(?:ERROR|ERRORS?)\s+([\w/.:\[\]-]+(?:::[\w\[\]-]+)*)\s*[-–—]", re.MULTILINE
)


def _parse_pytest(text: str) -> RunnerDigest | None:
    """Parse pytest output into a RunnerDigest, or None if not pytest."""
    failed_ids: set[str] = set()
    for m in _PYTEST_FAILED_LINE.finditer(text):
        failed_ids.add(m.group(1).strip())
    for m in _PYTEST_ERROR_LINE.finditer(text):
        failed_ids.add(m.group(1).strip())
    if not failed_ids:
        return None

    failures: list[FailureEntry] = []
    # Try to extract assertion and traceback from FAILURES section
    failures_section = ""
    for m in _PYTEST_SECTION.finditer(text):
        if m.group(1) == "FAILURES":
            end = text.find("\n===", m.end())
            failures_section = text[m.end():end] if end > 0 else text[m.end():]
            break

    headers = list(_PYTEST_HEADER.finditer(failures_section))
    for i, hdr in enumerate(headers):
        test_name = hdr.group(1).strip()
        start = hdr.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(failures_section)
        block = failures_section[start:end].strip()

        # Find the assertion line
        assertion = ""
        traceback_line = ""
        lines = block.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("AssertionError", "AssertionError:", "assert ",
                                   "AssertError", "E       ", "E  ",
                                   "> ", "raise ", "TypeError", "ValueError",
                                   "KeyError", "AttributeError", "ImportError",
                                   "ModuleNotFoundError", "NameError", "RuntimeError",
                                   "FileNotFoundError", "IndexError", "Exception")):
                if stripped.startswith("E "):
                    stripped = stripped[2:].strip()
                if not assertion:
                    assertion = stripped[:200]
            if stripped.startswith("File ") and "test" in stripped.lower():
                traceback_line = stripped[:200]

        # Match test_name to one of our failed_ids
        matched_id = ""
        for fid in failed_ids:
            if test_name in fid or fid.endswith(test_name):
                matched_id = fid
                break
        if not matched_id:
            matched_id = test_name

        failures.append(FailureEntry(
            test_id=matched_id,
            assertion=assertion or "(see traceback)",
            traceback=traceback_line,
        ))

    # Ensure every failed_id has an entry
    seen = {f.test_id for f in failures}
    for fid in sorted(failed_ids):
        if fid not in seen:
            failures.append(FailureEntry(test_id=fid, assertion="(see raw output)", traceback=""))

    return RunnerDigest(name="pytest", passed=False, failures=failures)


# ── vitest parser ─────────────────────────────────────────────────────────────

_VITEST_FAIL_LINE = re.compile(r"^\s*(?:FAIL|×)\s+(.+)$", re.MULTILINE)
_VITEST_ASSERT = re.compile(r"^\s*(AssertionError|Error|Expected|Received|expect\()", re.MULTILINE)


def _parse_vitest(text: str) -> RunnerDigest | None:
    """Parse vitest output into a RunnerDigest, or None if not vitest."""
    fail_matches = list(_VITEST_FAIL_LINE.finditer(text))
    candidates = [m.group(1).strip() for m in fail_matches if " > " in m.group(1)]
    if not candidates:
        return None

    failures: list[FailureEntry] = []
    for test_id in candidates:
        # Find assertion near the test id
        idx = text.find(test_id)
        if idx < 0:
            idx = 0
        block = text[idx:idx + 2000]
        assertion = ""
        traceback_line = ""
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith(("AssertionError", "Error:", "Expected", "Received",
                                   "expect(", "TypeError", "ReferenceError")):
                if not assertion:
                    assertion = stripped[:200]
            if stripped.startswith("❯") or (stripped.startswith("File ") and "/" in stripped):
                traceback_line = stripped[:200]
        failures.append(FailureEntry(
            test_id=test_id,
            assertion=assertion or "(see raw output)",
            traceback=traceback_line,
        ))

    return RunnerDigest(name="vitest", passed=False, failures=failures)


# ── node test runner parser ───────────────────────────────────────────────────

_NODE_TEST_FAIL = re.compile(r"^\s*(?:✖|not ok)\s+(?:\d+\s+)?(.+)$", re.MULTILINE)
_NODE_TEST_DIAG = re.compile(r"^\s+---$", re.MULTILINE)


def _parse_node_test_runner(text: str) -> RunnerDigest | None:
    """Parse node's built-in test runner output into a RunnerDigest."""
    fail_matches = list(_NODE_TEST_FAIL.finditer(text))
    if not fail_matches:
        return None
    # Confirm this is actually node's test runner (look for TAP-style or node:test markers)
    if "node --test" not in text and "# tests" not in text and "not ok" not in text and "✖" not in text:
        return None

    failures: list[FailureEntry] = []
    for m in fail_matches:
        test_id = m.group(1).strip()
        # Look for assertion info in the diagnostic block after this test
        idx = m.end()
        block = text[idx:idx + 1500]
        assertion = ""
        traceback_line = ""
        in_diag = False
        for line in block.splitlines():
            stripped = line.strip()
            if stripped == "---":
                in_diag = True
                continue
            if stripped == "...":
                break
            if in_diag:
                if stripped.startswith(("operator:", "expected:", "actual:", "message:")):
                    if not assertion:
                        assertion = stripped[:200]
                if "Error:" in stripped and not assertion:
                    assertion = stripped[:200]
            if stripped.startswith("at ") and ("test" in stripped.lower() or "/" in stripped):
                traceback_line = stripped[:200]
        failures.append(FailureEntry(
            test_id=test_id,
            assertion=assertion or "(see raw output)",
            traceback=traceback_line,
        ))

    return RunnerDigest(name="node_test_runner", passed=False, failures=failures)


# ── public API ────────────────────────────────────────────────────────────────

def digest_check_output(output: str) -> CheckDigest:
    """Parse check output and return a structured digest."""
    cleaned = _clean(output)
    digest = CheckDigest()

    # Try each runner parser
    pytest_result = _parse_pytest(cleaned)
    if pytest_result:
        digest.runners.append(pytest_result)

    vitest_result = _parse_vitest(cleaned)
    if vitest_result:
        digest.runners.append(vitest_result)

    node_result = _parse_node_test_runner(cleaned)
    if node_result:
        digest.runners.append(node_result)

    # Raw tail: last RAW_TAIL_LIMIT lines of cleaned output
    lines = cleaned.strip().splitlines()
    digest.raw_tail = "\n".join(lines[-RAW_TAIL_LIMIT:]) if lines else ""

    return digest


def format_digest(digest: CheckDigest) -> str:
    """Format a CheckDigest as human-readable text for the builder."""
    parts: list[str] = []
    for runner in digest.runners:
        if not runner.failures:
            continue
        parts.append(f"## Failing tests ({runner.name})\n")
        for f in runner.failures:
            line = f"- **{f.test_id}**"
            if f.assertion:
                line += f" — {f.assertion}"
            parts.append(line)
            if f.traceback:
                parts.append(f"  {f.traceback}")
        parts.append("")

    if digest.raw_tail:
        parts.append(f"## Raw tail (last {RAW_TAIL_LIMIT} lines)\n")
        parts.append(digest.raw_tail)

    return "\n".join(parts)


#: pytest options that take a value, so the value is not a positional path.
#: A flag missing here whose value happens to name a path would lose the value and
#: leave the flag dangling (`-n 4` became `-n refactory/tests/...` once).
_PYTEST_VALUE_OPTIONS = frozenset({
    "-n", "-k", "-m", "-p", "-o", "-W", "-c", "-r",
    "--dist", "--maxfail", "--durations", "--timeout", "--tb", "--rootdir",
    "--junitxml", "--junit-xml", "--deselect", "--ignore", "--ignore-glob",
    "--import-mode", "--confcutdir", "--basetemp", "--cov", "--cov-report",
    "--override-ini", "--color", "--capture", "--log-level", "--log-file",
    "--log-cli-level", "--log-format", "--assert",
})


def _strip_pytest_positional_paths(step: str) -> str:
    """Remove positional path arguments from a pytest command, keeping flags and values.

    ``pytest -q tests`` becomes ``pytest -q`` so that only the appended test
    IDs run, instead of the whole directory. A flag that takes a value keeps it:
    ``-n 4`` and ``-p no:cacheprovider`` stay whole, because dropping the value
    leaves the flag dangling and the targeted command runs nothing.
    """
    tokens = shlex.split(step)
    kept: list[str] = []
    past_cmd = False
    expect_value = False
    for tok in tokens:
        if not past_cmd:
            kept.append(tok)
            if "pytest" in tok:
                past_cmd = True
            continue
        if expect_value:
            kept.append(tok)
            expect_value = False
            continue
        if tok.startswith("-"):
            kept.append(tok)
            expect_value = tok in _PYTEST_VALUE_OPTIONS
            continue
        # A positional path like "tests" or "refactory/tests": dropped.
    return shlex.join(kept)


#: A test file vitest (or jest) runs whole when it is named: a spec or test script.
_SCRIPT_TEST_FILE = re.compile(r"\.(?:spec|test)\.[cm]?[jt]sx?$")


def _vitest_files(vitest_ids: list[str], whole_files: list[str]) -> tuple[str, str]:
    """The file arguments and ``-t`` pattern a vitest step gets for these ids and whole files.

    A file named whole runs every test in it, so once any is named the ids' files are
    run whole too: a ``-t`` pattern would filter the whole files down to the ids' tests.
    """
    files: set[str] = set(whole_files)
    patterns: set[str] = set()
    for vid in vitest_ids:
        parts = vid.split(" > ")
        if parts:
            files.add(parts[0].strip())
        if len(parts) > 1:
            patterns.add(re.escape(parts[-1].strip()))
    file_args = " ".join(shlex.quote(f) for f in sorted(files))
    pattern_args = (f"-t {shlex.quote('|'.join(sorted(patterns)))}"
                    if patterns and not whole_files else "")
    return file_args, pattern_args


def build_probe_command(check_command: str, test_ids: list[str], *, root: 'Path | None' = None,
                        test_files: list[str] | None = None) -> tuple[str, str | None]:
    """Build a targeted command per runner from the check command and failing test IDs.

    Returns (command, None) on success or ("", refusal_reason) on failure.
    Classifies each test id by runner so a mixed check gets only its own ids per step.

    Parameters
    ----------
    root:
        Directory to read ``package.json`` from for the pnpm/vitest check.
        Defaults to ``Path(".")`` (the process's cwd).
    test_files:
        Test files to run whole (the self-check's selection, b41a), as paths from the
        repository root. Both runners take a file path where they take a test id: a
        ``.py`` file goes to each pytest step, a spec or test script to each vitest
        step. A step with neither ids nor files of its runner is left out, as it is for
        ids alone, so a selection of Python files never starts the vitest step.
    """
    from pathlib import Path
    whole = list(test_files or ())
    if not test_ids and not whole:
        return "", "no test IDs to target"

    from refactory_dsh.base_check import _steps

    # Classify test IDs by runner
    pytest_ids: list[str] = []
    vitest_ids: list[str] = []
    node_ids: list[str] = []
    vitest_files: list[str] = []
    for tid in test_ids:
        if "::" in tid:
            pytest_ids.append(tid)
        elif " > " in tid:
            vitest_ids.append(tid)
        elif _SCRIPT_TEST_FILE.search(tid):
            vitest_files.append(tid)
        else:
            node_ids.append(tid)
    pytest_files = [name for name in whole if name.endswith(".py")]
    vitest_files.extend(name for name in whole if _SCRIPT_TEST_FILE.search(name))
    # An id in a file that runs whole is already run by it.
    pytest_ids = [tid for tid in pytest_ids if tid.split("::")[0] not in pytest_files]

    if node_ids and not pytest_ids and not vitest_ids and not pytest_files and not vitest_files:
        return "", "targeting node's test runner is not supported"

    steps = _steps(check_command)
    targeted_parts: list[str] = []
    found_recognized = False
    for step in steps:
        step = step.strip()
        if not step:
            continue
        # Detect runner for this step
        if "pytest" in step or "python -m pytest" in step:
            found_recognized = True
            if not pytest_ids and not pytest_files:
                continue  # skip steps with no failures
            # Strip positional paths and append only the named test IDs
            base = _strip_pytest_positional_paths(step)
            named = [*pytest_files, *pytest_ids]
            targeted_parts.append(f"{base} {' '.join(shlex.quote(tid) for tid in named)}")
        elif "vitest" in step:
            found_recognized = True
            if not vitest_ids and not vitest_files:
                continue
            # Map vitest ids to file filters plus -t patterns
            file_args, pattern_args = _vitest_files(vitest_ids, vitest_files)
            targeted_parts.append(f"{step} {file_args} {pattern_args}".strip())
        elif "pnpm test" in step:
            found_recognized = True
            if not vitest_ids and not vitest_files:
                continue  # No vitest IDs target this step — skip silently
            # Target only when package.json's "test" script is vitest.
            import json as _json
            pkg_root = root if root is not None else Path(".")
            try:
                pkg = _json.loads((pkg_root / "package.json").read_text())
                test_script = (pkg.get("scripts") or {}).get("test", "")
            except (OSError, _json.JSONDecodeError):
                test_script = ""
            if "vitest" not in test_script:
                return "", "pnpm test targeting requires vitest as the test script"
            file_args, pattern_args = _vitest_files(vitest_ids, vitest_files)
            targeted_parts.append(f"{step} {file_args} {pattern_args}".strip())
        elif "node " in step and ("--test" in step or "node_modules" not in step):
            found_recognized = True
            return "", "targeting node's test runner is not supported"
        else:
            # Keep build steps and other non-test steps
            targeted_parts.append(step)

    if not found_recognized:
        return "", "no recognised test runner in the check command"

    if not targeted_parts:
        return "", "no targetable test steps found in the check command"

    return " && ".join(targeted_parts), None
