#!/usr/bin/env python3
"""Refresh refactory/tests/ci_durations.json from CI's JUnit reports.

    uv run python scripts/ci-durations.py --latest 5          # last 5 green CI runs on main
    uv run python scripts/ci-durations.py junit/<run> [...]   # runs already downloaded with
        # gh run download <run> -p 'python-suite-times*' -D junit/<run>

Each test's seconds are averaged over the runs that ran it, then credited to the file
that holds it now (from `pytest --collect-only`), not the file the report names, so a
test moved between files keeps its time. The file holds one entry per test file (its
seconds) and, for a file longer than SPLIT_SECONDS, one entry per test function as
well (`file.py::test_name`, every parametrization summed). `--shard` packs whole files
and the functions of those long files one by one (tests/conftest.py), so no file holds
a shard up on its own. A file with no time yet counts as the median file, and a new
function in a long file as that file's median function. Run it where the shards run
(the code-node sandbox available), so environment-dependent cases are counted.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "refactory" / "tests"
OUT = TESTS / "ci_durations.json"
REPO = "boston-harness-group/refactory"
#: A file longer than this is packed test by test. A shard's four workers each get about
#: an eighth of the suite's half hour; a file of four minutes or more is too big a
#: piece to balance that well, and one of fourteen sets the whole job's time.
SPLIT_SECONDS = 240.0


def _collect() -> list[str]:
    """Every test the suite collects now, as ``file.py::rest`` (the durations keys)."""
    listed = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--collect-only", "-q",
         str(TESTS)],
        capture_output=True, text=True, cwd=ROOT, check=False).stdout
    return [line.split("::")[0].rsplit("/", 1)[-1] + line[len(line.split("::")[0]):]
            for line in listed.splitlines()
            if line.startswith("refactory/tests/") and "::" in line]


def _junit_id(case: ET.Element) -> tuple[str, str]:
    """(file, test id) a JUnit case names: ``refactory.tests.test_x.TestThing`` plus
    ``test_y[p]`` is ``test_x.py`` and ``test_x.py::TestThing::test_y[p]``."""
    parts = (case.get("classname") or "").split(".")
    index = next((i for i in range(len(parts) - 1, -1, -1)
                  if parts[i].startswith(("test_", "_"))), len(parts) - 1)
    module = parts[index] + ".py"
    return module, "::".join([module, *parts[index + 1:], case.get("name") or ""])


def _home(module: str, name: str, where: dict[str, set[str]]) -> str:
    """Where a test the report placed in ``module`` lives now, when its id is gone: still
    there, else the one file it moved to, else the file split off ``module`` that
    defines it."""
    files = where.get(name, set())
    if module in files or not files:
        return module
    if len(files) == 1:
        return next(iter(files))
    split = sorted(f for f in files if f.startswith(module[:-3] + "_"))
    return split[0] if split else module


def durations(run_dirs: list[Path], collected: list[str]) -> dict[str, float]:
    """The durations file's contents for these runs' reports and this collection."""
    current = set(collected)
    functions = {test.split("[", 1)[0] for test in collected}
    where: dict[str, set[str]] = collections.defaultdict(set)
    for test in collected:
        where[re.sub(r"\[.*", "", test.split("::")[-1])].add(test.split("::")[0])
    seconds: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for run in run_dirs:
        per_run: collections.Counter[tuple[str, str]] = collections.Counter()
        for report in sorted(run.rglob("*.xml")):
            for case in ET.parse(report).getroot().iter("testcase"):
                module, test = _junit_id(case)
                if test not in current:
                    name = re.sub(r"\[.*", "", case.get("name") or "")
                    module = _home(module, name, where)
                per_run[(module, test)] += float(case.get("time") or 0)
        for key, value in per_run.items():
            seconds[key].append(value)
    by_file: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for (module, test), values in seconds.items():
        by_file[module][test] = sum(values) / len(values)
    out: dict[str, float] = {}
    for module, tests in by_file.items():
        total = sum(tests.values())
        out[module] = round(total, 1)
        if total > SPLIT_SECONDS:
            per_function: collections.Counter[str] = collections.Counter()
            for test, value in tests.items():
                if test.split("[", 1)[0] in functions:
                    per_function[test.split("[", 1)[0]] += value
            out.update({key: round(value, 1) for key, value in per_function.items()})
    return dict(sorted(out.items()))


def _download(latest: int, into: Path) -> list[Path]:
    """The JUnit reports of the last ``latest`` green CI runs on main."""
    runs = json.loads(subprocess.run(
        ["gh", "run", "list", "-R", REPO, "-w", "ci.yml", "-b", "main", "-e", "push",
         "-s", "success", "-L", str(latest), "--json", "databaseId"],
        capture_output=True, text=True, check=True).stdout)
    dirs = []
    for run in runs:
        target = into / str(run["databaseId"])
        subprocess.run(["gh", "run", "download", str(run["databaseId"]), "-R", REPO,
                        "-p", "python-suite-times*", "-D", str(target)], check=True)
        dirs.append(target)
    return dirs


def _summary(table: dict[str, float]) -> str:
    files = {k: v for k, v in table.items() if "::" not in k}
    split = sorted(k for k in files if any(t.startswith(k + "::") for t in table))
    return (f"{len(files)} files, {sum(files.values()) / 60:.1f} min in all; "
            f"packed function by function: {', '.join(split) or 'none'}; "
            f"median file {statistics.median(files.values()):.1f}s")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="*", type=Path, help="one directory per CI run")
    parser.add_argument("--latest", type=int, help="download the last N green main runs")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        runs = list(args.runs) + (_download(args.latest, Path(tmp)) if args.latest else [])
        if not runs:
            parser.error("name run directories or pass --latest N")
        table = durations(runs, _collect())
    text = json.dumps(table, indent=1) + "\n"
    print(f"{len(runs)} run(s): {_summary(table)}")
    args.out.write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
