#!/usr/bin/env python3
"""Refresh refactory/tests/ci_durations.json from CI's JUnit reports.

    gh run download <run> -p 'python-suite-times*' -D junit/<run>   # one or more runs
    uv run python scripts/ci-durations.py junit/<run> [junit/<another run> ...]

Each test's time is added to the file that holds it now (from `pytest --collect-only`),
not the file the report names, so a test moved between files keeps its time. A run's
reports (one per shard) are summed per file, and the runs are averaged. `--shard` packs
files by these seconds (tests/conftest.py); a file with no time yet counts as the median.
"""

from __future__ import annotations

import collections
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "refactory" / "tests"
OUT = TESTS / "ci_durations.json"


def _current_files() -> dict[str, set[str]]:
    """Test function name (parameters dropped) -> the files that define it now."""
    listed = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--collect-only", "-q", str(TESTS)],
        capture_output=True, text=True, cwd=ROOT, check=False).stdout
    found: dict[str, set[str]] = collections.defaultdict(set)
    for line in listed.splitlines():
        if "::" in line:
            path, name = line.split("::", 1)
            found[re.sub(r"\[.*", "", name)].add(path.rsplit("/", 1)[-1])
    return found


def _home(module: str, name: str, where: dict[str, set[str]]) -> str:
    """Where a test the report placed in ``module`` lives now: still there, else the one
    file it moved to, else the file split off ``module`` that defines it."""
    files = where.get(name, set())
    if module in files or not files:
        return module
    if len(files) == 1:
        return next(iter(files))
    split = sorted(f for f in files if f.startswith(module[:-3] + "_"))
    return split[0] if split else module


def main(run_dirs: list[str]) -> None:
    """Each argument is one run's directory; every JUnit file under it is one of its shards."""
    if not run_dirs:
        raise SystemExit(__doc__)
    where = _current_files()
    runs: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for run in run_dirs:
        for report in sorted(Path(run).rglob("*.xml")):
            for case in ET.parse(report).getroot().iter("testcase"):
                name = re.sub(r"\[.*", "", case.get("name") or "")
                module = (case.get("classname") or "").rsplit(".", 1)[-1] + ".py"
                runs[run][_home(module, name, where)] += float(case.get("time") or 0)
    files = sorted(set().union(*runs.values()))
    durations = {name: round(sum(run[name] for run in runs.values()) / len(runs), 1) for name in files}
    OUT.write_text(json.dumps(durations, indent=1, sort_keys=True) + "\n")
    print(f"{OUT.relative_to(ROOT)}: {len(durations)} files from {len(runs)} run(s), "
          f"{sum(durations.values()) / 60:.1f} min in all")


if __name__ == "__main__":
    main(sys.argv[1:])
