#!/usr/bin/env python3
"""Check that CI's Python shards together cover exactly the unsharded suite.

    uv run python scripts/check-shards.py [N]           # plan: collect whole, and per shard
    python3 scripts/check-shards.py --reports DIR [N]   # what the N shard jobs really ran

The first collects `refactory/tests` once whole and once per `--shard I/N` (fast-checks
job). The second reads, under DIR, each shard job's JUnit report and the whole
collection that job took in its own environment (`collected-shardI.txt`): the four
collections must agree, and the JUnit test ids together must be that collection, each
once (the job after the shards). Either fails when a test is in no shard, in two, or in
a shard but not the suite. The second needs only the standard library.
"""

from __future__ import annotations

import collections
import xml.etree.ElementTree as ET
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def collect(*extra: str) -> list[str]:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "refactory/tests", "--collect-only", "-q",
         "-p", "no:cacheprovider", *extra],
        capture_output=True, text=True, cwd=ROOT, check=False)
    tests = [line for line in done.stdout.splitlines()
             if line.startswith("refactory/tests/") and "::" in line]
    if done.returncode not in (0, 5) or not tests:
        sys.stdout.write(done.stdout[-4000:] + done.stderr[-4000:])
        raise SystemExit(f"collection {' '.join(extra) or '(whole suite)'} failed")
    return tests


def _junit_id(case: ET.Element) -> str:
    """The test id a JUnit case names: ``refactory.tests.test_x.TestThing`` plus
    ``test_y[p]`` is ``refactory/tests/test_x.py::TestThing::test_y[p]``."""
    parts = (case.get("classname") or "").split(".")
    index = next((i for i in range(len(parts) - 1, -1, -1)
                  if parts[i].startswith(("test_", "_"))), len(parts) - 1)
    return "::".join(["/".join(parts[:index + 1]) + ".py", *parts[index + 1:],
                      case.get("name") or ""])


def from_reports(root: Path, shards: int) -> tuple[list[str], collections.Counter[str]]:
    collections_seen: dict[int, list[str]] = {}
    seen: collections.Counter[str] = collections.Counter()
    for shard in range(1, shards + 1):
        listed = sorted(root.rglob(f"collected-shard{shard}.txt"))
        reports = sorted(root.rglob(f"python-suite-shard{shard}.xml"))
        if len(listed) != 1 or len(reports) != 1:
            raise SystemExit(f"shard {shard}: expected one collection and one JUnit report "
                             f"under {root}, found {len(listed)} and {len(reports)}")
        collections_seen[shard] = [line for line in listed[0].read_text().splitlines()
                                   if line.startswith("refactory/tests/") and "::" in line]
        ran = [_junit_id(case) for case in ET.parse(reports[0]).getroot().iter("testcase")]
        print(f"shard {shard}/{shards}: ran {len(ran)} tests; its runner collected "
              f"{len(collections_seen[shard])}")
        seen.update(ran)
    full = collections_seen[1]
    for shard, other in collections_seen.items():
        if set(other) != set(full):
            print(f"shard {shard}'s runner collected a different suite from shard 1's: "
                  f"{len(set(full) - set(other))} fewer, {len(set(other) - set(full))} more")
            for test in sorted(set(full) ^ set(other))[:20]:
                print(f"  differs: {test}")
            raise SystemExit(1)
    return full, seen


def main(shards: int, reports: Path | None = None) -> int:
    if reports is not None:
        full, seen = from_reports(reports, shards)
    else:
        full = collect()
        seen = collections.Counter()
        for shard in range(1, shards + 1):
            mine = collect("--shard", f"{shard}/{shards}")
            print(f"shard {shard}/{shards}: {len(mine)} tests")
            seen.update(mine)
    missed = sorted(set(full) - set(seen))
    twice = sorted(test for test, count in seen.items() if count > 1)
    extra = sorted(set(seen) - set(full))
    print(f"whole suite: {len(full)} tests; shards together: {sum(seen.values())} "
          f"({len(missed)} missed, {len(twice)} in two shards, {len(extra)} not in the suite)")
    for label, tests in (("missed", missed), ("in two shards", twice), ("extra", extra)):
        for test in tests[:50]:
            print(f"  {label}: {test}")
    return 1 if missed or twice or extra else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    where = None
    if args[:1] == ["--reports"]:
        where, args = Path(args[1]), args[2:]
    raise SystemExit(main(int(args[0]) if args else 4, where))
