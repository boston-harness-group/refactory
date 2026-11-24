#!/usr/bin/env python3
"""Check that CI's Python shards together collect exactly the unsharded suite.

    uv run python scripts/check-shards.py [N]      # N shards, default 4

Collects `refactory/tests` once whole and once per `--shard I/N`, and fails when a test
is in no shard, in two, or in a shard but not the suite. The fast-checks job runs it.
"""

from __future__ import annotations

import collections
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


def main(shards: int) -> int:
    full = collect()
    seen: collections.Counter[str] = collections.Counter()
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
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4))
