"""How big the optimizer's requests are, and how much of them repeats.

Reads the calls a run's optimizer made (``optimizer_trace.jsonl``: every proposal,
analyst, merge, memory, comparison and step-review call with its prompt) and reports, per
call kind: how many calls, their size (max and mean characters), how much of them is
repeated text, and their largest sections. The measures are the ones the request tests
hold the renderers to (`repeated_share`, `own_headings`), so a size table in a PR and a
failing test say the same thing.

    python -m refactory_dsh.tools.request_sizes <run dir or optimizer_trace.jsonl> ...

Size is reported, never capped: a request may grow when it says more. Repetition is what
the tests fail on.
"""

from __future__ import annotations

import collections
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

#: A line shorter than this is not counted as repeated text (a label, a short heading, a
#: per-item status line that reads the same for different items).
REPEAT_LINE = 60
#: The renderers' own short pointers, which say where a repeated text was shown: not
#: repeated content themselves.
POINTERS = ("Sees: ", "Rest of its prompt: same as", "[a line of ", "[as in ")


def _outside_fences(text: str) -> list[str]:
    out, fence = [], None
    for line in text.splitlines():
        found = re.match(r"^(`{3,})", line)
        if found and (fence is None or found.group(1) == fence):
            fence = None if fence else found.group(1)
            continue
        if fence is None:
            out.append(line)
    return out


def own_headings(text: str) -> list[str]:
    """The request's own headings: Markdown headings outside any code fence."""
    return [line for line in _outside_fences(text) if re.match(r"#{1,6} ", line)]


def repeated_share(text: str) -> float:
    """The share of ``text`` in content lines (``REPEAT_LINE``+ characters) that appear more
    than once, the renderers' pointers aside."""
    lines = [" ".join(line.split()) for line in text.splitlines()
             if not line.startswith(POINTERS)]
    counts = collections.Counter(line for line in lines if len(line) >= REPEAT_LINE)
    return sum(len(line) * (n - 1) for line, n in counts.items() if n > 1) / max(1, len(text))


def sections(text: str) -> list[tuple[str, int]]:
    """Each top-level section (``## ``, outside fences) with its size, largest first."""
    out: list[tuple[str, int]] = []
    current, size = "(before the first section)", 0
    fence = None
    for line in text.splitlines():
        found = re.match(r"^(`{3,})", line)
        if found and (fence is None or found.group(1) == fence):
            fence = None if fence else found.group(1)
        elif fence is None and line.startswith("## "):
            out.append((current, size))
            current, size = line[3:].strip(), 0
        size += len(line) + 1
    out.append((current, size))
    return sorted(out, key=lambda pair: -pair[1])


def measure(text: str) -> dict[str, Any]:
    return {"chars": len(text), "headings": len(own_headings(text)),
            "repeated": round(repeated_share(text), 4), "largest": sections(text)[:3]}


def _calls(paths: Iterable[Path], skipped: list[str] | None = None
           ) -> Iterable[dict[str, Any]]:
    """Each call with a prompt; each line that is not JSON is added to ``skipped``
    (``path:line``), and a path with no trace too, so the report can say what it left out."""
    for path in paths:
        trace = path / "optimizer_trace.jsonl" if path.is_dir() else path
        if not trace.exists():
            if skipped is not None:
                skipped.append(f"{trace}: no such file")
            continue
        for number, line in enumerate(trace.read_text().splitlines(), 1):
            try:
                entry = json.loads(line)
            except ValueError:
                if skipped is not None and line.strip():
                    skipped.append(f"{trace}:{number}")
                continue
            if isinstance(entry, dict) and isinstance(entry.get("prompt"), str):
                yield entry


def report(paths: Iterable[Path], skipped: list[str] | None = None
           ) -> dict[str, dict[str, Any]]:
    """Per call kind: calls, chars max and mean, repeated share mean and max, and the
    largest sections of the longest call. What could not be read goes to ``skipped``."""
    by_kind: dict[str, list[str]] = collections.defaultdict(list)
    for entry in _calls(paths, skipped):
        by_kind[str(entry.get("kind"))].append(entry["prompt"])
    out = {}
    for kind, prompts in sorted(by_kind.items()):
        sizes = [len(p) for p in prompts]
        shares = [repeated_share(p) for p in prompts]
        longest = max(prompts, key=len)
        out[kind] = {"calls": len(prompts), "chars_max": max(sizes),
                     "chars_mean": round(statistics.mean(sizes)),
                     "repeated_mean": round(statistics.mean(shares), 4),
                     "repeated_max": round(max(shares), 4),
                     "largest_sections": sections(longest)[:3]}
    return out


def table(rows: dict[str, dict[str, Any]]) -> str:
    """The report as a Markdown table, for a PR's before/after."""
    lines = ["| call kind | calls | chars max | chars mean | repeated mean | repeated max | "
             "largest sections (longest call) |", "|---|---|---|---|---|---|---|"]
    for kind, row in rows.items():
        largest = "; ".join(f"{name[:40]} {size:,}" for name, size in row["largest_sections"])
        lines.append(f"| {kind} | {row['calls']} | {row['chars_max']:,} | {row['chars_mean']:,} | "
                     f"{row['repeated_mean']:.1%} | {row['repeated_max']:.1%} | {largest} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    paths = [Path(arg) for arg in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        print(__doc__)
        return 2
    skipped: list[str] = []
    print(table(report(paths, skipped)))
    if skipped:
        print(f"\n{len(skipped)} line(s) or file(s) could not be read and are not counted: "
              + ", ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
