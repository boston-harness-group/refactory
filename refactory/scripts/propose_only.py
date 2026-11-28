"""Propose-only replay: rebuild loop state from a run directory, run the analysts and
proposer with a real model, evaluate nothing, and write the rendered requests, replies
and proposals to an output directory.

    python -m refactory.scripts.propose_only <run dir> <output dir> [--epoch N]
        [--project DIR]

Works for runs recorded by older commits: the run's config and journal are read as-is,
the round is replayed with the current code's rendering. The run directory is only read
(a temporary copy is made internally). Asserts that no val/test item content enters any
request.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


def _load_val_test_content(run_dir: Path) -> set[str]:
    """Collect val/test item content strings from the run's data to guard against leakage."""
    tokens: set[str] = set()
    for name in ("val_items.json", "test_items.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            items = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("input", "expected", "question", "answer", "content", "text"):
                value = item.get(key)
                if isinstance(value, str) and len(value) > 20:
                    tokens.add(value.strip())
    return tokens


def _assert_no_leakage(text: str, val_test_tokens: set[str], label: str) -> None:
    """Raise if any val/test content appears in a request."""
    for token in val_test_tokens:
        if token in text:
            raise AssertionError(
                f"val/test content leaked into {label}: "
                f"{token[:80]!r}...")


def run(run_dir: Path, output_dir: Path, epoch: int, project: Path,
        client: Any = None) -> int:
    from refactory.refactory_dsh.tools.replay_round import (
        build_loop, cut_journal, SET_ASIDE,
    )

    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    val_test_tokens = _load_val_test_content(run_dir)

    with tempfile.TemporaryDirectory(prefix="propose_only_") as tmp:
        import shutil

        copy_dir = Path(tmp) / "run_copy"
        shutil.copytree(run_dir, copy_dir, symlinks=False, ignore_dangling_symlinks=True)
        for name in SET_ASIDE:
            path = copy_dir / name
            if path.exists():
                path.rename(copy_dir / f"{name}.before-replay")
        kept = cut_journal(copy_dir / "journal.jsonl", epoch)
        print(f"journal cut to its first {kept} events", file=sys.stderr)

        loop = build_loop(copy_dir, project, client)
        from refactory.refactory_dsh.outerloop import Run

        loop.init()
        loop._memory_pass = lambda: None  # noqa: SLF001
        loop.install = lambda run, proposals, parents: []
        loop.optimizer_step(Run(_loop=loop))

        trace_path = copy_dir / "optimizer_trace.jsonl"
        if not trace_path.exists():
            print("no optimizer trace written", file=sys.stderr)
            return 1

        calls = []
        for line in trace_path.read_text().splitlines():
            try:
                calls.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    for i, call in enumerate(calls):
        kind = call.get("kind", "unknown")
        prefix = f"{i:03d}_{kind}"

        prompt = call.get("prompt", "")
        if isinstance(prompt, str) and prompt:
            (output_dir / f"{prefix}_request.md").write_text(prompt)
            if val_test_tokens:
                _assert_no_leakage(prompt, val_test_tokens, f"call {i} ({kind}) request")

        reply = call.get("reply", "")
        if isinstance(reply, str) and reply:
            (output_dir / f"{prefix}_reply.md").write_text(reply)

        system = call.get("system", "")
        if isinstance(system, str) and system:
            (output_dir / f"{prefix}_system.md").write_text(system)

        findings = call.get("findings")
        if findings:
            (output_dir / f"{prefix}_findings.json").write_text(
                json.dumps(findings, indent=2, default=str))

    proposal_calls = [c for c in calls if str(c.get("kind", "")).startswith("proposal")]
    for i, call in enumerate(proposal_calls):
        reply = call.get("reply", "")
        if isinstance(reply, str) and reply:
            try:
                payload = json.loads(re.sub(r'^[^{]*', '', reply.strip(), count=1))
            except (json.JSONDecodeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                proposals = payload.get("proposals") or payload.get("edits") or []
                if proposals:
                    (output_dir / f"proposals_{i}.json").write_text(
                        json.dumps(proposals, indent=2, default=str))
                diagnosis = payload.get("diagnosis", "")
                if diagnosis:
                    (output_dir / f"diagnosis_{i}.md").write_text(str(diagnosis))

    summary_lines = [f"# Propose-only replay: epoch {epoch}", "",
                     f"Run: {run_dir}", f"Calls: {len(calls)}", ""]
    for i, call in enumerate(calls):
        kind = call.get("kind", "unknown")
        items = call.get("items")
        summary_lines.append(
            f"- {i:03d} {kind}"
            + (f" items={items}" if items else "")
            + (f" error={call['error']}" if call.get("error") else ""))
    summary_lines.append("")
    (output_dir / "summary.md").write_text("\n".join(summary_lines))
    print(f"wrote {len(list(output_dir.iterdir()))} files to {output_dir}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None, client: Any = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m refactory.scripts.propose_only",
        description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="the stopped run (only read)")
    parser.add_argument("output_dir", type=Path, help="where to write rendered output")
    parser.add_argument("--epoch", type=int, default=1, help="the round's epoch (1)")
    parser.add_argument("--project", type=Path, default=Path("."),
                        help="the project the run was made in")
    args = parser.parse_args(argv)
    return run(args.run_dir, args.output_dir, args.epoch, args.project.resolve(), client)


if __name__ == "__main__":
    raise SystemExit(main())
