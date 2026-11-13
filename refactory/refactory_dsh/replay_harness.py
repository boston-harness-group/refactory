"""Replay harness: score a verifier against the evidence log.

For each entry with a pre-fix commit, run the verifier with only that entry's
brief, diff and repository, and report how many known gaps it finds and how many
findings were noise.

Usage:
    python -m refactory_dsh.replay_harness [--evidence PATH] [--output PATH] [--repo PATH] --model MODEL

A post-build verifier does not cover gaps found at a design gate (which have no
diff to verify).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from refactory_dsh.create_ops import MARKER_RULE


@dataclass
class ReplayResult:
    entry_id: str
    method: str
    found: bool          # True if the verifier identified the known gap
    noise_count: int     # Number of findings that don't match the known gap
    findings: list[str]  # Raw findings from the verifier
    skipped: bool = False  # True when the commit was unreachable


@dataclass
class ReplaySummary:
    total_entries: int
    found: int           # Entries whose gap the verifier identified
    missed: int          # Entries whose gap the verifier did not identify
    noise: int           # Total noise findings across all entries
    skipped: int = 0     # Entries skipped (commit unreachable)
    per_method: dict[str, dict[str, int]] = field(default_factory=dict)
    results: list[ReplayResult] = field(default_factory=list)


def _finding_matches(finding: str, entry: dict) -> bool:
    """Check whether a finding matches an entry's expected gap.

    Matching uses the entry's ``expected_finding_file`` and
    ``expected_finding_terms``.  A finding matches when it mentions the
    expected file AND at least one of the key terms (case-insensitive).
    """
    lower = finding.lower()
    expected_file = entry.get("expected_finding_file", "")
    terms = entry.get("expected_finding_terms", [])
    if not expected_file or not terms:
        return False
    if expected_file.lower() not in lower:
        return False
    return any(t.lower() in lower for t in terms)


def replay_one(entry: dict, verifier_fn: Callable) -> ReplayResult:
    """Replay one evidence-log entry through the verifier.

    ``entry`` must have: id, brief, gap_description, method,
    expected_finding_file, expected_finding_terms.

    ``verifier_fn(brief, diff, repo_path) -> list[str]`` returns blocking findings.
    ``repo_path`` is the path to a worktree checked out at the pre-fix commit,
    or None when no worktree is available (e.g. in unit tests).

    Scoring: ``found`` is True when any finding mentions the expected file
    AND at least one key term (case-insensitive).
    ``noise_count`` counts findings that do NOT match.
    """
    diff = entry.get("diff", "")
    findings = verifier_fn(entry["brief"], diff, entry.get("repo_path"))
    matched = any(_finding_matches(f, entry) for f in findings)
    noise = sum(1 for f in findings if not _finding_matches(f, entry))
    return ReplayResult(
        entry_id=entry["id"],
        method=entry["method"],
        found=matched,
        noise_count=noise,
        findings=findings,
    )


def replay_all(entries: list[dict], verifier_fn: Callable) -> ReplaySummary:
    """Replay all entries and aggregate into a summary.

    Entries whose ``repo_path`` is ``None`` (commit unreachable) are skipped
    and reported in the summary.
    """
    results: list[ReplayResult] = []
    per_method: dict[str, dict[str, int]] = {}
    skipped_count = 0

    for entry in entries:
        # Actually skip unreachable entries
        if entry.get("repo_path") is None and "repo_path" in entry:
            results.append(ReplayResult(
                entry_id=entry["id"], method=entry["method"],
                found=False, noise_count=0, findings=[], skipped=True,
            ))
            skipped_count += 1
            continue

        result = replay_one(entry, verifier_fn)
        results.append(result)
        method = result.method
        if method not in per_method:
            per_method[method] = {"found": 0, "missed": 0, "noise": 0}
        if result.found:
            per_method[method]["found"] += 1
        else:
            per_method[method]["missed"] += 1
        per_method[method]["noise"] += result.noise_count

    found = sum(1 for r in results if r.found)
    missed = sum(1 for r in results if not r.found and not r.skipped)
    noise = sum(r.noise_count for r in results)

    return ReplaySummary(
        total_entries=len(entries),
        found=found,
        missed=missed,
        noise=noise,
        skipped=skipped_count,
        per_method=per_method,
        results=results,
    )


def _build_diff(repo: Path, commit: str) -> str:
    """Build the diff for a pre-fix commit against its merge-base with main."""
    import subprocess
    try:
        # Use merge-base with main to get the PR's change, not one attempt's increment.
        base_result = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "main", commit],
            capture_output=True, text=True, check=True,
        )
        merge_base = base_result.stdout.strip()
        diff_result = subprocess.run(
            ["git", "-C", str(repo), "diff", merge_base, commit],
            capture_output=True, text=True, check=True,
        )
        return diff_result.stdout
    except subprocess.CalledProcessError:
        return ""


def _make_verifier_fn(client: Any):
    """Create a verifier function from a model client.

    Uses the same prompt (VERIFIER_PROMPT template) the run's verifier node uses,
    with the entry's brief, diff and worktree as substitutions.
    """
    from refactory_dsh.verification import VERIFICATION_METHODS

    def verifier_fn(brief: str, diff: str, repo_path: str | None = None) -> list[str]:
        prompt = (
            f"Artifact directory (absolute): {repo_path or '(not available)'}\n"
            f"Read and run the delivered files at this exact path; do not infer it "
            f"from your working directory.\n"
            f"Changes:\n{diff}\n\n"
            f"Brief: {brief}\n\n"
            + VERIFICATION_METHODS +
            "\nFor each finding, report it as:\n"
            "BLOCKING: <file:line>: <description with a runnable counterexample or "
            "a cited file:line chain>\n"
            + MARKER_RULE + "\n\n"
            "design.md is in the run directory. Do NOT read it, or strategy.json, "
            "or any review files. You verify the change against the brief and the "
            "repository only.\n\n"
            "End with a line that is exactly VERDICT: PASS when there are no "
            "blocking findings, or VERDICT: FAIL when there are any."
        )
        answer = client.complete(prompt, workdir=repo_path)
        return [
            line for line in answer.splitlines()
            if line.strip().upper().startswith("BLOCKING:")
        ]
    return verifier_fn


if __name__ == "__main__":
    import argparse
    import subprocess
    import tempfile

    parser = argparse.ArgumentParser(
        prog="python -m refactory_dsh.replay_harness",
        description="Score a verifier against the evidence log.",
    )
    parser.add_argument("--evidence", default="evidence_log_entries.json",
                        help="path to the evidence log entries JSON file")
    parser.add_argument("--output", default="verifier_baseline.json",
                        help="path to write the summary output")
    parser.add_argument("--repo", default=".",
                        help="repository to create worktrees from pre-fix commits")
    parser.add_argument("--model", default=None,
                        help="model name for the verifier backend (required)")
    parser.add_argument("--provider", default="",
                        help="provider for the model backend")
    parser.add_argument("--gh-evidence", action="store_true",
                        help="read entries from #400 evidence log via gh api")
    args = parser.parse_args()

    if not args.model:
        print("error: --model is required; the replay harness needs a real "
              "verifier backend, not a stub. Example:\n"
              "  python -m refactory_dsh.replay_harness "
              "--evidence refactory/tests/fixtures/evidence_log_entries.json "
              "--model claude-sonnet-4-20250514 --repo .",
              file=sys.stderr)
        sys.exit(1)

    entries = json.loads(Path(args.evidence).read_text())
    replayable = [e for e in entries if e.get("pre_fix_commit")]
    skipped_entries = [e for e in entries if not e.get("pre_fix_commit")]
    print(f"Loaded {len(entries)} entries; {len(replayable)} replayable, "
          f"{len(skipped_entries)} skipped (no pre_fix_commit)")

    repo = Path(args.repo).resolve()
    worktrees_created: list[Path] = []

    def _create_worktree(commit: str) -> Path | None:
        """Create a temporary git worktree at the given commit."""
        try:
            tmpdir = tempfile.mkdtemp(prefix="replay-wt-")
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "add", "--detach",
                 tmpdir, commit],
                check=True, capture_output=True, text=True,
            )
            worktrees_created.append(Path(tmpdir))
            return Path(tmpdir)
        except subprocess.CalledProcessError:
            return None

    def _remove_worktrees() -> None:
        """Remove all temporary git worktrees."""
        for wt in worktrees_created:
            try:
                subprocess.run(
                    ["git", "-C", str(repo), "worktree", "remove", "--force",
                     str(wt)],
                    capture_output=True, text=True,
                )
            except (subprocess.CalledProcessError, OSError):
                pass

    # Build verifier function from the model client — do this BEFORE
    # creating worktrees so a crash here does not leak them.
    from refactory_dsh.model import DshModelClient
    client = DshModelClient(model=args.model, provider=args.provider)
    verifier = _make_verifier_fn(client)

    # Set up worktrees and build diffs for replayable entries.
    actually_replayable: list[dict] = []
    try:
        for entry in replayable:
            commit = entry["pre_fix_commit"]
            # Check reachability first.
            reachable = subprocess.run(
                ["git", "-C", str(repo), "cat-file", "-t", commit],
                capture_output=True, text=True,
            )
            if reachable.returncode != 0:
                print(f"  {entry['id']}: commit {commit} is not reachable, skipping")
                entry["repo_path"] = None
                actually_replayable.append(entry)
                continue
            wt = _create_worktree(commit)
            if wt is None:
                print(f"  {entry['id']}: commit {commit} worktree failed, skipping")
                entry["repo_path"] = None
                actually_replayable.append(entry)
                continue
            entry["repo_path"] = str(wt)
            entry["diff"] = _build_diff(repo, commit)
            actually_replayable.append(entry)

        summary = replay_all(actually_replayable, verifier)
    finally:
        _remove_worktrees()

    import datetime
    output = {
        "model": args.model,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_entries": summary.total_entries,
        "found": summary.found,
        "missed": summary.missed,
        "noise": summary.noise,
        "skipped": summary.skipped,
        "per_method": summary.per_method,
        "results": [
            {"entry_id": r.entry_id, "method": r.method,
             "found": r.found, "noise_count": r.noise_count,
             "skipped": r.skipped}
            for r in summary.results
        ],
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote summary to {args.output}: found={summary.found}, "
          f"missed={summary.missed}, noise={summary.noise}, "
          f"skipped={summary.skipped}")
