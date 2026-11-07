"""Reading a finished run: the past an optimizer is allowed to learn from.

A run leaves everything it did on disk. This module turns a run directory back
into objects, so an optimizer can consult a *previous* run — "the last three runs
all plateaued at grid=18" is a real prior, and it is only available if the
artifacts are readable rather than merely present.

It reads `report.json` when there is one and falls back to the per-candidate run
directories when there is not, so a run that crashed before writing its report is
still partly usable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The file every complete run writes, and the marker of a readable run.
REPORT = "report.json"

#: The run directory's layout version. Bumped when a field is added, renamed or
#: removed, so a reader fails loudly instead of reading an older report as if it
#: were current. The state dict already works this way; this is the same rule.
RUN_VERSION = 1

#: Where a run records its own version.
RUN_FILE = "run.json"

#: The append-only event log a run is replayed from, and the only place a
#: candidate's whole state is written down.
JOURNAL = "journal.jsonl"


class RunVersionError(ValueError):
    """A run directory was written by a different layout."""

#: How many candidates a summary keeps, so a prior cannot become a payload.
DEFAULT_LIMIT = 40


@dataclass(frozen=True)
class RunArchive:
    """A finished run, as much of it as is worth carrying forward."""

    path: Path
    mode: str = ""
    optimizer: str = ""
    bound: str = ""
    evaluations: int = 0
    seed_score: float | None = None
    best_score: float | None = None
    candidates: tuple[dict[str, Any], ...] = ()
    rejection_codes: dict[str, int] = field(default_factory=dict)
    accepted_op_mix: dict[str, int] = field(default_factory=dict)
    knob_attribution: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False

    def summary(self) -> dict[str, Any]:
        """The bounded form an optimizer prompt receives."""
        return {
            "path": str(self.path),
            "mode": self.mode,
            "optimizer": self.optimizer,
            "bound": self.bound,
            "evaluations": self.evaluations,
            "seed_score": self.seed_score,
            "best_score": self.best_score,
            "improved": (
                self.best_score > self.seed_score
                if self.best_score is not None and self.seed_score is not None
                else None
            ),
            "best_edits": self.candidates[0].get("edits") if self.candidates else None,
            "rejection_codes": self.rejection_codes,
            "accepted_op_mix": self.accepted_op_mix,
            "knob_attribution": self.knob_attribution,
            "candidates": len(self.candidates),
            "candidates_truncated": self.truncated,
        }


def read_run(path: Path, *, limit: int = DEFAULT_LIMIT) -> RunArchive:
    """Read one run directory. Raises when there is nothing readable there."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a run directory")
    _check_version(path)

    report_path = path / REPORT
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            report = None
        if isinstance(report, dict):
            return _from_report(path, report, limit=limit)
    candidates = _from_run_dirs(path, limit=limit)
    if not candidates:
        raise FileNotFoundError(f"{path} holds no report and no candidate runs")
    return RunArchive(path=path, candidates=tuple(candidates), truncated=len(candidates) >= limit)


def _check_version(path: Path) -> None:
    """Refuse a run written by a different layout, rather than half-read it.

    A directory with no ``run.json`` predates versioning, so its layout is unknown
    and reading it would be a guess.
    """
    version_file = path / RUN_FILE
    if not version_file.exists():
        if (path / REPORT).exists() or (path / "runs").is_dir():
            raise RunVersionError(
                f"{path} has no {RUN_FILE}, so it was written before runs were "
                "versioned and its layout cannot be assumed"
            )
        return
    try:
        payload = json.loads(version_file.read_text())
    except json.JSONDecodeError as exc:
        raise RunVersionError(f"{version_file} is not readable: {exc}") from exc
    version = payload.get("version") if isinstance(payload, dict) else None
    if version != RUN_VERSION:
        raise RunVersionError(
            f"{path} was written by layout version {version!r}, not {RUN_VERSION}"
        )


def _from_report(path: Path, report: dict[str, Any], *, limit: int) -> RunArchive:
    board = list(report.get("leaderboard") or ())
    return RunArchive(
        path=path,
        mode=str(report.get("mode", "")),
        optimizer=str(report.get("optimizer", "")),
        bound=str(report.get("bound", "")),
        evaluations=int(report.get("evaluations") or 0),
        seed_score=_number(report.get("seed_score")),
        # A validation-gated run's report gives its winner's val score as best_score
        # (best_score_basis); an optimizer is never shown a held-out number, so the
        # train score it reports beside it is carried forward instead.
        best_score=_number(report.get("best_train_score")
                           if str(report.get("best_score_basis") or "").startswith("val")
                           else report.get("best_score")),
        candidates=tuple(board[:limit]),
        rejection_codes=dict(report.get("rejection_codes") or {}),
        accepted_op_mix=dict(report.get("accepted_op_mix") or {}),
        knob_attribution=dict(report.get("knob_attribution") or {}),
        truncated=len(board) > limit,
    )


def _from_run_dirs(path: Path, *, limit: int) -> list[dict[str, Any]]:
    """Fall back to the per-candidate files a partial run still left behind."""
    found: list[dict[str, Any]] = []
    runs = path / "runs"
    if not runs.is_dir():
        return found
    for candidate_path in sorted(runs.glob("*/candidate.json")):
        try:
            payload = json.loads(candidate_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            found.append(payload)
    found.sort(key=lambda row: -(row.get("score") or 0.0))
    return found[:limit]


def find_runs(root: Path, *, limit: int = 3, exclude: Path | None = None) -> list[RunArchive]:
    """Readable runs under ``root``, newest first, excluding one directory.

    Sibling run directories, not nested ones: a run root is one run, and the
    previous runs a caller cares about are the ones beside it.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    exclude = Path(exclude).resolve() if exclude else None
    found: list[tuple[float, Path]] = []
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name == "runs":
            continue
        if exclude is not None and entry.resolve() == exclude:
            continue
        if not (entry / REPORT).exists() and not (entry / "runs").is_dir():
            continue
        found.append((entry.stat().st_mtime, entry))
    found.sort(reverse=True)

    archives: list[RunArchive] = []
    for _mtime, path in found[:limit]:
        try:
            archives.append(read_run(path))
        except (FileNotFoundError, RunVersionError):
            # A run we cannot read is not a prior we can use, and is not an error:
            # an older layout is expected to be sitting next to a newer one.
            continue
    return archives


def prior_summaries(root: Path, *, limit: int = 3, exclude: Path | None = None) -> list[dict[str, Any]]:
    """The bounded summaries an optimizer prompt carries."""
    return [archive.summary() for archive in find_runs(root, limit=limit, exclude=exclude)]


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
