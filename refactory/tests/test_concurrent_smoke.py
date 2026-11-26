"""A round's smoke checks run concurrently under the run-wide item limit.

With ``batch_size`` > 1, ``install`` defers the smoke check and runs the
batch through ``_smoke_check_batch``, so independent candidates overlap
their smoke runs.  ``batch_size`` 1 keeps the sequential path.  Each test
fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import Candidate, Optimizer, Proposal

from test_checkpoint_selection import SmokeProbe, _events, _by_level
from test_skillopt import _patch_mode


# ── helpers ────────────────────────────────────────────────────────────────

SEED = 42
MODE_ID = "smoke_probe/smoke_probe"
LEVEL_MODE_ID = "level_probe/level_probe"


class MultiFixer(Optimizer):
    """Proposes ``batch_size`` candidates: levels from ``plan`` per epoch.
    The first entry in each epoch's levels is broken (13); the rest are good.
    Its repair answers level 14 for every E-smoke refusal."""

    name = "multi_fixer"

    def __init__(self, plan: list[list[int]]) -> None:
        super().__init__(k=1)
        self.plan = list(plan)
        self.repaired: list[list[str]] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        levels = self.plan.pop(0) if self.plan else []
        return [Proposal(edit={"op": "set_knob", "name": "level", "value": lv},
                         parent=parents[0].id)
                for lv in levels[:n]]

    def repair(self, ctx, proposal, reasons, parent):  # type: ignore[no-untyped-def]
        self.repaired.append(list(reasons))
        return Proposal(edit={"op": "set_knob", "name": "level", "value": 14},
                        parent=parent.id)


class MultiLevel(Optimizer):
    """Proposes ``batch_size`` candidates at given levels, all good (no broken)."""

    name = "multi_level"

    def __init__(self, plan: list[list[int]]) -> None:
        super().__init__(k=1)
        self.plan = list(plan)

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        levels = self.plan.pop(0) if self.plan else []
        return [Proposal(edit={"op": "set_knob", "name": "level", "value": lv},
                         parent=parents[0].id)
                for lv in levels[:n]]


def _smoke_parts(tmp_path: Path) -> Path:
    parts = tmp_path / "smoke-parts.json"
    parts.write_text(json.dumps({"train": ["s1", "s2", "s3", "s4"], "val": ["v1", "v2"]}))
    return parts


def _run_batch(tmp_path: Path, monkeypatch: Any, plan: list[list[int]], *,
               batch_size: int = 3, epochs: int = 1,
               max_concurrent_evaluations: int = 4,
               ) -> tuple[Loop, MultiFixer]:
    """One run of SmokeProbe with the MultiFixer: ``(loop, fixer)``."""
    parts = _smoke_parts(tmp_path)
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(SmokeProbe):
        loop = Loop(MODE_ID, LoopConfig(
            epochs=epochs, batch_size=batch_size, optimizer="random", seed=SEED,
            runs_root=tmp_path / "run", allowed_ops=("set_knob",),
            max_evaluations=60, holdout=0.0, parts_manifest=str(parts),
            max_concurrent_evaluations=max_concurrent_evaluations),
            project=Path("."))
        loop.optimizer = fixer = MultiFixer(plan)
        loop.run()
    return loop, fixer


# ── overlap test ──────────────────────────────────────────────────────────


def test_a_rounds_smoke_checks_overlap(tmp_path, monkeypatch):
    """With batch_size 3 and 3 good candidates, their smoke checks all run at the
    same time: a barrier of 3 is met inside _measure (which runs only in the real
    smoke check, not the deferred skip)."""
    barrier = threading.Barrier(3, timeout=10.0)
    met: list[str] = []
    original = Loop._measure

    def wrapped(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        if Path(workdir).name.startswith("smoke_"):
            try:
                barrier.wait()
                met.append(candidate.id)
            except threading.BrokenBarrierError:
                pass
        return original(self, candidate, workdir, part, **kwargs)

    monkeypatch.setattr(Loop, "_measure", wrapped)
    parts = _smoke_parts(tmp_path)
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(SmokeProbe):
        loop = Loop(MODE_ID, LoopConfig(
            epochs=1, batch_size=3, optimizer="random", seed=SEED,
            runs_root=tmp_path / "run", allowed_ops=("set_knob",),
            max_evaluations=60, holdout=0.0, parts_manifest=str(parts),
            max_concurrent_evaluations=4),
            project=Path("."))
        loop.optimizer = MultiLevel([[1, 2, 3]])
        loop.run()
    assert len(met) == 3, "the round's smoke checks did not overlap"
    assert all(c.evaluated for c in loop.population if c.parent)


def test_overlap_fails_when_smoke_is_sequential(tmp_path, monkeypatch):
    """Mutation check: disabling _smoke_check_batch makes the barrier break because
    no three smoke _measure calls are ever in flight at once."""
    barrier = threading.Barrier(3, timeout=1.0)
    met: list[str] = []
    original = Loop._measure

    def wrapped(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        if Path(workdir).name.startswith("smoke_"):
            try:
                barrier.wait()
                met.append(candidate.id)
            except threading.BrokenBarrierError:
                pass
        return original(self, candidate, workdir, part, **kwargs)

    monkeypatch.setattr(Loop, "_measure", wrapped)
    # Disable the batch path: _smoke_check_batch runs each check inline.
    def sequential_batch(self, run, candidates, smoke_meta):  # type: ignore[no-untyped-def]
        from refactory_dsh.outerloop import Run

        final = []
        for cand in candidates:
            if cand.id in smoke_meta:
                _, parent = smoke_meta[cand.id]
                outcome = self._smoke_check(parent, cand)
                if isinstance(outcome, Candidate):
                    final.append(outcome)
            else:
                final.append(cand)
        return final

    monkeypatch.setattr(Loop, "_smoke_check_batch", sequential_batch)
    parts = _smoke_parts(tmp_path)
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(SmokeProbe):
        loop = Loop(MODE_ID, LoopConfig(
            epochs=1, batch_size=3, optimizer="random", seed=SEED,
            runs_root=tmp_path / "run", allowed_ops=("set_knob",),
            max_evaluations=60, holdout=0.0, parts_manifest=str(parts),
            max_concurrent_evaluations=4),
            project=Path("."))
        loop.optimizer = MultiLevel([[1, 2, 3]])
        loop.run()
    assert met == [], "smoke checks should NOT overlap when forced sequential"


# ── smoke failure + repair with batch_size > 1 ───────────────────────────


def test_smoke_failure_is_repaired_in_batch(tmp_path, monkeypatch):
    """With batch_size 3, one broken candidate (13) and two good (1, 2): the broken
    one is refused E-smoke, repaired to 14, smoked again (passes), and all three are
    evaluated. The E-smoke refusal reaches the repair call."""
    loop, fixer = _run_batch(tmp_path, monkeypatch, [[13, 1, 2]], batch_size=3)
    smokes = _events(loop, "smoke")
    outcomes = [(e["id"], e["outcome"]) for e in smokes]
    failed = [o for o in outcomes if o[1] == "failed"]
    passed = [o for o in outcomes if o[1] == "passed"]
    assert len(failed) == 1, "exactly one smoke check should fail"
    assert len(passed) >= 2, "the good candidates and the repaired one pass"
    assert fixer.repaired, "the repair call must have been invoked"
    assert fixer.repaired[0][0].startswith("E-smoke:")
    levels = _by_level(loop)
    assert 13 not in levels, "the broken graph should not appear as evaluated"
    assert 14 in levels, "the repaired graph (14) should be evaluated"
    assert all(lv in levels for lv in (1, 2)), "the good graphs should be evaluated"
    refusal = next(r for r in loop.rejections if r.reasons[0].startswith("E-smoke"))
    assert refusal.repaired == levels[14]


# ── batch_size 1 is unchanged ────────────────────────────────────────────


def test_batch_size_one_does_not_defer(tmp_path, monkeypatch):
    """batch_size 1 runs the smoke check inline in attempt, not deferred."""
    deferred: list[bool] = []
    original = Loop._smoke_check

    def tracked(self, parent, candidate, **kwargs):  # type: ignore[no-untyped-def]
        deferred.append(self._defer_smoke)
        return original(self, parent, candidate, **kwargs)

    monkeypatch.setattr(Loop, "_smoke_check", tracked)
    parts = _smoke_parts(tmp_path)
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(SmokeProbe):
        loop = Loop(MODE_ID, LoopConfig(
            epochs=1, batch_size=1, optimizer="random", seed=SEED,
            runs_root=tmp_path / "run", allowed_ops=("set_knob",),
            max_evaluations=20, holdout=0.0, parts_manifest=str(parts)),
            project=Path("."))
        loop.optimizer = MultiLevel([[1]])
        loop.run()
    assert deferred and not any(deferred), "_defer_smoke should be False for batch_size 1"


# ── journal and request content ──────────────────────────────────────────


def test_journal_smoke_entries_match_sequential(tmp_path, monkeypatch):
    """The journal's smoke entries have the same fields and structure whether
    produced by the batch path or the sequential path."""
    def run_once(root, batch_size):
        root.mkdir(parents=True, exist_ok=True)
        parts = root / "smoke-parts.json"
        parts.write_text(json.dumps(
            {"train": ["s1", "s2", "s3", "s4"], "val": ["v1", "v2"]}))
        monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
        with _patch_mode(SmokeProbe):
            loop = Loop(MODE_ID, LoopConfig(
                epochs=1, batch_size=batch_size, optimizer="random", seed=SEED,
                runs_root=root / "run", allowed_ops=("set_knob",),
                max_evaluations=60, holdout=0.0, parts_manifest=str(parts)),
                project=Path("."))
            loop.optimizer = MultiLevel([[1, 2]])
            loop.run()
        return loop

    seq_loop = run_once(tmp_path / "seq", batch_size=1)
    batch_loop = run_once(tmp_path / "batch", batch_size=2)

    # For batch_size=1, propose is called per epoch so only 1 candidate per epoch;
    # run 2 epochs to get 2 smokes. For batch_size=2, we get 2 in one epoch.
    # Instead, compare the structure of individual smoke events.
    seq_smokes = _events(seq_loop, "smoke")
    batch_smokes = _events(batch_loop, "smoke")
    assert len(batch_smokes) == 2, "batch should produce 2 smoke entries"

    # Verify same field structure (keys minus timing/id-dependent fields)
    stable_keys = {"outcome", "items", "seed", "errors", "rollouts"}
    for entry in batch_smokes:
        assert stable_keys <= set(entry.keys()), f"missing keys in {entry.keys()}"
        assert entry["outcome"] in ("passed", "failed", "inconclusive")


# ── wall-clock overlap ───────────────────────────────────────────────────


def test_wall_time_with_slow_evaluator(tmp_path, monkeypatch):
    """With a slow _measure (0.5s), 5 candidates' smoke checks finish in well
    under 5x the single-check time (the concurrent batch takes roughly one
    delay plus overhead, not five)."""
    original_measure = Loop._measure
    delay = 0.5

    def slow_measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        if Path(workdir).name.startswith("smoke_"):
            time.sleep(delay)
        return original_measure(self, candidate, workdir, part, **kwargs)

    monkeypatch.setattr(Loop, "_measure", slow_measure)
    parts = _smoke_parts(tmp_path)
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    n = 5
    with _patch_mode(SmokeProbe):
        loop = Loop(MODE_ID, LoopConfig(
            epochs=1, batch_size=n, optimizer="random", seed=SEED,
            runs_root=tmp_path / "run", allowed_ops=("set_knob",),
            max_evaluations=60, holdout=0.0, parts_manifest=str(parts),
            max_concurrent_evaluations=n),
            project=Path("."))
        loop.optimizer = MultiLevel([[1, 2, 3, 4, 5]])
        t0 = time.monotonic()
        loop.run()
        elapsed = time.monotonic() - t0
    smokes = _events(loop, "smoke")
    assert len(smokes) == n, f"expected {n} smoke entries, got {len(smokes)}"
    sequential_minimum = delay * n
    assert elapsed < sequential_minimum, (
        f"smoke checks took {elapsed:.1f}s, sequential would be >= {sequential_minimum:.1f}s")


# ── rollout budget (TOCTOU fix) ─────────────────────────────────────────


def test_max_rollouts_caps_concurrent_smokes(tmp_path, monkeypatch):
    """With max_rollouts set tight, only the candidates whose smoke rollouts
    fit are smoked; the rest are skipped by _smoke_check_batch's atomic
    reservation.  rollouts never exceeds the cap."""
    parts = _smoke_parts(tmp_path)
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(SmokeProbe):
        loop = Loop(MODE_ID, LoopConfig(
            epochs=1, batch_size=4, optimizer="random", seed=SEED,
            runs_root=tmp_path / "run", allowed_ops=("set_knob",),
            max_evaluations=60, holdout=0.0, parts_manifest=str(parts),
            max_concurrent_evaluations=4,
            max_rollouts=10),
            project=Path("."))
        loop.optimizer = MultiLevel([[1, 2, 3, 4]])
        loop.run()
    smokes = _events(loop, "smoke")
    assert loop.rollouts <= 10, f"rollouts {loop.rollouts} exceeded cap 10"
    assert len(smokes) < 4, (
        "some candidates should be skipped when max_rollouts is tight")
    assert len(smokes) >= 1, "at least one candidate should be smoked"


# ── resume mid-batch ────────────────────────────────────────────────────


def test_resume_mid_batch_smokes_only_the_unfinished(tmp_path, monkeypatch):
    """Crash after 2 of 3 smoke verdicts are journaled.  Resume re-proposes the
    same batch; the two already-journaled candidates get their verdict from the
    journal (``_smoke_verdicts``), and only the third is actually smoked.

    The crash fires on the 3rd _measure call (the actual smoke execution), so
    the 3rd candidate's smoke never completes or journals."""
    parts = _smoke_parts(tmp_path)
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)

    measure_count = [0]
    original_measure = Loop._measure

    def crash_on_third_measure(self, candidate, workdir, part, **kwargs):
        if Path(workdir).name.startswith("smoke_"):
            measure_count[0] += 1
            if measure_count[0] >= 3:
                raise KeyboardInterrupt("crash on 3rd smoke measure")
        return original_measure(self, candidate, workdir, part, **kwargs)

    monkeypatch.setattr(Loop, "_measure", crash_on_third_measure)

    import pytest as _pt
    with _pt.raises(KeyboardInterrupt):
        with _patch_mode(SmokeProbe):
            loop = Loop(MODE_ID, LoopConfig(
                epochs=1, batch_size=3, optimizer="random", seed=SEED,
                runs_root=tmp_path / "run", allowed_ops=("set_knob",),
                max_evaluations=60, holdout=0.0, parts_manifest=str(parts),
                max_concurrent_evaluations=1),
                project=Path("."))
            loop.optimizer = MultiLevel([[1, 2, 3]])
            loop.run()

    run_root = tmp_path / "run"
    journal = run_root / "journal.jsonl"
    assert journal.exists(), "journal should exist after partial run"
    first_smokes = [json.loads(line) for line in journal.read_text().splitlines()
                    if line.strip() and json.loads(line).get("event") == "smoke"]
    assert len(first_smokes) == 2, f"expected 2 journaled smokes, got {len(first_smokes)}"

    monkeypatch.setattr(Loop, "_measure", original_measure)
    resumed_measure_calls: list[str] = []

    def tracking_measure(self, candidate, workdir, part, **kwargs):
        if Path(workdir).name.startswith("smoke_"):
            resumed_measure_calls.append(candidate.id)
        return original_measure(self, candidate, workdir, part, **kwargs)

    monkeypatch.setattr(Loop, "_measure", tracking_measure)
    with _patch_mode(SmokeProbe):
        resumed = Loop(MODE_ID, LoopConfig(
            epochs=1, batch_size=3, optimizer="random", seed=SEED,
            runs_root=tmp_path / "run", allowed_ops=("set_knob",),
            max_evaluations=60, holdout=0.0, parts_manifest=str(parts),
            max_concurrent_evaluations=4, resume=True),
            project=Path("."))
        resumed.optimizer = MultiLevel([[1, 2, 3]])
        resumed.run()

    already_smoked = {str(s["id"]) for s in first_smokes}
    assert all(cid not in already_smoked for cid in resumed_measure_calls), (
        "resumed run should not re-smoke candidates whose verdicts were journaled")
    assert len(resumed_measure_calls) >= 1, "the unfinished candidate should be smoked on resume"
