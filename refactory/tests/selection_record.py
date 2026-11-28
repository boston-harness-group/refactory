"""Deterministic harness for recording selection decisions.

Drives REAL Loop objects through scripted scenarios.  Candidates are
created by the Loop's own evaluation path: ScenarioProbe returns scripted
train scores by level, and ``ValidationGate.val_score`` is stubbed to
return scripted val scores by level.  Every downstream decision — gate,
pool, checkpoint, incumbent, leaderboard, parent picking, early stop — is
the Loop's own code.

Usage::

    # generate the record
    SELECTION_RECORD_REGEN=1 uv run python -m pytest tests/test_selection_record.py::test_selection_record -x -q

    # check against the record
    uv run python -m pytest tests/test_selection_record.py -x -q

The record lives at ``tests/fixtures/selection_record.json``.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.nn.mode import RunResult
from refactory_dsh.optim import Optimizer, Proposal
from refactory_dsh.outerloop import ValidationGate

from test_skillopt import MODE_ID, TRAIN_IDS, VAL_IDS, _patch_mode, _write_parts
from test_validation_selection import FitProbe, _level

FIXTURE = Path(__file__).parent / "fixtures" / "selection_record.json"

# ---------------------------------------------------------------------------
# ScenarioProbe: FitProbe that can simulate train failures
# ---------------------------------------------------------------------------


class ScenarioProbe(FitProbe):
    """FitProbe extended to raise on specific levels (simulating a failed
    train run so the candidate gets ``exit_ok=False``)."""

    name = "scenario_probe"
    FAIL_LEVELS: set[int] = set()

    def run(self, workdir, **kwargs):  # type: ignore[no-untyped-def]
        from pathlib import Path

        level = int(self.values()["level"])
        if level in ScenarioProbe.FAIL_LEVELS:
            return RunResult(exit_ok=False, error="simulated train failure",
                             workdir=Path(workdir))
        score = float(ScenarioProbe.TRAIN.get(level, 0.0))
        subset = Path(workdir) / "subset.json"
        ids = VAL_IDS if subset.exists() and "val" in subset.read_text() else TRAIN_IDS
        records = [{"id": iid, "score": score} for iid in ids]
        return RunResult(
            exit_ok=True, workdir=Path(workdir), score=score,
            item_records=records,
            facts={"item_scores": records},
        )

    def score(self, run):  # type: ignore[no-untyped-def]
        level = int(self.values()["level"])
        if level in ScenarioProbe.FAIL_LEVELS:
            raise RuntimeError("simulated train failure")
        return float(ScenarioProbe.TRAIN.get(level, 0.0))


# ---------------------------------------------------------------------------
# ScenarioOptimizer: proposes children with scripted levels
# ---------------------------------------------------------------------------


class ScenarioOptimizer(Optimizer):
    """Each epoch, proposes children with the levels from the plan."""

    name = "scenario"

    def __init__(self, plan: list[list[int]]) -> None:
        super().__init__(k=1)
        self._plan = list(plan)
        self._pending: list[int] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        if not self._pending and self._plan:
            self._pending = list(self._plan.pop(0))
        children = self._pending[:n]
        self._pending = self._pending[n:]
        parent = parents[0] if parents else None
        return [
            Proposal(
                edit={"op": "set_knob", "name": "level", "value": child},
                parent=parent.id if parent else None,
            )
            for child in children
        ]


# ---------------------------------------------------------------------------
# Scenario specifications
# ---------------------------------------------------------------------------


@dataclass
class RoundSpec:
    """One round's candidates to inject."""

    candidates: list[dict[str, Any]]


@dataclass
class ScenarioSpec:
    label: str
    rounds: list[RoundSpec]
    early_stop_patience: int = 0
    early_stop_at_full_val: bool = False


@dataclass
class DecisionRecord:
    """What happened in one round."""

    round_idx: int
    gate_accepted: list[str]
    gate_rejected: list[str]
    pool: list[str]
    incumbent: str | None
    leaderboard: list[str]
    parents_chosen: list[str]
    picker_name: str
    early_stop: str
    val_evaluations: int


@dataclass
class ScenarioResult:
    decisions: list[DecisionRecord]
    final_winner: str | None
    final_winner_val: float | None
    total_val_evaluations: int


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _monotone_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        label="monotone",
        rounds=[
            RoundSpec(candidates=[{"label": "seed", "train_score": 0.3, "val_score": 0.3}]),
            RoundSpec(candidates=[{"label": "c1", "parent_label": "seed", "train_score": 0.4, "val_score": 0.4}]),
            RoundSpec(candidates=[{"label": "c2", "parent_label": "c1", "train_score": 0.5, "val_score": 0.5}]),
            RoundSpec(candidates=[{"label": "c3", "parent_label": "c2", "train_score": 0.6, "val_score": 0.6}]),
        ],
    )


def _noisy_flat_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        label="noisy_flat",
        rounds=[
            RoundSpec(candidates=[{"label": "seed", "train_score": 0.5, "val_score": 0.5}]),
            RoundSpec(candidates=[{"label": "n1", "parent_label": "seed", "train_score": 0.51, "val_score": 0.49}]),
            RoundSpec(candidates=[{"label": "n2", "parent_label": "seed", "train_score": 0.49, "val_score": 0.51}]),
            RoundSpec(candidates=[{"label": "n3", "parent_label": "seed", "train_score": 0.50, "val_score": 0.50}]),
        ],
    )


def _val_drop_scenario() -> ScenarioSpec:
    """Train gain but val drop: checkpoint keeps building on worse candidate."""
    return ScenarioSpec(
        label="val_drop",
        rounds=[
            RoundSpec(candidates=[{"label": "seed", "train_score": 0.4, "val_score": 0.5}]),
            RoundSpec(candidates=[{"label": "v1", "parent_label": "seed", "train_score": 0.6, "val_score": 0.3}]),
            RoundSpec(candidates=[{"label": "v2", "parent_label": "v1", "train_score": 0.7, "val_score": 0.2}]),
        ],
    )


def _ties_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        label="ties",
        rounds=[
            RoundSpec(candidates=[{"label": "seed", "train_score": 0.5, "val_score": 0.5}]),
            RoundSpec(candidates=[
                {"label": "t1", "parent_label": "seed", "train_score": 0.5, "val_score": 0.5},
                {"label": "t2", "parent_label": "seed", "train_score": 0.6, "val_score": 0.5},
                {"label": "t3", "parent_label": "seed", "train_score": 0.4, "val_score": 0.5},
            ]),
        ],
    )


def _failed_train_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        label="failed_train",
        rounds=[
            RoundSpec(candidates=[{"label": "seed", "train_score": 0.5, "val_score": 0.5}]),
            RoundSpec(candidates=[
                {"label": "f1", "parent_label": "seed", "train_score": None, "exit_ok": False, "val_score": 0.6},
            ]),
            RoundSpec(candidates=[
                {"label": "f2", "parent_label": "seed", "train_score": 0.3, "val_score": 0.7},
            ]),
        ],
    )


def _structural_mix_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        label="structural_mix",
        rounds=[
            RoundSpec(candidates=[{"label": "seed", "train_score": 0.4, "val_score": 0.4}]),
            RoundSpec(candidates=[
                {"label": "s1", "parent_label": "seed", "train_score": 0.5, "val_score": 0.5},
                {"label": "s2", "parent_label": "seed", "train_score": 0.45, "val_score": 0.45},
            ]),
            RoundSpec(candidates=[
                {"label": "s3", "parent_label": "s1", "train_score": 0.55, "val_score": 0.55},
            ]),
        ],
    )


def _early_stop_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        label="early_stop",
        rounds=[
            RoundSpec(candidates=[{"label": "seed", "train_score": 0.5, "val_score": 0.5}]),
            RoundSpec(candidates=[{"label": "e1", "parent_label": "seed", "train_score": 0.6, "val_score": 0.6}]),
            RoundSpec(candidates=[{"label": "e2", "parent_label": "e1", "train_score": 0.55, "val_score": 0.55}]),
            RoundSpec(candidates=[{"label": "e3", "parent_label": "e2", "train_score": 0.54, "val_score": 0.54}]),
            RoundSpec(candidates=[{"label": "e4", "parent_label": "e3", "train_score": 0.53, "val_score": 0.53}]),
        ],
        early_stop_patience=2,
    )


def _perfect_val_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        label="perfect_val",
        rounds=[
            RoundSpec(candidates=[{"label": "seed", "train_score": 0.9, "val_score": 0.9}]),
            RoundSpec(candidates=[{"label": "p1", "parent_label": "seed", "train_score": 1.0, "val_score": 1.0}]),
            RoundSpec(candidates=[{"label": "p2", "parent_label": "p1", "train_score": 0.95, "val_score": 0.95}]),
        ],
        early_stop_at_full_val=True,
    )


SYNTHETIC_SCENARIOS = [
    _monotone_scenario,
    _noisy_flat_scenario,
    _val_drop_scenario,
    _ties_scenario,
    _failed_train_scenario,
    _structural_mix_scenario,
    _early_stop_scenario,
    _perfect_val_scenario,
]

SELECTION_MODES = ["checkpoint", "single", "pool"]

PICKERS = ["", "map_elites", "islands", "archive", "evolutionary", "hill_climb", "pareto"]

BATCH_SIZES = [1, 3, 5]


def _is_compatible(mode: str, picker: str, batch_size: int) -> bool:
    """Whether a combination is valid."""
    if mode == "checkpoint" and picker:
        return False
    if mode == "checkpoint" and batch_size > 1:
        return False
    return True


# ---------------------------------------------------------------------------
# Real-run scenario extraction
# ---------------------------------------------------------------------------

_REAL_RUNS_DIR = Path(
    "/tmp/claude-1000/-home-churwitz-remote-factory/"
    "c6f7e470-53c1-40aa-8dd8-e7bcb5136076/scratchpad/r5/runs"
)


def _extract_real_scenario(run_name: str) -> ScenarioSpec | None:
    """Extract a scenario from a real run's journal."""
    run_dir = _REAL_RUNS_DIR / run_name / "run"
    journal = run_dir / "journal.jsonl"
    if not journal.exists():
        return None

    entries: list[dict[str, Any]] = []
    with open(journal) as f:
        for line in f:
            entries.append(json.loads(line))

    state_file = run_dir / "state.json"
    if not state_file.exists():
        return None
    state = json.loads(state_file.read_text())
    pop = state.get("population", [])
    if not pop:
        return None

    val_cache: dict[str, float] = {}
    for entry in reversed(entries):
        if entry.get("event") == "strategy_state":
            gate_state = (entry.get("state") or {}).get("val_gate") or {}
            val_cache = {str(k): float(v) for k, v in (gate_state.get("val_cache") or {}).items()}
            break

    seed = pop[0]
    seed_round = RoundSpec(candidates=[{
        "label": seed["id"],
        "train_score": seed.get("score"),
        "val_score": val_cache.get(seed["id"]),
        "exit_ok": seed.get("exit_ok", True),
    }])

    proposal_epochs: dict[int, list[str]] = {}
    for entry in entries:
        if entry.get("event") == "proposal":
            cid = entry.get("candidate")
            ep = entry.get("epoch")
            if cid and ep is not None:
                proposal_epochs.setdefault(ep, []).append(cid)

    rounds = [seed_round]
    seen_ids = {seed["id"]}
    for ep in sorted(proposal_epochs):
        epoch_cands = []
        for cid in proposal_epochs[ep]:
            cand = next((c for c in pop if c["id"] == cid), None)
            if cand and cid not in seen_ids:
                seen_ids.add(cid)
                parent_id = cand.get("parent")
                epoch_cands.append({
                    "label": cid,
                    "parent_label": parent_id or seed["id"],
                    "train_score": cand.get("score"),
                    "val_score": val_cache.get(cid),
                    "exit_ok": cand.get("exit_ok", True),
                })
        if epoch_cands:
            rounds.append(RoundSpec(candidates=epoch_cands))

    for cand in pop:
        if cand["id"] not in seen_ids:
            seen_ids.add(cand["id"])
            rounds.append(RoundSpec(candidates=[{
                "label": cand["id"],
                "parent_label": cand.get("parent") or seed["id"],
                "train_score": cand.get("score"),
                "val_score": val_cache.get(cand["id"]),
                "exit_ok": cand.get("exit_ok", True),
            }]))

    return ScenarioSpec(label=f"real_{run_name}", rounds=rounds)


def _real_scenarios() -> list[ScenarioSpec]:
    """Load real run scenarios (best-effort)."""
    result = []
    for name in ("real3-OL-s42", "real3-OLME-s42"):
        s = _extract_real_scenario(name)
        if s is not None:
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Mapping scenarios to FitProbe levels
# ---------------------------------------------------------------------------


def _map_scenario(
    scenario: ScenarioSpec,
) -> tuple[dict[int, float], dict[int, float], set[int], list[list[int]]]:
    """Map scenario labels to FitProbe integer levels.

    Returns (train_map, val_map, fail_levels, plan) where plan is a list of
    lists of child levels, one per epoch.
    """
    train_map: dict[int, float] = {}
    val_map: dict[int, float] = {}
    fail_levels: set[int] = set()
    plan: list[list[int]] = []

    level = 0
    for rnd in scenario.rounds:
        for cspec in rnd.candidates:
            ts = cspec.get("train_score")
            if ts is not None:
                train_map[level] = ts
            elif not cspec.get("exit_ok", True):
                fail_levels.add(level)
            vs = cspec.get("val_score")
            if vs is not None:
                val_map[level] = vs
            level += 1

    for rnd in scenario.rounds[1:]:
        plan.append([
            sum(len(r.candidates) for r in scenario.rounds[:i])
            + j
            for i, r in enumerate(scenario.rounds)
            if r is rnd
            for j in range(len(rnd.candidates))
        ])

    return train_map, val_map, fail_levels, plan


def _map_scenario_simple(
    scenario: ScenarioSpec,
) -> tuple[dict[int, float], dict[int, float], set[int], list[list[int]]]:
    """Map scenario labels to FitProbe integer levels.

    Returns (train_map, val_map, fail_levels, plan).
    Level assignment: seed round's candidates get levels 0..k-1,
    round 1's candidates get k..k+m-1, etc.
    Plan[i] = levels of round i+1's candidates.
    """
    train_map: dict[int, float] = {}
    val_map: dict[int, float] = {}
    fail_levels: set[int] = set()
    plan: list[list[int]] = []

    level = 0
    for rnd in scenario.rounds:
        round_levels: list[int] = []
        for cspec in rnd.candidates:
            ts = cspec.get("train_score")
            if ts is not None:
                train_map[level] = ts
            elif not cspec.get("exit_ok", True):
                fail_levels.add(level)
            vs = cspec.get("val_score")
            if vs is not None:
                val_map[level] = vs
            round_levels.append(level)
            level += 1
        plan.append(round_levels)

    # plan[0] is the seed (not proposed); plan[1:] are proposal rounds
    proposal_plan = plan[1:]
    return train_map, val_map, fail_levels, proposal_plan


# ---------------------------------------------------------------------------
# Patching context
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _patched_scores(
    train_map: dict[int, float],
    val_map: dict[int, float],
    fail_levels: set[int],
):  # type: ignore[no-untyped-def]
    """Patch ScenarioProbe.TRAIN, .FAIL_LEVELS, ValidationGate.val_score,
    Loop._seed_population, and expensive I/O helpers for the duration of a
    scenario run."""
    saved = {
        "TRAIN": dict(getattr(ScenarioProbe, "TRAIN", {})),
        "FAIL": set(getattr(ScenarioProbe, "FAIL_LEVELS", set())),
        "val_score": ValidationGate.val_score,
        "seed_pop": Loop._seed_population,
        "journal": Loop._journal,
        "traces": Loop._write_traces,
        "progress": Loop._progress,
        "write_state": Loop._write_state,
        "write_cand": Loop._write_candidate,
        "save_strat": Loop._save_strategy_state,
        "measure": Loop._measure,
    }

    def _fast_measure(  # type: ignore[no-untyped-def]
        self, candidate, workdir, part, *, seed=None, items=None,
        gold_feedback=False, item_gate=None, item_slots=False,
        stop_event=None, repeats=None, traces=True,
    ):
        workdir.mkdir(parents=True, exist_ok=True)
        from refactory_dsh import task as _t

        _t.write_subset(workdir, part, self.config.holdout, seed=seed,
                        gold_feedback=gold_feedback)
        if self._parts_manifest is not None:
            manifest_src = self.root / _t.MANIFEST
            if manifest_src.exists():
                import shutil as _sh

                _sh.copyfile(manifest_src, workdir / _t.MANIFEST)
        mode = self.mode_class(state=candidate.state)
        result = mode.evaluate(workdir)
        with self._state_lock:
            self.measurements += 1
        return result, {"count": 1, "complete": True}

    ScenarioProbe.TRAIN = dict(train_map)
    ScenarioProbe.FAIL_LEVELS = set(fail_levels)
    ValidationGate.val_score = staticmethod(  # type: ignore[assignment]
        lambda run, cand: float(val_map.get(_level(cand), 0.0))
    )
    Loop._seed_population = lambda self, seed: None  # type: ignore[assignment]
    Loop._journal = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._write_traces = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._progress = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._write_state = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._write_candidate = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._save_strategy_state = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._measure = _fast_measure  # type: ignore[assignment]

    try:
        yield
    finally:
        ScenarioProbe.TRAIN = saved["TRAIN"]
        ScenarioProbe.FAIL_LEVELS = saved["FAIL"]
        ValidationGate.val_score = saved["val_score"]  # type: ignore[assignment]
        Loop._seed_population = saved["seed_pop"]  # type: ignore[assignment]
        Loop._journal = saved["journal"]  # type: ignore[assignment]
        Loop._write_traces = saved["traces"]  # type: ignore[assignment]
        Loop._progress = saved["progress"]  # type: ignore[assignment]
        Loop._write_state = saved["write_state"]  # type: ignore[assignment]
        Loop._write_candidate = saved["write_cand"]  # type: ignore[assignment]
        Loop._save_strategy_state = saved["save_strat"]  # type: ignore[assignment]
        Loop._measure = saved["measure"]  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Run a scenario through the REAL Loop
# ---------------------------------------------------------------------------


def _run_scenario(
    scenario: ScenarioSpec,
    selection_mode: str,
    picker_name: str,
    batch_size: int,
    seed: int = 42,
    *,
    work_dir: Path | None = None,
) -> ScenarioResult:
    """Drive the real Loop through ``scenario`` and capture decisions."""

    train_map, val_map, fail_levels, plan = _map_scenario_simple(scenario)

    tmp_dir = work_dir or Path(tempfile.mkdtemp(prefix="selrec_"))
    own_tmp = work_dir is None
    try:
        parts_path = _write_parts(tmp_dir)

        with _patched_scores(train_map, val_map, fail_levels):
            with _patch_mode(ScenarioProbe):
                n_epochs = max(1, len(scenario.rounds) - 1)
                config = LoopConfig(
                    epochs=n_epochs,
                    batch_size=batch_size,
                    optimizer="random",
                    seed=seed,
                    runs_root=tmp_dir / "run",
                    allowed_ops=("set_knob",),
                    max_evaluations=500,
                    holdout=0.0,
                    selection=selection_mode,
                    parent_picker=picker_name,
                    parts_manifest=str(parts_path),
                    early_stop_patience=scenario.early_stop_patience,
                    early_stop_at_full_val=scenario.early_stop_at_full_val,
                )

                loop = Loop(MODE_ID, config, project=Path("."))

                # Replace the proposer, preserving the PickedParent wrapper
                proposer = ScenarioOptimizer(list(plan))
                if picker_name and hasattr(loop.optimizer, "proposer"):
                    loop.optimizer.proposer = proposer  # type: ignore[union-attr]
                else:
                    loop.optimizer = proposer

                # Hook _gate_population to capture per-round decisions
                snapshots: list[dict[str, Any]] = []
                _original_gate_pop = Loop._gate_population

                def _capturing_gate_pop(run):  # type: ignore[no-untyped-def]
                    cache_before = len(loop._gate().cache)
                    acc_before = len(loop._gate_accepted)
                    rej_before = len(loop._gate_rejected)

                    _original_gate_pop(loop, run)

                    new = (len(loop._gate_accepted) - acc_before
                           + len(loop._gate_rejected) - rej_before)
                    if new == 0:
                        return

                    loop._count_stale_epoch()
                    gate = loop._gate()
                    early = loop._early_stop_reason() if loop.config.val_gate else ""

                    snapshots.append({
                        "gate_accepted": list(loop._gate_accepted),
                        "gate_rejected": list(loop._gate_rejected),
                        "incumbent": gate.current_id,
                        "best_id": gate.best_id,
                        "best_score": gate.best_score,
                        "leaderboard": [c.id for c in loop.leaderboard()],
                        "pool": ([c.id for c in loop._pool()]
                                 if loop.selection_mode() == "pool" else []),
                        "early_stop": early,
                        "val_evaluations": len(gate.cache) - cache_before,
                        "parents": [c.id for c in loop.select_parents(batch_size)],
                    })

                loop._gate_population = _capturing_gate_pop  # type: ignore[assignment]

                loop.run()

        # Build the result from snapshots
        gate = loop._gate()
        total_val = sum(s["val_evaluations"] for s in snapshots)

        decisions = [
            DecisionRecord(
                round_idx=i,
                gate_accepted=s["gate_accepted"],
                gate_rejected=s["gate_rejected"],
                pool=s["pool"],
                incumbent=s["incumbent"],
                leaderboard=s["leaderboard"],
                parents_chosen=s["parents"],
                picker_name=picker_name or "default",
                early_stop=s["early_stop"],
                val_evaluations=s["val_evaluations"],
            )
            for i, s in enumerate(snapshots)
        ]

        return ScenarioResult(
            decisions=decisions,
            final_winner=gate.best_id,
            final_winner_val=gate.best_score if gate.best_id else None,
            total_val_evaluations=total_val,
        )
    finally:
        if own_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------


def _result_to_dict(r: ScenarioResult) -> dict[str, Any]:
    return {
        "decisions": [
            {
                "round_idx": d.round_idx,
                "gate_accepted": d.gate_accepted,
                "gate_rejected": d.gate_rejected,
                "pool": d.pool,
                "incumbent": d.incumbent,
                "leaderboard": d.leaderboard,
                "parents_chosen": d.parents_chosen,
                "picker_name": d.picker_name,
                "early_stop": d.early_stop,
                "val_evaluations": d.val_evaluations,
            }
            for d in r.decisions
        ],
        "final_winner": r.final_winner,
        "final_winner_val": r.final_winner_val,
        "total_val_evaluations": r.total_val_evaluations,
    }


def generate_record() -> dict[str, Any]:
    """Run all scenarios under all compatible combinations and return the record."""
    record: dict[str, Any] = {}
    tmp_root = Path(tempfile.mkdtemp(prefix="selrec_root_"))
    combo = 0

    try:
        all_scenarios = [(fn, fn()) for fn in SYNTHETIC_SCENARIOS]
        for real in _real_scenarios():
            all_scenarios.append((None, real))

        for _fn, scenario in all_scenarios:
            for mode in SELECTION_MODES:
                for picker in PICKERS:
                    for batch in BATCH_SIZES:
                        if not _is_compatible(mode, picker, batch):
                            continue
                        key = f"{scenario.label}|{mode}|{picker or 'none'}|batch{batch}"
                        work = tmp_root / str(combo)
                        work.mkdir()
                        result = _run_scenario(
                            scenario, mode, picker, batch, work_dir=work,
                        )
                        record[key] = _result_to_dict(result)
                        combo += 1
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return record


def load_record() -> dict[str, Any]:
    """Load the saved record."""
    return json.loads(FIXTURE.read_text())


def save_record(record: dict[str, Any]) -> None:
    """Save the record (sorted, deterministic)."""
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
