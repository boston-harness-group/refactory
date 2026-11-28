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

Extracting a real run into a fixture (one-off, run from refactory/)::

    python tests/selection_record.py /path/to/runs/real3-OL-s42
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HISTORY_DIR = Path(__file__).parent / "fixtures" / "selection_histories"
_REAL_HISTORY_NAMES = ["real3-OL-s42", "real3-OLME-s42"]

# ---------------------------------------------------------------------------
# ScenarioProbe: FitProbe that can simulate train failures
# ---------------------------------------------------------------------------


class ScenarioProbe(FitProbe):
    """FitProbe extended to raise on specific levels (simulating a failed
    train run so the candidate gets ``exit_ok=False``)."""

    name = "scenario_probe"
    FAIL_LEVELS: set[int] = set()

    def run(self, workdir, **kwargs):  # type: ignore[no-untyped-def]
        level = int(self.values()["level"])
        if level in ScenarioProbe.FAIL_LEVELS:
            score = ScenarioProbe.TRAIN.get(level)
            if score is not None:
                subset = Path(workdir) / "subset.json"
                ids = VAL_IDS if subset.exists() and "val" in subset.read_text() else TRAIN_IDS
                records = [{"id": iid, "score": float(score)} for iid in ids]
                return RunResult(
                    exit_ok=False, error="simulated partial failure",
                    workdir=Path(workdir), score=float(score),
                    item_records=records,
                    facts={"item_scores": records},
                )
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
    picker_parents: list[str]
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


def _partial_fail_scenario() -> ScenarioSpec:
    """Partial failure: exit_ok=False WITH a non-None train score, so
    checkpoint's exit_ok guard in ``_apply_round`` is exercised."""
    return ScenarioSpec(
        label="partial_fail",
        rounds=[
            RoundSpec(candidates=[{"label": "seed", "train_score": 0.4, "val_score": 0.4}]),
            RoundSpec(candidates=[
                {"label": "pf1", "parent_label": "seed",
                 "train_score": 0.8, "exit_ok": False, "val_score": 0.8},
            ]),
            RoundSpec(candidates=[
                {"label": "pf2", "parent_label": "seed", "train_score": 0.5, "val_score": 0.5},
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
    _partial_fail_scenario,
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
# Real-run scenario loading
# ---------------------------------------------------------------------------


def _real_scenarios() -> list[ScenarioSpec]:
    """Load real run scenarios from committed fixtures."""
    result = []
    for name in _REAL_HISTORY_NAMES:
        path = _HISTORY_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Real run fixture {path} is missing. Regenerate with:\n"
                f"  python {Path(__file__).name} /path/to/runs/{name}")
        data = json.loads(path.read_text())
        rounds = [RoundSpec(candidates=r["candidates"]) for r in data["rounds"]]
        result.append(ScenarioSpec(
            label=data["label"],
            rounds=rounds,
            early_stop_patience=data.get("early_stop_patience", 0),
            early_stop_at_full_val=data.get("early_stop_at_full_val", False),
        ))
    return result


# ---------------------------------------------------------------------------
# Mapping scenarios to FitProbe levels
# ---------------------------------------------------------------------------


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
            if not cspec.get("exit_ok", True):
                fail_levels.add(level)
            vs = cspec.get("val_score")
            if vs is not None:
                val_map[level] = vs
            round_levels.append(level)
            level += 1
        plan.append(round_levels)

    proposal_plan = plan[1:]
    return train_map, val_map, fail_levels, proposal_plan


# ---------------------------------------------------------------------------
# Patching context
# ---------------------------------------------------------------------------

_VAL_MAP: dict[int, float] = {}


@contextlib.contextmanager
def _patched_harness():  # type: ignore[no-untyped-def]
    """Patch Loop I/O helpers and ``ValidationGate.val_score`` for the
    duration of a full generation run.  Score maps are set per-scenario
    via ``_set_scenario_scores``."""
    saved = {
        "val_score": ValidationGate.val_score,
        "seed_pop": Loop._seed_population,
        "journal": Loop._journal,
        "traces": Loop._write_traces,
        "progress": Loop._progress,
        "write_state": Loop._write_state,
        "write_cand": Loop._write_candidate,
        "save_strat": Loop._save_strategy_state,
        "measure": Loop._measure,
        "context": Loop._context,
        "evaluate": Loop._evaluate,
    }

    def _fast_measure(  # type: ignore[no-untyped-def]
        self, candidate, workdir, part, *, seed=None, items=None,
        gold_feedback=False, item_gate=None, item_slots=False,
        stop_event=None, repeats=None, traces=True,
    ):
        from refactory_dsh import task as _t

        workdir.mkdir(parents=True, exist_ok=True)
        _t.write_subset(workdir, part, self.config.holdout, seed=seed,
                        gold_feedback=gold_feedback)
        if self._parts_manifest is not None:
            manifest_src = self.root / _t.MANIFEST
            if manifest_src.exists():
                shutil.copyfile(manifest_src, workdir / _t.MANIFEST)
        mode = self.mode_class(state=candidate.state)
        result = mode.evaluate(workdir)
        with self._state_lock:
            self.measurements += 1
        return result, {"count": 1, "complete": True}

    ValidationGate.val_score = staticmethod(  # type: ignore[assignment]
        lambda run, cand: float(_VAL_MAP.get(_level(cand), 0.0))
    )
    Loop._seed_population = lambda self, seed: None  # type: ignore[assignment]
    Loop._journal = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._write_traces = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._progress = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._write_state = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._write_candidate = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._save_strategy_state = lambda self, *a, **kw: None  # type: ignore[assignment]
    Loop._measure = _fast_measure  # type: ignore[assignment]

    from refactory_dsh.optim.base import ProposalContext

    def _minimal_context(self):  # type: ignore[no-untyped-def]
        return ProposalContext(
            epoch=self.epoch, generation=self.generation,
            mode_name=self.mode_id, elite=int(self.config.elite),
            val_scores=self._val_scores(),
            val_item_scores=self._val_item_scores(),
            preset=self.config.edit_preset,
            selection=self.selection_mode(),
        )

    Loop._context = _minimal_context  # type: ignore[assignment]

    _original_evaluate = saved["evaluate"]

    def _evaluate_keep_partial(self, candidate, **kwargs):  # type: ignore[no-untyped-def]
        _original_evaluate(self, candidate, **kwargs)
        if not candidate.exit_ok and "partial" in str(candidate.error or ""):
            level = _level(candidate)
            ts = ScenarioProbe.TRAIN.get(level)
            if ts is not None:
                candidate.score = float(ts)
                candidate.raw_score = float(ts)

    Loop._evaluate = _evaluate_keep_partial  # type: ignore[assignment]

    try:
        yield
    finally:
        ValidationGate.val_score = saved["val_score"]  # type: ignore[assignment]
        Loop._seed_population = saved["seed_pop"]  # type: ignore[assignment]
        Loop._journal = saved["journal"]  # type: ignore[assignment]
        Loop._write_traces = saved["traces"]  # type: ignore[assignment]
        Loop._progress = saved["progress"]  # type: ignore[assignment]
        Loop._write_state = saved["write_state"]  # type: ignore[assignment]
        Loop._write_candidate = saved["write_cand"]  # type: ignore[assignment]
        Loop._save_strategy_state = saved["save_strat"]  # type: ignore[assignment]
        Loop._context = saved["context"]  # type: ignore[assignment]
        Loop._measure = saved["measure"]  # type: ignore[assignment]
        Loop._evaluate = saved["evaluate"]  # type: ignore[assignment]


def _set_scenario_scores(
    train_map: dict[int, float],
    val_map: dict[int, float],
    fail_levels: set[int],
) -> None:
    """Update the per-scenario score maps (called inside ``_patched_harness``)."""
    global _VAL_MAP  # noqa: PLW0603
    ScenarioProbe.TRAIN = dict(train_map)
    ScenarioProbe.FAIL_LEVELS = set(fail_levels)
    _VAL_MAP = dict(val_map)


# ---------------------------------------------------------------------------
# Run a scenario through the REAL Loop
# ---------------------------------------------------------------------------


def _run_scenario(
    scenario: ScenarioSpec,
    selection_mode: str,
    picker_name: str,
    batch_size: int,
    plan: list[list[int]],
    parts_path: Path,
    work_dir: Path,
) -> ScenarioResult:
    """Drive the real Loop through ``scenario`` and capture decisions.

    Called inside ``_patched_harness`` and ``_patch_mode`` contexts, with
    score maps already set via ``_set_scenario_scores``.
    """
    n_epochs = max(1, len(scenario.rounds) - 1)
    config = LoopConfig(
        epochs=n_epochs,
        batch_size=batch_size,
        optimizer="random",
        seed=42,
        runs_root=work_dir / "run",
        allowed_ops=("set_knob",),
        max_evaluations=500,
        holdout=0.0,
        selection=selection_mode,
        parent_picker=picker_name,
        parts_manifest=str(parts_path),
        early_stop_patience=scenario.early_stop_patience,
        early_stop_at_full_val=scenario.early_stop_at_full_val,
        max_concurrent_evaluations=1,
    )

    loop = Loop(MODE_ID, config, project=PROJECT_ROOT)

    proposer = ScenarioOptimizer(list(plan))
    if picker_name and hasattr(loop.optimizer, "proposer"):
        loop.optimizer.proposer = proposer  # type: ignore[union-attr]
    else:
        loop.optimizer = proposer

    # Capture picker decisions via a wrapper on picker.select.
    # picker.select() is called during optimizer_step (via choose_parent),
    # AFTER _gate_population in the same epoch.  The picks from epoch N
    # produce the candidates gated in epoch N+1, so each snapshot flushes
    # the buffer of picks made since the last snapshot.
    _picker_calls: list[str] = []
    if picker_name and hasattr(loop.optimizer, "picker"):
        _orig_select = loop.optimizer.picker.select

        def _capturing_select(ctx, n, parents):  # type: ignore[no-untyped-def]
            result = _orig_select(ctx, n, parents)
            _picker_calls.extend(c.id for c in result)
            return result

        loop.optimizer.picker.select = _capturing_select  # type: ignore[assignment]

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
            "picker_parents": list(_picker_calls),
        })
        _picker_calls.clear()

    loop._gate_population = _capturing_gate_pop  # type: ignore[assignment]

    loop.run()

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
            picker_parents=s["picker_parents"],
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
                "picker_parents": d.picker_parents,
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


def generate_record(*, keys: set[str] | None = None) -> dict[str, Any]:
    """Run scenarios under compatible combinations and return the record.

    When *keys* is given, only the matching ``scenario|mode|picker|batch``
    entries are generated (for fast sabotage checks).
    """
    record: dict[str, Any] = {}
    tmp_root = Path(tempfile.mkdtemp(prefix="selrec_root_"))
    parts_path = _write_parts(tmp_root)
    combo = 0

    try:
        all_scenarios: list[ScenarioSpec] = [fn() for fn in SYNTHETIC_SCENARIOS]
        all_scenarios.extend(_real_scenarios())

        with _patch_mode(ScenarioProbe):
            with _patched_harness():
                for scenario in all_scenarios:
                    train_map, val_map, fail_levels, plan = (
                        _map_scenario_simple(scenario))
                    _set_scenario_scores(train_map, val_map, fail_levels)

                    for mode in SELECTION_MODES:
                        for picker in PICKERS:
                            for batch in BATCH_SIZES:
                                if not _is_compatible(mode, picker, batch):
                                    continue
                                key = (f"{scenario.label}|{mode}"
                                       f"|{picker or 'none'}|batch{batch}")
                                if keys is not None and key not in keys:
                                    continue
                                work = tmp_root / str(combo)
                                work.mkdir()
                                result = _run_scenario(
                                    scenario, mode, picker, batch,
                                    list(plan), parts_path, work,
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


# ---------------------------------------------------------------------------
# Real-run extraction helper (one-off, not used by tests)
# ---------------------------------------------------------------------------


def _extract_real_scenario(run_dir: Path) -> ScenarioSpec | None:
    """Extract a scenario from a real run's directory.

    ``run_dir`` is ``<runs>/<name>`` — it should contain ``run/journal.jsonl``
    and ``run/state.json``."""
    journal_path = run_dir / "run" / "journal.jsonl"
    state_path = run_dir / "run" / "state.json"
    if not journal_path.exists() or not state_path.exists():
        return None

    entries: list[dict[str, Any]] = []
    with open(journal_path) as f:
        for line in f:
            entries.append(json.loads(line))

    state = json.loads(state_path.read_text())
    pop = state.get("population", [])
    if not pop:
        return None

    val_cache: dict[str, float] = {}
    for entry in reversed(entries):
        if entry.get("event") == "strategy_state":
            gate_state = (entry.get("state") or {}).get("val_gate") or {}
            val_cache = {str(k): float(v)
                         for k, v in (gate_state.get("val_cache") or {}).items()}
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

    name = run_dir.name
    return ScenarioSpec(label=f"real_{name}", rounds=rounds)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} /path/to/runs/<name>")
        sys.exit(1)
    run_path = Path(sys.argv[1])
    spec = _extract_real_scenario(run_path)
    if spec is None:
        print(f"Could not extract scenario from {run_path}", file=sys.stderr)
        sys.exit(1)
    data = {
        "label": spec.label,
        "rounds": [{"candidates": r.candidates} for r in spec.rounds],
        "early_stop_patience": spec.early_stop_patience,
        "early_stop_at_full_val": spec.early_stop_at_full_val,
    }
    out = _HISTORY_DIR / f"{run_path.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out}")
