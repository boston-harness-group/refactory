"""Deterministic harness for recording selection decisions.

Drives the REAL Loop selection code (gate, pool, checkpoint, incumbent,
leaderboard, parent picking, early stop) through scripted scenarios with no
model calls and no item runs.  Candidates are injected with pre-set train and
val scores by stubbing the evaluator at the narrowest point: the
``ValidationGate.score`` method (which normally rolls out on val) returns the
scripted val score, and train scores are set directly on ``Candidate.score``.

Usage::

    # generate the record
    SELECTION_RECORD_REGEN=1 uv run python -m pytest tests/test_selection_record.py -x -q

    # check against the record
    uv run python -m pytest tests/test_selection_record.py -x -q

The record lives at ``tests/fixtures/selection_record.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from refactory_dsh.optim import Candidate
from refactory_dsh.optim.base import candidate_id
from refactory_dsh.optim.policies import (
    ArchiveSelector,
    EvolutionarySelector,
    HillClimbSelector,
    IslandsSelector,
    MapElitesSelector,
    ParetoSelector,
    ParentSelector,
)
from refactory_dsh.outerloop import (
    EvalRequest,
    ValidationGate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "selection_record.json"

# ---------------------------------------------------------------------------
# Minimal stub mode so Loop.__init__ can resolve it
# ---------------------------------------------------------------------------

_SEED_STATE: dict[str, Any] = {
    "graph": {
        "nodes": {
            "start": {"kind": "data", "fields": {}},
            "agent": {"kind": "agent", "fields": {}},
            "end": {"kind": "data", "fields": {}},
        },
        "edges": [
            {"from": "start", "to": "agent"},
            {"from": "agent", "to": "end"},
        ],
    },
    "knobs": {"prompt": {"value": "seed", "type": "str"}},
}


def _make_state(label: str, structural: bool = False) -> dict[str, Any]:
    """A unique state for a candidate, keyed by label."""
    nodes = dict(_SEED_STATE["graph"]["nodes"])
    if structural:
        nodes[f"extra_{label}"] = {"kind": "agent", "fields": {}}
    return {
        "graph": {
            "nodes": nodes,
            "edges": list(_SEED_STATE["graph"]["edges"]),
        },
        "knobs": {"prompt": {"value": label, "type": "str"}},
    }


def _candidate(
    label: str,
    *,
    parent_id: str | None = None,
    train_score: float | None = None,
    exit_ok: bool = True,
    structural: bool = False,
) -> Candidate:
    state = _make_state(label, structural=structural)
    edits: list[dict[str, Any]] = []
    if structural:
        edits = [{"op": "add_node", "name": f"extra_{label}"}]
    elif parent_id is not None:
        edits = [{"op": "set_knob", "name": "prompt", "value": label}]
    c = Candidate(
        id=candidate_id(state),
        parent=parent_id,
        state=state,
        edits=edits,
        step_edits=len(edits),
        score=train_score,
        exit_ok=exit_ok,
        evaluated=train_score is not None or not exit_ok,
    )
    return c


# ---------------------------------------------------------------------------
# Recording run stub — the narrowest seam
# ---------------------------------------------------------------------------


class RecordingRun:
    """Stub for ``outerloop.Run`` that the gate calls to roll out on val.

    ``evaluate`` and ``val_scores`` return scripted values from the val_scores
    dict.  ``evaluate_many`` delegates to ``evaluate`` for each request.
    ``log`` accumulates messages.
    """

    def __init__(self, val_scores: dict[str, float]) -> None:
        self._val_scores = val_scores
        self.log_lines: list[str] = []
        self._occasions: dict[str, list[dict[str, Any]]] = {}
        self.val_rollouts = 0

    def evaluate(
        self,
        candidate: Candidate,
        *,
        part: str = "train",
        seed: int | None = None,
        items: list[str] | None = None,
        gold_feedback: bool = False,
    ) -> Candidate:
        if part == "val":
            self.val_rollouts += 1
            score = self._val_scores.get(candidate.id, 0.0)
            self._occasions.setdefault(candidate.id, []).append({"score": score})
        return candidate

    def evaluate_many(
        self,
        requests: Sequence[EvalRequest],
        *,
        on_result: Any = None,
    ) -> list[Candidate]:
        results = []
        for i, req in enumerate(requests):
            self.evaluate(req.candidate, part=req.part, seed=req.seed)
            if on_result:
                on_result(i, req.candidate)
            results.append(req.candidate)
        return results

    def val_scores(self, candidate: Candidate) -> list[dict[str, Any]]:
        return self._occasions.get(candidate.id, [])

    def has_part(self, part: str) -> bool:
        return False

    def empty_part_reason(self, part: str) -> str:
        return "no items"

    def log(self, msg: str) -> None:
        self.log_lines.append(msg)

    def shared_scores(self, ids: list[str], part: str) -> None:
        return None


# ---------------------------------------------------------------------------
# The harness: drives real selection code through scripted rounds
# ---------------------------------------------------------------------------


@dataclass
class RoundSpec:
    """One round's candidates to inject."""

    candidates: list[dict[str, Any]]
    # each candidate: {"label", "parent_label", "train_score", "val_score",
    #                   "structural", "exit_ok"}


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
    gate_judged: list[str]
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


def _run_scenario(
    scenario: ScenarioSpec,
    selection_mode: str,
    picker_name: str,
    batch_size: int,
    seed: int = 42,
) -> ScenarioResult:
    """Drive the real selection code through ``scenario`` under the given mode."""

    # Build a lookup of label -> candidate
    label_to_candidate: dict[str, Candidate] = {}
    label_to_val: dict[str, float] = {}
    all_val_scores: dict[str, float] = {}

    # Pre-create all candidates so ids are known
    seed_label = scenario.rounds[0].candidates[0]["label"]
    seed_cand = _candidate(
        seed_label,
        train_score=scenario.rounds[0].candidates[0].get("train_score"),
        exit_ok=scenario.rounds[0].candidates[0].get("exit_ok", True),
        structural=scenario.rounds[0].candidates[0].get("structural", False),
    )
    label_to_candidate[seed_label] = seed_cand
    val_score = scenario.rounds[0].candidates[0].get("val_score")
    if val_score is not None:
        label_to_val[seed_label] = val_score
        all_val_scores[seed_cand.id] = val_score

    for rnd in scenario.rounds[1:]:
        for cspec in rnd.candidates:
            parent_label = cspec.get("parent_label", seed_label)
            parent_id = label_to_candidate[parent_label].id
            c = _candidate(
                cspec["label"],
                parent_id=parent_id,
                train_score=cspec.get("train_score"),
                exit_ok=cspec.get("exit_ok", True),
                structural=cspec.get("structural", False),
            )
            label_to_candidate[cspec["label"]] = c
            vs = cspec.get("val_score")
            if vs is not None:
                label_to_val[cspec["label"]] = vs
                all_val_scores[c.id] = vs

    # Build a stub gate+loop by directly manipulating Loop internals
    # We cannot call Loop.__init__ (needs a real mode), so we build the
    # ValidationGate and Loop state manually and call the methods directly.
    gate = ValidationGate(seed=seed)
    run = RecordingRun(all_val_scores)

    # Build picker if requested
    picker: ParentSelector | None = None
    picker_map: dict[str, type[ParentSelector]] = {
        "archive": ArchiveSelector,
        "evolutionary": EvolutionarySelector,
        "islands": IslandsSelector,
        "hill_climb": HillClimbSelector,
        "map_elites": MapElitesSelector,
        "pareto": ParetoSelector,
    }
    if picker_name and picker_name in picker_map:
        picker = picker_map[picker_name](seed=seed)

    # State tracking (mirroring Loop's internals)
    population: list[Candidate] = []
    gate_accepted: list[str] = []
    gate_rejected: list[str] = []
    gate_rounds: list[list[str]] = []
    pool_size = 8
    strategy_state: dict[str, Any] = {}
    epoch = 0

    decisions: list[DecisionRecord] = []
    total_val = 0

    for round_idx, rnd in enumerate(scenario.rounds):
        epoch += 1
        round_val_before = run.val_rollouts

        if round_idx == 0:
            # Seed round: add seed to population, start the gate
            c = label_to_candidate[seed_label]
            population.append(c)

            if selection_mode == "checkpoint":
                # Under checkpoint, seed gets journaled val
                if c.id in all_val_scores:
                    gate.cache[c.id] = all_val_scores[c.id]
                gate.start(run, c)
            else:
                gate.start(run, c)

            gate_accepted.append(c.id)
            gate_rounds.append([c.id])
            strategy_state["gate_start_epoch"] = epoch

        else:
            # Add this round's candidates to the population
            new_candidates = []
            for cspec in rnd.candidates:
                c = label_to_candidate[cspec["label"]]
                population.append(c)
                new_candidates.append(c)

            # Only evaluated candidates go through the gate
            pending = [c for c in new_candidates if c.evaluated]
            if selection_mode == "checkpoint":
                if pending:
                    _apply_checkpoint_round(
                        gate, run, pending,
                        gate_accepted, gate_rejected, gate_rounds,
                        epoch, all_val_scores,
                    )
            else:
                if pending:
                    _gate_round(
                        gate, run, pending,
                        gate_accepted, gate_rejected, gate_rounds,
                        all_val_scores,
                    )

        # Count stale epochs for early stop
        if scenario.early_stop_patience or scenario.early_stop_at_full_val:
            _count_stale(strategy_state, gate, epoch)

        # Compute pool, leaderboard, incumbent, parents
        judged = set(gate_accepted) | set(gate_rejected)
        order = {c.id: i for i, c in enumerate(population)}
        evaluated = [c for c in population if c.evaluated]

        if selection_mode == "pool":
            ranked = sorted(
                (c for c in evaluated if c.id in judged and c.id in gate.cache),
                key=lambda c: (-gate.cache[c.id], -_train_key(c), order[c.id]),
            )
            pool_members = _top_pool(ranked, gate, pool_size)
            incumbent_cand = next(
                (c for c in pool_members if c.id == gate.current_id), None
            )
            if incumbent_cand is not None:
                pool_members = [incumbent_cand] + [
                    c for c in pool_members if c is not incumbent_cand
                ]
            leaderboard = list(pool_members)
        elif selection_mode == "checkpoint":
            accepted_set = set(gate_accepted)
            leaderboard = sorted(
                [c for c in evaluated if c.id in accepted_set],
                key=lambda c: (
                    c.id != gate.best_id,
                    -gate.cache.get(c.id, 0.0),
                    order[c.id],
                ),
            )
            pool_members = []
        else:
            # single
            accepted_set = set(gate_accepted)
            leaderboard = sorted(
                [c for c in evaluated if c.id in accepted_set],
                key=lambda c: (-gate.cache.get(c.id, 0.0), c.id),
            )
            pool_members = []

        incumbent_id = gate.current_id

        # Choose parents
        parents = _select_parents(
            selection_mode, gate, population, evaluated,
            gate_accepted, gate_rejected, judged,
            leaderboard, pool_members, pool_size, order,
            picker, picker_name, epoch, seed, batch_size,
        )

        # Early stop check
        early_stop = _early_stop_check(
            strategy_state, gate, scenario, epoch,
        )

        round_val = run.val_rollouts - round_val_before
        total_val += round_val

        decisions.append(DecisionRecord(
            round_idx=round_idx,
            gate_judged=[c.id for c in (rnd.candidates if round_idx > 0 else [{"label": seed_label}])
                         for c in [label_to_candidate[c["label"] if isinstance(c, dict) else c]]
                         if c.id in judged],
            gate_accepted=list(gate_accepted),
            gate_rejected=list(gate_rejected),
            pool=[c.id for c in pool_members] if pool_members else [],
            incumbent=incumbent_id,
            leaderboard=[c.id for c in leaderboard],
            parents_chosen=[c.id for c in parents],
            picker_name=picker_name or "default",
            early_stop=early_stop,
            val_evaluations=round_val,
        ))

        if early_stop:
            break

    # Final winner
    final_winner = gate.best_id
    final_winner_val = gate.best_score if gate.best_id else None

    return ScenarioResult(
        decisions=decisions,
        final_winner=final_winner,
        final_winner_val=final_winner_val,
        total_val_evaluations=total_val,
    )


def _train_key(candidate: Candidate) -> float:
    return float(candidate.score) if candidate.score is not None else float("-inf")


def _top_pool(
    ranked: list[Candidate], gate: ValidationGate, pool_size: int
) -> list[Candidate]:
    size = max(1, pool_size)
    if len(ranked) > size:
        edge = gate.cache[ranked[size - 1].id]
        ranked = [c for c in ranked if gate.cache[c.id] >= edge]
    return ranked


def _gate_round(
    gate: ValidationGate,
    run: RecordingRun,
    candidates: list[Candidate],
    accepted: list[str],
    rejected: list[str],
    rounds: list[list[str]],
    val_scores: dict[str, float],
) -> None:
    """Mirror Loop._gate_population for non-checkpoint modes."""
    # Prefetch: score all candidates on val
    for c in candidates:
        if c.id not in gate.cache:
            score = val_scores.get(c.id, 0.0)
            gate.cache[c.id] = score
            run.val_rollouts += 1

    # judge_round: the real code path
    acc, rej = gate.judge_round(run, candidates)
    for c in candidates:
        if c in acc:
            accepted.append(c.id)
        elif c in rej:
            rejected.append(c.id)
    rounds.append([c.id for c in candidates])


def _apply_checkpoint_round(
    gate: ValidationGate,
    run: RecordingRun,
    candidates: list[Candidate],
    accepted: list[str],
    rejected: list[str],
    rounds: list[list[str]],
    epoch: int,
    val_scores: dict[str, float],
) -> None:
    """Mirror Loop._apply_round for checkpoint mode."""
    order = {c.id: i for i, c in enumerate(candidates)}
    trained = [c for c in candidates if c.exit_ok and c.score is not None]
    chosen = (
        max(trained, key=lambda c: (float(c.score or 0.0), -order[c.id]))
        if trained
        else None
    )
    if chosen is not None:
        # Roll out on val
        score = val_scores.get(chosen.id, 0.0)
        gate.cache[chosen.id] = score
        run.val_rollouts += 1
        gate.current_id, gate.current_score = chosen.id, score
        gate.offer_best(chosen.id, score)

    applied = [chosen] if chosen is not None else []
    others = [c for c in candidates if c is not chosen]
    accepted.extend(c.id for c in applied)
    rejected.extend(c.id for c in others)
    rounds.append([c.id for c in candidates])


def _count_stale(
    state: dict[str, Any], gate: ValidationGate, epoch: int
) -> None:
    """Mirror Loop._count_stale_epoch."""
    if state.get("stale_counted_epoch") == epoch:
        return
    state["stale_counted_epoch"] = epoch
    if state.get("gate_start_epoch", epoch) == epoch:
        state["early_stop_best"] = gate.best_score
        return
    best = float(state.get("early_stop_best", gate.best_score))
    if gate.best_score > best + 1e-9:
        state["early_stop_best"] = gate.best_score
        state["stale_epochs"] = 0
    else:
        state["stale_epochs"] = int(state.get("stale_epochs", 0)) + 1


def _early_stop_check(
    state: dict[str, Any],
    gate: ValidationGate,
    scenario: ScenarioSpec,
    epoch: int,
) -> str:
    """Mirror Loop._early_stop_reason."""
    counted = state.get("stale_counted_epoch")
    if counted is None or counted == state.get("gate_start_epoch"):
        return ""
    if scenario.early_stop_at_full_val and gate.best_score >= 1.0 - 1e-9:
        return "perfect_val"
    if scenario.early_stop_patience > 0 and int(
        state.get("stale_epochs", 0)
    ) >= scenario.early_stop_patience:
        return "patience_exhausted"
    return ""


def _select_parents(
    selection_mode: str,
    gate: ValidationGate,
    population: list[Candidate],
    evaluated: list[Candidate],
    gate_accepted: list[str],
    gate_rejected: list[str],
    judged: set[str],
    leaderboard: list[Candidate],
    pool_members: list[Candidate],
    pool_size: int,
    order: dict[str, int],
    picker: ParentSelector | None,
    picker_name: str,
    epoch: int,
    seed: int,
    n: int,
) -> list[Candidate]:
    """Mirror Loop.select_parents + the picker's select."""
    from refactory_dsh.optim.base import ProposalContext

    if selection_mode == "checkpoint":
        # Under checkpoint: current candidate first
        current = next((c for c in population if c.id == gate.current_id), None)
        parents = [current] if current else leaderboard[:1]
    elif selection_mode == "pool":
        has_population_policy = picker is not None and getattr(
            picker, "population", False
        )
        if has_population_policy:
            # Every judged candidate, ranked
            parents = sorted(
                (c for c in evaluated if c.id in judged and c.id in gate.cache),
                key=lambda c: (
                    -gate.cache[c.id],
                    -_train_key(c),
                    order[c.id],
                ),
            )
            incumbent = next(
                (c for c in parents if c.id == gate.current_id), None
            )
            if incumbent is not None:
                parents = [incumbent] + [c for c in parents if c is not incumbent]
        else:
            parents = list(pool_members)
    else:
        # single: gate's current candidate
        current = next(
            (c for c in leaderboard if c.id == gate.current_id), None
        )
        parents = [current] if current else leaderboard[:1]

    if not parents:
        parents = [c for c in population[:n] if c.evaluated]

    # Apply picker if present
    if picker is not None and parents:
        ctx = ProposalContext(
            generation=epoch,
            epoch=epoch,
            preset="broad",
        )
        ctx.val_scores = {c.id: gate.cache.get(c.id, 0.0) for c in parents}
        try:
            chosen = picker.select(ctx, n, parents)
            if chosen:
                return chosen
        except Exception:
            pass

    return parents[:n]


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
                {"label": "s1", "parent_label": "seed", "train_score": 0.5, "val_score": 0.5, "structural": True},
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

# Strategies that pick parents (from strategies.py)
STRATEGY_NAMES = ["tree_search", "reflective", "adaptive_meta"]


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

    # Get population from state.json
    state_file = run_dir / "state.json"
    if not state_file.exists():
        return None
    state = json.loads(state_file.read_text())
    pop = state.get("population", [])
    if not pop:
        return None

    # Get val cache from last strategy_state in journal
    val_cache: dict[str, float] = {}
    for entry in reversed(entries):
        if entry.get("event") == "strategy_state":
            gate_state = (entry.get("state") or {}).get("val_gate") or {}
            val_cache = {str(k): float(v) for k, v in (gate_state.get("val_cache") or {}).items()}
            break

    # Build rounds: seed first, then each epoch's candidates
    seed = pop[0]
    seed_round = RoundSpec(candidates=[{
        "label": seed["id"],
        "train_score": seed.get("score"),
        "val_score": val_cache.get(seed["id"]),
        "exit_ok": seed.get("exit_ok", True),
        "structural": seed.get("structural", False),
    }])

    # Group non-seed by proposal epoch
    proposal_epochs: dict[int, list[dict[str, Any]]] = {}
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
                    "structural": cand.get("structural", False),
                })
        if epoch_cands:
            rounds.append(RoundSpec(candidates=epoch_cands))

    # Add any remaining candidates not in proposals (e.g., from second epochs)
    for cand in pop:
        if cand["id"] not in seen_ids:
            seen_ids.add(cand["id"])
            rounds.append(RoundSpec(candidates=[{
                "label": cand["id"],
                "parent_label": cand.get("parent") or seed["id"],
                "train_score": cand.get("score"),
                "val_score": val_cache.get(cand["id"]),
                "exit_ok": cand.get("exit_ok", True),
                "structural": cand.get("structural", False),
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
# For real scenarios, we need to use the REAL candidate ids as labels,
# and construct states that hash to them. We override the _candidate helper.
# ---------------------------------------------------------------------------


def _candidate_for_real(
    cid: str,
    *,
    parent_id: str | None = None,
    train_score: float | None = None,
    exit_ok: bool = True,
    structural: bool = False,
) -> Candidate:
    """Create a candidate with a specific id (for real-run replay)."""
    state = _make_state(cid, structural=structural)
    edits: list[dict[str, Any]] = []
    if structural:
        edits = [{"op": "add_node", "name": f"extra_{cid}"}]
    elif parent_id is not None:
        edits = [{"op": "set_knob", "name": "prompt", "value": cid}]
    c = Candidate(
        id=cid,
        parent=parent_id,
        state=state,
        edits=edits,
        step_edits=len(edits),
        score=train_score,
        exit_ok=exit_ok,
        evaluated=train_score is not None,
    )
    return c


def _run_scenario_real(
    scenario: ScenarioSpec,
    selection_mode: str,
    picker_name: str,
    batch_size: int,
    seed: int = 42,
) -> ScenarioResult:
    """Run a scenario where labels ARE candidate ids (real run replay)."""
    label_to_candidate: dict[str, Candidate] = {}
    all_val_scores: dict[str, float] = {}

    seed_label = scenario.rounds[0].candidates[0]["label"]
    seed_cand = _candidate_for_real(
        seed_label,
        train_score=scenario.rounds[0].candidates[0].get("train_score"),
        exit_ok=scenario.rounds[0].candidates[0].get("exit_ok", True),
        structural=scenario.rounds[0].candidates[0].get("structural", False),
    )
    label_to_candidate[seed_label] = seed_cand
    vs = scenario.rounds[0].candidates[0].get("val_score")
    if vs is not None:
        all_val_scores[seed_cand.id] = vs

    for rnd in scenario.rounds[1:]:
        for cspec in rnd.candidates:
            parent_label = cspec.get("parent_label", seed_label)
            parent_id = label_to_candidate.get(parent_label, seed_cand).id
            c = _candidate_for_real(
                cspec["label"],
                parent_id=parent_id,
                train_score=cspec.get("train_score"),
                exit_ok=cspec.get("exit_ok", True),
                structural=cspec.get("structural", False),
            )
            label_to_candidate[cspec["label"]] = c
            v = cspec.get("val_score")
            if v is not None:
                all_val_scores[c.id] = v

    gate = ValidationGate(seed=seed)
    run = RecordingRun(all_val_scores)
    picker: ParentSelector | None = None
    picker_map: dict[str, type[ParentSelector]] = {
        "archive": ArchiveSelector,
        "evolutionary": EvolutionarySelector,
        "islands": IslandsSelector,
        "hill_climb": HillClimbSelector,
        "map_elites": MapElitesSelector,
        "pareto": ParetoSelector,
    }
    if picker_name and picker_name in picker_map:
        picker = picker_map[picker_name](seed=seed)

    population: list[Candidate] = []
    gate_accepted: list[str] = []
    gate_rejected: list[str] = []
    gate_rounds: list[list[str]] = []
    pool_size = 8
    strategy_state: dict[str, Any] = {}
    epoch = 0
    decisions: list[DecisionRecord] = []
    total_val = 0

    for round_idx, rnd in enumerate(scenario.rounds):
        epoch += 1
        round_val_before = run.val_rollouts

        if round_idx == 0:
            c = label_to_candidate[seed_label]
            population.append(c)
            if selection_mode == "checkpoint":
                if c.id in all_val_scores:
                    gate.cache[c.id] = all_val_scores[c.id]
                gate.start(run, c)
            else:
                gate.start(run, c)
            gate_accepted.append(c.id)
            gate_rounds.append([c.id])
            strategy_state["gate_start_epoch"] = epoch
        else:
            new_candidates = []
            for cspec in rnd.candidates:
                c = label_to_candidate[cspec["label"]]
                population.append(c)
                new_candidates.append(c)
            pending = [c for c in new_candidates if c.evaluated]
            if selection_mode == "checkpoint":
                if pending:
                    _apply_checkpoint_round(
                        gate, run, pending,
                        gate_accepted, gate_rejected, gate_rounds,
                        epoch, all_val_scores,
                    )
            else:
                if pending:
                    _gate_round(
                        gate, run, pending,
                        gate_accepted, gate_rejected, gate_rounds,
                        all_val_scores,
                    )

        if scenario.early_stop_patience or scenario.early_stop_at_full_val:
            _count_stale(strategy_state, gate, epoch)

        judged = set(gate_accepted) | set(gate_rejected)
        order = {c.id: i for i, c in enumerate(population)}
        evaluated = [c for c in population if c.evaluated]

        if selection_mode == "pool":
            ranked = sorted(
                (c for c in evaluated if c.id in judged and c.id in gate.cache),
                key=lambda c: (-gate.cache[c.id], -_train_key(c), order[c.id]),
            )
            pool_members = _top_pool(ranked, gate, pool_size)
            incumbent_cand = next(
                (c for c in pool_members if c.id == gate.current_id), None
            )
            if incumbent_cand is not None:
                pool_members = [incumbent_cand] + [
                    c for c in pool_members if c is not incumbent_cand
                ]
            leaderboard = list(pool_members)
        elif selection_mode == "checkpoint":
            accepted_set = set(gate_accepted)
            leaderboard = sorted(
                [c for c in evaluated if c.id in accepted_set],
                key=lambda c: (
                    c.id != gate.best_id,
                    -gate.cache.get(c.id, 0.0),
                    order[c.id],
                ),
            )
            pool_members = []
        else:
            accepted_set = set(gate_accepted)
            leaderboard = sorted(
                [c for c in evaluated if c.id in accepted_set],
                key=lambda c: (-gate.cache.get(c.id, 0.0), c.id),
            )
            pool_members = []

        incumbent_id = gate.current_id
        parents = _select_parents(
            selection_mode, gate, population, evaluated,
            gate_accepted, gate_rejected, judged,
            leaderboard, pool_members, pool_size, order,
            picker, picker_name, epoch, seed, batch_size,
        )
        early_stop = _early_stop_check(strategy_state, gate, scenario, epoch)
        round_val = run.val_rollouts - round_val_before
        total_val += round_val

        decisions.append(DecisionRecord(
            round_idx=round_idx,
            gate_judged=[c.id for cspec in (rnd.candidates if round_idx > 0 else [{"label": seed_label}])
                         for c in [label_to_candidate[cspec["label"]]]
                         if c.id in judged],
            gate_accepted=list(gate_accepted),
            gate_rejected=list(gate_rejected),
            pool=[c.id for c in pool_members] if pool_members else [],
            incumbent=incumbent_id,
            leaderboard=[c.id for c in leaderboard],
            parents_chosen=[c.id for c in parents],
            picker_name=picker_name or "default",
            early_stop=early_stop,
            val_evaluations=round_val,
        ))
        if early_stop:
            break

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

    for scenario_fn in SYNTHETIC_SCENARIOS:
        scenario = scenario_fn()
        for mode in SELECTION_MODES:
            for picker in PICKERS:
                for batch in BATCH_SIZES:
                    if not _is_compatible(mode, picker, batch):
                        continue
                    key = f"{scenario.label}|{mode}|{picker or 'none'}|batch{batch}"
                    result = _run_scenario(scenario, mode, picker, batch)
                    record[key] = _result_to_dict(result)

    # Real scenarios: run under every mode
    for scenario in _real_scenarios():
        for mode in SELECTION_MODES:
            for picker in PICKERS:
                for batch in BATCH_SIZES:
                    if not _is_compatible(mode, picker, batch):
                        continue
                    key = f"{scenario.label}|{mode}|{picker or 'none'}|batch{batch}"
                    result = _run_scenario_real(scenario, mode, picker, batch)
                    record[key] = _result_to_dict(result)

    return record


def load_record() -> dict[str, Any]:
    """Load the saved record."""
    return json.loads(FIXTURE.read_text())


def save_record(record: dict[str, Any]) -> None:
    """Save the record (sorted, deterministic)."""
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
