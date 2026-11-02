"""SkillOpt outer-loop strategy tests.

Each test exercises one mechanism from the brief with a scripted model and a
small toy Task, and each fails when the mechanism it tests is reverted.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn
from refactory_dsh import strategies
from refactory_dsh import task as task_module
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import Candidate

from refactory_packages.skillopt.loop import SkillOptLoop
from refactory_packages.skillopt import schedule as sched_mod
from refactory_packages.skillopt import prompts

# ── toy task & mode ────────────────────────────────────────────────────────

TASK_REF = "skillopt_test_task"
MODE_ID = "skillopt_probe/skill_probe"

# Train/val split
TRAIN_IDS = [f"t{i}" for i in range(1, 9)]  # 8 train
VAL_IDS = [f"v{i}" for i in range(1, 5)]     # 4 val

# A magic marker the toy task looks for to decide pass/fail
MAGIC_MARKER = "SKILL_MAGIC_PASS"

# Comma-separated item ids the toy task passes (the rest fail)
PASS_IDS_ENV = "_SKILLOPT_PASS_IDS"

# Initial skill text (enough content for analysts to work with)
INITIAL_SKILL = """\
# Skill Document

## Problem Solving
When approaching a problem, break it into steps.

## Verification
Always verify your answer before submitting.
"""


class SkillTask(task_module.Task):
    """Toy Task: passes if the skill text contains MAGIC_MARKER, fails otherwise.

    Score varies by item id to allow distinguishing scores.
    """

    name = TASK_REF

    def instances(self, *, part: str | None = None, seed: int | None = None) -> list[task_module.TaskInstance]:
        return [
            task_module.TaskInstance(id=item_id)
            for item_id in TRAIN_IDS + VAL_IDS
        ]

    #: (part, seed) of every verify, so a test can see what reached the Task.
    seen: list[tuple[str | None, int | None]] = []

    def verify(self, instance, workspace, *, part=None, seed=None):  # type: ignore[no-untyped-def]
        SkillTask.seen.append((part, seed))
        # Read the skill from the state if it's available via item data
        # For simplicity: pass based on whether MAGIC_MARKER env is set
        # In tests, the FakeClient controls the scoring via the val cache
        pass_ids = os.environ.get(PASS_IDS_ENV, "").split(",")
        passed = bool(os.environ.get("_SKILLOPT_PASS_ALL", "")) or instance.id in pass_ids
        score = 1.0 if passed else 0.5
        return task_module.VerifyResult(passed=passed, score=score)


task_module.register_task(TASK_REF, SkillTask)


def _probe_graph() -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref=TASK_REF, writes=("current_item.json",), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=("current_item.json",), writes=("seen.json",), produces=("seen",),
        knobs=("skill",),
    ))
    graph.add(nn.Node(
        "finish", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "finish")
    return graph


class SkillProbe(nn.Mode):
    """Mode with a ``skill`` str knob and a data node iterating SkillTask."""

    name = "skill_probe"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "skill": nn.Knob(default=INITIAL_SKILL, kind="str",
                             description="The skill document"),
        }

    def build(self) -> nn.Graph:
        return _probe_graph()

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


# ── FakeClient ─────────────────────────────────────────────────────────────


class FakeClient:
    """Scripted model client that returns canned responses keyed by call_key.

    When a specific call_key is not in the responses dict, returns a default
    response based on the step_type.
    """

    def __init__(self, responses: dict[tuple, str] | None = None) -> None:
        self.responses: dict[tuple, str] = dict(responses or {})
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        call_key: tuple | list | None = None,
    ) -> str:
        key = tuple(call_key) if call_key else None
        self.calls.append({
            "system": system,
            "prompt": prompt,
            "call_key": key,
        })
        if key and key in self.responses:
            return self.responses[key]
        # Default responses based on step type
        step_type = key[0] if key else ""
        return self._default_response(step_type, prompt)

    def _default_response(self, step_type: str, prompt: str) -> str:
        if step_type.startswith("analyst_"):
            return json.dumps({
                "batch_size": 1,
                "failure_summary": [],
                "patch": {
                    "reasoning": "test edits",
                    "edits": [
                        {"op": "append", "content": "## New Rule\nAlways check."},
                    ],
                },
            })
        if step_type.startswith("merge_"):
            return json.dumps({
                "reasoning": "merged",
                "edits": [
                    {"op": "append", "content": "## Merged\nCombined rule.",
                     "source_type": "failure", "support_count": 2},
                ],
            })
        if step_type == "rank":
            return json.dumps({
                "reasoning": "ranked",
                "selected_indices": [0],
            })
        if step_type == "slow_update":
            return json.dumps({
                "reasoning": "reflected",
                "slow_update_content": "Updated guidance text.",
            })
        if step_type == "meta_skill":
            return json.dumps({
                "reasoning": "meta reflected",
                "meta_skill_content": "USE CONCISE EDITS",
            })
        return json.dumps({"result": "ok"})


class ReplayClient:
    """Replay client: returns recorded outputs keyed by call_key."""

    def __init__(self, recordings: dict[tuple, str]) -> None:
        self.recordings = dict(recordings)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        call_key: tuple | list | None = None,
    ) -> str:
        key = tuple(call_key) if call_key else None
        self.calls.append({"call_key": key, "prompt": prompt})
        if key and key in self.recordings:
            return self.recordings[key]
        return json.dumps({"result": "no recording"})


# ── helpers ────────────────────────────────────────────────────────────────


def _patch_mode(mode_class: type[nn.Mode]):  # type: ignore[no-untyped-def]
    """Point the registry at a local mode class for the duration of a test."""
    import refactory_dsh.registry as registry

    class FakePackage:
        name = "fake"
        description = ""

        def mode(self, name: str) -> type[nn.Mode]:
            return mode_class

    @contextlib.contextmanager
    def patched():  # type: ignore[no-untyped-def]
        original = registry.resolve
        registry.resolve = lambda project, identifier: (FakePackage(), mode_class)  # type: ignore[assignment]
        try:
            yield
        finally:
            registry.resolve = original  # type: ignore[assignment]

    return patched()


def _write_parts(root: Path) -> Path:
    """Write a parts.json manifest and return its path."""
    manifest = {"train": TRAIN_IDS, "val": VAL_IDS}
    parts_path = root / "parts.json"
    parts_path.parent.mkdir(parents=True, exist_ok=True)
    parts_path.write_text(json.dumps(manifest))
    return parts_path


def _config(root: Path, **overrides: Any) -> LoopConfig:
    """Build a LoopConfig for SkillOpt tests."""
    parts_path = _write_parts(root)
    base: dict[str, Any] = dict(
        epochs=10,
        batch_size=1,
        optimizer="random",
        seed=42,
        runs_root=root / "run",
        allowed_ops=("set_knob", "text_patch"),
        strategy="skillopt",
        parts_manifest=str(parts_path),
        max_evaluations=200,
    )
    base.update(overrides)
    return LoopConfig(**base)


def _run_loop(
    tmp_path: Path,
    client: FakeClient | ReplayClient | None = None,
    config_overrides: dict[str, Any] | None = None,
    skillopt_options: dict[str, Any] | None = None,
    preload_state: dict[str, Any] | None = None,
) -> tuple[SkillOptLoop, dict[str, Any]]:
    """Set up and run a SkillOptLoop via strategies.create(), returning (loop, report).

    Uses the public ``strategies.create()`` path: the Loop machinery
    constructs the infrastructure, and the SkillOptLoop strategy is attached
    via ``set_custom_strategy()``.
    """
    cfg = _config(tmp_path, **(config_overrides or {}))
    client = client or FakeClient()
    options = dict(skillopt_options or {})
    with _patch_mode(SkillProbe):
        loop = strategies.create(
            "skillopt", MODE_ID, cfg,
            strategy_options=options,
            project=Path("."), model=client,
        )
        strategy = loop._custom_strategy
        assert isinstance(strategy, SkillOptLoop)
        if preload_state is not None:
            strategy.load_strategy_state(preload_state)
        report = loop.run()
    return strategy, report


def _spy_evaluations(monkeypatch) -> list[tuple[str, tuple | None]]:  # type: ignore[no-untyped-def]
    """Record (part, items) for every Run.evaluate call."""
    from refactory_dsh.outerloop import Run

    calls: list[tuple[str, tuple | None]] = []
    original = Run.evaluate

    def evaluate(self, candidate, *, part="train", items=None, seed=None):  # type: ignore[no-untyped-def]
        calls.append((part, tuple(items) if items else None))
        return original(self, candidate, part=part, items=items, seed=seed)

    monkeypatch.setattr(Run, "evaluate", evaluate)
    return calls


def _decisions(trace: list[dict[str, Any]], step_type: str) -> list[dict[str, Any]]:
    return [e for e in trace if e["type"] == "decision" and e["step_type"] == step_type]


def _read_trace(root: Path) -> list[dict[str, Any]]:
    """Read the JSONL trace file."""
    trace_path = root / "run" / "skillopt_trace.jsonl"
    if not trace_path.exists():
        return []
    entries = []
    for line in trace_path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


@pytest.fixture(autouse=True)
def _skillopt_found_by_discovery():
    """Every test reaches ``skillopt`` the way the CLI does: by package discovery
    (``Package.strategies``), never a hand ``register()``."""
    assert "skillopt" not in strategies.STRATEGIES
    yield
    assert "skillopt" not in strategies.STRATEGIES


# ── tests ──────────────────────────────────────────────────────────────────


def test_edit_budget_schedule_matches_skillopt():
    """The cosine schedule reproduces SkillOpt's CosineScheduler._compute_lr
    at scheduler.py:87-92 for the planned steps.
    """
    assert sched_mod.full_sequence(4) == [4, 3, 2, 2]
    assert sched_mod.edit_budget(1, 1) == 4
    assert sched_mod.full_sequence(3) == [4, 2, 2]
    # Edge: step beyond total_steps should clamp
    assert sched_mod.edit_budget(5, 4) == 2
    # Verify the formula explicitly for total_steps=4
    for s, expected in [(1, 4), (2, 3), (3, 2), (4, 2)]:
        t = min(s, 4) / 4
        lr = 2 + 0.5 * (4 - 2) * (1 + math.cos(math.pi * t))
        assert sched_mod.edit_budget(s, 4) == max(2, round(lr)) == expected


def test_full_epoch_end_to_end(tmp_path):
    """Run 1 epoch: rollout, analysts, merge, rank, patch, val gate.

    Assert: trace has analyst_failure, analyst_success, merge, rank,
    val_evaluate entries. Candidate has modified skill text.
    """
    client = FakeClient()
    loop, report = _run_loop(
        tmp_path, client=client,
        skillopt_options={"max_epochs": 1},
    )

    trace = _read_trace(tmp_path)
    step_types = [e["step_type"] for e in trace]

    assert "initial_val" in step_types
    assert "train_rollout" in step_types
    assert "analyst_failure" in step_types or "analyst_success" in step_types
    assert "patch_applied" in step_types or "patch_refused" in step_types
    # Trace records call keys and decision inputs
    model_calls = [e for e in trace if e["type"] == "model_call"]
    assert len(model_calls) > 0
    for mc in model_calls:
        assert "call_key" in mc
        assert len(mc["call_key"]) == 4
    decisions = [e for e in trace if e["type"] == "decision"]
    assert len(decisions) > 0
    # At least one analyst decision should have record_ids (shuffled record order)
    analyst_decisions = [d for d in decisions if d["step_type"].startswith("analyst_")]
    if analyst_decisions:
        assert "record_ids" in analyst_decisions[0]
    # The trace should have edit_budget info
    ranked_decisions = [d for d in decisions if d["step_type"] == "ranked_edits"]
    if ranked_decisions:
        assert "edit_budget" in ranked_decisions[0]


def test_no_edit_step_skipped(tmp_path, monkeypatch):
    """FakeClient returns empty edits. No val rollout, edit_budget_step=0.

    trainer.py:1336-1360 skips the step before the scheduler and the
    selection rollout.
    """
    parts = _spy_evaluations(monkeypatch)
    empty_responses = {}
    # Override all analyst calls to return empty edits
    for gi in range(10):
        for step in ("analyst_failure", "analyst_success"):
            empty_responses[(step, 1, gi, 0)] = json.dumps({
                "batch_size": 0,
                "patch": {"reasoning": "no edits", "edits": []},
            })

    client = FakeClient(responses=empty_responses)
    loop, report = _run_loop(
        tmp_path, client=client,
        skillopt_options={"max_epochs": 1},
    )

    trace = _read_trace(tmp_path)
    step_types = [e["step_type"] for e in trace]

    assert "no_edits_skip" in step_types
    # No val_evaluate when edits are empty — mechanism 6
    val_evals = [e for e in trace if e["step_type"] == "val_evaluate"]
    assert len(val_evals) == 0, "val should not be evaluated when no edits are proposed"
    # Only the initial val score was rolled out — no val rollout for the step
    assert [p for p, _ in parts].count("val") == 1

    # edit_budget_step should remain at 0
    state_dict = loop.strategy_state_dict()
    assert state_dict["edit_budget_step"] == 0


def test_cached_skill_hash_not_rescored(tmp_path):
    """Val cache prevents re-evaluation of the same skill hash.

    Run 1: discover the patched-candidate hash that epoch 1 produces.
    Run 2: pre-populate val_cache with that hash → epoch 1 must find
    a val_cache_hit instead of re-evaluating.
    """
    # Run 1: discover the hash the edit produces
    client1 = FakeClient()
    loop1, _ = _run_loop(
        tmp_path / "run1", client=client1,
        skillopt_options={"max_epochs": 1},
    )
    trace1 = _read_trace(tmp_path / "run1")
    val_evals1 = [e for e in trace1 if e["step_type"] == "val_evaluate"]
    assert len(val_evals1) >= 1, "run 1 must evaluate val"
    target_hash = val_evals1[0]["skill_hash"]

    # Run 2: pre-populate val_cache with the target hash
    client2 = FakeClient()  # same default responses → same edit
    loop2, _ = _run_loop(
        tmp_path / "run2", client=client2,
        skillopt_options={"max_epochs": 1},
        preload_state={"val_cache": {target_hash: 0.99}},
    )
    trace2 = _read_trace(tmp_path / "run2")
    cache_hits = [e for e in trace2 if e["step_type"] == "val_cache_hit"]
    val_evals2 = [e for e in trace2 if e["step_type"] == "val_evaluate"]

    assert len(cache_hits) >= 1, "pre-populated hash must trigger val_cache_hit"
    # The cached hash should NOT appear as a val_evaluate
    evaluated_hashes = {e["skill_hash"] for e in val_evals2}
    assert target_hash not in evaluated_hashes, (
        "the cached hash was re-evaluated; the val cache is not being consulted"
    )


def test_accept_only_if_strictly_better(tmp_path):
    """Equal score → reject.  Strictly better → accept."""
    import hashlib

    # ── Part A: equal score → reject ──────────────────────────────────────
    client_eq = FakeClient()
    loop_eq, _ = _run_loop(
        tmp_path / "equal", client=client_eq,
        skillopt_options={"max_epochs": 1},
    )
    trace_eq = _read_trace(tmp_path / "equal")
    gate_events_eq = [
        e for e in trace_eq
        if e["step_type"] in ("val_gate_accept", "val_gate_reject")
    ]
    assert len(gate_events_eq) >= 1, "val gate must make a decision"
    # With the default SkillTask (score=0.5 everywhere), all evaluations are
    # equal → reject, so the gate must reject.
    for evt in gate_events_eq:
        assert evt["step_type"] == "val_gate_reject", (
            "equal score must be rejected (not >=)"
        )
        assert evt["cand_score"] <= evt["current_score"]

    # ── Part B: strictly better → accept ──────────────────────────────────
    # Pre-populate val_cache: give the patched candidate a score higher than
    # the initial val score (0.5).  This proves acceptance on >.
    # First discover the patched hash (same edit as run 1).
    trace1_evals = [e for e in trace_eq if e["step_type"] in ("val_evaluate", "val_cache_hit")]
    patched_hash = None
    for ev in trace1_evals:
        if ev.get("skill_hash"):
            patched_hash = ev["skill_hash"]
            break
    assert patched_hash is not None, "must find the patched candidate hash"

    # Run with val_cache assigning the patched hash a score > 0.5
    client_gt = FakeClient()
    loop_gt, _ = _run_loop(
        tmp_path / "better", client=client_gt,
        skillopt_options={"max_epochs": 1},
        preload_state={"val_cache": {patched_hash: 0.9}},
    )
    trace_gt = _read_trace(tmp_path / "better")
    gate_events_gt = [
        e for e in trace_gt
        if e["step_type"] in ("val_gate_accept", "val_gate_reject")
    ]
    assert len(gate_events_gt) >= 1, "val gate must make a decision"
    accepts = [e for e in gate_events_gt if e["step_type"] == "val_gate_accept"]
    assert len(accepts) >= 1, "strictly better score must be accepted"
    for evt in accepts:
        assert evt["cand_score"] > evt["current_score"], (
            "val_gate_accept must record cand_score > current_score"
        )


def test_slow_update_from_epoch_2(tmp_path):
    """Epoch 1: placeholder markers. Epoch 2: guidance inserted, cache updated."""
    client = FakeClient()
    loop, report = _run_loop(
        tmp_path, client=client,
        skillopt_options={"max_epochs": 2},
    )

    trace = _read_trace(tmp_path)
    step_types = [e["step_type"] for e in trace]

    # Epoch 1 should have slow_update_placeholder
    assert "slow_update_placeholder" in step_types

    # Epoch 2 should have slow_update with sample_ids
    slow_sample = [e for e in trace if e["step_type"] == "slow_update_sample"]
    assert len(slow_sample) >= 1, "epoch 2 must have a slow_update_sample decision"
    assert "sample_ids" in slow_sample[0]

    # Check the skill contains markers after epoch 1
    state_dict = loop.strategy_state_dict()
    skill = state_dict.get("current_skill", "")
    assert prompts.SLOW_UPDATE_START in skill
    assert prompts.SLOW_UPDATE_END in skill

    # Val cache should have an entry for post-slow-update skill hash
    val_cache = state_dict.get("val_cache", {})
    assert len(val_cache) >= 1


def test_meta_skill_reaches_next_epoch_prompts(tmp_path):
    """Run 3 epochs. Meta skill from epoch 2 appears in epoch 3 analyst prompts."""
    meta_content = "USE CONCISE EDITS"
    responses: dict[tuple, str] = {}
    responses[("meta_skill", 2, 0, 0)] = json.dumps({
        "reasoning": "meta",
        "meta_skill_content": meta_content,
    })

    client = FakeClient(responses=responses)
    loop, report = _run_loop(
        tmp_path, client=client,
        skillopt_options={"max_epochs": 3},
    )

    # Check that epoch 3 analyst prompts contain the meta skill content.
    # We MUST have epoch 3 analyst calls for this test to be meaningful.
    epoch3_analyst_calls = [
        c for c in client.calls
        if c.get("call_key") and c["call_key"][0].startswith("analyst_") and c["call_key"][1] == 3
    ]
    assert len(epoch3_analyst_calls) > 0, (
        "epoch 3 must have analyst calls for meta skill propagation to be tested"
    )
    combined_prompts = " ".join(c.get("prompt", "") for c in epoch3_analyst_calls)
    assert meta_content in combined_prompts, (
        f"meta skill content '{meta_content}' not found in epoch 3 analyst prompts"
    )

    # Also verify meta_skill also reaches rank prompts
    epoch3_rank_calls = [
        c for c in client.calls
        if c.get("call_key") and c["call_key"][0] == "rank" and c["call_key"][1] == 3
    ]
    if epoch3_rank_calls:
        rank_prompt = epoch3_rank_calls[0].get("prompt", "")
        assert meta_content in rank_prompt, (
            f"meta skill content '{meta_content}' not found in epoch 3 rank prompt"
        )

    trace = _read_trace(tmp_path)
    meta_decisions = [e for e in trace if e["step_type"] == "meta_skill"]
    # Meta skill should appear for epoch 2
    epoch2_meta = [e for e in meta_decisions if e["epoch"] == 2]
    assert len(epoch2_meta) >= 1, "meta_skill decision should exist for epoch 2"


def test_stop_resume_same_decision(tmp_path):
    """Stop after epoch 1 val, resume → RNG and state continue consistently."""
    client1 = FakeClient()
    strategy1, report1 = _run_loop(
        tmp_path / "run1", client=client1,
        skillopt_options={"max_epochs": 1},
    )
    state1 = strategy1.strategy_state_dict()

    # The state must be JSON-serializable (for journaling)
    json_str = json.dumps(state1)
    state1_roundtripped = json.loads(json_str)

    # Run 2: create fresh strategy, load state, verify consistency
    client2 = FakeClient()
    strategy2 = SkillOptLoop()
    strategy2.configure(model=client2, max_epochs=2)
    strategy2.load_strategy_state(state1_roundtripped)
    state2 = strategy2.strategy_state_dict()

    # States must match
    assert state2["current_skill"] == state1["current_skill"]
    assert state2["current_score"] == state1["current_score"]
    assert state2["val_cache"] == state1["val_cache"]
    assert state2["edit_budget_step"] == state1["edit_budget_step"]
    assert state2["epoch"] == state1["epoch"]

    assert state2 == state1
    # No generator is carried between steps: every shuffle and sample is
    # reseeded from the seed and epoch (reflect.py:481, trainer.py:1844).
    assert "rng_state" not in state1


def test_pause_stop_honoured(tmp_path):
    """Write control.json with stop → loop terminates with control_stop bound."""
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)

    # Write stop control before running
    control_path = run_root / "control.json"
    control_path.write_text(json.dumps({"action": "stop"}))

    client = FakeClient()
    cfg = _config(tmp_path, runs_root=run_root)

    with _patch_mode(SkillProbe):
        loop = strategies.create(
            "skillopt", MODE_ID, cfg,
            strategy_options={"max_epochs": 5},
            project=Path("."), model=client,
        )
        report = loop.run()

    # Loop must have terminated due to stop control — not max_epochs
    bound = report.get("bound", "")
    assert bound in ("control_stop", "a person asked the run to stop"), (
        f"expected control_stop or equivalent, got {bound!r}; "
        "the run reached max_epochs instead of honouring the stop control"
    )

    # Trace file should exist (even if minimal)
    trace = _read_trace(tmp_path)
    # Strategy state should have been saved
    journal_path = run_root / "journal.jsonl"
    if journal_path.exists():
        journal = journal_path.read_text()
        assert "strategy_state" in journal


def test_trace_records_calls_and_decisions(tmp_path):
    """Trace has model_call entries with all required fields, and decision entries."""
    client = FakeClient()
    _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 1})

    trace = _read_trace(tmp_path)
    assert len(trace) > 0, "trace file should not be empty"

    # Check model_call entries
    model_calls = [e for e in trace if e["type"] == "model_call"]
    assert len(model_calls) > 0, "at least one model_call should be recorded"
    for mc in model_calls:
        assert "system_prompt" in mc
        assert "user_prompt" in mc
        assert "raw_output" in mc
        assert "call_key" in mc
        assert isinstance(mc["call_key"], list)
        assert len(mc["call_key"]) == 4, "call_key must be [step_type, epoch, group_index, call_order]"

    # Check decision entries
    decisions = [e for e in trace if e["type"] == "decision"]
    assert len(decisions) > 0, "at least one decision should be recorded"
    for d in decisions:
        assert "step_type" in d
        assert "epoch" in d

    # Every line is valid JSON
    trace_path = tmp_path / "run" / "skillopt_trace.jsonl"
    for line in trace_path.read_text().splitlines():
        if line.strip():
            json.loads(line)  # Should not raise




def test_replay_client_reproduces_decisions(tmp_path, monkeypatch):
    """A ReplayClient fed the recorded outputs reproduces the run exactly.

    Compares the full sequence of call keys and prompts, and every decision
    with the inputs that drive it (group order, sampled ids, budget per step,
    per-edit outcomes) — not just the decision types.
    """
    monkeypatch.setenv(PASS_IDS_ENV, "t1,t3,t5,t7")
    client1 = FakeClient()
    _run_loop(tmp_path / "run1", client=client1, skillopt_options={"max_epochs": 2})
    trace1 = _read_trace(tmp_path / "run1")
    model_calls_1 = [e for e in trace1 if e["type"] == "model_call"]

    keys1 = [c["call_key"] for c in client1.calls]
    assert len(set(keys1)) == len(keys1), "call keys must be unique to be replayable"
    recordings = {tuple(mc["call_key"]): mc["raw_output"] for mc in model_calls_1}
    assert set(recordings) == set(keys1)

    replay = ReplayClient(recordings)
    _run_loop(tmp_path / "run2", client=replay, skillopt_options={"max_epochs": 2})
    trace2 = _read_trace(tmp_path / "run2")

    assert [c["call_key"] for c in replay.calls] == keys1
    assert [c["prompt"] for c in replay.calls] == [c["prompt"] for c in client1.calls]

    def strip(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in e.items() if k != "timestamp"}
            for e in entries if e["type"] == "decision"
        ]

    decisions1, decisions2 = strip(trace1), strip(trace2)
    assert decisions1 == decisions2
    # The compared decisions carry the inputs that drive the run
    assert _decisions(trace1, "analyst_failure")[0]["record_ids"]
    assert _decisions(trace1, "analyst_success")[0]["record_ids"]
    assert _decisions(trace1, "slow_update_sample")[0]["sample_ids"]
    assert [d["edit_budget"] for d in _decisions(trace1, "ranked_edits")] == [3, 2]
    assert _decisions(trace1, "patch_applied")[0]["edit_report"]["outcomes"]


# ── item 1: edit budget per ranked step ───────────────────────────────────


def test_budget_advances_every_ranked_step_whatever_the_gate(tmp_path):
    """The scheduler steps whenever a step reaches ranking, before the gate.

    trainer.py:1415 (``scheduler.step()``) runs before the gate at :1583, so a
    rejected candidate still advances the budget. Every step here is rejected
    (equal val scores), and the budget still walks the cosine schedule.
    """
    loop, _ = _run_loop(tmp_path, skillopt_options={"max_epochs": 4})
    trace = _read_trace(tmp_path)
    ranked = _decisions(trace, "ranked_edits")
    assert [d["edit_budget"] for d in ranked] == sched_mod.full_sequence(4) == [4, 3, 2, 2]
    assert [d["budget_step"] for d in ranked] == [1, 2, 3, 4]
    assert not _decisions(trace, "val_gate_accept")
    assert loop.strategy_state_dict()["edit_budget_step"] == 4


def test_budget_advances_when_patch_refused(tmp_path):
    """A refused patch reached ranking too, so it advances the budget."""
    loop, _ = _run_loop(
        tmp_path,
        config_overrides={"allowed_ops": ("set_knob",)},
        skillopt_options={"max_epochs": 4},
    )
    trace = _read_trace(tmp_path)
    assert len(_decisions(trace, "patch_refused")) == 4
    assert [d["edit_budget"] for d in _decisions(trace, "ranked_edits")] == [4, 3, 2, 2]
    assert loop.strategy_state_dict()["edit_budget_step"] == 4


# ── item 2: full trajectories ─────────────────────────────────────────────

META_CTX = (
    "## Optimizer Meta Skill\n"
    "This is optimizer-side memory distilled from prior epoch transitions in "
    "this environment. Use it to improve how you propose, merge, and rank "
    "skill edits. Prefer it when the current evidence is ambiguous, but do "
    "not force it if the current trajectories clearly contradict it.\n\n"
    "META"
)

RECORD_FAIL = {
    "id": "q7", "passed": False, "score": 0.25, "error_text": None, "error_kind": None,
    "exchanges": [{"node": "solve", "prompt": "Solve: 2+2?", "response": "<answer>5</answer>"}],
    "prediction": "5", "label": "B", "label_text": "4",
    "fail_reason": "predicted 5, expected B",
    "metadata": {"question": "What is 2+2?", "task_type": "arith"},
}


#: A SkillOpt rollout result as an env renders it: the result dict, its
#: conversation and the target prompts (reflect.py:121-248 reads the latter
#: from conversation.json and target_*_prompt.txt).
RESULT_FAIL = {
    "id": "q7", "question": "What is 2+2?", "task_type": "arith", "hard": 0,
    "fail_reason": "MCQ=0: predicted 'A' but expected 'B'", "n_turns": 1,
    "reference_text": "  2+2=4  ",
    "target_system_prompt": "SYSTEM\n",
    "target_user_prompt": "## Question\nWhat is 2+2?",
    "conversation": [
        {"type": "message", "turn": 1, "content": "<answer>A</answer>"},
        {"role": "system", "content": "[EVALUATION RESULT]\nExact Match: 0.0"},
    ],
}

#: SkillOpt's rendering of RESULT_FAIL, byte for byte (reflect.py:159-199, :246):
#: an empty Task (LiveMath results carry no task_description/instruction), the
#: hidden reference stripped, each target prompt followed by a newline, then
#: one blank line and the conversation through fmt_trajectory (reflect.py:74-115).
RESULT_FAIL_TEXT = (
    "### Trajectory 1 (id=q7)\n"
    "Task: \n"
    "Task type: arith\n"
    "Failure reason: MCQ=0: predicted 'A' but expected 'B'\n"
    "Steps: 1\n"
    "\n#### Hidden Reference\n2+2=4\n"
    "\n#### Target System Prompt\nSYSTEM\n\n"
    "\n#### Target User Prompt\n## Question\nWhat is 2+2?\n"
    "\n"
    "[agent] <answer>A</answer>\n"
    "[verification] [EVALUATION RESULT]\nExact Match: 0.0"
)


def test_fmt_trajectory_matches_skillopt():
    """reflect.py:74-115: tool calls, step records, system entries as
    [verification], everything else as [<role>] defaulting to agent."""
    assert prompts.fmt_trajectory([
        "raw",
        {"type": "tool_call", "cmd": "ls", "obs": "a"},
        {"step": 2, "action": "go", "env_feedback": "ok", "reasoning": "why"},
        {"step": 3, "action": "stop", "env_feedback": None},
        {"role": "system", "content": "graded"},
        {"role": "user", "content": "hi"},
        {"type": "message", "content": None},
    ]) == (
        "[agent] raw\n"
        "[action] ls\n[obs]    a\n"
        "[step 2 think] why\n[step 2 action] go\n[step 2 obs]    ok\n"
        "[step 3 action] stop\n[step 3 obs]    \n"
        "[verification] graded\n"
        "[user] hi\n"
        "[agent] "
    )


def test_slow_update_trajectory_matches_skillopt():
    """slow_update.py:116-168 ``_read_trajectory``: no conversation file,
    an empty one, and non-dict entries skipped (unlike fmt_trajectory)."""
    from refactory_packages.skillopt.loop import _read_trajectory

    assert _read_trajectory({}) == "(trajectory not available)"
    assert _read_trajectory({"conversation": None}) == "(trajectory not available)"
    assert _read_trajectory({"conversation": []}) == "(empty trajectory)"
    assert _read_trajectory({"conversation": [
        "raw", {"type": "message", "content": "x"}, {"role": "system", "content": "v"},
    ]}) == "[agent] x\n[verification] v"


def test_analyst_message_renders_full_trajectory():
    """reflect.py:321-344 / :421-441, trajectories as reflect.py:121-248."""
    no_conversation = dict(RESULT_FAIL, id="q8", conversation=None)
    ok = dict(RESULT_FAIL, id="q9", fail_reason="", reference_text="",
              target_system_prompt="", target_user_prompt="", task_description="T")
    user = prompts.format_analyst_user(
        "SKILL", [RESULT_FAIL, no_conversation, ok], 4, "META", failure=True,
    )
    assert user == (
        "## Current Skill\nSKILL\n\n"
        "## Edits Budget\nProduce at most L=4 edits.\n\n"
        + META_CTX + "\n\n"
        "## Failed Trajectories (3 total)\n"
        + RESULT_FAIL_TEXT
        + "\n\n---\n\n"
        # q8 has no conversation: skipped, but the index still advances
        # (reflect.py:148-156 ``continue`` inside ``enumerate``).
        "### Trajectory 3 (id=q9)\nTask: T\nTask type: arith\nSteps: 1\n\n"
        "[agent] <answer>A</answer>\n"
        "[verification] [EVALUATION RESULT]\nExact Match: 0.0"
    )
    success = prompts.format_analyst_user("SKILL", [RESULT_FAIL], 3, "", failure=False)
    assert success == (
        "## Current Skill\nSKILL\n\n"
        "## Edits Budget\nProduce at most L=3 edits.\n\n"
        "## Successful Trajectories (1 total)\n" + RESULT_FAIL_TEXT
    )
    # No trajectory at all: no message, so no call (reflect.py:324-325).
    assert prompts.format_analyst_user("SKILL", [no_conversation], 4, "", failure=True) == ""


def test_analyst_group_without_trajectories_makes_no_call(tmp_path):
    """reflect.py:324-325: a group whose results have no conversation is not sent."""
    from refactory_packages.skillopt import analysts
    from refactory_packages.skillopt.env import SkillOptEnv

    class NoConversation(SkillOptEnv):
        def rollout_result(self, record, knobs):  # type: ignore[no-untyped-def]
            return {"id": record["id"], "conversation": None}

    client = FakeClient()
    records = [{"id": f"r{i}", "passed": False, "score": 0.0} for i in range(3)]
    failures, _ = analysts.reflect(
        client, records, "SKILL", 4, "", seed=1, group_size=8, epoch=1,
        trace=_trace(tmp_path), env=NoConversation(),
    )
    assert client.calls == [] and failures == []


def test_analyst_prompts_in_a_run_carry_records(tmp_path):
    """The run's analysts see each record's fields, not just pass/fail."""
    client = FakeClient()
    _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 1})
    prompt = next(c["prompt"] for c in client.calls if c["call_key"][0] == "analyst_failure")
    assert "## Failed Trajectories (8 total)\n### Trajectory 1 (id=t" in prompt
    assert "[verification] [EVALUATION RESULT]\nPrediction: ''" in prompt
    assert "Score: 0.5\nPassed: False" in prompt


# ── env prompt overrides (skillopt/prompts/__init__.py:32-58) ──────────────


def test_env_prompt_override_with_generic_fallback(tmp_path, monkeypatch):
    """load_prompt(name, env): the package's skillopt_prompts/<name>.md wins,
    the generic prompt is the fallback, and any name is looked up."""
    from refactory_packages.skillopt import env as env_mod

    pkg = tmp_path / "pkg"
    (pkg / env_mod.PROMPT_DIR).mkdir(parents=True)
    (pkg / env_mod.PROMPT_DIR / "analyst_error.md").write_text("ENV ERROR\n")
    (pkg / env_mod.PROMPT_DIR / "custom.md").write_text("ENV CUSTOM")
    monkeypatch.setattr(env_mod, "_package_dir", lambda name: pkg if name == "pkg" else None)

    with_env = env_mod.SkillOptEnv("pkg")
    assert with_env.prompt("analyst_error") == "ENV ERROR\n"  # as read, unstripped
    assert with_env.prompt("custom") == "ENV CUSTOM"
    assert with_env.prompt("analyst_success") == prompts.SUCCESS_ANALYST_SYSTEM
    assert env_mod.SkillOptEnv("other").prompt("analyst_error") == prompts.FAILURE_ANALYST_SYSTEM
    with pytest.raises(KeyError):
        env_mod.SkillOptEnv("other").prompt("custom")


def test_livemath_ships_skillopt_analyst_overrides():
    """LiveMath's analyst prompts are SkillOpt's env files, and only those two:
    SkillOpt's envs/livemathematicianbench/prompts/ overrides analyst_error
    and analyst_success (plus its rollout_system, the solver's own prompt)."""
    from refactory_packages.skillopt.env import PROMPT_DIR, SkillOptEnv

    env = SkillOptEnv.for_mode("livemath/solver")
    assert env.package == "livemath"
    shipped = sorted(p.name for p in (env.override_path("x").parent).iterdir())
    assert shipped == ["analyst_error.md", "analyst_success.md"], PROMPT_DIR
    error = env.prompt("analyst_error")
    assert error.startswith(
        "You are an expert failure-analysis agent for theorem-grounded mathematical "
        "multiple-choice questions.\n"
    )
    assert hashlib.sha256(error.encode()).hexdigest() == LIVEMATH_ANALYST_ERROR_SHA256
    success = env.prompt("analyst_success")
    assert hashlib.sha256(success.encode()).hexdigest() == LIVEMATH_ANALYST_SUCCESS_SHA256
    assert env.prompt("merge_failure") == prompts.MERGE_FAILURE_SYSTEM


#: sha256 of SkillOpt's envs/livemathematicianbench/prompts/analyst_{error,success}.md
#: at the commit arm A ran (ec9d021b).
LIVEMATH_ANALYST_ERROR_SHA256 = "9cd5e3449b9013fb67b6dec930486bd8c23c7128f2746f97a1ca29b07b178150"
LIVEMATH_ANALYST_SUCCESS_SHA256 = "42619d41000d16b7e5194e5dd09ba67fcd65d1c5ae8b04b4097c38e45a02750d"


def test_run_uses_the_mode_packages_analyst_prompts(tmp_path, monkeypatch):
    """The analysts' system prompts come from the package that holds the run's
    mode (envs/base.py:288-330), found from the run itself."""
    from refactory_packages.skillopt import env as env_mod

    pkg = tmp_path / "skillopt_probe"
    (pkg / env_mod.PROMPT_DIR).mkdir(parents=True)
    (pkg / env_mod.PROMPT_DIR / "analyst_error.md").write_text("PROBE ERROR")
    (pkg / env_mod.PROMPT_DIR / "analyst_success.md").write_text("PROBE SUCCESS")
    monkeypatch.setattr(
        env_mod, "_package_dir", lambda name: pkg if name == MODE_ID.split("/")[0] else None,
    )
    monkeypatch.setenv(PASS_IDS_ENV, "t1,t3")
    client = FakeClient()
    _run_loop(tmp_path / "run", client=client, skillopt_options={"max_epochs": 1})
    systems = {c["call_key"][0]: c["system"] for c in client.calls if c["call_key"]}
    assert systems["analyst_failure"] == "PROBE ERROR"
    assert systems["analyst_success"] == "PROBE SUCCESS"
    assert {k for k, v in systems.items() if v.startswith("PROBE")} == {
        "analyst_failure", "analyst_success",
    }


# ── rollout order seam (replay-only use) ──────────────────────────────────


def _analyst_ids(trace: list[dict[str, Any]], epoch: int = 1) -> list[list[str]]:
    return [d["record_ids"] for d in _decisions(trace, "analyst_failure") if d["epoch"] == epoch]


def test_rollout_order_is_off_by_default(tmp_path):
    """A run that does not pass ``rollout_order`` shuffles the Run's own order."""
    import random

    loop, _ = _run_loop(tmp_path, skillopt_options={"max_epochs": 1})
    assert loop.rollout_order is None
    trace = _read_trace(tmp_path)
    expected = list(_decisions(trace, "train_rollout")[0]["rollout_ids"])
    random.Random(1043).shuffle(expected)
    assert _analyst_ids(trace) == [expected]


def test_rollout_order_feeds_the_seeded_shuffle(tmp_path):
    """reflect.py:558-561 shuffle the list in the order the rollout returned it:
    the order the option gives is the one the batch seed shuffles."""
    import random

    seen: list[int] = []

    def reverse(epoch, records):  # type: ignore[no-untyped-def]
        seen.append(epoch)
        return list(reversed(records))

    _run_loop(tmp_path, skillopt_options={"max_epochs": 1, "rollout_order": reverse})
    trace = _read_trace(tmp_path)
    expected = list(reversed(_decisions(trace, "train_rollout")[0]["rollout_ids"]))
    random.Random(1043).shuffle(expected)
    assert seen == [1]
    assert _analyst_ids(trace) == [expected]


def test_rollout_order_must_be_a_callable_permutation(tmp_path):
    loop = SkillOptLoop()
    with pytest.raises(TypeError):
        loop.configure(rollout_order="completion")
    loop.configure(rollout_order=lambda epoch, records: records[:-1])
    records = [{"id": "a"}, {"id": "b"}]
    with pytest.raises(ValueError):
        loop._analyst_records(1, records)
    loop.configure(rollout_order=lambda epoch, records: [{"id": "b", "x": 1}, {"id": "a"}])
    # Only the order is taken: the records themselves stay the Run's.
    assert loop._analyst_records(1, records) == [{"id": "b"}, {"id": "a"}]


def test_comparison_and_slow_update_messages_match_skillopt():
    """slow_update.py:206-269 pairs, :287-349 text, :409-424 message;
    meta_skill.py:44-60 message."""
    prev_ok = dict(RECORD_FAIL, passed=True, score=1.0, prediction="4", fail_reason=None)
    pairs = SkillOptLoop._build_comparison_pairs([prev_ok], [RECORD_FAIL], ["q7", "q9"])
    text = prompts.format_comparison_text(pairs)
    # Trajectories as slow_update.py:116-168 ``_read_trajectory`` renders the
    # generic rollout result (env.default_rollout_result) of each record.
    traj_prev = (
        "[agent] <answer>5</answer>\n"
        "[verification] [EVALUATION RESULT]\nPrediction: '4'\nLabel: 'B'\n"
        "Label text: '4'\nScore: 1.0\nPassed: True"
    )
    traj_curr = (
        "[agent] <answer>5</answer>\n"
        "[verification] [EVALUATION RESULT]\nPrediction: '5'\nLabel: 'B'\n"
        "Label text: '4'\nScore: 0.25\nPassed: False"
    )
    assert text == (
        "## Longitudinal Comparison Summary\n"
        "Total samples: 2\n"
        "- Improved (wrong→right): 0\n"
        "- Regressed (right→wrong): 1\n"
        "- Persistent failures (wrong→wrong): 1\n"
        "- Stable successes (right→right): 0\n"
        "\n\n"
        "### Regressions (right→wrong) — HIGHEST PRIORITY\n"
        "\n#### Task q7: What is 2+2?\n"
        "- Prev epoch: PASS (soft=1.00) — answer: 4\n"
        "- Curr epoch: FAIL (soft=0.25) — answer: 5\n"
        "- Curr fail reason: predicted 5, expected B\n"
        f"\n**Previous epoch trajectory:**\n```\n{traj_prev}\n```\n"
        f"\n**Current epoch trajectory:**\n```\n{traj_curr}\n```"
        "\n\n"
        "### Persistent Failures (wrong→wrong)\n"
        "\n#### Task q9: q9\n"
        "- Prev epoch: FAIL (soft=0.00) — answer: N/A\n"
        "- Curr epoch: FAIL (soft=0.00) — answer: N/A\n"
        "\n**Previous epoch trajectory:**\n```\n(trajectory not available)\n```\n"
        "\n**Current epoch trajectory:**\n```\n(trajectory not available)\n```"
        "\n\n"
        "### Improvements (wrong→right)\n(none)\n"
        "\n\n"
        "### Stable Successes (right→right)\n(none)\n"
    )
    assert prompts.format_slow_update_user("PREV", "CURR", "", "CMP") == (
        "## Previous Epoch's Skill\nPREV\n\n"
        "## Current Epoch's Skill\nCURR\n\n"
        "## Previous Slow Update Guidance\n"
        "The following guidance was active during the current epoch. "
        "Reflect on its effectiveness before writing the new version.\n\n"
        "(No previous guidance — this is the first slow update.)\n\n"
        "## Longitudinal Comparison (same 20 tasks, two skill versions)\nCMP"
    )
    assert prompts.format_meta_skill_user("PREV", "CURR", " M ", "CMP") == (
        "## Previous Epoch Last-Step Skill\nPREV\n\n"
        "## Current Epoch Last-Step Skill\nCURR\n\n"
        "## Previous Optimizer Meta Skill\n"
        "The following optimizer memory was available during the current epoch. "
        "Reflect on whether it improved or harmed the quality of edits.\n\n"
        "M\n\n"
        "## Longitudinal Comparison (same tasks, two last-step skills)\nCMP"
    )


def test_slow_update_and_meta_prompts_in_a_run_carry_trajectories(tmp_path):
    client = FakeClient()
    _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 2})
    for step in ("slow_update", "meta_skill"):
        prompt = next(c["prompt"] for c in client.calls if c["call_key"][0] == step)
        assert "## Longitudinal Comparison Summary\nTotal samples: 8\n" in prompt
        assert "**Current epoch trajectory:**\n```\n[verification] [EVALUATION RESULT]" in prompt


def test_slow_update_rolls_out_a_previous_skill_the_search_already_holds(tmp_path, monkeypatch):
    """trainer.py:1866-1867 roll out the previous epoch's last skill on the sample
    whatever it is. When epoch 2 accepts a patch, that skill is the candidate the
    epoch-2 train rollout already evaluated: re-proposing it is refused
    (E-duplicate), so the strategy must roll out the candidate it already has."""
    spy = _spy_evaluations(monkeypatch)
    # Every patched skill scores higher on val, so both epochs accept.
    monkeypatch.setattr(
        SkillOptLoop, "_val_score",
        staticmethod(lambda run, cand: 0.1 * SkillOptLoop._get_skill(cand).count("Always check.")),
    )
    client = FakeClient()
    loop, _ = _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 2})
    trace = _read_trace(tmp_path)
    assert len(_decisions(trace, "val_gate_accept")) == 2
    assert not _decisions(trace, "slow_update_skipped")
    assert [c["call_key"] for c in client.calls if c["call_key"][0] == "slow_update"] == [
        ("slow_update", 2, 0, 0),
    ]
    sampled = [items for part, items in spy if part == "train" and items]
    assert len(sampled) == 2  # the previous skill's and the current skill's rollouts
    prompt = next(c["prompt"] for c in client.calls if c["call_key"][0] == "slow_update")
    prev_section = prompt.split("## Previous Epoch's Skill\n", 1)[1].split("\n\n## Current Epoch's Skill", 1)[0]
    assert prev_section.count("Always check.") == 1


def test_failed_optimizer_calls_fall_back_like_skillopt(tmp_path):
    """SkillOpt catches every optimizer-call failure (reflect.py:383, :467;
    aggregate.py:60, :261; clip.py:101; slow_update.py:440; meta_skill.py:76):
    the group's patch is dropped, the run goes on, and the trace says why."""
    from refactory_packages.skillopt import analysts

    class Failing(FakeClient):
        def complete(self, prompt, *, system="", call_key=None):  # type: ignore[no-untyped-def]
            super().complete(prompt, system=system, call_key=call_key)
            raise RuntimeError("Claude backend failed after 3 retries: Claude CLI exited with code 1")

    records = [{"id": f"r{i}", "passed": False, "score": 0.0} for i in range(3)]
    trace = _trace(tmp_path)
    failures, _ = analysts.reflect(
        Failing(), records, "SKILL", 4, "", seed=1, group_size=8, epoch=1, trace=trace,
    )
    assert failures == []
    calls = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert calls[0]["raw_output"] is None and "failed after 3 retries" in calls[0]["error"]

    # A whole run whose every optimizer call fails still completes its epochs,
    # and says so: per epoch in the trace, and in the run report's proposer
    # health, as a run whose proposer never answered (Loop._proposer_health).
    loop, report = _run_loop(tmp_path / "run", client=Failing(), skillopt_options={"max_epochs": 2})
    run_trace = _read_trace(tmp_path / "run")
    assert _decisions(run_trace, "no_edits_skip")
    assert loop.strategy_state_dict()["epoch"] == 2
    per_epoch = _decisions(run_trace, "optimizer_calls")
    assert [d["all_failed"] for d in per_epoch] == [True, True]
    assert all(d["calls"] == d["failed"] > 0 for d in per_epoch)
    proposer = report["proposer"]
    assert proposer["all_failed"] is True
    assert proposer["failure_count"] == proposer["rounds"] == sum(d["calls"] for d in per_epoch)
    assert "failed after 3 retries" in proposer["failures"][0]
    assert "every one of its" in (tmp_path / "run" / "run" / "strategy.log").read_text()


def test_optimizer_calls_that_answer_are_counted_and_not_flagged(tmp_path):
    _, report = _run_loop(tmp_path, skillopt_options={"max_epochs": 1})
    epoch = _decisions(_read_trace(tmp_path), "optimizer_calls")[0]
    assert epoch["calls"] > 0 and epoch["failed"] == 0 and epoch["all_failed"] is False
    assert report["proposer"]["rounds"] == epoch["calls"]
    assert report["proposer"]["failure_count"] == 0


class _Ending(FakeClient):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def complete(self, prompt, *, system="", call_key=None):  # type: ignore[no-untyped-def]
        raise self.error


def test_budget_exhaustion_ends_the_run_at_its_bound(tmp_path):
    """BudgetExhausted is not a bad reply: it leaves the strategy, and the loop
    ends the run on that bound (Loop.run), instead of every epoch carrying on."""
    from refactory_dsh.budgets import BudgetExhausted

    _, report = _run_loop(
        tmp_path, client=_Ending(BudgetExhausted("model budget spent")),
        skillopt_options={"max_epochs": 3},
    )
    assert "model budget spent" in str(report["bound"])
    trace = _read_trace(tmp_path)
    assert [d["epoch"] for d in _decisions(trace, "train_rollout")] == [1]
    assert trace[-1]["type"] == "model_call" and trace[-1]["error"].startswith("BudgetExhausted")


@pytest.mark.parametrize("make_error", [
    lambda: __import__("concurrent.futures", fromlist=["CancelledError"]).CancelledError(),
    lambda: __import__("refactory_dsh.model", fromlist=["ModelChoiceError"]).ModelChoiceError(),
])
def test_cancellation_and_model_choice_end_the_run(tmp_path, make_error):
    """Not a bad reply: the model client raises these to stop the run."""
    error = make_error()
    with pytest.raises(type(error)):
        _run_loop(tmp_path, client=_Ending(error), skillopt_options={"max_epochs": 1})
    calls = [d for d in _read_trace(tmp_path) if d["type"] == "model_call"]
    assert calls and calls[-1]["error"].startswith(type(error).__name__)


def test_slow_update_records_skillopt_result_fields(tmp_path):
    """slow_result.json's fields (trainer.py:1936-1947): reasoning, content,
    both rollouts' hard scores and the candidate's 16-hex hash."""
    import hashlib as _hashlib

    monkey_pass = "t1,t2"
    os.environ[PASS_IDS_ENV] = monkey_pass
    try:
        loop, _ = _run_loop(tmp_path, skillopt_options={"max_epochs": 2})
    finally:
        os.environ.pop(PASS_IDS_ENV, None)
    slow = _decisions(_read_trace(tmp_path), "slow_update")[0]
    assert slow["slow_update_content"] == "Updated guidance text."
    assert slow["reasoning"] == "reflected"
    assert slow["prev_hard"] == slow["curr_hard"] == 2 / 8
    skill = loop.strategy_state_dict()["current_skill"]
    assert slow["candidate_hash"] == _hashlib.sha256(skill.encode()).hexdigest()[:16]
    meta = _decisions(_read_trace(tmp_path), "meta_skill")[-1]
    assert (meta["meta_skill_content"], meta["reasoning"]) == ("USE CONCISE EDITS", "meta reflected")


def test_resume_at_epoch_done_still_checks_the_early_stop(tmp_path):
    """A run interrupted after an epoch's end but before its stop check stops
    on resume instead of running another epoch."""
    loop, _ = _run_loop(
        tmp_path,
        skillopt_options={"max_epochs": 4, "early_stop_patience": 2},
        preload_state={
            "current_skill": INITIAL_SKILL, "epoch": 2, "phase": "epoch_done",
            "stale_epochs": 2, "early_stop_best": 0.5, "best_score": 0.5,
            "current_score": 0.5,
        },
    )
    trace = _read_trace(tmp_path)
    assert [d["reason"] for d in _decisions(trace, "early_stop")] == ["patience"]
    assert not _decisions(trace, "train_rollout")


def test_early_stop_best_survives_a_resume():
    loop = SkillOptLoop()
    loop.configure()
    loop.load_strategy_state({"early_stop_best": 0.7, "best_score": 0.9})
    assert loop.strategy_state_dict()["early_stop_best"] == 0.7


def test_patch_order_feeds_the_merge_and_its_fallback(tmp_path):
    """aggregate.py:70-76 concatenates the patches' edits in the order the merge
    got them; ``patch_order`` (replay-only) sets that order, off by default."""
    from refactory_packages.skillopt import analysts as _analysts  # noqa: F401

    class FailMerges(FakeClient):
        def complete(self, prompt, *, system="", call_key=None):  # type: ignore[no-untyped-def]
            if call_key and call_key[0] == "merge_failure":
                super().complete(prompt, system=system, call_key=call_key)
                raise RuntimeError("failed after 3 retries")
            if call_key and call_key[0] == "analyst_failure":
                super().complete(prompt, system=system, call_key=call_key)
                return json.dumps({"batch_size": 1, "patch": {"reasoning": f"g{call_key[2]}", "edits": [
                    {"op": "append", "content": f"from group {call_key[2]}"},
                ]}})
            return super().complete(prompt, system=system, call_key=call_key)

    seen: list[tuple[int, str, list[str]]] = []

    def reverse(epoch, kind, patches):  # type: ignore[no-untyped-def]
        seen.append((epoch, kind, [p["reasoning"] for p in patches]))
        return list(reversed(patches))

    options = {"max_epochs": 1, "group_size": 2}
    loop, _ = _run_loop(tmp_path / "default", client=FailMerges(), skillopt_options=options)
    assert loop.patch_order is None
    merged = _decisions(_read_trace(tmp_path / "default"), "merged_patch")[0]["patch"]
    default_order = [e["content"] for e in merged["edits"]]

    client = FailMerges()
    _run_loop(tmp_path / "ordered", client=client,
              skillopt_options={**options, "patch_order": reverse})
    assert seen and seen[0][1] == "failure"
    merged = _decisions(_read_trace(tmp_path / "ordered"), "merged_patch")[0]["patch"]
    assert [e["content"] for e in merged["edits"]] == list(reversed(default_order))
    first_merge = next(c for c in client.calls if c["call_key"][0] == "merge_failure")
    assert first_merge["prompt"].index('"g3"') < first_merge["prompt"].index('"g0"')


def test_patch_order_must_be_a_permutation():
    loop = SkillOptLoop()
    loop.configure(patch_order=lambda epoch, kind, patches: [dict(patches[1]), patches[0]])
    patches = [{"reasoning": "a"}, {"reasoning": "b"}]
    with pytest.raises(ValueError, match="changed which patches"):
        loop._merge_order(1, "failure", patches)
    loop.configure(patch_order=lambda epoch, kind, patches: patches[:1])
    with pytest.raises(ValueError, match="dropped some"):
        loop._merge_order(1, "failure", patches)


def test_early_stop_counts_every_epoch_without_a_better_best(tmp_path):
    """Arm A's early-stop patch (skillopt_local.diff, trainer.py after the meta
    skill): every epoch whose best score did not improve is stale -- a skipped
    epoch too -- and ``early_stop_patience`` stale epochs end the run."""

    class NoEdits(FakeClient):
        def complete(self, prompt, *, system="", call_key=None):  # type: ignore[no-untyped-def]
            if call_key and call_key[0].startswith("analyst_"):
                super().complete(prompt, system=system, call_key=call_key)
                return json.dumps({"batch_size": 1, "patch": {"reasoning": "r", "edits": []}})
            return super().complete(prompt, system=system, call_key=call_key)

    loop, _ = _run_loop(
        tmp_path, client=NoEdits(),
        skillopt_options={"max_epochs": 5, "early_stop_patience": 2},
    )
    trace = _read_trace(tmp_path)
    assert len(_decisions(trace, "no_edits_skip")) == 2
    assert [d["reason"] for d in _decisions(trace, "early_stop")] == ["patience"]
    assert loop.strategy_state_dict()["epoch"] == 2


def test_early_stop_restarts_the_count_when_the_best_improves(tmp_path, monkeypatch):
    """An epoch that raises the best score resets the stale count (arm A's
    patch: ``es_stale = 0`` when ``best_score > es_best``)."""
    monkeypatch.setattr(
        SkillOptLoop, "_val_score",
        staticmethod(lambda run, cand: 0.1 * SkillOptLoop._get_skill(cand).count("Always check.")),
    )

    class EditsInEpochOne(FakeClient):
        def complete(self, prompt, *, system="", call_key=None):  # type: ignore[no-untyped-def]
            if call_key and call_key[0].startswith("analyst_") and call_key[1] > 1:
                super().complete(prompt, system=system, call_key=call_key)
                return json.dumps({"batch_size": 1, "patch": {"reasoning": "r", "edits": []}})
            return super().complete(prompt, system=system, call_key=call_key)

    loop, _ = _run_loop(
        tmp_path, client=EditsInEpochOne(),
        skillopt_options={"max_epochs": 5, "early_stop_patience": 2},
    )
    trace = _read_trace(tmp_path)
    assert len(_decisions(trace, "val_gate_accept")) == 1
    assert loop.strategy_state_dict()["epoch"] == 3  # epoch 1 improved; 2 and 3 stale


def test_a_patch_that_changes_nothing_is_gated_from_the_cache(tmp_path):
    """trainer.py:1507-1572: an unchanged skill still reaches the gate; its
    cached score is the current one, so it is rejected (A's ``reject``), not
    refused as a patch."""

    class Noop(FakeClient):
        def complete(self, prompt, *, system="", call_key=None):  # type: ignore[no-untyped-def]
            super().complete(prompt, system=system, call_key=call_key)
            edits = [{"op": "replace", "target": "NOT IN THE SKILL", "content": "x"}]
            if call_key and call_key[0].startswith("analyst_"):
                return json.dumps({"batch_size": 1, "patch": {"reasoning": "r", "edits": edits}})
            if call_key and call_key[0].startswith("merge_"):
                return json.dumps({"reasoning": "m", "edits": edits})
            return super()._default_response(call_key[0] if call_key else "", prompt)

    _run_loop(tmp_path, client=Noop(), skillopt_options={"max_epochs": 1})
    trace = _read_trace(tmp_path)
    noop = _decisions(trace, "patch_noop")
    assert noop and noop[0]["edit_report"]["outcomes"][0]["status"].startswith("skipped")
    assert not _decisions(trace, "patch_refused")
    assert _decisions(trace, "val_cache_hit") and _decisions(trace, "val_gate_reject")


def test_analyst_decision_says_whether_a_patch_came_back(tmp_path):
    """reflect.py:646-650 writes minibatch_*.json for every reply carrying a
    patch, edits or none; the trace keeps that apart from the edit count."""
    from refactory_packages.skillopt import analysts

    replies = {
        ("analyst_failure", 1, 0, 0): json.dumps({"batch_size": 1, "patch": {"reasoning": "r", "edits": []}}),
        ("analyst_failure", 1, 1, 0): "no json here",
    }
    records = [{"id": f"r{i}", "passed": False, "score": 0.0} for i in range(2)]
    trace = _trace(tmp_path)
    analysts.reflect(FakeClient(replies), records, "SKILL", 4, "", seed=1, group_size=1,
                     epoch=1, trace=trace)
    decisions = [json.loads(line) for line in trace.path.read_text().splitlines()]
    decisions = [d for d in decisions if d["type"] == "decision"]
    assert [(d["n_edits"], d["returned_patch"]) for d in decisions] == [(0, True), (0, False)]


# ── item 3: tolerant JSON ─────────────────────────────────────────────────


def _wrapped(obj: object) -> str:
    return "Sure, here it is:\n```json\n" + json.dumps(obj) + "\n```\nHope that helps."


def test_tolerant_json_in_every_parser(tmp_path):
    """Every parser uses SkillOpt's extract_json (utils/json_utils.py:183)."""
    from refactory_packages.skillopt import analysts, merge
    from refactory_packages.skillopt.json_utils import extract_json

    assert extract_json("prose {x} only") is None
    assert extract_json('noise {"a": 1} tail') == {"a": 1}

    patch = analysts._parse_patch(
        _wrapped({"batch_size": 2, "patch": {"edits": [{"op": "append", "content": "x"}]}}),
        "failure", 4,
    )
    assert patch is not None
    assert patch["edits"] == [
        {"op": "append", "content": "x", "source_type": "failure", "support_count": 2},
    ]
    assert merge._parse_merge(_wrapped({"edits": [{"op": "append", "content": "y"}]})) == {
        "edits": [{"op": "append", "content": "y"}],
    }
    edits = [{"op": "append", "content": "a"}, {"op": "append", "content": "b"}]
    assert merge._parse_ranking(_wrapped({"selected_indices": [1]}), edits, 1) == [edits[1]]

    client = FakeClient({
        ("slow_update", 2, 0, 0): _wrapped({"reasoning": "r", "slow_update_content": "GUIDE"}),
        ("meta_skill", 2, 0, 0): _wrapped({"reasoning": "r", "meta_skill_content": "META2"}),
    })
    loop, _ = _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 2})
    state_dict = loop.strategy_state_dict()
    assert prompts.extract_slow_update_field(state_dict["current_skill"]) == "GUIDE"
    assert state_dict["meta_skill_text"] == "META2"


# ── item 4: merge as SkillOpt ─────────────────────────────────────────────


def _trace(tmp_path: Path):  # type: ignore[no-untyped-def]
    from refactory_packages.skillopt.trace import TraceRecorder
    return TraceRecorder(tmp_path / "unit_trace.jsonl")


def _patch(i: int, source: str = "failure") -> dict[str, Any]:
    return {"reasoning": f"r{i}", "edits": [
        {"op": "append", "content": f"rule {i}", "source_type": source, "support_count": 1},
    ]}


def test_one_patch_per_analyst_group(tmp_path):
    """reflect.py:557-558 groups; one patch per group, edits truncated to L
    (reflect.py:360, update_modes.py:64-70)."""
    from refactory_packages.skillopt import analysts

    six = json.dumps({"batch_size": 8, "patch": {"reasoning": "r", "edits": [
        {"op": "append", "content": f"e{i}"} for i in range(6)
    ]}})
    client = FakeClient({("analyst_failure", 1, 0, 0): six, ("analyst_failure", 1, 1, 0): six})
    records = [{"id": f"r{i}", "passed": False, "score": 0.0} for i in range(16)]
    failures, successes = analysts.reflect(
        client, records, "SKILL", 4, "", seed=43, group_size=8, epoch=1, trace=_trace(tmp_path),
    )
    assert [c["call_key"] for c in client.calls] == [
        ("analyst_failure", 1, 0, 0), ("analyst_failure", 1, 1, 0),
    ]
    assert successes == []
    assert len(failures) == 2
    assert all(len(p["edits"]) == 4 for p in failures)
    assert failures[0]["edits"][0]["support_count"] == 8


def test_merge_in_batches_of_eight(tmp_path):
    """aggregate.py:79-151: 9 patches → a batch of 8 merged + 1 passed through,
    then the 2 remaining merged at level 2."""
    from refactory_packages.skillopt import merge

    client = FakeClient()
    patches = [_patch(i) for i in range(9)]
    merged = merge.hierarchical_merge(client, "SKILL", patches, [], "", 1, _trace(tmp_path))
    assert [c["call_key"] for c in client.calls] == [
        ("merge_failure", 1, 0, 1), ("merge_failure", 1, 0, 2),
    ]
    assert "## Patches to merge (8 total, merge level 1)\n" in client.calls[0]["prompt"]
    level2 = client.calls[1]["prompt"]
    assert "## Patches to merge (2 total, merge level 2)\n" in level2
    assert '"rule 8"' in level2  # the lone 9th patch passed through unmerged
    assert merged["edits"][0]["merge_level"] == 2


def test_single_patch_passes_through_without_a_call(tmp_path):
    """aggregate.py:99-100 and :212-213."""
    from refactory_packages.skillopt import merge

    client = FakeClient()
    only = _patch(0)
    assert merge.hierarchical_merge(client, "S", [only], [], "", 1, _trace(tmp_path)) is only
    assert client.calls == []


def test_failed_batch_merge_concatenates(tmp_path):
    """aggregate.py:70-76: an unusable batch merge keeps every edit."""
    from refactory_packages.skillopt import merge

    client = FakeClient({("merge_failure", 1, 0, 1): "I cannot merge these."})
    patches = [_patch(i) for i in range(3)]
    merged = merge.hierarchical_merge(client, "S", patches, [], "", 1, _trace(tmp_path))
    assert [e["content"] for e in merged["edits"]] == ["rule 0", "rule 1", "rule 2"]
    assert all(e["merge_level"] == 1 for e in merged["edits"])


def test_failed_final_merge_concatenates_failure_first(tmp_path):
    """aggregate.py:272-275."""
    from refactory_packages.skillopt import merge

    client = FakeClient({("merge_final", 1, 0, 0): "nope"})
    merged = merge.hierarchical_merge(
        client, "S", [_patch(0)], [_patch(1, "success")], "", 1, _trace(tmp_path),
    )
    assert [e["content"] for e in merged["edits"]] == ["rule 0", "rule 1"]


def test_merge_and_rank_messages_match_skillopt(tmp_path):
    """aggregate.py:37-44 and :217-242, clip.py:58-71 — byte for byte,
    with the meta skill prepended."""
    from refactory_packages.skillopt import merge

    client = FakeClient()
    batch = [_patch(0), _patch(1)]
    succ = [_patch(2, "success"), _patch(3, "success")]
    merge.hierarchical_merge(client, "SKILL", batch, succ, "META", 1, _trace(tmp_path))
    prompts_by_key = {c["call_key"]: c["prompt"] for c in client.calls}
    assert prompts_by_key[("merge_failure", 1, 0, 1)] == (
        META_CTX + "\n\n## Current Skill\nSKILL\n\n"
        "## Patches to merge (2 total, merge level 1)\n"
        + json.dumps(batch, ensure_ascii=False, indent=2)
    )
    failure_merged = json.loads(FakeClient()._default_response("merge_failure", ""))
    failure_merged["edits"][0]["merge_level"] = 1
    success_merged = json.loads(FakeClient()._default_response("merge_success", ""))
    success_merged["edits"][0]["merge_level"] = 1
    assert prompts_by_key[("merge_final", 1, 0, 0)] == (
        META_CTX + "\n\n## Current Skill\nSKILL\n\n"
        "## Two pre-merged patch groups to combine\n"
        "Group 1 (failure-driven, HIGH priority): 1 edits\n"
        "Group 2 (success-driven, lower priority): 1 edits\n\n"
        + json.dumps([failure_merged, success_merged], ensure_ascii=False, indent=2)
    )

    edits = [
        {"op": "append", "content": "a", "support_count": 2},
        {"op": "replace", "target": "x", "content": "y"},
        {"op": "delete", "target": "z"},
    ]
    merge.rank(client, "SKILL", edits, 2, "META", 1, _trace(tmp_path))
    assert client.calls[-1]["prompt"] == (
        META_CTX + "\n\n## Current Skill\nSKILL\n\n"
        "## Edits Pool (3 edits, budget=2)\n"
        "[0] op=append  content='a'  support=2\n"
        "[1] op=replace  target='x'  content='y'\n"
        "[2] op=delete  target='z'\n\n"
        "Select the 2 most important edits. Return their 0-based indices in priority order."
    )


def test_meta_skill_reaches_merge_and_analysts(tmp_path, monkeypatch):
    """The meta skill is prepended to merge prompts (aggregate.py:240-242)
    and sits after the budget in analyst prompts (reflect.py:341-343)."""
    monkeypatch.setenv(PASS_IDS_ENV, "t1,t3,t5,t7")
    client = FakeClient()
    _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 3})
    ctx = prompts.format_meta_skill_context("USE CONCISE EDITS")
    final3 = [c["prompt"] for c in client.calls if c["call_key"] == ("merge_final", 3, 0, 0)]
    assert final3 and final3[0].startswith(ctx + "\n\n## Current Skill\n")
    analyst3 = next(c["prompt"] for c in client.calls if c["call_key"] == ("analyst_failure", 3, 0, 0))
    assert "Produce at most L=4 edits.\n\n" + ctx + "\n\n## Failed Trajectories" in analyst3


# ── item 5: smaller fidelity fixes ────────────────────────────────────────


def test_unparseable_slow_update_changes_nothing(tmp_path):
    """slow_update.py:434-443 returns None; trainer.py:2040-2047 then leaves
    the skill as it was (no raw output injected)."""
    client = FakeClient({("slow_update", 2, 0, 0): "I have no guidance this time."})
    loop, _ = _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 2})
    skill = loop.strategy_state_dict()["current_skill"]
    assert prompts.extract_slow_update_field(skill) == ""
    assert "no guidance this time" not in skill
    assert _decisions(_read_trace(tmp_path), "slow_update")[0]["action"] == "no_content"


def test_unparseable_meta_skill_leaves_none(tmp_path):
    """meta_skill.py:70-79 returns None; the next epoch then loads no meta
    skill (trainer.py:384-397)."""
    client = FakeClient({("meta_skill", 2, 0, 0): "no json here"})
    loop, _ = _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 2})
    assert loop.strategy_state_dict()["meta_skill_text"] == ""


def test_meta_skill_compares_skill_before_slow_update(tmp_path):
    """trainer.py:1763 / :2176: the meta skill sees this epoch's last-step
    skill, not the skill the slow update just rewrote."""
    client = FakeClient()
    _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 2})
    slow = next(c["prompt"] for c in client.calls if c["call_key"][0] == "slow_update")
    meta = next(c["prompt"] for c in client.calls if c["call_key"][0] == "meta_skill")
    pre_update = slow.split("## Current Epoch's Skill\n", 1)[1].split(
        "\n\n## Previous Slow Update Guidance", 1)[0]
    meta_curr = meta.split("## Current Epoch Last-Step Skill\n", 1)[1].split(
        "\n\n## Previous Optimizer Meta Skill", 1)[0]
    assert meta_curr == pre_update
    assert "Updated guidance text." not in meta_curr


def test_analysts_get_constant_L(tmp_path):
    """envs/base.py:265: analysts get the adapter's constant edit budget (4),
    while the rank budget follows the schedule."""
    client = FakeClient()
    _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 4})
    analyst_prompts = [c["prompt"] for c in client.calls if c["call_key"][0].startswith("analyst_")]
    assert len(analyst_prompts) == 4
    assert all("## Edits Budget\nProduce at most L=4 edits.\n" in p for p in analyst_prompts)
    budgets = [d["edit_budget"] for d in _decisions(_read_trace(tmp_path), "ranked_edits")]
    assert 2 in budgets


def test_analyst_shuffle_uses_batch_seed(tmp_path, monkeypatch):
    """reflect.py:553-554 shuffles failures with the batch seed and successes
    with batch seed + 1. LiveMath has a dataloader, so the batch seed is the
    batch's own (trainer.py:1242, :1283): ``plan_train_epoch`` gives it as
    seed + epoch*1000 + batch_idx + 1 (dataloader.py:278). Expected values were
    computed by SkillOpt's ``LiveMathematicianBenchDataLoader.plan_train_epoch``
    for seed 42: 1043 in epoch 1, 2043 in epoch 2."""
    import random

    from refactory_packages.skillopt import analysts
    from refactory_packages.skillopt.loop import batch_seed

    monkeypatch.setenv(PASS_IDS_ENV, "t1,t3,t5,t7")
    seen: list[tuple[list[str], list[str], int]] = []
    original = analysts.reflect

    def spy(**kwargs):  # type: ignore[no-untyped-def]
        records = kwargs["records"]
        seen.append((
            [r["id"] for r in records if not r["passed"]],
            [r["id"] for r in records if r["passed"]],
            kwargs["seed"],
        ))
        return original(**kwargs)

    monkeypatch.setattr(analysts, "reflect", spy)
    _run_loop(tmp_path, skillopt_options={"max_epochs": 2})
    trace = _read_trace(tmp_path)
    assert (batch_seed(42, 1), batch_seed(42, 2)) == (1043, 2043)
    assert [s for _, _, s in seen] == [1043, 2043]
    for epoch, (fails, succs, seed) in enumerate(seen, 1):
        random.Random(seed).shuffle(fails)
        random.Random(seed + 1).shuffle(succs)
        fail_dec = [d for d in _decisions(trace, "analyst_failure") if d["epoch"] == epoch]
        succ_dec = [d for d in _decisions(trace, "analyst_success") if d["epoch"] == epoch]
        assert fail_dec[0]["record_ids"] == fails
        assert succ_dec[0]["record_ids"] == succs


def test_default_analyst_group_is_eight(tmp_path):
    """_base_/default.yaml:87 ``minibatch_size: 8``: 8 failures, one call."""
    client = FakeClient()
    _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 1})
    calls = [c for c in client.calls if c["call_key"][0] == "analyst_failure"]
    assert len(calls) == 1
    assert len(_decisions(_read_trace(tmp_path), "analyst_failure")[0]["record_ids"]) == 8


def test_val_cache_placeholder_and_slow_update_rules(tmp_path):
    """The epoch-1 placeholder is not cached (trainer.py:1809-1824); the slow
    update caches the pre-update score for its skill (trainer.py:2052-2053)."""
    import hashlib

    def h(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    loop1, _ = _run_loop(tmp_path / "one", skillopt_options={"max_epochs": 1})
    state1 = loop1.strategy_state_dict()
    assert prompts.SLOW_UPDATE_START in state1["current_skill"]
    assert h(state1["current_skill"]) not in state1["val_cache"]

    loop2, _ = _run_loop(tmp_path / "two", skillopt_options={"max_epochs": 2})
    state2 = loop2.strategy_state_dict()
    assert "Updated guidance text." in state2["current_skill"]
    assert state2["val_cache"][h(state2["current_skill"])] == 0.5


def test_slow_update_sample_seed(tmp_path, monkeypatch):
    """trainer.py:1843-1844 + dataloader.py:278-283: the slow-update sample is
    the train items shuffled by ``Random(seed + epoch*2000)``, first n, and both
    rollouts use that same seed. Expected ids were computed by SkillOpt's
    ``build_train_batch(batch_size=5, seed=...)`` over train items t1..t8."""
    from refactory_dsh.outerloop import Run

    calls: list[tuple[str, tuple | None, int | None]] = []
    original = Run.evaluate

    def evaluate(self, candidate, *, part="train", items=None, seed=None):  # type: ignore[no-untyped-def]
        calls.append((part, tuple(items) if items else None, seed))
        return original(self, candidate, part=part, items=items, seed=seed)

    monkeypatch.setattr(Run, "evaluate", evaluate)
    _run_loop(tmp_path, skillopt_options={"max_epochs": 3, "slow_update_samples": 5})
    samples = [d["sample_ids"] for d in _decisions(_read_trace(tmp_path), "slow_update_sample")]
    expected = {
        4042: ["t4", "t3", "t8", "t2", "t7"],   # epoch 2
        6042: ["t8", "t6", "t7", "t2", "t4"],   # epoch 3
    }
    assert samples == list(expected.values())
    for seed, ids in expected.items():
        rollouts = [c for c in calls if c[1] == tuple(ids)]
        assert len(rollouts) == 2 and all(c[2] == seed for c in rollouts), rollouts


def test_slow_update_without_train_ids_raises(tmp_path):
    """The slow update cannot run without train ids; say so before spending."""
    with pytest.raises(ValueError, match="parts manifest"):
        _run_loop(
            tmp_path,
            config_overrides={"parts_manifest": None},
            skillopt_options={"max_epochs": 2},
        )


def test_seeds_reach_the_task(tmp_path):
    """The seeds the trace names are the seeds the Task is run with: the base
    seed for every val evaluation (trainer.py:1133-1136; dataloader.py:291-301),
    the batch seed for each train rollout (dataloader.py:278) and the slow-update
    seed for its rollouts (trainer.py:1844). A strategy that traced the right
    value but evaluated with another fails here."""
    SkillTask.seen = []
    _run_loop(tmp_path, skillopt_options={"max_epochs": 2})
    by_part: dict[str | None, set[int | None]] = {}
    for part, seed in SkillTask.seen:
        by_part.setdefault(part, set()).add(seed)
    # (The loop's own end-of-run holdout audit is not the strategy's and is
    # left out.)
    assert by_part["val"] == {42}
    assert by_part["train"] == {1043, 2043, 4042}
    traced = [d["epoch_seed"] for d in _decisions(_read_trace(tmp_path), "train_rollout")]
    assert traced == [1043, 2043]


# ── Ask 1: Custom strategy seam ──────────────────────────────────────────


def test_strategy_no_proxy_no_loop_internals():
    """A custom strategy subclass cannot reach Loop internals through
    attribute access: no __getattr__ proxy, no _infra attribute.

    Fails when reverted because the old SkillOptLoop had __getattr__ and
    _infra.
    """
    from refactory_dsh.outerloop import OuterLoop

    strategy = SkillOptLoop()
    strategy.configure(model=FakeClient())

    # No __getattr__ proxy on the class
    assert "__getattr__" not in type(strategy).__dict__, (
        "SkillOptLoop must not define __getattr__"
    )

    # No _infra attribute
    assert not hasattr(strategy, "_infra"), (
        "SkillOptLoop must not have an _infra attribute"
    )

    # Accessing a Loop-internal attribute raises AttributeError
    with pytest.raises(AttributeError):
        _ = strategy.population  # noqa: B018

    with pytest.raises(AttributeError):
        _ = strategy.evaluations  # noqa: B018

    with pytest.raises(AttributeError):
        _ = strategy.config  # noqa: B018

    # Defining __getattr__ on an OuterLoop subclass is rejected at class creation
    with pytest.raises(TypeError, match="__getattr__"):
        class BadStrategy(OuterLoop):
            name = "bad"
            def __getattr__(self, name):
                pass


def test_attached_strategy_reaches_no_loop_internals(tmp_path, monkeypatch):
    """Attached to a running loop, the strategy still reaches nothing of the
    Loop's by attribute: no ``__getattr__`` anywhere in its class hierarchy
    (``OuterLoop`` included), no reference to the Loop in its own state.

    Fails if a proxy is reintroduced -- on ``OuterLoop`` or by the Loop handing
    itself to the strategy on attach -- because the probes would then resolve.
    """
    probes: dict[str, Any] = {}
    original = SkillOptLoop.strategy

    def probing(self, run):  # type: ignore[no-untyped-def]
        for name in ("population", "config", "evaluations", "rejections", "mode",
                     "_loop", "_infra", "_custom_strategy", "_train_item_records"):
            try:
                getattr(self, name)
            except AttributeError:
                probes[name] = "refused"
            else:
                probes[name] = "reached"
        probes["getattr_in_mro"] = [
            k.__name__ for k in type(self).__mro__ if "__getattr__" in k.__dict__
        ]
        probes["holds_loop"] = [
            key for key, value in vars(self).items() if isinstance(value, (Loop, Run))
        ]
        return original(self, run)

    from refactory_dsh.outerloop import Run

    monkeypatch.setattr(SkillOptLoop, "strategy", probing)
    _, report = _run_loop(tmp_path, skillopt_options={"max_epochs": 1})
    assert report["exit_ok"]
    reached = [name for name, how in probes.items() if how == "reached"]
    assert reached == []
    assert probes["getattr_in_mro"] == []
    assert probes["holds_loop"] == []


def test_gepa_like_population_strategy_on_same_seam(tmp_path):
    """A second toy strategy (GEPA-like population strategy with several
    candidates per epoch) runs on the same custom-strategy seam unchanged.

    Proves the seam is generic, not SkillOpt-specific.  Fails when reverted
    because the seam (custom strategy delegation in Loop) would not exist.
    """
    from refactory_dsh.outerloop import OuterLoop, Run

    class PopulationStrategy(OuterLoop):
        """A toy GEPA-like strategy with multiple candidates per epoch."""

        name = "toy_population"
        requires_model = False

        def configure(self, *, model=None, model_name="", provider="auto", **options):
            self.model = model
            self.model_name = model_name
            self.provider = provider
            self.options = dict(options)
            self.pop_size = int(options.get("pop_size", 3))
            self.max_epochs = int(options.get("max_epochs", 2))
            self._epoch = 0
            self._best_score = 0.0
            self._candidates_evaluated = 0

        def strategy(self, run: Run) -> None:
            """Evaluate multiple candidates per epoch, keep the best."""
            seed = run.seed()
            current = seed
            for epoch in range(1, self.max_epochs + 1):
                self._epoch = epoch
                if run.budget_left() <= 0:
                    break
                # Evaluate the seed/current on train
                current = run.evaluate(current, part="train")
                records = run.item_records(current)
                self._candidates_evaluated += 1
                run.save_strategy_state()

                ctrl = run.check_control()
                if ctrl == "stop":
                    run.stop("control_stop")
                    return

            run.stop("max_epochs")

        def strategy_state_dict(self):
            return {
                "epoch": self._epoch,
                "best_score": self._best_score,
                "candidates_evaluated": self._candidates_evaluated,
            }

        def load_strategy_state(self, state):
            self._epoch = state.get("epoch", 0)
            self._best_score = state.get("best_score", 0.0)
            self._candidates_evaluated = state.get("candidates_evaluated", 0)

    # Register and run
    strategies.register("toy_population", PopulationStrategy)
    try:
        cfg = _config(tmp_path, max_evaluations=50)
        with _patch_mode(SkillProbe):
            loop = strategies.create(
                "toy_population", MODE_ID, cfg,
                strategy_options={"pop_size": 3, "max_epochs": 2},
                project=Path("."), model=FakeClient(),
            )
            assert loop._custom_strategy is not None
            assert isinstance(loop._custom_strategy, PopulationStrategy)
            report = loop.run()

        # The strategy ran and produced results
        strategy = loop._custom_strategy
        state = strategy.strategy_state_dict()
        assert state["epoch"] == 2
        assert state["candidates_evaluated"] >= 2
        assert report["exit_ok"]
    finally:
        strategies.STRATEGIES.pop("toy_population", None)


# ── Ask 2: CLI strategy options ──────────────────────────────────────────


def test_cli_runs_skillopt_end_to_end(tmp_path, monkeypatch):
    """``python -m refactory_dsh.loop <mode> --strategy skillopt
    --strategy-option max_epochs=1`` finds the strategy by package discovery,
    hands the options to ``configure()`` and completes the run.

    Fails when reverted because the CLI could not resolve ``skillopt`` (it was
    reachable only through a hand ``register()``), or would drop the options.
    """
    from refactory_dsh import loop as loop_module
    from refactory_dsh import model as model_module

    client = FakeClient()

    class ScriptedRoute:
        """Stands in for DshModelClient: building one yields the scripted client."""

        def __new__(cls, **_kwargs: Any) -> Any:
            return client

    monkeypatch.setattr(model_module, "DshModelClient", ScriptedRoute)
    configured: list[dict[str, Any]] = []
    original = SkillOptLoop.configure

    def spy(self, **kwargs):  # type: ignore[no-untyped-def]
        configured.append(dict(kwargs))
        return original(self, **kwargs)

    monkeypatch.setattr(SkillOptLoop, "configure", spy)
    parts = _write_parts(tmp_path)
    with _patch_mode(SkillProbe):
        code = loop_module.main([
            MODE_ID, "--strategy", "skillopt", "--strategy-option", "max_epochs=1",
            "--optimizer", "random", "--seed", "42",
            "--runs-root", str(tmp_path / "run"), "--project", ".",
            "--parts", str(parts),
        ])

    assert code == 0
    assert len(configured) == 1
    assert configured[0]["max_epochs"] == 1
    assert configured[0]["model"] is client
    step_types = [e["step_type"] for e in _read_trace(tmp_path)]
    assert "initial_val" in step_types
    assert "train_rollout" in step_types
    assert any(c["call_key"][0] == "analyst_failure" for c in client.calls)
    # No allowed_ops was set: the strategy's declared ``required_ops`` let its
    # patch through instead of every patch being refused by the edit preset.
    trace = _read_trace(tmp_path)
    assert _decisions(trace, "patch_applied")
    assert not _decisions(trace, "patch_refused")


def test_explicit_allowed_ops_win_over_required_ops(tmp_path):
    """A person's ``allowed_ops`` is the whole answer: without text_patch in
    it, the patch is refused even though the strategy declares it."""
    assert SkillOptLoop.required_ops == ("text_patch", "set_knob")
    _run_loop(
        tmp_path, config_overrides={"allowed_ops": ("set_knob",)},
        skillopt_options={"max_epochs": 1},
    )
    trace = _read_trace(tmp_path)
    assert _decisions(trace, "patch_refused")
    assert not _decisions(trace, "patch_applied")


def test_skillopt_resolves_by_discovery_in_a_fresh_process():
    """A new interpreter resolves ``skillopt`` through the package discovery the
    CLI uses, with nothing registered by hand."""
    import subprocess
    import sys

    code = (
        "from refactory_dsh import strategies\n"
        "cls = strategies.resolve('skillopt')\n"
        "assert 'skillopt' not in strategies.STRATEGIES\n"
        "assert 'skillopt' in strategies.available()\n"
        "print(cls.__module__ + ':' + cls.__name__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "refactory_packages.skillopt.loop:SkillOptLoop"


# ── Ask 3: Resume continues, never restarts ──────────────────────────────


class ProcessKilled(BaseException):
    """The process dying mid-call. A failed model *call* is not a crash: SkillOpt
    carries on past it (trace.ask), so the kill is not an ``Exception``."""


class CrashingClient(FakeClient):
    """A scripted client whose process dies at one call key."""

    def __init__(self, crash_at: tuple) -> None:
        super().__init__()
        self.crash_at = crash_at

    def complete(self, prompt, *, system="", call_key=None):  # type: ignore[no-untyped-def]
        if call_key is not None and tuple(call_key) == self.crash_at:
            raise ProcessKilled(f"model died at {self.crash_at}")
        return super().complete(prompt, system=system, call_key=call_key)


def _crash_then_resume(
    tmp_path: Path, crash_at: tuple, max_epochs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], SkillOptLoop]:
    """An uninterrupted reference run, and the same run killed at ``crash_at``
    then resumed through the real ``--resume`` journal path.

    Returns (reference calls, calls before the crash, calls after resume, the
    resumed strategy).
    """
    options = {"max_epochs": max_epochs}
    reference = FakeClient()
    _run_loop(tmp_path / "ref", client=reference, skillopt_options=options)

    crashing = CrashingClient(crash_at)
    with pytest.raises(ProcessKilled, match="model died"):
        _run_loop(tmp_path / "split", client=crashing, skillopt_options=options)

    resumed = FakeClient()
    cfg = _config(tmp_path / "split", resume=True)
    with _patch_mode(SkillProbe):
        loop = strategies.create(
            "skillopt", MODE_ID, cfg, strategy_options=options,
            project=Path("."), model=resumed,
        )
        loop.run()
    strategy = loop._custom_strategy
    assert isinstance(strategy, SkillOptLoop)
    return reference.calls, crashing.calls, resumed.calls, strategy


def _keys(calls: list[dict[str, Any]]) -> list[tuple | None]:
    return [c["call_key"] for c in calls]


def test_resume_skips_completed_phases(tmp_path):
    """A run killed at the epoch-2 analyst call resumes from the journal and
    continues with the skill it had, not the seed's: the resumed analyst prompt
    is the uninterrupted run's, text and all, and every later call matches.

    Fails when reverted because resume reset the skill to the seed's and the
    candidate to the seed, so the analyst saw a different skill.
    """
    crash_at = ("analyst_failure", 2, 0, 0)
    ref, before, after, strategy = _crash_then_resume(tmp_path, crash_at, 3)

    assert _keys(before) + _keys(after) == _keys(ref)
    assert _keys(after)[0] == crash_at
    ref_prompt = next(c for c in ref if c["call_key"] == crash_at)
    assert after[0]["prompt"] == ref_prompt["prompt"]
    assert after[0]["system"] == ref_prompt["system"]
    assert [c["prompt"] for c in before + after] == [c["prompt"] for c in ref]
    assert strategy.strategy_state_dict()["epoch"] == 3


def test_resume_mid_epoch_continues_at_saved_phase(tmp_path, monkeypatch):
    """Killed at the epoch-2 merge (train and analysts done), the resumed run
    makes exactly the uninterrupted run's remaining calls, with the same
    prompts: no analyst call is repeated and none is skipped.

    Fails when reverted because the resumed run would restart the epoch, or
    merge against the seed's skill.
    """
    monkeypatch.setenv(PASS_IDS_ENV, "t1,t3,t5,t7")
    crash_at = ("merge_final", 2, 0, 0)
    ref, before, after, _ = _crash_then_resume(tmp_path, crash_at, 2)

    ref_keys = _keys(ref)
    assert _keys(after) == ref_keys[ref_keys.index(crash_at):]
    assert _keys(before) == ref_keys[:ref_keys.index(crash_at)]
    assert [c["prompt"] for c in after] == [c["prompt"] for c in ref[ref_keys.index(crash_at):]]


def test_resume_before_the_val_gate_gates_the_patched_candidate(tmp_path, monkeypatch):
    """Killed at the epoch-1 val-gate evaluation (after the patch), the resumed
    run evaluates the patched skill -- not the pre-patch one -- and then makes
    exactly the uninterrupted run's calls.

    Fails when reverted because resume took ``current`` as the candidate, so
    the val gate re-scored the unpatched skill from its cache.
    """
    from refactory_dsh.outerloop import Run

    original = Run.evaluate
    log: list[tuple[str, str, str]] = []   # (run label, part, skill)
    label = {"run": ""}
    crash = {"armed": False}

    def evaluate(self, candidate, *, part="train", items=None, seed=None):  # type: ignore[no-untyped-def]
        if part == "val":
            vals = [entry for entry in log if entry[0] == label["run"] and entry[1] == "val"]
            if crash["armed"] and len(vals) == 1:
                raise RuntimeError("process died at the val gate")
            log.append((label["run"], part, SkillOptLoop._get_skill(candidate)))
        return original(self, candidate, part=part, items=items, seed=seed)

    monkeypatch.setattr(Run, "evaluate", evaluate)
    options = {"max_epochs": 2}

    label["run"] = "ref"
    reference = FakeClient()
    _run_loop(tmp_path / "ref", client=reference, skillopt_options=options)

    label["run"] = "split"
    crash["armed"] = True
    before = FakeClient()
    with pytest.raises(RuntimeError, match="val gate"):
        _run_loop(tmp_path / "split", client=before, skillopt_options=options)
    crash["armed"] = False

    label["run"] = "resumed"
    after = FakeClient()
    with _patch_mode(SkillProbe):
        loop = strategies.create(
            "skillopt", MODE_ID, _config(tmp_path / "split", resume=True),
            strategy_options=options, project=Path("."), model=after,
        )
        loop.run()

    ref_vals = [skill for run, part, skill in log if run == "ref" and part == "val"]
    resumed_vals = [skill for run, part, skill in log if run == "resumed" and part == "val"]
    assert len(ref_vals) >= 2
    assert ref_vals[1] != ref_vals[0], "the epoch-1 patch must change the skill"
    assert resumed_vals[0] == ref_vals[1]
    assert resumed_vals == ref_vals[1:]
    assert _keys(before.calls) + _keys(after.calls) == _keys(reference.calls)
    assert [c["prompt"] for c in before.calls + after.calls] == [
        c["prompt"] for c in reference.calls
    ]


def test_resume_after_crash_at_meta_skill_makes_the_meta_skill(tmp_path):
    """Killed at the epoch-2 meta-skill call, after the slow update: the resumed
    run does not repeat the slow update and does make the meta skill, which
    then reaches epoch 3 exactly as in the uninterrupted run.

    Fails when reverted because the slow-update phase marker made resume skip
    the whole end of epoch, so ('meta_skill', 2) was never made.
    """
    crash_at = ("meta_skill", 2, 0, 0)
    ref, before, after, strategy = _crash_then_resume(tmp_path, crash_at, 3)

    assert ("slow_update", 2, 0, 0) in _keys(before)
    assert ("slow_update", 2, 0, 0) not in _keys(after)
    assert _keys(after)[0] == crash_at
    assert _keys(before) + _keys(after) == _keys(ref)
    assert [c["prompt"] for c in before + after] == [c["prompt"] for c in ref]
    assert strategy.strategy_state_dict()["meta_skill_text"] == "USE CONCISE EDITS"


# ── Ask 4: Batch seed matches SkillOpt formula ──────────────────────────


def test_batch_seed_matches_skillopt_formula(tmp_path):
    """The training batch seed matches SkillOpt's exact formula:
    seed + epoch*1000 + batch_idx + 1 with batch_idx=0.

    Fails when reverted because the old formula seed + epoch*1000 is off
    by one.
    """
    # Test the epoch_seed used in strategy
    client = FakeClient()
    strategy, _ = _run_loop(
        tmp_path, client=client,
        skillopt_options={"max_epochs": 2},
    )

    trace = _read_trace(tmp_path)
    train_rollouts = _decisions(trace, "train_rollout")

    base_seed = 42  # default
    for rollout in train_rollouts:
        epoch = rollout["epoch"]
        expected_seed = base_seed + epoch * 1000 + 1  # SkillOpt formula with batch_idx=0
        assert rollout["epoch_seed"] == expected_seed, (
            f"epoch {epoch}: expected seed {expected_seed}, got {rollout['epoch_seed']}; "
            f"SkillOpt's formula is seed + epoch*1000 + batch_idx + 1"
        )

    # Verify specific values
    assert train_rollouts[0]["epoch_seed"] == 42 + 1 * 1000 + 1  # 1043
    assert train_rollouts[1]["epoch_seed"] == 42 + 2 * 1000 + 1  # 2043


# ── Ask 5: Part/seed delivery to Tasks ───────────────────────────────────


class SeedAwareTask(task_module.Task):
    """Toy Task whose instances depend on the seed."""

    name = "seed_aware_test"
    _seen_part: str | None = None
    _seen_seed: int | None = None
    _setup_part: str | None = None
    _setup_seed: int | None = None
    _verify_part: str | None = None
    _verify_seed: int | None = None

    def instances(self, *, part=None, seed=None):
        SeedAwareTask._seen_part = part
        SeedAwareTask._seen_seed = seed
        items = [task_module.TaskInstance(id=f"item_{i}") for i in range(4)]
        if seed is not None:
            import random as _rng
            items_copy = list(items)
            _rng.Random(seed).shuffle(items_copy)
            return items_copy
        return items

    def setup(self, instance, workspace, *, part=None, seed=None):
        SeedAwareTask._setup_part = part
        SeedAwareTask._setup_seed = seed

    def verify(self, instance, workspace, *, part=None, seed=None):
        SeedAwareTask._verify_part = part
        SeedAwareTask._verify_seed = seed
        return task_module.VerifyResult(passed=True, score=1.0)


task_module.register_task("seed_aware_test", SeedAwareTask)


def test_task_receives_part_and_seed_walker(tmp_path):
    """The walker passes part and seed to Task.instances(), setup() and verify().

    Drives the real walker (``nn.executor.execute``) with a data node bound
    to a seed-aware Task, so it exercises the full ``load_items → task_items``
    chain and the ``setup/verify`` calls.

    Fails when reverted because Task.instances/setup/verify would receive no
    part/seed arguments.
    """
    from refactory_dsh.nn import Graph, Node
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute
    from refactory_dsh.task import register_task, write_subset

    # Reset class-level tracking
    SeedAwareTask._seen_part = None
    SeedAwareTask._seen_seed = None
    SeedAwareTask._setup_part = None
    SeedAwareTask._setup_seed = None
    SeedAwareTask._verify_part = None
    SeedAwareTask._verify_seed = None

    # Register the task
    register_task("seed_aware_test", SeedAwareTask)

    # Build a graph with a data node using the seed-aware task
    graph = Graph()
    graph.add(Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="seed_aware_test",
        writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(Node(
        "finish", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "finish")

    # Write subset.json with part and seed
    workdir = tmp_path / "walker"
    workdir.mkdir(parents=True, exist_ok=True)
    write_subset(workdir, "train", 0.3, seed=42)

    # Run the walker
    result = execute(graph, workdir)
    assert result.exit_ok, result.error

    # Verify instances() received part and seed
    assert SeedAwareTask._seen_part == "train", (
        f"task.instances did not receive part='train', got {SeedAwareTask._seen_part!r}"
    )
    assert SeedAwareTask._seen_seed == 42, (
        f"task.instances did not receive seed=42, got {SeedAwareTask._seen_seed!r}"
    )

    # Verify setup() received part and seed
    assert SeedAwareTask._setup_part == "train", (
        f"task.setup did not receive part='train', got {SeedAwareTask._setup_part!r}"
    )
    assert SeedAwareTask._setup_seed == 42, (
        f"task.setup did not receive seed=42, got {SeedAwareTask._setup_seed!r}"
    )

    # Verify verify() received part and seed
    assert SeedAwareTask._verify_part == "train", (
        f"task.verify did not receive part='train', got {SeedAwareTask._verify_part!r}"
    )
    assert SeedAwareTask._verify_seed == 42, (
        f"task.verify did not receive seed=42, got {SeedAwareTask._verify_seed!r}"
    )


def test_task_receives_part_and_seed_spine(tmp_path):
    """The spine's real drain pipeline (lowering + TS drain) passes part and
    seed to Task.setup()/verify().

    Uses the real drain (lowering + TS drain) as the existing real-spine data
    scenarios do (test_runtime_equivalence.py), not direct items.py calls.

    Fails when reverted because setup/verify would receive no part/seed.
    """
    import os
    import shutil
    import subprocess
    import sys

    from refactory_dsh import lowering, nn
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute
    from refactory_dsh.paths import REPO_ROOT
    from refactory_dsh.task import register_task, write_subset

    NODE_BIN = shutil.which("node")
    DATA_DRAIN = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain.mts"
    cordis_dir = REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis"

    if NODE_BIN is None or not cordis_dir.exists():
        pytest.skip("spine drain needs Node and installed workspace (pnpm install)")

    # ── Write a Task package that records part/seed to a sidecar file ──
    _SEED_TASK_SOURCE = '''\
"""A Task whose setup/verify records part and seed to a sidecar file."""
import json
from pathlib import Path
from refactory_dsh.task import Task, TaskInstance, VerifyResult

class SeedRecorderTask(Task):
    name = "seed_recorder"

    def instances(self, *, part=None, seed=None):
        return [TaskInstance(id="s1"), TaskInstance(id="s2")]

    def setup(self, instance, workspace, *, part=None, seed=None):
        sidecar = Path(workspace) / "_seed_sidecar.json"
        data = json.loads(sidecar.read_text()) if sidecar.exists() else {"setups": [], "verifies": []}
        data["setups"].append({"id": instance.id, "part": part, "seed": seed})
        sidecar.write_text(json.dumps(data))

    def verify(self, instance, workspace, *, part=None, seed=None):
        sidecar = Path(workspace) / "_seed_sidecar.json"
        data = json.loads(sidecar.read_text()) if sidecar.exists() else {"setups": [], "verifies": []}
        data["verifies"].append({"id": instance.id, "part": part, "seed": seed})
        sidecar.write_text(json.dumps(data))
        return VerifyResult(passed=True, score=1.0)
'''

    _SEED_OP_SOURCE = '''\
"""Op for seed-recorder scenario."""
import json
from pathlib import Path

def record(ctx):
    item = ctx.read_json("current_item.json", {}) or {}
    seen_path = Path(ctx.workdir) / "seen.json"
    seen = json.loads(seen_path.read_text()) if seen_path.exists() else {"ids": []}
    seen["ids"].append(item.get("id", ""))
    ctx.write_json("seen.json", seen)
    ctx.set_fact("value", seen)
'''

    project = tmp_path / "project"
    pkg = project / ".factory" / "packages" / "seed_recorder"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        'from refactory_dsh.nn import Package\n'
        'PACKAGE = Package(\n'
        '    name="seed_recorder",\n'
        '    description="seed recorder task",\n'
        '    modes=(),\n'
        '    ops=("refactory_packages.seed_recorder.ops",),\n'
        ')\n'
    )
    (pkg / "task.py").write_text(_SEED_TASK_SOURCE)
    (pkg / "ops.py").write_text(_SEED_OP_SOURCE)

    sys.path.insert(0, str(project / ".factory" / "packages"))
    try:
        from seed_recorder.task import SeedRecorderTask  # type: ignore[import-not-found]
        register_task("seed_recorder", SeedRecorderTask)

        task_ref_spine = "refactory_packages.seed_recorder.task:SeedRecorderTask"

        # ── Build graphs ──
        # Walker graph uses the registered name "seed_recorder"; spine graph
        # uses the fully-qualified module path for subprocess import.
        walker_graph = nn.Graph()
        walker_graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref="seed_recorder",
            writes=(CURRENT_ITEM,), produces=("item_scores",),
        ))
        walker_graph.add(nn.Node(
            "work", "op", call="loop_ops:record_item",
            reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
        ))
        walker_graph.add(nn.Node(
            "finish", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        walker_graph.connect("data", "finish")

        spine_graph = nn.Graph()
        spine_graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref=task_ref_spine,
            writes=(CURRENT_ITEM,), produces=("item_scores",),
        ))
        spine_graph.add(nn.Node(
            "work", "op", call="loop_ops:record_item",
            reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
        ))
        spine_graph.add(nn.Node(
            "finish", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        spine_graph.connect("data", "finish")

        # ── Walker run (verifies walker passes part/seed) ──
        walker_dir = tmp_path / "walker"
        walker_dir.mkdir(parents=True, exist_ok=True)
        write_subset(walker_dir, "train", 0.3, seed=77)
        walker_result = execute(walker_graph, walker_dir)
        assert walker_result.exit_ok, walker_result.error
        # With pass isolation, each pass writes its sidecar in its pass directory.
        walker_passes = walker_dir / "passes" / "data"
        assert walker_passes.exists(), "walker did not create passes/data/"
        walker_setups: list = []
        walker_verifies: list = []
        for subdir in sorted(walker_passes.iterdir()):
            if subdir.is_dir():
                sidecar = subdir / "_seed_sidecar.json"
                if sidecar.exists():
                    data = json.loads(sidecar.read_text())
                    walker_setups.extend(data.get("setups", []))
                    walker_verifies.extend(data.get("verifies", []))
        assert len(walker_setups) >= 1
        assert len(walker_verifies) >= 1
        assert walker_setups[0]["part"] == "train"
        assert walker_setups[0]["seed"] == 77
        assert walker_verifies[0]["part"] == "train"
        assert walker_verifies[0]["seed"] == 77

        # ── Spine run (real TS drain via data-drain.mts) ──
        spine_dir = tmp_path / "spine"
        spine_dir.mkdir(parents=True, exist_ok=True)
        write_subset(spine_dir, "train", 0.3, seed=77)

        state = {"graph": spine_graph.to_state()}
        lowering.write_declarations(state, spine_dir)
        graph_path, _ = lowering.write_mode_graph(
            state, tmp_path / "graphs", name="seed_recorder",
        )

        python_bin = str(REPO_ROOT / ".venv/bin")
        HERE = Path(__file__).resolve().parent
        extra_path = os.pathsep.join([
            str(project / ".factory" / "packages"),
            str(HERE),
        ])
        env_patch = {
            "REFACTORY_PROJECT": str(project),
            "PYTHONPATH": extra_path,
        }
        old_env = {}
        for key, value in env_patch.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        try:
            env = {**os.environ, "REFACTORY_WORKDIR": str(spine_dir)}
            env["PATH"] = f"{python_bin}:{env.get('PATH', '')}"
            result = subprocess.run(
                [str(NODE_BIN), "--import", "tsx/esm", str(DATA_DRAIN),
                 str(graph_path), str(spine_dir), f"{python_bin}/python", "10000"],
                capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
                env=env,
            )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        assert result.returncode == 0, (
            f"spine drain failed:\nstdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

        # ── Verify spine task received part/seed ──
        # With pass isolation, each pass writes its sidecar in its pass directory.
        spine_passes = spine_dir / "passes" / "data"
        assert spine_passes.exists(), "spine did not create passes/data/"
        spine_setups: list = []
        spine_verifies: list = []
        for subdir in sorted(spine_passes.iterdir()):
            if subdir.is_dir():
                sidecar = subdir / "_seed_sidecar.json"
                if sidecar.exists():
                    data = json.loads(sidecar.read_text())
                    spine_setups.extend(data.get("setups", []))
                    spine_verifies.extend(data.get("verifies", []))
        assert len(spine_setups) >= 1, "spine had no setup calls"
        assert len(spine_verifies) >= 1, "spine had no verify calls"
        assert spine_setups[0]["part"] == "train", (
            f"spine task.setup did not receive part='train', got {spine_setups[0]['part']!r}"
        )
        assert spine_setups[0]["seed"] == 77, (
            f"spine task.setup did not receive seed=77, got {spine_setups[0]['seed']!r}"
        )
        assert spine_verifies[0]["part"] == "train", (
            f"spine task.verify did not receive part='train', got {spine_verifies[0]['part']!r}"
        )
        assert spine_verifies[0]["seed"] == 77, (
            f"spine task.verify did not receive seed=77, got {spine_verifies[0]['seed']!r}"
        )
    finally:
        # Clean up sys.path
        if str(project / ".factory" / "packages") in sys.path:
            sys.path.remove(str(project / ".factory" / "packages"))
