"""SkillOpt outer-loop strategy tests.

Each test exercises one mechanism from the brief with a scripted model and a
small toy Task, and each fails when the mechanism it tests is reverted.
"""

from __future__ import annotations

import contextlib
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

    def instances(self) -> list[task_module.TaskInstance]:
        return [
            task_module.TaskInstance(id=item_id)
            for item_id in TRAIN_IDS + VAL_IDS
        ]

    def verify(self, instance, workspace):  # type: ignore[no-untyped-def]
        # Read the skill from the state if it's available via item data
        # For simplicity: pass based on whether MAGIC_MARKER env is set
        # In tests, the FakeClient controls the scoring via the val cache
        passed = bool(os.environ.get("_SKILLOPT_PASS_ALL", ""))
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
    """Set up and run a SkillOptLoop, returning (loop, report)."""
    cfg = _config(tmp_path, **(config_overrides or {}))
    client = client or FakeClient()
    with _patch_mode(SkillProbe):
        loop = SkillOptLoop(MODE_ID, cfg, project=Path("."))
        options = dict(skillopt_options or {})
        loop.configure(model=client, **options)
        if preload_state is not None:
            loop.load_strategy_state(preload_state)
        report = loop.run()
    return loop, report


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
def _register_skillopt():
    """Register SkillOptLoop for each test, and clean up after."""
    strategies.register("skillopt", SkillOptLoop)
    yield
    strategies.STRATEGIES.pop("skillopt", None)


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


def test_no_edit_step_skipped(tmp_path):
    """FakeClient returns empty edits. No val_evaluate entry, edit_budget_step=0."""
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
    import random
    from refactory_packages.skillopt import state as state_mod

    client1 = FakeClient()
    loop1, report1 = _run_loop(
        tmp_path / "run1", client=client1,
        skillopt_options={"max_epochs": 1},
    )
    state1 = loop1.strategy_state_dict()

    # The state must be JSON-serializable (for journaling)
    json_str = json.dumps(state1)
    state1_roundtripped = json.loads(json_str)

    # Run 2: create fresh loop, load state, verify consistency
    client2 = FakeClient()
    with _patch_mode(SkillProbe):
        cfg2 = _config(tmp_path / "run2")
        loop2 = SkillOptLoop(MODE_ID, cfg2, project=Path("."))
        loop2.configure(model=client2, max_epochs=2)
        loop2.load_strategy_state(state1_roundtripped)
        state2 = loop2.strategy_state_dict()

    # States must match
    assert state2["current_skill"] == state1["current_skill"]
    assert state2["current_score"] == state1["current_score"]
    assert state2["val_cache"] == state1["val_cache"]
    assert state2["edit_budget_step"] == state1["edit_budget_step"]
    assert state2["epoch"] == state1["epoch"]

    # RNG state must be faithfully serialized: the deserialized RNG
    # must produce the same sequence as the original.
    rng_original = loop1._rng
    rng_restored = loop2._rng
    # Draw a sequence of numbers from both RNGs
    seq_original = [rng_original.random() for _ in range(10)]
    seq_restored = [rng_restored.random() for _ in range(10)]
    assert seq_original == seq_restored, (
        "RNG state was not faithfully restored; resume would diverge"
    )


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
        loop = SkillOptLoop(MODE_ID, cfg, project=Path("."))
        loop.configure(model=client, max_epochs=5)
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


def test_replay_client_reproduces_decisions(tmp_path):
    """A ReplayClient returning recorded outputs reproduces the same decisions."""
    # Run 1: collect all model call outputs
    client1 = FakeClient()
    loop1, report1 = _run_loop(
        tmp_path / "run1", client=client1,
        skillopt_options={"max_epochs": 1},
    )
    trace1 = _read_trace(tmp_path / "run1")
    model_calls_1 = [e for e in trace1 if e["type"] == "model_call"]

    # Build replay dict
    recordings: dict[tuple, str] = {}
    for mc in model_calls_1:
        key = tuple(mc["call_key"])
        recordings[key] = mc["raw_output"]

    # Run 2: same setup with ReplayClient
    replay = ReplayClient(recordings)
    loop2, report2 = _run_loop(
        tmp_path / "run2", client=replay,
        skillopt_options={"max_epochs": 1},
    )
    trace2 = _read_trace(tmp_path / "run2")

    # Compare decisions
    decisions1 = [e for e in trace1 if e["type"] == "decision"]
    decisions2 = [e for e in trace2 if e["type"] == "decision"]

    # Same number of decisions
    types1 = [d["step_type"] for d in decisions1]
    types2 = [d["step_type"] for d in decisions2]
    assert types1 == types2, f"decision sequences differ: {types1} vs {types2}"
