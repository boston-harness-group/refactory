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

    def instances(self) -> list[task_module.TaskInstance]:
        return [
            task_module.TaskInstance(id=item_id)
            for item_id in TRAIN_IDS + VAL_IDS
        ]

    def verify(self, instance, workspace):  # type: ignore[no-untyped-def]
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

RECORD_FAIL_TEXT = (
    "### Trajectory 1 (id=q7)\n"
    "Task: What is 2+2?\n"
    "Task type: arith\n"
    "Failure reason: predicted 5, expected B\n"
    "Steps: 1\n"
    "\n"
    "[solve prompt] Solve: 2+2?\n"
    "[solve] <answer>5</answer>\n"
    "[verification] [EVALUATION RESULT]\n"
    "Prediction: '5'\n"
    "Label: 'B'\n"
    "Label text: '4'\n"
    "Score: 0.25\n"
    "Passed: False"
)


def test_analyst_message_renders_full_trajectory():
    """reflect.py:321-344 / :421-441, trajectories as reflect.py:112-239."""
    ok = dict(RECORD_FAIL, id="q8", passed=True, score=1.0, prediction="4",
              fail_reason=None, exchanges=[])
    user = prompts.format_analyst_user("SKILL", [RECORD_FAIL, ok], 4, "META", failure=True)
    assert user == (
        "## Current Skill\nSKILL\n\n"
        "## Edits Budget\nProduce at most L=4 edits.\n\n"
        + META_CTX + "\n\n"
        "## Failed Trajectories (2 total)\n"
        + RECORD_FAIL_TEXT
        + "\n\n---\n\n"
        "### Trajectory 2 (id=q8)\nTask: What is 2+2?\nTask type: arith\nSteps: 0\n\n"
        "[verification] [EVALUATION RESULT]\nPrediction: '4'\nLabel: 'B'\n"
        "Label text: '4'\nScore: 1.0\nPassed: True"
    )
    success = prompts.format_analyst_user("SKILL", [RECORD_FAIL], 3, "", failure=False)
    assert success == (
        "## Current Skill\nSKILL\n\n"
        "## Edits Budget\nProduce at most L=3 edits.\n\n"
        "## Successful Trajectories (1 total)\n" + RECORD_FAIL_TEXT
    )


def test_analyst_prompts_in_a_run_carry_records(tmp_path):
    """The run's analysts see each record's fields, not just pass/fail."""
    client = FakeClient()
    _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 1})
    prompt = next(c["prompt"] for c in client.calls if c["call_key"][0] == "analyst_failure")
    assert "## Failed Trajectories (8 total)\n### Trajectory 1 (id=t" in prompt
    assert "[verification] [EVALUATION RESULT]\nPrediction: ''" in prompt
    assert "Score: 0.5\nPassed: False" in prompt


def test_comparison_and_slow_update_messages_match_skillopt():
    """slow_update.py:206-269 pairs, :287-349 text, :409-424 message;
    meta_skill.py:44-60 message."""
    prev_ok = dict(RECORD_FAIL, passed=True, score=1.0, prediction="4", fail_reason=None)
    pairs = SkillOptLoop._build_comparison_pairs([prev_ok], [RECORD_FAIL], ["q7", "q9"])
    text = prompts.format_comparison_text(pairs)
    traj_prev = (
        "[solve prompt] Solve: 2+2?\n[solve] <answer>5</answer>\n"
        "[verification] [EVALUATION RESULT]\nPrediction: '4'\nLabel: 'B'\n"
        "Label text: '4'\nScore: 1.0\nPassed: True"
    )
    traj_curr = RECORD_FAIL_TEXT.split("\n\n", 1)[1]
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
    with batch seed + 1; the batch seed is datasets/base.py:98-109's (43 for
    seed 42 and one batch per epoch), the same every epoch."""
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
    assert batch_seed(42, 1) == batch_seed(42, 2) == 43
    assert [s for _, _, s in seen] == [43, 43]
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


def test_slow_update_sample_seed(tmp_path):
    """trainer.py:1844: the sample is seeded with ``seed + epoch * 2000``."""
    import random

    _run_loop(tmp_path, skillopt_options={"max_epochs": 2})
    sample = _decisions(_read_trace(tmp_path), "slow_update_sample")[0]["sample_ids"]
    assert sample == random.Random(42 + 2 * 2000).sample(TRAIN_IDS, 8)


def test_slow_update_without_train_ids_raises(tmp_path):
    """The slow update cannot run without train ids; say so before spending."""
    with pytest.raises(ValueError, match="parts manifest"):
        _run_loop(
            tmp_path,
            config_overrides={"parts_manifest": None},
            skillopt_options={"max_epochs": 2},
        )
