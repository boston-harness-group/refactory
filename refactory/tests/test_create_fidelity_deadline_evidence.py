"""Deadline scores inspect actual producer receipts and reject inconsistent claims."""
from __future__ import annotations

import copy
import json

import pytest

from benchmarks.create_fidelity.deadline_evidence import dispatch_deadline_row
from refactory_dsh.model import DshModelClient
from test_worker_deadline_telemetry import controlled_wire, settled_budget


@pytest.fixture
def dispatched(tmp_path, monkeypatch):
    budget = settled_budget(tmp_path)
    controlled_wire(monkeypatch, lambda *_: None)
    client = DshModelClient(attempts=1).with_budget(budget).for_node("review").with_node_timeout(75)
    assert client.complete("Inspect the actual input.", workdir=tmp_path)
    path = next((tmp_path / ".refactory/workers").glob("*.json"))
    return [(str(path), json.loads(path.read_text()))]


def test_actual_dispatch_receipt_passes_with_coordinator_excluded(dispatched):
    workers = [("coordinator.json", {"node_id": "", "role": "coordinator"}), *dispatched]
    score = dispatch_deadline_row(workers)
    assert score["status"] == "passed"
    assert score["evidence"] == [dispatched[0][0]]
    assert "expiry enforcement is assessed separately" in score["reason"]


@pytest.mark.parametrize("field,value", [
    ("stage_remaining_s", True), ("stage_remaining_s", 76),
    ("run_remaining_s", 80), ("effective_remaining_s", 75),
    ("effective_limiters", ["stage"]), ("sampled_at", "2000-01-01T00:00:00+00:00"),
    ("attempt", 2), ("sample_phase", "after_completion"),
])
def test_inconsistent_remaining_claims_fail(dispatched, field, value):
    sample = dispatched[0][1]["effective_deadlines"]["dispatches"][0]
    sample[field] = value
    assert dispatch_deadline_row(dispatched)["status"] == "failed"


@pytest.mark.parametrize("field,value", [
    ("retained_validation_reserve_s", 0), ("remaining_s", 80),
    ("path", "/foreign/resource-usage.json"), ("phase", "unknown"),
    ("elapsed_limit_status", "inactive"),
])
def test_budget_snapshot_must_match_exact_phase_reserve_and_ledger(dispatched, field, value):
    dispatched[0][1]["effective_deadlines"]["dispatches"][0]["run_budget_snapshot"][field] = value
    assert dispatch_deadline_row(dispatched)["status"] == "failed"


def test_unknown_idle_timer_cannot_be_scored_as_inactive(dispatched):
    dispatched[0][1]["effective_deadlines"]["dispatches"][0]["idle_timer"] = {"status": "unavailable", "timeout_s": None}
    assert dispatch_deadline_row(dispatched)["status"] == "failed"


def test_existing_receipts_without_new_samples_remain_unavailable(dispatched):
    del dispatched[0][1]["effective_deadlines"]["dispatches"]
    assert dispatch_deadline_row(dispatched)["status"] == "unavailable"
    assert dispatch_deadline_row([("coordinator.json", {"node_id": "", "role": "coordinator"})])["status"] == "unavailable"


def test_missing_enclosing_composite_clock_is_unavailable(dispatched):
    dispatched[0][1]["node_id"] = "research_director:v1:plan:a1"
    score = dispatch_deadline_row(dispatched)
    assert score["status"] == "unavailable" and "enclosing composite" in score["reason"]


def test_duplicate_or_stale_retry_observations_fail(dispatched):
    deadlines = dispatched[0][1]["effective_deadlines"]
    deadlines["dispatches"].append(copy.deepcopy(deadlines["dispatches"][0]))
    assert dispatch_deadline_row(dispatched)["status"] == "failed"


@pytest.mark.parametrize("budget_kind", ["none", "tokens", "validation"])
def test_absent_elapsed_limit_and_final_validation_reserve_are_scored_truthfully(tmp_path, monkeypatch, budget_kind):
    controlled_wire(monkeypatch, lambda *_: None)
    client = DshModelClient(attempts=1).for_node("work").with_node_timeout(10)
    if budget_kind == "tokens":
        client = client.with_budget(settled_budget(tmp_path, maximum=None, reserve=0, tokens=100))
    elif budget_kind == "validation":
        client = client.with_budget(settled_budget(tmp_path, phase="final_validation"))
    assert client.complete("Do the work.", workdir=tmp_path)
    path = next((tmp_path / ".refactory/workers").glob("*.json"))
    assert dispatch_deadline_row([(str(path), json.loads(path.read_text()))])["status"] == "passed"
