"""Served route and recovered/dotted stage evidence regression controls."""
from __future__ import annotations

import pytest

from benchmarks.create_fidelity.evidence import authoring_score, normalize, route_score


def stage_score(records):
    events = normalize(records, trial_id="controlled", source="raw.json")
    return next(row for row in authoring_score(events, {}) if row["requirement_id"] == "F.stage.gate_strategy")


@pytest.mark.parametrize("field,wrong", [("provider", "different-proxy"), ("model", "gpt-6.1-sol"), ("effort", "low"), ("worker_backend", "codex")])
def test_explicit_request_mismatch_never_passes_served_route_score(field, wrong):
    actual = {"worker_backend": "dsh", "provider": "openai", "model": "gpt-6-sol", "effort": "high", "source": "runtime-request"}
    request = {key: value for key, value in actual.items() if key != "source"}
    request[field] = wrong
    row = route_score([{"requested_route": request, "actual_route": actual}], model="gpt-6-sol")
    assert row["status"] == "failed"


def test_confirmed_route_stays_observed_when_work_fails_after_the_model_request():
    route = {"worker_backend": "dsh", "provider": "openai", "model": "gpt-6-sol", "effort": "high"}
    row = route_score([{"state": "failed", "requested_route": route, "actual_route": {**route, "source": "runtime-request"}}], model="gpt-6-sol")
    assert row["status"] == "passed"


def test_dotted_remote_events_normalize_and_retain_exact_raw_evidence():
    raw = [{"type": "node.started", "node_id": "gate_strategy"},
           {"type": "node.failed", "node_id": "gate_strategy"},
           {"type": "node.started", "node_id": "gate_strategy"},
           {"type": "node.completed", "node_id": "gate_strategy"}]
    events = normalize(raw, trial_id="remote-control", source="remote-events.json")
    assert [event["kind"] for event in events] == ["node_started", "node_failed", "node_started", "node_completed"]
    assert [event["visit"] for event in events] == [1, 1, 2, 2]
    assert [event["status"] for event in events] == ["running", "failed", "running", "completed"]
    assert [event["raw_kind"] for event in events] == [record["type"] for record in raw]
    assert [event["raw_evidence"] for event in events] == [f"remote-events.json#record={index}" for index in range(4)]
    assert stage_score(raw)["status"] == "passed"


def test_human_gate_pending_failure_followed_by_completed_resume_passes_latest_visit():
    row = stage_score([{"kind": "node_started", "node_id": "gate_strategy"},
                       {"kind": "node_failed", "node_id": "gate_strategy"},
                       {"kind": "node_started", "node_id": "gate_strategy"},
                       {"kind": "node_completed", "node_id": "gate_strategy"}])
    assert row["status"] == "passed"
    assert "earlier failed visits" in row["reason"]
    assert len(row["evidence"]) == 4


@pytest.mark.parametrize("ending,expected", [("node_failed", "failed"), ("node_started", "unavailable")])
def test_previous_success_cannot_mask_latest_failed_or_unfinished_visit(ending, expected):
    row = stage_score([{"kind": "node_completed", "node_id": "gate_strategy"},
                       {"kind": ending, "node_id": "gate_strategy"}])
    assert row["status"] == expected


def test_dotted_remote_first_stage_failure_is_a_failed_stage():
    events = normalize([{"type": "node.started", "node_id": "init_user_intent"},
                        {"type": "node.failed", "node_id": "init_user_intent", "error": "source shell quoting defect"}],
                       trial_id="reference", source="remote-events.json")
    row = next(row for row in authoring_score(events, {}) if row["requirement_id"] == "F.stage.init_user_intent")
    assert row["status"] == "failed"
