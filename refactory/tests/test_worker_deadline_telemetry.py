"""Deadline observations use real dispatch/ledger policy, with external I/O controlled."""
from __future__ import annotations

from datetime import datetime
import json
import subprocess
import time

import pytest

from refactory_dsh.budgets import ResourceLimits, RunBudget
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn import Graph, Node
from refactory_dsh.nn.executor import execute


def settled_budget(root, *, maximum=100, reserve=30, elapsed=20, phase="work", tokens=None):
    budget = RunBudget.create(root, ResourceLimits(max_elapsed_s=maximum,
                             validation_reserve_s=reserve, max_tokens=tokens))
    budget.pause()
    with budget._locked() as state:
        state["elapsed_before_s"] = elapsed
        state["phase"] = phase
    return budget


def receipt(root):
    return json.loads(next((root / ".refactory/workers").glob("*.json")).read_text())


def controlled_wire(monkeypatch, inspect):
    def wire(self, cmd, run_dir, stop_at=None, budget_s=None):
        record = json.loads(self._worker_receipt.read_text())
        response = inspect(record, stop_at, budget_s)
        record["actual_route"] = {**self.route(), "source": "controlled-external-I/O"}
        record["usage"] = {"status": "measured", "tokens": 0}
        self._worker_receipt.write_text(json.dumps(record))
        return response or subprocess.CompletedProcess(cmd, 0, "A complete fresh worker report.", "")
    monkeypatch.setattr(DshModelClient, "_run_completion", wire)


@pytest.mark.parametrize("phase,stage,remaining,limiter", [
    ("work", 75, 50, "run_budget"),
    ("final_validation", 10, 80, "stage"),
])
def test_real_executor_samples_budget_reserve_and_stage_clock_before_dispatch(
        tmp_path, monkeypatch, phase, stage, remaining, limiter):
    budget = settled_budget(tmp_path, phase=phase)
    observed = []

    def inspect(record, stop_at, wall):
        sample = record["effective_deadlines"]["dispatches"][-1]
        observed.append(sample)
        assert sample["attempt"] == 1 and sample["sample_phase"] == "pre_transport_dispatch"
        assert datetime.fromisoformat(sample["sampled_at"]).tzinfo is not None
        assert sample["run_remaining_s"] == remaining
        assert 0 < sample["stage_remaining_s"] <= wall == stage
        assert stop_at is not None
        assert sample["stage_clock"] == {"scope": "absolute_stage", "status": "active"}
        assert sample["effective_remaining_s"] == min(sample["stage_remaining_s"], remaining)
        assert sample["effective_limiters"] == [limiter]
        resources = sample["run_budget_snapshot"]
        assert resources["path"] == str(budget.path)
        assert resources["phase"] == phase and resources["elapsed_s"] == 20
        assert resources["max_elapsed_s"] == 100 and resources["validation_reserve_s"] == 30
        assert resources["retained_validation_reserve_s"] == (30 if phase == "work" else 0)
        assert sample["idle_timer"]["status"] == "inactive"
        assert sample["idle_timer"]["timeout_s"] is None

    controlled_wire(monkeypatch, inspect)
    graph = Graph()
    graph.add(Node("work", "agent", prompt_template="Deliver a substantive report.",
                   writes=("report.md",), timeout_s=stage))
    result = execute(graph, tmp_path, model=DshModelClient(attempts=1).with_budget(budget))
    assert result.exit_ok, result.error
    final = receipt(tmp_path)
    assert final["state"] == "completed"
    assert final["effective_deadlines"]["dispatches"] == observed


def test_retry_records_diminishing_absolute_stage_and_run_allowance_without_reset(tmp_path, monkeypatch):
    budget = RunBudget.create(tmp_path, ResourceLimits(max_elapsed_s=200, validation_reserve_s=25))
    observed = []
    boundaries = []

    def inspect(record, stop_at, wall):
        sample = record["effective_deadlines"]["dispatches"][-1]
        observed.append(sample)
        boundaries.append((stop_at, wall))
        if len(observed) == 1:
            time.sleep(.02)
            return subprocess.CompletedProcess([], 1, "", "TRANSPORT: connection reset")

    controlled_wire(monkeypatch, inspect)
    client = DshModelClient(attempts=2, backoff_s=0).with_budget(budget).for_node("retry").with_node_timeout(75)
    assert client.complete("Perform the work.", workdir=tmp_path)
    samples = receipt(tmp_path)["effective_deadlines"]["dispatches"]
    assert samples == observed and [sample["attempt"] for sample in samples] == [1, 2]
    assert boundaries[0] == boundaries[1]  # the same absolute wall, not a fresh retry timer
    assert samples[1]["stage_remaining_s"] < samples[0]["stage_remaining_s"] <= 75
    assert samples[1]["run_remaining_s"] < samples[0]["run_remaining_s"] <= 175
    assert all(sample["effective_limiters"] == ["stage"] for sample in samples)
    assert all(sample["idle_timer"]["status"] == "inactive" for sample in samples)


def test_plain_completion_marks_unstarted_window_and_absent_budget_without_invented_values(tmp_path, monkeypatch):
    controlled_wire(monkeypatch, lambda *_: None)
    assert DshModelClient(timeout_s=9, attempts=1).complete("Work.", workdir=tmp_path)
    sample = receipt(tmp_path)["effective_deadlines"]["dispatches"][0]
    assert sample["stage_remaining_s"] == 9
    assert sample["stage_clock"] == {"scope": "per_completion", "status": "window_not_started"}
    assert sample["run_remaining_s"] is None and sample["run_budget_snapshot"] is None
    assert sample["effective_remaining_s"] == 9 and sample["effective_limiters"] == ["stage"]
    assert sample["idle_timer"]["status"] == "inactive"


def test_token_only_ledger_does_not_invent_elapsed_deadline(tmp_path, monkeypatch):
    budget = settled_budget(tmp_path, maximum=None, reserve=0, tokens=100)
    controlled_wire(monkeypatch, lambda *_: None)
    client = DshModelClient(attempts=1).with_budget(budget).for_node("tokens").with_node_timeout(10)
    assert client.complete("Work.", workdir=tmp_path)
    sample = receipt(tmp_path)["effective_deadlines"]["dispatches"][0]
    assert sample["run_remaining_s"] is None
    assert sample["run_budget_snapshot"]["max_elapsed_s"] is None
    assert sample["run_budget_snapshot"]["elapsed_limit_status"] == "inactive"
    assert sample["run_budget_snapshot"]["token_status"] == "unavailable"
    assert sample["effective_limiters"] == ["stage"]


def test_native_worker_bridge_samples_actual_inherited_ledger_at_dispatch(tmp_path, monkeypatch):
    from refactory_dsh.worker_node import execute as execute_native
    budget = settled_budget(tmp_path)
    (tmp_path / "input.md").write_text("The original input.")
    controlled_wire(monkeypatch, lambda *_: None)
    client = DshModelClient(attempts=1).with_budget(budget).with_node_timeout(75)
    delivered = execute_native({"schema_version": 1, "node_id": "native-review", "role": "critic",
                               "reads": ["input.md"], "writes": ["report.md"],
                               "prompt": "Read the input and deliver a report.", "produces": [],
                               "model_context": client.context()}, tmp_path)
    assert delivered["status"] == "delivered"
    worker = receipt(tmp_path)
    assert worker["node_id"] == "native-review" and worker["role"] == "critic"
    sample = worker["effective_deadlines"]["dispatches"][0]
    assert sample["run_remaining_s"] == 50 and sample["effective_limiters"] == ["run_budget"]
    assert sample["run_budget_snapshot"]["path"] == str(budget.path)


def test_real_composite_dispatch_reports_existing_outer_timer_as_minimum(tmp_path, monkeypatch):
    from test_create_fidelity_directors import ExternalResponses, plan_for, seed_inputs
    from refactory_dsh.create_fidelity_directors import RESEARCH
    responses = ExternalResponses({RESEARCH.id: plan_for(RESEARCH, 3)})

    def wire(client, cmd, run_dir, stop_at=None, budget_s=None):
        output = responses.complete(client, cmd[-1], workdir=run_dir)
        return subprocess.CompletedProcess(cmd, 0, output, "")

    monkeypatch.setattr(DshModelClient, "_run_completion", wire)
    seed_inputs(tmp_path, RESEARCH)
    graph = Graph()
    graph.add(Node(RESEARCH.id, "op", call="refactory_dsh.create_fidelity_directors:research_director",
                   writes=(".factory/",), role="ceo"))
    result = execute(graph, tmp_path, model=DshModelClient(attempts=1))
    assert result.exit_ok, result.error
    samples = []
    for path in (tmp_path / ".refactory/workers").glob("*.json"):
        worker = json.loads(path.read_text())
        sample = worker["effective_deadlines"]["dispatches"][0]
        outer, = sample["enclosing_deadlines"]
        assert outer["scope"] == "composite_stage"
        assert "caller-owned timer" in outer["enforcement"]
        assert 0 < outer["remaining_s"] <= 3600
        assert sample["effective_remaining_s"] == min(sample["stage_remaining_s"], outer["remaining_s"])
        samples.append((worker["node_id"], sample))
    planner = next(sample for node, sample in samples if ":plan:" in node)
    assert planner["effective_limiters"] == ["composite_stage"]
    from benchmarks.create_fidelity.deadline_evidence import dispatch_deadline_row
    scored = dispatch_deadline_row([(str(path), json.loads(path.read_text()))
                                   for path in (tmp_path / ".refactory/workers").glob("*.json")])
    assert scored["status"] == "passed", scored["reason"]
