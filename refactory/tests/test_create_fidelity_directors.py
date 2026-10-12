"""Controlled external responses across real director/Executor boundaries."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from refactory_dsh import create_fidelity_directors as directors
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn.executor import Executor, RunContext
from refactory_dsh.nn.graph import Graph, Node


def plan_for(stage, count):
    if stage is directors.RESEARCH:
        required = [{"id": "workflow-patterns", "text": "Existing native workflow patterns and graph construction"},
                    {"id": "node-types", "text": "Node types, data flow, gate logic and recovery"},
                    {"id": "mode-requirements", "text": "Mode purpose and user requirements"}]
    elif stage is directors.STRATEGY:
        required = [{"id": "architecture", "text": "Concrete graph node IDs, types and edge wiring"},
                    {"id": "testing-verification", "text": "Explicit acceptance criteria and verification evidence"}]
    else:
        required = [{"id": "acceptance-1", "text": "Booking results preserve all IDs"},
                    {"id": "user-intent", "text": "Verify implementation against every original user ask"}]
    intent = [{"id": "booking-results", "text": "Preserve all booking IDs", "source_line": 4, "priority": "required"}]
    coverage = required + intent
    tasks = [{"id": f"child-{i + 1}", "slug": f"child-{i + 1}", "role": stage.role,
              "focus": f"Concrete independent perspective {i + 1}",
              "prompt": f"Investigate booking evidence, concrete native graph and acceptance obligations from perspective {i + 1}.",
              "output": f"{stage.reports}/child-{i + 1}.md",
              "coverage": [item["id"] for item in coverage]} for i in range(count)]
    return {"schema_version": 1, "stage_id": stage.id, "tasks": tasks,
            "intent_items": intent, "coverage_items": coverage}


class ExternalResponses:
    """Only the external model response is fake; routing and receipts are real."""

    def __init__(self, plans, problem=None, quality_retry=False):
        self.plans = plans
        self.problem = problem
        self.quality_retry = quality_retry
        self.calls = []
        self.finished = []
        self.full_quality_bodies = []
        self.planning_prompts = {}
        self.lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def complete(self, client, prompt, *, system="", workdir=None):
        node = client._node_id
        stage, visit, task, attempt = node.split(":")
        record = json.loads(client._worker_receipt.read_text())
        # This is the information the external DSH observer supplies in a real
        # completion. The production begin/finish functions still own the receipt.
        record["actual_route"] = client.route()
        if self.problem == "actual-codex" and task.startswith("child-"):
            record["actual_route"] = {"worker_backend": "codex", "model": "unexpected"}
        if self.problem == "actual-model" and task.startswith("child-"):
            record["actual_route"]["model"] = "unexpected-downgrade"
        client._worker_receipt.write_text(json.dumps(record))
        if self.problem == "foreign-inflight":
            (Path(workdir) / ".refactory/workers/foreign-inflight.json").write_text('{"node_id":')
            foreign_delivery = Path(workdir) / ".refactory/agent-delivery/foreign-inflight/receipt.json"
            foreign_delivery.parent.mkdir(parents=True, exist_ok=True)
            foreign_delivery.write_text('{"node":')
        with self.lock:
            self.calls.append((node, client.route()))
        if task == "plan":
            self.planning_prompts[stage] = prompt
            return json.dumps(self.plans[stage])
        if task == "quality":
            joined = prompt.split("COMPLETE JOINED CHILD REPORTS AND RECEIPTS:\n", 1)[1].split("\n\n=== COMPLETE INPUT", 1)[0]
            reports = json.loads(joined)
            self.full_quality_bodies.append([report["body"] for report in reports])
            decisions = []
            for report in reports:
                retry = report["status"] != "completed" or (self.quality_retry and task == "quality" and attempt == "a1" and report["task_id"] == "child-1")
                decisions.append({"id": report["task_id"], "verdict": "retry" if retry else "accept",
                                  "reason": "Assigned acceptance evidence is missing" if retry else "Full assigned evidence checked",
                                  "retry_prompt": "Run the missing acceptance probe and report exact evidence for this same assigned focus.",
                                  "outcome": "failed" if retry else "passed"})
            return json.dumps({"decisions": decisions, "summary": "Reviewed complete bodies and every original ask against concrete acceptance evidence.",
                               "coverage_complete": not any(decision["verdict"] == "retry" for decision in decisions)})
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            # Distinct peers complete in a different order than their plan.
            time.sleep(.03 if task == "child-1" else .005)
            if self.problem == "deadline":
                while True:
                    client._raise_if_cancelled()
                    time.sleep(.01)
            if task == "child-1" and self.problem == "failed":
                raise RuntimeError("External worker crashed during investigation")
            if task == "child-1" and self.problem in ("missing", "stale"):
                return ""
            if task == "child-1" and self.problem == "thin":
                return "One line."
            if self.problem == "duplicate" and task in ("child-1", "child-2"):
                task = "duplicate"
            body = f"# Actual {task} report\n\nCOMPLETE-SUBSTANTIVE-EVIDENCE-{stage}-{task}-{attempt}\nLiteral native template: {{{{name}}}}\n" + (
                "Verified concrete native graph IDs, gate edges and booking behavior against independently executed probes. "
                "Observed exact command output and preserved acceptance evidence for all assigned user asks. " * 3)
            if stage == directors.STRATEGY.id:
                body += "\n### Acceptance Criteria\n- [ ] Preserve all booking IDs\n"
            return body
        finally:
            with self.lock:
                self.active -= 1
                self.finished.append(node)


def seed_inputs(workdir, stage):
    for name in stage.reads:
        path = workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("research_director-findings.json"):
            path.write_text(json.dumps({"status": "completed", "reports": [{"body": "UPSTREAM-FULL-REPORT-CONSUMPTION " * 12}]}))
        elif name.endswith("research-plan.json"):
            path.write_text("{}")
        elif name.endswith("current.md"):
            path.write_text("# Booking strategy\n### Acceptance Criteria\n- [ ] Booking results preserve all IDs\n")
        else:
            path.write_text("Concrete project context and user asks: preserve all booking IDs.\n")


def run_stage(tmp_path, monkeypatch, stage=directors.RESEARCH, count=3, **options):
    responses = ExternalResponses({stage.id: plan_for(stage, count)},
                                  problem=options.pop("problem", None), quality_retry=options.pop("quality_retry", False))
    monkeypatch.setattr(DshModelClient, "_complete", lambda client, *args, **kwargs: responses.complete(client, *args, **kwargs))
    seed_inputs(tmp_path, stage)
    if responses.problem == "stale":
        stale = tmp_path / f"{stage.reports}/child-1.md"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("Stale report from a previous run must not be accepted. " * 20)
    graph = Graph()
    graph.add(Node(id=stage.id, kind="op", call=f"refactory_dsh.create_fidelity_directors:{stage.id}",
                   writes=(".factory/",), role="ceo"))
    client = DshModelClient(provider="fixture", model="sol-6.1", effort="high", attempts=1,
                            role_routes=options.pop("role_routes", None), node_routes=options.pop("node_routes", None))
    client = client.with_node_timeout(options.pop("stage_timeout", None))
    ctx = RunContext(tmp_path, model=client, knobs={"child_attempts": options.pop("child_attempts", 2), "director_parallelism": 7},
                     task={"trial_id": "controlled-director-trial"})
    assert not options
    result = Executor(graph, ctx).run()
    manifest = json.loads((tmp_path / f".factory/{stage.id}-findings.json").read_text()) if (tmp_path / f".factory/{stage.id}-findings.json").exists() else None
    return result, manifest, responses


@pytest.mark.parametrize("stage,count", [(directors.RESEARCH, 3), (directors.RESEARCH, 7),
                                         (directors.STRATEGY, 2), (directors.STRATEGY, 5),
                                         (directors.QA, 2), (directors.QA, 5)])
def test_real_executor_directors_extrema_and_full_report_delivery(tmp_path, monkeypatch, stage, count):
    result, manifest, responses = run_stage(tmp_path, monkeypatch, stage, count)
    assert result.exit_ok, result.error
    assert result.facts == {}
    assert manifest["status"] == "completed"
    assert len(manifest["reports"]) == count + (2 if stage is directors.QA else 0)
    assert responses.maximum_active >= 2
    assert responses.finished[0].split(":")[2] != "child-1"
    assert manifest["intent_items"][0]["source_line"] == 4
    assert manifest["intent_items"][0]["priority"] == "required"
    markdown = (tmp_path / f".factory/{stage.id}-findings.md").read_text()
    for report in manifest["reports"]:
        assert report["body"] in markdown
        assert report["body"] in responses.full_quality_bodies[0]
        assert report["plan_digest"] == manifest["plan_digest"]
        assert report["run_id"] == "controlled-director-trial"
        assert report["actual_route"]["worker_backend"] == "dsh"
        assert report["raw_evidence"]["worker_receipts"]
        assert report["raw_evidence"]["delivery_receipts"]
        delivery = json.loads((tmp_path / report["raw_evidence"]["delivery_receipts"][0]).read_text())
        supplied = (tmp_path / delivery["contexts"][-1]["prompt"]).read_bytes()
        assert report["input_digest"] == hashlib.sha256(supplied).hexdigest()
        assert report["steps"][0]["kind"] == "agent"
    if stage is directors.STRATEGY:
        assert "UPSTREAM-FULL-REPORT-CONSUMPTION" in responses.planning_prompts[stage.id]
    if stage is directors.QA:
        assert {report["task_id"] for report in manifest["reports"]} >= set(directors.MANDATORY)
    assert json.loads((tmp_path / stage.plan).read_text())["tasks"]
    assert (tmp_path / f".factory/{stage.id}-quality.md").read_text().strip()


@pytest.mark.parametrize("problem", ["missing", "thin", "duplicate", "stale", "failed", "actual-codex", "actual-model"])
def test_child_evidence_failures_never_complete(tmp_path, monkeypatch, problem):
    result, manifest, responses = run_stage(tmp_path, monkeypatch, problem=problem)
    assert not result.exit_ok
    assert manifest["status"] == "failed"
    assert "exhausted same-child retries" in result.error
    assert len(responses.full_quality_bodies) == 2
    failed = [report for report in manifest["reports"] if report["status"] != "completed"]
    assert failed
    assert all(report["attempt"] == 2 for report in failed)


def test_director_quality_retry_uses_same_child_and_preserves_prior_receipts(tmp_path, monkeypatch):
    result, manifest, responses = run_stage(tmp_path, monkeypatch, quality_retry=True)
    assert result.exit_ok, result.error
    assert len(manifest["attempts"]) == 4
    attempts = [report for report in manifest["attempts"] if report["task_id"] == "child-1"]
    assert [report["attempt"] for report in attempts] == [1, 2]
    assert attempts[0]["task_digest"] == attempts[1]["task_digest"]
    assert attempts[0]["input_digest"] != attempts[1]["input_digest"]
    assert attempts[0]["raw_evidence"] != attempts[1]["raw_evidence"]
    assert len(manifest["quality_reviews"]) == 2
    assert all(report["attempt"] == 1 for report in manifest["reports"] if report["task_id"] != "child-1")


def test_role_routes_are_resolved_and_actual_routes_recorded(tmp_path, monkeypatch):
    result, manifest, responses = run_stage(tmp_path, monkeypatch,
        role_routes={"researcher": {"provider": "research-provider", "model": "research-model", "effort": "high"},
                     "ceo": {"provider": "director-provider", "model": "director-model", "effort": "high"}})
    assert result.exit_ok, result.error
    assert all(report["requested_route"]["model"] == "research-model" for report in manifest["reports"])
    assert all(route["model"] == "director-model" for node, route in responses.calls if ":plan:" in node or ":quality:" in node)


def test_external_nonbuilder_route_is_rejected_before_child_dispatch(tmp_path, monkeypatch):
    result, manifest, responses = run_stage(tmp_path, monkeypatch, role_routes={"researcher": {"worker_backend": "codex"}})
    assert not result.exit_ok
    assert "must execute in DSH" in result.error
    assert not [node for node, _ in responses.calls if ":child-" in node]


def test_logical_node_route_overrides_survive_unique_execution_receipt_ids(tmp_path, monkeypatch):
    result, manifest, responses = run_stage(tmp_path, monkeypatch,
        role_routes={"researcher": {"model": "role-default"}, "ceo": {"model": "ceo-role-default"}},
        node_routes={"research_director": {"model": "director-node-override"}, "child-1": {"model": "child-node-override"}})
    assert result.exit_ok, result.error
    assert manifest["reports"][0]["requested_route"]["model"] == "child-node-override"
    assert manifest["reports"][1]["requested_route"]["model"] == "role-default"
    assert all(route["model"] == "director-node-override" for node, route in responses.calls if ":plan:" in node or ":quality:" in node)


def test_research_reports_reach_strategy_through_actual_executor_graph(tmp_path, monkeypatch):
    responses = ExternalResponses({directors.RESEARCH.id: plan_for(directors.RESEARCH, 3),
                                   directors.STRATEGY.id: plan_for(directors.STRATEGY, 2)})
    monkeypatch.setattr(DshModelClient, "_complete", lambda client, *args, **kwargs: responses.complete(client, *args, **kwargs))
    seed_inputs(tmp_path, directors.RESEARCH)
    graph = Graph()
    for stage in (directors.RESEARCH, directors.STRATEGY):
        graph.add(Node(id=stage.id, kind="op", call=f"refactory_dsh.create_fidelity_directors:{stage.id}", writes=(".factory/",)))
    graph.connect(directors.RESEARCH.id, directors.STRATEGY.id)
    ctx = RunContext(tmp_path, facts={"name": "booking_conflicts_bench"}, model=DshModelClient(provider="fixture", model="sol-6.1", effort="high", attempts=1))
    result = Executor(graph, ctx).run()
    assert result.exit_ok, result.error
    research = json.loads((tmp_path / ".factory/research_director-findings.json").read_text())
    strategy_prompt = responses.planning_prompts[directors.STRATEGY.id]
    for report in research["reports"]:
        assert report["body"] in strategy_prompt
    assert ctx.facts == {"name": "booking_conflicts_bench"}


def test_native_op_reconstructs_inherited_dsh_client(tmp_path, monkeypatch):
    responses = ExternalResponses({directors.RESEARCH.id: plan_for(directors.RESEARCH, 3)})
    monkeypatch.setattr(DshModelClient, "_complete", lambda client, *args, **kwargs: responses.complete(client, *args, **kwargs))
    monkeypatch.setenv("REFACTORY_WORKER_CONTEXT", json.dumps({"routes_resolved": True, "worker_backend": "dsh",
        "provider": "native-fixture-provider", "model": "sol-6.1", "effort": "high", "permission_mode": "skip"}))
    seed_inputs(tmp_path, directors.RESEARCH)
    graph = Graph()
    graph.add(Node(id=directors.RESEARCH.id, kind="op", call="refactory_dsh.create_fidelity_directors:research_director", writes=(".factory/",)))
    ctx = RunContext(tmp_path, model=None)
    result = Executor(graph, ctx).run()
    assert result.exit_ok, result.error
    assert all(route["provider"] == "native-fixture-provider" and route["model"] == "sol-6.1" and route["effort"] == "high" for _, route in responses.calls)


def test_composite_deadline_cancels_all_owned_children_and_settles_threads(tmp_path, monkeypatch):
    before = {thread.ident for thread in threading.enumerate()}
    result, manifest, responses = run_stage(tmp_path, monkeypatch, problem="deadline", stage_timeout=1)
    assert not result.exit_ok
    assert "deadline expired" in result.error
    assert manifest["status"] == "failed"
    assert all(report["status"] == "failed" for report in manifest["reports"])
    assert responses.active == 0
    assert not [thread for thread in threading.enumerate() if thread.ident not in before and thread.name.startswith("ThreadPoolExecutor")]


def test_receipt_collection_ignores_incomplete_concurrent_peer_writes(tmp_path, monkeypatch):
    result, manifest, responses = run_stage(tmp_path, monkeypatch, stage=directors.QA, count=5, problem="foreign-inflight")
    assert result.exit_ok, result.error
    assert len(manifest["reports"]) == 7
    assert all(len(report["raw_evidence"]["worker_receipts"]) == 1 for report in manifest["reports"])
    assert all(len(report["raw_evidence"]["delivery_receipts"]) == 1 for report in manifest["reports"])


@pytest.mark.parametrize("stage,count", [(directors.RESEARCH, 2), (directors.RESEARCH, 8),
                                         (directors.STRATEGY, 1), (directors.STRATEGY, 6),
                                         (directors.QA, 1), (directors.QA, 6)])
def test_invalid_plan_counts_rejected_before_execution(stage, count):
    with pytest.raises(directors.ExecutionError, match="custom children"):
        directors.normalize_plan(plan_for(stage, count), stage, [])


@pytest.mark.parametrize("mutation", ["duplicate-id", "duplicate-output", "wrong-role", "escape-output", "missing-coverage", "unknown-field", "reserved-slug", "drop-intent"])
def test_strict_plan_identity_route_output_and_coverage_validation(mutation):
    stage = directors.RESEARCH
    plan = plan_for(stage, 3)
    if mutation == "duplicate-id":
        plan["tasks"][1]["id"] = plan["tasks"][0]["id"]
    elif mutation == "duplicate-output":
        plan["tasks"][1]["slug"] = plan["tasks"][0]["slug"]
        plan["tasks"][1]["output"] = plan["tasks"][0]["output"]
    elif mutation == "wrong-role":
        plan["tasks"][0]["role"] = "builder"
    elif mutation == "escape-output":
        plan["tasks"][0]["output"] = "../outside.md"
    elif mutation == "missing-coverage":
        for task in plan["tasks"]:
            task["coverage"] = ["booking-results"]
    elif mutation == "unknown-field":
        plan["tasks"][0]["worker_backend"] = "codex"
    elif mutation == "reserved-slug":
        plan["tasks"][0]["slug"] = "workflow-validate"
    elif mutation == "drop-intent":
        plan["coverage_items"] = plan["coverage_items"][:-1]
    with pytest.raises(directors.ExecutionError):
        directors.normalize_plan(plan, stage, [])


def test_required_acceptance_cannot_be_removed_or_changed():
    plan = plan_for(directors.QA, 2)
    required = copy.deepcopy(plan["coverage_items"][:1])
    plan["coverage_items"][0]["text"] = "Weaker acceptance claim"
    with pytest.raises(directors.ExecutionError, match="dropped or changed"):
        directors.normalize_plan(plan, directors.QA, required)


def test_source_requirement_ids_preserve_case_and_metadata_through_typed_plan():
    plan = plan_for(directors.RESEARCH, 3)
    items = [{"id": "B01", "text": "An empty array is valid", "source": "user-intent.md:9"},
             {"id": "REQ:B02.integer", "text": "Booleans are not integers", "source_line": 10}]
    plan["intent_items"].extend(items)
    plan["coverage_items"].extend(copy.deepcopy(items))
    for task in plan["tasks"]:
        task["coverage"].extend(item["id"] for item in items)
    normalized = directors.normalize_plan(plan, directors.RESEARCH, [])
    assert normalized["intent_items"][-2:] == items
    assert normalized["coverage_items"][-2:] == items
    assert all("B01" in task["coverage"] for task in normalized["tasks"])


@pytest.mark.parametrize("identifier", ["", " B01", "B01 ", "B01/child", "../B01", "B01\nB02", "B" * 129])
def test_requirement_id_schema_refuses_ambiguous_or_unsafe_references(identifier):
    with pytest.raises(directors.ExecutionError, match="item ID"):
        directors._items([{"id": identifier, "text": "Required behavior"}], "intent_items")


def test_planning_prompt_explicitly_distinguishes_task_and_requirement_id_schemas(tmp_path):
    ctx = RunContext(tmp_path)
    prompt = directors._plan_prompt(ctx, directors.RESEARCH, {}, [])
    assert "Task id and slug must be lower-case kebab-case" in prompt
    assert "Preserve existing requirement IDs EXACTLY, including case and punctuation" in prompt
    assert "Task coverage must cite these exact IDs" in prompt


def test_requirement_ids_cross_actual_planning_dispatch_quality_and_publication(tmp_path, monkeypatch):
    original = plan_for

    def requirements_plan(stage, count):
        plan = original(stage, count)
        item = {"id": "B01", "text": "Input is a JSON array and an empty array is valid", "source": "user-intent.md:9"}
        plan["intent_items"].append(item)
        plan["coverage_items"].append(dict(item))
        for task in plan["tasks"]:
            task["coverage"].append("B01")
        return plan

    monkeypatch.setitem(plan_for.__globals__, "plan_for", requirements_plan)
    result, manifest, responses = run_stage(tmp_path, monkeypatch)
    assert result.exit_ok, result.error
    assert manifest["intent_items"][-1] == requirements_plan(directors.RESEARCH, 3)["intent_items"][-1]
    assert all(report["status"] == "completed" for report in manifest["reports"])
    assert responses.full_quality_bodies


def test_failed_plan_validation_retains_diagnostic_schema_and_raw_response(tmp_path, monkeypatch):
    original = plan_for

    def invalid_plan(stage, count):
        plan = original(stage, count)
        plan["intent_items"][0]["id"] = "invalid requirement ID with spaces"
        return plan

    monkeypatch.setitem(plan_for.__globals__, "plan_for", invalid_plan)
    result, manifest, responses = run_stage(tmp_path, monkeypatch)
    assert not result.exit_ok
    assert manifest is None
    validation = json.loads((tmp_path / ".factory/directors/research_director/v1/plan-validation.json").read_text())
    assert validation["status"] == "failed"
    assert "item ID" in validation["error"]
    assert validation["item_id_pattern"] == directors.ITEM_ID.pattern
    assert validation["required_coverage"]
    raw = json.loads((tmp_path / validation["raw_plan_receipt"]).read_text())
    assert "invalid requirement ID with spaces" in raw["body"]
    assert raw["raw_evidence"]["worker_receipts"]
    assert not [node for node, _ in responses.calls if ":child-" in node]
