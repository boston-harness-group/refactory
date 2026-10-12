"""Full candidate orchestration; only the external DSH response is controlled.

The generic document fixture is separate from every booking benchmark artifact.
It is authored by controlled builder I/O, then admitted and invoked by real
native entry points. Scripted evidence is never presented as live model work.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from refactory_dsh import create_fidelity_ops as ops
from refactory_dsh.create_fidelity import CreateFidelityMode
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn import Executor, RunContext
from refactory_dsh.steering import Ledger
from test_create_fidelity_graph_ops import make_package

BRIEF = "Create a reusable native document mode with runtime document_path; copy JSON objects to result.json in each invocation's own run directory and reject arrays honestly."
CRITERION = "Runtime JSON is copied to invocation-specific result.json and invalid arrays fail honestly"


class ExternalDshIO:
    def __init__(self, rework=False, fail_precheck=False):
        self.calls = []
        self.prompts = {}
        self.builder_count = 0
        self.rework = rework
        self.fail_precheck = fail_precheck
        self.builder_started = threading.Event()
        self.plan_archive_saw_builder = False
        self.lock = threading.Lock()

    def complete(self, client, prompt, *, system="", workdir=None):
        node, run = client._node_id, Path(workdir)
        # The fake supplies only observer information that the external DSH
        # process supplies. Production complete() owns start/finish receipts.
        record = json.loads(client._worker_receipt.read_text())
        record["actual_route"] = client.route()
        client._worker_receipt.write_text(json.dumps(record))
        with self.lock:
            self.calls.append(node)
            self.prompts.setdefault(node, []).append(prompt)
        if ":" in node:
            return self.director_response(node, prompt, run)
        if node.startswith("gate_"):
            request = json.JSONDecoder().raw_decode(prompt.split("Runtime evidence (read current artifacts before judging):\n", 1)[1])[0]
            verdict = "reloop" if node == "gate_build" and self.rework and self.builder_count == 1 else "proceed"
            return json.dumps({"schema_version": 1, "gate_id": node, "input_digest": request["input_digest"],
                               "evidence_digest": request["evidence_digest"], "verdict": verdict,
                               "rationale": "User requested documented reusable native discovery; README.md is missing and must be supplied." if verdict == "reloop" else "Reviewed the current full artifacts and actual native command evidence against every original requirement."})
        if node == "create_factory_md":
            return "# Factory\n\n## Goal\nBuild the original reusable document mode.\n\n## Scope\nNative local artifacts.\n\n## Guards\nPreserve runtime inputs and user requirements.\n\n## Eval command\nRun native structural admission and distinct-input execution.\n"
        if node == "graph_explorer":
            return "# Graph context\n\nSource main.py defines main and imports json. The native graph inventory establishes these exact module and function relationships; Python mode operations consume runtime knobs and produce run-relative artifacts.\n"
        if node == "synthesize_strategy":
            # Test visibility of full child substantive bodies, not just paths.
            assert "strategy_director-findings.md" in prompt
            findings = json.loads((run / ".factory/strategy_director-findings.json").read_text())
            assert all(report["body"] in prompt for report in findings["reports"])
            return ("# Approved native document strategy\n\n### Graph Topology\nread: op is the start and terminal of a complete one-node DAG.\n\n"
                    "### Node Definitions\nread: op, role ceo, call portable_probe.ops:read, knobs document_path, reads none, writes result.json; input is supplied at invocation time.\n\n"
                    "### Edge Wiring\nNo edges; the single operation is terminal.\n\n### Phased Plan\nImplement reusable JSON object copying with useful failures.\n\n"
                    "### Acceptance Criteria\n- [ ] " + CRITERION + "\n\n### MVP Scope\nAll original requested behaviors.\n\n### Deferred Features\nNone.\n\n"
                    "### Intent Coverage\n| User Ask | Where | Status |\n| Runtime document input, independent output, honest invalid arrays | read and acceptance criterion | Covered |\n")
        if node == "builder":
            self.builder_started.set()
            with self.lock:
                self.builder_count += 1
                count = self.builder_count
            root = make_package(run)
            if self.rework and count == 1:
                (root / "README.md").unlink()
            report = ops.admit_native_package(run, "portable_probe", requires_input=("document_path",))
            assert report["ok"], report
            test_input = run / ".factory/directors/controlled-input.json"
            test_input.parent.mkdir(parents=True, exist_ok=True)
            test_input.write_text('{"stage":"builder"}')
            result = ops.run_native_mode(run, "portable_probe/read", run / f"builder-check-{count}", {"document_path": str(test_input)})
            assert result["exit_ok"], result
            output = run / ".factory/reviews/builder-latest.md"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("# Builder attempt " + str(count) + "\n\nNative package is reusable; runtime document_path reads external input.\n\nCommand: candidate admit portable_probe --requires-input document_path\nActual output:\n" + json.dumps(report) + "\n\nCommand: candidate native run portable_probe/read\nActual output:\n" + json.dumps(result) + "\nREADME status: " + ("missing" if self.rework and count == 1 else "delivered") + "\n")
            return "Delivered the package and truthful builder report."
        if node in ("health_checker", "code_reviewer", "overwatch"):
            check = ops.validate_native_package(run, "portable_probe", requires_input=("document_path",))
            assert check["ok"], check
            if node == "overwatch" and self.fail_precheck:
                # Deliberately break a docs artifact after successful model QA.
                # CEO guards see admitted source unchanged; final precheck must
                # independently diagnose stale skill bytes and archive failure.
                skill = run / ".factory/packages/portable_probe/SKILL.md"
                skill.write_text(skill.read_text() + "\nUnexpected post-admission modification\n")
            return "# " + node + " report\n\nPASS: every original user ask and criterion was inspected against actual native package code and executed evidence.\nCommand: candidate validate portable_probe --requires-input document_path\nActual output:\n" + json.dumps(check) + "\nAcceptance outcome, connected scope and input/output evidence are present. No blocking code issues were observed.\n"
        if node == "archivist_plan":
            self.plan_archive_saw_builder = self.builder_started.wait(3)
            return "# Approved plan archive\n\nPreserved the approved original document-mode specification and its runtime input and output responsibilities. Builder started while this archive remained in flight.\n"
        if node == "archivist_build":
            precheck = json.loads((run / ".factory/reviews/precheck.json").read_text())
            return "# Build archive\n\nActual final precheck outcome: " + precheck["outcome"] + "\n\nComplete diagnostic evidence:\n" + json.dumps(precheck) + "\n"
        pytest.fail("unexpected controlled external node " + node)

    def director_response(self, node, prompt, run):
        stage, _, task, _ = node.split(":")
        if task == "plan":
            marker = "Required coverage items (may add others, must not change or drop these):\n"
            required = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]
            intent = [{"id": "runtime-document", "text": BRIEF, "source_line": 1}]
            coverage = required + intent
            role, directory, count = {
                "research_director": ("researcher", ".factory/strategy/research", 3),
                "strategy_director": ("strategist", ".factory/strategy/perspectives", 2),
                "qa_director": ("adversarial_tester", ".factory/reviews/testers", 2),
            }[stage]
            tasks = [{"id": f"focus-{i}", "slug": f"focus-{i}", "role": role,
                      "focus": "Architecture and concrete evidence" if i == 1 else "Testing and intent coverage",
                      "prompt": f"Inspect distinct perspective {i} for all original reusable runtime document obligations; report real native file relationships and accepted evidence.",
                      "output": f"{directory}/focus-{i}.md", "coverage": [item["id"] for item in coverage]} for i in range(1, count + 1)]
            return json.dumps({"schema_version": 1, "stage_id": stage, "tasks": tasks, "intent_items": intent, "coverage_items": coverage})
        if task == "quality":
            reports = json.JSONDecoder().raw_decode(prompt.split("COMPLETE JOINED CHILD REPORTS AND RECEIPTS:\n", 1)[1])[0]
            assert all(report["status"] == "completed" for report in reports), reports
            assert all(report["body"] and report["raw_evidence"]["worker_receipts"] for report in reports)
            return json.dumps({"decisions": [{"id": report["task_id"], "verdict": "accept", "reason": "Full assigned actual reports and original coverage were consumed", "outcome": "passed"} for report in reports],
                               "summary": "Reviewed every full substantive report and preserved the original runtime-document acceptance asks; mandatory native evidence passed.", "coverage_complete": True})
        body = "# " + stage + " " + task + "\n\n"
        if stage == "qa_director":
            validation = ops.validate_native_package(run, "portable_probe", requires_input=("document_path",))
            assert validation["ok"], validation
            body += "Command: python -m refactory_dsh.create_fidelity_ops validate portable_probe --project RUN --requires-input document_path\nActual output:\n" + json.dumps(validation) + "\nNative graph has exactly read:op with zero edges and reusable document_path input.\n"
            valid = run / f"qa-{task}-input.json"
            invalid = run / f"qa-{task}-invalid.json"
            valid.write_text(json.dumps({"tester": task}))
            invalid.write_text("[]")
            success = ops.run_native_mode(run, "portable_probe/read", run / f"qa-{task}-valid", {"document_path": str(valid)})
            failure = ops.run_native_mode(run, "portable_probe/read", run / f"qa-{task}-invalid", {"document_path": str(invalid)})
            assert success["exit_ok"] and not failure["exit_ok"]
            assert json.loads((run / f"qa-{task}-valid/result.json").read_text()) == {"tester": task}
            assert not (run / f"qa-{task}-invalid/result.json").exists()
            body += "\nCommands: native run with a distinct object and an invalid array. Actual complete outputs:\n" + json.dumps({"valid": success, "invalid": failure}) + "\n"
        else:
            body += "Substantive independent perspective " + task + ": main.py imports json and defines main. Native nn.Mode declare/build contracts preserve runtime document_path. The one-node operation reads only the selected external input and writes result.json in ctx.workdir. Consider invalid JSON, arrays, distinct invocation outputs and actual graph inspection evidence.\n"
        if stage == "strategy_director":
            body += "\n### Acceptance Criteria\n- [ ] " + CRITERION + "\nOutcome: correct document; Scope: connected native invocation; Evidence: two runtime files and honest array failure.\n"
        return body


def begin_trial(tmp_path, monkeypatch, *, rework=False, fail_precheck=False, public_run=False):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("import json\ndef main(): return json.dumps({'source': True})\n")
    responses = ExternalDshIO(rework, fail_precheck)
    monkeypatch.setattr(DshModelClient, "_complete", lambda client, *args, **kwargs: responses.complete(client, *args, **kwargs))
    client = DshModelClient(provider="fixture", model="sol-6.1", effort="high", attempts=1)
    knobs = {"name": "portable_probe", "brief": BRIEF, "project_path": str(source), "requires_input": "document_path", "director_parallelism": 7, "child_attempts": 2}
    ctx = RunContext(tmp_path / "run", knobs=knobs, model=client)
    executor = Executor(CreateFidelityMode().graph(), ctx)
    if public_run:
        mode = CreateFidelityMode()
        state = mode.state_dict()
        for name, value in knobs.items():
            state["knobs"][name]["value"] = value
        mode.load_state_dict(state)
        result = mode.run(ctx.workdir, model=client, task=knobs)
        ctx.facts = result.facts
        ctx.knobs = mode.values()
    else:
        result = executor.run()
    assert not result.exit_ok
    assert "fresh human approval" in result.error
    assert not responses.builder_count
    return executor.graph, ctx, responses, executor


def test_actual_mode_resume_consumes_fresh_human_approval_and_restores_recorded_route(tmp_path, monkeypatch):
    graph, ctx, responses, _ = begin_trial(tmp_path, monkeypatch, public_run=True)
    gate = graph.nodes["gate_strategy"]
    ledger = Ledger(ctx.workdir)
    ledger.verdict(gate.id, "proceed", review=ledger.review(gate.id, gate.reads))
    before = set((ctx.workdir / ".refactory/workers").glob("*.json"))
    result = CreateFidelityMode.resume(ctx.workdir, model=DshModelClient(provider="wrong", model="wrong", effort="low"))
    assert result.exit_ok, result.error
    assert result.facts["create_fidelity_precheck_ok"] is True
    assert responses.builder_count == 1
    receipts = [json.loads(path.read_text()) for path in set((ctx.workdir / ".refactory/workers").glob("*.json")) - before]
    assert receipts and all(receipt["requested_route"]["provider"] == "fixture" for receipt in receipts)
    assert all(receipt["requested_route"]["effort"] == "high" for receipt in receipts)


def test_actual_later_mode_resume_reuses_fresh_completed_gate_approvals(tmp_path, monkeypatch):
    graph, ctx, responses, _ = begin_trial(tmp_path, monkeypatch, public_run=True)
    gate = graph.nodes["gate_strategy"]
    ledger = Ledger(ctx.workdir)
    ledger.verdict(gate.id, "proceed", review=ledger.review(gate.id, gate.reads))
    failed = False
    def external(client, *args, **kwargs):
        nonlocal failed
        response = responses.complete(client, *args, **kwargs)
        if client._node_id == "gate_overwatch" and not failed:
            failed = True
            return "invalid verdict JSON"
        return response
    monkeypatch.setattr(DshModelClient, "_complete", external)
    result = CreateFidelityMode.resume(ctx.workdir)
    assert not result.exit_ok and "JSON" in result.error
    progress = json.loads((ctx.workdir / ".refactory/progress.json").read_text())
    assert progress["node"] == "gate_overwatch"
    result = CreateFidelityMode.resume(ctx.workdir)
    assert result.exit_ok, result.error
    assert result.facts["create_fidelity_precheck_ok"] is True
    assert responses.builder_count == 1


def test_actual_mode_resume_refuses_stale_human_approval_until_exact_review_is_answered(tmp_path, monkeypatch):
    graph, ctx, responses, _ = begin_trial(tmp_path, monkeypatch, public_run=True)
    gate = graph.nodes["gate_strategy"]
    ledger = Ledger(ctx.workdir)
    ledger.verdict(gate.id, "proceed", review=ledger.review(gate.id, gate.reads))
    specification = ctx.workdir / ".factory/strategy/current.md"
    specification.write_text(specification.read_text() + "\nClarification: preserve distinct invocation outputs.\n")
    stale = CreateFidelityMode.resume(ctx.workdir)
    assert not stale.exit_ok and "fresh human approval" in stale.error
    assert not responses.builder_count
    progress = json.loads((ctx.workdir / ".refactory/progress.json").read_text())
    assert progress["node"] == gate.id and progress["outcome"] == "failed"
    ledger.verdict(gate.id, "proceed", review=ledger.review(gate.id, gate.reads))
    fresh = CreateFidelityMode.resume(ctx.workdir)
    assert fresh.exit_ok, fresh.error
    assert fresh.facts["create_fidelity_precheck_ok"] is True
    assert responses.builder_count == 1


def test_candidate_resume_still_refuses_changed_authored_instructions(tmp_path, monkeypatch):
    _, ctx, _, _ = begin_trial(tmp_path, monkeypatch, public_run=True)
    path = ctx.workdir / ".refactory/progress.json"
    progress = json.loads(path.read_text())
    progress["graph"]["nodes"]["builder"]["prompt_template"] += "\nDifferent authored instruction."
    path.write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="graph or instructions changed"):
        CreateFidelityMode.resume(ctx.workdir)


def approve_and_resume(graph, old_ctx):
    ledger = Ledger(old_ctx.workdir)
    gate = graph.nodes["gate_strategy"]
    review = ledger.review(gate.id, gate.reads)
    ledger.verdict(gate.id, "proceed", review=review)
    ctx = RunContext(old_ctx.workdir, knobs=old_ctx.knobs, facts=old_ctx.facts, model=old_ctx.model)
    executor = Executor(graph, ctx)
    return executor.run(start=gate.id), ctx, executor


@pytest.mark.parametrize("rework", [False, True])
def test_full_candidate_human_resume_real_directors_qa_join_gates_and_archives(tmp_path, monkeypatch, rework):
    graph, ctx, responses, initial = begin_trial(tmp_path, monkeypatch, rework=rework)
    approved_inputs = {name: (ctx.workdir / name).read_bytes() for name in graph.nodes["gate_strategy"].reads}
    approved_review = Ledger(ctx.workdir).review("gate_strategy", graph.nodes["gate_strategy"].reads)
    result, resumed, executor = approve_and_resume(graph, ctx)
    assert result.exit_ok, result.error
    assert result.facts["create_fidelity_precheck_ok"] is True
    assert responses.builder_count == (2 if rework else 1)
    assert responses.plan_archive_saw_builder
    assert "Actual final precheck outcome: proceed" in (ctx.workdir / ".factory/archive/create-build.md").read_text()
    for stage, count in (("research_director", 3), ("strategy_director", 2), ("qa_director", 4)):
        findings = json.loads((ctx.workdir / f".factory/{stage}-findings.json").read_text())
        assert findings["status"] == "completed"
        assert len(findings["reports"]) == count
        for report in findings["reports"]:
            assert report["body"] in (ctx.workdir / f".factory/{stage}-findings.md").read_text()
    steps = initial.steps + executor.steps
    assert {step.node for step in steps} == set(graph.nodes)
    assert {step.node for step in steps if step.kind == "gate" and step.verdict == "proceed"} >= {"gate_strategy", "gate_build", "gate_qa", "gate_overwatch", "gate_doc_freshness", "gate_precheck"}
    if rework:
        assert any(step.node == "gate_build" and step.verdict == "reloop" for step in steps)
        assert "README.md is missing" in responses.prompts["builder"][-1]
        assert {name: (ctx.workdir / name).read_bytes() for name in graph.nodes["gate_strategy"].reads} == approved_inputs
        assert Ledger(ctx.workdir).review("gate_strategy", graph.nodes["gate_strategy"].reads) == approved_review
        assert "README.md is missing" in (ctx.workdir / ".factory/reviews/repair-feedback.md").read_text()
        assert "README.md is missing" in responses.prompts["qa_director:v1:plan:a1"][0]
    for gate in ("gate_build", "gate_qa", "gate_overwatch", "gate_doc_freshness"):
        assert len(responses.prompts[gate]) >= 1
    full = (ctx.workdir / ".factory/reviews/qa-synthesized.md").read_text()
    qa = json.loads((ctx.workdir / ".factory/qa_director-findings.json").read_text())
    assert all(report["body"] in full for report in qa["reports"])


def test_stale_human_approval_refuses_builder_then_exact_review_resumes(tmp_path, monkeypatch):
    graph, ctx, responses, _ = begin_trial(tmp_path, monkeypatch)
    ledger = Ledger(ctx.workdir)
    gate = graph.nodes["gate_strategy"]
    prior = ledger.review(gate.id, gate.reads)
    ledger.verdict(gate.id, "proceed", review=prior)
    specification = ctx.workdir / ".factory/strategy/current.md"
    specification.write_text(specification.read_text() + "\nClarification: preserve distinct invocation outputs.\n")
    stale = Executor(graph, ctx).run(start="gate_strategy")
    assert not stale.exit_ok
    assert not responses.builder_count
    result, _, _ = approve_and_resume(graph, ctx)
    assert result.exit_ok, result.error
    assert responses.builder_count == 1


def test_final_precheck_failure_is_archived_and_stays_failed(tmp_path, monkeypatch):
    graph, ctx, responses, _ = begin_trial(tmp_path, monkeypatch, fail_precheck=True)
    result, _, executor = approve_and_resume(graph, ctx)
    assert not result.exit_ok
    assert result.facts["create_fidelity_precheck_ok"] is False
    assert "SKILL.md" in result.error
    assert any(step.node == "gate_precheck" and step.verdict == "halt" for step in executor.steps)
    archive = (ctx.workdir / ".factory/archive/create-build.md").read_text()
    assert "Actual final precheck outcome: halt" in archive
    assert "SKILL.md" in archive


@pytest.mark.parametrize("relative", [".factory/strategy/current.md", ".factory/strategy/user-intent.md"])
def test_completed_approval_cannot_certify_materially_changed_strategy_or_human_intent(tmp_path, monkeypatch, relative):
    graph, ctx, _, _ = begin_trial(tmp_path, monkeypatch)
    result, completed, _ = approve_and_resume(graph, ctx)
    assert result.exit_ok, result.error
    file = ctx.workdir / relative
    file.write_text(file.read_text() + "\nNew material requirement: preserve a second output format.\n")
    recheck = Executor(graph, completed).run(start="gate_precheck")
    assert not recheck.exit_ok
    assert recheck.facts["gate_precheck_verdict"] == "halt"
    assert "human strategy approval is absent or stale" in completed.facts["create_fidelity_precheck_error"]
