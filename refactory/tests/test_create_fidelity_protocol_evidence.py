"""Independent protocol scoring over real controlled orchestration artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.create_fidelity.evidence import authoring_score
from test_create_fidelity_orchestration import begin_trial, approve_and_resume


@pytest.fixture(scope="module")
def trial(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("protocol-evidence")
    graph, ctx, responses, _ = begin_trial(root, patch, rework=True)
    outcome, _, _ = approve_and_resume(graph, ctx)
    assert outcome.exit_ok, outcome.error
    manifests = {stage: json.loads((ctx.workdir / f".factory/{stage}-findings.json").read_text())
                 for stage in ("research_director", "strategy_director", "qa_director")}
    yield ctx.workdir, manifests, {"exit_ok": outcome.exit_ok}
    patch.undo()


def scores(trial):
    root, manifests, outcome = trial
    return {row["requirement_id"]: row for row in authoring_score([], outcome, director_manifests=manifests, evidence_root=root)}


def test_real_raw_protocol_evidence_passes_without_semantic_self_certification(trial):
    rows = scores(trial)
    for requirement in ("F.obligations", "F.consumption", "F.approval", "F.repair", "F.deadlines"):
        assert rows[requirement]["status"] == "passed", rows[requirement]
        assert rows[requirement]["evidence"]
    assert rows["F.obligations.semantic"]["status"] == "unavailable"
    assert rows["F.approval.identity"]["status"] == "unavailable"
    assert rows["F.deadlines.remaining"]["status"] == "unavailable"
    assert rows["F.deadlines.expiry"]["status"] == "not_exercised"


def test_success_and_model_pass_claims_without_raw_files_cannot_certify_protocol(tmp_path):
    rows = {row["requirement_id"]: row for row in authoring_score([], {"exit_ok": True}, evidence_root=tmp_path)}
    for requirement in ("F.obligations", "F.consumption", "F.approval", "F.deadlines"):
        assert rows[requirement]["status"] == "unavailable", rows[requirement]
    assert rows["F.repair"]["status"] == "not_exercised"


def test_missing_inherited_prompt_and_upstream_content_are_independent_failures(trial):
    root, manifests, _ = trial
    paths = list((root / ".refactory/agent-delivery").glob("*/receipt.json"))
    receipt = next(json.loads(path.read_text()) for path in paths if json.loads(path.read_text()).get("node") == "synthesize_strategy")
    prompt = root / receipt["contexts"][-1]["prompt"]
    original = prompt.read_text()
    try:
        prompt.write_text("The model claims PASS without original duties or child evidence.")
        rows = scores(trial)
        assert rows["F.obligations"]["status"] == "failed"
        assert rows["F.consumption"]["status"] == "failed"
    finally:
        prompt.write_text(original)
    # Keeping full inherited instructions still cannot replace child report data.
    report = manifests["strategy_director"]["reports"][0]["body"]
    try:
        omitted = original.replace(report, "Substantive report omitted by an adapter.")
        for ascii_only in (True, False):
            omitted = omitted.replace(json.dumps(report, ensure_ascii=ascii_only), json.dumps("Report content omitted."))
        prompt.write_text(omitted)
        assert scores(trial)["F.consumption"]["status"] == "failed"
    finally:
        prompt.write_text(original)


@pytest.mark.parametrize("problem", ["standing", "reused", "wrong-digest", "changed-strategy"])
def test_human_approval_cannot_be_unconditional_reused_or_stale(trial, problem):
    root, _, _ = trial
    ledger = root / "intent-ledger.md"
    strategy = root / ".factory/strategy/current.md"
    original, original_strategy = ledger.read_text(), strategy.read_text()
    try:
        if problem == "standing":
            ledger.write_text(original.replace("source: steer", "source: upfront").replace("scope: ", "scope: run # "))
        elif problem == "reused":
            section = next(part for part in original.split("\n## ") if "] verdict-use\n" in part and "result: used" in part)
            ledger.write_text(original + "\n## " + section)
        elif problem == "wrong-digest":
            ledger.write_text(original.replace("scope: ", "scope: invalid-"))
        else:
            strategy.write_text(original_strategy + "\nNew material requirement\n")
        assert scores(trial)["F.approval"]["status"] == "failed"
    finally:
        ledger.write_text(original)
        strategy.write_text(original_strategy)


@pytest.mark.parametrize("problem", ["missing-branch", "missing-feedback", "self-reported-verdict"])
def test_repair_requires_raw_branch_real_feedback_and_actual_model_decision(trial, problem):
    root, _, _ = trial
    gate = next(path for path in (root / ".refactory/create-fidelity-gates").glob("*/verdict.json")
                if json.loads(path.read_text()).get("verdict") == "reloop")
    original = gate.read_text()
    gate_stat = gate.stat()
    prompt_path, prompt_original = None, None
    try:
        if problem == "missing-branch":
            gate.unlink()
            assert scores(trial)["F.repair"]["status"] == "unavailable"
        elif problem == "self-reported-verdict":
            record = json.loads(original)
            record["rationale"] = "The harness invented this critique without model output."
            gate.write_text(json.dumps(record))
            assert scores(trial)["F.repair"]["status"] == "failed"
        else:
            builders = [json.loads(path.read_text()) for path in (root / ".refactory/agent-delivery").glob("*/receipt.json") if json.loads(path.read_text()).get("node") == "builder"]
            receipt = max(builders, key=lambda record: record["started"])
            prompt_path = root / receipt["contexts"][-1]["prompt"]
            prompt_original = prompt_path.read_text()
            rationale = json.loads(original)["rationale"]
            prompt_path.write_text(prompt_original.replace(rationale, "Feedback omitted by adapter."))
            assert scores(trial)["F.repair"]["status"] == "failed"
    finally:
        gate.write_text(original)
        os.utime(gate, ns=(gate_stat.st_atime_ns, gate_stat.st_mtime_ns))
        if prompt_path is not None:
            prompt_path.write_text(prompt_original)


@pytest.mark.parametrize("problem", ["missing-layers", "short-builder", "inflight", "active-no-finished", "unconfirmed-cleanup", "overlong"])
def test_deadline_configuration_and_terminal_settlement_require_raw_records(trial, problem):
    root, _, _ = trial
    worker = next(path for path in (root / ".refactory/workers").glob("*.json") if json.loads(path.read_text()).get("node_id") == "builder")
    original = worker.read_text()
    try:
        record = json.loads(original)
        if problem == "missing-layers":
            record.pop("effective_deadlines")
        elif problem == "short-builder":
            record["effective_deadlines"]["declared_stage_timeout_s"] = 900
            record["effective_deadlines"]["stage_timeout_s"] = 900
        elif problem == "overlong":
            record["finished"] = "2099-01-01T00:00:00+00:00"
        elif problem == "active-no-finished":
            record["state"] = "starting"
            record.pop("finished")
        else:
            record["state"] = "running" if problem == "inflight" else "unknown"
        worker.write_text(json.dumps(record))
        status = scores(trial)["F.deadlines"]["status"]
        assert status == ("unavailable" if problem in ("missing-layers", "active-no-finished") else "failed")
    finally:
        worker.write_text(original)
