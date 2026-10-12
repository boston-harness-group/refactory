"""Independent scorer inspections of actual controlled director receipts."""
from __future__ import annotations

import copy
import json

import pytest

from benchmarks.create_fidelity.evidence import authoring_score
from refactory_dsh import create_fidelity_directors as directors
from test_create_fidelity_directors import run_stage


def score(manifest, root=None):
    rows = authoring_score([], {}, director_manifests={manifest["stage_id"]: manifest}, evidence_root=root)
    return next(row for row in rows if row["requirement_id"] == f"F.children.{manifest['stage_id']}")


@pytest.mark.parametrize("stage,count", [(directors.RESEARCH, 3), (directors.STRATEGY, 2), (directors.QA, 5)])
def test_scorer_dereferences_actual_dispatch_delivery_and_quality_receipts(tmp_path, monkeypatch, stage, count):
    result, manifest, _ = run_stage(tmp_path, monkeypatch, stage=stage, count=count, quality_retry=True)
    assert result.exit_ok, result.error
    scored = score(manifest, tmp_path)
    assert scored["status"] == "passed", scored
    assert scored["evidence"]
    assert any("worker-" in reference for reference in scored["evidence"])
    assert any("context-" in reference for reference in scored["evidence"])


def test_manifest_alone_and_missing_raw_receipts_are_unavailable(tmp_path, monkeypatch):
    _, manifest, _ = run_stage(tmp_path, monkeypatch)
    assert score(manifest)["status"] == "unavailable"
    (tmp_path / manifest["reports"][0]["raw_evidence"]["worker_receipts"][0]).unlink()
    assert score(manifest, tmp_path)["status"] == "unavailable"


@pytest.mark.parametrize("mutation", ["failed-manifest", "incomplete-quality", "thin-body", "wrong-body-hash",
                                       "wrong-input-hash", "wrong-plan-hash", "wrong-task-hash", "empty-refs",
                                       "stale-visit", "stale-run", "wrong-role", "wrong-route", "changed-file",
                                       "stale-worker", "changed-quality", "raw-route-mismatch", "stale-delivery"])
def test_invalid_or_stale_director_evidence_never_passes(tmp_path, monkeypatch, mutation):
    _, source, _ = run_stage(tmp_path, monkeypatch)
    manifest = copy.deepcopy(source)
    report = manifest["reports"][0]
    if mutation == "failed-manifest":
        manifest["status"] = "failed"
    elif mutation == "incomplete-quality":
        manifest["quality_reviews"][-1]["coverage_complete"] = False
    elif mutation == "thin-body":
        report["body"] = "x"
    elif mutation == "wrong-body-hash":
        report["output_digest"] = "incorrect"
    elif mutation == "wrong-input-hash":
        report["input_digest"] = "incorrect"
    elif mutation == "wrong-plan-hash":
        manifest["plan_digest"] = "incorrect"
    elif mutation == "wrong-task-hash":
        report["task_digest"] = "incorrect"
    elif mutation == "empty-refs":
        report["raw_evidence"] = {"worker_receipts": [], "delivery_receipts": []}
    elif mutation == "stale-visit":
        report["visit"] += 1
    elif mutation == "stale-run":
        report["run_id"] = "another-run"
    elif mutation == "wrong-role":
        report["role"] = "builder"
    elif mutation == "wrong-route":
        report["actual_route"]["worker_backend"] = "codex"
    elif mutation == "changed-file":
        (tmp_path / report["output"]).write_text(report["body"] + "\nchanged")
    elif mutation == "changed-quality":
        manifest["quality_reviews"][-1]["summary"] = "Parent-invented quality assessment"
    elif mutation in ("stale-worker", "raw-route-mismatch"):
        path = tmp_path / report["raw_evidence"]["worker_receipts"][0]
        receipt = json.loads(path.read_text())
        if mutation == "stale-worker":
            receipt["node_id"] = receipt["node_id"].replace(":v1:", ":v0:")
        else:
            receipt["actual_route"]["model"] = "unrequested-model"
        path.write_text(json.dumps(receipt))
    elif mutation == "stale-delivery":
        path = tmp_path / report["raw_evidence"]["delivery_receipts"][0]
        receipt = json.loads(path.read_text())
        receipt["status"] = "failed"
        path.write_text(json.dumps(receipt))
    scored = score(manifest, tmp_path)
    assert scored["status"] == "failed", scored


def test_failed_stage_thin_arbitrary_hashes_and_empty_refs_cannot_claim_child_success(tmp_path):
    manifest = {"stage_id": "research_director", "status": "failed", "quality_reviews": [{"coverage_complete": False}],
        "reports": [{"task_id": f"child-{index}", "status": "completed", "body": "x", "input_digest": "unverified",
                     "output_digest": "wrong", "raw_evidence": {"worker_receipts": [], "delivery_receipts": []}}
                    for index in range(3)]}
    assert score(manifest, tmp_path)["status"] == "failed"
