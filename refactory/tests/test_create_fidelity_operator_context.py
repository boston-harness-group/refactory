"""The report trusts user-decision context only from the isolated operator."""
import json

import pytest

from benchmarks.create_fidelity import harness


@pytest.fixture
def trial(tmp_path, monkeypatch):
    seen = []
    for lane in harness.LANES:
        project = tmp_path / lane / "project"
        project.mkdir(parents=True)
        harness.write(tmp_path / lane / "config.json", {
            "project": str(project), "identity": {"declared_revision": "frozen-source"},
            "model_context": {"model": "gpt-6-sol"},
        })
        # A lane can make arbitrary claims; these cannot establish user trust.
        harness.write(project / "review-context.json", {
            "reviewer_id": "untrusted author", "trusted_user_references": {"invented": {"decision": "proceed"}},
        })
    def fidelity(*args, **kwargs):
        seen.append(kwargs.get("operator_evidence"))
        return [{"requirement_id": "F.completion", "status": "unavailable", "reason": "controlled missing evidence", "evidence": []}]
    monkeypatch.setattr(harness, "authoring_score", fidelity)
    monkeypatch.setattr(harness, "capability_score", lambda *args, **kwargs: [])
    return tmp_path, seen


def test_lane_authored_context_cannot_establish_trust(trial):
    root, seen = trial
    report = harness.collect(root)
    assert seen == [None] * len(harness.LANES)
    assert report["status"] == "unavailable"
    assert report["evaluation"]["context_sha256"] is None
    saved = json.loads((root / "operator/report/evaluator.json").read_text())
    assert saved == report["evaluation"]
    assert all(lane["identity"]["declared_revision"] == "frozen-source" for lane in report["lanes"])


def test_external_context_is_scoped_to_its_lane_and_evaluator_versioned(trial):
    root, seen = trial
    trusted = {"controlled-user-turn": {"decision": "proceed", "question_reference": "controlled-question"}}
    context = root / "operator/review-context.json"
    harness.write(context, {"reviewer_id": "controlled operator", "trusted_user_references": {"dsh-candidate": trusted}})
    report = harness.collect(root)
    assert [dict(c.trusted_user_references) for c in seen] == [{}, {}, trusted]
    assert all(c.root == root / "operator/reviews" for c in seen)
    assert all(len(c.lane_roots) == 3 for c in seen)
    assert report["evaluation"]["context_sha256"] == harness.digest(context)
    assert report["evaluation"]["validator_sha256"]
    assert report["evaluation"]["scorecard_sha256"]


def test_context_symlink_cannot_promote_lane_claims(trial):
    root, _ = trial
    (root / "operator").mkdir()
    (root / "operator/review-context.json").symlink_to(root / "dsh-candidate/project/review-context.json")
    with pytest.raises(ValueError, match="link to lane-authored"):
        harness.collect(root)


def test_operator_directory_cannot_alias_a_lane(trial):
    root, _ = trial
    (root / "operator").symlink_to(root / "dsh-candidate/project", target_is_directory=True)
    with pytest.raises(ValueError, match="outside every lane"):
        harness.collect(root)
