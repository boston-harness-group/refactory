"""Negative controls for the independent harness, never model-backed acceptance."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

from benchmarks.create_fidelity.contract import FIXTURES
from benchmarks.create_fidelity.evidence import (
    aggregate, authoring_score, capability_score, normalize, route_score, write_report,
)
from benchmarks.create_fidelity.harness import ADAPTER_INSTRUCTIONS, authoring_brief, run_pair
from benchmarks.create_fidelity.isolation import profile
from benchmarks.create_fidelity import approval


def test_both_lanes_receive_identical_public_contract_and_external_check_ownership():
    briefs = {flavor: authoring_brief(flavor) for flavor in ("dsh", "remote")}
    public = (FIXTURES / "brief.md").read_text()
    boundary = (FIXTURES / "authoring-boundary.md").read_text()
    for flavor, brief in briefs.items():
        assert brief.startswith(public + "\n" + boundary)
        assert brief.count(boundary) == 1
        assert ADAPTER_INSTRUCTIONS[flavor] in brief
    assert briefs["dsh"].replace(ADAPTER_INSTRUCTIONS["dsh"], "<native>") == briefs["remote"].replace(ADAPTER_INSTRUCTIONS["remote"], "<native>")
    # External ownership cannot turn completed authoring into checker evidence.
    assert aggregate(capability_score(None, None)) != "passed"


def test_checker_runs_all_56_cases_and_rejects_bad_program(tmp_path):
    # Deliberately incorrect fixture, not a repaired benchmark deliverable.
    script = tmp_path / "wrong.py"
    script.write_text("import sys,json\nfrom pathlib import Path\nPath(sys.argv[4]).write_text(json.dumps({'conflicts':[]}))\n")
    cp = subprocess.run([sys.executable, str(FIXTURES / "acceptance.py"), str(script)], capture_output=True, text=True)
    result = json.loads(cp.stdout)
    assert cp.returncode != 0
    assert result["total"] == 56
    assert result["status"] == "failed"
    assert not next(c for c in result["cases"] if c["name"] == "simple_overlap")["passed"]
    assert not next(c for c in result["cases"] if c["name"] == "duplicate_across_rooms_preserves_output")["passed"]


def test_zero_checker_cases_cannot_pass():
    scores = capability_score({"status": "passed", "total": 0, "passed": 0, "cases": []}, None)
    assert aggregate(scores) == "failed"


def test_route_request_is_not_served_route_evidence():
    requested = {"worker_backend": "dsh", "model": "gpt-6.1-sol", "effort": "high"}
    assert route_score([{"requested_route": requested}])["status"] == "unavailable"
    assert route_score([{"actual_route": requested}])["status"] == "failed"
    assert route_score([{"actual_route": {**requested, "source": "runtime-request"}}])["status"] == "passed"
    fallback = {"actual_route": {**requested, "model": "gpt-6-sol", "source": "runtime-request"}}
    assert route_score([fallback])["status"] == "failed"
    assert route_score([fallback], model="gpt-6-sol")["status"] == "passed"


def test_missing_child_evidence_does_not_pass_completed_authoring():
    events = normalize([{"kind": "node_completed", "node_id": "research_director"}], trial_id="controlled", source="events.json")
    rows = authoring_score(events, {"exit_ok": True})
    assert next(r for r in rows if r["requirement_id"] == "F.children.research_director")["status"] == "unavailable"
    assert aggregate(rows) != "passed"


def test_qa_partial_order_allows_peer_completion_order_but_requires_full_barrier():
    raw = [{"kind": "node_started", "node_id": "fork_qa"}]
    raw += [{"kind": "node_completed", "node_id": n} for n in ["qa_director", "health_checker", "code_reviewer", "join_qa"]]
    raw += [{"kind": "node_started", "node_id": "synthesize_qa"}]
    def status(records):
        events = normalize(records, trial_id="controlled", source="raw.json")
        return next(r for r in authoring_score(events, {}) if r["requirement_id"] == "F.qa_barrier")["status"]
    assert status(raw) == "passed"
    assert status([r for r in raw if r["node_id"] != "health_checker"]) == "failed"


def test_interrupted_run_remains_failed_and_html_escapes_evidence(tmp_path):
    lane = {"lane": "<script>bad</script>", "capability": capability_score(None, None),
            "fidelity": authoring_score([], {"exit_ok": False, "error": "interrupted"})}
    result = write_report(tmp_path, [lane])
    assert result["status"] == "failed"
    assert "<script>bad</script>" not in (tmp_path / "comparison.html").read_text()
    assert "&lt;script&gt;bad&lt;/script&gt;" in (tmp_path / "comparison.html").read_text()


def test_live_command_without_preflight_does_not_launch(tmp_path):
    result = run_pair(tmp_path, ["dsh-baseline", "remote-reference"])
    assert all(r["status"] == "unavailable" for r in result)
    assert (tmp_path / "operator/launch-blocked.json").is_file()


def test_isolation_denies_peers_operator_checker_and_source_writes(tmp_path):
    lane, peer, source, hidden = (tmp_path / n for n in ["lane", "peer", "source", "operator"])
    text = profile(lane, [peer], [source], [hidden])
    assert f'(deny file-read* (subpath "{peer}"))' in text
    assert f'(deny file-read* (subpath "{hidden}"))' in text
    assert f'(deny file-write* (subpath "{source}"))' in text


def test_approval_requires_current_strategy_and_one_use_answer(tmp_path):
    strategy = tmp_path / ".factory/strategy"
    strategy.mkdir(parents=True)
    (strategy / "current.md").write_text("actual strategy")
    (strategy / "user-intent.md").write_text("actual intent")
    pending = approval.request(tmp_path, "Review it")
    approval.deliver(tmp_path, digest=pending["strategy_digest"], verdict="proceed", approver="controlled human fixture", feedback="")
    answer = approval.take(tmp_path, pending)
    assert answer["source"] == "explicit-human"
    assert approval.take(tmp_path, pending) is None
    (strategy / "current.md").write_text("changed strategy")
    with pytest.raises(ValueError, match="changed"):
        approval.deliver(tmp_path, digest=pending["strategy_digest"], verdict="proceed", approver="controlled human fixture", feedback="")
