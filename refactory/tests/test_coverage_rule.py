"""Coverage rule tests: every requirement must have probe coverage.

Tests for Ask 3: probe-coverage enforcement in the QA director and overwatch
prompts, and the _check_unverified_coverage helper in gate_qa.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh import authoring_evidence as evidence
from refactory_dsh import design_ops, judges
from refactory_dsh.nn import RunContext


# ── helpers ──────────────────────────────────────────────────────────────

def _gate_ctx(tmp_path: Path, judgment_text: str, **facts) -> RunContext:
    """A run directory with reviews and a QA judge verdict."""
    ctx = RunContext(tmp_path, facts=facts)
    for name, _ in design_ops.DESIGN_QA_REVIEWS:
        ctx.write_text(name, "ran it; fine\n")
    ctx.write_text(judges.QA_JUDGE, judgment_text)
    return ctx


def _qa_judgment(verdict: str, must_fix=None, follow_ups=None) -> str:
    return json.dumps({
        "verdict": verdict,
        "must_fix": must_fix or [],
        "follow_ups": follow_ups or [],
    })


# ── tests ────────────────────────────────────────────────────────────────

def test_qa_director_prompt_has_coverage_rule():
    """The QA director prompt contains the coverage-rule text."""
    from refactory_dsh import design
    graph = design.DesignMode().build()
    qa_node = graph.nodes.get("qa_director")
    assert qa_node is not None
    prompt = qa_node.prompt_template
    assert "every requirement" in prompt.lower()
    assert "exact command" in prompt


def test_overwatch_prompt_has_critical_probe_instruction():
    """The overwatch prompt contains text about running critical requirements."""
    from refactory_dsh import design
    assert "most critical requirements" in design.OVERWATCH_PROMPT


def test_unverified_requirement_triggers_qa_rerun(tmp_path, monkeypatch):
    """An unverified requirement in intent-audit.json → the gate returns a
    coverage_rerun verdict so the QA director re-runs via qa_director_rerun.
    The re-run fact and the qa_coverage_rerun_ids fact are both set."""
    monkeypatch.setattr(evidence, "current_problems", lambda ctx, workers: [])

    ctx = _gate_ctx(tmp_path, _qa_judgment("proceed"))
    ctx.write_json("intent-audit.json", {"requirements": [
        {"id": "ask-1", "requirement": "the tool prints built",
         "status": "unverified"},
        {"id": "ask-2", "requirement": "the tool exits 0",
         "status": "pass"},
    ]})

    verdict = design_ops.gate_qa(ctx)
    # First encounter: should trigger a coverage_rerun verdict (not proceed)
    assert verdict == "coverage_rerun"
    assert ctx.facts.get("qa_coverage_rerun_done") is True
    # The requirement ids reach the qa_director_rerun prompt via this fact.
    ids_fact = ctx.facts.get("qa_coverage_rerun_ids")
    assert ids_fact is not None
    assert "ask-1" in ids_fact
    assert "the tool prints built" in ids_fact


def test_still_unverified_after_rerun_listed_as_advisory(tmp_path, monkeypatch):
    """After the QA director re-run, requirement still unverified → listed as advisory
    in PR, no reloop occurs, and no kind='check' finding is injected."""
    monkeypatch.setattr(evidence, "current_problems", lambda ctx, workers: [])

    ctx = _gate_ctx(tmp_path, _qa_judgment("proceed"))
    ctx.set_fact("qa_coverage_rerun_done", True)  # Already re-run once
    ctx.write_json("intent-audit.json", {"requirements": [
        {"id": "ask-1", "requirement": "the tool prints built",
         "status": "unverified"},
    ]})

    verdict = design_ops.gate_qa(ctx)
    assert verdict == "proceed"  # No reloop

    follow_ups = (tmp_path / "reviews/follow-ups.md").read_text()
    assert "not verified" in follow_ups
    assert "ask-1" in follow_ups
    assert "still unverified" in follow_ups


def test_all_verified_proceeds(tmp_path, monkeypatch):
    """All requirements verified → no re-run triggered → proceed."""
    monkeypatch.setattr(evidence, "current_problems", lambda ctx, workers: [])

    ctx = _gate_ctx(tmp_path, _qa_judgment("proceed"))
    ctx.write_json("intent-audit.json", {"requirements": [
        {"id": "ask-1", "requirement": "the tool prints built",
         "status": "pass"},
        {"id": "ask-2", "requirement": "the tool exits 0",
         "status": "pass"},
    ]})

    verdict = design_ops.gate_qa(ctx)
    assert verdict == "proceed"

    # No unverified advisory
    follow_ups = (tmp_path / "reviews/follow-ups.md").read_text()
    assert "not verified" not in follow_ups
    # No re-run fact
    assert not ctx.facts.get("qa_coverage_rerun_done")


def test_audit_instructions_mention_unverified():
    """AUDIT_INSTRUCTIONS mentions that a requirement with no run probe gets unverified."""
    assert "unverified" in evidence.AUDIT_INSTRUCTIONS
    assert "no run probe evidence" in evidence.AUDIT_INSTRUCTIONS


# ── qa_director_rerun node tests ─────────────────────────────────────────

def test_qa_director_rerun_node_exists_with_coverage_prompt():
    """The graph has a qa_director_rerun agent node whose prompt mentions
    'coverage re-run' and writes reviews/qa-rerun.md.  Fails when the node
    does not exist (reverted)."""
    from refactory_dsh import design
    graph = design.DesignMode().build()
    node = graph.nodes.get("qa_director_rerun")
    assert node is not None, "qa_director_rerun node missing"
    assert node.kind == "agent"
    assert "reviews/qa-rerun.md" in node.writes
    assert "reviews/qa.md" in node.reads
    assert node.resource_role == "final_validation"
    assert node.outputs_only is True
    assert "coverage re-run" in node.prompt_template
    assert "qa_coverage_rerun_ids" in node.prompt_template


def test_qa_director_rerun_feedback_contains_requirement_ids(tmp_path, monkeypatch):
    """_check_unverified_coverage sets qa_coverage_rerun_ids with the
    requirement ids and texts, and includes them in the verdict reason."""
    monkeypatch.setattr(evidence, "current_problems", lambda ctx, workers: [])

    ctx = _gate_ctx(tmp_path, _qa_judgment("proceed"))
    ctx.write_json("intent-audit.json", {"requirements": [
        {"id": "ask-1", "requirement": "the tool prints built",
         "status": "unverified"},
        {"id": "ask-3", "requirement": "exits with code 0",
         "status": "unverified"},
    ]})

    verdict = design_ops.gate_qa(ctx)
    assert verdict == "coverage_rerun"
    ids = ctx.facts["qa_coverage_rerun_ids"]
    assert "ask-1" in ids
    assert "ask-3" in ids
    assert "the tool prints built" in ids
    assert "exits with code 0" in ids
    # The verdict reason also names the ids.
    gate_text = (tmp_path / "reviews/qa-gate.txt").read_text()
    assert "ask-1" in gate_text
    assert "ask-3" in gate_text


def test_synthesize_qa_includes_rerun_review(tmp_path):
    """synthesize_qa reads reviews/qa-rerun.md when it exists and includes
    it in the QA synthesis.  Fails when reverted because the file is ignored."""
    ctx = RunContext(tmp_path, facts={"check": "true", "validation": "ok",
                                      "qa_coverage_rerun_done": True})
    for name, _ in design_ops.DESIGN_QA_REVIEWS:
        ctx.write_text(name, "review: all good\n")
    ctx.write_text("check-record.json", '{"status": "pass"}')
    # Simulate a coverage re-run review
    ctx.write_text("reviews/qa-rerun.md",
                   "## ask-1 coverage probe\n\nran: python tool.py\nresult: built\n")
    design_ops.synthesize_qa(ctx)
    synthesis = ctx.read_text(design_ops.QA_SYNTHESIS)
    assert "QA director coverage re-run" in synthesis
    assert "ask-1 coverage probe" in synthesis
    assert "python tool.py" in synthesis


def test_coverage_rerun_graph_edges():
    """The coverage_rerun edge routes gate_qa → qa_director_rerun (not
    synthesize_qa), qa_director_rerun → synthesize_qa exists, and loop_edges
    classifies it as a retry.  Fails when reverted because the edge targets
    synthesize_qa directly."""
    from refactory_dsh import design
    from refactory_dsh.nn.loops import loop_edges
    graph = design.DesignMode().build()

    edges = {(e.target, e.condition) for e in graph.out_edges("gate_qa")}
    assert ("qa_director_rerun", "coverage_rerun") in edges
    assert ("synthesize_qa", "coverage_rerun") not in edges

    rerun_out = [(e.target, e.condition) for e in graph.out_edges("qa_director_rerun")]
    assert ("synthesize_qa", None) in rerun_out

    # The cycle gate_qa → qa_director_rerun → synthesize_qa → qa_judge → gate_qa
    # is a retry detected by loop_edges.
    loops = loop_edges(graph)
    # loop_edges returns (source, target, condition) tuples
    coverage_loops = [(s, t, c) for s, t, c in loops if c == "coverage_rerun"]
    assert len(coverage_loops) == 1
    assert coverage_loops[0] == ("gate_qa", "qa_director_rerun", "coverage_rerun")


def test_rerun_bounded_once_per_qa_round(tmp_path, monkeypatch):
    """When qa_coverage_rerun_done is set for the current round, a second
    unverified result does not trigger another re-run; gate_qa proceeds and
    lists the requirement as not verified (advisory).  The bound resets when
    the builder produces a new commit (new QA round).
    Fails when reverted because the bound check is absent."""
    monkeypatch.setattr(evidence, "current_problems", lambda ctx, workers: [])

    # Round 1: first unverified → triggers re-run
    ctx = _gate_ctx(tmp_path, _qa_judgment("proceed"))
    ctx.write_json("intent-audit.json", {"requirements": [
        {"id": "ask-1", "requirement": "the tool prints built",
         "status": "unverified"},
    ]})
    verdict1 = design_ops.gate_qa(ctx)
    assert verdict1 == "coverage_rerun"
    assert ctx.facts["qa_coverage_rerun_done"] is True

    # Round 1 continued: same round, still unverified → no second re-run
    ctx2 = _gate_ctx(tmp_path / "round1b", _qa_judgment("proceed"),
                     qa_coverage_rerun_done=True)
    ctx2.write_json("intent-audit.json", {"requirements": [
        {"id": "ask-1", "requirement": "the tool prints built",
         "status": "unverified"},
    ]})
    verdict2 = design_ops.gate_qa(ctx2)
    assert verdict2 == "proceed"  # Not another coverage_rerun
    follow_ups = (tmp_path / "round1b" / "reviews/follow-ups.md").read_text()
    assert "not verified" in follow_ups

    # Round 2: new QA round (builder commits, gate_check resets the bound).
    # Verify that gate_check produces qa_coverage_rerun_done (graph declaration)
    # — this fails when reverted because the old graph does not have it.
    from refactory_dsh import design
    graph = design.DesignMode().build()
    assert "qa_coverage_rerun_done" in graph.nodes["gate_check"].produces, (
        "gate_check must produce qa_coverage_rerun_done to reset the bound on new rounds"
    )
    # After gate_check resets it, unverified can trigger a re-run again.
    ctx3 = _gate_ctx(tmp_path / "round2", _qa_judgment("proceed"))
    ctx3.write_json("intent-audit.json", {"requirements": [
        {"id": "ask-1", "requirement": "the tool prints built",
         "status": "unverified"},
    ]})
    verdict3 = design_ops.gate_qa(ctx3)
    assert verdict3 == "coverage_rerun"


def test_new_qa_round_archives_qa_rerun_md(tmp_path):
    """After a coverage re-run writes reviews/qa-rerun.md, starting a new
    QA round (a rebuild after a coverage re-run) archives or removes it, so
    the next synthesize_qa does not carry the old qa-rerun.md into the
    synthesis.  Fails when reverted because the file persists."""
    ctx = RunContext(tmp_path, facts={"check": "true", "validation": "ok"})
    for name, _ in design_ops.DESIGN_QA_REVIEWS:
        ctx.write_text(name, "review: all good\n")
    ctx.write_text("check-record.json", '{"status": "pass"}')
    # Simulate a previous round's qa-rerun.md
    ctx.write_text("reviews/qa-rerun.md", "old probes from last round\n")

    # synthesize_qa starts the new round: it should archive the old qa-rerun.md
    design_ops.synthesize_qa(ctx)

    # The file should no longer be in the run directory
    assert not (tmp_path / "reviews/qa-rerun.md").exists()
    # The synthesis should NOT contain the old content
    synthesis = ctx.read_text(design_ops.QA_SYNTHESIS)
    assert "old probes from last round" not in synthesis
