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
    """An unverified requirement in intent-audit.json → the gate lists it as advisory.
    The QA director re-run is signalled by setting a fact."""
    monkeypatch.setattr(evidence, "current_problems", lambda ctx, workers: [])

    ctx = _gate_ctx(tmp_path, _qa_judgment("proceed"))
    ctx.write_json("intent-audit.json", {"requirements": [
        {"id": "ask-1", "requirement": "the tool prints built",
         "status": "unverified"},
        {"id": "ask-2", "requirement": "the tool exits 0",
         "status": "pass"},
    ]})

    verdict = design_ops.gate_qa(ctx)
    assert verdict == "proceed"

    # The unverified requirement should be listed as advisory
    follow_ups = (tmp_path / "reviews/follow-ups.md").read_text()
    assert "not verified" in follow_ups
    assert "ask-1" in follow_ups
    # The verified requirement should not be listed
    assert "ask-2" not in follow_ups or "not verified" not in follow_ups.split("ask-2")[0]

    # The re-run fact should be set
    assert ctx.facts.get("qa_coverage_rerun_done") is True


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
