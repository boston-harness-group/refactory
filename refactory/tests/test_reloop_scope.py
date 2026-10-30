"""Tests for the scope_reloop evaluator and gate_design_review fact clearing.

Each test fails when the scope_reloop function or the clearing code is reverted,
because the function will not exist, and the clearing lines will be absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh.design_ops import scope_reloop, _gate_reloop_feedback, gate_design_review
from refactory_dsh.nn import Verdict
from refactory_dsh.steering import LEDGER


# ── helpers ───────────────────────────────────────────────────────────────────

class FakeCtx:
    """Minimal context for evaluator unit tests."""

    def __init__(self, files=None, facts=None, workdir=None):
        self._files = dict(files or {})
        self.facts = dict(facts or {})
        self.workdir = workdir or Path(".")
        self._knobs = {"design_review_attempts": 1}
        self.knobs = self._knobs

    def exists(self, name):
        return name in self._files

    def read_text(self, name):
        if name not in self._files:
            raise FileNotFoundError(name)
        return self._files[name]

    def read_json(self, name, default=None):
        text = self._files.get(name)
        if text is None:
            return default
        return json.loads(text)

    def write_text(self, name, text):
        self._files[name] = text

    def set_fact(self, name, value):
        self.facts[name] = value

    def knob(self, name, default=""):
        return self._knobs.get(name, default)

    def path(self, name):
        return self.workdir / name


LEDGER_WITH_RELOOP = """\
# User Intent Ledger

## [2026-01-01T00:00:00+00:00] Initial Idea
Build a thing.

## [2026-01-01T01:00:00+00:00] verdict
id: abc123
gate: gate_strategy
verdict: reloop
scope: somehash
source: steer
consumed: yes

## [2026-01-01T01:01:00+00:00] correction
fix the typo
"""


def _make_ctx(scope_json: str | None, ledger_text: str = LEDGER_WITH_RELOOP,
              tmp_path: Path | None = None) -> FakeCtx:
    """Build a FakeCtx with the given scope json and a ledger on disk."""
    workdir = tmp_path or Path(".")
    if tmp_path is not None:
        (tmp_path / LEDGER).write_text(ledger_text)
    files = {}
    if scope_json is not None:
        files["reloop-scope.json"] = scope_json
    return FakeCtx(files=files, workdir=workdir)


# ── scope_reloop evaluator tests ─────────────────────────────────────────────

def test_scope_reloop_document(tmp_path):
    ctx = _make_ctx('{"scope": "document", "reason": "wording fix"}', tmp_path=tmp_path)
    verdict = scope_reloop(ctx)
    assert str(verdict) == "document"
    assert verdict.reason == "wording fix"
    assert ctx.facts["reloop_scope"] == "document"
    assert ctx.facts["reloop_scope_reason"] == "wording fix"
    assert ctx.facts["gate_strategy_feedback"] == "fix the typo"
    assert "Do not reconsider the strategy" in ctx.facts["gate_revision_instruction"]


def test_scope_reloop_approach(tmp_path):
    ctx = _make_ctx('{"scope": "approach", "reason": "wrong algorithm"}', tmp_path=tmp_path)
    verdict = scope_reloop(ctx)
    assert str(verdict) == "approach"
    assert ctx.facts["reloop_scope"] == "approach"
    assert ctx.facts["gate_strategy_feedback"] == "fix the typo"
    assert ctx.facts["gate_revision_instruction"] == ""


def test_classifier_paraphrase_does_not_affect_feedback(tmp_path):
    """The classifier's JSON can paraphrase/truncate — feedback comes from the ledger."""
    ledger = """\
# User Intent Ledger

## [2026-01-01T00:00:00+00:00] Initial Idea
Build a thing.

## [2026-01-01T01:00:00+00:00] verdict
id: abc123
gate: gate_strategy
verdict: reloop
scope: somehash
source: steer
consumed: yes

## [2026-01-01T01:01:00+00:00] correction
Add the missing test for edge case X
"""
    ctx = _make_ctx('{"scope": "document", "reason": "missing test"}',
                    ledger_text=ledger, tmp_path=tmp_path)
    scope_reloop(ctx)
    assert ctx.facts["gate_strategy_feedback"] == "Add the missing test for edge case X"


def test_scope_reloop_malformed_json(tmp_path):
    ctx = _make_ctx("not json at all {}", tmp_path=tmp_path)
    verdict = scope_reloop(ctx)
    assert str(verdict) == "approach"
    assert "defaulting to approach" in verdict.reason


def test_scope_reloop_missing_file(tmp_path):
    ctx = _make_ctx(None, tmp_path=tmp_path)
    verdict = scope_reloop(ctx)
    assert str(verdict) == "approach"


def test_scope_reloop_unrecognised_scope(tmp_path):
    ctx = _make_ctx('{"scope": "skip", "reason": "want to skip"}', tmp_path=tmp_path)
    verdict = scope_reloop(ctx)
    assert str(verdict) == "approach"


def test_scope_reloop_sets_all_facts(tmp_path):
    ctx = _make_ctx('{"scope": "document", "reason": "fix typo"}', tmp_path=tmp_path)
    scope_reloop(ctx)
    assert "reloop_scope" in ctx.facts
    assert "reloop_scope_reason" in ctx.facts
    assert "gate_strategy_feedback" in ctx.facts
    assert "gate_revision_instruction" in ctx.facts


# ── gate_design_review clearing tests ────────────────────────────────────────

def test_gate_design_review_clears_reloop_facts(tmp_path):
    design = ("# Fix\n\n## Summary\n\nFix.\n\n## Changes\n\n- tool.py\n\n"
              "## Tests\n\n- test.py\n")
    ctx = FakeCtx(
        files={"design.md": design, "brief.json": json.dumps({"brief": "a tool"})},
        facts={
            "name": "test",
            "has_pitfalls_step": False,
            "gate_strategy_feedback": "old feedback",
            "reloop_scope": "document",
            "reloop_scope_reason": "old reason",
            "gate_revision_instruction": "old instruction",
        },
        workdir=tmp_path,
    )
    gate_design_review(ctx)
    assert ctx.facts["gate_strategy_feedback"] == ""
    assert ctx.facts["reloop_scope"] == ""
    assert ctx.facts["reloop_scope_reason"] == ""
    assert ctx.facts["gate_revision_instruction"] == ""


def test_document_reloop_then_design_review_reloop(tmp_path):
    """After a document reloop followed by gate_design_review, the facts are cleared."""
    # Step 1: simulate scope_reloop setting facts for document scope
    (tmp_path / LEDGER).write_text(LEDGER_WITH_RELOOP)
    ctx_scope = _make_ctx('{"scope": "document", "reason": "fix typo"}', tmp_path=tmp_path)
    scope_reloop(ctx_scope)
    assert ctx_scope.facts["gate_strategy_feedback"] == "fix the typo"
    assert "Do not reconsider the strategy" in ctx_scope.facts["gate_revision_instruction"]

    # Step 2: simulate gate_design_review clearing the facts
    design = ("# Fix\n\n## Summary\n\nFix.\n\n## Changes\n\n- tool.py\n\n"
              "## Tests\n\n- test.py\n")
    ctx_review = FakeCtx(
        files={"design.md": design, "brief.json": json.dumps({"brief": "a tool"})},
        facts={
            "name": "test",
            "has_pitfalls_step": False,
            **{k: ctx_scope.facts[k] for k in (
                "gate_strategy_feedback", "reloop_scope",
                "reloop_scope_reason", "gate_revision_instruction")},
        },
        workdir=tmp_path,
    )
    gate_design_review(ctx_review)

    # Step 3: all four facts are empty — a subsequent design_doc prompt sees nothing
    assert ctx_review.facts["gate_strategy_feedback"] == ""
    assert ctx_review.facts["reloop_scope"] == ""
    assert ctx_review.facts["reloop_scope_reason"] == ""
    assert ctx_review.facts["gate_revision_instruction"] == ""

    # Rendering design_doc's prompt section with empty facts gives empty sections
    template = "\n## Strategy gate feedback\n{{gate_strategy_feedback}}\n\n{{gate_revision_instruction}}\n\n"
    rendered = template.replace("{{gate_strategy_feedback}}", ctx_review.facts["gate_strategy_feedback"])
    rendered = rendered.replace("{{gate_revision_instruction}}", ctx_review.facts["gate_revision_instruction"])
    assert "fix the typo" not in rendered
    assert "Do not reconsider the strategy" not in rendered
