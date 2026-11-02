"""Tests for the verifier-based hunk-revert verification.

Each test fails when its corresponding change is reverted.
"""

from __future__ import annotations

from refactory_dsh import design_ops, judges
from refactory_dsh.design import (
    DesignMode, TIMEOUT_RETRY, prepare_check_iterations,
)
from refactory_dsh.fast import DesignFastMode
from refactory_dsh.verification import DESIGN_VERIFIER_TIMEOUT_S


def test_verifier_prompt_carries_method_and_repo_access():
    """The verifier node's prompt includes the hunk-revert method, workspace access,
    brief.json requirements and gate corrections in its prompt context;
    'design.md' is NOT in its reads (the verifier is design-blind).
    """
    graph = DesignMode().graph()
    v = graph.nodes["verifier"]

    prompt = v.prompt_template.lower()

    # Hunk-revert method keywords
    assert "revert" in prompt, "prompt must describe the revert method"
    assert "hunk" in prompt, "prompt must mention hunks"
    assert "caught" in prompt, "prompt must mention caught/not caught"
    assert "not caught" in prompt, "prompt must mention not caught"

    # Scratch copy / workspace access
    assert v.workspace == {"kind": "clone", "of": "attempt", "collect": "files"}, \
        "verifier must get a clone workspace"
    assert "workspace" in prompt or "scratch" in prompt, \
        "prompt must mention workspace or scratch copy"

    # Requirements and corrections injected into prompt context
    assert "requirements_and_corrections" in v.prompt_template, \
        "prompt must carry the requirements_and_corrections template variable"

    # Design-blind: design.md is NOT in reads
    assert "design.md" not in v.reads, \
        "verifier must not read design.md (design-blind)"

    # Time budget stated
    assert str(DESIGN_VERIFIER_TIMEOUT_S) in v.prompt_template, \
        "prompt must state the time budget"
    assert v.timeout_s == DESIGN_VERIFIER_TIMEOUT_S

    # Docs-only / test-only requirements reported as NOT APPLICABLE
    assert "not applicable" in prompt, \
        "prompt must describe NOT APPLICABLE for docs/test-only requirements"

    # Unchecked requirements reported as UNCHECKED (advisory)
    assert "unchecked" in prompt, \
        "prompt must describe UNCHECKED for requirements not checked in time"

    # Probe instructions
    assert "probe" in prompt, "prompt must include probe instructions"


def test_uncaught_hunk_produces_blocking_check_finding():
    """A finding with kind 'check' and severity 'high' that breaks a requirement
    blocks via judges.blocks().
    """
    # This is the finding shape the verifier's judging instructions produce
    item = {
        "kind": "check",
        "severity": "high",
        "breaks": "ask",
        "what": "uncaught hunk: design_ops.py:2147 — requirement 'remove site-revert pass' "
                "is not caught by any test when its hunk is reverted",
    }
    assert judges.blocks(item) is True, \
        "a check finding with severity high must block"

    # Contrast: a follow-up item does not block
    followup = {"kind": "docs", "severity": "low", "breaks": "none"}
    assert judges.blocks(followup) is False, "a docs follow-up must not block"

    # A defect finding from a probe that fails also blocks when high severity
    probe_finding = {
        "kind": "defect",
        "severity": "high",
        "breaks": "ask",
        "what": "probe failure: edge case X not handled",
    }
    assert judges.blocks(probe_finding) is True, \
        "a high-severity defect finding must block"


def test_revert_uses_scratch_copy():
    """The verifier's workspace is a clone, so the attempt's worktree and the
    committed tree are untouched.
    """
    graph = DesignMode().graph()
    v = graph.nodes["verifier"]

    assert v.workspace == {"kind": "clone", "of": "attempt", "collect": "files"}, \
        "verifier must get a clone workspace"
    assert v.outputs_only is True, \
        "outputs_only must remain True (retry safety)"
    assert v.retry == TIMEOUT_RETRY, \
        "verifier must use the standard timeout retry"


def test_site_revert_fixups_knob_removed():
    """DesignMode.declare() has no site_revert_fixups key;
    DesignFastMode.declare() does not pop it.
    """
    design_knobs = DesignMode.declare()
    assert "site_revert_fixups" not in design_knobs, \
        "site_revert_fixups knob must be removed from DesignMode"

    fast_knobs = DesignFastMode.declare()
    assert "site_revert_fixups" not in fast_knobs, \
        "site_revert_fixups must not appear in DesignFastMode either"

    # The iteration formula changes from 54 to 45 with defaults
    result = prepare_check_iterations({})
    # (8 + 6 + 1) * 2 + 6 + 8 + 1 = 45
    assert result == 45, \
        f"prepare_check_iterations with defaults should be 45, got {result}"


def test_no_fixup_path():
    """The removed functions and classes no longer exist in their modules."""
    from refactory_dsh import design_contract

    # design_ops: removed functions
    assert not hasattr(design_ops, "_run_site_reverts"), \
        "_run_site_reverts must be removed"
    assert not hasattr(design_ops, "_pairs_from_design"), \
        "_pairs_from_design must be removed"
    assert not hasattr(design_ops, "_tests_for_source"), \
        "_tests_for_source must be removed"
    assert not hasattr(design_ops, "_site_revert_fixup"), \
        "_site_revert_fixup must be removed"
    assert not hasattr(design_ops, "_record_site_revert_advisory"), \
        "_record_site_revert_advisory must be removed"

    # design_contract: removed class and function
    assert not hasattr(design_contract, "parse_site_path"), \
        "parse_site_path must be removed"
    assert not hasattr(design_contract, "SiteParsed"), \
        "SiteParsed must be removed"

    # prepare_check node does not produce site_revert_fixup_count
    graph = DesignMode().graph()
    pc = graph.nodes["prepare_check"]
    assert "site_revert_fixup_count" not in pc.produces, \
        "prepare_check must not produce site_revert_fixup_count"
    assert "total_site_revert_fixups_used" not in pc.produces, \
        "prepare_check must not produce total_site_revert_fixups_used"
