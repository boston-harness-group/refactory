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


def test_design_fast_verifier_has_no_hunk_revert():
    """design_fast's verifier uses the standard prompt without hunk-revert instructions.

    It has no workspace clone, no retry, and uses VERIFIER_TIMEOUT_S (600 s) — the
    hunk-revert method requires capabilities design_fast does not have.
    """
    from refactory_dsh.verification import VERIFIER_TIMEOUT_S

    graph = DesignFastMode().graph()
    v = graph.nodes["verifier"]
    prompt_lower = v.prompt_template.lower()

    # design_fast must NOT have hunk-revert instructions
    assert "hunk-revert" not in prompt_lower, \
        "design_fast verifier must not contain hunk-revert instructions"
    assert "requirements_and_corrections" not in v.prompt_template, \
        "design_fast verifier must not reference requirements_and_corrections"
    # design_fast uses the shorter timeout and no workspace clone
    assert v.timeout_s == VERIFIER_TIMEOUT_S, \
        f"design_fast verifier must use VERIFIER_TIMEOUT_S (600), got {v.timeout_s}"
    assert v.workspace is None, \
        "design_fast verifier must not have a workspace clone"
    assert v.retry is None, \
        "design_fast verifier must not have a retry"


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


# ── edge-case sweep, per-function weakened-test check, replay model flags ──


def test_design_verifier_prompt_requires_the_edge_case_sweep():
    """The design verifier sweeps every edge-case class per requirement, not 2-3 probes."""
    from refactory_dsh.verification import DESIGN_VERIFIER_PROMPT

    prompt = DESIGN_VERIFIER_PROMPT
    assert "2–3 probes" not in prompt and "2-3 probes" not in prompt
    assert "For each requirement in the brief" in prompt
    for cls in ("Position:", "the LAST element", "All / none / exactly-one",
                "every item fails, no item fails", "Timeouts and hangs",
                "deadline actually\n  stops the work", "not retried",
                "Interaction with control flow", "gate, pause, halt, stop or resume",
                "Concurrency:", "forks or\n  threads"):
        assert cls in prompt, cls
    # Both runtimes: the same probe on walker and spine, compared.
    assert "Both runtimes" in prompt
    assert "shared runtime code" in prompt
    assert "on the walker AND on the\n  spine (through lowering" in prompt
    # Each probe gives its command and output; failures are findings; unprobed is listed.
    assert "exact command you ran and its output" in prompt
    assert "A failing probe is a\ndefect finding" in prompt
    assert "do not apply and why" in prompt
    assert "list every class you did not\nprobe as **UNCHECKED**" in prompt


def test_sweep_injects_failures_at_every_point_and_last_element():
    """Rule 1: each failure at every distinct point in the diff; LAST element at each."""
    from refactory_dsh.verification import DESIGN_VERIFIER_PROMPT as prompt

    assert "inject each failure at every distinct point in the diff" in prompt
    assert "each step/op inside the changed path" in prompt
    assert "not only the first point that works" in prompt
    assert "make\n  the LAST element fail at each of the failure points" in prompt


def test_sweep_probes_are_own_code_not_builder_tests():
    """Rule 2: a probe is new code the verifier writes; builder tests do not count."""
    from refactory_dsh.verification import DESIGN_VERIFIER_PROMPT as prompt

    assert "A probe is new code you write and run." in prompt
    assert "Running the builder's existing tests does\nnot count as a probe" in prompt


def test_sweep_read_is_not_probed_and_each_runtime_is_run():
    """Rule 3: PROBED means executed; reading is READ, not probed; unrun runtime UNCHECKED."""
    from refactory_dsh.verification import DESIGN_VERIFIER_PROMPT as prompt

    assert "PROBED means executed, with its\ncommand and output shown" in prompt
    assert 'reported as "READ, not\nprobed"' in prompt
    assert '"READ, not probed", NOT APPLICABLE' in prompt
    assert "through lowering + the TS drain" in prompt
    assert "A runtime you could not run the\n  probe on is UNCHECKED" in prompt


def test_sweep_fixes_environment_and_reruns_the_same_probe():
    """Rule 4: environment failures are fixed and the same probe rerun, never weakened."""
    from refactory_dsh.verification import DESIGN_VERIFIER_PROMPT as prompt

    assert "fails\nfor an environment reason" in prompt
    assert "set PYTHONPATH to the repo's test helpers" in prompt
    assert "rerun the same\nprobe; never substitute a weaker probe" in prompt


def test_m15_requires_a_per_function_comparison_against_the_base():
    """M15 diffs the test files, lists every modified test function, and may not say
    "no weakened tests" without that list."""
    from refactory_dsh.verification import (
        DESIGN_VERIFIER_PROMPT, FULL_VERIFICATION_METHODS, M15_METHOD, VERIFICATION_METHODS,
    )

    # Compare on collapsed whitespace: the wording is the contract, not its wrapping.
    flat = " ".join(M15_METHOD.split())
    for text in ('git diff <base> -- <test files>', "list every existing test function",
                 "file::name", "an assert removed or relaxed",
                 "an expected value changed to match the new behaviour",
                 "renamed or reworded to accept different behaviour", "skip/xfail added",
                 'You may not conclude "no weakened tests" without listing the '
                 'modified test functions you compared'):
        assert " ".join(text.split()) in flat, text
    # And M15 reports rather than blocks: the detector cannot tell a loosened
    # assertion from an expected value updated to match a required behaviour.
    assert "it does not block" in flat, M15_METHOD
    assert M15_METHOD in VERIFICATION_METHODS and M15_METHOD in FULL_VERIFICATION_METHODS
    assert "{{verification_methods}}" in DESIGN_VERIFIER_PROMPT


def _replay_run(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / ".refactory").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    (run_dir / "brief.json").write_text('{"brief": "do a thing"}')
    (run_dir / ".refactory" / "worktree.json").write_text(f'{{"path": "{work}"}}')
    return run_dir


def test_replay_verifier_parses_and_routes_the_model_flags(tmp_path, monkeypatch, capsys):
    """--provider/--model/--effort reach the route the verifier's session runs on."""
    import refactory_dsh.model as model_mod
    from refactory_dsh import cli

    args = cli.build_parser().parse_args([
        "replay-verifier", "--run-dir", "r", "--provider", "p1", "--model", "m1",
        "--effort", "high"])
    assert (args.provider, args.model, args.effort) == ("p1", "m1", "high")

    built: list[dict] = []

    class FakeClient:
        def __init__(self, **kwargs):
            built.append(kwargs)
            self.kwargs = kwargs

        def context(self):
            return {"worker_backend": "dsh", "provider": self.kwargs.get("provider", ""),
                    "model": self.kwargs.get("model", ""), "effort": self.kwargs.get("effort", ""),
                    "routes_resolved": True}

        def complete(self, prompt, **kwargs):
            return "VERDICT: PASS"

    monkeypatch.setattr(model_mod, "DshModelClient", FakeClient)
    run_dir = _replay_run(tmp_path)
    code = cli.main(["replay-verifier", "--run-dir", str(run_dir), "--provider", "p1",
                     "--model", "m1", "--effort", "high"])
    assert code == 0
    assert built[0]["provider"] == "p1" and built[0]["model"] == "m1"
    assert built[0]["effort"] == "high"
    # The session's client is built from the resolved route.
    assert built[-1]["provider"] == "p1" and built[-1]["model"] == "m1"
    assert built[-1]["effort"] == "high" and built[-1]["routes_resolved"] is True


def test_replay_verifier_refuses_half_a_route(tmp_path, monkeypatch, capsys):
    """A provider without a model is refused before any session, as a design run is."""
    import refactory_dsh.model as model_mod
    from refactory_dsh import cli

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def context(self):
            return {"worker_backend": "dsh", "provider": self.kwargs.get("provider", ""),
                    "model": ""}

        def complete(self, prompt, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("no session without a complete route")

    monkeypatch.setattr(model_mod, "DshModelClient", FakeClient)
    code = cli.main(["replay-verifier", "--run-dir", str(_replay_run(tmp_path)),
                     "--provider", "p1"])
    assert code == 1
    assert "refactory:" in capsys.readouterr().err
