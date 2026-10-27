"""R1 tests: the independent verification step exists and is wired correctly."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from refactory_dsh import nn
from refactory_dsh.design import DesignMode
from refactory_dsh.fast import DesignFastMode
from refactory_dsh.verification import VERIFICATION_METHODS, VERIFIER_PROMPT


# ── graph structure tests ─────────────────────────────────────────────────


#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def test_verifier_node_exists_in_design_mode():
    graph = DesignMode().build()
    assert "verifier" in graph.nodes
    assert "gate_verification" not in graph.nodes


def test_verifier_node_exists_in_design_fast_mode():
    graph = DesignFastMode().build()
    assert "verifier" in graph.nodes
    assert "gate_verification" not in graph.nodes


def test_verifier_is_design_blind():
    graph = DesignMode().build()
    node = graph.nodes["verifier"]
    assert "design.md" not in node.reads
    assert "strategy.json" not in node.reads
    assert "{{design}}" not in node.prompt_template
    assert "Read the approved design from" not in node.prompt_template


def test_verifier_prompt_says_design_md_is_in_run_dir():
    assert "design.md is in the run directory" in VERIFIER_PROMPT


def test_verifier_writes_verification_md():
    graph = DesignMode().build()
    assert "reviews/verification.md" in graph.nodes["verifier"].writes


def test_verifier_reads_changes_diff():
    graph = DesignMode().build()
    assert "changes.diff" in graph.nodes["verifier"].reads


def test_the_verifier_applies_only_the_methods_no_reviewer_applies():
    from refactory_dsh.authoring_evidence import (ARTIFACT_REVIEW_INSTRUCTIONS,
                                                  REPO_CHANGE_REVIEW_INSTRUCTIONS)
    from refactory_dsh.verification import (FULL_VERIFICATION_METHODS, FULL_VERIFIER_PROMPT,
                                            verification_methods)
    # A repository change: callers, claims and shared state are the code reviewers'.
    repo = verification_methods("repository_change")
    for method in ("M1 ", "M2 ", "M13 ", "M15 "):
        assert method in repo
    for method in ("M3 ", "M11 ", "M12 "):
        assert method not in repo
        assert method in REPO_CHANGE_REVIEW_INSTRUCTIONS
    # A new artifact's reviewers do not apply them, so the verifier keeps them.
    artifact = verification_methods("artifact")
    assert artifact == FULL_VERIFICATION_METHODS
    for method in ("M3 ", "M11 ", "M12 "):
        assert method in artifact and method not in ARTIFACT_REVIEW_INSTRUCTIONS
    # The review fork's prompt takes the run's set; fix_pack's has all of them.
    assert "{{verification_methods}}" in VERIFIER_PROMPT
    assert "M12 " in FULL_VERIFIER_PROMPT


def test_overwatch_does_not_declare_the_verifier_findings_as_an_input():
    # On the spine a declared input must exist, and in a run whose verifier found
    # nothing it does not.
    assert "reviews/verification-unresolved.md" not in DesignMode().build().nodes["overwatch"].reads


def test_verifier_prompt_instructs_output_format():
    assert "BLOCKING:" in VERIFIER_PROMPT
    assert "VERDICT: PASS" in VERIFIER_PROMPT
    assert "file:line" in VERIFIER_PROMPT


def test_graph_shape_full_mode():
    graph = DesignMode().build()
    fork = graph.nodes["fork_qa"]
    assert "verifier" in fork.targets
    assert "verifier" in fork.optional_targets
    assert "gate_verification" not in graph.nodes
    edges = [(e.source, e.target, e.condition) for e in graph.edges]
    assert ("check", "verifier", "proceed") not in edges
    assert ("prepare_check", "fork_qa", "proceed") in edges
    assert ("verifier", "join_qa", None) in edges


def test_graph_shape_fast_mode():
    graph = DesignFastMode().build()
    fork = graph.nodes["fork_qa"]
    assert "verifier" in fork.targets
    assert "focused_qa" in fork.targets
    assert "verifier" in fork.optional_targets
    assert "gate_check" in graph.nodes
    assert "gate_verification" not in graph.nodes
    edges = [(e.source, e.target, e.condition) for e in graph.edges]
    assert ("focused_check", "verifier", None) not in edges
    assert ("focused_check", "gate_check", None) in edges
    assert ("gate_check", "fork_qa", "proceed") in edges
    assert ("verifier", "join_qa", None) in edges


def test_verifier_not_in_reviewers():
    from refactory_dsh.authoring_evidence import REVIEWERS
    assert "verifier" not in REVIEWERS


def test_verifier_not_in_planning_steps():
    from refactory_dsh.design_ops import PLANNING_STEPS
    assert "verifier" not in PLANNING_STEPS


def test_verifier_has_final_validation_resource_role_full_mode():
    graph = DesignMode().build()
    assert graph.nodes["verifier"].resource_role == "final_validation"


# ── gate_qa verifier handling ─────────────────────────────────────────────


def test_the_verifier_never_gets_a_round_of_its_own():
    # No gate counts verifier rounds, because there are none to count.
    assert "verification_advisory_used" not in DesignMode().build().nodes["gate_qa"].produces
    assert "verification_advisory_used" not in DesignFastMode().build().nodes["gate_qa"].produces


def test_gate_qa_max_iterations_raised():
    graph = DesignMode().build()
    gate = graph.nodes["gate_qa"]
    assert gate.max_iterations >= 5


# ── wire_verification raises on missing edge ──────────────────────────────


def test_wire_verification_raises_on_missing_edge():
    from refactory_dsh.nn import Graph, Node
    from refactory_dsh.verification import wire_verification
    graph = Graph()
    graph.add(Node("check", "gate", evaluator="x:y"))
    graph.add(Node("emit", "agent", prompt_template="x"))
    graph.add(Node("report", "op", call="x:y"))
    graph.connect("check", "emit", condition="reloop")
    with pytest.raises(ValueError, match="no edge"):
        wire_verification(graph, check_node="check", proceed_target="report")


# ── fix_pack gets the verifier ────────────────────────────────────────────


def test_fix_pack_has_verifier():
    from refactory_dsh import registry
    from refactory_dsh.paths import REPO_ROOT
    mode = registry.resolve(REPO_ROOT, "fix_pack/make_pass")[1]()
    graph = mode.graph()
    assert "verifier" in graph.nodes
    assert "gate_verification" in graph.nodes
    edges = [(e.source, e.target, e.condition) for e in graph.edges]
    assert ("check", "verifier", "proceed") in edges
    assert ("gate_verification", "report", "proceed") in edges
    assert ("gate_verification", "emit", "reloop") in edges
    assert ("check", "report", "proceed") not in edges


# ── reviewer independence ────────────────────────────────────────────────


def test_reviewer_independence():
    """No reviewer reads reviews/verification.md or has it in its prompt."""
    graph = DesignMode().build()
    for name in ("health_checker", "code_reviewer", "qa_director"):
        node = graph.nodes[name]
        assert "reviews/verification.md" not in node.reads, f"{name} reads verification.md"
        assert "reviews/verification.md" not in node.prompt_template, (
            f"{name} prompt mentions verification.md")

    fast_graph = DesignFastMode().build()
    fqa = fast_graph.nodes["focused_qa"]
    assert "reviews/verification.md" not in fqa.reads, "focused_qa reads verification.md"


# ── scripted run tests ────────────────────────────────────────────────────


def test_verification_pass_proceeds_to_reviews_full_mode(tmp_path, monkeypatch):
    """A PASS verdict proceeds to the reviews without reloop."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import (DESIGN_DOC_ANSWER,
                               PITFALLS_ANSWER, RESEARCH_ANSWER,
                               FakeModel, render)
    VERIFICATION_PASS = "No blocking findings.\n\nVERDICT: PASS"
    answers = [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER,
        render({"tool.py": "print('built')\n"}),
        # fork_qa: health_checker, code_reviewer, qa_director, verifier
        "health: passed", "code review: passed", "qa: passed",
        VERIFICATION_PASS,
        # overwatch
        "VERDICT: PASS",
    ]
    result = DesignMode().run(
        tmp_path / "run", model=FakeModel(answers),
        verdicts={"gate_strategy": "proceed"},
        task={"name": "verify_pass", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error


def test_verifier_findings_go_to_the_qa_judge_unparsed_and_the_pr_not_a_round(tmp_path, monkeypatch):
    """Design mode: the verifier's report goes, whole, to the QA judge with the reviews,
    and is not parsed. A judge that lists its finding as a follow-up costs no builder
    round, and the pull request lists it."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import (DESIGN_DOC_ANSWER,
                               PITFALLS_ANSWER, RESEARCH_ANSWER,
                               FakeModel, overwatch_judgment, qa_judgment, render)
    VERIFICATION_FAIL = "BLOCKING: tool.py:1: negative values accepted\n\nVERDICT: FAIL"
    answers = [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER,
        render({"tool.py": "print('built')\n"}),
        "health: passed", "code: passed", "qa: passed",
        VERIFICATION_FAIL,
        qa_judgment("proceed", follow_ups=["tool.py:1: negative values accepted (verifier)"]),
        overwatch_judgment("pass"),
    ]
    model = FakeModel(answers)
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    mode.load_state_dict(state)
    result = mode.run(
        tmp_path / "run", model=model,
        verdicts={"gate_strategy": "proceed"},
        task={"name": "advisory_full", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error
    assert len([p for p in model.prompts if "What went wrong last time" in p]) == 1
    assert int(result.facts.get("review_attempts_used", 0)) == 0
    # A new artifact: the verifier was given every method.
    verifier = next(p for p in model.prompts if "design.md is in the run directory" in p)
    assert "M12 Shared state" in verifier
    synthesis = (tmp_path / "run" / "reviews" / "qa-synthesized.md").read_text()
    assert "## reviews/verification.md (the design-blind verifier)" in synthesis
    assert VERIFICATION_FAIL in synthesis
    assert not (tmp_path / "run" / "reviews" / "verification-unresolved.md").exists()
    assert "negative values accepted" in (tmp_path / "run" / "pull-request.md").read_text()


def test_overwatch_sends_back_a_verifier_finding_that_breaks_an_ask(tmp_path, monkeypatch):
    """The verifier's finding costs a round only when a judge finds it breaks an ask."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import (DESIGN_DOC_ANSWER,
                               PITFALLS_ANSWER, RESEARCH_ANSWER,
                               FakeModel, overwatch_judgment, render)
    VERIFICATION_FAIL = "BLOCKING: tool.py:1: negative values accepted\n\nVERDICT: FAIL"
    OVERWATCH_REDO = overwatch_judgment("redo", must_fix=[{
        "source": "verifier", "where": "tool.py:1",
        "what": "negative values accepted, which the brief forbids", "why": "the brief"}])
    answers = [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER,
        render({"tool.py": "print('built')\n"}),
        "health: passed", "code: passed", "qa: passed", VERIFICATION_FAIL,
        OVERWATCH_REDO,
        render({"tool.py": "print('built v2')\n"}),
        "health: passed", "code: passed", "qa: passed",
        "No blocking findings.\n\nVERDICT: PASS",
        overwatch_judgment("pass"),
    ]
    model = FakeModel(answers)
    result = DesignMode().run(
        tmp_path / "run", model=model,
        verdicts={"gate_strategy": "proceed"},
        task={"name": "overwatch_verifier", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error
    builders = [p for p in model.prompts if "What went wrong last time" in p]
    assert len(builders) == 2
    assert "which the brief forbids" in builders[1]
    assert not (tmp_path / "run" / "reviews" / "verification-unresolved.md").exists()


def test_joint_fix_round(tmp_path, monkeypatch):
    """A QA finding and a verifier finding the QA judge both must-fix: one reloop with both."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import (DESIGN_DOC_ANSWER,
                               PITFALLS_ANSWER, RESEARCH_ANSWER,
                               FakeModel, overwatch_judgment, qa_judgment, render)
    VERIFICATION_FAIL = "BLOCKING: tool.py:1: noise\n\nVERDICT: FAIL"
    QA_FAIL = "BLOCKING: tool.py:5: missing feature\n\nVERDICT: FAIL"
    answers = [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER,
        # Round 1: QA and verifier both report, and the judge must-fixes both
        render({"tool.py": "print('built')\n"}),
        "health: passed", "code: passed", QA_FAIL,
        VERIFICATION_FAIL,
        qa_judgment("redo", must_fix=[
            {"source": "qa_director", "where": "tool.py:5", "what": "missing feature", "why": "ask 1"},
            {"source": "verifier", "where": "tool.py:1", "what": "noise", "why": "ask 2"}]),
        # Reloop → emit (combined feedback)
        render({"tool.py": "print('fixed')\n"}),
        "health: passed", "code: passed", "qa: passed",
        "No blocking findings.\n\nVERDICT: PASS",
        overwatch_judgment("pass"),
    ]
    model = FakeModel(answers)
    result = DesignMode().run(
        tmp_path / "run", model=model,
        verdicts={"gate_strategy": "proceed"},
        task={"name": "joint_fix", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error
    # The emit prompt in round 2 must carry both findings
    emit_prompts = [p for p in model.prompts if "What went wrong last time" in p]
    assert len(emit_prompts) == 2
    second_emit = emit_prompts[1]
    assert "missing feature" in second_emit and "noise" in second_emit


def test_verifier_branch_failure(tmp_path, monkeypatch):
    """A verifier that returns empty is treated as missing in the synthesis."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import (DESIGN_DOC_ANSWER,
                               PITFALLS_ANSWER, RESEARCH_ANSWER,
                               FakeModel, render)
    # The verifier returns empty output — no VERDICT line, no findings.
    # This simulates a branch that produced nothing useful.
    VERIFICATION_EMPTY = ""
    answers = [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER,
        render({"tool.py": "print('ok')\n"}),
        # fork_qa: 3 reviews pass, verifier returns empty
        "health: passed", "code: passed", "qa: passed",
        VERIFICATION_EMPTY,
        # overwatch
        "VERDICT: PASS",
    ]
    model = FakeModel(answers)
    result = DesignMode().run(
        tmp_path / "run", model=model,
        verdicts={"gate_strategy": "proceed"},
        task={"name": "empty_verifier", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error
    # Design mode: a verifier that left nothing is simply absent from the QA judge's input.
    synthesis = (tmp_path / "run" / "reviews" / "qa-synthesized.md").read_text()
    assert "verifier" not in synthesis
    assert synthesis.startswith("# QA synthesis\n\n3 of 3 reviews reported.\n")


class _CrashingModel:
    """Scripted answers; any prompt carrying ``mark`` raises, as a crashed branch."""

    def __init__(self, answers, mark):
        from tests_support import FakeModel
        self.inner, self.mark, self.crashes = FakeModel(answers), mark, 0

    def complete(self, prompt, *, system="", workdir=None, **workspace):
        if self.mark in prompt:
            self.crashes += 1
            raise RuntimeError("branch crashed")
        return self.inner.complete(prompt, system=system, workdir=workdir, **workspace)


_VERIFIER_MARK = "design.md is in the run directory. Do NOT read it"


@pytest.mark.parametrize("crash, survives", [(_VERIFIER_MARK, True),
                                             ("check you did not run is not a check that passed", False)])
def test_crashed_branch_full_mode(tmp_path, monkeypatch, crash, survives):
    """A verifier that raises is missing, and the run completes; a reviewer that
    raises still fails the run."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import (DESIGN_DOC_ANSWER,
                               PITFALLS_ANSWER, RESEARCH_ANSWER, REVIEW_ANSWERS, render)
    reviews = REVIEW_ANSWERS[:3] + (["No blocking findings.\n\nVERDICT: PASS"] if not survives else [])
    if not survives:
        reviews = reviews[1:]
    model = _CrashingModel([RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
                            DESIGN_DOC_ANSWER,
                            render({"tool.py": "print('ok')\n"}), *reviews, "VERDICT: PASS"], crash)
    result = DesignMode().run(tmp_path / "run", model=model, verdicts={"gate_strategy": "proceed"},
                              task={"name": "crash", "brief": "a tool", "check": "python tool.py"})
    assert model.crashes == 1
    assert result.exit_ok is survives, result.error
    if survives:
        assert "proceed" in (tmp_path / "run" / "reviews" / "qa-gate.txt").read_text()
    else:
        assert "health_checker" in (result.error or "")


@pytest.mark.parametrize("crash, survives", [(_VERIFIER_MARK, True),
                                             ("Perform one focused QA review", False)])
def test_crashed_branch_fast_mode(tmp_path, monkeypatch, crash, survives):
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import render
    fast_qa = json.dumps({"status": "pass", "checks": [{"name": "runs", "command": "python tool.py",
                                                         "status": "pass", "evidence": "exit 0"}],
                          "findings": [], "not_run": []})
    other = fast_qa if survives else "No blocking findings.\n\nVERDICT: PASS"
    model = _CrashingModel(["Change only the requested file.", render({"tool.py": "print('ok')\n"}), other],
                           crash)
    result = DesignFastMode().run(tmp_path / "run", model=model,
                                  task={"name": "crash", "brief": "a tool", "check": "python tool.py"})
    assert model.crashes == 1
    assert result.exit_ok is survives, result.error
    if not survives:
        assert "focused_qa" in (result.error or "")


def test_finding_sources_json(tmp_path):
    """create_ops.gate_qa, which a run lowered before design mode's judges still calls
    with a verifier beside its reviews: round 1, QA and verifier findings go back
    together; round 2, QA is fixed, the builder rebuts the verifier's finding, and the gate
    proceeds. Each entry's fate and first_round/last_round are recorded."""
    from refactory_dsh import create_ops

    VERIFICATION_FAIL = "BLOCKING: tool.py:1: negative values accepted\n\nVERDICT: FAIL"
    QA_FAIL = "BLOCKING: tool.py:5: missing error handling\n\nVERDICT: FAIL"
    ctx = nn.RunContext(tmp_path)

    def attempt(qa, emit=""):
        ctx.write_text("reviews/health-check.md", "health: passed\n")
        ctx.write_text("reviews/code-review.md", "code: passed\n")
        ctx.write_text("reviews/qa.md", qa)
        ctx.write_text("reviews/verification.md", VERIFICATION_FAIL)
        if emit:
            ctx.write_text("emit.txt", emit)
        create_ops.synthesize_qa(ctx)
        return create_ops.gate_qa(ctx)

    assert attempt(QA_FAIL) == "reloop"
    # create's builder writes its own emission, so it answers the verifier there.
    assert "respond in emit.txt under ## Verifier findings" in ctx.read_text("reviews/qa-gate.txt")
    assert attempt("qa: passed", "## Verifier findings\n"
                   "negative values accepted → rebutted: false positive\n") == "proceed"
    path = tmp_path / "reviews" / "finding-sources.json"
    assert path.exists(), "finding-sources.json must be written"
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    for entry in data:
        assert "source" in entry
        assert "review_path" in entry
        assert "finding" in entry
        assert "fate" in entry
        assert entry["fate"] in ("fixed", "rebutted", "unresolved")
        assert entry["class"] == "core"
        assert "first_round" in entry
        assert "last_round" in entry
    qa_entries = [e for e in data if e["source"] == "qa_director"
                  and "missing error handling" in e["finding"]]
    assert len(qa_entries) >= 1, "QA finding must appear in finding-sources.json"
    assert qa_entries[0]["fate"] == "fixed", (
        f"QA finding fate should be 'fixed', got {qa_entries[0]['fate']!r}")
    assert qa_entries[0]["first_round"] == 1
    verifier_entries = [e for e in data if e["source"] == "verifier"
                        and "negative values accepted" in e["finding"]]
    assert len(verifier_entries) >= 1, "verifier finding must appear"
    assert verifier_entries[0]["fate"] == "rebutted", (
        f"verifier finding fate should be 'rebutted', got {verifier_entries[0]['fate']!r}")
    assert verifier_entries[0]["first_round"] == 1
    assert verifier_entries[0]["last_round"] == 2


def test_verifier_only_findings_cost_no_repair_fast_mode(tmp_path, monkeypatch):
    """Verifier FAIL with focused QA PASS: no builder round; the finding is listed."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import FakeModel, render
    FAST_PLAN = "Change only the requested file."
    FAST_QA = json.dumps({"status": "pass", "checks": [{"name": "runs", "command": "python tool.py",
                                                         "status": "pass", "evidence": "exit 0"}],
                          "findings": [], "not_run": []})
    VERIFICATION_FAIL = "BLOCKING: tool.py:1: negative values accepted\n\nVERDICT: FAIL"
    answers = [
        FAST_PLAN,
        render({"tool.py": "print('built')\n"}),
        # fork: focused_qa, verifier
        FAST_QA,
        VERIFICATION_FAIL,
    ]
    mode = DesignFastMode()
    state = mode.state_dict()
    state["knobs"]["repair_attempts"]["value"] = 2
    mode.load_state_dict(state)
    model = FakeModel(answers)
    result = mode.run(
        tmp_path / "run", model=model,
        task={"name": "advisory_fast", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error
    assert int(result.facts.get("repairs_used", 0)) == 0
    assert len(model.prompts) == 4, "plan, emit, focused QA, verifier: no second build"
    unresolved = (tmp_path / "run" / "reviews" / "verification-unresolved.md").read_text()
    assert "negative values accepted" in unresolved


def test_verification_pass_proceeds_fast_mode(tmp_path, monkeypatch):
    """A PASS verdict in fast mode proceeds to report."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import FakeModel, render
    FAST_PLAN = "Change only the requested file."
    FAST_QA = json.dumps({"status": "pass", "checks": [{"name": "runs", "command": "python tool.py",
                                                         "status": "pass", "evidence": "exit 0"}],
                          "findings": [], "not_run": []})
    VERIFICATION_PASS = "No blocking findings.\n\nVERDICT: PASS"
    answers = [FAST_PLAN, render({"tool.py": "print('built')\n"}),
               FAST_QA, VERIFICATION_PASS]
    result = DesignFastMode().run(
        tmp_path / "run", model=FakeModel(answers),
        task={"name": "verify_fast_pass", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error


def test_fast_mode_failing_check_routes_through_gate_check(tmp_path, monkeypatch):
    """A failing check routes through gate_check, not through the verifier."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import FakeModel, render
    FAST_PLAN = "Change only the requested file."
    FAST_QA = json.dumps({"status": "pass", "checks": [{"name": "runs", "command": "python tool.py",
                                                         "status": "pass", "evidence": "exit 0"}],
                          "findings": [], "not_run": []})
    VERIFICATION_PASS = "No blocking findings.\n\nVERDICT: PASS"
    answers = [
        FAST_PLAN,
        render({"tool.py": "import sys; sys.exit(1)\n"}),   # check fails
        # gate_check routes to _repair (no verifier runs yet)
        render({"tool.py": "print('fixed')\n"}),             # second emit
        FAST_QA, VERIFICATION_PASS,
    ]
    mode = DesignFastMode()
    state = mode.state_dict()
    state["knobs"]["repair_attempts"]["value"] = 2
    mode.load_state_dict(state)
    model = FakeModel(answers)
    result = mode.run(
        tmp_path / "run", model=model,
        task={"name": "check_fail_fast", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error
    assert int(result.facts.get("repairs_used", 0)) >= 1, "a repair must have been spent"


def test_finding_description_strips_file_line_prefix():
    """_finding_description strips the file:line: prefix used in BLOCKING findings."""
    from refactory_dsh.create_ops import _finding_description
    assert _finding_description("tool.py:1: negative values accepted") == "negative values accepted"
    assert _finding_description("src/app.py:42: missing check") == "missing check"
    assert _finding_description("the verifier did not state 'VERDICT: PASS'") == "the verifier did not state 'VERDICT: PASS'"
    assert _finding_description("simple finding") == "simple finding"


def test_rebuttal_matching_works_without_prefix():
    """Rebuttal matching strips the file:line: prefix so a builder's rebuttal
    like 'negative values accepted → rebutted: ...' matches the finding
    'tool.py:1: negative values accepted'."""
    from refactory_dsh.create_ops import _finding_description
    finding = "tool.py:1: negative values accepted"
    rebuttal = "negative values accepted → rebutted: false positive"
    desc = _finding_description(finding)
    assert desc.lower() in rebuttal.lower(), (
        f"description {desc!r} should be found in rebuttal {rebuttal!r}")


def test_rebuttal_fate_fast_mode(tmp_path, monkeypatch):
    """A verifier finding that rode along with a core repair and was rebutted is
    recorded with the 'rebutted' fate in finding-sources.json."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import FakeModel, render
    FAST_PLAN = "Change only the requested file."
    FAST_QA = json.dumps({"status": "pass", "checks": [{"name": "runs", "command": "python tool.py",
                                                         "status": "pass", "evidence": "exit 0"}],
                          "findings": [], "not_run": []})
    VERIFICATION_FAIL = "BLOCKING: tool.py:1: negative values accepted\n\nVERDICT: FAIL"
    FAST_QA_FAIL = json.dumps({"status": "fail", "checks": [
        {"name": "runs", "command": "python tool.py", "status": "pass", "evidence": "exit 0"}],
        "findings": ["tool.py: prints the wrong greeting"], "not_run": []})
    answers = [
        FAST_PLAN,
        render({"tool.py": "print('built')\n"}),
        FAST_QA_FAIL,
        VERIFICATION_FAIL,
        "## Verifier findings\nnegative values accepted → rebutted: false positive\n\n" +
        render({"tool.py": "print('built v2')\n"}),
        FAST_QA,
        VERIFICATION_FAIL,
    ]
    mode = DesignFastMode()
    state = mode.state_dict()
    state["knobs"]["repair_attempts"]["value"] = 2
    mode.load_state_dict(state)
    model = FakeModel(answers)
    result = mode.run(
        tmp_path / "run", model=model,
        task={"name": "advisory_fast", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error
    # design_fast's builder works in a workspace: it answers in its final message.
    emits = [prompt for prompt in model.prompts if "Current stage: make the change" in prompt]
    assert "answer in your final message, which is saved as verifier-response.md" in emits[1]
    sources_path = tmp_path / "run" / "reviews" / "finding-sources.json"
    assert sources_path.exists(), "finding-sources.json must be written"
    sources = json.loads(sources_path.read_text())
    verifier_entries = [e for e in sources if e["source"] == "verifier"
                        and "negative values accepted" in e["finding"]]
    assert len(verifier_entries) >= 1, "verifier finding must be in finding-sources.json"
    assert verifier_entries[0]["fate"] == "rebutted", (
        f"verifier finding fate should be 'rebutted', got {verifier_entries[0]['fate']!r}")


def test_a_repository_change_gives_the_verifier_only_the_unshared_methods(tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    import sys
    from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER,
                               RESEARCH_ANSWER, VERIFICATION_ANSWER, FakeModel, git_repo,
                               render)
    model = FakeModel([RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
                       DESIGN_DOC_ANSWER,
                       render({"tool.py": "def build():\n    print('built')\n"}),
                       "health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER,
                       "VERDICT: PASS"])
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    mode.load_state_dict(state)
    result = mode.run(tmp_path / "run", model=model, verdicts={"gate_strategy": "proceed"},
                      task={"name": "repo_verify", "brief": "a tool",
                            "check": f"{sys.executable} tool.py",
                            "target": str(git_repo(tmp_path / "repo"))})
    assert result.exit_ok, result.error
    verifier = next(p for p in model.prompts if "design.md is in the run directory" in p)
    assert "M15 Weakened tests" in verifier
    assert "M12 Shared state" not in verifier
    assert "Aim to finish within about seven minutes" in verifier
