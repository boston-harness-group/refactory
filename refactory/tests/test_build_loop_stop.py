"""Build-loop stop signal, refusal bound, and failure text (dm-202, dm-203).

Tests for ``BuildLoopStop``, ``format_failure_detail``, the refusal counter,
the walker's ``_try_visit`` bypass, ``main()``'s exit code 3, and the
cross-process refusal bound.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from refactory_dsh.run import (
    BuildLoopStop,
    FAILURE_MARKER,
    format_failure_detail,
    main,
    run_node,
)


# ── BuildLoopStop is a ValueError ─────────────────────────────────────────

#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def test_build_loop_stop_is_a_value_error():
    assert issubclass(BuildLoopStop, ValueError)
    assert isinstance(BuildLoopStop("x"), ValueError)


# ── format_failure_detail ──────────────────────────────────────────────────

def test_format_failure_detail_short_message_comes_back_whole():
    """A short multi-line message comes back whole (correction #3)."""
    exc = BuildLoopStop("check exhausted\nsome detail")
    result = format_failure_detail(exc)
    assert result == "BuildLoopStop: check exhausted\nsome detail"


def test_format_failure_detail_1200_char_message_no_repeat():
    """A 1200-char message does not repeat its first line (correction #3)."""
    first_line = "authoring unresolved after 3 attempt(s)"
    rest = "x" * (1200 - len(first_line) - 1)
    exc = BuildLoopStop(f"{first_line}\n{rest}")
    result = format_failure_detail(exc)
    assert result.startswith("BuildLoopStop: authoring unresolved")
    # Count how many times the reason line appears — must be exactly once
    assert result.count("authoring unresolved after 3 attempt(s)") == 1
    assert len(result) <= 2000


def test_format_failure_detail_reason_first():
    """The 20 KB fixture: reason first, FAILED test id present, total ≤ 2000."""
    # Build a 20 KB check output with one FAILED line in the middle and a vite warning at the end
    preamble = "some check output\n" * 200  # ~3600 chars
    failed_line = "FAILED refactory/tests/test_create.py::test_a_review_that_reloops_the_build_says_why_into_the_rebuild\n"
    vite_warning = "vite v5.2.0 building for production...\n" * 100  # ~3800 chars
    filler = "x" * (20000 - len(preamble) - len(failed_line) - len(vite_warning))
    check_output = preamble + failed_line + filler + vite_warning

    exc = BuildLoopStop(f"authoring unresolved after 3 attempt(s): {check_output}")
    result = format_failure_detail(exc)

    assert result.startswith("BuildLoopStop: authoring unresolved")
    assert "test_a_review_that_reloops_the_build_says_why_into_the_rebuild" in result
    assert len(result) <= 2000
    # The vite warning should not appear before the reason
    reason_pos = result.index("authoring unresolved")
    assert reason_pos == len("BuildLoopStop: ")


# ── main() exit codes ─────────────────────────────────────────────────────

def test_main_exits_3_for_build_loop_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "test_node", "type": "op"}))
    (tmp_path / "facts.json").write_text("{}")

    def _raise_stop(ctx):
        raise BuildLoopStop("emissions refused 3 consecutive times: edit text not found")

    with mock.patch("refactory_dsh.run.resolve", return_value=_raise_stop):
        import io
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stderr", captured)
        code = main(["test_module:_raise_stop", "--node", "test_node"])

    assert code == 3
    stderr_text = captured.getvalue()
    assert FAILURE_MARKER in stderr_text
    assert "emissions refused 3 consecutive times" in stderr_text


def test_main_exits_1_for_other_exceptions(tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "test_node", "type": "op"}))
    (tmp_path / "facts.json").write_text("{}")

    def _raise_error(ctx):
        raise RuntimeError("something went wrong: details")

    with mock.patch("refactory_dsh.run.resolve", return_value=_raise_error):
        import io
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stderr", captured)
        code = main(["test_module:_raise_error", "--node", "test_node"])

    assert code == 1
    stderr_text = captured.getvalue()
    assert FAILURE_MARKER in stderr_text


# ── run_node does not retry BuildLoopStop ──────────────────────────────────

def test_run_node_does_not_retry_build_loop_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "test_node", "type": "op"}))
    (tmp_path / "facts.json").write_text("{}")

    call_count = 0

    def _raise_stop(ctx):
        nonlocal call_count
        call_count += 1
        ctx.set_fact("counter", call_count)
        raise BuildLoopStop("limit reached")

    with mock.patch("refactory_dsh.run.resolve", return_value=_raise_stop):
        with pytest.raises(BuildLoopStop, match="limit reached"):
            run_node("test_module:_raise_stop", workdir=tmp_path, node_id="test_node")

    # The op should be called exactly once (no retry)
    assert call_count == 1
    # Facts should be written to disk
    facts = json.loads((tmp_path / "facts.json").read_text())
    assert facts["counter"] == 1


# ── walker _try_visit bypass ──────────────────────────────────────────────

def test_walker_stop_reaches_run_result_error(tmp_path):
    """BuildLoopStop propagates through the walker's Executor.run() with its reason."""
    from refactory_dsh.nn import RunContext, Node, Graph
    from refactory_dsh.nn.executor import Executor

    # Build a minimal graph with one op that raises BuildLoopStop
    graph = Graph()
    graph.add(Node("failing_op", "op", call="test_module:fail"))

    ctx = RunContext(tmp_path, facts={"name": "test"}, knobs={})

    with mock.patch("refactory_dsh.nn.executor.Executor._visit") as mock_visit:
        # First call raises, second call (retry) should NOT happen
        mock_visit.side_effect = BuildLoopStop("test stop reason")
        executor = Executor(graph, ctx, enable_recovery=True)
        result = executor.run()

    assert result.exit_ok is False
    assert "BuildLoopStop" in (result.error or "")
    assert "test stop reason" in (result.error or "")
    # _visit called only once — the bypass prevents retry
    assert mock_visit.call_count == 1


# ── consecutive refusals end the run (walker) ──────────────────────────────

def test_consecutive_refusals_end_the_build_loop(tmp_path):
    """A scripted walker run whose emissions keep changing nothing ends after refusal_attempts.
    Its first attempt applied, so it ends as a draft of that attempt (b45a), not a failure."""
    from refactory_dsh.design import DesignMode
    from tests_support import FakeModel, render, RESEARCH_ANSWER, PITFALLS_ANSWER
    from tests_support import DESIGN_DOC_ANSWER, REVIEW_ANSWERS, VERIFICATION_ANSWER
    from tests_support import git_repo

    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"

    # An emission that is identical to itself (will be refused as "changed nothing"
    # on the second and subsequent materialise calls because the worktree already
    # has these files from the first attempt).
    emission = render({"tool.py": "print('built')\n"})

    # We need enough answers: research, pitfalls, perspectives, plan, design doc,
    # design review, the first attempt and its reviews, an overwatch that sends it back,
    # then identical emissions for the refusal loop.
    from tests_support import overwatch_judgment

    redo = overwatch_judgment("redo", must_fix=[{"what": "print more"}])
    answers = [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER,
        emission, *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, redo,
        emission, emission, emission, emission,
    ]
    model = FakeModel(answers)
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["refusal_attempts"]["value"] = 3
    mode.load_state_dict(state)
    result = mode.run(
        run, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": "refusal_test", "brief": "a tool", "check": "python tool.py",
              "target": str(repo)},
    )

    assert result.exit_ok, result.error
    reason = result.facts["draft_delivery"]["reason"]
    assert reason.startswith("emissions refused 3 consecutive times"), reason
    # The bound, not the gates' iteration limits, ended it: the first emission and three refused.
    assert sum(1 for prompt in model.prompts if "Write the artifact `" in prompt) == 4


def test_interleaved_refusals_and_successes_converge(tmp_path):
    """Refusals interleaved with applied emissions converge: the counter resets. Attempt 1's
    check fails, its repeat is refused, and attempt 2 passes: the run is delivered whole,
    no draft, with the refusal count back at 0."""
    from refactory_dsh.design import DesignMode
    from tests_support import FakeModel, render, RESEARCH_ANSWER, PITFALLS_ANSWER
    from tests_support import DESIGN_DOC_ANSWER, REVIEW_ANSWERS, VERIFICATION_ANSWER
    from tests_support import git_repo

    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"

    # First emission: applied, its check fails. Second: identical (refused). Third: different
    # (applied, passes check).
    emit1 = render({"tool.py": "raise SystemExit(1)\n"})
    emit2 = render({"tool.py": "raise SystemExit(1)\n"})  # same as emit1 → refused
    emit3 = render({"tool.py": "print(1)\n"})  # different → applied, passes check "python tool.py"

    answers = [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER,
        emit1, emit2, emit3,
        *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3],
    ]
    model = FakeModel(answers)
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["refusal_attempts"]["value"] = 3
    mode.load_state_dict(state)
    result = mode.run(
        run, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": "interleave_test", "brief": "a tool", "check": "python tool.py",
              "target": str(repo)},
    )

    assert result.exit_ok, result.error
    assert "draft_delivery" not in result.facts
    assert not result.facts.get("consecutive_refusals")
    assert sum(1 for prompt in model.prompts if "Write the artifact `" in prompt) == 3
    assert "materialise could not apply emit.txt" in model.prompts[
        [i for i, prompt in enumerate(model.prompts) if "Write the artifact `" in prompt][2]]


# ── cross-process refusal bound ──────────────────────────────────────────

def test_refusal_bound_holds_across_node_processes(tmp_path, monkeypatch):
    """Run materialise three times through node_runner, each sharing one facts file.

    The third invocation exits 3 with the failure marker and the refusal reason.
    Removing consecutive_refusals from produces makes the test fail.
    """
    from refactory_dsh.design import DesignMode
    from refactory_dsh import lowering, worktree, repo_change
    from refactory_dsh.run import main as runner_main
    from tests_support import git_repo

    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"

    # Create worktree and write declarations
    worktree.create(repo, "HEAD", "abcdef999999", run)
    lowering.write_declarations(DesignMode().state_dict(), run)
    (run / "facts.json").write_text(json.dumps({
        "name": "refusal_test", "brief": "a tool", "check": "true",
    }))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({"check_attempts": 3, "refusal_attempts": 3}))

    # An edit whose search text is not in the file → AttemptRefused
    bad_emission = "===EDIT: app.py===\n<<<<<<< SEARCH\ntext not in file\n=======\nreplacement\n>>>>>>> REPLACE\n"

    import io

    for i in range(3):
        (run / "emit.txt").write_text(bad_emission)
        monkeypatch.setenv("REFACTORY_WORKDIR", str(run))
        monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "materialise", "type": "op"}))

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            code = runner_main([f"{repo_change.OPS}:materialise", "--node", "materialise"])
        finally:
            sys.stderr = old_stderr

        facts = json.loads((run / "facts.json").read_text())

        if i < 2:
            # First two: the refusal is recorded, exit 0 (refusal returned, not raised)
            assert code == 0, f"invocation {i+1}: expected exit 0, got {code}"
            assert facts.get("consecutive_refusals") == i + 1, (
                f"invocation {i+1}: expected consecutive_refusals={i+1}, got {facts.get('consecutive_refusals')}"
            )
        else:
            # Third: BuildLoopStop raised, exit 3, marker present
            assert code == 3, f"invocation 3: expected exit 3, got {code}. stderr={captured.getvalue()}"
            stderr_text = captured.getvalue()
            assert FAILURE_MARKER in stderr_text
            assert "emissions refused 3 consecutive times" in stderr_text


# ── snapshot guard: a removed produces breaks the cross-process test ──────

def test_removing_consecutive_refusals_from_produces_fails(tmp_path, monkeypatch):
    """Removing consecutive_refusals from the materialise node's produces must
    make the fact unavailable in a subsequent invocation."""
    from refactory_dsh.design import DesignMode
    from refactory_dsh import lowering, worktree, repo_change
    from refactory_dsh.run import main as runner_main
    from tests_support import git_repo

    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"

    worktree.create(repo, "HEAD", "abcdef888888", run)
    lowering.write_declarations(DesignMode().state_dict(), run)

    # Remove consecutive_refusals from declarations
    declarations = json.loads((run / lowering.DECLARATIONS).read_text())
    produces = declarations["materialise"]["produces"]
    if "consecutive_refusals" in produces:
        produces.remove("consecutive_refusals")
    (run / lowering.DECLARATIONS).write_text(json.dumps(declarations))

    (run / "facts.json").write_text(json.dumps({
        "name": "produces_test", "brief": "a tool", "check": "true",
    }))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({"check_attempts": 3, "refusal_attempts": 3}))

    bad_emission = "===EDIT: app.py===\n<<<<<<< SEARCH\ntext not in file\n=======\nreplacement\n>>>>>>> REPLACE\n"

    import io

    # First invocation — will try to set consecutive_refusals but it is not in produces
    (run / "emit.txt").write_text(bad_emission)
    monkeypatch.setenv("REFACTORY_WORKDIR", str(run))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "materialise", "type": "op"}))

    captured = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured
    try:
        code = runner_main([f"{repo_change.OPS}:materialise", "--node", "materialise"])
    finally:
        sys.stderr = old_stderr

    # The _check_produces should fail because consecutive_refusals is not declared
    # This manifests as exit 1 (general failure) rather than exit 0
    assert code == 1, (
        f"expected exit 1 (produces check failure), got {code}. stderr={captured.getvalue()}"
    )


# ── review of #496: the stop holds on every path, the count resets and survives ──

def _design_mode(**knobs):
    from refactory_dsh.design import DesignMode

    mode = DesignMode()
    state = mode.state_dict()
    for key, value in knobs.items():
        state["knobs"][key]["value"] = value
    mode.load_state_dict(state)
    return mode


def _run_node_main(run, target, node_id, node_type, monkeypatch):
    import io

    monkeypatch.setenv("REFACTORY_WORKDIR", str(run))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": node_id, "type": node_type}))
    err, out = io.StringIO(), io.StringIO()
    old_err, old_out = sys.stderr, sys.stdout
    sys.stderr, sys.stdout = err, out
    try:
        code = main([target, "--node", node_id])
    finally:
        sys.stderr, sys.stdout = old_err, old_out
    return code, err.getvalue()


def test_a_design_check_limit_ends_the_walker_run_in_a_check_failing_draft(tmp_path):
    """Design mode runs its check beside the reviews; at the check limit the build loop
    stops, and the applied attempt is delivered as a draft marked check failing (b45a)."""
    from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER,
                               RESEARCH_ANSWER, REVIEW_ANSWERS, VERIFICATION_ANSWER, FakeModel,
                               git_repo, render)

    repo = git_repo(tmp_path / "repo")
    emits = [render({"tool.py": f"print({i})\n"}) for i in range(8)]
    answers = [RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
               DESIGN_DOC_ANSWER, *emits,
               *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3]]
    result = _design_mode(check_attempts=2, check_attempt_cap=0).run(
        tmp_path / "run", model=FakeModel(answers), verdicts={"gate_strategy": "proceed"},
        task={"name": "stop_tool", "brief": "a tool", "check": "exit 1", "target": str(repo)})
    assert result.exit_ok, result.error
    fact = result.facts["draft_delivery"]
    assert fact["reason"] == "check attempts exhausted after 2 check failure(s)"
    assert fact["check_failing"] is True


def test_a_design_check_limit_goes_to_the_report_as_a_draft_on_the_spine(tmp_path, monkeypatch):
    """The spine runs materialise, prepare_check, the check beside the reviews and
    gate_check as separate processes; at the check limit gate_check routes ``draft``,
    straight to the report, which accepts the draft (b45a, correction 2)."""
    from refactory_dsh import lowering, worktree
    from tests_support import git_repo, render

    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    worktree.create(repo, "HEAD", "abcdef777777", run)
    lowering.write_declarations(_design_mode(check_attempts=1, check_attempt_cap=0).state_dict(), run)
    (run / "facts.json").write_text(json.dumps({"name": "stop_tool", "brief": "a tool", "check": "exit 1"}))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({"check_attempts": 1, "check_attempt_cap": 0,
                                                      "refusal_attempts": 3}))
    (run / "emit.txt").write_text(render({"tool.py": "print(1)\n"}))
    routes = _walk_on_the_spine(run, monkeypatch, *_ATTEMPT, _REPORT)
    assert routes == [*_TO_THE_CHECK, ("gate_check", 0, "draft"), ("report", 0, "")], routes
    facts = json.loads((run / "facts.json").read_text())
    assert facts["draft_delivery"]["reason"] == "check attempts exhausted after 1 check failure(s)"
    assert facts["draft_delivery"]["check_failing"] is True
    assert json.loads((run / "design.json").read_text())["draft"] is True


#: The nodes of one attempt, as the spine runs design's graph (`DesignMode.graph`): the
#: self-check in prepare_check, then the check beside the reviews in fork_qa.
_ATTEMPT = (("refactory_dsh.design_ops:design_materialise", "materialise", "op"),
            ("refactory_dsh.design_ops:prepare_check", "prepare_check", "gate"),
            ("refactory_dsh.design_ops:check_beside_reviews", "check", "op"),
            ("refactory_dsh.design_ops:gate_check", "gate_check", "gate"))
#: The attempt's routes up to the check beside the reviews.
_TO_THE_CHECK = [("materialise", 0, ""), ("prepare_check", 0, "proceed"), ("check", 0, "")]
_REPORT = ("refactory_dsh.design_ops:report_judged", "report", "op")


def _walk_on_the_spine(run, monkeypatch, *steps) -> list[tuple[str, int, str]]:
    """Run ``steps`` as the spine does, each its own process through `refactory_dsh.run`,
    stopping at the first that fails; each with its exit code and the verdict it printed."""
    import io

    walked = []
    for target, node, kind in steps:
        monkeypatch.setenv("REFACTORY_WORKDIR", str(run))
        monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": node, "type": kind}))
        err, out = io.StringIO(), io.StringIO()
        old_err, old_out = sys.stderr, sys.stdout
        sys.stderr, sys.stdout = err, out
        try:
            code = main([target, "--node", node])
        finally:
            sys.stderr, sys.stdout = old_err, old_out
        lines = [line for line in out.getvalue().splitlines() if line.strip()]
        walked.append((node, code, lines[-1] if lines and kind == "gate" else ""))
        if code != 0:
            break
    return walked


def _spine_attempt(tmp_path, monkeypatch, check: str, **knobs) -> Path:
    """A design run on the spine, its first attempt applied and checked."""
    from refactory_dsh import lowering, worktree
    from tests_support import git_repo, render

    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    worktree.create(repo, "HEAD", "abcdef555555", run)
    lowering.write_declarations(_design_mode(**knobs).state_dict(), run)
    (run / "facts.json").write_text(json.dumps({"name": "draft_tool", "brief": "a tool", "check": check}))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({"check_attempts": 3, "check_attempt_cap": 0,
                                                      "refusal_attempts": 3, **knobs}))
    (run / "emit.txt").write_text(render({"tool.py": "print(1)\n"}))
    return run


def test_the_refusal_bound_after_an_applied_commit_goes_to_the_report_on_the_spine(tmp_path, monkeypatch):
    """Attempt 1 applies and passes its check; the next emissions are refused to the bound.
    materialise then exits 0 with the draft, prepare_check routes ``draft``, and the report
    accepts it: the refusals' rewrite of check.txt is put back from the evidence's history."""
    from refactory_dsh import authoring_evidence as evidence
    import hashlib

    run = _spine_attempt(tmp_path, monkeypatch, "true", refusal_attempts=2)
    assert _walk_on_the_spine(run, monkeypatch, *_ATTEMPT) == [*_TO_THE_CHECK,
                                                               ("gate_check", 0, "proceed")]
    bad = "===EDIT: missing.py===\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"
    (run / "emit.txt").write_text(bad)
    assert _walk_on_the_spine(run, monkeypatch, *_ATTEMPT[:2]) == [
        ("materialise", 0, ""), ("prepare_check", 0, "reloop")]
    assert "materialise could not apply emit.txt" in (run / "check.txt").read_text()
    (run / "emit.txt").write_text(bad)
    assert _walk_on_the_spine(run, monkeypatch, *_ATTEMPT[:2], _REPORT) == [
        ("materialise", 0, ""), ("prepare_check", 0, "draft"), ("report", 0, "")]
    facts = json.loads((run / "facts.json").read_text())
    assert facts["draft_delivery"]["reason"].startswith("emissions refused 2 consecutive times")
    assert "check_failing" not in facts["draft_delivery"]
    receipt = json.loads((run / evidence.EVIDENCE).read_text())["receipts"]["check.txt"]
    assert hashlib.sha256((run / "check.txt").read_bytes()).hexdigest() == receipt


def test_a_review_limit_at_gate_qa_goes_on_as_a_draft_on_the_spine(tmp_path, monkeypatch):
    """At the review limit gate_qa routes ``proceed`` with the draft set (on to the audit
    and the overwatch); gate_overwatch then adds its own problem to the draft without
    spending an attempt, and the report accepts it."""
    run = _spine_attempt(tmp_path, monkeypatch, "true", review_attempts=1)
    _walk_on_the_spine(run, monkeypatch, *_ATTEMPT)
    steps = (("refactory_dsh.design_ops:gate_qa", "gate_qa", "gate"),
             ("refactory_dsh.design_ops:gate_overwatch", "gate_overwatch", "gate"), _REPORT)
    assert _walk_on_the_spine(run, monkeypatch, *steps) == [
        ("gate_qa", 0, "proceed"), ("gate_overwatch", 0, "proceed"), ("report", 0, "")]
    facts = json.loads((run / "facts.json").read_text())
    assert facts["draft_delivery"]["reason"] == "review reloop limit reached after 1 review cycle(s)"
    # No review signed off: gate_qa names each, and so does gate_overwatch, all on attempt 1.
    unresolved = facts["draft_delivery"]["unresolved"]
    assert sorted({item["source"] for item in unresolved}) == ["QA gate", "overwatch gate"]
    assert all(item["rounds"] == [1] for item in unresolved), unresolved
    assert any("health_checker has no current delivery" in item["text"] for item in unresolved)
    assert facts["review_attempts_used"] == 1 and facts["attempts_used"] == 1
    assert json.loads((run / "design.json").read_text())["draft"] is True


def test_a_review_limit_at_gate_overwatch_goes_to_the_report_as_a_draft_on_the_spine(tmp_path, monkeypatch):
    run = _spine_attempt(tmp_path, monkeypatch, "true", review_attempts=1)
    _walk_on_the_spine(run, monkeypatch, *_ATTEMPT)
    steps = (("refactory_dsh.design_ops:gate_overwatch", "gate_overwatch", "gate"), _REPORT)
    assert _walk_on_the_spine(run, monkeypatch, *steps) == [
        ("gate_overwatch", 0, "proceed"), ("report", 0, "")]
    facts = json.loads((run / "facts.json").read_text())
    assert facts["draft_delivery"]["reason"] == "review reloop limit reached after 1 review cycle(s)"
    assert [item["source"] for item in facts["draft_delivery"]["unresolved"]] == ["overwatch gate"]


def test_every_limit_raise_in_the_build_loop_is_a_stop():
    """No limit in repair or _repair raises a plain ValueError: each is final on the spine.
    Design mode's gate functions may guard BuildLoopStop with an if/return for draft
    delivery; the structural test allows that pattern alongside raises in create_ops."""
    import inspect

    from refactory_dsh import create_ops, fast_ops

    import ast
    import textwrap

    for fn in (create_ops.repair, fast_ops._repair):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        raised = [node.exc for node in ast.walk(tree) if isinstance(node, ast.Raise) and node.exc]
        assert len(raised) >= 3, fn.__name__
        for exc in raised:
            # Every raise in the build loop's repair is the stop, never another type.
            assert isinstance(exc, ast.Call) and getattr(exc.func, "id", "") == "BuildLoopStop", (
                fn.__name__, ast.unparse(exc)[:80])


def test_an_applied_emission_resets_the_refusal_count_across_processes(tmp_path, monkeypatch):
    """refuse, refuse, apply, refuse, refuse: the applied emission resets the count,
    so the run never stops, and the count is 2 at the end."""
    from refactory_dsh import lowering, repo_change, worktree
    from tests_support import git_repo, render

    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    worktree.create(repo, "HEAD", "abcdef666666", run)
    lowering.write_declarations(_design_mode().state_dict(), run)
    (run / "facts.json").write_text(json.dumps({"name": "reset_test", "brief": "a tool", "check": "true"}))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({"check_attempts": 3, "refusal_attempts": 3}))
    bad = "===EDIT: app.py===\n<<<<<<< SEARCH\ntext not in file\n=======\nreplacement\n>>>>>>> REPLACE\n"
    sequence = [bad, bad, render({"tool.py": "print(1)\n"}), bad, bad]
    counts = []
    for emission in sequence:
        (run / "emit.txt").write_text(emission)
        code, err = _run_node_main(run, f"{repo_change.OPS}:materialise", "materialise", "op", monkeypatch)
        assert code == 0, err
        counts.append(json.loads((run / "facts.json").read_text()).get("consecutive_refusals"))
    assert counts == [1, 2, 0, 1, 2], counts


def test_the_refusal_bound_follows_its_knob(tmp_path):
    from refactory_dsh import create_ops, nn

    ctx = nn.RunContext(tmp_path, facts={"name": "knob_pkg",
                                         "package_root": str(tmp_path / "stage/.factory/packages/knob_pkg")},
                        knobs={"refusal_attempts": 2})
    ctx.enforce_writes = False
    bad = "===EDIT: mode.py===\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"
    ctx.write_text("emit.txt", bad)
    assert create_ops.materialise(ctx).get("emission_refused")
    ctx.write_text("emit.txt", bad)
    with pytest.raises(BuildLoopStop, match="emissions refused 2 consecutive times"):
        create_ops.materialise(ctx)


def test_create_materialise_stops_at_the_bound(tmp_path):
    """The create side of the shared refusal helper counts and stops too."""
    from refactory_dsh import create_ops, nn

    ctx = nn.RunContext(tmp_path, facts={"name": "probe_pkg",
                                         "package_root": str(tmp_path / "stage/.factory/packages/probe_pkg")},
                        knobs={"refusal_attempts": 3})
    ctx.enforce_writes = False
    bad = "===EDIT: mode.py===\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"
    counts = []
    for _ in range(2):
        ctx.write_text("emit.txt", bad)
        assert create_ops.materialise(ctx).get("emission_refused")
        counts.append(ctx.facts.get("consecutive_refusals"))
    assert counts == [1, 2]
    ctx.write_text("emit.txt", bad)
    with pytest.raises(BuildLoopStop, match="emissions refused 3 consecutive times"):
        create_ops.materialise(ctx)


def test_a_run_lowered_before_the_refusal_count_still_materialises(tmp_path):
    from refactory_dsh import lowering, repo_change, worktree
    from tests_support import git_repo, render

    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    worktree.create(repo, "HEAD", "abcdef445566", run)
    lowering.write_declarations(_design_mode().state_dict(), run)
    declarations = json.loads((run / lowering.DECLARATIONS).read_text())
    declarations["materialise"]["produces"].remove("consecutive_refusals")
    (run / lowering.DECLARATIONS).write_text(json.dumps(declarations))
    (run / "facts.json").write_text(json.dumps({"name": "old_tool", "check": "python tool.py"}))
    (run / "emit.txt").write_text(render({"tool.py": "print(1)\n"}))
    run_node(f"{repo_change.OPS}:materialise", workdir=run, node_id="materialise")
    assert json.loads((run / "facts.json").read_text())["files"] == ["tool.py"]


def test_a_ceo_redo_keeps_the_refusal_count(tmp_path):
    """A CEO redo cannot reset a limit count; the refusal count is one."""
    from refactory_dsh import items, lowering

    run = tmp_path / "run"
    run.mkdir()
    lowering.write_declarations(_design_mode().state_dict(), run)
    (run / "facts.json").write_text(json.dumps({"consecutive_refusals": 2, "attempts_used": 1,
                                                "check_attempts_used": 1}))
    items.command_reset(run, {"nodes": ["emit", "materialise"], "start": "emit",
                              "preserveAttemptFacts": True})
    assert json.loads((run / "facts.json").read_text()).get("consecutive_refusals") == 2


def test_a_short_message_with_a_test_id_comes_back_exactly_whole():
    """The whole-message branch, not the reason-plus-ids fallback, answers a short message."""
    exc = BuildLoopStop("check failed\nFAILED tests/test_x.py::test_y - boom")
    assert format_failure_detail(exc) == "BuildLoopStop: check failed\nFAILED tests/test_x.py::test_y - boom"


def test_an_applied_package_resets_the_create_refusal_count(tmp_path, monkeypatch):
    """Create's materialise resets the count on an applied package: refuse, refuse,
    apply, refuse, refuse never stops."""
    from refactory_dsh import create_ops, nn

    monkeypatch.setattr(create_ops.registry, "install_workspace", lambda *a, **k: None)
    ctx = nn.RunContext(tmp_path, facts={"name": "reset_pkg",
                                         "package_root": str(tmp_path / "stage/.factory/packages/reset_pkg")},
                        knobs={"refusal_attempts": 3})
    ctx.enforce_writes = False
    bad = "===EDIT: mode.py===\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"
    from tests_support import render

    good = render({"mode.py": "VALUE = 1\n"})
    counts = []
    for emission in (bad, bad, good, bad, bad):
        ctx.write_text("emit.txt", emission)
        # The executor writes an op's returned facts back; do the same here.
        for key, value in create_ops.materialise(ctx).items():
            ctx.set_fact(key, value)
        counts.append(ctx.facts.get("consecutive_refusals"))
    assert counts == [1, 2, 0, 1, 2], counts


@pytest.mark.parametrize("mode_name", ["create", "edit"])
def test_create_and_edit_still_stop_at_their_review_limit_with_an_applied_commit(mode_name, tmp_path,
                                                                                 monkeypatch):
    """Correction 1: the draft route is design mode's own. create's and edit's gate_qa
    (`create_ops.gate_qa`, `create_ops.repair`) still end the run at the review limit, an
    applied commit or not: exit 3 with the marker, and no draft."""
    from refactory_dsh import lowering
    from refactory_dsh.create import CreateMode
    from refactory_dsh.edit import EditMode

    run = tmp_path / "run"
    run.mkdir()
    mode = {"create": CreateMode, "edit": EditMode}[mode_name]()
    lowering.write_declarations(mode.state_dict(), run)
    (run / "facts.json").write_text(json.dumps({"name": "pkg", "brief": "a mode", "commit": "a" * 40}))
    (run / "reviews").mkdir()
    (run / "reviews/qa-synthesized.md").write_text("# QA synthesis\n")
    # Check evidence, as an applied and checked attempt leaves: create's gate spends a
    # review attempt only then.
    (run / "authoring-evidence.json").write_text(json.dumps(
        {"identity": {"root": str(run)}, "validation": {"status": "pass"}}))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({"review_attempts": 1, "check_attempts": 3}))
    code, err = _run_node_main(run, "refactory_dsh.create_ops:gate_qa", "gate_qa", "gate", monkeypatch)
    assert code == 3, err
    assert FAILURE_MARKER in err and "review reloop limit reached after 1 review cycle(s)" in err
    assert "draft_delivery" not in json.loads((run / "facts.json").read_text())


def test_a_gate_check_that_runs_the_check_itself_drafts_at_its_limit_on_the_spine(tmp_path, monkeypatch):
    """When no decision stands for the attempt (the check command was corrected after the
    check ran), gate_check runs the whole check itself; at the limit it routes ``draft``."""
    run = _spine_attempt(tmp_path, monkeypatch, "exit 1", check_attempts=1)
    assert _walk_on_the_spine(run, monkeypatch, *_ATTEMPT[:-1]) == _TO_THE_CHECK
    facts = json.loads((run / "facts.json").read_text())
    (run / "facts.json").write_text(json.dumps({**facts, "check": "exit 2"}))
    assert _walk_on_the_spine(run, monkeypatch, _ATTEMPT[-1], _REPORT) == [
        ("gate_check", 0, "draft"), ("report", 0, "")]
    fact = json.loads((run / "facts.json").read_text())["draft_delivery"]
    assert fact["reason"] == "check attempts exhausted after 2 check failure(s)" and fact["check_failing"]


def test_a_stop_whose_commit_has_no_current_evidence_is_not_a_draft(tmp_path, monkeypatch):
    """A draft says what its check showed (`report_judged` requires current evidence), so a
    stop whose applied commit's evidence is stale stays a stop."""
    from refactory_dsh import design_ops, nn

    run = _spine_attempt(tmp_path, monkeypatch, "true")
    _walk_on_the_spine(run, monkeypatch, *_ATTEMPT)
    knobs = json.loads(os.environ["REFACTORY_KNOBS"])

    def context():
        return nn.RunContext(run, facts=json.loads((run / "facts.json").read_text()), knobs=knobs,
                             enforce_writes=False)

    assert design_ops._draft_for_stop(context(), BuildLoopStop("limit")) is not None
    (run / "design.md").write_text("# A design approved after the check ran\n")
    assert design_ops._draft_for_stop(context(), BuildLoopStop("limit")) is None


def test_the_overall_limit_at_gate_qa_goes_on_as_a_draft_on_the_spine(tmp_path, monkeypatch):
    run = _spine_attempt(tmp_path, monkeypatch, "true", check_attempts=1, review_attempts=3)
    _walk_on_the_spine(run, monkeypatch, *_ATTEMPT)
    steps = (("refactory_dsh.design_ops:gate_qa", "gate_qa", "gate"),
             ("refactory_dsh.design_ops:gate_overwatch", "gate_overwatch", "gate"), _REPORT)
    assert _walk_on_the_spine(run, monkeypatch, *steps) == [
        ("gate_qa", 0, "proceed"), ("gate_overwatch", 0, "proceed"), ("report", 0, "")]
    facts = json.loads((run / "facts.json").read_text())
    assert facts["draft_delivery"]["reason"] == "authoring unresolved after 1 attempt(s)"
    assert "check_failing" not in facts["draft_delivery"]


def test_the_refusal_bound_after_a_failing_check_is_a_check_failing_draft_on_the_spine(tmp_path, monkeypatch):
    run = _spine_attempt(tmp_path, monkeypatch, "exit 1", check_attempts=3, refusal_attempts=2)
    assert _walk_on_the_spine(run, monkeypatch, *_ATTEMPT)[-1] == ("gate_check", 0, "reloop")
    bad = "===EDIT: missing.py===\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"
    (run / "emit.txt").write_text(bad)
    assert _walk_on_the_spine(run, monkeypatch, *_ATTEMPT[:2])[-1] == ("prepare_check", 0, "reloop")
    (run / "emit.txt").write_text(bad)
    assert _walk_on_the_spine(run, monkeypatch, *_ATTEMPT[:2], _REPORT) == [
        ("materialise", 0, ""), ("prepare_check", 0, "draft"), ("report", 0, "")]
    fact = json.loads((run / "facts.json").read_text())["draft_delivery"]
    assert fact["check_failing"] is True and fact["reason"].startswith("emissions refused 2")


def test_a_redo_of_the_overwatch_keeps_the_qa_judges_draft(tmp_path, monkeypatch):
    """gate_qa's limit made the run a draft; a CEO redo of the overwatch resets gate_overwatch's
    facts, draft_delivery among them. gate_qa's stop is its own fact, so the redone
    gate_overwatch folds it back in, spending no attempt, and the report takes the draft."""
    from refactory_dsh import items

    run = _spine_attempt(tmp_path, monkeypatch, "true", review_attempts=1)
    _walk_on_the_spine(run, monkeypatch, *_ATTEMPT)
    gates = (("refactory_dsh.design_ops:gate_qa", "gate_qa", "gate"),
             ("refactory_dsh.design_ops:gate_overwatch", "gate_overwatch", "gate"))
    _walk_on_the_spine(run, monkeypatch, *gates)
    assert json.loads((run / "facts.json").read_text())["draft_delivery"]
    items.command_reset(run, {"nodes": ["overwatch", "gate_overwatch", "report", "pick_corrections",
                                        "deliver", "delivery_gate", "done"],
                              "start": "overwatch", "preserveAttemptFacts": True})
    facts = json.loads((run / "facts.json").read_text())
    assert "draft_delivery" not in facts and facts["qa_draft"]
    assert _walk_on_the_spine(run, monkeypatch, gates[1], _REPORT) == [
        ("gate_overwatch", 0, "proceed"), ("report", 0, "")]
    facts = json.loads((run / "facts.json").read_text())
    assert facts["draft_delivery"]["reason"] == "review reloop limit reached after 1 review cycle(s)"
    assert facts["review_attempts_used"] == 1
    assert "QA gate" in {item["source"] for item in facts["draft_delivery"]["unresolved"]}
    assert json.loads((run / "design.json").read_text())["draft"] is True
