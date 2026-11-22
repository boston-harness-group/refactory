"""Design mode runs its check beside the reviews, not before them.

`prepare_check` settles the commit the check and the reviewers both see; the check runs
as a branch of the review fork (`check_beside_reviews`) and records its decision;
`gate_check` routes on it once the reviews are done, because a node inside a fork may not
send the run outside it. A review that starts before the check records is bound to the
commit and to that check, and each node on the spine writes back only the facts it
changed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from refactory_dsh import authoring_evidence as evidence, design_ops, lowering, nn
from refactory_dsh.design import DesignMode
from refactory_dsh.invocations import CHECK_PREP, CHECK_VERDICT
from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER, RESEARCH_ANSWER,
                           SCOPED_RESEARCH_ANSWER,
                           VERIFICATION_ANSWER, FakeModel, git_repo, render)

BESIDE = "## The check runs beside this review"
RECEIPT_CONTEXT = "## Authoritative validation and intent baseline"
HEALTH = "You are the health checker."


def _edges(graph):
    return {(e.source, e.target, e.condition) for e in graph.edges}


def test_the_check_is_a_branch_of_the_review_fork():
    graph = DesignMode().build()
    edges = _edges(graph)
    assert ("materialise", "prepare_check", None) in edges
    assert ("prepare_check", "pick_corrections", "reloop") in edges
    assert ("prepare_check", "fork_qa", "proceed") in edges
    assert "check" in graph.nodes["fork_qa"].targets
    assert "check" in graph.nodes["join_qa"].sources
    assert "early_asks_check" in graph.nodes["join_qa"].sources
    assert "verifier" in graph.nodes["join_qa"].sources
    # The check sends the run nowhere: like the reviews, its branch drains into the join.
    assert not [e for e in edges if e[0] == "check"]
    assert ("join_qa", "gate_check", None) in edges
    assert ("gate_check", "pick_corrections", "reloop") in edges
    # On to the synthesis, through the rerun of the verifier's probe files.
    assert ("gate_check", "rerun_probes", "proceed") in edges
    assert ("rerun_probes", "synthesize_qa", None) in edges
    assert graph.nodes["check"].kind == "op"
    assert graph.nodes["check"].call.endswith("design_ops:check_beside_reviews")


def test_the_spine_gets_the_same_shape():
    workflow = lowering.to_workflow(DesignMode().state_dict())
    lowered = {(e.source, e.target, e.condition) for e in workflow.edges}
    assert ("join_qa", "gate_check", None) in lowered
    assert ("gate_check", "pick_corrections", "reloop") in lowered
    assert "check" in workflow.nodes["fork_qa"].targets
    assert workflow.nodes["check"].command.endswith(
        "refactory_dsh.design_ops:check_beside_reviews")


def test_reviews_beside_the_check_neither_read_nor_see_what_it_writes():
    graph = DesignMode().build()
    check = graph.nodes["check"]
    for name in ("health_checker", "code_reviewer", "qa_director"):
        node = graph.nodes[name]
        assert evidence.EVIDENCE not in node.reads
        assert BESIDE in node.prompt_template
        assert "{{runtime_evidence}}" not in node.prompt_template
        assert "{{validation}}" not in node.prompt_template
        assert not set(node.reads) & set(check.writes)
    assert not set(graph.nodes["verifier"].reads) & set(check.writes)
    # The audit runs after the check, and still reads its receipt.
    assert evidence.EVIDENCE in graph.nodes["overwatch"].reads
    assert RECEIPT_CONTEXT in graph.nodes["overwatch"].prompt_template


def _run(tmp_path, first_tool, answers_between):
    repo = git_repo(tmp_path / "repo")
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    mode.load_state_dict(state)
    reviews = ["health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER]
    model = FakeModel([SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER,
                       render({"tool.py": first_tool}), *reviews, *answers_between,
                       "VERDICT: PASS"])
    run = mode.run(tmp_path / "run", model=model, verdicts={"gate_strategy": "proceed"},
                   task={"name": "beside_tool", "brief": "Make tool.py print built.",
                         "check": f"{sys.executable} tool.py", "target": str(repo)})
    return run, model


def _signoffs(run_dir: Path) -> dict[str, dict]:
    from refactory_dsh.agent_delivery import receipts_in

    return {name: receipt.get("execution", {}).get("authoring")
            for name, receipt in receipts_in(run_dir).items()}


def test_a_passing_check_and_its_reviews_run_once_and_the_reviews_stay_current(tmp_path):
    run, model = _run(tmp_path, "print('built')\n", [])
    assert run.exit_ok, run.error
    health = [p for p in model.prompts if HEALTH in p]
    assert len(health) == 1
    assert BESIDE in health[0] and RECEIPT_CONTEXT not in health[0]
    # Each review is bound to the commit and to the check it ran beside, which the
    # evidence names; the gate accepted them as current.
    recorded = json.loads((tmp_path / "run" / evidence.EVIDENCE).read_text())
    attempt = recorded["validation"]["check_attempt"]
    assert attempt == json.loads((tmp_path / "run" / CHECK_PREP).read_text())["attempt"]
    signoffs = _signoffs(tmp_path / "run")
    for name in ("health_checker", "code_reviewer", "qa_director"):
        assert signoffs[name]["check_attempt"] == attempt
        assert "receipt" not in signoffs[name]
    assert "receipt" in signoffs["overwatch"]


def test_a_failed_check_sends_the_build_back_with_its_feedback_after_the_reviews(tmp_path):
    broken = "import sys\nprint('BROKEN-MARKER', file=sys.stderr)\nsys.exit(1)\n"
    reviews = ["health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER]
    run, model = _run(tmp_path, broken, [render({"tool.py": "print('built')\n"}), *reviews])
    assert run.exit_ok, run.error
    emits = [p for p in model.prompts if "Write the artifact `beside_tool`" in p]
    assert len(emits) == 2
    assert "BROKEN-MARKER" in emits[1]
    # Both attempts were reviewed beside their check; only the second reached gate_qa.
    assert len([p for p in model.prompts if HEALTH in p]) == 2
    assert json.loads((tmp_path / "run" / "check-record.json").read_text())["status"] == "pass"


class _QuickCheckDesign(DesignMode):
    """Design mode whose inactivity limit may be 1 s, so a silent check stops at once."""

    def declare(self):
        from refactory_dsh.nn import Knob

        knobs = super().declare()
        knobs["check_timeout_s"] = Knob(1, "int", lo=1, hi=1800,
                                        description=knobs["check_timeout_s"].description)
        return knobs


#: A check that prints one line and then goes silent: stopped at the inactivity limit.
SILENT_TOOL = "import time\nprint('STARTED-MARKER', flush=True)\ntime.sleep(30)\n"


def _assert_silent_check_draft(run_dir: Path, facts: dict) -> None:
    draft = facts["draft_delivery"]
    assert draft["check_failing"] is True
    assert "did not finish" in draft["reason"] and "inactivity" in draft["reason"]
    assert "STARTED-MARKER" in draft["check_failure"]
    assert "STARTED-MARKER" in (run_dir / "timeout-output.txt").read_text()
    assert json.loads((run_dir / "check-record.json").read_text())["status"] == "check_timeout"


def test_a_check_beside_the_reviews_stopped_for_inactivity_ends_in_a_draft(tmp_path):
    """The walker, on design mode's real graph: the check in the review fork goes silent,
    is stopped, and gate_check delivers the attempt as a draft marked check failing."""
    repo = git_repo(tmp_path / "repo")
    mode = _QuickCheckDesign()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    mode.load_state_dict(state)
    reviews = ["health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER]
    model = FakeModel([SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER,
                       render({"tool.py": SILENT_TOOL}), *reviews, *[""] * 10])
    run = mode.run(tmp_path / "run", model=model, verdicts={"gate_strategy": "proceed"},
                   task={"name": "silent_tool", "brief": "Make tool.py print built.",
                         "check": f"{sys.executable} tool.py", "target": str(repo)})
    assert run.exit_ok, run.error
    _assert_silent_check_draft(tmp_path / "run", run.facts)
    # One attempt: the stop went to the report, not back to the builder.
    assert len([p for p in model.prompts if "Write the artifact `silent_tool`" in p]) == 1


def test_a_check_beside_the_reviews_stopped_for_inactivity_ends_in_a_draft_on_the_spine(
        tmp_path):
    """The same route as the spine runs it: each node through ``python -m refactory_dsh.run``
    in its own process, with the lowered declarations, so an undeclared fact fails."""
    import os
    import subprocess

    from refactory_dsh import repo_change, worktree
    from refactory_dsh.run import NODE_ENV, WORKDIR_ENV, run_node

    repo = git_repo(tmp_path / "repo")
    run_dir = tmp_path / "run"
    worktree.create(repo, "HEAD", "silent0123456", run_dir)
    lowering.write_declarations(DesignMode().state_dict(), run_dir)
    (run_dir / "facts.json").write_text(json.dumps({
        "name": "silent_tool", "brief": "Make tool.py print built.",
        "check": f"{sys.executable} tool.py"}))
    (run_dir / "emit.txt").write_text(render({"tool.py": SILENT_TOOL}))
    run_node(f"{repo_change.OPS}:materialise", workdir=run_dir, node_id="materialise")
    facts = json.loads((run_dir / "facts.json").read_text())
    (run_dir / CHECK_PREP).write_text(json.dumps({
        "attempt": "a1", "commit": facts["commit"], "root": str(facts.get("artifact") or ""),
        "targeted_command": None}))
    env = {**os.environ, WORKDIR_ENV: str(run_dir),
           "REFACTORY_KNOBS": json.dumps({"check_timeout_s": 1})}

    def node(call: str, node_id: str, kind: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "refactory_dsh.run", f"{design_ops.__name__}:{call}",
             "--node", node_id],
            env={**env, NODE_ENV: json.dumps({"id": node_id, "type": kind})},
            capture_output=True, text=True, timeout=60, cwd=str(run_dir), check=False)

    checked = node("check_beside_reviews", "check", "OpNode")
    assert checked.returncode == 0, checked.stderr[-2000:]
    gated = node("gate_check", "gate_check", "GateNode")
    assert gated.returncode == 0, gated.stderr[-2000:]
    assert gated.stdout.strip().splitlines()[-1] == "draft", gated.stdout[-2000:]
    _assert_silent_check_draft(run_dir, json.loads((run_dir / "facts.json").read_text()))


def _ctx(tmp_path, **facts):
    ctx = nn.RunContext(tmp_path, facts={"check": "true", "commit": "c1", "brief": "b",
                                         "artifact": str(tmp_path), **facts})
    return ctx


def _decision(verdict, reason="", *, attempt="a2", command="true"):
    return {"attempt": attempt, "command": command, "verdict": verdict, "reason": reason}


def test_gate_check_routes_on_the_decision_for_this_attempt(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.write_json(CHECK_PREP, {"attempt": "a2", "commit": "c1", "root": str(tmp_path)})
    ctx.write_json(CHECK_VERDICT, _decision("reloop", "tests failed"))
    verdict = design_ops.gate_check(ctx)
    assert verdict == "reloop" and verdict.reason.startswith("tests failed")
    ctx.write_json(CHECK_VERDICT, _decision("proceed"))
    assert design_ops.gate_check(ctx) == "proceed"
    # A person who skipped the check this attempt: its skip receipt is the evidence.
    ctx.write_json(CHECK_VERDICT, _decision("proceed", attempt="a1"))
    ctx.write_json(evidence.EVIDENCE, {"validation": {"status": "skipped", "source": "person",
                                                      "check_attempt": "a2"}})
    assert design_ops.gate_check(ctx) == "proceed"
    # Another attempt's decision, a skip of another attempt, or a non-person skip does
    # not stand: the gate judges the attempt itself, bound to it.
    whole = []
    # Independent routing scenarios on one ctx; the send-back bound has its own test.
    ctx.set_fact("combined_sendback_count", 0)
    monkeypatch.setattr(design_ops, "check_artifact", lambda ctx, *, check_attempt=None:
                        whole.append(check_attempt) or nn.Verdict("reloop", "judged here"))
    for receipt in ({"status": "skipped", "source": "person", "check_attempt": "a1"},
                    {"status": "skipped", "source": "ceo", "check_attempt": "a2"}):
        ctx.write_json(CHECK_VERDICT, _decision("proceed", attempt="a1"))
        ctx.write_json(evidence.EVIDENCE, {"validation": receipt})
        assert design_ops.gate_check(ctx).reason.startswith("judged here")
    assert whole == ["a2", "a2"]
    # What it judged is recorded for this attempt: a rerun of the gate routes on it.
    ctx.set_fact("combined_sendback_count", 0)
    assert design_ops.gate_check(ctx).reason.startswith("judged here")
    assert whole == ["a2", "a2"]
    # The commit moved since prepare_check: nothing here is this attempt's.
    ctx.set_fact("commit", "c9")
    with pytest.raises(ValueError, match="not prepared for the attempt"):
        design_ops.gate_check(ctx)


def test_a_corrected_check_command_is_judged_again_not_routed_on_the_old_decision(
        tmp_path, monkeypatch):
    """graph-control amend-command changes the check and reruns gate_check: the decision
    the old command made, a stop included, must not route the run."""
    ctx = _ctx(tmp_path, check="pytest -q")
    ctx.write_json(CHECK_PREP, {"attempt": "a2", "commit": "c1", "root": str(tmp_path)})
    ctx.write_json(CHECK_VERDICT, {"attempt": "a2", "command": "python -m pytets -q",
                                   "error": "validation unavailable: no module pytets",
                                   "error_type": "ValueError"})
    monkeypatch.setattr(design_ops, "check_artifact",
                        lambda ctx, *, check_attempt=None: "proceed")
    assert design_ops.gate_check(ctx) == "proceed"
    assert ctx.read_json(CHECK_VERDICT)["command"] == "pytest -q"


def test_the_check_in_the_fork_runs_only_on_the_attempt_it_was_prepared_for(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    with pytest.raises(ValueError, match="not prepared for this attempt"):
        design_ops.check_beside_reviews(ctx)
    ctx.write_json(CHECK_PREP, {"attempt": "a1", "commit": "c0", "root": str(tmp_path)})
    with pytest.raises(ValueError, match="not prepared for this attempt"):
        design_ops.check_beside_reviews(ctx)
    ctx.write_json(CHECK_PREP, {"attempt": "a1", "commit": "c1", "root": str(tmp_path),
                                "targeted_command": "pytest t.py::x"})
    seen = {}

    def full_check(ctx, command, timeout, prev, targeted, *, beside_reviews, check_attempt=None):
        seen.update(targeted=targeted, beside_reviews=beside_reviews, attempt=check_attempt)
        return nn.Verdict("reloop", "it failed")

    monkeypatch.setattr(design_ops, "_full_check", full_check)
    assert design_ops.check_beside_reviews(ctx) == {}
    assert seen == {"targeted": "pytest t.py::x", "beside_reviews": True, "attempt": "a1"}
    assert ctx.read_json(CHECK_VERDICT) == _decision("reloop", "it failed", attempt="a1")
    # Decided already for this attempt and command (prepare_check ran the whole check
    # itself): not run again.
    seen.clear()
    ctx.write_json(CHECK_VERDICT, _decision("proceed", attempt="a1"))
    assert design_ops.check_beside_reviews(ctx) == {}
    assert seen == {}
    # A decision the old command made does not stand after the command changed.
    ctx.write_json(CHECK_VERDICT, _decision("proceed", attempt="a1", command="old"))
    design_ops.check_beside_reviews(ctx)
    assert seen["attempt"] == "a1"


def test_a_check_that_stops_the_run_is_recorded_once_and_raised_after_the_reviews(
        tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.write_json(CHECK_PREP, {"attempt": "a1", "commit": "c1", "root": str(tmp_path)})
    calls = []

    def unrepairable(*args, **kwargs):
        calls.append(1)
        raise ValueError("validation unavailable: the base check could not run")

    monkeypatch.setattr(design_ops, "_full_check", unrepairable)
    assert design_ops.check_beside_reviews(ctx) == {}  # the reviews still finish
    assert calls == [1]
    with pytest.raises(ValueError, match="the base check could not run"):
        design_ops.gate_check(ctx)


def test_prepare_check_sends_back_before_the_reviews_when_last_attempts_failures_remain(
        tmp_path, monkeypatch):
    from refactory_dsh import failfast, repo_change, test_selection
    from refactory_dsh.sandbox import SandboxResult

    ctx = _ctx(tmp_path)
    monkeypatch.setattr(repo_change, "prepare_attempt", lambda ctx, timeout: [])
    # Every attempt — first or fix-up — runs only touched tests in prepare_check.
    monkeypatch.setattr(test_selection, "select",
                        lambda *a, **k: test_selection.Selection(tests=["t.py"]))
    failed = SandboxResult(ok=False, returncode=1, elapsed_s=0.1,
                           stdout="FAILED t.py::x - AssertionError\n1 failed in 0.01s\n")
    monkeypatch.setattr(repo_change, "run_check", lambda *a, **kw: failed)
    early_stops = []
    monkeypatch.setattr(design_ops, "_early_stop", lambda ctx, prepass, prev, base: (
        early_stops.append(prepass.failing_ids) or nn.Verdict("reloop", "still failing")))
    # What the last attempt prepared and decided does not survive into this one.
    ctx.write_json(CHECK_PREP, {"attempt": "a0", "commit": "c1", "root": str(tmp_path)})
    ctx.write_json(CHECK_VERDICT, _decision("proceed", attempt="a0"))
    ctx.set_fact("validation", "what went wrong last time")
    # The first attempt now goes to _fixup_self_check (like a fix-up).
    # With an inconclusive self-check (timed out in chunks), it proceeds to fork_qa.
    sent_back = design_ops.prepare_check(ctx)
    # First attempt with no prior record: self-check result is inconclusive → proceed
    assert sent_back == "proceed"
    # A fix-up's self-check runs in prepare_check, on what failed and what the fix-up changed.
    go = failfast.PrepassResult(
        ran=True, early_stop=False, targeted_command="pytest t.py::x", check_mode="full",
        result=None, digest_text="", selection=["t.py::x"], ran_ids=["t.py::x"],
        commands=["pytest t.py::x"])
    monkeypatch.setattr(failfast, "targeted_prepass", lambda *a, **k: go)
    attempts_before = int(ctx.facts.get("attempts_used", 0))
    check_attempts_before = int(ctx.facts.get("check_attempts_used", 0))
    assert design_ops.prepare_check(ctx) == "proceed"
    prepared = ctx.read_json(CHECK_PREP)
    assert prepared["commit"] == "c1" and prepared["root"] == str(tmp_path)
    assert prepared["attempt"]
    # The reviewers judge the attempt against what sent it back.
    assert ctx.facts.get("review_feedback") == "what went wrong last time"
    assert ctx.facts.get("self_check_fixup_count", 0) == 0
    # prepare_check did not spend an attempt: it just prepared and proceeded
    assert int(ctx.facts.get("attempts_used", 0)) == attempts_before
    assert int(ctx.facts.get("check_attempts_used", 0)) == check_attempts_before

    # Now test the send-back path: exhaust the fix-up budget so _early_stop fires.
    # A self-check that exhausts fixups sends back with "reloop" from _early_stop,
    # and attempts_used must not change (no attempt spent in the send-back).
    failing = failfast.PrepassResult(
        ran=True, early_stop=True, targeted_command="pytest t.py::x", check_mode="full",
        result=None, digest_text="t.py::x failed", selection=["t.py::x"],
        ran_ids=["t.py::x"], commands=["pytest t.py::x"],
        failing_ids=["t.py::x"])
    monkeypatch.setattr(failfast, "targeted_prepass", lambda *a, **k: failing)
    # Exhaust fixup budget: set self_check_fixup_count to the allowed limit
    ctx.set_fact("self_check_fixup_count", 2)
    ctx.set_fact("total_self_check_fixups_used", 6)
    attempts_before2 = int(ctx.facts.get("attempts_used", 0))
    early_stops.clear()
    sent_back2 = design_ops.prepare_check(ctx)
    assert str(sent_back2) == "reloop", (
        f"expected reloop when fixup budget exhausted, got {sent_back2}")
    # _early_stop was called with the failing test IDs from the prepass
    assert len(early_stops) == 1, f"_early_stop must be called once, was called {len(early_stops)} times"
    assert "t.py::x" in early_stops[0], (
        "the send-back must name the failing test from the previous round")
    # attempts_used must not have changed during the send-back
    assert int(ctx.facts.get("attempts_used", 0)) == attempts_before2, (
        "prepare_check send-back must not spend an attempt")


def test_a_failed_preparation_is_judged_by_the_whole_check_at_once(tmp_path, monkeypatch):
    from refactory_dsh import repo_change

    ctx = _ctx(tmp_path)
    monkeypatch.setattr(repo_change, "prepare_attempt", lambda ctx, timeout: ["setup failed"])
    calls = []

    def whole(ctx, *, check_attempt=None):
        calls.append(check_attempt)
        return nn.Verdict("reloop", "the setup command failed")

    monkeypatch.setattr(design_ops, "check_artifact", whole)
    assert design_ops.prepare_check(ctx) == "reloop"
    attempt = ctx.read_json(CHECK_PREP)["attempt"]
    assert calls == [attempt]
    assert ctx.read_json(CHECK_VERDICT)["attempt"] == attempt


def _identity_ctx(tmp_path):
    (tmp_path / "art").mkdir()
    (tmp_path / "art" / "a.txt").write_text("42")
    ctx = nn.RunContext(tmp_path, facts={"brief": "b", "artifact": str(tmp_path / "art")})
    ctx.write_json("brief.json", {"brief": "b"})
    from refactory_dsh.steering import Ledger
    Ledger.seed(tmp_path, "b")
    return ctx


def test_a_review_beside_the_check_is_bound_to_that_check_and_its_commit(tmp_path):
    ctx = _identity_ctx(tmp_path)
    root = str(tmp_path / "art")
    ctx.write_json(CHECK_PREP, {"attempt": "a1", "commit": "c1", "root": root})
    beside = evidence.delivery_identity(tmp_path)
    assert set(beside) == {"identity", "check_attempt"} and beside["check_attempt"] == "a1"
    # The check records for that attempt: the gate's view is the full receipt.
    evidence.record(ctx, {"status": "pass", "command": "true", "check_attempt": "a1"})
    now = evidence.delivery_identity(tmp_path)
    assert set(now) == {"receipt", "identity"}
    recorded = ctx.read_json(evidence.EVIDENCE)
    assert evidence._signed_off(beside, now, recorded)
    # Another check, or another commit/intent, leaves it stale.
    assert not evidence._signed_off({**beside, "check_attempt": "a0"}, now, recorded)
    assert not evidence._signed_off({**beside, "identity": "other"}, now, recorded)
    assert not evidence._signed_off({"identity": beside["identity"]}, now, recorded)
    # A sign-off on the receipt itself is current as before.
    assert evidence._signed_off(dict(now), now, recorded)


def test_a_person_skipping_the_check_keeps_the_reviews_beside_it_current(tmp_path):
    ctx = _identity_ctx(tmp_path)
    ctx.set_fact("commit", "c1")
    ctx.write_json("facts.json", ctx.facts)  # record_skip reads the run's facts from disk
    ctx.write_json(CHECK_PREP, {"attempt": "a1", "commit": "c1", "root": str(tmp_path / "art")})
    payload = evidence.record_skip(tmp_path, "check", {"source": "person", "at": "t"})
    assert payload["validation"]["check_attempt"] == "a1"
    # A preparation for another commit is not the attempt the skip belongs to.
    ctx.write_json(CHECK_PREP, {"attempt": "a0", "commit": "c0", "root": str(tmp_path / "art")})
    payload = evidence.record_skip(tmp_path, "check", {"source": "person", "at": "t"})
    assert "check_attempt" not in payload["validation"]


def _sibling_writes_meanwhile(ctx):
    """A node that runs while a sibling branch records a fact."""
    facts = json.loads((ctx.workdir / "facts.json").read_text())
    facts["health_check"] = "written by a sibling review"  # a fact this node read, unchanged
    (ctx.workdir / "facts.json").write_text(json.dumps(facts))
    ctx.facts.pop("stale", None)
    return {"validation": "(the check passed)"}


def test_a_node_writes_back_only_the_facts_it_changed(tmp_path):
    from refactory_dsh.run import FACTS, run_node

    (tmp_path / FACTS).write_text(json.dumps({"brief": "b", "stale": 1,
                                              "health_check": "from the last attempt"}))
    declared = tmp_path / lowering.DECLARATIONS
    declared.parent.mkdir(parents=True, exist_ok=True)
    declared.write_text(json.dumps({"check": {"reads": [], "writes": [],
                                              "produces": ["validation"]}}))
    run_node(f"{__name__}:_sibling_writes_meanwhile", workdir=tmp_path, node_id="check")
    assert json.loads((tmp_path / FACTS).read_text()) == {
        "brief": "b", "health_check": "written by a sibling review",
        "validation": "(the check passed)"}


def test_a_facts_update_waits_for_the_one_in_progress(tmp_path):
    """Two branches finishing together take turns, so neither loses the other's fact."""
    import fcntl
    import threading

    from refactory_dsh.run import FACTS, FACTS_LOCK, update_facts

    (tmp_path / FACTS).write_text(json.dumps({"brief": "b"}))
    lock_path = tmp_path / FACTS_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        writer = threading.Thread(target=update_facts, args=(tmp_path, {"code_review": "ok"}))
        writer.start()
        writer.join(0.3)
        assert writer.is_alive(), "the update must wait for the lock"
        assert "code_review" not in json.loads((tmp_path / FACTS).read_text())
        fcntl.flock(held, fcntl.LOCK_UN)
    writer.join(5)
    assert json.loads((tmp_path / FACTS).read_text()) == {"brief": "b", "code_review": "ok"}


def test_a_prepared_check_runs_its_command_without_preparing_again(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from refactory_dsh import repo_change, sandbox, worktree

    found = SimpleNamespace(path=str(tmp_path), head="c1", target=str(tmp_path))
    monkeypatch.setattr(worktree, "recorded", lambda workdir: found)
    monkeypatch.setattr(worktree, "restore", lambda workdir: (found, []))
    monkeypatch.setattr(worktree, "shared_change", lambda found: None)
    monkeypatch.setattr(worktree, "check_environment", lambda path, home: {"PATH": "/bin"})

    def no_preparation(*args, **kwargs):
        raise AssertionError("a prepared check must not prepare again")

    monkeypatch.setattr(repo_change, "_prepare_in_worktree", no_preparation)
    ran = []
    monkeypatch.setattr(sandbox, "run_command", lambda command, **kw: ran.append(
        (command, kw["workdir"])) or SimpleNamespace(ok=True, cwd_outside=None))
    ctx = nn.RunContext(tmp_path, facts={"commit": "c1"})
    assert repo_change.run_check(ctx, "pytest -q", 30, prepared=True).ok
    assert ran == [("pytest -q", Path(str(tmp_path)))]
    ctx.set_fact("commit", "c2")  # the attempt moved since it was prepared
    with pytest.raises(ValueError, match="prepared for"):
        repo_change.run_check(ctx, "pytest -q", 30, prepared=True)


def _explode(ctx):
    raise RuntimeError("the optional branch crashed")


def _after(ctx):
    return {"after_ran": True}


@pytest.mark.parametrize("serial", [True, False])
def test_a_failed_optional_branch_does_not_stop_the_branches_after_it(tmp_path, serial):
    """The verifier is optional and the check now comes after it in the fork: a crashed
    verifier must not keep the check from running, on the walker's serial path (a custom
    model client) as on its concurrent one."""
    from refactory_dsh.nn import Graph, Node
    from refactory_dsh.nn.executor import execute

    graph = Graph()
    graph.add(Node("fork", "fork", targets=("explode", "after"), optional_targets=("explode",)))
    graph.add(Node("explode", "op", call=f"{__name__}:_explode"))
    graph.add(Node("after", "op", call=f"{__name__}:_after", produces=("after_ran",)))
    graph.add(Node("join", "join", sources=("after",)))
    graph.connect("fork", "join")
    graph.connect("explode", "join")
    graph.connect("after", "join")
    result = execute(graph, tmp_path / "run", model=FakeModel([]) if serial else None)
    assert result.exit_ok, result.error
    assert result.facts["after_ran"] is True


def test_an_unreadable_facts_file_is_refused_not_overwritten(tmp_path):
    from refactory_dsh.run import FACTS, RunnerError, update_facts

    (tmp_path / FACTS).write_text('{"brief": "b", "commit": ')  # cut off mid-write
    with pytest.raises(RunnerError, match="refusing to overwrite"):
        update_facts(tmp_path, {"code_review": "ok"})
    assert (tmp_path / FACTS).read_text() == '{"brief": "b", "commit": '


def test_the_check_beside_the_reviews_runs_in_its_own_copy(tmp_path, monkeypatch):
    """What a reviewer leaves in the run's worktree is not in the check's tree, and the
    check deletes nothing there; the copy follows the attempt to its next commit."""
    from tests_support import git

    from refactory_dsh import repo_change, worktree

    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    target = git_repo(tmp_path / "target")
    (target / "app.py").write_text("print(1)\n")
    git(target, "add", ".")
    git(target, "commit", "-q", "-m", "app")
    run = tmp_path / "run"
    run.mkdir()
    found = worktree.create(target, "HEAD", "abc123", run)
    ctx = nn.RunContext(run, facts={"commit": found.head, "artifact": found.path})
    # A reviewer's scratch file in the run's worktree, beside the check.
    (Path(found.path) / "probe_test.py").write_text("raise SystemExit('a reviewer probe')\n")
    command = "pwd > where.txt; ls > seen.txt; cp where.txt seen.txt \"$HOME\"/"
    result = repo_change.run_check(ctx, command, 60, beside_reviews=True)
    assert result.ok, result.stderr
    from refactory_dsh.invocations import CHECK_COPY_HOME

    home = run.resolve() / CHECK_COPY_HOME  # the copy's own HOME, not the attempt's
    ran_in = Path((home / "where.txt").read_text().strip())
    assert ran_in != Path(found.path).resolve() and ran_in.name.startswith("check-")
    assert "probe_test.py" not in (home / "seen.txt").read_text()
    assert (Path(found.path) / "probe_test.py").exists(), "the check deletes nothing there"
    assert not (Path(found.path) / "where.txt").exists()
    # The next attempt: the copy moves to its commit, without the last check's files.
    git(Path(found.path), "commit", "-q", "--allow-empty", "-m", "attempt 2")
    head = git(Path(found.path), "rev-parse", "HEAD")
    record = json.loads((run / ".refactory" / "worktree.json").read_text())
    (run / ".refactory" / "worktree.json").write_text(json.dumps({**record, "head": head}))
    ctx.set_fact("commit", head)
    assert repo_change.run_check(ctx, "git rev-parse HEAD > \"$HOME\"/head.txt; ls > \"$HOME\"/seen.txt",
                                 60, beside_reviews=True).ok
    assert (home / "head.txt").read_text().strip() == head
    assert "where.txt" not in (home / "seen.txt").read_text()
    repo_change.remove_check_copy(target, "abc123")
    assert not ran_in.exists()


# ── D13: a check copy's preparation that fails ─────────────────────────────────
#
# The check beside the reviews prepares its own copy of the attempt
# (`repo_change._check_copy`). A preparation failure the change did not cause ends the
# build in a draft, with no builder round; one the change caused goes back to the builder.
# The real-shell tests put a stub on the PATH or a ``setup`` command in the facts, which
# reach every op process on both runtimes: the walker (`DesignMode.run`) and the spine
# (each node through `refactory_dsh.run.main`).

#: A shell condition true only in the run's check copy (`repo_change.CHECK_COPY_PREFIX`).
IN_COPY = 'case "$(basename "$(pwd)")" in check-*) true;; *) false;; esac'
STUB_EMIT = "Write the artifact `stub_install`"
STUB_FRONT = [SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER]
STUB_REVIEWS = ["health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER]
PACKAGE = '{"name": "stub-test", "dependencies": {}}\n'
UPDATED = render({"app.py": "print('updated')\n"})
#: A long installer error: the draft keeps 500 characters of what the check said.
LONG_ERROR = " ".join(f"ERR_PNPM_FETCH_{n:02d} the registry did not answer." for n in range(30))


def _pnpm_target(tmp_path, monkeypatch, body: str) -> tuple[Path, Path]:
    """A target whose worktrees install with pnpm, and a pnpm stub on the PATH that logs
    the directory each install ran in and then runs ``body``."""
    from tests_support import git, stub

    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    target = git_repo(tmp_path / "target")
    (target / "package.json").write_text(PACKAGE)
    (target / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n")
    git(target, "add", ".")
    git(target, "commit", "-q", "-m", "with-pnpm")
    bin_dir = tmp_path / "bin"
    stub(bin_dir, "pnpm", f'basename "$(pwd)" >> "{bin_dir}/where.log"\n{body}')
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    return target, bin_dir / "where.log"


def _installs(log: Path, prefix: str) -> int:
    """How many installs ran in the worktrees whose directory starts with ``prefix``."""
    return sum(line.startswith(prefix) for line in log.read_text().splitlines())


def _stub_run(tmp_path, target: Path, answers: list[str], **task: str):
    """Design mode on the walker, on its real graph, with nothing of refactory's faked."""
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    mode.load_state_dict(state)
    model = FakeModel([*STUB_FRONT, *answers])
    run = mode.run(tmp_path / "run", model=model, verdicts={"gate_strategy": "proceed"},
                   task={"name": "stub_install", "brief": "Update app.py",
                         "check": f"{sys.executable} app.py", "target": str(target), **task})
    return run, [p for p in model.prompts if STUB_EMIT in p]


def _spine_node(run_dir: Path, call: str, node_id: str, kind: str, monkeypatch
                ) -> tuple[int, str, str]:
    """One node visit as the spine makes it: `refactory_dsh.run.main`, for that node."""
    import io

    from refactory_dsh.run import main

    monkeypatch.setenv("REFACTORY_WORKDIR", str(run_dir))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": node_id, "type": kind}))
    out, err = io.StringIO(), io.StringIO()
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        code = main([call, "--node", node_id])
    finally:
        sys.stdout, sys.stderr = saved
    return code, out.getvalue(), err.getvalue()


def _spine_run(tmp_path, monkeypatch, target: Path, emission: str, *, setup: str = "",
               through: str = "gate_check", **knobs: int) -> tuple[Path, dict[str, str]]:
    """The spine's route from materialise to ``through``, node by node, with nothing of
    refactory's faked: materialise, prepare_check, the check beside the reviews, and
    gate_check. Returns the run directory and the label each gate printed."""
    from refactory_dsh import repo_change, worktree

    run_dir = tmp_path / "run"
    worktree.create(target, "HEAD", "copyprep0123", run_dir)
    lowering.write_declarations(DesignMode().state_dict(), run_dir)
    facts = {"name": "stub_install", "brief": "Update app.py",
             "check": f"{sys.executable} app.py", **({"setup": setup} if setup else {})}
    (run_dir / "facts.json").write_text(json.dumps(facts))
    (run_dir / "emit.txt").write_text(emission)
    if knobs:
        monkeypatch.setenv("REFACTORY_KNOBS", json.dumps(knobs))
    labels: dict[str, str] = {}
    for call, node_id, kind in ((f"{repo_change.OPS}:materialise", "materialise", "op"),
                                (f"{design_ops.__name__}:prepare_check", "prepare_check", "gate"),
                                (f"{design_ops.__name__}:check_beside_reviews", "check", "op"),
                                (f"{design_ops.__name__}:gate_check", "gate_check", "gate")):
        code, out, err = _spine_node(run_dir, call, node_id, kind, monkeypatch)
        assert code == 0, f"{node_id} exited {code}:\n{out[-2000:]}\n{err[-3000:]}"
        lines = out.strip().splitlines()
        labels[node_id] = lines[-1].strip() if lines and kind == "gate" else ""
        if node_id == through:
            break
    return run_dir, labels


def _assert_copy_only_draft(run_dir: Path, draft: dict, how: str) -> None:
    """The draft names both facts: the copy failed twice, and the run's own worktree
    prepared cleanly at the same commit. Its clipped check_failure still says no repair
    was tried."""
    assert draft["check_failing"] is True
    assert "in the check copy, twice" in draft["reason"], draft["reason"]
    assert "run's own worktree prepared cleanly at the same commit" in draft["reason"]
    assert how in draft["reason"], draft["reason"]
    assert draft["check_failure"].startswith("code repair was not attempted"), draft
    assert json.loads((run_dir / "check-record.json").read_text())["status"] == "prep_failure"


def test_an_install_failing_only_in_the_check_copy_ends_in_a_draft(tmp_path, monkeypatch):
    """The walker: pnpm fails only in the check copy, as a long error, while the run's own
    worktree and the base install cleanly. The copy is prepared twice; then the run ends
    in a draft after exactly one emit."""
    target, log = _pnpm_target(tmp_path, monkeypatch,
                               f"if {IN_COPY}; then echo '{LONG_ERROR}' >&2; exit 1; fi")
    run, emits = _stub_run(tmp_path, target, [UPDATED, *STUB_REVIEWS, *[""] * 10])
    assert run.exit_ok, run.error
    assert len(emits) == 1
    _assert_copy_only_draft(tmp_path / "run", run.facts["draft_delivery"],
                            "`pnpm install --frozen-lockfile` failed (exit 1)")
    assert _installs(log, "run-") >= 1 and _installs(log, "check-") == 2, log.read_text()


def test_an_install_failing_only_in_the_check_copy_ends_in_a_draft_on_the_spine(
        tmp_path, monkeypatch):
    """The spine, node by node: prepare_check proceeds on the run's own worktree, the copy
    fails twice, and gate_check prints ``draft`` -- the edge to the report, not ``reloop``,
    the one back to the builder."""
    target, log = _pnpm_target(tmp_path, monkeypatch,
                               f"if {IN_COPY}; then echo '{LONG_ERROR}' >&2; exit 1; fi")
    run_dir, labels = _spine_run(tmp_path, monkeypatch, target, UPDATED)
    assert labels == {"materialise": "", "prepare_check": "proceed", "check": "",
                      "gate_check": "draft"}, labels
    facts = json.loads((run_dir / "facts.json").read_text())
    _assert_copy_only_draft(run_dir, facts["draft_delivery"],
                            "`pnpm install --frozen-lockfile` failed (exit 1)")
    assert not facts.get("combined_sendback_count")
    assert _installs(log, "check-") == 2, log.read_text()


def test_a_copy_install_that_fails_once_is_prepared_again_and_the_check_goes_on(
        tmp_path, monkeypatch):
    """The walker: pnpm fails only on the copy's first install. The second preparation
    works, the check runs and passes, and the run ends with no draft after one emit."""
    once = tmp_path / "copy-install-failed-once"
    target, log = _pnpm_target(
        tmp_path, monkeypatch,
        f"if {IN_COPY} && [ ! -e '{once}' ]; then : > '{once}'; "
        "echo 'ERR_PNPM_FETCH flaky' >&2; exit 1; fi")
    run, emits = _stub_run(tmp_path, target, [UPDATED, *STUB_REVIEWS, "VERDICT: PASS"])
    assert run.exit_ok, run.error
    assert len(emits) == 1
    assert not run.facts.get("draft_delivery")
    assert json.loads((tmp_path / "run" / "check-record.json").read_text())["status"] == "pass"
    assert once.exists() and _installs(log, "check-") == 2, log.read_text()


@pytest.mark.parametrize(("body", "knobs", "how"), [
    ("kill -9 $$", {}, "was killed by signal 9"),
    ("sleep 40", {"check_timeout_s": 3}, "did not finish (stopped at its time limit)"),
], ids=["killed", "timed-out"])
def test_a_killed_or_timed_out_copy_setup_ends_in_a_draft_on_the_spine(
        tmp_path, monkeypatch, body, knobs, how):
    """The spine: a ``setup`` killed by a signal, or silent past its inactivity limit, in the
    check copy only. Each ends in a draft; neither stops the run with no pull request."""
    setup = f"if {IN_COPY}; then {body}; fi"
    run_dir, labels = _spine_run(tmp_path, monkeypatch, git_repo(tmp_path / "target"), UPDATED,
                                 setup=setup, **knobs)
    assert labels["prepare_check"] == "proceed" and labels["gate_check"] == "draft", labels
    _assert_copy_only_draft(run_dir, json.loads((run_dir / "facts.json").read_text())[
        "draft_delivery"], f"`{setup}` {how}")


def test_a_killed_copy_setup_ends_in_a_draft(tmp_path):
    """The walker: a ``setup`` killed by a signal in the check copy only ends in a draft
    after exactly one emit."""
    setup = f"if {IN_COPY}; then kill -9 $$; fi"
    run, emits = _stub_run(tmp_path, git_repo(tmp_path / "target"),
                           [UPDATED, *STUB_REVIEWS, *[""] * 10], setup=setup)
    assert run.exit_ok, run.error
    assert len(emits) == 1
    _assert_copy_only_draft(tmp_path / "run", run.facts["draft_delivery"],
                            f"`{setup}` was killed by signal 9")


#: An emission that adds a dependency the registry does not have: its install fails.
BROKEN_MANIFEST = render({"package.json": '{"name": "stub-test", "dependencies": '
                                          '{"left-pad": "99.0.0"}}\n',
                          "app.py": "print('updated')\n"})
#: pnpm that refuses the unfrozen install a changed manifest calls for.
REFUSES_NEW_DEPENDENCY = ('case "$*" in *--no-frozen-lockfile*) '
                          'echo "ERR_PNPM_NO_MATCHING_VERSION left-pad@99.0.0" >&2; exit 1;; esac')


def test_a_change_caused_install_failure_goes_to_the_builder(tmp_path, monkeypatch):
    """The walker: the change breaks its manifest, so the install fails in the run's own
    worktree while the base installs cleanly (`base_check.compare_prep_failure`). The
    builder gets the error and fixes it: two emits, no draft."""
    target, _log = _pnpm_target(tmp_path, monkeypatch, REFUSES_NEW_DEPENDENCY)
    fixed = render({"package.json": PACKAGE, "app.py": "print('updated')\n"})
    run, emits = _stub_run(tmp_path, target,
                           [BROKEN_MANIFEST, fixed, *STUB_REVIEWS, "VERDICT: PASS"])
    assert run.exit_ok, run.error
    assert len(emits) == 2
    assert "ERR_PNPM_NO_MATCHING_VERSION" in emits[1]
    assert not run.facts.get("draft_delivery")


def test_a_change_caused_install_failure_goes_to_the_builder_on_the_spine(tmp_path, monkeypatch):
    """The spine: prepare_check prints ``reloop`` -- back to the builder -- with the
    install's error as the feedback, and no draft."""
    target, _log = _pnpm_target(tmp_path, monkeypatch, REFUSES_NEW_DEPENDENCY)
    run_dir, labels = _spine_run(tmp_path, monkeypatch, target, BROKEN_MANIFEST,
                                 through="prepare_check")
    assert labels["prepare_check"] == "reloop", labels
    facts = json.loads((run_dir / "facts.json").read_text())
    assert "ERR_PNPM_NO_MATCHING_VERSION" in facts["validation"]
    assert not facts.get("draft_delivery")


def test_real_failing_install_stub_ends_in_draft(tmp_path, monkeypatch):
    """Real failing install through _check_in_worktree → compare_prep_failure → _full_check,
    with the base failing the same way, ends in a draft (no builder round)."""
    target, _log = _pnpm_target(tmp_path, monkeypatch,
                                "echo 'ERR_PNPM_FETCH: network error' >&2; exit 1")
    run, emits = _stub_run(tmp_path, target, [UPDATED, *STUB_REVIEWS, *[""] * 10])
    assert run.exit_ok, run.error
    draft = run.facts["draft_delivery"]
    assert draft["check_failing"] is True
    assert draft["check_failure"].startswith("code repair was not attempted")
    assert len(emits) == 1, f"expected exactly 1 emit, got {len(emits)}"


def _copy_without_a_prep_record(ctx, found, timeout):
    """`repo_change._check_copy` whose preparation failed with problems but no record."""
    import child_patches

    from refactory_dsh import repo_change

    copy, env, home, _problems, _fail = child_patches.real(repo_change, "_check_copy")(
        ctx, found, timeout)
    return copy, env, home, ["install failed: some reason"], None


def test_a_copy_preparation_failing_without_a_record_ends_in_a_draft(tmp_path, monkeypatch):
    """The walker: when the copy's preparation reports problems but no prep-failure record,
    a record is made from the problems and the run ends in a draft after one emit."""
    import child_patches

    from refactory_dsh import repo_change

    child_patches.for_test(monkeypatch).setattr(repo_change, "_check_copy",
                                                _copy_without_a_prep_record)
    run, emits = _stub_run(tmp_path, git_repo(tmp_path / "target"),
                           [UPDATED, *STUB_REVIEWS, *[""] * 10])
    assert run.exit_ok, run.error
    assert len(emits) == 1
    draft = run.facts["draft_delivery"]
    assert draft["check_failing"] is True
    assert "`prepare` failed (exit 1)" in draft["reason"]
    assert "install failed: some reason" in draft["check_failure"]


def _spine_prep_attempt(tmp_path, monkeypatch, prep_result):
    """A spine-shaped run with one applied attempt, whose check beside the reviews returns
    ``prep_result``. Returns the run directory and a helper that runs one spine node."""
    from refactory_dsh import repo_change, worktree
    from refactory_dsh.run import run_node

    original = repo_change.run_check

    def _beside(ctx, command, timeout, **kw):
        return prep_result if kw.get("beside_reviews") else original(ctx, command, timeout, **kw)

    monkeypatch.setattr(repo_change, "run_check", _beside)
    repo = git_repo(tmp_path / "repo")
    run_dir = tmp_path / "run"
    worktree.create(repo, "HEAD", "prepfail01234", run_dir)
    lowering.write_declarations(DesignMode().state_dict(), run_dir)
    (run_dir / "facts.json").write_text(json.dumps({
        "name": "prep_tool", "brief": "Make tool.py print built.",
        "check": f"{sys.executable} tool.py"}))
    (run_dir / "emit.txt").write_text(render({"tool.py": "print('built')\n"}))
    run_node(f"{repo_change.OPS}:materialise", workdir=run_dir, node_id="materialise")
    facts = json.loads((run_dir / "facts.json").read_text())
    (run_dir / CHECK_PREP).write_text(json.dumps({
        "attempt": "a1", "commit": facts["commit"],
        "root": str(facts.get("artifact") or ""),
        "targeted_command": None}))

    def spine_node(call, node_id, kind):
        code, out, _err = _spine_node(run_dir, f"refactory_dsh.design_ops:{call}", node_id,
                                      kind, monkeypatch)
        lines = [line for line in out.splitlines() if line.strip()]
        return code, lines[-1] if lines and kind == "gate" else ""

    return run_dir, spine_node


def test_code_failure_check_unchanged(tmp_path, monkeypatch):
    """A normal code failure (no prep_failure, no change_caused_prep) goes to the builder
    as today — gate_check returns reloop, no BuildLoopStop, no draft."""
    from refactory_dsh.sandbox import SandboxResult

    code_result = SandboxResult(
        ok=False, returncode=1, elapsed_s=1.0,
        stdout="FAILED test_x\n1 failed in 0.01s\n", stderr="",
    )
    run_dir, spine_node = _spine_prep_attempt(tmp_path, monkeypatch, code_result)
    code_check, _ = spine_node("check_beside_reviews", "check", "op")
    assert code_check == 0
    code_gate, verdict = spine_node("gate_check", "gate_check", "gate")
    assert code_gate == 0
    # A code failure sends the builder back, not a draft.
    assert verdict == "reloop", f"expected 'reloop', got '{verdict}'"
    facts = json.loads((run_dir / "facts.json").read_text())
    assert "draft_delivery" not in facts or not facts.get("draft_delivery")


def test_the_check_copy_is_prepared_once_more_before_its_failure_stands(tmp_path, monkeypatch):
    """Unit: a failed copy preparation is retried once; the second failure is returned as
    the copy's, with the commit, and never as a change-caused failure."""
    from types import SimpleNamespace

    from refactory_dsh import repo_change, worktree

    found = SimpleNamespace(path=str(tmp_path), head="c1", target=str(tmp_path),
                            base="b1", run=str(tmp_path))
    monkeypatch.setattr(worktree, "recorded", lambda workdir: found)
    monkeypatch.setattr(worktree, "shared_change", lambda f: None)
    calls = []

    def failing_copy(ctx, found, timeout):
        calls.append(found.head)
        return (found, {"PATH": "/bin"}, tmp_path / "home", ["install failed"],
                {"step": "install", "returncode": 1, "timed_out": False})

    monkeypatch.setattr(repo_change, "_check_copy", failing_copy)
    ctx = nn.RunContext(tmp_path, facts={"commit": "c1", "check": "true"})
    result = repo_change.run_check(ctx, "true", 60, beside_reviews=True)
    assert calls == ["c1", "c1"]
    assert result.prep_failure == {"step": "install", "returncode": 1, "timed_out": False,
                                   "error_text": "install failed", "check_copy": True,
                                   "commit": "c1"}
    assert result.change_caused_prep is None


def test_check_in_worktree_defensive_synthetic_prep_failure(tmp_path, monkeypatch):
    """Unit: C2b — _check_in_worktree builds a synthetic prep_failure when _prepare_in_worktree
    returns problems but prep_fail is None (defensive case).  Without C2b, the result
    would have prep_failure=None and the builder would be sent back."""
    from types import SimpleNamespace

    from refactory_dsh import base_check, repo_change, worktree

    found = SimpleNamespace(path=str(tmp_path), head="c1", target=str(tmp_path),
                            base="b1", run=str(tmp_path))
    monkeypatch.setattr(worktree, "recorded", lambda workdir: found)
    monkeypatch.setattr(worktree, "shared_change", lambda f: None)
    monkeypatch.setattr(worktree, "restore", lambda workdir: (found, []))

    # _prepare_in_worktree returns problems but prep_fail=None (defensive case).
    monkeypatch.setattr(repo_change, "_prepare_in_worktree",
                        lambda ctx, f, t: (f, {"PATH": "/bin"}, tmp_path / "home",
                                           ["npm install failed: ETIMEOUT"], None))
    # compare_prep_failure returns None (not change-caused / unavailable).
    monkeypatch.setattr(base_check, "compare_prep_failure", lambda *a, **k: None)

    ctx = nn.RunContext(tmp_path, facts={"commit": "c1", "check": "true"})
    result = repo_change.run_check(ctx, "true", 60)
    # C2b: synthetic prep_failure is built from problems.
    assert result.prep_failure is not None
    assert result.prep_failure["step"] == "prepare"
    assert "npm install failed" in result.prep_failure["error_text"]
    assert result.change_caused_prep is None
