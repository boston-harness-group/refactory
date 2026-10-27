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
    # The check sends the run nowhere: like the reviews, its branch drains into the join.
    assert not [e for e in edges if e[0] == "check"]
    assert ("join_qa", "gate_check", None) in edges
    assert ("gate_check", "pick_corrections", "reloop") in edges
    assert ("gate_check", "synthesize_qa", "proceed") in edges
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
    assert verdict == "reloop" and verdict.reason == "tests failed"
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
    monkeypatch.setattr(design_ops, "check_artifact", lambda ctx, *, check_attempt=None:
                        whole.append(check_attempt) or nn.Verdict("reloop", "judged here"))
    for receipt in ({"status": "skipped", "source": "person", "check_attempt": "a1"},
                    {"status": "skipped", "source": "ceo", "check_attempt": "a2"}):
        ctx.write_json(CHECK_VERDICT, _decision("proceed", attempt="a1"))
        ctx.write_json(evidence.EVIDENCE, {"validation": receipt})
        assert design_ops.gate_check(ctx).reason == "judged here"
    assert whole == ["a2", "a2"]
    # What it judged is recorded for this attempt: a rerun of the gate routes on it.
    assert design_ops.gate_check(ctx).reason == "judged here"
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
    from tests_support import early_asks_judgment

    ctx = _ctx(tmp_path)
    monkeypatch.setattr(repo_change, "prepare_attempt", lambda ctx, timeout: [])
    # The attempt's first self-check (b48a) runs the whole check once, beside the judge.
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
    assert design_ops.prepare_check(ctx) == "first"
    assert not ctx.exists(CHECK_PREP) and not ctx.exists(CHECK_VERDICT)
    assert design_ops.first_self_check(ctx) == {}
    ctx.write_text(design_ops.EARLY_ASKS_VERDICT, early_asks_judgment())
    sent_back = design_ops.gate_self_check(ctx)
    # A still-failing test goes back as a fix-up first, spending nothing.
    assert sent_back == "reloop" and sent_back.reason.startswith("self-check fix-up 1 of 2")
    assert early_stops == [] and not ctx.facts.get("attempts_used")
    assert not ctx.exists(CHECK_PREP)  # no review starts on this attempt
    assert not ctx.exists(CHECK_VERDICT)
    # A fix-up's self-check runs in prepare_check, on what failed and what the fix-up changed.
    go = failfast.PrepassResult(
        ran=True, early_stop=False, targeted_command="pytest t.py::x", check_mode="full",
        result=None, digest_text="", selection=["t.py::x"], ran_ids=["t.py::x"],
        commands=["pytest t.py::x"])
    monkeypatch.setattr(failfast, "targeted_prepass", lambda *a, **k: go)
    assert design_ops.prepare_check(ctx) == "proceed"
    prepared = ctx.read_json(CHECK_PREP)
    assert prepared["commit"] == "c1" and prepared["root"] == str(tmp_path)
    assert prepared["targeted_command"] == "pytest t.py::x" and prepared["attempt"]
    # The reviewers judge the attempt against what sent it back, not against its fix-up.
    assert ctx.facts["review_feedback"] == "what went wrong last time"
    assert ctx.facts["self_check_fixup_count"] == 0


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
    monkeypatch.setattr(worktree, "shared_changed", lambda found: False)
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
