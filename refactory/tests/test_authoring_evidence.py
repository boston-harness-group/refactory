"""Behavioral controls for canonical execution, repair and faithful sign-off."""
import json
from pathlib import Path

import pytest

from refactory_dsh import authoring_evidence as evidence, create_ops, nn, sandbox
from refactory_dsh.agent_delivery import begin, finish
from refactory_dsh.create import CreateMode
from refactory_dsh.edit import EditMode
from refactory_dsh.fast import CreateFastMode, EditFastMode
from refactory_dsh.nn import Verdict
from refactory_dsh.steering import Ledger
from tests_support import FakeModel, create_script, good_files, render
from test_fast_modes import PLAN, QA_PASS


#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


@pytest.mark.parametrize("cls", [CreateMode, EditMode, CreateFastMode, EditFastMode])
def test_canonical_execution_repairs_before_any_review(tmp_path, monkeypatch, cls):
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    name = "repair_probe"
    good = good_files(name)
    bad = {**good, "ops.py": good["ops.py"].replace('target = int(ctx.knob("target", 3))',
                                                           'raise ValueError("deliberate runtime defect")')}
    is_fast = cls in (CreateFastMode, EditFastMode)
    answers = ([PLAN] if cls is CreateFastMode else []) + [render(bad), render(good), QA_PASS]
    if not is_fast:
        answers = create_script(name, [render(bad), render(good)])
    client = FakeModel(answers)
    mode = cls()
    state = mode.state_dict()
    state["knobs"]["single_shot"]["value"] = True
    mode.load_state_dict(state)
    calls = []
    execute = sandbox.smoke

    def observe(identifier, workdir, **kwargs):
        calls.append((identifier, workdir))
        return execute(identifier, workdir, **kwargs)

    monkeypatch.setattr(sandbox, "smoke", observe)
    run = mode.run(tmp_path / "run", task={"name": name, "brief": "Count to the target"},
                   model=client, verdicts={"gate_strategy": "proceed"})
    assert run.exit_ok, run.error
    assert len(calls) == 2 and calls[0][1] != calls[1][1]
    assert "deliberate runtime defect" in client.prompts[-2 if is_fast else -5]
    history = [json.loads(path.read_text()) for path in
               (tmp_path / "run/.refactory/validation-history").glob("*/authoring-evidence.json")]
    assert any(row["validation"]["status"] == "code_failure" for row in history)
    assert json.loads((tmp_path / "run/smoke.json").read_text())["ok"]
    review_prompt = next(prompt for prompt in client.prompts if "Authoritative validation" in prompt)
    assert '"status": "pass"' in review_prompt
    assert json.loads((tmp_path / "run/smoke.json").read_text())["error"] == ""


@pytest.mark.parametrize("error,timed_out,kind", [
    ("missing provider credentials", False, "unavailable"),
    ("worker timed out", True, "uncertain"),
    ("cancelled by user", False, "cancelled"),
    ("token budget exhausted", False, "resource_exhausted"),
])
def test_unrepairable_validation_does_not_recode(tmp_path, monkeypatch, error, timed_out, kind):
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    mode = CreateFastMode()
    state = mode.state_dict()
    state["knobs"]["single_shot"]["value"] = True
    mode.load_state_dict(state)
    monkeypatch.setattr(sandbox, "smoke", lambda *a, **k: sandbox.SandboxResult(
        ok=False, returncode=1, stderr=error, timed_out=timed_out))
    client = FakeModel([PLAN, render(good_files("blocked"))])
    run = mode.run(tmp_path / "run", task={"name": "blocked", "brief": "count"}, model=client)
    assert not run.exit_ok and kind in run.error
    assert len(client.prompts) == 2
    assert run.facts["repairs_used"] == 0
    assert not (tmp_path / "run/create.json").exists()


@pytest.mark.parametrize("original_failure", ["exception", "timeout"])
def test_comparison_bypass_never_executes_original(tmp_path, monkeypatch, original_failure):
    calls = []

    def execute(identifier, *args, **kwargs):
        calls.append(identifier)
        if identifier == "counting_pack/fixed_step":
            if original_failure == "exception":
                raise AssertionError("original must never run")
            return sandbox.SandboxResult(ok=False, timed_out=True)
        return sandbox.SandboxResult(ok=True, payload={"exit_ok": True, "score": 0.4})

    monkeypatch.setattr(sandbox, "smoke", execute)
    ctx = nn.RunContext(tmp_path, facts={"name": "circle_packing", "revision_of": "counting_pack/fixed_step"},
                        knobs={"allow_regression": True})
    result = create_ops.smoke(ctx)
    assert calls == ["circle_packing/adaptive"]
    assert result["before_score"] is result["regressed"] is None
    assert result["after_score"] == 0.4
    recorded = ctx.read_json("revision.json")
    assert recorded["revision_of_state"] and recorded["revision_of_digest"]
    assert "intentionally skipped" in recorded["note"]


def audit_context(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "answer.txt").write_text("42")
    ctx = nn.RunContext(tmp_path, facts={"brief": "Deliver the answer and explanation", "artifact": str(root)})
    ctx.write_json("brief.json", {"brief": ctx.facts["brief"]})
    ctx.write_json("strategy.json", {"acceptance_criteria": ["answer is 42", "explanation exists"]})
    Ledger.seed(tmp_path, ctx.facts["brief"])
    evidence.record(ctx, {"status": "pass", "command": "test answer.txt", "stdout": "42"})
    return ctx


def deliver_audit(ctx, *, omitted=(), exclusion=None):
    begin(ctx.workdir, "overwatch", "Audit each ask", ["reviews/overwatch.md", evidence.AUDIT])
    rows = []
    for item in ctx.read_json(evidence.EVIDENCE)["requirements"]:
        if item["id"] in omitted:
            continue
        row = {"id": item["id"], "requirement": item["text"], "status": "pass",
               "evidence": [{"path": "artifact/answer.txt", "detail": "read actual answer: 42"}]}
        if item["id"] == "criterion-2" and exclusion:
            change = next((item for item in ctx.read_json(evidence.EVIDENCE)["requirements"]
                           if item["id"] == exclusion), {})
            row.update(status="excluded", reason="explanation excluded", user_change=exclusion,
                       user_quote=change.get("text", ""))
        if item.get("kind") in ("correction", "constraint"):
            row["clauses"] = [{"clause": item.get("text", ""),
                               "evidence": [{"path": "artifact/answer.txt",
                                             "detail": "full constraint verified"}]}]
        rows.append(row)
    ctx.write_json(evidence.AUDIT, {"requirements": rows})
    ctx.write_text("reviews/overwatch.md", "VERDICT: PASS")
    finish(ctx.workdir, "overwatch")


def test_missing_requirement_and_reviewer_waiver_cannot_pass(tmp_path):
    ctx = audit_context(tmp_path)
    deliver_audit(ctx, omitted={"criterion-2"})
    # Audit-only problems with VERDICT: PASS return the "audit" label, not "reloop".
    result = create_ops.gate_overwatch(ctx)
    assert str(result) == "audit"
    assert "criterion-2: no requirement evidence" in ctx.facts["validation"]
    deliver_audit(ctx, exclusion="reviewer says out of scope")
    result = create_ops.gate_overwatch(ctx)
    assert str(result) == "audit"
    assert "reviewers cannot waive" in ctx.facts["validation"]


def test_explicit_user_change_requires_fresh_validation_and_audit(tmp_path):
    ctx = audit_context(tmp_path)
    deliver_audit(ctx)
    assert create_ops.gate_overwatch(ctx) == "proceed"
    Ledger(tmp_path).note("correction", "I explicitly exclude the explanation; only deliver 42")
    assert create_ops.gate_overwatch(ctx) == "reloop"
    assert "user intent changed" in ctx.facts["validation"]
    evidence.record(ctx, {"status": "pass", "stdout": "42"})
    change = next(item["id"] for item in ctx.read_json(evidence.EVIDENCE)["requirements"] if item.get("kind") == "correction")
    deliver_audit(ctx, exclusion=change)
    assert create_ops.gate_overwatch(ctx) == "proceed"


def test_repair_invalidates_audit_even_after_new_successful_validation(tmp_path):
    ctx = audit_context(tmp_path)
    deliver_audit(ctx)
    (tmp_path / "artifact/answer.txt").write_text("43")
    assert evidence.current_problems(ctx, ("overwatch",))
    evidence.record(ctx, {"status": "pass", "stdout": "43"})
    assert create_ops.gate_overwatch(ctx) == "reloop"
    assert "no current delivery/sign-off" in ctx.facts["validation"]


def test_audit_rejects_directory_reference_with_actionable_path(tmp_path):
    ctx = audit_context(tmp_path)
    deliver_audit(ctx)
    audit = ctx.read_json(evidence.AUDIT)
    audit["requirements"][0]["evidence"][0]["path"] = "artifact/"
    ctx.write_json(evidence.AUDIT, audit)
    assert "artifact'" in evidence.audit_problems(ctx)[0]
    assert "individual file" in evidence.audit_problems(ctx)[0]


def test_audit_accepts_a_path_relative_to_the_delivered_artifact(tmp_path):
    """Found live: a reviewer of a run in a worktree cites `datesum.py`, not an absolute path."""
    ctx = audit_context(tmp_path)
    deliver_audit(ctx)
    audit = ctx.read_json(evidence.AUDIT)
    audit["requirements"][0]["evidence"][0]["path"] = "answer.txt"
    ctx.write_json(evidence.AUDIT, audit)
    assert evidence.audit_problems(ctx) == []
    audit["requirements"][0]["evidence"][0]["path"] = "nowhere.txt"
    ctx.write_json(evidence.AUDIT, audit)
    assert "nowhere.txt" in evidence.audit_problems(ctx)[0]


def test_audit_instructions_tell_probing_reviewers_to_stage_fixtures_in_the_run():
    """A reviewer told to probe but not where to put fixtures cites /tmp and loses the cycle.

    The audit validator accepts only files inside the run directory or the delivered
    artifact; without staging instructions a spot-check fixture written to /tmp is
    uncitable, and its evidence dies with the gate verdict.
    """
    from refactory_dsh.create import CreateMode
    from refactory_dsh.design import DesignMode

    # The node that writes the evidence map: the overwatch in create, the QA judge in
    # design (so the correction audit can measure the map before the overwatch judges).
    for graph, writer in ((CreateMode().build(), "overwatch"), (DesignMode().build(), "qa_judge")):
        prompt = graph.nodes[writer].prompt_template
        assert "Write intent-audit.json as" in prompt
        assert "scratch/" in prompt
        assert "never /tmp" in prompt


def test_waiver_requires_user_quote_and_runtime_output_cannot_be_replaced(tmp_path):
    ctx = audit_context(tmp_path)
    deliver_audit(ctx)
    audit = ctx.read_json(evidence.AUDIT)
    audit["requirements"][0].update(status="excluded", user_change="brief", reason="reviewer chose to waive")
    ctx.write_json(evidence.AUDIT, audit)
    assert "reviewers cannot waive" in evidence.audit_problems(ctx)[0]
    output = tmp_path / "smoke/attempt/greeting.txt"
    output.parent.mkdir(parents=True)
    output.write_text("hello\n")
    ctx.write_json("smoke.json", {"workdir": str(output.parent), "ok": True})
    evidence.record(ctx, {"status": "pass"})
    assert not evidence.current_problems(ctx)
    output.write_text("different\n")
    assert "canonical runtime output files changed" in ";".join(evidence.current_problems(ctx))


def test_unavailable_audit_cannot_pass_or_request_code_repair(tmp_path):
    ctx = audit_context(tmp_path)
    deliver_audit(ctx)
    ctx.write_text("reviews/overwatch.md", "UNAVAILABLE: renderer missing; install the requested renderer\nVERDICT: PASS")
    with pytest.raises(ValueError, match="validation unavailable.*renderer"):
        create_ops.gate_overwatch(ctx)
    assert not ctx.facts.get("attempts_used")


def test_code_findings_about_timeout_and_local_modules_remain_repairable():
    assert evidence.failure_kind("Missing required timeout setting") == "code_failure"
    assert evidence.failure_kind("ModuleNotFoundError: No module named 'refactory_packages.new.ops'") == "code_failure"
    assert evidence.failure_kind("ModuleNotFoundError: No module named 'external_renderer'") == "unavailable"


def test_signal_death_classified_as_uncertain():
    assert evidence.failure_kind("killed by SIGABRT") == "uncertain"


def test_signal_death_stops_check_artifact_without_attempt(tmp_path, monkeypatch):
    """A signal-killed environment install stops the run as uncertain, not as a code failure."""
    from refactory_dsh import design_ops, repo_change

    # Set up a RunContext with the minimum facts check_artifact needs.
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "tool.py").write_text("print(1)\n")
    ctx = nn.RunContext(tmp_path, facts={
        "name": "signal_probe",
        "brief": "a tool that gets killed",
        "artifact": str(root),
        "check": "python tool.py",
    })
    ctx.write_json("brief.json", {"brief": ctx.facts["brief"]})
    Ledger.seed(tmp_path, ctx.facts["brief"])

    # Monkeypatch repo_change so check_artifact never touches the real filesystem
    # or worktree machinery: it sees the signal-death SandboxResult directly.
    # The prep_failure field is what _check_in_worktree now sets when an install
    # is killed by a signal (returncode < 0).
    monkeypatch.setattr(repo_change, "refused", lambda ctx: "")
    monkeypatch.setattr(repo_change, "run_check", lambda ctx, cmd, timeout: sandbox.SandboxResult(
        ok=False, returncode=1, stderr="the worktree's environment could not be prepared:\n"
        "`pnpm install --frozen-lockfile` failed: killed by SIGABRT", stdout="",
        prep_failure={"step": "pnpm install --frozen-lockfile", "returncode": -6, "timed_out": False}))

    # check_artifact must raise ValueError matching "uncertain" — stop_unrepairable's
    # message — proving the run does not reloop into emit or spend an attempt.
    with pytest.raises(ValueError, match="uncertain"):
        design_ops.check_artifact(ctx)

    # No attempt was spent: repair() is never reached because stop_unrepairable raises first.
    assert not ctx.facts.get("attempts_used")


def test_prep_failure_nonzero_reaches_builder():
    """A preparation failure exiting non-zero without a signal is code_failure.

    The builder may have broken a manifest, so it still goes to the builder.
    """
    from dataclasses import dataclass, field

    @dataclass
    class FakeResult:
        ok: bool = False
        timed_out: bool = False
        returncode: int = 1
        stdout: str = ""
        stderr: str = "setup command failed"
        prep_failure: dict = field(default_factory=lambda: {"step": "pnpm run build", "returncode": 1, "timed_out": False})
        def problem(self) -> str:
            return ""

    result = FakeResult()
    kind = evidence.check_failure_kind(result)
    assert kind == "code_failure"
    # stop_unrepairable does not raise for code_failure — failure reaches the builder
    evidence.stop_unrepairable(kind, "setup failed")


# ── asks 2 and 3: Initial Idea and info are not requirements ──────────────


def test_initial_idea_is_not_a_separate_requirement(tmp_path):
    """The Initial Idea duplicates the brief; it is not a second requirement."""
    ctx = audit_context(tmp_path)
    reqs = evidence.requirements(ctx)
    assert not any(r.get("kind") == "Initial Idea" for r in reqs), "Initial Idea should not be a requirement"
    # But intent() still includes it for identity tracking.
    entries = evidence.intent(ctx.workdir)
    assert any(e["kind"] == "Initial Idea" for e in entries), "intent() must still include Initial Idea"


def test_info_notes_are_not_requirements_but_corrections_are(tmp_path):
    """Info notes are context, not testable requirements."""
    ctx = audit_context(tmp_path)
    Ledger(tmp_path).note("info", "This is context about the process")
    Ledger(tmp_path).note("correction", "Fix the explanation too")
    reqs = evidence.requirements(ctx)
    assert not any(r.get("kind") == "info" for r in reqs), "info notes should not be requirements"
    assert any(r.get("kind") == "correction" for r in reqs), "corrections must remain requirements"
    entries = evidence.intent(ctx.workdir)
    assert any(e["kind"] == "info" for e in entries), "intent() must still include info entries"
    assert any(e["kind"] == "correction" for e in entries)


def test_ceo_notes_are_not_requirements(tmp_path):
    """C5: CEO-sourced corrections are context, not auditable requirements."""
    ctx = audit_context(tmp_path)
    Ledger(tmp_path).note("correction", "CEO noticed a typo", source="ceo")
    Ledger(tmp_path).note("correction", "User says fix the padding", source="user-relayed")
    reqs = evidence.requirements(ctx)
    ceo_texts = [r["text"] for r in reqs if "CEO noticed" in r.get("text", "")]
    assert not ceo_texts, "CEO-sourced corrections must not be requirements"
    user_texts = [r["text"] for r in reqs if "fix the padding" in r.get("text", "")]
    assert user_texts, "user-relayed corrections must remain requirements"


def test_person_skip_lets_require_completion_pass(tmp_path):
    """dm-115: A person's skip with matching evidence lets require_completion pass."""
    ctx = audit_context(tmp_path)
    # Write declarations.json with a check node that produces runtime_evidence
    decl_dir = tmp_path / ".refactory"
    decl_dir.mkdir(exist_ok=True)
    import json
    (decl_dir / "declarations.json").write_text(json.dumps({
        "check": {"produces": ["runtime_evidence"]}
    }))
    # Write a spine checkpoint with person skip provenance
    (decl_dir / "spine-checkpoint.json").write_text(json.dumps({
        "state": {
            "skipped": ["check"],
            "skip_provenance": {
                "check": {"source": "person", "at": "2026-10-20T22:25:29+00:00"}
            }
        }
    }))
    # Re-record evidence with the skip outcome using the proper API.
    # This produces an evidence file whose identity hash matches
    # delivery_identity(ctx.workdir), unlike manual file manipulation.
    evidence.record(ctx, {
        "status": "skipped",
        "source": "person",
        "at": "2026-10-20T22:25:29+00:00",
        "via": "graph-control",
        "node": "check",
    })
    # current_problems without workers proves the person-skip validation
    # path works: no "did not pass" and no identity mismatch.
    problems = evidence.current_problems(ctx)
    assert not problems, f"person skip should produce no problems, got: {problems}"


def test_ceo_skip_is_refused_by_require_completion(tmp_path):
    """dm-115: A CEO skip is refused by current_problems, naming the source."""
    ctx = audit_context(tmp_path)
    import json
    decl_dir = tmp_path / ".refactory"
    decl_dir.mkdir(exist_ok=True)
    (decl_dir / "declarations.json").write_text(json.dumps({
        "check": {"produces": ["runtime_evidence"]}
    }))
    (decl_dir / "spine-checkpoint.json").write_text(json.dumps({
        "state": {
            "skipped": ["check"],
            "skip_provenance": {
                "check": {"source": "ceo", "at": "2026-10-20T22:25:29+00:00"}
            }
        }
    }))
    problems = evidence.current_problems(ctx)
    assert any("ceo" in p for p in problems), f"CEO skip should be refused, got: {problems}"


def _skip_check(tmp_path, ctx, source):
    """The check node skipped by ``source``, and the evidence that skip records."""
    at = "2026-10-20T22:25:29+00:00"
    decl_dir = tmp_path / ".refactory"
    decl_dir.mkdir(exist_ok=True)
    (decl_dir / "declarations.json").write_text(json.dumps({"check": {"produces": ["runtime_evidence"]}}))
    (decl_dir / "spine-checkpoint.json").write_text(json.dumps({"state": {
        "skipped": ["check"], "skip_provenance": {"check": {"source": source, "at": at}}}}))
    evidence.record(ctx, {"status": "skipped", "source": source, "at": at, "via": "graph-control",
                          "node": "check"})


def test_ceo_skip_is_refused_on_the_draft_path(tmp_path):
    """A draft's report refuses a check the CEO skipped exactly as require_completion
    does: only a person's skip is accepted (#500)."""
    ctx = audit_context(tmp_path)
    _skip_check(tmp_path, ctx, "ceo")
    assert evidence.current_problems(ctx) == [
        "required runtime validation was skipped by ceo; only a person's skip is accepted"]
    with pytest.raises(ValueError, match="skipped by ceo; only a person's skip is accepted"):
        evidence.require_current_evidence(ctx)


def test_person_skip_is_accepted_on_the_draft_path(tmp_path):
    """The control: a person's skip, recorded as such, passes the draft path's checks."""
    ctx = audit_context(tmp_path)
    _skip_check(tmp_path, ctx, "person")
    evidence.require_current_evidence(ctx)


def test_fast_mode_ceo_skip_refused(tmp_path):
    """AC-13: fast_ops.report_artifact raises ValueError naming 'ceo' when the
    check was skipped by the CEO, independently of the current_problems path."""
    import json
    from refactory_dsh import fast_ops

    ctx = nn.RunContext(tmp_path, facts={"name": "test_pkg"})
    # focused-check.json with a non-pass status (the check never ran)
    ctx.write_json("focused-check.json", {"status": "skipped"})
    # declarations.json marking focused_check as a runtime_evidence producer
    decl_dir = tmp_path / ".refactory"
    decl_dir.mkdir(exist_ok=True)
    (decl_dir / "declarations.json").write_text(json.dumps({
        "focused_check": {"produces": ["runtime_evidence"]}
    }))
    # spine checkpoint with CEO skip provenance
    (decl_dir / "spine-checkpoint.json").write_text(json.dumps({
        "state": {
            "skipped": ["focused_check"],
            "skip_provenance": {
                "focused_check": {"source": "ceo", "at": "2026-10-20T22:25:29+00:00"}
            }
        }
    }))
    with pytest.raises(ValueError, match="ceo"):
        fast_ops.report_artifact(ctx)


def test_requirements_excludes_ceo_note_naming_gate_verdict(tmp_path):
    """dm-113: A CEO note naming a gate verdict is excluded from requirements."""
    ctx = audit_context(tmp_path)
    Ledger(tmp_path).note("correction", "gate_strategy: proceed. The person approved.", source="ceo")
    reqs = evidence.requirements(ctx)
    ceo_texts = [r["text"] for r in reqs if "gate_strategy" in r.get("text", "")]
    assert not ceo_texts, "CEO note naming a gate verdict must not be a requirement"


# ── ask 1: gate_overwatch says what actually failed ───────────────────────


def test_gate_overwatch_labels_evidence_problems(tmp_path):
    """When current_problems exist (stale evidence), the reason names them."""
    ctx = audit_context(tmp_path)
    deliver_audit(ctx)
    # Modify the artifact so evidence is stale.
    (tmp_path / "artifact" / "answer.txt").write_text("changed")
    result = create_ops.gate_overwatch(ctx)
    assert result == "reloop"
    assert "evidence problems" in ctx.facts["validation"]


def test_gate_overwatch_audit_only_returns_audit_label(tmp_path):
    """When VERDICT: PASS, no evidence problems, no blocking findings, but audit
    problems exist, the gate returns Verdict('audit', ...) not 'reloop'."""
    ctx = audit_context(tmp_path)
    deliver_audit(ctx, omitted={"criterion-2"})
    result = create_ops.gate_overwatch(ctx)
    assert isinstance(result, Verdict)
    assert str(result) == "audit"
    assert "intent audit problems" in result.reason
    assert not ctx.facts.get("attempts_used"), "audit-only path must not spend an attempt"


def test_gate_overwatch_labels_blocking_findings(tmp_path):
    """Blocking findings in overwatch.md are labeled 'blocking review findings'."""
    ctx = audit_context(tmp_path)
    deliver_audit(ctx)
    ctx.write_text("reviews/overwatch.md",
                   "BLOCKING: test failure\nVERDICT: FAIL")
    result = create_ops.gate_overwatch(ctx)
    assert result == "reloop"
    assert "blocking review findings" in ctx.facts["validation"]


def test_gate_overwatch_labels_missing_verdict(tmp_path):
    """A missing verdict is labeled 'overwatch verdict is missing'."""
    ctx = audit_context(tmp_path)
    deliver_audit(ctx)
    ctx.write_text("reviews/overwatch.md", "The review looks good but no verdict line")
    result = create_ops.gate_overwatch(ctx)
    assert result == "reloop"
    assert "overwatch verdict is missing" in ctx.facts["validation"]


def test_gate_overwatch_code_reloop_spends_attempt(tmp_path):
    """A reloop caused by blocking findings calls repair() and increments attempts_used."""
    ctx = audit_context(tmp_path)
    deliver_audit(ctx)
    ctx.write_text("reviews/overwatch.md",
                   "BLOCKING: test failure\nVERDICT: FAIL")
    result = create_ops.gate_overwatch(ctx)
    assert result == "reloop"
    assert ctx.facts.get("attempts_used", 0) >= 1


# ── design-document contract: clause-level audit ─────────────────────────


def test_overwatch_audit_refuses_constraint_row_with_no_clauses(tmp_path):
    """audit_problems refuses a constraint row whose clause list is empty."""
    ctx = audit_context(tmp_path)
    deliver_audit(ctx)
    audit = ctx.read_json(evidence.AUDIT)
    # Find a constraint/correction row and remove its clauses.
    ev = ctx.read_json(evidence.EVIDENCE)
    constraint_ids = {item["id"] for item in ev["requirements"]
                      if item.get("kind") in ("correction", "constraint")}
    if not constraint_ids:
        # Add one so the test has something to check.
        Ledger(tmp_path).note("constraint", "Apply rule A to every caller")
        evidence.record(ctx, {"status": "pass", "command": "test", "stdout": "ok"})
        deliver_audit(ctx)
        audit = ctx.read_json(evidence.AUDIT)
        ev = ctx.read_json(evidence.EVIDENCE)
        constraint_ids = {item["id"] for item in ev["requirements"]
                          if item.get("kind") in ("correction", "constraint")}
    for row in audit["requirements"]:
        if row["id"] in constraint_ids:
            row.pop("clauses", None)
            break
    ctx.write_json(evidence.AUDIT, audit)
    problems = evidence.audit_problems(ctx)
    assert any("no clauses" in p for p in problems), f"expected clause check failure, got {problems}"


def test_overwatch_audit_refuses_clause_with_no_evidence(tmp_path):
    """audit_problems refuses a clause that lists no evidence."""
    ctx = audit_context(tmp_path)
    Ledger(tmp_path).note("constraint",
                          "descendants(container, continuation), without following "
                          "continuations and skipping retry edges as planRedo's walk does")
    evidence.record(ctx, {"status": "pass", "command": "test", "stdout": "ok"})
    deliver_audit(ctx)
    audit = ctx.read_json(evidence.AUDIT)
    ev = ctx.read_json(evidence.EVIDENCE)
    constraint_ids = {item["id"] for item in ev["requirements"]
                      if item.get("kind") in ("correction", "constraint")}
    # Find the constraint row and give it clauses with one missing evidence.
    for row in audit["requirements"]:
        if row["id"] in constraint_ids:
            row["clauses"] = [
                {"clause": "descendants(container, continuation)",
                 "evidence": [{"path": "artifact/answer.txt", "detail": "checked"}]},
                {"clause": "skipping retry edges",
                 "evidence": []},
            ]
            break
    ctx.write_json(evidence.AUDIT, audit)
    problems = evidence.audit_problems(ctx)
    assert any("skipping retry edges" in p and "no evidence" in p for p in problems), (
        f"expected clause evidence failure, got {problems}"
    )


def test_audit_instructions_ask_for_clauses():
    """The AUDIT_INSTRUCTIONS text tells the overwatch agent to list clauses."""
    assert "clauses" in evidence.AUDIT_INSTRUCTIONS
    assert "every separate" in evidence.AUDIT_INSTRUCTIONS or "every distinct" in evidence.AUDIT_INSTRUCTIONS


def test_fakemodel_writes_clauses_for_constraints(tmp_path):
    """FakeModel's audit writer produces clause entries for constraint rows."""
    ctx = audit_context(tmp_path)
    Ledger(tmp_path).note("constraint", "Apply rule A; also apply rule B")
    evidence.record(ctx, {"status": "pass", "command": "test", "stdout": "ok"})

    from tests_support import FakeModel
    model = FakeModel(["VERDICT: PASS"])
    model.complete(
        "Write intent-audit.json as ... " + evidence.AUDIT_INSTRUCTIONS,
        workdir=str(tmp_path),
    )
    audit = json.loads((tmp_path / evidence.AUDIT).read_text())
    ev = ctx.read_json(evidence.EVIDENCE)
    constraint_ids = {item["id"] for item in ev["requirements"]
                      if item.get("kind") in ("correction", "constraint")}
    for row in audit["requirements"]:
        if row["id"] in constraint_ids:
            assert "clauses" in row, f"FakeModel should write clauses for constraint row {row['id']}"
            assert len(row["clauses"]) >= 1
            for clause in row["clauses"]:
                assert clause.get("clause"), "clause text must not be empty"
                assert clause.get("evidence"), "clause evidence must not be empty"


# ── ask 4: failure message count matches attempts_used ────────────────────


def test_failure_message_count_matches_attempts_used(tmp_path):
    """repair()'s message and saved attempts_used agree for a run that exhausts its attempts."""
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "answer.txt").write_text("42")
    ctx = nn.RunContext(tmp_path, facts={"brief": "count", "artifact": str(root)},
                        knobs={"attempts": 3})
    ctx.write_json("brief.json", {"brief": ctx.facts["brief"]})
    Ledger.seed(tmp_path, ctx.facts["brief"])
    evidence.record(ctx, {"status": "pass"})
    # Simulate three repair calls: first two return Verdict, third raises.
    result1 = create_ops.repair(ctx, ["problem 1"])
    assert str(result1) == "reloop"
    assert ctx.facts["attempts_used"] == 1
    result2 = create_ops.repair(ctx, ["problem 2"])
    assert str(result2) == "reloop"
    assert ctx.facts["attempts_used"] == 2
    with pytest.raises(ValueError, match=r"check attempts exhausted after 3 check failure\(s\)"):
        create_ops.repair(ctx, ["problem 3"])
    assert ctx.facts["attempts_used"] == 3, (
        "the saved attempts_used must match the number in the failure message"
    )
    # repair() no longer writes facts.json (that is run_node's job in spine mode);
    # the in-memory value is what run_node would persist.


# ── ask 4: audit reloop end-to-end ───────────────────────────────────────


def test_evidence_map_problems_reach_the_design_overwatch_as_evidence(tmp_path, monkeypatch):
    """Design mode has no free audit round: a map that leaves a requirement uncovered is
    listed in the correction audit the overwatch reads, the overwatch runs once, nothing
    is spent, and its pass decides. (create keeps the round: create_ops.gate_overwatch.)"""
    from refactory_dsh.design import DesignMode
    from test_edit_design import _design_answers

    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))

    overwatch_prompts = []

    class AuditModel(FakeModel):
        def complete(self, prompt, *, system="", workdir=None, **workspace):
            text = super().complete(prompt, system=system, workdir=workdir, **workspace)
            if "You are the overwatch" in prompt:
                overwatch_prompts.append(prompt)
            if "You are the QA judge" in prompt:
                # The QA judge writes the map; this one leaves `design` uncovered.
                run_dir = tmp_path / "run"
                ev = json.loads((run_dir / evidence.EVIDENCE).read_text())
                rows = [{"id": item["id"], "requirement": item.get("text", ""),
                         "status": "pass",
                         "evidence": [{"path": "brief.json", "detail": "verified requirement"}]}
                        for item in ev["requirements"] if item["id"] != "design"]
                run_dir.joinpath(evidence.AUDIT).write_text(json.dumps({"requirements": rows}))
            return text

    model = AuditModel(_design_answers([render({"answer.py": "print(42)"})]))
    result = DesignMode().run(
        tmp_path / "run", model=model,
        verdicts={"gate_strategy": "proceed"},
        task={"name": "audit_reloop",
              "brief": "Deliver the answer and explanation",
              "check": "python answer.py"},
    )
    assert result.exit_ok, result.error
    assert len(overwatch_prompts) == 1, "the overwatch runs once"
    assert "design: no requirement evidence in intent audit" in overwatch_prompts[0]
    assert result.facts.get("attempts_used", 0) == 0


# ── dm-33: process notes ─────────────────────────────────────────────────


def test_process_note_does_not_become_requirement(tmp_path):
    """A constraint with scope='process' is excluded from requirements()
    and does not block gate_overwatch."""
    ctx = audit_context(tmp_path)
    Ledger(tmp_path).note("constraint", "add a row to the audit table", scope="process")
    evidence.record(ctx, {"status": "pass", "command": "test", "stdout": "42"})

    reqs = evidence.requirements(ctx)
    req_texts = [r.get("text", "") for r in reqs]
    assert not any("add a row" in t for t in req_texts), (
        "process note became a requirement"
    )

    deliver_audit(ctx)
    result = create_ops.gate_overwatch(ctx)
    assert str(result) == "proceed", f"process note blocked the gate: {result}"


def test_process_note_handling_is_recorded_in_evidence(tmp_path):
    """The evidence payload contains a process_notes list with the note."""
    ctx = audit_context(tmp_path)
    Ledger(tmp_path).note("constraint", "add a row to the audit table", scope="process")
    evidence.record(ctx, {"status": "pass", "command": "test", "stdout": "42"})

    ev = ctx.read_json(evidence.EVIDENCE)
    assert "process_notes" in ev, "evidence payload has no process_notes field"
    assert len(ev["process_notes"]) == 1
    assert ev["process_notes"][0]["text"] == "add a row to the audit table"
    assert ev["process_notes"][0]["scope"] == "process"


def test_product_note_still_requires_product_evidence(tmp_path):
    """A product constraint is still a requirement and fails when evidence is missing."""
    ctx = audit_context(tmp_path)
    Ledger(tmp_path).note("constraint", "do not add numpy")
    evidence.record(ctx, {"status": "pass", "command": "test", "stdout": "42"})

    reqs = evidence.requirements(ctx)
    assert any(r.get("text") == "do not add numpy" for r in reqs), (
        "product constraint is not a requirement"
    )

    deliver_audit(ctx, omitted={r["id"] for r in reqs if r.get("text") == "do not add numpy"})
    problems = evidence.audit_problems(ctx)
    assert any("no requirement evidence" in p for p in problems), (
        f"omitting product evidence did not fail: {problems}"
    )


def test_self_citation_still_refused_for_product_note(tmp_path):
    """intent-audit.json cannot be cited as evidence for a product requirement."""
    ctx = audit_context(tmp_path)
    Ledger(tmp_path).note("constraint", "do not add numpy")
    evidence.record(ctx, {"status": "pass", "command": "test", "stdout": "42"})

    # Deliver an audit that cites intent-audit.json for the constraint.
    from refactory_dsh.agent_delivery import begin, finish

    begin(ctx.workdir, "overwatch", "Audit each ask", ["reviews/overwatch.md", evidence.AUDIT])
    ev = ctx.read_json(evidence.EVIDENCE)
    rows = []
    for item in ev["requirements"]:
        row = {"id": item["id"], "requirement": item.get("text", ""),
               "status": "pass",
               "evidence": [{"path": "artifact/answer.txt", "detail": "verified"}]}
        if item.get("kind") in ("correction", "constraint"):
            # Cite intent-audit.json for the constraint — should be refused.
            row["evidence"] = [{"path": evidence.AUDIT, "detail": "self-cited"}]
            row["clauses"] = [{"clause": item.get("text", ""),
                               "evidence": [{"path": evidence.AUDIT,
                                              "detail": "self-cited"}]}]
        rows.append(row)
    ctx.write_json(evidence.AUDIT, {"requirements": rows})
    ctx.write_text("reviews/overwatch.md", "VERDICT: PASS")
    finish(ctx.workdir, "overwatch")

    problems = evidence.audit_problems(ctx)
    assert any("missing or invalid evidence reference" in p for p in problems), (
        f"self-citation was not refused: {problems}"
    )


def test_console_steer_passes_scope(tmp_path):
    """The console's --scope argument reaches ledger.note()."""
    from refactory_dsh import console

    root = tmp_path / "run"
    root.mkdir()
    Ledger.seed(root, "build a mode for circle packing")

    payload = console._steer(
        root, note="add a row to the audit", kind="constraint", scope="process"
    )
    entries = payload["entries"]
    process_entries = [e for e in entries if e.get("scope") == "process"]
    assert len(process_entries) == 1
    assert process_entries[0]["text"] == "add a row to the audit"


def test_gui_tool_steer_exposes_scope():
    """The coverage.json chat mapping includes scope so the GUI tool can pass it."""
    coverage_path = Path(__file__).resolve().parent.parent / "refactory_dsh" / "data" / "coverage.json"
    coverage_data = json.loads(coverage_path.read_text())
    steer_chat = coverage_data["operations"]["steer"]["chat"]
    assert "scope" in steer_chat["fields"], "scope not in steer chat fields"
    assert steer_chat["fields"]["scope"] == "scope"
    assert "scope" in steer_chat["order"], "scope not in steer chat order"


def test_audit_instructions_mention_process_notes():
    """The overwatch audit instructions tell the agent about process notes."""
    assert "process_notes" in evidence.AUDIT_INSTRUCTIONS
    assert "followed" in evidence.AUDIT_INSTRUCTIONS


@pytest.mark.parametrize("cls", [CreateMode, EditMode, CreateFastMode, EditFastMode])
def test_native_smoke_gate_keeps_the_requested_execution_timeout(cls):
    from refactory_dsh.lowering import to_workflow
    from refactory_dsh.ir import Workflow

    state = cls().state_dict()
    state["knobs"]["smoke_timeout_s"]["value"] = 1200
    workflow = to_workflow(state)
    encoded = workflow.to_dict()
    restored = Workflow.from_dict(encoded)
    assert restored.nodes["smoke"].command_timeout_s >= 2400


def test_fast_mode_allows_all_three_declared_repairs(tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    good = good_files("three_repairs")
    bad = {**good, "ops.py": good["ops.py"].replace('target = int(ctx.knob("target", 3))', 'raise ValueError("broken")')}
    mode = EditFastMode()
    state = mode.state_dict()
    state["knobs"]["single_shot"]["value"] = True
    state["knobs"]["repair_attempts"]["value"] = 3
    mode.load_state_dict(state)
    run = mode.run(tmp_path / "run", task={"name": "three_repairs", "brief": "count"},
                   model=FakeModel([render(bad)] * 3 + [render(good), QA_PASS]))
    assert run.exit_ok, run.error
    assert run.facts["repairs_used"] == 3


def test_full_qa_repair_cannot_spend_final_validation_reserve(tmp_path, monkeypatch):
    from refactory_dsh.budgets import RunBudget

    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    answers = create_script("full_reserved", [render(good_files("full_reserved"))])
    answers[5:8] = ["BLOCKING: intentional check defect", "code review passed", "QA passed"]

    class MeteredModel(FakeModel):
        def with_budget(self, budget):
            self.budget = budget
            return self

        def complete(self, prompt, *, system="", workdir=None, **workspace):
            index = len(self.prompts)
            self.budget.record_usage(f"request-{index}", 60 if index == 4 else 10 if index >= 5 else 0)
            return super().complete(prompt, system=system, workdir=workdir, **workspace)

    mode = CreateMode()
    state = mode.state_dict()
    state["knobs"]["single_shot"]["value"] = True
    mode.load_state_dict(state)
    client = MeteredModel(answers)
    run = mode.run(tmp_path / "run", task={"name": "full_reserved", "brief": "count"}, model=client,
                   verdicts={"gate_strategy": "proceed"},
                   resource_limits={"max_tokens": 100, "validation_reserve_tokens": 20})
    assert not run.exit_ok and "reserve retained" in run.error
    assert len(client.prompts) == 8
    assert RunBudget.open(tmp_path / "run").snapshot()["phase"] == "work"


def test_unresolved_runtime_defect_stops_at_shared_repair_bound(tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    files = good_files("never_passes")
    files["ops.py"] = files["ops.py"].replace('target = int(ctx.knob("target", 3))', 'raise ValueError("still broken")')
    mode = CreateFastMode()
    state = mode.state_dict()
    state["knobs"]["single_shot"]["value"] = True
    mode.load_state_dict(state)
    client = FakeModel([PLAN, render(files), render(files)])
    run = mode.run(tmp_path / "run", task={"name": "never_passes", "brief": "count"}, model=client)
    assert not run.exit_ok and "after 1 repair(s)" in run.error
    assert len(client.prompts) == 3
    assert not (tmp_path / "run/create.json").exists()


def test_ceo_redo_preserves_attempt_facts(tmp_path):
    """command_reset with preserveAttemptFacts=True keeps attempt counters."""
    from refactory_dsh import items, lowering, nn

    graph = nn.Graph()
    graph.add(nn.Node('emit', 'op', call='loop_ops:write_number',
                      produces=('emit', 'strategy_used')))
    graph.add(nn.Node('check', 'op', call='loop_ops:write_number',
                      reads=('validation.txt',), writes=('validation.txt',),
                      produces=('validation', 'attempts_used')))
    graph.connect('emit', 'check')
    lowering.write_declarations({'graph': graph.to_state()}, tmp_path)
    (tmp_path / 'facts.json').write_text(json.dumps({
        'attempts_used': 2, 'validation': 'old', 'emit': 'prior', 'strategy_used': 'lean'
    }))
    (tmp_path / '.refactory').mkdir(exist_ok=True)
    request = {
        'requested': 'emit', 'start': 'emit',
        'nodes': ['emit', 'check'],
        'preserveAttemptFacts': True,
    }
    items.command_reset(tmp_path, request)
    facts = json.loads((tmp_path / 'facts.json').read_text())
    # The gate's attempt count is kept, so its limit still ends the run.
    assert facts['attempts_used'] == 2
    # Everything else the reopened nodes produced is cleared, including a fact that
    # only shares the suffix (lean_pack's strategy_used is not a count).
    assert 'validation' not in facts
    assert 'emit' not in facts
    assert 'strategy_used' not in facts


def test_graph_control_rerun_resets_attempt_facts(tmp_path):
    """command_reset without preserveAttemptFacts clears all facts including attempts."""
    from refactory_dsh import items, lowering, nn

    graph = nn.Graph()
    graph.add(nn.Node('emit', 'op', call='loop_ops:write_number',
                      produces=('emit',)))
    graph.add(nn.Node('check', 'op', call='loop_ops:write_number',
                      reads=('validation.txt',), writes=('validation.txt',),
                      produces=('validation', 'attempts_used')))
    graph.connect('emit', 'check')
    lowering.write_declarations({'graph': graph.to_state()}, tmp_path)
    (tmp_path / 'facts.json').write_text(json.dumps({
        'attempts_used': 2, 'validation': 'old', 'emit': 'prior'
    }))
    (tmp_path / '.refactory').mkdir(exist_ok=True)
    request = {
        'requested': 'emit', 'start': 'emit',
        'nodes': ['emit', 'check'],
    }
    items.command_reset(tmp_path, request)
    facts = json.loads((tmp_path / 'facts.json').read_text())
    assert 'attempts_used' not in facts
    assert 'validation' not in facts
    assert 'emit' not in facts


@pytest.mark.parametrize("artifact", [False, True])
def test_full_graph_first_pass_supplies_every_declared_worker_input(tmp_path, monkeypatch, artifact):
    """Native worker_node enforces reads; the walker alone used to hide this defect."""
    from refactory_dsh import agent_delivery
    from refactory_dsh.design import DesignMode
    from test_edit_design import _design_answers

    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    deliver = agent_delivery.begin

    def strict_begin(workdir, node, prompt, writes, **kwargs):
        for name in kwargs.get("reads", []):
            assert (Path(workdir) / name).exists(), f"{node}: required worker input is missing: {name}"
        return deliver(workdir, node, prompt, writes, **kwargs)

    monkeypatch.setattr(agent_delivery, "begin", strict_begin)
    mode = DesignMode() if artifact else CreateMode()
    if artifact:
        answers = _design_answers([render({"answer.py": "print(42)"})])
    else:
        state = mode.state_dict()
        state["knobs"]["single_shot"]["value"] = True
        mode.load_state_dict(state)
        answers = create_script("first_pass", [render(good_files("first_pass"))])
    run = mode.run(tmp_path / "run", model=FakeModel(answers),
                   task={"name": "first_pass", "brief": "produce a checked answer", "check": "python answer.py"},
                   verdicts={"gate_strategy": "proceed"})
    assert run.exit_ok, run.error
