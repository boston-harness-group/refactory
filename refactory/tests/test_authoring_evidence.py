"""Behavioral controls for canonical execution, repair and faithful sign-off."""
import json
from pathlib import Path

import pytest

from refactory_dsh import authoring_evidence as evidence, create_ops, nn, sandbox
from refactory_dsh.agent_delivery import begin, finish
from refactory_dsh.create import CreateMode
from refactory_dsh.edit import EditMode
from refactory_dsh.fast import CreateFastMode, EditFastMode
from refactory_dsh.steering import Ledger
from tests_support import FakeModel, create_script, good_files, render
from test_fast_modes import PLAN, QA_PASS


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
        rows.append(row)
    ctx.write_json(evidence.AUDIT, {"requirements": rows})
    ctx.write_text("reviews/overwatch.md", "VERDICT: PASS")
    finish(ctx.workdir, "overwatch")


def test_missing_requirement_and_reviewer_waiver_cannot_pass(tmp_path):
    ctx = audit_context(tmp_path)
    deliver_audit(ctx, omitted={"criterion-2"})
    assert create_ops.gate_overwatch(ctx) == "reloop"
    assert "criterion-2: no requirement evidence" in ctx.facts["validation"]
    deliver_audit(ctx, exclusion="reviewer says out of scope")
    assert create_ops.gate_overwatch(ctx) == "reloop"
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


def test_audit_instructions_tell_probing_reviewers_to_stage_fixtures_in_the_run():
    """A reviewer told to probe but not where to put fixtures cites /tmp and loses the cycle.

    The audit validator accepts only files inside the run directory or the delivered
    artifact; without staging instructions a spot-check fixture written to /tmp is
    uncitable, and its evidence dies with the gate verdict.
    """
    from refactory_dsh.design import DesignMode

    overwatch = DesignMode().build().nodes["overwatch"]
    prompt = overwatch.prompt_template
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

        def complete(self, prompt, *, system="", workdir=None):
            index = len(self.prompts)
            self.budget.record_usage(f"request-{index}", 60 if index == 4 else 10 if index >= 5 else 0)
            return super().complete(prompt, system=system, workdir=workdir)

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
