"""Tests for feedback re-fill: declared feedback keys are re-derived from persisted files.

Integration tests drive the redo through worker_node with a cleared fact, and
the design_fast route through focused_check → gate_check → worker_node's
prompt.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _make_workdir(tmp_path: Path) -> Path:
    """Create a minimal run directory."""
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / ".refactory").mkdir()
    return workdir


def test_refill_from_check_record(tmp_path):
    """Validation is re-derived from check-record.json."""
    from refactory_dsh.feedback_refill import refill

    workdir = _make_workdir(tmp_path)
    (workdir / "check-record.json").write_text(json.dumps({
        "status": "fail",
        "validation": "## Failing tests\n\n- test_foo.py::test_bar",
    }))
    values = {"name": "test", "brief": "test brief"}
    updated, refilled = refill(workdir, ["validation"], values)
    assert "validation" in refilled
    assert "Failing tests" in updated["validation"]


def test_refill_from_initial(tmp_path):
    """A redo before QA re-fills from feedback-initial.json, not refused."""
    from refactory_dsh.feedback_refill import refill

    workdir = _make_workdir(tmp_path)
    (workdir / "feedback-initial.json").write_text(json.dumps({
        "validation": "(no check has run yet)",
        "qa_gate": "(no review has refused anything yet)",
        "overwatch_review": "(no audit has refused anything yet)",
    }))
    values = {"name": "test", "brief": "test brief"}
    updated, refilled = refill(workdir, ["validation", "qa_gate", "overwatch_review"], values)
    assert set(refilled) == {"validation", "qa_gate", "overwatch_review"}
    assert updated["validation"] == "(no check has run yet)"


def test_refill_refuses_when_nothing_exists(tmp_path):
    """When no gate file or initial record exists, refill refuses."""
    from refactory_dsh.feedback_refill import refill

    workdir = _make_workdir(tmp_path)
    values = {"name": "test"}
    with pytest.raises(ValueError, match="unfilled"):
        refill(workdir, ["validation"], values)


def test_refill_skips_filled_keys(tmp_path):
    """Keys already in values are not re-filled."""
    from refactory_dsh.feedback_refill import refill

    workdir = _make_workdir(tmp_path)
    values = {"validation": "already set"}
    updated, refilled = refill(workdir, ["validation"], values)
    assert refilled == []
    assert updated["validation"] == "already set"


def test_refill_validation_from_newer_validation_txt(tmp_path):
    """When validation.txt is newer than check-record.json, use it."""
    from refactory_dsh.feedback_refill import refill
    import time

    workdir = _make_workdir(tmp_path)
    (workdir / "check-record.json").write_text(json.dumps({
        "status": "fail", "validation": "old digest",
    }))
    time.sleep(0.05)
    (workdir / "validation.txt").write_text("newer verification feedback\n")
    values = {}
    updated, refilled = refill(workdir, ["validation"], values)
    assert "validation" in refilled
    assert "newer verification" in updated["validation"]


def test_refill_qa_gate_from_file(tmp_path):
    """qa_gate is re-filled from reviews/qa-gate.txt."""
    from refactory_dsh.feedback_refill import refill

    workdir = _make_workdir(tmp_path)
    (workdir / "reviews").mkdir()
    (workdir / "reviews" / "qa-gate.txt").write_text("some QA feedback\n")
    values = {}
    updated, refilled = refill(workdir, ["qa_gate"], values)
    assert "qa_gate" in refilled
    assert "QA feedback" in updated["qa_gate"]


def test_refill_overwatch_from_file(tmp_path):
    """overwatch_review is re-filled from reviews/overwatch-gate.txt."""
    from refactory_dsh.feedback_refill import refill

    workdir = _make_workdir(tmp_path)
    (workdir / "reviews").mkdir()
    (workdir / "reviews" / "overwatch-gate.txt").write_text("audit feedback\n")
    values = {}
    updated, refilled = refill(workdir, ["overwatch_review"], values)
    assert "overwatch_review" in refilled
    assert "audit feedback" in updated["overwatch_review"]


def test_refill_repair_feedback_from_file(tmp_path):
    """repair_feedback is re-filled from repair.txt."""
    from refactory_dsh.feedback_refill import refill

    workdir = _make_workdir(tmp_path)
    (workdir / "repair.txt").write_text("repair notes\n")
    values = {}
    updated, refilled = refill(workdir, ["repair_feedback"], values)
    assert "repair_feedback" in refilled
    assert "repair notes" in updated["repair_feedback"]


def test_feedback_refill_entry_point():
    """The refactory.feedback_refill entry point is installed and resolves."""
    import importlib.metadata
    eps = importlib.metadata.entry_points(group='refactory.feedback_refill')
    assert len(list(eps)) > 0, "refactory.feedback_refill entry point should be installed"
    ep = list(eps)[0]
    fn = ep.load()
    assert callable(fn)


def test_fill_feedback_reads_declarations(tmp_path):
    """fill_feedback reads feedback_keys from declarations.json when not passed."""
    from refactory_dsh.agent_delivery import fill_feedback

    workdir = _make_workdir(tmp_path)
    decls = {
        "emit": {
            "kind": "agent", "reads": [], "writes": ["emit.txt"],
            "produces": ["emit"], "resource_role": "work",
            "feedback_keys": ["validation", "qa_gate"],
        }
    }
    (workdir / ".refactory" / "declarations.json").write_text(json.dumps(decls))
    (workdir / "feedback-initial.json").write_text(json.dumps({
        "validation": "(no check has run yet)",
        "qa_gate": "(no review has refused anything yet)",
    }))
    values = {"name": "test"}
    updated, refilled = fill_feedback(workdir, "emit", values)
    assert "validation" in refilled
    assert "qa_gate" in refilled


def test_redo_after_refusal_refills_from_check_record(tmp_path):
    """A redo after a materialise refusal re-fills from check-record.json's stored validation."""
    from refactory_dsh.feedback_refill import refill

    workdir = _make_workdir(tmp_path)
    refusal_msg = "materialise could not apply emit.txt: bad edit"
    (workdir / "check-record.json").write_text(json.dumps({
        "status": "refused", "validation": refusal_msg,
    }))
    values = {}
    updated, refilled = refill(workdir, ["validation"], values)
    assert "validation" in refilled
    assert updated["validation"] == refusal_msg


def test_redo_after_guardedness_refills_from_check_record(tmp_path):
    """A redo after a guardedness block re-fills from check-record.json."""
    from refactory_dsh.feedback_refill import refill

    workdir = _make_workdir(tmp_path)
    guarded_msg = "BLOCKING: these new tests pass without the change: test_foo"
    (workdir / "check-record.json").write_text(json.dumps({
        "status": "guarded", "validation": guarded_msg,
    }))
    values = {}
    updated, refilled = refill(workdir, ["validation"], values)
    assert "validation" in refilled
    assert updated["validation"] == guarded_msg


def test_refill_validation_round_trip(tmp_path):
    """Run check_artifact, record validation fact, clear fact, re-fill:
    the re-filled value matches the original via check-record.json.
    """
    from refactory_dsh.feedback_refill import refill

    workdir = _make_workdir(tmp_path)
    original_val = "## Failing tests\n\ntest_foo.py::test_bar — AssertionError"
    (workdir / "check-record.json").write_text(json.dumps({
        "status": "fail", "validation": original_val,
    }))
    # Simulate clearing the fact (as planRedo does)
    values = {}
    updated, refilled = refill(workdir, ["validation"], values)
    assert "validation" in refilled
    assert updated["validation"] == original_val, "re-filled value must match the original exactly"


# ── redo through worker_node ──────────────────────────────────────────────────

def test_digest_present_after_redo(tmp_path, monkeypatch):
    """Via worker_node with a cleared fact, validation is re-derived from
    check-record.json, not literal {{validation}}.

    The redo test lowers the real mode with write_declarations.
    """
    from refactory_dsh import worker_node
    from refactory_dsh.run import NODE_ENV
    from refactory_dsh.design import DesignMode
    from refactory_dsh.lowering import write_declarations
    from refactory_dsh.nn.executor import RunContext

    run = tmp_path / "run"
    run.mkdir()
    (run / ".refactory").mkdir(parents=True)

    # Lower the real design mode to get declarations.json
    mode = DesignMode()
    write_declarations(mode.state_dict(), run)

    # Write check-record.json with a known validation string
    validation_text = "## Failing tests\n\ntest_foo.py::test_bar — AssertionError: 1 != 2"
    (run / "check-record.json").write_text(json.dumps({
        "status": "fail", "validation": validation_text,
    }))
    # Write feedback-initial.json
    (run / "feedback-initial.json").write_text(json.dumps({
        "validation": "(no check has run yet)",
        "qa_gate": "(no review has refused anything yet)",
        "overwatch_review": "(no audit has refused anything yet)",
    }))

    # Write required input files
    (run / "brief.json").write_text(json.dumps({"name": "test", "brief": "test", "check": "echo ok"}))
    (run / "design.md").write_text("# Design\n")
    (run / "check.txt").write_text("failed\n")

    # Simulate a redo: facts are cleared (no validation key)
    (run / "facts.json").write_text(json.dumps({
        "name": "test", "brief": "test", "check": "echo ok",
    }))

    captured_prompt = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def for_node(self, node, role="worker"):
            return self
        def complete(self, prompt, workdir=None):
            captured_prompt.append(prompt)
            p = Path(workdir) if workdir else run
            (p / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
            return "done"
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "dsh"}

    spec = {
        "schema_version": 1,
        "node_id": "emit",
        "prompt": "Write the artifact `{{name}}`\n{{validation}}\n{{qa_gate}}\n{{overwatch_review}}",
        "reads": ["brief.json", "design.md", "check.txt", "check-record.json"],
        "writes": ["emit.txt"],
        "model_context": {"provider": "test", "model": "test"},
        "feedback_keys": ["validation", "qa_gate", "overwatch_review"],
    }

    manifest = {"id": "emit", "type": "AgentNode", "command": "echo"}
    monkeypatch.setenv(NODE_ENV, json.dumps(manifest))
    monkeypatch.chdir(run)

    with patch("refactory_dsh.model.DshModelClient", FakeClient):
        with patch("refactory_dsh.agent_delivery.begin", return_value={"prompt": "test", "node": "emit", "status": "pending"}):
            with patch("refactory_dsh.agent_delivery.finish", return_value={"node": "emit", "status": "delivered"}):
                receipt = worker_node.execute(spec, run)

    assert receipt["status"] == "delivered"
    assert captured_prompt, "model should have been called"
    # The validation should have been re-derived, not literal {{validation}}
    prompt_text = captured_prompt[0]
    assert "{{validation}}" not in prompt_text, \
        "validation placeholder must not appear literally after re-fill"


def test_unfilled_feedback_key_refuses_dispatch(tmp_path, monkeypatch):
    """When feedback key is absent and no gate file or initial record exists,
    dispatch is refused with a message naming the key.
    """
    from refactory_dsh import worker_node
    from refactory_dsh.run import NODE_ENV

    run = tmp_path / "run"
    run.mkdir()
    (run / ".refactory").mkdir(parents=True)
    (run / "facts.json").write_text(json.dumps({"name": "test"}))

    spec = {
        "schema_version": 1,
        "node_id": "emit",
        "prompt": "Write {{validation}}",
        "reads": [],
        "writes": ["emit.txt"],
        "model_context": {"provider": "test", "model": "test"},
        "feedback_keys": ["validation"],
    }

    manifest = {"id": "emit", "type": "AgentNode", "command": "echo"}
    monkeypatch.setenv(NODE_ENV, json.dumps(manifest))
    monkeypatch.chdir(run)

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def for_node(self, node, role="worker"):
            return self
        def complete(self, prompt, workdir=None):
            return "done"
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "dsh"}

    with patch("refactory_dsh.model.DshModelClient", FakeClient):
        with pytest.raises(ValueError, match="unfilled"):
            worker_node.execute(spec, run)


def test_redo_before_check_refills_from_initial(tmp_path):
    """A strategy redo before any check re-fills from feedback-initial.json."""
    from refactory_dsh.feedback_refill import refill

    workdir = _make_workdir(tmp_path)
    (workdir / "feedback-initial.json").write_text(json.dumps({
        "validation": "(no check has run yet)",
        "qa_gate": "(no review has refused anything yet)",
        "overwatch_review": "(no audit has refused anything yet)",
    }))
    # No check-record.json, no validation.txt — this is before any check
    values = {}
    updated, refilled = refill(workdir, ["validation", "qa_gate", "overwatch_review"], values)
    assert set(refilled) == {"validation", "qa_gate", "overwatch_review"}
    assert updated["validation"] == "(no check has run yet)"


# ── walker re-fill ────────────────────────────────────────────────────────────

def test_walker_refill_does_not_raise(tmp_path, monkeypatch):
    """The walker uses _internal_fact for re-filled feedback keys, so emit
    does not need to declare them as produced facts.

    Without the _internal_fact fix (nn/executor.py ~687), set_fact would add
    validation to _produced, and _check_produces would raise ExecutionError
    because validation is not in _declared_facts.
    """
    from refactory_dsh import nn
    from refactory_dsh.nn import Node, Graph

    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / ".refactory").mkdir()

    # Write check-record.json for re-derivation
    (workdir / "check-record.json").write_text(json.dumps({
        "status": "fail",
        "validation": "## Failing tests\n\ntest_foo.py::test_bar",
    }))

    class WalkerFakeModel:
        def complete(self, prompt, *, system="", workdir=None):
            if workdir:
                (Path(workdir) / "emit.txt").write_text("===FILE: tool.py===\nresult\n")
            return "done"
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "dsh"}

    graph = Graph()
    graph.add(Node("emit", "agent",
                   prompt_template="Write {{validation}}",
                   writes=("emit.txt",),
                   reads=("check-record.json",),
                   produces=("emit",),
                   feedback_keys=("validation",)))

    model = WalkerFakeModel()
    ctx = nn.RunContext(workdir, facts={"name": "test"}, model=model)
    # emit declares emit.txt as a write and "emit" as a produced fact.
    # validation is NOT in declared_facts — that is the point:
    # _internal_fact bypasses the declaration check.
    ctx.enforce_writes = False

    monkeypatch.setattr("refactory_dsh.agent_delivery.begin",
                        lambda workdir, node, prompt, writes, **kw:
                        {"prompt": prompt, "node": node, "status": "pending"})
    monkeypatch.setattr("refactory_dsh.agent_delivery.finish",
                        lambda workdir, node, **kw:
                        {"node": node, "status": "delivered"})

    executor = nn.Executor(graph, ctx)
    executor.run()

    # If _internal_fact was not used, set_fact would raise ExecutionError
    # because validation is not in _declared_facts.
    assert "validation" in ctx.facts
    assert "test_bar" in ctx.facts["validation"]


def test_worker_refill_without_spec_keys(tmp_path, monkeypatch):
    """When the worker spec has no feedback_keys, fill_feedback reads
    the node's keys from declarations.json and still re-fills.

    This pins the lowering round-trip: a real lower_workflow spec that
    drops feedback_keys must still be re-filled through the worker route.
    """
    from refactory_dsh import worker_node
    from refactory_dsh.run import NODE_ENV
    from refactory_dsh.design import DesignMode
    from refactory_dsh.lowering import write_declarations

    run = tmp_path / "run"
    run.mkdir()
    (run / ".refactory").mkdir(parents=True)

    # Lower the real design mode to get declarations.json with feedback_keys
    mode = DesignMode()
    write_declarations(mode.state_dict(), run)

    # Write check-record.json with a known validation string
    validation_text = "## Failing tests\n\ntest_foo.py::test_bar — AssertionError"
    (run / "check-record.json").write_text(json.dumps({
        "status": "fail", "validation": validation_text,
    }))
    (run / "feedback-initial.json").write_text(json.dumps({
        "validation": "(no check has run yet)",
        "qa_gate": "(no review has refused anything yet)",
        "overwatch_review": "(no audit has refused anything yet)",
    }))

    # Write required input files
    (run / "brief.json").write_text(json.dumps({
        "name": "test", "brief": "test", "check": "echo ok",
    }))
    (run / "design.md").write_text("# Design\n")
    (run / "check.txt").write_text("failed\n")

    # facts.json without validation (simulating a redo)
    (run / "facts.json").write_text(json.dumps({
        "name": "test", "brief": "test", "check": "echo ok",
    }))

    captured_prompt = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def for_node(self, node, **kw):
            return self
        def complete(self, prompt, workdir=None):
            captured_prompt.append(prompt)
            p = Path(workdir) if workdir else run
            (p / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
            return "done"
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "dsh"}

    # Spec WITHOUT feedback_keys — the worker must read them from declarations.json
    spec = {
        "schema_version": 1,
        "node_id": "emit",
        "prompt": "Write {{validation}}\n{{qa_gate}}\n{{overwatch_review}}",
        "reads": ["brief.json", "design.md", "check.txt", "check-record.json"],
        "writes": ["emit.txt"],
        "model_context": {"provider": "test", "model": "test"},
    }

    manifest = {"id": "emit", "type": "AgentNode", "command": "echo"}
    monkeypatch.setenv(NODE_ENV, json.dumps(manifest))
    monkeypatch.chdir(run)

    with patch("refactory_dsh.model.DshModelClient", FakeClient):
        with patch("refactory_dsh.agent_delivery.begin",
                   return_value={"prompt": "test", "node": "emit", "status": "pending"}):
            with patch("refactory_dsh.agent_delivery.finish",
                       return_value={"node": "emit", "status": "delivered"}):
                receipt = worker_node.execute(spec, run)

    assert receipt["status"] == "delivered"
    assert captured_prompt, "model should have been called"
    # The prompt must not contain literal {{validation}}
    prompt_text = captured_prompt[0]
    assert "{{validation}}" not in prompt_text, \
        "validation must be re-filled even when spec has no feedback_keys"


# ── design_fast full route ────────────────────────────────────────────────────

def test_digest_design_fast_full_route(tmp_path, monkeypatch):
    """The digest reaches the builder through the full design_fast route:
    focused_check → gate_check → worker_node's prompt.
    """
    from refactory_dsh import fast_ops, nn
    from refactory_dsh.sandbox import SandboxResult
    from refactory_dsh import repo_change

    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / ".refactory").mkdir()

    # Write the required files
    (run / "brief.json").write_text(json.dumps({"name": "test", "brief": "test", "check": "echo fail"}))
    (run / "emit.txt").write_text("===FILE: answer.py===\nassert False\n")
    (run / "focused-check.json").write_text("{}")
    (run / "focused-source.json").write_text("{}")
    (run / "authoring-evidence.json").write_text("{}")
    (run / "source-manifest.json").write_text(json.dumps({"source_project": "", "files": {}}))

    # Step 1: focused_check with a failing check
    fake_result = SandboxResult(ok=False, returncode=1, elapsed_s=1.0,
                                 stdout="FAILED tests/test_foo.py::test_bar - AssertionError: 1 != 2\n1 failed\n",
                                 stderr="")
    monkeypatch.setattr(repo_change, "run_check", lambda ctx, cmd, timeout: fake_result)
    from refactory_dsh import base_check, authoring_evidence
    monkeypatch.setattr(base_check, "compute_test_guardedness", lambda *a, **k: None)
    monkeypatch.setattr(authoring_evidence, "record", lambda *a, **k: None)
    monkeypatch.setattr(authoring_evidence, "stop_unrepairable", lambda *a, **k: None)

    ctx = nn.RunContext(run, facts={"check": "echo fail", "name": "test", "brief": "test",
                                    "artifact": str(run)},
                        knobs={"check_timeout_s": 60, "repair_attempts": 1})
    ctx._declared = ("focused-check.json", "focused-source.json", "authoring-evidence.json",
                     "check-record.json", "check.txt")
    ctx._declared_facts = ("focused_check", "output_path", "review_files", "runtime_evidence",
                          "inherited_failures")
    ctx.enforce_writes = False

    fast_ops.focused_check(ctx)

    # Step 2: Check that check-record.json has the digest
    cr = json.loads((run / "check-record.json").read_text())
    assert cr["status"] == "fail"
    assert cr.get("digest"), "check-record.json must have a digest"
    assert "test_bar" in cr.get("digest", ""), "digest must contain the failing test ID"

    # Step 3: gate_check reads the digest and routes to the builder
    (run / "fast-report.json").write_text("{}")
    (run / "repair.txt").write_text("(initial)\n")

    gc_ctx = nn.RunContext(run, facts={**ctx.facts, "focused_check": json.dumps({"status": "fail"}),
                                       "repair_feedback": "(initial)", "repairs_used": 0},
                           knobs={"repair_attempts": 1})
    gc_ctx._declared = ("fast-report.json", "repair.txt")
    gc_ctx._declared_facts = ("repair_feedback", "repairs_used",
                             "check_repairs_used")
    gc_ctx.enforce_writes = False

    result = fast_ops.gate_check(gc_ctx)
    # The result should be a reloop with the digest in the detail
    if hasattr(result, 'reason'):
        assert "test_bar" in result.reason, \
            "gate_check must pass the digest (not raw output) to the builder"
    else:
        repair_text = (run / "repair.txt").read_text()
        assert "test_bar" in repair_text, \
            "the repair feedback must contain the digest with the failing test ID"
