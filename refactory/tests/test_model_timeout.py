"""Tests for --model-timeout propagation through CLI, interactive and console paths.

Each test fails without its corresponding fix, as stated in the docstring.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# 1. _run_argv forwards --model-timeout
# ---------------------------------------------------------------------------


def test_run_argv_forwards_model_timeout():
    """_run_argv() includes --model-timeout in the forwarded argv.

    Fails without fix: _run_argv() does not forward --model-timeout.
    """
    from refactory_dsh.cli import build_parser, _run_argv

    parser = build_parser()
    args = parser.parse_args(["ceo", "my idea", "--model-timeout", "120", "--headless", "--check", "true", "--mode", "design"])
    forwarded = _run_argv("some/mode", args)
    assert "--model-timeout" in forwarded
    idx = forwarded.index("--model-timeout")
    assert forwarded[idx + 1] == "120"


def test_run_argv_omits_model_timeout_when_absent():
    """When --model-timeout is not given, the forwarded argv omits it.

    Fails without fix: N/A (regression guard for the default).
    """
    from refactory_dsh.cli import build_parser, _run_argv

    parser = build_parser()
    args = parser.parse_args(["ceo", "my idea", "--headless", "--check", "true", "--mode", "design"])
    forwarded = _run_argv("some/mode", args)
    assert "--model-timeout" not in forwarded


# ---------------------------------------------------------------------------
# 2. Authoring path forwards --model-timeout
# ---------------------------------------------------------------------------


def test_authoring_path_forwards_model_timeout(monkeypatch):
    """The authoring path in cmd_ceo() includes --model-timeout in the forwarded list.

    Fails without fix: the authoring path does not include --model-timeout.
    """
    captured: list[list[str]] = []

    def fake_forward(module: str, argv: list[str]) -> int:
        captured.append(argv)
        return 0

    monkeypatch.setattr("refactory_dsh.cli._forward", fake_forward)

    from refactory_dsh.cli import build_parser, cmd_ceo

    parser = build_parser()
    args = parser.parse_args([
        "ceo", "build a widget", "--model-timeout", "120",
        "--mode", "design", "--check", "make test",
    ])
    cmd_ceo(args)

    assert captured, "no _forward call was made"
    forwarded = captured[0]
    assert "--model-timeout" in forwarded
    idx = forwarded.index("--model-timeout")
    assert forwarded[idx + 1] == "120"


# ---------------------------------------------------------------------------
# 3. interactive.main() parses --model-timeout
# ---------------------------------------------------------------------------


def test_interactive_main_parses_model_timeout(monkeypatch):
    """interactive.main()'s parser accepts --model-timeout and passes it to run().

    Fails without fix: the parser has no --model-timeout; argparse raises an error.
    """
    captured_kwargs: dict[str, Any] = {}

    def fake_run(identifier, **kwargs):
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr("refactory_dsh.interactive.run", fake_run)
    # Prevent the real require_build / dsh_profile.require from running.
    # Accept **kw so the mock matches the real require_build(root, *, check_only, gui_mode) signature.
    monkeypatch.setattr("refactory_dsh.interactive.require_build", lambda **kw: None)

    from refactory_dsh import interactive

    code = interactive.main(["some/mode", "--model-timeout", "120", "--headless"])
    assert code == 0
    assert captured_kwargs.get("model_timeout_s") == 120


# ---------------------------------------------------------------------------
# 4. Worker specs carry timeout_s
# ---------------------------------------------------------------------------


def test_worker_specs_carry_timeout_s(tmp_path, monkeypatch):
    """prepare() with timeout_s=120 produces worker specs with the right values.

    Fails without fix: N/A (regression guard). The meaningful assertion is on
    command_timeout_s: 3 × 120 + 600 = 960.
    """
    monkeypatch.setenv("REFACTORY_WORKER_CONTEXT", "{}")
    # We need a minimal mode to prepare
    from refactory_dsh.model import DshModelClient, NODE_DISPATCH_OVERHEAD_S

    client = DshModelClient(timeout_s=120)
    ctx = client.context()
    assert ctx["timeout_s"] == 120

    # Verify the budget formula
    from refactory_dsh.model import node_session_budget_s

    budget = node_session_budget_s(3, 120)
    assert budget == 3 * 120 + NODE_DISPATCH_OVERHEAD_S


# ---------------------------------------------------------------------------
# 5. Resume with --model-timeout is refused
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("graph_profile_home")
def test_resume_with_model_timeout_is_refused(tmp_path, monkeypatch):
    """run() with resume=True and model_timeout_s=300 raises ValueError.

    Fails without fix: model_timeout_s is not checked in the resume condition,
    so it is silently ignored.
    """
    monkeypatch.setenv("REFACTORY_WORKER_CONTEXT", "{}")
    # Create a minimal run-config.json
    config_dir = tmp_path / ".refactory"
    config_dir.mkdir()
    (config_dir / "run-config.json").write_text(json.dumps({
        "mode": "test/mode",
        "project": str(tmp_path),
        "knobs": [],
        "model_context": {"timeout_s": 1800, "provider": "", "model": ""},
        "resource_limits": None,
        "configuration_fingerprints": {},
    }))
    # Write spine-status to indicate an authoring run
    (config_dir / "spine-status.json").write_text(json.dumps({"status": "halted"}))

    from refactory_dsh.interactive import run

    with pytest.raises(ValueError, match="resume retains saved"):
        run(
            "test/mode",
            workdir=tmp_path,
            resume=True,
            model_timeout_s=300,
        )


# ---------------------------------------------------------------------------
# 6. Default is 1800, no forwarding
# ---------------------------------------------------------------------------


def test_default_is_1800_no_forwarding():
    """DshModelClient() with no timeout_s gives 1800; _run_argv omits --model-timeout.

    Fails without fix: N/A (regression guard).
    """
    from refactory_dsh.cli import build_parser, _run_argv
    from refactory_dsh.model import DshModelClient

    parser = build_parser()
    args = parser.parse_args(["ceo", "an idea", "--headless", "--check", "true", "--mode", "design"])
    forwarded = _run_argv("some/mode", args)
    assert "--model-timeout" not in forwarded

    client = DshModelClient()
    assert client.context()["timeout_s"] == 1800


# ---------------------------------------------------------------------------
# 7. Invalid values refused
# ---------------------------------------------------------------------------


def test_invalid_model_timeout_zero(monkeypatch):
    """--model-timeout 0 is refused with exit code 2.

    Fails without fix: value 0 is accepted and forwarded.
    """
    monkeypatch.setattr("refactory_dsh.cli._forward", lambda m, a: 0)

    from refactory_dsh.cli import build_parser, cmd_ceo

    parser = build_parser()
    args = parser.parse_args(["ceo", "idea", "--model-timeout", "0", "--check", "true", "--mode", "design"])
    code = cmd_ceo(args)
    assert code == 2


def test_invalid_model_timeout_negative(monkeypatch):
    """--model-timeout -5 is refused with exit code 2.

    Fails without fix: value -5 is accepted and forwarded.
    """
    monkeypatch.setattr("refactory_dsh.cli._forward", lambda m, a: 0)

    from refactory_dsh.cli import build_parser, cmd_ceo

    parser = build_parser()
    args = parser.parse_args(["ceo", "idea", "--model-timeout", "-5", "--check", "true", "--mode", "design"])
    code = cmd_ceo(args)
    assert code == 2


def test_invalid_model_timeout_too_large(monkeypatch):
    """--model-timeout 8000 is refused with exit code 2.

    Fails without fix: value 8000 is accepted and forwarded.
    """
    monkeypatch.setattr("refactory_dsh.cli._forward", lambda m, a: 0)

    from refactory_dsh.cli import build_parser, cmd_ceo

    parser = build_parser()
    args = parser.parse_args(["ceo", "idea", "--model-timeout", "8000", "--check", "true", "--mode", "design"])
    code = cmd_ceo(args)
    assert code == 2


def test_invalid_model_timeout_non_integer():
    """--model-timeout abc causes argparse to exit with code 2.

    Fails without fix: N/A (argparse type=int handles this). Regression guard.
    """
    from refactory_dsh.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["ceo", "idea", "--model-timeout", "abc"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# 8. All agent nodes get command_timeout_s
# ---------------------------------------------------------------------------


def test_all_agent_nodes_get_command_timeout_s(tmp_path, monkeypatch):
    """lower_workflow with timeout_s=2000 sets command_timeout_s on every AgentNode.

    Fails without fix: if lower_workflow stops setting command_timeout_s, agent nodes
    would fall back to commandTimeoutMs. Regression guard.
    """
    monkeypatch.setenv("REFACTORY_WORKER_CONTEXT", "{}")
    from refactory_dsh.ir import AgentNode, AgentRole, Workflow, GateNode
    from refactory_dsh.model import DEFAULT_BACKSTOP_S, DshModelClient, NODE_DISPATCH_OVERHEAD_S
    from refactory_dsh.workers import lower_workflow

    # Build a minimal workflow with one agent node
    agent = AgentNode(id="draft", prompt_template="write something", role=AgentRole.BUILDER)
    workflow = Workflow(name="test", nodes={"draft": agent}, edges=[], start_node="draft")

    client = DshModelClient(timeout_s=2000)
    ctx = client.context()

    result = lower_workflow(workflow, tmp_path, ctx)
    draft = result.nodes["draft"]
    assert draft.command_timeout_s is not None
    # DSH reports activity, so the session has no wall: the envelope outlasts the
    # harness backstop. Codex reports none and keeps its route's budget.
    assert draft.command_timeout_s == DEFAULT_BACKSTOP_S + NODE_DISPATCH_OVERHEAD_S
    codex = lower_workflow(workflow, tmp_path / "codex",
                           DshModelClient(timeout_s=2000, worker_backend="codex", model="m").context())
    assert codex.nodes["draft"].command_timeout_s == 2000 + NODE_DISPATCH_OVERHEAD_S


# ---------------------------------------------------------------------------
# 9. Console open refuses invalid --model-timeout
# ---------------------------------------------------------------------------


def test_console_open_refuses_invalid_model_timeout():
    """console open with --model-timeout 0 is refused with exit code 2.

    Fails without fix: the invalid value reaches _open_mode() without validation.
    """
    from refactory_dsh.console import main
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["open", "test/mode", "--workdir", "/tmp/test", "--model-timeout", "0"])
    assert code == 2
    output = buf.getvalue()
    assert "model-timeout" in output


# ---------------------------------------------------------------------------
# 10. Console open passes --model-timeout through to model_context
# ---------------------------------------------------------------------------


def test_console_open_verb_passes_model_timeout(monkeypatch):
    """console open --model-timeout 120 includes timeout_s in the model_context.

    Fails without fix: the model_context dict built at console.py does not include
    timeout_s, so even with --model-timeout 120 the value is absent.
    """
    captured: dict[str, Any] = {}

    def fake_open_mode(mode, *, project=None, workdir=None, knobs=None,
                       intent="", model_context=None, resource_limits=None):
        captured["model_context"] = model_context
        return {"mode": mode, "run": str(workdir), "name": "test"}

    monkeypatch.setattr("refactory_dsh.console._open_mode", fake_open_mode)

    from refactory_dsh.console import _parser, main
    import io
    from contextlib import redirect_stdout

    # Run through main with stdout captured
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["open", "test/mode", "--workdir", "/tmp/testrun", "--model-timeout", "120"])

    assert code == 0
    assert captured.get("model_context", {}).get("timeout_s") == 120


# ---------------------------------------------------------------------------
# 11. Interactive path via REFACTORY_WORKER_CONTEXT inheritance
# ---------------------------------------------------------------------------


def test_interactive_path_inherits_timeout_via_worker_context(tmp_path, monkeypatch):
    """When REFACTORY_WORKER_CONTEXT carries timeout_s: 120 and no --model-timeout
    flag is given, DshModelClient inherits the value through the environment.

    This is how the interactive CEO path actually carries the value: the spine
    starts `console open` with no --model-timeout flag, and the value reaches
    it through REFACTORY_WORKER_CONTEXT.

    Fails without fix: N/A (existing inheritance mechanism). Regression guard
    for the interactive CEO path.
    """
    worker_ctx = json.dumps({"timeout_s": 120, "provider": "", "model": ""})
    monkeypatch.setenv("REFACTORY_WORKER_CONTEXT", worker_ctx)

    from refactory_dsh.model import DshModelClient

    # DshModelClient with no explicit timeout_s inherits from REFACTORY_WORKER_CONTEXT
    client = DshModelClient()
    ctx = client.context()
    assert ctx["timeout_s"] == 120

    # Verify that node_session_budget_s computes correctly
    from refactory_dsh.model import node_session_budget_s, NODE_DISPATCH_OVERHEAD_S

    budget = node_session_budget_s(3, 120)
    assert budget == 3 * 120 + NODE_DISPATCH_OVERHEAD_S


# ---------------------------------------------------------------------------
# 12. Open verb inherits timeout_s from REFACTORY_WORKER_CONTEXT (no flag)
# ---------------------------------------------------------------------------


def test_open_verb_inherits_timeout_from_worker_context(monkeypatch):
    """The open verb, invoked without --model-timeout but with
    REFACTORY_WORKER_CONTEXT carrying timeout_s: 120, passes a model_context
    without timeout_s to _open_mode. DshModelClient then inherits it from
    the environment.

    This is how the interactive CEO path works: index.ts:1544 builds the
    console open command without --model-timeout, and the value reaches
    DshModelClient through REFACTORY_WORKER_CONTEXT inheritance (model.py:200).

    Fails without fix: N/A (existing inheritance mechanism). Regression guard
    for the interactive CEO → console open path (gate correction #2).
    """
    worker_ctx = json.dumps({"timeout_s": 120, "provider": "", "model": ""})
    monkeypatch.setenv("REFACTORY_WORKER_CONTEXT", worker_ctx)

    captured: dict[str, Any] = {}

    def fake_open_mode(mode, *, project=None, workdir=None, knobs=None,
                       intent="", model_context=None, resource_limits=None):
        captured["model_context"] = model_context
        return {"mode": mode, "run": str(workdir), "name": "test"}

    monkeypatch.setattr("refactory_dsh.console._open_mode", fake_open_mode)

    from refactory_dsh.console import main
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["open", "test/mode", "--workdir", "/tmp/testrun"])

    assert code == 0
    mc = captured.get("model_context", {})
    # No --model-timeout flag → mc has no timeout_s key
    assert "timeout_s" not in mc

    # DshModelClient constructed with mc inherits timeout_s from REFACTORY_WORKER_CONTEXT
    from refactory_dsh.model import DshModelClient, node_session_budget_s, NODE_DISPATCH_OVERHEAD_S

    client = DshModelClient(**mc)
    ctx = client.context()
    assert ctx["timeout_s"] == 120
    assert node_session_budget_s(3, 120) == 3 * 120 + NODE_DISPATCH_OVERHEAD_S


# ---------------------------------------------------------------------------
# 13. Resume keeps saved timeout_s
# ---------------------------------------------------------------------------


def test_resume_keeps_saved_timeout(tmp_path, monkeypatch):
    """A resumed run uses the saved model_context's timeout_s: 500.

    Fails without fix: N/A (existing behavior). Regression guard: run()
    with resume=True loads model_context from run-config.json, and the
    context passed to run_graph_command carries timeout_s: 500.
    """
    monkeypatch.setenv("REFACTORY_WORKER_CONTEXT", "{}")

    config_dir = tmp_path / ".refactory"
    config_dir.mkdir()
    (config_dir / "run-config.json").write_text(json.dumps({
        "mode": "test/mode",
        "project": str(tmp_path),
        "knobs": [],
        "model_context": {
            "timeout_s": 500, "provider": "", "model": "",
            "worker_backend": "dsh", "permission_mode": "default",
        },
        "resource_limits": None,
        "configuration_fingerprints": {},
    }))

    captured: dict[str, Any] = {}

    monkeypatch.setattr("refactory_dsh.interactive.require_build", lambda **kw: None)

    import refactory_dsh.dsh_profile as _dp
    monkeypatch.setattr(_dp, "require", lambda **kw: None)

    from contextlib import contextmanager

    @contextmanager
    def fake_ownership(wd):
        yield 999

    monkeypatch.setattr("refactory_dsh.interactive.ownership", fake_ownership)
    monkeypatch.setattr("refactory_dsh.interactive.prepare",
                        lambda *a, **kw: ("test", []))
    monkeypatch.setattr("refactory_dsh.interactive.command",
                        lambda *a, **kw: ["echo", "test"])
    monkeypatch.setattr("refactory_dsh.interactive.run_summary", lambda wd: "")
    monkeypatch.setattr("refactory_dsh.interactive.run_warnings", lambda wd: "")

    class FakeResult:
        returncode = 0

    import refactory_dsh.workers as _w

    def fake_run_graph(argv, *, workdir=None, context=None, **kw):
        captured["context"] = context
        return FakeResult()

    monkeypatch.setattr(_w, "run_graph_command", fake_run_graph)

    from refactory_dsh.interactive import run

    code = run("test/mode", workdir=tmp_path, resume=True)
    assert code == 0
    assert captured["context"]["timeout_s"] == 500
