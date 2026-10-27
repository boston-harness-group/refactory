"""Tests for the worker node: feedback sidecar in dispatched prompt (dm-110)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest


def test_feedback_file_prepends_heading_to_prompt(tmp_path):
    """dm-110: A manifest with feedback_file results in feedback under its heading."""
    from refactory_dsh.run import NODE_ENV

    # Set up the feedback file.
    fb_path = tmp_path / "feedback.md"
    fb_path.write_text("Please fix the widget alignment.")

    # Set up the prompt file.
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Build the widget.")

    manifest = {
        "id": "emit",
        "type": "AgentNode",
        "command": "echo",
        "reads": [],
        "writes": [],
        "prompt_file": str(prompt_path),
        "feedback_file": str(fb_path),
    }

    spec = {
        "schema_version": 1,
        "node_id": "emit",
        "prompt": "fallback prompt",
        "reads": [],
        "writes": [],
        "model_context": {"provider": "test", "model": "test"},
    }

    captured_prompt = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def for_node(self, node, role="worker"):
            return self
        def complete(self, prompt, workdir=None):
            captured_prompt.append(prompt)
            return "done"
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "dsh"}

    class FakeDelivery:
        @staticmethod
        def begin(workdir, node, prompt, writes, *, reads=None, model_context=None):
            captured_prompt.clear()
            captured_prompt.append(prompt)
            return {"prompt": prompt, "node": node, "status": "pending"}
        @staticmethod
        def finish(workdir, node, *, final_response=None, capture_final=False):
            return {"node": node, "status": "delivered"}
        @staticmethod
        def _path(workdir, name):
            return Path(workdir) / name

    # Mock at the source modules (deferred imports inside execute() import from
    # these, not from worker_node module-level attributes).
    with mock.patch.dict(os.environ, {NODE_ENV: json.dumps(manifest)}):
        with mock.patch("refactory_dsh.model.DshModelClient", FakeClient):
            with mock.patch("refactory_dsh.agent_delivery.begin", FakeDelivery.begin):
                with mock.patch("refactory_dsh.agent_delivery.finish", FakeDelivery.finish):
                    with mock.patch("refactory_dsh.agent_delivery._path", FakeDelivery._path):
                        with mock.patch("refactory_dsh.run._knobs", return_value={}):
                            with mock.patch("refactory_dsh.run._read_facts", return_value={}):
                                with mock.patch("refactory_dsh.nn.executor.render", side_effect=lambda p, c: p):
                                    with mock.patch("refactory_dsh.nn.executor.RunContext"):
                                        from refactory_dsh import worker_node
                                        worker_node.execute(spec, tmp_path)

    assert len(captured_prompt) > 0
    final_prompt = captured_prompt[0]
    assert "## Feedback for this visit" in final_prompt
    assert "Please fix the widget alignment." in final_prompt
    assert "Build the widget." in final_prompt
    # Feedback comes before the main prompt.
    fb_pos = final_prompt.index("Feedback for this visit")
    prompt_pos = final_prompt.index("Build the widget.")
    assert fb_pos < prompt_pos


def test_end_to_end_drain_feedback_reaches_real_worker(tmp_path):
    """dm-110 end-to-end: verify that the file-level contract between
    drain() and worker_node.execute() holds across three visits.

    drain() writes ``.refactory/prompts/<id>.feedback.md`` and puts
    ``feedback_file`` in the DSH_WORKFLOW_NODE manifest.  worker_node.execute()
    reads that file and prepends its content under "## Feedback for this visit".

    The complementary vitest test (correction.spec.ts: 'drain dispatches to
    real worker_node.execute and feedback reaches prompt (dm-110 e2e)') crosses
    the real TS→Python boundary via a real shell; this test exercises the
    Python consumer side with the exact file layout that drain() produces.

    - Visit 1: no feedback sidecar → prompt has no "Feedback" heading.
    - Visit 2: feedback sidecar written → prompt carries marker under heading.
    - Visit 3: feedback sidecar removed → prompt has no stale feedback.
    """
    from refactory_dsh.run import NODE_ENV

    workdir = tmp_path / "run"
    workdir.mkdir()
    prompts_dir = workdir / ".refactory" / "prompts"
    prompts_dir.mkdir(parents=True)

    # Write the prompt file exactly as drain() does.
    prompt_file = prompts_dir / "a.md"
    prompt_file.write_text("Build it.")

    spec = {
        "schema_version": 1,
        "node_id": "a",
        "prompt": "fallback",
        "reads": [],
        "writes": [],
        "model_context": {"provider": "test", "model": "test"},
    }

    recorded_prompts: list[str] = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def for_node(self, node, role="worker"):
            return self
        def complete(self, prompt, workdir=None):
            recorded_prompts.append(prompt)
            return "done"
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "dsh"}

    class FakeDelivery:
        @staticmethod
        def begin(workdir, node, prompt, writes, *, reads=None, model_context=None):
            recorded_prompts.append(prompt)
            return {"prompt": prompt, "node": node, "status": "pending"}
        @staticmethod
        def finish(workdir, node, *, final_response=None, capture_final=False):
            return {"node": node, "status": "delivered"}
        @staticmethod
        def _path(workdir, name):
            return Path(workdir) / name

    marker = "MARKER-dm110-end-to-end-feedback"
    feedback_file = prompts_dir / "a.feedback.md"

    def run_worker(*, with_feedback: bool):
        """Simulate one visit: write or remove the sidecar, build the manifest
        exactly as runDeterministic() does, and call worker_node.execute()."""
        if with_feedback:
            feedback_file.write_text(marker)
        elif feedback_file.exists():
            feedback_file.unlink()

        manifest = {
            "id": "a",
            "type": "AgentNode",
            "command": "echo dispatched",
            "reads": [],
            "writes": [],
            "prompt_file": str(prompt_file),
        }
        if with_feedback:
            manifest["feedback_file"] = str(feedback_file)

        recorded_prompts.clear()
        with mock.patch.dict(os.environ, {NODE_ENV: json.dumps(manifest)}):
            with mock.patch("refactory_dsh.model.DshModelClient", FakeClient), \
                 mock.patch("refactory_dsh.agent_delivery.begin", FakeDelivery.begin), \
                 mock.patch("refactory_dsh.agent_delivery.finish", FakeDelivery.finish), \
                 mock.patch("refactory_dsh.agent_delivery._path", FakeDelivery._path), \
                 mock.patch("refactory_dsh.run._knobs", return_value={}), \
                 mock.patch("refactory_dsh.run._read_facts", return_value={}), \
                 mock.patch("refactory_dsh.nn.executor.render", side_effect=lambda p, c: p), \
                 mock.patch("refactory_dsh.nn.executor.RunContext"):
                from refactory_dsh import worker_node
                worker_node.execute(spec, workdir)
        return recorded_prompts[-1] if recorded_prompts else ""

    # Visit 1: no feedback sidecar — prompt is the prompt file content only.
    v1 = run_worker(with_feedback=False)
    assert "Feedback for this visit" not in v1
    assert "Build it." in v1

    # Visit 2: feedback sidecar written (simulating drain() after rerun --text).
    v2 = run_worker(with_feedback=True)
    assert "## Feedback for this visit" in v2
    assert marker in v2
    assert "Build it." in v2
    # Feedback comes before the main prompt.
    assert v2.index("Feedback for this visit") < v2.index("Build it.")

    # Visit 3: feedback cleared (simulating completeDue removing the sidecar).
    v3 = run_worker(with_feedback=False)
    assert "Feedback for this visit" not in v3
    assert marker not in v3


def test_gate_reloop_reason_reaches_reloop_target_prompt(tmp_path):
    """dm-110: A gate reloop's reason reaches the reloop target's prompt
    the same way as rerun text: under '## Feedback for this visit'.

    The spine writes the gate reason as the feedback sidecar; the worker
    reads it identically to a rerun text sidecar.
    """
    from refactory_dsh.run import NODE_ENV

    prompts_dir = tmp_path / ".refactory" / "prompts"
    prompts_dir.mkdir(parents=True)
    fb_path = prompts_dir / "builder.feedback.md"
    # The gate's reason, written by the spine's drain() for the reloop target.
    fb_path.write_text("GATE-REASON: the output did not pass review criteria")

    prompt_path = prompts_dir / "builder.md"
    prompt_path.write_text("Implement the feature.")

    manifest = {
        "id": "builder",
        "type": "AgentNode",
        "command": "refactory-worker",
        "reads": [],
        "writes": ["output.txt"],
        "prompt_file": str(prompt_path),
        "feedback_file": str(fb_path),
    }

    spec = {
        "schema_version": 1,
        "node_id": "builder",
        "prompt": "fallback",
        "reads": [],
        "writes": ["output.txt"],
        "model_context": {"provider": "test", "model": "test"},
    }

    recorded_prompts: list[str] = []

    class ScriptedModelClient:
        def __init__(self, **kwargs):
            pass
        def for_node(self, node, role="worker"):
            return self
        def complete(self, prompt, workdir=None):
            recorded_prompts.append(prompt)
            return "done"
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "dsh"}

    class FakeDelivery:
        @staticmethod
        def begin(workdir, node, prompt, writes, *, reads=None, model_context=None):
            recorded_prompts.append(prompt)
            return {"prompt": prompt, "node": node, "status": "pending"}
        @staticmethod
        def finish(workdir, node, *, final_response=None, capture_final=False):
            return {"node": node, "status": "delivered"}
        @staticmethod
        def _path(workdir, name):
            return Path(workdir) / name

    with mock.patch.dict(os.environ, {NODE_ENV: json.dumps(manifest)}):
        with mock.patch("refactory_dsh.model.DshModelClient", ScriptedModelClient):
            with mock.patch("refactory_dsh.agent_delivery.begin", FakeDelivery.begin):
                with mock.patch("refactory_dsh.agent_delivery.finish", FakeDelivery.finish):
                    with mock.patch("refactory_dsh.agent_delivery._path", FakeDelivery._path):
                        with mock.patch("refactory_dsh.run._knobs", return_value={}):
                            with mock.patch("refactory_dsh.run._read_facts", return_value={}):
                                with mock.patch("refactory_dsh.nn.executor.render", side_effect=lambda p, c: p):
                                    with mock.patch("refactory_dsh.nn.executor.RunContext"):
                                        from refactory_dsh import worker_node
                                        worker_node.execute(spec, tmp_path)

    assert len(recorded_prompts) > 0
    final_prompt = recorded_prompts[-1]
    assert "## Feedback for this visit" in final_prompt
    assert "GATE-REASON: the output did not pass review criteria" in final_prompt
    assert "Implement the feature." in final_prompt
    fb_pos = final_prompt.index("Feedback for this visit")
    prompt_pos = final_prompt.index("Implement the feature.")
    assert fb_pos < prompt_pos


def test_no_feedback_file_leaves_prompt_unchanged(tmp_path):
    """dm-110: Without a feedback_file, the prompt is the prompt file's content."""
    from refactory_dsh.run import NODE_ENV

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Build the widget.")

    manifest = {
        "id": "emit",
        "type": "AgentNode",
        "command": "echo",
        "reads": [],
        "writes": [],
        "prompt_file": str(prompt_path),
        # No feedback_file
    }

    spec = {
        "schema_version": 1,
        "node_id": "emit",
        "prompt": "fallback prompt",
        "reads": [],
        "writes": [],
        "model_context": {"provider": "test", "model": "test"},
    }

    captured_prompt = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def for_node(self, node, role="worker"):
            return self
        def complete(self, prompt, workdir=None):
            captured_prompt.append(prompt)
            return "done"
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "dsh"}

    class FakeDelivery:
        @staticmethod
        def begin(workdir, node, prompt, writes, *, reads=None, model_context=None):
            captured_prompt.clear()
            captured_prompt.append(prompt)
            return {"prompt": prompt, "node": node, "status": "pending"}
        @staticmethod
        def finish(workdir, node, *, final_response=None, capture_final=False):
            return {"node": node, "status": "delivered"}
        @staticmethod
        def _path(workdir, name):
            return Path(workdir) / name

    # Mock at the source modules (deferred imports inside execute() import from
    # these, not from worker_node module-level attributes).
    with mock.patch.dict(os.environ, {NODE_ENV: json.dumps(manifest)}):
        with mock.patch("refactory_dsh.model.DshModelClient", FakeClient):
            with mock.patch("refactory_dsh.agent_delivery.begin", FakeDelivery.begin):
                with mock.patch("refactory_dsh.agent_delivery.finish", FakeDelivery.finish):
                    with mock.patch("refactory_dsh.agent_delivery._path", FakeDelivery._path):
                        with mock.patch("refactory_dsh.run._knobs", return_value={}):
                            with mock.patch("refactory_dsh.run._read_facts", return_value={}):
                                with mock.patch("refactory_dsh.nn.executor.render", side_effect=lambda p, c: p):
                                    with mock.patch("refactory_dsh.nn.executor.RunContext"):
                                        from refactory_dsh import worker_node
                                        worker_node.execute(spec, tmp_path)

    assert len(captured_prompt) > 0
    final_prompt = captured_prompt[0]
    assert "Feedback for this visit" not in final_prompt
    assert final_prompt == "Build the widget."


def _run_worker(tmp_path, spec, *, prompt: str, feedback: str | None = None):
    """Run ``worker_node.execute`` with a recording client; return (prompts, roles)."""
    from refactory_dsh.run import NODE_ENV

    prompts_dir = tmp_path / ".refactory" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompts_dir / f"{spec['node_id']}.md"
    prompt_path.write_text(prompt)
    manifest = {"id": spec["node_id"], "type": "AgentNode", "command": "refactory-worker", "reads": [],
                "writes": [], "prompt_file": str(prompt_path)}
    if feedback is not None:
        fb_path = prompts_dir / f"{spec['node_id']}.feedback.md"
        fb_path.write_text(feedback)
        manifest["feedback_file"] = str(fb_path)
    prompts: list[str] = []
    roles: list[str] = []

    class RecordingClient:
        def __init__(self, **kwargs):
            pass

        def for_node(self, node, role="worker"):
            roles.append(role)
            return self

        def complete(self, prompt, workdir=None):
            return "done"

        def context(self):
            return {"provider": "test", "model": "test"}

        def route(self):
            return {"worker_backend": "dsh"}

    def begin(workdir, node, prompt, writes, *, reads=None, model_context=None):
        prompts.append(prompt)
        return {"prompt": prompt, "node": node, "status": "pending"}

    with mock.patch.dict(os.environ, {NODE_ENV: json.dumps(manifest)}), \
            mock.patch("refactory_dsh.model.DshModelClient", RecordingClient), \
            mock.patch("refactory_dsh.agent_delivery.begin", begin), \
            mock.patch("refactory_dsh.agent_delivery.finish",
                       lambda workdir, node, **kw: {"node": node, "status": "delivered"}), \
            mock.patch("refactory_dsh.run._knobs", return_value={}), \
            mock.patch("refactory_dsh.run._read_facts", return_value={}):
        from refactory_dsh import worker_node
        worker_node.execute(spec, tmp_path)
    return prompts, roles


def _spec(**extra):
    return {"schema_version": 1, "node_id": "builder", "prompt": "fallback", "reads": [], "writes": [],
            "produces": [], "model_context": {"provider": "test", "model": "test"}, **extra}


def test_feedback_the_rendered_prompt_already_carries_is_not_repeated(tmp_path):
    """dm-123 #436 gap 5: a template that renders the gate's reason says it once; the worker
    adds the feedback heading only for feedback the prompt does not carry."""
    reason = "GATE-REASON: tests/test_pin.py::test_pin fails"
    prompts, _ = _run_worker(tmp_path, _spec(), prompt=f"Implement it.\n\n## What went wrong last time\n{reason}\n",
                             feedback=reason)
    assert prompts[-1].count(reason) == 1
    assert "## Feedback for this visit" not in prompts[-1]


def test_the_worker_opens_the_session_with_the_nodes_role(tmp_path):
    """The receipt records the node's role, as the walker's does; an older spec keeps `worker`."""
    _, roles = _run_worker(tmp_path, _spec(role="code_reviewer"), prompt="Review it.")
    assert roles == ["code_reviewer"]
    _, roles = _run_worker(tmp_path / "old", _spec(), prompt="Review it.")
    assert roles == ["worker"]


def _workspace_spec(workdir: Path, script: Path) -> dict:
    """The spine's worker spec for one agent node that works in a scratch workspace."""
    import shlex

    from refactory_dsh import lowering, workers
    from refactory_dsh.model import DshModelClient
    from refactory_dsh.nn import Graph, Node

    graph = Graph()
    graph.add(Node("solve", "agent", prompt_template="Solve it.", writes=("answer.txt",), workspace="scratch"))
    state = {"graph": graph.to_state(), "knobs": {}}
    context = DshModelClient(worker_backend="dsh", dsh_cli=str(script), attempts=1).context()
    lowered = workers.lower_workflow(lowering.to_workflow(state), workdir, context,
                                     lowering.node_declarations(state))
    return json.loads(Path(shlex.split(lowered.nodes["solve"].dispatch_command)[-1]).read_text())


def test_the_spines_worker_runs_a_workspace_node_in_its_workspace_and_names_it(tmp_path, monkeypatch):
    """#516 on the spine: `worker_node.execute` asks the agent in the node's workspace
    (`workspace.ask`), not in the run directory, and the contract names that workspace
    (`begin(..., workspace=place.describe())`)."""
    import sys

    from refactory_dsh import worker_node, workspace

    monkeypatch.setattr(workspace, "_sandbox_runner", lambda root: ("landlock", ""))
    workdir = tmp_path / "run"
    workdir.mkdir()
    seen = tmp_path / "seen.json"
    script = tmp_path / "agent-cli"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "prompt = sys.stdin.read()\n"
        f"pathlib.Path({str(seen)!r}).write_text(json.dumps({{'cwd': str(pathlib.Path.cwd()), 'prompt': prompt}}))\n"
        "pathlib.Path('answer.txt').write_text('42\\n')\n")
    script.chmod(0o755)
    worker_node.execute(_workspace_spec(workdir, script), workdir)
    found = json.loads(seen.read_text())
    cwd = Path(found["cwd"])
    assert cwd.parent == workdir / workspace.ROOT / "solve" and cwd.name.startswith("scratch-")
    assert (workdir / "answer.txt").read_text() == "42\n", "the workspace's answer is collected"
    from refactory_dsh.agent_delivery import CONTRACT_END, CONTRACT_START
    contract = found["prompt"].split(CONTRACT_START, 1)[1].split(CONTRACT_END, 1)[0]
    execution = json.loads(contract[contract.rfind("\n{\n") + 1:])
    assert execution["workspace"]["kind"] == "scratch" and execution["workspace"]["path"] == str(cwd)
    assert f"Your workspace is {cwd}" in contract
