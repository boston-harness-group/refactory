"""``refactory-targeted-tests``: the builder runs, in its own session, the tests the self-check
will run for its change.

The self-check (`design_ops.prepare_check`) finds the existing tests a builder broke and
sends it back to a new, cold session. The command in the builder's workspace runs the same
selection (`design_ops.self_check_selection`) for the workspace's current diff, so the
builder finds those failures before it stops. Each test drives the real clone workspace and
the real command; the self-check's own record or selection function is the reference.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from refactory_dsh import design_ops, lowering, repo_change, targeted_tests, workers, workspace, worktree
from refactory_dsh.design import BUILDER_WORKSPACE
from refactory_dsh.nn import Graph
from refactory_dsh.nn.executor import RunContext
from test_self_check import (PINNED, _emit, _facts, _prepare, _records, _spine_run, _target)
from test_workspace import _agent, _client, _clone_state, _run_with_worktree, _seen, _walk
from tests_support import git, render

pytestmark = pytest.mark.usefixtures("fake_gh")

PIN = "tests/test_pin.py"
CHECK = "python -m pytest -q -p no:cacheprovider tests"


def _agent_env(place: workspace.Workspace) -> dict[str, str]:
    """The environment the builder's shell has: the run's, with the workspace's over it."""
    env = dict(os.environ)
    for name, value in place.env.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env


def _command(place: workspace.Workspace, *args: str) -> subprocess.CompletedProcess:
    """The command as the builder runs it: by name, from its workspace, on its PATH."""
    return subprocess.run(["refactory-targeted-tests", *args], cwd=place.path, env=_agent_env(place),
                          capture_output=True, text=True, timeout=300)


def _emission(place: workspace.Workspace) -> str:
    """What the runtime collects from the workspace, and materialise commits."""
    emission, refused, _ = workspace.derive_emission(
        place.path, place.base, skip=place.skip, generate_owned=place.generate_owned)
    assert not refused, refused
    return emission


def test_the_command_selects_what_the_self_check_selects_for_the_same_diff(tmp_path, monkeypatch):
    """On a first dispatch and on a fix-up, the command's selection for the workspace's diff
    is the selection of the self-check that then runs on the emitted attempt: a test the
    change broke, a test importing a module it changed, a test file it only created (still
    untracked when the builder runs the command), and, on the fix-up, the last failure."""
    repo = _target(tmp_path / "repo", {**PINNED, "tests/test_other.py": "def test_other():\n    pass\n"})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef600001")
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts=_facts(run))
    assert place.env["PATH"].split(os.pathsep)[0] == str(place.root / targeted_tests.BIN)
    (place.path / "tool.py").write_text("VERSION = 2\n")
    (place.path / "tests" / "test_new.py").write_text(
        "import tool\n\n\ndef test_new():\n    assert tool.VERSION == 2\n")
    listed = json.loads(_command(place, "--list").stdout)
    assert PIN in listed["selection"] and "tests/test_new.py" in listed["selection"]
    assert listed["touched"] == ["tests/test_new.py", "tool.py"]
    _emit(run, _emission(place), monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and out.strip().endswith("reloop"), err
    first = _records(run)[0]
    assert listed["selection"] == first["selection"]
    assert first["failing_ids"] == [f"{PIN}::test_pin"]

    # The fix-up's dispatch: nothing changed yet, so the last failure runs on its own; then the
    # builder fixes the pin and the command's selection is again the next self-check's.
    again = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts=_facts(run))
    assert json.loads(_command(again, "--list").stdout)["selection"] == [f"{PIN}::test_pin"]
    (again.path / PIN).write_text("import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n")
    listed = json.loads(_command(again, "--list").stdout)
    assert listed["touched"] == [PIN] and PIN in listed["selection"]
    _emit(run, _emission(again), monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and out.strip() == "proceed", err
    assert listed["selection"] == _records(run)[1]["selection"]


def test_the_command_runs_the_selection_with_the_checks_options_and_reports_the_result(tmp_path):
    run, _, _ = _run_with_worktree(tmp_path, PINNED)
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts={"check": CHECK})
    (place.path / "tool.py").write_text("VERSION = 2\n")
    failed = _command(place)
    assert failed.returncode != 0, failed.stdout
    assert f"running: python -m pytest -q -p no:cacheprovider {PIN}" in failed.stdout
    assert f"FAILED {PIN}::test_pin" in failed.stdout and "FAILED (exit" in failed.stdout
    (place.path / PIN).write_text("import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n")
    passed = _command(place)
    assert passed.returncode == 0 and "PASSED" in passed.stdout, passed.stdout
    # It reads the diff as it is when it is run: nothing changed, nothing to select.
    (place.path / PIN).write_text(PINNED[PIN])
    (place.path / "tool.py").write_text(PINNED["tool.py"])
    idle = _command(place)
    assert idle.returncode == 0 and "nothing is selected" in idle.stdout, idle.stdout


def test_the_command_keeps_the_check_commands_options(tmp_path):
    """The check's own options (`-n 4`) stay in the command, as the self-check keeps them."""
    run, _, _ = _run_with_worktree(tmp_path, PINNED)
    check = "python -m pytest -q -n 4 -p no:cacheprovider tests"
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts={"check": check})
    (place.path / "tool.py").write_text("VERSION = 2\n")
    listed = json.loads(_command(place, "--list").stdout)
    assert listed["command"] == f"python -m pytest -q -n 4 -p no:cacheprovider {PIN}"


def test_a_run_without_a_check_command_says_so_and_runs_nothing(tmp_path):
    run, _, _ = _run_with_worktree(tmp_path, PINNED)
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts={})
    said = _command(place)
    assert said.returncode == 2 and "no check command" in said.stdout


BUILDER = ("(cwd / 'tool.py').write_text('VERSION = 2\\n')\n"
           "where = pathlib.Path({seen!r}).parent\n"
           "listed = subprocess.run(['refactory-targeted-tests', '--list'], capture_output=True, text=True)\n"
           "(where / 'listed.json').write_text(listed.stdout)\n"
           "ran = subprocess.run(['refactory-targeted-tests'], capture_output=True, text=True)\n"
           "(where / 'ran.txt').write_text(f'{ran.returncode}\\n{ran.stdout}{ran.stderr}')\n"
           "print('done')\n")


@pytest.mark.parametrize("runtime", ["walker", "spine"])
def test_a_builder_on_either_runtime_finds_the_neighbour_it_broke_in_its_own_session(
        tmp_path, runtime, monkeypatch):
    """The walker's `_visit_agent` and the spine's `worker_node` both prepare the clone the
    command is installed in: the builder runs it by name, it finds the neighbouring test the
    change broke, and its selection is the self-check's for the committed attempt."""
    monkeypatch.setattr(workspace, "_sandbox_runner", lambda root: ("landlock", ""))
    run, repo, found = _run_with_worktree(tmp_path, PINNED)
    facts = {"name": "tool", "check": CHECK}
    (run / "facts.json").write_text(json.dumps(facts))
    script = _agent(tmp_path, BUILDER.replace("{seen!r}", repr(str(tmp_path / "seen.jsonl"))))
    state = _clone_state()
    if runtime == "walker":
        _walk(Graph.from_state(state["graph"]), run, _client(script), facts=facts)
    else:
        from refactory_dsh import worker_node
        lowered = workers.lower_workflow(lowering.to_workflow(state), run, _client(script).context(),
                                         lowering.node_declarations(state))
        spec = json.loads(Path(shlex.split(lowered.nodes["emit"].dispatch_command)[-1]).read_text())
        worker_node.execute(spec, run)
    [seen] = _seen(tmp_path)
    assert seen["cwd"] == str(run / workspace.ROOT / "emit" / "repo")
    code, _, output = (tmp_path / "ran.txt").read_text().partition("\n")
    assert code != "0" and f"FAILED {PIN}::test_pin" in output, output
    listed = json.loads((tmp_path / "listed.json").read_text())
    ctx = RunContext(run, knobs={}, facts={"name": "tool"}, task={})
    ctx.enforce_writes = False
    repo_change.materialise(ctx)
    attempt = worktree.recorded(run)
    touched, _since = design_ops._fixup_changes(Path(attempt.path), attempt, {})
    expected = design_ops.self_check_selection(Path(attempt.path), CHECK, touched, {}, [])
    assert listed["selection"] == expected.tests == [PIN]
    assert git(Path(attempt.path), "show", f"{attempt.head}:tool.py") == "VERSION = 2"
