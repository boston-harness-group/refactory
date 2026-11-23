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
import sys
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


def test_failed_reruns_only_the_last_failures_and_falls_back_without_a_record(tmp_path):
    files = {**PINNED, "tests/test_second.py": "import tool\n\n\ndef test_second():\n    assert tool\n"}
    run, _, _ = _run_with_worktree(tmp_path, files)
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts={"check": CHECK})
    (place.path / "tool.py").write_text("VERSION = 2\n")
    # No record yet: --failed is the full selection, and says so.
    full = json.loads(_command(place, "--failed", "--list").stdout)
    assert PIN in full["selection"] and "tests/test_second.py" in full["selection"]
    assert full["rerun_failed"] is False
    said = _command(place, "--failed")
    assert "no failures are recorded" in said.stdout and "tests/test_second.py" in said.stdout
    assert said.returncode != 0
    # The record is inside the workspace's own .refactory/ (inside the sandbox).
    record_path = place.path / ".refactory" / targeted_tests.LAST_RUN
    assert record_path.exists(), f"record not at {record_path}"
    assert record_path.relative_to(place.path)  # inside the sandbox
    # The log file exists and contains FAILED lines.
    log_path = place.path / ".refactory" / targeted_tests.LOG_FILE
    assert log_path.exists(), f"log not at {log_path}"
    assert "FAILED" in log_path.read_text()
    # The run that just failed recorded its failures; --failed reruns only those.
    listed = json.loads(_command(place, "--failed", "--list").stdout)
    assert listed["selection"] == [f"{PIN}::test_pin"] and listed["rerun_failed"] is True
    again = _command(place, "--failed")
    assert f"{PIN}::test_pin" in again.stdout and "test_second" not in again.stdout.split("running:")[1]
    assert again.returncode != 0
    # Fixed: the --failed run passes, the record is empty, and the next --failed is the full set.
    (place.path / PIN).write_text("import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n")
    assert _command(place, "--failed").returncode == 0
    after = json.loads(_command(place, "--failed", "--list").stdout)
    assert after["rerun_failed"] is False and "tests/test_second.py" in after["selection"]


def test_cutoff_run_records_failures_seen_so_far(tmp_path):
    """A run cut off by a timeout (SIGTERM) still leaves exactly the failures seen so far
    in the record, and --failed afterwards runs only those. No mocks: the real command
    runs a fixture that prints FAILED lines then sleeps, and is killed."""
    run, _, _ = _run_with_worktree(tmp_path, PINNED)
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts={"check": CHECK})
    (place.path / "tool.py").write_text("VERSION = 2\n")
    # Write a Python test script that prints two FAILED lines then blocks.
    # Using Python + sys.stdout.flush to ensure the lines reach the pipe before the hang.
    fake_test = place.path / "_fake_pytest.py"
    fake_test.write_text(
        "import sys, time\n"
        "print('FAILED tests/test_a.py::test_one - assert 1 == 2')\n"
        "sys.stdout.flush()\n"
        "print('FAILED tests/test_b.py::test_two - assert 3 == 4')\n"
        "sys.stdout.flush()\n"
        "time.sleep(300)\n"
    )
    record_dir = place.path / ".refactory"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / targeted_tests.LAST_RUN
    log_path = record_dir / targeted_tests.LOG_FILE
    # Launch _run_and_collect in a child process group so we can kill it.
    import signal
    import threading
    import time

    exc_holder: list[BaseException | None] = [None]

    def run_it():
        try:
            targeted_tests._run_and_collect(
                f"{shlex.quote(sys.executable)} {shlex.quote(str(fake_test))}",
                place.path, record_path=record_path, log_path=log_path)
        except BaseException as e:
            exc_holder[0] = e

    thread = threading.Thread(target=run_it, daemon=True)
    thread.start()
    # Wait for the record to be written with the two failures.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if record_path.exists():
            try:
                rec = json.loads(record_path.read_text())
                if len(rec.get("failed", [])) >= 2:
                    break
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.1)
    # The record holds exactly the failures printed so far.
    assert record_path.exists(), "record not written before cutoff"
    rec = json.loads(record_path.read_text())
    assert rec["failed"] == ["tests/test_a.py::test_one", "tests/test_b.py::test_two"]
    # Kill the hanging child process (the Python script's sleep).
    # Find it by its command line containing _fake_pytest.py.
    import signal as _sig
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode(errors="replace")
            if "_fake_pytest.py" in cmdline and str(pid_dir.name) != str(os.getpid()):
                os.kill(int(pid_dir.name), _sig.SIGKILL)
        except (OSError, PermissionError):
            continue
    thread.join(timeout=10)
    # Confirm --failed reads back exactly those two.
    failures = targeted_tests._last_failures(record_path)
    assert failures == ["tests/test_a.py::test_one", "tests/test_b.py::test_two"]


def test_remember_warns_on_write_failure(tmp_path, capsys):
    """When the record cannot be written, a warning naming the path is printed to stderr.
    Uses a directory in place of the file (works even as root, unlike chmod)."""
    record_path = tmp_path / "record" / targeted_tests.LAST_RUN
    # Create a directory where the JSON file should go, so writing the file fails.
    record_path.mkdir(parents=True)
    targeted_tests._remember(record_path, ["tests/test_x.py::test_y"])
    captured = capsys.readouterr()
    assert "warning" in captured.err
    assert str(record_path) in captured.err


def test_record_and_log_never_reach_the_change(tmp_path):
    """The record and log files are inside .refactory/, which is excluded from delivery.
    After a targeted run they must not appear in git status or the emitted diff."""
    run, _, _ = _run_with_worktree(tmp_path, PINNED)
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts={"check": CHECK})
    (place.path / "tool.py").write_text("VERSION = 2\n")
    result = _command(place)
    assert result.returncode != 0  # pin fails
    # The record and log exist in the workspace.
    assert (place.path / ".refactory" / targeted_tests.LAST_RUN).exists()
    assert (place.path / ".refactory" / targeted_tests.LOG_FILE).exists()
    # They do not appear in git status (the workspace's .gitignore or exclusion hides them).
    status = subprocess.run(
        ["git", "-C", str(place.path), "status", "--porcelain", "-uall"],
        capture_output=True, text=True, check=True).stdout
    assert targeted_tests.LAST_RUN not in status
    assert targeted_tests.LOG_FILE not in status
    # They do not appear in the emitted diff.
    emission = _emission(place)
    assert targeted_tests.LAST_RUN not in emission
    assert targeted_tests.LOG_FILE not in emission
