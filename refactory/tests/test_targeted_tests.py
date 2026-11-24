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
import re
import shlex
import subprocess
import tempfile
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
    """The command as the builder runs it: by name, from its workspace, on its PATH. A run
    cut off at the limit fails with what it printed, a stalled session's stack included."""
    try:
        return subprocess.run(["refactory-targeted-tests", *args], cwd=place.path,
                              env=_agent_env(place), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as cut:
        said = [part.decode(errors="replace") if isinstance(part, bytes) else part or ""
                for part in (cut.stdout, cut.stderr)]
        pytest.fail(f"refactory-targeted-tests ran past 300 s; it printed:\n{''.join(said)}")


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


CUT = {
    "tool.py": "VERSION = 1\n",
    # Collected first: sleeps before any test fails when CUT_FIRST names a marker.
    "tests/test_a_first.py": ("import os, pathlib, time\nimport tool\n\n\ndef test_first():\n"
                              "    if os.environ.get('CUT_FIRST'):\n"
                              "        pathlib.Path(os.environ['CUT_FIRST']).write_text('started')\n"
                              "        time.sleep(300)\n"),
    # Two tests the change breaks, then one that sleeps when CUT_MARKER names a marker.
    "tests/test_cut.py": ("import os, pathlib, time\nimport tool\n\n\n"
                          "def test_one():\n    assert tool.VERSION == 1\n\n\n"
                          "def test_two():\n    assert tool.VERSION == 1\n\n\n"
                          "def test_wait():\n    if os.environ.get('CUT_MARKER'):\n"
                          "        pathlib.Path(os.environ['CUT_MARKER']).write_text('started')\n"
                          "        time.sleep(300)\n"),
}
CUT_FAILED = ["tests/test_cut.py::test_one", "tests/test_cut.py::test_two"]


def _cut_off(place: workspace.Workspace, out: Path, until, **env: str) -> None:
    """Start the real command as the builder does, wait until ``until()`` holds, then cut
    it off as a session timeout does: SIGTERM to the process group this test started (and
    SIGKILL if it lingers). Only that group is signalled."""
    import signal
    import time

    with out.open("w") as stream:
        proc = subprocess.Popen(["refactory-targeted-tests"], cwd=place.path,
                                env={**_agent_env(place), **env}, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 120
        while not until():
            assert proc.poll() is None, out.read_text()
            assert time.monotonic() < deadline, out.read_text()
            time.sleep(0.1)
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


@pytest.mark.parametrize("check", [CHECK, "python -m pytest -q -n 2 -p no:cacheprovider tests"],
                         ids=["serial", "xdist"])
def test_a_real_pytest_run_cut_off_leaves_the_failures_it_saw(tmp_path, check):
    """The real command runs real pytest (``-q``, so no FAILED line is printed before the
    summary) and is cut off while a later test still runs: the record holds the failures
    seen so far, and ``--failed`` reruns exactly those. With xdist, the controller records
    what its workers report. A new run that is cut off before any failure leaves an empty
    record, not the last run's failures."""
    run, _, _ = _run_with_worktree(tmp_path, CUT)
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts={"check": check})
    (place.path / "tool.py").write_text("VERSION = 2\n")
    record = place.path / ".refactory" / targeted_tests.LAST_RUN
    marker = tmp_path / "waiting"

    def seen_both() -> bool:
        return marker.exists() and sorted(targeted_tests._last_failures(record)) == CUT_FAILED

    _cut_off(place, tmp_path / "first.out", seen_both, CUT_MARKER=str(marker))
    assert "PASSED" not in (tmp_path / "first.out").read_text()
    assert sorted(targeted_tests._last_failures(record)) == CUT_FAILED
    listed = json.loads(_command(place, "--failed", "--list").stdout)
    assert listed["rerun_failed"] is True and sorted(listed["selection"]) == CUT_FAILED
    if "-n" in check:
        return
    # The next --failed says why it reruns so few, and runs only those (not the waiting test).
    again = _command(place, "--failed")
    assert "the last run here did not finish" in again.stdout, again.stdout
    assert "2 test(s) had failed by then, and only those are rerun" in again.stdout
    assert "test_wait" not in again.stdout.split("running:")[1]
    assert again.returncode != 0 and "FAILED (exit" in again.stdout
    # The next run is cut off before anything fails: none of the last run's failures stay,
    # and --failed runs the full selection.
    first = tmp_path / "first-test"
    _cut_off(place, tmp_path / "second.out", first.exists, CUT_FIRST=str(first))
    assert targeted_tests._last_run(record) == {
        "finished": False, "failed": [], "seconds": targeted_tests._last_run(record)["seconds"]}
    listed = json.loads(_command(place, "--failed", "--list").stdout)
    assert listed["rerun_failed"] is False and "tests/test_cut.py" in listed["selection"]


NESTED = {
    "tool.py": "VERSION = 1\n",
    # A repo test that runs pytest itself, with a PYTHONPATH of its own: it passes in the
    # check, so it must pass under the command too.
    "tests/test_nested.py": (
        "import os, subprocess, sys\nimport tool\n\n\n"
        "def test_nested(tmp_path):\n"
        "    # What the repo's own tests see of the environment carries none of the plugin.\n"
        "    assert 'refactory_failure_log' not in os.environ.get('PYTEST_ADDOPTS', '')\n"
        "    assert not [part for part in os.environ.get('PYTHONPATH', '').split(os.pathsep)\n"
        "                if part.endswith(os.path.join('.refactory', 'plugins'))]\n"
        "    inner = tmp_path / 'test_inner.py'\n"
        "    inner.write_text('def test_inner():\\n    pass\\n')\n"
        "    own = tmp_path / 'inner-tmp'\n"
        "    own.mkdir()\n"
        "    done = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',\n"
        "                           '-o', 'faulthandler_timeout=60', str(inner)], cwd=tmp_path,\n"
        "                          stdin=subprocess.DEVNULL, capture_output=True, text=True,\n"
        "                          timeout=120,\n"
        "                          env={**os.environ, 'PYTHONPATH': str(tmp_path),\n"
        "                               'PYTEST_DEBUG_TEMPROOT': str(own), 'TMPDIR': str(own)})\n"
        "    assert done.returncode == 0, done.stdout + done.stderr\n"
        "    assert tool.VERSION == 2\n"),
}


#: A stalled session prints every thread's stack after this long, well inside `_command`'s 300 s.
STALL = "-o faulthandler_timeout=120"


@pytest.mark.parametrize("check", [f"python -m pytest -q {STALL} -p no:cacheprovider tests",
                                   f"python -m pytest -q -n 2 {STALL} -p no:cacheprovider tests"],
                         ids=["serial", "xdist"])
def test_a_repo_test_that_runs_pytest_itself_does_not_see_the_plugin(tmp_path, monkeypatch, check):
    """The plugin is the command's alone: a pytest the repo's own test starts, with its own
    PYTHONPATH, neither loads it nor fails importing it.

    The sessions the command starts keep their temporary directories under this test's
    own root (`PYTEST_DEBUG_TEMPROOT`, `TMPDIR`), not the machine's shared
    ``pytest-of-<user>``, whose cleanup at exit another suite's sessions can hold up. This
    process's own temporary directory is fixed first, so only the sessions it starts move."""
    tempfile.gettempdir()
    own = tmp_path / "check-tmp"
    own.mkdir()
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(own))
    monkeypatch.setenv("TMPDIR", str(own))
    run, _, _ = _run_with_worktree(tmp_path, NESTED)
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts={"check": check})
    (place.path / "tool.py").write_text("VERSION = 2\n")
    said = _command(place)
    assert said.returncode == 0 and "PASSED" in said.stdout, said.stdout + said.stderr
    assert "refactory_failure_log" not in said.stdout


def test_the_command_prints_the_selection_size_and_the_last_time_before_it_runs(tmp_path):
    run, _, _ = _run_with_worktree(tmp_path, PINNED)
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts={"check": CHECK})
    (place.path / "tool.py").write_text("VERSION = 2\n")
    first = _command(place).stdout
    plan = "this run: 1 test file(s), 0 test id(s) named; no earlier run here has been timed."
    assert plan in first and first.index(plan) < first.index("running:")
    (place.path / PIN).write_text("import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n")
    second = _command(place).stdout
    assert re.search(r"this run: 1 test file\(s\), 0 test id\(s\) named; the last finished run "
                     r"here took about \d+ s\.", second), second
    assert second.index("the last finished run") < second.index("running:")
    # A longer run is told in minutes, and an id counts as an id in its file.
    assert targeted_tests._took(330) == "the last finished run here took about 6 min"
    assert targeted_tests._size([f"{PIN}::test_pin", "tests/test_other.py"]) == (
        "2 test file(s), 1 test id(s) named")


def test_a_record_or_log_that_cannot_be_written_is_warned_about_and_the_run_goes_on(tmp_path):
    """The record's folder and the log cannot be made (a file stands where the folder goes):
    each is warned about by path, and the tests still run and report."""
    run, _, _ = _run_with_worktree(tmp_path, PINNED)
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts={"check": CHECK})
    (place.path / "tool.py").write_text("VERSION = 2\n")
    own = place.path / ".refactory"
    import shutil
    shutil.rmtree(own, ignore_errors=True)
    own.write_text("not a folder\n")
    said = _command(place)
    assert said.returncode != 0 and f"FAILED {PIN}::test_pin" in said.stdout, said.stdout
    assert f"warning: could not make {own}" in said.stderr, said.stderr
    assert f"could not open the output log {own / targeted_tests.LOG_FILE}" in said.stderr, \
        said.stderr


def test_remember_warns_on_write_failure(tmp_path, capsys):
    """When the record cannot be written, a warning naming the path is printed to stderr.
    Uses a directory in place of the file (works even as root, unlike chmod)."""
    record_path = tmp_path / "record" / targeted_tests.LAST_RUN
    # Create a directory where the JSON file should go, so writing the file fails.
    record_path.mkdir(parents=True)
    targeted_tests._remember(record_path, ["tests/test_x.py::test_y"], 1.0)
    captured = capsys.readouterr()
    assert "warning" in captured.err
    assert str(record_path) in captured.err


def test_record_and_log_never_reach_the_change(tmp_path):
    """The record, the failure stream, the pytest plugin and the log are inside .refactory/,
    which is excluded from delivery. After a targeted run they must not appear in git status
    or the emitted diff."""
    run, _, _ = _run_with_worktree(tmp_path, PINNED)
    place = workspace.prepare(run, "emit", BUILDER_WORKSPACE, facts={"check": CHECK})
    (place.path / "tool.py").write_text("VERSION = 2\n")
    result = _command(place)
    assert result.returncode != 0  # pin fails
    # The record and log exist in the workspace.
    assert (place.path / ".refactory" / targeted_tests.LAST_RUN).exists()
    assert (place.path / ".refactory" / targeted_tests.LOG_FILE).exists()
    assert (place.path / ".refactory" / targeted_tests.FAILURES).read_text() == f"{PIN}::test_pin\n"
    plugin = place.path / ".refactory" / targeted_tests.PLUGINS / f"{targeted_tests.PLUGIN}.py"
    assert plugin.exists()
    # They do not appear in git status (the workspace's .gitignore or exclusion hides them).
    status = subprocess.run(
        ["git", "-C", str(place.path), "status", "--porcelain", "-uall"],
        capture_output=True, text=True, check=True).stdout
    assert targeted_tests.LAST_RUN not in status
    assert targeted_tests.LOG_FILE not in status
    assert targeted_tests.FAILURES not in status and targeted_tests.PLUGIN not in status
    assert status == " M tool.py\n", status
    # They do not appear in the emitted diff.
    emission = _emission(place)
    assert targeted_tests.LAST_RUN not in emission
    assert targeted_tests.LOG_FILE not in emission
    assert targeted_tests.FAILURES not in emission and targeted_tests.PLUGIN not in emission
