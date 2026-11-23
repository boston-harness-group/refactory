"""Real runtime/model tests; --run-integration, optional REFACTORY_SMOKE_PROJECT."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from refactory_dsh import activity, sandbox
from refactory_dsh.dsh_runtime import track_run as _real_track
from refactory_dsh.dsh_runtime import untrack_run as _real_untrack
from refactory_dsh.model import DshModelClient
from test_sandbox import GOOD, NAME, _stage

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("input_exists", [False, True], ids=["missing-input", "valid-input"])
def test_smoke_enforces_production_agent_inputs(tmp_path, input_exists):
    files = dict(GOOD)
    files["mode.py"] = files["mode.py"].replace(
        "g.add(nn.Node('start', 'op', call=OPS + ':start',"
        " writes=('state.json',), produces=('value',), knobs=('target',)))",
        "g.add(nn.Node('start', 'agent', reads=('input.txt',), writes=('result.txt',),"
        " prompt_template='Write exactly ok followed by a newline to result.txt using a file-writing tool.'))",
    )
    assert files["mode.py"] != GOOD["mode.py"]
    _stage(tmp_path, files)
    workdir = tmp_path / "run"
    workdir.mkdir()
    if input_exists:
        (workdir / "input.txt").write_text("Available input.\n")
    client = DshModelClient(
        project=Path(os.environ.get("REFACTORY_SMOKE_PROJECT", str(tmp_path))),
    )
    result = sandbox.smoke(
        f"{NAME}/counter", workdir, project=tmp_path,
        model=True, model_context=client.context(),
        timeout_s=180 if input_exists else 40,
    )
    if input_exists:
        assert result.ok, result.problem()
        assert result.payload.get("exit_ok") is True, result.payload
        assert result.payload["score"] == 3.0
        assert (workdir / "result.txt").read_text() == "ok\n"
    else:
        # The graph driver may wait for steering after a worker failure. Its
        # persisted failure is authoritative even when smoke reaches its bound.
        status_path = workdir / ".refactory" / "spine-status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        detail = (result.problem() + str(result.payload.get("error", ""))
                  + str(status.get("last_failure", {}).get("error", "")))
        assert not result.timed_out, result.problem()
        assert not result.ok or result.payload.get("exit_ok") is False, result.payload
        assert "required worker input is missing: input.txt" in detail, detail
        assert not (workdir / "result.txt").exists()


# ── run_on_dsh inactivity tests (mocked, not integration) ──────────────────

class _FakeProcess:
    """A minimal subprocess.Popen stand-in for testing run_on_dsh poll loops."""

    def __init__(self, *, exit_after: int = 0):
        self.pid = os.getpid()
        self.returncode = None
        self._exit_after = exit_after
        self._calls = 0
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()
        self._terminated = False

    def poll(self):
        self._calls += 1
        if self._exit_after and self._calls >= self._exit_after:
            self.returncode = 0
            return 0
        if self._terminated:
            self.returncode = -15
            return -15
        return None

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        raise subprocess.TimeoutExpired("cmd", timeout or 0)

    def terminate(self):
        self._terminated = True
        self.returncode = -15

    def kill(self):
        self._terminated = True
        self.returncode = -9

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeStream:
    """A stream that immediately yields nothing (empty iterator)."""

    def __iter__(self):
        return iter([])

    def fileno(self):
        raise OSError("fake stream")


@pytest.mark.no_integration
class TestRunOnDshInactivity:
    """Tests for run_on_dsh inactivity parameter — not integration tests."""

    def _mock_dsh_deps(self, monkeypatch, tmp_path, process):
        """Patch all external dependencies of run_on_dsh."""
        from refactory_dsh import dsh_profile, dsh_runtime, lowering, workers

        monkeypatch.setattr(dsh_profile, "model_home", lambda x: tmp_path / "home")
        monkeypatch.setattr(dsh_profile, "requirement", lambda **kw: None)
        monkeypatch.setattr(dsh_profile, "stored_model_choice",
                            lambda h: {"provider": "test", "model": "test"})

        from refactory_dsh import paths
        monkeypatch.setattr(paths, "require_build", lambda **kw: None)
        monkeypatch.setattr(paths, "dsh_command", lambda **kw: ["echo"])
        monkeypatch.setattr(paths, "venv_bin", lambda: Path("/usr/bin"))

        monkeypatch.setattr(lowering, "write_mode_graph", lambda *a, **kw: None)
        monkeypatch.setattr(lowering, "write_declarations", lambda *a, **kw: None)
        monkeypatch.setattr(lowering, "knob_environment", lambda s: {lowering.KNOBS_ENV: "{}"})

        monkeypatch.setattr(workers, "environment", lambda ctx: {})
        monkeypatch.setattr(workers, "session_patch", lambda wd: None)

        import contextlib
        monkeypatch.setattr("refactory_dsh.owned_process._termination_as_exit",
                            contextlib.nullcontext)

        monkeypatch.setattr(dsh_runtime, "subprocess", type("FakeSub", (), {
            "Popen": lambda *a, **kw: process,
            "TimeoutExpired": subprocess.TimeoutExpired,
            "PIPE": subprocess.PIPE,
        })())
        monkeypatch.setattr(dsh_runtime, "track_run", lambda p: None)
        monkeypatch.setattr(dsh_runtime, "untrack_run", lambda p: None)
        # The fake child is this process: there is no group of its own to end.
        monkeypatch.setattr(dsh_runtime, "_run_group", lambda p: None)
        monkeypatch.setattr("refactory_dsh.owned_process.shutdown", lambda p, **kw: None)

    def test_run_on_dsh_inactivity_true_stops_on_idle(self, tmp_path, monkeypatch):
        """With inactivity=True, an idle run is stopped with timeout_kind='inactivity'."""
        from refactory_dsh.dsh_runtime import run_on_dsh

        proc = _FakeProcess()
        self._mock_dsh_deps(monkeypatch, tmp_path, proc)
        # Make time advance quickly so inactivity fires
        real_monotonic = time.monotonic
        call_count = [0]

        def fast_time():
            call_count[0] += 1
            # Each call jumps ahead 100 seconds
            return real_monotonic() + call_count[0] * 100

        monkeypatch.setattr(time, "monotonic", fast_time)
        monkeypatch.setattr(activity.Watch, "look", lambda self: ())

        workdir = tmp_path / "run"
        workdir.mkdir(parents=True)

        result = run_on_dsh(
            {"graph": {"nodes": {}}}, workdir,
            timeout_s=10, inactivity=True,
        )
        assert result.returncode == -1
        assert result.timeout_kind == "inactivity"
        assert "inactivity" in result.stderr

    def test_run_on_dsh_inactivity_false_preserves_wall_clock(self, tmp_path, monkeypatch):
        """With inactivity=False (default), wall-clock deadline is used."""
        from refactory_dsh.dsh_runtime import run_on_dsh

        proc = _FakeProcess()
        self._mock_dsh_deps(monkeypatch, tmp_path, proc)
        real_monotonic = time.monotonic
        call_count = [0]

        def fast_time():
            call_count[0] += 1
            return real_monotonic() + call_count[0] * 100

        monkeypatch.setattr(time, "monotonic", fast_time)

        workdir = tmp_path / "run"
        workdir.mkdir(parents=True)

        result = run_on_dsh(
            {"graph": {"nodes": {}}}, workdir,
            timeout_s=10, inactivity=False,
        )
        assert result.returncode == -1
        assert result.timeout_kind is None
        # Main's text, which `authoring_evidence.failure_kind` reads as a timeout.
        assert result.stderr == "dsh timed out after 10s; owned runtime shutdown requested"

    def test_run_on_dsh_inactivity_progress_files_count_as_activity(self, tmp_path, monkeypatch):
        """Writes under the inner workdir reset the activity timestamp.

        A long agent call that writes worker events but prints nothing must not
        trip the inner inactivity limit.  The test uses a short inactivity
        limit (1 s) and a process that exits after several watcher looks.
        Without the run directory watched the run would time out; with it the
        changes keep resetting the activity timestamp.
        """
        from refactory_dsh.dsh_runtime import run_on_dsh

        # Process exits normally after 5 poll() calls from the watcher thread.
        proc = _FakeProcess(exit_after=5)
        self._mock_dsh_deps(monkeypatch, tmp_path, proc)

        # The watcher looks every 0.05 s instead of every 0.5 s.
        monkeypatch.setattr(sandbox, "WATCH_INTERVAL_S", 0.05)

        # The run directory changes on every look, resetting activity.
        progress_call_count = [0]

        def changing_progress(self):
            progress_call_count[0] += 1
            return (("events", progress_call_count[0], 0),)

        monkeypatch.setattr(activity.Watch, "look", changing_progress)

        workdir = tmp_path / "run"
        workdir.mkdir(parents=True)

        result = run_on_dsh(
            {"graph": {"nodes": {}}}, workdir,
            timeout_s=1, inactivity=True,
        )
        # The run should complete normally (not time out) because progress
        # changes keep resetting the activity timestamp.
        assert result.returncode == 0
        assert result.timeout_kind is None
        assert progress_call_count[0] >= 3  # progress was actually checked

    @pytest.mark.parametrize("finishes", [True, False], ids=["finished", "timed-out"])
    def test_no_watcher_outlives_the_run(self, tmp_path, monkeypatch, finishes):
        """However the run ends, its watcher is stopped and joined before `run_on_dsh`
        returns: one left looking would mark the sandbox's activity file after the run
        was reported stopped. Each look takes 0.3 s here and the watcher barely pauses
        between them, so one that was not joined is still looking when the run returns."""
        import threading

        from refactory_dsh.dsh_runtime import run_on_dsh

        proc = _FakeProcess()
        if finishes:  # it ends on its own, after the watcher's first look
            threading.Timer(0.3, lambda: setattr(proc, "returncode", 0)).start()
        self._mock_dsh_deps(monkeypatch, tmp_path, proc)
        monkeypatch.setattr(sandbox, "WATCH_INTERVAL_S", 0.001)

        def slow_look(self):
            time.sleep(0.3)
            return ()

        monkeypatch.setattr(activity.Watch, "look", slow_look)
        workdir = tmp_path / "run"
        workdir.mkdir(parents=True)
        result = run_on_dsh({"graph": {"nodes": {}}}, workdir, timeout_s=1, inactivity=True,
                            activity_file=workdir / sandbox.ACTIVITY_FILE)
        assert (result.timeout_kind is None) == finishes, result.stderr
        assert [t for t in threading.enumerate()
                if t.name == "dsh-progress-watch" and t.is_alive()] == []


    def test_the_runs_own_budget_ledger_is_never_work(self, tmp_path, monkeypatch):
        """The run's budget ledger moves with every model call's accounting, not with the
        run's work: wherever it lives -- here among the workers' files, which are signs
        of work -- a run whose only change is its ledger is stopped for inactivity."""
        import threading

        from refactory_dsh.dsh_runtime import run_on_dsh

        proc = _FakeProcess()
        self._mock_dsh_deps(monkeypatch, tmp_path, proc)
        monkeypatch.setattr(sandbox, "WATCH_INTERVAL_S", 0.05)
        workdir = tmp_path / "run"
        ledger = workdir / ".refactory" / "workers" / "budget-ledger.json"
        ledger.parent.mkdir(parents=True)
        monkeypatch.setenv("REFACTORY_BUDGET_LEDGER", str(ledger))
        done = threading.Event()

        def account() -> None:
            tick = 0
            while not done.wait(0.05):
                tick += 1
                ledger.write_text(str(tick))

        accountant = threading.Thread(target=account, daemon=True)
        accountant.start()
        try:
            result = run_on_dsh({"graph": {"nodes": {}}}, workdir, timeout_s=1, inactivity=True)
        finally:
            done.set()
            accountant.join()
        assert result.timeout_kind == "inactivity", result.stderr

    def test_a_run_whose_bookkeeping_fails_is_no_longer_tracked(self, tmp_path, monkeypatch):
        """Whatever fails once the DSH child has started -- here reading its process
        group -- the run leaves the tracked set (`shutdown_tracked_runs` would otherwise
        signal a process that is no longer this run's)."""
        from refactory_dsh import dsh_runtime

        proc = _FakeProcess(exit_after=1)
        self._mock_dsh_deps(monkeypatch, tmp_path, proc)
        monkeypatch.setattr(dsh_runtime, "track_run", _real_track)
        monkeypatch.setattr(dsh_runtime, "untrack_run", _real_untrack)

        def broken(process):
            raise RuntimeError("the group could not be read")

        monkeypatch.setattr(dsh_runtime, "_run_group", broken)
        workdir = tmp_path / "run"
        workdir.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="could not be read"):
            dsh_runtime.run_on_dsh({"graph": {"nodes": {}}}, workdir, timeout_s=10, inactivity=True)
        assert proc not in dsh_runtime._TRACKED_RUNS

    def test_a_process_with_no_readable_group_is_run_and_cleaned_up_as_any(self):
        """A child whose group cannot be read -- no pid, or already gone -- has none to
        end; reading it never fails the run."""
        from refactory_dsh import dsh_runtime

        class _NoPid:
            pass

        assert dsh_runtime._run_group(_NoPid()) is None
        assert dsh_runtime._run_group(_FakeProcess()) is None  # this process's own group

# ── a node command in a watched run: its output, and its process group ─────

def _node(tmp_path: Path, body: str, *, watched: bool, stdout: object = subprocess.PIPE,
          wait: bool = True) -> object:
    """``python -m refactory_dsh.run probe:node``, as the spine runs a node command, with
    the run's watch directory named (`activity.WATCH_ENV`) or not."""
    import sys

    from refactory_dsh import lowering
    from refactory_dsh.run import NODE_ENV

    (tmp_path / "probe.py").write_text(body)
    declarations = tmp_path / lowering.DECLARATIONS
    declarations.parent.mkdir(parents=True, exist_ok=True)
    declarations.write_text(json.dumps({"node": {"reads": [], "writes": [], "produces": []}}))
    env = {key: value for key, value in os.environ.items() if key != activity.WATCH_ENV}
    env.update({lowering.KNOBS_ENV: "{}", "REFACTORY_WORKDIR": str(tmp_path),
                NODE_ENV: json.dumps({"id": "node", "type": "FnNode", "reads": [], "writes": []})})
    if watched:
        env[activity.WATCH_ENV] = str(activity.watch_dir(tmp_path))
    argv = [sys.executable, "-m", "refactory_dsh.run", "probe:node"]
    if not wait:
        return subprocess.Popen(argv, env=env, cwd=str(tmp_path), stdout=stdout,
                                stderr=subprocess.DEVNULL)
    return subprocess.run(argv, env=env, cwd=str(tmp_path), stdout=stdout,
                          stderr=subprocess.PIPE, timeout=60, check=False)


@pytest.mark.no_integration
@pytest.mark.parametrize("watched", [True, False])
def test_a_node_that_ends_abruptly_keeps_all_its_output(tmp_path, watched):
    """The tee is a process of its own, so a node ending with ``os._exit`` (no flush, no
    cleanup in the node's process) still has every byte it wrote reach the spine."""
    lines = [f"line {i:06d} " + "x" * 40 for i in range(4000)] + ["the last line"]
    body = ("import os, sys\n"
            "def node(ctx):\n"
            f"    data = {chr(10).join(lines) + chr(10)!r}.encode()\n"
            "    view = memoryview(data)\n"
            "    while view:\n"
            "        view = view[os.write(1, view):]\n"
            "    os._exit(0)\n")
    result = _node(tmp_path, body, watched=watched)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().splitlines() == lines
    assert (activity.watch_dir(tmp_path) / activity.MARK).exists() == watched


@pytest.mark.no_integration
@pytest.mark.parametrize("watched", [True, False])
def test_only_a_watched_node_goes_through_the_tee(tmp_path, watched):
    """A node of a run nobody watches writes straight to its own stdout, as before: no
    pipe, no mark, no process record. A watched one writes into the tee's pipe."""
    body = ("import json, os, stat\n"
            "from pathlib import Path\n"
            "def node(ctx):\n"
            "    mode = os.fstat(1).st_mode\n"
            "    Path('fd1.json').write_text(json.dumps({'fifo': stat.S_ISFIFO(mode),\n"
            "        'records': len(list(Path('.refactory/watch/processes').glob('*.json')))}))\n"
            "    print('done', flush=True)\n")
    with (tmp_path / "out.txt").open("wb") as out:
        result = _node(tmp_path, body, watched=watched, stdout=out)
    assert result.returncode == 0, result.stderr
    seen = json.loads((tmp_path / "fd1.json").read_text())
    assert seen == {"fifo": watched, "records": 1 if watched else 0}
    assert (tmp_path / "out.txt").read_text() == "done\n"
    # The record is gone once the node has finished.
    assert not list((activity.watch_dir(tmp_path) / activity.PROCESSES).glob("*.json"))


@pytest.mark.no_integration
def test_a_watched_gate_still_ends_on_its_verdict(tmp_path):
    """Everything a gate prints goes through the same pipe in order, so its verdict is
    still the last line of its stdout."""
    body = ("from refactory_dsh.nn import Verdict\n"
            "def node(ctx):\n"
            "    for i in range(2000):\n"
            "        print('log', i)\n"
            "    return Verdict('reloop', 'two tests failed')\n")
    result = _node(tmp_path, body, watched=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().splitlines()[-2:] == ["two tests failed", "reloop"]


@pytest.mark.no_integration
def test_a_watched_node_that_leaves_a_child_running_ends_at_once(tmp_path):
    """Nothing waits for the tee: a node whose child still holds its output ends as soon
    as it returns, and what the child prints later still arrives."""
    body = ("import subprocess\n"
            "def node(ctx):\n"
            "    subprocess.Popen(['sh', '-c', 'sleep 3; echo late'])\n"
            "    print('early', flush=True)\n")
    with (tmp_path / "out.txt").open("wb") as out:
        started = time.monotonic()
        node = _node(tmp_path, body, watched=True, stdout=out, wait=False)
        returncode = node.wait(timeout=30)
        took = time.monotonic() - started
    assert returncode == 0
    assert took < 2.5, took
    deadline = time.monotonic() + 10
    while "late" not in (tmp_path / "out.txt").read_text() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert (tmp_path / "out.txt").read_text() == "early\nlate\n"


@pytest.mark.no_integration
@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_a_stopped_run_ends_the_groups_its_nodes_recorded_and_no_other(tmp_path):
    """`activity.end_recorded` ends a node command's own process group -- as DSH spawns
    them, a group the run's own does not reach -- and leaves alone a group whose
    recorded process is no longer the one that wrote the record (pid reuse)."""
    import sys

    watch = activity.watch_dir(tmp_path)
    recorder = ("import os, sys, time\n"
                "from refactory_dsh import activity\n"
                "activity.record_process(sys.argv[1])\n"
                "time.sleep(60)\n")
    node = subprocess.Popen([sys.executable, "-c", recorder, str(watch)], start_new_session=True)
    stranger = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not (watch / activity.PROCESSES / f"{node.pid}.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        # A record whose process started at another time: its pid was reused.
        (watch / activity.PROCESSES / f"{stranger.pid}.json").write_text(json.dumps(
            {"pid": stranger.pid, "start": "another time", "group": stranger.pid,
             "group_start": None}))
        assert activity.end_recorded(watch) == [node.pid]
        assert node.wait(timeout=10) is not None
        assert stranger.poll() is None
    finally:
        for process in (node, stranger):  # only what this test started
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.no_integration
def test_a_watched_node_whose_tee_dies_still_gets_its_output_and_verdict_out(tmp_path):
    """Should the tee die mid-run, the node's own output goes back to where it went
    before: what it prints afterwards, its verdict last, still reaches the spine."""
    body = ("import os, signal, time\n"
            "from refactory_dsh import activity\n"
            "from refactory_dsh.nn import Verdict\n"
            "def node(ctx):\n"
            "    os.kill(activity.tee_pid, signal.SIGKILL)\n"
            "    time.sleep(0.5)\n"
            "    print('after the tee', flush=True)\n"
            "    return Verdict('reloop', 'two tests failed')\n")
    result = _node(tmp_path, body, watched=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().splitlines() == ["after the tee", "two tests failed", "reloop"]


@pytest.mark.no_integration
@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_a_recorded_group_whose_id_now_names_another_group_is_left_alone(tmp_path):
    """A node command that is gone, whose group's id now leads another process's group
    (the id was reused), is not signalled: the leader started at another time."""
    import sys

    watch = activity.watch_dir(tmp_path)
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    stranger = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        (watch / activity.PROCESSES).mkdir(parents=True)
        (watch / activity.PROCESSES / f"{gone.pid}.json").write_text(json.dumps(
            {"pid": gone.pid, "start": "when it ran", "group": stranger.pid,
             "group_start": "when the group's first leader ran"}))
        assert activity.end_recorded(watch) == []
        time.sleep(0.2)
        assert stranger.poll() is None
    finally:
        stranger.kill()  # only what this test started
        stranger.wait()


@pytest.mark.no_integration
@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_a_run_whose_dsh_ends_on_its_own_leaves_nothing_running(tmp_path, monkeypatch):
    """DSH exits by itself, leaving a process in its own group and a node command in a
    group of its own that recorded itself: both are ended before `run_on_dsh` returns."""
    import sys

    from refactory_dsh import dsh_profile, dsh_runtime, lowering

    left, detached = tmp_path / "left.pid", tmp_path / "detached.pid"
    node = ("import os, pathlib, sys, time\n"
            "from refactory_dsh import activity\n"
            "activity.record_process()\n"
            f"pathlib.Path({str(detached)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(60)\n")
    # Neither holds DSH's output open, so nothing but the cleanup can end them in time.
    child = ("import os, pathlib, subprocess, sys, time\n"
             "quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
             "kept = subprocess.Popen(['sleep', '60'], **quiet)\n"
             f"pathlib.Path({str(left)!r}).write_text(str(kept.pid))\n"
             f"subprocess.Popen([sys.executable, '-c', {node!r}], start_new_session=True, **quiet)\n"
             f"while not pathlib.Path({str(detached)!r}).exists():\n"
             "    time.sleep(0.05)\n")
    monkeypatch.setattr(dsh_profile, "requirement", lambda **kw: None)
    monkeypatch.setattr(dsh_runtime, "require_build", lambda **kw: None)
    monkeypatch.setattr(dsh_runtime, "dsh_command", lambda **kw: [sys.executable, "-c", child])
    monkeypatch.setattr(lowering, "write_mode_graph", lambda *a, **kw: None)
    monkeypatch.setattr(lowering, "write_declarations", lambda *a, **kw: None)
    started = time.monotonic()
    run = dsh_runtime.run_on_dsh({"knobs": {}}, tmp_path / "run", timeout_s=60, inactivity=True)
    pids = [int(left.read_text()), int(detached.read_text())]
    try:
        assert run.returncode == 0, run.stderr
        assert time.monotonic() - started < 20
        deadline = time.monotonic() + 5
        while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not any(_alive(pid) for pid in pids)
    finally:
        for pid in pids:  # only what this test started, should an assertion above fail
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
