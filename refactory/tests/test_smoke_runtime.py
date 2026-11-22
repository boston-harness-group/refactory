"""Real runtime/model tests; --run-integration, optional REFACTORY_SMOKE_PROJECT."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from refactory_dsh import sandbox
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
        from refactory_dsh import dsh_runtime, dsh_profile, lowering, workers, steering

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

        from refactory_dsh.owned_process import _termination_as_exit
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
        from refactory_dsh.owned_process import shutdown
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
        monkeypatch.setattr(sandbox, "_progress_at", lambda w: {})

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
        assert "timeout" in result.stderr

    def test_run_on_dsh_inactivity_progress_files_count_as_activity(self, tmp_path, monkeypatch):
        """Progress file changes in the inner workdir reset the activity timestamp.

        A long agent call that writes worker events but prints nothing must not
        trip the inner inactivity limit.  The test uses a short inactivity
        limit (1 s) and a process that exits after several progress-thread
        polls.  Without progress tracking the run would time out; with it the
        progress changes keep resetting the activity timestamp.
        """
        from refactory_dsh.dsh_runtime import run_on_dsh

        # Process exits normally after 5 poll() calls from the progress thread.
        proc = _FakeProcess(exit_after=5)
        self._mock_dsh_deps(monkeypatch, tmp_path, proc)

        # Speed up the progress watcher thread so it checks every 0.05 s
        # instead of 1 s — real time, no monkeypatch on time.monotonic needed.
        _real_sleep = time.sleep

        def _fast_sleep(secs: float) -> None:
            _real_sleep(min(secs, 0.05))

        monkeypatch.setattr(time, "sleep", _fast_sleep)

        # Progress changes on every call, resetting activity.
        progress_call_count = [0]

        def changing_progress(w):
            progress_call_count[0] += 1
            return {"step": progress_call_count[0]}

        monkeypatch.setattr(sandbox, "_progress_at", changing_progress)

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
