"""A builder's model session is stopped for inactivity, not for taking long.

The session limit of an agent node's model session is an inactivity limit: no model output
(the worker receipt the DSH observer rewrites) and no running tool command for N seconds.
A node's declared `timeout_s` stays as a hard maximum. The in-process tests drive the
model client's inactivity detection with a deterministic fake clock that the test advances,
so the assertions never depend on machine load. One real-subprocess integration test per
feature confirms that the detection works end to end.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from refactory_dsh import lowering, workers
from refactory_dsh.design import BUILDER_RETRY, BUILDER_TIMEOUT_S, DesignMode
from refactory_dsh.fast import DesignFastMode
from refactory_dsh.model import (DEFAULT_INACTIVITY_S, NODE_DISPATCH_OVERHEAD_S, CompletionTimeout,
                                 DeclaredTimeout, DshModelClient, InactivityTimeout)
from refactory_dsh.node_retry import RETRY_EVENT, deliver, node_allowance_s


# ── deterministic fake clock ────────────────────────────────────────────────────

class FakeClock:
    """A deterministic clock that advances only when the test says so."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


def _mock_run_completion(mode: str, duration: float, clock: FakeClock, wall_clock: FakeClock):
    """Return a ``_run_completion`` replacement that simulates subprocess behavior.

    ``mode`` is ``busy`` (receipt touching), ``silent`` (no activity), or ``tool``
    (a tool command running).  The mock advances the fake clock in 0.5 s steps and
    applies the same inactivity detection the real ``_run_completion`` uses.
    """
    def run_completion(self, cmd, run_dir, stop_at=None, budget_s=None, task=None,
                       target_workdir=None):
        inactivity_s = self._inactivity_limit()
        timeout_s = budget_s if budget_s is not None else self._timeout_s
        deadline = stop_at if stop_at is not None else clock() + timeout_s
        last_activity = clock()
        start = clock()
        step = 0.5

        while True:
            clock.advance(step)
            wall_clock.advance(step)
            now = clock()

            if mode == "busy":
                last_activity = now
            elif mode == "tool":
                last_activity = now
            # "silent": last_activity stays at start

            if now >= deadline:
                expired = subprocess.TimeoutExpired(cmd, timeout_s)
                expired.stream_quiet_s = now - last_activity if mode != "busy" else 0.0
                raise expired

            if inactivity_s and now - last_activity >= inactivity_s:
                raise InactivityTimeout(cmd, inactivity_s)

            if now - start >= duration:
                return subprocess.CompletedProcess(cmd, 0, "done\n", "")

    return run_completion


def _deterministic_client(tmp_path: Path, clock: FakeClock, wall_clock: FakeClock,
                          *, inactivity_s: int = 2, wall: int | None = None) -> DshModelClient:
    """A client with injected deterministic clocks, no real subprocess needed."""
    client = DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli="/bin/true",
                            attempts=1, timeout_s=60, inactivity_s=inactivity_s,
                            _monotonic=clock, _wall_clock=wall_clock)
    client = client.for_node("emit", role="builder")
    return client.with_wall(wall) if wall is not None else client


def _complete_deterministic(client: DshModelClient, tmp_path: Path,
                            prompt: str) -> tuple[float, str | Exception]:
    """Like _complete but measures elapsed time with the client's injected clock."""
    started = client._monotonic()
    try:
        result: str | Exception = client.complete(prompt, workdir=tmp_path / "run")
    except Exception as error:  # noqa: BLE001 - the test reports which one
        result = error
    return client._monotonic() - started, result


# ── in-process timing tests (deterministic, load-independent) ───────────────────

def test_a_builder_that_keeps_producing_output_past_the_old_wall_is_not_stopped(tmp_path, monkeypatch):
    # The old limit was a wall (here 3 s, standing for 1800 s). Output keeps arriving for 5 s.
    clock, wall_clock = FakeClock(100.0), FakeClock(1_000_000.0)
    client = _deterministic_client(tmp_path, clock, wall_clock, inactivity_s=2, wall=30)
    monkeypatch.setattr(DshModelClient, "_run_completion",
                        _mock_run_completion("busy", 5, clock, wall_clock))
    elapsed, result = _complete_deterministic(client, tmp_path, "busy 5")
    assert isinstance(result, str) and "done" in result, result
    assert elapsed > 3


def test_a_builder_busy_on_a_long_tool_command_is_not_stopped(tmp_path, monkeypatch):
    # A 5 s test command with no model output meanwhile, against a 2 s inactivity limit.
    clock, wall_clock = FakeClock(100.0), FakeClock(1_000_000.0)
    client = _deterministic_client(tmp_path, clock, wall_clock, inactivity_s=2, wall=30)
    monkeypatch.setattr(DshModelClient, "_run_completion",
                        _mock_run_completion("tool", 5, clock, wall_clock))
    elapsed, result = _complete_deterministic(client, tmp_path, "tool 5")
    assert isinstance(result, str) and "done" in result, result
    assert elapsed > 4


def test_a_silent_builder_is_stopped_at_the_inactivity_limit(tmp_path, monkeypatch):
    clock, wall_clock = FakeClock(100.0), FakeClock(1_000_000.0)
    client = _deterministic_client(tmp_path, clock, wall_clock, inactivity_s=2, wall=60)
    monkeypatch.setattr(DshModelClient, "_run_completion",
                        _mock_run_completion("silent", 40, clock, wall_clock))
    elapsed, result = _complete_deterministic(client, tmp_path, "silent 40")
    assert isinstance(result, CompletionTimeout), result
    assert "timed out after 2s of inactivity" in str(result)
    assert 1.5 < elapsed < 5


def test_the_hard_maximum_still_stops_an_endlessly_busy_builder(tmp_path, monkeypatch):
    clock, wall_clock = FakeClock(100.0), FakeClock(1_000_000.0)
    client = _deterministic_client(tmp_path, clock, wall_clock, inactivity_s=2, wall=5)
    monkeypatch.setattr(DshModelClient, "_run_completion",
                        _mock_run_completion("busy", 60, clock, wall_clock))
    elapsed, result = _complete_deterministic(client, tmp_path, "busy 60")
    # The wall the node declares: the graph's own limit (`DeclaredTimeout`), not retried.
    assert isinstance(result, DeclaredTimeout), result
    assert "declared timeout_s (5s) after 5s" in str(result) and "inactivity" not in str(result)
    assert 4 < elapsed < 8


# ── unit tests (no subprocess, no timing) ───────────────────────────────────────

def test_a_session_without_a_node_has_no_inactivity_limit(tmp_path):
    client = DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli="/bin/true",
                            attempts=1, timeout_s=60, inactivity_s=1)
    node_client = client.for_node("emit", role="builder")
    plain = DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli="/bin/true",
                           attempts=1, timeout_s=60, inactivity_s=1)
    assert node_client._inactivity_limit() == 1 and plain._inactivity_limit() == 0


def test_the_limit_defaults_to_ten_minutes_and_travels_to_the_worker(tmp_path):
    assert DEFAULT_INACTIVITY_S == 600
    client = DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli="/bin/true",
                            attempts=1, timeout_s=60, inactivity_s=7).for_node("emit")
    assert client.context()["inactivity_s"] == 7
    rebuilt = DshModelClient(**client.context()).for_node("emit")
    assert rebuilt._inactivity_limit() == 7
    # Unset, the context is what it was, so a recorded worker spec keeps its bytes.
    plain = DshModelClient(worker_backend="dsh", provider="p", model="m")
    assert "inactivity_s" not in plain.context()


# ── real-subprocess integration test (one per feature, for parity) ──────────────

#: What the scripted session does, by the first word of its prompt.
FAKE_DSH = '''import os, pathlib, subprocess, sys, time
prompt = sys.stdin.read()
receipt = pathlib.Path(os.environ["REFACTORY_WORKER_RECEIPT"])
mode, _, seconds = prompt.strip().splitlines()[-1].partition(" ")
seconds = float(seconds or 0)
end = time.time() + seconds
if mode == "busy":          # model output keeps arriving: the observer rewrites the receipt
    while time.time() < end:
        receipt.touch()
        time.sleep(0.2)
elif mode == "silent":      # nothing at all
    time.sleep(seconds)
elif mode == "tool":        # one long tool command, no model output meanwhile
    subprocess.run(["sleep", str(seconds)])
elif mode == "flaky":       # silent the first time, quick the second
    counter = pathlib.Path(sys.argv[0] + ".calls")
    calls = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(calls + 1))
    time.sleep(0 if calls else seconds)
print("done")
'''


def _client(tmp_path: Path, *, inactivity_s: int = 2, wall: int | None = None) -> DshModelClient:
    script = tmp_path / "fake_dsh.py"
    script.write_text(FAKE_DSH)
    cli = tmp_path / "dsh"
    cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
    cli.chmod(0o755)
    client = DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli=str(cli),
                            attempts=1, timeout_s=60, inactivity_s=inactivity_s)
    client = client.for_node("emit", role="builder")
    return client.with_wall(wall) if wall is not None else client


def _complete(client: DshModelClient, tmp_path: Path, prompt: str) -> tuple[float, str | Exception]:
    started = client._monotonic()
    try:
        result: str | Exception = client.complete(prompt, workdir=tmp_path / "run")
    except Exception as error:  # noqa: BLE001 - the test reports which one
        result = error
    return client._monotonic() - started, result


def test_a_timeout_retry_runs_on_inactivity(tmp_path):
    """#532's retry still fires: the silent session is retried once, fresh, and finishes."""
    run = tmp_path / "run"
    run.mkdir()
    client = _client(tmp_path, inactivity_s=2, wall=60)
    answer, _receipt = deliver(
        run, "emit", prompt="flaky 40", writes=[], reads=[], client=client,
        ask=lambda attempt, sent, place: attempt.complete(sent, workdir=run),
        capture_final=True, rule=BUILDER_RETRY, timeout_s=60)
    assert "done" in answer
    events = [json.loads(line) for line in (run / ".refactory/events.jsonl").read_text().splitlines()]
    retried = [event for event in events if event.get("kind") == RETRY_EVENT]
    assert len(retried) == 1 and "timed out after" in retried[0]["summary"]


@pytest.mark.parametrize("mode_class", [DesignMode, DesignFastMode])
def test_the_builder_declares_the_hard_maximum_and_the_spine_outlasts_it(mode_class, tmp_path):
    state = mode_class().state_dict()
    assert state["graph"]["nodes"]["emit"]["timeout_s"] == BUILDER_TIMEOUT_S == 14400
    workflow = lowering.to_workflow(state)
    lowered = workers.lower_workflow(workflow, tmp_path, DshModelClient().context(),
                                     lowering.node_declarations(state))
    emit = lowered.nodes["emit"]
    # The session ends at the hard maximum (or sooner, on inactivity); the spine's command
    # allows every attempt that maximum plus the dispatch overhead, so it never kills an
    # active builder first.
    attempts = 2 if mode_class is DesignMode else 1
    assert emit.command_timeout_s >= attempts * BUILDER_TIMEOUT_S + NODE_DISPATCH_OVERHEAD_S
    if mode_class is DesignMode:
        assert node_allowance_s(BUILDER_TIMEOUT_S, BUILDER_RETRY) == 2 * BUILDER_TIMEOUT_S
