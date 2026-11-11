"""A model session that answered is kept when its worker's cleanup fails afterwards.

Under load, the worker bridge's cleanup after a finished answer (waiting for the worker's
process scope to empty) sometimes failed: the bridge exited 1 with a teardown failure,
and the client threw the answer away and asked the model again, recording nothing and
overwriting the first session's usage. On the claude_code backend each retry also got a
fresh time limit, so nodes outran their wall. Now:

- the bridge exits 0 after the answer when only its cleanup failed, with a warning line;
  the client keeps such an answer either way and records a `worker_teardown_failed` event;
- the client re-asks only after a wire failure, recording each retry with its cause and
  the discarded attempt's usage, and the worker observer adds up a receipt's sessions;
- a node with a wall has one deadline on every backend, which its retries share.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from refactory_dsh import model as model_mod
from refactory_dsh.model import (RETRY_EVENT, SHUTDOWN_EVENT, TEARDOWN_EVENT, TEARDOWN_WARNING,
                                 DshModelClient)
from refactory_dsh.node_retry import RETRY_EVENT as NODE_RETRY_EVENT
from refactory_dsh.optim.base import item_errors
from test_crashed_step_retry import RIGHT, RUNTIMES, _events, _keys, _pass, _run, _sessions, _straight

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "refactory_dsh"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="needs Node")

#: What the worker reported in the live LiveMath run: the CLI exited 0, then cleanup failed.
TEARDOWN = ("dsh: worker failed: subagent-claude-code: Product subagent failure (product: "
            "Claude Code; stage: teardown; category: unknown; exit code: 0)")

FAKE = """import fcntl, json, pathlib, signal, subprocess, sys, time
root = pathlib.Path({root!r})
plan = json.loads((root / "plan.json").read_text())
sys.stdin.read()
with open(root / "lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    counter = root / "calls"
    calls = json.loads(counter.read_text()) if counter.exists() else 0
    counter.write_text(json.dumps(calls + 1))
step = plan[calls] if calls < len(plan) else plan[-1]
if step.get("hang"):
    # Outlive the deadline, then ignore the shutdown request.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(step["hang"])
if step.get("usage") is not None:
    subprocess.run([{node!r}, {driver!r}, {observer!r}, "session-" + str(calls), str(step["usage"])],
                   check=True)
if step.get("out") is not None:
    print(step["out"])
if step.get("err"):
    print(step["err"], file=sys.stderr)
sys.exit(step.get("code", 0))
"""


class Fake:
    """A scripted ``dsh``: each session takes the next step of the plan (the last
    repeats): ``out`` to stdout, ``err`` to stderr, exit ``code``; ``usage`` tokens are
    reported through the real worker observer, as DSH would."""

    def __init__(self, root: Path, plan: list[dict]):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        (root / "plan.json").write_text(json.dumps(plan))
        script = root / "fake_dsh.py"
        script.write_text(FAKE.format(root=str(root), node=str(NODE or "node"),
                                      driver=str(HERE / "observer_session_driver.mjs"),
                                      observer=str(PACKAGE / "worker_observer.mjs")))
        self.cli = root / "dsh"
        self.cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
        self.cli.chmod(0o755)

    def calls(self) -> int:
        path = self.root / "calls"
        return json.loads(path.read_text()) if path.exists() else 0

    def client(self) -> DshModelClient:
        # Three attempts and no waits: before the fix, the client re-asked a teardown.
        return DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli=str(self.cli),
                              attempts=3, retry_window_s=0, timeout_s=60)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_an_answer_whose_cleanup_failed_is_kept_on_both_runtimes(tmp_path, runtime):
    """[both runtimes] Every solve answers, then the worker's cleanup fails and it exits 1
    (the walker's Executor._visit_agent and the spine's worker_node both reach the client
    through node_retry.deliver): one session per item, every answer kept and scored, and
    a warning recorded in each item's pass."""
    dsh = Fake(tmp_path / "dsh", [{"out": "right", "err": TEARDOWN, "code": 1}])
    scores, run = _run(runtime, _straight(), tmp_path, dsh)
    assert _keys(scores) == RIGHT
    assert dsh.calls() == 4, "a finished answer was asked again"
    for item in "0123":
        directory = _pass(run, item)
        [event] = _events(directory, TEARDOWN_EVENT)
        assert event["node_id"] == "solve" and event["state"] == "warning"
        assert _events(directory, RETRY_EVENT) == []
        assert _sessions(directory) == ["completed"]


@needs_node
def test_a_retry_is_recorded_and_the_receipt_adds_up_both_sessions(tmp_path):
    """A real retry: the first session reports 30 tokens through the observer and fails
    on the wire, the second reports 12 and answers. The retry event carries the cause,
    the attempt and the discarded session's usage; the receipt's usage is 42, with the
    first session kept beside it."""
    dsh = Fake(tmp_path / "dsh", [{"usage": 30, "err": "dsh: TRANSPORT: socket hang up", "code": 1},
                                  {"usage": 12, "out": "right"}])
    run = tmp_path / "run"
    assert dsh.client().complete("Solve.", workdir=run).strip() == "right"
    assert dsh.calls() == 2
    [retried] = _events(run, RETRY_EVENT)
    assert (retried["attempt"], retried["attempts"]) == (1, 3)
    assert retried["cause"] == "dsh: TRANSPORT: socket hang up"
    assert retried["usage"]["tokens"] == 30
    [path] = list((run / ".refactory/workers").glob("worker-*.json"))
    receipt = json.loads(path.read_text())
    assert receipt["usage"]["tokens"] == 42
    assert receipt["session_usage"]["tokens"] == 12
    [earlier] = receipt["earlier_sessions"]
    assert earlier["usage"]["tokens"] == 30 and earlier["runtime_session_id"] == "session-0"
    assert receipt["runtime_session_id"] == "session-1"


@needs_node
def test_a_third_session_counts_each_earlier_one_once(tmp_path):
    dsh = Fake(tmp_path / "dsh", [{"usage": 30, "err": "dsh: TRANSPORT: socket hang up", "code": 1},
                                  {"usage": 12, "err": "dsh: TRANSPORT: ECONNRESET", "code": 1},
                                  {"usage": 5, "out": "right"}])
    run = tmp_path / "run"
    assert dsh.client().complete("Solve.", workdir=run).strip() == "right"
    [path] = list((run / ".refactory/workers").glob("worker-*.json"))
    receipt = json.loads(path.read_text())
    assert receipt["usage"]["tokens"] == 47
    assert [s["usage"]["tokens"] for s in receipt["earlier_sessions"]] == [30, 12]
    assert [e["usage"]["tokens"] for e in _events(run, RETRY_EVENT)] == [30, 12]


def test_retries_on_a_native_backend_share_the_nodes_wall(monkeypatch):
    """claude_code, wall == timeout_s: attempt 1 fails on the wire after 30 s, so attempt
    2 gets what remains of the same deadline (wall - 30 s - the wait), not a fresh
    timeout_s."""
    clock = [1000.0]
    seen: list[tuple[float | None, float]] = []

    class Wire:
        returncode, stdout, stderr = 1, "", "dsh: TRANSPORT: socket hang up"

    class Done:
        returncode, stdout, stderr = 0, "done\n", ""

    def run(self, cmd, run_dir, stop_at=None, budget_s=None, **kwargs):  # type: ignore[no-untyped-def]
        seen.append((stop_at, clock[0]))
        clock[0] += 30
        return Wire() if len(seen) == 1 else Done()

    monkeypatch.setattr(DshModelClient, "_run_completion", run)
    monkeypatch.setattr(model_mod.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(model_mod.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    client = (DshModelClient(worker_backend="claude_code", timeout_s=600, retry_window_s=10)
              .with_tool_policy("none").with_wall(600))
    assert client.retry_attempts() == 3
    assert client.complete("hello") == "done\n"
    (first_stop, first_at), (second_stop, second_at) = seen
    assert first_stop == pytest.approx(first_at + 600)
    assert second_stop == first_stop, "a retry got a fresh deadline"
    assert second_stop - second_at == pytest.approx(600 - 30 - (second_at - first_at - 30))


def _bridge(tmp_path: Path, stop: str, failing: str) -> dict:
    out = tmp_path / f"bridge-{stop}-{failing}.json"
    subprocess.run([str(NODE), str(HERE / "worker_bridge_teardown_test.mjs"), str(PACKAGE / "worker_bridge.mjs"),
                    stop, failing, str(out)], check=True, timeout=60, capture_output=True)
    return json.loads(out.read_text())


@needs_node
@pytest.mark.parametrize("failing, warnings", [("worker", 1), ("parent", 1), ("both", 2), ("none", 0)])
def test_the_bridge_exits_0_when_only_cleanup_failed_after_the_answer(tmp_path, failing, warnings):
    """The real bridge: the worker completed and its answer is on stdout; a failing
    cleanup (the worker's, the parent's or both) is a warning line, and the exit is 0."""
    done = _bridge(tmp_path, "completed", failing)
    assert done["code"] == 0
    assert done["stdout"] == "the answer\n"
    assert done["stderr"].count(TEARDOWN_WARNING) == warnings
    assert "dsh: worker failed" not in done["stderr"]
    assert all("stage: teardown" in line for line in done["stderr"].splitlines())


@needs_node
def test_the_bridge_still_exits_1_for_a_worker_that_did_not_complete(tmp_path):
    stopped = _bridge(tmp_path, "cancelled", "worker")
    assert stopped["code"] == 1
    assert "dsh: worker cancelled: stopped" in stopped["stderr"]
    assert TEARDOWN_WARNING in stopped["stderr"]


def test_the_bridge_and_the_client_name_the_same_warning():
    assert f"const TEARDOWN_WARNING = '{TEARDOWN_WARNING}'" in (PACKAGE / "worker_bridge.mjs").read_text()


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_timeout_whose_shutdown_hangs_is_still_a_timeout(tmp_path, runtime, monkeypatch):
    """[both runtimes] Item 0's session outlives its 2 s limit and then ignores the
    shutdown request (forced exit, cleanup unconfirmed). The item is a timeout -- error
    kind ``timeout``, a ``CompletionTimeout`` at ``solve`` whose message says it timed
    out and carries the shutdown failure -- with a warning recorded; it is not retried.
    The other items answer."""
    from refactory_dsh import owned_process

    monkeypatch.setattr(owned_process, "SHUTDOWN_GRACE_S", 1)  # the walker's; the spine's is 10 s
    dsh = Fake(tmp_path / "dsh", [{"hang": 30}, {"out": "right"}])
    scores, run = _run(runtime, _straight(timeout_s=2), tmp_path, dsh)
    by_id = {str(record["id"]): record for record in scores}
    assert [by_id[item]["passed"] for item in "123"] == [True, True, True]
    timed_out = by_id["0"]
    assert timed_out["error_kind"] == "timeout"
    assert "model completion timed out after 2s" in timed_out["error_text"]
    assert "did not finish shutdown" in timed_out["error_text"]
    [error] = item_errors([timed_out])
    assert (error["kind"], error["node"]) == ("CompletionTimeout", "solve")
    assert error["first"].startswith("model completion timed out")
    directory = _pass(run, "0")
    [warning] = _events(directory, SHUTDOWN_EVENT)
    assert warning["state"] == "warning" and "did not finish shutdown" in warning["error"]
    assert _events(directory, NODE_RETRY_EVENT) == []
    assert dsh.calls() == 4


def test_a_timeout_with_unconfirmed_cleanup_is_never_a_retried_timeout():
    from refactory_dsh.model import CompletionTimeout
    from refactory_dsh.node_retry import failure_kind

    plain = CompletionTimeout("model completion timed out after 2s; requested runtime shutdown")
    assert failure_kind(plain) == "timeout"
    hung = CompletionTimeout("model completion timed out after 2s; requested runtime shutdown; "
                             "model runtime did not finish shutdown within 10s; forced exit; "
                             "child cleanup unconfirmed; not retried")
    hung.cleanup_unconfirmed = True  # type: ignore[attr-defined]
    assert failure_kind(hung) is None
