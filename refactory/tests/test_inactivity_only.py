"""A node's model session ends on inactivity only, never on a wall clock.

A session that keeps showing activity runs as long as it needs; one that shows none for
the inactivity limit is stopped as a timeout; the harness backstop stops only a session
still active past it, and that is the model's own failure (``backstop``): it kept writing.

Each runtime runs the same one-item data pass against a scripted ``dsh`` through the real
model client: the walker (`nn.executor`) in process, the spine through its real drain,
whose dispatched worker (`worker_node`) rebuilds the client from the spec
`workers.lower_workflow` wrote. The scripted session shows activity as the worker
observer does: it touches the session's activity file (`model.activity_path`).
Test-only values stand for production's: 5 s of inactivity for 600 s, and a backstop of
seconds for 14400 s. The per-completion ``timeout_s`` is 3 s, so a session bounded by it
would end long before the active one finishes.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn
from refactory_dsh.items import harness_incident
from refactory_dsh.model import DshModelClient, activity_path
from refactory_dsh.nn.executor import CURRENT_ITEM, execute
from test_crashed_step_retry import RUNTIMES, _sessions, _spine

FAKE_DSH = '''import json, os, pathlib, re, subprocess, sys, time
prompt = sys.stdin.read()
mode, seconds = re.search(r"(active|idle|busy) (\\d+)", prompt).groups()
item = pathlib.Path("current_item.json")
if item.exists() and json.loads(item.read_text()).get("id") != "0":
    seconds = 0                # the quick item
activity = pathlib.Path(os.environ["REFACTORY_WORKER_RECEIPT"]).with_suffix(".activity")
end = time.time() + int(seconds)
if mode == "busy" and int(seconds):   # a process runs beneath a model that streams nothing
    subprocess.Popen(["sleep", seconds])
while time.time() < end:
    if mode == "active":     # a frame streamed every second, as the observer records it
        activity.touch()
    time.sleep(1)
print("right")
'''


class Session:
    """A scripted ``dsh`` whose session is active, or idle, for the seconds its prompt says."""

    def __init__(self, root: Path, *, backstop_s: int):
        root.mkdir(parents=True, exist_ok=True)
        script = root / "fake_dsh.py"
        script.write_text(FAKE_DSH)
        self.cli = root / "dsh"
        self.cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
        self.cli.chmod(0o755)
        self.backstop_s = backstop_s

    def client(self) -> DshModelClient:
        return DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli=str(self.cli),
                              attempts=1, timeout_s=3, inactivity_s=5, backstop_s=self.backstop_s)


def _graph(prompt: str, *, tools: bool = False) -> nn.Graph:
    """data -> [solve] -> report: two items, each answered by one session, with no tools
    unless ``tools``."""
    graph = nn.Graph()
    graph.add(nn.Node("data", "data", subgraph_entry="solve", subgraph_exit="solve",
                      writes=(CURRENT_ITEM,), produces=("item_scores",),
                      task_ref="inactivity_ops:TwoItemTask"))
    graph.add(nn.Node("solve", "agent", prompt_template=prompt, writes=("answer.txt",),
                      **({} if tools else {"tool_policy": "none"}), role="solver"))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("data", "report")
    return graph


def _run(runtime: str, prompt: str, tmp_path: Path, *, backstop_s: int,
         tools: bool = False) -> tuple[dict[str, Any], float, Path]:
    """Run the pass; return item 0's record, how long the pass took, and the run directory."""
    session = Session(tmp_path / "dsh", backstop_s=backstop_s)
    run = tmp_path / runtime
    started = time.monotonic()
    if runtime == "walker":
        result = execute(_graph(prompt, tools=tools), run, model=session.client(), max_steps=50)
        assert result.exit_ok and not result.halted, result.error
        records = result.facts["item_scores"]
    else:
        records = _spine(_graph(prompt, tools=tools), run, tmp_path, session)
    [quick] = [r for r in records if r["id"] == "1"]
    assert quick["passed"] is True, quick
    [record] = [r for r in records if r["id"] == "0"]
    return record, time.monotonic() - started, run


def _item(run: Path) -> Path:
    [path] = list(run.glob("passes/data/0_*"))
    return path


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_an_active_session_runs_past_any_wall(tmp_path, runtime):
    """[both runtimes] (a) Active every second for 20 s, inactivity 5 s, no wall: the
    session completes and its item is scored -- not stopped by inactivity, nor by the
    per-completion timeout_s (3 s), nor by the backstop (well past 20 s)."""
    record, elapsed, run = _run(runtime, "active 20", tmp_path, backstop_s=120)
    assert record["passed"] is True and record["error_kind"] is None, record
    assert elapsed >= 20
    assert _sessions(_item(run)) == ["completed"]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_an_idle_session_ends_on_inactivity(tmp_path, runtime):
    """[both runtimes] (b) Silent past the 5 s inactivity limit: stopped as a timeout,
    "of inactivity", long before its 60 s are up."""
    record, elapsed, run = _run(runtime, "idle 60", tmp_path, backstop_s=120)
    assert record["passed"] is False and record["error_kind"] == "timeout", record
    assert "timed out after 5s of inactivity" in record["error_text"]
    assert elapsed < 40
    assert _sessions(_item(run)) == ["timed_out"]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_the_backstop_fires_only_past_its_value_and_is_the_models_failure(tmp_path, runtime):
    """[both runtimes] (c) The backstop (10 s) leaves a session active for 6 s alone, and
    stops one still active past it: the model's own failure (``backstop``), neither a
    timeout nor a harness incident."""
    record, _, _ = _run(runtime, "active 6", tmp_path / "under", backstop_s=10)
    assert record["passed"] is True and record["error_kind"] is None, record

    record, elapsed, run = _run(runtime, "active 60", tmp_path / "past", backstop_s=10)
    assert record["passed"] is False and record["error_kind"] == "backstop", record
    assert not harness_incident(record), "scored 0, never rerun or left out of the score"
    assert "still writing" in record["error_text"]
    # The spine names it by its worker's own line (`items.WORKER_ERROR_MARKER`), with its
    # type, as the walker does: never by a `worker:` line a quoted reply could hold.
    assert record["error_text"].startswith("solve: HarnessBackstop: model session hit"), record
    assert "hit the harness backstop after 10s" in record["error_text"]
    assert "timed out" not in record["error_text"]
    assert 10 <= elapsed < 45
    assert _sessions(_item(run)) == ["harness_backstop"]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_running_process_is_no_activity_in_a_session_without_tools(tmp_path, runtime):
    """[both runtimes] (d) A session with no tools whose model streams nothing while a
    process runs beneath it (a background job, a loop) ends on inactivity: only the
    model's stream is its sign of life."""
    record, elapsed, run = _run(runtime, "busy 60", tmp_path, backstop_s=120)
    assert record["passed"] is False and record["error_kind"] == "timeout", record
    assert "timed out after 5s of inactivity" in record["error_text"]
    assert elapsed < 40
    assert _sessions(_item(run)) == ["timed_out"]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_tool_that_outlives_a_quiet_model_is_an_inactivity_timeout_at_the_backstop(tmp_path, runtime):
    """[both runtimes] (e) In a session with tools a running tool command keeps the
    session alive past the 5 s inactivity limit, up to the backstop (10 s). Stopped
    there with no model frame for longer than the limit, it timed out on inactivity
    (``timeout``), not a harness incident."""
    record, elapsed, run = _run(runtime, "busy 60", tmp_path, backstop_s=10, tools=True)
    assert record["passed"] is False and record["error_kind"] == "timeout", record
    assert re.search(r"model completion timed out after \d+s of inactivity \(no model output; a "
                     r"tool command was still running at the 10s harness backstop\)",
                     record["error_text"]), record["error_text"]
    assert 10 <= elapsed < 45
    assert _sessions(_item(run)) == ["timed_out"]


def test_the_backstop_reports_the_calls_whole_time_after_a_transport_retry(tmp_path, monkeypatch):
    """The backstop's message says how long the call ran, its 6 s, not the 4 s a
    transport retry had left of it."""
    from refactory_dsh.model import HarnessBackstop

    calls = []

    def run_completion(self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):
        calls.append(budget_s)
        if len(calls) == 1:    # the wire fails after 2 s; the client asks again
            time.sleep(2)
            return subprocess.CompletedProcess(cmd, 1, "", "dsh: rate limit exceeded (429)")
        time.sleep(max(0.0, stop_at - time.monotonic()))
        expired = subprocess.TimeoutExpired(cmd, budget_s)
        expired.stream_quiet_s = 0.0    # still streaming
        raise expired

    monkeypatch.setenv("REFACTORY_MODEL_SCRATCH", str(tmp_path))
    monkeypatch.setattr(DshModelClient, "_run_completion", run_completion)
    client = DshModelClient(worker_backend="dsh", provider="p", model="m", attempts=2, backoff_s=0,
                            inactivity_s=5, backstop_s=6).for_node("solve")
    with pytest.raises(HarnessBackstop, match=r"hit the harness backstop after 6s"):
        client.complete("hello", workdir=tmp_path)
    assert len(calls) == 2 and calls[1] == 4


def test_the_backstop_is_no_interruption():
    """A session the backstop stopped did not die with the run's processes: its item is a
    ``backstop`` failure of the model's, not an interrupted one that would rerun the
    evaluation."""
    from refactory_dsh.items import interrupted_item

    for error in ("solve: HarnessBackstop: model session hit the harness backstop after 14400s",
                  "solve: model session hit the harness backstop after 14400s\nworker: model "
                  "session hit the harness backstop after 14400s"):
        assert interrupted_item({"error_text": error, "error_kind": "backstop"}) is None


def test_the_activity_file_is_the_receipts_neighbour(tmp_path):
    """The file the observer touches is the one the client reads (`activity_path`)."""
    receipt = tmp_path / ".refactory" / "workers" / "worker-abc.json"
    assert activity_path(receipt) == receipt.with_name("worker-abc.activity")


def test_the_observer_touches_the_activity_file_on_every_streamed_frame():
    """A thinking delta, a text delta and a native worker message each count as activity;
    the receipt alone is written only at a request's start and end."""
    subprocess.run(["node", str(Path(__file__).with_name("worker_activity_test.mjs"))], check=True)


def test_the_backstop_and_inactivity_travel_to_the_worker():
    """A dispatched worker rebuilds its client from the spec: both limits go with it."""
    client = DshModelClient(worker_backend="dsh", provider="p", model="m", inactivity_s=7,
                            backstop_s=99)
    rebuilt = DshModelClient(**client.context()).for_node("solve")
    assert rebuilt._inactivity_limit() == 7 and rebuilt.backstop_limit() == 99
    # Unset, the context is what it was, so a recorded worker spec keeps its bytes.
    assert "backstop_s" not in DshModelClient(worker_backend="dsh", provider="p", model="m").context()
    # Both activity backends; Codex reports none and keeps its per-completion wall.
    for backend, observed in (("dsh", True), ("claude_code", True), ("codex", False)):
        route = {"provider": "p"} if backend == "dsh" else {}
        node = DshModelClient(worker_backend=backend, model="m", **route).for_node("solve")
        assert node.observes_activity() is observed, backend
        assert (node._inactivity_limit() > 0) is observed, backend
    assert json.loads(json.dumps(client.context()))["backstop_s"] == 99


def test_the_backstop_is_read_from_anchored_signals_only():
    """The type, or a line starting with a runtime's own prefix, makes an item a harness
    incident; the phrase quoted mid-line (a model's own text) does not."""
    from refactory_dsh.items import error_kind
    from refactory_dsh.model import HarnessBackstop

    assert error_kind(HarnessBackstop("model session hit the harness backstop after 9s")) == "backstop"
    assert error_kind(RuntimeError("solve: HarnessBackstop: model session hit the harness backstop")) \
        == "backstop"
    # The spine's item error (`nodeErrorText`): the node's line, then the worker's stderr.
    assert error_kind(RuntimeError("solve: model session hit the harness backstop after 9s\nworker: "
                                   "model session hit the harness backstop after 9s")) == "backstop"
    assert error_kind(RuntimeError("worker: model session hit the harness backstop after 9s")) \
        == "backstop"
    # A harness fault, by its type or its walker/spine line, is a harness incident.
    from refactory_dsh.model import HarnessIncident
    # A harness incident's text names its class, known once its module is loaded
    # (`model.HARNESS_INCIDENT_CLASSES`): the package that raises it is.
    import refactory_packages.spreadsheetbench.recalc  # noqa: F401
    assert error_kind(HarnessIncident("grader down")) == "harness_incident"
    assert error_kind(RuntimeError("data: RecalcFailed: harness incident: grader down")) \
        == "harness_incident"
    assert error_kind(RuntimeError("parse: ValueError: x\ndata: RecalcFailed: harness incident: "
                                   "y")) == "RuntimeError"
    # The backstop is no harness incident, whatever its type's ancestry.
    assert not issubclass(HarnessBackstop, HarnessIncident)
    assert error_kind(RuntimeError("solve: ValueError: the answer says it hit the harness backstop")) \
        == "RuntimeError"
    assert error_kind(RuntimeError("solve: CompletionTimeout: model completion timed out after 5s of "
                                   "inactivity")) == "timeout"
    # A node's declared wall: the pipeline's own limit, by its type or its anchored line.
    from refactory_dsh.model import DeclaredTimeout
    said = "model session timed out at its node's declared timeout_s (60s) after 60s"
    assert error_kind(DeclaredTimeout(said)) == "declared_timeout"
    assert error_kind(RuntimeError(f"solve: DeclaredTimeout: {said}")) == "declared_timeout"
    assert error_kind(RuntimeError(f"solve: {said}\nworker: {said}")) == "declared_timeout"
    assert error_kind(RuntimeError(f"solve: ValueError: the log says {said}")) == "RuntimeError"


#: Each harness-written line `items.error_kind` classifies, as a model's reply could quote
#: it on a later line of another failure's text.
QUOTED_LINES = [
    "worker: model session hit the harness backstop after 1s",
    "solve: HarnessBackstop: model session hit the harness backstop after 1s",
    "data: SandboxFailed: harness incident: the model-code sandbox failed to start",
    "solve: model session hit the harness backstop after 1s",
    "worker: model completion timed out after 5s",
    "solve: CompletionTimeout: model completion timed out after 5s",
    "worker: model session timed out at its node's declared timeout_s (60s) after 60s",
    "solve: DeclaredTimeout: model session timed out at its node's declared timeout_s (60s)",
    "solve: RetriesExhausted: solve timed out twice. Last error: model session timed out at "
    "its node's declared timeout_s (60s)",
    "worker: solve timed out twice. Last error: model completion timed out after 5s",
]


@pytest.mark.parametrize("line", QUOTED_LINES)
def test_a_harness_line_only_counts_as_the_errors_first_line(line):
    """Only an error's first line is the harness's (`items.first_line`): the same line
    quoted from a model's reply further down another failure's text classifies nothing,
    while as the first line it does."""
    import refactory_packages.spreadsheetbench.sandbox  # noqa: F401  (SandboxFailed)
    from refactory_dsh.items import error_kind

    assert error_kind(RuntimeError(f"parse: ValueError: bad reply:\n{line}")) == "RuntimeError"
    assert error_kind(RuntimeError(line)) in ("harness_incident", "backstop", "timeout",
                                              "declared_timeout")
