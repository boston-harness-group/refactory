"""Large DSH evidence crosses real process I/O without a model or argv limits."""
from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from refactory_dsh.model import DshModelClient


def above_argmax():
    ceiling = os.sysconf("SC_ARG_MAX") if hasattr(os, "sysconf") else 2 * 1024 * 1024
    row = "  COMPLETE REPORT: α🙂 literal {{name}} and \\n escape\r\n"
    return "\n  " + row * (ceiling // len(row.encode()) + 2048) + "\nEND OF EXACT EVIDENCE  \n"


def runtime(tmp_path, *, slow=False):
    script = tmp_path / "controlled_dsh.py"
    script.write_text(f"#!{sys.executable}\n" + """import hashlib,json,signal,sys,time
from pathlib import Path
def stop(*args):
    Path('transport-stopped').write_text('owned runtime settled')
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
""" + ("""chunks=[]
while True:
    chunk=sys.stdin.buffer.read(4096)
    if not chunk:break
    chunks.append(chunk)
    Path('transport-ready').write_text('reading stdin')
    time.sleep(.01)
body=b''.join(chunks)
""" if slow else "body=sys.stdin.buffer.read() if sys.argv[-1]=='-' else sys.argv[-1].encode('utf-8')\n") + """
print(json.dumps({'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body),
                  'stdin':sys.argv[-1]=='-','largest_argument':max(len(x.encode()) for x in sys.argv)}))
""")
    script.chmod(0o700)
    return script


@pytest.mark.skipif(os.name != "posix", reason="real POSIX ARG_MAX and SIGTERM ownership")
def test_prompt_above_host_argmax_reaches_actual_runtime_byte_for_byte(tmp_path):
    prompt = above_argmax() + "Embedded data delimiter: \x00 exact tail\n"
    client = DshModelClient(dsh_cli=str(runtime(tmp_path)), attempts=1, timeout_s=10)
    result = json.loads(client.for_node("quality").with_node_timeout(10).complete(prompt, workdir=tmp_path))
    assert result["stdin"] is True
    assert result["bytes"] == len(prompt.encode())
    assert result["sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert result["largest_argument"] < 65536
    receipt = json.loads(next((tmp_path / ".refactory/workers").glob("*.json")).read_text())
    assert receipt["state"] == "completed"
    assert receipt["prompt_transport"] == {"kind": "stdin_file", "encoding": "utf-8",
                                          "bytes": len(prompt.encode()), "sha256": result["sha256"]}


@pytest.mark.skipif(os.name != "posix", reason="real POSIX process cancellation")
def test_cancel_while_large_stdin_is_read_settles_runtime_without_resending(tmp_path):
    stopped = threading.Event()
    client = DshModelClient(dsh_cli=str(runtime(tmp_path, slow=True)), attempts=3, timeout_s=10)
    client = client.for_node("quality").with_node_timeout(10).with_cancellation(stopped)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.complete, above_argmax(), workdir=tmp_path)
        try:
            deadline = time.monotonic() + 5
            while not (tmp_path / "transport-ready").exists():
                if future.done():
                    future.result()
                assert time.monotonic() < deadline, "controlled runtime never began stdin read"
                time.sleep(.01)
            stopped.set()
            with pytest.raises(CancelledError):
                future.result(timeout=5)
        finally:
            stopped.set()
    assert (tmp_path / "transport-stopped").read_text() == "owned runtime settled"
    receipts = list((tmp_path / ".refactory/workers").glob("*.json"))
    assert len(receipts) == 1  # cancellation is never replayed
    receipt = json.loads(receipts[0].read_text())
    assert receipt["state"] == "cancelled"
    assert len(receipt["effective_deadlines"]["dispatches"]) == 1


def test_transport_hash_binds_full_real_executor_delivery_context(tmp_path):
    from refactory_dsh.nn import Graph, Node
    from refactory_dsh.nn.executor import execute
    graph = Graph()
    graph.add(Node("quality", "agent", prompt_template=above_argmax(), writes=("response.json",), timeout_s=10))
    result = execute(graph, tmp_path, model=DshModelClient(dsh_cli=str(runtime(tmp_path)), attempts=1))
    assert result.exit_ok, result.error
    delivery = json.loads(next((tmp_path / ".refactory/agent-delivery").glob("*/receipt.json")).read_text())
    supplied = (tmp_path / delivery["contexts"][0]["prompt"]).read_bytes()
    response = json.loads((tmp_path / "response.json").read_text())
    worker = json.loads(next((tmp_path / ".refactory/workers").glob("*.json")).read_text())
    assert worker["prompt_transport"]["bytes"] == response["bytes"] == len(supplied)
    assert worker["prompt_transport"]["sha256"] == response["sha256"] == hashlib.sha256(supplied).hexdigest()


def test_small_dsh_task_keeps_positional_transport_and_content(tmp_path):
    prompt = "  short UTF-8 α🙂 task\r\n  "
    result = json.loads(DshModelClient(dsh_cli=str(runtime(tmp_path)), attempts=1).complete(prompt, workdir=tmp_path))
    assert result["stdin"] is False
    assert result["sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    worker = json.loads(next((tmp_path / ".refactory/workers").glob("*.json")).read_text())
    assert worker["prompt_transport"]["kind"] == "argv"


def test_large_prompt_transport_closes_anonymous_storage_when_spawn_fails(tmp_path, monkeypatch):
    import tempfile
    original = tempfile.TemporaryFile
    handles = []

    def observe_storage(*args, **kwargs):
        handle = original(*args, **kwargs)
        handles.append((handle, kwargs["dir"]))
        return handle

    monkeypatch.setattr(tempfile, "TemporaryFile", observe_storage)
    client = DshModelClient(dsh_cli=str(tmp_path / "missing-dsh"), attempts=1)
    with pytest.raises(FileNotFoundError):
        client.complete(above_argmax(), workdir=tmp_path)
    assert len(handles) == 1 and handles[0][0].closed and handles[0][1] == tmp_path
    assert not list(tmp_path.glob("tmp*"))


def test_large_prompt_retry_reads_fresh_exact_stdin_and_closes_each_storage(tmp_path, monkeypatch):
    import tempfile
    original = tempfile.TemporaryFile
    handles = []

    def observe_storage(*args, **kwargs):
        handle = original(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(tempfile, "TemporaryFile", observe_storage)
    script = runtime(tmp_path)
    body = script.read_text()
    body = body.replace("print(json.dumps(", """with Path('received-hashes.jsonl').open('a') as stream:
    stream.write(json.dumps({'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})+'\\n')
if not Path('first-transport-failed').exists():
    Path('first-transport-failed').write_text('retry one external transport error')
    print('TRANSPORT: connection reset',file=sys.stderr)
    raise SystemExit(1)
print(json.dumps(""")
    script.write_text(body)
    prompt = above_argmax()
    assert DshModelClient(dsh_cli=str(script), attempts=2, backoff_s=0).for_node("quality").with_node_timeout(10).complete(prompt, workdir=tmp_path)
    observed = [json.loads(line) for line in (tmp_path / "received-hashes.jsonl").read_text().splitlines()]
    assert observed == [{"bytes": len(prompt.encode()), "sha256": hashlib.sha256(prompt.encode()).hexdigest()}] * 2
    assert len(handles) == 2 and all(handle.closed for handle in handles)
    worker = json.loads(next((tmp_path / ".refactory/workers").glob("*.json")).read_text())
    assert [sample["attempt"] for sample in worker["effective_deadlines"]["dispatches"]] == [1, 2]


@pytest.mark.skipif(os.name != "posix", reason="real POSIX runtime shutdown on wall expiry")
def test_wall_expiry_during_large_stdin_read_stops_runtime_without_retry(tmp_path):
    client = DshModelClient(dsh_cli=str(runtime(tmp_path, slow=True)), attempts=3, timeout_s=1)
    client = client.for_node("quality").with_node_timeout(1)
    with pytest.raises(RuntimeError, match="timed out.*not retried"):
        client.complete(above_argmax(), workdir=tmp_path)
    assert (tmp_path / "transport-stopped").exists()
    records = list((tmp_path / ".refactory/workers").glob("*.json"))
    assert len(records) == 1
    worker = json.loads(records[0].read_text())
    assert worker["state"] == "timed_out" and worker["prompt_transport"]["kind"] == "stdin_file"
    assert len(worker["effective_deadlines"]["dispatches"]) == 1


@pytest.mark.parametrize("limit", ["cancellation", "stage", "budget"])
def test_limit_reached_during_stdin_preparation_refuses_spawn_and_closes_storage(tmp_path, monkeypatch, limit):
    import tempfile
    import subprocess
    from refactory_dsh.budgets import BudgetExhausted, ResourceLimits, RunBudget

    actual_file = tempfile.TemporaryFile
    actual_popen = subprocess.Popen
    files, launched = [], []
    stopped = threading.Event()

    class PreparationWindow:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def write(self, body):
            count = self.handle.write(body)
            if limit == "cancellation":
                stopped.set()
            else:
                time.sleep(1.05)
            return count

    def prepared_file(*args, **kwargs):
        handle = actual_file(*args, **kwargs)
        files.append(handle)
        return PreparationWindow(handle)

    def observe_launch(*args, **kwargs):
        launched.append(args[0])
        return actual_popen(*args, **kwargs)

    monkeypatch.setattr(tempfile, "TemporaryFile", prepared_file)
    monkeypatch.setattr(subprocess, "Popen", observe_launch)
    client = DshModelClient(dsh_cli=str(runtime(tmp_path)), attempts=3).for_node("quality").with_node_timeout(1)
    if limit == "cancellation":
        client = client.with_cancellation(stopped)
        error, message = CancelledError, "cancelled; not retried"
    elif limit == "stage":
        error, message = RuntimeError, "timed out.*not retried"
    else:
        client = client.with_node_timeout(10).with_budget(RunBudget.create(tmp_path, ResourceLimits(max_elapsed_s=1)))
        error, message = BudgetExhausted, "elapsed-time budget exhausted"
    with pytest.raises(error, match=message):
        client.complete(above_argmax(), workdir=tmp_path)
    assert not launched, "preparation exhausted an existing limit but a runtime was still launched"
    assert len(files) == 1 and files[0].closed
    assert not (tmp_path / "transport-ready").exists()
    receipts = list((tmp_path / ".refactory/workers").glob("*.json"))
    assert len(receipts) == 1
    worker = json.loads(receipts[0].read_text())
    assert worker["state"] == {"cancellation": "cancelled", "stage": "timed_out", "budget": "budget_exhausted"}[limit]
    assert worker["actual_route"] is None
    assert len(worker["effective_deadlines"]["dispatches"]) == 1
