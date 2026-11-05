"""Parallel data node execution on the walker, and the spine's item loader verbs.

T1-T9 from the design, with the review's fixes: the step cap, the deadline span,
the stop semantics and the fallback reasons are the sequential loop's, and
``--knob data_parallelism=N`` reaches the graph ``Loop.run()`` evaluates.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import lowering, nn
from refactory_dsh.nn.executor import CURRENT_ITEM, execute
from refactory_dsh.task import Task, VerifyResult


# ── helpers ───────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent


def _data_graph(
    body_call: str,
    body_fn: str,
    *,
    items: tuple[dict[str, Any], ...],
    parallelism: int = 1,
    produces: tuple[str, ...] = ("item_scores",),
    body_reads: tuple[str, ...] = (CURRENT_ITEM,),
    body_produces: tuple[str, ...] = ("counted",),
) -> nn.Graph:
    """data(parallelism) -> [body_op] -> report."""
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data",
        subgraph_entry="body",
        subgraph_exit="body",
        writes=(CURRENT_ITEM,),
        produces=produces,
        inline_items=tuple(dict(item) for item in items),
        parallelism=parallelism,
    ))
    graph.add(nn.Node(
        "body", "op", call=f"{body_call}:{body_fn}",
        reads=body_reads, produces=body_produces,
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")
    return graph


# ── test ops (in test_ops_parallel module) ────────────────────────────────

_TEST_OPS = '''\
"""Test ops for data_parallel tests. Run inside forkserver child processes."""
import json
import os
import time
from pathlib import Path


def barrier_op(ctx):
    """Each item signals ready, then waits until K items are ready."""
    item = ctx.read_json("current_item.json")
    item_id = str(item.get("id", "?"))
    barrier_dir = Path(os.environ["BARRIER_DIR"])
    k = int(os.environ.get("BARRIER_K", "3"))

    # Signal ready.
    (barrier_dir / f"ready_{item_id}").write_text(str(time.monotonic()))

    # Wait for all K items to be ready.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        ready = list(barrier_dir.glob("ready_*"))
        if len(ready) >= k:
            break
        time.sleep(0.02)
    else:
        raise TimeoutError(f"barrier timeout: only {len(list(barrier_dir.glob('ready_*')))} of {k} ready")

    (barrier_dir / f"done_{item_id}").write_text(str(time.monotonic()))
    return {"counted": item_id}


def sleep_op(ctx):
    """Sleep for `delay` seconds (from the item)."""
    item = ctx.read_json("current_item.json")
    delay = float(item.get("delay", 0))
    barrier_dir_str = os.environ.get("BARRIER_DIR", "")
    item_id = str(item.get("id", "?"))

    if barrier_dir_str:
        barrier_dir = Path(barrier_dir_str)
        barrier_dir.mkdir(parents=True, exist_ok=True)
        (barrier_dir / f"start_{item_id}").write_text(str(time.time()))

    time.sleep(delay)

    if barrier_dir_str:
        (barrier_dir / f"end_{item_id}").write_text(str(time.time()))

    return {"counted": item_id}


def noop_op(ctx):
    """No-op: just record that this item was visited."""
    item = ctx.read_json("current_item.json")
    item_id = str(item.get("id", "?"))
    visited_dir_str = os.environ.get("VISITED_DIR", "")
    if visited_dir_str:
        visited_dir = Path(visited_dir_str)
        visited_dir.mkdir(parents=True, exist_ok=True)
        (visited_dir / item_id).write_text("1")
    return {"counted": item_id}


def long_sleep_op(ctx):
    """Sleep for a very long time (for deadline testing)."""
    item = ctx.read_json("current_item.json")
    delay = float(item.get("delay", 0))
    time.sleep(delay)
    return {"counted": item.get("id")}


def maybe_fail_op(ctx):
    """Fail if the item's 'fail' field is truthy; succeed otherwise."""
    item = ctx.read_json("current_item.json")
    item_id = str(item.get("id", "?"))
    if item.get("fail"):
        raise RuntimeError(f"item {item_id} forced to fail")
    visited_dir_str = os.environ.get("VISITED_DIR", "")
    if visited_dir_str:
        visited_dir = Path(visited_dir_str)
        visited_dir.mkdir(parents=True, exist_ok=True)
        (visited_dir / item_id).write_text("1")
    return {"counted": item_id}
'''


@pytest.fixture(autouse=True)
def _install_test_ops(tmp_path):
    """Install test ops module so forkserver children can import it."""
    ops_dir = REPO_ROOT / "refactory" / "tests"
    ops_path = ops_dir / "test_ops_parallel.py"
    ops_path.write_text(_TEST_OPS)
    yield
    ops_path.unlink(missing_ok=True)


# ── T1: walker overlap ────────────────────────────────────────────────────

def test_walker_overlap(tmp_path, monkeypatch):
    """T1 — items run concurrently with parallelism > 1.

    A file-based barrier in the op releases only when K items are in the stub
    simultaneously. With parallelism=K this succeeds; with parallelism=1
    it would deadlock (the barrier never reaches K).
    """
    K = 3
    items = tuple({"id": str(i)} for i in range(K))
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()

    monkeypatch.setenv("BARRIER_DIR", str(barrier_dir))
    monkeypatch.setenv("BARRIER_K", str(K))

    graph = _data_graph(
        "test_ops_parallel", "barrier_op",
        items=items, parallelism=K,
    )

    result = execute(graph, tmp_path / "run", max_steps=200)
    assert result.exit_ok, result.error

    # Verify all K items reached the barrier simultaneously.
    ready_files = list(barrier_dir.glob("ready_*"))
    done_files = list(barrier_dir.glob("done_*"))
    assert len(ready_files) == K, f"expected {K} ready files, got {len(ready_files)}"
    assert len(done_files) == K, f"expected {K} done files, got {len(done_files)}"

    # All items started before any finished.
    ready_times = [float(f.read_text()) for f in ready_files]
    done_times = [float(f.read_text()) for f in done_files]
    assert max(ready_times) <= min(done_times), "items did not overlap in time"


# ── T3: item order preserved with reverse finishing ───────────────────────

def test_item_order_reverse_finishing(tmp_path, monkeypatch):
    """T3 — item_records are in item order after concurrent execution.

    Items run concurrently (T1 proves overlap), and the merge step produces
    results in the original item index order. Without the merge, a concurrent
    executor would return results in arbitrary order.
    """
    K = 3
    items = tuple({"id": str(i)} for i in range(K))
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    monkeypatch.setenv("BARRIER_DIR", str(barrier_dir))
    monkeypatch.setenv("BARRIER_K", str(K))

    # Use a scored task so item_scores is populated.
    _SCORED = tmp_path / "scored_task.py"
    _SCORED.write_text(f"""\
from refactory_dsh.task import Task, TaskInstance, VerifyResult

class T(Task):
    name = "order_test"
    def instances(self):
        return [TaskInstance(id=str(i), metadata={{}}) for i in range({K})]
    def setup(self, instance, workspace, **kw):
        pass
    def verify(self, instance, workspace, **kw):
        return VerifyResult(passed=True, score=float(instance.id) / 10.0)
""")
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    from refactory_dsh.task import register_task
    import importlib
    if "scored_task" in sys.modules:
        importlib.reload(sys.modules["scored_task"])
    else:
        import scored_task  # noqa: F401
    from scored_task import T  # type: ignore[import-not-found]
    register_task("order_test", T)

    # Use barrier_op to force all items to overlap, proving concurrency.
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data",
        subgraph_entry="body",
        subgraph_exit="body",
        writes=(CURRENT_ITEM,),
        produces=("item_scores",),
        task_ref="order_test",
        parallelism=K,
    ))
    graph.add(nn.Node(
        "body", "op", call="test_ops_parallel:barrier_op",
        reads=(CURRENT_ITEM,), produces=("counted",),
    ))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    result = execute(graph, tmp_path / "run", max_steps=200)
    assert result.exit_ok, result.error
    scores = result.facts.get("item_scores", [])
    assert len(scores) == K, f"expected {K} scores, got {len(scores)}: {scores}"
    # The key assertion: scores are in item INDEX order (0, 1, 2)
    # regardless of which thread completed first.
    assert [s["id"] for s in scores] == [str(i) for i in range(K)]
    # Scores are the per-item values from the task.
    assert [s["score"] for s in scores] == [i / 10.0 for i in range(K)]

    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))
    sys.modules.pop("scored_task", None)


# ── T4: resume reruns only unfinished items ───────────────────────────────

def test_resume_reruns_only_unfinished(tmp_path, monkeypatch):
    """T4 — after an interrupt, only items without .done are re-run."""
    N = 4
    items = tuple({"id": str(i)} for i in range(N))

    # First run: sequential (parallelism=1), interrupt after 2 items.
    # Use a noop op with a visited-dir to track calls.
    visited_dir = tmp_path / "visited1"
    visited_dir.mkdir()
    monkeypatch.setenv("VISITED_DIR", str(visited_dir))

    graph1 = _data_graph(
        "test_ops_parallel", "noop_op",
        items=items, parallelism=1,
    )
    workdir = tmp_path / "run"
    result1 = execute(graph1, workdir, max_steps=200)
    assert result1.exit_ok

    # All 4 items ran.
    assert len(list(visited_dir.glob("*"))) == N

    # Simulate partial completion by removing .done for items 2 and 3.
    passes_root = workdir / "passes" / "data"
    for d in passes_root.iterdir():
        if d.is_dir():
            # Keep items 0 and 1 done, remove done for 2 and 3.
            item_name = d.name
            # Parse index from dir name (format: <id>_<index>).
            parts = item_name.rsplit("_", 1)
            idx = int(parts[-1]) if len(parts) > 1 else -1
            if idx >= 2:
                (d / ".done").unlink(missing_ok=True)

    # Second run: resume with parallelism > 1.
    visited_dir2 = tmp_path / "visited2"
    visited_dir2.mkdir()
    monkeypatch.setenv("VISITED_DIR", str(visited_dir2))

    graph2 = _data_graph(
        "test_ops_parallel", "noop_op",
        items=items, parallelism=N,
    )
    result2 = execute(graph2, workdir, max_steps=200, start="data")
    assert result2.exit_ok, result2.error

    # Only the unfinished items (2, 3) should have been re-run.
    visited = sorted(f.name for f in visited_dir2.glob("*"))
    assert visited == ["2", "3"], f"expected ['2', '3'], got {visited}"


# ── T5: the step cap is the sequential loop's, scaled by the item count ───

@pytest.mark.parametrize("parallelism", [1, 4])
def test_step_cap_scaled_like_sequential(tmp_path, monkeypatch, parallelism):
    """T5 — with 20 items under max_steps=10, every item runs, at any parallelism.

    The sequential loop scales the cap by the item count; the concurrent items
    reserve their steps against that same scaled cap, not the unscaled one (which
    left 9 steps for 20 items: 11 items failed "the run exceeded ... steps").
    """
    N = 20
    items = tuple({"id": str(i)} for i in range(N))
    visited_dir = tmp_path / "visited"
    visited_dir.mkdir()
    monkeypatch.setenv("VISITED_DIR", str(visited_dir))

    graph = _data_graph("test_ops_parallel", "noop_op", items=items, parallelism=parallelism)
    result = execute(graph, tmp_path / "run", max_steps=10)

    assert result.exit_ok, result.error
    assert len(list(visited_dir.glob("*"))) == N
    # No Task grades these items, so only a failed item would leave a record.
    assert result.item_records == []


@pytest.mark.parametrize("parallelism", [1, 3])
def test_nested_fork_in_item_shares_the_step_cap(tmp_path, monkeypatch, parallelism):
    """A fork inside a concurrent item charges its branches' steps to the cap the
    items share, so exactly the items the sequential loop fails fail here too.

    max_steps=3 over 3 items of a 5-step body (fork, a, b, join, last) scales to 15:
    the data step plus 14 body steps, one short of the 15 the items need.
    """
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="fork1", subgraph_exit="last",
        writes=(CURRENT_ITEM,), produces=("item_scores",),
        inline_items=tuple({"id": str(i)} for i in range(3)),
        parallelism=parallelism,
    ))
    graph.add(nn.Node("fork1", "fork", targets=("a", "b")))
    graph.add(nn.Node("a", "op", call="loop_ops:noop"))
    graph.add(nn.Node("b", "op", call="loop_ops:noop"))
    graph.add(nn.Node("join1", "join", sources=("a", "b")))
    graph.add(nn.Node("last", "op", call="loop_ops:noop"))
    graph.connect("fork1", "join1")
    graph.connect("a", "join1")
    graph.connect("b", "join1")
    graph.connect("join1", "last")
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("data", "report")

    result = execute(graph, tmp_path / "run", max_steps=3)
    # No Task grades these items: the records are the failed items'.
    assert len(result.item_records) == 1, result.item_records
    assert "exceeded" in result.item_records[0]["error_text"]


# ── T6: candidate ID stability ───────────────────────────────────────────

def test_candidate_id_stability():
    """T6 — to_state() omits parallelism when default (1), includes when > 1."""
    node_default = nn.Node(
        "data", "data",
        subgraph_entry="a", subgraph_exit="a",
        writes=("current_item.json",),
    )
    state = node_default.to_state()
    assert "parallelism" not in state, "default parallelism=1 should not appear in state"

    node_parallel = nn.Node(
        "data2", "data",
        subgraph_entry="a", subgraph_exit="a",
        writes=("current_item.json",),
        parallelism=4,
    )
    state2 = node_parallel.to_state()
    assert state2.get("parallelism") == 4


# ── T7: the per-item deadline covers setup and verify, as the sequential loop's

class _PhaseTask(Task):
    """Sleeps in the phase an item names, past the one-second deadline."""

    name = "phase_deadline"

    def setup(self, instance, workspace, **kw):
        if instance.metadata.get("phase") == "setup":
            time.sleep(4)

    def verify(self, instance, workspace, **kw):
        if instance.metadata.get("phase") == "verify":
            time.sleep(4)
        return VerifyResult(passed=True, score=1.0)


@pytest.mark.parametrize("parallelism", [1, 4])
def test_per_item_deadline_covers_setup_and_verify(tmp_path, monkeypatch, parallelism):
    """T7 — an item whose setup or verify outlasts the deadline is a timeout; an item
    whose walk does is not (the walk has no item deadline, on either path). The
    other items are not cancelled, and both paths record the same outcomes.
    """
    from refactory_dsh.nn.executor import Executor
    from refactory_dsh.task import register_task

    register_task("phase_deadline", _PhaseTask)
    monkeypatch.setattr(Executor, "_data_item_deadline", lambda self: 1.0)
    items = (
        {"id": "ok", "delay": "0"},
        {"id": "slow_setup", "delay": "0", "phase": "setup"},
        {"id": "slow_verify", "delay": "0", "phase": "verify"},
        {"id": "slow_walk", "delay": "2"},
    )
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="body", subgraph_exit="body",
        writes=(CURRENT_ITEM,), produces=("item_scores",), task_ref="phase_deadline",
        inline_items=items, parallelism=parallelism,
    ))
    graph.add(nn.Node("body", "op", call="test_ops_parallel:sleep_op",
                      reads=(CURRENT_ITEM,), produces=("counted",)))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    result = execute(graph, tmp_path / "run", max_steps=200)
    assert result.exit_ok, result.error
    outcomes = [(r["id"], r.get("error_kind"), r.get("passed")) for r in result.facts["item_scores"]]
    assert outcomes == [
        ("ok", None, True),
        ("slow_setup", "timeout", False),
        ("slow_verify", "timeout", False),
        ("slow_walk", None, True),
    ]


def test_per_item_deadline_holds_on_a_pool_thread(tmp_path, monkeypatch):
    """Run.evaluate_many runs evaluations on pool threads. The concurrent
    path's deadline is thread-based, so a hung item there is still a timeout
    (only the sequential loop's SIGALRM needs the main thread). Fails when the
    executor disarms the deadline off the main thread."""
    from concurrent.futures import ThreadPoolExecutor

    from refactory_dsh.nn.executor import Executor
    from refactory_dsh.task import register_task

    register_task("phase_deadline", _PhaseTask)
    monkeypatch.setattr(Executor, "_data_item_deadline", lambda self: 1.0)
    items = (
        {"id": "ok", "delay": "0"},
        {"id": "hung_verify", "delay": "0", "phase": "verify"},
    )
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="body", subgraph_exit="body",
        writes=(CURRENT_ITEM,), produces=("item_scores",), task_ref="phase_deadline",
        inline_items=items, parallelism=4,
    ))
    graph.add(nn.Node("body", "op", call="test_ops_parallel:sleep_op",
                      reads=(CURRENT_ITEM,), produces=("counted",)))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(execute, graph, tmp_path / "run", max_steps=200).result()
    assert result.exit_ok, result.error
    outcomes = [(r["id"], r.get("error_kind"), r.get("passed")) for r in result.facts["item_scores"]]
    assert outcomes == [("ok", None, True), ("hung_verify", "timeout", False)]


# ── T8: a serial fallback says exactly why, and is never silent ───────────

class _UnsharedModel:
    """A client that does not declare itself safe to share."""

    def complete(self, prompt: str, *, system: str = "", **kw: Any) -> str:
        return "ok"


def _noop_graph(tmp_path, monkeypatch, parallelism=4):
    visited_dir = tmp_path / "visited"
    visited_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("VISITED_DIR", str(visited_dir))
    return _data_graph("test_ops_parallel", "noop_op",
                       items=tuple({"id": str(i)} for i in range(3)), parallelism=parallelism)


def test_serial_fallback_records_reason(tmp_path, monkeypatch):
    """T8 — an unshareable client runs the items one at a time; the fact and the
    run warning say which client and why."""
    result = execute(_noop_graph(tmp_path, monkeypatch), tmp_path / "run",
                     model=_UnsharedModel(), max_steps=200)
    assert result.exit_ok, result.error
    fact = result.facts["data_parallelism_effective"]
    assert fact == {
        "node": "data", "requested": 4, "effective": 1,
        "reason": "model client _UnsharedModel does not declare thread_safe = True, "
                  "so it is not shared across concurrent items",
    }
    events = (tmp_path / "run" / ".refactory" / "events.jsonl").read_text()
    assert "data parallelism fallback to serial: model client _UnsharedModel" in events


def test_serial_fallback_names_a_memory_store_with_namespaces(tmp_path, monkeypatch):
    """A MemoryStore is named as one, with the namespaces that keep it serial."""
    from refactory_dsh.memory import MemoryStore

    store = MemoryStore((nn.Memory("notes"),), run_root=tmp_path / "mem",
                        project_root=tmp_path / "project")
    result = execute(_noop_graph(tmp_path, monkeypatch), tmp_path / "run",
                     memory=store, max_steps=200)
    reason = result.facts["data_parallelism_effective"]["reason"]
    assert "non-MemoryStore" not in reason
    assert reason == ("the MemoryStore declares namespace(s) notes, whose reads and "
                      "writes are not synchronised across concurrent items")


def test_serial_fallback_warning_that_cannot_be_written_is_reported(tmp_path, monkeypatch, capsys):
    """A run warning that fails to write is reported on stderr and in the fact."""
    from refactory_dsh import live

    def broken(*args: Any, **kwargs: Any) -> None:
        raise OSError("events file is read-only")

    monkeypatch.setattr(live, "emit", broken)
    result = execute(_noop_graph(tmp_path, monkeypatch), tmp_path / "run",
                     model=_UnsharedModel(), max_steps=200)
    assert result.facts["data_parallelism_effective"]["warning_error"] == "OSError: events file is read-only"
    assert '"warning": "data_parallelism_fallback"' in capsys.readouterr().err


def test_shareable_custom_client_runs_items_concurrently(tmp_path, monkeypatch):
    """A client that is not a DshModelClient but declares thread_safe = True is
    shared: no fallback, and the items overlap (the barrier needs all three)."""
    class _Shared(_UnsharedModel):
        thread_safe = True

    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    monkeypatch.setenv("BARRIER_DIR", str(barrier_dir))
    monkeypatch.setenv("BARRIER_K", "3")
    graph = _data_graph("test_ops_parallel", "barrier_op",
                        items=tuple({"id": str(i)} for i in range(3)), parallelism=3)
    result = execute(graph, tmp_path / "run", model=_Shared(), max_steps=200)
    assert result.exit_ok, result.error
    assert "data_parallelism_effective" not in result.facts


# ── T9: `--knob data_parallelism=N` reaches the evaluated graph ──────────

_LOOP_ITEMS = 8


class _SleepyClient:
    """Answers each solve after a pause, the same answer for the same prompt, and
    counts how many calls are in flight at once (shared across its copies)."""

    thread_safe = True

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.lock = threading.Lock()
        self.count = {"now": 0, "max": 0, "calls": 0}

    def complete(self, prompt: str, *, system: str = "", **kw: Any) -> str:
        import hashlib

        with self.lock:
            self.count["now"] += 1
            self.count["calls"] += 1
            self.count["max"] = max(self.count["max"], self.count["now"])
        time.sleep(self.delay)
        with self.lock:
            self.count["now"] -= 1
        return f"<answer>{'ABCD'[int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 4]}</answer>"

    def for_node(self, node_id, role=None):  # noqa: ANN001
        return self

    def with_tool_policy(self, policy):  # noqa: ANN001
        return self

    def with_wall(self, seconds):  # noqa: ANN001
        return self

    def route(self) -> dict[str, Any]:
        return {"worker_backend": "dsh"}


def _livemath_items(root: Path) -> Path:
    data = root / "data" / "202601"
    data.mkdir(parents=True)
    rows = [{
        "month": "202601", "no": no, "paper_link": f"http://arxiv.org/abs/2601.{no:05d}v1",
        "theorem": f"Theorem {no}.", "sketch": f"Sketch {no}.", "theorem_type": ["analysis"],
        "mcq": {"question": f"Question number {no}: which holds?",
                "choices": [{"label": label, "text": f"choice {label} of {no}"} for label in "ABCD"],
                "correct_choice": {"label": "ABCD"[no % 4], "text": f"choice {'ABCD'[no % 4]} of {no}"}},
    } for no in range(1, _LOOP_ITEMS + 1)]
    (data / "qa_202601_final.json").write_text(json.dumps(rows))
    return root / "data"


def _optimize_livemath(root: Path, parallelism: int) -> tuple[_SleepyClient, Path]:
    from refactory_dsh.loop import Loop, LoopConfig

    client = _SleepyClient(0.3)
    config = LoopConfig(knobs={"data_parallelism": parallelism}, epochs=1, batch_size=1,
                        max_evaluations=1, optimizer="random", seed=1,
                        runs_root=root / "run", holdout=0.0)
    Loop("livemath/solver", config, project=Path("."), model=client).run()
    (trial,) = sorted((root / "run" / "runs").iterdir())
    return client, trial


def test_loop_knob_reaches_the_evaluated_graph(tmp_path, monkeypatch):
    """T9 — through ``Loop.run()``, as ``refactory optimize --knob data_parallelism=N``
    runs it: knob 4 runs LiveMath's items four at a time, knob 1 one at a time, and
    both score every item the same."""
    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(_livemath_items(tmp_path)))
    serial, serial_trial = _optimize_livemath(tmp_path / "p1", 1)
    parallel, parallel_trial = _optimize_livemath(tmp_path / "p4", 4)

    assert serial.count["calls"] == parallel.count["calls"] == _LOOP_ITEMS
    assert serial.count["max"] == 1
    assert parallel.count["max"] == 4
    for name in ("item_records.json", "facts.json"):
        assert json.loads((serial_trial / name).read_text()) == json.loads((parallel_trial / name).read_text())


# ── M5: halts, gate stops and budget stops keep the sequential semantics ──

class _StopTask(Task):
    """Stops the run in item "stop"'s setup, the way the test's ``kind`` says."""

    name = "stop_task"
    kind = "halt"

    def setup(self, instance, workspace, **kw):
        from refactory_dsh.budgets import BudgetExhausted
        from refactory_dsh.nn.executor import GateStop, HaltStop

        if instance.id == "a":
            # Still in setup long after "stop" stopped the node, so with two workers
            # the only worker free to take "c" is the one "stop" freed, after the stop.
            time.sleep(1.0)
        if instance.id != "stop":
            return
        if self.kind == "halt":
            raise HaltStop("item_gate", reason="halted in an item")
        if self.kind == "gate":
            raise GateStop("item_gate", [], [])
        raise BudgetExhausted("budget gone in an item")

    def verify(self, instance, workspace, **kw):
        return VerifyResult(passed=True, score=1.0)


def _stop_graph(parallelism: int) -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="body", subgraph_exit="body",
        writes=(CURRENT_ITEM,), produces=("item_scores",), task_ref="stop_task",
        inline_items=({"id": "a"}, {"id": "stop"}, {"id": "c"}), parallelism=parallelism,
    ))
    graph.add(nn.Node("body", "op", call="loop_ops:noop"))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("data", "report")
    return graph


@pytest.mark.parametrize("parallelism", [1, 2])
@pytest.mark.parametrize("kind,outcome", [("halt", "halted"), ("gate", "refused")])
def test_item_stop_is_recorded_as_the_sequential_loop_records_it(
        tmp_path, monkeypatch, parallelism, kind, outcome):
    """A halt pauses nothing and ends the run halted; a gate stop ends it refused --
    both recorded in the progress record. The stopped item and the items after it
    are not finished, so a resume runs them."""
    from refactory_dsh.task import register_task

    monkeypatch.setattr(_StopTask, "kind", kind)
    register_task("stop_task", _StopTask)
    run = tmp_path / "run"
    result = execute(_stop_graph(parallelism), run, max_steps=200)

    assert result.exit_ok is False
    assert result.halted is (kind == "halt")
    progress = json.loads((run / ".refactory" / "progress.json").read_text())
    assert progress["outcome"] == outcome
    assert progress["node"] == "data"
    passes = run / "passes" / "data"
    assert not (passes / "stop_1" / ".done").exists()
    assert not (passes / "c_2").exists()


@pytest.mark.parametrize("parallelism", [1, 2])
def test_budget_stop_in_an_item_propagates(tmp_path, monkeypatch, parallelism):
    """``BudgetExhausted`` in an item ends the run with that exception, not an error string."""
    from refactory_dsh.budgets import BudgetExhausted
    from refactory_dsh.task import register_task

    monkeypatch.setattr(_StopTask, "kind", "budget")
    register_task("stop_task", _StopTask)
    with pytest.raises(BudgetExhausted, match="budget gone in an item"):
        execute(_stop_graph(parallelism), tmp_path / "run", max_steps=200)


# ── the spine's item loader: prepare, begin/end --concurrent, finish ──────

class _LoaderTask(Task):
    name = "loader_task"

    def verify(self, instance, workspace, **kw):
        return VerifyResult(passed=instance.id != "fail", score=1.0)


def _loader_run(tmp_path: Path, ids: tuple[str, ...],
                produces: tuple[str, ...] = ("item_scores",)) -> tuple[Path, Any]:
    from refactory_dsh.items import data_node
    from refactory_dsh.lowering import write_declarations
    from refactory_dsh.task import register_task

    register_task("loader_task", _LoaderTask)
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="body", subgraph_exit="body",
        writes=(CURRENT_ITEM,), produces=produces, task_ref="loader_task",
        inline_items=tuple({"id": item_id} for item_id in ids), parallelism=4,
    ))
    graph.add(nn.Node("body", "op", call="loop_ops:noop"))
    run = tmp_path / "run"
    write_declarations({"graph": graph.to_state()}, run)
    return run, data_node(run, "data")


def _scores(run: Path) -> list[str]:
    return [entry["id"] for entry in json.loads((run / "facts.json").read_text())["item_scores"]]


def test_concurrent_begin_never_resets_the_node(tmp_path):
    """Item 0's ``begin`` beside the others leaves their passes and scores alone:
    ``prepare`` reset the node, once, before any item began."""
    from refactory_dsh.items import command_begin, command_end, command_prepare

    run, node = _loader_run(tmp_path, ("x", "y", "z"))
    command_prepare(node, run, requested=4)
    command_begin(node, run, 1, concurrent=True)
    command_end(node, run, 1, concurrent=True)
    command_begin(node, run, 0, concurrent=True)
    assert (run / "passes" / "data" / "y_1" / ".done").exists()
    assert _scores(run) == ["y"]


def test_finish_puts_item_scores_in_item_order(tmp_path):
    """Items that end in reverse order are scored in item order after ``finish``."""
    from refactory_dsh.items import command_begin, command_end, command_finish, command_prepare

    run, node = _loader_run(tmp_path, ("x", "y", "z"))
    command_prepare(node, run, requested=4)
    for index in (2, 1, 0):
        command_begin(node, run, index, concurrent=True)
        command_end(node, run, index, concurrent=True)
    assert _scores(run) == ["z", "y", "x"]
    finished = command_finish(node, run)
    assert _scores(run) == ["x", "y", "z"]
    assert [record["id"] for record in finished["records"]] == ["x", "y", "z"]


def test_finish_applies_the_all_failed_rule_once(tmp_path):
    """The last index ending first does not fire the all-failed rule early; ``finish``
    fires it when every item failed."""
    from refactory_dsh.items import (
        AllItemsFailedError, command_begin, command_end, command_finish, command_prepare,
    )

    run, node = _loader_run(tmp_path, ("fail", "fail2"))
    command_prepare(node, run, requested=4)
    command_begin(node, run, 1, concurrent=True)
    command_end(node, run, 1, concurrent=True, failed_reason="boom")
    command_begin(node, run, 0, concurrent=True)
    command_end(node, run, 0, concurrent=True, failed_reason="boom")
    with pytest.raises(AllItemsFailedError):
        command_finish(node, run)


def test_concurrent_begin_keeps_a_finished_pass(tmp_path):
    """A resumed concurrent pass does not run a finished item again."""
    from refactory_dsh.items import command_begin, command_end, command_prepare

    run, node = _loader_run(tmp_path, ("x", "y"))
    command_prepare(node, run, requested=4)
    command_begin(node, run, 0, concurrent=True)
    kept = command_end(node, run, 0, concurrent=True)
    assert command_begin(node, run, 0, concurrent=True) == {"pass_finished": True, "record": kept}


def test_prepare_records_a_fallback_without_resetting(tmp_path):
    """``prepare --fallback`` records the walker's fact and leaves the reset to item 0."""
    from refactory_dsh.items import command_prepare

    run, node = _loader_run(tmp_path, ("x",))
    (run / "passes" / "data" / "kept").mkdir(parents=True)
    assert command_prepare(node, run, requested=4, fallback="why") == {"effective": 1, "reason": "why"}
    facts = json.loads((run / "facts.json").read_text())
    assert facts["data_parallelism_effective"] == {"node": "data", "requested": 4, "effective": 1, "reason": "why"}
    assert (run / "passes" / "data" / "kept").exists()


# ── partial item failure does not make the run fail ───────────────────────

def test_partial_item_failure_exit_ok(tmp_path, monkeypatch):
    """A single failed item does NOT set exit_ok=False when other items succeed.

    Without the _data_steps_start offset for _failed_item_step_indices, the
    post-loop error scan (executor.py ~:1329) would promote the failed item's
    step error to a run-level error, making exit_ok=False even though not all
    items failed.
    """
    _TASK = tmp_path / "partial_fail_task.py"
    _TASK.write_text("""\
from refactory_dsh.task import Task, TaskInstance, VerifyResult

class T(Task):
    name = "partial_fail_test"
    def instances(self):
        return [TaskInstance(id=str(i), metadata={}) for i in range(3)]
    def setup(self, instance, workspace, **kw):
        pass
    def verify(self, instance, workspace, **kw):
        return VerifyResult(passed=True, score=1.0)
""")
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    from refactory_dsh.task import register_task
    import importlib
    if "partial_fail_task" in sys.modules:
        importlib.reload(sys.modules["partial_fail_task"])
    else:
        import partial_fail_task  # noqa: F401
    from partial_fail_task import T  # type: ignore[import-not-found]
    register_task("partial_fail_test", T)

    items = (
        {"id": "0"},
        {"id": "1"},
        {"id": "2", "fail": True},  # this one fails
    )
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data",
        subgraph_entry="body",
        subgraph_exit="body",
        writes=(CURRENT_ITEM,),
        produces=("item_scores",),
        task_ref="partial_fail_test",
        inline_items=tuple(dict(item) for item in items),
        parallelism=3,
    ))
    graph.add(nn.Node(
        "body", "op", call="test_ops_parallel:maybe_fail_op",
        reads=(CURRENT_ITEM,), produces=("counted",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")

    result = execute(graph, tmp_path / "run", max_steps=200)
    # Not all items failed (only 1 of 3), so the run should succeed.
    assert result.exit_ok, f"expected exit_ok=True, got error: {result.error}"

    scores = result.facts.get("item_scores", [])
    assert len(scores) == 3
    # Item 2 should have an error.
    assert scores[2].get("error_text") is not None

    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))
    sys.modules.pop("partial_fail_task", None)


# ── lowering preserves parallelism ────────────────────────────────────────

def test_lowering_preserves_parallelism():
    """The lowering step passes node.parallelism through to the IR DataNode."""
    from refactory_dsh.lowering import to_workflow
    from refactory_packages.livemath.mode import Solver

    state = Solver().state_dict()
    state["knobs"]["data_parallelism"]["value"] = 8

    wf = to_workflow(state)
    data_node = wf.nodes["data"]
    assert data_node.parallelism == 8, (
        f"lowering should preserve parallelism=8, got {data_node.parallelism}"
    )


# ── pass_state locking under concurrent writes ───────────────────────────

def test_pass_state_locked_under_concurrent_writes(tmp_path):
    """Concurrent pass_state updates do not lose increments.

    Without fcntl.LOCK_EX, a concurrent read-modify-write could lose a counter
    increment. This test writes 20 increments concurrently and checks the final
    count.
    """
    from refactory_dsh.items import _locked_pass_state_update, _read_pass_state, _write_pass_state

    N = 20
    node_id = "test_node"
    # Initialise pass state.
    _write_pass_state(tmp_path, node_id, {"processed": 0})

    def increment(_unused: int) -> None:
        _locked_pass_state_update(
            tmp_path, node_id,
            lambda ps: ps.__setitem__("processed", ps.get("processed", 0) + 1),
        )

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(increment, range(N)))

    final = _read_pass_state(tmp_path, node_id)
    assert final["processed"] == N, (
        f"expected {N} increments, got {final['processed']} — likely a race"
    )


# ── additional: parallelism validation ────────────────────────────────────

def test_parallelism_validation():
    """Parallelism < 1 and parallelism != 1 on non-data nodes are refused."""
    with pytest.raises(nn.graph.GraphError, match="parallelism must be between 1 and 64"):
        nn.Node("x", "data", subgraph_entry="a", subgraph_exit="a",
                writes=("current_item.json",), parallelism=0)

    with pytest.raises(nn.graph.GraphError, match="parallelism is a data-node field"):
        nn.Node("x", "op", call="foo:bar", parallelism=2)


def test_from_state_roundtrip():
    """parallelism survives a to_state/from_state roundtrip."""
    node = nn.Node("d", "data", subgraph_entry="a", subgraph_exit="a",
                   writes=("current_item.json",), parallelism=8)
    state = node.to_state()
    restored = nn.Node.from_state(state)
    assert restored.parallelism == 8


# ── LiveMath knob test ────────────────────────────────────────────────────

def test_livemath_data_parallelism_knob():
    """LiveMath's data node takes its parallelism from the knob, whatever set it."""
    from refactory_packages.livemath.mode import Solver

    solver = Solver()
    node = solver.graph().nodes["data"]
    assert node.parallelism_knob == "data_parallelism"
    assert node.effective_parallelism(solver.values()) == 1
    assert node.effective_parallelism({**solver.values(), "data_parallelism": 24}) == 24


def test_parallelism_knob_must_be_one_of_the_nodes_knobs():
    with pytest.raises(nn.graph.GraphError, match="must be one of the node's knobs"):
        nn.Node("d", "data", subgraph_entry="a", subgraph_exit="a",
                writes=("current_item.json",), parallelism_knob="par")


# ── every path that sets the knob gets it: `refactory run`, resume ─────────

def test_run_knob_reaches_the_lowered_graph(tmp_path):
    """``refactory run livemath/solver --knob data_parallelism=4`` (interactive.prepare)
    lowers a data node that runs 4 items at once; without the knob, 1."""
    from refactory_dsh import interactive

    def lowered(run: Path, knobs: list[str]) -> dict[str, Any]:
        name, graphs = interactive.prepare("livemath/solver", workdir=run, knobs=knobs, project=REPO_ROOT)
        return json.loads((graphs / f"{name}.graph.json").read_text())["nodes"]["data"]

    assert lowered(tmp_path / "p4", ["data_parallelism=4"])["parallelism"] == 4
    assert lowered(tmp_path / "p1", [])["parallelism"] == 1


class _ResumeTask(Task):
    """Halts every item while the flag file exists, so the first run stops in the pass."""

    name = "resume_bound_task"

    def setup(self, instance, workspace, **kw):
        from refactory_dsh.nn.executor import HaltStop

        if Path(os.environ["HALT_FLAG"]).exists():
            raise HaltStop("item_gate", reason="stopped for the resume test")

    def verify(self, instance, workspace, **kw):
        return VerifyResult(passed=True, score=1.0)


class _BoundMode(nn.Mode):
    """Three items whose body needs all three in flight (a barrier), at the knob's parallelism."""

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"par": nn.Knob(1, "int", lo=1, hi=8, frozen=True, description="items at once")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data", subgraph_entry="body", subgraph_exit="body",
            writes=(CURRENT_ITEM,), produces=("item_scores",), task_ref="resume_bound_task",
            inline_items=tuple({"id": str(i)} for i in range(3)),
            parallelism_knob="par", knobs=("par",),
        ))
        graph.add(nn.Node("body", "op", call="test_ops_parallel:barrier_op",
                          reads=(CURRENT_ITEM,), produces=("counted",)))
        graph.add(nn.Node("report", "op", call="loop_ops:noop"))
        graph.connect("data", "report")
        return graph


def test_resume_runs_items_at_the_bound_knobs_parallelism(tmp_path, monkeypatch):
    """A run started with ``par=3`` and resumed (``Mode.resume``, which configures the
    recorded knobs on a fresh mode) runs its items three at a time again."""
    from refactory_dsh.task import register_task

    register_task("resume_bound_task", _ResumeTask)
    flag = tmp_path / "halt"
    flag.write_text("")
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    monkeypatch.setenv("HALT_FLAG", str(flag))
    monkeypatch.setenv("BARRIER_DIR", str(barrier))
    monkeypatch.setenv("BARRIER_K", "3")
    state = _BoundMode().state_dict()
    state["knobs"]["par"]["value"] = 3
    run = tmp_path / "run"

    first = _BoundMode(state=state).run(run, max_steps=200)
    assert first.halted
    flag.unlink()
    resumed = _BoundMode.resume(run)
    assert resumed.exit_ok, resumed.error
    assert [r.get("error_text") for r in resumed.facts["item_scores"]] == [None, None, None]
    assert len(list(barrier.glob("done_*"))) == 3


# ── the spine's loader folds a pass's declared facts in item order ────────

def test_finish_folds_declared_facts_in_item_order(tmp_path):
    """Items that end in reverse order leave the last item's declared fact, as the
    walker's item-order fold does, not the last to end."""
    from refactory_dsh.items import command_begin, command_end, command_finish, command_prepare

    run, node = _loader_run(tmp_path, ("x", "y", "z"), produces=("item_scores", "last"))
    command_prepare(node, run, requested=4)
    for index in (2, 1, 0):
        begun = command_begin(node, run, index, concurrent=True)
        (Path(begun["pass_dir"]) / "facts.json").write_text(json.dumps({"last": begun["id"]}))
        command_end(node, run, index, concurrent=True)
    # Nothing is folded while the items end; finish folds them, in item order.
    assert "last" not in json.loads((run / "facts.json").read_text())
    command_finish(node, run)
    assert json.loads((run / "facts.json").read_text())["last"] == "z"


# ── the per-item aliases: replaced whatever is there, cleared, not left behind ──

@pytest.mark.parametrize("leftover", ["directory", "dangling"])
def test_a_leftover_at_an_alias_path_does_not_crash_the_node(tmp_path, monkeypatch, leftover):
    """A resumed node whose alias path holds a real folder (an abandoned phase's
    ``mkdir(parents=True)``) or a dangling link (a stopped run) still runs its items."""
    graph = _noop_graph(tmp_path, monkeypatch, parallelism=2)
    run = tmp_path / "run"
    alias = run / ".refactory" / "live" / "data" / "0_0"
    alias.parent.mkdir(parents=True)
    if leftover == "directory":
        (alias / "copied").mkdir(parents=True)
    else:
        alias.symlink_to(tmp_path / "gone", target_is_directory=True)
    result = execute(graph, run, max_steps=200, start="data")
    assert result.exit_ok, result.error
    assert len(list((tmp_path / "visited").glob("*"))) == 3


def test_no_alias_folder_is_left_after_a_pass(tmp_path, monkeypatch):
    result = execute(_noop_graph(tmp_path, monkeypatch, parallelism=2), tmp_path / "run", max_steps=200)
    assert result.exit_ok, result.error
    assert not (tmp_path / "run" / ".refactory" / "live").exists()


def test_prepare_clears_an_earlier_visits_aliases(tmp_path):
    from refactory_dsh.items import command_prepare

    run, node = _loader_run(tmp_path, ("x",))
    leftover = run / ".refactory" / "live" / "data" / "x_0"
    (leftover / "copied").mkdir(parents=True)
    command_prepare(node, run, requested=4)
    assert not (run / ".refactory" / "live" / "data").exists()


# ── parallelism and its knob: one or the other, an integer in range ───────

def test_parallelism_and_its_knob_are_exclusive():
    with pytest.raises(nn.graph.GraphError, match="not both"):
        nn.Node("d", "data", subgraph_entry="a", subgraph_exit="a", writes=("current_item.json",),
                parallelism=4, parallelism_knob="par", knobs=("par",))


@pytest.mark.parametrize("knobs,message", [
    ({}, "has no value"),
    ({"par": "4"}, "must be an integer"),
    ({"par": True}, "must be an integer"),
    ({"par": 0}, "between 1 and 64"),
    ({"par": 1000}, "between 1 and 64"),
])
def test_a_bad_bound_knob_value_is_a_graph_error(knobs, message):
    node = nn.Node("d", "data", subgraph_entry="a", subgraph_exit="a", writes=("current_item.json",),
                   parallelism_knob="par", knobs=("par",))
    with pytest.raises(nn.graph.GraphError, match=message):
        node.effective_parallelism(knobs)


# ── a setup or verify that outlives its deadline cannot write into the pass ─

class _LateTask(Task):
    name = "late_task"

    def setup(self, instance, workspace, **kw):
        if instance.id == "slow":
            time.sleep(2.5)
            (Path(workspace) / "late.txt").write_text("written after the deadline")

    def verify(self, instance, workspace, **kw):
        return VerifyResult(passed=True, score=1.0)


@pytest.mark.parametrize("parallelism", [1, 4])
def test_timed_out_phase_cannot_write_into_the_pass(tmp_path, monkeypatch, parallelism):
    from refactory_dsh.nn.executor import Executor
    from refactory_dsh.task import register_task

    register_task("late_task", _LateTask)
    monkeypatch.setattr(Executor, "_data_item_deadline", lambda self: 1.0)
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="body", subgraph_exit="body",
        writes=(CURRENT_ITEM,), produces=("item_scores",), task_ref="late_task",
        inline_items=({"id": "ok"}, {"id": "slow"}), parallelism=parallelism,
    ))
    graph.add(nn.Node("body", "op", call="loop_ops:noop"))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("data", "report")
    run = tmp_path / "run"

    result = execute(graph, run, max_steps=200)
    time.sleep(2.5)  # past the abandoned setup's late write
    slow = run / "passes" / "data" / "slow_1"
    assert [r.get("error_kind") for r in result.facts["item_scores"]] == [None, "timeout"]
    assert not (slow / "late.txt").exists()
    if parallelism > 1:
        assert (slow / ".timed_out").read_text() == "setup"


# ── F21: fork/join inside a data node's per-item pass ─────────────────────
#
# One graph (_fork_graph) runs on the walker and, lowered with to_workflow, on the
# spine through the real drain. Its branches each write a file and the pass's exit
# scores the item from both (loop_ops.score_from_branches), so a runtime that
# skips or mis-runs a branch scores differently. The Task (loop_ops:ForkBranchTask)
# is one reference both runtimes import.

_NODE = shutil.which("node")
_DRAIN = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain.mts"
_DRAIN_AGENT = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
_PYTHON_BIN = str(REPO_ROOT / ".venv/bin")

_needs_spine = pytest.mark.skipif(
    _NODE is None
    or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists(),
    reason="the spine drain needs Node and the installed workspace (pnpm install)",
)

_FORK_TASK = "loop_ops:ForkBranchTask"

#: Item scores when both branches run: a gives 0.5 (0.25 for item 1), b adds 0.5.
_BOTH_BRANCHES = [("0", 1.0), ("1", 0.75), ("2", 1.0)]

#: Merged facts only one runtime writes, left out of the two-runtime comparison:
#: the spine records the parallelism its drain used; the walker does not.
_RUNTIME_ONLY_FACTS = frozenset({"data_parallelism_effective"})


def _fork_graph(
    *,
    parallelism: int = 1,
    branch_b: str = "loop_ops:fork_branch_b",
    after_join: nn.Node | None = None,
) -> nn.Graph:
    """data -> [d_entry -> fork1 -> (branch_a, branch_b) -> join1 -> d_exit] -> report.

    ``branch_b`` is an op's call, or ``"agent"`` for an agent whose answer lands in
    ``b_out.txt``: the walker captures its model's reply there, and the spine's
    agent drain writes its scripted reply there as a worker would. ``after_join`` goes between ``join1`` and ``d_exit``.
    """
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="d_entry", subgraph_exit="d_exit",
        writes=(CURRENT_ITEM,), produces=("item_scores",), parallelism=parallelism,
        task_ref=_FORK_TASK,
    ))
    graph.add(nn.Node("d_entry", "op", call="loop_ops:noop"))
    graph.add(nn.Node("fork1", "fork", targets=("branch_a", "branch_b")))
    graph.add(nn.Node("branch_a", "op", call="loop_ops:fork_branch_a",
                      reads=(CURRENT_ITEM,), writes=("a_out.txt",)))
    if branch_b == "agent":
        graph.add(nn.Node("branch_b", "agent", prompt_template="answer", writes=("b_out.txt",)))
    else:
        graph.add(nn.Node("branch_b", "op", call=branch_b,
                          reads=(CURRENT_ITEM,), writes=("b_out.txt",)))
    graph.add(nn.Node("join1", "join", sources=("branch_a", "branch_b")))
    graph.add(nn.Node("d_exit", "op", call="loop_ops:score_from_branches",
                      reads=("a_out.txt", "b_out.txt"), writes=("item_result.json",)))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("d_entry", "fork1")
    graph.connect("fork1", "join1")
    graph.connect("branch_a", "join1")
    graph.connect("branch_b", "join1")
    if after_join is None:
        graph.connect("join1", "d_exit")
    else:
        graph.add(after_join)
        graph.connect("join1", after_join.id)
        graph.connect(after_join.id, "d_exit",
                      condition="proceed" if after_join.kind == "gate" else None)
    graph.connect("data", "report")
    return graph


class _SimpleFakeModel:
    """A model client that answers every agent prompt with ``agent_answer``."""

    thread_safe = True

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, system: str = "", **kw: Any) -> str:
        self.prompts.append(prompt)
        return "agent_answer"

    def for_node(self, node_id: str, role: str | None = None) -> "_SimpleFakeModel":
        return self

    def with_wall(self, seconds: float) -> "_SimpleFakeModel":
        return self

    def route(self) -> dict[str, Any]:
        return {"worker_backend": "dsh"}


def _walker(graph: nn.Graph, run: Path, **kw: Any) -> Any:
    return execute(graph, run, max_steps=200, **kw)


def _spine(graph: nn.Graph, run: Path, tmp_path: Path, *, timeout_ms: int = 30_000,
           env: dict[str, str] | None = None, drain: Path = _DRAIN_AGENT) -> tuple[int, str]:
    """Lower ``graph`` into ``run`` and drive it through the real spine drain.

    Returns the drain's exit code and the status it ended on. The agent drain
    answers ``branch_b`` with ``agent_answer``, as the walker's model does; the
    plain drain (``_DRAIN``) is the one that resumes a run (``DRAIN_RESUME=1``).
    """
    run.mkdir(parents=True, exist_ok=True)
    state = {"graph": graph.to_state()}
    if not (run / ".refactory" / "declarations.json").exists():
        lowering.write_declarations(state, run)
    graph_path, _ = lowering.write_mode_graph(state, tmp_path / f"graphs-{run.name}", name="fork_in_data")
    responses = tmp_path / f"responses-{run.name}.json"
    responses.write_text(json.dumps({"branch_b": "agent_answer"}))
    run_env = {**os.environ, "REFACTORY_WORKDIR": str(run), "PYTHONPATH": str(HERE), **(env or {})}
    run_env["PATH"] = f"{_PYTHON_BIN}:{run_env.get('PATH', '')}"
    done = subprocess.run(
        [str(_NODE), "--import", "tsx/esm", str(drain), str(graph_path), str(run),
         f"{_PYTHON_BIN}/python", str(timeout_ms), str(responses)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180, env=run_env,
    )
    if done.returncode != 0:
        sys.stderr.write(f"spine drain: rc={done.returncode}\nstdout: {done.stdout[-2000:]}\n"
                         f"stderr: {done.stderr[-3000:]}\n")
    lines = done.stdout.strip().splitlines()
    status = json.loads(lines[-1]).get("status", "") if lines and lines[-1].startswith("{") else ""
    return done.returncode, status


def _spine_facts(run: Path) -> dict[str, Any]:
    return json.loads((run / "facts.json").read_text())


def _score_tuples(records: list[dict[str, Any]]) -> list[tuple[str, float]]:
    return sorted((str(r["id"]), float(r.get("score") or 0)) for r in records)


def _record_keys(records: list[dict[str, Any]]) -> list[tuple[str, bool, float, Any]]:
    return sorted((str(r["id"]), bool(r["passed"]), float(r.get("score") or 0), r.get("error_kind"))
                  for r in records)


def _visits(log: Path) -> list[str]:
    return log.read_text().split() if log.exists() else []


# ── walker ──


@pytest.mark.parametrize("parallelism", [1, 3])
def test_fork_inside_data_op_branches(tmp_path, parallelism):
    """[walker] Both op branches run for every item and their outputs score it."""
    result = _walker(_fork_graph(parallelism=parallelism), tmp_path / "run")
    assert result.exit_ok, result.error
    assert _score_tuples(result.facts["item_scores"]) == _BOTH_BRANCHES
    assert result.item_records == [] or all(r.get("error_kind") is None for r in result.item_records)


@pytest.mark.parametrize("parallelism", [1, 3])
def test_fork_inside_data_agent_branch(tmp_path, parallelism):
    """[walker] An agent branch's reply is captured into b_out.txt and scores the item."""
    model = _SimpleFakeModel()
    result = _walker(_fork_graph(parallelism=parallelism, branch_b="agent"), tmp_path / "run", model=model)
    assert result.exit_ok, result.error
    assert len(model.prompts) == 3, "the agent branch did not run once per item"
    assert _score_tuples(result.facts["item_scores"]) == _BOTH_BRANCHES


@pytest.mark.parametrize("parallelism", [1, 3])
def test_fork_inside_data_failed_branch(tmp_path, parallelism):
    """[walker] A branch raising ValueError fails its item only; the pass goes on.

    Item 0's branch_b raises ``ValueError``. The fork reports it as a failed branch
    and the data node records the item with ``error_kind == "RuntimeError"``, not
    ``ValueError``: the walker wraps a failed branch in ``RuntimeError``
    (executor.py, the ``_BranchFailed`` handler; data_parallel.py likewise) so the
    kind matches the spine's for a failed pass. The original message is kept in
    ``error_text``. Items 1 and 2 score from both branches as usual.
    """
    result = _walker(_fork_graph(parallelism=parallelism, branch_b="loop_ops:fork_branch_b_fail_on_0"),
                     tmp_path / "run")
    assert result.exit_ok, result.error
    assert _score_tuples(result.facts["item_scores"]) == [("0", 0.0), ("1", 0.75), ("2", 1.0)]
    failed = [r for r in result.item_records if r.get("error_kind")]
    assert [(r["id"], r["error_kind"]) for r in failed] == [("0", "RuntimeError")]
    assert "deliberate branch failure" in failed[0]["error_text"]


def test_fork_inside_data_resume(tmp_path, monkeypatch):
    """[walker] Stop inside item 1's fork, resume at the data node: item 0 is not re-run.

    branch_b halts the run once, on item 1, after item 0 has finished. The resume
    keeps item 0's score from its finished pass, replays the interrupted item 1
    (its pass has no ``.done``) and runs item 2.
    """
    log = tmp_path / "walker.log"
    monkeypatch.setenv("PASS_TEST_LOG", str(log))
    graph = _fork_graph(branch_b="loop_ops:fork_branch_b_stop_on_1")
    run = tmp_path / "run"
    first = _walker(graph, run)
    assert first.halted, first.error
    stopped = _visits(log)
    assert sorted(stopped) == ["0:a", "0:b", "1:a", "1:b"], stopped
    resumed = _walker(graph, run, start="data", facts=dict(first.facts))
    assert resumed.exit_ok, resumed.error
    after = _visits(log)[len(stopped):]
    assert sorted(after) == ["1:a", "1:b", "2:a", "2:b"], f"re-ran after the stop: {after}"
    assert _score_tuples(resumed.facts["item_scores"]) == _BOTH_BRANCHES


def test_data_inside_fork_validation(tmp_path):
    """[walker] A data node inside a fork branch is refused by validate (V13) and lowering."""
    from refactory_dsh.lowering import LoweringError, to_workflow
    from refactory_dsh.nn.validate import validate

    graph = nn.Graph()
    graph.add(nn.Node("start", "op", call="loop_ops:noop"))
    graph.add(nn.Node("fork1", "fork", targets=("data_branch", "other")))
    graph.add(nn.Node(
        "data_branch", "data", subgraph_entry="d_work", subgraph_exit="d_exit",
        writes=(CURRENT_ITEM,), produces=("item_scores",), inline_items=({"id": "0"},),
    ))
    graph.add(nn.Node("d_work", "op", call="loop_ops:noop"))
    graph.add(nn.Node("d_exit", "op", call="loop_ops:noop"))
    graph.add(nn.Node("other", "op", call="loop_ops:noop"))
    graph.add(nn.Node("join1", "join", sources=("data_branch", "other")))
    graph.add(nn.Node("end", "op", call="loop_ops:noop"))
    graph.connect("start", "fork1")
    graph.connect("fork1", "join1")
    graph.connect("d_work", "d_exit")
    graph.connect("join1", "end")

    v13 = [i for i in validate(graph, {}, declared_knobs={}) if i.code == "V13"]
    assert [i.node for i in v13] == ["fork1"]
    assert "data node 'data_branch' inside a fork branch is not supported" in v13[0].message
    with pytest.raises(LoweringError, match="data node 'data_branch' inside a fork branch"):
        to_workflow({"graph": graph.to_state()})


def _fork_with_conditional_edge_first(*, data_in_branch: bool) -> nn.Graph:
    """fork1's first out-edge is a conditional one to ``side``; its join edge comes second.

    With ``data_in_branch`` the data node is branch ``x``; otherwise it follows the join.
    """
    graph = nn.Graph()
    graph.add(nn.Node("fork1", "fork", targets=("x", "y")))
    graph.add(nn.Node("side", "op", call="loop_ops:noop"))
    if data_in_branch:
        graph.add(nn.Node("x", "data", subgraph_entry="d_work", subgraph_exit="d_work",
                          writes=(CURRENT_ITEM,), produces=("item_scores",), inline_items=({"id": "0"},)))
        graph.add(nn.Node("d_work", "op", call="loop_ops:noop"))
        after = nn.Node("end", "op", call="loop_ops:noop")
    else:
        graph.add(nn.Node("x", "op", call="loop_ops:noop"))
        after = nn.Node("end", "data", subgraph_entry="d_work", subgraph_exit="d_work",
                        writes=(CURRENT_ITEM,), produces=("item_scores",), inline_items=({"id": "0"},))
        graph.add(nn.Node("d_work", "op", call="loop_ops:noop"))
    graph.add(nn.Node("y", "op", call="loop_ops:noop"))
    graph.add(nn.Node("join1", "join", sources=("x", "y")))
    graph.add(after)
    graph.connect("fork1", "side", condition="other")
    graph.connect("fork1", "join1")
    graph.connect("x", "join1")
    graph.connect("y", "join1")
    graph.connect("join1", "end")
    return graph


def test_v13_takes_the_join_from_the_unconditional_edge():
    """V13 and lowering stop a branch walk at the fork's join, the target of its
    unconditional out-edge, even when a conditional edge is listed first.

    Taking the first out-edge instead walks past the join and flags a data node
    that follows it; and still flags one inside a branch.
    """
    from refactory_dsh.lowering import LoweringError, to_workflow
    from refactory_dsh.nn.validate import validate

    after = _fork_with_conditional_edge_first(data_in_branch=False)
    assert [i for i in validate(after, {}, declared_knobs={}) if i.code == "V13"] == []
    try:
        to_workflow({"graph": after.to_state()})
    except LoweringError as exc:
        assert "inside a fork branch" not in str(exc), exc

    inside = _fork_with_conditional_edge_first(data_in_branch=True)
    assert [i.node for i in validate(inside, {}, declared_knobs={}) if i.code == "V13"] == ["fork1"]
    with pytest.raises(LoweringError, match="data node 'x' inside a fork branch"):
        to_workflow({"graph": inside.to_state()})


@pytest.mark.parametrize("parallelism", [1, 3])
def test_fork_inside_data_deadline(tmp_path, monkeypatch, parallelism):
    """[walker] The per-item deadline covers a pass with a fork in it.

    Item 1's verify outlives the 1 s deadline and is recorded as a timeout; items 0
    and 2, whose fork walks finish well inside it, score from both branches.
    """
    from refactory_dsh.nn.executor import Executor

    monkeypatch.setattr(Executor, "_data_item_deadline", lambda self: 1.0)
    monkeypatch.setenv("FORK_TEST_SLOW_ITEM", "1")
    monkeypatch.setenv("FORK_TEST_SLOW_SECONDS", "4")
    result = _walker(_fork_graph(parallelism=parallelism), tmp_path / "run")
    assert result.exit_ok, result.error
    scores = {s["id"]: (s.get("error_kind"), s.get("score")) for s in result.facts["item_scores"]}
    assert scores == {"0": (None, 1.0), "1": ("timeout", 0.0), "2": (None, 1.0)}, scores


@pytest.mark.parametrize("parallelism", [1, 3])
def test_fork_inside_data_step_cap(tmp_path, parallelism):
    """[walker] The shared step cap counts the nodes a fork runs inside a pass.

    ``max_steps=2`` grows by (items - 1) x the body's node count for the data node,
    to 16. Each item walks six nodes (d_entry, fork1, branch_a, branch_b, join1,
    d_exit), so the data step and two items fit and the third does not: run one at
    a time, item 2 is the one that fails. Run concurrently, the items draw on the
    same 15 steps at once, so which items run short (one or two) depends on how
    they interleave. Either way every capped item carries the step-cap error and
    every other item scores from both branches.
    """
    result = execute(_fork_graph(parallelism=parallelism), tmp_path / "run", max_steps=2)
    capped = {r["id"]: r["error_text"] for r in result.item_records if r.get("error_kind")}
    if parallelism == 1:
        assert capped == {"2": "the run exceeded 16 steps"}, result.item_records
    else:
        assert capped and set(capped.values()) == {"the run exceeded 16 steps"}, result.item_records
    scored = [t for t in _score_tuples(result.facts["item_scores"]) if t[0] not in capped]
    assert scored == [t for t in _BOTH_BRANCHES if t[0] not in capped]


@pytest.mark.parametrize("parallelism", [1, 3])
def test_fork_inside_data_stop(tmp_path, monkeypatch, parallelism):
    """[walker] A halt raised inside item 1's fork stops the run there.

    The walker has no graph control; a halt is how it stops (the spine's stop
    control is test_fork_inside_data_stop_and_pause_spine). Item 0 finished before
    the stop; sequentially, item 2 never starts.
    """
    log = tmp_path / "walker.log"
    monkeypatch.setenv("PASS_TEST_LOG", str(log))
    result = _walker(_fork_graph(parallelism=parallelism, branch_b="loop_ops:fork_branch_b_stop_on_1"),
                     tmp_path / "run")
    assert result.halted, f"expected halted, got exit_ok={result.exit_ok}, error={result.error}"
    assert {"0:a", "0:b", "1:b"} <= set(_visits(log))
    if parallelism == 1:
        assert not any(v.startswith("2:") for v in _visits(log)), _visits(log)


def test_fork_inside_data_pauses_at_a_user_gate(tmp_path, monkeypatch):
    """[walker] A user gate after the join pauses the run in the first item's pass.

    The walker's pause is a user gate with nobody to answer it (GateStop): the run
    stops with the gate waiting, after item 0's fork ran both branches and before
    its d_exit or any later item.
    """
    log = tmp_path / "walker.log"
    monkeypatch.setenv("PASS_TEST_LOG", str(log))
    gate = nn.Node("ask", "gate", gate_kind="user", notes="Go on?")
    run = tmp_path / "run"
    result = _walker(_fork_graph(after_join=gate), run)
    assert not result.exit_ok
    assert "gate 'ask' is waiting for a person" in str(result.error), result.error
    assert sorted(_visits(log)) == ["0:a", "0:b"]
    assert list(run.glob("passes/data/*/item_result.json")) == []



# ── both runtimes ──


#: Item scores per branch_b: both branches run, or item 0's branch_b fails.
_TWO_RUNTIME_CASES = {
    "agent": _BOTH_BRANCHES,
    "loop_ops:fork_branch_b": _BOTH_BRANCHES,
    "loop_ops:fork_branch_b_fail_on_0": [("0", 0.0), ("1", 0.75), ("2", 1.0)],
}


def _assert_same_merged_facts(walker: dict[str, Any], spine: dict[str, Any]) -> None:
    """The facts both runtimes merge are the same; only _RUNTIME_ONLY_FACTS may differ.

    ``item_scores`` entries are compared on id, passed, score and error_kind: the
    records' other fields (exchanges, timings) are each runtime's own bookkeeping.
    """
    assert set(walker) ^ set(spine) <= _RUNTIME_ONLY_FACTS, set(walker) ^ set(spine)
    for key in sorted((set(walker) & set(spine)) - _RUNTIME_ONLY_FACTS):
        if key == "item_scores":
            assert _record_keys(walker[key]) == _record_keys(spine[key])
        else:
            assert walker[key] == spine[key], key


@_needs_spine
@pytest.mark.parametrize("branch_b", sorted(_TWO_RUNTIME_CASES))
@pytest.mark.parametrize("parallelism", [1, 3])
def test_fork_inside_data_two_runtimes(tmp_path, parallelism, branch_b):
    """[both runtimes] One graph, run by the walker and lowered onto the spine's drain.

    Each item's score comes from both fork branches' files, and item 1 scores
    differently from the others, so a runtime that skips a branch (or runs one item's
    branch for another) fails the comparison. Cases: an agent branch, an op-only
    fork, and a branch that raises on item 0 (the item scores 0 with its error kind
    kept, and the pass goes on to items 1 and 2).
    """
    graph = _fork_graph(parallelism=parallelism, branch_b=branch_b)
    expected = _TWO_RUNTIME_CASES[branch_b]

    model = _SimpleFakeModel()
    walker = _walker(graph, tmp_path / "walker", model=model)
    assert walker.exit_ok, walker.error
    assert len(model.prompts) == (3 if branch_b == "agent" else 0)

    spine_run = tmp_path / "spine"
    rc, status = _spine(graph, spine_run, tmp_path)
    assert (rc, status) == (0, "done")
    spine = _spine_facts(spine_run)

    assert _score_tuples(walker.facts["item_scores"]) == expected
    assert _score_tuples(spine["item_scores"]) == expected
    _assert_same_merged_facts(walker.facts, spine)
    failed = [(r["id"], r["error_kind"]) for r in spine["item_scores"] if r.get("error_kind")]
    assert failed == ([("0", "RuntimeError")] if expected[0][1] == 0.0 else [])


@_needs_spine
def test_fork_inside_data_resume_spine(tmp_path):
    """[spine] Killed inside item 1's fork, resumed from the checkpoint: item 0 is not re-run.

    branch_b kills the drain once, mid-command, in item 1's fork, after item 0
    has finished; the drain resumes the run (DRAIN_RESUME). The log shows item 0
    ran once, the interrupted item 1 ran again and item 2 ran, and the scores
    are the uninterrupted run's.
    """
    log = tmp_path / "spine.log"
    env = {"PASS_TEST_LOG": str(log), "PASS_TEST_RUNTIME": "spine"}
    graph = _fork_graph(branch_b="loop_ops:fork_branch_b_stop_on_1")
    run = tmp_path / "spine"
    rc, _ = _spine(graph, run, tmp_path, env=env, drain=_DRAIN)
    assert rc != 0
    stopped = _visits(log)
    assert {"0:a", "0:b", "1:b"} <= set(stopped) and not any(v.startswith("2:") for v in stopped), stopped
    rc, status = _spine(graph, run, tmp_path, env={**env, "DRAIN_RESUME": "1"}, drain=_DRAIN)
    assert (rc, status) == (0, "done")
    after = _visits(log)[len(stopped):]
    assert not any(v.startswith("0:") for v in after), f"finished item 0 re-ran: {after}"
    assert "1:b" in after and sorted(v for v in after if v.startswith("2:")) == ["2:a", "2:b"], after
    assert _score_tuples(_spine_facts(run)["item_scores"]) == _BOTH_BRANCHES


@_needs_spine
@pytest.mark.parametrize("control", ["stop", "pause"])
def test_fork_inside_data_stop_and_pause_spine(tmp_path, control):
    """[spine] A person's stop or pause filed during item 1's fork holds the run there.

    branch_b files the graph control in item 1's pass; the spine applies it at its
    next boundary inside the pass, so item 1's d_exit and item 2 never run.
    """
    log = tmp_path / "spine.log"
    run = tmp_path / "spine"
    env = {"PASS_TEST_LOG": str(log), "FORK_TEST_RUN": str(run), "FORK_TEST_CONTROL": control}
    rc, status = _spine(_fork_graph(branch_b="loop_ops:fork_branch_b_control_on_1"), run, tmp_path, env=env)
    recorded = json.loads((run / ".refactory/spine-status.json").read_text())["status"]
    assert recorded == {"stop": "halted", "pause": "paused"}[control], (rc, status, recorded)
    receipt = json.loads((run / ".refactory/graph-control-receipt.json").read_text())
    assert (receipt["action"], receipt["status"]) == (control, "applied")
    assert {"0:a", "0:b", "1:a", "1:b"} == set(_visits(log)), _visits(log)
    finished = sorted(p.parent.name for p in run.glob("passes/data/*/item_result.json"))
    assert finished == ["0_0"], finished


@_needs_spine
def test_fork_inside_data_deadline_spine(tmp_path):
    """[spine] The per-item deadline covers a pass with a fork in it.

    The spine bounds each loader command by the command timeout (runDataCommand);
    item 1's verify outlives it and the item is recorded as a timeout, while items
    0 and 2 score from both branches.
    """
    run = tmp_path / "spine"
    env = {"FORK_TEST_SLOW_ITEM": "1", "FORK_TEST_SLOW_SECONDS": "12"}
    rc, status = _spine(_fork_graph(), run, tmp_path, timeout_ms=6_000, env=env)
    assert (rc, status) == (0, "done")
    scores = {s["id"]: (s.get("error_kind"), s.get("score")) for s in _spine_facts(run)["item_scores"]}
    assert scores == {"0": (None, 1.0), "1": ("timeout", 0.0), "2": (None, 1.0)}, scores
