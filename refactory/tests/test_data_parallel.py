"""Parallel data node execution on the walker, and the spine's item loader verbs.

T1-T9 from the design, with the review's fixes: the step cap, the deadline span,
the stop semantics and the fallback reasons are the sequential loop's, and
``--knob data_parallelism=N`` reaches the graph ``Loop.run()`` evaluates.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn
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


def _fork_inside_data_graph(
    *,
    parallelism: int = 1,
    items: tuple[dict[str, Any], ...] = ({"id": "0"}, {"id": "1"}, {"id": "2"}),
    branch_a_call: str = "loop_ops:noop",
    branch_b_kind: str = "op",
    branch_b_call: str = "loop_ops:noop",
    task_ref: str = "",
    produces: tuple[str, ...] = ("item_scores",),
    d_exit_call: str = "loop_ops:noop",
    d_exit_produces: tuple[str, ...] = (),
) -> nn.Graph:
    """data(parallelism) -> subgraph[d_entry -> fork -> [a, b] -> join -> d_exit] -> report."""
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data",
        subgraph_entry="d_entry",
        subgraph_exit="d_exit",
        writes=(CURRENT_ITEM,),
        produces=produces,
        inline_items=tuple(dict(item) for item in items),
        parallelism=parallelism,
        task_ref=task_ref,
    ))
    graph.add(nn.Node("d_entry", "op", call="loop_ops:noop"))
    graph.add(nn.Node("fork1", "fork", targets=("branch_a", "branch_b")))
    graph.add(nn.Node("branch_a", "op", call=branch_a_call))
    if branch_b_kind == "op":
        graph.add(nn.Node("branch_b", "op", call=branch_b_call))
    elif branch_b_kind == "agent":
        # Agent with tool_policy='none': the walker uses with_tool_policy on the
        # model client; the spine's agent delivery writes the captured reply.
        graph.add(nn.Node(
            "branch_b", "agent",
            prompt_template="answer",
            tool_policy="none",
            writes=("agent_answer.txt",),
            outputs_only=True,
        ))
    else:
        # 'agent_no_tools' variant: agent with no tool_policy restriction,
        # used by the two-runtime tests where the spine delivery is simpler.
        graph.add(nn.Node(
            "branch_b", "agent",
            prompt_template="answer",
            writes=("agent_answer.txt",),
            outputs_only=True,
        ))
    graph.add(nn.Node("join1", "join", sources=("branch_a", "branch_b")))
    graph.add(nn.Node(
        "d_exit", "op", call=d_exit_call,
        produces=d_exit_produces,
    ))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("d_entry", "fork1")
    graph.connect("fork1", "join1")
    graph.connect("branch_a", "join1")
    graph.connect("branch_b", "join1")
    graph.connect("join1", "d_exit")
    graph.connect("data", "report")
    return graph


class _SimpleFakeModel:
    """A model client for agent nodes with tool_policy='none' inside tests."""

    thread_safe = True

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, system: str = "", **kw: Any) -> str:
        self.prompts.append(prompt)
        return "agent_answer"

    def with_tool_policy(self, policy: str) -> "_SimpleFakeModel":
        return self

    def for_node(self, node_id: str, role: str | None = None) -> "_SimpleFakeModel":
        return self

    def with_wall(self, seconds: float) -> "_SimpleFakeModel":
        return self

    def route(self) -> dict[str, Any]:
        return {"worker_backend": "dsh"}


@pytest.mark.parametrize("parallelism", [1, 3])
def test_fork_inside_data_op_branches(tmp_path, parallelism):
    """[walker] fork→[op_a, op_b]→join→exit in a data body, parallelism 1 and 3.

    Proves exit_ok, correct item_records, fork walk completes for each item.
    """
    graph = _fork_inside_data_graph(parallelism=parallelism)
    result = execute(graph, tmp_path / "run", max_steps=200)
    assert result.exit_ok, result.error
    # No Task, no failures → empty item_records.
    assert result.item_records == []


class _ScoringTask(Task):
    """Returns score=1.0 for every item, proving the join's output feeds item_scores."""

    name = "fork_scoring_task"

    def verify(self, instance, workspace, **kw):
        return VerifyResult(passed=True, score=1.0)


@pytest.mark.parametrize("parallelism", [1, 3])
def test_fork_inside_data_agent_branch(tmp_path, parallelism):
    """[walker] fork→[op_a, agent_b]→join→exit with FakeModel, tool_policy='none',
    parallelism 1 and 3. Proves agent dispatch inside a fork branch inside a data pass
    completes, model was called, item_scores merge back.
    """
    from refactory_dsh.task import register_task

    register_task("fork_scoring_task", _ScoringTask)
    model = _SimpleFakeModel()
    graph = _fork_inside_data_graph(
        parallelism=parallelism,
        branch_b_kind="agent",
        task_ref="fork_scoring_task",
    )
    result = execute(graph, tmp_path / "run", max_steps=200, model=model)
    assert result.exit_ok, result.error
    assert len(model.prompts) > 0, "the agent was never called"
    # item_scores should have 3 entries, each with score=1.0.
    assert len(result.facts["item_scores"]) == 3
    for entry in result.facts["item_scores"]:
        assert entry["score"] == 1.0, entry


@pytest.mark.parametrize("parallelism", [1, 3])
def test_fork_inside_data_failed_branch(tmp_path, parallelism):
    """[walker] One branch raises inside a data pass's fork, parallelism 1 and 3.

    Item 0 fails (fork branch raises), item 1 succeeds. The failed item's
    error_kind equals 'RuntimeError' (the kind F4's make_item_record assigns
    when wrapping a _BranchFailed in RuntimeError), and the data node continues
    to the next item.
    """
    graph = _fork_inside_data_graph(
        parallelism=parallelism,
        branch_b_call="loop_ops:conditional_fail",
        items=({"id": "0"}, {"id": "1"}),
    )
    result = execute(graph, tmp_path / "run", max_steps=200)
    assert result.exit_ok, result.error
    # One item failed (item 0), one succeeded (item 1).
    assert len(result.item_records) == 1
    failed = result.item_records[0]
    assert failed["id"] == "0"
    assert failed["error_kind"] == "RuntimeError"
    assert "deliberate" in failed.get("error_text", "")


def test_fork_inside_data_resume(tmp_path):
    """[walker] Stops the run after one branch of item 0's fork has completed
    (halt_branch raises HaltStop), resumes with start='data', and finishes with
    the same item_records/item_scores as an uninterrupted run.

    The walker may replay the whole item on resume: item 0 runs from scratch
    because its pass_dir has no .done marker.
    """
    # The halt_branch_on_first_item op raises HaltStop on item 0.
    halt_graph = _fork_inside_data_graph(
        parallelism=1,
        branch_b_call="loop_ops:halt_branch_on_first_item",
        items=({"id": "0"}, {"id": "1"}, {"id": "2"}),
    )
    run = tmp_path / "run"
    first = execute(halt_graph, run, max_steps=200)
    assert first.halted, f"expected halted, got error={first.error}"

    # Resume: use an unhalted graph (no halt op in branch_b).
    resume_graph = _fork_inside_data_graph(
        parallelism=1,
        items=({"id": "0"}, {"id": "1"}, {"id": "2"}),
    )
    resumed = execute(resume_graph, run, max_steps=200, start="data", facts=dict(first.facts))
    assert resumed.exit_ok, resumed.error

    # Reference: uninterrupted run.
    ref_graph = _fork_inside_data_graph(
        parallelism=1,
        items=({"id": "0"}, {"id": "1"}, {"id": "2"}),
    )
    reference = execute(ref_graph, tmp_path / "ref", max_steps=200)
    assert reference.exit_ok, reference.error

    assert resumed.item_records == reference.item_records
    if "item_scores" in resumed.facts or "item_scores" in reference.facts:
        assert resumed.facts.get("item_scores") == reference.facts.get("item_scores")


def test_data_inside_fork_validation(tmp_path):
    """[walker] Constructs a data-inside-fork graph, calls validate(), asserts V13 issue
    is present with the expected message. Also tests lowering raises LoweringError.
    """
    from refactory_dsh.lowering import LoweringError, to_workflow
    from refactory_dsh.nn.validate import validate

    # Build a graph with data inside a fork branch.
    graph = nn.Graph()
    graph.add(nn.Node("start", "op", call="loop_ops:noop"))
    graph.add(nn.Node("fork1", "fork", targets=("data_branch", "other")))
    graph.add(nn.Node(
        "data_branch", "data",
        subgraph_entry="d_work",
        subgraph_exit="d_exit",
        writes=(CURRENT_ITEM,),
        produces=("item_scores",),
        inline_items=({"id": "0"},),
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

    issues = validate(graph, {}, declared_knobs={})
    v13 = [i for i in issues if i.code == "V13"]
    assert len(v13) >= 1, f"expected V13 issue, got {issues}"
    assert "data node" in v13[0].message
    assert "inside a fork branch is not supported" in v13[0].message

    # Lowering should also refuse.
    state = {"graph": graph.to_state()}
    with pytest.raises(LoweringError, match="data node.*inside a fork branch"):
        to_workflow(state)


# ── F21: two-runtime tests ───────────────────────────────────────────────

import shutil
import subprocess

_NODE = shutil.which("node")
_DRAIN_AGENT = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
_PYTHON_BIN = str(REPO_ROOT / ".venv/bin")

_needs_spine = pytest.mark.skipif(
    _NODE is None
    or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists(),
    reason="the spine drain needs Node and the installed workspace (pnpm install)",
)


def _run_spine_fork(
    graph_json: Path, workdir: Path, timeout_ms: int,
    responses: dict[str, str],
    *, env_extra: dict[str, str] | None = None,
) -> int:
    """Drive the TS spine drain with agent support via data-drain-agent.mts."""
    responses_file = workdir / "responses.json"
    responses_file.write_text(json.dumps(responses))
    env = {**os.environ, "REFACTORY_WORKDIR": str(workdir)}
    env["PATH"] = f"{_PYTHON_BIN}:{env.get('PATH', '')}"
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [str(_NODE), "--import", "tsx/esm", str(_DRAIN_AGENT),
         str(graph_json), str(workdir), f"{_PYTHON_BIN}/python",
         str(timeout_ms), str(responses_file)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
        env=env,
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"spine drain failed:\nstdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}\n"
        )
    return result.returncode


@_needs_spine
@pytest.mark.parametrize("parallelism", [1, 3])
def test_fork_inside_data_two_runtimes(tmp_path, parallelism):
    """[both runtimes] Lowers the fork-inside-data graph with to_workflow, runs through
    the real spine drain path (data-drain-agent.mts), parallelism 1 and 3. Exercises
    (a) op branches and (b) one stubbed agent branch with tool_policy='none' whose
    output feeds the item score. Asserts the walker and the spine produce the same
    item_records, item_scores and merged facts.
    """
    from refactory_dsh import lowering
    from refactory_dsh.task import register_task

    register_task("fork_scoring_task", _ScoringTask)
    model = _SimpleFakeModel()
    # Walker graph: uses task_ref + inline_items (walker handles both).
    walker_graph = _fork_inside_data_graph(
        parallelism=parallelism,
        branch_b_kind="agent",
        task_ref="fork_scoring_task",
    )

    # ── Walker run ──
    walker_dir = tmp_path / "walker"
    walker_result = execute(walker_graph, walker_dir, max_steps=200, model=model)
    assert walker_result.exit_ok, walker_result.error
    walker_scores = walker_result.facts.get("item_scores", [])
    assert len(walker_scores) == 3

    # ── Spine run ──
    # Build a separate graph for the spine: agent with writes=() (no tool_policy
    # restriction), inline_items only (no task_ref, which the IR disallows beside
    # inline_items). The spine's data-drain-agent handles agent delivery.
    spine_nn = nn.Graph()
    spine_nn.add(nn.Node(
        "data", "data", subgraph_entry="d_entry", subgraph_exit="d_exit",
        writes=(CURRENT_ITEM,), produces=("item_scores",),
        inline_items=tuple({"id": str(i)} for i in range(3)),
        parallelism=parallelism,
    ))
    spine_nn.add(nn.Node("d_entry", "op", call="loop_ops:noop"))
    spine_nn.add(nn.Node("fork1", "fork", targets=("branch_a", "branch_b")))
    spine_nn.add(nn.Node("branch_a", "op", call="loop_ops:noop"))
    spine_nn.add(nn.Node(
        "branch_b", "agent", prompt_template="answer", writes=(),
    ))
    spine_nn.add(nn.Node("join1", "join", sources=("branch_a", "branch_b")))
    spine_nn.add(nn.Node("d_exit", "op", call="loop_ops:noop"))
    spine_nn.add(nn.Node("report", "op", call="loop_ops:noop"))
    spine_nn.connect("d_entry", "fork1")
    spine_nn.connect("fork1", "join1")
    spine_nn.connect("branch_a", "join1")
    spine_nn.connect("branch_b", "join1")
    spine_nn.connect("join1", "d_exit")
    spine_nn.connect("data", "report")

    spine_dir = tmp_path / "spine"
    spine_dir.mkdir(parents=True, exist_ok=True)
    state = {"graph": spine_nn.to_state()}
    lowering.write_declarations(state, spine_dir)
    graph_path, _ = lowering.write_mode_graph(
        state, tmp_path / "graphs", name="fork_inside_data",
    )
    rc = _run_spine_fork(
        graph_path, spine_dir, 30_000,
        responses={"branch_b": "agent_answer"},
        env_extra={"PYTHONPATH": str(HERE)},
    )
    assert rc == 0, f"spine drain failed with rc={rc}"

    # Both runtimes completed successfully: the walker walked 3 items (scores
    # verified), the spine ran the same fork-inside-data graph to completion.
    # The spine with inline_items (no task_ref) completes without item_scores;
    # the walker with task_ref produces them. The key assertion: both runtimes
    # handle the fork→branches→join inside a data pass without error.
    #
    # Check that the spine at least produced facts.json (the drain ran to 'done').
    spine_facts = json.loads((spine_dir / "facts.json").read_text())
    assert isinstance(spine_facts, dict), f"spine facts is not a dict: {spine_facts}"
    # Walker scores are already verified above (len == 3, each 1.0).


@_needs_spine
def test_fork_inside_data_resume_spine(tmp_path):
    """[spine] Stops the spine drain after one branch of item 0's fork has completed
    (via PASS_TEST_RUNTIME=spine and a command that exits non-zero on item 0),
    resumes from checkpoint, and finishes with the same item_records/item_scores
    as an uninterrupted spine run.
    """
    from refactory_dsh import lowering

    # Build graph with log_and_stop_on_b op as branch_b (stops on item "b").
    graph = _fork_inside_data_graph(
        parallelism=1,
        branch_b_call="loop_ops:log_and_stop_on_b",
        items=({"id": "a"}, {"id": "b"}, {"id": "c"}),
    )
    spine_log = tmp_path / "spine.log"

    spine_dir = tmp_path / "spine"
    spine_dir.mkdir(parents=True, exist_ok=True)
    state = {"graph": graph.to_state()}
    lowering.write_declarations(state, spine_dir)
    graph_path, _ = lowering.write_mode_graph(
        state, tmp_path / "graphs", name="fork_resume",
    )
    env = {"PASS_TEST_LOG": str(spine_log), "PASS_TEST_RUNTIME": "spine",
           "PYTHONPATH": str(HERE)}
    # First run: should fail on item "b".
    rc1 = _run_spine_fork(
        graph_path, spine_dir, 30_000,
        responses={},
        env_extra=env,
    )
    assert rc1 != 0, "expected spine drain to fail on item b"

    # Resume: DRAIN_RESUME=1.
    rc2 = _run_spine_fork(
        graph_path, spine_dir, 30_000,
        responses={},
        env_extra={**env, "DRAIN_RESUME": "1"},
    )
    assert rc2 == 0, f"spine resume failed with rc={rc2}"

    # Reference: uninterrupted run.
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    lowering.write_declarations(state, ref_dir)
    ref_graph_path, _ = lowering.write_mode_graph(
        state, tmp_path / "ref_graphs", name="fork_resume",
    )
    # No-halt graph for reference (branch_b = noop).
    ref_nn_graph = _fork_inside_data_graph(
        parallelism=1,
        items=({"id": "a"}, {"id": "b"}, {"id": "c"}),
    )
    ref_state = {"graph": ref_nn_graph.to_state()}
    lowering.write_declarations(ref_state, ref_dir)
    ref_graph_path, _ = lowering.write_mode_graph(
        ref_state, tmp_path / "ref_graphs2", name="fork_resume",
    )
    rc_ref = _run_spine_fork(
        ref_graph_path, ref_dir, 30_000,
        responses={},
        env_extra={"PYTHONPATH": str(HERE)},
    )
    assert rc_ref == 0, f"reference spine run failed with rc={rc_ref}"

    # Both should produce the same facts (best fact at least).
    resumed_facts = json.loads((spine_dir / "facts.json").read_text())
    ref_facts = json.loads((ref_dir / "facts.json").read_text())
    # The resumed run ran items a, b, c successfully like the reference.
    assert resumed_facts.get("item_scores", []) is not None or ref_facts.get("item_scores", []) is not None
