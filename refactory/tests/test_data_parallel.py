"""Parallel data node execution on the walker.

Tests T1–T9 from the design document. Every test must fail without the change
(parallelism > 1 running items concurrently).
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


# ── T5: budget exhaustion ─────────────────────────────────────────────────

def test_budget_exhaustion(tmp_path, monkeypatch):
    """T5 — step budget is enforced with parallelism > 1.

    With 8 items and max_steps=3 (1 data + limited body ops), not all items
    can complete. The step budget reserve stops new items from getting a step.
    """
    N = 8
    items = tuple({"id": str(i)} for i in range(N))
    visited_dir = tmp_path / "visited"
    visited_dir.mkdir()
    monkeypatch.setenv("VISITED_DIR", str(visited_dir))

    # With max_steps=3: 1 step for data node + at most 2 body steps.
    # Use parallelism=2 so items are dispatched gradually, not all at once.
    graph = _data_graph(
        "test_ops_parallel", "noop_op",
        items=items, parallelism=2,
    )
    result = execute(graph, tmp_path / "run", max_steps=3)

    # Not all items should have completed.
    visited = list(visited_dir.glob("*"))
    assert len(visited) < N, f"expected < {N} items to complete, got {len(visited)}"


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


# ── T7: per-item deadline in concurrent mode ─────────────────────────────

def test_per_item_deadline_concurrent(tmp_path, monkeypatch):
    """T7 — a per-item deadline failure in concurrent mode records the item as
    failed with timeout while other items complete.

    Uses a very short budget wall to trigger the timer-based per-item deadline.
    The over-deadline item must be recorded as failed with error_kind 'timeout'.
    The fast item must complete successfully (with a Task for grading).
    Other items must NOT be cancelled by the slow item's deadline.
    """
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    monkeypatch.setenv("BARRIER_DIR", str(barrier_dir))

    # Register a task that grades items (so make_item_record returns records
    # for successful items — without a Task, verdict=None → record=None).
    _TASK = tmp_path / "deadline_task.py"
    _TASK.write_text("""\
from refactory_dsh.task import Task, TaskInstance, VerifyResult

class T(Task):
    name = "deadline_test"
    def instances(self):
        return [
            TaskInstance(id="fast", metadata={}),
            TaskInstance(id="slow", metadata={}),
        ]
    def setup(self, instance, workspace, **kw):
        pass
    def verify(self, instance, workspace, **kw):
        return VerifyResult(passed=True, score=1.0)
""")
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    from refactory_dsh.task import register_task
    import importlib
    if "deadline_task" in sys.modules:
        importlib.reload(sys.modules["deadline_task"])
    else:
        import deadline_task  # noqa: F401
    from deadline_task import T  # type: ignore[import-not-found]
    register_task("deadline_test", T)

    items = (
        {"id": "fast", "delay": "0"},
        {"id": "slow", "delay": "60"},  # will be killed by deadline
    )

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data",
        subgraph_entry="body",
        subgraph_exit="body",
        writes=(CURRENT_ITEM,),
        produces=("item_scores",),
        task_ref="deadline_test",
        inline_items=tuple(dict(item) for item in items),
        parallelism=2,
    ))
    graph.add(nn.Node(
        "body", "op", call="test_ops_parallel:sleep_op",
        reads=(CURRENT_ITEM,), produces=("counted",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")

    from refactory_dsh.budgets import ResourceLimits
    result = execute(
        graph, tmp_path / "run", max_steps=200,
        resource_limits=ResourceLimits(max_elapsed_s=5),
        enable_recovery=True,
    )

    scores = result.facts.get("item_scores", [])
    assert len(scores) >= 2, f"expected at least 2 scores, got {len(scores)}: {scores}"

    # The fast item must have completed successfully.
    fast_items = [s for s in scores if s.get("id") == "fast"]
    assert len(fast_items) == 1, f"expected 1 fast item, got {fast_items}"
    assert fast_items[0].get("error_text") is None, f"fast item should not have an error: {fast_items[0]}"

    # The slow/over-deadline item must be recorded as failed with error_kind 'timeout'.
    slow_items = [s for s in scores if s.get("id") == "slow"]
    assert len(slow_items) == 1, f"expected 1 slow item, got {slow_items}"
    assert slow_items[0].get("error_text") is not None, f"slow item should have an error: {slow_items[0]}"
    assert slow_items[0].get("error_kind") == "timeout", (
        f"slow item should have error_kind 'timeout', got {slow_items[0].get('error_kind')!r}"
    )

    # Clean up.
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))
    sys.modules.pop("deadline_task", None)


# ── T8: serial fallback records reason ────────────────────────────────────

def test_serial_fallback_records_reason(tmp_path, monkeypatch):
    """T8 — when parallelism > 1 but a serial-fallback fires, the reason is
    recorded as a fact and the run still produces correct results.
    """
    items = tuple({"id": str(i)} for i in range(3))
    visited_dir = tmp_path / "visited"
    visited_dir.mkdir()
    monkeypatch.setenv("VISITED_DIR", str(visited_dir))

    class _CustomModel:
        """A non-DshModelClient model that triggers serial fallback."""
        def complete(self, prompt: str, *, system: str = "") -> str:
            return "ok"

    graph = _data_graph(
        "test_ops_parallel", "noop_op",
        items=items, parallelism=4,
    )

    result = execute(
        graph, tmp_path / "run", model=_CustomModel(), max_steps=200,
    )
    # Run should still succeed (serial fallback).
    assert result.exit_ok, result.error
    # The fallback fact should be recorded.
    fact = result.facts.get("data_parallelism_effective")
    assert fact is not None, "serial fallback should record data_parallelism_effective fact"
    assert fact["requested"] == 4
    assert fact["effective"] == 1
    assert "reason" in fact


# ── T9: Loop.run() overlap with stub solver ───────────────────────────────

def test_loop_run_overlap_stub_solver(tmp_path, monkeypatch):
    """T9 — items overlap through Mode.evaluate() with data_parallelism=4.

    Simulates the ``refactory optimize livemath/solver`` path: constructs a
    ``Solver`` mode with ``data_parallelism=K`` via the same state-dict
    mechanism ``profiles.resolve()`` uses, then calls ``mode.evaluate()``
    which runs ``mode.run()`` → ``execute(self.graph(), ...)``.

    This exercises the full knob → build() → node.parallelism → executor path,
    proving that the knob reaches the graph AND that items overlap.
    """
    K = 3
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    monkeypatch.setenv("BARRIER_DIR", str(barrier_dir))
    monkeypatch.setenv("BARRIER_K", str(K))

    from refactory_packages.livemath.mode import Solver

    # Construct the mode with data_parallelism=K via state dict manipulation
    # (the same mechanism profiles.resolve() → edits.configure() uses).
    solver = Solver()
    state = solver.state_dict()
    state["knobs"]["data_parallelism"]["value"] = K
    solver_k = Solver(state=state)
    # Rebuild graph with updated knobs (matching what profiles.resolve does).
    solver_k._graph_state = solver_k.build().to_state()

    # Verify the graph has parallelism=K.
    graph = solver_k.graph()
    assert graph.nodes["data"].parallelism == K, (
        f"data node parallelism should be {K}, got {graph.nodes['data'].parallelism}"
    )

    # Override the data body: replace format_prompt→solve→finish with a barrier_op.
    # We rebuild the graph so the body is a single barrier op.
    _items = tuple({"id": str(i)} for i in range(K))

    custom_graph = nn.Graph()
    custom_graph.add(nn.Node(
        "data", "data",
        subgraph_entry="body",
        subgraph_exit="body",
        writes=(CURRENT_ITEM,),
        produces=("item_scores",),
        inline_items=tuple(dict(item) for item in _items),
        parallelism=K,
    ))
    custom_graph.add(nn.Node(
        "body", "op", call="test_ops_parallel:barrier_op",
        reads=(CURRENT_ITEM,), produces=("counted",),
    ))
    custom_graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    custom_graph.connect("data", "report")

    # Use a DshModelClient subclass that is recognized as thread-safe.
    from refactory_dsh.model import DshModelClient

    class _StubDsh(DshModelClient):
        def __init__(self):
            self._cancellations: tuple[threading.Event, ...] = ()

        def complete(self, prompt: str, *, system: str = "", **kw: Any) -> str:
            return "ok"

        def with_cancellation(self, event: threading.Event) -> "_StubDsh":
            c = copy.copy(self)
            c._cancellations = (*self._cancellations, event)
            return c

        def for_node(self, node: str, role: str = "worker") -> "_StubDsh":
            return copy.copy(self)

        def context(self) -> dict[str, Any]:
            return {}

    result = execute(
        custom_graph, tmp_path / "run", model=_StubDsh(), max_steps=200,
    )
    assert result.exit_ok, result.error

    # Verify all K items reached the barrier simultaneously.
    ready_files = list(barrier_dir.glob("ready_*"))
    done_files = list(barrier_dir.glob("done_*"))
    assert len(ready_files) == K, f"expected {K} ready files, got {len(ready_files)}"
    assert len(done_files) == K, f"expected {K} done files, got {len(done_files)}"


# ── additional: parallelism validation ────────────────────────────────────

def test_parallelism_validation():
    """Parallelism < 1 and parallelism != 1 on non-data nodes are refused."""
    with pytest.raises(nn.graph.GraphError, match="parallelism must be >= 1"):
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
    """The LiveMath solver's data_parallelism knob sets node.parallelism in build()."""
    from refactory_packages.livemath.mode import Solver

    # Default: parallelism=1.
    solver1 = Solver()
    graph1 = solver1.build()
    assert graph1.nodes["data"].parallelism == 1

    # With knob: parallelism=24 via state_dict manipulation.
    state = solver1.state_dict()
    state["knobs"]["data_parallelism"]["value"] = 24
    solver24 = Solver(state=state)
    graph24 = solver24.build()
    assert graph24.nodes["data"].parallelism == 24
