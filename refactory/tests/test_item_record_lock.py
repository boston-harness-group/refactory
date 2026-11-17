"""Per-item records stay whole when a fork's parallel branches write them at once.

A fork inside a data pass runs its branches in one pass folder; on the spine each
branch is its own process, and each appends its model calls (`items.record_exchange`)
and op visits (`items.record_visit`) to the pass's ``.refactory/item_exchanges.json``.
Without a lock two appends read the same list and the later write drops the other's
entry, and a reader that catches a half-written file starts over from nothing.
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import items
from refactory_dsh.nn.executor import CURRENT_ITEM

import test_data_parallel as dp

WRITERS = 5
APPENDS = 20
EARLIER = 3


def _append_many(workdir: str, writer: int, start: Any) -> None:
    start.wait()
    for n in range(APPENDS):
        if n % 2:
            items.record_visit(Path(workdir), {"node": f"w{writer}", "kind": "op", "output": f"{n}" * 2000})
        else:
            items.record_exchange(Path(workdir), f"w{writer}", f"prompt {n}", "x" * 2000)


@pytest.mark.parametrize("trial", range(3))
def test_parallel_appends_keep_every_entry(tmp_path, trial):
    """Five processes append at once after three earlier visits: none is lost."""
    for n in range(EARLIER):
        items.record_visit(tmp_path, {"node": f"early{n}", "kind": "op", "output": "e"})
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    workers = [context.Process(target=_append_many, args=(str(tmp_path), w, start)) for w in range(WRITERS)]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(60)
    assert all(worker.exitcode == 0 for worker in workers), [w.exitcode for w in workers]
    kept = json.loads((tmp_path / items.EXCHANGES_SIDECAR).read_text())
    assert [e["node"] for e in kept[:EARLIER]] == [f"early{n}" for n in range(EARLIER)]
    for w in range(WRITERS):
        mine = [e for e in kept if e["node"] == f"w{w}"]
        assert len(mine) == APPENDS, f"writer {w} kept {len(mine)} of {APPENDS}"
    assert len(kept) == EARLIER + WRITERS * APPENDS


def test_unreadable_sidecar_is_refused_not_overwritten(tmp_path):
    """A record file that cannot be read is never taken for an empty one."""
    path = tmp_path / items.EXCHANGES_SIDECAR
    path.parent.mkdir(parents=True)
    path.write_text('[{"node": "a"')
    with pytest.raises(ValueError, match="refusing to overwrite"):
        items.record_exchange(tmp_path, "b", "p", "r")
    assert path.read_text() == '[{"node": "a"'
    assert items._read_and_clear_exchanges(tmp_path) == []  # noqa: SLF001
    assert (path.with_name(f"{path.name}.unreadable")).read_text() == '[{"node": "a"'


# ── end to end: an S4-like fork (precision, hypotheses) in a data pass ──

ITEMS = 10
BRANCH_OPS = ("branch_a", "branch_b", "branch_c", "branch_d")
AGENTS = ("precision", "hypotheses")


def _s4_graph(parallelism: int) -> Any:
    """data -> [d_entry -> fork1 -> (4 op branches, 2 agent branches) -> join1 -> d_exit]."""
    from refactory_dsh import nn

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="d_entry", subgraph_exit="d_exit",
        writes=(CURRENT_ITEM,), produces=("item_scores",), parallelism=parallelism,
        task_ref="loop_ops:ForkBranchTaskTen",
    ))
    graph.add(nn.Node("d_entry", "op", call="loop_ops:noop"))
    graph.add(nn.Node("fork1", "fork", targets=(*BRANCH_OPS, *AGENTS)))
    graph.add(nn.Node("branch_a", "op", call="loop_ops:fork_branch_a",
                      reads=(CURRENT_ITEM,), writes=("a_out.txt",)))
    graph.add(nn.Node("branch_b", "op", call="loop_ops:fork_branch_b",
                      reads=(CURRENT_ITEM,), writes=("b_out.txt",)))
    graph.add(nn.Node("branch_c", "op", call="loop_ops:fork_branch_c", writes=("c_out.txt",)))
    graph.add(nn.Node("branch_d", "op", call="loop_ops:fork_branch_d", writes=("d_out.txt",)))
    for agent in AGENTS:
        graph.add(nn.Node(agent, "agent", prompt_template=f"{agent} please", writes=(f"{agent}.txt",)))
    graph.add(nn.Node("join1", "join", sources=(*BRANCH_OPS, *AGENTS)))
    graph.add(nn.Node("d_exit", "op", call="loop_ops:score_from_branches",
                      reads=("a_out.txt", "b_out.txt"), writes=("item_result.json",)))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("d_entry", "fork1")
    graph.connect("fork1", "join1")
    for branch in (*BRANCH_OPS, *AGENTS):
        graph.connect(branch, "join1")
    graph.connect("join1", "d_exit")
    graph.connect("data", "report")
    return graph


def _assert_whole(records: list[dict[str, Any]]) -> None:
    assert sorted(str(r["id"]) for r in records) == [str(i) for i in range(ITEMS)]
    broken = {}
    for record in records:
        visited = sorted(step["node"] for step in record.get("steps") or ())
        called = sorted(call["node"] for call in record.get("exchanges") or ())
        if visited != sorted(("d_entry", *BRANCH_OPS, "d_exit")) or called != sorted(AGENTS):
            broken[record["id"]] = {"steps": visited, "exchanges": called}
    assert not broken, f"items lost record entries: {broken}"


@pytest.mark.parametrize("parallelism", [1, 3])
def test_s4_fork_records_whole_walker(tmp_path, parallelism):
    """[walker] Every item keeps every visit and every branch call."""
    result = dp._walker(_s4_graph(parallelism), tmp_path / "run", model=dp._SimpleFakeModel())  # noqa: SLF001
    assert result.exit_ok, result.error
    _assert_whole(result.item_records)


#: A dispatched agent, as the spine runs a worker (``dispatch_command``): writes the
#: node's answer to its outputs in the item's pass folder and records the call, as the
#: real worker's delivery does (`worker_node` -> `node_retry.deliver` ->
#: `agent_delivery.finish` -> `items.record_exchange`).
_FAKE_WORKER = """
import json, os, sys
from pathlib import Path
from refactory_dsh import items
node, writes = sys.argv[1:3]
workdir = Path(os.environ["REFACTORY_WORKDIR"])
for name in json.loads(writes):
    target = workdir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{node} answer")
# A long reply, as a model's is: the file takes a while to write.
items.record_exchange(workdir, node, f"{node} please", f"{node} answer " * 40000)
"""


def _spine_run(graph: Any, run: Path, tmp_path: Path) -> int:
    """Lower ``graph`` with its agents dispatched to a fake worker, so every body node
    is one the spine runs itself and a pass's fork peers run at once, and drain it."""
    import os
    import shlex
    import subprocess
    import sys

    from refactory_dsh import lowering

    worker = tmp_path / "fake_worker.py"
    worker.write_text(_FAKE_WORKER)

    def dispatch(workflow: Any) -> Any:
        for node in workflow.nodes.values():
            if node.id in AGENTS:
                node.dispatch_command = " ".join(shlex.quote(part) for part in (
                    f"{dp._PYTHON_BIN}/python", str(worker), node.id,  # noqa: SLF001
                    json.dumps(sorted(node.writes))))
                node.command_timeout_s = 60
        return workflow

    run.mkdir(parents=True)
    state = {"graph": graph.to_state()}
    lowering.write_declarations(state, run)
    graph_path, _ = lowering.write_mode_graph(state, tmp_path / "graphs", name="s4_like",
                                              transform=dispatch)
    env = {**os.environ, "REFACTORY_WORKDIR": str(run), "PYTHONPATH": str(dp.HERE)}
    env["PATH"] = f"{dp._PYTHON_BIN}:{env.get('PATH', '')}"  # noqa: SLF001
    done = subprocess.run(
        [str(dp._NODE), "--import", "tsx/esm", str(dp._DRAIN_AGENT), str(graph_path), str(run),  # noqa: SLF001
         f"{dp._PYTHON_BIN}/python", "60000"],  # noqa: SLF001
        capture_output=True, text=True, cwd=str(dp.REPO_ROOT), timeout=300, env=env)
    if done.returncode != 0:
        sys.stderr.write(f"spine drain: rc={done.returncode}\nstdout: {done.stdout[-2000:]}\n"
                         f"stderr: {done.stderr[-3000:]}\n")
    return done.returncode


@dp._needs_spine  # noqa: SLF001
@pytest.mark.parametrize("parallelism", [1, 3])
def test_s4_fork_records_whole_spine(tmp_path, parallelism):
    """[spine] A pass's fork peers run at once, each its own process writing the pass's
    record file; every item keeps every visit and every branch call."""
    run = tmp_path / "run"
    assert _spine_run(_s4_graph(parallelism), run, tmp_path) == 0
    _assert_whole(json.loads((run / items.RECORDS_FILE).read_text()))
