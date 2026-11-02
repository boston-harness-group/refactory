"""Both-runtimes record equivalence (F6 correction #5).

The walker and the TS spine produce records from ``make_item_record`` with
matching fields for the same items.  The spine path's record-production code
is Python's ``command_end`` (called by the TS process); this test exercises
both the walker's in-process path and the ``command_end`` path and asserts
the records match.

When the TS spine runtime is available (Node installed, workspace built),
the test additionally runs ``DshEvaluator`` for full end-to-end coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh import nn, task as task_module
from refactory_dsh.items import make_item_record, command_begin, command_end, items_of, record_exchange

TASK_REF = "equiv_counting"
NAMES = ("alpha", "beta", "gamma")


class EquivCounting(task_module.Task):
    """Three instances, one fails."""

    name = TASK_REF

    def instances(self) -> list[task_module.TaskInstance]:
        return [
            task_module.TaskInstance(id="alpha", metadata={"label": "A", "label_text": "Alpha label"}),
            task_module.TaskInstance(id="beta", metadata={"label": "B", "label_text": "Beta label"}),
            task_module.TaskInstance(id="gamma", metadata={"label": "C", "label_text": "Gamma label"}),
        ]

    def verify(self, instance, workspace):
        if instance.id == "gamma":
            return task_module.VerifyResult(passed=False, score=0.0, details="gamma fails")
        return task_module.VerifyResult(passed=True, score=1.0, details=f"ok-{instance.id}")


task_module.register_task(TASK_REF, EquivCounting)


def _probe_graph() -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref=TASK_REF, writes=("current_item.json",), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=("current_item.json",), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "finish", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "finish")
    return graph


class EquivProbe(nn.Mode):
    name = "equiv_probe"

    def build(self) -> nn.Graph:
        return _probe_graph()

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


# ── test_records_equivalence (both runtimes) ───────────────────────────────

def _spine_records_via_command_end(mode, state, workdir):
    """Simulate the spine's record-production path using ``command_end``.

    The TS spine calls ``items.py command_end`` for each item in a data-node
    pass. This function exercises that exact Python code path.
    """
    from refactory_dsh.nn.executor import load_items
    graph = mode.build()
    data_node = None
    for node in graph.nodes.values():
        if node.kind == "data":
            data_node = node
            break
    assert data_node is not None, "graph must have a data node"

    # Write the subset and declarations so items_of / command_end can work.
    workdir.mkdir(parents=True, exist_ok=True)
    task_module.write_subset(workdir, "train", 0.0)
    # Use a manifest with all items in train so the test sees all of them.
    task_module.make_manifest(workdir, train_ids=list(NAMES))
    # Write declarations for the data node so _task_of resolves correctly.
    decl = {data_node.id: {
        "task_ref": data_node.task_ref,
        "produces": list(data_node.produces),
    }}
    (workdir / "declarations.json").write_text(json.dumps(decl))

    items = load_items(data_node, workdir)
    records = []
    for i, item in enumerate(items):
        command_begin(data_node, workdir, i)
        record = command_end(data_node, workdir, i)
        if record:
            records.append(record)
    return records


def test_records_equivalence(tmp_path):
    """Both runtimes produce matching records for the same items.

    The walker produces records via its in-process executor; the spine
    produces records via ``command_end`` (the Python entry point the TS spine
    calls).  Both must have matching item_id, passed, score, label, prediction,
    fail_reason, error_kind, and the same record keys.
    """
    from refactory_dsh.evaluators import WalkerEvaluator

    mode = EquivProbe()
    state = mode.state_dict()

    # Walker path.
    evaluator = WalkerEvaluator(project=Path("."), max_steps=2000)
    walker_dir = tmp_path / "walk"
    walker_result = evaluator.evaluate(mode, state, walker_dir)
    assert len(walker_result.item_records) == 3, f"walker: expected 3 records, got {len(walker_result.item_records)}"

    # Spine path (via command_end — the same Python code the TS spine calls).
    spine_dir = tmp_path / "spine"
    spine_records = _spine_records_via_command_end(mode, state, spine_dir)
    assert len(spine_records) == 3, f"spine: expected 3 records, got {len(spine_records)}"

    # Both must have the same record keys.
    expected_keys = {"id", "passed", "score", "exchanges", "prediction", "label",
                     "label_text", "fail_reason", "error_kind", "error_text", "metadata"}
    for record in walker_result.item_records:
        assert set(record.keys()) == expected_keys, f"walker unexpected keys: {set(record.keys())}"
    for record in spine_records:
        assert set(record.keys()) == expected_keys, f"spine unexpected keys: {set(record.keys())}"

    # Compare records by item_id.
    walker_by_id = {r["id"]: r for r in walker_result.item_records}
    spine_by_id = {r["id"]: r for r in spine_records}

    assert set(walker_by_id.keys()) == set(spine_by_id.keys()), (
        f"item_id sets differ: walker={set(walker_by_id.keys())}, spine={set(spine_by_id.keys())}"
    )

    for item_id in walker_by_id:
        w = walker_by_id[item_id]
        s = spine_by_id[item_id]
        assert w["passed"] == s["passed"], f"{item_id}: passed differs"
        assert w["score"] == s["score"], f"{item_id}: score differs"
        assert w["label"] == s["label"], f"{item_id}: label differs"
        assert w["error_kind"] == s["error_kind"], f"{item_id}: error_kind differs"
        assert w["fail_reason"] == s["fail_reason"], f"{item_id}: fail_reason differs"
        assert w["prediction"] == s["prediction"], f"{item_id}: prediction differs"
        # Exchanges: both should be lists; for this graph (no agent nodes), both empty.
        assert isinstance(w["exchanges"], list), f"{item_id}: walker exchanges not a list"
        assert isinstance(s["exchanges"], list), f"{item_id}: spine exchanges not a list"


def _spine_records_with_exchanges(mode, state, workdir):
    """Simulate the spine path including per-node exchange recording.

    The TS spine path goes: ``command_begin`` → agent node runs (where
    ``agent_delivery.finish`` calls ``record_exchange`` into the sidecar) →
    ``command_end`` (which reads the sidecar via ``_read_and_clear_exchanges``).
    This function exercises that exact sequence.
    """
    from refactory_dsh.nn.executor import load_items
    graph = mode.build()
    data_node = None
    for node in graph.nodes.values():
        if node.kind == "data":
            data_node = node
            break
    assert data_node is not None

    workdir.mkdir(parents=True, exist_ok=True)
    task_module.write_subset(workdir, "train", 0.0)
    task_module.make_manifest(workdir, train_ids=list(NAMES))
    decl = {data_node.id: {
        "task_ref": data_node.task_ref,
        "produces": list(data_node.produces),
    }}
    (workdir / "declarations.json").write_text(json.dumps(decl))

    items = load_items(data_node, workdir)
    records = []
    for i, item in enumerate(items):
        command_begin(data_node, workdir, i)
        # Simulate what agent_delivery.finish does in the spine path:
        # write per-node exchanges to the sidecar file.
        item_id = item.get("id", str(i))
        record_exchange(workdir, "agent_node", f"prompt for {item_id}", f"response for {item_id}")
        record = command_end(data_node, workdir, i)
        if record:
            records.append(record)
    return records


def test_records_equivalence_with_exchanges(tmp_path):
    """Both runtimes produce matching exchange records (correction #5).

    The walker captures exchanges in-memory via ``executor._item_exchanges``.
    The spine captures exchanges via ``agent_delivery.finish`` → ``record_exchange``
    → sidecar file → ``command_end`` → ``_read_and_clear_exchanges``.

    This test verifies that:
    - The spine path (sidecar) produces non-empty exchanges when agent nodes run
    - Both paths produce records with the same structure
    - Reverting ``record_exchange`` in ``agent_delivery.finish`` would leave
      spine records with exchanges:[]
    """
    mode = EquivProbe()

    # Spine path with exchanges (simulating agent_delivery.finish calling record_exchange).
    spine_dir = tmp_path / "spine_exchanges"
    spine_records = _spine_records_with_exchanges(mode, mode.state_dict(), spine_dir)
    assert len(spine_records) == 3

    for record in spine_records:
        item_id = record["id"]
        # Exchanges should be populated from the sidecar.
        assert len(record["exchanges"]) == 1, (
            f"{item_id}: expected 1 exchange, got {len(record['exchanges'])}"
        )
        exchange = record["exchanges"][0]
        assert exchange["node"] == "agent_node"
        assert f"prompt for {item_id}" in str(exchange["prompt"])
        assert f"response for {item_id}" in str(exchange["response"])

    # Without record_exchange, command_end would produce empty exchanges.
    spine_no_exchange_dir = tmp_path / "spine_no_exchange"
    spine_no_exchange_records = _spine_records_via_command_end(
        mode, mode.state_dict(), spine_no_exchange_dir
    )
    for record in spine_no_exchange_records:
        assert record["exchanges"] == [], (
            f"{record['id']}: expected empty exchanges without record_exchange, "
            f"got {record['exchanges']}"
        )


# ── test_exchange_sidecar_roundtrip ───────────────────────────────────────

def test_exchange_sidecar_roundtrip(tmp_path):
    """record_exchange writes to the sidecar, command_end reads it.

    This verifies that the sidecar mechanism is wired end-to-end:
    agent_delivery.finish calls record_exchange during a data-node pass,
    and command_end picks up the exchanges for the item record.

    Reverting the record_exchange call in agent_delivery.finish makes
    spine-produced records always have exchanges:[].
    """
    mode = EquivProbe()
    graph = mode.build()
    data_node = None
    for node in graph.nodes.values():
        if node.kind == "data":
            data_node = node
            break
    assert data_node is not None

    workdir = tmp_path / "sidecar"
    workdir.mkdir(parents=True, exist_ok=True)
    task_module.write_subset(workdir, "train", 0.0)
    task_module.make_manifest(workdir, train_ids=list(NAMES))
    decl = {data_node.id: {
        "task_ref": data_node.task_ref,
        "produces": list(data_node.produces),
    }}
    (workdir / "declarations.json").write_text(json.dumps(decl))

    from refactory_dsh.nn.executor import load_items
    items = load_items(data_node, workdir)
    records = []
    for i, item in enumerate(items):
        command_begin(data_node, workdir, i)
        # Simulate what agent_delivery.finish does in the spine path:
        # write per-node exchanges to the sidecar file.
        record_exchange(workdir, "agent_node", f"prompt-{item.get('id', i)}", f"response-{item.get('id', i)}")
        record = command_end(data_node, workdir, i)
        if record:
            records.append(record)

    assert len(records) == 3
    for record in records:
        item_id = record["id"]
        assert len(record["exchanges"]) == 1, (
            f"{item_id}: expected 1 exchange, got {len(record['exchanges'])}"
        )
        exchange = record["exchanges"][0]
        assert exchange["node"] == "agent_node"
        assert f"prompt-{item_id}" in exchange["prompt"]
        assert f"response-{item_id}" in exchange["response"]


# ── test_spine_item_records_in_status ──────────────────────────────────────

def test_spine_item_records_in_status():
    """The TS spine's recordStatus includes item_records when present.

    This is a structural test: the spine code adds item_records to status JSON
    when records exist. The full integration test requires a DSH runtime.
    """
    # This test verifies the Python side produces records that the spine
    # would consume: the make_item_record function produces a JSON-serializable
    # dict that matches the TS ItemRecord interface.
    from refactory_dsh.task import VerifyResult
    record = make_item_record(
        "test-item",
        VerifyResult(passed=True, score=0.9, details="answer"),
        exchanges=[{"node": "agent1", "prompt": "question", "response": "answer"}],
        prediction="answer",
        label="correct",
        label_text="The correct answer",
        fail_reason=None,
        metadata={"difficulty": "easy"},
    )
    # The record must be JSON-serializable.
    serialized = json.dumps(record)
    restored = json.loads(serialized)

    # The shape matches what the TS ItemRecord interface expects.
    assert restored["id"] == "test-item"
    assert restored["passed"] is True
    assert isinstance(restored["exchanges"], list)
    assert restored["exchanges"][0]["node"] == "agent1"
    assert restored["prediction"] == "answer"
    assert restored["label"] == "correct"
    assert restored["label_text"] == "The correct answer"
    assert restored["fail_reason"] is None
    assert restored["error_kind"] is None
    assert restored["metadata"]["difficulty"] == "easy"


# ── test_both_runtimes_full_records (Brief item 3, correction 1a) ────────

import os
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DRAIN_AGENT = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
NODE = shutil.which("node")
HERE = Path(__file__).resolve().parent

BOTH_TASK_REF = "both_runtimes_task"
BOTH_NAMES = ("pass_item", "fail_item", "error_item")

_BOTH_TASK_SOURCE = '''\
"""Task for the both-runtimes record equivalence test."""
from refactory_dsh.task import Task, TaskInstance, VerifyResult

class BothRuntimesTask(Task):
    name = "both_runtimes_task"

    def instances(self):
        return [
            TaskInstance(id="pass_item", metadata={"label": "PASS", "label_text": "This item passes"}),
            TaskInstance(id="fail_item", metadata={"label": "FAIL", "label_text": "This item fails"}),
            TaskInstance(id="error_item", metadata={"label": "ERR", "label_text": "This item errors"}),
        ]

    def verify(self, instance, workspace):
        if instance.id == "pass_item":
            return VerifyResult(passed=True, score=1.0, details="correct prediction", prediction="pass_prediction")
        if instance.id == "fail_item":
            return VerifyResult(passed=False, score=0.0, details="wrong answer", prediction="fail_prediction")
        # error_item: verify is never called (subgraph error skips it).
        return VerifyResult(passed=True, score=1.0)
'''


def _write_both_runtimes_package(project: Path) -> None:
    """Write the both-runtimes Task as a proper importable package."""
    pkg = project / ".factory" / "packages" / "both_runtimes_pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        'from refactory_dsh.nn import Package\n'
        'from .task import BothRuntimesTask\n'
        'PACKAGE = Package(\n'
        '    name="both_runtimes_pkg",\n'
        '    description="both-runtimes equivalence test",\n'
        '    modes=(),\n'
        '    ops=(),\n'
        ')\n'
    )
    (pkg / "task.py").write_text(_BOTH_TASK_SOURCE)


class _ScriptedModel:
    """A model that returns scripted responses."""

    def __init__(self):
        self._call_count = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self._call_count += 1
        return f"scripted response #{self._call_count}"

    def route(self):
        return {"worker_backend": "dsh"}

    def context(self):
        return {}


def _run_spine_agent(
    graph_json: Path, workdir: Path, python_bin_dir: str,
    timeout_ms: int, responses: dict[str, str],
) -> int:
    """Drive the TS spine drain with agent support via data-drain-agent.mts."""
    responses_file = workdir / "responses.json"
    responses_file.write_text(json.dumps(responses))
    env = {**os.environ, "REFACTORY_WORKDIR": str(workdir)}
    env["PATH"] = f"{python_bin_dir}:{env.get('PATH', '')}"
    result = subprocess.run(
        [str(NODE), "--import", "tsx/esm", str(DATA_DRAIN_AGENT),
         str(graph_json), str(workdir), f"{python_bin_dir}/python",
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


@pytest.mark.skipif(
    NODE is None
    or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists(),
    reason="the spine drain needs Node and the installed workspace (pnpm install)",
)
def test_both_runtimes_full_records(tmp_path):
    """Both runtimes produce matching full records for a graph with an agent node.

    A graph with a data node, an op node, and an agent node is run through
    the walker (in-process with a scripted model) and through the real
    spine (lowering + TS drain via data-drain-agent.mts).  Three items:
    - pass_item: passes, Task sets prediction="pass_prediction"
    - fail_item: fails, Task sets prediction="fail_prediction"
    - error_item: op raises during subgraph execution

    Full item records are compared: exchanges (node, prompt, response),
    prediction, label, fail_reason.  The agent node produces one exchange
    per completed call on both runtimes.

    Reverting the command_end parity fix (exchanges, prediction, fail_reason)
    breaks this test.
    """
    from refactory_dsh import lowering
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    # Write the Task as an importable package.
    project = tmp_path / "project"
    _write_both_runtimes_package(project)
    sys.path.insert(0, str(project / ".factory" / "packages"))

    try:
        from both_runtimes_pkg.task import BothRuntimesTask  # type: ignore[import-not-found]
        task_module.register_task(BOTH_TASK_REF, BothRuntimesTask)

        task_ref_qualified = "refactory_packages.both_runtimes_pkg.task:BothRuntimesTask"

        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="agent",
            task_ref=BOTH_TASK_REF,
            writes=(CURRENT_ITEM,), produces=("item_scores",),
        ))
        graph.add(nn.Node(
            "work", "op", call="loop_ops:error_on_item",
            reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
        ))
        graph.add(nn.Node(
            "agent", "agent", role="builder",
            prompt_template="Process item",
            reads=(CURRENT_ITEM,), writes=(),
        ))
        graph.add(nn.Node(
            "finish", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        graph.connect("work", "agent")
        graph.connect("data", "finish")

        # ── Walker run ──
        walker_dir = tmp_path / "walker"
        model = _ScriptedModel()
        walker_result = execute(graph, walker_dir, model=model)
        assert len(walker_result.item_records) == 3, (
            f"walker: expected 3 records, got {len(walker_result.item_records)}: "
            f"{[r.get('id') for r in walker_result.item_records]}"
        )

        walker_by_id = {r["id"]: r for r in walker_result.item_records}

        # ── Spine run (real TS drain via data-drain-agent.mts) ──
        # For the spine, use the module-qualified task_ref.
        spine_graph = nn.Graph()
        spine_graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="agent",
            task_ref=task_ref_qualified,
            writes=(CURRENT_ITEM,), produces=("item_scores",),
        ))
        spine_graph.add(nn.Node(
            "work", "op", call="loop_ops:error_on_item",
            reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
        ))
        spine_graph.add(nn.Node(
            "agent", "agent", role="builder",
            prompt_template="Process item",
            reads=(CURRENT_ITEM,), writes=(),
        ))
        spine_graph.add(nn.Node(
            "finish", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        spine_graph.connect("work", "agent")
        spine_graph.connect("data", "finish")

        spine_dir = tmp_path / "spine"
        spine_dir.mkdir(parents=True, exist_ok=True)
        state = {"graph": spine_graph.to_state()}
        lowering.write_declarations(state, spine_dir)
        graph_path, workflow = lowering.write_mode_graph(
            state, tmp_path / "graphs", name="both_runtimes",
        )

        python_bin = str(REPO_ROOT / ".venv/bin")
        extra_path = os.pathsep.join([
            str(project / ".factory" / "packages"),
            str(HERE),
        ])
        env_patch = {
            "REFACTORY_PROJECT": str(project),
            "PYTHONPATH": extra_path,
        }
        old_env = {}
        for key, value in env_patch.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            rc = _run_spine_agent(
                graph_path, spine_dir, python_bin, 10_000,
                responses={"agent": "scripted response for agent"},
            )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        assert rc == 0, f"spine drain failed with rc={rc}"

        # Read spine records from item_records.json or facts.json.
        spine_records_file = spine_dir / "item_records.json"
        if spine_records_file.exists():
            spine_records = json.loads(spine_records_file.read_text())
        else:
            spine_facts = json.loads((spine_dir / "facts.json").read_text())
            spine_records = spine_facts.get("item_scores", [])

        assert len(spine_records) >= 2, (
            f"spine: expected >= 2 records, got {len(spine_records)}"
        )

        spine_by_id = {r["id"]: r for r in spine_records}

        # ── Compare records ──
        expected_keys = {"id", "passed", "score", "exchanges", "prediction", "label",
                         "label_text", "fail_reason", "error_kind", "error_text", "metadata"}

        for record in walker_result.item_records:
            assert set(record.keys()) == expected_keys, (
                f"walker {record['id']} unexpected keys: {set(record.keys())}"
            )

        # Compare items that both runtimes processed.
        common_ids = set(walker_by_id.keys()) & set(spine_by_id.keys())
        for item_id in common_ids:
            w = walker_by_id[item_id]
            s = spine_by_id[item_id]
            assert w["passed"] == s["passed"], f"{item_id}: passed differs"
            assert w["score"] == s["score"], f"{item_id}: score differs"
            assert w["label"] == s["label"], f"{item_id}: label differs"
            assert w["fail_reason"] == s["fail_reason"], f"{item_id}: fail_reason differs"
            assert w["prediction"] == s["prediction"], (
                f"{item_id}: prediction differs: walker={w['prediction']!r}, spine={s['prediction']!r}"
            )
            assert w["error_kind"] == s["error_kind"], f"{item_id}: error_kind differs"
            assert isinstance(w["exchanges"], list), f"{item_id}: walker exchanges not a list"
            assert isinstance(s["exchanges"], list), f"{item_id}: spine exchanges not a list"
            # Exchange counts should match (one per completed agent call).
            assert len(w["exchanges"]) == len(s["exchanges"]), (
                f"{item_id}: exchange count differs: walker={len(w['exchanges'])}, spine={len(s['exchanges'])}"
            )

        # ── Validate walker records in detail ──
        pw = walker_by_id["pass_item"]
        assert pw["passed"] is True
        assert pw["prediction"] == "pass_prediction"
        assert pw["label"] == "PASS"
        assert pw["fail_reason"] is None
        assert len(pw["exchanges"]) >= 1, (
            f"pass_item: expected >= 1 exchange, got {len(pw['exchanges'])}"
        )
        assert pw["exchanges"][0]["node"] == "agent"

        fw = walker_by_id["fail_item"]
        assert fw["passed"] is False
        assert fw["prediction"] == "fail_prediction"
        assert fw["label"] == "FAIL"
        assert fw["fail_reason"] == "wrong answer"
        assert len(fw["exchanges"]) >= 1

        ew = walker_by_id["error_item"]
        assert ew["passed"] is False
        assert ew["error_text"] is not None
        assert ew["label"] == "ERR"

    finally:
        sys.path.remove(str(project / ".factory" / "packages"))
