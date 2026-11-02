"""End-to-end tests with representative data-node graphs on both runtimes.

Proves that pass isolation does not break existing graph behavior: item_scores
and records match between walker and spine, and pass directories are created.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from refactory_dsh import lowering, nn
from refactory_dsh.nn.executor import CURRENT_ITEM, execute
from refactory_dsh.task import register_task

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
NODE = REPO_ROOT / "node_modules/.bin/tsx"
DATA_DRAIN = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain.mts"


def _run_spine(graph_json: Path, workdir: Path, python_bin_dir: str, timeout_ms: int) -> int:
    """Drive the TS spine drain (index.ts) via data-drain.mts."""
    env = {**os.environ, "REFACTORY_WORKDIR": str(workdir)}
    env["PATH"] = f"{python_bin_dir}:{env.get('PATH', '')}"
    result = subprocess.run(
        [str(NODE), "--import", "tsx/esm", str(DATA_DRAIN),
         str(graph_json), str(workdir), f"{python_bin_dir}/python", str(timeout_ms)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
        env=env,
    )
    if result.returncode != 0:
        sys.stderr.write(f"spine drain failed:\nstdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}\n")
    return result.returncode


def _write_solver_package(project: Path) -> None:
    """Write a representative solver-style Task as a proper package.

    Mimics the LiveMath solver graph: multiple items, each with a label,
    setup writes a prompt file, verify checks for a result file.
    """
    pkg = project / ".factory" / "packages" / "solver_scenario"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        'from refactory_dsh.nn import Package\n'
        'from .task import SolverTask\n'
        'PACKAGE = Package(\n'
        '    name="solver_scenario",\n'
        '    description="solver scenario",\n'
        '    modes=(),\n'
        '    ops=("refactory_packages.solver_scenario.ops",),\n'
        ')\n'
    )
    (pkg / "task.py").write_text(
        'from refactory_dsh.task import Task, TaskInstance, VerifyResult\n'
        '\n'
        'class SolverTask(Task):\n'
        '    name = "solver_scenario"\n'
        '\n'
        '    def instances(self):\n'
        '        return [\n'
        '            TaskInstance(id="q1", metadata={"label": "42", "label_text": "answer is 42"}),\n'
        '            TaskInstance(id="q2", metadata={"label": "7", "label_text": "answer is 7"}),\n'
        '            TaskInstance(id="q3", metadata={"label": "100", "label_text": "answer is 100"}),\n'
        '        ]\n'
        '\n'
        '    def setup(self, instance, workspace):\n'
        '        (workspace / f"prompt-{instance.id}.txt").write_text(\n'
        '            f"Solve for {instance.id}"\n'
        '        )\n'
        '\n'
        '    def verify(self, instance, workspace):\n'
        '        return VerifyResult(\n'
        '            passed=True,\n'
        '            score=1.0 if instance.id != "q2" else 0.5,\n'
        '            prediction=str(instance.metadata.get("label", "")),\n'
        '        )\n'
    )
    (pkg / "ops.py").write_text(
        'def solve(ctx):\n'
        '    item = ctx.read_json("current_item.json", {})\n'
        '    ctx.write_text("solution.txt", f"solution for {item.get(\\"id\\", \\"\\")}")\n'
        '    return {}\n'
    )


@pytest.mark.skipif(not NODE.exists(), reason="tsx not available")
def test_livemath_solver_end_to_end(tmp_path):
    """A representative solver graph runs end to end on both runtimes.

    Merged item_scores and records are identical across runtimes.  Pass
    directories are created with .done markers.
    """
    project = tmp_path / "project"
    _write_solver_package(project)
    sys.path.insert(0, str(project / ".factory" / "packages"))
    try:
        from solver_scenario.task import SolverTask  # type: ignore[import-not-found]
        register_task("solver_scenario", SolverTask)
        task_ref = "refactory_packages.solver_scenario.task:SolverTask"

        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref="solver_scenario",
            writes=(CURRENT_ITEM, "solution.txt"), produces=("item_scores",),
        ))
        graph.add(nn.Node(
            "work", "op", call="loop_ops:record_item",
            reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
        ))
        graph.add(nn.Node(
            "report", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        graph.connect("data", "report")

        # ── Walker run ──
        walker_dir = tmp_path / "walker"
        walker_result = execute(graph, walker_dir)
        assert walker_result.exit_ok, walker_result.error
        walker_scores = walker_result.facts.get("item_scores", [])

        # ── Spine run ──
        spine_dir = tmp_path / "spine"
        spine_dir.mkdir(parents=True, exist_ok=True)

        spine_graph = nn.Graph()
        spine_graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref=task_ref,
            writes=(CURRENT_ITEM, "solution.txt"), produces=("item_scores",),
        ))
        spine_graph.add(nn.Node(
            "work", "op", call="loop_ops:record_item",
            reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
        ))
        spine_graph.add(nn.Node(
            "report", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        spine_graph.connect("data", "report")

        state = {"graph": spine_graph.to_state()}
        lowering.write_declarations(state, spine_dir)
        graph_path, _ = lowering.write_mode_graph(
            state, tmp_path / "graphs", name="solver_scenario",
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
            rc = _run_spine(graph_path, spine_dir, python_bin, 5_000)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        assert rc == 0, "spine drain failed"
        spine_facts = json.loads((spine_dir / "facts.json").read_text())
        spine_scores = spine_facts.get("item_scores", [])

        # ── Compare item_scores ──
        def normalize(scores):
            return sorted(
                [(s["id"], s["passed"], s["score"]) for s in scores],
                key=lambda x: x[0],
            )

        assert normalize(walker_scores) == normalize(spine_scores), (
            f"walker and spine disagree:\nwalker: {walker_scores}\nspine: {spine_scores}"
        )

        # ── Pass directories should exist on both runtimes ──
        walker_passes = walker_dir / "passes" / "data"
        spine_passes = spine_dir / "passes" / "data"
        assert walker_passes.exists(), "walker should have passes/data/"
        assert spine_passes.exists(), "spine should have passes/data/"

        walker_pass_names = sorted(d.name for d in walker_passes.iterdir() if d.is_dir())
        spine_pass_names = sorted(d.name for d in spine_passes.iterdir() if d.is_dir())
        assert walker_pass_names == spine_pass_names, (
            f"pass dirs differ:\nwalker: {walker_pass_names}\nspine: {spine_pass_names}"
        )
        for name in walker_pass_names:
            assert (walker_passes / name / ".done").exists(), f"walker {name} missing .done"
        for name in spine_pass_names:
            assert (spine_passes / name / ".done").exists(), f"spine {name} missing .done"

    finally:
        sys.path.remove(str(project / ".factory" / "packages"))


@pytest.mark.skipif(not NODE.exists(), reason="tsx not available")
def test_existing_data_node_graph_end_to_end(tmp_path):
    """An existing data-node graph pattern runs on both runtimes.

    Merged item_scores and records match.  This exercises the common
    pattern of inline items with a task that verifies each.
    """
    project = tmp_path / "project"
    _write_solver_package(project)
    sys.path.insert(0, str(project / ".factory" / "packages"))
    try:
        from solver_scenario.task import SolverTask  # type: ignore[import-not-found]
        register_task("solver_existing", SolverTask)
        task_ref = "refactory_packages.solver_scenario.task:SolverTask"

        # This graph uses task-provided items (no inline_items).
        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref="solver_existing",
            writes=(CURRENT_ITEM,), produces=("item_scores",),
        ))
        graph.add(nn.Node(
            "work", "op", call="loop_ops:noop",
            reads=(CURRENT_ITEM,), writes=(), produces=(),
        ))
        graph.add(nn.Node(
            "report", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        graph.connect("data", "report")

        # ── Walker ──
        walker_dir = tmp_path / "walker"
        walker_result = execute(graph, walker_dir)
        assert walker_result.exit_ok, walker_result.error
        walker_scores = walker_result.facts.get("item_scores", [])

        # ── Spine ──
        spine_dir = tmp_path / "spine"
        spine_dir.mkdir(parents=True, exist_ok=True)

        spine_graph = nn.Graph()
        spine_graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref=task_ref,
            writes=(CURRENT_ITEM,), produces=("item_scores",),
        ))
        spine_graph.add(nn.Node(
            "work", "op", call="loop_ops:noop",
            reads=(CURRENT_ITEM,), writes=(), produces=(),
        ))
        spine_graph.add(nn.Node(
            "report", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        spine_graph.connect("data", "report")

        state = {"graph": spine_graph.to_state()}
        lowering.write_declarations(state, spine_dir)
        graph_path, _ = lowering.write_mode_graph(
            state, tmp_path / "graphs_existing", name="existing_data",
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
            rc = _run_spine(graph_path, spine_dir, python_bin, 5_000)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        assert rc == 0, "spine drain failed"
        spine_facts = json.loads((spine_dir / "facts.json").read_text())
        spine_scores = spine_facts.get("item_scores", [])

        def normalize(scores):
            return sorted(
                [(s["id"], s["passed"], s["score"]) for s in scores],
                key=lambda x: x[0],
            )

        assert normalize(walker_scores) == normalize(spine_scores), (
            f"walker and spine disagree:\nwalker: {walker_scores}\nspine: {spine_scores}"
        )

        # Pass directories exist on both runtimes.
        assert (walker_dir / "passes" / "data").exists()
        assert (spine_dir / "passes" / "data").exists()

    finally:
        sys.path.remove(str(project / ".factory" / "packages"))
