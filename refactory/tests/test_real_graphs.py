"""End-to-end tests with the real LiveMath package on both runtimes.

Proves that pass isolation does not break existing graph behavior: item_scores
and records match between walker and spine, and pass directories are created.

Uses the actual LiveMath package (refactory_packages/livemath) with the
fixture data, not a custom representative solver package.
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
FIXTURE_DIR = HERE / "fixtures" / "livemath"


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


@pytest.mark.skipif(not NODE.exists(), reason="tsx not available")
def test_livemath_solver_end_to_end(tmp_path):
    """The real LiveMath solver graph runs end to end on both runtimes.

    Uses the actual LiveMath package (refactory_packages/livemath) with the
    fixture data.  Walker uses FakeModel; spine uses a deterministic graph
    variant with the same LiveMathTask (same items, same verify/grading).

    Merged item_scores are identical across runtimes.  Pass directories are
    created with .done markers.
    """
    import refactory_packages.livemath.tasks as tasks_mod
    from refactory_packages.livemath.mode import Solver
    from refactory_packages.livemath.tasks import LiveMathTask

    # Patch LiveMathTask to use fixture data (no shuffling for reproducibility).
    original_init = tasks_mod.LiveMathTask.__init__

    def patched_init(self, data_path=None, shuffle_choices=True):
        original_init(self, data_path=str(FIXTURE_DIR), shuffle_choices=False)

    tasks_mod.LiveMathTask.__init__ = patched_init  # type: ignore[method-assign]
    try:
        register_task("livemath", LiveMathTask)

        # ── Walker run: real LiveMath Solver mode + FakeModel ──
        from tests_support import FakeModel
        model = FakeModel(["<answer>B</answer>"] * 10)
        mode = Solver()
        walker_dir = tmp_path / "walker"
        walker_result = mode.run(walker_dir, model=model)
        assert walker_result.exit_ok, f"walker failed: {walker_result.error}"
        walker_scores = walker_result.facts.get("item_scores", [])

        # ── Spine run: real LiveMathTask + deterministic graph variant ──
        # The spine data drain can only run deterministic (op/gate) nodes, so
        # the agent "solve" node is replaced by an op that writes the same
        # scripted response ("<answer>B</answer>") to answer.txt.
        # This exercises the REAL LiveMathTask (instances, setup, verify) and
        # the real graph structure through the spine drain.
        spine_dir = tmp_path / "spine"
        spine_dir.mkdir(parents=True, exist_ok=True)

        OPS = "refactory_packages.livemath.ops"
        TASK = "refactory_packages.livemath.tasks:LiveMathTask"

        spine_graph = nn.Graph()
        spine_graph.add(nn.Node(
            "data", "data",
            subgraph_entry="format_prompt",
            subgraph_exit="solve",
            task_ref=TASK,
            writes=(CURRENT_ITEM,),
            produces=("item_scores",),
        ))
        spine_graph.add(nn.Node(
            "format_prompt", "op",
            call=f"{OPS}:format_prompt",
            reads=(CURRENT_ITEM,),
            knobs=("skill", "use_theorem", "use_sketch"),
            produces=("user_prompt", "skill_section"),
        ))
        # Deterministic op replacing the agent node — writes the same response
        # FakeModel would give so the LiveMathTask's verify produces matching scores.
        spine_graph.add(nn.Node(
            "solve", "op",
            call="loop_ops:write_scripted_answer",
            writes=("answer.txt",),
            produces=(),
        ))
        spine_graph.add(nn.Node(
            "finish", "op",
            call=f"{OPS}:finish",
            writes=("result.json",),
            produces=("finished",),
        ))
        spine_graph.connect("format_prompt", "solve")
        spine_graph.connect("data", "finish")

        state = {"graph": spine_graph.to_state()}
        lowering.write_declarations(state, spine_dir)
        graph_path, _ = lowering.write_mode_graph(
            state, tmp_path / "graphs", name="livemath_solver",
        )

        python_bin = str(REPO_ROOT / ".venv/bin")
        extra_path = os.pathsep.join([
            str(REPO_ROOT / "refactory" / "refactory_packages"),
            str(HERE),
        ])
        env_patch = {
            "LIVEMATH_DATA_PATH": str(FIXTURE_DIR),
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

        # Items with correct label B should score 1.0
        for scores_list in [walker_scores, spine_scores]:
            correct_b = {"202501:1", "202503:1", "202503:2"}
            for entry in scores_list:
                if entry["id"] in correct_b:
                    assert entry["score"] == 1.0, (
                        f"{entry['id']} should score 1.0: {entry}"
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
        tasks_mod.LiveMathTask.__init__ = original_init  # type: ignore[method-assign]


@pytest.mark.skipif(not NODE.exists(), reason="tsx not available")
def test_existing_data_node_graph_end_to_end(tmp_path):
    """An existing data-node graph pattern runs on both runtimes.

    Uses the real LiveMathTask with the fixture data but a minimal
    deterministic graph.  Merged item_scores match.
    """
    import refactory_packages.livemath.tasks as tasks_mod
    from refactory_packages.livemath.tasks import LiveMathTask

    original_init = tasks_mod.LiveMathTask.__init__

    def patched_init(self, data_path=None, shuffle_choices=True):
        original_init(self, data_path=str(FIXTURE_DIR), shuffle_choices=False)

    tasks_mod.LiveMathTask.__init__ = patched_init  # type: ignore[method-assign]
    try:
        register_task("livemath_existing", LiveMathTask)
        task_ref = "refactory_packages.livemath.tasks:LiveMathTask"

        # This graph uses the real LiveMathTask for items and verify,
        # with a noop subgraph and a scripted answer writer.
        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref="livemath_existing",
            writes=(CURRENT_ITEM,), produces=("item_scores",),
        ))
        graph.add(nn.Node(
            "work", "op", call="loop_ops:write_scripted_answer",
            reads=(CURRENT_ITEM,), writes=("answer.txt",), produces=(),
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
            "work", "op", call="loop_ops:write_scripted_answer",
            reads=(CURRENT_ITEM,), writes=("answer.txt",), produces=(),
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
            str(REPO_ROOT / "refactory" / "refactory_packages"),
            str(HERE),
        ])
        env_patch = {
            "LIVEMATH_DATA_PATH": str(FIXTURE_DIR),
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
        tasks_mod.LiveMathTask.__init__ = original_init  # type: ignore[method-assign]
