"""One mode, two runtimes, the same answer.

The DSH issue's gate. The Python walker is a debug runtime; DSH is the runtime
that ships. A debug runtime that disagrees with the real one is worse than none, so
these tests run the same mode both ways and compare what came out.

They are marked `integration` because they spawn a real session and make real model
calls: `pytest --run-integration`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from refactory_dsh import lowering, registry
from refactory_dsh.paths import REPO_ROOT, dsh_command, node_bin, venv_bin
from refactory_dsh.run import FACTS

MODE = "circle_packing/adaptive"

#: A deterministic mode needs the CEO only to call `wf_next`; the spine runs every
#: node itself because the bundle mounts it with `executeCommands: true`.
TASK = "Call wf_next repeatedly until it reports done. Do nothing else."

pytestmark = pytest.mark.integration


def _mode():
    _package, mode_class = registry.resolve(Path("."), MODE)
    return mode_class()


def _run_on_dsh(
    state: dict, tmp_path: Path, *, knobs: dict, name: str = "circle_packing", timeout: int = 600
) -> dict:
    """Lower the mode, write it where the bundle loads it, and run DSH."""
    graphs = tmp_path / "graphs"
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    lowering.write_mode_graph(state, graphs, name=name)
    # The IR has no field for the facts a node produces, so the runner reads them
    # from the run directory.
    lowering.write_declarations(state, work)

    result = subprocess.run(
        [
            *dsh_command(),
            "--profile", "graph", "run",
            "--mode", name,
            "--task", TASK,
            "--budget", "1",
            "--graphs", str(graphs),
            "--workdir", str(work),
            "--python-bin", str(venv_bin()),
            "--knobs", json.dumps(knobs),
        ],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=timeout,
    )
    payload = {}
    for line in reversed(result.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            payload = json.loads(result.stdout[result.stdout.index(line) :])
            break
    payload["_returncode"] = result.returncode
    payload["_stderr"] = result.stderr[-2000:]
    payload["_facts"] = (
        json.loads((work / FACTS).read_text()) if (work / FACTS).exists() else {}
    )
    payload["_work"] = str(work)
    return payload


def _run_with_the_walker(state: dict, tmp_path: Path) -> dict:
    mode = type(_mode())(state=state)
    result = mode.evaluate(tmp_path / "walk", max_steps=5000)
    return {
        "score": result.score,
        "exit_ok": result.exit_ok,
        "error": result.error,
        "facts": result.facts,
        "iterations": result.iterations,
    }


def test_the_spine_itself_accepts_the_lowered_graph(tmp_path):
    """The runtime's own validator, not just refactory's."""
    state = _mode().state_dict()
    graphs = tmp_path / "graphs"
    path, _workflow = lowering.write_mode_graph(state, graphs, name="circle_packing")
    spine = REPO_ROOT / "packages" / "workflow" / "workflow-spine" / "src" / "spine.ts"
    script = (
        "import { readFileSync } from 'node:fs';"
        f"import {{ validateGraph }} from {spine.as_uri()!r};"
        f"const g = JSON.parse(readFileSync({str(path)!r}, 'utf8'));"
        "const issues = validateGraph(g as never);"
        "console.log(JSON.stringify(issues));"
    )
    result = subprocess.run(
        [str(node_bin()), "--import", "tsx/esm", "-e", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    issues = json.loads(result.stdout.strip().splitlines()[-1])
    assert issues == [], f"the spine refused the lowered graph: {issues}"


def _state_with(**knobs: object) -> dict:
    """One state with the knobs applied, so both runtimes get the same ones.

    Setting a knob only on the `--knobs` side would compare two different runs,
    which is the mistake this helper exists to prevent.
    """
    from refactory_dsh import nn

    state = _mode().state_dict()
    for name, value in knobs.items():
        state = nn.edits.apply({"op": "set_knob", "name": name, "value": value}, state)
    return state


def _knobs_of(state: dict) -> dict:
    return {name: entry["value"] for name, entry in state["knobs"].items()}


def test_a_mode_scores_the_same_on_dsh_as_with_the_walker(tmp_path):
    state = _state_with(budget=2)
    knobs = _knobs_of(state)

    dsh = _run_on_dsh(state, tmp_path / "dsh", knobs=knobs)
    assert dsh["_returncode"] == 0, dsh["_stderr"]
    walker = _run_with_the_walker(state, tmp_path)

    assert dsh["_facts"], "DSH produced no facts, so there is nothing to compare"
    # The runtime boundary: DSH produces facts on disk, Python turns them into a
    # number. Both runtimes must agree about that number.
    from refactory_dsh.nn import RunResult

    dsh_score = type(_mode())().score(RunResult(facts=dsh["_facts"]))
    assert dsh_score == pytest.approx(walker["score"]), (
        f"the walker scored {walker['score']} and DSH scored {dsh_score}; "
        f"facts {json.dumps(dsh['_facts'], default=str)[:400]}"
    )
    assert dsh["best_score"] == pytest.approx(dsh_score), (
        "the runner's own reported score must agree with the facts it left behind"
    )


def test_the_facts_dsh_produced_are_the_facts_the_walker_produced(tmp_path):
    state = _state_with(budget=2)
    knobs = _knobs_of(state)

    dsh = _run_on_dsh(state, tmp_path / "dsh", knobs=knobs)
    walker = _run_with_the_walker(state, tmp_path)
    for key in ("circles", "seed_score", "best_score", "spent"):
        assert key in dsh["_facts"], f"DSH did not produce {key}: {sorted(dsh['_facts'])}"
        assert dsh["_facts"][key] == walker["facts"][key], (
            f"{key}: DSH said {dsh['_facts'][key]!r}, the walker said "
            f"{walker['facts'][key]!r}"
        )


def test_a_node_that_lies_is_refused_on_the_dsh_path_too(tmp_path):
    """Declaration enforcement is not a property of the debug runtime."""
    state = _mode().state_dict()
    lowering.write_declarations(state, tmp_path)
    # Take the produce declaration away and the run must fail, not silently pass.
    path = tmp_path / lowering.DECLARATIONS
    payload = json.loads(path.read_text())
    payload["exit"]["produces"] = []
    path.write_text(json.dumps(payload))
    result = subprocess.run(
        [str(venv_bin() / "python"), "-m", "refactory_dsh.run",
         "refactory_packages.circle_packing.ops:finish", "--node", "exit"],
        env={**__import__("os").environ, "REFACTORY_WORKDIR": str(tmp_path),
             lowering.KNOBS_ENV: "{}"},
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode == 1, result.stdout
    assert "produced the fact" in result.stderr


def test_the_loop_can_use_dsh_as_its_evaluator(tmp_path):
    """The payoff: the same loop, the same mode, the shipping runtime.

    One epoch is enough to prove the wiring. The two runs must agree about the
    best score, which is the whole claim of this issue.
    """
    from refactory_dsh.loop import Loop, LoopConfig

    def config(root: Path, evaluator: str) -> LoopConfig:
        return LoopConfig(
            epochs=1, batch_size=3, optimizer="random", seed=5,
            runs_root=root, evaluator=evaluator, dsh_timeout_s=900,
        )

    walked = Loop(MODE, config(tmp_path / "walker", "walker"), project=Path(".")).run()
    shipped = Loop(MODE, config(tmp_path / "dsh", "dsh"), project=Path(".")).run()
    assert shipped["best_score"] == pytest.approx(walked["best_score"]), (
        f"walker said {walked['best_score']}, DSH said {shipped['best_score']}"
    )
    assert shipped["evaluations"] == walked["evaluations"]
    assert all(row["exit_ok"] for row in shipped["leaderboard"]), (
        "a DSH inner run failed: "
        + json.dumps([r for r in shipped["leaderboard"] if not r["exit_ok"]], default=str)
    )


def test_the_loop_refuses_an_unknown_evaluator(tmp_path):
    """The registry is the one authority, and it fails at construction."""
    from refactory_dsh.evaluators import EvaluatorError
    from refactory_dsh.loop import Loop, LoopConfig

    with pytest.raises(EvaluatorError, match="unknown evaluator"):
        Loop(
            MODE,
            LoopConfig(epochs=1, batch_size=2, optimizer="random", runs_root=tmp_path,
                       evaluator="magic"),
            project=Path("."),
        ).init()


# ── a data node: one pass per item, on both runtimes ──────────────────────────
#
# Issue #59. The walking executor runs a data node's loop itself; the spine drives
# it from outside through the item loader's three commands, because the subgraph a
# data node iterates can contain agent nodes. The acceptance is that the facts come
# out the same either way, so this is the test that says so.

#: A package whose single mode is a data node over a one-op subgraph, bound to a
#: Task that names two items and judges each pass. Written into a temporary project
#: so the node subprocesses resolve it the same way a created package resolves.
PROBE = {
    # The package's two modules, as a created package would write them.
    "__init__.py": '''
from refactory_dsh.nn import Package

from .probe import Dataset

PACKAGE = Package(
    name="dataset_probe",
    description="iterate a Task's items",
    modes=(Dataset,),
    ops=("refactory_packages.dataset_probe.probe",),
)
''',
    "probe.py": '''
from __future__ import annotations

from pathlib import Path

from refactory_dsh import nn
from refactory_dsh.task import Task, TaskInstance, VerifyResult

OPS = "refactory_packages.dataset_probe.probe"


class CountingTask(Task):
    """Two named items, each judged by whether the pass recorded it."""

    name = "counting"

    def instances(self):
        return [TaskInstance(id="alpha"), TaskInstance(id="beta")]

    def verify(self, instance, workspace: Path) -> VerifyResult:
        seen = workspace / "seen.json"
        passed = seen.exists() and instance.id in seen.read_text()
        return VerifyResult(passed=passed, score=1.0 if passed else 0.0)


class Dataset(nn.Mode):
    """One data node over a one-node subgraph, bound to a Task."""

    name = "dataset"
    description = "iterate a Task's items and score the passes"

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref="refactory_packages.dataset_probe.probe:CountingTask",
            writes=("current_item.json",), produces=("item_scores",),
        ))
        graph.add(nn.Node(
            "work", "op", call=f"{OPS}:record",
            reads=("current_item.json",), writes=("seen.json",), produces=("value",),
        ))
        graph.add(nn.Node("finish", "op", call=f"{OPS}:done", produces=("finished",)))
        graph.connect("data", "finish")
        return graph


def record(ctx) -> dict:
    """Append the current item's id, the way a per-item worker would."""
    item = ctx.read_json("current_item.json", {}) or {}
    seen = ctx.read_json("seen.json", {"ids": []}) or {"ids": []}
    seen.setdefault("ids", []).append(str(item.get("id")))
    ctx.write_json("seen.json", seen)
    return {"value": len(seen["ids"])}


def done(ctx) -> dict:
    return {"finished": True}
''',
}


def _probe_project(root: Path) -> Path:
    """Write the probe package into a temporary project and return the project."""
    package = root / ".factory" / "packages" / "dataset_probe"
    package.mkdir(parents=True)
    for filename, source in PROBE.items():
        (package / filename).write_text(source)
    return root


@pytest.mark.integration
def test_a_data_node_iterates_the_same_items_on_dsh_as_with_the_walker(
    tmp_path, monkeypatch
):
    """One pass per item, and the same facts, on the runtime that ships.

    The item loader is a command the composition configures, so this also proves
    the wiring: the bundle publishes `dataCommand`, the spine runs its three verbs
    around each pass, and the Task's `verify` lands in `item_scores` exactly as it
    does when the walker owns the loop.
    """
    project = _probe_project(tmp_path / "project")
    # The node subprocesses resolve workspace packages against the project, and a
    # `module:Class` Task reference is only importable where the package is.
    monkeypatch.setenv("REFACTORY_PROJECT", str(project))
    _package, mode_class = registry.resolve(project, "dataset_probe/dataset")
    state = mode_class().state_dict()

    walked = type(mode_class())(state=state).evaluate(tmp_path / "walk", max_steps=500)
    assert walked.exit_ok, walked.error

    dsh = _run_on_dsh(state, tmp_path / "dsh", knobs={}, name="dataset_probe")
    assert dsh["_returncode"] == 0, dsh["_stderr"]

    seen = json.loads((Path(dsh["_work"]) / "seen.json").read_text())
    assert seen["ids"] == ["alpha", "beta"], (
        "the subgraph did not run once per item, in order, on DSH: "
        f"{seen}"
    )
    assert dsh["_facts"].get("item_scores") == walked.facts["item_scores"], (
        f"the Task judged the passes differently: walker {walked.facts['item_scores']}, "
        f"DSH {dsh['_facts'].get('item_scores')}"
    )
    assert dsh["_facts"].get("value") == walked.facts["value"]
    assert dsh["_facts"].get("finished") is True
