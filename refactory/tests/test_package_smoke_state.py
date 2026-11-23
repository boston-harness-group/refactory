"""A created package is marked in `refactory list` until its smoke passes (dm-264).

The walkthrough's create run stopped when readme_pack's smoke failed, and `refactory
list` still showed readme_pack like any other package. Register has to put a package
where the smoke can import it, so it is listed from that moment; what the listing owes
the person is that it says the package has not been shown to run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from refactory_dsh import create_ops, nn, registry, sandbox
from refactory_dsh.create_ops import STAGING_DIR, register
from refactory_dsh.steering import Ledger
from tests_support import git_repo

NAME = "smoke_state_pkg"


def _project(tmp_path: Path, monkeypatch) -> Path:
    project = git_repo(tmp_path / "project")
    (project / ".factory" / "packages").mkdir(parents=True)
    monkeypatch.setattr("refactory_dsh.paths.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.create_ops.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.sandbox.describe_package", lambda name, **_: sandbox.SandboxResult(
        ok=True, payload={"describe": {"name": name, "modes": []}, "ids": [f"{name}/x"]}))
    return project


def _registered(tmp_path: Path, project: Path) -> nn.RunContext:
    """A create run that staged a package and registered it, as the graph does."""
    run = tmp_path / "run"
    staging = run / STAGING_DIR / ".factory" / "packages" / NAME
    staging.mkdir(parents=True)
    (staging / "__init__.py").write_text(
        "from refactory_dsh.nn import Package\n\n"
        f"PACKAGE = Package(name={NAME!r}, description='made by a create run')\n")
    (staging / ".origin.json").write_text(json.dumps({"origin": "create-mode", "trust": "generated"}))
    ctx = nn.RunContext(run, facts={"name": NAME, "package_root": str(staging), "source_hash": "",
                                    "source_existed": False, "brief": "make it",
                                    "artifact": str(run)}, enforce_writes=False)
    ctx.facts.update(register(ctx))
    ctx.write_json("brief.json", {"brief": "make it"})
    ctx.write_json("strategy.json", {"acceptance_criteria": ["it runs"]})
    Ledger.seed(run, "make it")
    return ctx


def _line(project: Path) -> str:
    return next(line for line in registry.render_catalogue(project).splitlines()
                if line.startswith(f"{NAME}/"))


def test_a_registered_package_is_unverified_until_its_smoke_passes(tmp_path, monkeypatch):
    project = _project(tmp_path, monkeypatch)
    ctx = _registered(tmp_path, project)
    shared = registry.workspace_root(project) / NAME
    assert json.loads((shared / registry.ORIGIN).read_text())["smoke"] == "pending"
    assert _line(project) == f"{NAME}/  made by a create run  [generated, unverified (smoke not passed)]"

    # The smoke fails, as readme_pack's did: still marked.
    def fails(ctx):
        raise ValueError("the new mode did not finish a run: WebAssembly.instantiate(): Out of memory")

    monkeypatch.setattr(create_ops, "smoke", fails)
    assert create_ops.gate_smoke(ctx) == "reloop"
    assert registry.smoke_state(NAME, project) == "pending"
    assert "unverified" in _line(project)

    # It passes: listed like any other package.
    monkeypatch.setattr(create_ops, "smoke", lambda ctx: {"smoke_score": 1.0, "smoke_ok": True})
    assert create_ops.gate_smoke(ctx) == "proceed"
    assert registry.smoke_state(NAME, project) == "passed"
    assert _line(project) == f"{NAME}/  made by a create run  [generated]"

    # A later attempt that fails takes the mark back: a pass is not kept past a failure.
    monkeypatch.setattr(create_ops, "smoke", fails)
    create_ops.gate_smoke(ctx)
    assert registry.smoke_state(NAME, project) == "pending"


def test_a_package_written_in_place_is_unverified_from_registration(tmp_path, monkeypatch):
    """Without a staging directory `materialise` writes the shared package itself."""
    project = _project(tmp_path, monkeypatch)
    shared = registry.workspace_root(project) / NAME
    shared.mkdir()
    (shared / "__init__.py").write_text(
        "from refactory_dsh.nn import Package\n\n"
        f"PACKAGE = Package(name={NAME!r}, description='made in place')\n")
    (shared / registry.ORIGIN).write_text(json.dumps({"origin": "create-mode", "trust": "generated"}))
    run = tmp_path / "run"
    run.mkdir()
    register(nn.RunContext(run, facts={"name": NAME}, enforce_writes=False))
    assert registry.smoke_state(NAME, project) == "pending"
    assert _line(project).endswith("[generated, unverified (smoke not passed)]")


def test_a_package_no_run_registered_is_not_marked(tmp_path):
    """Installed, added or adopted packages never went through a run's smoke."""
    root = registry.workspace_root(tmp_path) / "added_pkg_state"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text(
        "from refactory_dsh.nn import Package\n\nPACKAGE = Package(name='added_pkg_state', "
        "description='added by hand')\n")
    (root / registry.ORIGIN).write_text(json.dumps({"origin": "/somewhere", "trust": "generated"}))
    listing = registry.render_catalogue(tmp_path)
    assert "added_pkg_state/  added by hand  [generated]" in listing
    assert "unverified" not in listing
    (root / registry.ORIGIN).write_text("not json")
    assert registry.smoke_state("added_pkg_state", tmp_path) == "passed"


def _runnable(project: Path, name: str, smoke: str) -> None:
    """A workspace package with one mode that runs, as a create run left it."""
    root = registry.workspace_root(project) / name
    root.mkdir(parents=True)
    (root / "__init__.py").write_text(
        "from refactory_dsh.nn import Package\n"
        "from refactory_packages.circle_packing.mode import CirclePacking\n\n"
        f"PACKAGE = Package(name={name!r}, description='fixed by hand', modes=(CirclePacking,),\n"
        "                  ops=('refactory_packages.circle_packing.ops',))\n")
    (root / registry.ORIGIN).write_text(json.dumps({"origin": "create-mode", "trust": "generated",
                                                    "smoke": smoke}))


def test_a_passing_package_smoke_clears_the_mark(tmp_path):
    """A person who fixes a package whose run's smoke failed, and smokes it, sees it verified."""
    _runnable(tmp_path, "hand_fixed_pkg", "pending")
    assert "unverified" in registry.render_catalogue(tmp_path)
    result = registry.smoke(tmp_path, "hand_fixed_pkg/adaptive")
    assert result["exit_ok"] is True
    assert registry.smoke_state("hand_fixed_pkg", tmp_path) == "passed"
    assert "unverified" not in registry.render_catalogue(tmp_path)


def test_a_failing_package_smoke_leaves_the_mark(tmp_path):
    root = registry.workspace_root(tmp_path) / "still_broken_pkg"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text(
        "from refactory_dsh.nn import Package\n"
        "from refactory_dsh.nn.mode import RunResult\n"
        "from refactory_packages.circle_packing.mode import CirclePacking\n\n\n"
        "class Broken(CirclePacking):\n"
        "    def evaluate(self, workdir, model=None, **kwargs):\n"
        "        return RunResult(exit_ok=False, error='still broken')\n\n\n"
        "PACKAGE = Package(name='still_broken_pkg', description='not fixed yet', modes=(Broken,),\n"
        "                  ops=('refactory_packages.circle_packing.ops',))\n")
    (root / registry.ORIGIN).write_text(json.dumps({"origin": "create-mode", "smoke": "pending"}))
    assert registry.smoke(tmp_path, "still_broken_pkg/adaptive")["exit_ok"] is False
    assert registry.smoke_state("still_broken_pkg", tmp_path) == "pending"
    assert "still_broken_pkg/  not fixed yet  [unverified (smoke not passed)]" in (
        registry.render_catalogue(tmp_path))


def test_a_package_smoke_writes_nothing_for_a_package_with_no_record(tmp_path):
    _runnable(tmp_path, "unrecorded_pkg", "passed")
    origin = registry.workspace_root(tmp_path) / "unrecorded_pkg" / registry.ORIGIN
    origin.write_text(json.dumps({"origin": "/elsewhere", "trust": "generated"}))
    before = origin.read_bytes()
    assert registry.smoke(tmp_path, "unrecorded_pkg/adaptive")["exit_ok"] is True
    assert origin.read_bytes() == before, "a package nothing marked is not rewritten"


def test_the_json_catalogue_carries_the_smoke_state(tmp_path):
    """The GUI picker and the CEO read this catalogue; they can now see the mark."""
    from refactory_dsh import console

    _runnable(tmp_path, "catalogued_pkg", "pending")
    entries = {entry["id"]: entry for entry in console.catalogue(project=tmp_path)}
    assert entries["catalogued_pkg/adaptive"]["smoke"] == "pending"
    assert entries["circle_packing/adaptive"]["smoke"] == "passed"
    registry.record_smoke(registry.workspace_root(tmp_path) / "catalogued_pkg", "passed")
    entries = {entry["id"]: entry for entry in console.catalogue(project=tmp_path)}
    assert entries["catalogued_pkg/adaptive"]["smoke"] == "passed"


def test_the_package_record_changes_once_after_registering(tmp_path, monkeypatch):
    """`.origin.json` is in the hash a concurrent run's register compares: a passing smoke
    changes it once, not twice (pending again, then passed)."""
    project = _project(tmp_path, monkeypatch)
    ctx = _registered(tmp_path, project)
    writes = []
    original = registry.record_smoke
    monkeypatch.setattr(registry, "record_smoke", lambda root, status: (writes.append(status),
                                                                        original(root, status)))
    monkeypatch.setattr(create_ops, "smoke", lambda ctx: {"smoke_score": 1.0, "smoke_ok": True})
    assert create_ops.gate_smoke(ctx) == "proceed"
    assert writes == ["passed"]

    # A revision's smoke records the pass in its own provenance write (`_revision`), so
    # the gate writes nothing more.
    def revised(ctx):
        registry.record_smoke(Path(ctx.facts["package_path"]), "pending")  # set up, not counted
        writes.clear()
        origin = Path(ctx.facts["package_path"]) / registry.ORIGIN
        origin.write_text(json.dumps({**json.loads(origin.read_text()), "smoke": "passed",
                                      "revision_of": "circle_packing/adaptive"}))
        return {"smoke_score": 1.0, "smoke_ok": True}

    monkeypatch.setattr(create_ops, "smoke", revised)
    assert create_ops.gate_smoke(ctx) == "proceed"
    assert writes == [] and registry.smoke_state(NAME, project) == "passed"



# ── a smoke that stops before its run ends halts the run, once, on both runtimes ──

QUIET = "quiet_smoke_pkg"

#: The created package: one mode whose only work goes quiet for a minute, so its smoke
#: is stopped for inactivity, the outcome `create_ops.gate_smoke` cannot judge.
_QUIET_FILES = {
    "__init__.py": (
        "from refactory_dsh.nn import Package\n"
        "from .mode import Quiet\n\n"
        f"PACKAGE = Package(name={QUIET!r}, modes=(Quiet,), ops=('refactory_packages.{QUIET}.ops',))\n"
    ),
    "ops.py": (
        "import time\n\n"
        "def wait(ctx):\n"
        "    time.sleep(60)\n"
        "    ctx.write_json('best.json', {'best': 1.0})\n"
        "    return {'best': 1.0}\n"
    ),
    "mode.py": (
        "from refactory_dsh import nn\n\n"
        "class Quiet(nn.Mode):\n"
        "    name = 'quiet'\n"
        "    def build(self):\n"
        "        g = nn.Graph()\n"
        f"        g.add(nn.Node('wait', 'op', call='refactory_packages.{QUIET}.ops:wait',"
        " writes=('best.json',), produces=('best',)))\n"
        "        return g\n"
    ),
}

#: The reason the halt carries (`create_ops.gate_smoke`), as the person reads it.
_HALT = "smoke timed out (inactivity, limit 2s); code repair was not attempted"


class _SmokeGate(nn.Mode):
    """Create's smoke gate as create wires it (reloop to the authoring node), after a node
    that leaves what the gate reads. The real `create_ops.gate_smoke` runs it."""

    name = "smoke_gate"

    @classmethod
    def declare(cls):
        return {
            "package_name": nn.Knob(QUIET, "str"),
            "smoke_timeout_s": nn.Knob(2, "int", lo=1, hi=86400),
            "smoke_items": nn.Knob(3, "int", lo=1),
            "attempts": nn.Knob(10, "int", lo=0),
        }

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("emit", "op", call="loop_ops:name_the_smoked_package",
                          knobs=("package_name",), writes=("brief.json", "strategy.json"),
                          produces=("name", "brief", "artifact")))
        graph.add(nn.Node("smoke", "gate", evaluator="refactory_dsh.create_ops:gate_smoke",
                          knobs=("smoke_timeout_s", "smoke_items", "attempts"),
                          reads=("brief.json", "strategy.json"),
                          writes=("smoke.json", "revision.json", "authoring-evidence.json",
                                  "validation.txt"),
                          produces=("smoke_score", "smoke_ok", "smoke_model", "smoke_reason",
                                    "before_score", "after_score", "regressed", "revision_note",
                                    "runtime_evidence", "validation", "attempts_used",
                                    "check_attempts_used", "convergence_extras_used",
                                    "convergence_granted")))
        graph.add(nn.Node("done", "op", call="loop_ops:noop"))
        graph.connect("emit", "smoke")
        graph.connect("smoke", "done", condition="proceed")
        graph.connect("smoke", "emit", condition="reloop")
        return graph


def _quiet_project(tmp_path: Path, monkeypatch) -> Path:
    project = git_repo(tmp_path / "project")
    root = registry.workspace_root(project) / QUIET
    root.mkdir(parents=True)
    for filename, body in _QUIET_FILES.items():
        (root / filename).write_text(body)
    registry.install_workspace(project)
    monkeypatch.setenv("REFACTORY_PROJECT", str(project))
    return project


def _smokes(run: Path) -> list[Path]:
    """The smoke runs a run made: `create_ops.smoke` gives each its own directory."""
    return sorted(path for path in (run / "smoke").glob("*") if path.is_dir())


def test_a_smoke_stopped_for_inactivity_halts_the_walker_run_once(tmp_path, monkeypatch):
    """Walker: the real gate halts with its reason instead of failing or reworking the
    package, the reason reaches the run's status, and the smoke ran once."""
    _quiet_project(tmp_path, monkeypatch)
    run = tmp_path / "run"
    result = _SmokeGate().evaluate(run)
    assert result.halted is True, result.error
    assert _HALT in (result.error or "")
    assert result.gate_verdicts.get("smoke") == "halt"
    status = json.loads((run / ".refactory" / "spine-status.json").read_text())
    assert status["status"] == "halted" and _HALT in status["error"]
    assert len(_smokes(run)) == 1
    assert json.loads((run / "smoke.json").read_text())["timeout_kind"] == "inactivity"


def test_a_smoke_stopped_for_inactivity_halts_the_spine_run_once(tmp_path, monkeypatch,
                                                                 graph_profile_home):
    """Spine: the same graph, lowered and run by the graph profile's spine
    (`graph-owner.mjs`, the real spine row with no CEO model), the gate run as DSH runs
    it (`python -m refactory_dsh.run`). The gate's printed reason is the halt's, in the
    run's status, and the smoke ran once."""
    from refactory_dsh.dsh_runtime import run_on_dsh
    from tests_support import graph_owner_launcher

    launcher = graph_owner_launcher(tmp_path / "dsh")
    if launcher is None:
        pytest.skip("the spine owner needs Node and the installed workspace (pnpm install)")
    _quiet_project(tmp_path, monkeypatch)
    monkeypatch.setenv("DSH_CLI", str(launcher))
    tests = str(Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        part for part in (tests, os.environ.get("PYTHONPATH", "")) if part))
    run = tmp_path / "run"
    done = run_on_dsh(_SmokeGate().state_dict(), run, name="smoke-gate", timeout_s=600)
    status = json.loads((run / ".refactory" / "spine-status.json").read_text())
    assert status["status"] == "halted", (status, done.stderr[-3000:])
    assert _HALT in status["error"], status
    assert len(_smokes(run)) == 1
    assert json.loads((run / "smoke.json").read_text())["timeout_kind"] == "inactivity"


# ── a halt's reason reaches the walker's status from inside a fork or a data pass ──

def _fork_with_halting_branch() -> nn.Graph:
    """fork -> (body -> gate, side) -> join, the gate out of retries and declared ``halt``."""
    graph = nn.Graph()
    graph.add(nn.Node("fork", "fork", targets=("body", "side")))
    graph.add(nn.Node("body", "op", call="loop_ops:count_visit",
                      writes=("visits.json",), produces=("visits",)))
    graph.add(nn.Node("gate", "gate", evaluator="loop_ops:always_retry", max_iterations=2,
                      on_exhausted="halt"))
    graph.add(nn.Node("side", "op", call="loop_ops:noop"))
    graph.add(nn.Node("join", "join", sources=("gate", "side")))
    graph.add(nn.Node("done", "op", call="loop_ops:noop"))
    graph.connect("fork", "join")
    graph.connect("body", "gate")
    graph.connect("gate", "body", condition="retry")
    graph.connect("gate", "join", condition="proceed")
    graph.connect("side", "join")
    graph.connect("join", "done")
    return graph


@pytest.mark.parametrize("shape", ["fork", "pass", "pass-at-once"])
def test_a_halt_inside_a_fork_or_a_data_pass_keeps_its_reason_on_the_walker(tmp_path, shape):
    """The gate's reason -- here the cap that refused its retry -- is in the run's error
    and its status, wherever the halt happened, as the spine keeps it; not only which
    gate halted."""
    from refactory_dsh.nn.executor import execute
    from test_loop_exhaustion import _loop_graph

    graph = (_fork_with_halting_branch() if shape == "fork"
             else _loop_graph(parallelism=3 if shape == "pass-at-once" else 1, on_exhausted="halt"))
    run = tmp_path / "run"
    result = execute(graph, run, max_steps=400)
    assert result.halted, result.error
    assert "halted the run: max_iterations of 2" in str(result.error), result.error
    status = json.loads((run / ".refactory" / "spine-status.json").read_text())
    assert "max_iterations of 2" in status["error"], status
