"""A created package is marked in `refactory list` until its smoke passes (dm-264).

The walkthrough's create run stopped when readme_pack's smoke failed, and `refactory
list` still showed readme_pack like any other package. Register has to put a package
where the smoke can import it, so it is listed from that moment; what the listing owes
the person is that it says the package has not been shown to run.
"""
from __future__ import annotations

import json
from pathlib import Path

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


def test_smoke_timeout_halt_prevents_redispatch_walker(tmp_path):
    """Walker: a graph with smoke → gate_smoke runs through the executor.

    The smoke op writes smoke.json with status='uncertain' (as the real
    create_ops.smoke does before raising on timeout). The gate reads smoke.json
    and returns Verdict('halt', reason) — the same logic as create_ops.gate_smoke's
    uncertain intercept. The executor catches HaltStop, records outcome='halted',
    and the run ends. Smoke runs exactly once (no retry, no re-dispatch).
    """
    from refactory_dsh.nn import Graph, Node, execute

    # Build a graph: smoke (writes smoke.json) → gate_smoke (halts on uncertain) → register.
    # Uses loop_ops functions that replicate create_ops.smoke/gate_smoke paths
    # in the forkserver child where create_ops's registry deps are unavailable.
    graph = Graph()
    graph.add(Node("smoke", "op", call="loop_ops:smoke_timeout_op",
                   writes=("smoke.json",)))
    graph.add(Node("gate_smoke", "gate", gate_kind="fn",
                   evaluator="loop_ops:gate_smoke_halt_on_uncertain",
                   reads=("smoke.json",)))
    graph.add(Node("register", "op", call="loop_ops:noop"))
    graph.connect("smoke", "gate_smoke")
    graph.connect("gate_smoke", "register", condition="proceed")

    result = execute(graph, tmp_path, enforce_writes=False)

    # The run should be halted, not failed.
    assert result.halted is True, f"expected halted, got halted={result.halted}, error={result.error}"
    assert "inactivity" in (result.error or ""), f"halt reason should mention inactivity: {result.error}"
    assert "code repair was not attempted" in (result.error or "")

    # gate_smoke saw the smoke.json that the smoke op wrote: both ran.
    assert result.gate_verdicts.get("gate_smoke") == "halt", (
        f"gate_smoke should have recorded halt, got {result.gate_verdicts}")

    # spine-status.json should say halted with the reason.
    status_path = tmp_path / ".refactory" / "spine-status.json"
    assert status_path.exists(), "spine-status.json not written"
    status = json.loads(status_path.read_text())
    assert status["status"] == "halted"
    assert "inactivity" in (status.get("error") or "")


def test_halt_reason_in_spine_status(tmp_path):
    """The halt reason from a Verdict('halt', reason) appears in the walker's
    spine-status.json error field — the record a person sees.

    This runs through the actual executor code path: an op → gate that halts
    with a reason, through the executor, checking that _write_status records
    the reason in spine-status.json.
    """
    from refactory_dsh.nn import Graph, Node, execute

    reason_text = "smoke timed out (inactivity, limit 3600s); code repair was not attempted"

    # Build a graph: prepare → gate (halts with reason) → done.
    graph = Graph()
    graph.add(Node("prepare", "op", call="loop_ops:noop",
                   writes=("halt_reason.txt",)))
    graph.add(Node("gate", "gate", gate_kind="fn",
                   evaluator="loop_ops:halt_with_reason",
                   reads=("halt_reason.txt",)))
    graph.add(Node("done", "op", call="loop_ops:noop"))
    graph.connect("prepare", "gate")
    graph.connect("gate", "done", condition="proceed")

    # Write the halt reason so the gate reads it.
    (tmp_path / "halt_reason.txt").write_text(reason_text)

    result = execute(graph, tmp_path, enforce_writes=False)
    assert result.halted is True

    # The halt reason must appear in spine-status.json.
    status_path = tmp_path / ".refactory" / "spine-status.json"
    assert status_path.exists(), "spine-status.json not written"
    status = json.loads(status_path.read_text())
    assert status["status"] == "halted"
    assert reason_text in (status.get("error") or ""), (
        f"halt reason not in spine-status error: {status.get('error')}")
    # The error field also contains the gate id.
    assert "gate" in (status.get("error") or "")
