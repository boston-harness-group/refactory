"""Per-run staging for create/edit modes: two parallel runs never collide (#351).

Two parallel creates of the same package name both build in their own staging
directories.  The second to register refuses, naming the first, and leaves the
first's package intact.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import registry, sandbox
from refactory_dsh.create_ops import prepare_create, materialise, register, _dir_hash, STAGING_DIR
from refactory_dsh.paths import project_for_nodes
from tests_support import git_repo


def _mock_describe(name, **kwargs):
    """Return a successful SandboxResult for describe_package, bypassing subprocess import."""
    return sandbox.SandboxResult(
        ok=True,
        payload={"describe": {"name": name, "modes": []}, "ids": [name]},
    )


class FakeCtx:
    """A minimal context for create_ops functions."""

    def __init__(self, workdir: Path, facts: dict[str, Any] | None = None,
                 knobs_dict: dict[str, Any] | None = None,
                 task: dict[str, Any] | None = None, texts: dict[str, str] | None = None):
        self.workdir = workdir
        self.facts = dict(facts or {})
        self._knobs = dict(knobs_dict or {})
        self.task = dict(task or {})
        self._texts: dict[str, str] = dict(texts or {})
        self._jsons: dict[str, Any] = {}
        self.model = None
        self.budget = None

    def knob(self, name: str, default: Any = None) -> Any:
        return self._knobs.get(name, default)

    def read_text(self, name: str) -> str:
        return self._texts.get(name, "")

    def write_text(self, name: str, content: str) -> None:
        path = self.workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self._texts[name] = content

    def read_json(self, name: str, default: Any = None) -> Any:
        path = self.workdir / name
        if path.exists():
            return json.loads(path.read_text())
        return default

    def write_json(self, name: str, payload: Any) -> None:
        path = self.workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        self._jsons[name] = payload

    def set_fact(self, name: str, value: Any) -> None:
        self.facts[name] = value

    def exists(self, name: str) -> bool:
        return (self.workdir / name).exists()

    @property
    def iteration(self) -> int:
        return 1


def _make_project(tmp_path: Path) -> Path:
    """Create a project with .factory/packages/ directory."""
    project = git_repo(tmp_path / "project")
    pkgs = project / ".factory" / "packages"
    pkgs.mkdir(parents=True, exist_ok=True)
    return project


def _make_staged_package(run: Path, name: str, content: str = "# generated\n") -> Path:
    """Create a staged package in a run directory with a valid .origin.json."""
    staging = run / STAGING_DIR / ".factory" / "packages" / name
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "__init__.py").write_text(content)
    (staging / ".origin.json").write_text(json.dumps({
        "origin": "create-mode", "ref": "", "trust": "generated",
    }, indent=2))
    return staging


def test_two_parallel_creates_both_build_second_register_refuses(tmp_path, monkeypatch):
    """Two runs through prepare_create; the second register refuses naming the first.

    Drives through the real production path: prepare_create → write content →
    register.  Assert that the second register's refusal names the first run.
    """
    project = _make_project(tmp_path)
    monkeypatch.setattr("refactory_dsh.paths.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.create_ops.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.sandbox.describe_package", _mock_describe)
    monkeypatch.setattr("refactory_dsh.create_ops.prepare", lambda ctx: {
        "name": "mypkg",
        "brief": "test parallel creates",
    })

    shared = registry.workspace_root(project) / "mypkg"
    assert not shared.exists()

    # Run 1: prepare_create sets up staging.
    run1 = tmp_path / "run1"
    run1.mkdir()
    ctx1 = FakeCtx(run1)
    result1 = prepare_create(ctx1)
    ctx1.facts.update(result1)
    # Simulate materialise writing into the staged package.
    pkg1 = Path(result1["package_root"])
    (pkg1 / "__init__.py").write_text("# run1\n")

    # Run 2: prepare_create sets up its own staging.
    run2 = tmp_path / "run2"
    run2.mkdir()
    ctx2 = FakeCtx(run2)
    result2 = prepare_create(ctx2)
    ctx2.facts.update(result2)
    pkg2 = Path(result2["package_root"])
    (pkg2 / "__init__.py").write_text("# run2\n")

    # Both staging directories are separate.
    assert str(pkg1) != str(pkg2)
    assert pkg1.exists() and pkg2.exists()

    # Run 1 registers successfully.
    register(ctx1)
    assert shared.exists()
    assert (shared / "__init__.py").read_text() == "# run1\n"
    origin = json.loads((shared / ".origin.json").read_text())
    assert origin.get("run_id") == "run1"

    # Run 2 tries to register: refuses because the shared package appeared.
    with pytest.raises(ValueError, match="did not exist when this run's prepare"):
        register(ctx2)

    # Run 1's package is intact.
    assert (shared / "__init__.py").read_text() == "# run1\n"


def test_second_register_refuses_when_shared_changed(tmp_path, monkeypatch):
    """The second run to register refuses, naming the first, when the shared package changed."""
    project = _make_project(tmp_path)
    monkeypatch.setattr("refactory_dsh.paths.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.create_ops.project_for_nodes", lambda: project)

    shared = registry.workspace_root(project) / "mypkg"
    shared.mkdir(parents=True, exist_ok=True)
    (shared / "__init__.py").write_text("# original\n")
    (shared / ".origin.json").write_text('{"origin": "create-mode"}\n')
    original_hash = _dir_hash(shared)

    # First run registers successfully: simulate by changing the shared package.
    (shared / "__init__.py").write_text("# run1 registered\n")
    (shared / ".origin.json").write_text(json.dumps({
        "origin": "create-mode", "run_id": "run1",
    }, indent=2))
    new_hash = _dir_hash(shared)
    assert new_hash != original_hash

    # Second run has the original hash and a staged package.
    run2 = tmp_path / "run2"
    run2.mkdir()
    staging = _make_staged_package(run2, "mypkg", "# run2\n")

    ctx2 = FakeCtx(run2, facts={
        "name": "mypkg",
        "package_root": str(staging),
        "source_hash": original_hash,
        "source_existed": True,
    })

    with pytest.raises(ValueError, match="changed since this run's prepare"):
        register(ctx2)

    # The first run's package is intact.
    assert (shared / "__init__.py").read_text() == "# run1 registered\n"


def test_reloop_register_does_not_refuse_itself(tmp_path, monkeypatch):
    """A register that reloops does not refuse its own prior registration."""
    project = _make_project(tmp_path)
    monkeypatch.setattr("refactory_dsh.paths.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.create_ops.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.sandbox.describe_package", _mock_describe)

    run1 = tmp_path / "run1"
    run1.mkdir()
    staging = _make_staged_package(run1, "mypkg", "# attempt 1\n")

    # First register.
    ctx = FakeCtx(run1, facts={
        "name": "mypkg",
        "package_root": str(staging),
        "source_hash": "",
        "source_existed": False,
    })
    register(ctx)
    shared = registry.workspace_root(project) / "mypkg"
    assert shared.exists()

    # The run's origin is stamped in the shared package.
    origin = json.loads((shared / ".origin.json").read_text())
    assert origin.get("run_id") == "run1"

    # Simulate a reloop: materialise writes a new version to staging.
    (Path(staging) / "__init__.py").write_text("# attempt 2\n")

    # Second register of the same run: must not refuse.
    register(ctx)
    assert (shared / "__init__.py").read_text() == "# attempt 2\n"


def test_edit_whose_source_changed_refuses_at_register(tmp_path, monkeypatch):
    """An edit whose source package changed mid-run refuses at register."""
    project = _make_project(tmp_path)
    monkeypatch.setattr("refactory_dsh.paths.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.create_ops.project_for_nodes", lambda: project)

    shared = registry.workspace_root(project) / "mypkg"
    shared.mkdir(parents=True, exist_ok=True)
    (shared / "__init__.py").write_text("# original\n")
    (shared / ".origin.json").write_text('{"origin": "create-mode"}\n')
    original_hash = _dir_hash(shared)

    # Edit run prepares with the original hash.
    run = tmp_path / "edit_run"
    run.mkdir()
    staging = _make_staged_package(run, "mypkg", "# edited\n")

    # Another run changes the shared package.
    (shared / "__init__.py").write_text("# changed by another\n")

    ctx = FakeCtx(run, facts={
        "name": "mypkg",
        "package_root": str(staging),
        "source_hash": original_hash,
        "source_existed": True,
    })

    with pytest.raises(ValueError, match="changed since this run's prepare"):
        register(ctx)


def test_validation_imports_staged_copy(tmp_path, monkeypatch):
    """Validation imports the staged copy even when a shared copy exists."""
    project = _make_project(tmp_path)
    monkeypatch.setattr("refactory_dsh.paths.project_for_nodes", lambda: project)

    # Create a shared package.
    shared = registry.workspace_root(project) / "testpkg"
    shared.mkdir(parents=True, exist_ok=True)
    (shared / "__init__.py").write_text("SHARED = True\n")

    # Create a staging directory with a different package.
    run = tmp_path / "run"
    run.mkdir()
    staging_base = run / STAGING_DIR
    staging = staging_base / ".factory" / "packages" / "testpkg"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "__init__.py").write_text("STAGED = True\n")

    # After install_workspace with project_override, the staged one takes priority.
    import refactory_packages
    original_path = list(refactory_packages.__path__)
    try:
        registry.install_workspace(project, project_override=staging_base)
        # The staging base's packages dir should be first in the path.
        staged_pkgs = str((staging_base / ".factory" / "packages").resolve())
        assert staged_pkgs in refactory_packages.__path__
        # It should be before the shared one.
        shared_pkgs = str(registry.workspace_root(project).resolve())
        if shared_pkgs in refactory_packages.__path__:
            assert refactory_packages.__path__.index(staged_pkgs) < refactory_packages.__path__.index(shared_pkgs)
    finally:
        refactory_packages.__path__[:] = original_path


def test_edit_revision_of_differs_from_name(tmp_path, monkeypatch):
    """An edit whose revision_of differs from its name copies source from revision_of."""
    project = _make_project(tmp_path)
    monkeypatch.setattr("refactory_dsh.paths.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.create_ops.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.sandbox.describe_package", _mock_describe)

    # Create the source package under source_pkg (different from dest name).
    source_shared = registry.workspace_root(project) / "source_pkg"
    source_shared.mkdir(parents=True, exist_ok=True)
    (source_shared / "__init__.py").write_text("SOURCE = True\n")
    (source_shared / ".origin.json").write_text('{"origin": "create-mode"}\n')

    run = tmp_path / "edit_run"
    run.mkdir()

    monkeypatch.setattr("refactory_dsh.create_ops.prepare", lambda ctx: {
        "name": "dest_pkg",
        "revision_of": "source_pkg/some_mode",
        "brief": "test edit with different revision_of",
    })

    ctx = FakeCtx(run)
    result = prepare_create(ctx)

    # The staging directory starts from the source package's content.
    pkg_root = Path(result["package_root"])
    assert pkg_root.exists()
    assert (pkg_root / "__init__.py").read_text() == "SOURCE = True\n"
    assert result["source_existed"]
    assert result["source_hash"] != ""

    # Modify the source package mid-run (simulating another run changing it).
    (source_shared / "__init__.py").write_text("CHANGED = True\n")
    (source_shared / ".origin.json").write_text(json.dumps({
        "origin": "create-mode", "run_id": "other_run",
    }, indent=2))

    # Register should refuse because the source changed since prepare.
    ctx.facts.update(result)
    with pytest.raises(ValueError, match="changed since this run's prepare"):
        register(ctx)

    # The source package is intact (not overwritten by register).
    assert (source_shared / "__init__.py").read_text() == "CHANGED = True\n"


def test_staged_package_can_import_shared_workspace_package(tmp_path, monkeypatch):
    """A staged package whose ops import another workspace package validates.

    On main, ``check_package`` ran with the shared project and both packages were
    on the import path.  With staging, ``check_package`` passes
    ``project_override`` pointing to the staging base, so the subprocess saw only
    the staged workspace and failed with "No module named refactory_packages.base_ws".
    After the fix, ``verify_sandboxed`` receives ``shared_project`` and installs
    the shared workspace behind the staging root so the staged copy wins for its
    own name and every other package still resolves.
    """
    project = _make_project(tmp_path)
    monkeypatch.setattr("refactory_dsh.paths.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.create_ops.project_for_nodes", lambda: project)

    # Shared workspace has a helper package ``base_ws``.
    base_pkg = registry.workspace_root(project) / "base_ws"
    base_pkg.mkdir(parents=True, exist_ok=True)
    (base_pkg / "__init__.py").write_text(
        "from refactory_dsh.nn import Package\n"
        "HELPER = 42\n"
        "PACKAGE = Package(name='base_ws')\n"
    )

    # The staged package ``new_pkg`` imports from ``base_ws``.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staging_base = run_dir / STAGING_DIR
    new_pkg = staging_base / ".factory" / "packages" / "new_pkg"
    new_pkg.mkdir(parents=True, exist_ok=True)
    (new_pkg / "__init__.py").write_text(
        "from refactory_packages.base_ws import HELPER\n"
        "from refactory_dsh.nn import Package\n"
        "from .mode import MyMode\n\n"
        "PACKAGE = Package(name='new_pkg', modes=(MyMode,),"
        " ops=('refactory_packages.new_pkg.ops',))\n"
    )
    (new_pkg / "mode.py").write_text(
        "from refactory_dsh import nn\n"
        "OPS = 'refactory_packages.new_pkg.ops'\n\n"
        "class MyMode(nn.Mode):\n"
        "    name = 'mymode'\n"
        "    @classmethod\n"
        "    def declare(cls):\n"
        "        return {'target': nn.Knob(1, 'int', lo=1, hi=9)}\n"
        "    def build(self):\n"
        "        g = nn.Graph()\n"
        "        g.add(nn.Node('start', 'op', call=OPS + ':start',"
        " writes=('out.json',), produces=('v',)))\n"
        "        return g\n"
        "    def score(self, run):\n"
        "        return 1.0\n"
    )
    (new_pkg / "ops.py").write_text(
        "from refactory_packages.base_ws import HELPER\n\n"
        "def start(ctx):\n"
        "    ctx.write_json('out.json', {'v': HELPER})\n"
        "    return {'v': HELPER}\n"
    )

    from refactory_dsh.create_ops import check_package
    problems = check_package("new_pkg", project_override=staging_base)
    import_problems = [p for p in problems if "base_ws" in p or "does not import" in p]
    assert not import_problems, f"Staged package cannot import shared workspace package: {import_problems}"


def test_edit_into_existing_destination_succeeds(tmp_path, monkeypatch):
    """An edit whose destination already exists at prepare is not falsely refused.

    When ``revision_of`` differs from ``name`` and the destination package
    already exists at prepare time, register must not refuse with 'did not
    exist when this run's prepare ran' because the destination's state is
    recorded separately (``dest_hash`` / ``dest_existed``).
    """
    project = _make_project(tmp_path)
    monkeypatch.setattr("refactory_dsh.paths.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.create_ops.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.sandbox.describe_package", _mock_describe)

    # Create the source package.
    source_shared = registry.workspace_root(project) / "source_pkg"
    source_shared.mkdir(parents=True, exist_ok=True)
    (source_shared / "__init__.py").write_text("SOURCE = True\n")
    (source_shared / ".origin.json").write_text('{"origin": "create-mode"}\n')

    # Create the destination package (already exists before the edit).
    dest_shared = registry.workspace_root(project) / "dest_pkg"
    dest_shared.mkdir(parents=True, exist_ok=True)
    (dest_shared / "__init__.py").write_text("OLD_DEST = True\n")
    (dest_shared / ".origin.json").write_text('{"origin": "create-mode"}\n')

    run = tmp_path / "edit_run"
    run.mkdir()

    monkeypatch.setattr("refactory_dsh.create_ops.prepare", lambda ctx: {
        "name": "dest_pkg",
        "revision_of": "source_pkg/some_mode",
        "brief": "test edit into existing destination",
    })

    ctx = FakeCtx(run)
    result = prepare_create(ctx)
    ctx.facts.update(result)

    # Verify prepare recorded the destination's state.
    assert result["dest_existed"] is True
    assert result["dest_hash"] != ""

    # Write edited content into staging.
    pkg_root = Path(result["package_root"])
    (pkg_root / "__init__.py").write_text("EDITED = True\n")

    # Register succeeds because the destination has not changed since prepare.
    register(ctx)
    assert (dest_shared / "__init__.py").read_text() == "EDITED = True\n"


# ── the project is never refactory's own checkout (C1) ──────────────────────────────

def _prepared(monkeypatch, project: Path) -> None:
    monkeypatch.setattr("refactory_dsh.paths.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.create_ops.project_for_nodes", lambda: project)
    monkeypatch.setattr("refactory_dsh.create_ops.prepare", lambda ctx: {"name": "mypkg", "brief": "b"})


def test_prepare_create_refuses_refactory_own_checkout_as_the_project(tmp_path, monkeypatch):
    from refactory_dsh.paths import REPO_ROOT, CheckoutProjectError

    _prepared(monkeypatch, REPO_ROOT)
    before = sorted((REPO_ROOT / ".factory" / "packages").glob("*"))
    with pytest.raises(CheckoutProjectError, match="refusing to write packages into refactory's own checkout"):
        prepare_create(FakeCtx(tmp_path / "run"))
    assert sorted((REPO_ROOT / ".factory" / "packages").glob("*")) == before
    assert not (tmp_path / "run" / STAGING_DIR).exists(), "refused before anything is written"


def test_prepare_create_refuses_the_runs_recorded_driver_as_the_project(tmp_path, monkeypatch):
    from refactory_dsh.paths import CheckoutProjectError

    driver = git_repo(tmp_path / "driver")
    run = tmp_path / "run"
    (run / ".refactory").mkdir(parents=True)
    (run / ".refactory" / "run-config.json").write_text(json.dumps({"driver": {"checkout": str(driver)}}))
    _prepared(monkeypatch, driver)
    with pytest.raises(CheckoutProjectError, match=str(driver.resolve())):
        prepare_create(FakeCtx(run))


def test_prepare_create_records_the_project_so_agents_are_watched(tmp_path, monkeypatch):
    """The baseline that turns the per-node watch of refactory's checkout on for create/edit."""
    from refactory_dsh import target_guard

    project = _make_project(tmp_path)
    driver = git_repo(tmp_path / "driver")
    run = tmp_path / "run"
    (run / ".refactory").mkdir(parents=True)
    (run / ".refactory" / "run-config.json").write_text(json.dumps({"driver": {"checkout": str(driver)}}))
    _prepared(monkeypatch, project)
    prepare_create(FakeCtx(run))
    assert target_guard._read(run)["root"] == str(project.resolve())
    assert target_guard.checkouts(run, tmp_path / "nowhere") == [driver.resolve()]


def test_register_refuses_when_the_project_became_the_checkout(tmp_path, monkeypatch):
    from refactory_dsh.paths import REPO_ROOT, CheckoutProjectError

    project = _make_project(tmp_path)
    _prepared(monkeypatch, project)
    run = tmp_path / "run"
    ctx = FakeCtx(run)
    ctx.facts.update(prepare_create(ctx))
    _prepared(monkeypatch, REPO_ROOT)
    with pytest.raises(CheckoutProjectError):
        register(ctx)
    assert not (REPO_ROOT / ".factory" / "packages" / "mypkg").exists()


def test_a_project_that_is_another_refactory_clone_is_written_to(tmp_path, monkeypatch):
    from refactory_dsh.paths import project_for_writes

    clone = git_repo(tmp_path / "clone")
    (clone / "refactory" / "refactory_dsh").mkdir(parents=True)
    assert project_for_writes(clone) == clone.resolve()
    monkeypatch.chdir(clone)
    monkeypatch.delenv("REFACTORY_PROJECT", raising=False)
    assert project_for_writes() == clone.resolve(), "running from inside a project keeps working"


def test_spine_register_writes_the_stated_project_and_refuses_the_checkout(tmp_path):
    """`register` as the spine runs it: `python -m refactory_dsh.run` in the run directory,
    with the project the graph bundle states (`REFACTORY_PROJECT`, startup.ts). A bundle
    started from refactory's checkout states the checkout, and that is refused."""
    import os
    import subprocess
    import sys

    from refactory_dsh import lowering
    from refactory_dsh.paths import REPO_ROOT
    from refactory_dsh.run import FACTS, NODE_ENV
    from tests_support import good_files

    name = "c1_spine_pkg"
    project = _make_project(tmp_path)
    run = tmp_path / "run"
    staged = run / STAGING_DIR / ".factory" / "packages" / name
    staged.mkdir(parents=True)
    for relative, content in good_files(name).items():
        (staged / relative).parent.mkdir(parents=True, exist_ok=True)
        (staged / relative).write_text(content)
    (run / FACTS).write_text(json.dumps({"name": name, "package_root": str(staged),
                                         "source_hash": "", "source_existed": False}))
    (run / lowering.DECLARATIONS).parent.mkdir(parents=True, exist_ok=True)
    (run / lowering.DECLARATIONS).write_text(json.dumps({"register": {
        "reads": [], "writes": ["registered.json", "staging/package-diff.json"],
        "produces": ["registered", "package_path"]}}))

    def register_on_the_spine(stated: Path) -> subprocess.CompletedProcess:
        env = {**os.environ, "REFACTORY_PROJECT": str(stated), "REFACTORY_WORKDIR": str(run),
               NODE_ENV: json.dumps({"id": "register", "type": "FnNode", "reads": [], "writes": []})}
        return subprocess.run([sys.executable, "-m", "refactory_dsh.run", "refactory_dsh.create_ops:register"],
                              cwd=str(run), env=env, capture_output=True, text=True, timeout=120, check=False)

    refused = register_on_the_spine(REPO_ROOT)
    assert refused.returncode != 0
    assert "refusing to write packages into refactory's own checkout" in refused.stderr, refused.stderr
    assert not (REPO_ROOT / ".factory" / "packages" / name).exists()

    done = register_on_the_spine(project)
    assert done.returncode == 0, done.stderr
    assert (project / ".factory" / "packages" / name / "__init__.py").exists()
    assert not (REPO_ROOT / ".factory" / "packages" / name).exists()
