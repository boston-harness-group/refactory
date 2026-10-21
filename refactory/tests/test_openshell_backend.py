"""Tests for the OpenShell sandbox backend composition layer.

These test the pure plan-building and workspace-packing logic — no
gateway, no SDK, no sandbox.  The execution path (``run_in_openshell``)
is integration-tested when the credential adapter lands.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from refactory_dsh.openshell_backend import (
    PACK_EXCLUDES,
    PROVIDER_NAME,
    RUN_ARTIFACTS,
    compose_plan,
    pack_workspace,
    provenance_checks,
)


class TestComposePlan:
    def test_deterministic_name_from_workdir(self, tmp_path: Path):
        plan = compose_plan(tmp_path, ["dsh", "run"])
        assert plan.name
        assert len(plan.name) <= 19

        plan2 = compose_plan(tmp_path, ["dsh", "run"])
        assert plan.name == plan2.name

    def test_project_dir_under_workspace_root(self, tmp_path: Path):
        plan = compose_plan(tmp_path, ["dsh", "run"])
        assert plan.project_dir.startswith("/workspace/")
        assert plan.project_dir.endswith(tmp_path.name)

    def test_provider_is_claude_code(self, tmp_path: Path):
        plan = compose_plan(tmp_path, ["dsh", "run"])
        assert plan.provider == PROVIDER_NAME

    def test_run_command_contains_argv(self, tmp_path: Path):
        plan = compose_plan(tmp_path, ["dsh", "--profile", "graph", "run"])
        assert "dsh --profile graph run" in plan.run_command

    def test_env_defaults(self, tmp_path: Path):
        plan = compose_plan(tmp_path, ["dsh", "run"], env={})
        assert plan.env["PYTHONUNBUFFERED"] == "1"
        assert plan.env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"

    def test_env_preserves_user_values(self, tmp_path: Path):
        plan = compose_plan(tmp_path, ["dsh", "run"], env={"FOO": "bar"})
        assert plan.env["FOO"] == "bar"

    def test_secret_env_warns(self, tmp_path: Path):
        plan = compose_plan(
            tmp_path, ["dsh", "run"],
            env={"ANTHROPIC_API_KEY": "sk-test"},
        )
        assert plan.warnings
        assert "credential" in plan.warnings[0].lower()

    def test_custom_name(self, tmp_path: Path):
        plan = compose_plan(tmp_path, ["dsh", "run"], name="my-sandbox")
        assert plan.name == "my-sandbox"

    def test_invalid_name_rejected(self, tmp_path: Path):
        with pytest.raises(Exception, match="sandbox name"):
            compose_plan(tmp_path, ["dsh", "run"], name="UPPER_CASE_BAD!")

    def test_labels_include_factory_contained(self, tmp_path: Path):
        plan = compose_plan(tmp_path, ["dsh", "run"])
        assert plan.labels["factory.contained"] == "true"

    def test_gateway_and_image_forwarded(self, tmp_path: Path):
        plan = compose_plan(
            tmp_path, ["dsh", "run"],
            gateway="my-gateway", image="custom:v1",
        )
        assert plan.gateway == "my-gateway"
        assert plan.image == "custom:v1"

    def test_default_image(self, tmp_path: Path):
        plan = compose_plan(tmp_path, ["dsh", "run"])
        assert plan.image == "factory-runtime:latest"


class TestPackWorkspace:
    def test_creates_tarball(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("print('hello')")
        tarball = pack_workspace(tmp_path, "test-run")
        assert tarball.exists()
        assert tarball.suffix == ".gz"

    def test_tarball_contains_files(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "input.json").write_text("{}")
        tarball = pack_workspace(tmp_path, "test-run")
        with tarfile.open(tarball) as archive:
            names = archive.getnames()
        assert any("main.py" in name for name in names)
        assert any("input.json" in name for name in names)

    def test_excludes_venv_and_node_modules(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("print('hello')")
        for excluded in PACK_EXCLUDES:
            d = tmp_path / excluded
            d.mkdir(exist_ok=True)
            (d / "file.txt").write_text("excluded")
        tarball = pack_workspace(tmp_path, "test-run")
        with tarfile.open(tarball) as archive:
            names = archive.getnames()
        for excluded in PACK_EXCLUDES:
            assert not any(excluded in name for name in names), \
                f"{excluded} should be excluded from tarball"

    def test_excludes_stale_run_artifacts(self, tmp_path: Path):
        factory = tmp_path / ".factory"
        factory.mkdir()
        for artifact in RUN_ARTIFACTS:
            (factory / artifact).write_text("stale")
        (tmp_path / "main.py").write_text("print('hello')")
        tarball = pack_workspace(tmp_path, "test-run")
        with tarfile.open(tarball) as archive:
            names = archive.getnames()
        for artifact in RUN_ARTIFACTS:
            assert not any(name.endswith(artifact) for name in names), \
                f".factory/{artifact} should be excluded"


class TestProvenanceChecks:
    def test_returns_probes(self, tmp_path: Path):
        plan = compose_plan(tmp_path, ["dsh", "run"])
        probes = provenance_checks(plan, tmp_path)
        assert isinstance(probes, list)
        assert all(isinstance(p, tuple) or hasattr(p, 'name') for p in probes)


class TestVerifyGatewayProvider:
    def test_missing_provider_raises_with_fix(self, monkeypatch):
        from refactory_dsh import openshell_backend
        from factory.contained.errors import ContainedError

        def fake_connect(gateway=None):
            class FakeClient:
                def get_provider(self, name):
                    raise RuntimeError("not found")
            return FakeClient()

        monkeypatch.setattr(openshell_backend, "connect", fake_connect)
        with pytest.raises(ContainedError, match="no 'claude-code' provider"):
            openshell_backend.verify_gateway_provider()

    def test_matching_provider_passes(self, monkeypatch):
        from refactory_dsh import openshell_backend
        from factory.contained.openshell import load_provider_profile

        expected = load_provider_profile()

        def fake_connect(gateway=None):
            class FakeClient:
                def get_provider(self, name):
                    return expected
            return FakeClient()

        monkeypatch.setattr(openshell_backend, "connect", fake_connect)
        openshell_backend.verify_gateway_provider()

    def test_divergent_profile_raises(self, monkeypatch):
        from refactory_dsh import openshell_backend
        from factory.contained.errors import ContainedError

        def fake_connect(gateway=None):
            class FakeClient:
                def get_provider(self, name):
                    return {"endpoints": [{"url": "https://wrong.example.com"}]}
            return FakeClient()

        monkeypatch.setattr(openshell_backend, "connect", fake_connect)
        with pytest.raises(ContainedError, match="diverges"):
            openshell_backend.verify_gateway_provider()


class TestWorkersBackendRegistration:
    def test_openshell_in_backends(self):
        from refactory_dsh.workers import BACKENDS, PERMISSIONS
        assert 'openshell' in BACKENDS
        assert 'openshell' in PERMISSIONS

    def test_openshell_route_validates(self):
        from refactory_dsh.workers import validate_route
        route = validate_route({'worker_backend': 'openshell'})
        assert route['worker_backend'] == 'openshell'

    def test_openshell_gateway_field_accepted(self):
        from refactory_dsh.workers import validate_route
        route = validate_route({
            'worker_backend': 'openshell',
            'openshell_gateway': 'local',
        })
        assert route['openshell_gateway'] == 'local'

    def test_openshell_capabilities(self):
        from refactory_dsh.workers import capabilities
        caps = capabilities('openshell')
        assert isinstance(caps, dict)
        assert 'model_selection' in caps
