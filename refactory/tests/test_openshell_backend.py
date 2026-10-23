"""Tests for the OpenShell sandbox backend composition layer.

These test the pure plan-building and workspace-packing logic — no
gateway, no SDK, no sandbox.  The execution path (``run_in_openshell``)
is integration-tested when the credential adapter lands.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from refactory_dsh.openshell_backend import (
    PACK_EXCLUDES,
    PROVIDER_NAME,
    REQUIRED_DOWNLOADS,
    RUN_ARTIFACTS,
    _ENV_ALLOWLIST,
    _sandbox_env,
    _terminate_remote,
    _translate_argv,
    compose_plan,
    download_artifacts,
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

    def test_env_preserves_allowed_values(self, tmp_path: Path):
        plan = compose_plan(
            tmp_path, ["dsh", "run"],
            env={"REFACTORY_WORKER_CONTEXT": "test", "LANG": "en_US.UTF-8"},
        )
        assert plan.env["REFACTORY_WORKER_CONTEXT"] == "test"
        assert plan.env["LANG"] == "en_US.UTF-8"

    def test_env_strips_secrets(self, tmp_path: Path):
        plan = compose_plan(
            tmp_path, ["dsh", "run"],
            env={"ANTHROPIC_API_KEY": "sk-test", "AWS_SECRET_ACCESS_KEY": "x"},
        )
        assert "ANTHROPIC_API_KEY" not in plan.env
        assert "AWS_SECRET_ACCESS_KEY" not in plan.env

    def test_no_warnings_without_secrets(self, tmp_path: Path):
        plan = compose_plan(tmp_path, ["dsh", "run"], env={"LANG": "C"})
        assert not plan.warnings

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


class TestSandboxEnv:
    def test_allowlisted_keys_pass_through(self):
        env = _sandbox_env({"LANG": "en_US", "HOME": "/home/user", "TZ": "UTC"})
        assert env["LANG"] == "en_US"
        assert env["HOME"] == "/home/user"
        assert env["TZ"] == "UTC"

    def test_refactory_prefix_passes(self):
        env = _sandbox_env({"REFACTORY_FOO": "bar", "REFACTORY_BUDGET_LEDGER": "/tmp/b"})
        assert env["REFACTORY_FOO"] == "bar"
        assert env["REFACTORY_BUDGET_LEDGER"] == "/tmp/b"

    def test_secrets_are_excluded(self):
        env = _sandbox_env({
            "ANTHROPIC_API_KEY": "sk-test",
            "AWS_SECRET_ACCESS_KEY": "x",
            "GITHUB_TOKEN": "ghp_abc",
            "OPENAI_API_KEY": "sk-oa",
        })
        assert "ANTHROPIC_API_KEY" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env
        assert "GITHUB_TOKEN" not in env
        assert "OPENAI_API_KEY" not in env

    def test_defaults_applied(self):
        env = _sandbox_env({})
        assert env["PYTHONUNBUFFERED"] == "1"
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"

    def test_user_override_of_defaults(self):
        env = _sandbox_env({"PYTHONUNBUFFERED": "0"})
        assert env["PYTHONUNBUFFERED"] == "0"


class TestTranslateArgv:
    def test_host_workdir_paths_translated(self, tmp_path: Path):
        workdir = tmp_path / "myproject"
        workdir.mkdir()
        argv = [str(workdir / "run.py"), "--config", str(workdir / "config.json")]
        result = _translate_argv(argv, workdir, "/workspace/myproject")
        assert result == ["/workspace/myproject/run.py", "--config", "/workspace/myproject/config.json"]

    def test_non_workdir_paths_kept(self, tmp_path: Path):
        workdir = tmp_path / "myproject"
        workdir.mkdir()
        argv = ["dsh", "--flag", "value"]
        result = _translate_argv(argv, workdir, "/workspace/myproject")
        assert result == ["dsh", "--flag", "value"]

    def test_python_path_simplified(self, tmp_path: Path):
        workdir = tmp_path / "myproject"
        workdir.mkdir()
        argv = ["/usr/local/bin/python3", "-m", "dsh"]
        result = _translate_argv(argv, workdir, "/workspace/myproject")
        assert result[0] == "python3"

    def test_dsh_path_simplified(self, tmp_path: Path):
        workdir = tmp_path / "myproject"
        workdir.mkdir()
        argv = ["/home/user/.local/bin/dsh", "run"]
        result = _translate_argv(argv, workdir, "/workspace/myproject")
        assert result[0] == "dsh"


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

    def test_excludes_dotenv_files(self, tmp_path: Path):
        (tmp_path / ".env").write_text("SECRET=value")
        (tmp_path / ".env.local").write_text("LOCAL_SECRET=value")
        (tmp_path / "main.py").write_text("print('hello')")
        tarball = pack_workspace(tmp_path, "test-run")
        with tarfile.open(tarball) as archive:
            names = archive.getnames()
        assert not any(".env" in name for name in names)


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

        def fake_run(argv, **kwargs):
            return MagicMock(returncode=0, stdout='[]', stderr='')

        monkeypatch.setattr(openshell_backend.subprocess, "run", fake_run)
        with pytest.raises(ContainedError, match="no 'claude-code' provider"):
            openshell_backend.verify_gateway_provider()

    def test_matching_provider_passes(self, monkeypatch):
        from refactory_dsh import openshell_backend
        from factory.contained.openshell import load_provider_profile

        expected = load_provider_profile()

        call_count = {"n": 0}
        def fake_run(argv, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps([{"name": "claude-code", "type": "claude-code"}]),
                    stderr='',
                )
            return MagicMock(
                returncode=0,
                stdout=json.dumps(expected),
                stderr='',
            )

        monkeypatch.setattr(openshell_backend.subprocess, "run", fake_run)
        openshell_backend.verify_gateway_provider()

    def test_divergent_profile_raises(self, monkeypatch):
        from refactory_dsh import openshell_backend
        from factory.contained.errors import ContainedError

        call_count = {"n": 0}
        def fake_run(argv, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps([{"name": "claude-code", "type": "claude-code"}]),
                    stderr='',
                )
            return MagicMock(
                returncode=0,
                stdout=json.dumps({"endpoints": [{"host": "wrong.example.com", "port": 443}]}),
                stderr='',
            )

        monkeypatch.setattr(openshell_backend.subprocess, "run", fake_run)
        with pytest.raises(ContainedError, match="differs"):
            openshell_backend.verify_gateway_provider()

    def test_cli_failure_raises(self, monkeypatch):
        from refactory_dsh import openshell_backend
        from factory.contained.errors import ContainedError

        def fake_run(argv, **kwargs):
            return MagicMock(returncode=1, stdout='', stderr='connection refused')

        monkeypatch.setattr(openshell_backend.subprocess, "run", fake_run)
        with pytest.raises(ContainedError, match="cannot list gateway providers"):
            openshell_backend.verify_gateway_provider()


class TestTerminateRemote:
    def test_sends_kill_via_exec(self):
        from factory.contained.openshell import OpenShellPlan
        session = MagicMock()
        plan = OpenShellPlan(
            name="test", image="img", project_dir="/workspace/test",
            env={}, labels={}, provider="claude-code", policy_path=None,
            run_command="cmd", factory_command="cmd",
        )
        with patch("refactory_dsh.openshell_backend.exec_argv") as mock_exec:
            _terminate_remote(session, plan)
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            argv = call_args[0][1]
            assert "kill" in " ".join(argv)

    def test_swallows_errors(self):
        from factory.contained.openshell import OpenShellPlan
        session = MagicMock()
        plan = OpenShellPlan(
            name="test", image="img", project_dir="/workspace/test",
            env={}, labels={}, provider="claude-code", policy_path=None,
            run_command="cmd", factory_command="cmd",
        )
        from factory.contained.errors import ContainedError
        with patch("refactory_dsh.openshell_backend.exec_argv", side_effect=ContainedError("fail")):
            _terminate_remote(session, plan)


class TestDownloadArtifacts:
    def test_downloads_required_paths(self, tmp_path: Path):
        from factory.contained.openshell import OpenShellPlan
        plan = OpenShellPlan(
            name="test", image="img", project_dir="/workspace/test",
            env={}, labels={}, provider="claude-code", policy_path=None,
            run_command="cmd", factory_command="cmd",
        )
        call_log = []
        def fake_run(argv, **kwargs):
            call_log.append(argv)
            return MagicMock(returncode=0)

        with patch("refactory_dsh.openshell_backend.subprocess.run", fake_run):
            failures = download_artifacts(plan, tmp_path)

        assert failures == []
        assert len(call_log) == len(REQUIRED_DOWNLOADS)

    def test_reports_failed_downloads(self, tmp_path: Path):
        from factory.contained.openshell import OpenShellPlan
        plan = OpenShellPlan(
            name="test", image="img", project_dir="/workspace/test",
            env={}, labels={}, provider="claude-code", policy_path=None,
            run_command="cmd", factory_command="cmd",
        )
        def fake_run(argv, **kwargs):
            return MagicMock(returncode=1)

        with patch("refactory_dsh.openshell_backend.subprocess.run", fake_run):
            failures = download_artifacts(plan, tmp_path)

        assert len(failures) == len(REQUIRED_DOWNLOADS)


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

    def test_openshell_write_patch_uses_dsh(self, tmp_path: Path):
        from refactory_dsh.workers import write_patch
        patch_path = tmp_path / "test.patch.yml"
        route = {'worker_backend': 'openshell', 'permission_mode': 'default'}
        write_patch(patch_path, tmp_path, route)
        assert patch_path.exists()
        content = json.loads(patch_path.read_text())
        assert isinstance(content, list)


class TestWorkerConfigAcceptsOpenshellFields:
    def test_openshell_gateway_in_route_fields(self):
        from refactory_dsh.worker_config import ROUTE_FIELDS
        assert "openshell_gateway" in ROUTE_FIELDS
        assert "openshell_image" in ROUTE_FIELDS
