"""Worker dependency bytes and bindings are an independent benchmark input."""
import json
from benchmarks.create_fidelity import harness


def test_runtime_identity_detects_dependency_bytes_launcher_and_binding_changes(tmp_path):
    root = tmp_path / "runtime"
    store = root / "node_modules/.pnpm"
    package = store / "runtime-package"
    package.mkdir(parents=True)
    (root / "package.json").write_text('{"name":"controlled-runtime"}')
    script = package / "index.js"
    script.write_text("export const version=1")
    binary = root / "node_modules/.bin/dsh"
    binary.parent.mkdir()
    binary.write_text("controlled launcher")
    binding = root / "node_modules/runtime-package"
    binding.symlink_to(package, target_is_directory=True)
    before = harness.runtime_identity(root)
    script.write_text("export const version=2")
    changed = harness.runtime_identity(root)
    assert before["dependencies"] != changed["dependencies"]
    assert before["source"] == changed["source"], "Source identity alone cannot see dependency drift"
    binary.write_text("changed launcher")
    assert changed["launcher"] != harness.runtime_identity(root)["launcher"]
    other = store / "other"
    other.mkdir()
    binding.unlink()
    binding.symlink_to(other, target_is_directory=True)
    assert before["module_bindings"] != harness.runtime_identity(root)["module_bindings"]


def test_post_operation_runtime_drift_cannot_report_completed(tmp_path, monkeypatch):
    project = tmp_path / "dsh-candidate/project"
    project.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    identity = {"declared_revision": "controlled"}
    pin = {"root": str(source), "version": 1}
    config = {"source": str(source), "project": str(project), "identity": identity,
              "runtime_identity": pin, "adapter_root": str(tmp_path / "adapter"),
              "python": "/controlled/python", "policy": str(tmp_path / "policy.sb"),
              "total_timeout_s": 60, "shutdown_grace_s": 1,
              "runtime_state_isolation": {"dsh_home": str(project / "runtime"),
                  "claude_config_dir": str(project / "claude"), "private_tmp": str(project / "tmp")}}
    (project.parent / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(harness, "source_identity", lambda *args: identity)
    snapshots = iter([pin, {**pin, "version": 2}])
    monkeypatch.setattr(harness, "runtime_identity", lambda *args: next(snapshots))
    def controlled_command(*args, **kwargs):
        assert kwargs["env"]["PYTHONPATH"].split(":")[0] == config["adapter_root"]
        return {"status": "completed", "returncode": 0}
    monkeypatch.setattr(harness, "execute_command", controlled_command)
    result = harness.run_lane(tmp_path, "dsh-candidate", "author")
    assert result["status"] == "failed"
    assert "runtime changed" in result["integrity_error"]
    receipt = json.loads((project / "raw/author/runtime-integrity.json").read_text())
    assert receipt["status"] == "failed"
