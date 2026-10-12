"""Native adapter containment and immutable artifact boundary regressions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from benchmarks.create_fidelity import harness, lane


def controlled_os_boundary(monkeypatch):
    # Replace only external OS boundary availability. Native registry, imports,
    # executor, operations, child process, model guards and fingerprints are real.
    monkeypatch.setattr(lane, "_offline_argv", lambda code: ([sys.executable, "-c", code],
        {"status": "passed", "kind": "controlled OS-boundary availability; Python guards active"}))


def package(project, *, import_mutates=False, operation="normal"):
    root = project / ".factory/packages" / lane.NAME
    root.mkdir(parents=True)
    mutation = "from pathlib import Path\nPath(__file__).with_name('import-mutation.txt').write_text('changed before inspection')\n" if import_mutates else ""
    (root / "__init__.py").write_text(mutation + f"from refactory_dsh import nn\nfrom .mode import Bookings\nPACKAGE=nn.Package(name={lane.NAME!r},description='Offline bookings',modes=(Bookings,),ops=('refactory_packages.{lane.NAME}.ops',))\n")
    (root / "mode.py").write_text(f"""from refactory_dsh import nn
class Bookings(nn.Mode):
    name='run'
    description='Compute runtime booking conflicts offline'
    unscored_reason='Produces deterministic output'
    @classmethod
    def declare(cls): return {{'bookings_path':nn.Knob('', 'str')}}
    def build(self):
        graph=nn.Graph()
        graph.add(nn.Node('compute','op',call='refactory_packages.{lane.NAME}.ops:compute',knobs=('bookings_path',),writes=('conflicts.json',)))
        return graph
""")
    extra = ""
    if operation == "model":
        extra = " from refactory_dsh.model import DshModelClient\n DshModelClient().complete('An unauthorized paid computation')\n"
    elif operation == "network":
        extra = " import socket\n socket.socket()\n"
    elif operation == "mutation":
        extra = " Path(__file__).with_name('invocation-mutation.txt').write_text('changed during computation')\n"
    (root / "ops.py").write_text("import json\nfrom pathlib import Path\ndef compute(ctx):\n" + extra + " rows=json.loads(Path(ctx.knob('bookings_path')).read_text())\n if len({row['id'] for row in rows}) != len(rows): raise ValueError('duplicate booking ID')\n pairs=sorted(sorted((a['id'],b['id'])) for i,a in enumerate(rows) for b in rows[i+1:] if a['room']==b['room'] and max(a['start'],b['start'])<min(a['end'],b['end']))\n ctx.write_json('conflicts.json', {'conflicts':pairs})\n")
    inputs = project / "inputs"
    inputs.mkdir()
    a = {"id": "a", "room": "A", "start": 0, "end": 10}
    b = {"id": "b", "room": "A", "start": 5, "end": 15}
    for name, rows in (("overlap", [a, b]), ("touching", [a, {**b, "start": 10, "end": 20}]), ("invalid-duplicate", [a, a])):
        (inputs / f"{name}.json").write_text(json.dumps(rows))
    return root, inputs


def test_native_adapter_freezes_before_import_and_uses_real_native_runtime(tmp_path, monkeypatch):
    controlled_os_boundary(monkeypatch)
    root, inputs = package(tmp_path)
    result = lane.dsh_native(tmp_path, inputs)
    assert result["validated"], result
    assert result["package_before_import"] == result["package_after_inspection"] == result["package_after_native"]
    assert result["package_unchanged"]
    assert result["invocations"][0]["exit_ok"]
    assert result["invocations"][0]["output"] == {"conflicts": [["a", "b"]]}
    assert result["invocations"][1]["output"] == {"conflicts": []}
    assert result["invocations"][2]["exit_ok"] is False
    assert result["invocations"][2]["output_exists"] is False
    assert all(invocation["package_before"] == result["package_before_import"] for invocation in result["invocations"])
    assert lane._native_complete(result)


def test_import_time_mutation_fails_before_native_invocation(tmp_path, monkeypatch):
    controlled_os_boundary(monkeypatch)
    root, inputs = package(tmp_path, import_mutates=True)
    result = lane.dsh_native(tmp_path, inputs)
    assert result["status"] == "failed"
    assert result["validated"] is False
    assert not result["package_unchanged"]
    assert result["invocations"] == []
    assert result["package_before_import"] != result["package_after_inspection"]
    assert not lane._native_complete(result)


@pytest.mark.parametrize("operation", ["model", "network", "mutation"])
def test_internal_model_network_or_package_mutation_cannot_pass_native_result(tmp_path, monkeypatch, operation):
    controlled_os_boundary(monkeypatch)
    root, inputs = package(tmp_path, operation=operation)
    result = lane.dsh_native(tmp_path, inputs)
    assert result["validated"], result
    assert all(invocation.get("exit_ok") is False for invocation in result["invocations"])
    assert not lane._native_complete(result)
    if operation == "mutation":
        assert result["status"] == "failed" and not result["package_unchanged"]
    else:
        assert "must not invoke a model" in result["invocations"][0]["error"] if operation == "model" else "remain offline" in result["invocations"][0]["error"]


def test_no_os_boundary_is_unavailable_and_does_not_import_generated_package(tmp_path, monkeypatch):
    root, inputs = package(tmp_path, import_mutates=True)
    monkeypatch.setattr(lane, "_offline_argv", lambda code: (None, {"status": "unavailable", "reason": "OS boundary rejected"}))
    result = lane.dsh_native(tmp_path, inputs)
    assert result["status"] == "unavailable"
    assert result["native_boundary"]["status"] == "unavailable"
    assert not (root / "import-mutation.txt").exists()
    assert result["invocations"] == []
    assert not lane._native_complete(result)


@pytest.mark.parametrize("status", ["failed", "unavailable"])
def test_native_cli_returns_nonzero_for_failed_or_unavailable_verification(tmp_path, monkeypatch, status):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"project": str(tmp_path), "inputs": str(tmp_path / "inputs"), "lane": "dsh-candidate"}))
    monkeypatch.setattr(sys, "argv", ["lane", "native", str(config)])
    # Only external adapter result is substituted; command/result persistence
    # and the final success criterion remain the real CLI implementation.
    monkeypatch.setattr(lane, "dsh_native", lambda *args: {"status": status, "error": "verification unavailable or failed"})
    assert lane.main() == 1
    assert json.loads((tmp_path / "native-result.json").read_text())["status"] == status


def test_native_launch_applies_network_denial_at_lane_owner(tmp_path, monkeypatch):
    directory = tmp_path / "dsh-candidate"
    directory.mkdir()
    project = directory / "project"
    project.mkdir()
    policy = directory / "isolation.sb"
    policy.write_text('(version 1)\n(allow default)\n(deny file-read* (subpath "/peer"))\n')
    config = {"source": str(tmp_path / "source"), "project": str(project), "identity": {"declared_revision": "fixed"},
              "python": sys.executable, "adapter_root": str(tmp_path), "policy": str(policy), "total_timeout_s": 1000, "shutdown_grace_s": 15,
              "runtime_state_isolation": {"dsh_home": str(project / "runtime/dsh"), "claude_config_dir": str(project / "runtime/claude"), "private_tmp": str(project / "tmp")}}
    (directory / "config.json").write_text(json.dumps(config))
    # Fake only external git observation/command I/O; policy construction is real.
    monkeypatch.setattr(harness, "source_identity", lambda *args: config["identity"])
    commands = []
    monkeypatch.setattr(harness, "execute_command", lambda argv, **kwargs: commands.append(argv) or {"status": "completed"})
    harness.run_lane(tmp_path, "dsh-candidate", "native")
    applied = Path(commands[0][2])
    assert applied.name == "native-isolation.sb"
    assert policy.read_text() in applied.read_text()
    assert "(deny network*)" in applied.read_text()


def test_real_inherited_kernel_boundary_and_native_model_guard(tmp_path):
    _, boundary = lane._offline_argv("print('kernel boundary probe')")
    if boundary["status"] != "passed":
        pytest.skip(boundary.get("reason", "No inherited OS network boundary available"))
    _, inputs = package(tmp_path, operation="model")
    result = lane.dsh_native(tmp_path, inputs)
    assert result["native_boundary"]["status"] == "passed"
    assert result["native_boundary"]["parent_checks"] == result["native_boundary"]["child_checks"] == {"network-outbound": 1, "network-inbound": 1}
    assert result["invocations"][0]["exit_ok"] is False
    assert "must not invoke a model" in result["invocations"][0]["error"]
