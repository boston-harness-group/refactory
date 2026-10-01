"""The pinned harness and the harness serving the profile are compared legibly.

Issue #279: every acceptance path booted the pinned `node_modules/.bin/dsh` while
the profile is served by the `dsh` on `PATH`. The strict `TypertCodec` differed
between the two, and the version string alone could not see it — the fork's tree
and the published `0.1.6-alpha.1` both report `0.1.6-alpha.1`. These assert the
version comparison, the contract-face comparison that closes that blind spot, and
that the plain-clone case (no separate `dsh` on `PATH`) stays green.
"""

from __future__ import annotations

from pathlib import Path

from refactory_dsh import harness_check


def _launcher(directory: Path, version: str) -> Path:
    """A standalone launcher that answers `--version` with `version`."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "dsh"
    script.write_text(f"#!/bin/sh\necho {version}\n")
    script.chmod(0o755)
    return script


def _protocol(root: Path, face: str) -> Path:
    """A protocol package under `root` declaring the given strict codec face."""
    package = root / "node_modules" / "@deepseek-ai" / "dsh-typert-protocol"
    (package / "lib" / "types").mkdir(parents=True, exist_ok=True)
    (package / "package.json").write_text("{}")
    (package / "lib" / "types" / "types.d.ts").write_text(
        f"export type TypertCodec = {{\n  readonly mode: 'strict';\n  readonly {face}: unknown;\n}}\n"
    )
    return package


def test_the_protocol_package_is_found_in_a_node_modules_layout(tmp_path):
    launcher = _launcher(tmp_path / "bin", "0.1.6-alpha.1")
    package = _protocol(tmp_path, "schema")
    assert harness_check.protocol_package(launcher) == package


def test_the_protocol_package_is_found_in_a_workspace_layout(tmp_path):
    """A monorepo checkout keeps it at packages/typert/protocol, not node_modules."""
    launcher = _launcher(tmp_path / "apps" / "cli" / "lib", "0.1.6-alpha.1")
    package = tmp_path / "packages" / "typert" / "protocol"
    package.mkdir(parents=True)
    (package / "package.json").write_text("{}")
    assert harness_check.protocol_package(launcher) == package


def test_the_codec_face_reads_both_declarations(tmp_path):
    assert harness_check.codec_face(_protocol(tmp_path / "a", "create")) == "create"
    assert harness_check.codec_face(_protocol(tmp_path / "b", "schema")) == "schema"
    assert harness_check.codec_face(None) is None


def test_a_version_difference_is_a_disagreement(tmp_path):
    pinned = harness_check.inspect("pinned", _launcher(tmp_path / "p", "0.1.6-alpha.1"))
    serving = harness_check.inspect("serving", _launcher(tmp_path / "s", "0.1.6-alpha.2"))
    reason = harness_check.disagreement(pinned, serving)
    assert reason is not None
    assert "0.1.6-alpha.1" in reason and "0.1.6-alpha.2" in reason


def test_the_same_version_with_a_different_face_is_a_disagreement(tmp_path):
    """The #278 blind spot: equal versions, different strict codec contracts."""
    pinned = harness_check.inspect("pinned", _launcher(tmp_path / "p", "0.1.6-alpha.1"))
    _protocol(tmp_path / "p", "schema")
    serving = harness_check.inspect("serving", _launcher(tmp_path / "s", "0.1.6-alpha.1"))
    _protocol(tmp_path / "s", "create")
    pinned = harness_check.inspect("pinned", pinned.launcher)
    serving = harness_check.inspect("serving", serving.launcher)
    reason = harness_check.disagreement(pinned, serving)
    assert reason is not None, "the same version string must not hide a contract difference"
    assert "same version" in reason
    assert "schema" in reason and "create" in reason


def test_matching_versions_and_faces_agree(tmp_path):
    _protocol(tmp_path / "p", "schema")
    _protocol(tmp_path / "s", "schema")
    pinned = harness_check.inspect("pinned", _launcher(tmp_path / "p", "0.1.6-alpha.1"))
    serving = harness_check.inspect("serving", _launcher(tmp_path / "s", "0.1.6-alpha.1"))
    assert harness_check.disagreement(pinned, serving) is None


def test_the_same_launcher_is_the_same_harness(tmp_path):
    launcher = _launcher(tmp_path / "bin", "0.1.6-alpha.1")
    one = harness_check.inspect("pinned", launcher)
    two = harness_check.inspect("serving", launcher)
    assert harness_check.disagreement(one, two) is None


def test_a_serving_launcher_that_does_not_answer_is_reported(tmp_path):
    pinned = harness_check.inspect("pinned", _launcher(tmp_path / "p", "0.1.6-alpha.1"))
    serving = harness_check.Harness("serving", tmp_path / "missing", None, None)
    reason = harness_check.disagreement(pinned, serving)
    assert reason is not None and "did not answer" in reason


def test_check_names_both_sides_of_the_fork_case(monkeypatch, tmp_path):
    _protocol(tmp_path / "pinned", "schema")
    _protocol(tmp_path / "serving", "create")
    pinned = _launcher(tmp_path / "pinned", "0.1.6-alpha.1")
    serving = _launcher(tmp_path / "serving", "0.1.6-alpha.1")
    monkeypatch.setattr(harness_check, "pinned_launcher", lambda: pinned)
    monkeypatch.setattr(harness_check, "serving_launcher", lambda: serving)
    outcome = harness_check.check()
    assert not outcome.ok
    assert "0.1.6-alpha.1" in outcome.reason
    assert "schema" in outcome.reason and "create" in outcome.reason


def test_check_is_green_when_no_serving_harness_exists(monkeypatch, tmp_path):
    _protocol(tmp_path / "pinned", "schema")
    monkeypatch.setattr(harness_check, "pinned_launcher", lambda: _launcher(tmp_path / "pinned", "0.1.6-alpha.1"))
    monkeypatch.setattr(harness_check, "serving_launcher", lambda: None)
    outcome = harness_check.check()
    assert outcome.ok
    assert outcome.serving is None


def test_serving_launcher_prefers_the_environment_override(monkeypatch, tmp_path):
    fake = tmp_path / "elsewhere" / "dsh"
    monkeypatch.setenv(harness_check.SERVING_DSH_ENV, str(fake))
    assert harness_check.serving_launcher() == fake


def test_serving_launcher_falls_back_to_path(monkeypatch):
    monkeypatch.delenv(harness_check.SERVING_DSH_ENV, raising=False)
    monkeypatch.setattr(harness_check.shutil, "which", lambda _name: "/opt/dsh")
    assert harness_check.serving_launcher() == Path("/opt/dsh")


def test_serving_launcher_is_none_without_path(monkeypatch):
    monkeypatch.delenv(harness_check.SERVING_DSH_ENV, raising=False)
    monkeypatch.setattr(harness_check.shutil, "which", lambda _name: None)
    assert harness_check.serving_launcher() is None


def test_main_exits_nonzero_and_prints_the_reason(monkeypatch, capsys):
    outcome = harness_check.HarnessCheck(
        pinned=harness_check.Harness("pinned", Path("/p/dsh"), "1.0.0", "schema"),
        serving=harness_check.Harness("serving", Path("/s/dsh"), "1.0.0", "create"),
        reason="the two disagree",
    )
    monkeypatch.setattr(harness_check, "check", lambda: outcome)
    assert harness_check.main([]) == 1
    assert "the two disagree" in capsys.readouterr().out


def test_main_is_green_without_a_serving_harness(monkeypatch, capsys):
    outcome = harness_check.HarnessCheck(
        pinned=harness_check.Harness("pinned", Path("/p/dsh"), "1.0.0", "schema"),
        serving=None,
        reason=None,
    )
    monkeypatch.setattr(harness_check, "check", lambda: outcome)
    assert harness_check.main([]) == 0
    assert "serving harness" in capsys.readouterr().out


def test_main_accepts_argv_for_module_invocation(monkeypatch, capsys):
    """`python -m refactory_dsh.harness_check` passes sys.argv; it must be ignored."""
    outcome = harness_check.HarnessCheck(
        pinned=harness_check.Harness("pinned", Path("/p/dsh"), "1.0.0", "schema"),
        serving=None,
        reason=None,
    )
    monkeypatch.setattr(harness_check, "check", lambda: outcome)
    assert harness_check.main(["--unused"]) == 0
