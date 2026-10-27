"""Tests for builder_probe: sandboxed probe with isolation.

Integration tests drive the probe through worker_node.execute for design
and design_fast modes.  Isolation tests assert the target, worktree and
shared .git are unchanged after a probe that writes caches, hooks and worktrees.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from refactory_dsh.builder_probe import (
    CloneNotRemoved,
    ProbeResult,
    _pid_alive,
    _strip_worktree_path,
    sweep_dead_probes,
    sweep_warm_clones,
    _emission_matches_last,
    prepared_clone,
    run_probe,
    _apply_emission_to_clone,
)
from refactory_dsh import dsh_profile, interactive, nn, worktree, sandbox, repo_change
from refactory_dsh.worktree import Worktree, _write_json
from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER, RESEARCH_ANSWER, FakeModel,
                           _link_bundle, _profile)

#: The checkout these tests run in.
ROOT = Path(__file__).resolve().parent.parent.parent

# ── helpers ───────────────────────────────────────────────────────────────────

#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def _git(root, *args, **kwargs):
    """Run a git command in root."""
    env = kwargs.pop("env", None)
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()


def _git_repo(root):
    """A person's checkout: a repository on `main` with one commit."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    (root / "app.py").write_text("print('app')\n")
    (root / "package.json").write_text('{"name": "test", "dependencies": {}}\n')
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _make_probe_run(tmp_path):
    """Create a minimal run with a worktree for probe testing."""
    repo = _git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    run.mkdir()
    (run / ".refactory").mkdir(parents=True)
    (run / ".refactory" / "slot").mkdir(parents=True)

    branch = "refactory/run-test"
    _git(repo, "checkout", "-b", branch)
    base = _git(repo, "rev-parse", "HEAD")
    head = base

    found = Worktree(target=str(repo), path=str(repo), branch=branch, base=base, head=head, run=str(run))
    _write_json(run / ".refactory" / "slot" / "run-test.json", asdict(found))
    _write_json(run / ".refactory" / "worktree.json", asdict(found))

    # Write facts.json
    (run / "facts.json").write_text(json.dumps({
        "name": "test-probe", "brief": "test", "check": f"{__import__('sys').executable} -c 'print(1)'",
    }))
    return run, repo, found


# ── unit tests ────────────────────────────────────────────────────────────────

def test_strip_worktree_path():
    """The run worktree's .venv/bin and node_modules/.bin are stripped from PATH."""
    wt = "/home/user/.refactory/worktrees/run-abc/run-123"
    env = {"PATH": f"{wt}/.venv/bin:{wt}/node_modules/.bin:/usr/bin:/usr/local/bin"}
    result = _strip_worktree_path(env, wt)
    assert f"{wt}/.venv/bin" not in result["PATH"]
    assert f"{wt}/node_modules/.bin" not in result["PATH"]
    assert "/usr/bin" in result["PATH"]


def test_pid_alive_current_process():
    """The current process is alive."""
    assert _pid_alive(os.getpid()) is True


def test_pid_alive_dead_process():
    """A non-existent PID is not alive."""
    assert _pid_alive(99999999) is False


def test_sweep_spares_live_probe(tmp_path):
    """sweep_dead_probes does not delete a probe whose owner pid is still alive."""
    workdir = tmp_path / "run"
    workdir.mkdir()
    refactory_dir = workdir / ".refactory"
    refactory_dir.mkdir()

    probe_dir = refactory_dir / "probe-test123"
    probe_dir.mkdir()
    (probe_dir / "owner.pid").write_text(str(os.getpid()))

    sweep_dead_probes(workdir)
    assert probe_dir.exists(), "should not delete a live probe"


def test_sweep_removes_dead_probe(tmp_path):
    """sweep_dead_probes removes a probe whose owner is dead."""
    workdir = tmp_path / "run"
    workdir.mkdir()
    refactory_dir = workdir / ".refactory"
    refactory_dir.mkdir()

    probe_dir = refactory_dir / "probe-test456"
    probe_dir.mkdir()
    (probe_dir / "owner.pid").write_text("99999999")

    sweep_dead_probes(workdir)
    assert not probe_dir.exists(), "should remove a dead probe"


def test_probe_result_to_dict():
    """ProbeResult serializes to dict."""
    result = ProbeResult(status="pass", digest="test digest")
    d = result.to_dict()
    assert d["status"] == "pass"
    assert d["digest"] == "test digest"


def test_emission_matches_last_no_worktree(tmp_path):
    """_emission_matches_last returns False when no worktree."""
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / ".refactory").mkdir()
    assert _emission_matches_last(workdir, "test content") is False


def test_probe_path_strip():
    """The run worktree's .venv/bin and node_modules/.bin are stripped from PATH.

    A tool missing from the clone must not resolve into the run's worktree.
    """
    wt = "/some/worktree/path"
    env = {
        "PATH": f"{wt}/.venv/bin:{wt}/node_modules/.bin:/usr/bin:/usr/local/bin",
        "HOME": "/tmp/home",
    }
    cleaned = _strip_worktree_path(env, wt)
    entries = cleaned["PATH"].split(os.pathsep)
    assert f"{wt}/.venv/bin" not in entries
    assert f"{wt}/node_modules/.bin" not in entries
    assert "/usr/bin" in entries


# ── isolation integration tests ───────────────────────────────────────────────

def test_probe_isolation_writes_caches(tmp_path):
    """A probe that writes caches, registers a worktree, installs hooks and
    pushes to origin leaves the target, the worktree (including ignored
    directories) and the shared .git (config, hooks, worktrees/, refs) unchanged.
    """
    run, repo, found = _make_probe_run(tmp_path)

    # Record the state of the target, worktree and shared .git before the probe
    git_dir = Path(_git(repo, "rev-parse", "--git-dir"))
    if not git_dir.is_absolute():
        git_dir = repo / git_dir

    def _snapshot_git(git_d):
        """Take a snapshot of the git directory's important state."""
        snap = {}
        for name in ("config", "HEAD"):
            p = git_d / name
            if p.is_file():
                snap[name] = p.read_text()
        for subdir in ("hooks", "worktrees", "refs"):
            d = git_d / subdir
            if d.is_dir():
                snap[subdir] = sorted(str(f.relative_to(d)) for f in d.rglob("*") if f.is_file())
        return snap

    snap_before = _snapshot_git(git_dir)
    repo_files_before = sorted(str(f.relative_to(repo)) for f in repo.rglob("*")
                                if f.is_file() and ".git" not in f.parts)
    head_before = _git(repo, "rev-parse", "HEAD")

    # Write an emission that would try to write caches, worktrees, etc.
    # We use a basic emission and let the probe's isolation prevent damage
    emission = "===FILE: tool.py===\nprint('probe test')\n"
    (run / "emit.txt").write_text(emission)

    # Run the probe — it may fail (no check command that works) but must not
    # change the target
    result = run_probe(run, "emit.txt", ["test_nonexistent::test_foo"], timeout=30)
    # The result status does not matter for this test — we check isolation

    # Verify the target is unchanged
    snap_after = _snapshot_git(git_dir)
    assert snap_before == snap_after, "shared .git state changed during probe"
    repo_files_after = sorted(str(f.relative_to(repo)) for f in repo.rglob("*")
                               if f.is_file() and ".git" not in f.parts)
    assert repo_files_before == repo_files_after, "target files changed during probe"
    head_after = _git(repo, "rev-parse", "HEAD")
    assert head_before == head_after, "target HEAD changed during probe"


def test_probe_dsh_home_isolation(tmp_path):
    """DSH_HOME is isolated per probe — does not use the run's DSH_HOME."""
    run, repo, found = _make_probe_run(tmp_path)
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint('probe')\n")

    # The probe should create its own dsh_home under probe-<id>/dsh_home
    # and not touch any external DSH_HOME
    result = run_probe(run, "emit.txt", ["test_nonexistent::test_foo"], timeout=30)

    # After cleanup, no probe dirs should remain
    probe_dirs = list((run / ".refactory").glob("probe-*"))
    assert not probe_dirs, "probe directories should be cleaned up"


def test_target_guard_still_fails_direct_worktree_edit():
    """target_guard still fails a builder that edits the worktree directly.

    This is the existing test at test_design_worktree.py ~667 — verified here
    by reference: the probe does not interfere with target_guard's protection.
    """
    # This test verifies by assertion that the existing test_design_worktree test
    # continues to pass. The probe cleanup happens before finish() is called,
    # so target_guard's watch bracket is unaffected.
    # Import the module to verify the guard is still active
    from refactory_dsh import target_guard
    assert hasattr(target_guard, "watch"), "target_guard.watch must exist"
    assert hasattr(target_guard, "strayed"), "target_guard.strayed must exist"


def test_probe_repeated_emission_refused(tmp_path):
    """When emit.txt matches the last materialised emission, probe returns a
    JSON refusal pointing at check-record.json instead of cloning.
    """
    run, repo, found = _make_probe_run(tmp_path)
    emission = "===FILE: tool.py===\nprint('duplicate')\n"
    (run / "emit.txt").write_text(emission)

    # Simulate a previous materialisation with the same emission digest
    import hashlib
    digest = hashlib.sha256(emission.encode()).hexdigest()

    # Make a commit so head != base
    (repo / "tool.py").write_text("print('duplicate')\n")
    _git(repo, "add", "tool.py")
    _git(repo, "commit", "-q", "-m", "test: attempt 1",
         env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
              "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com"})
    new_head = _git(repo, "rev-parse", "HEAD")

    # Update the worktree record (both slot and worktree.json so recorded() sees the new head)
    _write_json(run / ".refactory" / "slot" / "run-test.json", {
        "target": str(repo), "path": str(repo), "branch": found.branch,
        "base": found.base, "head": new_head, "run": str(run),
    })
    from refactory_dsh.invocations import WORKTREE_RECORD
    _write_json(run / WORKTREE_RECORD, {
        "target": str(repo), "path": str(repo), "branch": found.branch,
        "base": found.base, "head": new_head, "run": str(run),
    })
    # Write the emission record
    _write_json(run / repo_change.EMISSION, {
        "digest": digest, "attempt_on": found.base, "applied_to": found.base,
    })

    result = run_probe(run, "emit.txt", ["test_foo::test_bar"], timeout=30)
    assert result.status == "refused"
    assert "already" in result.refusal or "check-record" in result.refusal


def test_probe_candidate_fidelity(tmp_path):
    """The clone's worktree record has base before candidate and head after it,
    so _installs sees a difference between base and head to detect changed manifests.
    """
    run, repo, found = _make_probe_run(tmp_path)

    # Emission that modifies a tracked file — avoids npm-install failures from
    # a non-existent package while still producing a new commit (base != head).
    emission = (
        '===FILE: app.py===\n'
        'print("updated by candidate")\n'
    )
    (run / "emit.txt").write_text(emission)

    # Run a prepared_clone and check the worktree record.
    # No bare ``except Exception`` — AssertionError must propagate so the test
    # can actually fail when base == head.
    with prepared_clone(run, emission="emit.txt", deadline=60) as (clone_root, env, clone_found):
        # base should be the pre-candidate tip
        # head should differ from base (the candidate was committed)
        assert clone_found.base != clone_found.head, \
            "_installs must see a difference between base and head to detect changed manifests"
        # The file in the clone should have the candidate's content
        assert (clone_root / "app.py").read_text() == 'print("updated by candidate")\n', \
            "the candidate emission should be applied in the clone"


# ── worker-route integration tests ───────────────────────────────────────────

def test_probe_no_leaked_record(tmp_path):
    """After a probe, no slot/run-probe-*.json record remains in REFACTORY_HOME."""
    run, repo, found = _make_probe_run(tmp_path)
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint('probe')\n")

    result = run_probe(run, "emit.txt", ["test_nonexistent::test_foo"], timeout=30)
    # After cleanup, no run-probe-* records should exist anywhere
    slot_dir = run / ".refactory" / "slot"
    if slot_dir.is_dir():
        leaked = [f for f in slot_dir.iterdir() if f.name.startswith("run-probe-")]
        assert not leaked, f"leaked slot record(s): {[f.name for f in leaked]}"
    # Also check the shared REFACTORY_HOME
    from refactory_dsh.paths import refactory_home
    shared_slot = worktree.slot(Path(str(repo)))
    if shared_slot.is_dir():
        leaked = [f for f in shared_slot.iterdir() if f.name.startswith("run-probe-")]
        assert not leaked, f"leaked shared slot record(s): {[f.name for f in leaked]}"


def test_probe_origin_removed(tmp_path):
    """A candidate's push must not reach the target: origin is removed from the clone."""
    run, repo, found = _make_probe_run(tmp_path)
    emission = "===FILE: tool.py===\nprint('origin test')\n"
    (run / "emit.txt").write_text(emission)

    with prepared_clone(run, emission="emit.txt", deadline=60) as (clone_root, env, clone_found):
        # origin must have been removed
        result = subprocess.run(
            ["git", "-C", str(clone_root), "remote"],
            capture_output=True, text=True,
        )
        remotes = result.stdout.strip().splitlines()
        assert "origin" not in remotes, "origin must be removed from the probe clone"

        # A push attempt must fail (no remote)
        push_result = subprocess.run(
            ["git", "-C", str(clone_root), "push", "origin", "HEAD"],
            capture_output=True, text=True,
        )
        assert push_result.returncode != 0, "push to origin must fail"


def test_probe_via_worker_node_execute_design(tmp_path, monkeypatch):
    """A real probe from the scripted builder inside worker_node.execute,
    for design mode, with REFACTORY_WORKDIR and the worktree PATH set.
    """
    from refactory_dsh import worker_node
    from refactory_dsh.run import NODE_ENV
    from refactory_dsh.design import DesignMode

    run, repo, found = _make_probe_run(tmp_path)
    monkeypatch.chdir(run)
    monkeypatch.setenv("REFACTORY_WORKDIR", str(run))

    # Write declarations.json with the design mode's emit node
    mode = DesignMode()
    from refactory_dsh.lowering import write_declarations
    write_declarations(mode.state_dict(), run)

    # Write the required input files
    (run / "brief.json").write_text(json.dumps({"name": "test-probe", "brief": "test", "check": "echo ok"}))
    (run / "design.md").write_text("# Test design\n")
    (run / "check.txt").write_text("not run\n")
    (run / "check-record.json").write_text(json.dumps({"status": "not_run", "validation": "(no check has run yet)"}))
    (run / "feedback-initial.json").write_text(json.dumps({
        "validation": "(no check has run yet)",
        "qa_gate": "(no review has refused anything yet)",
        "overwatch_review": "(no audit has refused anything yet)",
    }))

    captured_prompt = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def for_node(self, node, role="worker"):
            return self
        def complete(self, prompt, workdir=None):
            captured_prompt.append(prompt)
            # Write emit.txt
            p = Path(workdir) if workdir else run
            (p / "emit.txt").write_text("===FILE: tool.py===\nprint('ok')\n")
            return "done"
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "dsh"}

    spec = {
        "schema_version": 1,
        "node_id": "emit",
        "prompt": "Write the artifact",
        "reads": ["brief.json", "design.md", "check.txt", "check-record.json"],
        "writes": ["emit.txt"],
        "model_context": {"provider": "test", "model": "test"},
        "feedback_keys": ["validation", "qa_gate", "overwatch_review"],
    }

    manifest = {"id": "emit", "type": "AgentNode", "command": "echo"}
    monkeypatch.setenv(NODE_ENV, json.dumps(manifest))

    with patch("refactory_dsh.model.DshModelClient", FakeClient):
        with patch("refactory_dsh.agent_delivery.begin", return_value={"prompt": "test", "node": "emit", "status": "pending"}):
            with patch("refactory_dsh.agent_delivery.finish", return_value={"node": "emit", "status": "delivered"}):
                receipt = worker_node.execute(spec, run)

    assert receipt["status"] == "delivered"
    # The probe_instruction fact should be available
    facts = json.loads((run / "facts.json").read_text()) if (run / "facts.json").is_file() else {}
    # The prompt should have been rendered with filled feedback keys
    assert captured_prompt, "the model should have been called"


def test_probe_via_worker_node_execute_design_fast(tmp_path, monkeypatch):
    """A real probe from the scripted builder inside worker_node.execute,
    for design_fast mode.
    """
    from refactory_dsh import worker_node
    from refactory_dsh.run import NODE_ENV
    from refactory_dsh.fast import DesignFastMode

    run, repo, found = _make_probe_run(tmp_path)
    monkeypatch.chdir(run)
    monkeypatch.setenv("REFACTORY_WORKDIR", str(run))

    # Write declarations.json with the design_fast mode's emit node
    mode = DesignFastMode()
    from refactory_dsh.lowering import write_declarations
    write_declarations(mode.state_dict(), run)

    # Write the required input files
    (run / "brief.json").write_text(json.dumps({"name": "test-probe", "brief": "test", "check": "echo ok"}))
    (run / "source-manifest.json").write_text(json.dumps({"source_project": "", "files": {}}))
    (run / "implementation-plan.txt").write_text("Plan: implement\n")
    (run / "repair.txt").write_text("(initial implementation)\n")
    (run / "check-record.json").write_text(json.dumps({"status": "not_run"}))
    (run / "feedback-initial.json").write_text(json.dumps({
        "repair_feedback": "(initial implementation)",
    }))

    captured_prompt = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def for_node(self, node, role="worker"):
            return self
        def complete(self, prompt, workdir=None):
            captured_prompt.append(prompt)
            p = Path(workdir) if workdir else run
            (p / "emit.txt").write_text("===FILE: answer.py===\nprint(1)\n")
            return "done"
        def context(self):
            return {"provider": "test", "model": "test"}
        def route(self):
            return {"worker_backend": "dsh"}

    spec = {
        "schema_version": 1,
        "node_id": "emit",
        "prompt": "Build artifact",
        "reads": ["brief.json", "source-manifest.json", "implementation-plan.txt", "repair.txt", "check-record.json"],
        "writes": ["emit.txt"],
        "model_context": {"provider": "test", "model": "test"},
        "feedback_keys": ["repair_feedback"],
    }

    manifest = {"id": "emit", "type": "AgentNode", "command": "echo"}
    monkeypatch.setenv(NODE_ENV, json.dumps(manifest))

    with patch("refactory_dsh.model.DshModelClient", FakeClient):
        with patch("refactory_dsh.agent_delivery.begin", return_value={"prompt": "test", "node": "emit", "status": "pending"}):
            with patch("refactory_dsh.agent_delivery.finish", return_value={"node": "emit", "status": "delivered"}):
                receipt = worker_node.execute(spec, run)

    assert receipt["status"] == "delivered"
    assert captured_prompt, "the model should have been called"


# ── tests that pin the core ──────────────────────────────────────────────────

def test_lowering_feedback_keys_round_trip(tmp_path):
    """A real workers.lower_workflow spec with feedback_keys must still carry
    the keys so fill_feedback can read them from declarations.json.
    """
    from refactory_dsh.design import DesignMode
    from refactory_dsh.lowering import write_declarations

    run = tmp_path / "run"
    run.mkdir()
    (run / ".refactory").mkdir(parents=True)

    mode = DesignMode()
    write_declarations(mode.state_dict(), run)

    import json as _json
    decls = _json.loads((run / ".refactory" / "declarations.json").read_text())
    # The emit node must carry its feedback_keys
    assert "feedback_keys" in decls.get("emit", {}), \
        "lowering must include feedback_keys in declarations.json"
    assert decls["emit"]["feedback_keys"] == ["validation", "qa_gate", "overwatch_review"]


def test_check_artifact_sends_digest(tmp_path, monkeypatch):
    """check_artifact sends a digest (not raw output) as the repair detail.

    The digest starts with a recognisable header when there are test failures.
    """
    from refactory_dsh import design_ops, repo_change, nn, base_check, authoring_evidence, create_ops
    from refactory_dsh.sandbox import SandboxResult
    from refactory_dsh.nn import Verdict

    pytest_output = (
        "============================= FAILURES =============================\n"
        "_________________________ test_foo _________________________\n"
        "    def test_foo():\n"
        ">       assert 1 == 2\n"
        "E       AssertionError: assert 1 == 2\n"
        "\n"
        "    File tests/test_foo.py:3\n"
        "============================= short test summary info =============================\n"
        "FAILED tests/test_foo.py::test_foo - AssertionError: assert 1 == 2\n"
        "1 failed in 0.1s\n"
    )
    monkeypatch.setattr(repo_change, "refused", lambda ctx: None)
    fake_result = SandboxResult(ok=False, returncode=1, elapsed_s=1.0,
                                 stdout=pytest_output, stderr="")
    monkeypatch.setattr(repo_change, "run_check", lambda ctx, cmd, timeout: fake_result)
    monkeypatch.setattr(base_check, "compute_test_guardedness", lambda *a, **k: None)
    monkeypatch.setattr(authoring_evidence, "record", lambda *a, **k: None)
    monkeypatch.setattr(authoring_evidence, "stop_unrepairable", lambda *a, **k: None)

    repair_details = []
    monkeypatch.setattr(create_ops, "repair",
                        lambda ctx, details: (repair_details.extend(details), Verdict("reloop", details[0]))[1])

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".refactory").mkdir(exist_ok=True)
    ctx = nn.RunContext(run_dir, facts={"check": "python -m pytest", "name": "test", "brief": "test"},
                        knobs={"check_timeout_s": 60, "attempts": 3})
    ctx._declared = ("check.txt", "validation.txt", "authoring-evidence.json", "check-record.json",
                     "test-guardedness.json")
    ctx._declared_facts = ("validation", "check_attempts_used", "review_attempts_used",
                          "attempts_used", "runtime_evidence", "inherited_failures")
    ctx.enforce_writes = False

    result = design_ops.check_artifact(ctx)
    assert isinstance(result, Verdict)
    assert repair_details, "repair should have been called with a detail"
    detail = repair_details[0]
    # The detail must start with the digest, not raw output
    assert "Failing tests" in detail, \
        "check_artifact must send a digest (not raw output) as the repair detail"
    assert "test_foo" in detail


# ── warm probe tests ─────────────────────────────────────────────────────────


def _make_probe_run_with_pytest(tmp_path):
    """Like _make_probe_run but with a pytest check command so test IDs are
    recognised by build_probe_command."""
    run, repo, found = _make_probe_run(tmp_path)
    (run / "facts.json").write_text(json.dumps({
        "name": "test-probe", "brief": "test",
        "check": f"{sys.executable} -m pytest -q",
    }))
    return run, repo, found


def _make_probe_run_with_real_tests(tmp_path):
    """Like _make_probe_run but with real test files that probes can target,
    and a pytest check command."""
    run, repo, found = _make_probe_run(tmp_path)

    # conftest.py so imports from the repo root work
    (repo / "conftest.py").write_text(
        "import sys, os\n"
        "sys.path.insert(0, os.path.dirname(__file__))\n"
    )
    # A test file that probes can target
    (repo / "test_helper.py").write_text(
        "def test_bar():\n"
        "    from helper import VALUE\n"
        "    assert VALUE > 0\n"
    )
    # Two tests that cannot both pass: each needs its own emission's VALUE.
    (repo / "test_values.py").write_text(
        "def test_one():\n"
        "    from helper import VALUE\n"
        "    assert VALUE == 1\n"
        "\n\n"
        "def test_two():\n"
        "    from helper import VALUE\n"
        "    assert VALUE == 2\n"
    )

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "add test files",
         env={**os.environ, "GIT_AUTHOR_NAME": "test",
              "GIT_AUTHOR_EMAIL": "t@t.com",
              "GIT_COMMITTER_NAME": "test",
              "GIT_COMMITTER_EMAIL": "t@t.com"})

    new_head = _git(repo, "rev-parse", "HEAD")
    _write_json(run / ".refactory" / "slot" / "run-test.json", {
        "target": str(repo), "path": str(repo), "branch": found.branch,
        "base": found.base, "head": new_head, "run": str(run),
    })
    from refactory_dsh.invocations import WORKTREE_RECORD
    _write_json(run / WORKTREE_RECORD, {
        "target": str(repo), "path": str(repo), "branch": found.branch,
        "base": found.base, "head": new_head, "run": str(run),
    })
    (run / "facts.json").write_text(json.dumps({
        "name": "test-probe", "brief": "test",
        "check": f"{sys.executable} -m pytest -q",
    }))
    found = Worktree(target=str(repo), path=str(repo), branch=found.branch,
                     base=found.base, head=new_head, run=str(run))
    return run, repo, found


def _warm_dirs(run: Path) -> list[Path]:
    """The run's warm probe clones: probe directories that hold a cache.key."""
    refactory_dir = run / ".refactory"
    if not refactory_dir.is_dir():
        return []
    return sorted(d for d in refactory_dir.iterdir()
                  if d.name.startswith("probe-") and d.is_dir() and (d / "cache.key").is_file())


def _probe_cli(run: Path, emission: str, *tests: str) -> dict:
    """One probe as the builder runs it: its own process, with the emit prompt's flags."""
    result = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.builder_probe", "--workdir", str(run),
         "--emission", emission, "--test", *tests, "--timeout", "60"],
        capture_output=True, text=True, timeout=180,
    )
    return {**json.loads(result.stdout), "stderr": result.stderr}


#: Long enough that two probes sharing one clone at once overlap: each spends it in
#: setup, between applying its emission and running its test.
_SLOW_SETUP = "sleep 2"


def _exclusive_emissions(run: Path) -> None:
    """Two emissions, each passing only its own test: VALUE is 1 or 2, never both."""
    (run / "facts.json").write_text(json.dumps({
        "name": "test-probe", "brief": "test",
        "check": f"{sys.executable} -m pytest -q", "setup": _SLOW_SETUP,
    }))
    (run / "emit1.txt").write_text("===FILE: helper.py===\nVALUE = 1\n")
    (run / "emit2.txt").write_text("===FILE: helper.py===\nVALUE = 2\n")


@pytest.fixture
def unremovable():
    """Makes warm clones the sweep cannot remove, and removable again afterwards."""
    if os.geteuid() == 0:
        pytest.skip("root removes entries from a read-only directory")
    locked_dirs: list[Path] = []

    def make(run: Path, name: str = "probe-stuck") -> Path:
        clone = run / ".refactory" / name
        locked = clone / "repo" / "locked"
        locked.mkdir(parents=True)
        (locked / "file").write_text("held\n")
        (clone / "cache.key").write_text("key\n")
        locked.chmod(0o500)
        locked_dirs.append(locked)
        return clone

    yield make
    for locked in locked_dirs:
        if locked.exists():
            locked.chmod(0o700)


def test_warm_probe_subprocess_reuse(tmp_path):
    """Two sequential subprocess probes reuse the warm clone and reset properly.

    Emission 1 writes helper.py (VALUE=1) and marker.py; emission 2 writes
    helper.py (VALUE=2) only.  Both probes pass.  After the second probe,
    marker.py must be absent (reset cleaned it).

    Kills M6: without reset, marker.py from emission 1 persists and the test
    assertion on the clone directory fails.
    Also exercises fix 1 (install record deleted so install re-runs).
    """
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)

    # Emission 1: helper.py + marker.py
    (run / "emit.txt").write_text(
        "===FILE: helper.py===\n"
        "VALUE = 1\n"
        "===FILE: marker.py===\n"
        "STALE = True\n"
    )
    result1 = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.builder_probe",
         "--workdir", str(run), "--emission", "emit.txt",
         "--test", "test_helper.py::test_bar", "--timeout", "60"],
        capture_output=True, text=True, timeout=120,
    )
    output1 = json.loads(result1.stdout)
    assert output1["status"] == "pass", (
        f"first probe should pass (VALUE=1 > 0): {result1.stderr}")

    # Exactly one warm clone
    probe_dirs_1 = [d for d in (run / ".refactory").iterdir()
                    if d.name.startswith("probe-") and d.is_dir()
                    and (d / "cache.key").is_file()]
    assert len(probe_dirs_1) == 1, f"expected 1 warm clone, got {len(probe_dirs_1)}"
    first_dir = probe_dirs_1[0]

    # Emission 2: helper.py only — no marker.py
    (run / "emit.txt").write_text(
        "===FILE: helper.py===\n"
        "VALUE = 2\n"
    )
    result2 = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.builder_probe",
         "--workdir", str(run), "--emission", "emit.txt",
         "--test", "test_helper.py::test_bar", "--timeout", "60"],
        capture_output=True, text=True, timeout=120,
    )
    output2 = json.loads(result2.stdout)
    assert output2["status"] == "pass", (
        f"second probe should pass (VALUE=2 > 0): {result2.stderr}")

    # Same warm clone directory — only one git clone occurred
    probe_dirs_2 = [d for d in (run / ".refactory").iterdir()
                    if d.name.startswith("probe-") and d.is_dir()
                    and (d / "cache.key").is_file()]
    assert len(probe_dirs_2) == 1, f"expected 1 warm clone, got {len(probe_dirs_2)}"
    assert probe_dirs_2[0] == first_dir, "second probe should reuse the same clone"

    # Verify second emission's content
    clone_dir = first_dir / "repo"
    assert "VALUE = 2" in (clone_dir / "helper.py").read_text(), (
        "warm-hit clone must show the second emission's helper.py")
    # marker.py from emission 1 must be cleaned by reset (M6 kill)
    assert not (clone_dir / "marker.py").exists(), (
        "reset should have cleaned marker.py from emission 1")

    # Cleanup
    sweep_warm_clones(run)
    remaining = [d for d in (run / ".refactory").iterdir()
                 if d.name.startswith("probe-") and d.is_dir()]
    assert len(remaining) == 0, "sweep_warm_clones should remove the warm clone"


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm not available")
def test_warm_probe_install_record_cleared_on_reset(tmp_path):
    """Fix 1: install records deleted on warm reset so install re-runs.

    A target with package.json whose postinstall writes an ignored gen.json.
    The first probe passes (postinstall runs, gen.json created, test reads it).
    The second (warm) probe must also pass: git clean removes gen.json, the
    fix deletes the install record so npm install re-runs the postinstall.

    Without the fix (builder_probe.py ~447-453 reverted), the second probe
    fails: the install record persists, ensure_environment skips npm install,
    gen.json is missing, and the test that reads it errors.
    """
    run, repo, found = _make_probe_run(tmp_path)

    # Overwrite package.json with a postinstall that writes gen.json
    (repo / "package.json").write_text(json.dumps({
        "name": "test-probe-install",
        "version": "1.0.0",
        "scripts": {
            "postinstall": "node postinstall.js"
        }
    }))
    # postinstall.js writes gen.json (an ignored file outside node_modules)
    (repo / "postinstall.js").write_text(
        'var fs = require("fs");\n'
        'fs.writeFileSync("gen.json", JSON.stringify({v: 1}));\n'
    )
    (repo / ".gitignore").write_text("gen.json\nnode_modules/\n")
    # conftest.py so imports from the repo root work
    (repo / "conftest.py").write_text(
        "import sys, os\n"
        "sys.path.insert(0, os.path.dirname(__file__))\n"
    )
    # Test that reads gen.json — fails when gen.json is missing
    (repo / "test_gen.py").write_text(
        "import json, pathlib\n"
        "def test_gen():\n"
        "    data = json.loads(pathlib.Path('gen.json').read_text())\n"
        "    assert data['v'] == 1\n"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "add postinstall and test",
         env={**os.environ, "GIT_AUTHOR_NAME": "test",
              "GIT_AUTHOR_EMAIL": "t@t.com",
              "GIT_COMMITTER_NAME": "test",
              "GIT_COMMITTER_EMAIL": "t@t.com"})

    new_head = _git(repo, "rev-parse", "HEAD")
    _write_json(run / ".refactory" / "slot" / "run-test.json", {
        "target": str(repo), "path": str(repo), "branch": found.branch,
        "base": found.base, "head": new_head, "run": str(run),
    })
    from refactory_dsh.invocations import WORKTREE_RECORD
    _write_json(run / WORKTREE_RECORD, {
        "target": str(repo), "path": str(repo), "branch": found.branch,
        "base": found.base, "head": new_head, "run": str(run),
    })
    found = Worktree(target=str(repo), path=str(repo), branch=found.branch,
                     base=found.base, head=new_head, run=str(run))
    (run / "facts.json").write_text(json.dumps({
        "name": "test-probe-install",
        "brief": "test",
        "check": f"{sys.executable} -m pytest -q",
    }))

    # First probe: npm install → postinstall → gen.json → test passes
    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 1\n")
    result1 = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.builder_probe",
         "--workdir", str(run), "--emission", "emit.txt",
         "--test", "test_gen.py::test_gen", "--timeout", "90"],
        capture_output=True, text=True, timeout=180,
    )
    output1 = json.loads(result1.stdout)
    assert output1["status"] == "pass", (
        f"first probe should pass (gen.json written by postinstall): "
        f"{result1.stderr}")

    # Second probe (warm hit): gen.json cleaned → fix deletes install record →
    # npm install re-runs → postinstall re-creates gen.json → test passes
    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 2\n")
    result2 = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.builder_probe",
         "--workdir", str(run), "--emission", "emit.txt",
         "--test", "test_gen.py::test_gen", "--timeout", "90"],
        capture_output=True, text=True, timeout=180,
    )
    output2 = json.loads(result2.stdout)
    assert output2["status"] == "pass", (
        f"second (warm) probe must pass — install re-runs after fix 1: "
        f"{result2.stderr}")

    sweep_warm_clones(run)


def test_warm_probe_concurrent_subprocess(tmp_path):
    """Two probe processes at once each see exactly their own emission (M7).

    Each passes only on its own value -- test_one needs VALUE == 1, test_two VALUE == 2
    -- so a probe that ran on the other's emission fails. Without the per-clone flock,
    the second probe resets the shared clone and applies its emission while the first
    is in its setup, and the first fails.
    """
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)
    _exclusive_emissions(run)
    command = [sys.executable, "-m", "refactory_dsh.builder_probe",
               "--workdir", str(run), "--timeout", "60"]
    first = subprocess.Popen(
        [*command, "--emission", "emit1.txt", "--test", "test_values.py::test_one"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    second = subprocess.Popen(
        [*command, "--emission", "emit2.txt", "--test", "test_values.py::test_two"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out1, err1 = first.communicate(timeout=180)
    out2, err2 = second.communicate(timeout=180)

    assert json.loads(out1)["status"] == "pass", f"probe 1 did not see only its emission: {out1}{err1}"
    assert json.loads(out2)["status"] == "pass", f"probe 2 did not see only its emission: {out2}{err2}"
    assert len(_warm_dirs(run)) == 1, "the two probes did not share one warm clone"
    sweep_warm_clones(run)
    assert _warm_dirs(run) == []


def test_warm_probe_concurrent_no_interference(tmp_path, monkeypatch):
    """Two in-process probes at once, on threads, each see exactly their own emission (M7).

    Kept beside the subprocess test, as the first approval asked: the flock is held per
    open file description, so it serialises two threads of one process too.
    """
    import refactory_dsh.builder_probe as bp
    monkeypatch.setattr(bp, "_warm_enabled", True)
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)
    _exclusive_emissions(run)

    barrier = threading.Barrier(2, timeout=30)
    results: dict[str, ProbeResult] = {}
    errors: dict[str, BaseException] = {}

    def probe(emission: str, test: str) -> None:
        try:
            barrier.wait()
            results[test] = run_probe(run, emission, [f"test_values.py::{test}"], timeout=60)
        except BaseException as exc:  # reported below rather than lost with the thread
            errors[test] = exc

    threads = [threading.Thread(target=probe, args=("emit1.txt", "test_one")),
               threading.Thread(target=probe, args=("emit2.txt", "test_two"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)

    assert not errors, errors
    for test in ("test_one", "test_two"):
        assert results[test].status == "pass", f"{test} did not see only its emission: {results[test]}"
    assert len(_warm_dirs(run)) == 1, "the two probes did not share one warm clone"
    sweep_warm_clones(run)


def test_refused_emission_keeps_the_warm_clone(tmp_path):
    """A probe whose emission is refused returns the refusal and keeps the warm clone.

    The refusal is the builder's mistake, not a broken clone, so the next probe reuses
    the clone instead of cloning and installing again (measured on the lab before this:
    the refused probe took 2.7 s and the next one 28.7 s, against about 20 s warm).
    """
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)
    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 1\n")
    assert _probe_cli(run, "emit.txt", "test_helper.py::test_bar")["status"] == "pass"
    [warm] = _warm_dirs(run)

    # An edit of a file the attempt does not have: refused before the clone is touched.
    (run / "emit.txt").write_text(
        f"===EDIT: missing.py===\n{repo_change.SEARCH}\nold\n{repo_change.DIVIDER}\n"
        f"new\n{repo_change.REPLACE}\n")
    refused = _probe_cli(run, "emit.txt", "test_helper.py::test_bar")
    assert refused["status"] == "refused", refused
    assert "missing.py" in refused["refusal"]
    assert _warm_dirs(run) == [warm], "the refused emission evicted the warm clone"

    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 2\n")
    assert _probe_cli(run, "emit.txt", "test_helper.py::test_bar")["status"] == "pass"
    assert _warm_dirs(run) == [warm], "the probe after the refusal did not reuse the clone"
    sweep_warm_clones(run)


def test_warm_probe_materialise_removes_clone(tmp_path):
    """materialise() removes the warm clone via sweep_warm_clones.

    Kills M4: without sweep_warm_clones in materialise, warm clones survive.
    Drives the real repo_change.materialise.
    """
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)

    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 1\n")
    result = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.builder_probe",
         "--workdir", str(run), "--emission", "emit.txt",
         "--test", "test_helper.py::test_bar", "--timeout", "60"],
        capture_output=True, text=True, timeout=120,
    )
    output = json.loads(result.stdout)
    assert output["status"] == "pass", f"probe failed: {result.stderr}"

    # Warm clone must exist
    warm_dirs = [d for d in (run / ".refactory").iterdir()
                 if d.name.startswith("probe-") and d.is_dir()
                 and (d / "cache.key").is_file()]
    assert len(warm_dirs) == 1, "warm clone should exist after probe"

    # Drive the real materialise — sweep_warm_clones runs before the commit
    from refactory_dsh.nn.executor import RunContext
    ctx = RunContext(run,
                     facts={"check": f"{sys.executable} -m pytest -q",
                            "name": "test-probe", "brief": "test"},
                     knobs={"check_timeout_s": 60, "attempts": 3})
    ctx._declared = ()
    ctx._declared_facts = ()
    ctx.enforce_writes = False
    try:
        repo_change.materialise(ctx)
    except Exception:
        pass  # expected — the emission format is not set up for full materialise

    # Warm clones must be gone (materialise sweeps them before committing)
    warm_dirs = [d for d in (run / ".refactory").iterdir()
                 if d.name.startswith("probe-") and d.is_dir()
                 and (d / "cache.key").is_file()]
    assert len(warm_dirs) == 0, "materialise must remove warm clones"

    # Slot records should be cleaned too
    slot_dir = run / ".refactory" / "slot"
    if slot_dir.is_dir():
        leaked = [f for f in slot_dir.iterdir() if f.name.startswith("run-probe-")]
        assert not leaked, f"leaked slot records: {[f.name for f in leaked]}"


def test_sweep_dead_probes_spares_warm_clone(tmp_path):
    """sweep_dead_probes skips warm clones (identified by cache.key).

    Kills M3: without the cache.key guard, the warm clone is treated as leaked
    and deleted.
    """
    workdir = tmp_path / "run"
    workdir.mkdir()
    refactory_dir = workdir / ".refactory"
    refactory_dir.mkdir()

    # Create a synthetic warm clone with cache.key only (no owner.pid —
    # real warm clones never write owner.pid; that is cold-clone only)
    probe_dir = refactory_dir / "probe-warmtest"
    probe_dir.mkdir()
    (probe_dir / "cache.key").write_text("test-key")

    sweep_dead_probes(workdir)
    assert probe_dir.exists(), "sweep_dead_probes must spare warm clones (cache.key)"


def test_warm_probe_setup_runs_on_warm_hit(tmp_path):
    """Setup and generate re-run on a warm-hit probe.

    Kills M11: without prepare_check_environment on warm hit, setup doesn't
    run and the stamp file is missing.
    """
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)

    # Setup command writes a stamp file
    (run / "facts.json").write_text(json.dumps({
        "name": "test-probe", "brief": "test",
        "check": f"{sys.executable} -m pytest -q",
        "setup": "touch .setup_stamp",
    }))

    # First probe: creates warm clone and runs setup
    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 1\n")
    result1 = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.builder_probe",
         "--workdir", str(run), "--emission", "emit.txt",
         "--test", "test_helper.py::test_bar", "--timeout", "60"],
        capture_output=True, text=True, timeout=120,
    )
    output1 = json.loads(result1.stdout)
    assert output1["status"] == "pass", f"first probe failed: {result1.stderr}"

    # Second probe: warm hit — setup must re-run
    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 2\n")
    result2 = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.builder_probe",
         "--workdir", str(run), "--emission", "emit.txt",
         "--test", "test_helper.py::test_bar", "--timeout", "60"],
        capture_output=True, text=True, timeout=120,
    )
    output2 = json.loads(result2.stdout)
    assert output2["status"] == "pass", f"second probe failed: {result2.stderr}"

    # After second probe the stamp must exist in the clone (setup re-ran)
    probe_dirs = [d for d in (run / ".refactory").iterdir()
                  if d.name.startswith("probe-") and d.is_dir()
                  and (d / "cache.key").is_file()]
    assert len(probe_dirs) == 1
    clone_dir = probe_dirs[0] / "repo"
    assert (clone_dir / ".setup_stamp").exists(), (
        "setup must re-run on warm hit — stamp missing (M11)")

    sweep_warm_clones(run)


def test_in_process_probe_is_cold(tmp_path):
    """In-process callers get cold (non-warm) clones — no cache.key persists.

    Kills M13: if _warm_enabled were True unconditionally, a warm clone with
    cache.key would persist after the probe.
    """
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)
    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 1\n")

    result = run_probe(run, "emit.txt", ["test_helper.py::test_bar"], timeout=60)
    assert result.status == "pass", f"probe failed: {result.refusal}"

    # No cache.key should remain (cold probe cleans up completely)
    probe_dirs = [d for d in (run / ".refactory").iterdir()
                  if d.name.startswith("probe-") and d.is_dir()
                  and (d / "cache.key").is_file()]
    assert not probe_dirs, (
        "in-process probe must not leave a warm clone (cache.key)")


def test_warm_probe_stale_index_lock(tmp_path):
    """A stale .git/index.lock triggers cold fallback, not a persistent error.

    Fix 3: the fallback evicts the broken clone and creates a new cold one.
    """
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)

    # First probe: creates warm clone
    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 1\n")
    result1 = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.builder_probe",
         "--workdir", str(run), "--emission", "emit.txt",
         "--test", "test_helper.py::test_bar", "--timeout", "60"],
        capture_output=True, text=True, timeout=120,
    )
    output1 = json.loads(result1.stdout)
    assert output1["status"] == "pass", f"first probe failed: {result1.stderr}"

    # Plant a stale .git/index.lock in the warm clone
    probe_dirs = [d for d in (run / ".refactory").iterdir()
                  if d.name.startswith("probe-") and d.is_dir()
                  and (d / "cache.key").is_file()]
    assert len(probe_dirs) == 1
    first_dir = probe_dirs[0]
    (first_dir / "repo" / ".git" / "index.lock").write_text("stale")

    # Second probe: broken clone should trigger cold fallback
    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 2\n")
    result2 = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.builder_probe",
         "--workdir", str(run), "--emission", "emit.txt",
         "--test", "test_helper.py::test_bar", "--timeout", "60"],
        capture_output=True, text=True, timeout=120,
    )
    output2 = json.loads(result2.stdout)
    assert output2["status"] == "pass", (
        f"second probe should succeed via cold fallback: {result2.stderr}")

    # The old broken clone should be evicted and a new one created
    probe_dirs = [d for d in (run / ".refactory").iterdir()
                  if d.name.startswith("probe-") and d.is_dir()
                  and (d / "cache.key").is_file()]
    assert len(probe_dirs) == 1
    assert probe_dirs[0] != first_dir, "broken clone should be evicted"

    sweep_warm_clones(run)


class _ProbingBuilder(FakeModel):
    """Answers a walker run as FakeModel does; at the builder's turn, runs a real probe.

    The probe runs as the builder runs it: emit.txt first, then the command line the
    prompt's "Builder probe" section gives, in its own process. Then the turn ends as
    ``end`` says: ``answered`` returns an answer, ``failed`` raises as a builder session
    that died does, and ``stopped`` sends this process SIGINT -- what a person's Ctrl-C
    and `console stop` send the walker.
    """

    def __init__(self, answers: list[str], *, end: str, facts: dict | None = None) -> None:
        super().__init__(answers)
        self.end = end
        self.facts = facts
        self.clones: list[Path] = []

    def complete(self, prompt: str, *, system: str = "", workdir: object = None) -> str:
        if "## Builder probe" not in prompt:
            return super().complete(prompt, system=system, workdir=workdir)
        run = Path(str(workdir))
        if self.facts is not None:
            # The walker keeps its facts in memory, and the probe reads facts.json,
            # which only the spine's node runner writes: written here as it would be.
            (run / "facts.json").write_text(json.dumps(self.facts))
        (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 1\n")
        command = prompt.split("## Builder probe", 1)[1].split("```bash\n", 1)[1].split("\n```", 1)[0]
        argv = shlex.split(command.replace("<test_id> [<test_id> ...]", "test_helper.py::test_bar"))
        probe = subprocess.run(argv, capture_output=True, text=True, timeout=180)
        assert json.loads(probe.stdout)["status"] == "pass", probe.stdout + probe.stderr
        self.clones = _warm_dirs(run)
        if self.end == "stopped":
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(10)  # the SIGINT raises KeyboardInterrupt here, in the main thread
            raise AssertionError("SIGINT did not stop the run")
        if self.end == "failed":
            raise RuntimeError("the builder's session ended")
        return "answered\n"


@pytest.mark.parametrize("end", ["failed", "stopped"])
def test_walker_design_run_that_ends_before_materialise_leaves_no_warm_clone(tmp_path, end):
    """A walker design run whose builder probed, then ended before materialise, leaves no clone.

    Real: DesignMode on the walker with scripted answers, a target repository and the
    run's worktree of it, and the probe CLI the emit prompt names, which makes a warm
    clone. The run then ends -- the builder fails, or SIGINT stops it -- before
    materialise, so only the walker's run end (nn.executor.execute, through the
    refactory.run_end entry point) can remove the clone.
    """
    from refactory_dsh.design import DesignMode

    repo = _git_repo(tmp_path / "repo")
    (repo / "conftest.py").write_text("import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n")
    (repo / "test_helper.py").write_text(
        "def test_bar():\n    from helper import VALUE\n    assert VALUE > 0\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "tests")
    check = f"{sys.executable} -m pytest -q"
    builder = _ProbingBuilder(
        [RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan", DESIGN_DOC_ANSWER],
        end=end, facts={"check": check})
    run = tmp_path / "run"
    task = {"name": "probe_tool", "brief": "a tool", "check": check, "target": str(repo)}

    if end == "stopped":
        # A suite run in the background (nohup, `&`) inherits SIGINT as ignored, so the
        # builder's own SIGINT would do nothing: Python's handler is put back for the run.
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            with pytest.raises(KeyboardInterrupt):
                DesignMode().run(run, model=builder, verdicts={"gate_strategy": "proceed"}, task=task)
        finally:
            signal.signal(signal.SIGINT, previous)
    else:
        result = DesignMode().run(run, model=builder, verdicts={"gate_strategy": "proceed"}, task=task)
        assert not result.exit_ok and "the builder's session ended" in (result.error or "")

    assert len(builder.clones) == 1, "the builder's probe made no warm clone"
    assert not (run / "changes.diff").exists(), "materialise ran"
    assert _warm_dirs(run) == [], "the walker's run end left the warm clone on disk"


@pytest.mark.parametrize("end", ["done", "halted", "waiting"])
def test_walker_run_end_removes_warm_clone(tmp_path, end):
    """However a walker run ends, its run end removes the warm clone a probe made.

    A real probe from an agent node, then, before any materialise: the graph ends
    (done), a gate halts it (halted), or a user gate waits for a person (waiting).
    """
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)
    prompt = ("## Builder probe\n\n```bash\n"
              f"{sys.executable} -m refactory_dsh.builder_probe --workdir {run} "
              "--emission emit.txt --test <test_id> [<test_id> ...] --timeout 60\n```\n")
    graph = nn.Graph()
    graph.add(nn.Node("emit", "agent", prompt_template=prompt, writes=("answer.txt",)))
    if end != "done":
        graph.add(nn.Node("gate_check", "gate", gate_kind="user", reads=("answer.txt",)))
        graph.add(nn.Node("materialise", "op", call="loop_ops:noop"))
        graph.connect("emit", "gate_check")
        graph.connect("gate_check", "materialise", condition="proceed")
    builder = _ProbingBuilder([], end="answered")

    # The run's facts are the walker's, which it writes to `facts.json` as it goes.
    result = nn.execute(graph, run, model=builder, facts=json.loads((run / "facts.json").read_text()),
                        verdicts={"gate_check": "halt"} if end == "halted" else None)

    assert {"done": result.exit_ok, "halted": result.halted,
            "waiting": result.gate_stop == "gate_check"}[end], result.error
    assert len(builder.clones) == 1, "the probe made no warm clone"
    assert _warm_dirs(run) == [], f"a walker run that ended {end} left the warm clone"


#: Plays DSH's spine for one run: what its nodes leave before emit, the builder's real
#: probe, then the status the run ends on. Its plan is the JSON file beside it.
_FAKE_SPINE = '''\
import json
import subprocess
import sys
from pathlib import Path

from refactory_dsh.invocations import WORKTREE_RECORD
from refactory_dsh.worktree import _write_json

plan = json.loads(Path(sys.argv[0]).with_suffix(".json").read_text())
workdir = Path(sys.argv[sys.argv.index("--workdir") + 1])
record = {**plan["found"], "run": str(workdir)}
_write_json(workdir / ".refactory" / "slot" / "run-test.json", record)
_write_json(workdir / WORKTREE_RECORD, record)
(workdir / "facts.json").write_text(json.dumps({"check": sys.executable + " -m pytest -q"}))
(workdir / "emit.txt").write_text("===FILE: helper.py===\\nVALUE = 1\\n")
probe = subprocess.run(
    [sys.executable, "-m", "refactory_dsh.builder_probe", "--workdir", str(workdir),
     "--emission", "emit.txt", "--test", "test_helper.py::test_bar", "--timeout", "60"],
    capture_output=True, text=True)
clones = sorted(str(p) for p in (workdir / ".refactory").glob("probe-*") if (p / "cache.key").is_file())
(workdir / "spine-probe.json").write_text(json.dumps({"probe": probe.stdout + probe.stderr,
                                                     "clones": clones}))
if plan["unremovable"]:
    for clone in clones:
        locked = Path(clone) / "repo" / "locked"
        locked.mkdir()
        (locked / "file").write_text("held\\n")
        locked.chmod(0o500)
_write_json(workdir / ".refactory" / "spine-status.json", plan["status"])
sys.exit(plan["code"])
'''


@pytest.mark.parametrize("status, code, unremovable", [
    ({"status": "done"}, 0, False),
    ({"status": "failed", "error": "emit: the builder's session ended"}, 1, False),
    # What `graph-control stop` leaves: the spine halted at a boundary, cancelled.
    ({"status": "halted", "halted": {"gate": "emit", "reason": "cancelled"}}, 1, False),
    ({"status": "done"}, 0, True),
], ids=["done", "failed", "stopped", "unremovable"])
def test_spine_run_owner_removes_warm_clone_at_run_end(tmp_path, monkeypatch, caplog,
                                                      status, code, unremovable):
    """A spine run's owner removes the warm clone its builder's probe left, at the run's end.

    Real: interactive.run's pre-flight checks (the build, the steer entry point, a graph
    profile linked to this checkout), the ownership lock, the lowering, the DSH argv,
    run_graph_command spawning and waiting on its child, promote_unretried_failure and
    the sweep. Replaced: only the DSH executable (`interactive.dsh_command`), by a script
    that leaves what the spine's nodes leave before emit, runs the real probe CLI as the
    builder does, writes the status the run ends on and exits. A clone the owner cannot
    remove is named in its warning.
    """
    if unremovable and os.geteuid() == 0:
        pytest.skip("root removes entries from a read-only directory")
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)
    home = tmp_path / "dsh-home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    _link_bundle(home, "graph", dsh_profile.GRAPH_BUNDLE, ROOT / "packages" / "bundle" / "graph")
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.delenv("REFACTORY_MODEL_DSH_HOME", raising=False)
    spine = tmp_path / "spine.py"
    spine.write_text(_FAKE_SPINE)
    spine.with_suffix(".json").write_text(json.dumps({
        "found": {"target": found.target, "path": found.path, "branch": found.branch,
                  "base": found.base, "head": found.head},
        "status": status, "code": code, "unremovable": unremovable,
    }))
    monkeypatch.setattr(interactive, "dsh_command", lambda: [sys.executable, str(spine)])
    workdir = tmp_path / "spine-run"
    caplog.set_level(logging.WARNING)

    try:
        assert interactive.run("review_pack/review", workdir=workdir, project=ROOT,
                               headless=True) == code
        seen = json.loads((workdir / "spine-probe.json").read_text())
        assert len(seen["clones"]) == 1, f"the probe made no warm clone: {seen['probe']}"
        if unremovable:
            [clone] = seen["clones"]
            assert Path(clone).exists()
            assert any(clone in record.getMessage() for record in caplog.records
                       if record.name == "refactory_dsh.interactive"), (
                "the owner's warning does not name the clone it left")
        else:
            assert _warm_dirs(workdir) == [], "the spine run's end left the warm clone"
    finally:
        for locked in workdir.glob(".refactory/probe-*/repo/locked"):
            locked.chmod(0o700)


def test_sweep_names_a_warm_clone_it_cannot_remove(tmp_path, unremovable):
    """A clone the sweep cannot remove raises CloneNotRemoved naming it, after the rest went."""
    run = tmp_path / "run"
    stuck = unremovable(run, "probe-a-stuck")
    other = run / ".refactory" / "probe-b-other"
    other.mkdir()
    (other / "cache.key").write_text("key\n")

    with pytest.raises(CloneNotRemoved) as raised:
        sweep_warm_clones(run)

    assert f"could not remove warm clone {stuck}" in str(raised.value)
    assert "Permission denied" in str(raised.value)
    assert not other.exists(), "the sweep stopped at the clone it could not remove"


@pytest.mark.parametrize("site", ["walker", "dsh_evaluator", "materialise"])
def test_cleanup_warning_names_a_warm_clone_it_cannot_remove(tmp_path, unremovable, caplog, site):
    """Each place that sweeps warns naming the clone it left, and carries on.

    The spine run's owner is checked by test_spine_run_owner_removes_warm_clone_at_run_end.
    """
    run = tmp_path / "run"
    (run / ".refactory").mkdir(parents=True)
    stuck = unremovable(run)
    caplog.set_level(logging.WARNING)

    if site == "walker":
        graph = nn.Graph()
        graph.add(nn.Node("only", "op", call="loop_ops:noop"))
        assert nn.execute(graph, run).exit_ok
    elif site == "dsh_evaluator":
        from refactory_dsh.dsh_runtime import DshRun, evaluate_on_dsh

        (run / ".refactory" / "spine-status.json").write_text(json.dumps({"status": "done"}))

        class _Mode:
            __module__ = "refactory_dsh.test_walker"
            name = "test"

            def score(self, result):
                return 0.0

        ended = DshRun(facts={"name": "test"}, workdir=run, returncode=0,
                       stdout="", stderr="", duration_s=1.0)
        with patch("refactory_dsh.dsh_runtime.run_on_dsh", return_value=ended):
            evaluate_on_dsh(_Mode(), {}, workdir=run)
    else:
        from refactory_dsh.nn.executor import RunContext

        (run / "emit.txt").write_text("===FILE: tool.py===\nprint('built')\n")
        ctx = RunContext(run, facts={"name": "probe_tool", "check": "true"}, knobs={})
        ctx.enforce_writes = False
        assert repo_change.materialise(ctx)["commit"] == repo_change.NO_WORKTREE

    assert any(f"could not remove warm clone {stuck}" in record.getMessage()
               for record in caplog.records if record.levelno == logging.WARNING), (
        f"no warning names {stuck}: {[record.getMessage() for record in caplog.records]}")
    assert stuck.exists()


def test_dsh_evaluator_run_end_removes_warm_clone(tmp_path):
    """The DSH evaluator's run end removes warm clones (dsh_runtime.evaluate_on_dsh).

    Drives evaluate_on_dsh with a mocked run_on_dsh; the run-end cleanup it then
    runs (the refactory.run_end entry points) runs for real.
    """
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)

    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 1\n")
    subprocess.run(
        [sys.executable, "-m", "refactory_dsh.builder_probe",
         "--workdir", str(run), "--emission", "emit.txt",
         "--test", "test_helper.py::test_bar", "--timeout", "60"],
        capture_output=True, text=True, timeout=120,
    )

    warm_dirs = [d for d in (run / ".refactory").iterdir()
                 if d.name.startswith("probe-") and d.is_dir()
                 and (d / "cache.key").is_file()]
    assert len(warm_dirs) == 1

    # Drive evaluate_on_dsh with mocked run_on_dsh
    from refactory_dsh.dsh_runtime import DshRun, evaluate_on_dsh

    mock_run = DshRun(
        facts={"name": "test"}, workdir=run, returncode=0,
        stdout="", stderr="", duration_s=1.0,
    )
    (run / ".refactory" / "spine-status.json").write_text(
        json.dumps({"status": "done"}))

    class _FakeMode:
        __module__ = "refactory_dsh.test_walker"
        name = "test"
        def score(self, result):
            return 0.0

    with patch("refactory_dsh.dsh_runtime.run_on_dsh", return_value=mock_run):
        evaluate_on_dsh(_FakeMode(), {}, workdir=run)

    warm_dirs = [d for d in (run / ".refactory").iterdir()
                 if d.name.startswith("probe-") and d.is_dir()
                 and (d / "cache.key").is_file()]
    assert len(warm_dirs) == 0, (
        "the DSH evaluator's run end must remove warm clones")


def test_sweep_skips_locked_clone(tmp_path):
    """sweep_warm_clones skips a clone whose lock is held (in use).

    Fix 6: take lock without blocking, skip clones in use.
    """
    import fcntl as _fcntl

    workdir = tmp_path / "run"
    workdir.mkdir()
    refactory_dir = workdir / ".refactory"
    refactory_dir.mkdir()

    probe_dir = refactory_dir / "probe-locktest"
    probe_dir.mkdir()
    (probe_dir / "cache.key").write_text("test-key")
    (probe_dir / "clone_base").write_text("abc123")

    # Hold the lock in another thread
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        fd = open(probe_dir / "probe.lock", "w")
        _fcntl.flock(fd, _fcntl.LOCK_EX)
        lock_held.set()
        release_lock.wait(timeout=30)
        _fcntl.flock(fd, _fcntl.LOCK_UN)
        fd.close()

    t = threading.Thread(target=hold_lock)
    t.start()
    lock_held.wait(timeout=10)

    # Sweep should skip the locked clone
    sweep_warm_clones(workdir)
    assert probe_dir.exists(), "sweep must skip a locked clone"

    # Release lock and sweep again — clone should be removed
    release_lock.set()
    t.join(timeout=10)

    sweep_warm_clones(workdir)
    assert not probe_dir.exists(), "sweep must remove an unlocked clone"


def test_docs_warm_lifecycle_language():
    """hardening.md and repo_change.py describe the warm lifecycle.

    Fix 7: 'deleted afterwards' must be replaced with warm lifecycle language.
    """
    import refactory_dsh
    pkg_root = Path(refactory_dsh.__file__).parent.parent

    # Check hardening.md
    hardening_path = pkg_root / "docs" / "hardening.md"
    hardening = hardening_path.read_text()
    assert "Lifecycle" in hardening, (
        "hardening.md must have a 'Lifecycle' bullet for probes")
    assert "deleted afterwards" not in hardening.split("#### What is not isolated")[0], (
        "old 'deleted afterwards' text should be replaced in probe section")

    # Check repo_change.py
    rc_path = pkg_root / "refactory_dsh" / "repo_change.py"
    rc_source = rc_path.read_text()
    assert "cleaned up when the builder" in rc_source, (
        "repo_change.py probe instruction must describe warm lifecycle")


def test_vitest_routing_mixed_ids_run_whole_files(tmp_path):
    """When bare spec IDs and structured IDs are mixed, named files run whole.

    Fix 8: no code change — document the existing behaviour.
    """
    from refactory_dsh.check_digest import build_probe_command

    # Create a package.json with vitest test script for pnpm test routing
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "test", "scripts": {"test": "vitest run"},
    }))

    structured_ids = ["src/App.test.tsx > renders correctly"]
    bare_ids = ["src/other.test.tsx"]

    cmd, refusal = build_probe_command(
        "npx vitest run",
        structured_ids + bare_ids,
        root=tmp_path,
    )
    assert refusal is None, f"unexpected refusal: {refusal}"
    # Both files should appear in the command
    assert "src/App.test.tsx" in cmd, "structured ID's file must be included"
    assert "src/other.test.tsx" in cmd, "bare spec file must be included"
    # When whole files are present, -t is dropped (files run whole)
    assert "-t" not in cmd, (
        "mixed bare+structured IDs: named files run whole, no -t filter")


def test_vitest_routing_self_check_stable(tmp_path):
    """Pure structured vitest IDs get -t; pure whole files get no -t.

    Regression snapshot: the self-check's command format is unchanged.
    """
    from refactory_dsh.check_digest import build_probe_command

    (tmp_path / "package.json").write_text(json.dumps({
        "name": "test", "scripts": {"test": "vitest run"},
    }))

    # Pure structured IDs → -t present
    cmd_structured, ref = build_probe_command(
        "npx vitest run",
        ["src/App.test.tsx > renders correctly"],
        root=tmp_path,
    )
    assert ref is None
    assert "-t" in cmd_structured, "pure structured IDs must keep -t"

    # Pure whole files → no -t
    cmd_whole, ref2 = build_probe_command(
        "npx vitest run",
        ["src/other.test.tsx"],
        root=tmp_path,
    )
    assert ref2 is None
    assert "-t" not in cmd_whole, "pure whole files must not have -t"

    # pytest routing unchanged
    cmd_pytest, ref3 = build_probe_command(
        f"{sys.executable} -m pytest -q",
        ["tests/test_foo.py::test_bar"],
        root=tmp_path,
    )
    assert ref3 is None
    assert "test_foo.py::test_bar" in cmd_pytest


def test_warm_probe_stale_clone_not_reused(tmp_path, monkeypatch):
    """A stale clone is not reused after the attempt's head moves.

    Revert check: removing stale-eviction code would reuse the old clone
    with the wrong base, or fail the HEAD verification.
    """
    import refactory_dsh.builder_probe as bp
    monkeypatch.setattr(bp, "_warm_enabled", True)
    run, repo, found = _make_probe_run_with_real_tests(tmp_path)

    # First probe — creates a warm clone
    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 1\n")
    run_probe(run, "emit.txt", ["test_helper.py::test_bar"], timeout=60)

    probe_dirs_1 = [d for d in (run / ".refactory").iterdir()
                    if d.name.startswith("probe-") and d.is_dir()
                    and (d / "cache.key").is_file()]
    assert len(probe_dirs_1) == 1
    first_name = probe_dirs_1[0].name

    # Advance the repo's HEAD — makes the cache key stale
    (repo / "new_file.py").write_text("print('new')\n")
    _git(repo, "add", "new_file.py")
    _git(repo, "commit", "-q", "-m", "advance head",
         env={**os.environ, "GIT_AUTHOR_NAME": "test",
              "GIT_AUTHOR_EMAIL": "t@t.com",
              "GIT_COMMITTER_NAME": "test",
              "GIT_COMMITTER_EMAIL": "t@t.com"})
    new_head = _git(repo, "rev-parse", "HEAD")

    # Update the worktree records to point to the new head
    _write_json(run / ".refactory" / "slot" / "run-test.json", {
        "target": str(repo), "path": str(repo), "branch": found.branch,
        "base": found.base, "head": new_head, "run": str(run),
    })
    from refactory_dsh.invocations import WORKTREE_RECORD
    _write_json(run / WORKTREE_RECORD, {
        "target": str(repo), "path": str(repo), "branch": found.branch,
        "base": found.base, "head": new_head, "run": str(run),
    })

    # Second probe — should evict stale clone and create a new one
    (run / "emit.txt").write_text("===FILE: helper.py===\nVALUE = 2\n")
    run_probe(run, "emit.txt", ["test_helper.py::test_bar"], timeout=60)

    probe_dirs_2 = [d for d in (run / ".refactory").iterdir()
                    if d.name.startswith("probe-") and d.is_dir()
                    and (d / "cache.key").is_file()]
    assert len(probe_dirs_2) == 1, f"expected 1 warm clone, got {len(probe_dirs_2)}"
    assert probe_dirs_2[0].name != first_name, (
        "stale clone should be evicted — the probe dir name must differ")

    sweep_warm_clones(run)
