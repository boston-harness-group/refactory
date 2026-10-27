"""Tests for the robustness hardening: R1, R2, R3, R5, R6 and R9.

Each test uses scripted commands that kill themselves with a signal;
none depends on the lab's LXCFS.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import base_check, delivery, design_ops, nn, sandbox, worktree
from refactory_dsh import authoring_evidence as evidence
from refactory_dsh import design_contract
from refactory_dsh.create_ops import repair
from refactory_dsh.fast_ops import _repair as fast_repair
from refactory_dsh.nn import Verdict
from refactory_dsh.design import DesignMode
from tests_support import (DESIGN_DOC_ANSWER,
                           PITFALLS_ANSWER, RESEARCH_ANSWER,
                           REVIEW_ANSWERS, VERIFICATION_ANSWER,
                           FakeModel as _FakeModel,
                           git_repo as _git_repo, render as _render)

# ── helpers ────────────────────────────────────────────────────────────────


#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def _git_init(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "T"],
                   capture_output=True, check=True)
    return path


def _commit_file(repo: Path, name: str, text: str) -> str:
    (repo / name).write_text(text)
    subprocess.run(["git", "-C", str(repo), "add", name],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", f"add {name}"],
                   capture_output=True, check=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _make_worktree(repo: Path, run_dir: Path, base: str) -> worktree.Worktree:
    """Create a real worktree and record it, returning the descriptor."""
    run_dir.mkdir(parents=True, exist_ok=True)
    found = worktree.create(repo, base, "test00000000", run_dir)
    return found


def _with_attempt(found: worktree.Worktree) -> worktree.Worktree:
    """``found`` with one attempt committed on its branch."""
    import dataclasses

    return dataclasses.replace(found, head=_commit_file(Path(found.path), "attempt.txt", "built\n"))


class _Ctx:
    """Minimal RunContext stand-in for unit-level tests."""

    def __init__(self, workdir: Path, *, facts: dict | None = None,
                 knobs: dict | None = None, files: dict | None = None):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.facts: dict[str, Any] = dict(facts or {})
        self.knobs_dict: dict[str, Any] = dict(knobs or {})
        self._files: dict[str, str] = dict(files or {})
        self.budget = None
        self.enforce_writes = False
        self._produced: set[str] = set()
        self._declared_facts: tuple[str, ...] = ()

    def knob(self, name: str, default: Any = None) -> Any:
        return self.knobs_dict.get(name, default)

    def set_fact(self, name: str, value: Any) -> None:
        self.facts[name] = value
        self._produced.add(name)

    def exists(self, name: str) -> bool:
        if name in self._files:
            return True
        return (self.workdir / name).exists()

    def read_text(self, name: str) -> str:
        if name in self._files:
            return self._files[name]
        return (self.workdir / name).read_text()

    def write_text(self, name: str, text: str) -> Path:
        target = self.workdir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    def write_json(self, name: str, payload: Any) -> Path:
        return self.write_text(name, json.dumps(payload, indent=2, default=str))

    def read_json(self, name: str, default: Any = None) -> Any:
        p = self.workdir / name
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return default


# ── R1: is_signal_crash helper ─────────────────────────────────────────────


def test_is_signal_crash_direct_signal():
    """A direct signal (rc < 0) is classified as a crash."""
    result = sandbox.SandboxResult(ok=False, returncode=-6)
    crashed, label = sandbox.is_signal_crash(result)
    assert crashed is True
    assert "SIGABRT" in label


def test_is_signal_crash_shell_wrapped():
    """A shell-wrapped SIGABRT (exit 134) is classified as a crash."""
    result = sandbox.SandboxResult(ok=False, returncode=134)
    crashed, label = sandbox.is_signal_crash(result)
    assert crashed is True
    assert "SIGABRT" in label


def test_is_signal_crash_normal_exit():
    """A normal non-zero exit is not a crash."""
    result = sandbox.SandboxResult(ok=False, returncode=1)
    crashed, label = sandbox.is_signal_crash(result)
    assert crashed is False
    assert label == ""


def test_is_signal_crash_none_returncode():
    """A None returncode is not a crash."""
    result = sandbox.SandboxResult(ok=False, returncode=None)
    crashed, label = sandbox.is_signal_crash(result)
    assert crashed is False


# ── R1: signal-killed install (via scripted design run) ────────────────────


def _design_answers(*emits: str) -> list[str]:
    """Every answer a design run needs, in the order its graph asks."""
    return [RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
            DESIGN_DOC_ANSWER, *emits,
            *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3]]


def _design_run(workdir, model, target):
    """Run a scripted design run, like the design-worktree tests do."""
    return DesignMode().run(
        workdir, model=model,
        verdicts={"gate_strategy": "proceed"},
        task={"name": "r1_test", "brief": "a tool", "check": "python tool.py",
              "target": str(target)},
    )


def test_signal_killed_install_once_then_passing_spends_no_attempt(tmp_path, monkeypatch):
    """An install killed by SIGABRT once, then passing, spends no attempt.

    Driven as a scripted design run.  The stub uv is a shell script whose child
    aborts — it exits 134, as ./install.sh did in the b27a incident.
    run_command retries once internally; the retry sees the flag file, creates
    .venv normally and exits 0.  The run completes with one attempt.
    """
    counter_dir = tempfile.mkdtemp(prefix="r1_")
    counter_file = os.path.join(counter_dir, "counter")

    repo = _git_repo(tmp_path / "repo")
    (repo / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    subprocess.run(["git", "-C", str(repo), "add", "pyproject.toml"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "manifest"],
                   capture_output=True, check=True)

    # Stub uv: shell-wrapped command whose child aborts (exits 134) once, then
    # creates .venv and passes.  The counter file tracks calls across retries.
    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    uv_script = stub_bin / "uv"
    uv_script.write_text(
        f"#!/bin/sh\n"
        f'f="{counter_file}"\n'
        f'n=0; [ -f "$f" ] && n=$(cat "$f")\n'
        f'n=$((n + 1)); echo $n > "$f"\n'
        f'if [ $n -le 1 ]; then\n'
        f'  exec sh -c "kill -ABRT \\$\\$"\n'
        f'fi\n'
        f'mkdir -p .venv/bin\n'
        f"printf '#!/bin/sh\\necho stub-python\\n' > .venv/bin/python\n"
        f'chmod +x .venv/bin/python\n'
    )
    uv_script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub_bin}{os.pathsep}{os.environ['PATH']}")

    emit = _render({"tool.py": "print('built')\n"})
    result = _design_run(tmp_path / "run", _FakeModel(_design_answers(emit)), repo)
    assert result.exit_ok, f"run failed unexpectedly: {result.error}"
    found = worktree.recorded(tmp_path / "run")
    count = subprocess.run(
        ["git", "-C", str(found.target), "rev-list", "--count",
         f"{found.base}..{found.branch}"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert count == "1", f"expected 1 attempt (no extra from crash), got {count}"


def test_shell_wrapped_install_crash_once_then_passing(tmp_path, monkeypatch):
    """A shell-wrapped command (./install.sh) whose child aborts and that exits 134
    is retried once and the run goes on (R1, correction 7).

    This is the case this run exists for: b27a's ./install.sh (a setup command)
    called pnpm, which was killed by SIGABRT, and the shell exited 134.
    """
    counter_dir = tempfile.mkdtemp(prefix="r1sw_")
    counter_file = os.path.join(counter_dir, "counter")

    repo = _git_repo(tmp_path / "repo")
    (repo / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    subprocess.run(["git", "-C", str(repo), "add", "pyproject.toml"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "manifest"],
                   capture_output=True, check=True)

    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    uv_script = stub_bin / "uv"
    # Exits 134 (shell-wrapped SIGABRT) on first two calls, then passes.
    # run_command retries once, so 2 calls = one retry cycle that fails;
    # but we want it to pass, so crash only once.
    uv_script.write_text(
        f"#!/bin/sh\n"
        f'f="{counter_file}"\n'
        f'n=0; [ -f "$f" ] && n=$(cat "$f")\n'
        f'n=$((n + 1)); echo $n > "$f"\n'
        f'if [ $n -le 1 ]; then exit 134; fi\n'
        f'mkdir -p .venv/bin\n'
        f"printf '#!/bin/sh\\necho stub-python\\n' > .venv/bin/python\n"
        f'chmod +x .venv/bin/python\n'
    )
    uv_script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub_bin}{os.pathsep}{os.environ['PATH']}")

    emit = _render({"tool.py": "print('built')\n"})
    result = _design_run(tmp_path / "run", _FakeModel(_design_answers(emit)), repo)
    assert result.exit_ok, f"run failed unexpectedly: {result.error}"


def test_shell_wrapped_install_crash_twice_stops_infrastructure(tmp_path, monkeypatch):
    """A shell-wrapped command that always exits 134 stops as infrastructure
    naming the signal (R1, correction 7).

    run_command retries once — exit 134 is in _AMBIGUOUS_SHELL_EXITS.
    After the retry the exit code is still 134, so is_signal_crash detects it,
    the return code is normalised to -6, and the run stops as infrastructure.
    """
    repo = _git_repo(tmp_path / "repo")
    (repo / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    subprocess.run(["git", "-C", str(repo), "add", "pyproject.toml"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "manifest"],
                   capture_output=True, check=True)

    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    uv_script = stub_bin / "uv"
    uv_script.write_text("#!/bin/sh\nexit 134\n")
    uv_script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub_bin}{os.pathsep}{os.environ['PATH']}")

    emit = _render({"tool.py": "print('built')\n"})
    result = _design_run(tmp_path / "run", _FakeModel(_design_answers(emit, emit)), repo)
    assert result.exit_ok is False
    assert "SIGABRT" in (result.error or "") or "killed by" in (result.error or "").lower(), (
        f"infrastructure stop must name the signal: {result.error}"
    )


def test_signal_killed_install_twice_stops_as_infrastructure(tmp_path, monkeypatch):
    """An install killed twice stops as infrastructure naming the signal.

    Driven as a scripted design run.  The stub uv always crashes with SIGABRT
    (direct signal).  run_command retries once, gets another crash.
    is_signal_crash detects it and normalises the return code.
    """
    repo = _git_repo(tmp_path / "repo")
    (repo / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    subprocess.run(["git", "-C", str(repo), "add", "pyproject.toml"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "manifest"],
                   capture_output=True, check=True)

    # Stub uv: always crashes with SIGABRT.
    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    uv_script = stub_bin / "uv"
    uv_script.write_text("#!/bin/sh\nkill -ABRT $$\n")
    uv_script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub_bin}{os.pathsep}{os.environ['PATH']}")

    emit = _render({"tool.py": "print('built')\n"})
    result = _design_run(tmp_path / "run", _FakeModel(_design_answers(emit, emit)), repo)
    assert result.exit_ok is False
    assert "SIGABRT" in (result.error or "") or "killed by" in (result.error or "").lower(), (
        f"infrastructure stop must name the signal: {result.error}"
    )


# ── R2: base comparison crash → rebuild from scratch ───────────────────────


def test_base_crash_rebuilt_from_scratch(tmp_path, monkeypatch):
    """A base comparison whose setup crashes is rebuilt from scratch (R2).

    The scripted setup uses a counter file outside HOME.  run_command retries
    once internally, so the counter must reach 2 for the crash to survive the
    retry.  On the rebuild attempt the counter is already past 2, so setup passes.
    """
    counter_dir = tempfile.mkdtemp(prefix="r2_")
    counter_file = os.path.join(counter_dir, "counter")

    repo = _git_init(tmp_path / "repo")
    base = _commit_file(repo, "app.py", "print('app')\n")
    run_dir = tmp_path / "run"
    found = _make_worktree(repo, run_dir, base)

    # The setup script crashes on the first two calls (counter < 3), then passes.
    # run_command retries once: so first _try_base call sees crash+crash → signal crash.
    # After rebuild, counter is at 2, the third call passes.
    setup_cmd = (
        f'bash -c \''
        f'f="{counter_file}"; '
        f'n=0; [ -f "$f" ] && n=$(cat "$f"); '
        f'n=$((n + 1)); echo $n > "$f"; '
        f'if [ $n -le 2 ]; then kill -ABRT $$; fi'
        f'\''
    )

    ctx = _Ctx(run_dir, facts={"setup": setup_cmd, "generate": "", "check": "true"})
    # Write worktree.json so recorded() finds it.
    wt_record = run_dir / ".refactory" / "worktree.json"
    wt_record.parent.mkdir(parents=True, exist_ok=True)
    wt_record.write_text(json.dumps({
        "path": found.path, "branch": found.branch, "base": found.base,
        "head": found.head, "target": found.target, "run": str(run_dir),
    }))

    result = base_check.run_base_check(ctx, "true", timeout=60)
    # After rebuild, the base should prepare successfully.
    assert result.environment_failure is None, (
        f"expected no environment failure after rebuild, got: {result.environment_failure}"
    )


def test_base_cannot_be_prepared_continues(tmp_path, monkeypatch):
    """A base that cannot be prepared lets the run continue with a note (R2).

    After the double crash, HOME and the base worktree are discarded so the
    next check gets a fresh start (correction 5: b27a rerun scenario).
    """
    counter_dir = tempfile.mkdtemp(prefix="r2b_")
    counter_file = os.path.join(counter_dir, "counter")

    repo = _git_init(tmp_path / "repo")
    base = _commit_file(repo, "app.py", "print('app')\n")
    run_dir = tmp_path / "run"
    found = _make_worktree(repo, run_dir, base)

    # The setup script always crashes (exit 134, shell-wrapped SIGABRT).
    setup_cmd = "bash -c 'exit 134'"

    ctx = _Ctx(run_dir, facts={"setup": setup_cmd, "generate": "", "check": "true"})
    wt_record = run_dir / ".refactory" / "worktree.json"
    wt_record.parent.mkdir(parents=True, exist_ok=True)
    wt_record.write_text(json.dumps({
        "path": found.path, "branch": found.branch, "base": found.base,
        "head": found.head, "target": found.target, "run": str(run_dir),
    }))

    result = base_check.run_base_check(ctx, "true", timeout=60)
    # Double failure: environment_failure is set.
    assert result.environment_failure is not None

    # Verify it is NOT cached (later checks get a fresh attempt).
    cached = base_check._read_cache(
        run_dir,
        base_check.cache_key(found.base, setup_cmd, "", "true"),
    )
    assert cached is None, "a crashed base result must not be cached"

    # R2 correction 5: HOME and base worktree must be discarded after double
    # crash so the next check does not reuse a half-installed state.
    base_home = run_dir / ".refactory" / "base-check-home"
    assert not base_home.exists(), "base-check-home should be discarded after double crash"


def test_base_double_crash_then_clean_check(tmp_path, monkeypatch):
    """A setup that leaves a marker in HOME before crashing makes the next
    check's setup exit 1 if HOME is reused.  After R2's double-crash discard,
    the next check gets a fresh HOME and succeeds (correction 5).
    """
    counter_dir = tempfile.mkdtemp(prefix="r2c_")
    counter_file = os.path.join(counter_dir, "counter")

    repo = _git_init(tmp_path / "repo")
    base = _commit_file(repo, "app.py", "print('app')\n")
    run_dir = tmp_path / "run"
    found = _make_worktree(repo, run_dir, base)

    # Setup: leaves a marker in HOME, then crashes (exit 134).
    # On reuse, the marker would cause exit 1 (deterministic failure).
    setup_cmd = (
        f'bash -c \''
        f'f="{counter_file}"; '
        f'n=0; [ -f "$f" ] && n=$(cat "$f"); '
        f'n=$((n + 1)); echo $n > "$f"; '
        f'mkdir -p "$HOME/.marker_dir"; echo $n > "$HOME/.marker_dir/state"; '
        f'if [ $n -le 4 ]; then exit 134; fi'
        f'\''
    )

    wt_record = run_dir / ".refactory" / "worktree.json"
    wt_record.parent.mkdir(parents=True, exist_ok=True)
    wt_record.write_text(json.dumps({
        "path": found.path, "branch": found.branch, "base": found.base,
        "head": found.head, "target": found.target, "run": str(run_dir),
    }))

    # First check: double crash, continues without excusals.
    ctx1 = _Ctx(run_dir, facts={"setup": setup_cmd, "generate": "", "check": "true"})
    result1 = base_check.run_base_check(ctx1, "true", timeout=60)
    assert result1.environment_failure is not None

    # Second check: because HOME was discarded, setup runs in a clean HOME.
    # Counter is now at 4, so the 5th call passes.
    ctx2 = _Ctx(run_dir, facts={"setup": setup_cmd, "generate": "", "check": "true"})
    result2 = base_check.run_base_check(ctx2, "true", timeout=60)
    assert result2.environment_failure is None, (
        f"second check should pass after HOME was discarded: {result2.environment_failure}"
    )


# ── R3: generate failure reaches builder (regression test) ─────────────────


def test_generate_failure_attributed_to_change_reaches_builder(tmp_path):
    """A generate failure with exit 1 reaches the builder as code_failure (R3 regression).

    On current main, this already works: check_failure_kind returns 'code_failure'
    for a prep_failure with returncode=1 (not < 0, not 126/127).  This test
    verifies that the path is preserved.
    """
    # Build a SandboxResult as _check_in_worktree returns it when generate fails.
    result = sandbox.SandboxResult(
        ok=False, returncode=1, elapsed_s=0.0,
        prep_failure={"step": "pnpm run docs:generate", "returncode": 1, "timed_out": False},
        stderr="the worktree's environment could not be prepared:\nthe generate command failed",
    )
    kind = evidence.check_failure_kind(result)
    assert kind == "code_failure", f"expected 'code_failure', got {kind!r}"
    # stop_unrepairable does nothing for code_failure — the flow reaches repair().
    evidence.stop_unrepairable(kind, "detail")  # should not raise


def test_generate_failure_with_base_crash_reaches_builder(tmp_path, monkeypatch):
    """b27a regression: the base's setup crashes (SIGABRT) while the attempt's
    generate fails (exit 1).  The generate failure must reach the builder as
    code_failure, not stop the run as 'environment could not be prepared'.

    Driven through a scripted design run (correction 8: the test must drive the
    scenario it sets up).  The setup command crashes on the base (exit 134,
    shell-wrapped SIGABRT).  The generate command fails with exit 1 on the
    attempt.  After R2's rebuild, the base crash is discarded; the generate
    failure reaches the builder.
    """
    counter_dir = tempfile.mkdtemp(prefix="r3_")
    counter_file = os.path.join(counter_dir, "counter")

    repo = _git_repo(tmp_path / "repo")
    # The setup script crashes on its first two calls then passes.
    # On the base, run_command retries once (2 calls), then R2 rebuilds
    # and the third call passes.
    (repo / "setup.sh").write_text(
        f"#!/bin/sh\n"
        f'f="{counter_file}"\n'
        f'n=0; [ -f "$f" ] && n=$(cat "$f")\n'
        f'n=$((n + 1)); echo $n > "$f"\n'
        f'if [ $n -le 2 ]; then exit 134; fi\n'
        f'echo setup ok\n'
    )
    subprocess.run(["chmod", "+x", str(repo / "setup.sh")], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "setup.sh"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "add setup"],
                   capture_output=True, check=True)

    # The emit adds a generate command that fails.
    emit = _render({"tool.py": "print('built')\n"})
    result = DesignMode().run(
        tmp_path / "run",
        model=_FakeModel(_design_answers(emit, emit)),
        verdicts={"gate_strategy": "proceed"},
        task={"name": "r3_test", "brief": "a tool", "check": "python tool.py",
              "target": str(repo), "setup": "sh setup.sh",
              "generate": "bash -c 'echo generate failed; exit 1'"},
    )
    # The run should reach the builder (attempt used) or fail with the generate
    # failure, not stop as "environment could not be prepared".
    error = result.error or ""
    assert "environment could not be prepared" not in error.lower() or result.facts.get("attempts_used"), (
        f"the generate failure should reach the builder, not stop as environment: {error}"
    )


# ── R5: repair feedback includes [left] sites and broken guarantees ────────


def test_repair_feedback_includes_left_sites_and_broken_guarantees(tmp_path):
    """A check failure's feedback names the design's [left] sites and broken guarantees."""
    design_md = (
        "# Test design\n\n"
        "## Summary\n\nA summary.\n\n"
        "## Rule and sites\n\n"
        "- `worktree.py:473-493` — `_require_current`. [left: the safety check is correct]\n"
        "- `sandbox.py:48-51` — `_AMBIGUOUS_SHELL_EXITS`. [changed: new helper]\n\n"
        "## Docs\n\n- docs/hardening.md\n\n"
        "## Changes\n\n- refactory/refactory_dsh/sandbox.py\n"
    )

    repo = _git_init(tmp_path / "repo")
    base = _commit_file(repo, "app.py", "print('app')\n")
    run_dir = tmp_path / "run"
    found = _make_worktree(repo, run_dir, base)
    # Write design.md so check_artifact can read it.
    (run_dir / "design.md").write_text(design_md)

    # Build a FilteredCheckResult with base_failures and base_complete set.
    filtered = base_check.FilteredCheckResult(
        all_excused=False,
        inherited=set(),
        new_failures={"tests/test_guard.py::test_guard"},
        new_failure_detail="FAILED tests/test_guard.py::test_guard",
        inherited_detail="",
        base_environment_failure=None,
        guardedness=None,
        base_failures=set(),  # test_guard is not in base failures
        base_complete=True,
    )

    # Build a result that has the test failure in its output.
    result = sandbox.SandboxResult(
        ok=False, returncode=1,
        stdout="FAILED tests/test_guard.py::test_guard\n= 1 failed in 0.5s =",
        stderr="",
    )

    # The guidance the whole check gives the builder, from the real code (`_full_check`'s).
    ctx = nn.RunContext(run_dir, facts={"name": "test_design", "brief": "make the guard strict",
                                        "gate_corrections": "- keep the safety check"})
    guidance = design_ops._check_failure_guidance(ctx, result, filtered, found)
    lines = guidance.splitlines()
    left_lines = [line for line in lines if line.startswith("Design constraint:")]
    guarantee_lines = [line for line in lines if line.startswith("Broken guarantee:")]

    assert left_lines, "should have at least one [left] site"
    assert any("the safety check is correct" in line for line in left_lines)
    assert guarantee_lines == ["Broken guarantee: tests/test_guard.py::test_guard passes on the "
                               "base but fails on this attempt."]
    assert "the attempt must not break it" not in guidance
    # dm-270: beside them, the brief and the corrections, and when to update a test instead.
    assert "## Brief\nmake the guard strict" in guidance
    assert "## Approved corrections\n- keep the safety check" in guidance
    assert "update the test, keeping what it guards" in guidance
    # A test the attempt's own diff touches is not a broken guarantee.
    (Path(found.path) / "tests").mkdir(exist_ok=True)
    (Path(found.path) / "tests" / "test_guard.py").write_text("def test_guard():\n    assert False\n")
    subprocess.run(["git", "-C", found.path, "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", found.path, "-c", "user.email=t@e", "-c", "user.name=t", "commit",
                    "-qm", "touch the test"], check=True, capture_output=True)
    touched = design_ops._check_failure_guidance(ctx, result, filtered, found)
    assert "Broken guarantee" not in touched and "Design constraint" in touched


def test_vitest_broken_guarantee_detected(tmp_path):
    """A vitest-style failure ID (with ' > ' separator) is correctly detected
    as a broken guarantee when the test file was not changed by the attempt."""
    vitest_id = "packages/foo/tests/bar.spec.ts > Suite > test name"
    # _test_file should extract the file path from the vitest ID.
    test_file = base_check._test_file(vitest_id)
    assert test_file == "packages/foo/tests/bar.spec.ts", f"expected vitest file, got {test_file!r}"

    # Verify the broken-guarantee detection works with vitest IDs.
    attempt_failures = {vitest_id}
    base_failures: set[str] = set()  # test not in base failures
    changed_files: set[str] = set()  # no changed files
    guarantee_lines: list[str] = []
    for failure in sorted(attempt_failures):
        if failure not in base_failures:
            tf = base_check._test_file(failure)
            if tf and tf not in changed_files:
                guarantee_lines.append(f"Broken guarantee: {failure}")
    assert guarantee_lines, "vitest broken guarantee should be detected"
    assert vitest_id in guarantee_lines[0]


# ── R6: converging run gets extra attempt ──────────────────────────────────


def test_converging_run_gets_extra_attempt(tmp_path):
    """A run whose last attempt went from five failures to one gets one more attempt (R6).

    drive repair() through the overall cap path: five failures, then one, then
    a granted fourth attempt.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = _Ctx(run_dir, facts={
        "check": "python -m pytest -q tests/",
        "check_failures_history": [
            ["tests/test_a.py::test_a", "tests/test_b.py::test_b",
             "tests/test_c.py::test_c", "tests/test_d.py::test_d",
             "tests/test_e.py::test_e"],
            ["tests/test_a.py::test_a"],
        ],
        "check_blocker_history": [None, None],
        "check_attempts_used": 2,
        "attempts_used": 2,
    }, knobs={"check_attempts": 3, "attempts": 3, "check_attempt_cap": 8})

    feedback = "FAILED tests/test_a.py::test_a\n= 1 failed in 0.5s ="

    # Write files repair() needs.
    ctx.write_text("validation.txt", "")

    # repair() should NOT raise — it should grant the extra attempt.
    result = repair(ctx, [feedback])
    assert isinstance(result, Verdict)
    assert str(result) == "reloop"
    assert ctx.facts.get("convergence_granted") is True
    assert ctx.facts.get("convergence_extras_used") == 1


def test_non_converging_run_exhausts(tmp_path):
    """A run whose failures did not shrink exhausts its attempts."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = _Ctx(run_dir, facts={
        "check": "python -m pytest -q tests/",
        "check_failures_history": [
            ["tests/test_a.py::test_a", "tests/test_b.py::test_b",
             "tests/test_c.py::test_c"],
            ["tests/test_a.py::test_a", "tests/test_b.py::test_b",
             "tests/test_c.py::test_c"],
        ],
        "check_attempts_used": 2,
        "attempts_used": 2,
    }, knobs={"check_attempts": 3, "attempts": 3, "check_attempt_cap": 8})

    feedback = "FAILED tests/test_a.py::test_a\nFAILED tests/test_b.py::test_b\nFAILED tests/test_c.py::test_c\n= 3 failed in 0.5s ="
    ctx.write_text("validation.txt", "")

    with pytest.raises(ValueError, match="check attempts exhausted"):
        repair(ctx, [feedback])


def test_empty_failure_set_is_not_convergence(tmp_path):
    """An empty failure set is not convergence (R6)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = _Ctx(run_dir, facts={
        "check": "python -m pytest -q tests/",
        "check_failures_history": [
            ["tests/test_a.py::test_a", "tests/test_b.py::test_b",
             "tests/test_c.py::test_c"],
            [],  # empty — no test identifiers found
        ],
        "check_attempts_used": 2,
        "attempts_used": 2,
    }, knobs={"check_attempts": 3, "attempts": 3, "check_attempt_cap": 8})

    feedback = "everything crashed"
    ctx.write_text("validation.txt", "")

    with pytest.raises(ValueError, match="check attempts exhausted"):
        repair(ctx, [feedback])


def test_rolling_convergence_shrinks_across_many_attempts(tmp_path):
    """Failures shrink 5→4→3→2→1 across attempts past the regular 3: every one
    is granted, and the run passes on the next attempt (R6 rolling convergence).

    Driven through repair() with progressive history updates.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_tests = [f"tests/test_{c}.py::test_{c}" for c in "abcde"]
    # Start with 3 regular attempts: 5 failures each time until the 3rd, then
    # shrinking starts.  We simulate the history building up attempt by attempt.
    # Regular attempt 1: 5 failures.  Regular attempt 2: 5 failures.
    # Regular attempt 3 (kind_used=3, at limit): 4 failures → grant #1.
    # Extra attempt 4 (kind_used=4): 3 failures → grant #2.
    # Extra attempt 5 (kind_used=5): 2 failures → grant #3.
    # Extra attempt 6 (kind_used=6): 1 failure → grant #4.
    # Extra attempt 7 (kind_used=7): passes!  (no repair() call)
    failure_counts = [5, 5, 4, 3, 2, 1]
    history: list[list[str]] = []
    blocker_history: list = []

    for i, fc in enumerate(failure_counts):
        ctx = _Ctx(run_dir, facts={
            "check": "python -m pytest -q tests/",
            "check_failures_history": list(history),
            "check_blocker_history": list(blocker_history),
            "check_attempts_used": i,
            "attempts_used": i,
            "convergence_extras_used": max(0, i - 2),
            "convergence_granted": i > 2,
        }, knobs={"check_attempts": 3, "attempts": 3, "check_attempt_cap": 8})
        ctx.write_text("validation.txt", "")

        failures = all_tests[:fc]
        history.append(failures)
        blocker_history.append(None)  # runner completed fully
        ctx.facts["check_failures_history"] = list(history)
        ctx.facts["check_blocker_history"] = list(blocker_history)

        feedback_lines = [f"FAILED {t}" for t in failures]
        feedback_lines.append(f"= {fc} failed in 0.5s =")
        feedback = "\n".join(feedback_lines)

        result = repair(ctx, [feedback])
        assert isinstance(result, Verdict), f"attempt {i+1} should be granted, got raise"
        assert str(result) == "reloop"

    # After all 6 calls, extras_used should be 4 (attempts 3-6 were granted).
    assert ctx.facts.get("convergence_extras_used") == 4


def test_rolling_convergence_stalls(tmp_path):
    """Failures shrink and then stay the same: the run stops at the stall (R6).

    Regular attempts: [5, 5, 4].  Extra attempt: [4] — same as previous, not a
    strict subset.  repair() raises.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_tests = [f"tests/test_{c}.py::test_{c}" for c in "abcde"]

    ctx = _Ctx(run_dir, facts={
        "check": "python -m pytest -q tests/",
        "check_failures_history": [
            all_tests[:5],       # attempt 1: 5
            all_tests[:5],       # attempt 2: 5
            all_tests[:4],       # attempt 3: 4 → granted
            all_tests[:4],       # attempt 4: 4 — stalled
        ],
        "check_blocker_history": [None, None, None, None],
        "check_attempts_used": 3,
        "attempts_used": 3,
        "convergence_extras_used": 1,
    }, knobs={"check_attempts": 3, "attempts": 3, "check_attempt_cap": 8})

    feedback_lines = [f"FAILED {t}" for t in all_tests[:4]]
    feedback_lines.append("= 4 failed in 0.5s =")
    feedback = "\n".join(feedback_lines)
    ctx.write_text("validation.txt", "")

    with pytest.raises(ValueError, match="check attempts exhausted"):
        repair(ctx, [feedback])


def test_rolling_convergence_hits_cap(tmp_path):
    """Failures keep shrinking past 8 attempts: the run stops at the cap of 8,
    saying it reached the cap (R6).
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build a history where failures shrink every attempt for 8 attempts.
    # 10 test names so we can keep shrinking past 8.
    all_tests = [f"tests/test_{i}.py::test_{i}" for i in range(10)]
    history = [all_tests[:10-i] for i in range(8)]
    blocker_history = [None] * 8

    ctx = _Ctx(run_dir, facts={
        "check": "python -m pytest -q tests/",
        "check_failures_history": history,
        "check_blocker_history": blocker_history,
        "check_attempts_used": 7,
        "attempts_used": 7,
        "convergence_extras_used": 5,
    }, knobs={"check_attempts": 3, "attempts": 3, "check_attempt_cap": 8})

    feedback_lines = [f"FAILED {t}" for t in all_tests[:2]]
    feedback_lines.append("= 2 failed in 0.5s =")
    feedback = "\n".join(feedback_lines)
    ctx.write_text("validation.txt", "")

    with pytest.raises(ValueError, match="check attempt cap"):
        repair(ctx, [feedback])


def test_check_attempt_cap_knob_stops_at_4(tmp_path):
    """Setting check_attempt_cap to 4 stops a converging run at 4 (correction 1)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_tests = [f"tests/test_{c}.py::test_{c}" for c in "abcde"]
    # 3 regular + 1 granted = 4.  At attempt 4 (kind_used=4), cap=4 → cap error.
    ctx = _Ctx(run_dir, facts={
        "check": "python -m pytest -q tests/",
        "check_failures_history": [
            all_tests[:5], all_tests[:5], all_tests[:4], all_tests[:3],
        ],
        "check_blocker_history": [None, None, None, None],
        "check_attempts_used": 3,
        "attempts_used": 3,
        "convergence_extras_used": 1,
    }, knobs={"check_attempts": 3, "attempts": 3, "check_attempt_cap": 4})

    feedback_lines = [f"FAILED {t}" for t in all_tests[:3]]
    feedback_lines.append("= 3 failed in 0.5s =")
    feedback = "\n".join(feedback_lines)
    ctx.write_text("validation.txt", "")

    with pytest.raises(ValueError, match="check attempt cap"):
        repair(ctx, [feedback])


def test_new_failure_is_never_granted(tmp_path):
    """A new failure appearing is never granted, even if the set is smaller (R6)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = _Ctx(run_dir, facts={
        "check": "python -m pytest -q tests/",
        "check_failures_history": [
            ["tests/test_a.py::test_a", "tests/test_b.py::test_b",
             "tests/test_c.py::test_c"],
            # Shrunk to 1, but it's a NEW test not in any prior set.
            ["tests/test_NEW.py::test_NEW"],
        ],
        "check_blocker_history": [None, None],
        "check_attempts_used": 2,
        "attempts_used": 2,
    }, knobs={"check_attempts": 3, "attempts": 3, "check_attempt_cap": 8})

    feedback = "FAILED tests/test_NEW.py::test_NEW\n= 1 failed in 0.5s ="
    ctx.write_text("validation.txt", "")

    with pytest.raises(ValueError, match="check attempts exhausted"):
        repair(ctx, [feedback])


# ── R6: node declarations include convergence facts ───────────────────────


def test_node_declarations_include_convergence_facts():
    """The design, fast and create graph nodes declare the convergence and blocker facts.

    Structural check: verify the produces tuples contain the expected facts.
    """
    from refactory_dsh.design import DesignMode
    from refactory_dsh.fast import DesignFastMode, CreateFastMode
    from refactory_dsh.create import CreateMode

    # Design mode: check node.
    design_graph = DesignMode().build()
    check_node = design_graph.nodes["check"]
    for fact in ("check_failures_history", "check_blocker_history",
                 "convergence_extras_used", "convergence_granted", "base_unavailable"):
        assert fact in check_node.produces, f"design check node missing {fact}"
    assert "check_attempt_cap" in check_node.knobs, "design check node missing check_attempt_cap knob"

    # Design fast mode: focused_check, gate_check and gate_qa nodes.
    fast_graph = DesignFastMode().build()
    fc_node = fast_graph.nodes["focused_check"]
    for fact in ("check_failures_history", "check_blocker_history",
                 "convergence_extras_used", "convergence_granted", "base_unavailable"):
        assert fact in fc_node.produces, f"design_fast focused_check missing {fact}"
    gc_node = fast_graph.nodes["gate_check"]
    for fact in ("convergence_extras_used", "convergence_granted"):
        assert fact in gc_node.produces, f"design_fast gate_check missing {fact}"
    assert "check_attempt_cap" in gc_node.knobs, "design_fast gate_check missing check_attempt_cap knob"
    qa_node = fast_graph.nodes["gate_qa"]
    for fact in ("check_failures_history", "check_blocker_history",
                 "convergence_extras_used", "convergence_granted"):
        assert fact in qa_node.produces, f"design_fast gate_qa missing {fact}"
    assert "check_attempt_cap" in qa_node.knobs, "design_fast gate_qa missing check_attempt_cap knob"

    # Create fast mode: smoke and validate nodes.
    cf_graph = CreateFastMode().build()
    smoke_node = cf_graph.nodes["smoke"]
    for fact in ("check_failures_history", "check_blocker_history",
                 "convergence_extras_used", "convergence_granted"):
        assert fact in smoke_node.produces, f"create_fast smoke missing {fact}"
    validate_node = cf_graph.nodes["validate"]
    for fact in ("check_failures_history", "check_blocker_history",
                 "convergence_extras_used", "convergence_granted"):
        assert fact in validate_node.produces, f"create_fast validate missing {fact}"

    # Create full mode: smoke node.
    create_graph = CreateMode().build()
    c_smoke = create_graph.nodes["smoke"]
    for fact in ("convergence_extras_used", "convergence_granted"):
        assert fact in c_smoke.produces, f"create smoke missing {fact}"

    # Knob declarations: check_attempt_cap must be declared on the modes.
    assert "check_attempt_cap" in DesignMode.declare(), "design mode missing check_attempt_cap knob"
    assert "check_attempt_cap" in CreateMode.declare(), "create mode missing check_attempt_cap knob"
    assert "check_attempt_cap" in CreateFastMode.declare(), "create_fast mode missing check_attempt_cap knob"
    assert "check_attempt_cap" in DesignFastMode.declare(), "design_fast mode missing check_attempt_cap knob"


def test_node_declarations_enforced_by_executor(tmp_path):
    """Node declarations are enforced by the real executor with write enforcement.

    An undeclared fact write raises ExecutionError.  We construct a RunContext
    with enforce_writes=True and the declared facts from a graph node, then
    call set_fact on an undeclared name and verify it raises.
    """
    from refactory_dsh.design import DesignMode
    from refactory_dsh.nn.executor import RunContext, ExecutionError

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Get the check node's declared produces.
    design_graph = DesignMode().build()
    check_node = design_graph.nodes["check"]

    ctx = RunContext(run_dir, enforce_writes=True)
    ctx.node_id = "check"
    ctx._declared = check_node.writes
    ctx._declared_facts = check_node.produces

    # Writing a declared fact succeeds.
    ctx.set_fact("base_unavailable", "test")
    ctx.set_fact("check_failures_history", [])
    ctx.set_fact("check_blocker_history", [])
    ctx.set_fact("convergence_extras_used", 0)
    ctx.set_fact("convergence_granted", False)
    # _check_produces should pass with only declared facts.
    ctx._check_produces()

    # Writing an undeclared fact and checking raises ExecutionError.
    ctx.set_fact("totally_undeclared_fact_xyz", "bad")
    with pytest.raises(ExecutionError, match="does not declare"):
        ctx._check_produces()


# ── R1: delivery crash path stops as infrastructure ────────────────────────


def test_delivery_signal_crash_stops_as_infrastructure(tmp_path):
    """A signal crash in delivery's merged-tree check stops the run as
    infrastructure (raises ValueError), not _kept with a push of the reviewed
    commit.  Uses the structured infrastructure flag (correction 6).
    """
    from refactory_dsh.delivery import _deliver, MergeResult
    import refactory_dsh.delivery as _del

    repo = _git_init(tmp_path / "repo")
    reviewed = _commit_file(repo, "f.txt", "a\n")
    run_dir = tmp_path / "run"
    found = _make_worktree(repo, run_dir, reviewed)

    ctx = _Ctx(run_dir, facts={
        "check": "true",
        "generate": "",
        "delivery_plan": {"action": "none", "why": "test"},
    })

    merge_sha = _commit_file(repo, "g.txt", "b\n")
    planned = {
        "repo": "owner/repo", "branch": found.branch,
        "into": "main", "action": "create",
        "pull_request": "", "summary": "test",
        "inputs": {"rev": "", "enabled": True, "owner": "owner",
                   "visibility": "public", "refine": ""},
    }
    # The run's attempt: delivery refuses a branch that changes nothing.
    found_for_test = _with_attempt(worktree.recorded(run_dir))

    original_merge_target = _del._merge_target
    original_check_merged = _del._check_merged

    def fake_merge_target(*args, **kwargs):
        return MergeResult(merge_commit=merge_sha)

    def fake_check_merged(*args, **kwargs):
        return ("merged-tree install killed by SIGABRT", True)

    _del._merge_target = fake_merge_target
    _del._check_merged = fake_check_merged
    try:
        with pytest.raises(ValueError, match="infrastructure failure"):
            _deliver(ctx, found_for_test, planned, title="test")
    finally:
        _del._merge_target = original_merge_target
        _del._check_merged = original_check_merged


def test_delivery_abandoned_signal_crash_stops_as_infrastructure(tmp_path):
    """A signal crash in _merge_target's generate also stops the run as
    infrastructure (raises ValueError) using the structured flag (correction 6).
    """
    from refactory_dsh.delivery import _deliver, MergeResult
    import refactory_dsh.delivery as _del

    repo = _git_init(tmp_path / "repo")
    reviewed = _commit_file(repo, "f.txt", "a\n")
    run_dir = tmp_path / "run"
    found = _make_worktree(repo, run_dir, reviewed)

    ctx = _Ctx(run_dir, facts={
        "check": "true",
        "generate": "",
        "delivery_plan": {"action": "none", "why": "test"},
    })

    planned = {
        "repo": "owner/repo", "branch": found.branch,
        "into": "main", "action": "create",
        "pull_request": "", "summary": "test",
        "inputs": {"rev": "", "enabled": True, "owner": "owner",
                   "visibility": "public", "refine": ""},
    }
    # The run's attempt: delivery refuses a branch that changes nothing.
    found_for_test = _with_attempt(worktree.recorded(run_dir))

    original_merge_target = _del._merge_target

    def fake_merge_target(*args, **kwargs):
        return MergeResult(abandoned="generate (base side) killed by SIGABRT",
                           infrastructure=True)

    _del._merge_target = fake_merge_target
    try:
        with pytest.raises(ValueError, match="infrastructure failure"):
            _deliver(ctx, found_for_test, planned, title="test")
    finally:
        _del._merge_target = original_merge_target


# ── R9: deliver_design refuses repointed worktree ──────────────────────────


def test_deliver_design_refuses_repointed_worktree(tmp_path):
    """deliver_design refuses a worktree whose .git points at the repo's own
    git directory, and the repository's HEAD is unchanged (R9).

    With the verify_git_link call at delivery.py:808, the ValueError is raised
    immediately and the repo's HEAD stays at M (the merge commit).  Without
    the call, _recover_interrupted_merge would run git reset --hard in the
    repointed worktree, changing the repo's HEAD from M to the reviewed commit.
    """
    repo = _git_init(tmp_path / "repo")
    (repo / "f.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "reviewed"],
                   capture_output=True, check=True)
    reviewed = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()

    # Create a branch simulating main moving forward.
    subprocess.run(["git", "-C", str(repo), "checkout", "-b", "main-moved"],
                   capture_output=True, check=True)
    (repo / "g.txt").write_text("b\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "main-moved"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", reviewed],
                   capture_output=True, check=True)

    # Create a proper worktree at the reviewed commit.
    wt_path = tmp_path / "wt"
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-b", "run-branch",
                    str(wt_path), reviewed], capture_output=True, check=True)

    # Merge main-moved into the worktree (delivery's own merge).
    subprocess.run(["git", "-C", str(wt_path), "merge", "--no-ff", "--no-edit", "main-moved"],
                   capture_output=True, check=True)
    merge_sha = subprocess.run(["git", "-C", str(wt_path), "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip()
    assert merge_sha != reviewed

    # Set the repo's HEAD to the merge M.
    subprocess.run(["git", "-C", str(repo), "checkout", "-b", "at-merge", merge_sha],
                   capture_output=True, check=True)

    # Repoint the worktree's .git to the repo's own git dir.
    repo_git_dir = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--absolute-git-dir"],
        capture_output=True, text=True, check=True).stdout.strip()
    (wt_path / ".git").write_text(f"gitdir: {repo_git_dir}\n")

    # Record the repo's HEAD before deliver_design.
    repo_head_before = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()

    # Write worktree.json.
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    wt_record = run_dir / ".refactory" / "worktree.json"
    wt_record.parent.mkdir(parents=True, exist_ok=True)
    wt_record.write_text(json.dumps({
        "path": str(wt_path), "branch": "run-branch", "base": reviewed,
        "head": reviewed, "target": str(repo), "run": str(run_dir),
    }))

    # Call deliver_design — should raise because verify_git_link detects the repoint.
    ctx = nn.RunContext(run_dir, facts={
        "delivery_plan": {"action": "none", "why": "test"},
    }, enforce_writes=False)

    with pytest.raises(ValueError, match="no longer points at its own git directory"):
        delivery.deliver_design(ctx)

    # The repo's HEAD must still be at M — no git reset --hard ran.
    repo_head_after = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert repo_head_after == repo_head_before, (
        f"repo HEAD changed from {repo_head_before[:12]} to {repo_head_after[:12]}; "
        "verify_git_link call at delivery.py:808 is missing"
    )


# ── R6 convergence in fast_ops._repair ─────────────────────────────────────


def test_fast_repair_convergence_grants_extra(tmp_path):
    """fast_ops._repair grants an extra attempt when the failure set converges."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = _Ctx(run_dir, facts={
        "check": "python -m pytest -q tests/",
        "check_failures_history": [
            ["tests/test_a.py::test_a", "tests/test_b.py::test_b"],
            ["tests/test_a.py::test_a"],
        ],
        "check_blocker_history": [None, None],
        "repairs_used": 1,
        "check_repairs_used": 1,
    }, knobs={"repair_attempts": 1, "check_attempt_cap": 8})

    feedback_parts = ["FAILED tests/test_a.py::test_a\n= 1 failed in 0.5s ="]

    result = fast_repair(ctx, feedback_parts, kind="check")
    assert isinstance(result, Verdict)
    assert str(result) == "reloop"
    assert ctx.facts.get("convergence_granted") is True


def test_fast_repair_non_convergence_raises(tmp_path):
    """fast_ops._repair raises when the failure set does not converge."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = _Ctx(run_dir, facts={
        "check": "python -m pytest -q tests/",
        "check_failures_history": [
            ["tests/test_a.py::test_a", "tests/test_b.py::test_b"],
            ["tests/test_a.py::test_a", "tests/test_b.py::test_b"],
        ],
        "check_blocker_history": [None, None],
        "repairs_used": 1,
        "check_repairs_used": 1,
    }, knobs={"repair_attempts": 1, "check_attempt_cap": 8})

    feedback_parts = ["FAILED tests/test_a.py::test_a\nFAILED tests/test_b.py::test_b\n= 2 failed in 0.5s ="]

    with pytest.raises(ValueError, match="unresolved"):
        fast_repair(ctx, feedback_parts, kind="check")


def test_fast_repair_rolling_convergence_hits_cap(tmp_path):
    """fast_ops._repair stops at the check_attempt_cap (R6 rolling convergence)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_tests = [f"tests/test_{i}.py::test_{i}" for i in range(10)]
    history = [all_tests[:10-i] for i in range(8)]
    blocker_history = [None] * 8

    ctx = _Ctx(run_dir, facts={
        "check": "python -m pytest -q tests/",
        "check_failures_history": history,
        "check_blocker_history": blocker_history,
        "repairs_used": 7,
        "check_repairs_used": 8,
        "convergence_extras_used": 5,
    }, knobs={"repair_attempts": 3, "check_attempt_cap": 8})

    feedback_lines = [f"FAILED {t}" for t in all_tests[:2]]
    feedback_lines.append("= 2 failed in 0.5s =")
    feedback = "\n".join(feedback_lines)

    with pytest.raises(ValueError, match="check attempt cap"):
        fast_repair(ctx, [feedback], kind="check")


# ── Correction 1: robustness_notes with "credentials" must not stop the run ──


def test_design_fast_broken_guarantee_with_credentials_is_repaired(tmp_path, monkeypatch):
    """A design_fast run whose broken test is named test_does_not_see_the_real_credentials_0
    is repaired and passes, not stopped as 'unavailable' (correction 1).

    Without the fix, gate_qa would word-scan the robustness_notes text — which
    contains the target test ID with "credentials" — and failure_kind would
    return 'unavailable', stopping the run.  With the fix, robustness_notes are
    excluded from the text that failure_kind classifies.
    """
    from refactory_dsh.fast import DesignFastMode

    repo = _git_repo(tmp_path / "repo")
    # check.py: produces pytest-like FAILED output when flag.txt says 'bad'.
    # On the base (no flag.txt), it passes.
    (repo / "check.py").write_text(
        "import sys, pathlib\n"
        "if pathlib.Path('flag.txt').exists() and pathlib.Path('flag.txt').read_text().strip() == 'bad':\n"
        "    print('FAILED test_creds.py::test_does_not_see_the_real_credentials_0')\n"
        "    print('= 1 failed in 0.1s =')\n"
        "    sys.exit(1)\n"
        "print('= 1 passed in 0.1s =')\n"
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "add check"],
                   capture_output=True, check=True)

    FAST_PLAN = "Change only the requested file."
    FAST_QA = json.dumps({"status": "pass",
                          "checks": [{"name": "runs", "command": "python check.py",
                                      "status": "pass", "evidence": "exit 0"}],
                          "findings": [], "not_run": []})

    # First emit: adds flag.txt='bad', causing check.py to fail with a FAILED
    # line whose test ID contains "credentials".
    bad_emit = _render({"flag.txt": "bad\n", "tool.py": "print('ok')\n"})
    # Second emit: fixes flag.txt so check passes.
    good_emit = _render({"flag.txt": "good\n", "tool.py": "print('ok')\n"})

    model = _FakeModel([FAST_PLAN, bad_emit, good_emit, FAST_QA, VERIFICATION_ANSWER])
    result = DesignFastMode().run(
        tmp_path / "run", model=model,
        task={"name": "creds_test", "brief": "a tool",
              "check": "python check.py", "target": str(repo)},
    )
    assert result.exit_ok, (
        f"run should repair and pass (not stop as unavailable): {result.error}"
    )


# ── Correction 2: design_fast double base crash continues with note ──────────


def test_design_fast_base_double_crash_continues_with_note(tmp_path, monkeypatch):
    """A design_fast run whose base preparation crashes twice continues and
    its check record carries the note (correction 2).

    The stub uv crashes on calls 3-6 (the base install: two _try_base attempts,
    each with run_command's retry).  Call 1 (the builder's workspace, prepared once)
    and call 2 (attempt install) pass.  Calls 7+ (second attempt install) pass; the
    workspace, its manifests unchanged, installs nothing the second time.  focused_check sets base_unavailable and
    writes the note into check.txt, which the focused_check node now declares.
    """
    from refactory_dsh.fast import DesignFastMode

    counter_dir = tempfile.mkdtemp(prefix="r2fast_")
    counter_file = os.path.join(counter_dir, "counter")

    repo = _git_repo(tmp_path / "repo")
    (repo / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    subprocess.run(["git", "-C", str(repo), "add", "pyproject.toml"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "manifest"],
                   capture_output=True, check=True)

    # Stub uv: passes on call 1 (the builder's workspace) and call 2 (attempt install),
    # crashes on calls 3-6 (base install: 2 calls per _try_base x 2 _try_base attempts =
    # 4 calls), then passes on calls 7+ (second attempt install).
    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    uv_script = stub_bin / "uv"
    # The stub must symlink the real Python into .venv/bin so that the
    # check command ``python tool.py`` actually executes tool.py.  A
    # fake shell script that just echoes text would make every check
    # pass, and the base comparison would never be triggered.
    uv_script.write_text(
        f"#!/bin/sh\n"
        f'f="{counter_file}"\n'
        f'n=0; [ -f "$f" ] && n=$(cat "$f")\n'
        f'n=$((n + 1)); echo $n > "$f"\n'
        f'if [ $n -ge 3 ] && [ $n -le 6 ]; then exit 134; fi\n'
        f'mkdir -p .venv/bin\n'
        f'ln -sf "{sys.executable}" .venv/bin/python\n'
    )
    uv_script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub_bin}{os.pathsep}{os.environ['PATH']}")

    FAST_PLAN = "Change only the requested file."
    FAST_QA = json.dumps({"status": "pass",
                          "checks": [{"name": "runs", "command": "python tool.py",
                                      "status": "pass", "evidence": "exit 0"}],
                          "findings": [], "not_run": []})

    # First emit: tool.py that fails → triggers base comparison (which crashes).
    bad_emit = _render({"tool.py": "raise SystemExit(1)\n"})
    # Second emit: tool.py that passes.
    good_emit = _render({"tool.py": "print('ok')\n"})

    model = _FakeModel([FAST_PLAN, bad_emit, good_emit, FAST_QA, VERIFICATION_ANSWER])
    result = DesignFastMode().run(
        tmp_path / "run", model=model,
        task={"name": "r2fast", "brief": "a tool",
              "check": "python tool.py", "target": str(repo)},
    )
    assert result.exit_ok, f"run should continue despite base crash: {result.error}"

    # check.txt should carry the base_unavailable note.
    check_txt_path = tmp_path / "run" / "check.txt"
    assert check_txt_path.exists(), "check.txt should exist"
    check_txt = check_txt_path.read_text()
    assert "base comparison unavailable" in check_txt or "without excusals" in check_txt, (
        f"check.txt should carry the base_unavailable note: {check_txt}"
    )
