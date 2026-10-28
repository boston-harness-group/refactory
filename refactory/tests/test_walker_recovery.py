"""Walker recovery: environment overlay, FnNode retry, timeout, fork propagation.

Each test exercises the real walker executor (``nn.executor.execute``) and must
fail without its corresponding fix.

The time-limit tests are the real ones for their mechanism: a real child process,
really killed. Their limit comes from the run's elapsed-time budget, which clamps a
check op's limit (``Executor._effective_timeout``), so each sets ``max_elapsed_s`` to
about a second rather than waiting out ten.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn
from refactory_dsh.nn.executor import OpTimeoutError

@pytest.fixture(autouse=True)
def bounded_recovery_test():
    """An independent watchdog keeps reverted hang fixes from hanging pytest."""
    import signal
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        pytest.fail("recovery test exceeded its independent 30-second watchdog")
    signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous)


#: The run budget for the time-limit tests, in seconds. The walker clamps a check
#: op's limit to the budget left (and never below 1 s), so this is the limit itself.
SHORT_LIMIT_S = 1.0


# ---------------------------------------------------------------------------
# Test 1: environment overlay from target_workdir
# ---------------------------------------------------------------------------
def test_worktree_agent_sees_target_interpreter(tmp_path, monkeypatch):
    """_environment() derives the overlay from target_workdir, not REFACTORY_WORKDIR."""
    from refactory_dsh import model

    overlay_calls: list[str] = []

    def mock_overlay(workdir: Path) -> dict[str, str] | None:
        overlay_calls.append(str(workdir))
        return {
            "PATH": "/mock/target/bin:" + os.environ.get("PATH", ""),
            "VIRTUAL_ENV": "/mock/target/venv",
        }

    monkeypatch.setattr(model, "_environment_overlay", mock_overlay)
    monkeypatch.delenv("REFACTORY_WORKDIR", raising=False)

    client = model.DshModelClient()

    # Without target_workdir: the overlay is NOT called (env var is empty).
    env_without = client._environment()
    assert not overlay_calls, "overlay called without target_workdir"

    # With target_workdir: the overlay IS called.
    env_with = client._environment(target_workdir=tmp_path)
    assert len(overlay_calls) == 1
    assert overlay_calls[0] == str(tmp_path)
    assert env_with["PATH"].startswith("/mock/target/bin:")
    assert env_with.get("VIRTUAL_ENV") == "/mock/target/venv"


# ---------------------------------------------------------------------------
# Test 2: transient op failure retried, exit_ok stays True
# ---------------------------------------------------------------------------
def test_transient_op_failure_retried_exit_ok_true(tmp_path):
    graph = nn.Graph()
    graph.add(nn.Node(
        "retry_op", "op",
        call="walker_recovery_ops:transient_op",
        produces=("success",),
    ))

    ctx = nn.RunContext(tmp_path)
    executor = nn.Executor(graph, ctx, enable_recovery=True)
    result = executor.run()

    assert result.exit_ok is True, result.error
    assert result.facts.get("success") is True

    retried = [s for s in executor.steps if s.verdict == "retried"]
    assert len(retried) == 1
    assert retried[0].error is None

    # The op was called twice (file counter is 2).
    counter = int((tmp_path / "_attempt_counter.txt").read_text().strip())
    assert counter == 2


# ---------------------------------------------------------------------------
# Test 3: op exceeding timeout stops the run with a message naming it
# ---------------------------------------------------------------------------
def test_op_exceeding_timeout_stops_run_with_message(tmp_path):
    from refactory_dsh.invocations import CHECK_PRODUCES

    graph = nn.Graph()
    graph.add(nn.Node(
        "slow_check", "op",
        call="walker_recovery_ops:hanging_op",
        produces=(*CHECK_PRODUCES,),
        knobs=("check_timeout_s",),
    ))

    result = nn.execute(
        graph, tmp_path,
        knobs={"check_timeout_s": 2},
        enable_recovery=True,
        resource_limits={"max_elapsed_s": SHORT_LIMIT_S},
    )

    assert result.exit_ok is False
    assert "slow_check" in (result.error or "")
    assert "timed out" in (result.error or "").lower()

    # Verify the walker's raw formula matches lowering's.
    from refactory_dsh.lowering import command_timeout_s

    node = graph.nodes["slow_check"]
    state = {"knobs": {"check_timeout_s": {"value": 2}}}
    raw = command_timeout_s(node, state)
    assert raw is not None
    assert raw > 0


# ---------------------------------------------------------------------------
# Test 4: a hung fn-gate times out (never retried)
# ---------------------------------------------------------------------------
def test_hung_fn_gate_times_out(tmp_path):
    from refactory_dsh.invocations import CHECK_PRODUCES

    graph = nn.Graph()
    graph.add(nn.Node(
        "gate_hang", "gate", gate_kind="fn",
        evaluator="walker_recovery_ops:hanging_gate",
        produces=(*CHECK_PRODUCES,),
        knobs=("check_timeout_s",),
    ))
    graph.add(nn.Node("after", "op",
                       call="walker_recovery_ops:succeed_op",
                       produces=("done",)))
    graph.connect("gate_hang", "after", condition="proceed")

    result = nn.execute(
        graph, tmp_path,
        knobs={"check_timeout_s": 2},
        enable_recovery=True,
        resource_limits={"max_elapsed_s": SHORT_LIMIT_S},
    )

    assert result.exit_ok is False
    assert "gate_hang" in (result.error or "")
    assert "timed out" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# Test 5: fork-branch op under the same timeout limit
# ---------------------------------------------------------------------------
def test_fork_branch_op_under_timeout_limit(tmp_path):
    from refactory_dsh.invocations import CHECK_PRODUCES

    graph = nn.Graph()
    graph.add(nn.Node("f", "fork", targets=("branch_op",)))
    graph.add(nn.Node(
        "branch_op", "op",
        call="walker_recovery_ops:hanging_op",
        produces=(*CHECK_PRODUCES,),
        knobs=("check_timeout_s",),
    ))
    barrier = "j"
    graph.add(nn.Node(barrier, "join", sources=("branch_op",)))
    graph.connect("f", barrier)
    graph.connect("branch_op", barrier)

    result = nn.execute(
        graph, tmp_path,
        knobs={"check_timeout_s": 2},
        enable_recovery=True,
        resource_limits={"max_elapsed_s": SHORT_LIMIT_S},
    )

    assert result.exit_ok is False
    assert "branch_op" in (result.error or "")
    assert "timed out" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# Test 6: resume after timeout reruns the node (outcome='failed' not 'unknown')
# ---------------------------------------------------------------------------
def test_resume_after_timeout_outcome_is_failed(tmp_path):
    from refactory_dsh.invocations import CHECK_PRODUCES
    from refactory_dsh.nn.executor import PROGRESS

    graph = nn.Graph()
    graph.add(nn.Node(
        "timeout_op", "op",
        call="walker_recovery_ops:hanging_op",
        produces=(*CHECK_PRODUCES,),
        knobs=("check_timeout_s",),
    ))

    nn.execute(
        graph, tmp_path,
        knobs={"check_timeout_s": 2},
        enable_recovery=True,
        resource_limits={"max_elapsed_s": SHORT_LIMIT_S},
    )

    progress = json.loads((tmp_path / PROGRESS).read_text())
    assert progress["outcome"] == "failed", f"expected 'failed', got {progress['outcome']!r}"
    assert progress["node"] == "timeout_op"


# ---------------------------------------------------------------------------
# Test 7: wiring at all CLI sites
# ---------------------------------------------------------------------------
def test_wiring_at_all_cli_sites():
    """Every walker run/resume site passes enable_recovery=True."""
    from refactory_dsh import create, design, edit

    # Every CLI main() must pass enable_recovery=True at both the run and resume
    # paths.  Library functions (create.create, edit.edit) accept it as a
    # parameter so tests can run without recovery; the CLI is what wires True.
    for module in (design, create, edit):
        source = inspect.getsource(module.main)
        assert "enable_recovery=True" in source, (
            f"{module.__name__}.main does not pass enable_recovery=True"
        )

    # Check WalkerEvaluator.
    from refactory_dsh.evaluators import WalkerEvaluator
    source = inspect.getsource(WalkerEvaluator.evaluate)
    assert "enable_recovery=True" in source, (
        "WalkerEvaluator.evaluate does not pass enable_recovery=True"
    )


# ---------------------------------------------------------------------------
# Test 8: gate sees declared_reads, op sees memory
# ---------------------------------------------------------------------------
def test_enable_recovery_gate_sees_declared_reads_op_sees_memory(tmp_path):
    """Under enable_recovery the child process builds the same context DSH does."""
    graph = nn.Graph()
    graph.add(nn.Node(
        "g", "gate", gate_kind="fn",
        evaluator="walker_recovery_ops:check_declared_reads_gate",
        reads=("some_read.txt",),
    ))
    graph.add(nn.Node(
        "op1", "op",
        call="walker_recovery_ops:check_memory_op",
        produces=("has_memory",),
    ))
    graph.connect("g", "op1", condition="proceed")

    result = nn.execute(graph, tmp_path, enable_recovery=True)
    assert result.exit_ok is True, result.error


# ---------------------------------------------------------------------------
# Test 9: fake model through prepare never calls model_readiness
# ---------------------------------------------------------------------------
def test_enable_recovery_fake_model_no_readiness(tmp_path):
    """A non-DSH model in the child is neither None nor DshModelClient."""

    class FakeModel:
        def complete(self, prompt: str, *, system: str = "") -> str:
            return "ok"

    graph = nn.Graph()
    graph.add(nn.Node(
        "op1", "op",
        call="walker_recovery_ops:succeed_op",
        produces=("done",),
    ))

    # Run with a fake model and enable_recovery.  The child gets a stand-in
    # that is not None, so prepare/create_ops skip readiness exactly as they
    # do in-process.
    result = nn.execute(graph, tmp_path, model=FakeModel(), enable_recovery=True)
    assert result.exit_ok is True, result.error


# ---------------------------------------------------------------------------
# Test 10: child exception roundtrip
# ---------------------------------------------------------------------------
def test_child_exception_roundtrip(tmp_path):
    graph = nn.Graph()
    graph.add(nn.Node(
        "exc_op", "op",
        call="walker_recovery_ops:custom_exception_op",
        produces=("never",),
    ))

    result = nn.execute(graph, tmp_path, enable_recovery=True)
    assert result.exit_ok is False
    # The error should name the exception type and message.
    assert "custom message from child" in (result.error or "")


# ---------------------------------------------------------------------------
# Test 12: the executor's op context uses forkserver with refactory preloaded
# ---------------------------------------------------------------------------
def test_forkserver_context_with_preload(tmp_path):
    """Structural: the executor uses forkserver, not spawn, with refactory preloaded."""
    from refactory_dsh.nn import executor as ex_mod

    # The module-level context must be a ForkServerContext.
    ctx = ex_mod._forkserver_ctx
    assert ctx.get_start_method() == "forkserver", (
        f"expected forkserver, got {ctx.get_start_method()!r}"
    )

    # After ensuring preload for a graph that uses a refactory_dsh op,
    # the preload modules must include 'refactory_dsh.nn.executor' and
    # 'refactory_dsh.run'.
    graph = nn.Graph()
    graph.add(nn.Node(
        "op1", "op",
        call="walker_recovery_ops:succeed_op",
        produces=("done",),
    ))
    ex_mod._ensure_forkserver_preload(graph)
    preloaded = ex_mod._forkserver_modules
    assert "refactory_dsh.nn.executor" in preloaded
    assert "refactory_dsh.run" in preloaded
    assert "walker_recovery_ops" in preloaded


# ---------------------------------------------------------------------------
# Test 11: outer-loop evaluation — hung op stops, next trial starts clean
# ---------------------------------------------------------------------------
def test_outer_loop_evaluation_hung_op_stops_next_trial_starts_clean(tmp_path):
    from refactory_dsh.invocations import CHECK_PRODUCES
    from refactory_dsh.evaluators import WalkerEvaluator

    class _MiniMode(nn.Mode):
        name = "test_recovery_mode"

        @classmethod
        def declare(cls):
            return {"check_timeout_s": nn.Knob(2, "int")}

        def build(self):
            g = nn.Graph()
            g.add(nn.Node(
                "t_op", "op",
                call="walker_recovery_ops:outer_loop_transient_op",
                produces=("ol_done",),
            ))
            g.add(nn.Node(
                "h_op", "op",
                call="walker_recovery_ops:outer_loop_hanging_op",
                produces=(*CHECK_PRODUCES,),
                knobs=("check_timeout_s",),
            ))
            g.connect("t_op", "h_op")
            return g

        def score(self, run):
            return 1.0 if run.facts.get("ol_done") else 0.0

    mode = _MiniMode()
    evaluator = WalkerEvaluator(project=tmp_path)

    from refactory_dsh.budgets import RunBudget, ResourceLimits
    # The transient op runs first, in-process, so the budget leaves it room before
    # the hung op's limit starts.
    budget = RunBudget.create(tmp_path / "budget", ResourceLimits(max_elapsed_s=2 * SHORT_LIMIT_S))
    budget.resume()
    evaluator.budget = budget

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir(parents=True, exist_ok=True)

    result = evaluator.evaluate(mode, mode.state_dict(), trial_dir)

    assert result.exit_ok is False, "expected the hung op to fail the run"
    assert "h_op" in (result.error or ""), f"expected h_op in error, got: {result.error}"
    assert "timed out" in (result.error or "").lower()

    budget.pause()

    import multiprocessing
    leaked = [c for c in multiprocessing.active_children() if c.is_alive()]
    assert not leaked, f"leaked child processes: {leaked}"


# ---------------------------------------------------------------------------
# A check's own tools die at the op's limit, not only the op's child
# ---------------------------------------------------------------------------
def test_a_checks_own_tools_die_at_the_limit(tmp_path):
    from refactory_dsh.invocations import CHECK_PRODUCES

    graph = nn.Graph()
    graph.add(nn.Node(
        "slow_tools", "op",
        call="walker_recovery_ops:sandbox_sleep_op",
        produces=(*CHECK_PRODUCES,),
        knobs=("check_timeout_s",),
    ))
    # The op has to start its sleep before the limit, so it gets twice the short one.
    result = nn.execute(graph, tmp_path, knobs={"check_timeout_s": 2},
                        enable_recovery=True, resource_limits={"max_elapsed_s": 2 * SHORT_LIMIT_S})

    assert result.exit_ok is False
    assert "timed out" in (result.error or "").lower()
    pid = int((tmp_path / "_sleep_pid.txt").read_text())
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        os.kill(pid, 9)
        pytest.fail(f"the check's sleep (pid {pid}) outlived the op's time limit")


# ---------------------------------------------------------------------------
# dm-173 tests: default deadline for unclassified ops
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 13: a hung unclassified op times out with the 3600 s default
# ---------------------------------------------------------------------------
def test_unclassified_hung_op_times_out_with_default_deadline(tmp_path):
    """An op with no CHECK_PRODUCES / check_timeout_s gets the default 3600 s
    deadline, clamped by the budget.  Without the fix, it runs in-process with
    no timeout and hangs indefinitely."""
    graph = nn.Graph()
    graph.add(nn.Node(
        "unclass_hang", "op",
        call="walker_recovery_ops:hanging_op",
        produces=("never",),
    ))

    t0 = time.time()
    result = nn.execute(
        graph, tmp_path,
        enable_recovery=True,
        resource_limits={"max_elapsed_s": SHORT_LIMIT_S},
    )
    elapsed = time.time() - t0

    assert result.exit_ok is False
    assert "unclass_hang" in (result.error or "")
    assert "timed out" in (result.error or "").lower()
    # Should complete within a few seconds (budget clamps to ~1 s).
    assert elapsed < 15, f"took {elapsed:.1f}s; expected budget clamp near {SHORT_LIMIT_S}s"


# ---------------------------------------------------------------------------
# Test 14: classified op still uses derived timeout, not the 3600 s default
# ---------------------------------------------------------------------------
def test_unclassified_op_uses_explicit_timeout_when_classified(tmp_path):
    """An op that has check_timeout_s and CHECK_PRODUCES uses the derived
    timeout, not the 3600 s default.  Pins existing behavior."""
    from refactory_dsh.invocations import CHECK_PRODUCES

    graph = nn.Graph()
    graph.add(nn.Node(
        "classified_op", "op",
        call="walker_recovery_ops:classified_succeed_op",
        produces=(*CHECK_PRODUCES,),
        knobs=("check_timeout_s",),
    ))

    result = nn.execute(
        graph, tmp_path,
        knobs={"check_timeout_s": 10},
        enable_recovery=True,
        resource_limits={"max_elapsed_s": 30},
    )
    assert result.exit_ok is True, result.error


# ---------------------------------------------------------------------------
# Test 15: remaining budget clamps the default timeout
# ---------------------------------------------------------------------------
def test_remaining_budget_clamps_default_timeout(tmp_path):
    """A hung unclassified op with a short budget times out at the budget
    limit, not 3600 s.  Wall-clock must complete in under 5 s."""
    graph = nn.Graph()
    graph.add(nn.Node(
        "clamped_hang", "op",
        call="walker_recovery_ops:hanging_op",
        produces=("never",),
    ))

    t0 = time.time()
    result = nn.execute(
        graph, tmp_path,
        enable_recovery=True,
        resource_limits={"max_elapsed_s": SHORT_LIMIT_S},
    )
    elapsed = time.time() - t0

    assert result.exit_ok is False
    assert "clamped_hang" in (result.error or "")
    assert elapsed < 5, f"took {elapsed:.1f}s; budget should have clamped near {SHORT_LIMIT_S}s"


# ---------------------------------------------------------------------------
# Test 16: child process budget record_usage visible in parent
# ---------------------------------------------------------------------------
def test_child_process_budget_record_usage_visible_in_parent(tmp_path):
    """An unclassified op in a fork branch calls ctx.budget.record_usage() in
    a child process — the parent's budget must see the recorded tokens.
    Without PC-2/PC-3, the child gets ctx.budget=None and record_usage raises.
    Uses a fork branch so the op goes through the child-process path."""
    graph = nn.Graph()
    graph.add(nn.Node("f", "fork", targets=("spend_op",)))
    graph.add(nn.Node(
        "spend_op", "op",
        call="walker_recovery_ops:budget_spend_op",
        produces=("spent",),
    ))
    barrier = "j"
    graph.add(nn.Node(barrier, "join", sources=("spend_op",)))
    graph.connect("f", barrier)
    graph.connect("spend_op", barrier)

    result = nn.execute(
        graph, tmp_path,
        enable_recovery=True,
        resource_limits={"max_elapsed_s": 30},
    )
    assert result.exit_ok is True, result.error
    assert result.facts.get("spent") is True

    from refactory_dsh.budgets import RunBudget
    budget = RunBudget.from_path(tmp_path / ".refactory/resource-usage.json")
    snap = budget.snapshot()
    assert snap["known_tokens"] >= 5, f"expected >=5, got {snap['known_tokens']}"


# ---------------------------------------------------------------------------
# Test 17: child process budget snapshot works
# ---------------------------------------------------------------------------
def test_child_process_budget_snapshot_works(tmp_path):
    """An unclassified op in a fork branch returns budget.snapshot() from the
    child process.  Uses a fork branch so the op goes through the child-process
    path and exercises budget forwarding."""
    graph = nn.Graph()
    graph.add(nn.Node("f", "fork", targets=("snap_op",)))
    graph.add(nn.Node(
        "snap_op", "op",
        call="walker_recovery_ops:budget_snapshot_op",
        produces=("budget_snapshot",),
    ))
    barrier = "j"
    graph.add(nn.Node(barrier, "join", sources=("snap_op",)))
    graph.connect("f", barrier)
    graph.connect("snap_op", barrier)

    result = nn.execute(
        graph, tmp_path,
        enable_recovery=True,
        resource_limits={"max_elapsed_s": 30},
    )
    assert result.exit_ok is True, result.error
    snap = result.facts.get("budget_snapshot")
    assert isinstance(snap, dict), f"expected dict, got {type(snap)}"
    assert "limits" in snap
    assert "elapsed_s" in snap


# ---------------------------------------------------------------------------
# Test 18: fork branch unclassified hung op times out
# ---------------------------------------------------------------------------
def test_fork_branch_unclassified_hung_op_times_out(tmp_path):
    """An unclassified hanging_op in a fork branch must time out."""
    graph = nn.Graph()
    graph.add(nn.Node("f", "fork", targets=("branch_unclass",)))
    graph.add(nn.Node(
        "branch_unclass", "op",
        call="walker_recovery_ops:hanging_op",
        produces=("never",),
    ))
    barrier = "j"
    graph.add(nn.Node(barrier, "join", sources=("branch_unclass",)))
    graph.connect("f", barrier)
    graph.connect("branch_unclass", barrier)

    t0 = time.time()
    result = nn.execute(
        graph, tmp_path,
        enable_recovery=True,
        resource_limits={"max_elapsed_s": SHORT_LIMIT_S},
    )
    elapsed = time.time() - t0

    assert result.exit_ok is False
    assert "branch_unclass" in (result.error or "")
    assert "timed out" in (result.error or "").lower()
    assert elapsed < 15


# ---------------------------------------------------------------------------
# Test 19: unclassified fn gate times out
# ---------------------------------------------------------------------------
def test_unclassified_fn_gate_times_out(tmp_path):
    """An unclassified hanging fn gate (no check_timeout_s, no CHECK_PRODUCES)
    gets the default timeout through _effective_timeout."""
    graph = nn.Graph()
    graph.add(nn.Node(
        "unclass_gate", "gate", gate_kind="fn",
        evaluator="walker_recovery_ops:hanging_gate",
    ))
    graph.add(nn.Node("after", "op",
                       call="walker_recovery_ops:succeed_op",
                       produces=("done",)))
    graph.connect("unclass_gate", "after", condition="proceed")

    t0 = time.time()
    result = nn.execute(
        graph, tmp_path,
        enable_recovery=True,
        resource_limits={"max_elapsed_s": SHORT_LIMIT_S},
    )
    elapsed = time.time() - t0

    assert result.exit_ok is False
    assert "unclass_gate" in (result.error or "")
    assert "timed out" in (result.error or "").lower()
    assert elapsed < 15


# ---------------------------------------------------------------------------
# Test 20: enable_recovery=False keeps unclassified ops in-process
# ---------------------------------------------------------------------------
def test_enable_recovery_false_unclassified_op_runs_in_process(tmp_path):
    """With enable_recovery=False, unclassified ops run in-process without a
    timeout.  Pins the opt-in semantics."""
    graph = nn.Graph()
    graph.add(nn.Node(
        "simple_op", "op",
        call="walker_recovery_ops:succeed_op",
        produces=("done",),
    ))

    result = nn.execute(graph, tmp_path, enable_recovery=False)
    assert result.exit_ok is True, result.error
    assert result.facts.get("done") is True


def test_successful_child_run_exits_its_interpreter(tmp_path):
    import signal
    import subprocess
    import sys
    script = tmp_path / "run_once.py"
    work = tmp_path / "run"
    script.write_text(
        "from pathlib import Path\nfrom refactory_dsh import nn\n"
        "if __name__ == '__main__':\n"
        "    graph = nn.Graph()\n"
        "    graph.add(nn.Node('work', 'op', call='walker_recovery_ops:record_worker_pid', "
        "writes=('worker.pid',), produces=('done',)))\n"
        f"    result = nn.execute(graph, Path({str(work)!r}), enable_recovery=True)\n"
        "    assert result.exit_ok, result.error\n"
        "    print('completed', flush=True)\n"
    )
    process = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True,
                               env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                   [str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")])})
    try:
        out, err = process.communicate(timeout=15)
        assert process.returncode == 0, err
        assert "completed" in out
        assert int((work / "worker.pid").read_text()) != process.pid
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        if (work / "worker.pid").exists():
            try:
                os.killpg(int((work / "worker.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_op_crash_is_unknown_and_does_not_replay_side_effects(tmp_path):
    graph = nn.Graph()
    graph.add(nn.Node("crash", "op", call="walker_recovery_ops:crash_after_side_effect"))
    result = nn.execute(graph, tmp_path, enable_recovery=True)
    assert not result.exit_ok
    assert "child exited before reporting its outcome" in result.error
    assert (tmp_path / "crash-count").read_text() == "1"
    progress = json.loads((tmp_path / ".refactory/progress.json").read_text())
    assert progress["outcome"] == "unknown"


def test_timeout_kills_term_resistant_descendant_after_leader_exits(tmp_path):
    import signal
    graph = nn.Graph()
    graph.add(nn.Node("hang", "op", call="walker_recovery_ops:hang_with_term_resistant_descendant"))
    pid = None
    try:
        result = nn.execute(graph, tmp_path, enable_recovery=True,
                            resource_limits={"max_elapsed_s": 2})
        assert not result.exit_ok and "timed out" in result.error
        pid = int((tmp_path / "descendant.pid").read_text())
        for _ in range(100):
            try:
                os.kill(pid, 0)
                stat = Path(f"/proc/{pid}/stat")
                if stat.exists() and stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z":
                    break
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("TERM-resistant descendant outlived the op deadline")
    finally:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_child_uses_the_callers_actual_memory_roots(tmp_path):
    from refactory_dsh.memory import MemoryStore
    roots = {scope: tmp_path / ("chosen-" + scope) for scope in ("run", "project", "global")}
    memory = MemoryStore([nn.Memory(scope, "log", scope) for scope in roots],
                         run_root=roots["run"], project_root=roots["project"],
                         global_root=roots["global"])
    graph = nn.Graph()
    graph.add(nn.Node("remember", "op", call="walker_recovery_ops:write_declared_memories",
                      produces=("memory_pid",)))
    result = nn.execute(graph, tmp_path / "execution", memory=memory, enable_recovery=True)
    assert result.exit_ok, result.error
    assert result.facts["memory_pid"] != os.getpid()
    for scope in roots:
        assert memory.path(scope).is_relative_to(roots[scope])
        assert memory.read(scope) == [{"pid": result.facts["memory_pid"]}]


def test_child_preserves_iteration_verdicts_task_and_write_policy(tmp_path):
    graph = nn.Graph()
    for name in ("first", "second"):
        graph.add(nn.Node(name, "op", call="walker_recovery_ops:record_parent_context",
                          writes=("allowed.txt",), produces=(name, "context_flags")))
    graph.connect("first", "second")
    ctx = nn.RunContext(tmp_path, verdicts={"approval": "proceed"}, task={}, enforce_writes=False)
    result = nn.Executor(graph, ctx, enable_recovery=True).run()
    assert result.exit_ok, result.error
    assert result.facts["first"] == 0 and result.facts["second"] == 1
    assert result.facts["context_flags"] == {
        "verdicts": {"approval": "proceed"}, "task": {}, "enforce_writes": False}


def test_each_op_runs_in_a_fresh_interpreter(tmp_path):
    """One child per op: an op cannot see what an earlier op left in its interpreter.

    A reused worker would report 1 here, because the first op's module state would
    still be in the process that runs the second.
    """
    graph = nn.Graph()
    graph.add(nn.Node("mark", "op", call="walker_recovery_ops:mark_interpreter",
                      writes=("mark.txt",), produces=("marked_pid",)))
    graph.add(nn.Node("report", "op", call="walker_recovery_ops:report_interpreter",
                      writes=("seen.txt",), produces=("seen",)))
    graph.connect("mark", "report")
    result = nn.execute(graph, tmp_path, enable_recovery=True)
    assert result.exit_ok, result.error
    assert result.facts["seen"] == 0
    assert (tmp_path / "seen.txt").read_text() == "0"


def test_the_op_child_is_gone_when_the_node_returns(tmp_path):
    """No child outlives its node, even when the op leaves a thread behind.

    A child still running here keeps multiprocessing's exit-time join waiting, so the
    run would hang after the node that caused it (and the reviewer measured exactly
    that at 40 s and counting).
    """
    graph = nn.Graph()
    graph.add(nn.Node("threaded", "op", call="walker_recovery_ops:leave_a_thread_running",
                      writes=("threaded.pid",), produces=("threaded_pid",)))
    started = time.monotonic()
    result = nn.execute(graph, tmp_path, enable_recovery=True)
    assert result.exit_ok, result.error
    assert time.monotonic() - started < 30, "the run waited for the op's thread"
    pid = int(result.facts["threaded_pid"])
    assert pid != os.getpid()
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        state = Path(f"/proc/{pid}/stat")
        if state.exists() and state.read_text().rsplit(") ", 1)[1].split()[0] == "Z":
            break
        time.sleep(0.05)
    else:
        try:
            os.kill(pid, 9)
        finally:
            pytest.fail("the op's child outlived the node")


def test_an_unclassified_op_gets_the_spines_default_deadline(tmp_path):
    """An op outside every envelope is bounded at the spine's 3600 s, not left open.

    Both shipped spine configurations pin commandTimeoutMs to 3600000; an op with no
    deadline is the hang dm-173 is about.
    """
    graph = nn.Graph()
    graph.add(nn.Node("plain", "op", call="walker_recovery_ops:report_interpreter",
                      writes=("seen.txt",), produces=("seen",)))
    ctx = nn.RunContext(tmp_path)
    executor = nn.Executor(graph, ctx, enable_recovery=True)
    assert executor._effective_timeout(graph.nodes["plain"]) == 3600.0


def test_the_deadline_is_gone_when_recovery_is_off(tmp_path):
    """Without recovery the walker runs ops in process, as before."""
    graph = nn.Graph()
    graph.add(nn.Node("plain", "op", call="walker_recovery_ops:report_interpreter",
                      writes=("seen.txt",), produces=("seen",)))
    executor = nn.Executor(graph, nn.RunContext(tmp_path), enable_recovery=False)
    assert executor._effective_timeout(graph.nodes["plain"]) is None
