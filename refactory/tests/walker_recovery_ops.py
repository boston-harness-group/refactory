"""Module-level op/gate functions for walker recovery tests.

Under ``multiprocessing.get_context("forkserver")``, the child process imports
by name, so every callable must be a plain module-level function.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any


def transient_op(ctx: Any) -> dict[str, Any]:
    """Fail on the first call, succeed on the second.  File-based counter."""
    counter_file = ctx.workdir / "_attempt_counter.txt"
    attempt = 1
    if counter_file.exists():
        attempt = int(counter_file.read_text().strip()) + 1
    counter_file.write_text(str(attempt))
    if attempt == 1:
        raise RuntimeError("transient failure")
    return {"success": True}


def hanging_op(ctx: Any) -> dict[str, Any]:
    """Hang forever on a threading Event."""
    threading.Event().wait()
    return {}  # pragma: no cover


def hanging_gate(ctx: Any) -> str:
    """A fn-gate evaluator that hangs forever."""
    threading.Event().wait()
    return "proceed"  # pragma: no cover


def succeed_op(ctx: Any) -> dict[str, Any]:
    """An op that always succeeds."""
    return {"done": True}


def noop_gate(ctx: Any) -> str:
    """A fn-gate evaluator that always proceeds."""
    return "proceed"


def custom_exception_op(ctx: Any) -> dict[str, Any]:
    """Raise a custom exception for round-trip testing."""
    raise _CustomTestError("custom message from child")


class _CustomTestError(RuntimeError):
    """A test-only exception class that the parent can rebuild."""
    pass


# Alias so resolve() finds it at module level (the class itself is private).
CustomTestError = _CustomTestError


def check_declared_reads_gate(ctx: Any) -> str:
    """A gate that asserts declared_reads is populated."""
    if not ctx.declared_reads:
        raise RuntimeError("declared_reads is empty in child")
    return "proceed"


def check_memory_op(ctx: Any) -> dict[str, Any]:
    """An op that asserts memory is present."""
    # Just check that ctx.memory is not None (if declarations are set up).
    return {"has_memory": ctx.memory is not None}


def outer_loop_transient_op(ctx: Any) -> dict[str, Any]:
    """Fail transiently, using file counter."""
    counter_file = ctx.workdir / "_ol_attempt.txt"
    attempt = 1
    if counter_file.exists():
        attempt = int(counter_file.read_text().strip()) + 1
    counter_file.write_text(str(attempt))
    if attempt == 1:
        raise RuntimeError("outer loop transient failure")
    return {"ol_done": True}


def outer_loop_hanging_op(ctx: Any) -> dict[str, Any]:
    """Hang forever for outer-loop timeout testing."""
    threading.Event().wait()
    return {}  # pragma: no cover


def sandbox_sleep_op(ctx: Any) -> dict[str, Any]:
    """Start a long sleep through sandbox.run_command, as a check's tools run, recording its pid."""
    from refactory_dsh import sandbox

    pid_file = ctx.workdir / "_sleep_pid.txt"
    sandbox.run_command(f"echo $$ > {pid_file}; exec sleep 600", workdir=ctx.workdir, timeout_s=900)
    return {}


def classified_succeed_op(ctx: Any) -> dict[str, Any]:
    """An op that succeeds and produces check-like facts."""
    return {"commit": "abc123", "files": [], "inherited_failures": [], "changed_functions": []}


def budget_spend_op(ctx: Any) -> dict[str, Any]:
    """Record usage through ctx.budget in a child process (dm-173 test)."""
    ctx.budget.record_usage(str(ctx.workdir), 5)
    return {"spent": True}


def budget_snapshot_op(ctx: Any) -> dict[str, Any]:
    """Return budget.snapshot() as a fact to prove it works in a child (dm-173 test)."""
    snap = ctx.budget.snapshot()
    return {"budget_snapshot": snap}


def record_worker_pid(ctx):
    import os
    ctx.write_text("worker.pid", str(os.getpid()))
    return {"done": True}


def crash_after_side_effect(ctx):
    import os
    path = ctx.workdir / "crash-count"
    count = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(count))
    if count == 1:
        os._exit(17)
    return {}


def hang_with_term_resistant_descendant(ctx):
    import subprocess
    import sys
    import time
    path = ctx.workdir / "descendant.pid"
    source = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(path)!r}).write_text(str(os.getpid())); time.sleep(600)"
    )
    subprocess.Popen([sys.executable, "-c", source])
    while not path.exists():
        time.sleep(0.01)
    threading.Event().wait()


def write_declared_memories(ctx):
    import os
    for name in ctx.memory.namespaces():
        ctx.memory.append(name, {"pid": os.getpid()})
    return {"memory_pid": os.getpid()}


def record_parent_context(ctx):
    ctx.write_text("extra.txt", "permitted by parent policy")
    return {ctx.node_id: ctx.iteration, "context_flags": {
        "verdicts": ctx.verdicts, "task": ctx.task, "enforce_writes": ctx.enforce_writes}}


_SEEN_IN_THIS_INTERPRETER: dict[str, int] = {}


def mark_interpreter(ctx):
    """Record that this interpreter has run an op. A reused child would remember."""
    _SEEN_IN_THIS_INTERPRETER["marked"] = 1
    ctx.write_text("mark.txt", str(os.getpid()))
    return {"marked_pid": os.getpid()}


def report_interpreter(ctx):
    """Report whether this interpreter had already run an op of this graph."""
    ctx.write_text("seen.txt", str(_SEEN_IN_THIS_INTERPRETER.get("marked", 0)))
    return {"seen": _SEEN_IN_THIS_INTERPRETER.get("marked", 0)}


def leave_a_thread_running(ctx):
    """Return normally while a non-daemon thread keeps this interpreter alive."""
    threading.Thread(target=lambda: time.sleep(120), daemon=False).start()
    ctx.write_text("threaded.pid", str(os.getpid()))
    return {"threaded_pid": os.getpid()}
