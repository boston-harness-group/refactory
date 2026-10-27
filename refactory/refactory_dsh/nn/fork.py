"""Bounded execution of independent fork branches on the Python walker.

The shared workspace is not a sandbox. Declared cross-branch dependencies and
writes are refused before dispatch; arbitrary effects of user code remain its
responsibility, as with the sequential executor.
"""
from __future__ import annotations

import copy
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .executor import Executor
    from .graph import Node

# A bound on simultaneous branch work, not another mode knob or API setting. Design
# mode's review fork has five branches: three reviews, the verifier and the check.
MAX_WORKERS = 5


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents or (
        left.is_file() and right.is_file() and left.samefile(right)
    )


def _settle_missing(workdir: Path, members: list[Node], error: str) -> None:
    """Settle the deliveries of an optional branch recorded missing (dm-232).

    An agent whose model call raised -- a session past its wall, say -- never reaches
    ``agent_delivery.finish``, so its pending marker would outlive the branch, and a
    later recovery of the run directory would refuse on it as an unknown outcome. The
    spine's worker settles it the same way (``worker_node``); ``invalidate`` is a no-op
    for a member that left no marker.
    """
    from refactory_dsh.agent_delivery import invalidate

    agents = [member.id for member in members if member.kind == "agent"]
    if not agents:
        return
    try:
        invalidate(Path(workdir), agents, f"optional branch missing: {error}")
    except Exception as settle_error:  # the branch stays missing either way
        import logging

        logging.getLogger(__name__).warning(
            "could not settle the delivery markers of %s: %s", ", ".join(agents), settle_error)


def _record_missing(workdir: Path, node: str, error: str) -> None:
    """Record why an optional branch's node failed, where the spine records it: the worker
    and `refactory_dsh.run` write ``reviews/<node>-missing.txt`` with the error, and a gate
    that reads the branch's output says why it is missing from it (the early asks judge's
    error reaches `design_ops._read_early_asks` this way on both runtimes)."""
    reviews = Path(workdir) / "reviews"
    reviews.mkdir(parents=True, exist_ok=True)
    (reviews / f"{node}-missing.txt").write_text(f"{error}\n")


def _branches(executor: Executor, fork: Node, barrier: str) -> list[list[Node]]:
    """Admit disjoint flat branches before any branch can have side effects."""
    from .executor import ExecutionError

    graph, ctx = executor.graph, executor.ctx
    branches: list[list[Node]] = []
    owned: set[str] = set()
    for target in fork.targets:
        seen: set[str] = set()
        pending = [target]
        while pending:
            current = pending.pop()
            if current == barrier or current in seen:
                continue
            if current == fork.id or current not in graph.nodes:
                raise ExecutionError(f"fork {fork.id!r} has an invalid branch at {current!r}")
            node = graph.nodes[current]
            if node.kind not in ("op", "agent", "gate"):
                raise ExecutionError(
                    f"fork {fork.id!r}: nested {node.kind!r} node {current!r} is not "
                    "supported inside a Python fork branch"
                )
            seen.add(current)
            pending.extend(edge.target for edge in graph.out_edges(current))
        if not seen or seen & owned:
            raise ExecutionError(f"fork {fork.id!r} has empty or overlapping branches at {target!r}")
        owned.update(seen)
        branches.append([graph.nodes[name] for name in sorted(seen)])

    def paths(nodes: list[Node], field: str) -> list[Path]:
        result = []
        root = ctx.workdir.resolve()
        for node in nodes:
            for name in getattr(node, field):
                path = ctx.path(name).resolve()
                if not path.is_relative_to(root):
                    raise ExecutionError(f"fork {fork.id!r}: {field} path {name!r} escapes its workspace")
                result.append(path)
        return result

    writes = [paths(nodes, "writes") for nodes in branches]
    reads = [paths(nodes, "reads") for nodes in branches]
    facts = [{name for node in nodes for name in node.produces} for nodes in branches]
    # A sibling's newly produced fact is no more available than its file. A fact
    # from before the fork can be read by all branches, but not concurrently reset.
    import re
    consumed = [{match for node in nodes for match in re.findall(
        r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", node.prompt_template)}
        for nodes in branches]
    for index, branch_writes in enumerate(writes):
        if any(_overlap(path, ctx.workdir.resolve() / ".refactory") for path in branch_writes):
            raise ExecutionError(f"fork {fork.id!r}: a branch declares executor metadata as output")
        for other in range(index):
            if facts[index] & facts[other]:
                raise ExecutionError(f"fork {fork.id!r}: branches declare the same produced facts")
            if facts[index] & consumed[other] or facts[other] & consumed[index]:
                raise ExecutionError(f"fork {fork.id!r}: a branch prompt depends on a sibling's facts")
            if (any(_overlap(a, b) for a in branch_writes for b in writes[other] + reads[other])
                    or any(_overlap(a, b) for a in writes[other] for b in reads[index])):
                raise ExecutionError(f"fork {fork.id!r}: branches have conflicting declared file access")
    return branches


def run_fork(executor: Executor, node: Node) -> str | None:
    """Wait for all scheduled work; merge in target order; never join on failure."""
    from .executor import ExecutionError, Executor, RunContext
    from .record import RunRecord

    edges = executor.graph.out_edges(node.id)
    if len(edges) != 1 or edges[0].condition is not None:
        raise ExecutionError(f"fork {node.id!r} needs one unconditional join edge")
    barrier = edges[0].target
    join = executor.graph.nodes.get(barrier)
    if join is None or join.kind != "join":
        raise ExecutionError(f"fork {node.id!r} does not lead to a join")
    branches = _branches(executor, node, barrier)
    owned = {member.id for branch in branches for member in branch}
    if set(join.sources) - owned:
        raise ExecutionError(f"fork {node.id!r}: join sources must belong to its branches")
    ctx = executor.ctx
    from refactory_dsh.memory import MemoryStore
    from refactory_dsh.model import DshModelClient

    # Memory namespaces have no per-node ownership. Preserve the old serial
    # behavior when a store could be shared, rather than rejecting valid modes.
    serial_memory = ctx.memory is not None and (
        type(ctx.memory) is not MemoryStore or bool(ctx.memory.describe())
    )
    # The protocol never required custom clients to be thread-safe. Only the
    # shipped exact client is known to use independent subprocess invocations.
    serial_model = ctx.model is not None and type(ctx.model) is not DshModelClient
    serial_gate = any(member.kind == "gate" and member.gate_kind != "fn"
                      and not ctx.verdicts.get(member.id)
                      for branch in branches for member in branch)
    serial = serial_memory or serial_model or serial_gate
    workers = 1 if serial else min(MAX_WORKERS, len(branches))
    schedule = {"workers": workers, "execution": "caller-thread" if serial else "concurrent",
                "shared_memory_serial": serial_memory, "custom_model_serial": serial_model,
                "live_gate_answers_serial": serial_gate}

    lock = threading.Lock()
    cancelled = threading.Event()
    branch_model = (ctx.model.with_cancellation(cancelled)
                    if not serial and type(ctx.model) is DshModelClient else ctx.model)
    remaining = executor.max_steps - len(executor.steps)
    status: dict[str, dict[str, Any]] = {target: {"status": "queued"} for target in node.targets}

    def reserve() -> bool:
        nonlocal remaining
        with lock:
            if cancelled.is_set():
                raise CancelledError("fork cancelled before the next branch step")
            if remaining <= 0:
                return False
            remaining -= 1
            return True

    def record(target: str, state: str, error: str | None = None) -> None:
        with lock:
            status[target] = {"status": state, **({"error": error} if error else {})}
            executor._record(node.id, branches=copy.deepcopy(status), schedule=schedule)

    children = []
    # Clone all inputs before scheduling any work: an uncopyable value cannot
    # leave half a fork running. The shipped client uses separate subprocesses;
    # arbitrary custom clients retain caller-thread serial execution.
    for _ in branches:
        child_ctx = RunContext(
            ctx.workdir, knobs=copy.deepcopy(ctx.knobs), facts=copy.deepcopy(ctx.facts),
            verdicts=copy.deepcopy(ctx.verdicts), task=copy.deepcopy(ctx.task),
            model=branch_model, memory=ctx.memory, enforce_writes=ctx.enforce_writes,
        )
        # Each branch keeps its own record, starting from the run's pending feedback;
        # the branches' records are folded into the run's once they are done.
        child = Executor(executor.graph, child_ctx, max_steps=executor.max_steps,
                         enable_recovery=executor.enable_recovery,
                         record=RunRecord(feedback=dict(executor.record.feedback)))
        child_ctx.budget = ctx.budget
        # Branch gate IDs are disjoint. Keep their own counters across a later
        # outer re-entry of this fork, as the DSH spine does; a new child object
        # must not grant a fresh retry allowance.
        child._gate_iterations = executor._gate_iterations
        child._reserve_step = reserve
        children.append(child)

    def walk(index: int) -> str | None:
        from .executor import GateStop, HaltStop

        target, child = node.targets[index], children[index]
        if cancelled.is_set():
            record(target, "not-started")
            return f"{target}: fork cancelled before dispatch"
        record(target, "running")
        branch_state: str | None = None
        try:
            error = child._walk_branch(target, barrier)
            allowed = {key for member in branches[index] for key in member.produces}
            changed = {key for key in ctx.facts.keys() | child.ctx.facts.keys()
                       if key not in ctx.facts or key not in child.ctx.facts
                       or ctx.facts[key] != child.ctx.facts[key]}
            if changed - allowed:
                raise ExecutionError(f"branch {target!r} changed undeclared facts: {sorted(changed - allowed)}")
        except GateStop:
            branch_state = "refused"
            error = f"{target}: gate stop"
        except HaltStop:
            branch_state = "halted"
            error = f"{target}: halted"
        except Exception as exc:
            error = f"{target}: {type(exc).__name__}: {exc}"
        if branch_state is None:
            branch_state = "cancelled" if error and cancelled.is_set() else "failed" if error else "completed"
        record(target, branch_state, error)
        return error

    errors: list[str | None] = [None] * len(children)
    optional = set(node.optional_targets or ())
    try:
        if serial:
            for index in range(len(children)):
                errors[index] = walk(index)
                # A failed optional branch is missing, as it is when branches run at
                # once: the branches after it still run.
                if errors[index] and node.targets[index] not in optional:
                    for target in node.targets[index + 1:]:
                        record(target, "not-started")
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {}
                try:
                    for index in range(len(children)):
                        futures[pool.submit(walk, index)] = index
                    for future in as_completed(futures):
                        index = futures[future]
                        errors[index] = future.result()
                except BaseException:
                    # Set the scope before executor shutdown waits for workers.
                    # Only shipped DSH clients observe it during a completion;
                    # arbitrary operations are awaited, never thread-signalled.
                    cancelled.set()
                    for future in futures:
                        future.cancel()
                    raise
    except BaseException as exc:
        # Concurrent work has been awaited; a custom/serial call unwound on its
        # own calling thread and retains its own cleanup contract. Keep the fork
        # resumable without merging facts or acknowledging its join as successful.
        cancelled.set()
        for child in children:
            executor.steps.extend(child.steps)
            executor.model_calls += child.model_calls
            ctx.writes_made.extend(child.ctx.writes_made)
        for target in node.targets:
            if status[target]["status"] == "queued":
                status[target] = {"status": "not-started"}
            elif status[target]["status"] == "running":
                status[target] = {"status": "interrupted", "error": type(exc).__name__}
        executor._record(node.id, error=f"{type(exc).__name__}: fork cancelled",
                         branches=status, schedule=schedule)
        raise
    # All workers have finished, so no branch can still write after we return or
    # after the join begins. Order does not depend on completion timing.
    executor.record.merge_branches([child.record for child in children],
                                   [{member.id for member in branch} for branch in branches])
    for index, child in enumerate(children):
        if errors[index] and node.targets[index] in optional:
            # A failed optional branch is missing, not failed: keep its steps in
            # the record, but move the error to the reason so the run's final
            # step scan (Executor.run) does not fail the run on it.
            for step in child.steps:
                if step.error:
                    _record_missing(ctx.workdir, step.node, step.error)
                    step.reason, step.error = f"missing: {step.error}", None
            _settle_missing(ctx.workdir, branches[index], errors[index])
        executor.steps.extend(child.steps)
        executor.model_calls += child.model_calls
        ctx.writes_made.extend(child.ctx.writes_made)
    error = next((e for i, e in enumerate(errors) if e and node.targets[i] not in optional), None)
    visited = {step.node for child in children for step in child.steps}
    missing = set(join.sources) - visited
    if error is None and missing:
        error = f"fork {node.id!r} did not complete join sources: {sorted(missing)}"
    if error is None:
        for index, child in enumerate(children):
            target = node.targets[index]
            # A failed optional branch is silently skipped in the fact merge.
            if errors[index] and target in optional:
                continue
            produced = {key for member in branches[index] for key in member.produces}
            for key in sorted(produced):
                if key in child.ctx.facts:
                    ctx.facts[key] = child.ctx.facts[key]
                else:
                    ctx.facts.pop(key, None)
    # On failure, facts remain the pre-fork snapshot and resume replays the fork.
    # File side effects and completed delivery receipts remain available for audit.
    executor._record(node.id if error else barrier, error=error, branches=status, schedule=schedule,
                     outcome="failed" if error else "pending")
    return error
